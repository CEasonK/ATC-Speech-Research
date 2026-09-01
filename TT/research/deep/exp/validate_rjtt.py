"""RJTT 段级 NLL 校验：v3 中立裁判在 {atc_beam5, qwen} 转写间逐段挑选。
去自偏置：atc 自家转写不由 atc 自己判（只参考），由 v3 判。
输出 results/rjtt_final.txt + rjtt_validation.json
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
SEG = DEEP / "segments" / "RJTT_CONTROL"

idx = json.loads((DEEP / "results" / "decode_rjtt_index.json").read_text())
by_id = {e["id"]: e["text"] for e in idx}
MANIFEST = json.loads((SEG / "manifest.json").read_text())
SEGS = {f"seg{k:02d}": m for k, m in enumerate(MANIFEST)}


def main():
    sc = NLLScorer("openai/whisper-large-v3")
    out = {}
    final_lines = []
    for s, m in SEGS.items():
        apath = str(SEG / f"seg_{s[3:]}.wav")  # 磁盘文件名带下划线：seg_00.wav
        center = (m["t0"] + m["t1"]) / 2 - m["t0"]  # 段内相对时间（段音频从 t0 切出）
        half = max(m["dur"] / 2 + 2.0, 5.0)
        cands = {}
        for tag in ["atc_beam5", "qwen"]:
            t = by_id[f"{tag}_{s}"]
            r = sc.score_constrained(apath, t, center, half_width=half)
            cands[tag] = {"text": t, "nll_v3": r["score"]}
            print(f"[{s}] v3|{tag}: {r['score']:.4f}  {t[:60]}", flush=True)
        win = min(cands, key=lambda k: cands[k]["nll_v3"])
        out[s] = cands
        out[s]["winner"] = win
        final_lines.append(f"[{s}] ({win}) {cands[win]['text'].strip()}")
        print(f"==> {s}: WIN {win}", flush=True)
    del sc
    torch.cuda.empty_cache()

    (DEEP / "results" / "rjtt_final.txt").write_text("\n\n".join(final_lines) + "\n")
    (DEEP / "results" / "rjtt_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("saved rjtt_final.txt / rjtt_validation.json")


if __name__ == "__main__":
    main()
