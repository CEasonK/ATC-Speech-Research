"""P4f 温度段破局实验：双裁判语言先验对峙，用音素级证据裁决。

方法 A：切片解码 —— 定位温度段中心，切出 16s 短音频，多引擎独立解码。
        短片段无长上下文干扰，解码器被迫只靠这段发音。
方法 B：增量计分 —— 两条假设短语逐词累加，per-token NLL 曲线揭示
        每个词位置的声学支持度（哪个词开始陡增 = 音频里没有它）。

假设 H_atc : TEMPERATURE ONE DECIMAL NINER ONE ALTITUDE ...
假设 H_v3  : TEMPERATURE ONE DEW POINT MINUS ONE ALTITUDE ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}
OUT = DEEP / "results" / "temp_slices"
OUT.mkdir(parents=True, exist_ok=True)

JUDGES = [
    ("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
    ("v3", "openai/whisper-large-v3"),
]

# ---------- 方法 B：增量计分 ----------
HYP_A = ["TEMPERATURE", "ONE", "DECIMAL", "NINER", "ONE", "ALTITUDE", "THREE"]
HYP_B = ["TEMPERATURE", "ONE", "DEW", "POINT", "MINUS", "ONE", "ALTITUDE", "THREE"]


def incremental(scorer, ch, words_list, t_center):
    """逐词前缀计分：prefix_k = words[:k]，返回每级 per-token NLL（同窗）。"""
    rows = []
    for k in range(2, len(words_list) + 1):
        text = " ".join(words_list[:k])
        r = scorer.score_constrained(CH[ch], text, t_center, half_width=5.0)
        rows.append({"k": k, "text": text, **r})
        print(f"  [{ch}] +{words_list[k-1]:10s} -> {r['score']:.4f}", flush=True)
    return rows


def main():
    import librosa
    import soundfile as sf

    results = {"slices": {}, "incremental": {}}
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        sc = NLLScorer(mdir)

        # 定位温度段中心（锚 = 温度前后无争议词）
        for ch in CH:
            t_a = sc.find_anchor_window(CH[ch], "THOUSAND FEET ALTITUDE THREE ZERO")
            # 温度段约在锚窗内 "THOUSAND FEET" 之后。取锚窗中心偏后作为切片中心。
            center = t_a + 15.0
            results["incremental"].setdefault(ch, {})[tag] = {
                "H_atc": incremental(sc, ch, HYP_A, t_a),
                "H_v3": incremental(sc, ch, HYP_B, t_a),
            }

            # ---------- 方法 A：切片 ----------
            if tag == "atc":  # 切片只需做一次
                wav, _ = librosa.load(CH[ch], sr=16000, mono=True)
                s0 = max(0, int((center - 8) * 16000))
                s1 = min(len(wav), int((center + 8) * 16000))
                sp = OUT / f"{ch}_tempslice.wav"
                sf.write(sp, wav[s0:s1], 16000)
                print(f"[slice] {sp} ({(s1-s0)/16000:.1f}s @ {center:.0f}s)", flush=True)
                results["slices"][ch] = {"path": str(sp), "center": center}

        del sc
        torch.cuda.empty_cache()

    (DEEP / "results" / "adjudication_v8_incremental.json").write_text(
        json.dumps(results["incremental"], ensure_ascii=False, indent=1))
    print("saved incremental json")


if __name__ == "__main__":
    main()
