"""P5b wind 段独立裁决（a/b 分开！范式：不同日期→风可不同）。
b 信道 wind 行在最终验证中正位=错位=2.42（不贴合），需独立找真文。
锚：visibility 行（已双裁判定案）→ wind 在其前方。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}

VARIANTS = [
    "WIND TWO FOUR ZERO AT FIVE",
    "WINDS TWO FOUR ZERO AT FIVE",
    "WIND TWO FOUR ZERO AT FIVE KNOTS",
    "SURFACE WIND TWO FOUR ZERO AT FIVE",
    "WIND TWO FOUR ZERO AT ONE FIVE",
    "WIND TWO FOUR ZERO AT ONE ZERO",
    "WIND TWO FOUR ZERO AT EIGHT",
    "WIND TWO FOUR ZERO AT ZERO FIVE",
    "WIND TWO ZERO ZERO AT FIVE",
    "WIND TWO SEVEN ZERO AT FIVE",
    "WIND TWO THREE ZERO AT FIVE",
    "WIND THREE ONE ZERO AT FIVE",
    "WIND TWO FOUR ZERO DEGREES AT FIVE",
]

JUDGES = [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
          ("v3", "openai/whisper-large-v3")]


def main():
    results = {}
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        sc = NLLScorer(mdir)
        for ch, apath in CH.items():
            # 锚 = visibility 行（已定案），wind 在其前方 ~2.5s 结束
            t_vis = sc.find_anchor_window(apath, "ONE FIVE TWO FOUR THOUSAND FEET")
            t_wind = t_vis - 2.5
            ranked = []
            for v in VARIANTS:
                r = sc.score_constrained(apath, v, t_wind, half_width=5.0)
                ranked.append({"variant": v, **r})
            ranked.sort(key=lambda x: x["score"])
            results.setdefault(ch, {})[tag] = ranked
            for r in ranked:
                print(f"[wind|{tag}|{ch}] {r['score']:.4f}  {r['variant']}", flush=True)
            print(f"==> wind|{tag}|{ch}: WIN -> {ranked[0]['variant']}", flush=True)
        del sc
        torch.cuda.empty_cache()

    (DEEP / "results" / "adjudication_v10_wind.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1))
    print("saved adjudication_v10_wind.json")


if __name__ == "__main__":
    main()
