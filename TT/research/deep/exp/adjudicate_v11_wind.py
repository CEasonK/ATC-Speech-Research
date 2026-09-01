"""P5d wind 精确组合裁决：用切片实测中心定位，测 WIND 前缀有无 + AT 有无。
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
# 切片实测中心（slice_wind.py: open@55/230 + 7）
CENTER = {"CYYT_ATIS_a": 62.0, "CYYT_ATIS_b": 237.0}

VAR_A = [
    "WIND TWO FOUR ZERO AT FIVE",
    "WIND TWO FOUR ZERO FIVE",
    "TWO FOUR ZERO AT FIVE",
    "WIND TWO SEVEN ZERO AT FIVE",
]
VAR_B = [
    "WIND TWO FOUR ZERO AT FIVE",
    "TWO FOUR ZERO AT FIVE",
    "ZULU WIND TWO FOUR ZERO AT FIVE",
    "WIND TWO FOUR ZERO FIVE",
]

JUDGES = [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
          ("v3", "openai/whisper-large-v3")]


def main():
    results = {}
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        sc = NLLScorer(mdir)
        for ch, variants in [("CYYT_ATIS_a", VAR_A), ("CYYT_ATIS_b", VAR_B)]:
            ranked = []
            for v in variants:
                r = sc.score_constrained(CH[ch], v, CENTER[ch], half_width=5.0)
                ranked.append({"variant": v, **r})
            ranked.sort(key=lambda x: x["score"])
            results.setdefault(ch, {})[tag] = ranked
            for r in ranked:
                print(f"[wind2|{tag}|{ch}] {r['score']:.4f}  {r['variant']}", flush=True)
            print(f"==> wind2|{tag}|{ch}: WIN -> {ranked[0]['variant']}", flush=True)
        del sc
        torch.cuda.empty_cache()

    (DEEP / "results" / "adjudication_v11_wind.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1))
    print("saved adjudication_v11_wind.json")


if __name__ == "__main__":
    main()
