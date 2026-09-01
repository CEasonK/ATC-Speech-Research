"""RJTT v13 裁决：语言审计后的候选池扩充轮。
新证据源：v3 自由解码（此前只给 atc/qwen/turbo 三家打分，v3 自己的解码从未进池）。
裁决项：
  A) seg01+seg08 呼号 4 假设交叉验证：
     SIERRA(turbo) / SHAMROCK(atc=Aer Lingus 呼号) / SHANGHAI AIR(qwen+v3 双独立)
     —— 两段必须同向才定案，单段领先不数
  B) seg04 ORANGE 后词：NINER(现案) vs LINER(v3-ja 佐证) vs LINE(qwen)
  C) seg06 X 槽位：JOHNSON(现案) vs LEAVING FLIGHT(标准报告词语法假设)
       vs PASSING FLIGHT / JUST ON
判据：v3 同窗对立 NLL；差值 <1.4 nat 落入 LM 污染区间不单独定案。
输出 results/adjudication_v13.json
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

SEG = DEEP / "segments" / "RJTT_CONTROL"
OUT = DEEP / "results" / "adjudication_v13.json"

manifest = {f"seg{k:02d}": m for k, m in
            enumerate(json.loads((SEG / "manifest.json").read_text()))}


def path_of(s):
    return str(SEG / f"seg_{s[3:]}.wav")


def win(s):
    m = manifest[s]
    center = (m["t0"] + m["t1"]) / 2 - m["t0"]
    half = max(m["dur"] / 2 + 2.0, 5.0)
    return center, half


# ---------- 候选构造 ----------
CALLSIGNS = {
    "SIERRA": "SIERRA EIGHT NINER SIX",
    "SHAMROCK": "SHAMROCK EIGHT NINER SIX",
    "SHANGHAI_AIR": "SHANGHAI AIR EIGHT NINER SIX",
}

S01_TAIL = ("CLIMBING FLIGHT LEVEL TWO HUNDRED {cs} TOKYO CONTROL "
            "CLIMB MAINTAIN FLIGHT LEVEL THREE ZERO ZERO REQUEST DIRECT TO MAIDA "
            "CLIMBING FLIGHT LEVEL THREE ZERO ZERO DIRECT TO MAIDA {cs}")
s01_cands = {k: f"TOKYO CONTROL GOOD AFTERNOON {cs} " + S01_TAIL.format(cs=cs)
             for k, cs in CALLSIGNS.items()}

S08_HEAD = "TOKYO CONTROL {cs} REQUEST FLIGHT LEVEL THREE EIGHT ZERO OR FOUR ZERO ZERO"
s08_cands = {k: S08_HEAD.format(cs=cs) for k, cs in CALLSIGNS.items()}

s04_cands = {
    "NINER":  "ORANGE NINER ONE EIGHT THREE QUEBEC ONE THREE",
    "LINER":  "ORANGE LINER ONE EIGHT THREE QUEBEC ONE THREE",
    "LINE":   "ORANGE LINE ONE EIGHT THREE QUEBEC ONE THREE",
}

S06_TAIL = ("FEDEX ONE FIVE TOKYO CONTROL CLIMBING LEVEL THREE TWO ZERO "
            "INITIALLY CLIMB THREE TWO ZERO FEDEX ONE FIVE")
s06_cands = {
    "JOHNSON":         f"TOKYO CONTROL FEDEX ONE FIVE HEAVY JOHNSON LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO {S06_TAIL}",
    "LEAVING_FLIGHT":  f"TOKYO CONTROL FEDEX ONE FIVE HEAVY LEAVING FLIGHT LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO {S06_TAIL}",
    "PASSING_FLIGHT":  f"TOKYO CONTROL FEDEX ONE FIVE HEAVY PASSING FLIGHT LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO {S06_TAIL}",
    "JUST_ON":         f"TOKYO CONTROL FEDEX ONE FIVE HEAVY JUST ON LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO {S06_TAIL}",
}

TASKS = [
    ("callsign_seg01", "seg01", s01_cands),
    ("callsign_seg08", "seg08", s08_cands),
    ("orange_slot",    "seg04", s04_cands),
    ("x_slot_seg06",   "seg06", s06_cands),
]

CONTAM = 1.4


def main():
    sc = NLLScorer("openai/whisper-large-v3")
    out = {"contamination_band": CONTAM}
    for name, seg, cands in TASKS:
        apath = path_of(seg)
        t_center, half = win(seg)
        scores = {}
        for k, text in cands.items():
            r = sc.score_constrained(apath, text, t_center, half_width=half)
            scores[k] = round(r["score"], 4)
        best = min(scores, key=scores.get)
        order = sorted(scores.items(), key=lambda kv: kv[1])
        margin = order[1][1] - order[0][1]
        out[name] = {
            "scores": scores,
            "winner": best,
            "margin_to_2nd": round(margin, 4),
            "decidable": bool(margin > CONTAM),
            "ranking": [k for k, _ in order],
        }
        print(f"[{name}] winner={best} margin={margin:.3f} "
              f"decidable={margin > CONTAM}", flush=True)
        for k, v in order:
            print(f"    {k:16s} {v:.4f}", flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
