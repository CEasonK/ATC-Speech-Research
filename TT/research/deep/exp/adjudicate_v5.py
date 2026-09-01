"""P4b/c 终审：微窗 + 多模板 + （可选）第二裁判。

裁决项：
  1. altimeter_b     : b 信道 3023 vs 3033（beam8 孤证 vs greedy/qwen 多票）
  2. temperature_a/b : 温度+露点短语结构（6 模板，双信道）
  3. visibility_v5   : VISIBILITY vs DESCENDING LEVEL 微窗复核（v4 差距仅 0.07 且双信道反向）
  4. approach_b      : ZULU vs LIMA b 信道微窗记录（外部证据已定案 ZULU，此处存档声学证据）

方法：find_anchor_window 用无争议锚定位 -> score_constrained(half_width) 同窗计分。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL_ATC = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
MODEL_V3 = "openai/whisper-large-v3"

A = str(DEEP / ".." / ".." / "audio" / "CYYT_ATIS_a.wav")
B = str(DEEP / ".." / ".." / "audio" / "CYYT_ATIS_b.wav")
# 兜底：音频实际路径
for p in [A, B]:
    if not Path(p).exists():
        cand = list((DEEP / ".." / "..").glob("*.wav")) + list((DEEP / ".." / ".." / "audio").glob("*.wav"))
        print("audio fallback candidates:", cand)

CH = {"CYYT_ATIS_a": A, "CYYT_ATIS_b": B}

# ---------------- 争议定义 ----------------
# 1) b 信道 altimeter：上下文取争议词前 3 词 + 后 4 词
ALTI_PRE = "NINER ONE ALTITUDE"          # 温度尾接 altimeter（温度结构未定，用短前缀）
ALTI_POST = "APPROACH RNAV ZULU RUNWAY"
alti_variants = [
    f"{ALTI_PRE} THREE ZERO TWO THREE {ALTI_POST}",
    f"{ALTI_PRE} THREE ZERO THREE THREE {ALTI_POST}",
]

# 2) 温度段：前缀 = visibility 尾，后缀 = altimeter 头
TEMP_PRE = "ONE FIVE TWO FOUR THOUSAND FEET"
TEMP_POST = "ALTITUDE THREE ZERO TWO THREE"
temp_variants = [
    "TEMPERATURE ONE DECIMAL NINER DEW POINT MINUS ONE",
    "TEMPERATURE ONE POINT NINER DEW POINT MINUS ONE",
    "TEMPERATURE TWO DEW POINT MINUS ONE",
    "TEMPERATURE ONE NINER DEW POINT ONE",
    "TEMPERATURE ONE DECIMAL NINER DEW POINT ONE",
    "TEMPERATURE ONE DEW POINT MINUS ONE",
    "TEMPERATURE ONE DECIMAL NINER ONE",
]

# 3) visibility：争议词前后各带 3-4 词
VIS_VAR = [
    "ZERO AT FIVE VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE",
    "ZERO AT FIVE DESCENDING LEVEL ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE",
    "ZERO AT FIVE VISIBILITY ONE FIVE TWO TWO THOUSAND FEET TEMPERATURE",
]

# 4) approach b 信道存档复核
APPR_VAR = [
    "TWO THREE APPROACH RNAV ZULU RUNWAY TWO EIGHT",
    "TWO THREE APPROACH ARNAV ZULU RUNWAY TWO EIGHT",
    "TWO THREE APPROACH RNAV LIMA RUNWAY TWO EIGHT",
    "TWO THREE APPROACH ARNAV LIMA RUNWAY TWO EIGHT",
]

CONTESTS = [
    # (name, channels, anchor, variants, half_width)
    ("altimeter_b", ["CYYT_ATIS_b"],
     "APPROACH RUNWAY TWO EIGHT INFORM", alti_variants, 5.0),
    ("temperature_a", ["CYYT_ATIS_a"],
     "THOUSAND FEET ALTITUDE THREE ZERO TWO THREE", [f"{TEMP_PRE} {t} {TEMP_POST}" for t in temp_variants], 5.0),
    ("temperature_b", ["CYYT_ATIS_b"],
     "THOUSAND FEET ALTITUDE THREE ZERO TWO THREE", [f"{TEMP_PRE} {t} {TEMP_POST}" for t in temp_variants], 5.0),
    ("visibility_v5", ["CYYT_ATIS_a", "CYYT_ATIS_b"],
     "THOUSAND FEET TEMPERATURE", VIS_VAR, 3.0),
    ("approach_b_archive", ["CYYT_ATIS_b"],
     "RUNWAY TWO EIGHT INFORM CENTER", APPR_VAR, 5.0),
]


def run(scorer, tag):
    out = []
    for name, chans, anchor, variants, hw in CONTESTS:
        rec = {"contest": name, "judge": tag, "channels": {}}
        for ch in chans:
            t_anchor = scorer.find_anchor_window(CH[ch], anchor)
            ranked = []
            for v in variants:
                r = scorer.score_constrained(CH[ch], v, t_anchor, half_width=hw)
                ranked.append({"variant": v, **r})
                print(f"[{name}|{tag}|{ch}] {r['score']:.4f} (med {r['median_score']:.4f}) @{r['t_start']:.0f}s  {v[:70]}", flush=True)
            ranked.sort(key=lambda x: x["score"])
            rec["channels"][ch] = {"anchor_t": t_anchor, "ranked": ranked}
            win = ranked[0]["variant"]
            lose = ranked[1]["variant"]
            margin = ranked[1]["score"] - ranked[0]["score"]
            print(f"==> {name}|{tag}|{ch}: WIN by {margin:.3f} -> {win[:80]}", flush=True)
        # 双信道一致性
        winners = [c["ranked"][0]["variant"] for c in rec["channels"].values()]
        rec["dual_agree"] = len(set(winners)) == 1
        rec["winner"] = winners[0] if rec["dual_agree"] else f"CONFLICT({winners[0][:40]} vs {winners[1][:40]})"
        out.append(rec)
    return out


def main():
    results = []
    print("===== judge 1: whisper-atc =====", flush=True)
    sc1 = NLLScorer(MODEL_ATC)
    results += run(sc1, "atc")
    del sc1
    torch.cuda.empty_cache()

    print("===== judge 2: whisper-large-v3 (vanilla) =====", flush=True)
    try:
        sc2 = NLLScorer(MODEL_V3)
        results += run(sc2, "v3")
        del sc2
        torch.cuda.empty_cache()
    except Exception as e:
        print("judge2 failed:", e, flush=True)

    outf = DEEP / "results" / "adjudication_v5.json"
    outf.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
