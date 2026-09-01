"""裁决 v4：同窗对立比较（paired-window）——终审所有遗留争议。
协议：无争议锚片段定位 → 所有变体在锚窗口 ±5s 内计分 → 同声学材料比较。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav") for a in ["CYYT_ATIS_a", "CYYT_ATIS_b"]}

# (争议名, 锚片段[无争议词], [变体])
CONTESTS = [
    ("visibility_final",
     "WIND TWO FOUR ZERO AT FIVE VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET",
     ["WIND TWO FOUR ZERO AT FIVE VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET",
      "WIND TWO FOUR ZERO AT FIVE DESCENDING LEVEL ONE FIVE TWO FOUR THOUSAND FEET",
      "WIND TWO FOUR ZERO AT FIVE VISIBILITY ONE FIVE TWO TWO THOUSAND FEET"]),
    ("approach_final",
     "ALTITUDE THREE ZERO TWO THREE APPROACH RNAV ZULU RUNWAY TWO EIGHT INFORM",
     ["ALTITUDE THREE ZERO TWO THREE APPROACH RNAV ZULU RUNWAY TWO EIGHT INFORM",
      "ALTITUDE THREE ZERO TWO THREE APPROACH ARNAV ZULU RUNWAY TWO EIGHT INFORM",
      "ALTITUDE THREE ZERO TWO THREE APPROACH RNAV LIMA RUNWAY TWO EIGHT INFORM",
      "ALTITUDE THREE ZERO TWO THREE APPROACH ARNAV LIMA RUNWAY TWO EIGHT INFORM"]),
    ("airport_v4",
     "INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR",
     ["SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR",
      "EIGHT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR",
      "SIENT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR",
      "TUCSON INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR"]),
]

sc = NLLScorer(MODEL)
report = []
for name, anchor, variants in CONTESTS:
    entry = {"contest": name, "channels": {}}
    for ch, apath in AUDIO.items():
        t_anchor = sc.find_anchor_window(apath, anchor)
        res = []
        for v in variants:
            r = sc.score_constrained(apath, v, t_anchor, half_width=5.0)
            res.append({"variant": v, **{k: r[k] for k in ("score", "median_score", "t_start")}})
            print(f"[{name}|{ch}] {r['score']:.4f} (med {r['median_score']:.4f}) @{r['t_start']:.0f}s  {v[:52]}", flush=True)
        res.sort(key=lambda r: r["score"])
        entry["channels"][ch] = {"anchor_t": t_anchor, "ranked": res}
    wa = entry["channels"]["CYYT_ATIS_a"]["ranked"][0]["variant"]
    wb = entry["channels"]["CYYT_ATIS_b"]["ranked"][0]["variant"]
    entry["dual_witness_agree"] = (wa == wb)
    entry["winner"] = wa if wa == wb else f"CONFLICT({wa[:40]} vs {wb[:40]})"
    report.append(entry)
    print(f"==> {name}: {entry['winner']}\n", flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "adjudication_v4_paired.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
print("===== 同窗终审摘要 =====")
for e in report:
    mark = "OK " if e["dual_witness_agree"] else "!! "
    print(f"{mark}{e['contest']:18s} -> {e['winner']}")
