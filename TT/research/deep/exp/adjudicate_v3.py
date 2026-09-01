"""裁决 v3：decode_v2 发现的新争议（双信道 anchored）。
1) altimeter 3023 vs 3033（a 全体 vs b beam8）
2) approach ZULU vs LIMA（b 信道提出）
3) 机场名 SAINT JOHNS vs EIGHT JOHNS 终审（Qwen 支持 Saint John's）
4) inst_03 异常：DESCENDING LEVEL vs VISIBILITY
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav") for a in ["CYYT_ATIS_a", "CYYT_ATIS_b"]}

CONTESTS = [
    ("altimeter",
     ["ONE ALTITUDE THREE ZERO TWO THREE APPROACH",
      "ONE ALTITUDE THREE ZERO THREE THREE APPROACH"]),
    ("approach_name",
     ["THREE APPROACH RNAV ZULU RUNWAY TWO EIGHT",
      "THREE APPROACH ARNAV ZULU RUNWAY TWO EIGHT",
      "THREE APPROACH RNAV LIMA RUNWAY TWO EIGHT",
      "THREE APPROACH ARNAV LIMA RUNWAY TWO EIGHT"]),
    ("airport_final",
     ["JOHNS INFORMATION FOXTROT WEATHER AT",
      "SAINT JOHNS INFORMATION FOXTROT WEATHER AT",
      "EIGHT JOHNS INFORMATION FOXTROT WEATHER AT",
      "SIENT JOHNS INFORMATION FOXTROT WEATHER AT"]),
    ("visibility_inst03",
     ["ZERO WIND TWO FOUR ZERO AT FIVE VISIBILITY ONE FIVE TWO FOUR",
      "ZERO WIND TWO FOUR ZERO AT FIVE DESCENDING LEVEL ONE FIVE TWO FOUR"]),
]

sc = NLLScorer(MODEL)
report = []
for name, variants in CONTESTS:
    entry = {"contest": name, "channels": {}}
    for ch, apath in AUDIO.items():
        res = []
        for v in variants:
            r = sc.score(apath, v)
            res.append({"variant": v, **{k: r[k] for k in ("score", "t_start")}})
            print(f"[{name}|{ch}] {r['score']:.4f} @{r['t_start']:.0f}s  {v[:55]}", flush=True)
        res.sort(key=lambda r: r["score"])
        entry["channels"][ch] = res
    wa = entry["channels"]["CYYT_ATIS_a"][0]["variant"]
    wb = entry["channels"]["CYYT_ATIS_b"][0]["variant"]
    entry["dual_witness_agree"] = (wa == wb)
    entry["winner"] = wa if wa == wb else f"CONFLICT({wa[:36]} vs {wb[:36]})"
    report.append(entry)
    print(f"==> {name}: {entry['winner']}\n", flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "adjudication_v3.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
print("===== 裁决 v3 摘要 =====")
for e in report:
    mark = "OK " if e["dual_witness_agree"] else "!! "
    print(f"{mark}{e['contest']:16s} -> {e['winner']}")
