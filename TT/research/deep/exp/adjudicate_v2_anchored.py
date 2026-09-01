"""CONFLICT 二审：长上下文锚定片段。
问题：短片段 min-over-windows 各自找窗，可能落在报文不同位置/不同循环实例。
方案：给争议词加上下文锚（前 6 词 + 后 6 词），迫使 min 锁定到报文中
争议词的真实位置，对立对在同一声学材料上比较。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav") for a in ["CYYT_ATIS_a", "CYYT_ATIS_b"]}

# 长上下文锚定对立对（锚文本来自双信道共识结构）
CONTESTS = [
    ("visibility",
     ["WIND TWO FOUR ZERO AT FIVE VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET",
      "WIND TWO FOUR ZERO AT FIVE DISABILITY ONE FIVE TWO FOUR THOUSAND FEET"]),
    ("frequency",
     ["CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE AS REQUESTED",
      "CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE SIX AS REQUESTED",
      "CENTER ON FREQUENCY ONE TWO TWO ONE FIVE AS REQUESTED"]),
    ("time_zulu",
     ["WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR",
      "WEATHER AT ZERO TWO ZERO ZERO JULIETT WIND TWO FOUR",
      "WEATHER AT ZERO TWO ZERO ZERO JULIET WIND TWO FOUR"]),
    # runway 复核：加长锚确认 EIGHT 不是窗口漂移假象
    ("runway_long",
     ["APPROACH RNAV ZULU RUNWAY TWO EIGHT INFORM GANDER",
      "APPROACH RNAV ZULU RUNWAY TWO NINE INFORM GANDER"]),
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
            print(f"[{name}|{ch}] {r['score']:.4f} @{r['t_start']:.0f}s  {v[:60]}", flush=True)
        res.sort(key=lambda r: r["score"])
        entry["channels"][ch] = res
    wa = entry["channels"]["CYYT_ATIS_a"][0]["variant"]
    wb = entry["channels"]["CYYT_ATIS_b"][0]["variant"]
    entry["dual_witness_agree"] = (wa == wb)
    entry["winner"] = wa if wa == wb else f"CONFLICT({wa[:40]} vs {wb[:40]})"
    report.append(entry)
    print(f"==> {name}: {entry['winner']}", flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "adjudication_v2_anchored.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
print("\n===== 二审摘要 =====")
for e in report:
    mark = "OK " if e["dual_witness_agree"] else "!! "
    print(f"{mark}{e['contest']:12s} -> {e['winner']}")
