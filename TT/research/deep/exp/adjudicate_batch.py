"""批量争议字段裁决（P4）：一次加载模型，跑全部最小对立对，双信道交叉。

对立对设计原则：短片段（争议字段±2词），对立对之间词数相同 → 公平比较。
min-over-windows 语义下，真片段应在循环广播的多个实例中稳定胜出。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav") for a in ["CYYT_ATIS_a", "CYYT_ATIS_b"]}

# (争议名, [变体文本...])  —— 每组内词数严格相同
CONTESTS = [
    ("visibility", ["VISIBILITY ONE FIVE", "DISABILITY ONE FIVE"]),
    ("runway",     ["RUNWAY TWO EIGHT", "RUNWAY TWO NINE"]),
    ("center",     ["INFORM GANDER CENTER", "INFORM FANDAS CENTER",
                    "INFORM SANDGREN CENTER", "INFORM MONCTON CENTER"]),
    ("frequency",  ["FREQUENCY ONE TWO THREE DECIMAL ONE FIVE",
                    "FREQUENCY ONE TWO TWO ONE FIVE",
                    "FREQUENCY ONE TWO THREE DECIMAL ONE SIX"]),
    ("time_zulu",  ["ZERO TWO ZERO ZERO ZULU", "ZERO TWO ZERO ZERO JULIETT",
                    "ZERO TWO ZERO ZERO JULIET"]),
    ("airport",    ["SAINT JOHNS INFORMATION FOXTROT", "EIGHT JOHNS INFORMATION FOXTROT"]),
    ("approach",   ["APPROACH RNAV ZULU", "APPROACH ARNAV ZULU", "APPROACH ARNAV NEW"]),
    ("ceiling",    ["TWO FOUR THOUSAND FEET TEMPERATURE", "CEILING TWO FOUR THOUSAND FEET TEMPERATURE"]),
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
            print(f"[{name}|{ch}] {r['score']:.4f} @{r['t_start']:.0f}s  {v}", flush=True)
        res.sort(key=lambda r: r["score"])
        entry["channels"][ch] = res
    # 一致性：两信道冠军是否相同
    wa = entry["channels"]["CYYT_ATIS_a"][0]["variant"]
    wb = entry["channels"]["CYYT_ATIS_b"][0]["variant"]
    entry["dual_witness_agree"] = (wa == wb)
    entry["winner"] = wa if wa == wb else f"CONFLICT({wa} vs {wb})"
    report.append(entry)
    print(f"==> {name}: {entry['winner']}", flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "adjudication_v1.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
print("saved:", out)

# 摘要
print("\n===== 裁决摘要 =====")
for e in report:
    mark = "OK " if e["dual_witness_agree"] else "!! "
    print(f"{mark}{e['contest']:12s} -> {e['winner']}")
