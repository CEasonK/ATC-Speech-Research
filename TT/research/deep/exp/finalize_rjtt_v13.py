"""RJTT 定稿更新器：把 v13 裁决落进共识文件 + 终稿，并重算多数票覆盖率。
翻案项（证据见 adjudication_v13.json 与 rjtt_lang_audit.json）：
  1) seg01/seg08 呼号 SIERRA → SHANGHAI AIR：
     - v13 同窗配对计分两段同向（0.485<0.537；1.350<1.516）
     - v3 自由解码两段独立产出 "Shanghai Air 896"（新证据源，此前未入池）
     - qwen seg01 独立产出 "Shanghai A96"
     - 域先验：SHANGHAI AIR 为真实航司呼号且飞羽田；SIERRA 非航司呼号格式；
       SHAMROCK(Aer Lingus) 无日本航线。turbo 的 SIERRA 记为竞争假设保留。
  2) seg04 ORANGE NINER → ORANGE LINER：
     - v3-ja 「オレンジライナー」+ qwen "Orange Line" 双引擎独立 + v13 计分方向一致
     （margin 小于污染带，靠多源同向升格，置信度标 low-medium）
不变项：
  seg06 JOHNSON 击退 LEAVING/PASSING/JUST_ON 全部标准短语假设 → 维持+未决标注；
  seg05 维持碎片低置信；seg00 维持原案，KTX10Y(v3-en) 记为竞争假设。
覆盖率口径：终稿 token 在 ≥2 个证据源（atc/qwen/turbo/v3free/v3ja）token 集中出现的比例。
输出：rjtt_consensus.json（覆盖）、rjtt_final.txt（覆盖）、rjtt_coverage.json
"""
import json
import re
from pathlib import Path

DEEP = Path(__file__).resolve().parents[1]
RES = DEEP / "results"

cons = json.loads((RES / "rjtt_consensus.json").read_text())
idx = {e["id"]: e["text"] for e in
       json.loads((RES / "decode_rjtt_index.json").read_text())}
lang = json.loads((RES / "rjtt_lang_audit.json").read_text())

# ---- 1) 翻案 ----
V13_NOTE = ("；v13 语言审计后翻案：v3 自由解码 seg01/seg08 双段独立产出 "
            "'Shanghai Air 896'+qwen 'Shanghai A96'+同窗配对计分双段同向"
            "(0.485<0.537 / 1.350<1.516)+域先验(真实呼号格式)，SIERRA(turbo)记为竞争假设")

cons["seg01"]["text"] = cons["seg01"]["text"].replace(
    "SIERRA EIGHT NINER SIX", "SHANGHAI AIR EIGHT NINER SIX")
cons["seg01"]["conf"] = "medium"
cons["seg01"]["basis"] += V13_NOTE

cons["seg08"]["text"] = cons["seg08"]["text"].replace(
    "SIERRA EIGHT NINER SIX", "SHANGHAI AIR EIGHT NINER SIX")
cons["seg08"]["conf"] = "medium"
cons["seg08"]["basis"] += V13_NOTE

cons["seg04"]["text"] = cons["seg04"]["text"].replace("ORANGE NINER", "ORANGE LINER")
cons["seg04"]["conf"] = "low-medium"
cons["seg04"]["basis"] += ("；v13 翻案：ORANGE LINER——v3-ja『オレンジライナー』与 qwen "
                           "'Orange Line' 双引擎独立佐证+v13 计分方向一致(LINER 2.419<NINER 2.649)"
                           "，但 margin<污染带，故仅凭多源同向升格")

cons["seg00"]["basis"] += ("；v13 补记：v3-en 自由解码给出竞争假设 'KTX10Y'，与 "
                           "GOLF ZERO JULIETT 冲突不可裁决，维持原案")

(RES / "rjtt_consensus.json").write_text(json.dumps(cons, ensure_ascii=False, indent=1))


# ---- 2) 覆盖率 ----
def toks(t):
    return re.sub(r"[^A-Z0-9]+", " ", t.upper()).split()


SOURCES = ["atc_beam5", "qwen", "turbo"]
cov_rows, tot_n, tot_k = [], 0, 0
for s, m in cons.items():
    ftoks = toks(m["text"])
    src_sets = [set(toks(idx[f"{tag}_{s}"])) for tag in SOURCES]
    src_sets.append(set(toks(lang[s]["en"]["text"])))          # v3 自由解码(en)
    src_sets.append(set(re.findall(r"[A-Za-z0-9]+", lang[s]["ja"]["text"])))
    k = sum(1 for w in ftoks if sum(w in ss for ss in src_sets) >= 2)
    cov_rows.append((s, len(ftoks), k, round(k / max(1, len(ftoks)), 3), m["conf"]))
    tot_n += len(ftoks)
    tot_k += k

coverage = {
    "definition": "终稿 token 被 ≥2 个证据源(atc/qwen/turbo/v3free_en/v3free_ja)支持的占比；可信下界",
    "overall": {"tokens": tot_n, "supported": tot_k,
                "coverage": round(tot_k / tot_n, 3)},
    "per_segment": [{"seg": s, "tokens": n, "supported": k, "coverage": c,
                     "conf": cf} for s, n, k, c, cf in cov_rows],
}
(RES / "rjtt_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=1))

# ---- 3) 终稿 ----
lines = ["# RJTT_CONTROL 最终文本 v4（v13 语言审计+翻案轮后）", ""]
for s in sorted(cons):
    m = cons[s]
    lines.append(f"[{s}] {m['t0']}-{m['t1']}s (conf={m['conf']}) "
                 f"(coverage={dict((r[0], r[3]) for r in cov_rows)[s]})")
    lines.append(m["text"])
    lines.append("")
lines.append("---")
lines.append(f"总体多数票覆盖率: {tot_k}/{tot_n} = {tot_k/tot_n:.1%}（≥2 源支持的可信下界）")
lines.append("翻案记录: seg01/seg08 呼号→SHANGHAI AIR EIGHT NINER SIX；"
             "seg04→ORANGE LINER（详见 results/adjudication_v13.json / rjtt_lang_audit.json）")
(RES / "rjtt_final.txt").write_text("\n".join(lines))

print(f"overall coverage: {tot_k}/{tot_n} = {tot_k/tot_n:.1%}")
for s, n, k, c, cf in cov_rows:
    print(f"  {s}: {k}/{n} = {c:.0%}  conf={cf}")
print("wrote rjtt_consensus.json / rjtt_final.txt / rjtt_coverage.json")
