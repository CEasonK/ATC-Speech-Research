"""RJTT 定稿更新器 v14 轮：seg06 X 槽位改判落盘 + 终稿 v5 + 覆盖率重算。
改判：JOHNSON LEVEL → JUST OUT OF（置信 low-medium，JOIN YOU OUT OF 记竞争假设）
证据链：v3b 自由解码 'we join you out of 180' + v3-ja『ジョンシュアドブ』+
turbo 'jet of'/qwen 'junk jet of' 同音族 + v14 配对计分 out-of 族双胜 JOHNSON
(0.707/0.764 vs 0.890)。JOHNSON 为 atc 孤证+v13 轮旧胜者（当时未测 out-of 族）。
守案记录：seg03 NINE TWO THREE / seg02 NINE SEVEN TWO THREE / FEDEX ONE FIVE
在 v14 中全部击退新假设（margin 均小于污染带但方向一致）。
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

NOTE = ("；v14 改判：JOHNSON LEVEL→JUST OUT OF——v3b 自由解码 'we join you out of 180'"
        "+v3-ja『ジョンシュアドブ』+turbo 'jet of'/qwen 'junk jet of' 同音族+配对计分 "
        "JOIN_YOU_OUT_OF 0.707/JUST_OUT_OF 0.764 双胜 JOHNSON 0.890；"
        "JUST OUT OF 与 JOIN YOU OUT OF 近平手(Δ0.057)，取惯用语前者，后者记竞争假设")
cons["seg06"]["text"] = cons["seg06"]["text"].replace(
    "HEAVY JOHNSON LEVEL ONE EIGHT ZERO", "HEAVY JUST OUT OF ONE EIGHT ZERO")
cons["seg06"]["conf"] = "medium"
cons["seg06"]["basis"] += NOTE

GUARD = ("；v14 守案：{item} 击退新假设 {alt}（{scores}），方向一致维持原案")
cons["seg03"]["basis"] += GUARD.format(
    item="呼号数字 NINE TWO THREE", alt="SEVEN TWO THREE(v3b 'Japan Air 723')",
    scores="1.615<1.897")
cons["seg02"]["basis"] += GUARD.format(
    item="NINE SEVEN TWO THREE", alt="JAPAN AIR SEVEN TWO THREE 统一假设",
    scores="2.798<2.948<3.018<3.041")
cons["seg06"]["basis"] += GUARD.format(
    item="机号 FEDEX ONE FIVE", alt="ONE FIVE ZERO(v3b 'FedEx 150')",
    scores="0.890<1.122")

(RES / "rjtt_consensus.json").write_text(json.dumps(cons, ensure_ascii=False, indent=1))

# ---- 覆盖率 ----
def toks(t):
    return re.sub(r"[^A-Z0-9]+", " ", t.upper()).split()


SOURCES = ["atc_beam5", "qwen", "turbo"]
cov_rows, tot_n, tot_k = [], 0, 0
for s, m in cons.items():
    ftoks = toks(m["text"])
    src_sets = [set(toks(idx[f"{tag}_{s}"])) for tag in SOURCES]
    src_sets.append(set(toks(lang[s]["en"]["text"])))
    src_sets.append(set(re.findall(r"[A-Za-z0-9]+", lang[s]["ja"]["text"])))
    k = sum(1 for w in ftoks if sum(w in ss for ss in src_sets) >= 2)
    cov_rows.append((s, len(ftoks), k, round(k / max(1, len(ftoks)), 3), m["conf"]))
    tot_n += len(ftoks)
    tot_k += k

coverage = {
    "definition": "终稿 token 被 ≥2 个证据源支持的占比；可信下界",
    "overall": {"tokens": tot_n, "supported": tot_k,
                "coverage": round(tot_k / tot_n, 3)},
    "per_segment": [{"seg": s, "tokens": n, "supported": k, "coverage": c,
                     "conf": cf} for s, n, k, c, cf in cov_rows],
}
(RES / "rjtt_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=1))

# ---- 终稿 v5 ----
lines = ["# RJTT_CONTROL 最终文本 v5（v14 守案+seg06 改判后）", ""]
for s in sorted(cons):
    m = cons[s]
    cmap = dict((r[0], r[3]) for r in cov_rows)[s]
    lines.append(f"[{s}] {m['t0']}-{m['t1']}s (conf={m['conf']}) (coverage={cmap})")
    lines.append(m["text"])
    lines.append("")
lines.append("---")
lines.append(f"总体多数票覆盖率: {tot_k}/{tot_n} = {tot_k/tot_n:.1%}（≥2 源支持的可信下界）")
lines.append("v5 变更: 仅 seg06 一处——HEAVY 后槽位 JOHNSON LEVEL → JUST OUT OF；"
             "其余八段经 v14 全部守案。证据: results/adjudication_v14.json")
(RES / "rjtt_final.txt").write_text("\n".join(lines))

print(f"overall coverage: {tot_k}/{tot_n} = {tot_k/tot_n:.1%}")
for s, n, k, c, cf in cov_rows:
    print(f"  {s}: {k}/{n} = {c:.0%}  conf={cf}")
