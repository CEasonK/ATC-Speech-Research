"""多系统翻译结果客观汇总：两两 chrF 一致性矩阵 + 审计指标对照 + 最终推荐译文。

红线：只用确定性规则（chrF / 数字审计 / 术语审计），无主观打分。
用法：python summarize_translate.py --tag a_final
输出：results/<tag>/summary.md 与 final_zh.txt（推荐译文）
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from glossary import chr_f, numeric_fidelity, term_audit  # noqa: E402

RES = SRC.parent / "results"


def audit_tag(en_lines, zh_lines):
    """返回 (numeric_fidelity, term_hit_rate) 双指标；zh_lines 允许 None 占位。"""
    zl = [z if z else "" for z in zh_lines]
    if any(not z for z in zh_lines):
        return 0.0, 0.0
    r1, _ = numeric_fidelity(en_lines, zl)
    r2, _ = term_audit(en_lines, zl)
    return round(r1, 4), round(r2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ARGS = ap.parse_args()

    rd = RES / ARGS.tag
    en = json.load(open(rd / "template.json"))["en"]

    systems = {}
    for name, key in [("template", "zh"), ("m2m_direct", "zh")]:
        p = rd / f"{name}.json"
        if p.exists():
            systems[name] = json.load(open(p))[key]

    # 回译一致性（基于 EN 原文的 chrF）
    back_scores = {}
    for p in sorted(rd.glob("m2m_back_*.json")):
        d = json.load(open(p))
        back_scores[d["based_on"]] = d["chrf_mean"]
        if d["based_on"] not in systems and "qwen" in d["based_on"]:
            pass  # zh 来自对应 qwen json

    for variant in ("free", "constrained"):
        p = rd / f"qwen_{variant}.json"
        if p.exists():
            d = json.load(open(p))
            if all(z for z in d["zh"]):
                systems[f"qwen_{variant}"] = d["zh"]
            else:
                bad = [i + 1 for i, z in enumerate(d["zh"]) if not z]
                print(f"[warn] qwen_{variant} 存在未解析行 {bad}，"
                      f"已剔除出对比矩阵与终稿选举池")

    names = list(systems)
    n = len(names)
    lines = [f"# {ARGS.tag} 翻译多系统对比", "", f"EN 行数: {len(en)}",
             f"系统: {', '.join(names)}", ""]

    # 审计指标表
    lines += ["| 系统 | 数字保真 | 术语命中 | 回译chrF |",
              "|---|---|---|---|"]
    for nm in names:
        numf, termh = audit_tag(en, systems[nm])
        bc = back_scores.get(nm, "-")
        lines.append(f"| {nm} | {numf} | {termh} | {bc} |")
    lines.append("")

    # 两两 chrF 一致性矩阵
    if n >= 2:
        lines += ["## 两两 chrF 一致性（越高=译文越接近）", "",
                  "| | " + " | ".join(names) + " |",
                  "|---|" + "---|" * n]
        for i, a in enumerate(names):
            row = [a]
            for j, b in enumerate(names):
                if i == j:
                    row.append("—")
                elif j < i:
                    sc = sum(chr_f(x, y)
                             for x, y in zip(systems[a], systems[b])) / len(en)
                    row.append(f"{sc:.3f}")
                else:
                    row.append("——")  # 对称矩阵，上三角省略
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 推荐规则（确定性）：优先数字+术语双满分的 Qwen constrained；
    # 否则取 (numf*2 + termh + back_chrf) 综合分最高的系统。
    cand = []
    for nm in names:
        numf, termh = audit_tag(en, systems[nm])
        bc = back_scores.get(nm, 0.0)
        cand.append((nm, numf, termh, bc, 2 * numf + termh + bc))

    pref = [c for c in cand if c[1] == 1.0 and c[2] == 1.0 and c[0].startswith("qwen")]
    best = max(pref or cand, key=lambda c: c[4])
    lines += [f"## 推荐系统：{best[0]}", "",
              f"综合得分(2*数字保真+术语命中+回译chrF): {best[4]:.4f}", ""]

    finals = systems[best[0]]
    lines += ["## 最终推荐译文", ""]
    for i, (e, z) in enumerate(zip(en, finals)):
        lines += [f"**{i+1}.** `{e}`", f"→ {z}", ""]
    (rd / "final_zh.txt").write_text("\n".join(finals) + "\n")
    (rd / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"saved: {rd/'summary.md'}  推荐: {best[0]}")


if __name__ == "__main__":
    main()
