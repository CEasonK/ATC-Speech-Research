"""P5 最终组装：a/b 两份 ATIS 文本 + 逐行锚定 NLL 验证。

验证逻辑：每行在其真实锚位计分（正位）vs 在其他行锚位计分（错位）。
若全文正确，正位 NLL 应显著低于所有错位（行序乱序基线）。
输出 results/a_final.txt, b_final.txt, final_validation.json

口径修正（P2/D3，含主审自查再修正）：计分窗固定 WIN_S=30s；实测广播周期
a≈28.143s (exp/english_hardening.py 拟合)、b≈27.85s (exp/cut_b_anchored.py)。
窗长≥周期 ⇒ 错位窗仍含整份报文，检验在构造上失去内容排他性，只剩
"窗-文本相位对齐度"成分——DEGRADED 判定由此结构性论证即成立。
实测四组 Δ=-0.032/+0.143/+0.156/+0.536 nat（final_validation.json）：Δ 的
零点与噪声带未经标定（a-atc 已反向），不可作独立裁决证据；注意 v8 的
1.2 nat 是"同窗插词"口径的污染上限，属他口径迁移值，对本统计量未标定，
仅作数量级参考、不构成阈值。历史 weighted_nll_wrongpos 按此降级解读
（正位绝对 NLL 列不受影响）。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text, WIN_S

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}

# 实测广播周期（见模块 docstring）；新信道务必先测周期再加进来
PERIOD_S = {"CYYT_ATIS_a": 28.143, "CYYT_ATIS_b": 27.85}


def position_test_validity(win_len_s, period_s):
    """正位/错位检验"完全有效"当且仅当 窗长 < 广播周期。

    窗长≥周期时错位窗仍覆盖整份报文，检验构造上失去内容排他性，
    Δ只剩窗-文本相位对齐成分且零点未经标定——可算但不得作独立裁决证据。"""
    return bool(win_len_s < period_s)

# (行文本, 锚文本) —— 锚 = 该行内无争议词组，find_anchor_window 定位
LINES_COMMON = [
    ("SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
     "INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO"),
    ("WIND TWO FOUR ZERO AT FIVE",
     "WIND TWO FOUR ZERO AT FIVE"),
    ("VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET",
     "THOUSAND FEET"),
    ("TEMPERATURE ONE DEW POINT MINUS ONE",
     "TEMPERATURE ONE"),
]
# v12+能量探针定案：两信道 wind 行同文 "WIND TWO FOUR ZERO AT FIVE"
# （a 的弱化 AT @57.90-58.01；b 的 WIND @233.44-233.81 鼻音平台包络）
LINES_TAIL = [
    ("APPROACH RNAV ZULU RUNWAY TWO EIGHT",
     "APPROACH RNAV ZULU RUNWAY TWO EIGHT"),
    ("APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
     "LANDING AND DEPARTING RUNWAY"),
    ("INFORM ATC THAT YOU HAVE INFORMATION FOXTROT",
     "HAVE INFORMATION FOXTROT"),
]
FREQ = "INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE"
# closing 词形按信道独立（v7b 三裁判一致：a=AS REQUESTED / b=WHEN REQUESTED）
LINES_A = (LINES_COMMON
           + [("ALTIMETER THREE ZERO TWO THREE", "ALTIMETER THREE ZERO TWO THREE")]
           + LINES_TAIL[:1]
           + [(FREQ + " AS REQUESTED", "CENTER ON FREQUENCY")]
           + LINES_TAIL[1:])
LINES_B = (LINES_COMMON
           + [("ALTIMETER THREE ZERO THREE THREE", "ALTIMETER THREE ZERO THREE THREE")]
           + LINES_TAIL[:1]
           + [(FREQ + " WHEN REQUESTED", "CENTER ON FREQUENCY")]
           + LINES_TAIL[1:])


def validate(scorer, ch, lines):
    """逐行正位计分 + 错位矩阵（窗长≥周期时照常计分，但 position_test
    标 DEGRADED，Δ 仅作参考）。

    返回 (rows, position_test) —— position_test 记录守卫判定与依据。"""
    period = PERIOD_S.get(ch)
    assert period is not None, f"unknown period for {ch}; measure it first"
    valid = position_test_validity(WIN_S, period)
    ptest = {"valid": valid, "win_len_s": WIN_S, "period_s": period,
             "note": ("ok" if valid else
                      f"DEGRADED: 窗长{WIN_S}s ≥ 周期{period}s，错位窗仍含整份"
                      "报文，构造上无内容排他性；Δ零点未标定（实测含反向组），"
                      "不得作独立裁决证据")}
    anchors = {}
    for text, anchor in lines:
        if anchor not in anchors:
            anchors[anchor] = scorer.find_anchor_window(CH[ch], anchor)
    rows = []
    for i, (text, anchor) in enumerate(lines):
        t = anchors[anchor]
        r = scorer.score_constrained(CH[ch], text, t, half_width=5.0)
        nt = len(scorer.tok.encode(normalize_text(text), add_special_tokens=False))
        rows.append({"line": text, "anchor": anchor, "t": t,
                     "nll_correct_pos": r["score"], "n_tokens": nt})
        print(f"  [{ch}] {r['score']:.4f}  {text[:58]}", flush=True)
    if not valid:
        print(f"  [{ch}] position test DEGRADED (win {WIN_S}s >= period "
              f"{period}s) — wrongpos Δ 仅作参考", flush=True)
    # 错位基线：每行放到相邻行的锚位
    for i, row in enumerate(rows):
        j = (i + 1) % len(lines)
        t_wrong = anchors[lines[j][1]]
        r = scorer.score_constrained(CH[ch], row["line"], t_wrong, half_width=5.0)
        row["nll_wrong_pos"] = r["score"]
    return rows, ptest


def main():
    out = {"a": {}, "b": {}}
    # 尾三行为音频中的真实复诵（用户听音确认；exp/recheck_tail_repeat.py 探针
    # 佐证：a 信道锚窗深谷按 主报文尾/复诵尾 成对出现 (5,25)(85,105)s，周期≈80s；
    # b 信道 (215,230)s 同构），故在 9 行主体后显式追加复诵段 LINES_TAIL。
    texts = {
        "CYYT_ATIS_a": [t for t, _ in LINES_A + LINES_TAIL],
        "CYYT_ATIS_b": [t for t, _ in LINES_B + LINES_TAIL],
    }
    judges = [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
              ("v3", "openai/whisper-large-v3")]
    for tag, mdir in judges:
        print(f"===== judge: {tag} =====", flush=True)
        sc = NLLScorer(mdir)
        for ch, lines in [("CYYT_ATIS_a", LINES_A + LINES_TAIL),
                          ("CYYT_ATIS_b", LINES_B + LINES_TAIL)]:
            print(f"--- {ch} ---", flush=True)
            rows, ptest = validate(sc, ch, lines)
            correct = sum(r["nll_correct_pos"] * r["n_tokens"] for r in rows)
            total_tok = sum(r["n_tokens"] for r in rows)
            wrong = sum(r["nll_wrong_pos"] * r["n_tokens"] for r in rows)
            out["a" if ch.endswith("_a") else "b"][tag] = {
                "rows": rows,
                "position_test": ptest,
                "weighted_nll_correct": correct / total_tok,
                "weighted_nll_wrongpos": wrong / total_tok,
            }
            flag = "" if ptest["valid"] else " [DEGRADED]"
            print(f"  ==> {ch} {tag}: correct={correct/total_tok:.4f} "
                  f"wrongpos={wrong/total_tok:.4f}{flag}", flush=True)
        del sc
        torch.cuda.empty_cache()

    # 写最终文本
    for ch_key, fname in [("a", "a_final.txt"), ("b", "b_final.txt")]:
        body = "\n".join(texts[f"CYYT_ATIS_{ch_key}"])
        (DEEP / "results" / fname).write_text(body + "\n")
        print(f"wrote results/{fname}")

    (DEEP / "results" / "final_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("saved final_validation.json")


if __name__ == "__main__":
    main()
