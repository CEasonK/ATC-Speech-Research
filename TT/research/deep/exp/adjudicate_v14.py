"""RJTT v14 裁决：增强解码后的第二轮。
新证据源：v3 beam5 裸奔(v3b) 与 ATC 提示词(v3p)，见 decode_enrich_rjtt.py。
裁决项：
  A) seg03 呼号数字：NINE TWO THREE(现案) vs SEVEN TWO THREE
     —— v3b 'Japan Air 723' + qwen 'Japan Air 720' 挑战现案
  B) seg02 呼号：NINE SEVEN TWO THREE(现案) vs JAPAN AIR SEVEN TWO THREE
     （统一假设：seg02=管制指令、seg03=同一航班 JAL723 的复诵）
     —— v3 greedy '723'/v3b '1773' 支持；atc 'THAI...' 为孤例
  C) seg06 机号：FEDEX ONE FIVE(现案) vs ONE FIVE ZERO —— v3b 'FedEx 150'
  D) seg06 X 槽位第二轮：JOHNSON LEVEL(现案) vs JOIN YOU OUT OF / JUST OUT OF
     —— v3b 'we join you out of 180' 新证词
输出 results/adjudication_v14.json

口径修正（P2/D2）：score 是 per-token 均值，不同 token 数的候选直接比大小存在
长度混淆（短假设的 LM 先验稀释程度不同）。现改为：同一 contest 所有候选共用同
一音频窗（adjudicate_contest 纯函数保证），winner/decidable 只在同 n_tok 组内
判定，跨长度优势标记 length_confounded。历史 seg06_x_slot2 结论
(JOIN_YOU vs JUST_OUT_OF vs JOHNSON_LEVEL) 需按本口径重裁后方可采信。
"""
import json
import sys
from pathlib import Path

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text

SEG = DEEP / "segments" / "RJTT_CONTROL"
OUT = DEEP / "results" / "adjudication_v14.json"
CONTAM = 1.4

manifest = {f"seg{k:02d}": m for k, m in
            enumerate(json.loads((SEG / "manifest.json").read_text()))}


def win(s):
    m = manifest[s]
    return (m["t0"] + m["t1"]) / 2 - m["t0"], max(m["dur"] / 2 + 2.0, 5.0)


S02_TAIL = "CROSS MADIGU AT FLIGHT LEVEL THREE TWO ZERO"
s02_cands = {
    "NINE_SEVEN_TWO_THREE":   f"NINE SEVEN TWO THREE {S02_TAIL}",
    "JAL_SEVEN_TWO_THREE":    f"JAPAN AIR SEVEN TWO THREE {S02_TAIL}",
    "SEVEN_TWO_THREE":        f"SEVEN TWO THREE {S02_TAIL}",
    "JAL_NINE_TWO_THREE":     f"JAPAN AIR NINE TWO THREE {S02_TAIL}",
}
s03_cands = {
    "NINE_TWO_THREE":  f"JAPAN AIR NINE TWO THREE {S02_TAIL}",
    "SEVEN_TWO_THREE": f"JAPAN AIR SEVEN TWO THREE {S02_TAIL}",
}


def s06(x, num="ONE FIVE"):
    return (f"TOKYO CONTROL FEDEX {num} HEAVY {x} CLIMBING TWO FOUR ZERO "
            f"FEDEX {num} TOKYO CONTROL CLIMBING LEVEL THREE TWO ZERO "
            f"INITIALLY CLIMB THREE TWO ZERO FEDEX {num}")


s06_num = {"ONE_FIVE": s06("JOHNSON LEVEL"), "ONE_FIVE_ZERO": s06("JOHNSON LEVEL", "ONE FIVE ZERO")}
s06_x = {
    "JOHNSON_LEVEL":    s06("JOHNSON LEVEL"),
    "JOIN_YOU_OUT_OF":  s06("JOIN YOU OUT OF ONE EIGHT ZERO"),
    "JUST_OUT_OF":      s06("JUST OUT OF ONE EIGHT ZERO"),
}

TASKS = [
    ("seg03_digits", "seg03", s03_cands),
    ("seg02_callsign", "seg02", s02_cands),
    ("seg06_fedex_num", "seg06", s06_num),
    ("seg06_x_slot2", "seg06", s06_x),
]


def adjudicate_contest(scorer, apath, t_center, half, cands, contam=CONTAM):
    """同窗裁决纯函数：所有候选共用 (apath, t_center, half) 同一切片与配置，
    差异只在文本假设。返回 per-token NLL + 等长组内判定。
    - winner: 全局最优候选（其 per-token 最低）
    - fair_margin: 与同 n_tok 组内次优的差（组内无第二名为 None）
    - length_confounded: 全局次优与全局最优 n_tok 不同 → 优势可能来自长度而非声学
    - decidable: fair_margin 存在且 > contam"""
    scores, n_tok = {}, {}
    for k, text in cands.items():
        r = scorer.score_constrained(apath, text, t_center, half_width=half)
        scores[k] = round(r["score"], 4)
        n_tok[k] = len(scorer.tok.encode(normalize_text(text),
                                         add_special_tokens=False))
    order = sorted(scores.items(), key=lambda kv: kv[1])
    best, best_s = order[0]
    same_len = [(k, v) for k, v in order if n_tok[k] == n_tok[best]]
    fair_margin = round(same_len[1][1] - best_s, 4) if len(same_len) > 1 else None
    length_confounded = (len(order) > 1
                         and n_tok[order[1][0]] != n_tok[best])
    decidable = fair_margin is not None and fair_margin > contam
    return {"scores": scores, "n_tok": n_tok,
            "winner": best,
            "margin_to_2nd": round(order[1][1] - best_s, 4) if len(order) > 1 else None,
            "fair_margin": fair_margin,
            "length_confounded": length_confounded,
            "decidable": decidable,
            # 下游默认消费这个：等长对决可裁 且 无跨长度嫌疑
            "decided_strict": decidable and not length_confounded,
            "calib_note": "contam=1.4 标定于旧 global-margin 口径，"
                          "对 fair_margin 分布尚未重标定",
            "ranking": [k for k, _ in order]}


def main():
    sc = NLLScorer("openai/whisper-large-v3")
    out = {}
    for name, seg, cands in TASKS:
        apath = str(SEG / f"seg_{seg[3:]}.wav")
        t_center, half = win(seg)
        res = adjudicate_contest(sc, apath, t_center, half, cands, CONTAM)
        out[name] = res
        print(f"[{name}] winner={res['winner']} margin={res['margin_to_2nd']} "
              f"fair={res['fair_margin']} len_conf={res['length_confounded']} "
              f"decidable={res['decidable']}", flush=True)
        for k in res["ranking"]:
            print(f"    {k:22s} {res['scores'][k]:.4f}  ({res['n_tok'][k]}tok)",
                  flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
