"""尾三行复诵复核探针（2026-08-25）。
背景：assemble_final 的尾三行拼接曾被 review 误判为 bug；用户听音确认
音频中尾三行在周期内真实复诵。本脚本用锚窗 NLL 曲线独立验证：
锚 "YOU HAVE INFORMATION FOXTROT" 为尾段特有措辞（区别于开头的
"SAINT JOHNS INFORMATION FOXTROT WEATHER"），若复诵存在，则每个广播
周期内应出现两个相邻深谷，间隔≈尾三行朗读时长（约 10~20s），
且完整尾三行文本在两处约束计分应同样低。
输出：results/tail_repeat_check.json
"""
import json, sys
from pathlib import Path
import numpy as np

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
ANCHOR = "YOU HAVE INFORMATION FOXTROT"
TRIO = ("APPROACH RNAV ZULU RUNWAY TWO EIGHT "
        "APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT "
        "INFORM ATC THAT YOU HAVE INFORMATION FOXTROT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")


def local_minima(t_starts, ws):
    """低于 10 分位且为 ±2 窗局部最小的点；5s 内合并。"""
    lo = np.percentile(ws, 10)
    out = []
    for i in range(len(ws)):
        if ws[i] > lo:
            continue
        lft = ws[max(0, i - 2):i]
        rgt = ws[i + 1:i + 3]
        if (not len(lft) or ws[i] <= lft.min()) and (not len(rgt) or ws[i] <= rgt.min()):
            if not out or t_starts[i] - out[-1][0] > 5:
                out.append((float(t_starts[i]), round(float(ws[i]), 3)))
    return out


def main():
    sc = NLLScorer(MODEL)
    res = {}
    for ch in ("CYYT_ATIS_a", "CYYT_ATIS_b"):
        wav = str(TT / "audio" / f"{ch}.wav")
        entry = sc.load_audio(wav)
        ids = sc.tok.encode(normalize_text(ANCHOR), add_special_tokens=False)
        ws = sc._score_ids(entry, ids)
        mins = local_minima(entry["win_starts"], np.asarray(ws))
        print(f"\n== {ch} 锚窗深谷 [t(s), NLL]: {[(round(t,1), s) for t,s in mins]}")

        pairs = [(mins[i], mins[i + 1]) for i in range(len(mins) - 1)
                 if 8 <= mins[i + 1][0] - mins[i][0] <= 25]
        print(f"   周期内复诵对(间隔8~25s): "
              f"{[((round(a[0],1)), (round(b[0],1))) for a,b in pairs]}")

        trio_scores = []
        for (t1, _), (t2, _) in pairs[:4]:
            # 锚短语位于尾三行末尾附近，整段中心约在锚前 4s
            s1 = sc.score_constrained(wav, TRIO, t1 - 4.0)
            s2 = sc.score_constrained(wav, TRIO, t2 - 4.0)
            trio_scores.append({"t_first": round(t1, 1), "t_second": round(t2, 1),
                                "trio_first": round(s1["score"], 3),
                                "trio_second": round(s2["score"], 3)})
            print(f"   尾三行约束分 @第一处{t1:.0f}s={s1['score']:.3f} "
                  f"@第二处{t2:.0f}s={s2['score']:.3f}")
        res[ch] = {"minima": [[round(t, 1), s] for t, s in mins],
                   "pairs": [[a[0], b[0]] for a, b in pairs],
                   "trio_scores": trio_scores}

    out = DEEP / "results" / "tail_repeat_check.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
