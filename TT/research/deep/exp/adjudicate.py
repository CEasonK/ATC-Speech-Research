"""争议字段声学裁决（P4）：最小对立对 + 双信道交叉。

方法：
  1. 以某骨架文本为底，对争议字段构造变体（minimal pair）
  2. 教师强制 NLL 计分——声学上哪个词更贴合音频
  3. 双信道交叉：a 的裁决结果在 b 上复验（同广播双录音 = 双证人）
  4. 领域知识（ICAO 语法/机场事实）作为先验记录，不替代声学证据

用法：
  python exp/adjudicate.py --base-file research/deep/results/a_final.txt \
      --variants 'RUNWAY_TWO_EIGHT=RUNWAY TWO EIGHT' 'RUNWAY_TWO_NINE=RUNWAY TWO NINE' ...
变体语法：NAME=FULL_TEXT（完整替换后的全文，简单可靠）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav")
         for a in ["CYYT_ATIS_a", "CYYT_ATIS_b", "RJTT_CONTROL"]}
CROSS = {"CYYT_ATIS_a": "CYYT_ATIS_b", "CYYT_ATIS_b": "CYYT_ATIS_a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, choices=list(AUDIO))
    ap.add_argument("--base-file", help="骨架文本文件（变体=骨架时省略）")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="NAME=FULL_TEXT，至少 2 个互斥变体")
    ap.add_argument("--tag", default="adjudication", help="输出文件标签")
    args = ap.parse_args()

    sc = NLLScorer(MODEL)
    pairs = []
    for v in args.variants:
        name, _, text = v.partition("=")
        pairs.append((name.strip(), text.strip()))

    audios = [args.audio]
    other = CROSS.get(args.audio)
    if other:
        audios.append(other)   # 双证人

    rows = []
    for a in audios:
        res = []
        for name, text in pairs:
            r = sc.score(AUDIO[a], text)
            res.append({"name": name, **r})
            print(f"[{a}] {r['score']:.4f} @{r['t_start']:.0f}s {name}", flush=True)
        res.sort(key=lambda r: r["score"])
        best = res[0]
        margin = best["score"] - res[1]["score"] if len(res) > 1 else 0.0
        rows.append({"audio": a, "winner": best["name"],
                     "margin": round(margin, 4),
                     "ranked": [{k: r[k] for k in ("name", "score", "t_start")} for r in res]})

    # 双证人一致性
    verdict = {"tag": args.tag, "base": args.base_file or "(inline)", "rows": rows}
    if len(rows) == 2:
        agree = rows[0]["winner"] == rows[1]["winner"]
        verdict["dual_witness_agree"] = agree
        verdict["final"] = rows[0]["winner"] if agree else "CONFLICT->manual"
    print(json.dumps(verdict, ensure_ascii=False, indent=1))

    out = Path(__file__).resolve().parents[1] / "results" / f"adjudicate_{args.tag}.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
