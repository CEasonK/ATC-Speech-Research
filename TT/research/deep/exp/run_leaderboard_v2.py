"""排行榜 v2：乱序基线归一，解决跨信道基线差。

动机（sanity 关卡2b）：a_text@a=0.452 vs a_text@b=1.559，gap=1.107——
b 信道整体 NLL 基线更高（信道质量差），直接比较跨信道分数不公平。
归一化：Δ = NLL(shuffled) - NLL(text) = "文本相对乱序文本的声学优势"。
乱序文本在同一窗口计分，消除信道/位置基线，Δ 才跨信道可比。

每候选：NLL_own、NLL_x（跨信道）、SHUF_own（k=3 个乱序样本均值）、SHUF_x
Δ_own = SHUF_own - NLL_own；Δ_x = SHUF_x - NLL_x；共识 = mean(Δ_own, Δ_x)
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer, normalize_text  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav")
         for a in ["CYYT_ATIS_a", "CYYT_ATIS_b", "RJTT_CONTROL"]}
CROSS = {"CYYT_ATIS_a": "CYYT_ATIS_b", "CYYT_ATIS_b": "CYYT_ATIS_a"}
K_SHUF = 3


def shuffle_text(text: str, seed: int) -> str:
    words = normalize_text(text).split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


pool = json.load((DEEP / "results" / "candidate_pool.json").open())
sc = NLLScorer(MODEL)

rows = []
for r in pool:
    if r["words"] < 12:
        continue
    a_own, a_x = AUDIO[r["audio"]], AUDIO.get(CROSS.get(r["audio"], ""), "")
    s_own = sc.score(a_own, r["text"])
    shuf_own = sum(sc.score(a_own, shuffle_text(r["text"], s))["score"]
                   for s in range(K_SHUF)) / K_SHUF
    rec = {**r,
           "nll_own": round(s_own["score"], 4),
           "shuf_own": round(shuf_own, 4),
           "delta_own": round(shuf_own - s_own["score"], 4),
           "t": s_own["t_start"]}
    if a_x:
        s_x = sc.score(a_x, r["text"])
        shuf_x = sum(sc.score(a_x, shuffle_text(r["text"], s))["score"]
                     for s in range(K_SHUF)) / K_SHUF
        rec.update(nll_x=round(s_x["score"], 4),
                   shuf_x=round(shuf_x, 4),
                   delta_x=round(shuf_x - s_x["score"], 4),
                   delta_consensus=round((shuf_own - s_own["score"] + shuf_x - s_x["score"]) / 2, 4))
    rows.append(rec)
    print(f"[{r['audio']}] d_own={rec['delta_own']:.3f} "
          f"d_x={rec.get('delta_x', '-')} {r['id']} {r['label'][:36]}", flush=True)

out = []
for audio in AUDIO:
    sub = sorted([r for r in rows if r["audio"] == audio],
                 key=lambda r: r.get("delta_consensus", r["delta_own"]), reverse=True)
    out.append({"audio": audio, "ranked": sub})
(DEEP / "results" / "leaderboard_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

md = ["# 排行榜 v2（乱序基线归一 · Δ = NLL_shuffled − NLL_text，越大越好）", ""]
for blk in out:
    md += [f"## {blk['audio']}", "",
           "| # | Δ_own | Δ_cross | 共识Δ | NLL@own | @t | 词数 | id | 来源 | 标签 |",
           "|---|-------|---------|-------|---------|----|------|----|------|------|"]
    for i, r in enumerate(blk["ranked"], 1):
        md.append("| {} | {:.3f} | {} | {} | {:.3f} | {} | {} | {} | {} | {} |".format(
            i, r["delta_own"],
            f"{r['delta_x']:.3f}" if "delta_x" in r else "-",
            f"{r['delta_consensus']:.3f}" if "delta_consensus" in r else "-",
            r["nll_own"], r["t"], r["words"], r["id"],
            Path(r["source"]).name, r["label"][:36].replace("|", "/")))
    md.append("")
(DEEP / "results" / "leaderboard_v2.md").write_text("\n".join(md))
print("saved leaderboard_v2.md / .json")
