"""第一版 NLL 排行榜：候选池全部候选 × 本音频 NLL 排名。

输出：
  results/leaderboard_v1.md —— 每音频的客观排名表
  双信道共识分 = mean(NLL@own, NLL@other) 仅对 CYYT 两条计算（同广播双信道）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
AUDIO = {a: str(TT / "audio" / f"{a}.wav")
         for a in ["CYYT_ATIS_a", "CYYT_ATIS_b", "RJTT_CONTROL"]}
CROSS = {"CYYT_ATIS_a": "CYYT_ATIS_b", "CYYT_ATIS_b": "CYYT_ATIS_a"}

pool = json.load((DEEP / "results" / "candidate_pool.json").open())
sc = NLLScorer(MODEL)

rows = []
for r in pool:
    if r["words"] < 12:            # 过短候选无排名意义（Thank you 行等）
        continue
    s_own = sc.score(AUDIO[r["audio"]], r["text"])
    rec = {**r, "nll_own": s_own["score"], "t": s_own["t_start"]}
    other = CROSS.get(r["audio"])
    if other:
        s_x = sc.score(AUDIO[other], r["text"])
        rec["nll_cross"] = s_x["score"]
        rec["consensus"] = (s_own["score"] + s_x["score"]) / 2
    rows.append(rec)
    print(f"[{r['audio']}] {rec['nll_own']:.3f} {r['id']} {r['label'][:40]}", flush=True)

out = []
for audio in AUDIO:
    sub = sorted([r for r in rows if r["audio"] == audio],
                 key=lambda r: r.get("consensus", r["nll_own"]))
    out.append({"audio": audio, "ranked": sub})

(DEEP / "results" / "leaderboard_v1.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1))

# markdown 报告
md = ["# 排行榜 v1（NLL 裁判 · ATC-whisper-large-v3）", "",
      "排序键：双信道共识分（无交叉者用本信道分）。NLL 越低越贴合音频。", ""]
for blk in out:
    md += [f"## {blk['audio']}", "",
           "| # | NLL@own | NLL@cross | 共识 | @t | 词数 | id | 来源 | 标签 |",
           "|---|---------|-----------|------|----|------|----|------|------|"]
    for i, r in enumerate(blk["ranked"], 1):
        md.append("| {} | {:.3f} | {} | {} | {} | {} | {} | {} | {} |".format(
            i, r["nll_own"],
            f"{r['nll_cross']:.3f}" if "nll_cross" in r else "-",
            f"{r['consensus']:.3f}" if "consensus" in r else "-",
            r["t"], r["words"], r["id"],
            Path(r["source"]).name, r["label"][:38].replace("|", "/")))
    md.append("")
(DEEP / "results" / "leaderboard_v1.md").write_text("\n".join(md))
print("saved leaderboard_v1.md / .json")
