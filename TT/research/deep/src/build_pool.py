"""候选池收集器：把散落在 results/（TT 级与 deep 级）的
全部文本候选统一成 {audio, source, label, text, words} 记录，供 NLL 排行榜用。

原则：永不丢证据。短/垃圾候选保留但标 keep=False。
"""
import json
import re
from pathlib import Path

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
OUT = Path(__file__).resolve().parents[1] / "results" / "candidate_pool.json"

AUDIOS = ["CYYT_ATIS_a", "CYYT_ATIS_b", "RJTT_CONTROL"]


def guess_audio(name: str) -> str:
    for a in AUDIOS:
        if a in name:
            return a
    return "unknown"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


pool = []
n = 0

# 1) results/ 每文件一候选
for f in sorted((TT / "results").rglob("*.txt")):
    text = norm(f.read_text(errors="ignore"))
    if not text:
        continue
    pool.append({
        "id": f"R{n:03d}", "audio": guess_audio(f.name + str(f.parent)),
        "source": str(f.relative_to(TT)), "label": f.stem, "text": text,
        "words": len(text.split()),
    })
    n += 1

# 2) deep 终稿交付物（原扫描 research/best/、research/scratch/，
#    两目录已于 2026-08-25 清理；现只收仍存在的来源。
#    注意：results/candidate_pool.json 为含旧来源的历史版本，
#    重跑本脚本会重新编号且不含已删目录的候选）
for f in sorted((Path(__file__).resolve().parents[1] / "results").glob("CYYT_ATIS_*.txt")):
    text = norm(f.read_text())
    pool.append({"id": f"R{n:03d}", "audio": guess_audio(f.name),
                 "source": str(f.relative_to(TT)), "label": "deep_final", "text": text,
                 "words": len(text.split())})
    n += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(pool, OUT.open("w"), ensure_ascii=False, indent=1)
by_audio = {}
for r in pool:
    by_audio.setdefault(r["audio"], [0, 0])
    by_audio[r["audio"]][0] += 1
    if r["words"] >= 20:
        by_audio[r["audio"]][1] += 1
print(f"pool: {len(pool)} candidates -> {OUT}")
for a, (tot, big) in sorted(by_audio.items()):
    print(f"  {a}: {tot} 条 (>=20词: {big})")
