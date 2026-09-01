"""NLL 裁判 sanity check v3 —— 反映真实数据结构。

关键事实（d2 发现）：a/b 原始文件名 20260817160051 vs 160101，
同一 ATIS 广播(FOXTROT班次)相隔 10s 的两次录音 => 同报文双信道。

关卡1 内容敏感：全长候选中 a 正确文本显著优于 乱序/倒序/幻觉
关卡2a 内容敏感(b)：b 音频上，真文本(a或b版) 必须远优于乱序
关卡2b 跨信道迁移：a文本@a 与 a文本@b 分数应接近（同广播证据）
关卡3 定位合理：argmin 在语音区
关卡4 长度偏置：量化记录（完整性归语法裁判管）
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

MODEL = "/siyuan/FunASR_extracted/FunASR-main/TT/models/whisper-large-v3-finetuned-for-ATC"
AUDIO_A = "/siyuan/FunASR_extracted/FunASR-main/TT/audio/CYYT_ATIS_a.wav"
AUDIO_B = "/siyuan/FunASR_extracted/FunASR-main/TT/audio/CYYT_ATIS_b.wav"
FINAL_DIR = Path("/siyuan/FunASR_extracted/FunASR-main/TT/research/deep/results")

text_a = (FINAL_DIR / "a_final.txt").read_text().strip()
text_b = (FINAL_DIR / "b_final.txt").read_text().strip()
words_a = text_a.split()

sc = NLLScorer(MODEL)

print("=" * 70)
print("关卡1: a 音频 · 全长候选")
full_variants = {
    "a_best(正确)": text_a,
    "shuffle(乱序)": " ".join([words_a[0]] + random.Random(42).sample(words_a[1:], len(words_a) - 1)),
    "reverse(倒序)": " ".join(reversed(words_a)),
    "halluc(幻觉×10)": "Thank you. Goodbye. Thank you. Goodbye. " * 10,
}
r1 = sc.score_many(AUDIO_A, list(full_variants.values()))
for name, r in zip(full_variants.keys(), r1):
    print(f"  [{r['score']:.4f}] @{r['t_start']:5.1f}s  {name}")
best_a = r1[0]
ok1 = best_a["score"] == min(r["score"] for r in r1)
margin1 = sorted(r["score"] for r in r1)[1] - best_a["score"]

print("=" * 70)
print("关卡2a: b 音频 · 真文本 vs 乱序")
shuf_b = " ".join(random.Random(7).sample(text_b.split(), len(text_b.split())))
r2 = sc.score_many(AUDIO_B, [text_b, text_a, shuf_b])
for name, r in zip(["b_text@b", "a_text@b", "shuffle_b@b"], r2):
    print(f"  [{r['score']:.4f}] @{r['t_start']:5.1f}s  {name}")
real_min = min(r2[0]["score"], r2[1]["score"])
ok2a = r2[2]["score"] > real_min + 0.3

print("=" * 70)
print("关卡2b: 跨信道迁移（同广播假设检验）")
s_aa = sc.score(AUDIO_A, text_a)
s_ab = sc.score(AUDIO_B, text_a)
gap = abs(s_aa["score"] - s_ab["score"])
print(f"  a_text@a: {s_aa['score']:.4f} (@{s_aa['t_start']}s)")
print(f"  a_text@b: {s_ab['score']:.4f} (@{s_ab['t_start']}s)")
print(f"  迁移差 {gap:.3f} nats/tok —— 小则支持'同广播双信道'")
ok2b = gap < 0.6

print("=" * 70)
print(f"关卡3: 定位 argmin @{best_a['t_start']}s")
ok3 = best_a["t_start"] < 150

print("=" * 70)
print("关卡4: 长度偏置（前缀）")
pref = {"1/4": " ".join(words_a[: len(words_a) // 4]),
        "2/4": " ".join(words_a[: len(words_a) // 2]),
        "3/4": " ".join(words_a[: 3 * len(words_a) // 4]),
        "4/4": text_a}
r4 = sc.score_many(AUDIO_A, list(pref.values()))
for name, r in zip(pref.keys(), r4):
    print(f"  [{r['score']:.4f}] {name} ({r['n_words']}词)")

print("=" * 70)
print(f"关卡1 内容敏感(a): {'PASS' if ok1 else 'FAIL'} (+{margin1:.3f})")
print(f"关卡2a 内容敏感(b): {'PASS' if ok2a else 'FAIL'}")
print(f"关卡2b 跨信道迁移: {'PASS' if ok2b else 'FAIL'} (gap={gap:.3f})")
print(f"关卡3 定位合理: {'PASS' if ok3 else 'FAIL'}")
verd = ok1 and ok2a and ok3
print("VERDICT:", "裁判可信（fidelity 职能）✓" if verd else "裁判不可信 ✗")
print("附注: 关卡2b", "支持同广播双信道假设" if ok2b else "存疑，需进一步实验")
