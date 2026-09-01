"""P5f-2 细粒度能量探针：10ms 原始值逐点打印，回答两个槽位问题。
Q1: a 的 ZERO(57.74-57.88)→FIVE(58.08-58.22) 间隙有无弱化 AT？
Q2: b 的 233.44-233.76 burst 是 WIND 还是 ZULU 的一部分？
对照：b 已知 AT=234.99-235.08；a 词间静音参照 57.64-57.74、57.09-57.23。
"""
from pathlib import Path

import librosa
import numpy as np

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
SR = 16000
HOP = 160


def profile(ch, t0, t1, label):
    wav, _ = librosa.load(str(TT / "audio" / f"CYYT_ATIS_{ch}.wav"), sr=SR, mono=True)
    seg = wav[int(t0 * SR):int(t1 * SR)]
    n = len(seg) // HOP
    e = np.sqrt(np.mean(seg[:n * HOP].reshape(n, HOP) ** 2, axis=1))
    print(f"\n--- [{ch}] {label} ({t0}-{t1}s, 10ms/frame) ---")
    line = []
    for i, v in enumerate(e):
        bar = "#" * int(min(v / 0.05, 40))
        line.append(f"{t0 + i * 0.01:7.3f}  {v:.4f}  {bar}")
    print("\n".join(line))


# Q1: a 的 AT 槽位 + 两侧词尾/词首
profile("a", 57.70, 58.30, "ZERO尾→[AT?]→FIVE头")
# 静音参照：a 的词间间隙
profile("a", 57.62, 57.76, "FOUR→ZERO 间隙(已知无词)")
# Q2: b 的争议 burst 及前后
profile("b", 233.00, 234.00, "ZULU?→[WIND?]→TWO")
# 对照：b 已知 AT
profile("b", 234.90, 235.45, "AT(已知)→FIVE")
