"""P3 前置：能量包络自相关找 ATIS 循环周期。
报文周期 T 一旦确定，就能相位对齐切出完整报文实例（不切断词），
供各引擎整周期重解码。
"""
import sys
from pathlib import Path

import librosa
import numpy as np

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
SR = 16000


def envelope(wav, hop_s=0.05):
    hop = int(hop_s * SR)
    n = len(wav) // hop
    e = np.array([np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                  for i in range(n)])
    return e, hop_s


def find_period(env, hop_s, tmin=15.0, tmax=120.0):
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lo, hi = int(tmin / hop_s), min(int(tmax / hop_s), len(ac) - 1)
    k = lo + int(np.argmax(ac[lo:hi]))
    # 抛物线插值细化峰位
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = (y0 - 2 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    return (k + shift) * hop_s, ac, k


for name in ["CYYT_ATIS_a", "CYYT_ATIS_b", "RJTT_CONTROL"]:
    wav, _ = librosa.load(str(TT / "audio" / f"{name}.wav"), sr=SR, mono=True)
    dur = len(wav) / SR
    env, hop_s = envelope(wav)
    # 能量门限：找有声/静音段结构
    thr = np.percentile(env, 40) * 0.5
    voiced = env > thr
    # 周期
    T, ac, k = find_period(env, hop_s)
    print(f"== {name}: dur={dur:.1f}s  period≈{T:.2f}s  (dur/T≈{dur/T:.2f})")
    # 静音间隙位置（报文边界候选）
    gaps, start = [], None
    for i, v in enumerate(voiced):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if (i - start) * hop_s >= 1.0:   # >=1s 静音算间隙
                gaps.append((start * hop_s, (i - start) * hop_s))
            start = None
    print(f"   silences>=1s: {[(round(g,1), round(d,1)) for g, d in gaps[:12]]}")
