"""P5f wind 词间能量物理探测（LM-free 裁判）：
用 10ms RMS 包络定位 a/b wind 区的语音 burst，回答：
  Q1: a 的 ZERO 与 FIVE 之间有无 AT 的发音能量？
  Q2: b 的 ZULU 与 TWO FOUR 之间有无 WIND 的发音能量？
标定参照：a 的 WIND burst（已知存在，实词能量）、b 的 AT burst（已知存在，功能词能量）、
局部帧最小值 = 底噪。切片解码给出的粗定位：a wind≈56.5-60s，b wind≈234-238.5s。
"""
import json
from pathlib import Path

import librosa
import numpy as np

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SR = 16000
HOP = 160  # 10ms

WAV = {"a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
       "b": str(TT / "audio" / "CYYT_ATIS_b.wav")}
REG = {"a": (55.0, 62.0), "b": (231.5, 240.0)}


def envelope(ch):
    wav, _ = librosa.load(WAV[ch], sr=SR, mono=True)
    t0, t1 = REG[ch]
    seg = wav[int(t0 * SR):int(t1 * SR)]
    n = len(seg) // HOP
    e = np.sqrt(np.mean(seg[:n * HOP].reshape(n, HOP) ** 2, axis=1))
    return e, t0


def show(ch, e, t0):
    floor = float(np.percentile(e, 10))
    peak = float(e.max())
    print(f"\n===== {ch}  region={REG[ch]}  floor(p10)={floor:.5f}  peak={peak:.5f} =====")
    # ASCII 包络：0.05s/字符，能量分 9 级
    step = 5
    bins = np.sqrt(np.mean((e[:len(e)//step*step].reshape(-1, step))**2, axis=1))
    scale = " .:-=+*#%@"
    line1, line2 = [], []
    for i, v in enumerate(bins):
        t = t0 + i * 0.05
        line1.append("|" if abs(t - round(t)) < 0.026 and round(t) % 1 == 0 else " ")
        lvl = int(np.clip((v - floor) / (peak - floor + 1e-9) * 9, 0, 9))
        line2.append(scale[lvl])
    for s in range(0, len(bins), 100):
        print(f"{t0+s*0.05:6.2f}s |{''.join(line1[s:s+100])}")
        print(f"        |{''.join(line2[s:s+100])}")
    return floor


def bursts_and_stats(ch, e, t0, floor):
    th = max(floor * 3.5, 0.004)
    out, in_b, s, pk = [], False, 0, 0.0
    for i, v in enumerate(e):
        if v > th and not in_b:
            in_b, s, pk = True, i, float(v)
        elif in_b:
            pk = max(pk, float(v))
            if v <= th:
                in_b = False
                if i - s >= 8:
                    out.append({"t0": round(t0 + s * 0.01, 2),
                                "t1": round(t0 + i * 0.01, 2),
                                "dur": round((i - s) * 0.01, 2), "peak": round(float(pk), 4)})
    if in_b:
        out.append({"t0": round(t0 + s * 0.01, 2), "t1": round(t0 + len(e) * 0.01, 2),
                    "dur": round((len(e) - s) * 0.01, 2), "peak": round(pk, 4)})
    print(f"[{ch}] threshold={th:.5f} bursts:")
    for b in out:
        print(f"    {b['t0']:7.2f}-{b['t1']:7.2f}  dur={b['dur']:4.2f}s  peak={b['peak']}")
    return out


def gap_stats(ch, e, t0, g0, g1, label):
    i0, i1 = int((g0 - t0) * 100), int((g1 - t0) * 100)
    seg = e[max(0, i0):min(len(e), i1)]
    if len(seg) == 0:
        print(f"[{ch}] {label}: EMPTY")
        return
    print(f"[{ch}] {label} [{g0:.2f},{g1:.2f}]s  mean={seg.mean():.5f}  "
          f"max={seg.max():.5f}  p90={np.percentile(seg, 90):.5f}")


res = {}
for ch in ("a", "b"):
    e, t0 = envelope(ch)
    floor = show(ch, e, t0)
    res[ch] = {"floor": floor,
               "bursts": [{**b} for b in bursts_and_stats(ch, e, t0, floor)]}

print("\n----- slot stats (round 1, auto regions) -----")
ea, ta = envelope("a")
eb, tb = envelope("b")
fa, fb = res["a"]["floor"], res["b"]["floor"]
# a: wind 区 55-62；b: wind 区 231.5-240 —— 细节窗口等看完包络后二次指定
(DEEP / "results" / "wind_energy_probe.json").write_text(json.dumps(res, indent=1))
print("saved wind_energy_probe.json")
