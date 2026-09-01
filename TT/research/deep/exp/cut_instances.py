"""P3：相位对齐整周期切分。
用静音间隙的线性拟合求精确周期 T 与相位 φ，切出每个完整报文实例。
输出：research/deep/segments/{audio}/inst_XX.wav + manifest.json
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SR = 16000
OUT = DEEP / "segments"


def gaps_of(wav, min_gap_s=0.8, pct=40):
    hop_s = 0.02
    hop = int(hop_s * SR)
    n = len(wav) // hop
    e = np.array([np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                  for i in range(n)])
    thr = np.percentile(e, pct) * 0.5
    voiced = e > thr
    gaps, start = [], None
    for i, v in enumerate(voiced):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if (i - start) * hop_s >= min_gap_s:
                gaps.append((start * hop_s, (i - start) * hop_s))
            start = None
    if start is not None and (n - start) * hop_s >= min_gap_s:
        gaps.append((start * hop_s, (n - start) * hop_s))
    return gaps


def fit_period(gaps):
    """间隙中心线性拟合：center_k = phi + k*T"""
    centers = np.array([g + d / 2 for g, d in gaps])
    k = np.arange(len(centers))
    T, phi = np.polyfit(k, centers, 1)   # center = phi + k*T
    resid = centers - (phi + k * T)
    return T, phi, float(np.max(np.abs(resid)))


def cut_instances(name, wav, T, phi, dur):
    """以 phi 为周期末尾（静音中心），实例 k 覆盖 [phi+kT-T/2, phi+kT+T/2) 内的有声区。
    更直接：实例 k 起点 = phi + k*T - T + gap_half，终点 = phi + k*T + gap_half。
    即把静音中心放在两实例之间。"""
    insts = []
    half_gap = 0.9  # 静音半宽余量
    k = 0
    while True:
        end = phi + k * T + half_gap          # 本实例结束（静音中心前）
        start = end - T                        # 上个静音中心后
        k += 1
        if end > dur + 1:
            break
        if start < -1:
            continue                           # 首个不完整实例跳过
        s0, e0 = max(0, int(start * SR)), min(len(wav), int(end * SR))
        if e0 - s0 > SR:                       # 至少 1s
            insts.append({"k": k - 1, "t0": round(s0 / SR, 2), "t1": round(e0 / SR, 2)})
    return insts


for name, gap_kw in [("CYYT_ATIS_a", dict(min_gap_s=0.8)),
                     ("CYYT_ATIS_b", dict(min_gap_s=0.4, pct=55))]:
    wav, _ = librosa.load(str(TT / "audio" / f"{name}.wav"), sr=SR, mono=True)
    dur = len(wav) / SR
    gaps = gaps_of(wav, **gap_kw)
    print(f"== {name} dur={dur:.1f}s gaps={[(round(g,1),round(d,1)) for g,d in gaps]}")
    if len(gaps) < 2:
        print("   !! 间隙不足，跳过")
        continue
    T, phi, resid = fit_period(gaps)
    print(f"   fit: T={T:.3f}s phi={phi:.2f}s max_resid={resid:.2f}s")
    insts = cut_instances(name, wav, T, phi, dur)
    outdir = OUT / name
    outdir.mkdir(parents=True, exist_ok=True)
    for it in insts:
        s0 = int(it["t0"] * SR)
        e0 = int(it["t1"] * SR)
        sf.write(str(outdir / f"inst_{it['k']:02d}.wav"), wav[s0:e0], SR)
    (outdir / "manifest.json").write_text(json.dumps(
        {"period": T, "phase": phi, "max_resid": resid,
         "instances": insts}, indent=1))
    print(f"   saved {len(insts)} instances -> {outdir}")
