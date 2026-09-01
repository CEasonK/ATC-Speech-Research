"""RJTT_CONTROL 分段：管制对话非循环广播，按 >=1s 静音切通话段。
输出 segments/RJTT_CONTROL/seg_XX.wav + manifest.json
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SR = 16000

wav, _ = librosa.load(str(TT / "audio" / "RJTT_CONTROL.wav"), sr=SR, mono=True)
dur = len(wav) / SR

hop_s = 0.02
hop = int(hop_s * SR)
n = len(wav) // hop
e = np.array([np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2) + 1e-12)
              for i in range(n)])
thr = np.percentile(e, 35) * 0.5
voiced = e > thr

segs, start = [], None
for i, v in enumerate(voiced):
    if v and start is None:
        start = i
    elif not v and start is not None:
        if (i - start) * hop_s >= 1.0:
            segs.append((start * hop_s, i * hop_s))
        start = None
if start is not None and (n - start) * hop_s >= 1.0:
    segs.append((start * hop_s, n * hop_s))

# 合并间隔 <0.8s 的相邻段（呼吸停顿）
merged = []
for s in segs:
    if merged and s[0] - merged[-1][1] < 0.8:
        merged[-1] = (merged[-1][0], s[1])
    else:
        merged.append(list(s))

outdir = DEEP / "segments" / "RJTT_CONTROL"
outdir.mkdir(parents=True, exist_ok=True)
man = []
for k, (s, t) in enumerate(merged):
    s0, e0 = int(s * SR), int(t * SR)
    sf.write(str(outdir / f"seg_{k:02d}.wav"), wav[s0:e0], SR)
    man.append({"k": k, "t0": round(s, 2), "t1": round(t, 2), "dur": round(t - s, 2)})
    print(f"seg_{k:02d} [{s:.1f}-{t:.1f}] {t - s:.1f}s")

(outdir / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"{len(man)} segments saved")
