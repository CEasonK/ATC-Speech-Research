"""b 信道专用切分：锚定外推 + NLL 质量验证。
b 前半段无可靠静音，但尾部两真间隙（228.1、256.0，间隔 27.85s）给出本地周期。
策略：T=27.85 从锚点向两侧外切实例；每实例用报文文本 NLL 验证完整性。
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer  # noqa: E402

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SR = 16000

name = "CYYT_ATIS_b"
wav, _ = librosa.load(str(TT / "audio" / f"{name}.wav"), sr=SR, mono=True)
dur = len(wav) / SR

T = 27.85
ANCHOR_START, ANCHOR_END = 229.8, 256.0   # 真实报文实例（有声区）
MSG = ("INFORM ATC THAT YOU HAVE INFORMATION FOXTROT INFORMATION FOXTROT "
       "WEATHER AT ZERO TWO ZERO ZERO ZULU WIND TWO FOUR ZERO AT FIVE "
       "VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE ONE DECIMAL "
       "NINER ALTITUDE THREE ZERO TWO THREE APPROACH RNAV ZULU RUNWAY TWO "
       "EIGHT INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE "
       "FIVE AS REQUESTED APPROACH ON INITIAL CONTACT LANDING AND DEPARTING")

insts = []
k = 0
while True:
    t1 = ANCHOR_END - k * T
    t0 = ANCHOR_START - k * T
    if t0 < 0:
        break
    insts.append({"k": k, "t0": round(max(0, t0), 2), "t1": round(min(dur, t1), 2)})
    k += 1
insts.sort(key=lambda x: x["t0"])

sc = NLLScorer(str(TT / "models" / "whisper-large-v3-finetuned-for-ATC"))
outdir = DEEP / "segments" / name
outdir.mkdir(parents=True, exist_ok=True)

for it in insts:
    s0, e0 = int(it["t0"] * SR), int(it["t1"] * SR)
    seg_path = str(outdir / f"inst_{it['k']:02d}.wav")
    sf.write(seg_path, wav[s0:e0], SR)
    r = sc.score(seg_path, MSG)
    it.update(nll=round(r["score"], 4), t_argmin=r["t_start"])
    print(f"inst_{it['k']:02d} [{it['t0']:.1f}-{it['t1']:.1f}] NLL={r['score']:.3f} argmin@{r['t_start']:.0f}", flush=True)

(outdir / "manifest.json").write_text(json.dumps(
    {"period": T, "anchor": [ANCHOR_START, ANCHOR_END], "instances": insts}, indent=1))
print("saved manifest")
