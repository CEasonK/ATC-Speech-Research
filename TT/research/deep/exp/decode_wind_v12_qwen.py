"""P5e-3 v12 重切片 Qwen3-ASR 解码（用 lingbot-map python 运行）。"""
import json
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
SL = DEEP / "results" / "wind_slices" / "v12"
SLICES = sorted(SL.glob("*.wav"))

index = []
from qwen_asr import Qwen3ASRModel

qm = Qwen3ASRModel.from_pretrained(
    "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
    device_map="cuda", dtype=torch.bfloat16, max_inference_batch_size=1)
for sp in SLICES:
    r = qm.transcribe(audio=str(sp), language="English")
    (SL / f"{sp.stem}_qwen.txt").write_text(r[0].text.strip())
    index.append({"id": f"{sp.stem}_qwen", "text": r[0].text.strip()})
    print(f"[saved] {sp.stem}_qwen: {r[0].text.strip()[:120]}", flush=True)

(SL / "decode_index_qwen.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
print("done")
