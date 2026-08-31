"""Qwen3-ASR 常驻旁证 worker（lingbot-map 环境专用进程）。

主进程（streaming 环境，无 qwen_asr）通过 stdin/stdout 与本进程通信：
  - 主进程每行发一个 wav 路径
  - 本进程回一行 JSON {"text": "..."} 或 {"text": "", "error": "..."}

用法：python qwen_worker.py <model_path>   （由 run_2pass.py --qwen_python 派生）
"""
import json
import sys

MODEL = sys.argv[1]

import torch  # noqa: E402

from qwen_asr import Qwen3ASRModel  # noqa: E402

m = Qwen3ASRModel.from_pretrained(
    MODEL, device_map="auto", dtype=torch.bfloat16,
    max_inference_batch_size=1)

# 预热：首次推理含 CUDA 内核编译/缓存（实测 30s+），放到 ready 之前，
# 避免计入主进程墙钟（R6 优化）
import io  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402

buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes((np.zeros(8000, dtype=np.int16)).tobytes())
buf.seek(0)
import tempfile  # noqa: E402

with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
    tf.write(buf.getvalue())
    tf.flush()
    try:
        m.transcribe(audio=tf.name, language="English")
    except Exception:  # noqa: BLE001
        pass

print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    try:
        txt = m.transcribe(audio=path, language="English")[0].text
        print(json.dumps({"text": txt}, ensure_ascii=False), flush=True)
    except Exception as ex:  # noqa: BLE001
        print(json.dumps({"text": "", "error": str(ex)}, ensure_ascii=False),
              flush=True)
