"""P5 前置：多引擎 × 有效实例批量重解码（候选池 v2 的原料）。

有效实例（NLL 验证过）：
  a: inst_01..04（24.4-137.0s 的 4 个循环）
  b: inst_00（229.8-256.0s 唯一完整实例）
引擎：whisper-atc(4 变体) / Qwen3-ASR / SenseVoiceSmall
输出：results/decode_v2/*.txt + decode_v2_index.json
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SEG = DEEP / "segments"
OUT = DEEP / "results" / "decode_v2"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCES = [
    ("a_inst01", str(SEG / "CYYT_ATIS_a" / "inst_01.wav")),
    ("a_inst02", str(SEG / "CYYT_ATIS_a" / "inst_02.wav")),
    ("a_inst03", str(SEG / "CYYT_ATIS_a" / "inst_03.wav")),
    ("a_inst04", str(SEG / "CYYT_ATIS_a" / "inst_04.wav")),
    ("b_inst00", str(SEG / "CYYT_ATIS_b" / "inst_00.wav")),
]

index = []


def save(name, text):
    p = OUT / f"{name}.txt"
    p.write_text(text.strip())
    index.append({"id": name, "file": str(p), "text": text.strip()})
    print(f"[saved] {name}: {text.strip()[:90]}", flush=True)


def free(m=None):
    if m is not None:
        del m
    gc.collect()
    torch.cuda.empty_cache()


# ---------- 引擎 1：whisper-atc 4 变体 ----------
print("=== whisper-atc ===", flush=True)
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
proc = WhisperProcessor.from_pretrained(MODEL)
wm = (WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16)
      .to("cuda").eval())
PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]]).cuda()

VARIANTS = [
    ("greedy", dict(num_beams=1, do_sample=False)),
    ("beam5", dict(num_beams=5, do_sample=False)),
    ("beam8", dict(num_beams=8, do_sample=False)),
    ("beam5_t02", dict(num_beams=5, do_sample=True, temperature=0.2, top_k=0)),
]

for iname, apath in INSTANCES:
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    xf = feats.input_features.to("cuda", torch.float16)
    for vname, kw in VARIANTS:
        with torch.no_grad():
            seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=224, **kw)
        save(f"atc_{vname}_{iname}", proc.tokenizer.decode(seq[0], skip_special_tokens=True))
free(wm)
free()

# ---------- 引擎 2：Qwen3-ASR ----------
print("=== Qwen3-ASR ===", flush=True)
try:
    from qwen_asr import Qwen3ASRModel
    qm = Qwen3ASRModel.from_pretrained(
        "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
        device_map="cuda", dtype=torch.bfloat16, max_inference_batch_size=1)
    for iname, apath in INSTANCES:
        r = qm.transcribe(audio=apath, language="English")
        save(f"qwen_{iname}", r[0].text)
    free(qm)
except Exception as e:
    print("Qwen failed:", e, flush=True)

# ---------- 引擎 3：SenseVoiceSmall ----------
print("=== SenseVoice ===", flush=True)
try:
    from funasr import AutoModel
    sm = AutoModel(
        model="/home/w-main/.cache/modelscope/hub/models/iic/SenseVoiceSmall",
        trust_remote_code=True, device="cuda:0")
    for iname, apath in INSTANCES:
        r = sm.generate(input=apath, cache={}, language="en", use_itn=True,
                        batch_size_s=60, merge_vad=False)
        save(f"sv_{iname}", r[0]["text"])
    free(sm)
except Exception as e:
    print("SenseVoice failed:", e, flush=True)

(DEEP / "results" / "decode_v2_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
print(f"done, {len(index)} transcripts")
