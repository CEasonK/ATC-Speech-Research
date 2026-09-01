"""RJTT_CONTROL 9 段多引擎解码（此前从未解码过）。
引擎：whisper-atc(beam5) + Qwen3-ASR。SenseVoice 已证实失效，跳过。
输出：results/decode_rjtt/*.txt + decode_rjtt_index.json
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SEG = DEEP / "segments" / "RJTT_CONTROL"
OUT = DEEP / "results" / "decode_rjtt"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads((SEG / "manifest.json").read_text())
SEGS = [(f"seg{k:02d}", str(SEG / f"seg_{k:02d}.wav")) for k in range(len(manifest))]

index = []


def save(name, text):
    p = OUT / f"{name}.txt"
    p.write_text(text.strip())
    index.append({"id": name, "file": str(p), "text": text.strip()})
    print(f"[saved] {name}: {text.strip()[:100]}", flush=True)


def free(m=None):
    if m is not None:
        del m
    gc.collect()
    torch.cuda.empty_cache()


# ---------- whisper-atc ----------
print("=== whisper-atc (beam5) ===", flush=True)
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
proc = WhisperProcessor.from_pretrained(MODEL)
wm = (WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16)
      .to("cuda").eval())
PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]]).cuda()

for sname, apath in SEGS:
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    xf = feats.input_features.to("cuda", torch.float16)
    with torch.no_grad():
        seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=224,
                          num_beams=5, do_sample=False)
    save(f"atc_beam5_{sname}", proc.tokenizer.decode(seq[0], skip_special_tokens=True))
free(wm)
free()

# ---------- Qwen3-ASR ----------
print("=== Qwen3-ASR ===", flush=True)
try:
    from qwen_asr import Qwen3ASRModel
    qm = Qwen3ASRModel.from_pretrained(
        "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
        device_map="cuda", dtype=torch.bfloat16, max_inference_batch_size=1)
    for sname, apath in SEGS:
        r = qm.transcribe(audio=apath, language="English")
        save(f"qwen_{sname}", r[0].text)
    free(qm)
except Exception as e:
    print("Qwen failed:", e, flush=True)

(DEEP / "results" / "decode_rjtt_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
print(f"done, {len(index)} transcripts")
