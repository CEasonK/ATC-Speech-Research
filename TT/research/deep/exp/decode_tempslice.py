"""P4f-2 温度段切片多引擎解码（依赖 adjudicate_v8_temp.py 产出的切片）。
短片段无长上下文干扰 → 各引擎只能靠这段发音，语言先验影响最小化。
引擎：whisper-atc / whisper-large-v3 / Qwen3-ASR。
输出：results/temp_slices/*.txt + tempslice_decode_index.json
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SL = DEEP / "results" / "temp_slices"
SLICES = sorted(SL.glob("*_tempslice.wav"))

index = []


def save(name, text):
    p = SL / f"{name}.txt"
    p.write_text(text.strip())
    index.append({"id": name, "text": text.strip()})
    print(f"[saved] {name}: {text.strip()[:120]}", flush=True)


def free(m=None):
    if m is not None:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def load16k(apath):
    import librosa
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    return wav


# ---------- whisper-atc + whisper-v3 ----------
print("=== whisper judges (beam5) ===", flush=True)
from transformers import WhisperForConditionalGeneration, WhisperProcessor

for tag, mdir in [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
                  ("v3", "openai/whisper-large-v3")]:
    proc = WhisperProcessor.from_pretrained(mdir)
    wm = (WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float16)
          .to("cuda").eval())
    PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]]).cuda()
    for sp in SLICES:
        xf = proc.feature_extractor(load16k(sp), sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=96,
                              num_beams=5, do_sample=False)
        save(f"{sp.stem}_{tag}", proc.tokenizer.decode(seq[0], skip_special_tokens=True))
    free(wm)
    free()

# ---------- Qwen3-ASR ----------
print("=== Qwen3-ASR ===", flush=True)
try:
    from qwen_asr import Qwen3ASRModel
    qm = Qwen3ASRModel.from_pretrained(
        "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
        device_map="cuda", dtype=torch.bfloat16, max_inference_batch_size=1)
    for sp in SLICES:
        r = qm.transcribe(audio=str(sp), language="English")
        save(f"{sp.stem}_qwen", r[0].text)
    free(qm)
except Exception as e:
    print("Qwen failed:", e, flush=True)

(SL / "tempslice_decode_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
print(f"done, {len(index)} transcripts")
