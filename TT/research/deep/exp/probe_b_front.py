"""解码 b 信道前半段实例，确认其内容（是否为其他班次/其他内容）。"""
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
MODEL = str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")
proc = WhisperProcessor.from_pretrained(MODEL)
model = (WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16)
         .to("cuda").eval())

for seg in ["inst_08", "inst_05", "inst_02", "inst_00"]:
    p = str(TT / "research/deep/segments/CYYT_ATIS_b" / f"{seg}.wav")
    import librosa
    wav, _ = librosa.load(p, sr=16000, mono=True)
    if len(wav) < 16000 * 30:
        import numpy as np
        wav = np.pad(wav, (0, 16000 * 30 - len(wav)))
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    ids = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                         ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]])
    with torch.no_grad():
        seq = model.generate(feats.input_features.to("cuda", torch.float16),
                             decoder_input_ids=ids.to("cuda"), max_new_tokens=224)
    text = proc.tokenizer.decode(seq[0], skip_special_tokens=True)
    print(f"== {seg}: {text}", flush=True)
