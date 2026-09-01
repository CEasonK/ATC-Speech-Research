"""P4f-3 温度切片时间戳解码：量出温度段各词的实际时长。
DECIMAL(3音节~0.9s) vs DEW(1音节~0.3s) 时长差异显著 → 直接定案。
输出词级时间轴到 stdout + results/temp_slices/timestamps.json
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SL = DEEP / "results" / "temp_slices"
SLICES = sorted(SL.glob("*_tempslice.wav"))

TS = re.compile(r"<\|([\d.]+)\|>([^<]*)")


def load16k(apath):
    import librosa
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    return wav


out = {}
from transformers import WhisperForConditionalGeneration, WhisperProcessor

for tag, mdir in [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
                  ("v3", "openai/whisper-large-v3")]:
    proc = WhisperProcessor.from_pretrained(mdir)
    wm = (WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float16)
          .to("cuda").eval())
    # 保留时间戳：去掉 <|notimestamps|>
    PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]]]).cuda()
    for sp in SLICES:
        xf = proc.feature_extractor(load16k(sp), sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=120,
                              num_beams=5, do_sample=False)
        raw = proc.tokenizer.decode(seq[0], skip_special_tokens=False)
        words = [(float(t), txt.strip()) for t, txt in TS.findall(raw) if txt.strip()]
        out[f"{sp.stem}_{tag}"] = words
        print(f"--- {sp.stem}_{tag} ---", flush=True)
        for t, txt in words:
            print(f"  {t:6.2f}s  {txt}", flush=True)
    del wm
    torch.cuda.empty_cache()

(SL / "timestamps.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("saved timestamps.json")
