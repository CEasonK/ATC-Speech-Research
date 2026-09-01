"""P5e-2 v12 重切片 whisper 解码：atc + turbo_atcosim + v3(词级时间戳)。
turbo 是 ATIS 微调 → 若音频真有 WIND/AT 它最该听到；三中心投票破 LM 先验僵局。
"""
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
DEEP = Path(__file__).resolve().parents[1]
SL = DEEP / "results" / "wind_slices" / "v12"
SLICES = sorted(SL.glob("*.wav"))

TURBO_DIR = Path.home() / ".cache/huggingface/hub/models--tclin--whisper-large-v3-turbo-atcosim-finetune"
turbo_snaps = list((TURBO_DIR / "snapshots").glob("*")) if (TURBO_DIR / "snapshots").exists() else []
TURBO = next((str(s) for s in turbo_snaps if (s / "model.safetensors").exists()), None)
print(f"turbo dir: {TURBO}", flush=True)

index = []


def save(name, text):
    (SL / f"{name}.txt").write_text(text.strip())
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


TS_RE = re.compile(r"<\|[0-9]+\.[0-9]+\|>")


def decode_ts(proc, seq):
    """带时间戳 token 的可读解码（vanilla v3 支持）。"""
    out = []
    for t in seq[0].tolist():
        s = proc.tokenizer.decode([t])
        st = s.strip()
        if TS_RE.fullmatch(st):
            out.append(st)
        elif not (st.startswith("<|") and st.endswith("|>")):
            out.append(s)
    return " ".join(out)


from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODELS = [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC"), False)]
if TURBO:
    MODELS.append(("turbo", TURBO, False))
MODELS.append(("v3ts", "openai/whisper-large-v3", True))

for tag, mdir, want_ts in MODELS:
    print(f"=== {tag} ({mdir}) ===", flush=True)
    proc = WhisperProcessor.from_pretrained(mdir)
    wm = (WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float16)
          .to("cuda").eval())
    PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]]).cuda()
    for sp in SLICES:
        xf = proc.feature_extractor(load16k(str(sp)), sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            if want_ts:
                # 去掉 <|notimestamps|>，允许时间戳 token 输出
                seq = wm.generate(xf, decoder_input_ids=PROMPT[:, :3],
                                  max_new_tokens=96, num_beams=5, do_sample=False,
                                  return_timestamps=True)
                save(f"{sp.stem}_{tag}", decode_ts(proc, seq))
            else:
                seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=96,
                                  num_beams=5, do_sample=False)
                save(f"{sp.stem}_{tag}", proc.tokenizer.decode(seq[0], skip_special_tokens=True))
    free(wm)
    free()

(SL / "decode_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
print("done")
