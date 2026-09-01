"""P5c wind 段切片解码：opening 后的 wind 段切出来让三引擎自由听写。
锚 = INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU（opening 已定案）。
wind 段紧随其后。输出 results/wind_slices/ + 解码 txt
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}
OUT = DEEP / "results" / "wind_slices"
OUT.mkdir(parents=True, exist_ok=True)


def load16k(p):
    import librosa
    wav, _ = librosa.load(p, sr=16000, mono=True)
    return wav


def free(m=None):
    if m is not None:
        del m
    gc.collect()
    torch.cuda.empty_cache()


# ---------- 定位并切片 ----------
sc = NLLScorer(str(TT / "models" / "whisper-large-v3-finetuned-for-ATC"))
centers = {}
for ch, apath in CH.items():
    t_open = sc.find_anchor_window(apath, "INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO")
    # opening 文本 ~4.5s 长，从窗起点算 wind 大约始于 t_open+5
    c = t_open + 7.0
    wav = load16k(apath)
    s0, s1 = int((c - 6) * 16000), int((c + 9) * 16000)
    import soundfile as sf
    sp = OUT / f"{ch}_windslice.wav"
    sf.write(sp, wav[s0:s1], 16000)
    centers[ch] = {"t_open": t_open, "slice_center": c, "path": str(sp)}
    print(f"[slice] {ch}: open@{t_open:.1f}s slice=[{c-6:.1f},{c+9:.1f}]s", flush=True)
free(sc)

# ---------- 三引擎解码 ----------
index = []


def save(name, text):
    (OUT / f"{name}.txt").write_text(text.strip())
    index.append({"id": name, "text": text.strip()})
    print(f"[saved] {name}: {text.strip()[:110]}", flush=True)


print("=== whisper judges (beam5) ===", flush=True)
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

for tag, mdir in [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
                  ("v3", "openai/whisper-large-v3")]:
    proc = WhisperProcessor.from_pretrained(mdir)
    wm = (WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float16)
          .to("cuda").eval())
    PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]]).cuda()
    for ch, info in centers.items():
        wav, _ = librosa.load(info["path"], sr=16000, mono=True)
        if len(wav) < 480000:
            wav = np.pad(wav, (0, 480000 - len(wav)))
        xf = proc.feature_extractor(wav, sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=64,
                              num_beams=5, do_sample=False)
        save(f"{ch}_windslice_{tag}", proc.tokenizer.decode(seq[0], skip_special_tokens=True))
    free(wm)

print("=== Qwen3-ASR ===", flush=True)
try:
    from qwen_asr import Qwen3ASRModel
    qm = Qwen3ASRModel.from_pretrained(
        "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
        device_map="cuda", dtype=torch.bfloat16, max_inference_batch_size=1)
    for ch, info in centers.items():
        r = qm.transcribe(audio=info["path"], language="English")
        save(f"{ch}_windslice_qwen", r[0].text)
    free(qm)
except Exception as e:
    print("Qwen failed:", e, flush=True)

(OUT / "wind_slice_index.json").write_text(
    json.dumps({"centers": centers, "decodes": index}, ensure_ascii=False, indent=1))
print("done")
