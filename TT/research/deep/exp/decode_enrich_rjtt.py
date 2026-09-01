"""RJTT 解码增强轮：v3 vanilla × {beam5 裸奔, beam5+ATC 提示词} 两模式补入候选池。
动机：语言审计用的是 greedy；弱段（seg00/02/04/05）值得用更强解码再挖一次，
且 ATC 提示词可把模型先验拉向管制词汇（prompt conditioning）。
提示词设计：只给台名/呼号风格词，不给任何具体争议词，避免污染裁决。
输出：条目 id = v3b_{seg}（裸 beam5）、v3p_{seg}（提示词 beam5），幂等追加进
results/decode_rjtt_index.json
"""
import json
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEEP = Path(__file__).resolve().parents[1]
SEG = DEEP / "segments" / "RJTT_CONTROL"
IDX = DEEP / "results" / "decode_rjtt_index.json"

INITIAL_PROMPT = (
    "Tokyo Control, Japan Air, FedEx, Shanghai Air, Aeroflot, "
    "climb maintain flight level, cleared direct, crossing restriction."
)


def load16k(apath):
    import librosa
    import numpy as np
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    return wav


def main():
    manifest = json.loads((SEG / "manifest.json").read_text())
    SEGS = {f"seg{k:02d}": m for k, m in enumerate(manifest)}
    idx = json.loads(IDX.read_text())
    have = {e["id"] for e in idx}

    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    tok = proc.tokenizer
    model = (WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", dtype=torch.float16).to("cuda").eval())

    sot = torch.tensor([[tok.convert_tokens_to_ids(t) for t in
                         ["<|startoftranscript|>", "<|en|>", "<|transcribe|>",
                          "<|notimestamps|>"]]]).cuda()
    # 提示词 token 拼在 SOT 之后（Whisper prompt conditioning 标准做法）
    pfx = tok.encode(" " + INITIAL_PROMPT, add_special_tokens=False)
    prompt_ids = torch.tensor([sot[0].tolist() + pfx]).cuda()

    new = []
    for s in SEGS:
        apath = str(SEG / f"seg_{s[3:]}.wav")
        xf = proc.feature_extractor(load16k(apath), sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        for tag, ids in [("v3b", sot), ("v3p", prompt_ids)]:
            key = f"{tag}_{s}"
            if key in have:
                continue
            with torch.no_grad():
                seq = model.generate(xf, decoder_input_ids=ids, max_new_tokens=200,
                                     num_beams=5, do_sample=False)
            txt = tok.decode(seq[0], skip_special_tokens=True).strip()
            idx.append({"id": key, "file": f"<generated:{key}>", "text": txt})
            new.append(key)
            print(f"[{key}] {txt[:100]}", flush=True)

    IDX.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    print(f"\nappended {len(new)} entries -> {IDX}")


if __name__ == "__main__":
    main()
