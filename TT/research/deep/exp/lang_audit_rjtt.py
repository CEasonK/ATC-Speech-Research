"""RJTT 语言审计：9 段 × {auto / 强制英语 / 强制日语} 三模式 v3 解码对比。
动机：用户指出 FunASR 原始输出全是日语（实为片假名拼英语读音，但需排除真日语段）。
日本空域对日籍航班允许日语通话，此前管线全程强制 <|en|>，存在系统性盲区。
判据：
  1) auto 模式读出 v3 自己预测的语言 token
  2) 各模式平均 token logprob（greedy，compute_transition_scores）
  3) 文本合理性目检（日语 ATC 应产出可读日语管制用语，乱码则非）
输出 results/rjtt_lang_audit.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEEP = Path(__file__).resolve().parents[1]
SEG = DEEP / "segments" / "RJTT_CONTROL"
OUT = DEEP / "results" / "rjtt_lang_audit.json"


def load16k(apath):
    import librosa
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    return wav


def main():
    manifest = json.loads((SEG / "manifest.json").read_text())
    SEGS = {f"seg{k:02d}": m for k, m in enumerate(manifest)}

    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    tok = proc.tokenizer
    model = (WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", dtype=torch.float16).to("cuda").eval())

    def sot(lang):  # lang: None=auto, "en", "ja"
        ids = ["<|startoftranscript|>"]
        if lang:
            ids.append(f"<|{lang}|>")
        ids += ["<|transcribe|>", "<|notimestamps|>"]
        return torch.tensor([[tok.convert_tokens_to_ids(t) for t in ids]]).cuda()

    results = {}
    for s, m in SEGS.items():
        apath = str(SEG / f"seg_{s[3:]}.wav")
        xf = proc.feature_extractor(load16k(apath), sampling_rate=16000,
                                    return_tensors="pt").input_features.to("cuda", torch.float16)
        # 语言检测：标准 auto 流程——prompt 只给 <|startoftranscript|>，
        # 模型生成的第一个 token 即语言预测。（若 prompt 硬编码 <|transcribe|>，
        # 模型不会再输出语言 token，detected 恒空——2026-08-25 实测确认）
        with torch.no_grad():
            dseq = model.generate(
                xf, decoder_input_ids=torch.tensor(
                    [[tok.convert_tokens_to_ids("<|startoftranscript|>")]]).cuda(),
                max_new_tokens=1, num_beams=1, do_sample=False)
        dname = tok.convert_ids_to_tokens(int(dseq.reshape(-1)[-1]))
        det = dname[2:-2] if dname and dname.startswith("<|") and dname.endswith("|>") else None

        row = {"detected": det}
        for tag, lang in [("auto", None), ("en", "en"), ("ja", "ja")]:
            with torch.no_grad():
                seq = model.generate(
                    xf, decoder_input_ids=sot(lang), max_new_tokens=200,
                    num_beams=1, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
            txt = tok.decode(seq.sequences[0], skip_special_tokens=True).strip()
            # 平均 token logprob（只算正文 token）
            ts = model.compute_transition_scores(
                seq.sequences, seq.scores, normalize_logits=True)[0]
            n_gen = len(seq.scores)
            body = ts[-n_gen:] if ts.shape[0] > n_gen else ts  # 含提示时截尾
            avg_lp = float(body.mean())
            row[tag] = {"text": txt, "avg_logprob": round(avg_lp, 4),
                        "n_tokens": int(n_gen)}
        results[s] = row
        print(f"[{s}] det={row.get('detected')} "
              f"lp: auto={row['auto']['avg_logprob']:.3f} "
              f"en={row['en']['avg_logprob']:.3f} ja={row['ja']['avg_logprob']:.3f}",
              flush=True)
        print(f"    en : {row['en']['text'][:80]}", flush=True)
        print(f"    ja : {row['ja']['text'][:80]}", flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
