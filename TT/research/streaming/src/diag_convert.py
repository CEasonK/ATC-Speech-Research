"""转换正确性诊断：同一段音频、同一份 mel 特征，对比 HF 与 openai 格式模型的
encoder 输出与 decoder 首 token 分布。
用法：python diag_convert.py <hf_dir> <pt_path> <wav>
"""
import sys

sys.path.insert(
    0,
    "/siyuan/FunASR_extracted/FunASR-main/TT/research/refs/SimulStreaming-main/simulstreaming/whisper/simul_whisper",
)

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

import whisper as owhisper


def main():
    hf_dir, pt_path, wav = sys.argv[1], sys.argv[2], sys.argv[3]
    x, sr = sf.read(wav, dtype="float32")

    proc = AutoProcessor.from_pretrained(hf_dir)
    feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features.cuda().half()

    hf = (
        WhisperForConditionalGeneration.from_pretrained(hf_dir, dtype=torch.float16)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        enc_hf = hf.model.encoder(feats)[0]  # (1500, d)
        print("HF enc mean/std:", enc_hf.float().mean().item(), enc_hf.float().std().item())
        dec_prompt = proc.get_decoder_prompt_ids(language="en", task="transcribe")
        toks = [t for _, t in dec_prompt]
        ids = torch.tensor([toks]).cuda()
        hid_hf = hf.model.decoder(ids, encoder_hidden_states=enc_hf.unsqueeze(0)).last_hidden_state
        emb_hf = hf.model.decoder.embed_tokens.weight
        logits_hf = hid_hf.float() @ emb_hf.float().T
        print("HF dec logits mean:", logits_hf.mean().item())
        # 打印首步 top token
        tk = proc.tokenizer
        top = torch.topk(logits_hf[0, -1], 5)
        print("HF first-step top:", [(tk.decode([i]), round(float(p), 2)) for i, p in zip(top.indices.tolist(), top.values.tolist())])
        del hf
        torch.cuda.empty_cache()

    # openai format，直接用同一份 mel（HF 特征即 log-mel）
    m = owhisper.load_model(pt_path, device="cuda").half()
    mel = feats[0].float()  # (128, 3000)
    with torch.no_grad():
        enc_o = m.encoder(mel.unsqueeze(0).half())[0]
        print("openai enc mean/std:", enc_o.float().mean().item(), enc_o.float().std().item())
        diff = (enc_o.float() - enc_hf[0].float()).abs()
        print("enc |diff| mean=%.5f max=%.5f" % (diff.mean(), diff.max()))
        toks_o = torch.tensor([toks]).cuda()
        dec_out = m.decoder(toks_o, enc_o.unsqueeze(0))
        print("openai dec logits mean:", dec_out.float().mean().item())
        dd = (dec_out.float() - logits_hf[0].float()).abs()
        print("dec |diff| mean=%.5f max=%.5f" % (dd.mean(), dd.max()))
        top = torch.topk(dec_out.float()[0, -1], 5)
        otok = owhisper.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        print("openai first-step top:", [(otok.decode([i]), round(float(p), 2)) for i, p in zip(top.indices.tolist(), top.values.tolist())])


if __name__ == "__main__":
    main()
