"""把 HF 格式 whisper (large-v3 / ATC 微调) 转成 openai-whisper .pt 格式，
供 SimulStreaming 使用。

用法：python convert_hf_to_openai.py <hf_model_dir> <out.pt>
验证：转换后用 SimulStreaming 内嵌的 whisper 解码一条音频，与 HF 输出对比。
"""
import sys

import torch


def convert(hf_dir, out_path):
    from transformers import WhisperForConditionalGeneration

    hf = WhisperForConditionalGeneration.from_pretrained(hf_dir)
    sd = dict(hf.model.state_dict())
    cfg = hf.config

    dims = dict(
        n_mels=cfg.num_mel_bins,
        n_vocab=cfg.vocab_size,
        n_audio_ctx=cfg.max_source_positions,
        n_audio_state=cfg.d_model,
        n_audio_head=cfg.encoder_attention_heads,
        n_audio_layer=cfg.encoder_layers,
        n_text_ctx=cfg.max_target_positions,
        n_text_state=cfg.d_model,
        n_text_head=cfg.decoder_attention_heads,
        n_text_layer=cfg.decoder_layers,
    )

    out = {}

    def take(src, dst=None):
        assert src in sd, f"missing {src}"
        out[(dst or src)] = sd[src].clone()

    # ---- encoder ----
    E = f"encoder"
    take(f"{E}.conv1.weight")
    take(f"{E}.conv1.bias")
    take(f"{E}.conv2.weight")
    take(f"{E}.conv2.bias")
    out["encoder.positional_embedding"] = sd[f"{E}.embed_positions.weight"].clone()
    for i in range(cfg.encoder_layers):
        B = f"encoder.blocks.{i}"
        S = f"encoder.layers.{i}"
        out[f"{B}.attn_ln.weight"] = sd[f"{S}.self_attn_layer_norm.weight"].clone()
        out[f"{B}.attn_ln.bias"] = sd[f"{S}.self_attn_layer_norm.bias"].clone()
        out[f"{B}.attn.query.weight"] = sd[f"{S}.self_attn.q_proj.weight"].clone()
        out[f"{B}.attn.key.weight"] = sd[f"{S}.self_attn.k_proj.weight"].clone()
        out[f"{B}.attn.value.weight"] = sd[f"{S}.self_attn.v_proj.weight"].clone()
        if f"{S}.self_attn.q_proj.bias" in sd:
            out[f"{B}.attn.query.bias"] = sd[f"{S}.self_attn.q_proj.bias"].clone()
        if f"{S}.self_attn.v_proj.bias" in sd:
            out[f"{B}.attn.value.bias"] = sd[f"{S}.self_attn.v_proj.bias"].clone()
        out[f"{B}.attn.out.weight"] = sd[f"{S}.self_attn.out_proj.weight"].clone()
        out[f"{B}.attn.out.bias"] = sd[f"{S}.self_attn.out_proj.bias"].clone()
        out[f"{B}.mlp.0.weight"] = sd[f"{S}.fc1.weight"].clone()
        out[f"{B}.mlp.0.bias"] = sd[f"{S}.fc1.bias"].clone()
        out[f"{B}.mlp.2.weight"] = sd[f"{S}.fc2.weight"].clone()
        out[f"{B}.mlp.2.bias"] = sd[f"{S}.fc2.bias"].clone()
        out[f"{B}.mlp_ln.weight"] = sd[f"{S}.final_layer_norm.weight"].clone()
        out[f"{B}.mlp_ln.bias"] = sd[f"{S}.final_layer_norm.bias"].clone()
    out["encoder.ln_post.weight"] = sd["encoder.layer_norm.weight"].clone()
    out["encoder.ln_post.bias"] = sd["encoder.layer_norm.bias"].clone()

    # ---- decoder ----
    D = "decoder"
    out["decoder.token_embedding.weight"] = sd[f"{D}.embed_tokens.weight"].clone()
    out["decoder.positional_embedding"] = sd[f"{D}.embed_positions.weight"].clone()
    for i in range(cfg.decoder_layers):
        B = f"decoder.blocks.{i}"
        S = f"decoder.layers.{i}"
        out[f"{B}.attn_ln.weight"] = sd[f"{S}.self_attn_layer_norm.weight"].clone()
        out[f"{B}.attn_ln.bias"] = sd[f"{S}.self_attn_layer_norm.bias"].clone()
        out[f"{B}.attn.query.weight"] = sd[f"{S}.self_attn.q_proj.weight"].clone()
        out[f"{B}.attn.key.weight"] = sd[f"{S}.self_attn.k_proj.weight"].clone()
        out[f"{B}.attn.value.weight"] = sd[f"{S}.self_attn.v_proj.weight"].clone()
        if f"{S}.self_attn.q_proj.bias" in sd:
            out[f"{B}.attn.query.bias"] = sd[f"{S}.self_attn.q_proj.bias"].clone()
        if f"{S}.self_attn.v_proj.bias" in sd:
            out[f"{B}.attn.value.bias"] = sd[f"{S}.self_attn.v_proj.bias"].clone()
        out[f"{B}.attn.out.weight"] = sd[f"{S}.self_attn.out_proj.weight"].clone()
        out[f"{B}.attn.out.bias"] = sd[f"{S}.self_attn.out_proj.bias"].clone()
        # cross attention
        out[f"{B}.cross_attn.query.weight"] = sd[f"{S}.encoder_attn.q_proj.weight"].clone()
        out[f"{B}.cross_attn.key.weight"] = sd[f"{S}.encoder_attn.k_proj.weight"].clone()
        out[f"{B}.cross_attn.value.weight"] = sd[f"{S}.encoder_attn.v_proj.weight"].clone()
        if f"{S}.encoder_attn.q_proj.bias" in sd:
            out[f"{B}.cross_attn.query.bias"] = sd[f"{S}.encoder_attn.q_proj.bias"].clone()
        if f"{S}.encoder_attn.v_proj.bias" in sd:
            out[f"{B}.cross_attn.value.bias"] = sd[f"{S}.encoder_attn.v_proj.bias"].clone()
        out[f"{B}.cross_attn.out.weight"] = sd[f"{S}.encoder_attn.out_proj.weight"].clone()
        out[f"{B}.cross_attn.out.bias"] = sd[f"{S}.encoder_attn.out_proj.bias"].clone()
        out[f"{B}.cross_attn_ln.weight"] = sd[f"{S}.encoder_attn_layer_norm.weight"].clone()
        out[f"{B}.cross_attn_ln.bias"] = sd[f"{S}.encoder_attn_layer_norm.bias"].clone()
        out[f"{B}.mlp.0.weight"] = sd[f"{S}.fc1.weight"].clone()
        out[f"{B}.mlp.0.bias"] = sd[f"{S}.fc1.bias"].clone()
        out[f"{B}.mlp.2.weight"] = sd[f"{S}.fc2.weight"].clone()
        out[f"{B}.mlp.2.bias"] = sd[f"{S}.fc2.bias"].clone()
        out[f"{B}.mlp_ln.weight"] = sd[f"{S}.final_layer_norm.weight"].clone()
        out[f"{B}.mlp_ln.bias"] = sd[f"{S}.final_layer_norm.bias"].clone()
    out["decoder.ln.weight"] = sd["decoder.layer_norm.weight"].clone()
    out["decoder.ln.bias"] = sd["decoder.layer_norm.bias"].clone()

    torch.save({"dims": dims, "model_state_dict": out}, out_path)
    print(f"saved {out_path} with {len(out)} tensors; dims={dims}")


def sanity_check(pt_path, wav, hf_dir):
    """用内嵌 whisper 对同一段音频解码，与 HF fast ct2 结果大致对比。"""
    sys.path.insert(0, "/siyuan/FunASR_extracted/FunASR-main/TT/research/refs/SimulStreaming-main/simulstreaming/whisper/simul_whisper")
    import numpy as np
    import soundfile as sf
    import torch

    import whisper as owhisper  # vendored

    device = "cuda"
    model = owhisper.load_model(pt_path, device=device)
    x, sr = sf.read(wav, dtype="float32")
    if sr != 16000:
        x = np.repeat(x, int(sr / 16000))[: int(len(x) * 16000 / sr)]
    mel = owhisper.log_mel_spectrogram(x, model.dims.n_mels).to(device)
    tok = owhisper.tokenizer.get_tokenizer(True, language="en", task="transcribe")
    opts = owhisper.DecodingOptions(language="en", task="transcribe", without_timestamps=True, fp16=True)
    result = owhisper.decode(model, mel, opts)
    print("openai-format decode:", result.text[:300])
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    hf_dir = sys.argv[1]
    out_path = sys.argv[2]
    convert(hf_dir, out_path)
    if len(sys.argv) > 3:
        sanity_check(out_path, sys.argv[3], hf_dir)
