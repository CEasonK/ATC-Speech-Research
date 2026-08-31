"""SimulStreaming (AlignAtt) 流式引擎封装。

模拟实时：按 chunk 把评测音频喂入，收集事件流与最终文本。
用法：python run_simulstreaming.py <eval_wav> <out_dir> [--model PATH] [--frame_threshold N]
      [--chunk SEC] [--beams N] [--static_prompt FILE] ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SSROOT = Path(
    "/siyuan/FunASR_extracted/FunASR-main/TT/research/refs/SimulStreaming-main"
)
sys.path.insert(0, str(SSROOT))

from simulstreaming.whisper.simul_whisper.config import AlignAttConfig
from simulstreaming.whisper.simul_whisper.simul_whisper import PaddedAlignAttWhisper
from simulstreaming.whisper.simul_whisper import simul_whisper as _swm
from simulstreaming.whisper.simul_whisper.whisper import _ALIGNMENT_HEADS

# ---- P3/S3: alignment heads 兜底注入（vendored 零改动）----
# 事实链：vendored load_model 的文件路径分支 alignment_heads=None
# (whisper/__init__.py L141) → Whisper 保留"后半层全部头"默认糙掩码
# (model.py L303-308；large-v3=16层×20头=320)，词级 most_attended_frames
# 从 320 糙头 argmax → 时间戳系统性劣化。官方调优头按 decoder 架构注入：
# (32,20)→large-v3（覆盖 ATC 微调，checkpoint dims 已实测），(4,20)→turbo。
# 注意 simul_whisper.py 以 `from .whisper import load_model` 把函数复制进
# 自己的命名空间，补丁必须打在 simul_whisper 模块上才拦得住。
_HEADS_BY_ARCH = {(32, 20): "large-v3", (4, 20): "large-v3-turbo"}

if not getattr(_swm.load_model, "_p3_tuned", False):
    _p3_orig_load = _swm.load_model

    def _load_model_tuned(name, **kw):
        m = _p3_orig_load(name, **kw)
        d = m.dims
        key = _HEADS_BY_ARCH.get((d.n_text_layer, d.n_text_head))
        if key is not None:
            # 注：load_model 已 .to(device)，故 set_alignment_heads 注册的
            # bool 掩码 buffer 落在 CPU。安全依据：simul_whisper 仅消费
            # alignment_heads.indices()（层/头号对，与 buffer 所在设备无关，
            # simul_whisper.py L92）；half() 亦不改 bool buffer。
            m.set_alignment_heads(_ALIGNMENT_HEADS[key])
            m._p3_heads_key = key
        else:
            m._p3_heads_key = "default"
            print(f"[simul][WARN] 未知架构 dims=({d.n_text_layer},"
                  f"{d.n_text_head})，保留默认 heads 掩码（不假装会映射）",
                  flush=True)
        return m

    _load_model_tuned._p3_tuned = True
    _swm.load_model = _load_model_tuned

ARGS = None


class Online:
    def __init__(self):
        # P3/S4: --static_prompt 语义是文件路径，但上游 from_text 把入参
        # 字符串本身当 prompt 内容(simul_whisper.py L156-157)。直接传路径
        # 会把路径 token 混进解码上下文，且不出现在输出文本里——不可见污染，
        # ON/OFF 对照作废。读内容传入；文件不存在 fail-fast。
        sp_text, self.sp_sha8 = None, None
        if ARGS.static_prompt:
            pf = Path(ARGS.static_prompt)
            if not pf.is_file():
                raise SystemExit(f"--static_prompt 文件不存在: {pf}"
                                 "（旧版会静默把路径字符串喂给解码器）")
            sp_text = pf.read_text().strip()
            import hashlib
            self.sp_sha8 = hashlib.sha256(sp_text.encode()).hexdigest()[:8]
        cfg = AlignAttConfig(
            model_path=ARGS.model,
            segment_length=ARGS.chunk,
            frame_threshold=ARGS.frame_threshold,
            language="en",
            audio_max_len=ARGS.audio_max_len,
            audio_min_len=ARGS.audio_min_len,
            cif_ckpt_path=None,
            decoder_type="greedy" if ARGS.beams == 1 else "beam",
            beam_size=ARGS.beams,
            task="transcribe",
            never_fire=True,
            init_prompt=None,
            max_context_tokens=None,
            static_init_prompt=sp_text,
            logdir=None,
        )
        self.asr = PaddedAlignAttWhisper(cfg)
        if ARGS.half:
            # 运行时半精度：权重转 fp16，并在 encoder 入口把 fp32 mel 转回 fp16
            self.asr.model = self.asr.model.half()
            _orig_enc_fwd = self.asr.model.encoder.forward

            def _enc_fp16(mel, _f=_orig_enc_fwd):
                return _f(mel.half() if mel.dtype == torch.float32 else mel)

            self.asr.model.encoder.forward = _enc_fp16
            # decoder 侧 token embedding lookup 自动保持 fp16；logits 需要 float 的位置已有 .float()
        self.chunks = []
        self.offset = 0.0  # 已插入音频总长
        self.buf_offset = 0.0

    def insert(self, seg):
        import torch

        self.chunks.append(torch.from_numpy(seg))

    def iter_once(self, is_last=False):
        import torch

        if not self.chunks:
            audio = None
        else:
            audio = torch.cat(self.chunks, dim=0)
            if audio.shape[0] == 0:
                audio = None
        self.chunks = []
        # P3/S1: 上游返回值=滑窗头部**被驱逐的秒数**（内部已 /16000，
        # simul_whisper.py L277/L285），不是新增样本数。旧代码再 /16000
        # 把秒当样本 → buf_offset 恒≈0，超窗长音频的词时间整体钉死在窗内。
        # P3/S2 顺序修正: 驱逐立即推进 buf_offset 之后再 infer——本次词帧
        # 轴相对的是驱逐后的窗；旧代码先算词后加偏移，差一个驱逐周期。
        # 前提（codex 质疑轮 2026-08-31 指出，vendored 侧不改）：
        # simul_whisper.py L277 的 removed_len 是赋值而非累加，"返回值=
        # 本轮驱逐总秒数"依赖每轮至多驱逐一段——当前成立（定长 chunk
        # 0.5s、入口不变式 segments_len<=audio_max_len、首轮 len>1 守卫）。
        # 改 chunk 大小/audio_max_len 或使用不等长 chunk 前必须重证，
        # 否则多段驱逐轮 buf_offset 偏小（词时间整体偏早）。
        self.buf_offset += self.asr.insert_audio(audio)
        toks, prog = self.asr.infer(is_last=is_last)
        # 词级输出（most attended frames → 发音时刻估计）
        out_words = []
        if toks and prog and "result" in prog:
            sw, st = prog["result"]["split_words"], prog["result"]["split_tokens"]
            frames = [p["most_attended_frames"][0] for p in prog["progress"]]
            i = 0
            for w, ts in zip(sw, st):
                fs = []
                while len(fs) < len(ts) and i < len(frames):
                    fs.append(frames[i])
                    i += 1
                if fs:
                    rel_s, rel_e = fs[0] * 0.02, fs[-1] * 0.02
                    out_words.append(
                        {"text": w,
                         "start": rel_s + self.buf_offset,
                         "end": rel_e + self.buf_offset,
                         "start_rel": rel_s, "end_rel": rel_e}
                    )
        # 第1个元素保留为空串占位（无 flush 语义），调用方按 (_, toks, words) 解包
        return "", toks, out_words


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("outdir")
    ap.add_argument("--model", default="../downloads/whisper-atc-openai.pt")
    ap.add_argument("--frame_threshold", type=int, default=25)
    ap.add_argument("--chunk", type=float, default=0.5)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--audio_max_len", type=float, default=30.0)
    ap.add_argument("--audio_min_len", type=float, default=0.25)
    ap.add_argument("--static_prompt", type=str, default=None)
    ap.add_argument("--half", action="store_true", help="fp16 权重运行")
    ARGS = ap.parse_args()

    # PaddedAlignAttWhisper 用 basename 去 load_model，需要在权重目录下运行
    import os

    x, sr = sf.read(ARGS.wav, dtype="float32")
    assert sr == 16000
    outdir = Path(ARGS.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(ARGS.model).resolve().parent)
    online = Online()

    events = []  # {"emit_t","word","att_start"}
    snapshots = []  # (t, prefix_text)
    prefix_words = []
    full_parts = []

    t_start = time.time()
    chunk_n = int(ARGS.chunk * sr)
    n_chunks = int(np.ceil(len(x) / chunk_n))
    for ci in range(n_chunks):
        seg = x[ci * chunk_n : (ci + 1) * chunk_n]
        t_end_of_chunk = (ci + 1) * ARGS.chunk  # 音频时刻（chunk 完全说出的时刻）
        online.insert(seg)
        _, toks, ws = online.iter_once()
        # decode toks
        txt = online.asr.tokenizer.decode(toks) if toks else ""
        if txt.strip():
            full_parts.append(txt)
            for w in ws:
                prefix_words.append(w)
                events.append({
                    "emit_audio_t": t_end_of_chunk,  # 算法层：该 chunk 处理完成的音频时刻
                    "word": w["text"],
                    "att_start": round(w["start"], 3),
                    "att_end": round(w["end"], 3),
                    "att_start_rel": round(w.get("start_rel", -1.0), 3),
                    "att_end_rel": round(w.get("end_rel", -1.0), 3),
                })
        snapshots.append((round(t_end_of_chunk, 2), [w["text"] for w in prefix_words]))
    o = online.iter_once(is_last=True)
    txt_last = online.asr.tokenizer.decode(o[1]) if o[1] else ""
    if txt_last.strip():
        full_parts.append(txt_last)
        for w in o[2]:
            prefix_words.append(w)
            events.append({"emit_audio_t": len(x) / sr, "word": w["text"],
                           "att_start": round(w["start"], 3), "att_end": round(w["end"], 3),
                           "att_start_rel": round(w.get("start_rel", -1.0), 3),
                           "att_end_rel": round(w.get("end_rel", -1.0), 3)})
    wall = time.time() - t_start

    final_text = " ".join(full_parts)
    with open(Path(ARGS.outdir) / "transcript_final.txt", "w") as f:
        f.write(final_text + "\n")
    with open(Path(ARGS.outdir) / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(Path(ARGS.outdir) / "snapshots.jsonl", "w") as f:
        for s in snapshots:
            f.write(json.dumps({"t": s[0], "words": s[1]}, ensure_ascii=False) + "\n")
    meta = {
        "engine": "simulstreaming",
        "model": ARGS.model,
        "frame_threshold": ARGS.frame_threshold,
        "chunk": ARGS.chunk,
        "beams": ARGS.beams,
        "rtf": round(wall / (len(x) / sr), 4),
        "wall_sec": round(wall, 2),
        "final_text": final_text,
        "att_time_semantics": ("att_start/att_end=绝对音频秒(滑窗驱逐偏移"
                               "+窗内帧轴)；att_*_rel=相对当前窗起点"),
        "alignment_heads": getattr(online.asr.model, "_p3_heads_key", "unknown"),
        "static_prompt_sha8": online.sp_sha8,
    }
    with open(Path(ARGS.outdir) / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("RTF:", meta["rtf"])
    print("FINAL:", final_text[:400])


if __name__ == "__main__":
    main()
