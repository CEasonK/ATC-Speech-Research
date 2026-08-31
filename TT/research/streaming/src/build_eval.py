"""构造流式评测资产：
1. 从原音频切单循环音源（deep manifest 锚定窗口）
2. offline whisper-large-v3 对 <30s 单循环做 word timestamps → 参考发音时间轴 θ
3. 拼 K 遍生成合成评测音频 + eval_manifest.json（含逐 token 发音时刻）

产出（全部写入 streaming/results/eval_assets/，不触碰其他目录）。
用法：python build_eval.py
"""
import json
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf

from common import AUDIO_FILES, RESULTS, load_audio, load_ref, save_json, tokens

EVAL_ASSETS = RESULTS / "eval_assets"

# deep segments manifest 的锚定窗口（物理周期）
# a: 前 137s 内 4 个循环（正常语音电平），任选其一
# b: 前 224s 为恒定宽带噪声底（RMS≈223、CV≈0.10 无语音调制）；真实 ATIS 语音
#    仅在锚定最优窗 k=0（manifest nll=1.187）。
# 2026-08-27 修正（J10）：b 的 win 旧值 (229.8, 256.0) 长 26.2s 但标注 period=27.85s，
# 逐周期参考漂移 1.65s/3.30s/4.95s，且窗口尾部切在句末静音内（256.35-257.27s 的
# 周期间停顿被截掉）导致 K4b 拼接后无任何 >=0.75s 句末静音、VAD 全片只定稿一次。
# 现以真实句末停顿中点 257.65s 收窗：win=(229.8, 257.65) 长 27.85s == period，
# 每周期参考零漂移（a 口径 0.007s 同量级）。
CYCLES = {
    "CYYT_ATIS_a": {"period": 28.143, "win": (80.69, 108.84), "K": 4},
    "CYYT_ATIS_b": {"period": 27.85, "win": (229.8, 257.65), "K": 4},
}


def whisper_words_short(x, sr, model_path=None, initial_prompt=None):
    """<30s 片段的 word timestamps：faster-whisper（cross-attention DTW 对齐）。
    默认用 deep 项目已转好的 CT2 ATC 微调模型（streaming/downloads/_ct2_atc，
    从 research/deep 只读复制而来，并补了 preprocessor_config.json/num_mel_bins=128）。"""
    model_path = model_path or str(EVAL_ASSETS.parent.parent / "downloads" / "_ct2_atc")
    import tempfile

    from faster_whisper import WhisperModel

    m = WhisperModel(model_path, device="cuda", compute_type="float16")
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=str(EVAL_ASSETS)) as f:
        sf.write(f.name, x, sr)
        segs, _ = m.transcribe(
            f.name,
            language="en",
            word_timestamps=True,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
        )
        words = [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in s.words]
    del m
    return words


def align(words, ref):
    """whisper 词 ↔ 参考 token（双方都做 ICAO 数字归一化），返回 ref token (start,end) 与质量。"""
    from common import atoks

    ref_toks = tokens(ref)
    at_ref = [atoks(t)[0] if atoks(t) else t for t in ref_toks]
    # 把 whisper 词展开为 token 列表（"240"→[two,four,zero]，共享同一时间区间）
    pairs = []
    for w, s, e in words:
        for t in atoks(w):
            if t:
                pairs.append((t, s, e))
    w_norm = [p[0] for p in pairs]
    sm = SequenceMatcher(a=w_norm, b=at_ref, autojunk=False)
    starts = [0.0] * len(at_ref)
    ends = [0.0] * len(at_ref)
    n_eq = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            n_eq += i2 - i1
            for off in range(i2 - i1):
                starts[j1 + off] = pairs[i1 + off][1]
                ends[j1 + off] = pairs[i1 + off][2]
        elif tag == "insert":
            n_ins = j2 - j1
            t0 = pairs[i1][1] if 0 <= i1 < len(pairs) else 0.0
            t1 = pairs[i1][2] if 0 <= i1 < len(pairs) else (pairs[-1][2] if pairs else 1.0)
            for off in range(n_ins):
                f0, f1 = off / n_ins, (off + 1) / n_ins
                starts[j1 + off] = t0 + (t1 - t0) * f0
                ends[j1 + off] = t0 + (t1 - t0) * f1
        elif tag == "replace":
            t0 = pairs[i1][1] if 0 <= i1 < len(pairs) else 0.0
            t1 = pairs[i2 - 1][2] if 0 <= i2 - 1 < len(pairs) else (t0 + 0.3)
            n_ins = j2 - j1
            for off in range(n_ins):
                f0, f1 = off / n_ins, (off + 1) / n_ins
                starts[j1 + off] = t0 + (t1 - t0) * f0
                ends[j1 + off] = t0 + (t1 - t0) * f1
    quality = {
        "whisper_words": len(w_norm),
        "ref_tokens": len(at_ref),
        "matched": n_eq,
        "match_ratio": round(n_eq / max(len(at_ref), 1), 3),
    }
    heard = " ".join(t for t, _, _ in pairs)
    return starts, ends, quality, heard


def first_tok(s):
    from common import atoks

    t = atoks(s)
    return t[0] if t else "?"


def norm_ok(s):
    from common import atoks

    return bool(atoks(s))


def ref_body(name):
    """参考正文：deep 终稿文件的前 9 行（截至闭合句 INFORM ATC... 首次出现）。

    J12 核实：deep 终稿第 10-12 行"尾三行复诵"经时间轴重建（每周期 26.6s 连续
    语音 × 实测语速 2-2.9 tok/s ≈ 69 token，复诵 23 token 无处安放）+ 5 种解码
    配置零复诵产出 + deep 自身锚窗探针的"第二谷"实为下一周期开头
    （SAINT JOHNS INFORMATION FOXTROT 含锚词 INFORMATION FOXTROT），
    判定为 phantom——单周期物理上不含复诵。参考必须匹配模板窗实际内容。"""
    lines = [l.strip() for l in load_ref(name) if l.strip()]
    closing = "INFORM ATC THAT YOU HAVE INFORMATION FOXTROT"
    for i, l in enumerate(lines):
        if l.upper() == closing:
            return lines[: i + 1]
    return lines


def main():
    EVAL_ASSETS.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, cfg in CYCLES.items():
        x, sr = load_audio(name)
        t0, t1 = cfg["win"]
        seg = x[int(t0 * sr) : int(t1 * sr)]
        src_path = EVAL_ASSETS / f"{name}_cycle.wav"
        sf.write(str(src_path), seg, sr)

        ref_one = " ".join(ref_body(name))
        words = whisper_words_short(seg, sr)
        starts, ends, quality, whispertext = align(words, ref_one)

        K = cfg["K"]
        P = cfg["period"]
        # 拼接合成音频
        big = np.tile(seg, K)
        wav_path = EVAL_ASSETS / f"{name}_evalK{K}.wav"
        sf.write(str(wav_path), big, sr)

        rstarts, rends = [], []
        ref_toks = tokens(ref_one)
        for k in range(K):
            off = k * P
            rstarts += [s + off for s in starts]
            rends += [e + off for e in ends]
        man = {
            "audio": name,
            "cycle_win": cfg["win"],
            "period": P,
            "K": K,
            "dur_eval": round(len(big) / sr, 3),
            "src_wav": str(src_path),
            "eval_wav": str(wav_path),
            "ref_text_one_cycle": ref_one,
            "ref_tokens_K": ref_toks * K,
            "ref_starts": np.round(rstarts, 3).tolist(),
            "ref_ends": np.round(rends, 3).tolist(),
            "align_quality": quality,
            "whisper_words_one_cycle": whispertext,
        }
        p = save_json(man, EVAL_ASSETS / f"eval_manifest_{name}.json")
        print(f"[{name}] align match_ratio={quality['match_ratio']} "
              f"(whisper {quality['whisper_words']} vs ref {quality['ref_tokens']})")
        print("  whisper heard:", whispertext[:180])
        print("  saved:", p)


if __name__ == "__main__":
    main()
