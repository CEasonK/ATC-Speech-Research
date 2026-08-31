"""offline whisper-large-v3 word-level timestamps + 与 deep 终稿对齐。

用 transformers 官方长音频 pipeline（内部 30s 滑窗拼接），word 级时间戳全局对齐。
产出 results/align_ref/<audio>.json：
  - whisper words [(w,start,end)]
  - 循环周期检测结果（ATIS 自相关）
  - 每个 deep 参考 token 的 [start,end]（与第一遍循环实例对齐）
用法：python align_ref.py <CYYT_ATIS_a|CYYT_ATIS_b>
"""
import json
import sys
from difflib import SequenceMatcher

import numpy as np

from common import RESULTS, load_audio, ref_text, save_json, tokens


def run_whisper_pipeline(audio_name):
    """返回 [(word, start, end)] 全局时间轴 + offline 全文。"""
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda",
    )
    x, sr = load_audio(audio_name)
    x = x[: int(125 * sr)]  # 只分析有效语音段（ATIS 循环在前 ~114s，后段为削波噪声，且省显存）
    out = pipe(
        {"array": x, "sampling_rate": sr},
        return_timestamps="word",
        chunk_length_s=30,
        batch_size=2,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    words = []
    for ch in out["chunks"]:
        s, e = ch["timestamp"]
        for w in ch["text"].split():
            words.append((w, float(s), float(e)))
    return words, out["text"]


def detect_cycle(words, dur):
    """用 'SAINT'（ATIS 开头词）出现时刻估计播报周期。"""
    starts = [s for w, s, e in words if tokens(w) and tokens(w)[0] == "saint"]
    # 合并距离过近的（<10s 视为同一处）
    uniq = []
    for t in sorted(starts):
        if not uniq or t - uniq[-1] > 10:
            uniq.append(t)
    periods = np.diff(uniq).tolist() if len(uniq) > 1 else []
    return {
        "saint_onsets": uniq,
        "periods": periods,
        "median_period": float(np.median(periods)) if periods else None,
        "dur": dur,
    }


def main():
    audio_name = sys.argv[1] if len(sys.argv) > 1 else "CYYT_ATIS_a"
    import soundfile as sf

    from common import AUDIO_FILES

    x, sr = sf.read(str(AUDIO_FILES[audio_name]), dtype="float32")
    dur = len(x) / sr
    ref = ref_text(audio_name)
    ref_toks = tokens(ref)

    words, full_text = run_whisper_pipeline(audio_name)
    cyc = detect_cycle(words, dur)
    print("SAINT onsets:", [round(t, 2) for t in cyc["saint_onsets"]])
    print("periods:", [round(p, 2) for p in cyc["periods"]])

    # ---- 将 whisper words 循环对齐到 ref×N ----
    # 取前 K 个循环（覆盖有语音段）：K = saint 出现次数
    K = max(len(cyc["saint_onsets"]), 1)
    exp_ref_toks = ref_toks * K
    w_norm = [tokens(w[0])[0] for w in words if tokens(w[0])]
    words_f = [w for w in words if tokens(w[0])]
    sm = SequenceMatcher(a=w_norm, b=exp_ref_toks, autojunk=False)
    # 参考第 k 遍的 token 时间
    L = len(ref_toks)
    rstarts = [[] for _ in range(K)]
    rends = [[] for _ in range(K)]
    matched_cnt = [[] for _ in range(K)]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                j = j1 + off
                k, idx = divmod(j, L)
                rstarts[k].append(words_f[i1 + off][1])
                rends[k].append(words_f[i1 + off][2])
                matched_cnt[k].append(idx)
    per_cycle = []
    for k in range(K):
        ok = matched_cnt[k]
        cov = len(set(ok)) / L if L else 0
        span = (min(rstarts[k]), max(rends[k])) if rstarts[k] else None
        per_cycle.append({"coverage": round(cov, 3), "span": span})
    print("per-cycle coverage of ref:", [p["coverage"] for p in per_cycle])

    # 选覆盖率最高的周期作为权威实例 → 给参考 token 定时间
    best_k = int(np.argmax([p["coverage"] for p in per_cycle]))
    tok_time = {}
    for j_idx, tk in zip(matched_cnt[best_k], rends[best_k]):
        tok_time.setdefault(j_idx, []).append(tk)
    ref_ends, ref_starts = [], []
    last_end = 0.0
    wtimes = [
        (words_f[i][1], words_f[i][2]) for i in range(len(words_f))
    ]
    sm_map = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                sm_map[j1 + off] = i1 + off
    base = best_k * L
    for idx in range(L):
        gi = sm_map.get(base + idx)
        if gi is not None:
            ref_starts.append(wtimes[gi][0])
            ref_ends.append(wtimes[gi][1])
            last_end = wtimes[gi][1]
        else:
            ref_starts.append(last_end)
            ref_ends.append(last_end)

    out = {
        "audio": audio_name,
        "dur": dur,
        "ref_text": ref,
        "ref_tokens": ref_toks,
        "best_cycle": best_k,
        "cycles": cyc,
        "per_cycle_coverage": per_cycle,
        "whisper_full_text": full_text,
        "ref_starts": ref_starts,
        "ref_ends": ref_ends,
    }
    p = save_json(out, RESULTS / "align_ref" / f"{audio_name}.json")
    # offline 单遍 WER：取最佳周期的 whisper 词 vs ref
    st = cyc["saint_onsets"]
    lo = st[best_k] if st and best_k < len(st) else 0
    hi = st[best_k + 1] if st and best_k + 1 < len(st) else dur
    seg_words = " ".join(w[0] for w in words_f if lo <= w[1] < hi)
    from metrics import wer

    print("offline(one-cycle) WER vs deep-final:", round(wer(seg_words, ref), 4))
    print("saved:", p)


if __name__ == "__main__":
    main()
