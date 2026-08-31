"""跨周期共识精修：ATIS 通播循环播发 N 遍，多遍识别结果按周期对齐投票，
消除单遍幻觉/漂移（如 "box drop"、数字改写），输出去重后的单篇终稿。

原理（客观、确定性，无主观打分）：
1. 能量包络自相关估计循环周期 T；
2. 用词级时间戳(att_start)把识别词分到各周期窗口 -> N 遍独立转写；
3. 以第一遍为骨架，其余遍 pairwise 对齐投票：某 token 至少 max(2,N/2) 遍一致
   才保留；纯骨架独有且无人支持的 token 视为幻觉删除；多点一致的新增片段并入。

用法：python consensus_refine.py <exp_dir> <wav> <out_prefix> [--period auto]
输出：<out_prefix>_consensus.txt、<out_prefix>_cycles.json（每遍文本+共识对照）
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf


def estimate_period(x, sr, lo=15.0, hi=45.0):
    """能量包络自相关求循环周期（秒）。"""
    win = int(sr * 0.01)
    n = len(x) // win
    env = np.sqrt(np.mean(np.abs(x[: n * win]).reshape(n, win) ** 2, axis=1))
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[n - 1:]
    ac /= (ac[0] + 1e-12)
    lo_i, hi_i = int(lo * 100), int(hi * 100)
    k = lo_i + int(np.argmax(ac[lo_i:hi_i]))
    k, ac = int(k), [float(v) for v in ac]
    # 抛物线插值细化峰位
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        return round((k + d) / 100.0, 3), round(float(b), 3)
    return round(k / 100.0, 3), round(float(ac[k]), 3)


def read_words(exp_dir):
    words = []
    with open(Path(exp_dir) / "events.jsonl") as f:
        for ln in f:
            e = json.loads(ln)
            w = e.get("word", "").strip()
            s = e.get("att_start")
            if w and s is not None:
                words.append((float(s), w))
    words.sort()
    return words


def speech_bounds(x, sr, thr_rel=2.5, min_gap=0.45):
    """能量 VAD：返回静音间隙中点列表（秒，递增）与语音 runs。"""
    win = int(sr * 0.01)
    n = len(x) // win
    rms = np.sqrt(np.mean(np.abs(x[: n * win]).reshape(n, win) ** 2, axis=1))
    thr = max(np.percentile(rms, 20) * thr_rel, rms.max() * 0.03)
    sp = rms > thr
    gaps = []
    run_start = None
    gap_len = 0
    for i, s in enumerate(sp):
        if not s:
            if run_start is not None:
                gap_len += 1
                if gap_len * 0.01 >= min_gap:
                    gaps.append((i - gap_len / 2) * 0.01)
                    run_start = None
                    gap_len = 0
        else:
            if run_start is None:
                run_start = i
            gap_len = 0
    return gaps


def cycle_cutpoints(x, sr, T, tol=0.8):
    """以自相关周期 T 为固定步长铺满全轴，但相位由'落点附近存在静音间隙'
    的支持票数决定：得票最多的相位边界即真实循环切割点集合。"""
    gaps = [g for g in speech_bounds(x, sr) if g > 0.6]
    dur = len(x) / sr
    best_t0, best_score = None, -1
    cands = [g % T for g in gaps]
    for t0 in sorted(set(round(c, 2) for c in cands)):
        n_hits = 0
        t = t0
        while t < dur:
            if any(abs(t - g) < tol for g in gaps):
                n_hits += 1
            t += T
        # 均匀覆盖优先：命中数/期望数
        expect = dur / T
        score = (n_hits + 1) / (expect + 1)
        if score > best_score:
            best_score, best_t0 = score, t0
    cuts = []
    t = best_t0
    while t < dur:
        cuts.append(round(t, 3))
        t += T
    return [c for c in cuts if 0.5 < c]


def split_cycles(words, cuts, dur):
    """按周期边界 cuts 分桶。返回每桶 token 列表（保留出现顺序）。"""
    edges = [0.0] + list(cuts) + [dur + 1.0]
    buckets = defaultdict(list)
    for s, w in words:
        k = 0
        for i in range(len(edges) - 1):
            if edges[i] <= s < edges[i + 1]:
                k = i
                break
        else:
            continue
        buckets[k].append(w)
    return [buckets.get(k, []) for k in range(len(edges) - 1)]


def consensus(cycles):
    """以最长遍为骨架做多遍对齐投票，返回共识 token 列表。

    规则（确定性）：
    - 接近空的"残尾周期"（token 数 < 最长遍的 30%）不参与投票分母，
      否则会抬高 need_keep 造成本应保留的词被误删；
    - 覆盖数 covered[j] = 除骨架外与骨架位 j 对齐成功的遍数；
    - 一致数 votes[j]  = 其中给出相同 token 的遍数；
    - need_keep = max(2, ceil(n/2))（n=完整周期数）：一致数达标 -> 保留该 token；
    - 一致数为 0 且覆盖数 >= need_keep（多遍都认为此处是别的内容）-> 删除骨架词；
    - 其余情况保留骨架词（首轮音频通常最干净）。
    """
    cyc = [c for c in cycles if c]
    if not cyc:
        return [], {"mode": "empty"}
    if len(cyc) == 1:
        return cyc[0], {"mode": "single-pass"}
    skel = max(cyc, key=len)
    # 残尾周期过滤：只保留 >= 30% 骨架长度的周期参与投票
    min_len = max(1, int(len(skel) * 0.3))
    full = [c for c in cyc if len(c) >= min_len]
    tail_dropped = [c for c in cyc if c is not skel and len(c) < min_len]
    n_full = len(full)
    votes = Counter()
    covered = np.zeros(len(skel), dtype=int)

    for p in full:
        if p is skel:
            continue
        sm = SequenceMatcher(a=skel, b=p, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for off in range(i2 - i1):
                    votes[i1 + off] += 1
                    covered[i1 + off] += 1
            elif tag == "replace":
                # 对齐区间内的逐位置弱支持（帮助判断骨架独有词）
                for off in range(min(i2 - i1, j2 - j1)):
                    covered[i1 + off] += 1

    need_keep = max(2, math.ceil(n_full * 0.5))
    out = []
    dropped = []
    for j, tok in enumerate(skel):
        if votes[j] >= need_keep - 1:      # 含骨架自身的一票
            out.append(tok)
        elif votes[j] == 0 and covered[j] >= need_keep:
            dropped.append(tok)            # 多遍一致反对 -> 幻觉删除
        else:
            out.append(tok)
    return out, {"mode": "vote", "n_cycles": n_full,
                 "n_cycles_before_filter": len(cyc),
                 "tail_cycles_excluded": len(tail_dropped),
                 "need_keep": need_keep, "dropped": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("wav")
    ap.add_argument("out_prefix")
    ap.add_argument("--period", default="auto",
                    help="'auto' 自相关估计，或直接给秒数")
    ARGS = ap.parse_args()

    x, sr = sf.read(ARGS.wav, dtype="float32")
    dur = len(x) / sr
    if ARGS.period == "auto":
        T, peak = estimate_period(x, sr)
        print(f"[period] autocorr T={T}s peak={peak}")
    else:
        T, peak = float(ARGS.period), None

    cuts = cycle_cutpoints(x, sr, T)
    print(f"[cuts] n={len(cuts)} -> {cuts[:8]}{'...' if len(cuts) > 8 else ''}")

    words = read_words(ARGS.exp_dir)
    cycles = split_cycles(words, cuts, dur)
    nonempty = [c for c in cycles if c]
    cons, info = consensus(nonempty)

    report = {
        "period_sec": T, "ac_peak": peak, "dur": round(dur, 2),
        "cuts": cuts, "n_words": len(words),
        "n_nonempty_cycles": len(nonempty),
        "cycle_texts": [" ".join(c) for c in nonempty],
        "consensus_info": info, "consensus_text": " ".join(cons),
    }
    Path(ARGS.out_prefix + "_consensus.txt").write_text(
        " ".join(cons) + "\n")
    Path(ARGS.out_prefix + "_cycles.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    for i, c in enumerate(nonempty[:12]):
        print(f"[cycle {i}] {' '.join(c)[:150]}")
    print("[CONSENSUS]", " ".join(cons)[:600])


if __name__ == "__main__":
    main()
