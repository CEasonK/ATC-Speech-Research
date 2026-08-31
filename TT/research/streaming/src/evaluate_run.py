"""统一评测：读取某引擎实验目录的 events/snapshots/final_text，
对照 eval_manifest 的参考时间轴，输出 metrics.json（WER/LAG/token延迟/RTF）。

用法：python evaluate_run.py <exp_dir> <eval_manifest.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import atoks  # noqa: E402
from metrics import levenshtein  # noqa: E402


def load_exp(exp_dir):
    exp_dir = Path(exp_dir)
    final = (exp_dir / "transcript_final.txt").read_text().strip()
    events = [json.loads(l) for l in open(exp_dir / "events.jsonl")]
    snaps = [json.loads(l) for l in open(exp_dir / "snapshots.jsonl")]
    meta = json.load(open(exp_dir / "meta.json"))
    return final, events, snaps, meta


def main():
    exp_dir, man_path = sys.argv[1], sys.argv[2]
    man = json.load(open(man_path))
    final, events, snaps, meta = load_exp(exp_dir)

    ref_K = man["ref_tokens_K"]                    # norm 风格（字母拼读）
    ref_starts, ref_ends = man["ref_starts"], man["ref_ends"]

    # ---- WER（norm_asr 双侧归一）----
    hyp_toks = atoks(final)
    # 官方口径：token 编辑距离按参考长度归一
    r_norm = []
    for t in ref_K:
        r_norm += atoks(t)
    h_norm = atoks(final)
    dist = levenshtein(h_norm, r_norm)
    wer2 = dist / max(len(r_norm), 1)

    # ---- token 延迟：对齐 hyp→ref×K（SequenceMatcher），emit=chunk 结束时刻 ----
    from difflib import SequenceMatcher

    sm = SequenceMatcher(a=h_norm, b=r_norm, autojunk=False)
    # 显式建立「归一化 token 下标 → 词级事件下标」映射：
    # 归一化会把数字词展开成多个 token，直接拿 token 下标索引 events 会错位。
    evt_of_tok = []
    for ei, e in enumerate(events):
        n_tok = max(len(atoks(e.get("word", ""))), 1)
        evt_of_tok.extend([ei] * n_tok)
    lat = []
    matched = 0
    unmatched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for off in range(i2 - i1):
            gi = i1 + off
            if gi < len(evt_of_tok):
                lat.append(events[evt_of_tok[gi]]["emit_audio_t"]
                           - ref_starts[j1 + off])
                matched += 1
            else:
                unmatched += 1
    if unmatched:
        print(f"[warn] token→事件映射溢出 {unmatched} 处"
              f"（hyp 归一 token 数 {len(h_norm)} vs 词级事件数 {len(events)} 不一致），"
              f"对应样本已从延迟统计剔除")

    def pct(a, q):
        return round(float(np.percentile(a, q)), 3) if a else None

    tok_lat = {
        "n": matched,
        "match_ratio": round(matched / max(len(events), 1), 3),
        "median": pct(lat, 50),
        "p95": pct(lat, 95),
        "mean": round(float(np.mean(lat)), 3) if lat else None,
    }

    # ---- LAG(τ) 曲线 ----
    taus, lags = [], []
    words_so_far = [s["words"] for s in snaps]
    ts_list = [s["t"] for s in snaps]
    ends_arr = np.array([r for r in ref_ends])
    rflat = r_norm
    # ref token per-K 发音终点已展开
    for tau in np.arange(5.0, min(meta.get("dur", 999), ends_arr.max() + 2), 5.0):
        n_ref = int(np.sum(ends_arr <= tau))
        if n_ref <= 0:
            continue
        k = max((i for i, t in enumerate(ts_list) if t <= tau), default=-1)
        p = words_so_far[k] if k >= 0 else []
        pn = []
        for w in p:
            pn += atoks(w)
        lags.append(levenshtein(pn[: n_ref + 12], rflat[:n_ref]) / n_ref)
        taus.append(float(tau))
    mean_lag = float(np.mean(lags)) if lags else None

    out = {
        "wer_token_level": round(wer2, 4),
        "wer_edit_over_len": round(wer2, 4),
        "hyp_len": len(h_norm),
        "ref_len": len(r_norm),
        "token_latency": tok_lat,
        "lag_mean": round(mean_lag, 4) if mean_lag is not None else None,
        "lag_curve_n": len(lags),
        "rtf": meta.get("rtf"),
        "events": len(events),
        "latency_unmatched": unmatched,
    }
    with open(Path(exp_dir) / "metrics.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
