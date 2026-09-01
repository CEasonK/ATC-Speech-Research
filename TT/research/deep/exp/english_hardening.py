"""英语信道加固轮（CYYT_ATIS_a / b 优先深挖）。

补三个此前缺口：
A. turbo 第三裁判整篇正位/错位验证 —— 此前 final_validation.json 只有 atc+v3 四组，
   a-atc 边缘失败后没有第三独立引擎的整篇复核。
B. 循环一致性检验（英语 ATIS 特有强证据）—— a 音频为同一报文循环广播（周期 28.143s，约 4 遍）。
   真文本的 per-window NLL 低谷必须以广播周期重复出现；乱序文本无此结构。
   曲线版（原音频逐窗）+ 实例版（inst_01..04 切片各自计分）互证。
C. 差异字段 turbo 独立复核 —— altimeter(a:3023 vs b:3033) 与 closing(a:AS vs b:WHEN REQUESTED)。

输出 results/english_hardening.json
"""
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
AUDIO = {
    "a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}
SEG_A = DEEP / "segments" / "CYYT_ATIS_a"
PERIOD_S = 28.143          # Day 2/P3：静音间隙线性拟合周期，残差 0.15s
TURBO_DIR = Path.home() / ".cache/huggingface/hub/models--tclin--whisper-large-v3-turbo-atcosim-finetune"
TURBO = next(str(s) for s in (TURBO_DIR / "snapshots").glob("*")
             if (s / "model.safetensors").exists())

A_TEXT = (DEEP / "results" / "a_final.txt").read_text().strip()
B_TEXT = (DEEP / "results" / "b_final.txt").read_text().strip()

# 与 assemble_final.py 相同的行结构（正位/错位验证用）
LINES_COMMON = [
    ("SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
     "INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO"),
    ("WIND TWO FOUR ZERO AT FIVE", "WIND TWO FOUR ZERO AT FIVE"),
    ("VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET", "THOUSAND FEET"),
    ("TEMPERATURE ONE DEW POINT MINUS ONE", "TEMPERATURE ONE"),
]
LINES_TAIL = [
    ("APPROACH RNAV ZULU RUNWAY TWO EIGHT", "APPROACH RNAV ZULU RUNWAY TWO EIGHT"),
    ("APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
     "LANDING AND DEPARTING RUNWAY"),
    ("INFORM ATC THAT YOU HAVE INFORMATION FOXTROT", "HAVE INFORMATION FOXTROT"),
]
FREQ = "INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE"
LINES = {
    "a": LINES_COMMON + [("ALTIMETER THREE ZERO TWO THREE", "ALTIMETER THREE ZERO TWO THREE")]
         + LINES_TAIL[:1] + [(FREQ + " AS REQUESTED", "CENTER ON FREQUENCY")] + LINES_TAIL[1:],
    "b": LINES_COMMON + [("ALTIMETER THREE ZERO THREE THREE", "ALTIMETER THREE ZERO THREE THREE")]
         + LINES_TAIL[:1] + [(FREQ + " WHEN REQUESTED", "CENTER ON FREQUENCY")] + LINES_TAIL[1:],
}


def shuffle_text(text: str, seed: int = 42) -> str:
    words = normalize_text(text).split()
    return " ".join(random.Random(seed).sample(words, len(words)))


# ---------- Part A：turbo 整篇正位/错位 ----------
def part_a(sc):
    out = {}
    for ch in ["a", "b"]:
        lines = LINES[ch]
        anchors = {a: sc.find_anchor_window(AUDIO[ch], a) for _, a in lines}
        rows, tot_c, tot_w, tok_sum = [], 0.0, 0.0, 0
        for i, (text, anchor) in enumerate(lines):
            t = anchors[anchor]
            r_ok = sc.score_constrained(AUDIO[ch], text, t, half_width=5.0)
            j = (i + 1) % len(lines)
            r_bad = sc.score_constrained(AUDIO[ch], text, anchors[lines[j][1]], half_width=5.0)
            nt = len(sc.tok.encode(normalize_text(text), add_special_tokens=False))
            rows.append({"line": text, "t_correct": t,
                         "nll_correct_pos": r_ok["score"], "nll_wrong_pos": r_bad["score"]})
            tot_c += r_ok["score"] * nt
            tot_w += r_bad["score"] * nt
            tok_sum += nt
            print(f"  [turbo/{ch}] ok={r_ok['score']:.4f} bad={r_bad['score']:.4f}  {text[:52]}", flush=True)
        out[ch] = {"rows": rows,
                   "weighted_nll_correct": tot_c / tok_sum,
                   "weighted_nll_wrongpos": tot_w / tok_sum}
        print(f"  ==> turbo {ch}: correct={tot_c/tok_sum:.4f} wrongpos={tot_w/tok_sum:.4f}", flush=True)
    return out


# ---------- Part B：a 信道循环一致性 ----------
def part_b(sc):
    out = {}
    full = A_TEXT
    shuf = shuffle_text(full)
    entry = sc.load_audio(AUDIO["a"])
    ids_t = sc.tok.encode(normalize_text(full), add_special_tokens=False)
    ids_s = sc.tok.encode(normalize_text(shuf), add_special_tokens=False)
    cur_t = sc._score_ids(entry, ids_t)
    cur_s = sc._score_ids(entry, ids_s)
    starts = np.array(entry["win_starts"])

    # 相位对齐 binning：(窗起点 mod 周期) → 平均 NLL。真文本应在固定相位出现低谷。
    phase_t = starts % PERIOD_S
    bins = np.linspace(0, PERIOD_S, 8)
    bt, _ = np.histogram(phase_t, bins=bins, weights=cur_t)
    cnt, _ = np.histogram(phase_t, bins=bins)
    prof_t = (bt / np.maximum(cnt, 1)).tolist()
    bs, _ = np.histogram(phase_t, bins=bins, weights=cur_s)
    prof_s = (bs / np.maximum(cnt, 1)).tolist()
    k_best = int(cur_t.argmin())
    # 次低谷：与主低谷相位差 <3s 的其余窗口里的最小值
    best_phase = phase_t[k_best]
    phase_mask = np.abs(((phase_t - best_phase + PERIOD_S / 2) % PERIOD_S) - PERIOD_S / 2) < 3.0
    repeats = [round(float(cur_t[i]), 4) for i in np.where(phase_mask)[0]]
    out["curve"] = {
        "period_s": PERIOD_S,
        "win_starts": [round(float(x), 1) for x in starts],
        "nll_text_per_window": [round(float(x), 4) for x in cur_t],
        "nll_shuffle_per_window": [round(float(x), 4) for x in cur_s],
        "text_best_window": {"t_start": round(float(starts[k_best]), 1),
                             "nll": round(float(cur_t[k_best]), 4)},
        "shuffle_min_nll": round(float(cur_s.min()), 4),
        "text_phase_profile_mean_nll": [round(x, 4) for x in prof_t],
        "shuffle_phase_profile_mean_nll": [round(x, 4) for x in prof_s],
        "phase_bin_edges": [round(float(x), 2) for x in bins],
        "same_phase_windows_text_nll": repeats,
    }

    # 实例版：inst_01..08 各自能量筛查 + 有效者计分全文 vs 乱序
    insts = {}
    for p in sorted(SEG_A.glob("inst_*.wav")):
        wav, _ = librosa.load(str(p), sr=16000, mono=True)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        rec = {"rms": round(rms, 4)}
        if rms > 0.01:   # 有效语音实例；削波噪声段(≈0.22 全段噪声)与静音(<0.01)排除
            r_t = sc.score(str(p), full)
            r_s = sc.score(str(p), shuf)
            rec.update({"nll_text": round(r_t["score"], 4),
                        "nll_shuffle": round(r_s["score"], 4),
                        "delta": round(r_s["score"] - r_t["score"], 4)})
        insts[p.stem] = rec
        print(f"  [inst] {p.stem}: {rec}", flush=True)
    out["instances"] = insts
    return out


# ---------- Part C：差异字段 turbo 复核 ----------
def part_c(sc):
    cases = {
        "altimeter_a": ("a", "ALTIMETER",
                        ["ALTIMETER THREE ZERO TWO THREE", "ALTIMETER THREE ZERO THREE THREE"]),
        "altimeter_b": ("b", "ALTIMETER",
                        ["ALTIMETER THREE ZERO TWO THREE", "ALTIMETER THREE ZERO THREE THREE"]),
        "closing_a": ("a", "CENTER ON FREQUENCY",
                      [FREQ + " AS REQUESTED", FREQ + " WHEN REQUESTED"]),
        "closing_b": ("b", "CENTER ON FREQUENCY",
                      [FREQ + " AS REQUESTED", FREQ + " WHEN REQUESTED"]),
    }
    out = {}
    for name, (ch, anchor, cands) in cases.items():
        t = sc.find_anchor_window(AUDIO[ch], anchor)
        res = [{"text": c, **sc.score_constrained(AUDIO[ch], c, t, half_width=5.0)}
               for c in cands]
        win = min(res, key=lambda r: r["score"])
        delta = abs(res[0]["score"] - res[1]["score"])
        out[name] = {"anchor_t": t, "cands": res, "winner": win["text"],
                     "delta": round(delta, 4),
                     "decisive": bool(delta > 1.4)}   # >污染区间才可单独定案
        for r in res:
            print(f"  [{name}] {r['score']:.4f}  {r['text'][-40:]}", flush=True)
        print(f"  [{name}] winner={win['text'][-20:]} delta={delta:.3f}", flush=True)
    return out


def main():
    result = {"part_A_turbo_full_validation": None,
              "part_B_cycle_consistency_a": None,
              "part_C_diff_fields_turbo": None}
    print("===== load turbo judge =====", flush=True)
    sc = NLLScorer(TURBO)
    print("--- Part A: turbo full-text correct/wrong position ---", flush=True)
    result["part_A_turbo_full_validation"] = part_a(sc)
    print("--- Part B: cycle consistency on channel a ---", flush=True)
    result["part_B_cycle_consistency_a"] = part_b(sc)
    print("--- Part C: diff fields under turbo ---", flush=True)
    result["part_C_diff_fields_turbo"] = part_c(sc)
    del sc
    torch.cuda.empty_cache()

    (DEEP / "results" / "english_hardening.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    print("saved results/english_hardening.json")


if __name__ == "__main__":
    main()
