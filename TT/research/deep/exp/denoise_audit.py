"""P6 降噪重审：早期"降噪无用"结论是在无裁判时代得出的，现用 NLL 裁判复核。
方法（公用方案，非单条特调）：
  谱减法降噪（STFT 噪声底 = 每频点 10 分位帧幅度，gain=2.0，floor=0.02）
  → 对 a/b 终稿逐行：锚窗在原音频定位一次（降噪不改变时间轴，锚位通用），
  同一锚位在 原音频 vs 降噪音频 上各计分（atc + v3 双裁判）。
采纳规则：仅当 4 组(2信道×2裁判)全部改善 >2% 才采纳；否则维持原音频。
产出 results/denoise_audit.json
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
OUT = DEEP / "results" / "denoise"
JUDGES = [("atc", TT / "models" / "whisper-large-v3-finetuned-for-ATC"),
          ("v3", "openai/whisper-large-v3")]
HALF = 6.0


def specsub(wav, sr=16000, n_fft=1024, hop=256, gain=2.0, floor=0.02):
    st = librosa.stft(wav, n_fft=n_fft, hop_length=hop).astype(np.float32)
    mag, ph = np.abs(st), np.angle(st)
    noise = np.percentile(mag, 10, axis=1, keepdims=True)
    clean = np.maximum(mag - gain * noise, floor * mag)
    return librosa.istft(clean * np.exp(1j * ph), hop_length=hop, length=len(wav))


def lines_of(p):
    # 尾三行为真实复诵（听音确认+recheck_tail_repeat.py 佐证），
    # 重复行是交付文本的真实内容，逐行审计需保留
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    for ch in ("CYYT_ATIS_a", "CYYT_ATIS_b"):
        src = TT / "audio" / f"{ch}.wav"
        wav, _ = librosa.load(src, sr=16000, mono=True)
        dn = specsub(wav)
        dn_path = OUT / f"{ch}_specsub.wav"
        sf.write(dn_path, dn, 16000)
        report[ch] = {}
        for tag, mpath in JUDGES:
            sc = NLLScorer(str(mpath))
            rows, dsum = [], []
            for n, ln in enumerate(lines_of(DEEP / "results"
                                            / ("a_final.txt" if ch.endswith("a")
                                               else "b_final.txt")), 1):
                anchor = " ".join(ln.split()[-5:])
                t0 = sc.find_anchor_window(str(src), anchor)
                r_o = sc.score_constrained(str(src), ln, t0, half_width=HALF)
                r_d = sc.score_constrained(str(dn_path), ln, t0, half_width=HALF)
                rows.append({"n": n, "orig": round(r_o["score"], 4),
                             "denoi": round(r_d["score"], 4)})
                dsum.append(r_d["score"] - r_o["score"])
                print(f"[{ch}|{tag}] L{n}: orig={r_o['score']:.3f} "
                      f"denoi={r_d['score']:.3f} Δ={dsum[-1]:+.3f}", flush=True)
            mean_o = float(np.mean([r["orig"] for r in rows]))
            mean_d = float(np.mean([r["denoi"] for r in rows]))
            report[ch][tag] = {"rows": rows,
                               "mean_orig": round(mean_o, 4),
                               "mean_denoised": round(mean_d, 4),
                               "improve_pct": round((mean_o - mean_d) / mean_o * 100, 2)}
            print(f"==> {ch}|{tag}: mean {mean_o:.4f} -> {mean_d:.4f} "
                  f"(improve {report[ch][tag]['improve_pct']}%)", flush=True)
            del sc
            torch.cuda.empty_cache()
    adopt = all(v["improve_pct"] > 2.0 for ch in report for v in report[ch].values())
    report["_verdict"] = ("ADOPT" if adopt else
                          "KEEP-ORIGINAL：未达全组>2%改善门槛，维持无降噪管线")
    (DEEP / "results" / "denoise_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(report["_verdict"])


if __name__ == "__main__":
    main()
