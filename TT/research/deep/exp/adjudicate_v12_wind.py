"""P5e wind 终审补充证据（v12）：
1) 精确重切片 ×3 中心/信道 → results/wind_slices/v12/
2) 增量计分：b 的 WIND 步进曲线、a 的 AT 步进曲线（atc+v3 双裁判）
背景：v11 forced-alignment 给出 b-WIND Δ1.27 / a-AT Δ0.46，但 v8 已证明
单词插入的 decoder LM 先验污染可达 1.2-1.4 nat，需自由解码第三方证据三角定位。

口径修正（P2/D1）：score_constrained 返回的 score 本身已是 per-token 均值
（nll_scorer L213 nll.mean(dim=1)），旧版再除 nt 属双重归一、短前缀被系统性
压低；本文件历史 adjudication_v12_wind.json 中 per_tok 列及由其得出的排序作废。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer, normalize_text

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}
OUT = DEEP / "results" / "wind_slices" / "v12"
OUT.mkdir(parents=True, exist_ok=True)

# a: wind 始于 ~56s（首切片 [56,71] 中 WIND 为第一词）；b: wind 数字 ≈ 235-242
CENTERS = {
    "CYYT_ATIS_a": [59.0, 61.0, 63.0],
    "CYYT_ATIS_b": [236.0, 238.0, 240.0],
}

# ---------- 1) 切片 ----------
def load16k(p):
    import librosa
    wav, _ = librosa.load(p, sr=16000, mono=True)
    return wav


import soundfile as sf

slices = {}
for ch, apath in CH.items():
    wav = load16k(apath)
    for c in CENTERS[ch]:
        s0, s1 = int((c - 6) * 16000), int((c + 9) * 16000)
        sp = OUT / f"{ch}_c{int(c)}.wav"
        sf.write(sp, wav[s0:s1], 16000)
        slices[f"{ch}_c{int(c)}"] = str(sp)
        print(f"[slice] {ch} c={c} -> [{c-6},{c+9}]s", flush=True)

# ---------- 2) 增量计分 ----------
CURVES = {
    # 问题1：b 的 WIND 前缀有无（ZULU 之后、数字之前）
    ("CYYT_ATIS_b", 237.0): {
        "H_wind": ["WIND", "TWO", "FOUR", "ZERO", "AT", "FIVE"],
        "H_nowind": ["TWO", "FOUR", "ZERO", "AT", "FIVE"],
    },
    # 问题2：a 的 AT 有无（ZERO 与 FIVE 之间）
    ("CYYT_ATIS_a", 62.0): {
        "H_at": ["WIND", "TWO", "FOUR", "ZERO", "AT", "FIVE"],
        "H_noat": ["WIND", "TWO", "FOUR", "ZERO", "FIVE"],
    },
}

JUDGES = [("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
          ("v3", "openai/whisper-large-v3")]

results = {"slices": slices, "incremental": {}}
for tag, mdir in JUDGES:
    print(f"===== incremental judge: {tag} =====", flush=True)
    sc = NLLScorer(mdir)
    for (ch, c), hyps in CURVES.items():
        for hname, words in hyps.items():
            rows = []
            for k in range(1, len(words) + 1):
                text = " ".join(words[:k])
                r = sc.score_constrained(CH[ch], text, c, half_width=5.0)
                nt = len(sc.tok.encode(normalize_text(text), add_special_tokens=False))
                # P2/D1: r["score"] 已是 per-token NLL，禁止再除 nt（双重归一 bug）
                rows.append({"prefix": text, "per_tok": r["score"],
                             "score": r["score"], "n_tok": nt})
                print(f"[inc|{tag}|{ch}|{hname}] {rows[-1]['per_tok']:.3f}  ({nt}tok)  {text}",
                      flush=True)
            results["incremental"][f"{tag}|{ch}|{hname}"] = rows
    del sc
    torch.cuda.empty_cache()

(DEEP / "results" / "adjudication_v12_wind.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1))
print("saved adjudication_v12_wind.json")
