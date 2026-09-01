"""RJTT 段级终审 v2：三引擎候选 + 语法合理性硬过滤 + 多数聚类投票 + 簇内 v3-NLL 决胜。
v1 教训：单 v3 裁判受 LM 先验污染，把 seg04/05 判给日常英语垃圾（"That's possible, anyway"
NLL 3.41 < 5.42）——管制频率上不可能的话反而 NLL 低。对应方法论：
  语法层否决声学层 → 群体投票（三引擎） → NLL 只做簇内微决胜。
流程：
  1) turbo_atcosim 对 9 段自由解码（第三引擎，ATIS 微调域先验强）
  2) ATC 核心词覆盖率 frac + 会话词黑名单 → 硬过滤（不合理的候选直接出局）
  3) 幸存候选按归一化 token Jaccard>=0.30 聚类，最大簇胜出（2/3 多数即簇）
  4) 簇内取 v3 NLL 最低者为代表；全单例时退回全候选 v3 NLL
输出 results/rjtt_final.txt + rjtt_validation.json（覆盖 v1）
"""
import gc
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
SEG = DEEP / "segments" / "RJTT_CONTROL"
IDX = DEEP / "results" / "decode_rjtt_index.json"
V1 = DEEP / "results" / "rjtt_validation.json"

TURBO_DIR = Path.home() / ".cache/huggingface/hub/models--tclin--whisper-large-v3-turbo-atcosim-finetune"
TURBO = next(str(s) for s in (TURBO_DIR / "snapshots").glob("*")
             if (s / "model.safetensors").exists())
print(f"turbo: {TURBO}", flush=True)

ALPHA = {"ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL",
         "INDIA", "JULIETT", "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA",
         "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM", "VICTOR", "WHISKEY",
         "XRAY", "YANKEE", "ZULU"}
NUMS = {"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
        "NINER", "NINE", "HUNDRED", "THOUSAND", "DECIMAL", "POINT"}
PHRASE = {
    "TOWER", "CONTROL", "GROUND", "APPROACH", "DEPARTURE", "CENTER", "RADIO",
    "FLIGHT", "LEVEL", "CLIMB", "CLIMBING", "DESCEND", "DESCENDING", "MAINTAIN",
    "PASSING", "LEAVING", "REQUEST", "REQUESTS", "REQUESTING", "DIRECT", "TO",
    "AND", "THE", "HEAVY", "SUPER", "GOOD", "MORNING", "AFTERNOON", "EVENING",
    "DAY", "SPEED", "HEADING", "VIA", "EXPECT", "CONTACT", "FREQUENCY", "SQUAWK",
    "ROGER", "WILCO", "READBACK", "CORRECT", "AFFIRMATIVE", "NEGATIVE", "STANDBY",
    "MONITOR", "RADAR", "TERMINATE", "SIMULATED", "ALTIMETER", "FEET", "KNOTS",
    "RUNWAY", "WIND", "VISIBILITY", "TEMPERATURE", "DEW", "INFORMATION", "ATIS",
    "CLEARANCE", "TAKEOFF", "LANDING", "LINE", "UP", "WAIT", "HOLD", "SHORT",
    "REPORT", "ESTABLISHED", "ON", "COURSE", "TRACK", "MILES", "FROM", "WITH",
    "YOU", "HAVE", "IS", "AT", "FOR", "OF", "IN", "IT", "AS", "WHEN", "ALL",
    "THANK", "CLEAR", "JET", "STAR", "AIR", "AIRLINE", "AIRLINES", "AIRWAYS",
    "AIRLINES", "CENTRAL", "SHANGHAI", "JAPAN", "TOKYO", "OSAKA", "NARITA",
    "AEROFLOT", "SHAMROCK", "THAI", "RYANAIR", "FEDEX", "JAPANAIR", "JETSTAR",
    "ORANGE", "EVAAIR", "KOREAN", "ASIANA", "SINGAPORE", "CATHAY", "PACIFIC",
    "CARGO", "EXPRESS", "CONNECTION", "INITIALLY", "MAINTAINING", "REACH",
    "LEVELS", "FL", "AER", "LINGUS", "CHANGI", "AIRBUS", "BOEING",
}
CORE = ALPHA | NUMS | PHRASE
BLACK = {"THAT'S", "POSSIBLE", "ANYWAY", "BIGGER", "THAN", "EQUALS", "I'M",
         "HELLO", "OKAY", "YEAH", "PLEASE", "SORRY", "WELL", "JUST", "MAYBE"}


def toks(t):
    return re.sub(r"[^A-Z0-9]+", " ", t.upper()).split()


def plausible(text):
    """返回 (核心词覆盖率, 拒绝理由列表)。空/黑名单/覆盖率<0.5 → 不合理。"""
    tk = toks(text)
    if not tk:
        return 0.0, ["EMPTY"]
    hits = sorted({w for w in tk if w in BLACK})
    core = sum(1 for w in tk if w in CORE or re.fullmatch(r"[A-Z]", w)
               or re.fullmatch(r"[A-Z]+\d+", w) or w.isdigit())
    frac = core / len(tk)
    why = ([f"blacklist:{hits}"] if hits else []) + \
          ([f"frac={frac:.2f}<0.5"] if frac < 0.5 else [])
    return frac, why


def jaccard(a, b):
    sa, sb = set(toks(a)), set(toks(b))
    return len(sa & sb) / max(1, len(sa | sb))


def load16k(apath):
    import librosa
    wav, _ = librosa.load(apath, sr=16000, mono=True)
    if len(wav) < 480000:
        wav = np.pad(wav, (0, 480000 - len(wav)))
    return wav


def free(m=None):
    if m is not None:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def main():
    manifest = json.loads((SEG / "manifest.json").read_text())
    SEGS = {f"seg{k:02d}": m for k, m in enumerate(manifest)}
    idx = json.loads(IDX.read_text())
    by_id = {e["id"]: e["text"] for e in idx}
    v1 = json.loads(V1.read_text())

    # ---- 1) turbo 第三引擎自由解码（幂等：已有条目跳过） ----
    new_entries = []
    have = {e["id"] for e in idx}
    need = [s for s in SEGS if f"turbo_{s}" not in have]
    if need:
        proc = WhisperProcessor.from_pretrained(TURBO)
        wm = (WhisperForConditionalGeneration.from_pretrained(TURBO, dtype=torch.float16)
              .to("cuda").eval())
        PROMPT = torch.tensor([[proc.tokenizer.convert_tokens_to_ids(t) for t in
                                ["<|startoftranscript|>", "<|en|>", "<|transcribe|>",
                                 "<|notimestamps|>"]]]).cuda()
        for s in need:
            xf = proc.feature_extractor(
                load16k(str(SEG / f"seg_{s[3:]}.wav")), sampling_rate=16000,
                return_tensors="pt").input_features.to("cuda", torch.float16)
            with torch.no_grad():
                seq = wm.generate(xf, decoder_input_ids=PROMPT, max_new_tokens=128,
                                  num_beams=5, do_sample=False)
            txt = proc.tokenizer.decode(seq[0], skip_special_tokens=True).strip()
            by_id[f"turbo_{s}"] = txt
            new_entries.append({"id": f"turbo_{s}", "text": txt})
            print(f"[turbo {s}] {txt[:90]}", flush=True)
        free(wm)
        free()
        idx.extend(new_entries)
        IDX.write_text(json.dumps(idx, ensure_ascii=False, indent=1))

    # ---- 2) v3 NLL：atc/qwen 复用 v1，turbo 补算（同窗同参） ----
    sc = NLLScorer("openai/whisper-large-v3")
    cands = {}
    for s, m in SEGS.items():
        apath = str(SEG / f"seg_{s[3:]}.wav")
        center = (m["t0"] + m["t1"]) / 2 - m["t0"]
        half = max(m["dur"] / 2 + 2.0, 5.0)
        cands[s] = {}
        for tag in ["atc_beam5", "qwen"]:
            cands[s][tag] = {"text": by_id[f"{tag}_{s}"],
                             "nll_v3": v1[s][tag]["nll_v3"]}
        t = by_id[f"turbo_{s}"]
        r = sc.score_constrained(apath, t, center, half_width=half)
        cands[s]["turbo"] = {"text": t, "nll_v3": r["score"]}
        print(f"[{s}] v3|turbo: {r['score']:.4f}  {t[:60]}", flush=True)
    del sc
    torch.cuda.empty_cache()

    # ---- 3) 过滤 + 投票 + 决胜 ----
    out, final_lines = {}, []
    for s, m in SEGS.items():
        entries = []
        for tag, d in cands[s].items():
            frac, why = plausible(d["text"])
            entries.append({"tag": tag, **d, "frac": round(frac, 2), "reject": why})
            print(f"[{s}] {tag}: frac={frac:.2f} reject={why} "
                  f"nll={d['nll_v3']:.3f}  {d['text'][:55]}", flush=True)
        ok = [e for e in entries if not e["reject"]]
        pool = ok if ok else entries
        fallback = "ALL-REJECTED" if not ok else ""
        n = len(pool)
        sim = [[jaccard(a["text"], b["text"]) for b in pool] for a in pool]
        best = None
        for r in range(n, 0, -1):
            for sub in itertools.combinations(range(n), r):
                if all(sim[i][j] >= 0.30 for i, j in itertools.combinations(sub, 2)):
                    key = (-r, sum(pool[i]["nll_v3"] for i in sub) / r)
                    if best is None or key < best[0]:
                        best = (key, sub)
            if best:
                break
        win_i = min(best[1], key=lambda i: pool[i]["nll_v3"])
        w = pool[win_i]
        cluster = [pool[i]["tag"] for i in best[1]]
        out[s] = {"candidates": entries, "winner": w["tag"], "cluster": cluster,
                  "fallback": fallback,
                  "t0": m["t0"], "t1": m["t1"]}
        final_lines.append(f"[{s}] {m['t0']:.1f}-{m['t1']:.1f}s "
                           f"({w['tag']}{',' + fallback if fallback else ''}) "
                           f"{w['text'].strip()}")
        print(f"==> {s}: WIN {w['tag']} cluster={cluster} {fallback}", flush=True)

    (DEEP / "results" / "rjtt_final.txt").write_text("\n\n".join(final_lines) + "\n")
    (DEEP / "results" / "rjtt_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("saved rjtt_final.txt / rjtt_validation.json (v2)")


if __name__ == "__main__":
    main()
