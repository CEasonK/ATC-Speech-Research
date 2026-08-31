"""端到端推演：流式识别 -> 流式翻译 的延迟预算（x6 实验，客观测算/推演）。

组成：
- 识别侧：直接读 streaming/ 交付的事件流（draft 轨 token 时刻 / final 轨句级触发时刻）；
- 翻译侧：实测 Qwen2.5-7B 对单句 ATIS 中文生成的墙钟时间（GPU 上逐句生成，
  相当于"该句流式识别定稿后再起翻译请求"的保守口径）。

用法：python end_to_end.py <streaming_exp_dir(draft 轨)> [en_lines_file]
输出：results/e2e_latency.json（含 P50/P95 句级端到端延迟）
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TRANS_RES = SRC.parent / "results"


def load_events(exp_dir):
    """返回 {句子锚点: draft tokens(时刻,词)}；这里简化为按 att_start 重聚句。"""
    words = []
    with open(Path(exp_dir) / "events.jsonl") as f:
        for ln in f:
            e = json.loads(ln)
            w = e.get("word", "").strip()
            s = e.get("att_start")
            if w and s is not None:
                words.append((float(s), e.get("emit_audio_t"), w))
    words.sort()
    return words


def measure_translate_latency(en_sents):
    """实测：Qwen bf16 单句 ATIS 翻译墙钟时间列表（秒）。"""
    model_dir = SRC.parent / "models" / "qwen2.5-7b-instruct"
    if not model_dir.exists():
        print("[warn] qwen 模型缺失，跳过实测，用 1.2s/sent 文献估计值")
        return [1.2] * len(en_sents), True
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="cuda")
    mdl.eval()
    sysmsg = ("你是空管翻译引擎。把英文ATIS语句译成简洁中文，只输出译文。"
              "数字严格对应。")
    lat = []
    for s in en_sents[:12]:  # 抽样≤12句
        msgs = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": s}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(mdl.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=160, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        lat.append(time.time() - t0)
    del mdl
    torch.cuda.empty_cache()
    return lat, False


def split_sentences(words):
    """按词间静音间隙分句；阈值自适应：从宽松到收紧逐档尝试，
    保证至少聚出 2 句（否则长音频会整篇聚成 1 句，统计失真）。"""
    starts = [w[0] for w in words]
    for thr in (8.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6):
        sents, cur = [], []
        for i, s0 in enumerate(starts):
            cur.append(i)
            nxt = starts[i + 1] if i + 1 < len(words) else None
            if nxt is None or nxt - s0 > thr:
                if len(cur) >= 3:
                    sents.append(cur)
                cur = []
        if len(sents) >= 2:
            print(f"[split] 阈值 {thr}s -> {len(sents)} 句")
            return [[words[k] for k in g] for g in sents], thr
    # 全部档位都只聚出 1 句：退回最大间隔处硬切一次，避免单句退化
    print("[warn] 无有效句间停顿，按中位切分兜底")
    k = max(1, len(words) // 2)
    return [words[:k], words[k:]], None


def pctl(vals, q):
    """小样本安全的百分位（nearest-rank：ceil(n*q/100)，索引钳制在合法范围）。"""
    if not vals:
        return None
    sv = sorted(vals)
    k = min(len(sv) - 1, max(0, math.ceil(len(sv) * q / 100) - 1))
    return round(float(sv[k]), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="streaming draft 轨实验目录")
    ap.add_argument("en_lines", nargs="?", default=None)
    ap.add_argument("--tag", default="e2e")
    ARGS = ap.parse_args()

    words = load_events(ARGS.exp_dir)
    if not words:
        print("[error] 该实验目录无可用的词级事件（att_start 缺失或为空）")
        sys.exit(1)
    sents, split_thr = split_sentences(words)
    # 每句识别完成时刻 = 该句最后词的 emit_audio_t
    asr_finish = [(c[0][0], c[-1][1]) for c in sents]
    asr_lat = [f - s for s, f in asr_finish]

    en_lines = None
    if ARGS.en_lines and Path(ARGS.en_lines).exists():
        en_lines = [l.strip() for l in Path(ARGS.en_lines).read_text().splitlines()
                    if l.strip()]
    # 未提供英文行时翻译侧用文献估计值（与模型缺失分支同口径），保证脚本可独立运行
    tr_lat, estimated = ([1.2], True) if not en_lines else \
        measure_translate_latency(en_lines)
    sampled = len(en_lines or []) > len(tr_lat)

    per_sent_e2e = []
    for i, (s, f) in enumerate(asr_finish):
        t_lat = tr_lat[i % len(tr_lat)]
        per_sent_e2e.append({"sent_audio_start": round(s, 2),
                             "asr_final_delay": round(f - s, 2),
                             "translate_sec": round(t_lat, 2),
                             "e2e": round(f - s + t_lat, 2),
                             # 抽样耗时循环复用到更多句子时，句级配对为近似值
                             "pairing": "sampled" if sampled else
                                        ("estimated" if estimated else "measured")})
    e2e_vals = [p["e2e"] for p in per_sent_e2e]
    res = {
        "n_sentences": len(sents),
        "n_words_total": len(words),
        "split_threshold_sec": split_thr,
        "asr_final_delay": {"median": pctl(asr_lat, 50), "p95": pctl(asr_lat, 95)},
        "translate_latency_median": round(statistics.median(tr_lat), 2),
        "translate_estimated": estimated,
        "translate_sampled_n": len(tr_lat),
        "e2e_p50": pctl(e2e_vals, 50),
        "e2e_p95": pctl(e2e_vals, 95),
        "small_sample_warn": len(e2e_vals) < 20,
        "per_sentence": per_sent_e2e[:40],
    }
    out = TRANS_RES / f"{ARGS.tag}_latency.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k != "per_sentence"}, ensure_ascii=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
