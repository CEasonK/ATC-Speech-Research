#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最优 ATIS 识别管线 v3 (2026-08-21 循环广播研究结论固化)
========================================================
用法:
  python run_best_asr.py audio/CYYT_ATIS_a.wav            # 默认: 单条报文模式
  python run_best_asr.py audio/CYYT_ATIS_a.wav --full     # 完整转写 (保留全部循环)
  python run_best_asr.py audio/CYYT_ATIS_a.wav --no-clean # 跳过清洗

核心发现 (实验证据, 见 research/best/README.md):
  1. ATIS 是循环广播 —— CYYT_a 原始音频经 mel 自相关证实每 28.6s 重复
     一遍 (corr=0.625), 274s 里约 5 遍同一报文 + 后 124s 满能量噪声
  2. 噪声段能量与语音一样满, 能量 VAD 无效; Whisper 的 no_speech_prob
     (pipeline 内置 VAD) 才能过滤 → 消除 "Thank you"/"BEEP" 幻觉
  3. 旧 25s 滑窗每窗只含报文片段, Whisper 补全偏差 → 每窗补一遍完整
     报文 → 输出 4 遍冗余 (362 词)
  4. 各遍解码乱码不同 (循环投票仅 44% 共识, 逐词投票失败), 但术语打分
     能选出最干净的一遍 (唯一 GANDER+VISIBILITY+3023 全对的那遍)

单条报文模式 (默认):
  pipeline(30s 分块 + no_speech VAD) → 按 WIND 锚点切循环实例
  → 每实例术语打分 → 保留最优一条 → clean()
完整模式 (--full):
  同上解码, 不切实例, 输出全部 (清洗后)
输出: results/best_pipeline/<音频>/<名>_best.txt (+ .raw.txt)
"""
import argparse
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TT = HERE.parent
MODEL_DIR = TT / "models" / "whisper-large-v3-finetuned-for-ATC"
RESULTS = TT / "results" / "best_pipeline"

# ---- 解码参数 (v3: 30s 非重叠分块, 无滑窗重叠 → 无重复拼接) ----
CHUNK_S = 30                 # 分块秒数 (非重叠)
N_SPPEECH_THR = 0.6          # pipeline VAD: no_speech_prob 高于此判无语音
LOGPROB_THR = -1.0           # 低置信片段阈值
MSG_LEN = 75                 # 一条 ATIS 报文约 70-75 词
MIN_INST_WORDS = 40          # 最优实例词数低于此视为伪锚点误切, 不启用单条模式
DEFAULT_ANCHOR = "WIND TWO FOUR ZERO"  # 完整短语锚点 (避免误中单词内部)
TEMPERATURE_FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # pipeline 内置回退序列

# ---- 幻觉短语 (整句删; 判读: ATIS 报文不可能出现这些话) ----
# 用整句删而非词删: Whisper 对噪声段编的是完整句子, 整句才是幻觉的粒度。
# 教训: 早期全局词删正则 (如 r"thank you[.]*\s*") 会跨句边界吞掉相邻真内容。
PHRASE_HALLUC = [
    r"thank you(?:,?\s*thank you)*",
    r"b[e]{2,}p(?:,?\s*b[e]{2,}p)*",
    r"i don'?t know if you'?re ready or not",
    r"i'?ll talk to you later",
    r"if you'?re interested in talking\s+about any other things",
    r"austrian air force",
    r"siren\s+waiting",
    r"cleared for touch and go",
    r"speedbird(?:\s+ready)?",
    r"silent takeoff",
    r"softly touched to landing",
    r"bee air",
    r"box\s+(?:strong|dog)",
    r"takeoff time",
    r"\btwo seven four\b",
    r"\bryna?ir\b",
]

# ---- 术语打分: 干净命中 (+) 与已知乱码形态 (-) ----
# 依据: 同一报文 5 遍独立解码, 干净形态只在最干净那遍出现;
#       乱码形态每遍不同 (RECEIVABLITY/REQUESTABILITY/...)
GOOD_TERMS = [
    r"\bVISIBILITY\b", r"\bALTIMETER\b", r"\bGANDER\b", r"\bRNAV\b",
    r"\bFOXTROT\b", r"\bZULU\b", r"\bFREQUENCY\b",
]
BAD_TERMS = [
    r"\bRECEIVABLITY\b", r"\bREQUESTABILITY\b", r"\bDISABILITY\b",
    r"\bALCIMETER\b", r"\bSANDURK\b", r"\bFANDAS\b", r"\bFIENDER\b",
    r"\bDEANDER\b", r"\bARNAB\b", r"\bARNAP\b", r"\bALNAB\b",
    r"\bARNAV\b", r"\bARNAD\b", r"\bCLEARED FOR FEW\b",
    r"\bTHE INTERCEPT\b", r"\bBEEP\b",
]

# ---- 单字/多字乱码 → 真值 (清洗前先修, 避免被当幻觉误删) ----
# disability 是 VISIBILITY 的乱码形态 (其后跟 1.5 数字), 转回而非删
PRE_FIXES = {
    "DISABILITY": "VISIBILITY",
}

# ---- 术语纠错 (选出的最优实例再做规范化) ----
TERM_FIXES = {
    "ALCIMETER": "ALTIMETER",
    "ARNAB": "RNAV", "ARNAP": "RNAV", "ALNAB": "RNAV",
    "ARNAV": "RNAV", "ARNAD": "RNAV",
    "VISABILITY": "VISIBILITY",
    "JULIETT": "JULIET",
    "SYDNEY": "GANDER",
    "SANDURK": "GANDER", "FIENDER": "GANDER",
    "DEANDER": "GANDER", "FANDAS": "GANDER",
    "ENTERSTANDING": "STANDING",
    "INTERCEPTION": "INTERSECTION",
    "DESCENDING LEVEL": "VISIBILITY",
    "APPROVED": "APPROACH",
}


def _sent_halluc(sent: str) -> bool:
    """一个分句是否应删: 只按幻觉短语清单判, 不做长度/数字启发式。

    教训: 早期 '≤4词且无数字→删' 规则误杀过真内容 (如 3 词航班呼号
    HOTEL GOLF PAPA)。宁可漏删孤立幻觉碎片, 不可误删真实广播内容。
    """
    sl = sent.lower()
    return any(re.search(p, sl) for p in PHRASE_HALLUC)


def clean(text: str) -> str:
    """幻觉清洗 + 术语纠错 (v3 定稿)。

    翻车记录 (为什么是这个形状):
      v1 全局词删: 'thank you[.]*\\s*' 跨句边界吞掉相邻真内容
      v2 整句删 + '≤4词无数字→删': 误杀 3 词真呼号 HOTEL GOLF PAPA
      v3 定稿: 单字乱码先修复 → 分句 → 含幻觉短语的分句整句删 → 术语映射
    """
    # 0. 单字乱码先行修复 (避免误判为幻觉)
    for bad, good in PRE_FIXES.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text, flags=re.IGNORECASE)
    # 1. 分句 (按 . , ; ! ? 切, 保留无标点尾段)
    sents = re.findall(r"\S[^\n]*?[\.,;!?]|\S[^\n]*", text)
    keep = [s for s in sents if not _sent_halluc(s)]
    t = " ".join(keep)
    # 2. 术语映射 (全大写输出, ATIS 惯例)
    up = t.upper()
    for bad, good in TERM_FIXES.items():
        up = re.sub(rf"\b{re.escape(bad)}\b", good, up)
    # 3. 空白/标点规范化
    t = re.sub(r"\s{2,}", " ", up)
    t = re.sub(r"\s+([.,])", r"\1", t)
    return t.strip()


def score_instance(text: str) -> int:
    """循环实例质量分: 干净术语命中数 - 乱码形态数。

    判读依据: 最干净那遍是唯一同时含 VISIBILITY+GANDER 干净形态的。
    """
    up = text.upper()
    s = 0
    for p in GOOD_TERMS:
        s += len(re.findall(p, up))
    for p in BAD_TERMS:
        s -= len(re.findall(p, up))
    return s


def split_instances(text: str, anchor: str,
                    msg_len: int = MSG_LEN, min_gap: int = 20):
    """按锚点 (整词) 切循环实例, 返回 [(score, instance_text), ...]。

    同一循环圈可能多次出现锚点 (30s 分块卡位), 词距 <min_gap 的合并。
    """
    words = text.upper().split()
    aw = anchor.upper().split()
    if len(words) < len(aw):
        return []
    pos = [i for i in range(len(words) - len(aw) + 1)
           if words[i:i + len(aw)] == aw]
    kept = []
    for p in pos:
        if not kept or p - kept[-1] >= min_gap:
            kept.append(p)
    return [(score_instance(" ".join(words[p:p + msg_len])),
             " ".join(words[p:p + msg_len])) for p in kept]


def load_engine(model_dir: Path):
    """加载模型 + pipeline, 返回可复用的 decode 闭包。
    模型加载重 (数十秒), 多音频批量时只加载一次。"""
    import torch
    from transformers import (AutoProcessor, WhisperForConditionalGeneration,
                              pipeline)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float16).to("cuda").eval()
    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=CHUNK_S,
                    no_speech_threshold=N_SPPEECH_THR,
                    logprob_threshold=LOGPROB_THR,
                    return_timestamps=False)

    def decode(audio_path: Path) -> str:
        """30s 非重叠分块 + 内置 no_speech VAD 解码整条音频, 返回原始文本。"""
        out = pipe(str(audio_path), language="en", task="transcribe",
                   temperature=TEMPERATURE_FALLBACK)
        return out["text"].strip()

    return decode


def split_instances(text: str, anchor: str,
                    msg_len: int = MSG_LEN, min_gap: int = 20):
    """按锚点 (整词) 切循环实例, 返回 [(score, instance_text), ...]。

    同一循环圈可能多次出现锚点 (30s 分块卡位), 词距 <min_gap 的合并。
    """
    words = text.upper().split()
    aw = anchor.upper().split()
    if len(words) < len(aw):
        return []
    pos = [i for i in range(len(words) - len(aw) + 1)
           if words[i:i + len(aw)] == aw]
    kept = []
    for p in pos:
        if not kept or p - kept[-1] >= min_gap:
            kept.append(p)
    insts = []
    for p in kept:
        seg = " ".join(words[p:p + msg_len])
        insts.append((score_instance(seg), seg))
    return insts


def process_one(decode, src: Path, anchor: str, full: bool, do_clean: bool):
    """处理单条音频, 返回 (raw, final, 描述)。decode 为已加载的解码闭包。"""
    raw = decode(src)
    insts = split_instances(raw, anchor)
    if full:
        final = clean(raw) if do_clean else raw
        desc = f"完整模式, 检出循环实例 {len(insts)} 个, 保留全部"
    elif (not insts or len(insts) < 2
          or len(insts[0][1].split()) < MIN_INST_WORDS):
        final = clean(raw) if do_clean else raw
        desc = f"未检出有效循环实例 ({len(insts)} 个), 按完整转写处理"
    else:
        insts.sort(key=lambda x: -x[0])
        best_score, best_txt = insts[0]
        final = clean(best_txt) if do_clean else best_txt
        desc = (f"检出循环实例 {len(insts)} 个, 术语打分 "
                f"{[s for s, _ in insts[:3]]} → 保留最优 "
                f"(score={best_score}, {len(best_txt.split())}词)")
    return raw, final, desc


def main():
    ap = argparse.ArgumentParser(description="最优 ATIS 识别管线 v3")
    ap.add_argument("audio", nargs="+", help="wav/mp3 路径 (可多个)")
    ap.add_argument("--full", action="store_true",
                    help="完整转写 (保留全部循环), 默认只输出最优一条")
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR,
                    help="循环锚点短语 (默认 WIND TWO FOUR ZERO)")
    ap.add_argument("--no-clean", action="store_true", help="跳过清洗")
    args = ap.parse_args()

    t0 = time.time()
    print(f"加载模型 {MODEL_DIR.name} ...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    decode = load_engine(MODEL_DIR)  # 只加载一次, 多音频复用
    print(f"模型就绪 {time.time()-t0:.0f}s", flush=True)
    for a in args.audio:
        src = Path(a)
        if not src.exists():
            print(f"[跳过] 文件不存在: {a}", flush=True)
            continue
        name = src.stem
        print(f"\n=== {name} ===", flush=True)
        try:
            raw, final, desc = process_one(decode, src, args.anchor,
                                           args.full, not args.no_clean)
        except Exception as e:  # 单条失败不阻断其余
            print(f"  [错误] {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  {desc}", flush=True)
        out_dir = RESULTS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_best.raw.txt").write_text(
            raw + "\n", encoding="utf-8")
        (out_dir / f"{name}_best.txt").write_text(
            final + "\n", encoding="utf-8")
        print(f"  最终 {len(final.split())} 词, "
              f"耗时 {time.time()-t0:.0f}s → {out_dir / (name + '_best.txt')}",
              flush=True)
    print("\n完成", flush=True)


if __name__ == "__main__":
    main()
