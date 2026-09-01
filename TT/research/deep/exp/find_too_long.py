"""离线定位毒候选：token 数 + 4 prompt > 448 会触发 WhisperPositionalEmbedding 越界。
无需 GPU，秒级完成。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nll_scorer import NLLScorer, normalize_text

MODEL = "/siyuan/FunASR_extracted/FunASR-main/TT/models/whisper-large-v3-finetuned-for-ATC"
POOL = Path("/siyuan/FunASR_extracted/FunASR-main/TT/research/deep/results/candidate_pool.json")

# 只要 tokenizer，不加载模型权重
proc = NLLScorer.__new__(NLLScorer)
from transformers import WhisperProcessor
proc.tok = WhisperProcessor.from_pretrained(MODEL).tokenizer

pool = json.loads(POOL.read_text())
LIMIT = 448 - 4  # decoder 位置嵌入上限 - prompt 长度

print(f"pool={len(pool)} candidates, token limit={LIMIT}")
bad = []
for c in pool:
    n_tok = len(proc.tok.encode(normalize_text(c["text"]), add_special_tokens=False))
    c["_n_tok"] = n_tok
    if n_tok > LIMIT:
        bad.append(c)

bad.sort(key=lambda c: -c["_n_tok"])
print(f"\n=== TOO LONG (>448 with prompt): {len(bad)} candidates ===")
for c in bad:
    print(f"  {c['id']:40s} audio={c['audio']:20s} words={c['words']:4d} tokens={c['_n_tok']}  src={c['source']}")

# 分布概览
toks = sorted(c["_n_tok"] for c in pool)
import numpy as np
print(f"\ntoken distribution: min={toks[0]} p50={int(np.percentile(toks,50))} "
      f"p90={int(np.percentile(toks,90))} p99={int(np.percentile(toks,99))} max={toks[-1]}")

# 保存修复方案所需的清单
out = POOL.parent / "too_long_candidates.json"
out.write_text(json.dumps([{"id": c["id"], "n_tok": c["_n_tok"]} for c in bad], indent=1))
print("saved:", out)
