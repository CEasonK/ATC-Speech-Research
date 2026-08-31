"""M2M100 两用：1) EN->ZH 直译（多系统共识第三方）2) ZH->EN 回译（一致性度量）。

用法：
  python run_m2m.py --tag a_final --dir en2zh     # 对 results/<tag>/ 下 en 做直译
  python run_m2m.py --tag a_final --backto qwen_constrained   # 回译该 zh
输出：results/<tag>/m2m_<...>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from glossary import chr_f  # noqa: E402

MODEL_DIR = SRC.parent / "models" / "m2m100_418M"
RES = SRC.parent / "results"


def load(dev="cuda"):
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tok = M2M100Tokenizer.from_pretrained(str(MODEL_DIR))
    mdl = M2M100ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32)
    mdl = mdl.to(dev).eval()
    return tok, mdl


def translate_batch(tok, mdl, lines, src, tgt, bs=8):
    import torch

    dev = next(mdl.parameters()).device
    tok.src_lang = src
    out = []
    for i in range(0, len(lines), bs):
        batch = lines[i: i + bs]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = mdl.generate(**enc, forced_bos_token_id=tok.get_lang_id(tgt),
                               num_beams=5, max_length=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default="en2zh", choices=["en2zh"])
    ap.add_argument("--backto", default=None,
                    help="results/<tag>/<backto>.json 里的 zh -> en 回译")
    ap.add_argument("--cpu", action="store_true")
    ARGS = ap.parse_args()

    rdir = RES / ARGS.tag
    tok, mdl = load("cpu" if ARGS.cpu else "cuda")

    if ARGS.backto:
        src_json = json.load(open(rdir / f"{ARGS.backto}.json"))
        t0 = time.time()
        back = translate_batch(tok, mdl, src_json["zh"], "zh", "en")
        el = round(time.time() - t0, 1)
        scores = [chr_f(e, b) for e, b in zip(src_json["en"], back)]
        res = {"system": "m2m100_418M-back", "based_on": ARGS.backto,
               "back_en": back, "chrf_per_line": [round(s, 4) for s in scores],
               "chrf_mean": round(sum(scores) / max(len(scores), 1), 4),
               "elapsed_sec": el}
        outfile = rdir / f"m2m_back_{ARGS.backto}.json"
    else:
        src_json_path = rdir / "template.json"  # 仅取 en 行作输入源
        en_lines = json.load(open(src_json_path))["en"]
        t0 = time.time()
        zh = translate_batch(tok, mdl, en_lines, "en", "zh")
        el = round(time.time() - t0, 1)
        res = {"system": "m2m100_418M-direct", "en": en_lines, "zh": zh,
               "elapsed_sec": el}
        outfile = rdir / "m2m_direct.json"

    outfile.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved:", outfile)
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("chrf_mean", "elapsed_sec")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
