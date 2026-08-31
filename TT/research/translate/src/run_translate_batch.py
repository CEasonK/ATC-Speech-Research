"""批量翻译：单次加载 Qwen，跑完 constrained/free × a/b 四个组合（省去重复 16min 加载）。

输出与 run_translate.py 完全兼容：results/<tag>/qwen_<variant>.json
用法：python run_translate_batch.py [--model models/qwen2.5-7b-instruct]
"""
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from glossary import numeric_fidelity, term_audit          # noqa: E402
from run_translate import (GLOSSARY_HINT, SYSTEM_BASE, USER_TMPL,  # noqa: E402
                           chat, load_model, parse_numbered, retry_failed)

DEEP = SRC.parent.parent / "deep" / "results"
RES = SRC.parent / "results"


def run_one(tok, mdl, tag, variant):
    en_lines = [l.strip() for l in (DEEP / f"{tag}.txt").read_text().splitlines()
                if l.strip()]
    t0 = time.time()
    zh_lines, raw = None, ""
    body = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(en_lines))
    system = SYSTEM_BASE + ("\n" + GLOSSARY_HINT if variant == "constrained" else "")
    for attempt in range(3):  # 解析失败重试
        txt = chat(tok, mdl, system, USER_TMPL.format(N=len(en_lines), body=body))
        raw = txt
        zh_lines = parse_numbered(txt, len(en_lines))
        if all(z for z in zh_lines):
            break
    parse_sec = round(time.time() - t0, 1)

    ratio_num, det_num = numeric_fidelity(en_lines, zh_lines)
    ratio_term, audit_pack = term_audit(en_lines, zh_lines)
    retries = []
    if ratio_num < 1.0 or audit_pack["misses"]:
        zh_lines, retries = retry_failed(
            tok, mdl, en_lines, zh_lines,
            det_num, audit_pack["misses"])
        ratio_num, det_num = numeric_fidelity(en_lines, zh_lines)
        ratio_term, audit_pack = term_audit(en_lines, zh_lines)

    outdir = RES / tag
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"qwen_{variant}.json"
    result = {
        "source": str(DEEP / f"{tag}.txt"),
        "system": "qwen2.5-7b-instruct", "variant": variant,
        "gen_sec": parse_sec,
        "en": en_lines, "zh": zh_lines,
        "raw_output_preview": raw[:600],
        "unparsed_lines": sum(1 for z in zh_lines if not z),
        "metrics": {"numeric_fidelity": round(ratio_num, 4),
                    "term_hit_rate": round(ratio_term, 4)},
        "numeric_detail": det_num,
        "term_misses": audit_pack["misses"],
        "retries": retries,
    }
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("saved:", outfile, json.dumps(result["metrics"], ensure_ascii=False))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(SRC.parent / "models/qwen2.5-7b-instruct"))
    ARGS = ap.parse_args()

    t0 = time.time()
    tok, mdl = load_model(ARGS.model)
    print(f"[load] done in {time.time()-t0:.0f}s")
    for tag in ("a_final", "b_final"):
        for variant in ("constrained", "free"):
            run_one(tok, mdl, tag, variant)
    print("BATCH_DONE")


if __name__ == "__main__":
    main()
