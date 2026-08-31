"""整篇对照：任意 transcript_final.txt vs deep 权威终稿（atoks 归一口径）。

用法：python cmp_ref.py <transcript.txt> [a|b]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REF_DIR, atoks  # noqa: E402
from metrics import levenshtein  # noqa: E402


def main():
    hyp_path = Path(sys.argv[1])
    tag = (sys.argv[2] if len(sys.argv) > 2 else "a").strip().lower()
    ref_lines = (REF_DIR / f"{tag}_final.txt").read_text().splitlines()
    ref = " ".join(x.strip() for x in ref_lines if x.strip())
    hyp = hyp_path.read_text()
    r_toks, h_toks = atoks(ref), atoks(hyp)
    dist = levenshtein(h_toks, r_toks)
    wer = dist / max(len(r_toks), 1)
    from difflib import SequenceMatcher

    sm = SequenceMatcher(a=h_toks, b=r_toks, autojunk=False)
    matched = sum(i2 - i1 for tag_, i1, i2, _, _ in sm.get_opcodes() if tag_ == "equal")
    match_ratio = round(matched / max(len(h_toks), 1), 3)
    print(f"== {hyp_path.name} vs {tag}_final")
    print(f"WER={wer:.4f}  (dist={dist}/{len(r_toks)})  hyp={len(h_toks)}toks"
          f"  match_ratio={match_ratio}")


if __name__ == "__main__":
    main()
