"""P4e-b v7 补跑：v3 裁判（atc 部分已从 v7 log 提取）。turbo 若可用则一并跑。
用法: HF_HUB_OFFLINE=1 python exp/adjudicate_v7b.py
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from nll_scorer import NLLScorer
from adjudicate_v7 import CONTESTS, CH, run

JUDGES = [("v3", "openai/whisper-large-v3")]
TURBO_DIR = Path.home() / ".cache/huggingface/hub/models--tclin--whisper-large-v3-turbo-atcosim-finetune"
snaps = list((TURBO_DIR / "snapshots").glob("*")) if (TURBO_DIR / "snapshots").exists() else []
if snaps and any((s / "model.safetensors").exists() for s in snaps):
    JUDGES.append(("turbo_atcosim", "tclin/whisper-large-v3-turbo-atcosim-finetune"))
print("judges:", [j[0] for j in JUDGES], flush=True)


def main():
    results = []
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        sc = NLLScorer(mdir)
        results += run(sc, tag)
        del sc
        torch.cuda.empty_cache()

    outf = DEEP / "results" / "adjudication_v7b.json"
    outf.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
