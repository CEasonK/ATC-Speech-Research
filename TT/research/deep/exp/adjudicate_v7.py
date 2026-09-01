"""P4e v7 收尾裁决：报文骨架词。

争议：
  1. opening   : SAINT JOHNS INFORMATION FOXTROT 前有无 THIS IS / CYYT
  2. altimeter : ALTITUDE(解码全票) vs ALTIMETER(ATIS 标准措辞) vs QNH
  3. closing   : WHEN REQUESTING APPROACH vs IF REQUESTED vs WHEN REQUESTED
                 + 尾句 ADVISE ON INITIAL CONTACT YOU HAVE INFORMATION FOXTROT
裁判：三裁判同 v6。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}

openings = [
    "SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
    "THIS IS SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
    "CYYT SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
    "SAINT JOHNS INTERNATIONAL INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU",
]

alti_words = [
    "TEMPERATURE ONE ALTITUDE THREE ZERO TWO THREE APPROACH",
    "TEMPERATURE ONE ALTIMETER THREE ZERO TWO THREE APPROACH",
    "TEMPERATURE ONE QNH THREE ZERO TWO THREE APPROACH",
]

closings = [
    "ONE FIVE AS REQUESTED APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
    "ONE FIVE WHEN REQUESTING APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
    "ONE FIVE IF REQUESTING APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
    "ONE FIVE WHEN REQUESTED APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT",
]

tails = [
    "LANDING AND DEPARTING RUNWAY TWO EIGHT ADVISE ON INITIAL CONTACT YOU HAVE INFORMATION FOXTROT",
    "LANDING AND DEPARTING RUNWAY TWO EIGHT THAT YOU HAVE INFORMATION FOXTROT",
    "LANDING AND DEPARTING RUNWAY TWO EIGHT INFORM ATC THAT YOU HAVE INFORMATION FOXTROT",
]

CONTESTS = [
    ("opening", ["CYYT_ATIS_a", "CYYT_ATIS_b"], "WEATHER AT ZERO TWO ZERO ZERO", openings, 5.0),
    ("alti_word_a", ["CYYT_ATIS_a"], "THREE ZERO TWO THREE APPROACH RNAV ZULU", alti_words, 5.0),
    ("alti_word_b", ["CYYT_ATIS_b"], "THREE ZERO THREE THREE APPROACH RNAV ZULU", alti_words, 5.0),
    ("closing", ["CYYT_ATIS_a", "CYYT_ATIS_b"], "RUNWAY TWO EIGHT INFORM CENTER ON FREQUENCY", closings, 5.0),
    ("tail", ["CYYT_ATIS_a", "CYYT_ATIS_b"], "LANDING AND DEPARTING RUNWAY TWO EIGHT", tails, 5.0),
]

JUDGES = [
    ("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
    ("v3", "openai/whisper-large-v3"),
    ("turbo_atcosim", "tclin/whisper-large-v3-turbo-atcosim-finetune"),
]


def run(scorer, tag):
    out = []
    for name, chans, anchor, variants, hw in CONTESTS:
        rec = {"contest": name, "judge": tag, "channels": {}}
        for ch in chans:
            t_anchor = scorer.find_anchor_window(CH[ch], anchor)
            ranked = []
            for v in variants:
                r = scorer.score_constrained(CH[ch], v, t_anchor, half_width=hw)
                ranked.append({"variant": v, **r})
            ranked.sort(key=lambda x: x["score"])
            rec["channels"][ch] = {"anchor_t": t_anchor, "ranked": ranked}
            for r in ranked:
                print(f"[{name}|{tag}|{ch}] {r['score']:.4f}  {r['variant'][:72]}", flush=True)
            print(f"==> {name}|{tag}|{ch}: WIN -> {ranked[0]['variant'][:80]}", flush=True)
        out.append(rec)
    return out


def main():
    results = []
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        try:
            sc = NLLScorer(mdir)
            results += run(sc, tag)
            del sc
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"judge {tag} failed: {e}", flush=True)

    outf = DEEP / "results" / "adjudication_v7.json"
    outf.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
