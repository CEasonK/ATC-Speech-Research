"""Offline 上限参照：faster-whisper CT2 全音频 beam 离线转写。
输出目录结构与流式实验一致（events.jsonl 的 emit_t 全部=dur，
evaluate_run 兼容；WER 为关注重点，latency 无意义）。

用法：python run_offline.py <wav> <outdir> [--ct2 _ct2_v3] [--beams 5] [--prompt/--noprompt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("outdir")
    ap.add_argument("--ct2", default="_ct2_v3")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--prompt", action="store_true")
    a = ap.parse_args()

    x, sr = sf.read(a.wav, dtype="float32")
    dur = len(x) / sr
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from faster_whisper import WhisperModel

    base = SRC.parent / "downloads" / a.ct2
    fw = WhisperModel(str(base.resolve()), device="cuda", compute_type="float16")
    prompt = (SRC / "static_prompt_atc.txt").read_text().strip() if a.prompt else None

    t0 = time.time()
    segs, info = fw.transcribe(
        x, language="en", beam_size=a.beams, word_timestamps=True,
        condition_on_previous_text=True, initial_prompt=prompt, vad_filter=False,
    )
    parts, events = [], []
    for st in segs:
        if st.text.strip():
            parts.append(st.text.strip())
        for w in st.words:
            events.append({"emit_audio_t": round(w.end, 3), "word": w.word.strip(),
                           "att_start": round(w.start, 3), "att_end": round(w.end, 3)})
    wall = time.time() - t0
    text = " ".join(parts)
    (outdir / "transcript_final.txt").write_text(text + "\n")
    with open(outdir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (outdir / "snapshots.jsonl").write_text("")
    json.dump({"engine": "offline_ct2", "ct2": str(base), "beams": a.beams,
               "prompt": bool(a.prompt), "lang": info.language,
               "dur": dur, "wall_sec": round(wall, 2), "rtf": round(wall / dur, 4)},
              open(outdir / "meta.json", "w"), ensure_ascii=False, indent=1)
    print("RTF:", round(wall / dur, 4))
    print("FINAL:", text[:400])


if __name__ == "__main__":
    main()
