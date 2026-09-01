"""RJTT 逐词时间对齐轮（用户触发：部分词"怎么听都听不出来"，要求识别仔细对应）。

方法：faster-whisper（CT2 转换版），word_timestamps=True，
  A) 全曲解码（vad_filter=False, condition_on_previous_text=False）
  B) 9 个已知段切片分别解码（时间戳加回 manifest 偏移）
引擎：v3 = openai/whisper-large-v3（中立裁判）；atc = ATIS 微调版（域裁判）
输出 results/rjtt_word_align.json：每词 (t0,t1,word,prob,engine)
"""
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

DEEP = Path(__file__).resolve().parents[1]
TT = DEEP.parents[1]
SEG = DEEP / "segments" / "RJTT_CONTROL"
FULL = TT / "audio" / "RJTT_CONTROL.wav"
OUTJ = DEEP / "results" / "rjtt_word_align.json"

MODELS = {
    "v3": str(DEEP / "segments" / "_ct2_v3"),
    "atc": str(DEEP / "segments" / "_ct2_atc"),
}

manifest = json.loads((SEG / "manifest.json").read_text())
OFFS = {f"seg{k:02d}": m["t0"] for k, m in enumerate(manifest)}


def run(engine, model, src, path, offset=0.0):
    segs, info = model.transcribe(
        str(path), language="en", beam_size=5,
        word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False)
    n = 0
    for s in segs:
        for w in s.words:
            yield {"engine": engine, "src": src, "sid": s.id,
                   "t0": round(w.start + offset, 2), "t1": round(w.end + offset, 2),
                   "word": w.word.strip(), "prob": round(w.probability, 2)}
            n += 1
    print(f"[{engine}/{src}] {n} words", flush=True)


def main():
    engines = sys.argv[1:] or ["v3"]
    all_words = []
    for eng in engines:
        model = WhisperModel(MODELS[eng], device="cuda", compute_type="float16")
        all_words += list(run(eng, model, "full", FULL))
        for s in sorted(OFFS):
            all_words += list(run(eng, model, s, SEG / f"seg_{s[3:]}.wav",
                                  offset=OFFS[s]))
        del model
    OUTJ.write_text(json.dumps(all_words, ensure_ascii=False, indent=1))
    print(f"saved {len(all_words)} word events -> {OUTJ}")


if __name__ == "__main__":
    main()
