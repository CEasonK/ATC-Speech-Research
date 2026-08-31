#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATC-Whisper 识别工具（基础版）
==============================
对指定音频用 whisper-large-v3-finetuned-for-ATC 直接识别，看结果如何。
只实现基础功能：滑窗转录长音频，输出纯文本到 results/ATC_Whisper/。

用法：
  python run_atc_whisper.py CYYT_ATIS_a              # 识别 audio/CYYT_ATIS_a.wav
  python run_atc_whisper.py CYYT_ATIS_a --in denoise/output/CYYT_ATIS_a__m1__dn1.wav
  python run_atc_whisper.py --all                    # 识别 audio/ 全部录音
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-large-v3-finetuned-for-ATC"
TT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = TT_ROOT / "audio"
RESULT_DIR = TT_ROOT / "results" / "ATC_Whisper"

WINDOW = 30          # Whisper 特征窗口 30s
OVERLAP = 2          # 滑窗重叠 2s，避免句子截断在窗口边缘


def load_model():
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    print(f"加载模型: {MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16).to("cuda")
    model.eval()
    return model, processor


def transcribe(model, processor, audio_path, language=None):
    import librosa
    audio, sr = librosa.load(str(audio_path), sr=16000)
    total = len(audio) / sr
    print(f"  音频: {Path(audio_path).name}  ({total:.1f}s)  语言: {language or '自动检测'}")

    parts = []
    t = 0.0
    while t < total:
        start = int(t * sr)
        end = int(min(t + WINDOW, total) * sr)
        seg = audio[start:end]
        if len(seg) < sr * 0.2:
            break
        inputs = processor(seg, sampling_rate=16000, return_tensors="pt")
        gen_kwargs = {"language": language, "task": "transcribe"}
        with torch.no_grad():
            gen = model.generate(
                inputs.input_features.to("cuda", torch.float16),
                **gen_kwargs)
        txt = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if txt:
            parts.append(txt)
        t += WINDOW - OVERLAP

    return " ".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser(description="ATC-Whisper 基础识别")
    parser.add_argument("audio", nargs="?", help="音频名(不含扩展名)，来自 audio/")
    parser.add_argument("--in", dest="in_path", default=None,
                        help="指定完整音频文件路径（可指向降噪产物）")
    parser.add_argument("--all", action="store_true", help="识别 audio/ 全部录音")
    parser.add_argument("--lang", default=None,
                        help="手动指定语言，如 en/ja；默认不传=自动检测")
    args = parser.parse_args()

    model, processor = load_model()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.in_path:
        sources = [Path(args.in_path)]
    elif args.all:
        sources = sorted(AUDIO_DIR.glob("*.wav"))
    else:
        if not args.audio:
            print("请指定音频名，或加 --all，或用 --in 指定文件")
            sys.exit(1)
        sources = [AUDIO_DIR / f"{args.audio}.wav"]

    for src in sources:
        if not src.exists():
            print(f"⚠ 找不到: {src}")
            continue
        print(f"\n=== 识别 {src.name} ===")
        text = transcribe(model, processor, src, language=args.lang)
        # 结果按音频分子目录存放，避免不同音频结果互扰
        audio_name = extract_audio_name(src)
        out_dir = RESULT_DIR / audio_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "result.txt"
        out_file.write_text(text.strip() + "\n", encoding="utf-8")
        # 同步写 json（含元数据），与 Qwen 结果格式对齐
        out_json = out_dir / "result.json"
        out_json.write_text(json.dumps({
            "audio": src.name,
            "model": "whisper-large-v3-finetuned-for-ATC",
            "language": args.lang or "auto",
            "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "text": text.strip(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果: {text.strip()[:200]}...")
        print(f"已保存: {out_file}")
        print(f"已保存: {out_json}")

    print("\n✅ 完成")


def extract_audio_name(src: Path) -> str:
    """从音频路径提取音频名。若为降噪产物(含__m..__dn..)，取 __ 之前的基名。"""
    return src.stem.split("__")[0]


if __name__ == "__main__":
    main()