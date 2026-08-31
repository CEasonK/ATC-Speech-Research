#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-ASR 识别工具（基础版）
===========================
对指定音频用 Qwen3-ASR-1.7B 直接识别，看结果如何。
输出纯文本到 results/Qwen3ASR/。

用法：
  python run_qwen.py CYYT_ATIS_a                 # 识别 audio/CYYT_ATIS_a.wav
  python run_qwen.py CYYT_ATIS_a --in denoise/output/CYYT_ATIS_a__m1__dn1.wav
  python run_qwen.py --all                       # 识别 audio/ 全部录音
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch

# 模型路径: 默认走项目约定位置, 可用环境变量 QWEN_ASR_MODEL 覆盖 (换机时设这个)
MODEL_PATH = Path(
    os.environ.get("QWEN_ASR_MODEL", "/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B"))
TT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = TT_ROOT / "audio"
RESULT_DIR = TT_ROOT / "results" / "Qwen3ASR"

# 语言：None = 自动检测（同段里英日/多语都行）；也可用 --lang 手动覆盖
LANGUAGE = None


def load_model():
    from qwen_asr import Qwen3ASRModel
    print(f"加载模型: {MODEL_PATH}")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        max_inference_batch_size=1,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 基础识别")
    parser.add_argument("audio", nargs="?", help="音频名(不含扩展名)，来自 audio/")
    parser.add_argument("--in", dest="in_path", default=None,
                        help="指定完整音频文件路径（可指向降噪产物）")
    parser.add_argument("--all", action="store_true", help="识别 audio/ 全部录音")
    parser.add_argument("--lang", default=LANGUAGE,
                        help="手动指定语言，如 English/Japanese；默认不传=自动检测")
    args = parser.parse_args()
    language = args.lang

    model = load_model()
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
        results = model.transcribe(audio=str(src), language=language)
        if not results:
            # 空结果 (弱信号被 no-speech 检测判无语音): 写空文件并提示, 不崩溃
            print("⚠ 模型返回空 (可能 no-speech 检测误判; 已知语言请用 --lang English)")
            audio_name = extract_audio_name(src)
            out_dir = RESULT_DIR / audio_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.txt").write_text("", encoding="utf-8")
            continue
        text = results[0].text.strip()
        # 结果按音频分子目录存放，避免不同音频结果互扰
        audio_name = extract_audio_name(src)
        out_dir = RESULT_DIR / audio_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "result.txt"
        out_file.write_text(text + "\n", encoding="utf-8")
        # 同步写 json（含元数据），与 ATC-Whisper 结果格式对齐
        out_json = out_dir / "result.json"
        out_json.write_text(json.dumps({
            "audio": src.name,
            "model": "Qwen3-ASR-1.7B",
            "language": language or "auto",
            "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "text": text,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果: {text[:200]}...")
        print(f"已保存: {out_file}")
        print(f"已保存: {out_json}")

    print("\n✅ 完成")


def extract_audio_name(src: Path) -> str:
    """从音频路径提取音频名。若为降噪产物(含__m..__dn..)，取 __ 之前的基名。"""
    raw = src.stem
    return raw.split("__")[0]


if __name__ == "__main__":
    main()