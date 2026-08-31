#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频质量检查
============
对指定音频计算质量指标并打印/保存（底噪、整体RMS、峰值、SNR、动态范围）。

用法：
  python qc_check.py audio/CYYT_ATIS_a.wav
  python qc_check.py denoise/output/CYYT_ATIS_a__m1__dn1.wav --save  # 保存json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def analyze_audio(path):
    import librosa
    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    duration = len(audio) / sr

    # 底噪：用最低5%帧能量估计
    frame = int(sr * 0.05)
    hop = int(sr * 0.025 )
    rms_list = []
    for i in range(0, max(1, len(audio) - frame), hop):
        f = audio[i:i + frame]
        r = np.sqrt(np.mean(f ** 2))
        rms_list.append(20 * np.log10(r) if r > 1e-12 else -100)
    noise_db = float(np.percentile(rms_list, 5))

    rms = np.sqrt(np.mean(audio ** 2))
    overall_db = 20 * np.log10(rms) if rms > 1e-12 else -100
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak) if peak > 1e-12 else -100
    snr_db = overall_db - noise_db
    dynamic_range = peak_db - noise_db

    return {
        "file": str(path),
        "duration_sec": round(duration, 1),
        "sample_rate": sr,
        "noise_db": round(noise_db, 1),
        "overall_db": round(overall_db, 1),
        "peak_db": round(peak_db, 1),
        "snr_db": round(snr_db, 1),
        "dynamic_range_db": round(dynamic_range, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="音频质量检查")
    parser.add_argument("files", nargs="+", help="音频文件路径")
    parser.add_argument("--save", action="store_true", help="保存结果到 denoise/qc_report/")
    args = parser.parse_args()

    qc_dir = Path(__file__).resolve().parent.parent / "denoise" / "qc_report"

    for f in args.files:
        if not Path(f).exists():
            print(f"⚠ 找不到: {f}")
            continue
        qc = analyze_audio(f)
        print("\n" + "=" * 50)
        print(f"🎵 {Path(f).name}")
        print("=" * 50)
        print(f"  时长:      {qc['duration_sec']} 秒")
        print(f"  采样率:    {qc['sample_rate']} Hz")
        print(f"  底噪:      {qc['noise_db']} dB")
        print(f"  整体RMS:   {qc['overall_db']} dB")
        print(f"  峰值:      {qc['peak_db']} dB")
        print(f"  SNR:       {qc['snr_db']} dB")
        print(f"  动态范围:  {qc['dynamic_range_db']} dB")
        if qc['snr_db'] < 5:
            print(f"  ⚠ 信噪比极低，识别困难")
        elif qc['snr_db'] < 15:
            print(f"  ⚠ 信噪比较低，识别可能受限")
        elif qc['snr_db'] < 25:
            print(f"  ✓ 信噪比中等，识别应该可以")
        else:
            print(f"  ✓ 信噪比良好，识别应该没问题")

        if args.save:
            # 原始录音放 raw/ 子目录，降噪产物(文件名含 __m..__dn..)放根目录，避免混淆
            stem = Path(f).stem
            out_dir = qc_dir / ("raw" if "__m" not in stem else ".")
            out = out_dir / (stem + ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  已保存: {out}")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()