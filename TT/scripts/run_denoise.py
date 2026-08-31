#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频降噪工具
============
对 audio/ 下的原始录音做降噪，输出到 denoise/output/，并生成质量报告到 denoise/qc_report/。

目录约定（见 TT/README）：
  audio/                原始录音，只读不动
  denoise/methods/      降噪方法定义（每个方法一个 .py）
  denoise/output/       降噪产物，命名: <音频>__m<方法编号>__dn<版本>.wav
  denoise/qc_report/    降噪质量报告(.json机器/可读.md)，命名与产物同名，且 json 内含方法参数快照

用法：
  python run_denoise.py                    # 对全部音频、全部方法处理
  python run_denoise.py --audio CYYT_ATIS_a  # 只处理某段音频
  python run_denoise.py --method 1         # 只跑 1 号降噪方法
  python run_denoise.py --version 2        # 指定产物版本号 dn2（不覆盖旧版本）
  python run_denoise.py --list-methods     # 列出可用降噪方法
"""

import re
import sys
import json
import argparse
import importlib.util
from pathlib import Path

# 项目根目录（本文件在 scripts/ 下）
TT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = TT_ROOT / "audio"
METHODS_DIR = TT_ROOT / "denoise" / "methods"
OUTPUT_DIR = TT_ROOT / "denoise" / "output"
QC_DIR = TT_ROOT / "denoise" / "qc_report"


def list_methods():
    """扫描 methods/ 下的方法文件，返回 [(编号, 名称)]。文件名形如 01_noisereduce.py"""
    methods = []
    if not METHODS_DIR.exists():
        return methods
    for p in sorted(METHODS_DIR.glob("*.py")):
        m = re.match(r"(\d+)_(.+)\.py", p.name)
        if m:
            methods.append((int(m.group(1)), m.group(2), p))
    return methods


def load_method(py_path):
    """加载方法模块，要求模块实现 denoise(y, sr) -> audio"""
    spec = importlib.util.spec_from_file_location("dn_method", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "denoise"):
        raise ValueError(f"方法文件 {py_path.name} 缺少 denoise(y, sr) 函数")
    return mod


def main():
    parser = argparse.ArgumentParser(description="音频降噪")
    parser.add_argument("--audio", default=None, help="只处理指定音频(不含扩展名)，默认全部")
    parser.add_argument("--method", type=int, default=None, help="只用某编号的方法，默认全部")
    parser.add_argument("--version", type=int, default=None, help="指定版本号，默认自动递增")
    parser.add_argument("--list-methods", action="store_true", help="列出可用降噪方法并退出")
    args = parser.parse_args()

    # 列出方法
    methods = list_methods()
    if args.list_methods or not methods:
        print("可用降噪方法:")
        for num, name, path in methods:
            desc = ""
            mod = load_method(path)
            desc = getattr(mod, "DESC", "")
            print(f"  m{num}: {name}  --  {desc}")
        if not methods:
            print("  (无方法文件，请在 denoise/methods/ 下添加)")
        return

    # 输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    # 音频文件（只取 .wav，mp3 为冗余格式避免重复处理）
    audio_files = sorted(
        p for p in AUDIO_DIR.iterdir()
        if p.suffix.lower() == ".wav" and not p.name.startswith(".")
    )
    if args.audio:
        all_stems = [p.stem for p in audio_files]
        audio_files = [p for p in audio_files if p.stem == args.audio]
        if not audio_files:
            print(f"未找到音频: {args.audio}（audio/ 下有: {all_stems}）")
            return

    print("=" * 60)
    print("音频降噪")
    print("=" * 60)

    # 方法模块只加载一次 (exec_module 较重, 不在 音频×方法 循环内重复)
    loaded = [(num, name, load_method(path)) for num, name, path in methods]

    for ap in audio_files:
        print(f"\n▶ 原始音频: {ap.name}")
        audio, sr = _load_audio(ap)

        for num, name, mod in loaded:
            if args.method is not None and num != args.method:
                continue

            denoised = mod.denoise(audio, sr)

            # 版本号
            version = args.version or _next_version(ap.stem, num)
            out_name = f"{ap.stem}__m{num}__dn{version}.wav"
            out_path = OUTPUT_DIR / out_name
            _save_audio(out_path, denoised, sr)

            # 质量报告（json 机器可读 + md 人类可读）
            qc = _compute_qc(audio, denoised, sr)
            qc["audio"] = ap.stem
            qc["method"] = num
            qc["method_name"] = name
            qc["version"] = version
            qc["input"] = ap.name
            qc_stem = Path(out_name).stem
            qc_path = QC_DIR / (qc_stem + ".json")
            # 若方法提供了 DUMP_CONFIG()，把参数快照一并写入，便于追溯这版用了什么参数
            dump_cfg = getattr(mod, "DUMP_CONFIG", None)
            if callable(dump_cfg):
                qc["config"] = dump_cfg()
            qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_qc_md(QC_DIR / (qc_stem + ".md"), qc, out_name)

            print(f"  [m{num}] {name}: 输出 {out_name}")
            print(f"         SNR {qc['snr_original_db']:.1f} → {qc['snr_denoised_db']:.1f} dB")

    print("\n✅ 降噪完成")


def _load_audio(path):
    """读音频为单声道 16k。mp3/flac 用 librosa；wav 优先 soundfile。"""
    import soundfile as sf
    import librosa
    try:
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
        return audio, sr
    except Exception:
        audio, sr = sf.read(str(path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio, sr


def _save_audio(path, audio, sr):
    import soundfile as sf
    peak = max(abs(audio)) if len(audio) else 1.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)  # 防止削波
    sf.write(str(path), audio, sr)


def _next_version(audio_stem, method_num):
    """自动计算下一个版本号 dn"""
    max_v = 0
    for p in OUTPUT_DIR.glob(f"{audio_stem}__m{method_num}__dn*.wav"):
        m = re.search(r"__dn(\d+)\.wav", p.name)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def _compute_qc(orig, denoised, sr):
    """计算质量指标：底噪、整体RMS、峰值、SNR、动态范围"""
    import numpy as np
    qc = {}

    def stats(x):
        x = np.asarray(x, dtype=np.float64)
        rms = np.sqrt(np.mean(x**2))
        peak = np.max(np.abs(x))
        q = {
            "rms_db": 20 * np.log10(rms) if rms > 1e-12 else -100,
            "peak_db": 20 * np.log10(peak) if peak > 1e-12 else -100,
        }
        return q

    so = _noise_est(orig, sr)
    sd = _noise_est(denoised, sr)
    qc["noise_db_orig"] = so
    qc["noise_db_denoised"] = sd
    # 每个输入只算一次 stats (避免重复 rms/peak)
    qo, qd = stats(orig), stats(denoised)
    qc["overall_db_orig"] = qo["rms_db"]
    qc["overall_db_denoised"] = qd["rms_db"]
    qc["peak_db_orig"] = qo["peak_db"]
    qc["peak_db_denoised"] = qd["peak_db"]
    qc["snr_original_db"] = qc["overall_db_orig"] - qc["noise_db_orig"]
    qc["snr_denoised_db"] = qc["overall_db_denoised"] - qc["noise_db_denoised"]
    return qc


def _grade_label(value, threshold_good, threshold_ok, label_good, label_ok, label_bad):
    """根据阈值返回评级: 良好 / 一般 / 差"""
    if value >= threshold_good:
        return label_good
    if value >= threshold_ok:
        return label_ok
    return label_bad


def _write_qc_md(path, qc, out_name):
    """生成人类可读的质量报告 .md"""
    snr = qc["snr_denoised_db"]

    noise_grade = _grade_label(-qc["noise_db_denoised"], 40, 30, "底噪很干净", "还有一些底噪", "底噪大，会掩盖语音")
    snr_grade = _grade_label(snr, 25, 15, "优秀，识别应该很准", "中等，识别基本可用", "很低，语音会被噪声淹没")
    peak = qc["peak_db_denoised"]
    clip_warn = "⚠ 峰值接近 0 dB，有削波风险！语音可能被削掉。可降低放大系数。" if peak > -2 else "无削波风险"

    lines = []
    lines.append(f"# 降噪质量报告")
    lines.append("")
    lines.append(f"- 降噪产物：`{out_name}`")
    lines.append(f"- 输入音频：`{qc.get('input','')}`")
    lines.append(f"- 方法：m{qc['method']} · {qc['method_name']}（版本 dn{qc['version']}）")
    lines.append("")
    lines.append("## 指标对比（dB，越大越好，除非标注）")
    lines.append("")
    lines.append("| 指标 | 降噪前 | 降噪后 | 怎么理解 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 底噪 noise floor | {qc['noise_db_orig']:.1f} | {qc['noise_db_denoised']:.1f} | 越负越安静。降噪后应更负 |")
    lines.append(f"| 整体响度 RMS | {qc['overall_db_orig']:.1f} | {qc['overall_db_denoised']:.1f} | 语音整体音量。合理降噪不应大幅下降 |")
    lines.append(f"| 峰值 Peak | {qc['peak_db_orig']:.1f} | {qc['peak_db_denoised']:.1f} | 最大振幅。接近 0 会削波 |")
    lines.append(f"| 信噪比 SNR | {qc['snr_original_db']:.1f} | {qc['snr_denoised_db']:.1f} | 信号比噪声高多少，最重要的指标 |")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 信噪比：{snr_grade}（{snr:.1f} dB）")
    lines.append(f"- 底噪：{noise_grade}")
    lines.append(f"- {clip_warn}")
    lines.append("")
    lines.append("### 怎么才算“降噪成功”？")
    lines.append("理想的三条：**① SNR 明显提升（比降噪前高 5 dB 以上），② 语音响度没有明显变闷/变小，③ 不削波、不出现失真**。")
    lines.append("如果 SNR 提上去了但语音变得听不清/变细，说明方法过度了（把语音也砍了），需要调小强度。")
    lines.append("")
    lines.append("### 想进一步优化，可以改哪？")
    lines.append("去 `denoise/methods/01_noisereduce.py`（或复制一个新方法文件）：")
    lines.append("- `PROP_DECREASE`：降噪强度（0~1）。太大语音会削弱，太小降不干净。")
    lines.append("- `HIGH_PASS_CUTOFF`：高通截止频率。增大可去更多低频隆隆声。")
    lines.append("- 改成 `stationary=False`：适合噪声不平稳的情况。")
    lines.append("改完用 `--version 1` 重新跑，会生成下一版 `dn2/dn3...`，不会覆盖，可以来回对比。")
    path.write_text("\n".join(lines), encoding="utf-8")


def _noise_est(audio, sr):
    """用最低帧能量估底噪(dB)"""
    import numpy as np
    frame = int(sr * 0.05)
    hop = int(sr * 0.025)
    x = np.asarray(audio, dtype=np.float64)
    rms_list = []
    for i in range(0, max(1, len(x) - frame), hop):
        f = x[i:i + frame]
        r = np.sqrt(np.mean(f**2))
        rms_list.append(20 * np.log10(r) if r > 1e-12 else -100)
    return float(np.percentile(rms_list, 5)) if rms_list else -100.0


if __name__ == "__main__":
    main()