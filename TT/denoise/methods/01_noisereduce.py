# -*- coding: utf-8 -*-
"""
降噪方法 m1: 高通滤波 + noisereduce 频谱门限降噪
================================================
依赖: numpy, scipy, noisereduce

接口约定（run_denoise.py 会调用）:
    denoise(y, sr) -> 降噪后的 audio (同形状同采样率)

可通过 DESC 描述方法；如需可调参数，请在下方用模块级常量定义
并提供 DUMP_CONFIG() 返回参数快照。
"""

import numpy as np
from scipy import signal
import noisereduce as nr

DESC = "高通滤波(300Hz) + noisereduce 频谱门限降噪"

HIGH_PASS_CUTOFF = 300      # 高通截止频率 Hz
PROP_DECREASE = 0.8         # 降噪强度 0-1


def _high_pass(audio, sr):
    """4阶巴特沃斯高通滤波，去除低频噪声"""
    nyquist = sr / 2
    b, a = signal.butter(4, HIGH_PASS_CUTOFF / nyquist, btype="high")
    return signal.filtfilt(b, a, audio)


def denoise(y, sr):
    audio = np.asarray(y, dtype=np.float64)
    # 1. 高通滤波
    audio = _high_pass(audio, sr)
    # 2. noisereduce 频谱门限降噪（stationary=True 稳态噪声估计）
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=PROP_DECREASE,
        n_fft=512,
        hop_length=128,
    )
    return audio


def DUMP_CONFIG():
    """参数快照（写入 qc_report 用，便于追溯这版用了什么参数）"""
    return {
        "method": 1,
        "desc": DESC,
        "high_pass_cutoff": HIGH_PASS_CUTOFF,
        "prop_decrease": PROP_DECREASE,
    }