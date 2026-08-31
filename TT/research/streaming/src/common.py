"""共享工具：音频加载、参考文本、目录约定。"""
import json
import os
from pathlib import Path

import soundfile as sf

TT_ROOT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
STREAMING_ROOT = Path(__file__).resolve().parent.parent
RESULTS = STREAMING_ROOT / "results"
AUDIO = TT_ROOT / "audio"
REF_DIR = TT_ROOT / "research" / "deep" / "results"

AUDIO_FILES = {
    "CYYT_ATIS_a": AUDIO / "CYYT_ATIS_a.wav",
    "CYYT_ATIS_b": AUDIO / "CYYT_ATIS_b.wav",
}


def load_audio(name="CYYT_ATIS_a"):
    """16k 单声道 float32。"""
    x, sr = sf.read(str(AUDIO_FILES[name]), dtype="float32")
    if sr != 16000:
        import torchaudio

        x = torchaudio.functional.resample(
            torch.from_numpy(x).float(), sr, 16000
        ).numpy()
    return x, 16000


import torch  # noqa: E402  (放在函数外也行，这里保持简单)


def load_ref(name="CYYT_ATIS_a"):
    """deep 权威终稿参考文本（列表 of 行）。"""
    key = "a_final" if name.endswith("a") else "b_final"
    lines = (REF_DIR / f"{key}.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def ref_text(name="CYYT_ATIS_a"):
    return " ".join(load_ref(name))


def norm_text(s):
    """WER 归一化：小写、去标点、保数字读法原样（deep 终稿即字母拼读，不做阿拉伯数字转换）。"""
    import re

    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(s.split())


_ICAO_DIGIT = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# 常见等价词归并（小写域）
_ALIAS = {
    "st": "saint",
    "st.": "saint",
    "john's": "johns",
    "ft": "feet",
    "hpa": "",
}


def norm_asr(s):
    """跨风格 ASR 归一化：norm_text + 阿拉伯数字逐位 ICAO 展开（240→two four zero）+ 别名归并。
    使 whisper 风格输出与 deep 字母拼读终稿可公平对比 WER。"""
    import re

    s = norm_text(s)

    def repl_num(m):
        body = m.group(0)
        out = []
        for ch in body:
            if ch.isdigit():
                out.append(_ICAO_DIGIT[ch])
            elif ch == ".":
                out.append("decimal")
        return " ".join(out)

    # 连续数字（含小数点连接）逐位展开
    s = re.sub(r"\d+(?:\.\d+)?", repl_num, s)
    toks = [_ALIAS.get(t, t) for t in s.split()]
    return " ".join(t for t in toks if t)


def atoks(s):
    """norm_asr 后的 token 列表。"""
    return norm_asr(s).split()


def tokens(s):
    return norm_text(s).split()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def exp_dir(engine, audio, **kw):
    """实验输出目录：results/<engine>/<audio>[_k=v...]"""
    name = audio
    for k, v in kw.items():
        name += f"_{k}{v}"
    d = RESULTS / engine / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_audio(x, chunk_sec=0.5, hop=None):
    """把波形切成 (chunk, is_final) 序列。hop 默认=chunk（无重叠，真实流式）。"""
    hop = hop or chunk_sec
    n = len(x)
    i = 0
    while i < n:
        seg = x[i : i + int(chunk_sec * 16000)]
        yield seg, (i + int(chunk_sec * 16000) >= n), i / 16000.0
        i += int(hop * 16000)
