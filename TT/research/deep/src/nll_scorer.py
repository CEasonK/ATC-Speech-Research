"""NLL 声学似然评分器 —— 无对照文本时代的"耳朵"。

原理（教师强制）：
    Score(text | audio) = min_over_windows  mean_per_token_NLL( text | window )
把候选文本强行喂给 Whisper decoder，看它在某 30s 窗口的声学表征下
解释这段文本需要多少"惊讶度"。文本越贴合音频，NLL 越低。
min-over-windows 解决对齐问题：报文只需被某个窗口完整覆盖。

裁判有效性必须先过 sanity check（exp/sanity_nll.py）才能用于决策。

用法：
    python nll_scorer.py --model <whisper_dir> --audio <wav> --text-file <txt>
"""
import argparse
import json
import math
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000
WIN_S = 30.0
STRIDE_S = 5.0        # 密集窗口：任何 <=25s 的报文必被某窗完整覆盖
ENC_BATCH = 8         # encoder 批大小，防 OOM
DEC_BATCH = 8         # decoder 批大小
MAX_DEC_TOK = 440     # decoder 位置嵌入上限 448 - 4 prompt - 4 余量
                      # 超过则分块计分（device-side assert 预防：越界会毒化 CUDA 上下文）


def normalize_text(t: str) -> str:
    """统一候选文本形态：大写、去标点、压空白。
    实证（sanity v1 vs v2）：ATC 微调模型在大写转写上训练，
    小写输入属 OOD（全体 NLL 暴涨 ~4.7 nats/tok，且被通用英语幻觉文本反超）。
    故约定 = 大写 + 去标点，消除大小写/标点造成的不公平比较。"""
    t = t.upper()
    t = re.sub(r"(?<=\d)\.(?=\d)", " POINT ", t)   # 135.15 -> 135 POINT 15
    t = re.sub(r"[^A-Z0-9'\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class NLLScorer:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        self.device = device
        self.tok = self.processor.tokenizer
        # 手工构建 SOT 提示：<|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        self.prompt = [
            self.tok.convert_tokens_to_ids("<|startoftranscript|>"),
            self.tok.convert_tokens_to_ids("<|en|>"),
            self.tok.convert_tokens_to_ids("<|transcribe|>"),
            self.tok.convert_tokens_to_ids("<|notimestamps|>"),
        ]
        self.eot = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self._cache = {}  # audio_path -> dict(win_starts, enc_states_cpu_fp16)

    # ---------- 音频侧：窗口切分 + encoder 状态缓存 ----------
    def load_audio(self, audio_path: str):
        key = str(audio_path)
        if key in self._cache:
            return self._cache[key]

        wav, _ = librosa.load(key, sr=SR, mono=True)
        total_s = len(wav) / SR
        win_samples = int(WIN_S * SR)
        stride_samples = int(STRIDE_S * SR)

        starts = []
        chunks = []
        s = 0
        while True:
            seg = wav[s : s + win_samples]
            if len(seg) < win_samples:
                seg = np.pad(seg, (0, win_samples - len(seg)))
            starts.append(s / SR)
            chunks.append(seg)
            if s + win_samples >= len(wav):
                break
            s += stride_samples

        feats = self.processor.feature_extractor(
            chunks, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
        )
        input_features = feats.input_features.to(self.device, dtype=torch.float16)

        enc_list = []
        with torch.no_grad():
            for i in range(0, len(chunks), ENC_BATCH):
                out = self.model.model.encoder(
                    input_features=input_features[i : i + ENC_BATCH]
                )
                enc_list.append(out.last_hidden_state.cpu())
        enc_states = torch.cat(enc_list, dim=0)  # (n_win, 1500, d) fp16 cpu

        entry = {"win_starts": starts, "enc": enc_states, "duration": total_s}
        self._cache[key] = entry
        return entry

    # ---------- 文本侧：教师强制 NLL ----------
    @torch.no_grad()
    def _score_ids(self, entry, text_ids: list) -> np.ndarray:
        """单个 token 序列在全部窗口下的 per-token NLL（教师强制）。
        返回 shape=(n_win,) 数组。"""
        enc_all = entry["enc"]
        n_win = enc_all.shape[0]

        dec_in_1 = torch.tensor([self.prompt + text_ids])            # (1, P+T)
        targets_1 = torch.tensor([text_ids + [self.eot]])            # (1, T+1)
        P = len(self.prompt)

        dec_in = dec_in_1.expand(n_win, -1).to(self.device)
        scores = []
        for i in range(0, n_win, DEC_BATCH):
            enc_slice = enc_all[i : i + DEC_BATCH].to(self.device)
            di = dec_in[i : i + DEC_BATCH]
            out = self.model(encoder_outputs=(enc_slice,), decoder_input_ids=di)
            logits = out.logits.float()                               # (b, P+T, V)
            sl = logits[:, P - 1 :]                                   # T+1 个位置: 文本各词+EOT
            tgt = targets_1.expand(sl.shape[0], -1).to(self.device)
            nll = F.cross_entropy(
                sl.transpose(1, 2), tgt, reduction="none"
            )                                                          # (b, T+1)
            scores.extend(nll.mean(dim=1).tolist())
        return np.array(scores)

    @torch.no_grad()
    def score(self, audio_path: str, text: str) -> dict:
        """返回 dict(score, t_start, t_end, n_words, n_chunks)。score 越低越好。
        超长文本（>MAX_DEC_TOK）按 token 块分别计分再取均值——
        对循环广播物理合理：各块可在不同循环里找最佳对齐窗。"""
        entry = self.load_audio(audio_path)
        text_ids = self.tok.encode(normalize_text(text), add_special_tokens=False)

        if len(text_ids) <= MAX_DEC_TOK:
            win_scores = self._score_ids(entry, text_ids)
            k = int(win_scores.argmin())
            t0 = entry["win_starts"][k]
            return {
                "score": float(win_scores[k]),
                "t_start": round(t0, 1),
                "t_end": round(t0 + WIN_S, 1),
                "n_words": len(text.split()),
                "n_windows": len(win_scores),
                "median_score": float(np.median(win_scores)),
                "n_chunks": 1,
            }

        # 超长：分块（按词边界切，避免切断 BPE 词片）
        words = normalize_text(text).split()
        chunks_ids, cur = [], []
        for w in words:
            w_ids = self.tok.encode(w, add_special_tokens=False)
            if cur and len(cur) + len(w_ids) > MAX_DEC_TOK:
                chunks_ids.append(cur)
                cur = []
            cur.extend(w_ids)
        if cur:
            chunks_ids.append(cur)

        chunk_best = [float(self._score_ids(entry, ch).min()) for ch in chunks_ids]
        return {
            "score": float(np.mean(chunk_best)),
            "t_start": -1.0,          # 分块无单一窗口
            "t_end": -1.0,
            "n_words": len(text.split()),
            "n_windows": entry["enc"].shape[0],
            "chunk_mean_score": float(np.mean(chunk_best)),  # 分块路径无 median，勿与 score_constrained 的 median_score 混用
            "n_chunks": len(chunks_ids),
        }

    # ---------- 同窗对立比较（paired-window adjudication） ----------
    def find_anchor_window(self, audio_path: str, anchor_text: str) -> float:
        """用无争议锚片段定位争议词所在窗口。

        返回语义（P2/D3 诚实化）：最佳覆盖锚文本的 **窗起点秒**（WIN_S=30s 窗、
        5s 网格）。这是窗起点，不是锚短语的精确时刻；本函数不做窗内亚窗定位，
        调用方按短语时刻使用时自带 ±(窗长-短语长) 级的不确定性。
        另注：循环广播语料中窗长(30s)≥广播周期时，各窗内容近乎等价，
        argmin 选的是"最强周期"而非唯一位置——定位结论须过
        exp/assemble_final.py::position_test_validity 一类窗长<周期守卫。"""
        entry = self.load_audio(audio_path)
        ids = self.tok.encode(normalize_text(anchor_text), add_special_tokens=False)
        assert len(ids) <= MAX_DEC_TOK, "anchor too long"
        ws = self._score_ids(entry, ids)
        return entry["win_starts"][int(ws.argmin())]

    @torch.no_grad()
    def score_constrained(self, audio_path: str, text: str,
                          t_center: float, half_width: float = 5.0) -> dict:
        """只在 [t_center-half_width, t_center+half_width] 的窗口上计分。
        消除静音窗口假象：对立对全部在同一声学材料上比较。"""
        entry = self.load_audio(audio_path)
        enc_all = entry["enc"]
        keep = [i for i, t0 in enumerate(entry["win_starts"])
                if abs(t0 - t_center) <= half_width]
        assert keep, f"no window near {t_center}"
        text_ids = self.tok.encode(normalize_text(text), add_special_tokens=False)
        dec_in_1 = torch.tensor([self.prompt + text_ids])
        targets_1 = torch.tensor([text_ids + [self.eot]])
        P = len(self.prompt)
        scores = []
        for i in range(0, len(keep), DEC_BATCH):
            ks = keep[i:i + DEC_BATCH]
            enc_slice = enc_all[ks].to(self.device)
            di = dec_in_1.expand(len(ks), -1).to(self.device)
            out = self.model(encoder_outputs=(enc_slice,), decoder_input_ids=di)
            logits = out.logits.float()
            sl = logits[:, P - 1:]
            tgt = targets_1.expand(sl.shape[0], -1).to(self.device)
            nll = F.cross_entropy(sl.transpose(1, 2), tgt, reduction="none")
            scores.extend(nll.mean(dim=1).tolist())
        scores = np.array(scores)
        k = int(scores.argmin())
        return {"score": float(scores[k]),
                "median_score": float(np.median(scores)),
                "t_start": round(entry["win_starts"][keep[k]], 1),
                "n_windows": len(keep)}

    # ---------- 变体批量评分 ----------
    def score_many(self, audio_path: str, texts: list) -> list:
        return [{"text": t, **self.score(audio_path, t)} for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text-file", action="append", default=[],
                    help="可多次传入；每个文件整体作为一条候选")
    ap.add_argument("--text", action="append", default=[], help="直接传文本候选")
    args = ap.parse_args()

    sc = NLLScorer(args.model)
    cands = [Path(p).read_text().strip() for p in args.text_file] + args.text
    results = sc.score_many(args.audio, cands)
    results.sort(key=lambda r: r["score"])
    for r in results:
        print(f"[{r['score']:.4f}] @{r['t_start']:.0f}s ({r['n_words']}w) {r['text'][:80]}...")
    out = Path(args.audio).with_suffix(".nll.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()
