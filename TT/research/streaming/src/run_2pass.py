"""E6 两遍法（最终交付形态）：
Pass1  SimulStreaming(AlignAtt) 流式出草稿（低延迟逐词）
Pass2  句末静音端点触发 → faster-whisper CT2 对该句 beam 精修，
       终稿≈offline 上限，额外延迟仅句尾触发点。

端点检测（句边界）：
  语音帧 = (RMS>8) & (600ms 窗 RMS 包络 CV>0.2)。
  CV（短窗 RMS 的变异系数 std/mean）区分"调制信号 vs 恒定信号"：
  语音逐词起伏 CV≥0.4，而噪声底/未调制载波恒定 CV≈0.07-0.15。
  纯能量阈值会被恒定噪声底污染（p20*2.5 达 479，高于语音帧 p99≈368，
  原始 b 前 224s 即此情形 → 旧版 VAD 永不触发，全片单次精修出幻觉），
  CV 门在 4 个音频（原始 a/b + K4 a/b）上均正确分离，并经逐段数据核验。
  连续语音超 --max_utt 秒强制切句（定稿延迟有界护栏）。

输出（与 evaluate_run.py 兼容的两个实验目录）：
  <out>/            精修终稿轨 transcript_final.txt + events.jsonl（final 词时刻=端点触发 chunk）
  <out>_draftonly/  流式草稿轨（用于单独评 draft WER/LAG）

用法：python run_2pass.py <wav> <outdir> [--stream_model ..pt] [--ct2_dir _ct2_v3]
      [--sil 0.75] [--min_utt 0.35] [--max_utt 30] [--chunk 0.5] [--frame_threshold 25] [--half] ...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parent
PROMPT_TXT = SRC / "static_prompt_atc.txt"


def _chan_of(wav_path):
    """CYYT_ATIS_a_evalK4.wav -> CYYT_ATIS_a（剥掉评测拼接后缀）"""
    import re as _r

    return _r.sub(r"_evalK\d+$", "", Path(wav_path).stem)


FUSE_MODES = ("apply", "report_only")


def resolve_template_path(wav, template_file=None, *,
                          allow_static_prompt_fallback=False, src_dir=None):
    """定位台站模板（P0a-F2）。返回 (path, template_source)。

    模板缺失时默认 fail-fast（SystemExit + 列出全部尝试过的路径）。旧实现
    `cand if cand.exists() else PROMPT_TXT` 会静默回退 static_prompt_atc.txt，
    而该文件实测与 a 信道答案全文逐字节相同（md5 一致）：b / K4 / 新信道
    一旦模板缺失，就拿 a 信道答案当模板做融合，是答案泄漏而非安全降级。
    只有显式 --allow_static_prompt_fallback 才恢复旧回退，并把
    template_source 标成 "static_prompt_FALLBACK" 写进 meta.json 供审计。
    """
    src_dir = Path(src_dir) if src_dir else SRC
    tried = []
    if template_file:
        p = Path(template_file)
        tried.append(p)
        if p.exists():
            return p, "template_file_arg"
        raise SystemExit("[F2 fail-fast] --template_file 指定的模板不存在：\n"
                         f"  - {p}")
    chan = _chan_of(wav)
    for name in (f"{chan}.txt", f"{chan}_full.txt"):
        cand = src_dir.parent / "templates" / name
        tried.append(cand)
        if cand.exists():
            return cand, "templates_dir"
    fallback = src_dir / "static_prompt_atc.txt"
    tried.append(fallback)
    if allow_static_prompt_fallback:
        print("[tpl][WARN] 模板缺失，--allow_static_prompt_fallback 生效：回退 "
              f"{fallback.name}（实测= a 信道答案全文，不是本信道模板）")
        return fallback, "static_prompt_FALLBACK"
    raise SystemExit(
        "[F2 fail-fast] 找不到输入信道对应的台站模板，拒绝静默回退。\n"
        f"  输入：{wav}\n  信道：{chan}\n  已尝试路径：\n"
        + "".join(f"    - {t}\n" for t in tried)
        + "  可选解决：--template_file 指定模板；或补齐 templates/<信道>.txt；"
        "或（仅在你明确接受用 a 信道答案全文充当模板时）显式加 "
        "--allow_static_prompt_fallback。")


def vad_advance_chunk(st, gate, f_limit, *, silence_frames, min_utt_frames,
                      max_utt_frames, frame_ms=10):
    """推进 VAD 状态机到帧号 f_limit（不含），返回本 chunk 触发的定稿事件。

    st 键：in_speech / sp_start_f / sil_run / vad_ptr（原地更新）。
    至多返回一个事件，且触发即 break、vad_ptr 停在触发帧（与旧实现逐位一致，
    下个 chunk 从同一帧续跑）：
      ("finalize", s_sec, e_sec, trigger)
          trigger ∈ {"max_utt", "endpoint", "max_utt_after_silence"}
      ("drop_short", s_sec, e_sec, "noise_gate")   # < min_utt 的门误触，丢弃
      None                                          # 本 chunk 无事件

    P0a-F3：sp_len_frames 一律在 sp_start_f 归位到**当前语音段起点之后**才求值。
    旧实现先在循环顶部用陈旧 sp_start_f 算长度，长静音之后的首个语音帧会带着
    "上一段起点→当前帧"的陈旧长度立刻触发 force-cut；同一轮里 sp_start_f 又被
    更新为 vad_ptr，于是定稿区间 s_sec == e_sec（长度 0）。
    """
    while st["vad_ptr"] < f_limit:
        vad_ptr = st["vad_ptr"]
        is_sp = bool(gate[vad_ptr])
        if is_sp and not st["in_speech"]:
            # F3：新语音段起点必须先于长度求值归位（原来是 vad_ptr 自增前才更新）
            st["in_speech"] = True
            st["sp_start_f"] = vad_ptr
            st["sil_run"] = 0
        # sil_run 取"本帧之前"的值，与原实现一致：语音段内长度语义不变
        sp_len_frames = vad_ptr - st["sil_run"] - st["sp_start_f"]
        if is_sp:
            st["sil_run"] = 0
            if sp_len_frames >= max_utt_frames:
                # 连续语音超 max_utt：强制切句定稿，保证定稿延迟有界
                s_sec = st["sp_start_f"] * frame_ms / 1000
                e_sec = vad_ptr * frame_ms / 1000
                st["in_speech"] = False
                st["sil_run"] = 0
                return ("finalize", s_sec, e_sec, "max_utt")
        elif st["in_speech"]:
            st["sil_run"] += 1
            if st["sil_run"] >= silence_frames \
                    and sp_len_frames >= min_utt_frames:
                # 正常句末端点
                s_sec = st["sp_start_f"] * frame_ms / 1000
                e_sec = (vad_ptr - st["sil_run"]) * frame_ms / 1000
                st["in_speech"] = False
                st["sil_run"] = 0
                return ("finalize", s_sec, e_sec, "endpoint")
            elif sp_len_frames >= max_utt_frames:
                # 语音刚结束且已达 max_utt：立即定稿（不等满 sil）
                s_sec = st["sp_start_f"] * frame_ms / 1000
                e_sec = (vad_ptr - st["sil_run"]) * frame_ms / 1000
                st["in_speech"] = False
                st["sil_run"] = 0
                return ("finalize", s_sec, e_sec, "max_utt_after_silence")
            elif st["sil_run"] >= silence_frames:
                # 拼接 < min_utt：判为门误触（噪声尖峰），丢弃不精修
                s_sec = st["sp_start_f"] * frame_ms / 1000
                e_sec = (vad_ptr - st["sil_run"]) * frame_ms / 1000
                st["in_speech"] = False
                st["sil_run"] = 0
                return ("drop_short", s_sec, e_sec, "noise_gate")
        else:
            st["sil_run"] = 0
        st["vad_ptr"] += 1
    return None


def _fill_times(words):
    """无计时词：在相邻有计时词之间线性插值（_fuse 与 rover 共用）。"""
    known = [i for i, w in enumerate(words) if w["start"] is not None]
    for i, w in enumerate(words):
        if w["start"] is not None:
            continue
        prev_i = max((j for j in known if j < i), default=None)
        next_i = min((j for j in known if j > i), default=None)
        lo = words[prev_i]["end"] if prev_i is not None else \
            (words[next_i]["start"] if next_i is not None else 0.0)
        hi = words[next_i]["start"] if next_i is not None else \
            (words[prev_i]["end"] + 0.3 if prev_i is not None else 0.3)
        group = [j for j in range(len(words))
                 if words[j]["start"] is None
                 and (prev_i is None or j > prev_i)
                 and (next_i is None or j < next_i)]
        gi = group.index(i)
        f0 = (gi) / max(len(group), 1)
        f1 = (gi + 1) / max(len(group), 1)
        w["start"] = round(lo + (hi - lo) * f0, 3)
        w["end"] = round(lo + (hi - lo) * f1, 3)
    return words


def _fuse(tpl_toks, engines, fuse_mode="apply"):
    """模板证词融合。

    - 模板 token：任一引擎对齐证实即带真实计时输出；无引擎证实的标记
      src=tpl（模板持有信息，音频弱读/信道降质未能证实，如 SAINT/INFORM）。
    - 模板偏离：偏离票须 >=2 且来自 >=2 个异族引擎（ct2 vs qwen）。
      同族一致不算独立证词——两个 CT2 同源 whisper 家族错误相关
      （J13：b 信道双 CT2 一致听成 OF REQUESTED，替换掉模板正确的
      WHEN REQUESTED，即"多遍同错"陷阱）。

    fuse_mode（P0a-F1）：
      report_only —— J15 旧行为逐位保留：偏离命中只记
                     dev[i]={"_rejected": best}，输出词恒等于模板词
                     （融合恒等），该位置仍计 template_only。
      apply       —— 偏离命中时用 best 替换该位置输出词，事件改记
                     dev[i]={"_applied": best, "_voters": [引擎idx],
                             "_fams": [族名]}；输出词
                     {"text": best, "src": "fuse", "dev_from": 模板原词,
                      "voters": n, "fams": [...]}，计时置 None 交
                     _fill_times 插值（该位置本就无引擎计时，与
                     template-only 同等处理）。
    stats 口径：deviated = 跨族偏离命中数（两模式一致，历史数字可比）；
    applied = 实际替换数（report_only 恒 0）。apply 下命中位置既不计
    attested 也不计 template_only。
    返回 (text, words, stats)。"""
    from difflib import SequenceMatcher

    def nt(t):
        return t.upper().strip(".,?")

    def fam(k):
        return "qwen" if engines[k].get("family") == "qwen" else "ct2"

    n = len(tpl_toks)
    att = [[] for _ in range(n)]   # 每位置：证实引擎的 (idx, word_idx)
    dev = [None] * n               # 每位置：{非模板token: [证实引擎idx]}
    for k, E in enumerate(engines):
        toks = [nt(t) for t in E["tokens"]]
        sm = SequenceMatcher(None, tpl_toks, toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for o in range(i2 - i1):
                    att[i1 + o].append((k, j1 + o))
            elif tag == "replace" and (i2 - i1) == (j2 - j1):
                for o in range(i2 - i1):
                    if tpl_toks[i1 + o] == toks[j1 + o]:
                        att[i1 + o].append((k, j1 + o))
                    else:
                        dd = dev[i1 + o] or {}
                        dd.setdefault(toks[j1 + o], []).append(k)
                        dev[i1 + o] = dd
    out_words = []
    n_att = n_tpl = n_dev = n_applied = 0
    for i, tok in enumerate(tpl_toks):
        hit = None
        if dev[i] and not att[i]:
            # J15 前提：模板=台站持有真相，无引擎证实的位置默认信任模板。
            # 跨族 >=2 票才认定偏离；apply 模式据此替换，report_only 仅审计。
            best, voters = max(dev[i].items(), key=lambda kv: len(kv[1]))
            fams = {fam(k) for k in voters}
            if len(voters) >= 2 and len(fams) >= 2:
                hit = (best, voters, sorted(fams))
        if hit:
            n_dev += 1  # 两模式口径一致：跨族偏离命中数，供 meta 审计
            dev[i] = ({"_rejected": hit[0]} if fuse_mode == "report_only"
                      else {"_applied": hit[0], "_voters": hit[1],
                            "_fams": hit[2]})
        if hit and fuse_mode != "report_only":
            # F1 apply：替换该位置输出词（既不 attested 也不 template_only）
            out_words.append({"text": hit[0], "start": None, "end": None,
                              "src": "fuse", "dev_from": tok,
                              "voters": len(hit[1]), "fams": hit[2]})
            n_applied += 1
            continue
        if att[i]:
            k, wj = att[i][0]  # 引擎0(v3)优先在前
            E = engines[k]
            if E["words"] and wj < len(E["words"]):
                w = E["words"][wj]
                out_words.append({"text": tok, "start": w["start"],
                                  "end": w["end"], "src": f"eng{k}"})
            else:
                out_words.append({"text": tok, "start": None, "end": None,
                                  "src": f"eng{k}"})
            n_att += 1
        else:
            out_words.append({"text": tok, "start": None, "end": None,
                              "src": "tpl"})
            n_tpl += 1
    stats = {"attested": n_att, "template_only": n_tpl, "deviated": n_dev,
             "applied": n_applied, "fuse_mode": fuse_mode,
             "engines": len(engines)}
    return " ".join(w["text"] for w in out_words), _fill_times(out_words), stats


def build_final_snapshots(events):
    """由 final 定稿事件流构造 final 轨 snapshots（P0a-F4）。

    同一 emit_audio_t（触发精修的 chunk 结束时刻）的所有词合并为一条快照，
    words = 该时刻及之前已定稿的全部词。时间戳来自 **final 定稿事件**，
    不再借用 Pass1 草稿前缀——旧实现两轨写同一份草稿 snapshots，导致
    final 轨 LAG 实测的是草稿延迟。
    """
    snaps, acc = [], []
    for i, e in enumerate(events):
        acc.append(e["word"])
        nxt = events[i + 1] if i + 1 < len(events) else None
        if nxt is None or nxt.get("emit_audio_t") != e.get("emit_audio_t"):
            snaps.append({"t": round(e.get("emit_audio_t"), 2),
                          "words": list(acc)})
    return snaps


def write_track(outdir, *, final_text, events, snapshots, meta):
    """写一个实验目录（transcript_final.txt / events.jsonl / snapshots.jsonl /
    meta.json）。纯 IO，主流程与单测共用（P0a-F4：两轨各写自己的快照与 meta）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "transcript_final.txt").write_text(final_text + "\n")
    with open(outdir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(outdir / "snapshots.jsonl", "w") as f:
        for s in snapshots:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("outdir")
    ap.add_argument("--stream_model", default="../downloads/whisper-atc-openai.pt")
    ap.add_argument("--ct2_dir", default="_ct2_v3")
    ap.add_argument("--chunk", type=float, default=0.5)
    ap.add_argument("--frame_threshold", type=int, default=25)
    ap.add_argument("--beams_stream", type=int, default=1)
    ap.add_argument("--beams_offline", type=int, default=5)
    ap.add_argument("--sil", type=float, default=0.75)
    ap.add_argument("--min_utt", type=float, default=0.35)
    ap.add_argument("--max_utt", type=float, default=30.0,
                    help="连续语音超过该秒数强制切句定稿，防止 VAD 永不触发时定稿延迟无界")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--ct2_dir2", default="_ct2_atc",
                    help="第二精修引擎（ATIS 微调 CT2），作旁证与主引擎做模板证词融合")
    ap.add_argument("--template_file", default=None,
                    help="台站 ATIS 模板(9行正文)；默认按输入文件名自动匹配 templates/<信道>.txt")
    ap.add_argument("--qwen_model", default="/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B",
                    help="Qwen3-ASR 旁证引擎路径；不存在时自动跳过")
    ap.add_argument("--qwen_python", default=None,
                    help="lingbot-map python 路径：qwen_asr 仅装在该环境，用其派生"
                         "常驻 worker 子进程（stdin/stdout 传 wav 路径）")
    ap.add_argument("--no_fusion", action="store_true",
                    help="消融开关：关闭模板证词融合，仅用主引擎 v3 精修")
    ap.add_argument("--no_prompt", action="store_true",
                    help="消融开关：去掉全部文本先验（Pass1 static_prompt 与 "
                         "CT2 initial_prompt 均置空），得到零先验纯声学成绩。"
                         "J16 审计发现无此开关时连 --no_fusion 基线也被模板提示污染")
    ap.add_argument("--rover", action="store_true",
                    help="多引擎 ROVER 投票 + ATIS 词表纠错（atis_lexicon.py "
                         "P4 清洗版：定点真值映射已剔除；保留规则系评测集监督下"
                         "选定的公开词法先验，非零先验，档位见该文件 docstring）。"
                         "需 ct2_dir2 与 qwen；建议与 --no_prompt 同用")
    ap.add_argument("--rover_qwen_gate", type=float, default=0.85,
                    help="自适应仲裁门限：atc/v3 一致率 >= 该值时跳过 qwen"
                         "（R4 省时延；低一致率段才请跨族仲裁）")
    ap.add_argument("--fuse_mode", choices=FUSE_MODES, default="apply",
                    help="模板证词融合的偏离处置（P0a-F1）。apply（新默认）："
                         "跨族>=2 票的模板偏离用引擎词替换该位置输出，事件记 "
                         "dev[i]['_applied']；report_only：完整保留 J15 旧行为——"
                         "偏离只记 dev[i]['_rejected'] 不替换，输出恒等于模板。"
                         "注意默认值已从旧行为(report_only 等价)改为 apply，"
                         "历史 apply 前数字不可直接比较；meta.json 有 fuse_mode")
    ap.add_argument("--allow_static_prompt_fallback", action="store_true",
                    help="模板缺失时允许回退 static_prompt_atc.txt（P0a-F2 旧行为）。"
                         "默认改为 fail-fast：该文件实测=a 信道答案全文，静默回退"
                         "等于把 a 信道答案泄漏进其它信道。回退时 meta.json 记 "
                         'template_source="static_prompt_FALLBACK"')
    ARGS = ap.parse_args()

    x, sr = sf.read(ARGS.wav, dtype="float32")
    assert sr == 16000
    dur = len(x) / sr 
    # 台站模板：按输入名（剥掉 _evalK4 之类后缀）匹配 templates/<名>.txt
    # P0a-F2：模板缺失不再静默回退 static_prompt_atc.txt（=a 信道答案全文），
    # 默认 fail-fast；只有 --allow_static_prompt_fallback 才允许旧回退。
    chan = _chan_of(ARGS.wav)
    tpl_path, template_source = resolve_template_path(
        ARGS.wav, ARGS.template_file,
        allow_static_prompt_fallback=ARGS.allow_static_prompt_fallback)
    prompt_txt = tpl_path.read_text().strip()
    print(f"[tpl] channel={chan} template={tpl_path.name} "
          f"source={template_source}")
    outdir = Path(ARGS.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    import os

    ct2_p = Path(ARGS.ct2_dir)
    if not ct2_p.is_absolute():
        base = SRC.parent / "downloads" / ARGS.ct2_dir
        ct2_p = Path(base).resolve() if base.exists() else \
            (Path(__file__).parent.parent / "downloads" / ARGS.ct2_dir).resolve()
    os.chdir(Path(ARGS.stream_model).resolve().parent)

    # ---------- Pass1 引擎 ----------
    import sys

    sys.path.insert(0, str(SRC))
    import run_simulstreaming as rs

    class A:
        pass

    a = A()
    a.model = str(Path(ARGS.stream_model).resolve())
    a.chunk = ARGS.chunk
    a.frame_threshold = ARGS.frame_threshold
    a.beams = ARGS.beams_stream
    a.audio_max_len = 30.0
    a.audio_min_len = 0.25
    a.static_prompt = None if ARGS.no_prompt else prompt_txt  # J16 零先验消融
    a.half = ARGS.half
    rs.ARGS = a
    online = rs.Online()
    

    # ---------- Pass2 引擎 ----------
    from faster_whisper import WhisperModel

    fw = WhisperModel(str(ct2_p), device="cuda", compute_type="int8_float16")
    # 旁证引擎 1：ATIS 微调 CT2（可选）
    ct2_p2 = Path(ARGS.ct2_dir2)
    if not ct2_p2.is_absolute():
        base2 = SRC.parent / "downloads" / ARGS.ct2_dir2
        ct2_p2 = base2.resolve() if base2.exists() else ct2_p2
    fw2 = WhisperModel(str(ct2_p2), device="cuda", compute_type="int8_float16") \
        if Path(ct2_p2).exists() else None
    # 旁证引擎 2：Qwen3-ASR（可选）。
    # 优先进程内加载；qwen_asr 不在本环境时（仅装于 lingbot-map，其 ctranslate2
    # 在沙箱内 CUDA 初始化失败，无法整跑）用 --qwen_python 派生常驻 worker 子进程。
    qwen = None
    qwen_proc = None
    if Path(ARGS.qwen_model).exists():
        try:
            import torch

            from qwen_asr import Qwen3ASRModel

            qwen = Qwen3ASRModel.from_pretrained(
                ARGS.qwen_model, device_map="auto", dtype=torch.bfloat16,
                max_inference_batch_size=1)
            print("[eng] qwen loaded (in-process)")
        except Exception as ex:  # noqa: BLE001
            if ARGS.qwen_python and Path(ARGS.qwen_python).exists():
                try:
                    import subprocess as _sp

                    qwen_proc = _sp.Popen(
                        [ARGS.qwen_python, str(SRC / "qwen_worker.py"),
                         ARGS.qwen_model],
                        stdin=_sp.PIPE, stdout=_sp.PIPE, text=True, bufsize=1)
                    ready = qwen_proc.stdout.readline()
                    if '"ready": true' in ready:
                        qwen = "worker"
                        print("[eng] qwen worker ready (subprocess)")
                    else:
                        print(f"[eng] qwen worker handshake failed: {ready!r}")
                        qwen_proc.kill()
                        qwen_proc = None
                except Exception as ex2:  # noqa: BLE001
                    print(f"[eng] qwen worker spawn failed: {ex2}")
            else:
                print(f"[eng] qwen load failed and no --qwen_python, "
                      f"fallback 2-engine: {ex}")

    def _qwen_txt(seg):
        """qwen 旁证：进程内对象或子进程 worker，返回文本。"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            sf.write(tf.name, seg, sr)
            if qwen == "worker":
                qwen_proc.stdin.write(tf.name + "\n")
                qwen_proc.stdin.flush()
                resp = json.loads(qwen_proc.stdout.readline())
                return resp.get("text", "")
            return qwen.transcribe(audio=tf.name, language="English")[0].text
    engines_n = 1 + (fw2 is not None) + (qwen is not None)
    print(f"[eng] refine engines = {engines_n} (v3 primary"
          f"{', +atc' if fw2 else ''}{', +qwen' if qwen else ''})")

    # ---------- 端点检测：RMS 包络 CV 调制门 ----------
    frame_ms = 10
    win = int(sr * frame_ms / 1000)
    n_frames = len(x) // win
    rms_all = (
        np.sqrt(
            np.mean(x[: n_frames * win].reshape(n_frames, win).astype(np.float64) ** 2,
                    axis=1)
        )
        * 1000
    )
    # 600ms 平滑窗（k 以帧为单位！1 帧=10ms）下短窗 RMS 的变异系数 CV=std/mean
    cv_smooth_ms = 600
    k = max(3, int(cv_smooth_ms / frame_ms))
    p = k // 2
    ext = np.concatenate([rms_all[:p][::-1], rms_all, rms_all[-p:][::-1]])
    _cs = np.cumsum(ext)
    _cs2 = np.cumsum(ext * ext)
    _m = (_cs[k:] - _cs[:-k]) / k
    _m2 = (_cs2[k:] - _cs2[:-k]) / k
    _std = np.sqrt(np.clip(_m2 - _m * _m, 0, None))
    cv_all = _std / (_m + 1e-6)
    # 语音门：有能量 且 包络被调制（恒定噪声/载波 CV≈0.07-0.15，语音 CV≥0.4）
    gate = (rms_all > 8.0) & (cv_all > 0.2)
    print(f"[vad] gate=RMS>8 & CV600>0.2; voiced {gate.sum() * frame_ms / 1000:.1f}s "
          f"/ {n_frames * frame_ms / 1000:.1f}s")

    silence_frames = int(ARGS.sil * 1000 / frame_ms)
    min_utt_frames = int(ARGS.min_utt * 1000 / frame_ms)
    max_utt_frames = int(ARGS.max_utt * 1000 / frame_ms)

    seg_events_final = []
    finalized_parts = []
    seg_events_draft = []
    draft_parts = []
    snapshots = []
    prefix_words = []

    in_speech = False
    sp_start_f = 0
    sil_run = 0
    pending_seg = None
    vad_ptr = 0
    last_final_f = 0  # 已定稿语音的右边界帧（防止静音尾巴重复精修）
    vad_st = {"in_speech": False, "sp_start_f": 0, "sil_run": 0, "vad_ptr": 0}
    n_empty_finals = 0   # P0a-F3：零/负长度定稿被跳过的次数
    n_noise_gates = 0    # < min_utt 的门误触（原本就丢弃，此处仅计数）
    refine_wall = 0.0    # P0a-F4：Pass2 精修累计墙钟（final 轨 RTF 用）

    import re as _re

    _HALLUC = _re.compile(
        r"\s*(?:bye[\s-]*bye|"
        r"thank(?:s|,?\s+you(?:\s+for\s+(?:watching|listening)|,?\s*everyone)?)"
        r"|sub(?:scribe)?\s*now|amara\.org)\s*[.!]?\s*$", _re.I)
    # 段内短语级幻觉（whisper 经典幻听，可出现在句首/句中，非 ATIS 词汇）
    _HALLUC_PHRASE = _re.compile(
        r"\s*(?:box\s+drop|the\s+end|thank\s+you)\.?\s*", _re.I)

    def _dec_ct2(fwm, seg, off, beam=None, word_ts=True):
        """单 CT2 引擎解码，返回 (tokens, words)；words 供计时。
        beam：旁证引擎传 1 降载；word_ts=False 跳过词级对齐（旁证不需要）。"""
        segments, _ = fwm.transcribe(
            seg,
            language="en",
            beam_size=beam or ARGS.beams_offline,
            word_timestamps=word_ts,
            condition_on_previous_text=False,
            initial_prompt=None if ARGS.no_prompt else prompt_txt,
            vad_filter=False,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
        )
        txt_parts, words = [], []
        for st in segments:
            t = st.text.strip()
            if st.no_speech_prob > 0.55 and st.avg_logprob < -0.9:
                continue
            while True:
                m = _HALLUC.search(t)
                if not m:
                    break
                t = t[: m.start()].strip()
            if not t:
                continue
            txt_parts.append(t)
            for w in (st.words or []):
                wt = w.word.strip()
                if _HALLUC.fullmatch(wt):
                    continue
                words.append({"text": wt, "start": round(w.start + off, 3),
                              "end": round(w.end + off, 3)})
        txt = " ".join(t for t in txt_parts if t)
        txt = _HALLUC_PHRASE.sub(" ", txt)
        txt = _re.sub(r"\s+", " ", txt).strip()
        if _HALLUC.fullmatch(txt):
            return "", []
        # F1 修复：文本经幻觉清洗后与 words 失配（短语删了词还在），按文本
        # 重对齐 words，保证下游 rover/fuse 位置对齐时计时不错位
        from difflib import SequenceMatcher as _SMd

        wtexts = [w["text"] for w in words]
        ttoks = txt.split()
        norm = lambda s: s.lower().strip(".,?")  # noqa: E731
        sm2 = _SMd(None, [norm(w) for w in wtexts],
                   [norm(t) for t in ttoks], autojunk=False)
        words = [words[j1 + o]
                 for tag, i1, i2, j1, j2 in sm2.get_opcodes()
                 if tag == "equal"
                 for o in range(j2 - j1)]
        if len(words) != len(ttoks):  # 对齐丢失（罕见）：宁缺勿错位
            words = words[: len(ttoks)] + [
                {"text": ttoks[k], "start": None, "end": None}
                for k in range(len(words), len(ttoks))]
        return txt, words

    # P0a: _fill_times / _fuse 已提到模块级（纯函数，便于 CPU 单测；
    # 融合模式 --fuse_mode 由模块级 _fuse 的 fuse_mode 参数实现）。

    def refine(s, e):
        """对 [s,e] 秒做 offline 精修：多引擎解码 + 台站模板证词融合。

        模板 = 系统合法持有的台站 ATIS 播报模板（ATIS 本质是固定模板循环播报）。
        精修不再单引擎自由解码（系统性丢弱读词：SAINT/ZERO/TWO FOUR/INFORM，
        deep 教训4），改为引擎证词与模板对齐：
        audio-attested 词带真实计时，template-only 词如实标记（meta 可审计）。
        --no_fusion 退回单引擎自由解码（消融对照）。"""
        import torch

        torch.cuda.empty_cache()  # 归还流式阶段的缓存，缓解与 CT2 的显存争用
        lead, tail = 0.22, 0.10
        s2 = max(0.0, last_final_f * frame_ms / 1000 - 0.05, min(s - lead, s))
        e2 = min(dur, max(e + tail, s2 + 0.3))
        seg = x[int(s2 * sr): int(e2 * sr)]
        _t = {"t0": time.time()}
        t0_txt, t0_words = _dec_ct2(fw, seg, s2)
        _t["atc"] = time.time() - _t["t0"]
        if ARGS.rover:
            # 先验无关共识（R 方向）：atc 主 + v3 + qwen 三引擎 ROVER 投票 +
            # ATIS 公开词表模糊纠错。无模板、无 prompt（配合 --no_prompt）。
            import atis_lexicon
            from rover import rover as _rover_fn

            sec = []
            jobs = []          # 旁证并行解码（CT2 释放 GIL；qwen worker 独立进程）
            from concurrent.futures import ThreadPoolExecutor
            from difflib import SequenceMatcher as _SM

            t1_txt = ""
            _ts = time.time()
            if fw2 is not None:
                t1_txt = _dec_ct2(fw2, seg, s2, beam=1, word_ts=False)[0]
                sec.append((t1_txt.split(), None))
            _t["v3"] = time.time() - _ts

            def _qwen_job():
                try:
                    return _qwen_txt(seg)
                except Exception as ex:  # noqa: BLE001
                    print(f"[warn] qwen transcribe failed: {ex}")
                    return ""

            # 自适应跨族仲裁（R4）：atc/v3 高一致时跳过 qwen 省时延；
            # 分歧段才请跨族引擎做 >=2 票共识。一致率低时无单票覆盖
            # （rover.py 已改：覆盖主引擎一律 >=2 票）。
            if qwen is not None:
                _ts = time.time()
                a_toks, b_toks = t0_txt.split(), t1_txt.split()
                agr = 0.0
                if a_toks or b_toks:
                    sm_ = _SM(None, a_toks, b_toks, autojunk=False)
                    agr = sum(i2 - i1 for tag, i1, i2, j1, j2
                              in sm_.get_opcodes() if tag == "equal") \
                        / max(max(len(a_toks), len(b_toks)), 1)
                if agr < ARGS.rover_qwen_gate:
                    jobs.append(("qwen", _qwen_job))
            if jobs:
                with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    for txt_ in pool.map(lambda j: j[1](), jobs):
                        sec.append((txt_.split(), None))
                _t["qwen"] = time.time() - _ts
            rtoks, rwords, rstats = _rover_fn(t0_txt.split(), t0_words, sec)
            print(f"[rover-timing] atc={_t['atc']:.2f}s v3={_t['v3']:.2f}s "
                  f"qwen={_t.get('qwen', 0):.2f}s")
            rtxt = atis_lexicon.normalize(" ".join(rtoks))
            rtxt, n_g = atis_lexicon.grammar_fix(rtxt)
            rstats["grammar_fixes"] = n_g
            # 词表纠错会改词数（短语规则并/拆词）：计时对齐到主引擎词序列，
            # 多出的词计时置 None 交 _fill_times 插值
            out_words = []
            for k, t in enumerate(rtxt.split()):
                w = rwords[k] if k < len(rwords) else None
                out_words.append({"text": t,
                                  "start": w["start"] if w else None,
                                  "end": w["end"] if w else None,
                                  "src": "rover"})
            return rtxt, _fill_times(out_words), {"mode": "rover", **rstats}
        if ARGS.no_fusion:
            return t0_txt, t0_words, {"mode": "no_fusion"}
        engines = [{"tokens": t0_txt.split(), "words": t0_words, "family": "ct2"}]
        if fw2 is not None:
            t1_txt, _ = _dec_ct2(fw2, seg, s2)
            engines.append({"tokens": t1_txt.split(), "words": None, "family": "ct2"})
        if qwen is not None:
            try:
                qtxt = _qwen_txt(seg)
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] qwen transcribe failed: {ex}")
                qtxt = ""
            engines.append({"tokens": (qtxt or "").split(), "words": None,
                            "family": "qwen"})
        tpl_toks = [w.upper() for w in prompt_txt.split()]
        txt, words, stats = _fuse(tpl_toks, engines, ARGS.fuse_mode)
        # 低证词率护栏：音频几乎未证实任何模板词（如噪声尖峰触发的幻影段）时
        # 拒绝输出——输出模板文本等于无中生有（J13：K4b 幽灵第 5 周期 att 1/71）。
        if "attested" in stats and not ARGS.no_fusion:
            ratio = stats["attested"] / max(len(tpl_toks), 1)
            stats["att_ratio"] = round(ratio, 3)
            if ratio < 0.3:
                print(f"[guard] reject low-attest seg "
                      f"(att_ratio={ratio:.2f} < 0.3)")
                return "", [], {**stats, "mode": "rejected_lowatt"}
        return txt, words, stats

    def do_finalize(s_sec, e_sec, emit_t):
        """定稿一段。P0a-F3：零/负长度定稿一律跳过（不送精修）并计数。"""
        nonlocal last_final_f, n_empty_finals, refine_wall
        if e_sec - s_sec <= 0.0:
            # F3：长度非正的区间是状态机退化产物，精修它只会拿到
            # "整段静音/幻影"结果（deliver3_b.log 的 <empty> 即此类）
            n_empty_finals += 1
            print(f"[final @{emit_t:6.2f}s] SKIP empty-range "
                  f"({s_sec:.2f}-{e_sec:.2f}s) n_empty_finals={n_empty_finals}")
            return
        _tw = time.perf_counter()
        try:
            rtxt, rwords, rstats = refine(s_sec, e_sec)
        finally:
            refine_wall += time.perf_counter() - _tw  # F4：final 轨精修耗时
        if rtxt:
            finalized_parts.append(rtxt)
            for w in rwords:
                ev = {"emit_audio_t": emit_t, "word": w["text"],
                      "att_start": w["start"], "att_end": w["end"],
                      "src": w.get("src", "eng0")}
                if w.get("dev_from") is not None:  # F1 apply 替换事件
                    ev["dev_from"] = w["dev_from"]
                    ev["dev_voters"] = w.get("voters")
                    ev["dev_fams"] = w.get("fams")
                seg_events_final.append(ev)
            st_txt = ""
            if "attested" in rstats:
                st_txt = (f" | att {rstats['attested']} tpl {rstats['template_only']}"
                          f" dev {rstats['deviated']}")
                if rstats.get("applied"):
                    st_txt += f" applied {rstats['applied']}"
            print(f"[final @{emit_t:6.2f}s] {rtxt}{st_txt}")
        else:
            print(f"[final @{emit_t:6.2f}s] <empty>")
        last_final_f = max(last_final_f, int(e_sec * 1000 / frame_ms))

    t0 = time.time()
    chunk_n = int(ARGS.chunk * sr)
    n_chunks = int(np.ceil(len(x) / chunk_n))
    for ci in range(n_chunks):
        seg = x[ci * chunk_n: (ci + 1) * chunk_n]
        t_chunk_end = (ci + 1) * ARGS.chunk
        # ---- Pass1 喂入 ----
        online.insert(seg)
        _, toks, ws = online.iter_once()
        txt = online.asr.tokenizer.decode(toks) if toks else ""
        if txt.strip():
            draft_parts.append(txt)
            for w in ws:
                prefix_words.append(w)
                seg_events_draft.append({
                    "emit_audio_t": t_chunk_end, "word": w["text"],
                    "att_start": round(w["start"], 3), "att_end": round(w["end"], 3)})
        snapshots.append({"t": round(t_chunk_end, 2),
                          "words": [w["text"] for w in prefix_words]})
        # ---- VAD 推进（CV 门）----
        # P0a-F3：状态机搬到模块级 vad_advance_chunk，长度一律基于"当前语音段
        # 起点"求值（原实现在 sp_start_f 归位前用陈旧起点算 sp_len_frames）。
        f_limit = min(int(t_chunk_end * 1000 / frame_ms), n_frames)
        vad_st.update({"in_speech": in_speech, "sp_start_f": sp_start_f,
                       "sil_run": sil_run, "vad_ptr": vad_ptr})
        ev = vad_advance_chunk(vad_st, gate, f_limit,
                               silence_frames=silence_frames,
                               min_utt_frames=min_utt_frames,
                               max_utt_frames=max_utt_frames,
                               frame_ms=frame_ms)
        (in_speech, sp_start_f, sil_run,
         vad_ptr) = (vad_st["in_speech"], vad_st["sp_start_f"],
                     vad_st["sil_run"], vad_st["vad_ptr"])
        pending_seg = None
        if ev and ev[0] == "finalize":
            pending_seg = (ev[1], ev[2])
        elif ev and ev[0] == "drop_short":
            n_noise_gates += 1  # 与原实现一致：判为门误触，丢弃不精修
        # ---- Pass2 触发 ----
        if pending_seg:
            s_sec, e_sec = pending_seg
            emit_t = t_chunk_end  # 客观可复现：本 chunk 处理完成时刻
            do_finalize(s_sec, e_sec, emit_t)
            pending_seg = None
    # 收尾：flush 流式引擎 & 最后一句话
    o = online.iter_once(is_last=True)
    txt_last = online.asr.tokenizer.decode(o[1]) if o[1] else ""
    if txt_last.strip():
        draft_parts.append(txt_last)
        for w in o[2]:
            seg_events_draft.append({
                "emit_audio_t": dur, "word": w["text"],
                "att_start": round(w["start"], 3), "att_end": round(w["end"], 3)})
    # 尾句精修：
    # - 若文件在语音中结束（末句没有 >=sil 的句末静音），主循环不会触发它，
    #   这里强制定稿 [sp_start_f, 文件尾]，避免末句丢失（raw_b 尾部即此情形）；
    # - 若文件在静音中结束，末句已由句末静音触发定稿，无需补。
    # 不再用"最后一个门语音帧±窗口"（对噪声底失效且会与已定稿区重复）。
    if in_speech:
        s_sec = sp_start_f * frame_ms / 1000
        e_sec = dur
        # 与主循环同一门槛：尾段 < min_utt 视为噪声/尾音尖峰，不单独定稿
        if e_sec - s_sec >= ARGS.min_utt:
            do_finalize(s_sec, e_sec, dur)
    wall = time.time() - t0
    # P0a-F4：两轨墙钟分离。refine() 在主循环内同步执行，wall 含 Pass1+Pass2；
    # 草稿轨只该计 Pass1（wall - refine_wall），final 轨计全程。旧实现两轨同写
    # wall/dur，46/46 对 meta.rtf 完全相同，草稿轨 RTF 被精修耗时污染。
    draft_wall = max(wall - refine_wall, 0.0)

    # ---------- 写 two dirs ----------
    final_text = " ".join(finalized_parts)
    draft_text = " ".join(draft_parts)
    meta_common = {
        "engine": "twopass",
        "chunk": ARGS.chunk, "frame_threshold": ARGS.frame_threshold,
        "half": ARGS.half,
        "sil": ARGS.sil, "ct2": str(ct2_p), "stream_model": ARGS.stream_model,
        "vad_gate": "RMS>8 & CV600>0.2", "max_utt": ARGS.max_utt,
        "channel": chan, "template": tpl_path.name,
        "template_source": template_source,          # P0a-F2 审计字段
        "fuse_mode": ARGS.fuse_mode,                 # P0a-F1 审计字段
        "n_empty_finals": n_empty_finals,            # P0a-F3 审计字段
        "n_noise_gates": n_noise_gates,
        "refine_engines": 1 if ARGS.no_fusion else engines_n,
        "fusion": not ARGS.no_fusion,
        "no_prompt": ARGS.no_prompt,
        "rover": ARGS.rover,
        "prior": ("none_rover" if (ARGS.no_prompt and ARGS.rover) else
                  "none" if ARGS.no_prompt else
                  ("template_fusion" if not ARGS.no_fusion else "template_prompt")),
        "qwen": ("worker" if qwen == "worker" else
                 "in-process") if qwen else None,
        "dur": dur,
    }
    # 逐词证词来源统计（audio-attested vs template-only）
    srcs = [e.get("src") for e in seg_events_final if e.get("src")]
    if srcs:
        from collections import Counter

        meta_common["word_src_counts"] = dict(Counter(srcs))
    # final 轨：snapshots 来自 final 定稿事件（P0a-F4，不再借用草稿前缀）
    write_track(
        outdir, final_text=final_text, events=seg_events_final,
        snapshots=build_final_snapshots(seg_events_final),
        meta={**meta_common, "track": "refined",
              "n_sent": len(finalized_parts),
              "wall_sec": round(wall, 2), "rtf": round(wall / dur, 4),
              "rtf_note": "final 轨=Pass1+Pass2 全程墙钟"})

    dout = Path(str(outdir) + "_draftonly")
    # 草稿轨：snapshots 本就是逐 chunk 的草稿前缀（语义正确，予以保留）
    write_track(
        dout, final_text=draft_text, events=seg_events_draft,
        snapshots=snapshots,
        meta={**meta_common, "track": "draft", "engine": "simulstreaming-draft",
              "wall_sec": round(draft_wall, 2),
              "rtf": round(draft_wall / dur, 4) if dur else None,
              "rtf_note": "draft 轨=Pass1 流式墙钟（wall-refine_wall），"
                          "不含 Pass2 精修"})

    print("RTF final:", round(wall / dur, 4),
          "| RTF draft:", round(draft_wall / dur, 4))
    print("FINAL:", final_text[:500])
    if qwen_proc is not None:
        try:
            qwen_proc.stdin.close()
            qwen_proc.terminate()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
