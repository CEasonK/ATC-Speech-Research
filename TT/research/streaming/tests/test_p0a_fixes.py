"""P0a 修复回归单测（纯 CPU，不加载任何模型）。
覆盖：F1 _fuse apply/report_only、F2 resolve_template_path fail-fast、
F3 vad_advance_chunk 陈旧起点修复、F4 build_final_snapshots 事件来源。
运行：cd streaming && python -m pytest tests/test_p0a_fixes.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_2pass as rp  # noqa: E402


# ---------------- F1: _fuse ----------------

def _eng(tokens, family, words=None):
    return {"tokens": tokens, "words": words or [], "family": family}


def test_fuse_apply_replaces_crossfamily_deviation():
    tpl = ["ALPHA", "BRAVO", "CHARLIE"]
    engines = [_eng(["ALPHA", "XRAY", "CHARLIE"], "ct2"),
               _eng(["alpha", "xray", "charlie"], "qwen")]
    text, words, stats = rp._fuse(tpl, engines, "apply")
    assert text == "ALPHA XRAY CHARLIE"
    assert stats["applied"] == 1 and stats["deviated"] == 1
    w = words[1]
    assert w["src"] == "fuse" and w["dev_from"] == "BRAVO"


def test_fuse_report_only_keeps_identity():
    tpl = ["ALPHA", "BRAVO", "CHARLIE"]
    engines = [_eng(["ALPHA", "XRAY", "CHARLIE"], "ct2"),
               _eng(["alpha", "xray", "charlie"], "qwen")]
    text, _, stats = rp._fuse(tpl, engines, "report_only")
    assert text == "ALPHA BRAVO CHARLIE"   # 旧恒等行为逐位保留
    assert stats["applied"] == 0 and stats["deviated"] == 1
    assert stats["fuse_mode"] == "report_only"


def test_fuse_same_family_not_replaced():
    tpl = ["ALPHA", "BRAVO", "CHARLIE"]
    engines = [_eng(["ALPHA", "XRAY", "CHARLIE"], "ct2"),
               _eng(["alpha", "xray", "charlie"], "ct2")]
    text, _, stats = rp._fuse(tpl, engines, "apply")
    assert text == "ALPHA BRAVO CHARLIE"   # 同族双票不算独立证词
    assert stats["applied"] == 0 and stats["deviated"] == 0


# ---------------- F2: resolve_template_path ----------------

def test_template_missing_fails_fast(tmp_path):
    with pytest.raises(SystemExit):
        rp.resolve_template_path(tmp_path / "NEWCH_evalK4.wav", src_dir=tmp_path)


def test_template_explicit_missing_fails_fast(tmp_path):
    with pytest.raises(SystemExit):
        rp.resolve_template_path(tmp_path / "a.wav",
                                 template_file=str(tmp_path / "nope.txt"),
                                 src_dir=tmp_path)


def test_template_explicit_ok(tmp_path):
    tpl = tmp_path / "tpl.txt"
    tpl.write_text("ALPHA")
    p, src = rp.resolve_template_path(tmp_path / "a.wav",
                                      template_file=str(tpl), src_dir=tmp_path)
    assert p == tpl and src == "template_file_arg"


def test_template_fallback_explicit(tmp_path):
    p, src = rp.resolve_template_path(
        tmp_path / "NEWCH.wav", src_dir=tmp_path,
        allow_static_prompt_fallback=True)
    assert p.name == "static_prompt_atc.txt" and src == "static_prompt_FALLBACK"


def test_template_found_in_dir(tmp_path):
    # 函数按真实布局查找：src_dir.parent/"templates"/<chan>.txt
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    tdir = tmp_path / "templates"
    tdir.mkdir()
    (tdir / "NEWCH.txt").write_text("TANGO")
    p, src = rp.resolve_template_path(tmp_path / "NEWCH_evalK4.wav",
                                      src_dir=src_dir)
    assert p.name == "NEWCH.txt" and src == "templates_dir"


# ---------------- F3: vad_advance_chunk ----------------

def _st():
    return {"in_speech": False, "sp_start_f": 0, "sil_run": 0, "vad_ptr": 0}


def test_vad_no_zero_length_forced_cut_after_long_silence():
    """旧 bug：短段被丢弃后 sp_start_f 陈旧，长静音后首个语音帧带陈旧长度
    立刻 force-cut 出 s_sec==e_sec 的零长度定稿（deliver3_b.log <empty>）。"""
    sil, min_u, max_u = 5, 10, 20
    gate = [1] * 5 + [0] * 30 + [1] * 25          # 短语音-长静音-长语音
    st = _st()
    events = []
    for f in range(len(gate)):                    # 逐帧推进=最恶劣的碎片 chunk
        ev = rp.vad_advance_chunk(st, gate, f + 1, silence_frames=sil,
                                  min_utt_frames=min_u, max_utt_frames=max_u)
        if ev:
            events.append(ev)
    kinds = [e[0] for e in events]
    assert "drop_short" in kinds                  # 前 5 帧短段判噪声（旧语义保留）
    finals = [e for e in events if e[0] == "finalize"]
    assert all(e[2] > e[1] for e in finals), finals          # 无零/负长度定稿
    # 第二个长语音段应以其真实起点触发 max_utt 定稿（35*10ms=350s? 否：帧→毫秒）
    assert finals and finals[-1][3] == "max_utt"
    assert finals[-1][1] == 35 * 10 / 1000        # 起点=新段首帧（35 帧处）


def test_vad_normal_endpoint_untouched():
    sil, min_u, max_u = 5, 3, 100
    gate = [1] * 20 + [0] * 10
    st = _st()
    events = []
    for f in range(len(gate)):
        ev = rp.vad_advance_chunk(st, gate, f + 1, silence_frames=sil,
                                  min_utt_frames=min_u, max_utt_frames=max_u)
        if ev:
            events.append(ev)
    assert len(events) == 1                      # 不多不少恰好一个定稿
    assert events[0][0] == "finalize" and events[0][3] == "endpoint"
    # e_sec = (触发帧24 - sil_run5) 帧 = 19 帧（与旧实现语义一致：不含静音尾巴）
    assert events[0][1] == 0.0 and events[0][2] == 19 * 10 / 1000


# ---------------- F1: 计数矩阵（质疑者 E3 要求：引擎证实场景） ----------------

def test_fuse_count_matrix_three_scenarios():
    # 场景矩阵：位置1=跨族偏离(替换)，其余3位引擎证实（attested 带真实计时），
    # 另设第 3 引擎缺 CHARLIE 证实的场景在 template_only 单测中覆盖。
    tpl = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    eng_words = [{"start": 0.0, "end": 0.3}, {"start": 0.6, "end": 0.9},
                 {"start": 0.9, "end": 1.2}, {"start": 1.2, "end": 1.5}]
    engines = [_eng(["ALPHA", "XRAY", "CHARLIE", "DELTA"], "ct2", eng_words),
               _eng(["alpha", "xray", "charlie", "delta"], "qwen", eng_words)]
    text, words, stats = rp._fuse(tpl, engines, "apply")
    assert text == "ALPHA XRAY CHARLIE DELTA"
    assert stats["applied"] == 1                 # BRAVO→XRAY 替换
    assert stats["deviated"] == 1
    assert stats["template_only"] == 0
    assert stats["attested"] == 3                # ALPHA/CHARLIE/DELTA
    assert stats["attested"] + stats["template_only"] + stats["applied"] == 4
    assert words[0]["src"] == "eng0" and words[0]["start"] == 0.0
    assert words[1]["src"] == "fuse" and words[1]["dev_from"] == "BRAVO"
    # apply 命中位计时置 None，由 _fill_times 插值（非 None）
    assert words[1]["start"] is not None


def test_fuse_template_only_scenario():
    # 引擎完全未覆盖中段：模板持有词 template_only 计数
    tpl = ["ALPHA", "SAINT", "CHARLIE"]
    engines = [_eng(["ALPHA", "SAINT", "CHARLIE"], "ct2"),
               _eng(["ALPHA", "SAINT", "CHARLIE"], "qwen")]
    # 全证实 → template_only=0 对照
    _, _, stats = rp._fuse(tpl, engines, "apply")
    assert stats["attested"] == 3 and stats["template_only"] == 0
    # 双引擎同幻听偏离但**单族** → 不算独立证词：deviated=0，保持模板词
    engines1 = [_eng(["ALPHA", "SENT", "CHARLIE"], "ct2"),
                _eng(["ALPHA", "SENT", "CHARLIE"], "ct2")]
    text, _, stats = rp._fuse(tpl, engines1, "apply")
    assert text == "ALPHA SAINT CHARLIE"
    assert stats["deviated"] == 0 and stats["applied"] == 0
    assert stats["template_only"] == 1   # 仅 SAINT：ALPHA/CHARLIE 已被 equal 块证实
    assert stats["attested"] == 2


# ---------------- F4: build_final_snapshots ----------------

def test_final_snapshots_from_events_only():
    ev = [{"emit_audio_t": 1.0, "word": "A"}, {"emit_audio_t": 1.0, "word": "B"},
          {"emit_audio_t": 2.0, "word": "C"}]
    snaps = rp.build_final_snapshots(ev)
    assert snaps == [{"t": 1.0, "words": ["A", "B"]},
                     {"t": 2.0, "words": ["A", "B", "C"]}]


def test_final_snapshots_empty_events_no_draft_leak():
    assert rp.build_final_snapshots([]) == []


# ---------------- 集成守卫（质疑者 E1：防"孤儿函数假覆盖"） ----------------

def test_main_actually_wires_f4_helpers():
    """AST 断言 main() 源码里存在对 write_track / build_final_snapshots 的
    调用，以及 meta 的四个审计字段。曾经发生：两函数只有定义零调用、
    写盘段退回基线，而单测全绿——本测试专门让这种回退变红。"""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rp.main))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "write_track" in called
    assert "build_final_snapshots" in called
    src = inspect.getsource(rp.main)
    for field in ('"template_source"', '"fuse_mode"',
                  '"n_empty_finals"', "draft_wall"):
        assert field in src, f"main() 写盘 meta 缺少 {field}"
