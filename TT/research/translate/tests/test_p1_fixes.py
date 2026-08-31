"""P1 修复回归单测（纯 CPU，不加载模型；chat 一律 monkeypatch）。
覆盖：T1 重试环严格改进+陈旧反馈、T2 free 消融无词表、T3 裁判不改卷、
T4 数字审计符号/千分位矩阵、AST 集成守卫（重复定义/影子函数拦截）。
运行：cd translate && python -m pytest tests/test_p1_fixes.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import glossary as g        # noqa: E402
import run_translate as rt  # noqa: E402


# ---------------- T4: 数字对账矩阵 ----------------

@pytest.mark.parametrize("en,zh", [
    ("WIND TWO FOUR ZERO AT FIVE", "风240 度5"),
    ("TEMPERATURE ONE DEW POINT MINUS ONE", "气温1 露点零下1"),
    ("DEW POINT MINUS TWO ZERO", "露点零下20"),
    ("ALTIMETER THREE ZERO TWO THREE", "修正海压3023"),
    ("ALTIMETER THREE ZERO TWO THREE", "修正海压30.23"),
    ("VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET", "能见度15 云高24000"),
    ("FREQUENCY ONE TWO THREE DECIMAL ONE FIVE", "频率123.15"),
    ("RUNWAY TWO SEVEN LEFT", "跑道27左"),
    ("TEMPERATURE MINUS THREE DEW POINT NEGATIVE TWO", "温度零下三 露点负二"),
])
def test_number_matrix_en_zh(en, zh):
    assert sorted(g.en_numbers(en)) == sorted(g.zh_numbers(zh))


def test_minus_sign_not_dropped():
    # 基线缺陷：'DEW POINT MINUS ONE' 曾提成 ['1']（负号丢失，且把 MINUS 读作 0?）
    assert g.en_numbers("DEW POINT MINUS ONE") == ["-1"]
    assert g.zh_numbers("露点零下1") == ["-1"]


def test_thousands_comma_merged_zh():
    assert g.zh_numbers("云高15,240米") == ["15240"]


def test_neg_without_digit_void():
    assert g.en_numbers("MINUS") == []
    assert g.zh_numbers("温度零下") == []


# ---------------- T2: 词表注入按 variant ----------------

def test_system_prompt_variant():
    s_gloss = rt.system_prompt("constrained")
    s_free = rt.system_prompt("free")
    assert rt.GLOSSARY_HINT in s_gloss
    assert rt.GLOSSARY_HINT not in s_free


# ---------------- T1: 重试环（monkeypatch chat） ----------------

def _mk_chat(responses):
    """返回 (chat_fn, calls)。responses 按轮次给出模型输出文本；calls 记录 prompt。"""
    calls = []

    def chat(tok, mdl, system, user, max_new=512):
        calls.append({"system": system, "user": user})
        return responses[min(len(calls) - 1, len(responses) - 1)]
    return chat, calls


def test_retry_accepts_strict_improvement(monkeypatch):
    en = ["WIND TWO FOUR ZERO AT FIVE"]
    zh = ["风500"]                       # 数字不一致
    chat, calls = _mk_chat(["1. 风240 度5"])
    monkeypatch.setattr(rt, "chat", chat)
    out, rounds = rt.retry_failed(None, None, en, zh, variant="free")
    assert out == ["风240 度5"]
    assert rounds[0]["accepted"] == [1]


def test_retry_rejects_worse_candidate_keeps_old(monkeypatch):
    # 旧 bug 复现靶点：候选更差/无改进时曾被无条件覆盖。现在必须拒绝、保留原行。
    en = ["WIND TWO FOUR ZERO AT FIVE"]
    zh = ["风500"]
    chat, _ = _mk_chat(["1. 风777"])     # 仍然数字不一致（无改进）
    monkeypatch.setattr(rt, "chat", chat)
    out, rounds = rt.retry_failed(None, None, en, zh, variant="free")
    assert out == ["风500"]              # 原行保留
    assert rounds[0]["rejected"] == [1]


def test_retry_free_variant_no_glossary_hint(monkeypatch):
    en = ["WIND TWO FOUR ZERO AT FIVE"]
    zh = ["风500"]
    chat, calls = _mk_chat(["1. 风240 度5"])
    monkeypatch.setattr(rt, "chat", chat)
    rt.retry_failed(None, None, en, zh, variant="free")
    assert all(rt.GLOSSARY_HINT not in c["system"] for c in calls)
    # 对照：constrained 变体重试仍带词表
    chat2, calls2 = _mk_chat(["1. 风240 度5"])
    monkeypatch.setattr(rt, "chat", chat2)
    rt.retry_failed(None, None, en, ["风500"], variant="constrained")
    assert any(rt.GLOSSARY_HINT in c["system"] for c in calls2)


def test_retry_bad_idx_recomputed_each_round(monkeypatch):
    # 旧 bug 靶点：bad_idx 陈旧——第 1 轮修好行1后，行2 的问题在旧实现里不会被处理
    en = ["WIND TWO FOUR ZERO AT FIVE", "ALTIMETER THREE ZERO TWO THREE"]
    zh = ["风500", "修正海压9999"]
    chat, calls = _mk_chat(["1. 风240 度5", "2. 修正海压3023"])
    monkeypatch.setattr(rt, "chat", chat)
    out, rounds = rt.retry_failed(None, None, en, zh, variant="free")
    assert out == ["风240 度5", "修正海压3023"]
    assert len(rounds) >= 2          # 第二轮必然处理行2（陈旧反馈做不到）


# ---------------- T3: 裁判不改卷 ----------------

def test_term_suggestions_never_writes_zh():
    en = ["REQUEST INFORMATION"]
    zh = ["索要信息"]                    # 缺术语（期望词以词表为准）
    misses = [{"line": 1, "expected": "请求"}]
    zh_out, sugg = rt.term_suggestions(en, zh, misses)
    assert zh_out == ["索要信息"]        # 逐字不变
    assert isinstance(sugg, list)


# ---------------- AST 集成守卫 ----------------

def test_no_duplicate_top_level_functions():
    """2026-08-31 事故：新 retry_failed 被同名旧函数定义遮蔽（后者覆盖前者），
    调用点拿到旧实现。守卫：顶层函数名不得重复。"""
    import ast

    src_txt = (Path(__file__).resolve().parents[1]
               / "src/run_translate.py").read_text()
    tree = ast.parse(src_txt)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    dup = {x for x in names if names.count(x) > 1}
    assert not dup, f"顶层重复定义（后者遮蔽前者）: {dup}"


def test_scoring_path_free_of_fallback_terms():
    """fallback_terms 作为函数/调用必须绝迹（docstring 提及允许），
    suggestions 通道存在。"""
    import ast

    src_txt = (Path(__file__).resolve().parents[1]
               / "src/run_translate.py").read_text()
    tree = ast.parse(src_txt)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            assert n.name != "fallback_terms"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id != "fallback_terms", "打分/主路径仍在调用 fallback_terms"
    assert "term_suggestions" in src_txt and "suggestions.json" in src_txt
