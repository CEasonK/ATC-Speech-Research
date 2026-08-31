"""P2 修复单测（deep 计分口径）——纯 CPU 桩模型，禁止加载 whisper/跑 GPU。

覆盖：
  D1 score_constrained 返回 per-token 均值（v12 不得再除 nt）
  D2 adjudicate_contest 同窗共用 + 等长组判定/length_confounded
  D3 position_test_validity 窗长<周期守卫；find_anchor_window 语义诚实
  AST 集成守卫：main 真的接线了新函数（防孤儿函数假覆盖）
"""
import ast
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
sys.path.insert(0, str(DEEP / "exp"))

from nll_scorer import NLLScorer, WIN_S, normalize_text  # noqa: E402
import adjudicate_v14  # noqa: E402  (模块级仅读 manifest，安全)
import assemble_final as af  # noqa: E402  (main 在 __main__ 守卫下，安全)

EXP = DEEP / "exp"


# ---------------- 桩 ----------------
class FakeTok:
    """词→单 token：token 数 == 词数，长度语义与真实 BPE 单调一致，够测归一逻辑。"""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10 + (hash(w) % 100) for w in text.split()]


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """按 target 位置 j 给 logit g_j = 2.0 + j，其余类 logit=0。
    每个窗口的 per-position NLL 完全相同 → 便于解析算期望。"""
    V = 128

    def __init__(self, prompt_len):
        self.P = prompt_len

    def __call__(self, encoder_outputs=None, decoder_input_ids=None):
        b, L = decoder_input_ids.shape
        logits = torch.zeros(b, L, self.V)
        # 与真实计分对齐：sl = logits[:, P-1:]，第 j 位预测 targets[j]，
        # targets = text_ids + [eot]，text_ids = decoder_input_ids[P:]
        text_ids = decoder_input_ids[0, self.P:].tolist()
        targets = text_ids + [99]
        for j, tok in enumerate(targets):
            logits[:, self.P - 1 + j, tok] = 2.0 + j
        return FakeOut(logits)


def make_stub_scorer(scores_map=None, calls=None):
    """不触发 __init__ 的裸 NLLScorer（CPU）。"""
    sc = object.__new__(NLLScorer)
    sc.device = "cpu"
    sc.prompt = [1, 2, 3, 4]
    sc.eot = 99
    sc.tok = FakeTok()
    sc.model = FakeModel(len(sc.prompt))
    entry = {"win_starts": [0.0, 5.0, 10.0],
             "enc": torch.zeros(3, 4, 4), "duration": 40.0}
    sc._cache = {}
    sc.load_audio = lambda p: entry
    if scores_map is not None:  # 直接编排 score_constrained 返回值（D2 用）
        def _sc(path, text, t_center, half_width=5.0):
            if calls is not None:
                calls.append((path, t_center, half_width))
            return {"score": scores_map[text]}
        sc.score_constrained = _sc
    return sc


# ---------------- D1 ----------------
def test_score_constrained_returns_per_token_mean():
    sc = make_stub_scorer()
    text = "WIND TWO FOUR"  # 3 tok + eot = 4 targets
    r = sc.score_constrained("fake.wav", text, 5.0, half_width=5.0)
    V = FakeModel.V
    nll = [np.log((V - 1) + np.exp(2.0 + j)) - (2.0 + j) for j in range(4)]
    expected_mean = float(np.mean(nll))
    expected_sum = float(np.sum(nll))
    assert abs(r["score"] - expected_mean) < 1e-4, \
        f"score 应为 per-token 均值 {expected_mean}，实得 {r['score']}"
    assert abs(r["score"] - expected_sum) > 1.0, "score 不得是总 NLL"


def test_v12_no_double_norm():
    tree = ast.parse((EXP / "adjudicate_v12_wind.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Name) and node.right.id == "nt":
                raise AssertionError(
                    f"L{node.lineno}: 检测到对 nt 的除法——score 已是 per-token，"
                    "再除属双重归一（P2/D1）")


# ---------------- D2 ----------------
def test_contest_all_candidates_share_same_window():
    calls = []
    cands = {"A": "JUST OUT OF ONE EIGHT ZERO", "B": "JOIN YOU OUT OF ONE EIGHT ZERO",
             "C": "JOHNSON LEVEL"}
    smap = {cands["A"]: 1.00, cands["B"]: 1.10, cands["C"]: 0.90}
    sc = make_stub_scorer(smap, calls)
    adjudicate_v14.adjudicate_contest(sc, "seg.wav", 12.5, 6.0, cands, contam=1.4)
    assert len(calls) == 3
    assert all(c == calls[0] for c in calls), f"候选切片参数不一致: {calls}"
    assert calls[0] == ("seg.wav", 12.5, 6.0)


def test_contest_equal_length_fair_win():
    sc = make_stub_scorer({"ONE TWO": 1.0, "NINE EIGHT": 2.6}, None)
    res = adjudicate_v14.adjudicate_contest(
        sc, "s.wav", 5.0, 5.0, {"X1": "ONE TWO", "X2": "NINE EIGHT"}, contam=1.4)
    assert res["winner"] == "X1"
    assert res["fair_margin"] == 1.6
    assert res["decidable"] is True
    assert res["decided_strict"] is True
    assert "contam" in res["calib_note"]
    assert res["length_confounded"] is False
    assert res["n_tok"] == {"X1": 2, "X2": 2}


def test_contest_length_confounded_flag():
    # 全局第2名与最优候选长度不同 → length_confounded。
    # 分数构造有讲究：global margin=0.5(≤1.4) 而 fair margin=1.6(>1.4)，
    # 数学上 fair≥global 恒成立，只有这个方向能让新旧口径分歧——
    # 变异测试（decidable 改回全局 margin）曾存活，就是缺了这条判别用例。
    smap = {"A B C D E F": 1.00, "G H I J K L": 2.60, "X Y": 1.50}
    sc = make_stub_scorer(smap, None)
    res = adjudicate_v14.adjudicate_contest(
        sc, "s.wav", 5.0, 5.0,
        {"BEST6": "A B C D E F", "ALT6": "G H I J K L", "SHORT2": "X Y"},
        contam=1.4)
    assert res["winner"] == "BEST6"
    assert res["ranking"] == ["BEST6", "SHORT2", "ALT6"]
    assert res["length_confounded"] is True   # 对 SHORT2 的优势跨长度，未证实
    assert res["margin_to_2nd"] == 0.5        # 全局口径不可裁
    assert res["fair_margin"] == 1.6          # 等长组内 1.6 > 1.4 可裁
    assert res["decidable"] is True           # 新口径=fair；旧口径(global)会被测出
    # codex 质疑 Q-b(ii)：decidable 语义已漂移为"等长组内可裁决"，
    # 跨长度嫌疑必须经 decided_strict 才可见——下游默认消费后者
    assert res["decided_strict"] is False


# ---------------- D3 ----------------
def test_position_validity_guard():
    assert af.position_test_validity(10.0, 27.85) is True
    assert af.position_test_validity(30.0, 28.34) is False
    # 本项目实测：两信道周期都 < 30s 窗 ⇒ 现状正位/错位检验必判无效
    assert af.position_test_validity(WIN_S, af.PERIOD_S["CYYT_ATIS_a"]) is False
    assert af.position_test_validity(WIN_S, af.PERIOD_S["CYYT_ATIS_b"]) is False


class AnchorStub:
    def __init__(self):
        self.n_calls = 0
        self.tok = FakeTok()

    def find_anchor_window(self, path, anchor):
        return 100.0

    def score_constrained(self, path, text, t, half_width=5.0):
        self.n_calls += 1
        return {"score": 1.23}


def test_validate_invalid_flags_degraded(monkeypatch):
    # 窗长≥周期：数字照出，但标 DEGRADED（自查修正：Δ测得≤0.54nat 非零，
    # 隐藏数字不如标注口径——它只剩相位对齐信息，不能独立裁决）
    monkeypatch.setitem(af.CH, "CH_B", "b.wav")
    monkeypatch.setitem(af.PERIOD_S, "CH_B", 27.85)
    stub = AnchorStub()
    lines = [("LINE ONE", "ANCHOR ONE"), ("LINE TWO", "ANCHOR TWO")]
    rows, ptest = af.validate(stub, "CH_B", lines)
    assert ptest["valid"] is False and ptest["note"].startswith("DEGRADED")
    assert all(r["nll_wrong_pos"] == 1.23 for r in rows)
    assert stub.n_calls == 4  # 正位+错位都算，信息由 note 降级而非删数


def test_validate_valid_runs_wrongpos(monkeypatch):
    monkeypatch.setitem(af.CH, "CH_LONG", "l.wav")
    monkeypatch.setitem(af.PERIOD_S, "CH_LONG", 80.0)
    stub = AnchorStub()
    lines = [("LINE ONE", "ANCHOR ONE"), ("LINE TWO", "ANCHOR TWO")]
    rows, ptest = af.validate(stub, "CH_LONG", lines)
    assert ptest["valid"] is True
    assert all(r["nll_wrong_pos"] == 1.23 for r in rows)
    assert stub.n_calls == 4  # 正位 2 + 错位 2


def test_find_anchor_window_docstring_honest():
    doc = NLLScorer.find_anchor_window.__doc__
    assert "窗起点" in doc
    assert "不做窗内亚窗定位" in doc
    assert "精确时刻" in doc  # 明确否认精度


# ---------------- AST 集成守卫（防孤儿函数假覆盖） ----------------
def _called_names(fn):
    tree = ast.parse(inspect.getsource(fn))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_v14_main_wires_contest():
    called = _called_names(adjudicate_v14.main)
    assert "adjudicate_contest" in called
    src = inspect.getsource(adjudicate_v14.main)
    assert "score_constrained" not in src, "main 不得绕过纯函数直接计分"


def test_assemble_validate_wires_guard():
    called = _called_names(af.validate)
    assert "position_test_validity" in called
    src = inspect.getsource(af.main)
    assert "position_test" in src, "main 必须把守卫判定写进输出 json"


def test_no_duplicate_top_level_defs():
    for fname in ("adjudicate_v14.py", "assemble_final.py"):
        tree = ast.parse((EXP / fname).read_text())
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        dup = {n for n in names if names.count(n) > 1}
        assert not dup, f"{fname} 顶层函数重名(后者遮蔽前者): {dup}"
