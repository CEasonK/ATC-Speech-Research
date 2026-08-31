"""P3 修复单测（run_simulstreaming 封装层）——纯 CPU，sys.modules 注入假
simulstreaming 包，禁止加载真模型/音频。

覆盖：S1 驱逐秒直累加（双除回归）/ S2 先驱逐后 infer 的词时间轴 +
att_*_rel / S3 alignment heads 按 dims 注入 / S4 static_prompt 读内容。
"""
import ast
import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


# ---------------- 假 simulstreaming 包 ----------------
class AlignAttConfig:
    last = None

    def __init__(self, **kw):
        type(self).last = kw


class FakeModel:
    DIMS = (32, 20)  # large-v3 / ATC 微调（checkpoint 实測 dims）

    def __init__(self):
        self.dims = types.SimpleNamespace(n_text_layer=self.DIMS[0],
                                          n_text_head=self.DIMS[1])
        self.alignment_heads = None
        self.set_heads_calls = []

    def set_alignment_heads(self, dump):
        self.set_heads_calls.append(dump)
        self.alignment_heads = dump


FAKE_HEADS_TABLE = {"large-v3": b"TUNED32x20", "large-v3-turbo": b"TUNED4x20"}
MODELS_CREATED = []


def fake_load_model(name, **kw):
    m = FakeModel()
    MODELS_CREATED.append(m)
    return m


class FakeASR:
    EVICT_SCRIPT = []  # 每次 insert_audio 的返回值（驱逐秒数）
    INFER_N = 0

    def __init__(self, cfg):
        # 与真实 simul_whisper.py 一致：它调用的是自己模块命名空间里的
        # load_model 全局名（P3/S3 的补丁正打在那里），必须走同样的间接层
        self.model = sys.modules[
            "simulstreaming.whisper.simul_whisper.simul_whisper"
        ].load_model(name="fake", download_root=".")
        self.tokenizer = types.SimpleNamespace(decode=lambda ids: "W1 W2")
        self._evict = list(type(self).EVICT_SCRIPT)

    def insert_audio(self, audio):
        return type(self).EVICT_SCRIPT.pop(0) if type(self).EVICT_SCRIPT else 0.0

    def infer(self, is_last=False):
        FakeASR.INFER_N += 1
        f = 10 + FakeASR.INFER_N
        prog = {"result": {"split_words": ["W1", "W2"], "split_tokens": [[1], [2]]},
                "progress": [{"most_attended_frames": [f - 1, 0]},
                             {"most_attended_frames": [f, 0]}]}
        return [1, 2], prog


def _install_fake_pkg():
    def mk(name, is_pkg=False):
        m = types.ModuleType(name)
        if is_pkg:
            m.__path__ = []
        sys.modules[name] = m
        return m

    for nm in ("simulstreaming", "simulstreaming.whisper",
               "simulstreaming.whisper.simul_whisper"):
        if nm not in sys.modules or not hasattr(sys.modules[nm], "__path__"):
            mk(nm, is_pkg=True)
    cfgm = mk("simulstreaming.whisper.simul_whisper.config")
    cfgm.AlignAttConfig = AlignAttConfig
    wh = mk("simulstreaming.whisper.simul_whisper.whisper")
    wh._ALIGNMENT_HEADS = FAKE_HEADS_TABLE
    sim = mk("simulstreaming.whisper.simul_whisper.simul_whisper")
    sim.PaddedAlignAttWhisper = FakeASR
    sim.load_model = fake_load_model


_install_fake_pkg()
rs = importlib.import_module("run_simulstreaming")


def _args(tmp_path=None, static_prompt=None):
    import argparse
    return argparse.Namespace(model="fake.pt", frame_threshold=25, chunk=0.5,
                              beams=1, audio_max_len=30.0, audio_min_len=0.25,
                              static_prompt=static_prompt, half=False)


@pytest.fixture
def fresh_args(monkeypatch, tmp_path):
    def make(**kw):
        a = _args(tmp_path, **kw)
        monkeypatch.setattr(rs, "ARGS", a)
        return a
    return make


# ---------------- S3 ----------------
def test_s3_tuned_heads_injected(fresh_args):
    fresh_args()
    FakeModel.DIMS = (32, 20)
    o = rs.Online()
    assert o.asr.model.set_heads_calls == [FAKE_HEADS_TABLE["large-v3"]]
    assert o.asr.model._p3_heads_key == "large-v3"


def test_s3_turbo_dims(fresh_args):
    fresh_args()
    FakeModel.DIMS = (4, 20)
    try:
        o = rs.Online()
        assert o.asr.model._p3_heads_key == "large-v3-turbo"
    finally:
        FakeModel.DIMS = (32, 20)


def test_s3_unknown_dims_honest_default(fresh_args):
    fresh_args()
    FakeModel.DIMS = (7, 7)
    try:
        o = rs.Online()
        assert o.asr.model.set_heads_calls == []
        assert o.asr.model._p3_heads_key == "default"  # 不假装映射
    finally:
        FakeModel.DIMS = (32, 20)


# ---------------- S1 / S2 ----------------
def test_s1_evicted_seconds_added_directly(fresh_args, monkeypatch):
    fresh_args()
    FakeASR.EVICT_SCRIPT = [0.0, 2.0, 0.0]
    o = rs.Online()
    o.insert(__import__("numpy").zeros(8000, dtype="float32"))
    o.iter_once()
    assert o.buf_offset == 0.0
    o.iter_once()
    assert o.buf_offset == 2.0, "驱逐 2.0s 应直接累加（旧双除 bug 会得 1.25e-4）"
    FakeASR.EVICT_SCRIPT = []


def test_s2_word_times_use_post_evict_offset(fresh_args):
    # 驱逐与出词同轮：词帧轴相对驱逐后的窗 ⇒ start = 2.0 + f*0.02
    fresh_args()
    FakeASR.EVICT_SCRIPT = [2.0]
    FakeASR.INFER_N = 0
    o = rs.Online()
    o.insert(__import__("numpy").zeros(8000, dtype="float32"))
    _, toks, ws = o.iter_once()
    assert toks == [1, 2]
    assert ws and ws[0]["start"] == pytest.approx(2.0 + (11 - 1) * 0.02)
    assert ws[0]["start_rel"] == pytest.approx(10 * 0.02)
    FakeASR.EVICT_SCRIPT = []


def test_iter_once_no_double_division_in_source():
    src = inspect.getsource(rs.Online.iter_once)
    assert "/ 16000" not in src, "iter_once 不得再出现 /16000（S1 双除回归）"


# ---------------- S4 ----------------
def test_s4_prompt_file_read_as_content(fresh_args, tmp_path):
    pf = tmp_path / "prompt.txt"
    pf.write_text("ALPHA BRAVO RUNWAY TWO EIGHT\n")
    fresh_args(static_prompt=str(pf))
    o = rs.Online()
    content = AlignAttConfig.last["static_init_prompt"]
    assert content == "ALPHA BRAVO RUNWAY TWO EIGHT"
    assert content != str(pf), "S4 回归：路径字符串本身被喂进解码上下文"
    assert o.sp_sha8 == hashlib.sha256(content.encode()).hexdigest()[:8]


def test_s4_missing_file_fails_fast(fresh_args):
    fresh_args(static_prompt="/nonexistent/prompt.txt")
    with pytest.raises(SystemExit):
        rs.Online()


def test_main_wires_audit_fields():
    src = inspect.getsource(rs.main)
    assert "att_start_rel" in src
    assert "att_time_semantics" in src
    assert "static_prompt_sha8" in src
    assert "alignment_heads" in src


def test_no_duplicate_top_level_defs():
    tree = ast.parse((SRC / "run_simulstreaming.py").read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"顶层函数重名(后者遮蔽前者): {dup}"
