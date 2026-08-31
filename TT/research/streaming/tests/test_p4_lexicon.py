"""P4：atis_lexicon.py 金参考泄漏清洗的行为钉死测试。

裁决原则（见 /tmp/codex_review/p4_spec.md）：保留规则须可由公开标准
（ICAO/FAA ATIS 模板与读数标准）独立推导，且不删除引擎已输出词。
定点"观测误听→本句真值"映射与定点幻觉删词全部剔除。
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import atis_lexicon as L  # noqa: E402

# ---------- 泄漏清除：定点规则不得存在 ----------

LEAK_PHRASE = [
    "REPORT NINER", "DESCENDING LEVEL", "TAROM", "DECIMALITY",
    "SHAMROCK", "FANDAS", "FIENDER", "DEANDER", "ALTITUDE THREE", "TIME.",
]


def test_no_leaked_phrase_rules():
    pats = {p for p, _ in L.PHRASE_RULES}
    for bad in LEAK_PHRASE:
        assert bad not in pats, f"定点规则未清除: {bad}"


def test_no_leaked_mappings_behaviorally():
    # 删掉的映射必须真的不再发生（不只看表）
    assert L.normalize("REPORT NINER THREE") == "REPORT NINE THREE"
    assert L.normalize("DESCENDING LEVEL GOOD") == "DESCENDING LEVEL GOOD"
    assert L.normalize("FANDAS CENTER") == "FANDAS CENTER"
    assert L.normalize("SHAMROCK JOHNS") == "SHAMROCK JOHNS"
    assert L.normalize("TAROM REPORT NINER") == "TAROM REPORT NINE"


def test_foxtrot_removed_but_true_positive_kept():
    # 当日信息代号不得入表；正确识别的 FOXTROT（OOV、无相似词）原样保留
    assert "FOXTROT" not in L.VOCAB
    assert L.normalize("INFORMATION FOXTROT") == "INFORMATION FOXTROT"


def test_hallucination_words_no_longer_deleted():
    # 定点幻觉删词列表（TIME/BOX/BYE/SAME/GERMANS）已整条删除：不删词
    t, n = L.grammar_fix("TIME BOX GERMANS BYE SAME")
    assert t == "TIME BOX GERMANS BYE SAME" and n == 0


def test_surface_form_rules_removed():
    # DEGREES→ZULU WIND 与 VISIBILITY+THOUSAND FEET 已删
    t, _ = L.grammar_fix("DEGREES ONE THREE ZERO AT")
    assert "ZULU" not in t
    t, _ = L.grammar_fix("VISIBILITY ONE TWO THREE FOUR NEXT")
    assert "THOUSAND" not in t


# ---------- 保留规则行为正确 ----------

def test_phrase_rules_table_guard():
    # 表守卫：保留集必须恰为公开词表词的正字法变体（防定点规则回潮）
    assert set(L.PHRASE_RULES) == {
        ("ARNAV", "RNAV"), ("VISABILITY", "VISIBILITY"),
        ("WETHER", "WEATHER"), ("WHETHER", "WEATHER"),
        ("ALCIMBER", "ALTIMETER"),
    }


def test_kept_orthography_rules():
    assert (L.normalize("ARNAV VISABILITY WETHER ALCIMBER")
            == "RNAV VISIBILITY WEATHER ALTIMETER")


def test_is_it_insertion_paid_honestly():
    # 质疑轮 P4Q2：WEATHER IS→AT 与 AT(IS|IT)→AT 均删——删 IS/IT 是观测
    # 驱动的定点删词。零先验下插入错误照实付出，不得修复：
    t, n = L.grammar_fix("WEATHER AT IT ONE THREE ZERO ZERO")
    assert t == "WEATHER AT IT ONE THREE ZERO ZERO"
    assert L.normalize("WEATHER IS ONE TWO") == "WEATHER IS ONE TWO"


def test_altitude_covered_by_general_mechanism():
    # 冗余定点规则删除后，通用模糊纠错（0.66+首字母约束）必须接住 ALTITUDE
    assert L.normalize("ALTITUDE THREE ZERO TWO THREE") == \
        "ALTIMETER THREE ZERO TWO THREE"


def test_first_letter_constraint():
    assert L.normalize("NEW") == "NEW"  # 防 NEW→ONE 跨首字母误纠


# ---------- grammar_fix 只增不删 ----------

def test_closing_sentence_three_forms():
    t, _ = L.grammar_fix("INFORM ATC THAT YOU HAVE INFORMATION A")
    assert t == "INFORM ATC THAT YOU HAVE INFORMATION A"  # 完整形不重复
    t, _ = L.grammar_fix("INFORM THAT YOU HAVE INFORMATION A")
    assert t == "INFORM ATC THAT YOU HAVE INFORMATION A"  # 补漏 ATC
    t, _ = L.grammar_fix("NUMBER TIME THAT YOU HAVE INFORMATION A")
    # 裸锚点：前置整句，观测受损词保留（删词=真值监督渠道，禁止）
    assert t == "NUMBER TIME INFORM ATC THAT YOU HAVE INFORMATION A"


def test_zulu_completion_and_idempotence():
    t, n = L.grammar_fix("WEATHER AT ONE THREE ZERO ZERO WIND")
    assert t == "WEATHER AT ONE THREE ZERO ZERO ZULU WIND" and n == 1
    t2, n2 = L.grammar_fix(t)
    assert t2 == t and n2 == 0  # 幂等


def test_zulu_wind_completion():
    t, _ = L.grammar_fix("ZULU TWO THREE ZERO AT ONE TWO")
    assert t == "ZULU WIND TWO THREE ZERO AT ONE TWO"


def test_frequency_decimal_completion():
    t, _ = L.grammar_fix("FREQUENCY ONE TWO THREE ONE TWO")
    assert t == "FREQUENCY ONE TWO THREE DECIMAL ONE TWO"
    t, n = L.grammar_fix("FREQUENCY ONE TWO THREE DECIMAL ONE TWO")
    assert n == 0  # 已有 DECIMAL 不重复


def test_saint_completion_no_double_prefix():
    t, n = L.grammar_fix("JOHNS INFORMATION A")
    assert t == "SAINT JOHNS INFORMATION A" and n == 1
    # 原版潜在 bug：完整台站名会被再前缀成 SAINT SAINT——清洗版必须不触发
    t, n = L.grammar_fix("SAINT JOHNS INFORMATION A WIND")
    assert t == "SAINT JOHNS INFORMATION A WIND" and n == 0


def test_no_values_invented():
    # 补全只动结构词：任何数字词序列不得增删改
    src = "WEATHER AT ONE THREE ZERO ZERO WIND"
    t, _ = L.grammar_fix(src)
    digits = lambda s: [w for w in s.split()
                        if w in L._DIGITS]  # noqa: E731
    assert digits(t) == digits(src)
