"""ATIS 公开标准词表与短语语法（P4 清洗版，2026-08-31）。

清洗原则：只保留可由公开标准（ICAO Annex 3/10、FAA ATIS 播报模板、
ICAO/FAA 读数标准）独立推导、且**不删除引擎已输出词**的规则——删词必须
先"知道受损形式"= 观测产物 = 金参考监督渠道。
已剔除（原为 R 系列对着 K4 评测集真值残错分析调出的定点规则，见
streaming/JOURNAL.md 2026-08-28 R1-R9）：
  - 定点误听→真值映射：REPORT NINER→DEW POINT MINUS、DESCENDING LEVEL→
    VISIBILITY、TAROM( REPORT)→TEMPERATURE( DEW POINT)、DECIMALITY→
    VISIBILITY、SHAMROCK→SAINT、FANDAS/FIENDER/DEANDER→GANDER、
    ALTITUDE THREE→ALTIMETER THREE（含当日数值；ALTITUDE 由词表通用
    模糊纠错覆盖）
  - 定点幻觉删词：TIME/BOX/BYE/SAME/GERMANS、("TIME.","")
  - 表面形式规则：DEGREES→ZULU WIND、VISIBILITY <4位数>→+THOUSAND FEET
  - 定点删词（质疑轮 P4Q2，违反"不删引擎词"判据的自审漏检）：
    WEATHER IS→WEATHER AT、AT (IS|IT)→AT——左侧候选集恰等于 JOURNAL
    L326 观测残错清单，词形证据为零（IS/AT 相似度 0），与已删的
    TAROM→TEMPERATURE 同案同证；零先验下插入错误无合法修复路径，
    多出的 IS/IT 如实计为插入错误
  - 词表 FOXTROT（信息代号字母每班次更换 = 当日具体内容）

档位声明（质疑轮 P4Q1 采纳双轴判据后降级）：规则合法性的完整判据是
**双轴**同时无监督——内容可由公开标准推导 且 选择未被真值触达。本文件
保留的规则（含 VOCAB 选定、0.66 阈值、全部脱落补全与正字法变体）在
选择轴上均可追至 JOURNAL R1-R9 对着 K4 评测集真值的残错清单（哪条规则
值得存在=观测产物），故一律定性为"**评测集监督下选定的公开词法先验**"，
不得称"零先验/合法补全"。旧 R 系列 WER（0.1303/0.2711/0.0211/0.2254）
归"评测集监督自适应"档；真 L0 须禁用本文件（或仅用可独立重导规则另臂
评测）。
"""

# 域词表（大写）。台站名属公开地理信息（CYYT=SAINT JOHNS）。
# P4：已删 FOXTROT（当日信息代号）。
VOCAB = set("""
SAINT JOHNS INFORMATION WEATHER AT ZERO ZULU WIND VISIBILITY
TEMPERATURE DEW POINT MINUS ALTIMETER APPROACH RNAV ILS RUNWAY INFORM
GANDER CENTER ON FREQUENCY DECIMAL AS WHEN REQUESTED INITIAL CONTACT
LANDING AND DEPARTING THAT YOU HAVE ONE TWO THREE FOUR FIVE SIX SEVEN
EIGHT NINE HUNDRED THOUSAND HEAVY
""".split())

# 短语级正字法纠错：公开词表词的常见误拼变体（= normalize 逐词模糊纠错的
# 短语级补充，内容不依赖当日真值），按序应用。
PHRASE_RULES = [
    ("ARNAV", "RNAV"),
    ("VISABILITY", "VISIBILITY"),
    ("WETHER", "WEATHER"),
    ("WHETHER", "WEATHER"),
    ("ALCIMBER", "ALTIMETER"),
]

_NINER = {"NINER": "NINE"}

# 数字词（ICAO 读法；NINER 已归一为 NINE）
_DIGITS = {"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
           "EIGHT", "NINE", "OH"}


def grammar_fix(text):
    """ATIS 公开播报语法层：固定槽位校验与补全（P4 清洗版）。

    只增补公开结构词（ZULU/WIND/DECIMAL/INFORM ATC/SAINT），不删除引擎
    已输出词、不填任何具体数值/代号。返回 (text, n_fix)。
    """
    import re

    t = text
    n = 0

    def sub_count(pat, rep, s):
        nonlocal n
        s2, k = re.subn(pat, rep, s)
        n += k
        return s2

    D = r"(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|OH)"

    # 结尾句标准句式（FAA/ICAO ATIS）：INFORM ATC THAT YOU HAVE INFORMATION
    # <代号>。安全补全（不删词）：保护完整形 → 补漏 ATC → 裸锚点前置整句。
    _ph = "\x01"
    t, _k0 = re.subn(r"\bINFORM ATC THAT YOU HAVE INFORMATION\b", _ph, t)
    t, k2 = re.subn(r"\bINFORM THAT YOU HAVE INFORMATION\b", _ph, t)
    t, k3 = re.subn(r"\bTHAT YOU HAVE INFORMATION\b", _ph, t)
    t = t.replace(_ph, "INFORM ATC THAT YOU HAVE INFORMATION")
    n += k2 + k3
    # WEATHER AT <HHMM> [ZULU]：缺 ZULU 补全（ICAO 时组须带 ZULU）
    t = sub_count(
        rf"\bWEATHER AT ((?:{D} ){{3}}{D})(?! ZULU)\b",
        r"WEATHER AT \1 ZULU", t)
    t = sub_count(rf"\bZULU (?!WIND )((?:{D} ){{2}}{D} AT )", r"ZULU WIND \1", t)
    # FREQUENCY <数字组>：缺 DECIMAL 补全（ICAO/FAA 读数标准连接词；
    # 3+2 优先，2+2 兜底）
    t = sub_count(
        rf"\bFREQUENCY ((?:{D} ){{2}}{D}) ((?:{D} ){{1,2}}{D})\b",
        r"FREQUENCY \1 DECIMAL \2", t)
    t = sub_count(
        rf"\bFREQUENCY ((?:{D} ){{1}}{D}) ((?:{D} ){{1,2}}{D})\b",
        r"FREQUENCY \1 DECIMAL \2", t)
    # 开场固定句式 = 公开台站名 + INFORMATION <代号>；SAINT 弱读脱落时补全。
    # (?<!SAINT ) 防已有完整台站名被二次前缀（P4 顺带修复原版重复 bug）。
    t = sub_count(r"(?<!SAINT )\bJOHNS INFORMATION\b", "SAINT JOHNS INFORMATION", t)
    # P4Q2：AT(IS|IT)→AT 已删——删引擎输出词=观测驱动的定点删词，违反本层
    # "只增不删"完整性约束；插入错误照实付出（多 1 词计 1 插入）。
    t = re.sub(r"\s+", " ", t).strip()
    return t, n


def normalize(text):
    """短语规则 → 逐词模糊纠错（编辑距离阈值 0.66，首字母须一致）。

    注：0.66 阈值为对评测集调参的超参（非内容泄漏，但引用数字须注明）。
    """
    import re
    from difflib import SequenceMatcher

    t = text.upper()
    for pat, rep in PHRASE_RULES:
        t = t.replace(pat, rep)
    t = re.sub(r"\s+", " ", t).strip()

    out = []
    for w in t.split():
        if w in _NINER:
            out.append(_NINER[w])
            continue
        if w in VOCAB or w.rstrip(".,?").isdigit() or len(w) < 4:
            out.append(w)
            continue
        best, sc = w, 0.0
        for v in VOCAB:
            # 双向首字母约束：防 NEW→ONE 这类跨首字母误纠
            if not (v[0] == w[0] or (w[0] in v[:2] and v[0] in w[:2])):
                continue
            s = SequenceMatcher(None, w, v).ratio()
            if s > sc:
                best, sc = v, s
        out.append(best if sc >= 0.66 else w)
    return " ".join(out)
