"""ATC 域术语表、数字读法转换与客观审计工具。

红线：全部为确定性规则，无主观打分。
"""
import re

# ---------- 英文拼读数字 ----------
_DIG = {
    "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
    "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
}
# 触发数值边界/修饰的词
_BOUNDARY = {"THOUSAND", "FEET", "AT", "DECIMAL", "POINT"}
# 负号前缀（ATC 读法：MINUS/NEGATIVE/NEG 是符号本身，不是数字载荷）
_NEG_WORDS = {"MINUS", "NEGATIVE", "NEG"}

WORD_RE = re.compile(r"[A-Z]+")

# 千分位：只认半角逗号，首段 1-3 位 + 后续每组恰好 3 位，前后不接数字/小数点。
# 全角「，」在本语料中是小句分隔符（「能见度 15，云底高 24000 英尺」），不合并。
_THOUSANDS_COMMA_RE = re.compile(r"(?<![\d.])(\d{1,3})(?:,\d{3})+(?![\d.])")


def merge_thousands(s):
    """千分位合并：'15,240'->'15240'，'1,234,567'->'1234567'。
    不合并非连续 3 位的形态（'1,5'、'1234,567' 原样保留），两侧同规则调用。"""
    return _THOUSANDS_COMMA_RE.sub(lambda m: m.group(0).replace(",", ""), s)


def canon_num(tok):
    """数字 token 规范化（比较语义的唯一形式）：可选负号 + 数码串，
    去掉小数点与千分位逗号、保留前导零、去掉多余前导负号。
    例：'-15'->'-15'，'30.23'->'3023'，'15,240'->'15240'，'0200'->'0200'。
    保留前导零是刻意的：时刻 '0200' 不应与 '200' 混同。"""
    t = merge_thousands(str(tok)).replace(".", "").replace("．", "")
    neg = t.lstrip().startswith("-")
    t = t.lstrip().lstrip("-")
    return ("-" if neg else "") + t


def en_numbers(line):
    """把一行英文转成数字载荷列表（小数点在比较中忽略）。

    规则（确定性，ATIS 格式先验）：
    - 连续拼读数字 → 串接为 ASCII 数字；
    - THOUSAND → 紧邻数字段 ×1000；但 VISIBILITY 子句例外：形如
      'VISIBILITY D1 D2 D3 D4 THOUSAND FEET'（编码式读法）时按对半拆：
      前半 = 能见度值，后半 ×1000 = 云底/云量高。本语料全部命中该模式。
    - DECIMAL → 并入前段（比较时省略小数点，'123.15'=='12315'）。
    - MINUS/NEGATIVE/NEG → 负号，作用于紧随其后的数字段，输出带符号 token；
      符号后（跨空格）无数字则作废，不会外溢到再下一个数字段。
    例：'WIND TWO FOUR ZERO AT FIVE' -> ['240', '5']
        'FREQUENCY ONE TWO THREE DECIMAL ONE FIVE' -> ['12315']
        'VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET' -> ['15', '24000']
        'WEATHER AT ZERO TWO ZERO ZERO ZULU' -> ['0000']
        'ALTIMETER THREE ZERO TWO THREE' -> ['3023']   # 中文若写 30.23 同样可比
        'TEMPERATURE ONE DEW POINT MINUS ONE' -> ['1', '-1']
        'DEW POINT MINUS TWO ZERO' -> ['-20']
        'MINUS' -> []          # 符号后无数字：作废
    """
    u = " ".join(WORD_RE.findall(merge_thousands(line.upper())))
    mvis = re.search(
        r"VISIBILITY\s+((?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)"
        r"(?:\s+(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE))*)\s+THOUSAND",
        u,
    )
    if mvis:
        digs = mvis.group(1).split()
        # ATIS 编码式读法先验：VISIBILITY 后为偶数个拼读数字时对半拆
        # （前半=能见度编码，后半×1000=云底/云量高）
        if len(digs) >= 4 and len(digs) % 2 == 0:
            h = len(digs) // 2
            va = "".join(_DIG[d] for d in digs[:h])
            vb = str(int("".join(_DIG[d] for d in digs[h:])) * 1000)
            rest = en_numbers(u[: mvis.start()] + " " + u[mvis.end():])
            return [va, vb] + rest

    words = WORD_RE.findall(u)
    out, buf = [], []
    pending_decimal = False
    pending_neg = False

    def flush():
        nonlocal pending_neg
        if buf:
            tok = "".join(buf)
            out.append(("-" + tok) if pending_neg else tok)
            buf.clear()
        pending_neg = False

    for w in words:
        if w in _DIG:
            if pending_decimal and out:
                out[-1] += _DIG[w]
            else:
                buf.append(_DIG[w])
            # 保持 pending_decimal 直到出现非数字词
        elif w in _NEG_WORDS:
            flush()
            pending_decimal = False
            pending_neg = True
        elif w == "THOUSAND":
            pending_decimal = False
            flush()
            if out:  # 防护：THOUSAND 前无数字时跳过，避免 IndexError
                out[-1] = str(int(out[-1]) * 1000)
        elif w == "DECIMAL":
            flush()
            pending_decimal = True
        else:
            pending_decimal = False
            flush()
    flush()
    return out


# ---------- 中文侧数字提取 ----------
_ZH_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# 中文数字 -> ASCII 解析（覆盖逐位读法与位值组合两类）
_ZH_DIG = {"零": "0", "〇": "0", "一": "1", "壹": "1", "二": "2", "两": "2",
           "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8",
           "九": "9"}
_ZH_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_seq_to_ascii(seq):
    """一段连续中文数字字符 -> ASCII 数字串；无法解析返回 None。
    支持：逐位读法（二四零->240）；位值组合（一万五千二百四十->15240）。"""
    if not seq:
        return None
    # 逐位全可解为单数字且无单位 -> 直接拼接
    if all(c in _ZH_DIG for c in seq):
        return "".join(_ZH_DIG[c] for c in seq)
    # 位值组合解析
    total, section, cur = 0, 0, 0     # total/段(万)/当前累积
    for ch in seq:
        if ch in _ZH_DIG:
            cur = cur * 10 + int(_ZH_DIG[ch])
        elif ch in _ZH_UNIT:
            u = _ZH_UNIT[ch]
            if u == 10000:
                section = (section + cur) * 10000
                total += section
                section, cur = 0, 0
            else:
                section += (cur if cur else 1) * u
                cur = 0
        else:
            return None
    val = total + section + cur
    return str(val) if val else None


_ZH_NEG_RE = re.compile(r"(零下|負|负|−|－|(?<![\d.])-)")


def zh_numbers(line):
    """提取中文译文里的数字载荷，输出**带符号** token（与 en_numbers 同一形式）。

    规范化：阿拉伯/中文数字统一为数码串，小数点剥离（'30.23'->'3023'），千分位
    逗号合并（'15,240'->'15240'），前导零保留（'0200' 不与 '200' 混同）。
    负号来源：零下 / 负 / 負 / − / － / 紧邻数字的半角 '-'，作用域 = 紧随其后的
    一个数字段；符号后（跨任意非数字字符）若始终没有数字则作废，不外溢到下一段。
    先剥离修压类括注「（xx.xx 英寸汞柱）」，避免等值注释被二次计数。
    例：'零下五度' -> ['-5']；'温度 1 露点零下 1' -> ['1', '-1']；
        '能见度 15，云底高 24000 英尺' -> ['15', '24000']；'15,240' -> ['15240']
    """
    s = re.sub(r"[（(][^（）()]*英寸汞柱[^（）()]*[）)]", "", line)
    s = merge_thousands(s)
    nums = []
    pending_neg = False
    i, n = 0, len(s)
    while i < n:
        if _ZH_NEG_RE.match(s, i):
            pending_neg = True          # 符号本身不产出 token；连续符号取最后一个
            i += 2 if s.startswith("零下", i) else 1
            continue
        m = _ZH_NUM_RE.match(s, i)
        if m:
            tok = m.group(0).replace(".", "").lstrip("-")
            nums.append(("-" if pending_neg else "") + (tok or "0"))
            pending_neg = False
            i = m.end()
            continue
        j = i
        while j < n and (s[j] in _ZH_DIG or s[j] in _ZH_UNIT):
            j += 1
        if j > i:
            v = _cn_seq_to_ascii(s[i:j])
            if v is not None:
                nums.append(("-" if pending_neg else "") + v)
            pending_neg = False         # 解析失败的碎片同样让符号作废
            i = j
        else:
            i += 1
    return [d for d in nums if d not in ("", "-")]


def numeric_fidelity(en_lines, zh_lines):
    """逐行数字保真审计：两侧载荷经 canon_num 规范化后逐位序列一致→pass。

    比较语义 = 带符号数码串的有序序列相等（不做多重集/子集宽松匹配）：
    canon_num 只做 去小数点 + 千分位合并 + 负号归一，因此 '30.23' 与 '3023'
    等价、'-5' 与 '零下5' 等价，而 '0200' 与 '200' 不等价、'-5' 与 '5' 不等价。
    负号不再交给 term_audit 代管（MINUS 词条仅作术语提示，不参与数字判定）。
    返回 (ratio, details)；details 每项 {line,en,zh,ok,en_only,zh_only}。"""
    assert len(en_lines) == len(zh_lines), (len(en_lines), len(zh_lines))
    det = []
    good = 0
    for k, (e, z) in enumerate(zip(en_lines, zh_lines)):
        en_n = [canon_num(x) for x in en_numbers(e)]
        zh_n = [canon_num(x) for x in zh_numbers(z)]
        ok = en_n == zh_n
        good += ok
        d = {"line": k + 1, "en": en_n, "zh": zh_n, "ok": ok}
        if not ok:
            d["en_only"] = _seq_diff(en_n, zh_n)
            d["zh_only"] = _seq_diff(zh_n, en_n)
        det.append(d)
    return good / max(len(en_lines), 1), det


def _seq_diff(a, b):
    """a 中未被 b 逐位匹配掉的元素（保序，供反馈文案使用）。"""
    from collections import Counter
    left, out = Counter(b), []
    for x in a:
        if left[x] > 0:
            left[x] -= 1
        else:
            out.append(x)
    return out


# ---------- 术语审计 ----------
TERMS = [
    # (EN 正则, 中文期望正则(任一命中即过), 说明)
    (r"\bSAINT JOHNS\b|\bSAINT JOHN\b", r"圣约翰斯|圣约翰"),
    (r"\bINFORMATION FOXTROT\b|\bINFORMATION [A-Z]\b", r"信息|通播|情报"),
    (r"\bWEATHER\b", r"天气|气象"),
    # 仅行尾 ZULU 或 WEATHER AT ... ZULU 属时间语义；RNAV ZULU(进近名) 由 RNAV 条目负责
    (r"ZULU$(?<!RNAV ZULU)", r"世界协调时|协调世界时|UTC"),
    (r"\bRNAV ZULU\b", r"RNAV\s*Z"),
    (r"\bWIND\b", r"风向|风"),
    (r"\bVISIBILITY\b", r"能见度"),
    (r"\bFEET\b", r"英尺"),
    (r"\bTEMPERATURE\b", r"温度"),
    (r"\bDEW POINT\b", r"露点"),
    (r"\bMINUS\b", r"零下|-|负"),
    (r"\bALTIMETER\b", r"修正海压|高度表拨正|海压|修压"),
    (r"\bAPPROACH\b(?! RNAV)", r"进近"),
    (r"\bRUNWAY\b", r"跑道"),
    (r"\bGANDER\b", r"冈德|甘德|Gander"),
    (r"\bCENTER\b", r"中心"),
    (r"\bFREQUENCY\b", r"频率"),
    (r"\bLANDING\b", r"落地|着陆|降落"),
    (r"\bDEPARTING\b", r"起飞|离场|出发"),
    (r"\bATC\b", r"空管|管制|ATC"),
    (r"\bINITIAL CONTACT\b", r"首次联系|初次联系|第一次联系"),
    (r"\bAS REQUESTED\b|\bWHEN REQUESTED\b", r"按需|如需|需要时|要求时|按请求|按管制要求|当被要求"),
]


def term_audit(en_lines, zh_lines):
    """术语命中率审计。返回总命中率 + 未命中文案清单。

    total==0（该行集没有任何词表可审的术语）时命中率返回 **None**（不可比），
    不再默认 1.0——否则「无术语可审」会被当成「术语全命中」虚增指标。
    调用方兼容见 run_translate.py / 报告；misses 恒为列表，语义不变。"""
    hits, miss = 0, []
    total = 0
    for li, (e, z) in enumerate(zip(en_lines, zh_lines)):
        eu = e.upper()
        for pat, want in TERMS:
            if re.search(pat, eu):
                total += 1
                if re.search(want, z):
                    hits += 1
                else:
                    miss.append({"line": li + 1, "term": pat, "expected": want,
                                 "zh": z})
    rate = hits / total if total else None
    return rate, {"hit_rate": rate, "total": total, "hits": hits, "misses": miss}


def chr_f(ref, hyp, n_max=6, beta=2.0):
    """字符级 chrF（无参考系统间的回译一致性度量）。"""
    def ngrams(s, n):
        s = re.sub(r"\s+", " ", s.strip().lower())
        if len(s) < n:
            return [s] if s else []
        return [s[i:i + n] for i in range(len(s) - n + 1)]

    import math
    tp = fp = fn = 0.0
    eps = 1e-16
    for n in range(1, n_max + 1):
        g_ref, g_hyp = ngrams(ref, n), ngrams(hyp, n)
        from collections import Counter
        cr, ch = Counter(g_ref), Counter(g_hyp)
        overlap = sum((cr & ch).values())
        tp += overlap
        fp += sum(ch.values()) - overlap
        fn += sum(cr.values()) - overlap
    p, r = tp / (tp + fp + eps), tp / (tp + fn + eps)
    if p + r < eps:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
