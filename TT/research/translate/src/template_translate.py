"""确定性模板译法（空管通播规范文体）：作为多系统共识的第三方。

本语料为标准 ATIS 六类句型，正则槽位化解析后按中文空管播报体生成。
用法：python template_translate.py --tag a_final
输出：results/<tag>/template.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from glossary import en_numbers  # noqa: E402

TT_ROOT = SRC.parent.parent.parent

AP_VAL = {"FOXTROT": "Foxtrot（F）", "ALFA": "Alfa（A）", "BRAVO": "Bravo（B）",
          "CHARLIE": "Charlie（C）", "DELTA": "Delta（D）", "ECHO": "Echo（E）"}

_SPELLED2D = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
              "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}
_W_RAW = r"(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)"
_NUM_RUN = rf"({_W_RAW}(?:\s+{_W_RAW})*)"


def dnum(s):
    return "".join(_SPELLED2D[w] for w in s.split())


def translate_line(line, warns=None):
    """按句型模板翻译一行；槽位位数异常时输出保底译法并向 warns 追加告警。"""
    u = re.sub(r"\s+", " ", line.strip().upper())
    # T1 机场信息+天气时间
    m = re.match(rf"SAINT JOHNS INFORMATION (\w+) WEATHER AT {_NUM_RUN} ZULU$", u)
    if m:
        code = AP_VAL.get(m.group(1), m.group(1))
        return f"圣约翰斯机场通播信息 {code}，气象观测时间 {dnum(m.group(2))} 世界协调时。"
    # T2 风
    m = re.match(rf"WIND {_NUM_RUN} AT {_NUM_RUN}$", u)
    if m:
        return f"风向 {dnum(m.group(1))}，风速 {int(dnum(m.group(2)))} 节。"
    # T3 能见度/云（编码式读法：偶数个数字词对半拆 前半=能见度 后半×1000=云高）
    m = re.match(rf"VISIBILITY {_NUM_RUN} THOUSAND FEET$", u)
    if m:
        digs = m.group(1).split()
        if len(digs) >= 4 and len(digs) % 2 == 0:
            h = len(digs) // 2
            vis = dnum(" ".join(digs[:h]))
            cel = str(int(dnum(" ".join(digs[h:]))) * 1000)
        else:
            vis, cel = dnum(m.group(1)), ""
        return f"能见度 {vis}，云底高 {cel} 英尺。" if cel else f"能见度 {vis}。"
    # T4 温度露点
    m = re.match(rf"TEMPERATURE {_NUM_RUN} DEW POINT MINUS {_NUM_RUN}$", u)
    if m:
        return f"温度 {int(dnum(m.group(1)))}℃，露点零下 {int(dnum(m.group(2)))}℃。"
    # T5 修压
    m = re.match(rf"ALTIMETER {_NUM_RUN}$", u)
    if m:
        v = dnum(m.group(1))
        if len(v) == 4:
            return f"修正海压 {v}（{v[:2]}.{v[2:]} 英寸汞柱）。"
        if warns is not None:
            warns.append(f"ALTIMETER 位数异常({len(v)}位): {v}，省略英寸汞柱括注")
        return f"修正海压 {v}。"
    # T6 进近
    m = re.match(rf"APPROACH RNAV ZULU RUNWAY {_NUM_RUN}$", u)
    if m:
        v = dnum(m.group(1))
        if warns is not None and len(v) > 2:
            warns.append(f"跑道号位数异常({len(v)}位): {v}，按原样输出")
        rw = f"{int(v):02d}" if len(v) == 2 else v
        return f"进近方式 RNAV Z，使用跑道 {rw}。"
    # T7 频率通报（a: AS REQUESTED / b: WHEN REQUESTED）
    m = re.match(
        rf"INFORM GANDER CENTER ON FREQUENCY {_NUM_RUN} DECIMAL {_NUM_RUN}"
        rf" (AS REQUESTED|WHEN REQUESTED)$", u)
    if m:
        freq = dnum(m.group(1)) + "." + dnum(m.group(2))
        tail = "如需引导可联系" if m.group(3) == "AS REQUESTED" else "当被要求时联系"
        return f"{tail}冈德中心，频率 {freq}。"
    # T8 起降跑道+首次联系
    m = re.match(rf"APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY {_NUM_RUN}$",
                 u)
    if m:
        v = dnum(m.group(1))
        if warns is not None and len(v) > 2:
            warns.append(f"跑道号位数异常({len(v)}位): {v}，按原样输出")
        rw = f"{int(v):02d}" if len(v) == 2 else v
        return f"落地和起飞使用跑道 {rw}，首次联系进近时报告。"
    # T9 收到信息
    m = re.match(r"INFORM ATC THAT YOU HAVE INFORMATION (\w+)$", u)
    if m:
        code = AP_VAL.get(m.group(1), m.group(1))
        return f"联系时告知管制（ATC）你已收到通播信息 {code}。"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    inp = TT_ROOT / "research/deep/results" / f"{a.tag}.txt"
    lines = [l.strip() for l in inp.read_text().splitlines() if l.strip()]
    warns = []
    zh = [translate_line(l, warns) for l in lines]
    unparsed = [i + 1 for i, z in enumerate(zh) if z is None]
    outdir = SRC.parent / "results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    res = {"system": "rule-template-v1", "en": lines,
           "zh": [z or "" for z in zh], "unparsed_lines": len(unparsed),
           "unparsed_idx": unparsed, "warnings": warns}
    for w in warns:
        print(f"[warn] {w}")
    (outdir / "template.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["zh"], ensure_ascii=False, indent=0))
    print("unparsed:", unparsed)


if __name__ == "__main__":
    main()
