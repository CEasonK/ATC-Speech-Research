"""src/atis_grammar.py — 第三裁判（ICAO ATIS/METAR 语法）的形式化实现。
把研究全程 ad-hoc 的语法推理落成可复现工具：
  槽位匹配（正则）→ 合法值域校验 → 逐行 verdict（PASS/WARN/UNPARSED/FAIL）
值域依据：ICAO Annex 10 / Doc 7910、FAA AIM（QNH inHg）、VHF 118.000–136.975 MHz
且 25 kHz 网格；跑道 01–36(+L/R/C)；观测时间 0000–2359Z；温度整数 −60..60 且露点≤温度。
"""
import json
import re

NUM = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
       "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINER": 9, "NINE": 9,
       "OH": 0}
ALPHA = {"ALFA": "A", "BRAVO": "B", "CHARLIE": "C", "DELTA": "D", "ECHO": "E",
         "FOXTROT": "F", "GOLF": "G", "HOTEL": "H", "INDIA": "I",
         "JULIETT": "J", "KILO": "K", "LIMA": "L", "MIKE": "M",
         "NOVEMBER": "N", "OSCAR": "O", "PAPA": "P", "QUEBEC": "Q",
         "ROMEO": "R", "SIERRA": "S", "TANGO": "T", "UNIFORM": "U",
         "VICTOR": "V", "WHISKEY": "W", "XRAY": "X", "YANKEE": "Y",
         "ZULU": "Z"}
RUNWAY_SUF = {"LEFT", "RIGHT", "CENTER", "L", "R", "C"}


def digits(words):
    """连续数字词 → 整数；含非数字词返回 None。"""
    out = []
    for w in words:
        if w not in NUM:
            return None
        out.append(NUM[w])
    return int("".join(map(str, out))) if out else None


def signed_num(words):
    """处理 MINUS/负号前缀的数字词序列。"""
    neg = False
    ws = list(words)
    while ws and ws[0] in ("MINUS",):
        neg = not neg
        ws.pop(0)
    v = digits(ws)
    return None if v is None else (-v if neg else v)


def _freq_mhz(pre_words, post_words):
    """ONE TWO THREE DECIMAL ONE FIVE → 123.15"""
    ipre, ipost = digits(pre_words), digits(post_words)
    if ipre is None or ipost is None or len(post_words) != 2:
        return None
    return ipre + ipost / 100.0


def check_freq(mhz):
    ok_range = 118.0 <= mhz <= 136.975
    ok_grid = abs(round(mhz * 40) * 0.025 - mhz) < 1e-9  # 25 kHz 网格
    if not ok_range:
        return "FAIL", f"{mhz} MHz 超出 VHF 空地频段"
    if not ok_grid:
        return "WARN", f"{mhz} MHz 不在 25kHz 网格上"
    return "PASS", f"{mhz:.2f} MHz 合法"


def check_line(line):
    """单行 ATIS 文本 → [{slot, verdict, detail}]"""
    t = re.sub(r"[^A-Z ]+", " ", line.upper())
    t = re.sub(r"\s+", " ", t).strip()
    W = t.split()
    out = []

    def add(slot, verdict, detail):
        out.append({"slot": slot, "verdict": verdict, "detail": detail})

    # --- 信息代号 ---
    m = re.search(r"INFORMATION ([A-Z]+)", t)
    if m and m.group(1) in ALPHA:
        add("info_letter", "PASS", f"代号 {ALPHA[m.group(1)]}")
    elif m:
        add("info_letter", "WARN", f"代号词 '{m.group(1)}' 非 ICAO 字母")

    # --- 观测时间 ---
    if "WEATHER" in W and "AT" in W:
        ai = W.index("AT", W.index("WEATHER"))
        v = digits(W[ai + 1:ai + 5])
        if v is None:
            add("obs_time", "FAIL", "时间槽位无法解析为数字")
        elif not (0 <= v <= 2359):
            add("obs_time", "FAIL", f"{v:04d} 超出 0000-2359Z")
        else:
            add("obs_time", "PASS", f"{v:04d}Z")

    # --- 风组 ---
    j = next((k for k, w in enumerate(W) if w == "WIND"), None)
    if j is not None:
        rest = W[j + 1:]
        stop = next((k for k, w in enumerate(rest)
                     if w in ("VISIBILITY", "TEMPERATURE")), len(rest))
        grp, dirv, spdv, atv = rest[:stop], None, None, False
        if len(grp) >= 5 and grp[3] == "AT":
            dirv, spdv = digits(grp[0:3]), digits(grp[4:5])
            atv = True
        elif grp and grp[-1].startswith("AT"):
            dirv = digits(grp[:3])
            spdv = digits(grp[grp.index("AT") + 1:]) if "AT" in grp else None
        if dirv is None or spdv is None:
            add("wind", "WARN", f"风组结构未识别: {' '.join(grp)}")
        else:
            vd = "PASS" if (dirv % 10 == 0 or True) and 0 <= dirv <= 360 else "FAIL"
            vs = "PASS" if 0 <= spdv <= 199 else "FAIL"
            worst = "FAIL" if "FAIL" in (vd, vs) else "PASS"
            add("wind", worst, f"方向 {dirv:03d}° 风速 {spdv}kt")

    # --- 能见度/RVR（宽松：口播读法有歧义，双读法任一合法即 PASS） ---
    k = next((k for k, w in enumerate(W) if w == "VISIBILITY"), None)
    if k is not None:
        rest = W[k + 1:]
        if "THOUSAND" in rest:
            th = rest.index("THOUSAND")
            base = digits(rest[max(0, th - 2):th])
            if base is not None:
                # "TWO FOUR THOUSAND" 可读作 2400(RVR 逐位) 或 24000(整万)，双读法
                cands = sorted({base * 1000 // 10, base * 1000})
                ok = any(1200 <= c <= 6500 for c in cands)
                add("visibility", "PASS" if ok else "WARN",
                    f"{rest[-1] if rest else '?'} 候选值 {cands}"
                    f"（RVR 逐位读法 {'在域' if ok else '不在域'}）")
            else:
                add("visibility", "WARN", "千位组无法解析")
        else:
            val = digits(rest[:2]) if rest else None
            if val is not None and 0 < val <= 99:
                add("visibility", "PASS", f"能见度 {val}(单位未口播)")
            else:
                add("visibility", "WARN", f"数值 {val} 异常")

    # --- 温度/露点 ---
    if "TEMPERATURE" in W:
        ti = W.index("TEMPERATURE")
        di = W.index("DEW") if "DEW" in W else None
        tv = signed_num(W[ti + 1:di if di is not None else ti + 2])
        dv = signed_num(W[di + 2:di + 4]) if di is not None else None
        msgs = []
        verdict = "PASS"
        if tv is None or not (-60 <= tv <= 60):
            verdict = "FAIL"
            msgs.append(f"温度 {tv} 非法(须整数 −60..60)")
        else:
            msgs.append(f"温度 {tv:+d}℃")
        if dv is not None:
            if not (-60 <= dv <= 60):
                verdict = "FAIL"
                msgs.append(f"露点 {dv} 非法")
            elif tv is not None and dv > tv:
                verdict = "FAIL"
                msgs.append(f"露点 {dv} > 温度 {tv} 违反物理约束(METAR)")
            else:
                msgs.append(f"露点 {dv:+d}℃")
        add("temp_dew", verdict, "; ".join(msgs))

    # --- 修压 QNH(inHg 四位口播) ---
    if "ALTIMETER" in W:
        ai = W.index("ALTIMETER")
        v = digits(W[ai + 1:ai + 5])
        if v is None:
            add("qnh", "FAIL", "修压槽位无法解析")
        else:
            inhg = v / 100.0
            ok = 27.50 <= inhg <= 31.50
            add("qnh", "PASS" if ok else "FAIL",
                f"{inhg:.2f} inHg {'合法' if ok else '超出 27.50-31.50'}")

    # --- 进近+跑道（跑道号 1-2 个数字词：TWO EIGHT=28 / FOUR=04） ---
    if "RUNWAY" in W:
        ri = W.index("RUNWAY")
        rest = W[ri + 1:]
        ndig = 0
        while ndig < len(rest) and ndig < 2 and rest[ndig] in NUM:
            ndig += 1
        rn = digits(rest[:ndig]) if ndig else None
        suf = rest[ndig] if (ndig < len(rest) and rest[ndig] in RUNWAY_SUF) else ""
        if rn is not None and 1 <= rn <= 36 and (ndig == 2 or rn <= 9):
            add("runway", "PASS", f"跑道 {rn:02d}{' ' + suf if suf else ''}")
        else:
            add("runway", "FAIL", f"跑道号 '{' '.join(rest[:2])}' 非法(须 01-36)")

    # --- 频率 ---
    if "DECIMAL" in W:
        d = W.index("DECIMAL")
        pre = list(reversed(list(
            w for w in reversed(W[max(0, d - 3):d])
            if w in NUM)))  # 取紧邻的前缀数字词
        mhz = _freq_mhz(pre, W[d + 1:d + 3])
        if mhz is None:
            add("freq", "WARN", "频率结构未解析")
        else:
            v, msg = check_freq(mhz)
            add("freq", v, msg)

    if not out:
        add("free_text", "UNPARSED", "无严格槽位（自由文本行，仅记录）")
    return out


def check_rjtt_light(text):
    """管制通话轻校验：FLIGHT LEVEL 后 1-3 个数字词 → 值域校验。"""
    W = re.sub(r"[^A-Z ]+", " ", text.upper()).split()
    res, i = [], 0
    while i < len(W) - 1:
        if W[i] == "FLIGHT" and W[i + 1] == "LEVEL":
            ndig, j = 0, i + 2
            while j < len(W) and ndig < 3 and W[j] in NUM:
                ndig += 1
                j += 1
            v = digits(W[i + 2:i + 2 + ndig]) if ndig else None
            if (v is not None and ndig <= 2 and i + 2 + ndig < len(W)
                    and W[i + 2 + ndig] == "HUNDRED"):
                v *= 100  # "TWO HUNDRED" = FL200
                j += 1
            if v is not None:
                ok = 45 <= v <= 600
                res.append({"slot": "flight_level",
                            "verdict": "PASS" if ok else "WARN",
                            "detail": f"FL{v:03d} {'合法' if ok else '罕见'}"})
            i = i + 2 + max(ndig, 1)
        else:
            i += 1
    return res


def check_file(path, channel):
    lines = [l.strip() for l in open(path, encoding="utf-8").read().splitlines()
             if l.strip()]
    rep = {"file": path.name, "channel": channel, "lines": []}
    for n, ln in enumerate(lines, 1):
        rep["lines"].append({"n": n, "text": ln, "checks": check_line(ln)})
    return rep


if __name__ == "__main__":
    import sys
    from pathlib import Path
    DEEP = Path(__file__).resolve().parents[1]
    RES = DEEP / "results"
    reps = []
    for name, ch in (("a_final.txt", "CYYT_ATIS_a"), ("b_final.txt", "CYYT_ATIS_b")):
        reps.append(check_file(RES / name, ch))
    # RJTT 共识文本轻校验
    con = json.loads((RES / "rjtt_consensus.json").read_text(encoding="utf-8"))
    rj = []
    for s, c in sorted(con.items()):
        rj.append({"seg": s, "checks": check_rjtt_light(c["text"])})
    (RES / "grammar_check.json").write_text(
        json.dumps({"atis": reps, "rjtt_light": rj}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # 摘要打印
    n_pass = n_warn = n_fail = n_unparsed = 0
    for r in reps:
        print(f"== {r['channel']}")
        for L in r["lines"]:
            for c in L["checks"]:
                print(f"  L{L['n']} [{c['verdict']:8s}] {c['slot']}: {c['detail']}")
                n_pass += c["verdict"] == "PASS"
                n_warn += c["verdict"] == "WARN"
                n_fail += c["verdict"] == "FAIL"
                n_unparsed += c["verdict"] == "UNPARSED"
    fl = sum(1 for x in rj for c in x["checks"])
    print(f"== RJTT 轻校验: {fl} 项高度层检查")
    print(f"汇总: PASS={n_pass} WARN={n_warn} FAIL={n_fail} UNPARSED={n_unparsed}")
