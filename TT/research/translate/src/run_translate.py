"""Qwen 主翻译引擎（约束解码式提示词 + 客观审计反馈重试闭环）。

用法：python run_translate.py --input /path/a_final.txt --tag a_final \
        [--variant free|constrained] [--model models/qwen2.5-7b-instruct]
输出：results/<tag>/qwen_<variant>.json {en,zh,audit,retries}
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from glossary import (canon_num, numeric_fidelity,  # noqa: E402
                      term_audit)

TT_ROOT = SRC.parent.parent.parent
DEEP = TT_ROOT / "research" / "deep" / "results"

SYSTEM_BASE = (
    "你是一名资深民航空管（ATC）翻译专家，负责把英文 ATIS 通播/管制指令翻译成符合中国民航"
    "表达习惯的简体中文。要求：\n"
    "1) 所有数字必须与原文严格对应，不得增删改换；编码式读法保持位数"
    "（如 ALTIMETER 3023 表示 30.23 英寸汞柱）；\n"
    "2) 使用规范空管术语；\n"
    "3) 语气简洁正式，适合字幕/转写场景；\n"
    "4) 只输出译文本身。"
)

GLOSSARY_HINT = """
参考术语表（必须遵守）：
- SAINT JOHNS -> 圣约翰斯；GANDER CENTER -> 冈德中心
- INFORMATION FOXTROT -> 通播信息 Foxtrot（可写作 信息F）
- ZULU（时间上下文）-> 世界协调时（UTC）
- WIND xxx AT x -> 风向xxx，风速x节
- VISIBILITY -> 能见度；FEET -> 英尺
- ★VISIBILITY 编码式读法硬规则：
  `VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET` 必须译作「能见度 15，云底高 24000 英尺」；
  前半段拼读=能见度码，后半段×1000=云底高，严禁合并写作「15240 英尺」或「一万五千二百四十英尺」。
- TEMPERATURE -> 温度；DEW POINT -> 露点（负值写"零下x℃"）
- ALTIMETER -> 修正海压（如 三零二三 写作 3023，可注明30.23英寸汞柱）
- RNAV ZULU -> RNAV Z；RUNWAY TWO EIGHT -> 跑道28
- FREQUENCY ... DECIMAL ... -> 频率（如 ONE TWO THREE DECIMAL ONE FIVE -> 123.15）
- INITIAL CONTACT -> 首次联系；LANDING AND DEPARTING RUNWAY -> 落地和起飞跑道
- AS REQUESTED -> 按需/按管制要求；WHEN REQUESTED -> 当被要求时
"""

USER_TMPL = (
    "请将下面的英文 ATIS 通播逐行翻译为简体中文。共{N}行，"
    "按相同顺序输出编号列表（格式：`N. 译文`），不要增减行数。\n\n{body}"
)

# T2：只有词表变体（constrained，等价旧称 glossed）允许注入 GLOSSARY_HINT。
# free 变体**全程**（首轮 + 所有重试轮）都不得出现词表，否则消融对照失效
# （旧实现重试轮硬写 SYSTEM_BASE + GLOSSARY_HINT，qwen_free.json retries=3 已被污染）。
GLOSS_VARIANTS = {"constrained", "glossed"}


def system_prompt(variant):
    """首轮与重试轮共用同一个装配函数：词表只可能从这一处进入 prompt。"""
    return SYSTEM_BASE + ("\n" + GLOSSARY_HINT if variant in GLOSS_VARIANTS else "")


def load_model(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    mdl.eval()
    return tok, mdl


def chat(tok, mdl, system, user, max_new=3072):
    import torch

    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_numbered(text, n_expect):
    """从 '1. xxx' 编号列表里解析出 n 行译文；缺失行返回 None 占位。
    兼容模型把提示词占位符 'N.' 字面抄进行首的情况（一并剥离）。"""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[\.、\)）]\s*(.+)", ln.strip())
        if m:
            k = int(m.group(1))
            body = re.sub(r"^N[\.、\)）]\s*", "", m.group(2).strip())
            if body and k not in out:
                out[k] = body
    return [out.get(i + 1) for i in range(n_expect)]


def translate_lines(tok, mdl, en_lines, variant):
    body = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(en_lines))
    user = USER_TMPL.format(N=len(en_lines), body=body)
    txt = chat(tok, mdl, system_prompt(variant), user)
    return parse_numbered(txt, len(en_lines)), txt


def audit_problems(en_lines, zh_lines):
    """对当前 (en, zh) 全量重跑数字+术语审计，给出逐行问题向量。

    problems[i] = (数字问题数, 术语 miss 数)，全部来自确定性审计，无主观分。
    verdict 为同一信息的字符串形态，用于写盘/轨迹对比。
    重试环每轮都重新调用本函数（T1 的核心：反馈不得陈旧）。"""
    ratio_num, det = numeric_fidelity(en_lines, zh_lines)
    ratio_term, pack = term_audit(en_lines, zh_lines)
    miss_cnt = {}
    for m in pack["misses"]:
        miss_cnt[m["line"] - 1] = miss_cnt.get(m["line"] - 1, 0) + 1
    problems = [(0 if d["ok"] else 1, miss_cnt.get(i, 0))
                for i, d in enumerate(det)]
    bad_idx = [i for i, pv in enumerate(problems) if pv[0] or pv[1]]
    return {"ratio_num": ratio_num, "det": det, "ratio_term": ratio_term,
            "pack": pack, "problems": problems, "bad_idx": bad_idx,
            "verdicts": [f"num={'ok' if p[0] == 0 else 'bad'}|term_miss={p[1]}"
                         for p in problems]}


def _feedback_text(en_lines, zh_lines, bad_idx, cur):
    """把失败行连同**本轮**审计出的具体原因拼成回喂文本。"""
    fb = []
    for i in bad_idx:
        errs = []
        d = cur["det"][i]
        if not d["ok"]:
            errs.append(f"数字不一致: 原文应为 {d['en']}，译文给出 {d['zh']}")
        for mm in cur["pack"]["misses"]:
            if mm["line"] == i + 1:
                errs.append(f"术语缺失: 应含 {mm['expected']}")
        fb.append(f"{i+1}. 原文: {en_lines[i]}\n   当前译文: {zh_lines[i]}\n"
                  f"   问题: {'; '.join(errs) if errs else '审计未通过'}")
    return ("以下几行的中文译文存在数字/术语错误，请逐行修正后重新输出编号列表"
            "（只含这几行）。格式要求：每行以实际行号开头，如第3行则写 "
            "`3. 修正后的译文`——行号必须是真实数字，严禁照抄字母 N：\n\n"
            + "\n\n".join(fb))


def retry_failed(tok, mdl, en_lines, zh_lines, det_num=None, misses=None,
                 max_round=3, variant="constrained"):
    """自审计重试环（T1）：每轮重算审计，且只接受严格改进的候选行。

    与旧实现的三点差异：
    1) bad_idx 每轮基于**当前 zh_lines** 重新审计得出（旧实现只在进入时算一次，
       后续轮拿首轮反馈回喂已修好的行）；
    2) 候选行仅在 (数字问题数, 术语 miss 数) 两个分量上都不劣于当前行、且至少
       一个分量严格更优时才落盘，否则保留旧行（旧实现 `if fixed[i]` 无条件覆盖，
       实测会把审计通过的行改坏）；
    3) 重试 prompt 的词表注入走 system_prompt(variant)（T2）。
    det_num / misses 保留为形参只为兼容旧调用方（run_translate_batch 位置传参），
    判定一律以函数内重算为准。返回 (zh_lines, rounds)——仍是二元组，兼容旧解包。"""
    system = system_prompt(variant)
    zh_lines = list(zh_lines)
    rounds = []
    cur = audit_problems(en_lines, zh_lines)
    for r in range(1, max_round + 1):
        bad_idx = cur["bad_idx"]
        if not bad_idx:
            break
        txt = chat(tok, mdl, system,
                   _feedback_text(en_lines, zh_lines, bad_idx, cur), max_new=512)
        fixed = parse_numbered(txt, len(en_lines))
        cand, proposed = list(zh_lines), []
        for i in bad_idx:
            if fixed[i] and fixed[i] != zh_lines[i]:
                cand[i] = fixed[i]
                proposed.append(i + 1)
        if not proposed:
            rounds.append({"round": r, "bad": len(bad_idx), "proposed": 0,
                           "accepted": [], "rejected": [i + 1 for i in bad_idx],
                           "verdict_changes": [], "note": "no_candidate"})
            break
        nxt = audit_problems(en_lines, cand)
        accepted, rejected, changes = [], [], []
        for i in bad_idx:
            old_p, new_p = cur["problems"][i], nxt["problems"][i]
            strict = (new_p[0] <= old_p[0] and new_p[1] <= old_p[1]
                      and new_p != old_p)
            ok = (i + 1) in proposed and strict
            (accepted if ok else rejected).append(i + 1)
            changes.append({"line": i + 1, "before": cur["verdicts"][i],
                            "after": nxt["verdicts"][i] if ok else cur["verdicts"][i],
                            "candidate": nxt["verdicts"][i],
                            "accepted": ok})
        for i in accepted:                       # 只落地被接受的行
            zh_lines[i - 1] = cand[i - 1]
        rounds.append({"round": r, "bad": len(bad_idx), "proposed": len(proposed),
                       "accepted": accepted, "rejected": rejected,
                       "verdict_changes": changes})
        cur = audit_problems(en_lines, zh_lines)  # 落地后重新审计，供下一轮反馈
        if not accepted:
            break
    return zh_lines, rounds


def per_line_retry_stats(rounds):
    """把 rounds 还原成逐行 retry_rounds 与 verdict 轨迹（写进 meta/report）。"""
    per = {}
    for rd in rounds:
        for ev in rd.get("verdict_changes", []):
            st = per.setdefault(ev["line"], {"retry_rounds": 0, "accepted": 0,
                                             "rejected": 0, "verdicts": []})
            st["retry_rounds"] += 1
            st["accepted" if ev["accepted"] else "rejected"] += 1
            if not st["verdicts"] or st["verdicts"][-1] != ev["before"]:
                st["verdicts"].append(ev["before"])
            if ev["accepted"]:
                st["verdicts"].append(ev["after"])
    return [dict({"line": k}, **v) for k, v in sorted(per.items())]


def term_suggestions(en_lines, zh_lines, misses):
    """T3：裁判不改卷。审计只**建议**，绝不写回译文。

    旧 `fallback_terms` 直接把词表期望词替换进 zh_lines，之后 term_audit 再跑一遍
    就必然命中——术语命中率可被机械刷满，审计与被审计对象不再独立。现在改成：
    输出 suggestions（行号 / 命中原文 / 建议中文 / 理由），译文保持逐字不变，
    写入单独的 suggestions.json 供人工参考，且不参与任何打分路径。
    返回 (zh_lines 原样, suggestions 列表)。"""
    import re as _re

    sugg = []
    for m in misses or []:
        i, en_pat, expect = m["line"] - 1, m["term"], m["expected"]
        if i < 0 or i >= len(zh_lines):
            continue
        hit = _re.search(en_pat, en_lines[i].upper())
        if not hit:
            continue
        sugg.append({"line": i + 1, "en_term": hit.group(0),
                     "suggested_zh": expect.split("|")[0].strip(),
                     "accepted_alternatives": expect,
                     "current_zh": zh_lines[i],
                     "reason": "术语审计未命中期望中文；仅供参考，未写入译文"})
    return zh_lines, sugg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", required=True, help="来源标签（决定结果目录名）")
    ap.add_argument("--variant", default="constrained",
                    choices=["free", "constrained"])
    ap.add_argument("--model", default=str(SRC.parent / "models/qwen2.5-7b-instruct"))
    ARGS = ap.parse_args()

    inp = Path(ARGS.input)
    if not inp.exists() and (DEEP / f"{ARGS.tag}.txt").exists():
        inp = DEEP / f"{ARGS.tag}.txt"
    en_lines = [l.strip() for l in inp.read_text().splitlines() if l.strip()]

    outdir = SRC.parent / "results" / ARGS.tag
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"qwen_{ARGS.variant}.json"

    t0 = time.time()
    tok, mdl = load_model(ARGS.model)
    load_sec = time.time() - t0

    zh_lines, raw_out = translate_lines(tok, mdl, en_lines, ARGS.variant)
    # None 行（解析失败/截断）以空串占位，保证指标与重试闭环可继续
    none_cnt = sum(1 for z in zh_lines if z is None)
    zh_lines = [z or "" for z in zh_lines]

    before = audit_problems(en_lines, zh_lines)
    retries = []
    if before["bad_idx"]:
        zh_lines, retries = retry_failed(
            tok, mdl, en_lines, zh_lines,
            before["det"], before["pack"]["misses"], variant=ARGS.variant)

    after = audit_problems(en_lines, zh_lines)
    ratio_num, det_num = after["ratio_num"], after["det"]
    ratio_term, audit_pack = after["ratio_term"], after["pack"]
    # T3：这里只产出建议，不写回 zh_lines；打分子路径不再出现 fallback_terms。
    zh_lines, suggestions = term_suggestions(en_lines, zh_lines,
                                             audit_pack["misses"])
    (outdir / "suggestions.json").write_text(json.dumps(
        {"tag": ARGS.tag, "variant": ARGS.variant, "note": "人工参考，未参与打分/译文",
         "suggestions": suggestions}, ensure_ascii=False, indent=1))

    # 术语 total==0 -> ratio_term is None（不可比），不与 1.0 混计
    term_observable = audit_pack["total"] > 0

    result = {
        "source": str(inp), "system": "qwen2.5-7b-instruct", "variant": ARGS.variant,
        "load_sec": round(load_sec, 1),
        "en": en_lines, "zh": zh_lines,
        "raw_output_preview": raw_out[:600],
        "unparsed_lines": none_cnt,
        "metrics": {
            "numeric_fidelity": round(ratio_num, 4),
            # total==0 时 None=不可比（旧实现虚报 1.0）；命中数/总数一并落盘
            "term_hit_rate": (round(ratio_term, 4) if term_observable else None),
            "term_total": audit_pack["total"],
            "term_hits": audit_pack["hits"],
            "term_observable": term_observable,
        },
        "metrics_before_retry": {
            "numeric_fidelity": round(before["ratio_num"], 4),
            "term_hit_rate": (round(before["ratio_term"], 4)
                              if before["pack"]["total"] > 0 else None),
        },
        "numeric_detail": det_num,
        "term_misses": audit_pack["misses"],
        "retries": retries,
        "per_line_retries": per_line_retry_stats(retries),
        "suggestions_file": "suggestions.json",
    }
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["metrics"], ensure_ascii=False))
    print("saved:", outfile)


if __name__ == "__main__":
    main()
