"""流式识别输出 → 翻译行结构。

把 streaming 融合输出的长文本按台站 ATIS 模板行切回多行
（内容 100% 来自流式输出，模板只用于插入换行，不改词），
供 run_translate.py --input 消费。

用法：python stream_to_lines.py <transcript_final.txt> <template.txt> <out.txt>
"""
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# ATIS 句首关键词（模板文件是单行长文本，按此切成播报行结构）
LINE_BREAKS = re.compile(
    r"\s*(?=SAINT JOHNS|WIND |VISIBILITY |TEMPERATURE |ALTIMETER |"
    r"APPROACH RNAV|INFORM GANDER|APPROACH ON INITIAL|INFORM ATC THAT)")


def main():
    hyp_path, tpl_path, out_path = map(Path, sys.argv[1:4])
    hyp = hyp_path.read_text().split()
    tpl_text = " ".join(tpl_path.read_text().split())
    tpl_lines = [l.strip() for l in LINE_BREAKS.split(tpl_text) if l.strip()]

    # 循环贪心对齐：ATIS 循环播报，模板 9 行用尽后从头再来，
    # 直至 hyp 耗尽（尾部截断/残余同样按最近行结构输出，不丢弃）
    pos = 0
    li = 0
    out_lines = []
    while pos < len(hyp):
        want = tpl_lines[li % len(tpl_lines)].split()
        end = pos
        best, best_score = pos, -1.0
        # 行长按 ±40% 容差扫候选终点，取相似度最高的切点
        for cand in range(pos + 1, min(len(hyp), pos + int(len(want) * 2.2)) + 1):
            sm = SequenceMatcher(None, want, hyp[pos:cand], autojunk=False)
            sc = sm.ratio() - abs(cand - pos - len(want)) * 0.01
            if sc > best_score:
                best_score, best = sc, cand
        if best == pos:  # 防空切死循环
            best = min(pos + 1, len(hyp))
        out_lines.append(" ".join(hyp[pos:best]))
        pos = best
        li += 1

    Path(out_path).write_text("\n".join(out_lines) + "\n")
    print(f"{len(out_lines)} lines -> {out_path}")


if __name__ == "__main__":
    main()
