"""P4d 三裁判终审：温度结构 / b频率 / visibility云况结构。

背景（v5 结论）：
  - a/b 是不同日期录音（b altimeter=3033 双裁判一致 vs a=3023）-> 字段独立裁决
  - visibility=VISIBILITY（v3+语法定案），但 "ONE FIVE TWO FOUR THOUSAND FEET" 内部结构存疑
  - temperature 双裁判分歧：atc="ONE DECIMAL NINER ONE" vs v3="ONE DEW POINT MINUS ONE"
裁判：whisper-atc / whisper-large-v3 / whisper-large-v3-turbo-atcosim

2026-08-25 review 修复：原 vis_struct 锚 "ZERO AT FIVE TEMPERATURE" 与
temperature 锚 "THOUSAND FEET ALTITUDE ..." 在定稿转写中均不相邻
（前者跳过 VISIBILITY 行、后者 ALTITUDE 并不存在），三裁判 anchor_t 各自
漂移到不同循环实例，同窗比较失效。已改为跨行相邻锚文本。
⚠ results/adjudication_v6.json 中 vis_struct_* 与 temperature_* 的旧数值为
   坏锚产物、不可采信（温度结论已由 v8 增量曲线+切片解码独立支撑，不受影响）。
2026-08-25 已修复锚文本并重跑刷新 JSON。注：各裁判 anchor_t 不同属正常——
   锚句在循环广播中每周期重现一次，不同裁判命中不同周期的同一内容，
   裁判内部排序仍可比（v6 结论以 v8+ 增量曲线为准）。
"""
import json
import sys
from pathlib import Path

import torch

DEEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEP / "src"))
from nll_scorer import NLLScorer

TT = Path("/siyuan/FunASR_extracted/FunASR-main/TT")
CH = {
    "CYYT_ATIS_a": str(TT / "audio" / "CYYT_ATIS_a.wav"),
    "CYYT_ATIS_b": str(TT / "audio" / "CYYT_ATIS_b.wav"),
}

# ---------------- 变体 ----------------
# 温度：覆盖 NINER/NINE、DECIMAL/POINT/直连、有无 DEW POINT
temps = [
    "TEMPERATURE ONE DECIMAL NINER ONE",                 # atc v5 冠军
    "TEMPERATURE ONE DEW POINT MINUS ONE",               # v3 v5 冠军
    "TEMPERATURE ONE DECIMAL NINER DEW POINT MINUS ONE",
    "TEMPERATURE ONE DECIMAL NINE DEW POINT MINUS ONE",
    "TEMPERATURE ONE NINER DEW POINT MINUS ONE",
    "TEMPERATURE ONE NINE DEW POINT MINUS ONE",
    "TEMPERATURE ONE POINT NINER DEW POINT MINUS ONE",
    "TEMPERATURE ONE DEW POINT MINUS TWO",
]
temp_full = [f"THOUSAND FEET {t} ALTITUDE" for t in temps]

# b 频率：是否漏听 ONE TWO
freqs = [
    "CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE AS REQUESTED",
    "CENTER ON FREQUENCY THREE DECIMAL ONE FIVE AS REQUESTED",
    "CENTER ON FREQUENCY ONE TWO TWO DECIMAL ONE FIVE AS REQUESTED",
    "CENTER ON FREQUENCY ONE TWO THREE POINT ONE FIVE AS REQUESTED",
]

# visibility 云况结构
vis_struct = [
    "AT FIVE VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE",
    "AT FIVE VISIBILITY ONE FIVE FEW TWO FOUR THOUSAND FEET TEMPERATURE",
    "AT FIVE VISIBILITY ONE FIVE CLOUDS TWO FOUR THOUSAND FEET TEMPERATURE",
    "AT FIVE VISIBILITY FIFTEEN KILOMETERS FEW TWO FOUR THOUSAND FEET TEMPERATURE",
    "AT FIVE VISIBILITY ONE AND ONE HALF MILES FEW TWO FOUR THOUSAND FEET TEMPERATURE",
    "AT FIVE VISIBILITY ONE FIVE SCATTERED TWO FOUR THOUSAND FEET TEMPERATURE",
]

CONTESTS = [
    # 锚必须取定稿转写中的跨行相邻文本，否则 find_anchor_window 会漂移
    ("temperature_a", ["CYYT_ATIS_a"], "THOUSAND FEET TEMPERATURE ONE", temp_full, 5.0),
    ("temperature_b", ["CYYT_ATIS_b"], "THOUSAND FEET TEMPERATURE ONE", temp_full, 5.0),
    ("freq_b", ["CYYT_ATIS_b"], "RUNWAY TWO EIGHT INFORM CENTER", freqs, 5.0),
    ("freq_a", ["CYYT_ATIS_a"], "RUNWAY TWO EIGHT INFORM CENTER", freqs, 5.0),
    ("vis_struct_a", ["CYYT_ATIS_a"], "ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE", vis_struct, 5.0),
    ("vis_struct_b", ["CYYT_ATIS_b"], "ONE FIVE TWO FOUR THOUSAND FEET TEMPERATURE", vis_struct, 5.0),
]

JUDGES = [
    ("atc", str(TT / "models" / "whisper-large-v3-finetuned-for-ATC")),
    ("v3", "openai/whisper-large-v3"),
    ("turbo_atcosim", "tclin/whisper-large-v3-turbo-atcosim-finetune"),
]


def run(scorer, tag):
    out = []
    for name, chans, anchor, variants, hw in CONTESTS:
        rec = {"contest": name, "judge": tag, "channels": {}}
        for ch in chans:
            t_anchor = scorer.find_anchor_window(CH[ch], anchor)
            ranked = []
            for v in variants:
                r = scorer.score_constrained(CH[ch], v, t_anchor, half_width=hw)
                ranked.append({"variant": v, **r})
            ranked.sort(key=lambda x: x["score"])
            rec["channels"][ch] = {"anchor_t": t_anchor, "ranked": ranked}
            for r in ranked:
                print(f"[{name}|{tag}|{ch}] {r['score']:.4f}  {r['variant'][:72]}", flush=True)
            print(f"==> {name}|{tag}|{ch}: WIN -> {ranked[0]['variant'][:80]}", flush=True)
        winners = [c["ranked"][0]["variant"] for c in rec["channels"].values()]
        rec["dual_agree"] = len(set(winners)) == 1
        out.append(rec)
    return out


def main():
    results = []
    for tag, mdir in JUDGES:
        print(f"===== judge: {tag} =====", flush=True)
        try:
            sc = NLLScorer(mdir)
            results += run(sc, tag)
            del sc
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"judge {tag} failed: {e}", flush=True)

    outf = DEEP / "results" / "adjudication_v6.json"
    outf.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("saved:", outf)


if __name__ == "__main__":
    main()
