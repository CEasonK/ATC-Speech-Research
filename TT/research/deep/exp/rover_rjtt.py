"""RJTT 共识合成（ROVER 思想的裁决版）：基于 validate_rjtt2 的三引擎候选+NLL+过滤+投票结果，
逐段合成最终文本。每段裁决依据（多引擎一致性 / NLL / ATC 域语法 / 跨段一致性）记入 basis。
输出 results/rjtt_final.txt（覆盖 v2）+ rjtt_consensus.json
"""
import json
from pathlib import Path

DEEP = Path(__file__).resolve().parents[1]

MANIFEST = json.loads((DEEP / "segments" / "RJTT_CONTROL" / "manifest.json").read_text())
SEGS = {f"seg{k:02d}": m for k, m in enumerate(MANIFEST)}

CONSENSUS = {
    "seg00": {
        "conf": "medium-low",
        "text": "GOLF ZERO JULIETT CONTROL ONE ZERO YANKEE AEROFLOT PASSING ONE EIGHT NINER FOUR FLIGHT LEVEL TWO FOUR ZERO JULIETT CONTROL ONE ZERO YANKEE AEROFLOT THANK YOU GOLF ZERO CLIMB MAINTAIN FLIGHT LEVEL THREE TWO ZERO CLIMB MAINTAIN FLIGHT LEVEL THREE TWO ZERO JULIETT CONTROL ONE ZERO YANKEE AEROFLOT",
        "basis": "atc 基础版（NLL 1.080 胜出，且与 turbo 结构聚类）；TOWER→CONTROL 按 3:1 多数（turbo×2 'control'、qwen 'Control'）及 FL320 管制放行语义（塔台不放行航路高度）；呼号 GOLF ZERO JULIETT / AEROFLOT 与 turbo 的 'juliett airfr' 冲突无法裁决（各引擎内部自洽），保留 NLL 胜者原词",
    },
    "seg01": {
        "conf": "medium",
        "text": "TOKYO CONTROL GOOD AFTERNOON SIERRA EIGHT NINER SIX CLIMBING FLIGHT LEVEL TWO HUNDRED SIERRA EIGHT NINER SIX TOKYO CONTROL CLIMB MAINTAIN FLIGHT LEVEL THREE ZERO ZERO REQUEST DIRECT TO MAIDA CLIMBING FLIGHT LEVEL THREE ZERO ZERO DIRECT TO MAIDA SIERRA EIGHT NINER SIX",
        "basis": "开场 TOKYO CONTROL 取 qwen+turbo 2:1（atc 的 PAPA KILO KILO 为孤例，且 seg06/07/08 三引擎均确认本频率台名为 TOKYO CONTROL）；呼号 SIERRA EIGHT NINER SIX 取 turbo——跨段决定性证据：turbo 在 seg01 与 seg08 两次独立产出同一呼号，且 v3 NLL 两段均全场最优（0.749 / 1.506），atc 的 SHAMROCK 记为竞争假设；中段 REQUEST DIRECT TO 取 atc（域标准申请短语，turbo 的 'you can' 非标准）；第二次台名 DEPARTURE→TOKYO CONTROL 取 qwen（2:1 结构）",
    },
    "seg02": {
        "conf": "low",
        "text": "NINE SEVEN TWO THREE CROSS MADIGU AT FLIGHT LEVEL THREE TWO ZERO",
        "basis": "turbo NLL 胜出（2.780 < atc 3.222，qwen 语法否决）；CROSS MADIGU AT 为域标准越点指令且与 seg03 交叉印证（同一定位点的两次放行）；段首疑被 VAD 截断（呼号可能不完整），置信度低",
    },
    "seg03": {
        "conf": "medium",
        "text": "JAPAN AIR NINE TWO THREE CROSS MADIGU AT FLIGHT LEVEL THREE TWO ZERO",
        "basis": "呼号 JAPAN AIR 取 qwen+turbo 2:1（atc 的 JETSTAR 孤例；turbo 产出 japanair 与 qwen 的 Japan Air 同源）；数字 NINE TWO THREE 取 turbo（NLL 最优引擎 2.491）；CROSS MADIGU AT 按域语法补全（turbo 有 CROSS、atc 漏动词、qwen 'equals' 乱码）；定位点拼写统一为 MADIGU（seg02/seg03 交叉归一）",
    },
    "seg04": {
        "conf": "medium",
        "text": "ORANGE NINER ONE EIGHT THREE QUEBEC ONE THREE",
        "basis": "atc 为唯一合理候选（qwen 'bigger than Duffy' 黑名单否决；turbo 'air inter/sabena' NLL 5.726 全场最差且无结构对应）；无第二引擎佐证，置信度中",
    },
    "seg05": {
        "conf": "low",
        "text": "TAXI POSSIBLY ONE TWO NINE ONE EIGHT",
        "basis": "turbo NLL 显著胜出（2.682 vs atc 5.416；qwen \"That's possible, anyway\" 双重否决）；1.6s 切段疑为发射中截，POSSIBLY 疑为劣化音频误听（会话词出现在频率上不合常理），置信度低",
    },
    "seg06": {
        "conf": "medium-high",
        "text": "TOKYO CONTROL FEDEX ONE FIVE HEAVY JOHNSON LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO FEDEX ONE FIVE TOKYO CONTROL CLIMBING LEVEL THREE TWO ZERO INITIALLY CLIMB THREE TWO ZERO FEDEX ONE FIVE",
        "basis": "三引擎全簇一致（唯一 3/3 结构对应段），atc NLL 最优（0.725）；HEAVY 后的 JOHNSON 与 turbo/qwen 的 'jet of / junk jet of' 冲突未裁决（三方各不相同），保留 NLL 胜者原词；其余全部字段三引擎一致",
    },
    "seg07": {
        "conf": "medium",
        "text": "TOKYO CONTROL JAPAN AIR TWO THREE NINER LEAVING FLIGHT LEVEL ONE SEVEN FIVE CLIMBING FLIGHT LEVEL TWO ZERO ZERO JAPAN AIR TWO THREE NINER TOKYO CONTROL CLEARED DIRECT IGOTO CLIMB FLIGHT LEVEL THREE ZERO ZERO DIRECT IGOTO CLIMB FLIGHT LEVEL THREE ZERO ZERO JAPAN AIR TWO THREE NINER",
        "basis": "呼号 JAPAN AIR TWO THREE NINER 取 atc+turbo 2:1（qwen 'Japania T three nine' 乱码）；管制指令 CLEARED DIRECT 取 turbo+域语法（atc 的 VICTOR/OSCAR 为孤例插入；turbo 'cleared to' 为放行标准词）；定位点拼写 IGOTO 取 atc，turbo 的片假名リゴート为同音转写佐证；'DIRECT…CLIMB FL300' 双重出现为 atc+turbo 独立一致→判定为音频事实（管制指令+飞行员复诵）",
    },
    "seg08": {
        "conf": "medium",
        "text": "TOKYO CONTROL SIERRA EIGHT NINER SIX REQUEST FLIGHT LEVEL THREE EIGHT ZERO OR FOUR ZERO ZERO",
        "basis": "turbo NLL 最优（1.506）；TOKYO CONTROL 3:1（atc 的 'CAN YOU' 孤例）；呼号 SIERRA EIGHT NINER SIX 与 seg01 跨段一致（turbo 两次独立产出同一呼号，v3 NLL 两段均最优），atc 的 SHAMROCK / qwen 的 CHANGI 记为竞争假设",
    },
}


def main():
    lines, out = [], {}
    for s, m in SEGS.items():
        c = CONSENSUS[s]
        out[s] = {"t0": m["t0"], "t1": m["t1"], **c}
        lines.append(f"[{s}] {m['t0']:.1f}-{m['t1']:.1f}s (conf={c['conf']})\n{c['text']}")
    (DEEP / "results" / "rjtt_final.txt").write_text("\n\n".join(lines) + "\n")
    (DEEP / "results" / "rjtt_consensus.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("saved rjtt_final.txt (v3 consensus) / rjtt_consensus.json")


if __name__ == "__main__":
    main()
