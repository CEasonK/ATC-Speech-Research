# ATC 空管内容 EN→ZH 翻译研究 · 详细规划（translate/）

> **【状态 2026-08-27】本计划已执行完毕** → 结论见 `FINAL_REPORT.md`，过程日志见 `JOURNAL.md`。
> 与计划的偏差（T2 NLLB 未落地、回译绝对阈值改相对口径、批量加载改造）均已在新目录下记录。

> 目标：把 CYYT_ATIS_a / CYYT_ATIS_b 的权威识别文本（deep 终稿）翻译成**专业、准确**的中文空管术语对话。
> 用户对此领域零背景，要求"翻译准"。无中文参考译文 → 全程客观裁判制（与 deep 同纪律）。
> 红线：只允许在 `research/translate/` 内写文件；`audio/`、`deep/`、`results/`（TT 级）只读。
> 翻译输入 = deep 终稿（用户语境："在已有的语音识别下面进行翻译"），翻译模块与流式识别解耦，
> 但最后给出"流式识别→流式翻译"的端到端延迟推演（用 streaming/ 的事件流数据）。

---

## 0. 任务特殊性（调研结论）

1. **ATIS 广播不是对话，是结构化信息播报**：
   `SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU / WIND TWO FOUR ZERO AT FIVE / ...`
   中文民航有对应的标准术语体系（MH/T 4014-2003《空中交通无线电通话用语》+ 国内 ATIS 实际用法），
   例如标准中文 ATIS 播报格式："圣约翰斯信息 Foxtrot：天气：0200Z，风向240，风速5节，能见度15000英尺…"
   → 翻译不是自由 MT，而是**术语映射 + 模板重排**，领域词表可穷举。
2. **关键术语难点**（逐词确定，不许 MT 自由发挥）：
   - 数字读法还原：`TWO FOUR ZERO AT FIVE` → "240 度，风速 5 节"（风向度数、风速节）
   - `ONE FIVE TWO FOUR THOUSAND` → 15000 英尺（千位拆分读法）
   - `TEMPERATURE ONE DEW POINT MINUS ONE` → 气温 1 摄氏度，露点 -1 摄氏度
   - `ALTIMETER THREE ZERO TWO THREE` → 修正海压 1023 hPa（ATIS 省略千位 1）
   - `RNAV ZULU` → RNAV Zulu（朱鹭/字母名保留国际名）
   - `GANDER CENTER` → 甘德中心（Gander 国际通用译名）
   - `INFORM ATC THAT YOU HAVE INFORMATION FOXTROT` → "告知管制已接收信息 F"
3. **无参考译文的客观裁判**（沿用 deep 方法论）：
   - ① **回译一致性**：ZH→EN 回译后与原文 EN 做 token 级 F1/语义字段核对（数字、方位、单位一个都不能丢）
   - ② **字段完整性审计**：从英文原文解析出结构化字段表（wind_dir, wind_speed, vis, temp, dew, qnh, approach, freq, runway...），
     逐字段检查译文是否覆盖且数值正确 → 生成 `field_audit.json`（机器可验证，零主观）
   - ③ **多系统共识**：≥2 个独立翻译系统输出一致处直接采信；不一致处由字段审计裁决
   - ④ **术语表硬约束**：ICAO/民航术语表（~80 条，人工从 MH/T 4014 + 公开中文 ATIS 样例整理）作为
     post-hoc 校验 + LLM 翻译的 system prompt 注入

## 1. 翻译系统候选

| # | 系统 | 规模 | 部署 | 角色 |
|---|------|------|------|------|
| T1 | **Qwen2.5/Qwen3 系列 7B（bf16，hf-mirror 下载）** | 15GB | transformers 4.57.6 直接跑，3090 24G 足够 | 主力：LLM 翻译 + 术语 system prompt + few-shot 中文 ATIS 样例 |
| T2 | **NLLB-200-distilled-600M**（Meta） | 2.5GB | transformers | 对照组：通用 NMT 无术语约束，证明"裸 MT 在 ATC 域的失败模式" |
| T3 | **模板化术语翻译器（规则+词表）** | 0 | 纯 Python | 兜底+裁判：对 ATIS 这种模板化文本做结构化字段抽取→按中文 ATIS 标准模板重排，
     确定性输出，作为 T1 的交叉校验基线 |
| T4 | （可选）M2M-100-418M | 1.7GB | transformers | 第三投票源 |

**裁决规则**：三系统字段级投票 + T3 模板器做"数值正确性"终审（数值只信 T3 的解析，因为 T3 是从
英文数字读法按规则还原的，无幻觉）。

## 2. 目录结构

```
translate/
├── PLAN.md / JOURNAL.md / FINAL_REPORT.md
├── src/
│   ├── atis_parser.py    # 英文 ATIS 文本 → 结构化字段（正则+词表，确定性）
│   ├── zh_atis_template.py # 字段 → 中文标准 ATIS 播报（T3 模板翻译器）
│   ├── run_llm_translate.py  # T1：Qwen LLM 翻译（system prompt 注入术语表+中文 ATIS 样例，batch 两篇）
│   ├── run_nllb.py       # T2：NLLB 对照
│   ├── run_m2m.py        # T4：M2M 对照
│   ├── backtranslate.py  # 回译（T1 反向 ZH→EN）+ 与原文字段对齐
│   ├── field_audit.py    # ②字段完整性审计（对每个系统的译文跑）
│   ├── glossary.py       # 术语表（EN-ZH 对照，~80 条，含来源注释）
│   ├── consensus.py      # 多系统字段级共识 → 最终译文
│   └── end_to_end.py     # 端到端推演：接 streaming/ 的事件流，模拟"流式识别出句→即时翻译"延迟
├── exp/    # 各实验脚本与 prompt 版本
├── results/
│   ├── t1_qwen/  t2_nllb/  t3_template/  t4_m2m/
│   ├── field_audit.json / backtranslation.json / consensus.json
│   ├── a_final_zh.txt / b_final_zh.txt   # ★ 交付：权威中文译文
│   └── e2e_latency.md                    # 流式识别→流式翻译 延迟推演
└── downloads/  # 模型权重（hf-mirror）
```

## 3. 术语表建设（glossary.py，翻译质量的地基）

来源（公开、可核查）：
1. MH/T 4014-2003《空中交通无线电通话用语》（民航行业标准，含中英对照标准词）
2. ICAO Doc 9432（Annex 10）标准发音与中文惯例
3. 国内机场中文 ATIS 实际播报格式（公开样例：各机场"信息 A/B"中文文本）
4. deep 项目已定案的词（SAINT JOHNS=CYYT 圣约翰斯、GANDER CENTER=甘德等）

术语表示例（节选）：
```
INFORMATION FOXTROT → 信息F（ATIS 字母序号惯例，不译 Foxtrot 全文）
WIND ... AT ... → 风向…，风速…节
VISIBILITY ... FEET → 能见度…英尺
DEW POINT → 露点
ALTIMETER → 修正海压（ATIS 语境；气象报告语境=气压高度表）
RNAV ZULU → RNAV Zulu程序
ON INITIAL CONTACT → 首次联络时
LANDING AND DEPARTING RUNWAY → 着陆和起飞跑道
```

## 4. 实验与验收

| 实验 | 内容 |
|------|------|
| x1 | atis_parser 对 a_final/b_final 解析，人工核对字段表（写入 JOURNAL） |
| x2 | T3 模板翻译 → 中文 ATIS 播报稿 v0 |
| x3 | T1 Qwen 翻译（3 版 prompt：无术语/术语表/术语表+few-shot）→ 选优 |
| x4 | T2/T4 对照翻译 |
| x5 | 回译一致性 + 字段审计 + 共识裁决 → 终稿 |
| x6 | 端到端延迟推演（用 streaming/ 事件流，若 streaming 未完成则用 offline timestamps 估算） |

**验收标准**：
- 字段审计 100% 覆盖（每个数值字段在译文中出现且数值正确）
- 回译 token-F1 ≥ 0.8（与英文原文）
- 终稿中所有术语与 glossary 一致（自动校验脚本）
- 人读版：给出"专业播报体"与"直译对照体"两个版本（前者面向管制员阅读习惯）

## 5. 执行顺序

1. glossary.py + atis_parser.py + zh_atis_template.py（T3 先行，它是裁判，必须先立起来）
2. 下模型（T2/T4 小模型先下，T1 Qwen 7B 后台下）
3. T1/T2/T4 翻译 → 回译 + 字段审计
4. 共识裁决 → 终稿
5. 端到端推演 + FINAL_REPORT
