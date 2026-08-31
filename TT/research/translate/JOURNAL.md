# ATC 空管内容 EN→ZH 翻译研究 · 实验日志（translate/JOURNAL.md）

> 纪律：客观指标先行（数字保真/术语命中/回译一致性），全部为确定性规则，无主观打分。

---

## 2026-08-27

### T1 基础设施（承接前次会话）

- `glossary.py`：ATC 术语审计 + 数字读法转换 + chrF。
  关键坑修复：
  - VISIBILITY 编码式读法 `ONE FIVE TWO FOUR THOUSAND` 的对半拆分规则
    （前半=能见度码 15，后半×1000=云底高 24000）；
  - 中文侧数字提取需剥离「（30.23 英寸汞柱）」等值括注，防止注释被二次计数；
  - ZULU 歧义处理：时间语义（→UTC）与 RNAV ZULU（进近程序名）分流审计。
- `template_translate.py`（T3 规则模板翻译器，充当"数值正确性裁判"）完成 a/b 两稿，
  数字保真 = 术语命中 = **1.0**（生成式系统无法自证，必须由规则系统背书）。

### T2 模型资产

- Qwen2.5-7B-Instruct（bf16，15GB）经 hf-mirror 下载完毕；
- M2M-100-418M 权重首次下载中断（缺 pytorch_model.bin）+ 残留 .lock 与重复下载进程
  互相等待 → 清理后单进程补齐。

### T3 裸通用 MT 对照（T4 M2M-100 直译）——失败模式存档

样例（WIND TWO FOUR ZERO AT FIVE → 「赢得两四零到五」；ALTIMETER → 「最大」）：

| 失败类型 | 例证 |
|---|---|
| 同形异义 | WIND(风)→赢得；ALTIMETER(修压)→最大 |
| 拼读数字破坏 | ONE FIVE TWO FOUR THOUSAND→一五二千五千 |
| 音译地名 | SAINT JOHNS→圣约翰(丢失斯)、FOXTROT→福克斯特、ZULU→苏卢 |

结论：**通用 MT 不可用于空管域**，佐证"术语约束 + 模板裁决"路线必要性（PLAN §0.1）。
但 M2M 保留作为多系统共识的独立第三方（差异大的行触发人工级复核信号）。

### T4 Qwen2.5-7B 主力翻译（T1）

- 发现：模型从本地磁盘逐 shard 加载耗时 ~16min（IO 慢），按任务重载不可接受
  → 改造为 `run_translate_batch.py`：**单次加载**顺序跑 constrained/free × a/b 四组合，
  输出格式与 run_translate.py 完全兼容。
- 设计要点：
  - free variant：仅 SYSTEM_BASE（域角色设定，不给术语表）——衡量术语表注入的增益；
  - constrained variant：SYSTEM_BASE + GLOSSARY_HINT 术语硬约束；
  - 审计反馈闭环：numeric_fidelity / term_audit 不过 → 把失败行连同错误原因回喂修正，
    最多 3 轮，直至双满分或停止改善。

### T5 多系统共识与终稿选举（summarize_translate.py）

- 确定性选举规则：优先取"数字保真=1 且 术语命中=1"的 qwen_constrained；
  否则按 2×数字保真 + 术语命中 + 回译chrF 综合分最高者。
- 输出：results/<tag>/summary.md（对比矩阵）+ final_zh.txt（交付译文）。

### T6 回译一致性（M2M zh→en back-translation, chrF vs 原文 EN）

- 口径说明：回译 chrF 度量的是"中文译文可逆性"，数值受 M2M 自身回译能力上限约束
  （418M 小模型），因此只作系统间相对比较，不做绝对阈值判定。
- PLAN §4 设定的"回译 token-F1 ≥ 0.8"在实测前修正为相对口径（记录偏差理由：
  无多参考条件下绝对阈值缺乏依据，违反零主观纪律的反面即"拍脑袋阈值"）。

### T7 端到端延迟推演脚本（end_to_end.py）

- 组成：ASR 延迟（读 streaming events）+ Qwen 单句翻译墙钟实测；分句聚合逻辑初版
  对 draft 轨长音频不适用（J 轨词级数据稀疏），最终口径改为"词级 ASR 延迟 + 句翻译耗时"
  直接合成（见 T9）。

### T8 Qwen constrained 双满分达成（batch v2）

- 强化 GLOSSARY_HINT（VISIBILITY 编码式读法给出硬性目标译文 + 反例），重跑 batch：
  - **a/b 的 qwen_constrained 均达 numeric_fidelity=1.0、term_hit_rate=1.0**
    （含审计反馈闭环自动修复 b 版 ALTIMETER 3034→3033 错字）；
  - qwen_free 仅 SYSTEM_BASE：a 0.9167/0.8182，b 0.8333/0.8182
    → **术语表注入的量化增益**成立；
- 新坑记录：retry 提示词写 "格式 `N. 修正译文`" 被模型字面理解，输出行首出现 "N." 残留
  → parse_numbered 增加 "N." 剥离 + retry 提示改为明确真实行号示例。
- 终稿选举（summarize_translate.py）：a/b 均推荐 qwen_constrained，
  与 template 的 chrF 一致性 0.379 为全矩阵最高（双满分互证）；m2m_direct ≤0.146 远离全体。

### T9 端到端预算实测（e2e）

- Qwen 单句 ATC 翻译墙钟 median=**0.44s**（GPU bf16 greedy，抽样实测）；
- 联动 streaming draft 词延迟（1.35~2.08s）→ 端到端预算 ≈ **1.8–2.5s**；
- 产物：results/e2e_budget.json / e2e_latency.json。

### T10 修复轮：代码审查整改与全量重跑（2026-08-27 下午）

审查发现的缺陷全部修复后，用修复版代码把所有依赖产物重算了一遍：

1. `end_to_end.py` 分句失效（长音频整篇 1 句、p95 公式小样本取最大值）：
   改为阈值自适应降档 + nearest-rank 百分位。历史产物中的失真值
   （n_sentences=1，asr_final_delay median=218s）已被替换为真实值：
   - a 轨：2 句 / ASR 句定稿 median=2.86s / e2e p50=13.30s p95=52.38s
   - b 轨：3 句 / median=13.20s / e2e p50=14.66s p95=35.66s
   注：句级定稿延迟 ≠ 词级可用延迟；"听到词→中文可用 ≈1.8–2.5s"
   的预算结论（T9 词级口径）不受影响且经 K4 对齐复核成立。
2. `end_to_end.py` 无英文行时崩溃路径加固（改文献估计值口径）；
   翻译耗时抽样复用循环的句级配对显式标注 pairing 字段。
3. `glossary.py` 两处防护：「負」单字负号只跳 1 位（旧逻辑吞掉后续数字）；
   THOUSAND 前无数字时不再 IndexError。原语料未触发，审计结论（双满分）不变。
4. `template_translate.py` 定长假设告警：修压括注仅 4 位生成、跑道号>2 位
   打 warnings 并落盘 template.json；正常语料零告警，译文逐行不变。
5. `summarize_translate.py`：qwen 结果被剔除出选举池时打印警告；
   chrF 矩阵上三角补占位符。
6. **e2e_budget.json 重写**：旧文件 b 轨仅 2 个词样本（streaming flush 词
   缺 att_start 的连带受害者），现改为引用 evaluate_run K4 对齐口径
   （a n=242 p50=1.72 → 预算 2.16s；b n=200 p50=1.34 → 预算 1.78s），
   "≈1.8–2.5s" 结论经修复后全量数据复核依然成立。
7. 验证：py_compile 全部通过；glossary 边界单测通过；重跑 a/b 两 tag 的
   summarize 推荐终稿不变（qwen_constrained 双满分）。

### T11 随 streaming J10/J11 联动修订（2026-08-27 晚）

- streaming 侧端点检测从能量 VAD 改为 RMS-CV 调制门 + K4b 评测窗对齐周期
  （详见 streaming JOURNAL J10/J11），draft 轨词延迟 a 轨逐位不变
  （median 1.72 / p95 2.87，n=242）→ **a 轨预算 2.16s 不变**。
- b 轨 draft 词延迟随 K4b 新窗重算：median 1.34→**1.94s**、p95 6.41→**5.74s**
  （n=197，match_ratio 0.853）→ **b 轨预算 1.78→2.38s**。
  旧 1.34s 是窗错配（周期间停顿被切）下的边界伪象，偏乐观。
- 修订落点：e2e_budget.json（b 轨 + note）、FINAL_REPORT §4.2 表格。
  综合结论由"≈1.8–2.5s"更新为"**≈2.2–2.4s**"（a/b 同口径，final 轨句级
  定稿延迟 p50 a 13.8s / b 14.1s 是另一口径，不参与"词级可用"预算）。

### J8 翻译输入源切换：deep 终稿 → 流式识别输出（2026-08-28）

- **问题确认**：此前 translate 全部以 deep 终稿（a_final/b_final）为输入，
  从未消费 streaming 的流式识别输出（用户指出该脱节）。
- **新管线**：
  1. `stream_to_lines.py`：流式长文本按台站 ATIS 模板行循环贪心切行
     （内容 100% 来自流式输出，模板只用于插换行）；a=60 行（5×12）、b=12 行。
  2. `run_translate.py --input stream_en.txt`：术语表约束 + 数字/术语审计重试闭环。
- **修复两处管线缺陷**：
  - 批量 60 行时部分行解析失败（None）导致 numeric_fidelity 崩溃 → None 行
    以空串占位；chat max_new 1024→3072（60 行译文长度需要）；
  - 模型对周期首行 "SAINT JOHNS" 顽固保留英文、反馈重试 3 轮无效 →
    新增 `fallback_terms` 确定性兜底（重试穷尽后仍有 miss 时按术语表 EN 正则
    字面替换为首选中文）。
- **结果（qwen2.5-7b-instruct constrained）**：
  - stream_a：数字保真 1.0 / 术语命中 1.0（兜底后）/ 未解析 0；
  - stream_b：数字保真 1.0 / 术语命中 1.0 / 未解析 0；
  - 与 deep 输入版（a_final/b_final 均 1.0/1.0）持平。
