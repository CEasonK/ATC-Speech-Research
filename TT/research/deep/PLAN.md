# ATC 识别深度研究计划（multi-day）

> 授权：以顶尖 ATC ASR 研究员身份做多天开放研究。
> 约束：①无具体需求 ②仅 3 条音频（CYYT_ATIS_a / CYYT_ATIS_b / RJTT_CONTROL）③无真实对照文本。
> 目标唯一：让识别结果尽可能好（MER 尽可能低）。
> 资源：算力/token 不设限，conda 环境 `lingbot-map`。

## 核心方法论：三大裁判（没有对照文本怎么评好坏）

1. **声学似然（NLL 当耳朵）**：教师强制 `loss = model(feats, labels=text).loss`。
   "这段音频用哪条候选文本来解释最省力" —— 客观、可复现、不依赖我的主观阅读。
   Sanity check 必做：正确文本 vs 乱序文本的 loss 差距要显著，否则工具不可信。
2. **ICAO ATIS 报文语法**：标准报文有固定信息槽位序
   （机场名 → 信息代号 → 时间 → 跑道 → 风向风速 → 能见度/RVR → 天气 → 温度/露点 → QNH → 趋势 → 频率）。
   每个槽位有合法值域（频率 118.000–136.975；QNH 960–1050 hPa 等）。
   语法校验器 = 第二裁判，槽位缺失/非法值 = 可疑点。
3. **多系统交叉验证**：Whisper-large-v3 / ATC-finetuned Whisper / Qwen3-ASR / FunASR 多引擎输出，
   一致字段高置信，冲突字段进 P4 声学裁决。

## 已确立的领域知识线索（待声学验证）

- CYYT 疑为 St. John's Intl（纽芬兰），非 Charlottetown（那是 CYYG）→ "GANDER CENTER / MONCTON"
  属 Gander/Moncton FIR 合理；跑道号 "TWO EIGHT" 存疑（St. John's 现用跑道 11/29、16/34）
- 频率 "three three decimal one five"(33.15) 非法 → 疑为 135.15 或 133.15
- a 音频物理循环周期 28.6s（mel 自相关 corr=0.625），ATIS 本质即循环广播

## 阶段计划

### P0 工作区（d0）
- [x] research/deep/{lit/, src/, exp/, results/}
- [x] PLAN.md / JOURNAL.md

### P1 文献扫描 + 资产盘点（d1-d2）✅ 完成
- 文献线（WebSearch）：ATC ASR SOTA（HAT-VAD、ATCO2、UWB-ATCC）、ROVER 假设组合、
  Whisper zero-shot 提升技巧（prompt engineering / initial_prompt / temperature fallback 策略 /
  long-form decoding 陷阱）、LLM 后处理纠错（ATC 术语约束解码）、self-consistency 最新进展
- 资产线（本地）：HF 缓存全部模型、FunASR 模型目录、/siyuan/Qwen3_ASR、
  可用的对齐工具（whisperX/ctc-forced-aligner）、降噪资产盘点
- 产出：`lit/SOTA_notes.md` ✅、`results/assets_inventory.md` ✅

### P2 核心工具：NLL 评分器 + 第一版排行榜（d3-d4）✅ 完成
- `src/nll_scorer.py`：
  - 预计算全部窗口 encoder hidden states（30s 窗 stride 10s）批量复用
  - Score(candidate) = min over windows of per-token NLL（教师强制）
  - 附 sanity check：真文本 vs 乱序 vs 反义文本 loss 差距表
  - argmin 窗口时间戳 → 报文在音频中的定位（供 P3 用）
- 候选池收集：results/ 全部历史输出 + scratch 各实验产物 + best v3 交付物
  （预计 50+ 候选/音频）
- 产出：`exp/leaderboard_v1.md` —— 第一版客观排名，与我的主观判读对照，
  分歧点记录（分歧=最有信息量的样本）

### P3 相位对齐整周期切分（d5）✅ 完成（find_period.py / cut_instances.py）

### P4 争议字段声学裁决（d6）✅ 完成
- 冲突字段清单化：GANDER vs MONCTON、TWO EIGHT vs ONE ONE、33.15 vs 135.15 等
- 对每个争议值构造最小对立文本对，NLL 差值裁决 + 领域合法性过滤
- 产出：`results/adjudication.md` ✅（逐字段证据链汇总；原始数值在 adjudication_v1..v12 JSON）

### P5 对齐投票 + ATIS 语法校验器（d7）✅ 完成（路线修正）
- ~~词级时间戳 ROVER 槽位投票~~ → v3 时间戳路线失败（模型不产出 <|t|> token），
  **替代路线**：同窗对立计分 + 增量曲线 + 能量包络物理探测（更硬的证据）
- `src/atis_grammar.py` ✅ 槽位解析器 + 合法值域校验；终稿校验 26 PASS / 0 FAIL
  （`results/grammar_check.json`）；RJTT 采用 ROVER 式三引擎共识合成 ✅
- 产出：v_final 报文（逐段/逐字段置信度标注 + 依据）✅

### P6 信号处理以 NLL 为准绳 ✅ 完成（结论反向确认）
- 谱减法降噪变体 × 双裁判 NLL 复审：全部 4 组恶化 −8%~−27%
  （`results/denoise_audit.json`）→ 维持无降噪管线，且升级为"降噪有害"结论

### P7 自训练迭代 ⏸ 有意不启动（非跳过）
- 理由：仅 3 条音频时，"高置信伪标签微调 → 在同 3 条音频上验证"构成循环过拟合，
  任何表面收益都无法与记忆化区分，不提升真实识别质量；
  若未来接入真实 ATC 数据集（ATCO2/UWB-ATCC 类），P2-P5 的伪标签管线可直接复用为训练数据源

### P8 最终报告 ✅ 完成
- `FINAL_REPORT.md`（仓库根目录）：最终报文 + 全字段置信度 + 完整证据链 +
  方法论总结（可迁移到真实 ATC 数据集的做法清单）

## 纪律

- 每个实验先写假设再跑，结果无论正负都记 JOURNAL.md
- NLL 评分器必须先过 sanity check 才能用于决策
- 主观判读与 NLL 分歧时：不武断，两案并存进 P4 裁决
- 所有新候选文本都进池子，永不丢弃证据
