# ATC ASR SOTA 文献提炼（d1）

## 与本任务直接相关的方法（按可落地性排序）

### 1. 多引擎假设融合（DLR OpenSky 2026 — 最强背书）
Wüstenbecker et al., "Can YouTube Stream Recordings Improve Speech Recognition for ATC?"
- 三种互补 ASR 架构并行转写 + **LLM transcript fusion** 合成伪标签
- 相对最佳单模型 **37% 相对提升**（controller 10.2% WER）
- 关键洞察：不同架构错误特性互补（complementary error characteristics）
- 论文自述未来方向：ATC grammar validators + multi-LLM consensus ← 正是我们 P4/P5
- **落地**：我们恰好有 3 引擎（ATC-Whisper / Qwen3-ASR / SenseVoice），错误特性差异大

### 2. ROVER + 假设质量排序（Jalalvand et al. 2017）
"Automatic Quality Estimation for ASR System Combination"
- 标准 ROVER 弱点：①依赖第一个假设当 skeleton，顺序影响大 ②置信分数常高估
- 方案：融合前先做 **segment 级假设质量排序**（QE）→ 比 oracle 只差一点
- **落地**：NLL 评分器 = 我们的 QE。排行榜排序 → 排序后的假设进对齐投票（P5），
  骨架用 NLL 最好的假设

### 3. N-best T5 约束解码（Ma et al., Interspeech 2023, Cambridge）
- 纠错模型输出**约束在 N-best 词表/格内** → 杜绝 LLM 自由发挥编造
- **落地**：若用 LLM 融合，输出必须由候选池中出现的词组合而成（P5 实现约束）

### 4. RLLM-CF 三阶段 LLM 纠错（2025, HUST/SJTU）
- error pre-detection → CoT 子任务迭代纠错 → **答案验证**（防 LLM 幻觉改对为错）
- GPT-4o 相对 CER/WER 降 9-21%
- **落地**：LLM 纠错后必须用 NLL 复核：纠错版 NLL 不降则回滚

### 5. Whisper-ATC（TU Delft ICRAT 2024）
- Whisper 微调 ATC 数据 SOTA：ATCO2 13.5%、ATCOSIM 1.17%（随机切分）
- **区域特定数据微调可再提升 60%** ← CYYT 是加拿大纽芬兰区域，
  jacktol 模型训练分布未必覆盖，这解释了它在 b 上的挣扎

### 6. Calm-Whisper（2025）
- Whisper-large-v3 幻觉主要由 3/20 个 decoder head 贡献（75%）
- 微调这三个头 → 非语音段幻觉降 80%，WER 仅涨 0.1%
- **启发**：a 音频 150-274s 满能量噪声段的幻觉 = 这些 head 的锅；
  我们用 no_speech_prob 过滤等效绕开

### 7. Whisper 原生解码机制（源码级）
- temperature fallback (0.0→1.0)：compression_ratio>2.4（复读）或 avg_logprob<-1.0 触发重试
- no_speech_prob>0.6 且 avg_logprob 低 → 判静音跳过（不触发 fallback）
- initial_prompt + **carry_initial_prompt=True**：每窗都带领域 prompt（openai/whisper 原生支持）
- **落地**：ATC 术语 prompt 注入是零成本提升项，值得实验（P6）

### 8. 中文专利（2025）：空管语境纠错
- 动态呼号池 + 静态指令参数库（航路点/移交频率）做语境校验
- **落地**：ATIS 报文的槽位值域 = 我们的"静态参数库"（P5 语法校验器）

## 方法论总结 → 我们的管线

```
候选池(3引擎×多参数×多切分) 
  → NLL 排序(裁判1: 声学似然, QE式前置排序)
  → 争议字段裁决(裁判2: ICAO语法+值域, 裁判3: 跨引擎一致性)
  → 对齐投票(ROVER变体: NLL最优当骨架)
  → 约束融合(输出限候选池词表; LLM若有)
  → NLL 复核回滚
```

## 待验证假设
- H1: NLL 排名与人工判读一致率 >80%（d4 验证）
- H2: 整周期切分（P3）候选的 NLL 系统性优于滑窗候选
- H3: ATC 术语 initial_prompt 降低幻觉词（P6 验证）
- H4: 3 引擎槽位级投票 > 任何单引擎（P5 验证）
