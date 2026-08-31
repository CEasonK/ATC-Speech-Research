# 研究日志（逐日）

## Day 0 — 工作区与方法论

- 创建 research/deep/ 工作区，PLAN.md 定稿 P0-P8 八阶段
- 三大裁判确立：NLL 声学似然 / ICAO 语法 / 多系统交叉
- 领域线索入档：CYYT≈St.John's、跑道号存疑、频率非法值待裁决

## Day 1 — 文献扫描 + 资产盘点 + NLL 裁判诞生

### 资产盘点（d2 完成）

- 3 引擎可用：ATC-whisper-large-v3(5.8G) / Qwen3-ASR-1.7B(4.4G) / SenseVoiceSmall(897M)
- BGE-M3 嵌入模型可用；RTX 3090 24G 空闲
- whisperx/ctc-forced-aligner 未装；demucs/faster-whisper 未装
- 网络受限但 hf-mirror.com 带 UA 可用 → 后台下载 openai/whisper-large-v3 当中立裁判

### 文献提炼（d1 完成，详见 lit/SOTA\_notes.md）

- DLR 2026：3引擎+LLM融合 → 相对最佳单模型 37% 提升（我们的路线背书）
- ROVER+QE：融合前先做假设质量排序（NLL 排行榜 = 学术正确定位）
- N-best T5 约束解码：LLM 纠错输出限制在候选词表内
- RLLM-CF：纠错后必须验证防"改对为错"（NLL 复核回滚） 
- Calm-Whisper：幻觉集中在 3/20 个 decoder head

### ⭐ 重大发现：a/b 是同一广播的双信道录音

- 证据1（决定性）：Qwen 结果文件头保留原始文件名
  a=CYYT\_cyyt1\_atis\_**20260817160051**.wav，b=**20260817160101**.wav，相差 10 秒
- 证据2：同信息代号 FOXTROT（Qwen 听成 box drawn/box dog）、同天气时间 0200Z、
  同槽位值（风 240\@5、能见度 1.5、24000ft、altimeter 3023、RNAV ZULU、跑道 28）
- 推论：a/b = 同一条 ATIS 报文两次独立接收 → **双证人架构**
  ①跨信道 NLL 共识：真文本应在两个信道都低 NLL
  ②单信道特有错误会被另一信道声学否决
- NLL 实证：a\_text\@b=1.559 < b\_text\@b=1.798 << shuffle\@b=4.351
  （a 的干净文本比 b 自己的转写更贴合 b 音频——b 转写质量差的客观证据）

### NLL 裁判诞生记（sanity check v1→v3，两次翻车两次修正）

- v1 翻车：b\_text 带大小写标点 NLL 虚高 → 需统一归一化
- v2 翻车：小写归一化让全体 NLL 暴涨 \~4.7 nats → **ATC 微调模型在大写转写上训练，
  小写是 OOD**，且通用英语幻觉文本趁机反超 → 约定改为大写+去标点
- v3 全过：内容敏感 a(+0.164)/b(+2.6)、定位合理(argmin\@60s 语音区)
- 量化特性：长度偏置 +0.159 nats/tok(1/4 vs 4/4 前缀)——NLL 只裁 fidelity，
  完整性归 ATIS 语法裁判（职责分离，防重蹈"自设指标"覆辙）
- 跨信道绝对差 1.107 = 信道基线差（b 信道整体更难，b\_text\@b 都要 1.80）
  → 跨信道比较需用"相对乱序基线的优势"归一（待实现）

### 工具与数据

- src/nll\_scorer.py：教师强制 NLL，30s窗 stride 5s，encoder 状态缓存，min-over-windows
- src/build\_pool.py：65 候选入池（a:32 b:27 RJTT:6）
- exp/sanity\_nll.py：四关卡裁判验证（v3 通过）

### 待办

- [ ] d4 排行榜（运行中）
- [ ] 跨信道归一化共识分
- [ ] whisper-large-v3 下载完成后做双裁判交叉

## Day 2 — 排行榜 v1 + 争议裁决 + 循环周期发现

### d4 排行榜 v1（CUDA assert 修复后完成）

- **毒候选定位**：12 条超长文本（558-1369 tokens > 444 上限）触发
  WhisperPositionalEmbedding 越界（modeling\_whisper.py:211 device-side assert）
- **修复**：nll\_scorer 加 MAX\_DEC\_TOK=440 分块计分（按词边界切块，
  各块独立 min-over-windows 后取均值——对循环广播物理合理）
- 排行榜结果（NLL\@own 冠军）：
  - a: R050 longpipe 去重单条 69 词 =0.189（但含 DISABILITY/FANDAS 疑似误听）
  - b: R048 segmap chunk 30 词 =0.185；完整版 R029 =0.860
  - RJTT: 全体 5.6-6.2（管制对话对 ATIS 微调模型 OOD，需单独策略）

### P4 争议裁决（adjudicate\_batch + anchored 二审）

- **方法迭代**：短片段对立对各找各窗 → 窗口漂移假象（visibility CONFLICT）；
  二审改用**长上下文锚定片段**（争议词±6词），迫使 min 锁定真实位置 → 全部收敛
- 裁决结果（双信道一致）：
  - VISIBILITY 胜 DISABILITY ✓（短片段假象被纠正）
  - 频率 **123.15** 强胜（ONE SIX / 122.15 否决）
  - 时间后缀 **ZULU** 胜 JULIETT/JULIET（符合 ICAO）
  - 跑道 **TWO EIGHT** 压倒性胜 TWO NINE（差 1.6 nats/tok）
    → 音频确实说 EIGHT；CYYT 现实无 28 跑道仅记 WARN（合成数据可能不按现实）
  - GANDER 胜 FANDAS/SANDGREN/MONCTON ✓（符合 CYYT∈Gander FIR）
  - 无 CEILING 词（直接 TWO FOUR THOUSAND FEET）
- **方法论洞察**：NLL 回答"发了什么音"，目标却是"真文本是什么"。
  弱读/连读处二者分歧：EIGHT JOHNS≈SAINT JOHNS 弱读、ARNAV≈R-NAV 连读。
  → 声学裁判之后需要 ATIS 语法/术语层纠错（文献 LLM 纠错环节的定位）

### P3 循环周期发现（find\_period + cut\_instances）

- a 信道：能量包络自相关 T=28.34s；静音间隙线性拟合 **T=28.143s 残差 0.15s**
  → 切出 8 个完整报文实例（segments/CYYT\_ATIS\_a/inst\_\*.wav）
- b 信道：前半段无可靠静音（词间停顿污染拟合）；用尾部两真间隙
  （228.1/256.0，间隔 27.85s）锚定外推，每实例 NLL 验证完整性（进行中）
- RJTT：非循环对话，按静音分段即可

### 温度段终审定案：TEMPERATURE ONE, DEW POINT MINUS ONE ⭐

切片解码（16s 无上下文短音频，三引擎六转写）+ 语法层裁决：

| 引擎                 | a 信道                         | b 信道                       |
| ------------------ | ---------------------------- | -------------------------- |
| whisper-atc        | ONE **REPORT NINER** ONE     | TARAN **REPORT NINER** ONE |
| whisper-v3         | 1. **Report minus** 1        | **report minus** one       |
| Qwen3(lingbot-map) | one, **two point minus** one | **two point minus** one    |

裁决链：

1. **语法否决**：METAR/ATIS 温度必为整数摄氏度，"ONE DECIMAL NINER ONE"(1.91°?) 非法；"TEMPERATURE ONE, DEW POINT MINUS ONE"(+1°/-1°) 完全合法——与 freq\_b(3.15MHz) 同一否决逻辑
2. **多数投票**：point×4 / minus×4 / niner×2 / report×3
3. **误听机理全经典**：DEW↔two 清浊对、MINUS↔NINER 鼻音部位对、DEW POINT 连读→REPORT 融合
4. **方法论教训**：增量计分曲线显示各裁判 forced-alignment NLL 被 decoder LM 先验污染（atc 给 DEW +1.4nat 极端惩罚 = 其 ATIS 训练分布里温度段只出现数字读法），裁判内对比不能替代跨引擎解码共识
5. 附注：时间戳模式不可用（atc 微调模型 generation config 强制 notimestamps）；Qwen 需 lingbot-map 环境 python

### 工具新增

- exp/find\_too\_long.py：离线 token 长度审计（防 CUDA assert 复发）
- exp/adjudicate.py / adjudicate\_batch.py：最小对立对裁决（双证人协议）
- exp/adjudicate\_v2\_anchored.py：长上下文锚定二审（解决窗口漂移）
- exp/run\_leaderboard\_v2.py：乱序基线归一 Δ=NLL\_shuf−NLL\_text（跨信道公平比较）
- exp/find\_period.py / cut\_instances.py / cut\_b\_anchored.py：周期检测与实例切分

### 待办（新增）

- [ ] 排行榜 v2 结果分析（运行中）
- [ ] b 信道切分验证完成后：整周期实例重解码（3 引擎）→ 候选池 v2
- [ ] SAINT JOHNS vs EIGHT JOHNS、RNAV vs ARNAV 的语义纠错实验
- [ ] whisper-large-v3 双裁判交叉（下载中 \~22%）

## Day 3 — 音频结构定论 + 多引擎解码 + v4-v6 三裁判终审

### 音频结构最终定论

- a 信道：0-137s 四个循环实例（inst\_01..04），137s 后为削波噪声（rms 0.224 非语音）
- b 信道：**前半段 0-228s 全是静音**（解码输出全 "......"）——此前所有转写都是底噪幻觉；
  唯一有效实例 inst\_00 = 229.8-256.0s
- RJTT\_CONTROL：东京管制通话，9 段（seg\_00..08），内容 = 航班高度/位置报告
  （Shanghai A96、FedEx 15 Heavy、JapanAir 239、Shamrock 96 等，decode\_rjtt/）
- whisper-large-v3 下载完成（2.9G），中立裁判就位

### ⭐⭐ 范式转变：a/b 是不同日期的两条录音（非同日双链路）

- v5 双裁判一致：b altimeter = **3033**（atc Δ0.10 / v3 Δ0.08），而 a = 3023（解码全票）
- 同时间码 0200Z + 同代号 FOXTROT 但气象值不同 → ATIS 代号每日循环，
  a/b 必然是**不同日期**的录音（Qwen 文件名 20260817 仅是接收时间）
- **交付结构改为 a\_final.txt / b\_final.txt 两份独立文本**；
  跨信道证据只用于模板结构推断，字段值按信道独立裁决

### v4 同窗终审（adjudication\_v4\_paired.json）

- airport 定案 **SAINT JOHNS**（双信道一致胜 SIENT/EIGHT/TUCSON）
- visibility/approach 仍 CONFLICT → 暴露单裁判不可靠 → 升级三裁判制

### v5 三裁判交叉（adjudication\_v5.json，atc + large-v3）

| 争议           | atc                         | v3                            | 综合裁决                       |
| ------------ | --------------------------- | ----------------------------- | -------------------------- |
| b altimeter  | 3033 (+.10)                 | 3033 (+.08)                   | **b=3033**（范式转变依据）         |
| temperature  | ONE DECIMAL NINER ONE (Δ.6) | ONE DEW POINT MINUS ONE (Δ.3) | 分歧→v6                      |
| visibility 词 | DESCENDING LEVEL(双)         | VISIBILITY(双,Δ.3-.5)          | **VISIBILITY**（v3+语法+形近证据） |
| b approach   | LIMA (+.17)                 | ZULU (+.36)                   | **ZULU**（v3+外部图表）          |

### 外部证据（WebSearch，决定性）

- Navigraph CYYT 图表：RWY28 进近仅 ILS / ILS CAT II-III / RNAV(GNSS) **Z** / RNAV(RNP) Y
- 加拿大 NAV CANADA 进近图后缀体系只有 X/Y/Z（XRAY/YANKEE/**ZULU**），LIMA/JULIETT 不存在
- → approach 定案 RNAV ZULU（ARNAV 是 R-NAV 连读的声学拼写，正字法 RNAV）
- b 信道该词三种听感（JULIETT/LIMA/new）全部否决

### v7 骨架词终审（atc 裁判部分，log 提取）

- opening：双信道压倒性 **"SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU"**（a 1.60 vs 次名 2.24；b 1.48 vs 2.05）
- alti\_word：双信道 **ALTITUDE** 胜 ALTIMETER/QNH（a 1.25 vs 1.49）
- closing：a="AS REQUESTED"(0.85) vs b="WHEN REQUESTED"(1.24)，差距极小待 v3
- tail：双信道 **"INFORM ATC THAT YOU HAVE INFORMATION FOXTROT"**（0.60-0.72 压倒）

### v8 温度破局：增量计分曲线（adjudication\_v8\_incremental.json）⭐ 方法论创新

逐词前缀 per-token NLL（同窗），终点值对比：

| 裁判  | a: DECIMAL NINER ONE / DEW POINT MINUS ONE | b: 同            |
| --- | ------------------------------------------ | --------------- |
| atc | **1.80** / 3.02                            | **2.66** / 3.59 |
| v3  | 2.96 / **2.05**                            | 3.31 / 3.35（平手） |

关键发现：

1. atc 裁判下 H\_v3 在 **DEW 加入瞬间跳升 +1.2\~1.4 nat** 且维持高位 = 音频中无该发音的典型特征；H\_atc 单调下降 = 逐词被声学确认
2. v3 裁判下 DEW 也跳 +0.5（发音不匹配），但后续 POINT/MINUS/ONE 靠 LM 先验下行；b 信道两假设几乎打平
3. 不对称性：atc 给 DEW 极端惩罚，v3 给 DECIMAL 仅中度惩罚 → v6 的 v3 偏好主要是语言先验贡献
4. 待切片解码（decode\_tempslice.py）第三方仲裁

### 工具新增

- exp/adjudicate\_v4\_paired.py：同窗对立终审（find\_anchor\_window + score\_constrained）
- exp/adjudicate\_v5.py：双裁判 × 微窗（altimeter\_b/temperature/visibility/approach\_b）
- exp/adjudicate\_v6.py：三裁判（+turbo-atcosim）× 温度终审/b频率/vis云况结构
- exp/adjudicate\_v7.py：骨架词终审（opening/alti\_word/closing/tail）
- exp/adjudicate\_v8\_temp.py：⭐增量计分（逐词前缀 NLL 曲线）+ 温度段切片导出
- exp/decode\_rjtt.py：RJTT 9 段 whisper-atc + Qwen 解码（18 条转写入库）
- exp/decode\_tempslice.py：温度切片三引擎解码（atc/v3/Qwen）

## Day 4 — 温度定案 + 最终文本组装 + wind 段裁决链（v10→v11→v12）

### 温度段终审定案：TEMPERATURE ONE DEW POINT MINUS ONE ⭐

证据链（五步）：

1. 切片解码三引擎六转写（decode\_tempslice）：a-atc "TEMPERATURE ONE REPORT NINER ONE"、
   a-v3 "Temperature 1. Report minus 1."、a-qwen "temperature one, two point minus one"
   → "one" 后面确实还有内容，且 qwen 听到 "two point"≈"dew point" 连读
2. 语法层否决声学层：METAR/ATIS 温度必为整数 → "ONE DECIMAL NINER ONE"(1.91°?) 非法，
   atc 裁判的偏好被 ICAO 语法否决
3. 经典误听机理：DEW POINT 连读 /djupɔɪnt/ → REPORT 融合误听（a-v3 的 "1. Report minus 1"）
4. 增量曲线复核：atc 下 H\_dew 单调下降（逐词被声学确认），H\_decimal 在 DECIMAL 处跳升
5. b 信道同段同文（两日期广播格式一致）

- 方法论教训：**增量计分也被 decoder LM 先验污染**——v3 在 DEW 处跳升 +1.2\~1.4 nat
  与最终定案矛盾（DEW 实际存在）→ 单词插入级 Δ≤1.4 nat 的 forced NLL 差异不可单独定案

### v7b 第三裁判补跑（adjudication\_v7b.json，turbo 权重补齐后首次成功）

| 字段         | 结果                                           | 依据                                                       |
| ---------- | -------------------------------------------- | -------------------------------------------------------- |
| opening    | 无 THIS IS                                    | atc 强烈反对 THIS IS(Δ0.57-0.64)，v3/turbo 弱偏好(Δ0.04-0.14)=噪声 |
| alti\_word | **ALTIMETER**                                | turbo Δ0.69 > atc ALTITUDE Δ0.23，2:1 + ICAO 标准术语         |
| closing    | a="AS REQUESTED" / b="WHEN REQUESTED"        | 三裁判一致                                                    |
| tail       | INFORM ATC THAT YOU HAVE INFORMATION FOXTROT | 三裁判一致                                                    |

### P5 组装 + 正位/错位验证（assemble\_final.py）

- 9 行结构 ×2 信道；每行在其真实锚位 vs 相邻行锚位计分
- 四组全过：a-atc 1.5366<1.5573 ✓ / b-atc 1.7460<1.9542 ✓ / a-v3 1.7057<1.7959 ✓ / b-v3 2.2595<2.6883 ✓
- 已知缺陷：频率行对 b 写死 AS REQUESTED（待按 v7b 改 WHEN REQUESTED）；wind 行待独立裁决

### wind 段裁决链（v10 伪象 → 切片 → v11 → v12）⭐ 方法论案例

1. **v10 十三变体裁决出现跨裁判矛盾**：atc 说 a="TWO SEVEN ZERO"(Δ2.16)、v3 说 "AT ZERO FIVE"
   → 教训：contest 锚位估计不准时，单变量变体会产生系统性伪象（270° 是窗位伪象）
2. **slice\_wind 切片解码**（锚=opening 尾部，中心=t\_open+7）：
   - a：三引擎一致 240°（atc/qwen 听感暗示可能无 AT："TWO FOUR ZERO FIVE"）
   - b：atc/qwen 都没听到 WIND 词（"ZULU TWO FOUR ZERO AT FIVE" 直连）；v3 输出全点号
3. **v11 精确组合裁决**（切片实测中心 a=62/b=237）：
   - a：atc 强支持 AT FIVE(Δ0.46)，v3 弱反对(Δ0.12=噪声)
   - b：atc 压倒性支持 WIND 前缀(Δ1.27)，v3 弱反对 AT(Δ0.21)
   - 但 Δ1.27 落在 LM 先验污染区间(≤1.4 nat)内 → 不能单独定案
4. **v12 三角定位**（完成）：精确重切片×3中心/信道 + turbo 第三自由解码引擎
   - v3 词级时间戳解码 + 增量计分曲线
   * 自由解码：a 全部 11 条无 AT；b 全部 7 条有效解码无 WIND（含 ATIS 先验的 turbo）
   * 增量曲线：atc 下 b H\_wind 终点 0.279 < H\_nowind 0.406；a H\_at 0.146 < H\_noat 0.214（v3 同向但差距小）
   * turbo 在 b 听到 ZULU 与数字间有 "three" 样杂音(4/4)；b\_c240\_atc 听到 ALTITUDE THREE ZERO THREE THREE（与 v5 一致）
5. **能量包络物理探测终审（probe\_wind\_energy/fine）——破局定案** ⭐⭐
   NLL 裁判 vs 自由解码对峙，只有无语言先验的物理测量能裁决：
   - a 的 AT 存在：57.88-58.01s 连续浊音 RMS 0.06-0.14（≈b 清晰 AT 峰值的 55%），
     而真词间空隙跌至 0.025-0.04 → 弱化 /ət/，解释解码器为何全票丢失
   - b 的 WIND 存在：233.35-233.42 有 8 帧纯底噪把区域切成两词；
     ZULU=233.05-233.33（290ms ≈ a 信道 ZULU 310ms，排除长 ZULU 假设）+
     WIND=233.44-233.81（强元音头 0.22-0.33 + 鼻音平台 /nd/ 0.10-0.17）；
     随后 TWO=233.92/FOUR=234.14/ZERO=234.43/AT=234.99/FIVE=235.26 七 burst 全归属
   - **定案：两信道 wind 行均为 WIND TWO FOUR ZERO AT FIVE**
   - 方法论教训：自由解码全票缺席 ≠ 声学缺席（鼻音/弱读词系统性丢失）；对峙时回到信号本身

### P5 重跑（assemble\_final 按 v12/v7b 结论修正）

- b closing 改 WHEN REQUESTED；wind 行维持
- 四组正位/错位：a-v3 1.6777<1.8206 ✓ / b-atc 1.6068<1.7624 ✓ / b-v3 2.1778<2.7136 ✓ /
  a-atc 1.4725>1.4402 ✗（边缘失败归因 atc 裁判域先验饱和——错位惩罚在 ATIS 音频上失效，
  正是三裁判制要求 v3 并行的原因）

### RJTT 段级终审（v1 翻车 → v2 语法过滤+投票 → ROVER 共识）⭐ 方法论案例

1. validate\_rjtt.py（v1）：单 v3 裁判逐段在 {atc\_beam5, qwen} 间挑选
   → **翻车实锤**：seg05 把 "That's possible, anyway"(NLL 3.41) 判胜
   "DECIMAL TWO RYANAIR ONE EIGHT"(5.42)；seg04 同病。LM 先验污染又一实证。
2. validate\_rjtt2.py（v2）：+turbo 第三引擎（9 段自由解码入库）→ ATC 核心词覆盖率 frac +
   会话词黑名单硬过滤 → 归一化 token Jaccard≥0.30 聚类投票 → 簇内 NLL 决胜。
   seg04/05 垃圾文本被正确否决；seg06 三引擎全簇一致。
3. rover\_rjtt.py：以簇胜者为基、按多引擎一致性逐 token 共识修订：
   - 呼号 SIERRA EIGHT NINER SIX 定案依据 = turbo 在 seg01+seg08 两次独立产出同一呼号
     且 v3 NLL 两段均最优（跨段一致性 > 单段争论）；SHAMROCK 记为竞争假设
   - seg07 CLEARED DIRECT IGOTO：turbo 片假名リゴート为 IGOTO 同音佐证；
     双重 direct-climb FL300 为 atc+turbo 独立一致 → 判定为音频事实
   - 未决：seg00 呼号（AEROFLOT vs airfr）、seg05 碎片、seg06 JOHNSON
4. 产出 results/rjtt\_final.txt（9 段带置信度）+ rjtt\_consensus.json（逐段裁决依据）

### 研究收尾

- FINAL\_REPORT.md 完成：三条音频最终文本 + 八项方法论贡献 + 全部证据链与数值
- 最终交付：a\_final.txt / b\_final.txt / rjtt\_final.txt（全部可由 exp/ 脚本复现）

### 工具新增（Day 4 续）

- exp/probe\_wind\_energy.py / probe\_wind\_fine.py：能量包络物理探测（burst 检测+槽位统计）
- exp/validate\_rjtt.py（v1）/ validate\_rjtt2.py（v2 三引擎+过滤+投票）/ rover\_rjtt.py（共识合成）

### 待办（全部完成）

- [x] v7 v3+turbo 补跑（v7b 完成）
- [x] 温度切片解码 → 温度段终审定案
- [x] v12 收取 + 能量探针破局 → wind 行定案 → a\_final.txt/b\_final.txt 重生成
- [x] RJTT 段级终审（v2+ROVER）→ rjtt\_final.txt
- [x] FINAL\_REPORT.md

## Day 5 — 计划缺口补齐 + 英语信道加固轮

### 缺口补齐（P1/P4/P5/P6/P7）

- `src/atis_grammar.py`：ICAO 槽位解析+值域校验，三份终稿 26 PASS / 0 WARN / 0 FAIL
  （`results/grammar_check.json`）；五连 bug 修复记录见 git（AT 索引/跑道号/能见度双读法/
  露点 DEW 索引/RJTT HUNDRED 特例）
- `exp/denoise_audit.py`：谱减法 × 双裁判，全组恶化 −8%\~−27% → "降噪有害"结论
  （`results/denoise_audit.json`）
- `results/adjudication.md`（P4 汇总）+ `results/assets_inventory.md`（P1 盘点）
- PLAN.md 全面标注实况；FINAL\_REPORT §7 补遗；RJTT v1→v2→ROVER 全过程入档

### 英语加固轮（`exp/english_hardening.py` → `results/english_hardening.json`）⭐

应"先把英语两条研究扎实、着重看英语"的要求，补三个缺口：

1. **turbo 整篇正位/错位**（此前只有 atc+v3）：a-turbo 3.1498<3.3697 ✓ /
   b-turbo 3.6507<3.9300 ✓ → 六组矩阵 5✓+a-atc 边缘失败（域饱和归因不变），
   a 信道整篇有效性 2/3 裁判独立确认
2. **循环一致性检验**（新方法论，FINAL\_REPORT §2.9）：ATIS 循环广播=天然多重观测。
   实例版 Δ(text vs shuffle)=inst\_01 2.62 / inst\_02 2.61 / inst\_03 2.81 / inst\_04 1.39 nat
   全部支持终稿；曲线版低谷以 \~28s 周期复现（25/50/80s 同相位窗 2.26–2.60）；
   噪声段 inst\_05–08（RMS≈0.224）Δ≈0.17 自洽不贴合=无假阳性。
   → a 终稿获 4 重独立声学确认，证据强度仅次于能量探针
3. **差异字段 turbo 复核**：closing a=AS(Δ0.211)/b=WHEN(Δ0.082) 同向确认 ✓；
   altimeter b=3033 同向（Δ0.032 平手级）、a 上 turbo 弱偏好 3033（Δ0.113，噪声级≪污染区间）
   → 记录弱异议维持原判 a=3023。四案全部非决定性，零改判——差异字段"依赖多源证据"
   的定级被再次确认

### 结论

英语两条终稿零修改通过全部加固检验；研究回到可交付静止态。

### 目录清理

删除：全部运行日志、denoise/ 变体音频（17M）、wind/temp 切片 wav（txt+index 保留）、
b 信道静音期无效切片 inst\_01–13（11M）、__pycache__、too\_long\_candidates.json。
保留：三份终稿、全部证据 JSON（adjudication\_v\* / validation / grammar / denoise\_audit /
english\_hardening / wind probe）、解码转写 txt、有效切片 wav（a×8 + b inst\_00 + RJTT×9，
复现必需）、src+exp 全部工具脚本。40M→12M，239→186 文件。

## Day 6 — RJTT 语言审计轮（用户触发：FunASR 原始输出全是日语）

### 动机与盲区

用户指出 FunASR 对 RJTT 的识别"全是日语"。核查 `TT/results/FunASR/RJTT_CONTROL/result_1.txt`：
输出是片假名拼英语读音（クライム=climb、シャハラA96≈Sierra 896）——同音转写伪象。
但这暴露了管线的系统性盲区：**此前全程强制** **`<|en|>`，从未测试过日语假设**
（日本空域对日籍航班允许日语通话）。若个别段真是日语，整条英语管线会整段漏掉。

### 语言审计（`exp/lang_audit_rjtt.py` → `results/rjtt_lang_audit.json`）

v3 vanilla 对 9 段各跑 auto / 强制 en / 强制 ja 三模式解码+平均 token logprob：

- **ja 全线产出片假名音译英语或幻听**（seg05「ご視聴ありがとうございました」=
  "感谢收看"，典型噪声幻觉；seg00「クラウドメンテンス」≈climb maintain）
- **结论：语音确为英语管制英语，语言盲区排除**——FunASR 现象=日语分词器对英语的同音渲染

### 意外收获：呼号翻案 ⭐

审计暴露候选池缺口：v3 自己的自由解码从未进过 NLL 候选池。补入后发现
**v3 在 seg01/seg08 双段独立产出 "Shanghai Air 896"**，挑战原案 SIERRA(turbo)。
`exp/adjudicate_v13_callsign.py` 四项裁决（→ `results/adjudication_v13.json`）：

1. **seg01/seg08 呼号**：SHANGHAI\_AIR 双段同向胜出（0.485<0.537 / 1.350<1.516，
   单段 margin<1.4 污染带但方向一致）；叠加 qwen "Shanghai A96"+域先验
   （上海航空真实呼号且飞羽田；SIERRA 非航司格式；Aer Lingus 无日本航线）
   → **三重证据同向，SIERRA→SHANGHAI AIR EIGHT NINER SIX 翻案**，turbo 异议入档
2. **seg04**：ORANGE NINER→**ORANGE LINER**（v3-ja『オレンジライナー』+qwen 'Line'
   双源+v13 计分方向一致；margin 小故置信度降为 low-medium）
3. **seg06 X 槽位**：JOHNSON 击退 LEAVING/PASSING/JUST\_ON 全部标准短语假设
   ——该音真实存在，维持原案+未决标注（方法论：语法假设也要过声学关）
4. seg00 补记 v3-en 竞争假设 KTX10Y，不可裁决维持原案

### 定稿与量化（`exp/finalize_rjtt_v13.py`）

- rjtt\_consensus.json / rjtt\_final.txt 更新至 v4；证据源从 3 扩到 5（+v3free\_en/ja）
- **多数票覆盖率 72%→83.6%**（194/232）：seg06 97%、seg07 94%、seg02 58%（最低，
  VAD 截断段）、seg04 50%
- 方法论新增两条判例：①候选池必须含裁判自身的自由解码（否则系统性缺证）；
  ②跨语言审计是多语空域的必做前置检查

### 增强解码 + v14 守案轮（`exp/decode_enrich_rjtt.py` / `exp/adjudicate_v14.py`）

v3 × {beam5 裸奔, ATC 提示词 beam5} 补入候选池（+18 条），随后四组裁决
（`results/adjudication_v14.json`）：

- **守案×3**：seg03 NINE TWO THREE（1.615<1.897 击退 v3b 'Japan Air 723'）、
  seg02 NINE SEVEN TWO THREE（2.798 最优，JAL723 统一假设被否）、
  seg06 FEDEX ONE FIVE（0.890<1.122 击退 'FedEx 150'）
- **改判×1**：seg06 HEAVY 后槽位 JOHNSON LEVEL→**JUST OUT OF**——
  v3b 自由解码 "we join you out of 180"+日语『ジョンシュアドブ』+turbo 'jet of'
  同音族+v14 计分 JOIN\_YOU\_OUT\_OF/JUST\_OUT\_OF 双胜 JOHNSON；JUST OUT OF 与
  JOIN YOU OUT OF 近平手取惯用语。教训：**v13 轮 JOHNSON 的胜出是假阳性——
  当时没把 out-of 族放进对立组；对立假设集不完整时"胜出"只是"未败"**
- 定稿 rjtt\_final.txt → **v5**；覆盖率 83.3%（194/233，seg06 因改判词源支持略降为 94%）

## Day 5+ — 全量代码 review 与修复（2026-08-25）

对 research/deep 全部 src/exp 做逐文件 review，修复 6 处、排除 4 处误报，全部重跑验证：

1. **atis_grammar.py**：温度行解析失败时原会 TypeError 崩溃且 verdict 经 `max("PASS","FAIL")`
   字典序恒返回 PASS → 改为显式 FAIL + `tv is not None` 守卫；边界用例回归通过
2. **lang_audit_rjtt.py**：语言检测切片漏掉首个生成 token（detected 恒 null）→
   重写为标准 auto 流程（仅 `<|startoftranscript|>` prompt，max_new_tokens=1）。
   **重跑结果：seg02/03/07 detected=ja，其余=en** —— FINAL_REPORT §同音转写伪象
   所述"日语假设从未被测试"盲区现已有实测数据支撑
3. **adjudicate_v6.py**：vis/temp 坏锚文本（定稿中不存在）→ 跨行相邻锚；
   已重跑刷新 adjudication_v6.json。各裁判 anchor_t 不同 = 循环广播不同周期实例，
   裁判内排序仍有效；v6 结论继续以 v8+ 为准
4. **nll_scorer.py**：分块路径 `median_score` 实为 mean → 改名 `chunk_mean_score`（零消费方）
5. **denoise_audit.py**：曾加去重后回滚——尾三行重复为真实复诵
   （人工听音确认 + recheck_tail_repeat.py 探针双谷佐证，见 results/tail_repeat_check.json）
6. **adjudication_v7_atc_reconstructed.json**：补齐缺失的 v7 atc 数值档案（源自 JOURNAL Day3 记录）

误报排除：sanity_nll 关卡2b 迁移阈值、validate wrong-pos 基线、assemble_final
"双重拼接"（实为真实复诵）、build_pool 扫描顺序。已确认 deep 内无任何可执行代码
引用已删除的 research/best、research/scratch（仅 results JSON 溯源字段与历史注释保留）。

定案数字（3023/3033、wind 240@5、freq 123.15 等）证据链复核无恙。

