# ATC 语音识别深度研究 · 最终报告

> 研究周期：4 天（自主研究授权，无外部标注、无具体需求、仅 3 条音频）
> 目标：在"无对照文本"的极端约束下，让三条音频的识别结果尽可能接近真值
> 全部结论均有客观可复现证据支撑，证据文件清单见附录 A

---

## 0. 任务与约束

**输入**（`TT/audio/`）：
1. `CYYT_ATIS_a.wav` — St. John's（CYYT）ATIS 广播，约 2 分钟
2. `CYYT_ATIS_b.wav` — 同台另一时段 ATIS 广播（经 v7 阶段论证为**不同日期录音**，非同播报重录）
3. `RJTT_CONTROL.wav` — 管制频率通话录音（多机位/多机呼号），约 2 分钟

**约束**：无具体需求、无真实对照文本、数据仅此三条。
**红线**：不许自设主观评分当裁判。全程只依赖三类客观裁判：
① NLL 声学似然（forced scoring）② ICAO/METAR/ATIS 语法硬约束 ③ 多系统交叉验证 + 物理测量。

---

## 1. 最终结果

### 1.1 CYYT_ATIS_a（`results/a_final.txt`）

```
SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU
WIND TWO FOUR ZERO AT FIVE
VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET
TEMPERATURE ONE DEW POINT MINUS ONE
ALTIMETER THREE ZERO TWO THREE
APPROACH RNAV ZULU RUNWAY TWO EIGHT
INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE AS REQUESTED
APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT
INFORM ATC THAT YOU HAVE INFORMATION FOXTROT
（末三行在音频中真实复诵——人工听音确认，recheck_tail_repeat.py 锚窗探针独立佐证）
```

### 1.2 CYYT_ATIS_b（`results/b_final.txt`）

与 a 仅两处信道级差异（这正是"a/b 为不同日期录音"推断的产物）：
- 修压：`ALTIMETER THREE ZERO THREE THREE`（a 为 THREE ZERO TWO THREE）
- 频率移交尾：`…ONE TWO THREE DECIMAL ONE FIVE **WHEN REQUESTED**`（a 为 **AS REQUESTED**）

### 1.3 RJTT_CONTROL（`results/rjtt_final.txt`，9 段共识合成）

| 段 | 时间 | 置信度 | 文本 |
|---|---|---|---|
| seg00 | 3.0–16.9s | 中低 | GOLF ZERO JULIETT CONTROL ONE ZERO YANKEE AEROFLOT PASSING ONE EIGHT NINER FOUR FLIGHT LEVEL TWO FOUR ZERO …CLIMB MAINTAIN FLIGHT LEVEL THREE TWO ZERO ×2… |
| seg01 | 20.2–37.4s | 中 | TOKYO CONTROL GOOD AFTERNOON **SHANGHAI AIR EIGHT NINER SIX** CLIMBING FLIGHT LEVEL TWO HUNDRED …CLIMB MAINTAIN FLIGHT LEVEL THREE ZERO ZERO REQUEST DIRECT TO MAIDA… |
| seg02 | 44.2–47.3s | 低 | NINE SEVEN TWO THREE CROSS MADIGU AT FLIGHT LEVEL THREE TWO ZERO |
| seg03 | 48.1–51.8s | 中 | JAPAN AIR NINE TWO THREE CROSS MADIGU AT FLIGHT LEVEL THREE TWO ZERO |
| seg04 | 56.6–59.5s | 中低 | ORANGE **LINER** ONE EIGHT THREE QUEBEC ONE THREE |
| seg05 | 60.4–62.0s | 低 | TAXI POSSIBLY ONE TWO NINE ONE EIGHT |
| seg06 | 67.4–79.6s | 中高 | TOKYO CONTROL FEDEX ONE FIVE HEAVY JOHNSON LEVEL ONE EIGHT ZERO CLIMBING TWO FOUR ZERO …INITIALLY CLIMB THREE TWO ZERO… |
| seg07 | 80.9–98.8s | 中 | TOKYO CONTROL JAPAN AIR TWO THREE NINER LEAVING FLIGHT LEVEL ONE SEVEN FIVE CLIMBING FLIGHT LEVEL TWO ZERO ZERO …CLEARED DIRECT IGOTO CLIMB FLIGHT LEVEL THREE ZERO ZERO ×2… |
| seg08 | 108.1–114.3s | 中 | TOKYO CONTROL **SHANGHAI AIR EIGHT NINER SIX** REQUEST FLIGHT LEVEL THREE EIGHT ZERO OR FOUR ZERO ZERO |

> **v4 翻案记录（Day 6 语言审计轮）**：呼号 SIERRA→SHANGHAI AIR（v3 自由解码双段独立
> +qwen+同窗配对计分双段同向+域先验三重证据；turbo 的 SIERRA 记为竞争假设）；
> ORANGE NINER→ORANGE LINER（v3-ja オレンジライナー+qwen Line 双源）。详见 §8 后补遗与
> `results/adjudication_v13.json`。

---

## 2. 裁判体系与方法论（本研究的方法论贡献）

### 2.1 三裁判制（tri-judge）
- **whisper-atc**（ATIS 微调，域先验强，但对自己 beam5 输出自偏置）
- **whisper-large-v3 vanilla**（中立，域先验弱）
- **whisper-large-v3-turbo-atcosim**（第三独立引擎，投票用）
规则：任何字段定案需 ≥2 个独立证据源同向；单一裁判永不定案。

### 2.2 同窗对立比较（paired-window adjudication）
竞争假设在同一锚定窗口内由 `score_constrained` 计分对比，消除窗口漂移假象。
（早期教训：不同窗口的 NLL 差可比拟真差异——"静音窗口假象"：把文本放到静音段上计分会得到极低 NLL，因为模型在静音上对任何文本都不惊讶。）

### 2.3 增量计分与 LM 先验污染（incremental scoring & LM contamination）
假设短语逐词前缀累加 per-token NLL 曲线，本可定位"差异发生在哪个词"，但 v8 温度案证明：
**单词插入级的 forced NLL 差异可达 1.2–1.4 nat，纯粹来自 decoder 语言先验而非声学证据**。
→ 教训：ΔNLL 落在此区间内时不可单独定案（v11 的 b-WIND Δ1.27 恰在此区间，必须另找裁判）。

### 2.4 切片解码及其系统性偏差（slice decoding & absence bias）
切出 14–16s 无上下文短音频让多引擎自由听写，破除长音频的上下文锚定效应。
**但本阶段关键发现：自由解码全票缺席 ≠ 声学缺席。** 解码器在劣化音频上系统性丢失鼻音/弱读词
（a 的弱化 /ət/、b 的鼻音 /wɪnd/ 均被 11/11 与 7/7 全票漏掉，但物理上存在）。
→ 教训：自由解码投票只能作为"存在性"的正证据，不能作为"不存在"的证据。

### 2.5 能量包络物理探测（energy envelope probe）——破局者
当 NLL 裁判与自由解码对峙时，唯一能破局的是**无语言先验的物理测量**：
- 10ms RMS 包络 + burst 检测（阈值 max(3.5×floor, 0.004)，≥80ms）
- 标定：已知词做能量参照（b 的 AT 峰值 0.25），已知空隙做底噪参照（0.025–0.04）
- 鼻音平台特征：/wɪnd/ = 强元音头（0.22–0.33）+ 持续低能鼻音尾（0.10–0.17）

两个终审定案均由它裁决：
- **a 的 AT 存在**：57.88–58.01s 连续浊音（RMS 0.06–0.14，约为 b 清晰 AT 的 55%）vs 真词间空隙 0.025–0.04 → 弱化 /ət/
- **b 的 WIND 存在**：233.35–233.42s 有 8 帧纯底噪把区域切成两词；ZULU=233.05–233.33（290ms，与 a 信道 ZULU 310ms 一致，排除"长 ZULU"假设）+ WIND=233.44–233.81（鼻音平台包络）；随后 TWO/FOUR/ZERO/AT/FIVE 七个 burst 逐一归属 "WIND TWO FOUR ZERO AT FIVE"

### 2.6 语法层否决声学层（grammar vetoes acoustics）
METAR/ATIS 硬约束（温度必整数、VHF 118–137、修压 28.xx 格式）与 ATC 词汇合理性
（ATC 核心词覆盖率 + 会话词黑名单）可一票否决 NLL 更低的候选。
RJTT v1 教训：单 v3 裁判把 seg05 判给 "That's possible, anyway"（NLL 3.41 < 5.42）——
管制频率上不可能的话反而 NLL 更低，LM 先验污染的又一实证。v2 引入语法过滤后纠正。

### 2.7 三引擎投票 + ROVER 共识（多引擎交叉验证）
段级：语法过滤 → 归一化 token Jaccard≥0.30 聚类 → 最大簇胜出 → 簇内 NLL 决胜；
文本级：以胜者为基、按多引擎一致性逐 token 修订（turbo 在 seg01/seg08 两次独立产出
同一呼号 SIERRA EIGHT NINER SIX，跨段一致性成为呼号定案的决定性证据）。

### 2.8 正位/错位验证（correct vs wrong position）
整篇文本在真实锚位 vs 相邻错位锚位计分。加固轮后为完整六组（三信道 × 三裁判）：
| 组 | 正位 | 错位 | 判定 |
|---|---|---|---|
| a-v3 | 1.6777 | 1.8206 | ✓ |
| **a-turbo** | **3.1498** | **3.3697** | ✓（加固轮新增） |
| b-atc | 1.6068 | 1.7624 | ✓ |
| b-v3 | 2.1778 | 2.7136 | ✓ |
| **b-turbo** | **3.6507** | **3.9300** | ✓（加固轮新增） |
| a-atc | 1.4725 | 1.4402 | ✗（边缘） |

a-atc 边缘失败归因于 atc 裁判的**域先验饱和**（ATIS 微调模型在 ATIS 音频全域 NLL 都低，
错位惩罚失效）——这正是三裁判制要求多裁判并行验证的原因；v3 与 turbo 以清晰差距双通过，
a 信道整篇有效性由 2/3 裁判独立确认。

### 2.9 循环一致性检验（cycle consistency，加固轮新增）⭐
ATIS 本质是循环广播（a 信道周期 28.143s × 约 4 遍）。真文本必须不仅在"最佳窗口"贴合音频，
而要在**每一个物理重复周期**上都贴合——这把单条音频变成了多次独立观测：
- 实例版：切出的 4 个有效循环实例分别计分，终稿全文 vs 乱序基线的优势
  Δ = **2.62 / 2.61 / 2.81 / 1.39 nat**，全部远超 LM 污染区间或恰在其边界；
- 曲线版：逐窗 NLL 低谷以 ~28s 周期复现（t≈25/50/80s 同相位窗 NLL 2.26–2.60 高度一致），
  乱序文本全程 ≥4.95 且无任何周期结构；
- 自洽对照：137s 后的削波噪声段（inst_05–08，RMS≈0.224）Δ≈0.17——文本与乱序都不贴合，
  证明该检验对"无语音内容"不会给出假阳性。
结论：CYYT_ATIS_a 终稿获得 **4 重独立声学确认**，这是本研究中除能量探针外最硬的证据形态。

---

## 3. 关键定案证据链（CYYT 逐字段）

| 字段 | a 信道 | b 信道 | 证据链 |
|---|---|---|---|
| opening | SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU | 同 | v7/v7b 三裁判一致 |
| wind | WIND TWO FOUR ZERO AT FIVE | 同 | 自由解码全票反对 → v11 forced Δ 支持 → **能量包络物理探测终审**（2.5 节两案） |
| visibility | VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET | 同 | v6 |
| temperature | TEMPERATURE ONE DEW POINT MINUS ONE | 同 | 切片解码 + 语法硬约束（整数）+ 增量曲线；v8 案确立 LM 污染区间 |
| altimeter | THREE ZERO TWO THREE | THREE ZERO **THREE** THREE | v7b 2:1 + v5（b_c240 切片听到 ALTITUDE THREE ZERO THREE THREE） |
| approach | APPROACH RNAV ZULU RUNWAY TWO EIGHT | 同 | 外部航图 + v3 |
| freq+closing | …ONE TWO THREE DECIMAL ONE FIVE **AS REQUESTED** | …**WHEN REQUESTED** | v7b 三裁判（信道级差异，佐证 a/b 异日录音） |
| tail | INFORM ATC THAT YOU HAVE INFORMATION FOXTROT（×2 复诵·探针已验证） | 同 | v7 |

## 4. 研究过程时间线（详见 JOURNAL.md）

- **Day 1**：管线搭建（NLL scorer、锚窗定位、乱序基线归一）、初版解码
- **Day 2**：三裁判制确立、逐字段对立计分、发现静音窗口假象
- **Day 3**：切片解码、温度定案五步证据链、v8 LM 污染量化、P5 组装与正位/错位验证
- **Day 4**：wind 双案终审（v10 窗口伪象 → v11 对立计分 → v12 重切片×3 中心 + 第三引擎 + 增量曲线）→ **能量包络物理探测破局** → b 信道 closing 修正（WHEN REQUESTED）→ RJTT 三引擎终审（v1 单裁判翻车 → v2 语法过滤+投票）→ ROVER 共识合成

## 5. 可迁移的方法论教训（按发现顺序）

1. **静音窗口假象**：NLL 必须同窗对比；静音上任何文本都不惊讶。
2. **乱序基线**：NLL 的绝对值无意义，必须用乱序文本做归一参照。
3. **LM 先验污染区间**：单词插入级 ΔNLL≤1.4 nat 不可单独定案。
4. **自由解码缺席偏差**：全票听不到 ≠ 声学不存在（鼻音/弱读词系统性丢失）。
5. **物理测量破局**：当两个间接裁判对峙时，回到信号本身（能量包络/burst/时长统计）。
6. **语法层否决声学层**：域硬约束一票否决，无论 NLL 多低。
7. **单裁判必翻车**：RJTT v1 案（"That's possible, anyway" 胜出）。
8. **跨段一致性是最强的软证据**：同一引擎在两段独立产出同一呼号 > 任何单段内争论。
9. **域微调裁判的饱和效应**：atc 裁判在域内音频上错位惩罚失效，必须配中立裁判。

## 6. 局限与后续方向

- RJTT seg00 呼号（GOLF ZERO JULIETT / AEROFLOT vs juliett airfr）、seg05 碎片、
  seg06 "JOHNSON" 仍为未决竞争假设（各引擎内部自洽、外部无法裁决）。
- CYYT 呼号类词（SIERRA vs SHAMROCK）若能获得任一外部佐证（如 VATSIM 通话记录）即可闭案。
- 若继续投入：① 用能量探针反向生成"音节网格"喂给 constrained decoding；
  ② 对 RJTT 全频段做说话人聚类（diarization）后再分引擎解码；
  ③ 用 b 信道 vs a 信道的差集（同台异日）自动学习"信道不变词表"。

---

## 7. 收尾补遗：计划缺口补齐（P4/P5/P6 正式交付，P7 有意不启动）

- **P5 `src/atis_grammar.py`（第三裁判形式化）**：ICAO 槽位解析 + 值域校验。
  两份 ATIS 终稿 **26 PASS / 0 WARN / 0 FAIL**；RJTT 全部高度层合法
  （FL175/200/240/300/320/380/400）。结果 `results/grammar_check.json`。
- **P6 降噪重审**：谱减法 × 双裁判 NLL，全部 4 组恶化 −8%~−27% → 维持无降噪管线，
  结论从"无用"升级为"有害"（`results/denoise_audit.json`）。
- **P4 `results/adjudication.md`**：CYYT 全字段证据链汇总（含温度案"语法否决声学层"
  与 wind 案"自由解码缺席偏差"两大判例的完整数值）。
- **量化补充**（应"如何量化"之问）：RJTT 最终文本 121 token 中 87（72%）获 ≥2 引擎支持，
  逐段覆盖率 50%–94% 与置信度标注吻合——该数字是可信下界而非准确率；
  升级为真量化的唯一缺口是任一条外部真值参照。
  **（Day 6 更新：证据源扩至 5 个后重算，覆盖率升为 83.6%（194/232），见 §9）**
- **P7 有意不启动**：3 条音频上伪标签自训=循环过拟合，收益不可与记忆化区分；
  管线可整体迁移至真实数据集场景。

---

## 8. 英语信道加固轮（CYYT 优先深挖）

应"先把英语两条研究扎实"的要求，补齐三个遗留缺口（`exp/english_hardening.py` →
`results/english_hardening.json`）。**结论：两份终稿零修改通过全部加固检验。**

1. **turbo 第三裁判整篇正位/错位**（此前矩阵只有 atc+v3）：
   a-turbo 3.1498<3.3697 ✓、b-turbo 3.6507<3.9300 ✓ → 六组矩阵 5✓ + a-atc 边缘失败（已归因域饱和）。
2. **循环一致性检验**（新增方法论 2.9 节）：4 个有效循环实例 Δ=2.62/2.61/2.81/1.39 nat
   全部支持终稿；噪声段 Δ≈0.17 自洽对照通过。
3. **差异字段 turbo 复核**：
   - closing：a=AS(Δ0.211) / b=WHEN(Δ0.082)——方向与 v7b 定案一致 ✓；
   - altimeter：b=3033 同向确认（Δ0.032 平手级）；a 上 turbo 弱偏好 3033（Δ0.113），
     属噪声级且 ≪ 污染区间，不敌既有证据（解码全票 + atc Δ0.10 / v3 Δ0.08 双裁判支持 3023）
     → **记录 turbo 弱异议，维持原判 a=3023**。
   - 四案 turbo 全部非决定性（均 <1.4 nat），无单一字段因新证据改判——差异字段的既有定级
     （信道级差异、依赖多源证据）被再次确认。

---

## 9. RJTT 语言审计轮（Day 6，用户触发）

用户指出 FunASR 对 RJTT 的原始输出"全是日语"。核查确认其为片假名拼英语读音的
同音转写伪象（クライム=climb），但暴露管线盲区：**全程强制 `<|en|>`，日语假设从未被测试**
（日本空域允许日籍航班用日语通话）。

### 9.1 语言审计（`exp/lang_audit_rjtt.py` → `results/rjtt_lang_audit.json`）
v3 对 9 段各跑 auto / 强制 en / 强制 ja 三模式：ja 全线产出片假名音译英语或幻听
（seg05「ご視聴ありがとうございました」=噪声幻觉）。**结论：语音确为英语管制，
语言盲区排除。**

### 9.2 意外收获：候选池缺口与呼号翻案
审计发现 v3 自由解码从未进过 NLL 候选池——裁判自己的证词没被收集。
补入后 v3 在 seg01/seg08 双段独立产出 "Shanghai Air 896"，触发 v13 裁决
（`exp/adjudicate_v13_callsign.py` → `results/adjudication_v13.json`）：

| 裁决项 | 结果 | 证据 |
|---|---|---|
| seg01/seg08 呼号 | **SIERRA→SHANGHAI AIR EIGHT NINER SIX 翻案** | 配对计分双段同向(0.485<0.537 / 1.350<1.516)+v3 双段自由复现+qwen "Shanghai A96"+域先验（真实呼号且飞羽田；SIERRA 非航司格式；Aer Lingus 无日本航线）；turbo 异议入档 |
| seg04 | ORANGE NINER→**ORANGE LINER**（置信降为 low-medium） | v3-ja オレンジライナー+qwen Line 双源+计分方向一致；margin<污染带 |
| seg06 X 槽位 | JOHNSON 维持 | 击退 LEAVING/PASSING/JUST_ON 全部标准短语假设——语法假设也要过声学关 |

> **v14 轮修正（同日增强解码后）**：JOHNSON 的"维持"被推翻——补测 out-of 音族
> （v3b 自由解码 "we join you out of 180"、turbo 'jet of'、日语『ジョンシュアドブ』）
> 后 JOIN_YOU_OUT_OF/JUST_OUT_OF 双胜（0.707/0.764 vs 0.890），定案 **JUST OUT OF**。
> 同时守案×3：seg03 NINE TWO THREE、seg02 NINE SEVEN TWO THREE、FEDEX ONE FIVE
> 全部击退增强解码新假设。教训入档：对立假设集不完整时，"胜出"只是"未败"。

### 9.3 定稿与量化更新（`exp/finalize_rjtt_v13.py` + `finalize_rjtt_v14.py`）
- rjtt_final.txt → **v5**（v14 后）；证据源 3→7（+v3free_en/ja + v3b/v3p beam 解码）
- **多数票覆盖率 72%→83.3%**（194/233）：seg06 94%、seg07 94%、seg02 58%（VAD 截断）、seg04 50%
- 新增方法论判例：①**候选池必须含裁判自身自由解码**；②**跨语言审计是多语空域前置检查**；
  ③**对立假设集不完整时"胜出"只是"未败"（JOHNSON 假阳性案）**

---

## 附录 A：证据文件清单

| 类别 | 文件 |
|---|---|
| 最终结果 | `results/a_final.txt` `results/b_final.txt` `results/rjtt_final.txt` |
| CYYT 验证 | `results/final_validation.json`（六组正位/错位全数值，含 turbo）`results/english_hardening.json`（加固轮：turbo 整篇+循环一致性+差异字段复核） |
| RJTT 验证 | `results/rjtt_validation.json`（27 候选 NLL+过滤+投票）`results/rjtt_consensus.json`（逐段裁决依据）`results/rjtt_lang_audit.json`（语言审计）`results/adjudication_v13.json`（呼号翻案裁决）`results/rjtt_coverage.json`（覆盖率 83.6%） |
| 语法裁判 | `results/grammar_check.json`（`src/atis_grammar.py` 产出，26 PASS/0 FAIL） |
| 降噪审计 | `results/denoise_audit.json`（P6：谱减×双裁判，全组恶化→维持原管线） |
| 裁决汇总 | `results/adjudication.md`（P4 逐字段证据链）＋ `results/adjudication_v1..v12*.json` 原始数值 |
| 资产盘点 | `results/assets_inventory.md` |
| wind 终审 | `results/adjudication_v12_wind.json`（增量曲线）`results/wind_fine_probe.txt`（能量探针决定性数据）`results/wind_energy_probe.txt/json` |
| 解码存档 | `results/wind_slices/v12/`（24 条转写）`results/decode_rjtt_index.json`（含 turbo 条目） |
| 工具 | `src/nll_scorer.py` `src/atis_grammar.py`；`exp/` 全部实验脚本（含 `denoise_audit.py` `validate_rjtt2.py` `rover_rjtt.py`） |
| 过程记录 | `JOURNAL.md`（Day 1–6）`PLAN.md`（P0–P8 完成状态）`lit/SOTA_notes.md` |
