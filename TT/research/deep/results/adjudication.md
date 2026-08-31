# 争议字段声学裁决汇总（P4 交付）

> 每个字段 = 竞争假设清单 → 三裁判对立计分 → （对峙时）物理测量终审。
> NLL 单位：nat/token。**判例基准：单词插入级 ΔNLL ≤ 1.4 nat 视为 LM 先验污染区间，不可单独定案。**

## CYYT_ATIS_a / b

### opening — `SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU`
竞争假设：`THIS IS SAINT JOHNS…` / `SAINT JOHNS INTERNATIONAL…` / `CYYT SAINT JOHNS…`
| 变体 | v3 NLL |
|---|---|
| **THIS IS** SAINT JOHNS … | 1.879 |
| SAINT JOHNS …（采纳） | 1.959 |
| SAINT JOHNS INTERNATIONAL … | 2.131 |
| CYYT SAINT JOHNS … | 2.458 |
裁决：THIS-IS 仅以 Δ0.080 领先（≪1.4 污染区间），其余裁判不支持插入词；INTERNATIONAL/CYYT 被双裁判明确否决。按多数维持无 THIS-IS。证据 `adjudication_v7b.json`。

### wind — `WIND TWO FOUR ZERO AT FIVE`（a、b 同）⭐ 全案最硬
- v10：窗口漂移伪象（不同窗不可比）→ 弃
- v11 同窗对立：atc 裁判 b-WIND Δ1.27 支持 WIND——落在污染区间内，**不能单独定案**
- v12 增量曲线（atc）：b 终点 H_wind 0.279 < H_nowind 0.406；v3 同向但差距小
- 自由解码：a 11/11 无 AT；b 7/7 无 WIND（含 turbo）→ 与 forced 计分**对峙**
- **能量包络终审（`wind_fine_probe.txt`）**：
  - a：AT 槽位 57.88–58.01s 连续浊音 RMS 0.06–0.14（真词间空隙 0.025–0.04）→ 弱化 /ət/ 存在
  - b：233.35–233.42s 8 帧纯底噪把区域切成两词；ZULU=233.05–233.33（290ms≈a 信道 ZULU 310ms，
    排除"长 ZULU"假设）；WIND=233.44–233.81（强元音头 0.22–0.33 + /nd/ 鼻音平台 0.10–0.17）；
    TWO@233.92(峰0.4567)/FOUR@234.14/ZERO@234.43/AT@234.99/FIVE@235.26 七 burst 全归属
定案：**WIND 存在**。"自由解码全票缺席 ≠ 声学缺席"判例来源。

### visibility — `VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET`
v6 定案。语法器双读法校验：2400ft(RVR 逐位读法)在域 [1200,6500] ✓。

### temperature — `TEMPERATURE ONE DEW POINT MINUS ONE` ⭐ 语法否决声学层判例
atc 裁判原始偏好非法值 `ONE DECIMAL NINER ONE`（NLL 1.530 < 合法值 2.512，`adjudication_v6.json`）
——但 METAR 温度必为整数，语法层一票否决；切片解码自由听写亦未听到 DECIMAL；
v8 证明该量级差异属 LM 插入污染（≤1.4 nat）。三重证据下采纳合法值。
露点 MINUS ONE 与温度构成 dew(−1) ≤ temp(+1) 物理自洽 ✓。

### altimeter — a: `THREE ZERO TWO THREE` / b: `THREE ZERO THREE THREE`
v5（atc，dual_agree）：b 的 3033=2.205 < 3023=2.304 ✓；v7b 第三裁判复核 2:1。
切片佐证：b_c240_atc 听到 "ALTITUDE THREE ZERO THREE THREE"。
语法：30.23 / 30.33 inHg ∈ [27.50,31.50] ✓（`grammar_check.json` PASS）。
加固轮 turbo 复核：b 同向（Δ0.032 平手级）；a 上 turbo 弱偏好 3033（Δ0.113）——
噪声级 ≪ 污染区间，不敌解码全票+双裁判 → 记录弱异议，维持原判。

### approach + runway — `APPROACH RNAV ZULU RUNWAY TWO EIGHT`
外部航图 + v3 一致；跑道号格式合法（01–36）。注：PLAN 早期存疑"CYYT 现用 11/29"，
但音频证据与外部资料一致指向 TWO EIGHT，语法器只校验编号合法性不裁机场平面图。

### freq + closing — `INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE`
- 频率：123.15 MHz，VHF 频段内且恰在 25kHz 网格 ✓（早期候选 33.15 非法已弃）
- 尾词：a=`AS REQUESTED` / b=`WHEN REQUESTED`（v7b 三裁判信道级差异成立，
  佐证 a/b 为异日录音）；整篇正位/错位验证 b-v3 2.178<2.714 ✓
  四组汇总见 `final_validation.json`（a-atc 组边缘失败=atc 裁判域饱和已知弱点）。
  加固轮：turbo 整篇验证 a/b 双通过；turbo 复核尾词 a=AS(Δ0.211)/b=WHEN(Δ0.082) 同向 ✓。

### 循环一致性（加固轮新增，a 信道）
4 个有效广播周期实例 Δ(text vs shuffle)=2.62/2.61/2.81/1.39 nat 全部支持终稿；
噪声段 Δ≈0.17 自洽对照。详见 `english_hardening.json`。

### tail — `INFORM ATC THAT YOU HAVE INFORMATION FOXTROT`（×2 复诵为音频事实）

## RJTT_CONTROL（要点，全文见 `rjtt_consensus.json`）
1. 单 v3 裁判翻车案：seg05 "That's possible, anyway"(3.41) 胜过域文本(5.42) → 语法过滤+投票纠正
2. 呼号 SIERRA EIGHT NINER SIX：turbo 在 seg01+seg08 跨段独立复现 + 两段 NLL 双最优
3. seg07 IGOTO：turbo 片假名リゴート同音佐证；双重 direct-climb FL300 = atc+turbo 独立一致
4. 未决（各引擎内部自洽无法外部裁决）：seg00 呼号（AEROFLOT vs airfr）、seg05 碎片、seg06 JOHNSON

## 附：P6 降噪复核结论（`denoise_audit.json`）
谱减法在全部 4 组（2信道×2裁判）使 NLL 恶化 −8%∼−27% → 正式维持无降噪管线，
且升级原结论：降噪失真主动伤害声学模型特征，非"无用"而是"有害"。
