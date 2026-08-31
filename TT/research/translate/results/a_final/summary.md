# a_final 翻译多系统对比

EN 行数: 12
系统: template, m2m_direct, qwen_free, qwen_constrained

| 系统 | 数字保真 | 术语命中 | 回译chrF |
|---|---|---|---|
| template | 1.0 | 1.0 | 0.4312 |
| m2m_direct | 0.4167 | 0.2121 | - |
| qwen_free | 0.9167 | 0.8182 | - |
| qwen_constrained | 1.0 | 1.0 | 0.3684 |

## 两两 chrF 一致性（越高=译文越接近）

| | template | m2m_direct | qwen_free | qwen_constrained |
|---|---|---|---|---|
| template | — | —— | —— | —— |
| m2m_direct | 0.136 | — | —— | —— |
| qwen_free | 0.157 | 0.054 | — | —— |
| qwen_constrained | 0.379 | 0.129 | 0.309 | — |

## 推荐系统：qwen_constrained

综合得分(2*数字保真+术语命中+回译chrF): 3.3684

## 最终推荐译文

**1.** `SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU`
→ 圣约翰斯 信息F 天气世界协调时零二零零

**2.** `WIND TWO FOUR ZERO AT FIVE`
→ 风向二四零，风速五节

**3.** `VISIBILITY ONE FIVE TWO FOUR THOUSAND FEET`
→ 能见度 15，云底高 24000 英尺

**4.** `TEMPERATURE ONE DEW POINT MINUS ONE`
→ 温度一，露点零下一

**5.** `ALTIMETER THREE ZERO TWO THREE`
→ 修正海压三零二三

**6.** `APPROACH RNAV ZULU RUNWAY TWO EIGHT`
→ 进近RNAV Z 跑道28

**7.** `INFORM GANDER CENTER ON FREQUENCY ONE TWO THREE DECIMAL ONE FIVE AS REQUESTED`
→ 首次联系通知冈德中心，频率123.15，按需

**8.** `APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT`
→ 进近首次联系，落地和起飞跑道28

**9.** `INFORM ATC THAT YOU HAVE INFORMATION FOXTROT`
→ 通知管制，您已收到信息F

**10.** `APPROACH RNAV ZULU RUNWAY TWO EIGHT`
→ 进近RNAV Z 跑道28

**11.** `APPROACH ON INITIAL CONTACT LANDING AND DEPARTING RUNWAY TWO EIGHT`
→ 进近首次联系，落地和起飞跑道28

**12.** `INFORM ATC THAT YOU HAVE INFORMATION FOXTROT`
→ 通知管制，您已收到信息F

