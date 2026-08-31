# ATC 空管语音识别研究（ATC-ASR-Research）

> 基于 FunASR 框架的空管（ATC）语音研究工作区：在**无真实对照文本**的极端约束下，
> 完成 3 条空管录音的**权威转写 → 真流式识别 → 专业中文翻译**全链路，
> 全程仅依赖客观裁判（声学似然 / ICAO·METAR·ATIS 语法硬约束 / 多系统交叉验证），不自设主观评分。

硬件与环境：RTX 3090 24GB，conda 环境 `lingbot-map`。

## 1. 研究语料（`TT/audio/`，只读）

| 音频 | 内容 | 语言 |
|---|---|---|
| `CYYT_ATIS_a.wav` | St. John's (CYYT) ATIS 循环广播，约 2 分钟 | 英文 |
| `CYYT_ATIS_b.wav` | 同台另一时段 ATIS（经论证为**不同日期**录音） | 英文 |
| `RJTT_CONTROL.wav` | 羽田（RJTT）管制频率通话，多机呼号 | 英/日 |

## 2. 仓库结构

```
FunASR-main/
├── funasr/ · runtime/ · examples/ ...   # FunASR 上游框架（依赖，未改动）
└── TT/                                  # ★ 本研究的全部工作区
    ├── audio/          # 原始录音（只读）
    ├── denoise/        # 降噪实验：methods/ 方法、output/ 产物、qc_report/ 质检
    ├── results/        # 各模型识别结果（best_pipeline / ATC_Whisper / Qwen3ASR / FunASR）
    ├── scripts/        # 正式脚本（run_best_asr / run_denoise / run_atc_whisper / run_qwen / qc_check）
    ├── models/         # 模型权重占位（权重不入库，见 §5）
    └── research/
        ├── deep/        # 阶段一：离线深度研究，产出权威终稿 a_final / b_final / rjtt_final
        ├── streaming/   # 阶段二：真流式识别（SimulStreaming + 2-pass 精修 + 模板融合）
        └── translate/   # 阶段三：EN→ZH 专业翻译（规则模板 + LLM + 审计闭环）
```

每项研究自带 `PLAN.md`（计划）→ `JOURNAL.md`（逐轮日志）→ `FINAL_REPORT.md`（终稿与全部证据），
审查记录见 `TT/REVIEW_LOG.md`。

## 3. 三个阶段的核心结论

### 阶段一 · deep：无真值条件下的权威转写
- 三条音频全部产出权威终稿（`research/deep/results/`），关键结论均有可复现证据：
  NLL forced-scoring、ICAO/ATIS 语法硬校验、多模型交叉共识、人工听音锚窗复核。
- 论证了 a/b 为**不同日期**播报（仅修压 3023/3033 与 AS/WHEN REQUESTED 两处信道级差异）。

### 阶段二 · streaming：真流式识别（`research/streaming/`）
- 方案：SimulStreaming(AlignAtt) 流式引擎 + ATC 微调 whisper-large-v3，
  RMS-CV 调制门在句末触发 offline 精修，再经台站模板证词融合（2-pass）。
- 关键数字（受控 K4 口径，含先验声明审计）：
  - L2 模板融合档：a/b WER = 0.0 / 0.0（成色：含当日 ATIS 文本先验）
  - L0 真零先验档：三引擎 ROVER + 公开语法层后 a=0.25→0.0211 / b=0.338→0.2254
  - 草稿轨词延迟中位 ~1.7-1.9s，final 轨 ~14.5s；RTF 满足实时约束
- 诚实性纪律：所有"含先验"数字均标注成色（src=tpl 占比等），模板污染的数字已勘误归档。

### 阶段三 · translate：EN→ZH 专业翻译（`research/translate/`）
- 方案：规则模板翻译器做数值裁判 + Qwen2.5-7B 术语约束翻译 + 审计反馈闭环（≤3 轮重译）。
- 终稿指标：**数字保真 1.0 / 术语命中 1.0**；并用裸通用 MT 对照组证明
  空管翻译是"术语映射 + 数字读法还原"问题而非自由 MT。
- 流式输出直接作为翻译输入时同样 1.0/1.0（L2 档）；零先验档端到端指标见 streaming 报告。

## 4. 日常工作流（两步）

```bash
cd TT

# ① 降噪（可选）：全部音频 × 全部方法，产物进 denoise/output/，质检进 denoise/qc_report/
conda run -n lingbot-map python scripts/run_denoise.py
conda run -n lingbot-map python scripts/run_denoise.py --list-methods   # 查看方法

# ② 识别：最优管线（调参+清洗固化版，日常用）
conda run -n lingbot-map python scripts/run_best_asr.py

# 单模型对比
conda run -n lingbot-map python scripts/run_atc_whisper.py --all
conda run -n lingbot-map python scripts/run_qwen.py CYYT_ATIS_a

# 音频质检
conda run -n lingbot-map python scripts/qc_check.py audio/CYYT_ATIS_a.wav
```

结果落盘规则：`results/<模型>/<录音名>/result.txt + result.json`；
降噪产物命名 `<录音>__m<方法号>__dn<版本>.wav`，永不覆盖历史版本。
脚本逐行讲解见 `TT/scripts/README.md`。

## 5. 模型依赖（权重不入库）

| 模型 | 用途 | 获取 |
|---|---|---|
| [whisper-large-v3-finetuned-for-ATC](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC) | ATC 主识别引擎 | HF 下载至 `TT/models/` |
| openai/whisper-large-v3 | 交叉验证 / ROVER 旁证 | `from_pretrained` 自动缓存 |
| Qwen/Qwen3-ASR-1.7B | 旁证 ASR | ModelScope/HF |
| Qwen2.5-7B-Instruct | 翻译主力 | HF |
| facebook/m2m100_418M | 回译对照 | HF |
| SimulStreaming (AlignAtt) | 流式解码引擎 | `TT/research/refs/SimulStreaming-main` |

## 6. 复现研究结论

- 离线终稿复现：`TT/research/deep/`（PLAN 定义协议，src/ 内含复现实验脚本）
- 流式全档复现（L0/L1/L2）：`TT/research/streaming/`，meta 自带 `prior=` 自描述
- 翻译复现：`TT/research/translate/`（含数字保真/术语命中审计脚本）

各 `FINAL_REPORT.md` 附录列有全部证据文件清单；评测口径变更历史（如 J12 参考修正）
在 `JOURNAL.md` 中逐条可查。

## 7. 许可

上游 FunASR 遵循其原 License（MIT）；`TT/` 研究内容为本仓库作者所有。
