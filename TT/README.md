# TT 目录说明（ATC 语音识别测试工作区）

> 每段录音、每次降噪、每个模型的识别结果，都在**对固定路径**里，文件名自带标识，任何人都能看懂。

## 目录结构

```
TT/
├── audio/                 # 原始录音（只读，跑任何实验都不改动）
│   ├── CYYT_ATIS_a.wav     # a = 录于 16:00:51（英文，ATIS 播报）
│   ├── CYYT_ATIS_b.wav     # b = 录于 16:01:01（英文，ATIS 播报）
│   └── RJTT_CONTROL.wav    # 日文（管制通话）
├── denoise/               # 降噪实验区
│   ├── methods/           #   降噪方法（每个方法一个 .py，可并存对比）
│   ├── output/            #   降噪产物  [见下方命名规则]
│   ├── qc_report/         #   降噪质量报告(.json机器/可读.md)，和产物同名一一对应
│   └── legacy/            #   旧脚本产物隔离区（仅供参考，不参与新流程）
├── results/               # 识别结果（按模型分）
│   ├── best_pipeline/     #   【最优管线】run_best_asr.py 的输出（日常用）
│   ├── ATC_Whisper/       #   ATC-Whisper 基础版（默认参数，作对比）
│   ├── Qwen3ASR/          #   Qwen3-ASR-1.7B
│   └── FunASR/            #   旧 FunASR 结果（含 5 次重复 + 降噪版）
├── scripts/               # 所有脚本
│   ├── run_best_asr.py    #   【最优管线】研究结论固化：调参+清洗（日常用）
│   ├── run_denoise.py     #   降噪
│   ├── run_atc_whisper.py #   ATC-Whisper 识别（基础版）
│   ├── run_qwen.py        #   Qwen 识别
│   ├── qc_check.py        #   音频质量检查
│   └── README.md          #   【重要】脚本讲解：实现原理 + 参数含义 + 指标判断 + 能改哪
├── research/              # 深度研究区（含最终结果 + 报告 + 实验脚本，见 deep/FINAL_REPORT.md）
│   └── deep/             #   权威终稿：a_final/b_final/rjtt_final + 方法论 + 复现实验
└── models/                # 模型权重（唯一不动的目录）
    └── whisper-large-v3-finetuned-for-ATC/
```

## 文件名命名规则（解决"尝试很多次"的问题）

- **原始录音**：`机场_内容.wav`，如 `CYYT_ATIS_a.wav`
- **降噪产物**：`<录音>__m<方法编号>__dn<版本号>.wav`
  - `m1` = 1 号降噪方法，`dn1` = 第 1 版（下次用 `dn2`、`dn3`…不会覆盖旧版）
  - 例：`CYYT_ATIS_a__m1__dn1.wav`
- **识别结果**：存到 `results/<模型>/<录音名>/result.txt` + `result.json`（json 含模型/语言/时间/文本），每段录音一个子目录，互不干扰
  - 例：`results/Qwen3ASR/CYYT_ATIS_a/result.txt`
  - 语言默认自动检测（中英日都行），要强制某语言用 `--lang en` / `--lang ja`
  - 历史迁移数据保留原文件名（如 `qwen_result.txt`、`result_1.txt`），新脚本统一生成 `result.txt`

## 常用命令

```bash
cd /siyuan/FunASR_extracted/FunASR-main/TT

# 降噪：全部音频 × 全部方法
conda run -n lingbot-map python scripts/run_denoise.py

# 只看有哪些降噪方法
conda run -n lingbot-map python scripts/run_denoise.py --list-methods

# 识别：ATC-Whisper 识别全部录音
conda run -n lingbot-map python scripts/run_atc_whisper.py --all

# 识别：Qwen 识别某段录音
conda run -n lingbot-map python scripts/run_qwen.py CYYT_ATIS_a

# 质量检查某段音频
conda run -n lingbot-map python scripts/qc_check.py audio/CYYT_ATIS_a.wav
```

## 工作流两步即可

1. **降噪**（可选）：`run_denoise.py`，产物进 `denoise/output/`，指标进 `denoise/qc_report/`
2. **识别**：`run_atc_whisper.py` 或 `run_qwen.py`，结果进 `results/<模型>/`

> 提示：改脚本靠 git 管理（不用复制多个脚本文件），降噪方法才需要并存多份。