# ATC 空管语音识别研究（ATC-Speech-Research）

[简体中文](./README.md) | [English](./README_en.md)

> 基于 FunASR 框架的空管（Air Traffic Control）语音研究工作区。
> 核心命题：**在无真实对照文本、仅 3 条录音的极端约束下**，完成
> 权威转写 → 真流式识别 → 专业中文翻译 的全链路，并建立一套
> **客观可复现的评测体系**（声学似然 / ICAO·METAR·ATIS 语法硬约束 / 多系统交叉验证），
> 全程不自设主观评分当裁判。
>
> **项目状态：进行中。** deep / translate 阶段已出权威终稿；streaming 阶段已完成
> P4 审计并定版 L2/L1 档，**清洗词表后的干净 L0 复跑与重报正在进行**（见 §10）。

---

## 0. 结果速览（真实端到端产物，非示意）

```
🎧 CYYT_ATIS_a.wav（弱信道 ATIS 广播，循环 ~5 遍）
        │  deep 阶段：四重客观证据定稿
        ▼
SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU
WIND TWO FOUR ZERO AT FIVE
        │  translate 阶段：术语约束翻译 + 审计闭环（数字保真 1.0 / 术语命中 1.0）
        ▼
圣约翰斯 信息F 天气世界协调时零二零零
风向二四零，风速五节
```

流式档（streaming）同一管线以 ~1.7s 中位词延迟增量出草稿、句末自动精修，
L2 档 WER 0.0 / L1 档 0.0845（口径与成色见 §4、§7）。

## 1. 研究语料（`TT/audio/`，只读，任何实验不得改动）

| 音频 | 内容 | 语言 | 特性 |
|---|---|---|---|
| `CYYT_ATIS_a.wav` | St. John's (CYYT) ATIS 循环广播，~270s | 英文 | 每 ~28s 循环一遍，共 ~5 遍，各遍信道质量不同 |
| `CYYT_ATIS_b.wav` | 同台另一时段 ATIS（论证为**不同日期**录音） | 英文 | 信号更弱，自动语言检测会误判"无语音"；末尾有削波噪声段 |
| `RJTT_CONTROL.wav` | 羽田 (RJTT) 管制频率通话，多机呼号 | 英/日 | 多说话人、多呼号、真实管制交互 |

## 2. 环境与依赖

- 硬件：RTX 3090 24GB
- conda 环境：`lingbot-map`（torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 本地安装）
- 网络（国内机器实测配置）：
  - huggingface.co 直连不通 → 用 `export HF_ENDPOINT=https://hf-mirror.com`
  - pip 走清华源；clone GitHub 仓库需 ghproxy 代理
- FunASR 框架（仓库根目录）为本研究依赖的上游代码，未做修改；所有研究工作只在 `TT/` 内书写。

## 3. 目录结构

```
FunASR-main/
├── funasr/ · runtime/ · examples/ ...      # FunASR 上游框架（依赖，未改动）
└── TT/                                     # ★ 本研究的全部工作区
    ├── audio/            # 原始录音（只读）
    ├── denoise/
    │   ├── methods/      #   降噪方法：NN_方法名.py，实现 denoise(y, sr) 即自动纳入
    │   ├── output/       #   降噪产物 <录音>__m<方法号>__dn<版本>.wav（自动递增不覆盖）
    │   ├── qc_report/    #   质检报告（json 机器读 + md 人读，与产物同名对应）
    │   └── legacy/       #   旧脚本产物隔离区
    ├── results/          # 正式管线识别结果（best_pipeline / ATC_Whisper / Qwen3ASR / FunASR）
    ├── scripts/          # 正式脚本（逐行讲解见 TT/scripts/README.md）
    ├── models/           # 模型权重存放处（权重不入库，见 §7）
    ├── REVIEW_LOG.md     # 代码审查日志（增量）
    └── research/         # 深度研究区
        ├── deep/         #   阶段一：离线权威转写
        ├── streaming/    #   阶段二：真流式识别（进行中）
        ├── translate/    #   阶段三：EN→ZH 专业翻译
        └── refs/         #   第三方参考实现（SimulStreaming 等）
```

每个研究子项目固定三件套：`PLAN.md`（协议与红线）→ `JOURNAL.md`（逐轮实验日志，含
负结果与勘误）→ `FINAL_REPORT.md`（终稿结论 + 证据文件清单）。

## 4. 评测体系（本项目的方法论核心）

- **客观裁判三件套**：① NLL 声学似然（forced scoring，不给文本就无法"觉得对"）
  ② ICAO/METAR/ATIS 语法硬约束（槽位结构、NATO 字母表、数字读法）
  ③ 多系统交叉验证（跨模型族共识）+ 物理测量（能量/周期/锚窗探针）。
- **先验分档（streaming 阶段，防止先验污染指标）**：
  - **L0 真零先验**：无任何当日文本，纯声学 + 公开标准词法
  - **L1 +文本提示**：ATIS 文本作解码 prompt（工业对应"系统持有公开播报文本"）
  - **L2 +模板融合**：台站模板证词融合（模板文本源自 deep 终稿）
  - 所有交付数字必须标注档位；meta 自带 `prior=` 自描述；含模板输出的词标注 `src=tpl`。
- **受控评测口径 K4**：对修正后的参考文本（284 token 口径，J12 参考修正）计算 token-WER，
  历史口径（幻影复诵参考）已作废并在 JOURNAL 留痕。

## 5. 快速开始与新机器部署

```bash
# ① 克隆
git clone git@github.com:CEasonK/ATC-Speech-Research.git && cd ATC-Speech-Research

# ② 环境（版本参考 §2；funasr 直接装仓库本地这份，保证与研究环境一致）
conda create -n atc python=3.10 -y && conda activate atc
pip install -e .
pip install torch transformers modelscope funasr noisereduce soundfile  # 按需补齐

# ③ 权重：按 §8 表格下载到 TT/models/（HF 直连不通先 export HF_ENDPOINT=https://hf-mirror.com）

# ④ 跑通第一条识别
cd TT && python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
# 结果在 results/best_pipeline/CYYT_ATIS_a/result.txt
```

研究复现（deep / streaming / translate）不需要额外部署，按 §9 用对应 conda 环境跑即可。

## 6. 日常工作流（正式管线，两步）

```bash
cd TT

# ① 降噪（可选，默认不降噪——见下方教训）
conda run -n lingbot-map python scripts/run_denoise.py            # 全部音频 × 全部方法
conda run -n lingbot-map python scripts/run_denoise.py --list-methods
conda run -n lingbot-map python scripts/run_denoise.py --method 2 --audio CYYT_ATIS_a

# ② 识别（最优管线 run_best_asr.py = 研究结论固化，日常用这个）
conda run -n lingbot-map python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
#   --full      保留全部循环遍（默认锚点去重只留最干净的 1 遍）
#   --no-clean  只要原始解码输出（关后处理）
#   --anchor    更换循环锚点短语（换报文类型时）

# 单模型对比
conda run -n lingbot-map python scripts/run_atc_whisper.py --all        # 30s 滑窗 + 2s 重叠
conda run -n lingbot-map python scripts/run_qwen.py CYYT_ATIS_a --lang English
conda run -n lingbot-map python scripts/qc_check.py audio/CYYT_ATIS_a.wav   # 只看 SNR/响度/峰值
```

结果落盘：`results/<模型>/<录音名>/result.txt + result.json`（json 含模型/语言/时间/文本）。

**已固化的关键工程决策**（均有实验依据，改前先读 `TT/scripts/README.md`）：
- 原始音频直通，不降噪：a 有噪声降噪有效、b 太安静降噪有害，不存在两全参数；
  流式 R7 实验再次证实降噪前端使 WER 恶化（抹掉弱读词声学线索）。
- ATIS 循环去重用"整遍术语打分"而非逐词投票（逐词共识 44% 失败）。
- 幻觉过滤用 whisper 内置 no_speech VAD（能量 VAD 对满能量噪声段无效）。
- Qwen3-ASR 必须显式 `--lang English`：弱信号下自动检测返回空（2026-08-21 实测教训）。

## 7. 三阶段研究详情

### 阶段一 · deep —— 无真值条件下的权威转写（已完成）
- **产出**：`results/a_final.txt`、`b_final.txt`、`rjtt_final.txt`（三条音频权威终稿），
  成为后续所有研究的对照基准。
- **代表结论**：a/b 为不同日期播报（仅修压 3023 vs 3033、AS vs WHEN REQUESTED 两处
  信道级差异）；a 末尾三行是真实复诵（人工听音 + 锚窗探针独立佐证），非模型幻觉。
- 复现：`research/deep/`（PLAN 定义四重证据协议，src/ 内含复现脚本，附录 A 证据清单）。

### 阶段二 · streaming —— 真流式识别（进行中）
- **架构**：SimulStreaming(AlignAtt) 流式引擎 + ATC 微调 whisper-large-v3 主干，
  RMS-CV 调制门在句末触发 offline 精修（beam=5），再经台站模板证词融合
  （ATC CT2 主精修 + Qwen3-ASR 旁证 worker + 低证词率护栏），2-pass 交付。
- **当前数字**（K4 口径）：
  | 档位 | a | b | 说明 |
  |---|---|---|---|
  | L2 模板融合 | 0.0 | 0.0 | 与 deep 终稿逐字一致；成色：含模板先验，35-39% 输出词 src=tpl |
  | L1 文本提示 | 0.0845 | 0.1162 | 模板作文本提示（非零先验） |
  | L0 三引擎 ROVER | ⚠ 0.1303 / 0.2711 → **作废待重跑** | | 见 §9 P4 勘误 |
  - 延迟：草稿轨词延迟中位 ~1.7-1.9s（RTF ~0.5 满足实时约束）；final 轨 ~14.5s。
- **负结果存档**（同样是结论）：单旁证尤其同源 v3 反噬主引擎；m1 降噪前端使 WER 恶化；
  跨周期共识路线不成立；qwen 预热必须放 worker 启动期（首调用 32s）。

### 阶段三 · translate —— EN→ZH 专业翻译（已完成，随识别输入档位联动）
- **架构**：T3 规则模板翻译器（0 参数，做数值正确性裁判+兜底）+ T1 Qwen2.5-7B
  术语约束翻译（主力生成）+ 审计反馈闭环（不合格行带错误原因回喂重译，≤3 轮）；
  T4 M2M-100 做独立对照与 zh→en 回译。
- **指标**：数字保真（英文拼读数字→载荷序列全等）与术语命中（21 条 EN→ZH 词对）
  两项审计，终稿 **1.0 / 1.0**；以 deep 终稿或 L2 流式输出为输入均可达标。
- **已证明**：空管翻译不是自由 MT 问题，是"术语映射 + 数字读法还原"问题
  （裸 M2M 失败模式存档：WIND→赢得、ALTIMETER→最大）。
- **已知边界**：以零先验识别输出为输入时端到端指标大幅下降（a 数字 0.359/术语 0.833，
  b 更差）——端到端达标依赖先验档。

## 8. 模型依赖（权重不入库，按此表自行获取到 `TT/models/` 或 HF 缓存）

| 模型 | 用途 | 获取 |
|---|---|---|
| [whisper-large-v3-finetuned-for-ATC](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC) | ATC 主识别引擎 | HF → `TT/models/` |
| openai/whisper-large-v3 | 交叉验证 / ROVER 旁证 | `from_pretrained`（走 hf-mirror） |
| Qwen/Qwen3-ASR-1.7B | 旁证 ASR worker | ModelScope / HF |
| Qwen2.5-7B-Instruct | 翻译主力 | HF |
| facebook/m2m100_418M | 回译对照 | HF |
| SimulStreaming (AlignAtt) | 流式解码引擎 | 已随仓库：`TT/research/refs/SimulStreaming-main` |

## 9. 复现指南

1. **日常管线**：§6 命令直接可跑（需先备齐 §8 权重）。
2. **deep 终稿**：按 `research/deep/PLAN.md` 协议依次跑 `src/` 内脚本；每步产物与
   `results/*_final.txt` 逐字对比。
3. **streaming 全档**：`research/streaming/src/run_2pass.py <wav> <out> --chunk 1.0 --half
   --rover --no_prompt`（L0 零先验档）；去掉 `--no_prompt` 为 L1；加模板融合为 L2。
   评测统一走 `src/metrics.py`（K4 口径）。
4. **translate**：`research/translate/` 内含数字保真/术语命中审计脚本，可对任意
   译文独立复算两项指标。
5. 任何数字有疑问 → 按 `JOURNAL.md` 逐轮日志回放（口径变更、勘误全部留痕）。

## 10. 当前进度与待办（2026-08-31）

- ✅ deep：三音频权威终稿定版，证据链闭环
- ✅ translate：终稿 1.0/1.0，审计工具可独立复用
- ✅ streaming：L1/L2 定版；P1-P3 修复 + 回归测试（`tests/test_p3_fixes.py`）
- 🔶 **streaming P4 审计勘误（2026-08-31）**：R1-R9 的词表/语法规则经审计发现系对着
  K4 评测集真值残错分析选定（金参考监督下的自适应），**L0 数字 0.1303/0.2711/0.0211/0.2254
  全部作废**；`atis_lexicon.py` 已按"公开标准可推导 + 不删引擎词"清洗。
  **待办：用清洗后规则重跑干净 L0 并重新报告**（a/b 双轨，含全片交付与端到端翻译联动）。
- ⬜ 计划中未落地：Kyutai STT、NeMo FastConformer 流式对照（streaming PLAN 中 E3/E4 路线）
- ⬜ 弱信道鲁棒性：b 轨残错（周期接缝错位、降质段数值崩塌）待解，属声学底层能力问题

## 11. 许可

上游 FunASR 遵循其原 License（MIT）；`TT/` 研究内容为本仓库作者所有。
