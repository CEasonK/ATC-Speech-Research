# ATC 空管语音识别研究（ATC-Speech-Research）

[简体中文](./README.md) | [English](./README_en.md)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![last commit](https://img.shields.io/github/last-commit/CEasonK/ATC-Speech-Research)

一个基于 [FunASR](https://github.com/modelscope/FunASR) 的**空管（Air Traffic Control）语音识别**研究仓库，
在**无真实对照文本、仅 3 条录音**的极端约束下，用一套**全客观可复现的评测体系**
（声学似然 / ICAO·METAR·ATIS 语法硬约束 / 多系统交叉验证），完成
**权威转写 → 真流式识别 → 专业中文翻译**的全链路。

> 方法论底线：全程只认客观裁判，**不自设主观评分**。
>
> **项目状态：进行中。** deep / translate 阶段已出权威终稿；streaming 阶段已完成
> P4 审计并定版 L2/L1 档，**清洗词表后的干净 L0 复跑与重报正在进行**（见 §10）。

---

## 0. 结果速览（真实端到端产物，非示意）

```
🎧 CYYT_ATIS_a.wav（ATIS 循环广播，多遍质量不一）
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
- conda 环境：研究机为 `lingbot-map`（torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 本地安装）；新机器按 §5 新建独立环境即可
- 网络（国内机器实测配置）：
  - huggingface.co 直连不通 → 用 `export HF_ENDPOINT=https://hf-mirror.com`
  - pip 走清华源；clone GitHub 仓库需 ghproxy 代理
- FunASR 框架（仓库根目录）为本研究依赖的上游代码，未做修改；所有研究工作只在 `TT/` 内书写。

## 3. 目录结构

```
FunASR-main/
├── funasr/                               # FunASR 框架源码（研究依赖，未改动；上游其余目录已清理）
└── TT/                                   # ★ 本研究的全部工作区
    ├── audio/            # 原始录音（只读）
    ├── denoise/
    │   ├── methods/      #   降噪方法：NN_方法名.py，实现 denoise(y, sr) 即自动纳入
    │   ├── output/       #   降噪产物 <录音>__m<方法号>__dn<版本>.wav（自动递增不覆盖）
    │   ├── qc_report/    #   质检报告（json 机器读 + md 人读，与产物同名对应）
    │   └── legacy/       #   旧脚本产物隔离区
    ├── results/          # 正式管线识别结果（best_pipeline / ATC_Whisper / Qwen3ASR / FunASR）
    ├── scripts/          # 正式脚本（逐行讲解见 TT/scripts/README.md）
    ├── models/           # 模型权重存放处（权重不入库，安装见 §5）
    ├── REVIEW_LOG.md     # 代码审查日志（增量）
    └── research/         # 深度研究区
        ├── deep/         #   阶段一：离线权威转写
        ├── streaming/    #   阶段二：真流式识别（进行中）
        ├── translate/    #   阶段三：EN→ZH 专业翻译
        └── refs/         #   第三方参考实现（SimulStreaming 等）
```
每个研究子项目固定三件套：`PLAN.md`（协议与红线）→ `JOURNAL.md`（逐轮实验日志，
含负结果与勘误）→ `FINAL_REPORT.md`（终稿结论 + 证据文件清单）。具体方法在 §7。

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

## 5. 快速开始：模型安装与跑通

模型权重一律不入库，且**各脚本按固定路径读取模型**——下表的路径一列必须严格照做，
不是"下载到哪都行"。

**前置**（版本参考 §2）：

```bash
conda create -n atc python=3.10 -y && conda activate atc
pip install -e .                # 装仓库本地这份 funasr，保证与研究环境一致
pip install -U huggingface_hub modelscope   # 下载工具
export HF_ENDPOINT=https://hf-mirror.com    # 国内机器必设；HF 直连可通则跳过
```

**模型安装表**（路径 = 脚本默认读取位置，写错目录脚本会直接报错）：

| # | 模型 | 下载命令 | 落盘位置（相对仓库根） | 谁在用 |
|---|---|---|---|---|
| 1 | whisper-large-v3-finetuned-for-ATC | `hf download jacktol/whisper-large-v3-finetuned-for-ATC --local-dir TT/models/whisper-large-v3-finetuned-for-ATC` | `TT/models/whisper-large-v3-finetuned-for-ATC/` | `run_best_asr` / `run_atc_whisper` / streaming 主引擎 |
| 2 | Qwen3-ASR-1.7B | `modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir TT/models/Qwen3-ASR-1.7B` | `TT/models/Qwen3-ASR-1.7B/` + **必须** `export QWEN_ASR_MODEL=$PWD/TT/models/Qwen3-ASR-1.7B` | `run_qwen` / streaming 旁证 worker（`run_2pass.py --qwen_model` 亦可显式指定） |
| 3 | Qwen2.5-7B-Instruct | `hf download Qwen/Qwen2.5-7B-Instruct --local-dir TT/research/translate/models/qwen2.5-7b-instruct` | `TT/research/translate/models/qwen2.5-7b-instruct/` | translate 主力生成 |
| 4 | m2m100_418M | `hf download facebook/m2m100_418M --local-dir TT/research/translate/models/m2m100_418M` | `TT/research/translate/models/m2m100_418M/` | translate 对照 / 回译 |
| 5 | whisper-large-v3（原版） | 免手动：`from_pretrained("openai/whisper-large-v3")` 自动走镜像进 HF 缓存 | `~/.cache/huggingface/` | 交叉验证 / streaming 第二精修 |
| 6 | faster-whisper(CT2) 转换版 | 由 #1/#5 用 `TT/research/streaming/src/convert_hf_to_openai.py` 本地转换生成 | `TT/research/streaming/downloads/_ct2_atc/`、`_ct2_v3/` | 仅 streaming 阶段 |

**环境变量与路径依赖汇总**：

- `QWEN_ASR_MODEL`：覆盖 #2 的旧机器默认路径（不设则指向不存在的 `/siyuan/Qwen3_ASR/...`，直接报错）。
- `HF_ENDPOINT=https://hf-mirror.com`：所有走 HF 的脚本（#5 及 deep 复现）都需要。
- `TT/research/streaming/src/common.py` 中 `TT_ROOT` 硬编码了研究机路径——换机器部署
  streaming 复现时改这一处，或按相同路径部署。

**跑通验证**（装完 #1 即可验证日常管线）：

```bash
cd TT && python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
# 预期结果可与已入库的 results/best_pipeline/CYYT_ATIS_a/result.txt 逐字对比
```

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

**任务**：没有任何对照文本，只有 3 条录音，要让转写尽可能接近真值——且每个字都拿得出证据。

**方法论核心：三裁判 × 五类客观工具**（全程禁止"我觉得像"当裁判）：

1. **三裁判制**：whisper-atc（域先验强但自我偏置）/ whisper-large-v3 原版（中立）/
   turbo-atcosim（第三独立引擎）。铁律：**任何字段定案需 ≥2 个独立证据源同向，单一裁判永不定案**。
2. **同窗对立计分**：竞争假设放进**同一锚定窗口**做 forced-NLL 对比，消除窗口漂移假象
   （早期踩坑：把文本放到静音段计分 NLL 会假性极低——"静音窗口假象"）。
3. **LM 先验污染标定**：量化出单词插入级 ΔNLL 可达 1.2–1.4 nat 而纯来自 decoder 语言先验
   ——ΔNLL 落在该区间时**禁止单独定案**，必须另找裁判（v11 WIND 案 Δ1.27 即触发此规则）。
4. **切片自由解码**：切 14–16s 无上下文片段让多引擎独立听写，破除长音频上下文锚定。
   关键发现：**全票缺席 ≠ 声学不存在**——劣化音频上弱读词 /ət/、鼻音 /wɪnd/ 被 11/11 全票
   漏掉但物理上存在 → 自由解码只能作"存在"的正证据，不能作"不存在"的证据。
5. **能量包络物理探测（破局者）**：NLL 与解码对峙时，用无语言先验的 10ms RMS 包络 +
   burst 检测终审。两个疑难定案均由它裁决——a 的弱化 AT（57.88s 连续浊音 RMS 0.06–0.14
   vs 真词间空隙 0.025–0.04）；b 的 WIND（鼻音平台特征 + 7 个 burst 逐一归属后续词串）。
6. **语法层否决声学层**：METAR/ATIS 硬约束（温度必整数、VHF 频率 118–137、修压 28.xx
   格式合法域）——语法非法的"听感最优解"直接否决。

**代表性战果**（每个都有 `exp/adjudicate_v*.py` 复现脚本）：
- **v4 翻案**：呼号 SIERRA→SHANGHAI AIR（v3 自由解码双段独立命中 + qwen 同窗配对同向 +
  域先验三重证据，turbo 的 SIERRA 记为竞争假设存档）；ORANGE NINER→ORANGE LINER（日语
  轨 オレンジライナー + qwen 双源）。
- **a/b 关系论证**：非重录而是**不同日期**播报——全文仅修压（3023 vs 3033）与
  AS/WHEN REQUESTED 两处差异，且均为信道级/播报级。
- **末三行复诵真伪**：人工听音 + 锚窗探针物理佐证，确认是真实复诵而非模型幻觉。

**产出**：`results/a_final.txt`、`b_final.txt`、`rjtt_final.txt`（RJTT 为 9 段共识合成，
每段带置信度分层），成为后续 streaming / translate 全部研究的对照基准。
完整证据链见 `research/deep/FINAL_REPORT.md` 与附录 A 清单。

### 阶段二 · streaming —— 真流式识别（进行中）
- **架构**：SimulStreaming(AlignAtt) 流式引擎 + ATC 微调 whisper-large-v3 主干，
  RMS-CV 调制门在句末触发 offline 精修（beam=5），再经台站模板证词融合
  （ATC CT2 主精修 + Qwen3-ASR 旁证 worker + 低证词率护栏），2-pass 交付。
- **当前数字**（K4 口径）：
  | 档位 | a | b | 说明 |
  |---|---|---|---|
  | L2 模板融合 | 0.0 | 0.0 | 与 deep 终稿逐字一致；成色：含模板先验，35-39% 输出词 src=tpl |
  | L1 文本提示 | 0.0845 | 0.1162 | 模板作文本提示（非零先验） |
  | L0 三引擎 ROVER | ⚠ 0.1303 / 0.2711 → **作废待重跑** | | 见 §10 P4 勘误 |
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

## 8. 模型依赖总览

全部 6 项模型权重的**下载命令、落盘路径、环境变量**见 §5 安装表（权重不入库）。
一句话版：ATC 微调 whisper（主识别）· whisper-large-v3 原版（交叉验证）·
Qwen3-ASR（旁证）· Qwen2.5-7B（翻译）· m2m100（回译对照）· SimulStreaming（流式引擎，代码已随仓库）。

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

本项目采用 **MIT License**（见根目录 [LICENSE_ATC.md](./LICENSE_ATC.md)，可由任何用途自由使用、修改、再分发）。

> 注：仓库根目录另有一份 `LICENSE` 为上游 FunASR 的原始许可（MIT，© 2025 FunASR，
> 仅约束其框架代码）。本仓库的 ATC 研究内容（`TT/` 及 Chinese README）以
> 上面的项目 MIT License 为准。

## 12. 引用（Citation）

如果你在论文或报告里用到了本项目，请按此引用：

```bibtex
@misc{atc_speech_research,
  author = {CEasonK},
  title  = {ATC-Speech-Research: Air Traffic Control Speech Recognition via Fully Objective Evaluation},
  year   = {2026},
  url    = {https://github.com/CEasonK/ATC-Speech-Research}
}
```
