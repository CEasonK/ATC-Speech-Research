<a id="zh"></a>

# ATC 空管语音识别研究（ATC-Speech-Research）

**简体中文** | [English Version ↓](#en)

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

上游 FunASR 遵循其原 License（MIT）；`TT/` 研究内容为本仓库作者所有。

---

<a id="en"></a>

# English Version · ATC Speech Recognition Research (ATC-Speech-Research)

[↑ 返回简体中文](#zh)

> An air-traffic-control (ATC) speech research workspace built on top of the FunASR framework.
> Core question: under the extreme constraint of **no ground-truth transcripts and only three recordings**,
> can we deliver the full pipeline of **authoritative transcription → true streaming recognition →
> domain-grade Chinese translation**, backed by a **fully objective, reproducible evaluation system**
> (acoustic likelihood / hard ICAO·METAR·ATIS grammar constraints / cross-system verification)?
> No subjective scoring is ever used as the judge.
>
> **Status: in progress.** The deep and translate phases have shipped authoritative final results;
> the streaming phase has completed the P4 audit and frozen the L2/L1 tiers.
> **A clean re-run of L0 with the sanitized lexicon is underway** (see §10).

---

## 0. Results at a glance (real end-to-end output, not a mock-up)

```
🎧 CYYT_ATIS_a.wav (weak-channel ATIS broadcast, ~5 looped passes)
        │  deep phase: frozen by four-fold objective evidence
        ▼
SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU
WIND TWO FOUR ZERO AT FIVE
        │  translate phase: term-constrained translation + audit loop (numeric fidelity 1.0 / term hit 1.0)
        ▼
圣约翰斯 信息F 天气世界协调时零二零零
风向二四零，风速五节
```

In streaming mode the same pipeline emits incremental drafts at ~1.7 s median word
delay with automatic end-of-utterance refinement: L2 WER 0.0 / L1 0.0845
(protocol and caveats in §4, §7).

## 1. Research corpus (`TT/audio/`, read-only — no experiment may modify it)

| Recording | Content | Language | Characteristics |
|---|---|---|---|
| `CYYT_ATIS_a.wav` | St. John's (CYYT) ATIS looped broadcast, ~270 s | English | Repeats every ~28 s (~5 passes), each pass with different channel quality |
| `CYYT_ATIS_b.wav` | Same station, different time (argued to be a **different date**) | English | Much weaker signal — automatic language detection misfires as "no speech"; clipped noise tail |
| `RJTT_CONTROL.wav` | Haneda (RJTT) control-frequency comms, multiple callsigns | EN/JA | Multi-speaker, real controller–pilot interaction |

## 2. Environment & dependencies

- GPU: RTX 3090 24 GB
- Conda env: `lingbot-map` (torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 installed locally)
- Network (measured on a mainland-China machine):
  - huggingface.co unreachable → `export HF_ENDPOINT=https://hf-mirror.com`
  - pip via the Tsinghua mirror; GitHub clones need a ghproxy mirror
- The FunASR framework at the repo root is upstream code used as a dependency, unmodified. All research work lives strictly inside `TT/`.

## 3. Repository layout

```
FunASR-main/
├── funasr/                               # upstream FunASR framework (dependency, unmodified; other upstream dirs pruned)
└── TT/                                     # ★ the entire research workspace
    ├── audio/            # raw recordings (read-only)
    ├── denoise/
    │   ├── methods/      #   denoisers: NN_name.py implementing denoise(y, sr) is auto-registered
    │   ├── output/       #   outputs <rec>__m<ID>__dn<ver>.wav (version auto-increments, never overwritten)
    │   ├── qc_report/    #   QC reports (JSON for machines + MD for humans, name-matched to outputs)
    │   └── legacy/       #   quarantined legacy artifacts
    ├── results/          # production-pipeline ASR results (best_pipeline / ATC_Whisper / Qwen3ASR / FunASR)
    ├── scripts/          # production scripts (line-by-line docs in TT/scripts/README.md, zh)
    ├── models/           # model weights location (weights not in git, see §8)
    ├── REVIEW_LOG.md     # incremental code-review log
    └── research/
        ├── deep/         #   Phase 1: offline authoritative transcription
        ├── streaming/    #   Phase 2: true streaming recognition (in progress)
        ├── translate/    #   Phase 3: EN→ZH domain translation
        └── refs/         #   third-party references (SimulStreaming, etc.)
```

Every sub-project follows the same triad: `PLAN.md` (protocol & red lines) →
`JOURNAL.md` (per-round experiment log incl. negative results and errata) →
`FINAL_REPORT.md` (conclusions + evidence file manifest).

## 4. Evaluation system (the methodological core)

- **Three objective judges**:
  ① NLL acoustic likelihood (forced scoring — a hypothesis cannot "feel right" without surviving the audio);
  ② hard ICAO/METAR/ATIS grammar constraints (slot structure, NATO alphabet, number reading formats);
  ③ cross-system verification (cross-model-family consensus) + physical measurements (energy / period / anchor-window probes).
- **Prior tiers (streaming phase — prevents prior leakage from inflating metrics)**:
  - **L0 — true zero prior**: no day-specific text at all; pure acoustics + published-standard lexicon
  - **L1 — +text prompt**: ATIS text used as a decoding prompt (industry analogue: system holds the publicly broadcast text)
  - **L2 — +template fusion**: station-template evidence fusion (template text derived from the deep final drafts)
  - Every reported number must state its tier; metadata carries a self-describing `prior=` field;
    tokens coming from template output are tagged `src=tpl`.
- **Controlled K4 protocol**: token-WER computed against the corrected reference (284-token edition, J12 reference fix).
  The earlier protocol (phantom-repetition reference) is retired, with the change recorded in `JOURNAL.md`.

## 5. Quick start: model installation & first run

Weights are never committed, and **every script loads models from fixed paths** —
the Location column below must be followed exactly; "anywhere convenient" will not run.

**Prerequisites** (versions in §2):

```bash
conda create -n atc python=3.10 -y && conda activate atc
pip install -e .                # install the bundled funasr to match the research env
pip install -U huggingface_hub modelscope   # download tools
export HF_ENDPOINT=https://hf-mirror.com    # required in mainland China; skip if HF is reachable
```

**Model installation table** (Location = the path scripts read by default; wrong directories = immediate errors):

| # | Model | Download command | Location (repo-root relative) | Used by |
|---|---|---|---|---|
| 1 | whisper-large-v3-finetuned-for-ATC | `hf download jacktol/whisper-large-v3-finetuned-for-ATC --local-dir TT/models/whisper-large-v3-finetuned-for-ATC` | `TT/models/whisper-large-v3-finetuned-for-ATC/` | `run_best_asr` / `run_atc_whisper` / streaming primary engine |
| 2 | Qwen3-ASR-1.7B | `modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir TT/models/Qwen3-ASR-1.7B` | `TT/models/Qwen3-ASR-1.7B/` + **required** `export QWEN_ASR_MODEL=$PWD/TT/models/Qwen3-ASR-1.7B` | `run_qwen` / streaming side-witness worker (or pass `run_2pass.py --qwen_model`) |
| 3 | Qwen2.5-7B-Instruct | `hf download Qwen/Qwen2.5-7B-Instruct --local-dir TT/research/translate/models/qwen2.5-7b-instruct` | `TT/research/translate/models/qwen2.5-7b-instruct/` | primary translation model |
| 4 | m2m100_418M | `hf download facebook/m2m100_418M --local-dir TT/research/translate/models/m2m100_418M` | `TT/research/translate/models/m2m100_418M/` | translation baseline / back-translation |
| 5 | whisper-large-v3 (vanilla) | no manual step: `from_pretrained("openai/whisper-large-v3")` auto-fetches via mirror | `~/.cache/huggingface/` | cross-verification / streaming 2nd refiner |
| 6 | faster-whisper (CT2) builds | converted locally from #1/#5 via `TT/research/streaming/src/convert_hf_to_openai.py` | `TT/research/streaming/downloads/_ct2_atc/`, `_ct2_v3/` | streaming phase only |

**Environment variables & path dependencies**:

- `QWEN_ASR_MODEL`: overrides the legacy default path for #2 (without it, scripts point at a
  nonexistent `/siyuan/Qwen3_ASR/...` and fail immediately).
- `HF_ENDPOINT=https://hf-mirror.com`: needed by everything that touches HF (#5 and the deep reproductions).
- `TT/research/streaming/src/common.py` hard-codes `TT_ROOT` to the research machine's path —
  edit this one constant when deploying streaming reproductions elsewhere, or mirror the same path.

**Smoke test** (works as soon as #1 is installed):

```bash
cd TT && python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
# compare verbatim against the committed results/best_pipeline/CYYT_ATIS_a/result.txt
```

## 6. Production workflow (two steps)

```bash
cd TT

# ① denoise (optional — the default pipeline uses raw audio; see lessons below)
conda run -n lingbot-map python scripts/run_denoise.py            # all audios × all methods
conda run -n lingbot-map python scripts/run_denoise.py --list-methods
conda run -n lingbot-map python scripts/run_denoise.py --method 2 --audio CYYT_ATIS_a

# ② recognition (run_best_asr.py = frozen research conclusions; use this day-to-day)
conda run -n lingbot-map python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
#   --full      keep every loop pass (default: anchor-based dedup, keep the single cleanest pass)
#   --no-clean  raw decoder output only (post-processing off)
#   --anchor    change the loop anchor phrase (for other message types)

# single-model comparisons
conda run -n lingbot-map python scripts/run_atc_whisper.py --all        # 30 s sliding window + 2 s overlap
conda run -n lingbot-map python scripts/run_qwen.py CYYT_ATIS_a --lang English
conda run -n lingbot-map python scripts/qc_check.py audio/CYYT_ATIS_a.wav   # SNR / loudness / peak only
```

Outputs land in `results/<model>/<recording>/result.txt + result.json`.

**Key engineering decisions already frozen** (each backed by experiments; read `TT/scripts/README.md` before changing):
- Raw audio passes through with **no denoising**: denoising helps the noisy recording (a) but hurts the quiet one (b);
  no single parameter set wins on both. Confirmed again by streaming experiment R7 (denoiser front-end worsens WER
  by erasing weak-speech acoustic cues).
- ATIS loop dedup uses **whole-pass terminology scoring**, not per-word voting (per-word consensus fails 44% of the time).
- Hallucination filtering uses Whisper's built-in no_speech VAD (energy VAD is useless on full-energy noise segments).
- Qwen3-ASR must be given an explicit `--lang English`: auto-detection returns empty on the weak-signal recording
  (measured lesson, 2026-08-21).

## 7. The three research phases

### Phase 1 · deep — authoritative transcription without ground truth (done)

**Task**: no reference text of any kind, only three recordings — get as close to ground
truth as possible, with evidence available for every single word.

**Method core: three judges × five classes of objective tools** ("it sounds right" is never a judge):

1. **Tri-judge system**: whisper-atc (strong domain prior, self-biased) / whisper-large-v3
   vanilla (neutral) / turbo-atcosim (third independent engine). Iron rule: **any field is
   frozen only with ≥2 independent evidence sources agreeing; a single judge never decides**.
2. **Paired-window adjudication**: competing hypotheses are forced-NLL scored inside the
   *same* anchor window, removing window-drift artifacts (early trap: scoring text against a
   silent window yields deceptively low NLL — the "silent-window illusion").
3. **LM-prior contamination calibration**: measured that single-word-insertion ΔNLL can reach
   1.2–1.4 nat purely from decoder language priors — within that band ΔNLL **must not decide
   alone** (case v11 WIND, Δ1.27, triggered exactly this rule).
4. **Slice decoding**: cut 14–16 s context-free snippets for independent free dictation by all
   engines, breaking long-audio context anchoring. Key finding: **unanimous absence ≠ acoustic
   absence** — reduced /ət/ and the nasal /wɪnd/ were missed 11/11 and 7/7 yet physically
   present → free-decode voting is positive evidence of presence only, never of absence.
5. **Energy-envelope physical probe (the tie-breaker)**: when NLL and decoding deadlock, a
   language-free 10 ms RMS envelope + burst detection gives the final ruling. Both hard cases
   were decided by it — a's reduced AT (voiced run at 57.88 s, RMS 0.06–0.14 vs 0.025–0.04 in
   genuine inter-word gaps) and b's WIND (nasal-plateau signature + 7 bursts mapped word by word).
6. **Grammar vetoes acoustics**: METAR/ATIS hard constraints (integer temperature, VHF
   118–137 MHz, QNH 28.xx format) outright reject "best-sounding" hypotheses that are illegal.

**Representative wins** (each reproducible via `exp/adjudicate_v*.py`):
- **The v4 retrial**: callsign SIERRA→SHANGHAI AIR (v3 free decode hit it independently in two
  segments + qwen paired-window agreement + domain prior — three sources; turbo's SIERRA kept
  on file as the rival hypothesis); ORANGE NINER→ORANGE LINER (Japanese-track
  オレンジライナー + qwen, two sources).
- **a/b relationship**: not re-recordings but broadcasts from **different dates** — the whole
  corpus differs in exactly two spots (QNH 3023 vs 3033; AS vs WHEN REQUESTED), both channel-level.
- **Trailing readback**: human listening + anchor-window physical probe confirm the last three
  lines of a are a genuine repetition, not model hallucination.

**Deliverables**: `results/a_final.txt`, `b_final.txt`, `rjtt_final.txt` (RJTT synthesized from
9 consensus segments with per-segment confidence tiers) — the reference standard for all
streaming / translate research. Full evidence chain in `research/deep/FINAL_REPORT.md`, Appendix A.

### Phase 2 · streaming — true streaming recognition (in progress)
- **Architecture**: SimulStreaming (AlignAtt) streaming engine on the ATC-finetuned whisper-large-v3 backbone;
  an RMS-CV modulation gate triggers offline refinement (beam=5) at utterance end;
  station-template evidence fusion (ATC CT2 primary refiner + Qwen3-ASR side-witness worker +
  low-evidence-rate guardrail); delivered 2-pass.
- **Current numbers** (K4 protocol):
  | Tier | a | b | Notes |
  |---|---|---|---|
  | L2 template fusion | 0.0 | 0.0 | word-for-word match with deep finals; caveat: template prior included, 35–39 % of output tokens tagged `src=tpl` |
  | L1 text prompt | 0.0845 | 0.1162 | template used as a text prompt (not zero-prior) |
  | L0 3-engine ROVER | ⚠ 0.1303 / 0.2711 → **retired, pending re-run** | | see the P4 erratum in §9 |
  - Latency: draft-track median word delay ~1.7–1.9 s (RTF ~0.5, meets the real-time constraint); final track ~14.5 s.
- **Negative results archived** (equally conclusions): a single side-witness — especially the same-family whisper-v3 —
  can drag the primary engine down; the m1 denoiser front-end worsens WER; the cross-period-consensus route does not hold;
  Qwen warm-up must happen at worker startup (first call costs 32 s).

### Phase 3 · translate — EN→ZH domain translation (done; quality tied to the input tier)
- **Architecture**: T3 rule-template translator (0 parameters; numeric-correctness judge + fallback) +
  T1 Qwen2.5-7B term-constrained translation (primary generator) + audit feedback loop
  (failing lines are re-fed with error reasons, ≤3 rounds); T4 M2M-100 as an independent baseline and zh→en back-translation.
- **Metrics**: numeric fidelity (English-spelled numbers → payload sequences must match exactly) and
  terminology hit rate (21 EN→ZH term pairs). Final drafts score **1.0 / 1.0** when fed the deep finals or L2 streaming output.
- **Established**: ATC translation is not free MT — it is a "terminology mapping + number-reading restoration" problem
  (bare M2M failure modes archived: WIND→"win", ALTIMETER→"maximum").
- **Known boundary**: feeding zero-prior ASR output end-to-end drops metrics sharply
  (a: numeric 0.359 / terms 0.833; b worse) — end-to-end quality depends on the prior tier.

## 8. Model dependency overview

Download commands, exact locations and environment variables for all 6 model artifacts
are in the §5 installation table (weights are not in git).
One-liner: ATC-finetuned whisper (primary ASR) · whisper-large-v3 vanilla (cross-verification) ·
Qwen3-ASR (side witness) · Qwen2.5-7B (translation) · m2m100 (back-translation baseline) ·
SimulStreaming (streaming engine — code bundled in this repo).

## 9. Reproduction guide

1. **Production pipeline**: commands in §6 run out of the box (after fetching the weights in §8).
2. **deep finals**: follow the protocol in `research/deep/PLAN.md`, running `src/` scripts in order;
   diff every step's output against `results/*_final.txt` verbatim.
3. **streaming, all tiers**: `research/streaming/src/run_2pass.py <wav> <out> --chunk 1.0 --half --rover --no_prompt`
   for the L0 zero-prior tier; drop `--no_prompt` for L1; add template fusion for L2.
   Scoring always goes through `src/metrics.py` (K4 protocol).
4. **translate**: `research/translate/` ships standalone audit scripts that recompute numeric fidelity and
   terminology hit rate for any translation.
5. Doubt any number? Replay it round-by-round through `JOURNAL.md` (protocol changes and errata are all on record).

## 10. Current progress & TODO (2026-08-31)

- ✅ deep: authoritative finals for all three recordings, evidence chain closed
- ✅ translate: finals at 1.0/1.0; audit tooling independently reusable
- ✅ streaming: L1/L2 frozen; P1–P3 fixes + regression tests (`tests/test_p3_fixes.py`)
- 🔶 **streaming P4 audit erratum (2026-08-31)**: audit found that the R1–R9 lexicon/grammar rules had been
  selected by analyzing residual errors against the K4 evaluation-set ground truth (adaptation supervised by the gold
  reference). **L0 numbers 0.1303/0.2711/0.0211/0.2254 are all retired.** `atis_lexicon.py` has been sanitized to
  "derivable from published standards + never deletes engine tokens".
  **TODO: re-run a clean L0 with the sanitized rules and re-report** (both a/b tracks, full-file delivery and
  end-to-end translation linkage).
- ⬜ Planned but not yet landed: Kyutai STT and NeMo FastConformer streaming baselines (routes E3/E4 in the streaming PLAN)
- ⬜ Weak-channel robustness: residual errors on track b (period-seam misalignment, degraded-segment numeric collapse)
  remain open — an acoustic frontiers problem

## 11. License

Upstream FunASR remains under its original MIT License; the research content under `TT/` is owned by the repository author.
