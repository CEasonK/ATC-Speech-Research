# 流式语音识别研究 · 详细规划（streaming/）

> **【状态 2026-08-27】本计划已执行完毕** → 结论见 `FINAL_REPORT.md`，过程日志见 `JOURNAL.md`。
> 主要偏差：原始音频实为 ~270s 多周期长录音（非 120s）；Kyutai/NeMo 对照路线未落地；
> 新增"跨周期共识"探索实验（negative result）。

> 目标：对 CYYT_ATIS_a / CYYT_ATIS_b 两条英文 ATIS 广播音频做**真流式（chunk 喂入、增量出字）识别**，
> 达到"快（延迟低）+ 准（与 deep 权威终稿对比 WER 低）"，并与 offline 上限做对比分析。
> 红线：只允许在 `research/streaming/` 内写文件；`audio/`、`deep/`、`results/`（TT 级）只读。

---

## 0. 背景与已知事实

1. **权威对照文本**（deep 项目四重证据定稿）：
   - `deep/results/a_final.txt`（13 行，含尾部复诵）、`deep/results/b_final.txt`
   - a 与 b 仅 2 处信道级差异：`ALTIMETER THREE ZERO TWO THREE`（a）vs `THREE ZERO THREE THREE`（b）；
     `AS REQUESTED`（a）vs `WHEN REQUESTED`（b）
2. **音频**：`audio/CYYT_ATIS_a.wav` ≈120s、`b.wav` ≈120s，16kHz 单声道，ATIS 循环广播
   （a 信道周期 28.143s，约 4 遍重复）。音频末尾 ~137s 后为削波噪声段（b）。
3. **环境**：conda `lingbot-map`（torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 本地安装），
   RTX 3090 24GB。
4. **网络**：huggingface.co 直连不通；**hf-mirror.com 可达**（`HF_ENDPOINT=https://hf-mirror.com`）；
   pip 走清华源；GitHub 需 ghproxy 代理。

## 1. 候选方案（调研结论，2024-2026 SOTA）

| # | 方案 | 架构/原理 | 延迟 | 英文精度预期 | 备注 |
|---|------|-----------|------|--------------|------|
| E1 | **SimulStreaming (ufal) + whisper-large-v3** | AlignAtt 策略：用 cross-attention 位置引导解码到"音频末端 - frame_threshold"处停笔，逐 chunk 增量 | 低（0.5-1s 级） | 接近 offline whisper | IWSLT2025 冠军方案；代码已在 `refs/SimulStreaming-main`；依赖环境全有 |
| E2 | **SimulStreaming + ATC 微调 whisper**（TT/models 下 jacktol 微调版） | 同上，但模型域先验强 | 同 E1 | ATC 域可能更强（对 "WIND TWO FOUR ZERO AT FIVE" 这类弱读词） | 需确认对齐 head 兼容性（同架构，应兼容） |
| E3 | **Kyutai STT 1B/2.6B**（delayed streams modeling） | 原生流式：Mimi codec 12.5Hz + 文本流恒定延迟 2.5s 的 decoder-only LM，词级时间戳免费 | 恒定 2.5s | OpenASR 榜 SOTA（2.6B 6.4%）；但英文域、ATC 数字串是其已知痛点（issue #152 数字误识别） | 模型小（1B 2.4GB / 2.6B 16.8GB），3090 轻松跑；需装 moshi 包或手写推理 |
| E4 | **NeMo FastConformer-Hybrid 流式**（stt_en_..._streaming_multi，80/480/1040ms lookahead） | cache-aware streaming conformer + CTC/RNNT，原生 chunk 流式 | 0.08-1s | 通用英文好，ATC 数字弱 | 需装 nemo（重，~1.5GB 依赖），从 hf-mirror 下 nemo 权重 |
| E5 | **FunASR 流式（英文）** | Paraformer-streaming 主做中文；英文无强流式模型，仅 SenseVoice 离线 | - | 英文弱 | 作为 baseline/对照组跑通"2-pass 工业范式"演示即可 |
| E6 | 2-pass 混合方案（**推荐最终形态**） | Pass1: E1/E3 流式出草稿 → 端点后 Pass2: offline 大模型（whisper-large-v3 / ATC 微调）对该句精修 | 首字=流式延迟；终句+0.3~1s | 终句≈offline 上限 | 工业界标准做法（FunASR 官方博客/NeMo 均推荐），"又快又准"的最优解 |

**策略**：E1、E3 为主攻（各自代表"离线模型流式化"与"原生流式"两条技术路线），
E4 作为低延迟对照组（若环境安装顺利），E5 快速跑作工业范式对照，E6 做最终交付形态。

## 2. 评测框架（`src/`，所有引擎共用同一套裁判）

```
src/
├── stream_sim.py      # 模拟实时流式播放器：按 chunk（默认 0.5s，可 0.1/0.25/0.5/1.0）从 wav 切片喂入，
│                      #   记录每个 token 的"说话时刻"与"产出时刻"，产出事件流 JSON
├── run_simulstreaming.py  # E1/E2 引擎封装（import refs/SimulStreaming-main）
├── run_kyutai.py          # E3 引擎封装
├── run_nemo_stream.py     # E4 引擎封装（可选）
├── run_funasr_stream.py   # E5 引擎封装
├── metrics.py       # 客观指标（红线：不用主观评分）：
│                      #   ① WER（jiwer，对照 a_final/b_final；ATIS 数字读法归一：保留字母数字读法原样比）
│                      #   ② 延迟指标：
│                      #      - LAG(τ) 经典：在音频时刻 τ，已输出前缀与参考的 Levenshtein 对齐距离均值
│                      #      - first-token latency：首字产出时刻 - 首字发音时刻（能量包络定首字起点）
│                      #      - token 延迟中位数/95 分位：token 产出时刻 - 该 token 对齐到参考的发音时刻
│                      #        （发音时刻用 offline whisper-large-v3 的 word timestamps 对齐，客观）
│                      #   ③ RTF：总推理墙钟 / 音频时长（模拟单流，3090）
│                      #   ④ 稳定性：逐 chunk 输出的"已定稿前缀回退"次数（流式质量关键体验指标）
├── align_ref.py     # 用 offline whisper-large-v3 word-level timestamps + difflib 对齐参考文本，
│                    #   得到每个参考 token 的 [start,end]（供 token 延迟计算）
└── common.py        # 音频加载、16k 重采样、归一化、结果目录约定
```

**对照文本处理**：a_final 含尾部复诵 3 行（deep 已验证真实存在），整篇对照；
WER 同时报"全篇"与"去重复首遍"两个口径，避免重复段稀释指标。

## 3. 实验矩阵（`exp/`，每个实验一个脚本+固定 seed/参数，输出进 `results/<engine>/<audio>/`）

| 实验 | 引擎 | 参数扫描 |
|------|------|----------|
| s1 | SimulStreaming + large-v3 | frame_threshold ∈ {10,25,50,100}，chunk_stride ∈ {0.1,0.5}，beams=1/5 |
| s2 | SimulStreaming + ATC 微调 | 同 s1 最优参数 |
| s3 | Kyutai 1B | 默认（2.5s 延迟不可调）+ prompt 注入 ATIS 术语（"SAINT JOHNS INFORMATION ... WIND ... VISIBILITY ..."） |
| s4 | Kyutai 2.6B | 同 s3（若磁盘/速度允许） |
| s5 | NeMo streaming multi | lookahead ∈ {80,480,1040}ms |
| s6 | FunASR streaming（中文模型对英文，仅证伪/对照） | 默认 |
| s7 | **2-pass 最终方案**：s1/s2 最优流式 + offline 精修（端点=VAD 静音>0.8s） | 端点参数扫描 |

每个实验产出：
- `events.jsonl`：逐 token 事件（time_spoken, time_emitted, text, is_final）
- `transcript_final.txt`：流式定稿全文
- `metrics.json`：WER / LAG / token 延迟 / RTF / 回退次数
- `timeline.md`：人读版时间线（每 10s 快照当前已输出文本）

## 4. 优化手段（按预期收益排序）

1. **术语 prompt**：SimulStreaming 支持 `--init_prompt/--static_init_prompt`，
   注入 ATIS 词汇表（ICAO 数字读法 + ATIS 固定句式），CUNI 论文证实 prompt 注入域术语显著提质。
2. **ATC 微调模型**：域先验对弱读词（WIND/AT/DEW POINT）帮助大——deep 项目已反复证实弱读词是错误源。
3. **frame_threshold 调优**：阈值小=延迟低但边界词易错；阈值大=更准但延迟升。在两条音频上扫。
4. **2-pass 精修**：终句质量直接对齐 offline 上限，只增加句级延迟，不伤首字延迟。
5. **Kyutai 数字问题缓解**：prompt 中显式给 "spell out numbers one by one" 类指令（issue #152 的社区解法）。

## 5. 验收标准（"效果好"的客观定义）

- 流式终稿 WER ≤ offline whisper-large-v3 WER + 2.5pt（即流式代价可接受）；
- 若 2-pass 方案：终句 WER ≤ offline + 0.5pt；
- 首字延迟 ≤ 1.5s（流式引擎），token 延迟 p95 ≤ 2.5s；
- RTF < 0.3（3090 单流）；定稿回退 ≤ 3 次/分钟音频。
- 若某项达不到，在 JOURNAL 里记录原因与已尝试的缓解。

## 6. 执行顺序

1. 写 `src/common.py` + `src/metrics.py` + `src/stream_sim.py` + `src/align_ref.py`（半天内完成框架）
2. 跑 offline whisper-large-v3 出 word timestamps（参考对齐 + offline WER 基线）
3. E1（SimulStreaming+large-v3）跑通 → 参数扫描
4. E2（ATC 微调）→ 参数扫描
5. 下 Kyutai 1B → E3（2.6B 视速度决定）
6. E4/E5 快速对照
7. E6 2-pass 集成 + 端点调优
8. 汇总 `results/leaderboard.md`，写 JOURNAL/FINAL_REPORT
