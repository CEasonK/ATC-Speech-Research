# 模型资产盘点（P1 交付，收尾时复核）

## ASR 引擎（本研究实际使用）
| 资产 | 路径 | 角色 |
|---|---|---|
| whisper-large-v3 (vanilla) | HF 缓存 `models--openai--whisper-large-v3` | 中立裁判 + 自由解码 |
| whisper-large-v3-finetuned-for-ATC | 本地 `TT/models/whisper-large-v3-finetuned-for-ATC`（同 HF 缓存 jacktol） | ATIS 域裁判 |
| whisper-large-v3-turbo-atcosim | HF 缓存 `models--tclin--whisper-large-v3-turbo-atcosim-finetune` | 第三引擎（atcosim 域先验） |
| whisper-large-v3-atco2 | HF 缓存 `models--jlvdoorn--whisper-large-v3-atco2-asr-atcosim` | 备用（未启用） |
| Qwen3-ASR | `/siyuan/Qwen3_ASR`（lingbot-map python 运行） | 独立架构第二意见 |

## 未启用的备选资产
- `/siyuan/ATC_whisper_model`、`/siyuan/BAAI`：调研阶段候选，未进入管线
- FunASR 主仓库模型：与 whisper 系架构差异大，早期对比后未纳入三裁判制

## 音频与切片
- 原始：`TT/audio/{CYYT_ATIS_a,b,RJTT_CONTROL}.wav`
- 切片：`segments/RJTT_CONTROL/seg_00..08.wav`（VAD+manifest）；`results/wind_slices/v12/`（15s 重切片×3中心×信道）

## 运行环境
- conda env `lingbot-map`；`HF_HUB_OFFLINE=1` 全程离线；CUDA 单卡 fp16
- 依赖核心：torch/transformers/librosa/soundfile（谱减法审计未新增依赖）
