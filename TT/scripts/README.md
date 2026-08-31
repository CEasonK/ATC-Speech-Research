# 脚本说明（scripts/）

这一页讲清楚每个脚本**是怎么实现的、你能改哪里**。目的：不看我解释也能自己明白、自己调。

---

## 总览

| 脚本 | 干什么 | 读哪 | 写哪 |
|---|---|---|---|
| `run_denoise.py` | 对 audio 降噪 | `audio/` | `denoise/output/` + `denoise/qc_report/` |
| `run_atc_whisper.py` | ATC-Whisper 识别 | 指定的音频 | `results/ATC_Whisper/<音频>/result.txt` + `result.json` |
| `run_qwen.py` | Qwen3-ASR 识别 | 指定的音频 | `results/Qwen3ASR/<音频>/result.txt` + `result.json` |
| `qc_check.py` | 单看一段音频质量 | 指定的音频 | 可选 `denoise/qc_report/`（原始录音进 `raw/` 子目录） |

---

## 1. run_denoise.py（降噪）

### 流程（代码怎么走的）
1. 扫描 `denoise/methods/` 下所有 `NN_方法名.py` 方法文件，编号即文件名前缀
2. 扫描 `audio/` 下所有 `.wav`
3. 对每段音频 × 每个方法
   - 读音频 → 交给方法里的 `denoise(y, sr)` 得到降噪后音频
   - 算出版本号（`dn1/dn2…`，自动递增不覆盖）
   - 存 wav → 存 `qc_report`：一个 json（机器读，含 `config` 参数快照）+ 一个 md（人读）
4. 版本号算法 `_next_version()`：数 `output/` 里已存在的最大 `dnN`，+1。所以 **重复跑永不覆盖旧版**。
5. 每个方法文件可提供 `DUMP_CONFIG()` 返回参数字典，会写进对应 json 的 `config` 字段，方便追溯这版降噪用的参数。

### 你能改什么
- **加新降噪方法**：复制 `denoise/methods/01_noisereduce.py` 改成 `02_xxx.py`，改里面的 `denoise()` 和参数即可。多个方法自动并存。
- **改降噪强度/参数**：进具体方法文件。
- **只跑某个方法**：`--method 2`；**只处理某段**：`--audio CYYT_ATIS_a`
- **手动指定版本**：`--version 3`（想覆盖时用）

---

## 2. run_atc_whisper.py（ATC-Whisper 识别）

### 原理 / 代码怎么走的
- 加载 `models/whisper-large-v3-finetuned-for-ATC`（float16 半精度，省显存）
- 长音频**滑窗**处理：Whisper 一次最多处理 30 秒，所以脚本按 `30 秒窗口 + 2 秒重叠` 把长音频切成一段段，每段单独识别，再按顺序拼起来。重叠是为了避免句子正好被切断在窗口边缘。
- 结果存到 `results/ATC_Whisper/<音频>/result.txt`（纯文本）+ `result.json`（含模型/语言/时间/文本）
- **语言默认自动检测**（`language=None`），中英日混着跑都对；要强制某语言用 `--lang en` / `--lang ja`

### 你能改什么
- `WINDOW` / `OVERLAP`（顶部常量）：窗口秒数 / 重叠秒数。若你发现句子老是被切断，可加大 `OVERLAP`。
- `--lang`：手动指定语言（默认自动检测）。
- 打开更多按句输出：目前是纯文本拼接。想要带时间戳/分句，可以在这里再加。

---

## 3. run_qwen.py（Qwen 识别）

### 原理 / 代码怎么走的
- 加载 `/siyuan/Qwen3_ASR/models/Qwen3-ASR-1.7B`（bfloat16）
- 调用 `model.transcribe(audio=..., language=None)`（**默认自动检测语种**），Qwen 内部自动处理长音频
- 结果存到 `results/Qwen3ASR/<音频>/result.txt`（纯文本）+ `result.json`（含模型/语言/时间/文本）

### 你能改什么
- `--lang`：手动指定语言（**ATC 场景强烈建议显式 `--lang English`**）。
  ⚠ 重要教训（2026-08-21 研究实测）：弱信号音频（如 CYYT_ATIS_b）用默认
  自动检测（`language=None`）会被误判"无语音"返回**空**；显式指定
  `--lang English` 后正常出完整结果。已知语言的场景永远显式指定。
- 想加空管术语热词/prompt：Qwen 的 `transcribe` 支持额外参数，可后续扩展

---

## 4. qc_check.py（单段音频质量检查）

- 只算**一段音频**的底噪/响度/峰值/SNR，不识别
- 传文件路径即可：`python qc_check.py audio/CYYT_ATIS_a.wav`
- 加 `--save` 会把结果存一份：原始录音进 `denoise/qc_report/raw/`，降噪产物（文件名含 `__m..__dn..`）进 `denoise/qc_report/`，两者分开不混淆
- 作用和降噪脚本里生成的 qc_report 相同，只是它只对"已有文件"检查，不重新降噪

---

## 5. run_best_asr.py（最优管线 v3，日常用这个）

2026-08-21 研究结论固化（完整推导见 `research/best/README.md`）：

1. **原始音频直通**——不存在对两类音频都有用的降噪参数（a 有噪声降噪
   有效，b 太安静降噪有害），所以不做降噪
2. **ATC-Whisper pipeline 解码**：30s **非重叠**分块 + 内置 no_speech VAD
   （滤掉噪声段的 "Thank you"/"BEEP" 幻觉，能量 VAD 对满能量噪声无效）
3. **循环去重**（关键）：ATIS 是循环广播，CYYT_a 物理上每 28.6s 播一遍
   （mel 自相关 corr=0.625 证实）。按 `WIND TWO FOUR ZERO` 锚点切出各遍
   实例 → **术语打分选最干净的一遍**（5 遍乱码形态不同，逐词投票共识仅
   44% 失败，改用整遍打分）→ 只输出 1 条
4. **后处理 clean()**：含幻觉短语的分句整句删 + 术语纠错（SANDURK→GANDER
   等）+ 单字乱码先修复（DISABILITY→VISIBILITY，避免被误当幻觉删）

用法：
```bash
python run_best_asr.py audio/CYYT_ATIS_a.wav        # 默认: 去重留 1 条报文
python run_best_asr.py audio/CYYT_ATIS_a.wav --full # 保留全部循环
python run_best_asr.py audio/xxx.wav --no-clean     # 只要原始输出
python run_best_asr.py audio/xxx.wav --anchor "WIND TWO FOUR"  # 改循环锚点
```
非循环音频会自动检测（锚点实例 <2 或最短）→ 退化为完整转写，不会误切。

### 你能改什么
- `--anchor`：循环锚点短语（换报文类型时改这里）
- `PHRASE_HALLUC`：新音频出现新静音幻觉时，把幻语句加进去（整句删，
  不要写成跨句的正则）
- `GOOD_TERMS`/`BAD_TERMS`：术语打分词表（决定选哪一遍循环）
- `TERM_FIXES`：发现新术语误识时加一条映射
- `CHUNK_S`/`N_SPPEECH_THR`：分块长度、无语音阈值

---

## 关于"指标"怎么看（帮你判断降噪是不是成功了）

核心就盯 **SNR（信噪比）**：

| SNR | 含义 |
|---|---|
| > 25 dB | 优秀，语音明显盖过噪声，识别应该很准 |
| 15~25 dB | 中等，基本可用 |
| < 15 dB | 很低，语音被噪声淹没，识别会大量出错 |

**降噪成功的三条标准（同时满足）**
1. 降噪后 SNR **至少比降噪前高 5 dB**
2. 语音整体响度（RMS）**没被过度削弱**（别掉个位数 dB 以上）
3. **不削波**（峰值别接近 0 dB）、**不出现失真/听不清**

**注意一点**：SNR 是"整体"指标，它不区分"降噪后语音变尖还是变闷"。所以数值好看只是第一步，**最终还是要靠实际识别结果和听感**判断——这正是你接下来要做的（拿去识别对比）。