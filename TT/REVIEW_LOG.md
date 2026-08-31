# 代码审查日志（增量记录）

审查对象：`TT/scripts/` 正式脚本 + `denoise/methods/` + `research/scratch/postprocess.py`
审查维度：① 逻辑结构 ② 规范性 ③ 性能 ④ 错误处理/边界 ⑤ 安全 ⑥ 可维护性
语法基线：`python -m py_compile` 全部 7 个文件通过 ✅

---

## 2026-08-21 · 第 1 轮（全量首审）

### 🔴 需修（影响正确性/明显性能）

| # | 文件:行 | 维度 | 问题 | 建议 |
|---|---------|------|------|------|
| 1 | `run_denoise.py:90-93` | ①逻辑 | **`--audio` 报错 bug**：过滤后 `audio_files` 已被赋值为空列表，错误消息 `[p.stem for p in audio_files]` 打印空 `[]`，而非可用音频名 | 报错时打印过滤前的候选列表 |
| 2 | `run_best_asr.py` `decode_pipeline` | ③性能 | 模型 `from_pretrained` 在函数内，`main` 对多音频逐条调用 → **模型每文件重载一次**（函数 docstring 已自注"理想应外提复用"） | main 里加载一次 model/pipe，传入处理函数 |
| 3 | `run_qwen.py:71` | ④边界 | `results[0]` 未判空，`transcribe` 返回空列表时 IndexError | 加 `if not results` 保护 |
| 4 | `run_denoise.py:108` | ③性能 | `load_method(path)` 在 音频×方法 双层循环内，同一方法文件被反复 `exec_module` | 方法模块加载一次，循环内复用 |
| 5 | `run_denoise.py:188-193` | ③性能 | `_compute_qc` 对 `stats(orig)`/`stats(denoised)` 各调用 3 次，重复算 rms/peak | 各算一次存局部变量 |

### 🟡 建议（规范性/可维护性）

| # | 文件:行 | 维度 | 问题 | 建议 |
|---|---------|------|------|------|
| 6 | `run_qwen.py:21` | ⑥可维护 | `MODEL_PATH` 硬编码绝对路径 `/siyuan/Qwen3_ASR/...`，换机即失效 | 改相对 `TT_ROOT` 或环境变量 |
| 7 | `run_qwen.py:27` | ①逻辑 | 默认 `LANGUAGE=None`（自动检测）正是 CYYT_b 空输出根因（README 已警告但默认值仍不安全） | ATC 场景默认 `English` 或首次运行告警 |
| 8 | `run_qwen.py:80-81` / `run_atc_whisper.py:105-106` | ②规范 | `import json`/`datetime` 写在循环体内 | 提到模块顶部 |
| 9 | `run_atc_whisper.py:15` | ②规范 | `import os` 未使用 | 删除 |
| 10 | `run_best_asr.py:172-173` | ③性能 | `split_instances` 列表推导里 `" ".join(words[p:p+msg_len])` 每锚点算两次 | 先算 seg 再打分 |
| 11 | `postprocess.py:4-11` | ②规范 | 顶部 docstring 仍写"重复段压缩"，但 v2 起已不压缩（L88 注释才说明） | 更新 docstring |
| 12 | `postprocess.py:74` | ⑥一致 | `KNOWN_BAD` 含 DISABILITY，与正式脚本把 DISABILITY 映射回 VISIBILITY 矛盾 | scratch 属研究过程，标注即可 |
| 13 | `run_denoise.py:139-150` | ④边界 | `_load_audio` librosa 分支重采样 16k，soundfile fallback 不重采样 → 两路径 sr 可能不一致 | 统一 resample |

### ⑤ 安全维度
- 无网络调用、无凭据、无注入面（纯本地推理+文件 IO）。
- `run_denoise.py` 动态 `importlib` 加载 `methods/*.py` —— 仅限本目录白名单文件，可接受；若后续开放用户投放方法文件，需注意任意代码执行面（当前不涉及）。

### ✅ 做得好的地方
- `run_best_asr.py` 结构清晰：clean/score/split/decode/process_one 职责分离；单条失败 `continue` 不阻断批量；防误切保护（锚点 <2 或最短退化为完整转写）。
- `qc_check.py` 简洁，`range(0, max(1, len-frame), hop)` 保证 rms_list ≥1 元素，percentile 不会空列表报错。
- `01_noisereduce.py` 接口契约（`denoise`/`DESC`/`DUMP_CONFIG`）清晰，参数快照可追溯。
- 各脚本 `extract_audio_name` 统一处理降噪产物 `__m__dn__` 命名，结果按音频分子目录不互扰。

### 本轮结论
- 语法/安全无 BLOCKER。
- 1 个真实逻辑 bug（#1 报错信息）、1 个明显性能项（#2 模型重载）、1 个边界隐患（#3 空结果）。
- 其余为规范性/可维护性建议，非必须。

### 修复记录（2026-08-21 同日完成）
| # | 状态 | 说明 |
|---|------|------|
| 1 | ✅ 已修 | `run_denoise.py` 报错先存 `all_stems` 再过滤，提示信息正确 |
| 2 | ✅ 已修 | `run_best_asr.py` 拆出 `load_engine()` 返回 decode 闭包，main 只加载一次模型（验证：两条音频共享，8s 加载） |
| 3 | ✅ 已修 | `run_qwen.py` `results` 空保护：写空文件 + 提示用 `--lang English`，不再 IndexError |
| 4 | ✅ 已修 | `run_denoise.py` 方法模块 `loaded` 只加载一次，循环复用 |
| 5 | ✅ 已修 | `run_denoise.py` `stats()` 每输入只算一次 |
| 6 | ✅ 已修 | `run_qwen.py` `MODEL_PATH` 支持 `QWEN_ASR_MODEL` 环境变量覆盖 |
| 7 | 🟡 保留 | `LANGUAGE=None` 默认值保留（用户要"自动检测"能力），README+空结果提示双重告警 |
| 8 | ✅ 已修 | `run_qwen.py`/`run_atc_whisper.py` 循环内 import 提至模块顶 |
| 9 | ✅ 已修 | `run_atc_whisper.py` 删除未用的 `import os` |
| 10 | ✅ 已修 | `split_instances` 先算 seg 再打分，去重复 join |
| 11-13 | ⏸ 不修 | scratch docstring 过时 / KNOWN_BAD 矛盾（研究过程文件，标注即可）/ sr 一致性（wav 路径下不触发） |

**回归验证**：5 个正式脚本 `py_compile` 通过；`run_best_asr.py` 重构后重跑两条音频
结果与修复前逐字节一致（a 75 词单条 / b 88 词完整）。

---
（后续审查在此文件下方增量追加）
