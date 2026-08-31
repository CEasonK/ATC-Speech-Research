# 降噪方法（denoise/methods/）

这个目录存放所有降噪方法。**每个方法一个 `.py` 文件**，多个方法并存可以随意对比。

## 命名规则
文件名必须是 `NN_方法名.py`，其中 `NN` 是两位编号（如 `01`、`02`）。

## 方法文件怎么写（模板）

每个方法文件只需实现一个 `denoise(y, sr)` 函数，可选提供 `DESC` 描述和 `DUMP_CONFIG()` 参数快照。

```python
# 02_spectral.py
import numpy as np

DESC = "简单谱减法"   # 可选：对方法的描述

def denoise(y, sr):
    """输入 y: 音频数组, sr: 采样率; 返回降噪后的音频数组"""
    audio = np.asarray(y, dtype=np.float64)
    # ... 在这里写你的降噪逻辑 ...
    return audio

def DUMP_CONFIG():
    """可选：返回参数字典，会写入 qc_report 便于追溯"""
    return {"method": 2, "desc": DESC, "cutoff_hz": 200}
```

## 现有方法
| 编号 | 文件 | 说明 |
|---|---|---|
| 01 | `01_noisereduce.py` | 高通滤波(300Hz) + noisereduce 频谱门限降噪 |

## 怎么跑
```bash
cd /siyuan/FunASR_extracted/FunASR-main/TT
conda run -n lingbot-map python scripts/run_denoise.py --list-methods   # 看有哪些方法
conda run -n lingbot-map python scripts/run_denoise.py                  # 全部音频、全部方法
conda run -n lingbot-map python scripts/run_denoise.py --method 2       # 只跑 2 号方法
```