"""客观指标基础件：Levenshtein 距离与 token-WER。

裁判纪律（继承 deep）：只用客观可复现指标，不用主观评分。
注：LAG(τ)/token 延迟/定稿回退的高层封装已统一收敛到 evaluate_run.py 内联实现
（原先此处留有第二份实现，从未被调用，为防双源漂移已删除）。
"""
from common import tokens


def levenshtein(a, b):
    """标准 DP 距离。"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def wer(hyp, ref):
    h, r = tokens(hyp), tokens(ref)
    if not r:
        return 0.0
    return levenshtein(h, r) / len(r)
