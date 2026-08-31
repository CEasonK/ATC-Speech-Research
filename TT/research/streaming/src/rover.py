"""先验无关多引擎 ROVER：把旁证引擎的 token 序列对齐到主引擎，逐位置投票。

与模板证词融合不同：这里没有任何答案文本参与，只有引擎间共识（L0/L1 兼容）。
对齐用 SequenceMatcher；每个主引擎位置收齐旁证票后取最高票，平票取主引擎。
返回 (tokens, words, stats)：words 与主引擎对齐（旁证胜出位置计时取 None，
由上层插值）。
"""
from difflib import SequenceMatcher


def rover(primary_toks, primary_words, secondary):
    """secondary: [(tokens, words|None), ...]"""
    votes = [[] for _ in primary_toks]      # 每位置旁证票
    sec_extra = []                          # 旁证独有 token（主引擎缺失段）

    for stoks, swords in secondary:
        if not stoks:
            continue
        sm = SequenceMatcher(None, primary_toks, stoks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                for o in range(min(i2 - i1, j2 - j1)):
                    if stoks[j1 + o] != primary_toks[i1 + o]:
                        votes[i1 + o].append(stoks[j1 + o])
            elif tag == "insert":
                sec_extra.append((i1, stoks[j1:j2]))

    out_toks, out_words, n_win = [], [], 0
    for i, tok in enumerate(primary_toks):
        w = primary_words[i] if primary_words and i < len(primary_words) else None
        if votes[i]:
            best = max(set(votes[i]), key=votes[i].count)
            # 覆盖主引擎一律需 >=2 票（R2b 教训：单旁证覆盖会让同源引擎的
            # 相关错误反噬主引擎；旁证不足 2 票时信任主引擎）
            if votes[i].count(best) >= 2:
                out_toks.append(best)
                out_words.append(None)      # 旁证词无计时，交上层插值
                n_win += 1
                continue
        out_toks.append(tok)
        out_words.append(w)
    return out_toks, out_words, {
        "engines": 1 + len(secondary), "rover_wins": n_win,
        "sec_extra": sum(len(t) for _, t in sec_extra)}
