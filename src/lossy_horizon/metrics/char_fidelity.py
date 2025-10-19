from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def char_fidelity(ref: str, hyp: str) -> float:
    """Character-level fidelity = 1 - CER (normalized by max length)."""
    ed = levenshtein(ref, hyp)
    denom = max(1, max(len(ref), len(hyp)))
    return 1.0 - (ed / denom)
