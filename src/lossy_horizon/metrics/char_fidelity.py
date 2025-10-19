from __future__ import annotations


_fast_distance = None
try:
    from rapidfuzz.distance import Levenshtein as _RFLev  # type: ignore

    def _fast_distance(a: str, b: str) -> int:  # type: ignore
        return int(_RFLev.distance(a, b))

except Exception:
    try:
        import Levenshtein as _Lev  # type: ignore

        def _fast_distance(a: str, b: str) -> int:  # type: ignore
            return int(_Lev.distance(a, b))

    except Exception:
        _fast_distance = None


def _levenshtein_dp(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            temp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def char_fidelity(ref: str, hyp: str) -> float:
    if _fast_distance is not None:
        ed = _fast_distance(ref, hyp)
    else:
        ed = _levenshtein_dp(ref, hyp)
    denom = max(1, max(len(ref), len(hyp)))
    return 1.0 - (ed / denom)
