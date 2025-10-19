from __future__ import annotations

import math
from typing import Dict, List, Tuple, Iterable


def _build_freq_from_probs(
    probs: Iterable[float], M: int
) -> Tuple[List[int], List[int]]:
    ps = list(probs)
    freqs = [max(1, int(round(p * M))) for p in ps]
    s = sum(freqs)
    if s != M:
        diff = M - s
        sign = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            idx = (
                max(range(len(ps)), key=lambda i: ps[i])
                if sign > 0
                else min(range(len(ps)), key=lambda i: ps[i])
            )
            freqs[idx] = max(1, freqs[idx] + sign)
    cdf = [0]
    acc = 0
    for f in freqs:
        acc += f
        cdf.append(acc)
    return freqs, cdf


def _rans_bits(
    symbols: Iterable[int],
    freqs: List[int],
    cdf: List[int],
    M: int = 1 << 12,
    R: int = 16,
) -> float:
    x = 1
    total_bits = 0
    for s in symbols:
        f = freqs[s]
        c = cdf[s]
        while x >= (f << R):
            total_bits += R
            x >>= R
        x = (x // f) * M + (x % f) + c
    final_bits = int(math.ceil(math.log2(max(2, x))))
    return float(total_bits + final_bits)


def rans_bits_bernoulli(
    bits: Iterable[int], p1: float, M: int = 1 << 12, R: int = 16
) -> float:
    p0 = max(1e-12, min(1.0 - 1e-12, 1.0 - p1))
    p1 = max(1e-12, min(1.0 - 1e-12, p1))
    freqs, cdf = _build_freq_from_probs([p0, p1], M)
    return _rans_bits(bits, freqs, cdf, M=M, R=R)


def rans_bits_from_pmf(
    symbols: Iterable[int], probs: List[float], M: int = 1 << 12, R: int = 16
) -> float:
    freqs, cdf = _build_freq_from_probs(probs, M)
    return _rans_bits(symbols, freqs, cdf, M=M, R=R)


def rans_bits_adaptive_unigram(
    symbols: Iterable[int],
    vocab_size: int,
    M: int = 1 << 12,
    R: int = 16,
    alpha: int = 1,
) -> float:
    counts = [alpha] * vocab_size
    total = vocab_size * alpha
    x = 1
    total_bits = 0
    for s in symbols:
        s = int(s)
        f_s = counts[s]
        f_rest = total - f_s
        if f_rest <= 0:
            f_rest = 1
            total += 1
        freqs = [f_rest, f_s]
        cdf = [0, f_rest, f_rest + f_s]
        while x >= (freqs[1] << R):
            total_bits += R
            x >>= R
        x = (x // freqs[1]) * (freqs[0] + freqs[1]) + (x % freqs[1]) + cdf[1]
        counts[s] += 1
        total += 1
    final_bits = int(math.ceil(math.log2(max(2, x))))
    return float(total_bits + final_bits)


def ac_bits_adaptive_unigram_ideal(
    symbols: Iterable[int], vocab_size: int, alpha: float = 1.0
) -> float:
    counts = [float(alpha)] * int(vocab_size)
    total = float(vocab_size) * float(alpha)
    bits = 0.0
    for s in symbols:
        s = int(s)
        p = counts[s] / total if total > 0 else 1.0 / max(1, vocab_size)
        bits += -math.log2(max(p, 1e-12))
        counts[s] += 1.0
        total += 1.0
    return float(bits)
