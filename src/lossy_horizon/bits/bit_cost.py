from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from ..utils.rle import rle_encode


def binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -p * math.log2(p) - q * math.log2(q)


def comb_log2_nCk(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("inf")
    if k == 0 or k == n:
        return 0.0
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(2.0)


def elias_gamma_bits(x: int) -> float:
    if x <= 0:
        return 0.0
    m = int(math.floor(math.log2(x)))
    return float(2 * m + 1)


def rle_exact_bits(bitvec: List[int]) -> float:
    n = len(bitvec)
    if n == 0:
        return 0.0
    runs = rle_encode(bitvec)
    bits = 1.0
    for _, run_len in runs:
        bits += elias_gamma_bits(int(run_len))
    return bits


def positional_bits_cost_from_indices(
    n: int, keep_indices: List[int]
) -> Tuple[float, str, float, float]:
    keep_set = set(keep_indices)
    k = len(keep_set)
    bits_enum = comb_log2_nCk(n, k)
    bitvec = [1 if i in keep_set else 0 for i in range(n)]
    bits_rle = rle_exact_bits(bitvec)
    if bits_enum <= bits_rle:
        return bits_enum, "enumerative", bits_enum, bits_rle
    else:
        return bits_rle, "rle", bits_enum, bits_rle


def token_entropy_bits(neg_log2_probs: Iterable[float]) -> float:
    return float(sum(neg_log2_probs))
