from __future__ import annotations

from typing import Iterable, List, Tuple


def rle_encode(bits: Iterable[int]) -> List[Tuple[int, int]]:
    run: List[Tuple[int, int]] = []
    it = iter(bits)
    try:
        prev = int(next(it))
    except StopIteration:
        return []
    length = 1
    for b in it:
        b = int(b)
        if b == prev:
            length += 1
        else:
            run.append((prev, length))
            prev, length = b, 1
    run.append((prev, length))
    return run


def rle_decode(runs: Iterable[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    for v, l in runs:
        out.extend([int(v)] * int(l))
    return out
