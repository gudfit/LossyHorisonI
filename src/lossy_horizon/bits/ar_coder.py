from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

import math


class NGramARCoder:

    def __init__(self, order: int = 3, vocab_size: int = 30522):
        self.n = max(1, int(order))
        self.V = int(vocab_size)
        self.counts: Dict[Tuple[int, ...], Dict[int, int]] = {}

    def _context_key(self, hist: List[int]) -> Tuple[int, ...]:
        if self.n <= 1:
            return tuple()
        k = tuple(hist[-(self.n - 1) :]) if len(hist) >= self.n - 1 else tuple(hist)
        return k

    def neglog2_stream(self, ids: Iterable[int]) -> List[float]:
        hist: List[int] = []
        out: List[float] = []
        for x in ids:
            ctx = self._context_key(hist)
            dist = self.counts.get(ctx)
            if dist is None or len(dist) == 0:
                p = 1.0 / float(self.V)
            else:
                total = sum(dist.values()) + self.V
                cnt = dist.get(int(x), 0) + 1
                p = cnt / total
            out.append(-math.log2(max(p, 1e-12)))
            if ctx not in self.counts:
                self.counts[ctx] = {int(x): 1}
            else:
                self.counts[ctx][int(x)] = self.counts[ctx].get(int(x), 0) + 1
            hist.append(int(x))
        return out
