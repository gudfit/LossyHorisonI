from __future__ import annotations

import math
import numpy as np
from typing import Dict, Iterable, Optional, List, Tuple
from ..policies.masking import select_mask_set


class RankPMF:

    def __init__(self, K: int, alpha: float = 1.0):
        assert K >= 2
        self.K = int(K)
        self.alpha = float(alpha)
        self.counts: Dict[int, int] = {r: 0 for r in range(2, self.K + 1)}
        self.total = 0

    def update(self, r: int) -> None:
        if 2 <= r <= self.K:
            self.counts[r] = self.counts.get(r, 0) + 1
            self.total += 1

    def bulk_update(self, ranks: Iterable[int]) -> None:
        for r in ranks:
            self.update(int(r))

    def prob(self, r: int) -> float:
        if not (2 <= r <= self.K):
            return 0.0
        numer = self.counts.get(r, 0) + self.alpha
        denom = self.total + (self.K - 1) * self.alpha
        return float(numer) / float(max(denom, 1.0))

    def bits(self, r: int) -> float:
        p = max(self.prob(r), 1e-12)
        return -math.log2(p)

    def pmf_list(self) -> List[float]:
        denom = self.total + (self.K - 1) * self.alpha
        return [
            (self.counts.get(r, 0) + self.alpha) / max(denom, 1.0)
            for r in range(2, self.K + 1)
        ]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "K": int(self.K),
            "alpha": float(self.alpha),
            "counts": {int(k): int(v) for k, v in self.counts.items()},
            "total": int(self.total),
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, object]) -> "RankPMF":
        K = int(data.get("K", 2))
        alpha = float(data.get("alpha", 1.0))
        obj = cls(K=K, alpha=alpha)
        counts = data.get("counts", {})
        if isinstance(counts, dict):
            for k, v in counts.items():
                rk = int(k)
                if 2 <= rk <= K:
                    obj.counts[rk] = int(v)
        obj.total = int(data.get("total", sum(obj.counts.values())))
        return obj


class RankModel(RankPMF):

    def fit_zipf(self) -> Tuple[float, float, float]:
        pmf = self.pmf_list()
        r = np.arange(2, self.K + 1, dtype=float)
        p = np.array(pmf, dtype=float)
        mask = p > 0
        r = r[mask]
        p = p[mask]
        if len(p) < 2:
            return 1.0, 0.0, 0.0
        x = np.log(r)
        y = np.log(p)
        A = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = float(sol[0]), float(sol[1])
        y_pred = A @ sol
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        s = -slope
        return s, intercept, r2

    def train_on_texts(
        self, scorer, texts: List[str], p_mask: float, selection: str = "window"
    ) -> None:
        for text in texts:
            surprisals, tok, eligible = scorer.token_surprisal(text)
            specials = {
                scorer.tokenizer.cls_token_id,
                scorer.tokenizer.sep_token_id,
                scorer.tokenizer.pad_token_id,
            }
            ids = tok.input_ids[0].tolist()
            forbidden = {i for i, tid in enumerate(ids) if tid in specials}
            mask_idx, _ = select_mask_set(
                surprisals,
                p_mask=p_mask,
                selection=selection,
                forbidden=forbidden,
                eligible=eligible,
            )
            ranks, _topk, _ = scorer.ranks_and_topk(text, mask_idx, k=self.K)
            self.bulk_update([r for r in ranks.values() if 2 <= int(r) <= self.K])
