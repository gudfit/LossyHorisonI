from __future__ import annotations

from typing import List, Optional


def chrf_corpus(hyps: List[str], refs: List[str]) -> Optional[float]:
    try:
        import sacrebleu
    except Exception:
        return None
    try:
        score = sacrebleu.corpus_chrf(hyps, [refs]).score
        return float(score) / 100.0
    except Exception:
        return None
