from __future__ import annotations

from typing import List, Optional


def bertscore_corpus(
    hyps: List[str], refs: List[str], lang: str = "en"
) -> Optional[float]:
    try:
        from bert_score import score as bert_score
    except Exception:
        return None
    try:
        P, R, F1 = bert_score(hyps, refs, lang=lang)
        return float(F1.mean().item())
    except Exception:
        return None
