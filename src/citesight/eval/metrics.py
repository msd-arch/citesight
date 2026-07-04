"""Retrieval metrics against gold pages: nDCG@k, Recall@k, MRR."""
from __future__ import annotations

import math
from typing import Sequence

PageKey = tuple[str, int]  # (doc_id, page_num)


def ndcg_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int = 5) -> float:
    """Binary-relevance nDCG@k."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int = 5) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: Sequence[PageKey], gold: set[PageKey]) -> float:
    for i, p in enumerate(ranked, 1):
        if p in gold:
            return 1.0 / i
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,100]); 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[idx]
