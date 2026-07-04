"""Hybrid text retrieval: bge-m3 dense + BM25 sparse, RRF fusion, cross-encoder rerank.

Complements the visual path for text-heavy queries (long narrative prose where
late-interaction over page images underperforms). Page text comes from the source
document during rendering — the visual retrieval path remains OCR-free.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from citesight.config.settings import Settings
from citesight.models.registry import get_registry, resolve_device

logger = logging.getLogger(__name__)

DENSE_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


@dataclass
class TextPage:
    doc_id: str
    page_num: int
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of ranked id lists (higher fused score first)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class TextEncoder:
    """bge-m3 dense embeddings (1024-dim, normalized)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = get_registry().get_or_load(
                f"text_encoder:{DENSE_MODEL_ID}", self._load
            )

    def _load(self):
        from sentence_transformers import SentenceTransformer

        device = resolve_device(self.settings.device)
        model = SentenceTransformer(DENSE_MODEL_ID, device=device)
        logger.info("text encoder loaded: model=%s device=%s", DENSE_MODEL_ID, device)
        return model

    def encode(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        self._ensure_loaded()
        return np.asarray(
            self._model.encode(
                list(texts), batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class Reranker:
    """bge-reranker-v2-m3 cross-encoder over (query, page_text) pairs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = get_registry().get_or_load(
                f"reranker:{RERANKER_MODEL_ID}", self._load
            )

    def _load(self):
        from sentence_transformers import CrossEncoder

        device = resolve_device(self.settings.device)
        model = CrossEncoder(RERANKER_MODEL_ID, device=device, max_length=1024)
        logger.info("reranker loaded: model=%s device=%s", RERANKER_MODEL_ID, device)
        return model

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        self._ensure_loaded()
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class HybridTextRetriever:
    """BM25 + dense over page texts -> RRF -> cross-encoder rerank.

    BM25 is built in-memory from the manifest's page texts (corpus is small by
    design until Phase 5); dense vectors live in the Qdrant text collection.
    """

    def __init__(
        self,
        settings: Settings,
        pages: Sequence[TextPage],
        encoder: TextEncoder | None = None,
        reranker: Reranker | None = None,
        dense_search=None,  # callable(query_vec, top_k, ticker, filing_type) -> [(key, score)]
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.settings = settings
        self.pages = list(pages)
        self.by_key = {f"{p.doc_id}:{p.page_num}": p for p in self.pages}
        self._bm25 = BM25Okapi([tokenize(p.text) or ["empty"] for p in self.pages])
        self.encoder = encoder or TextEncoder(settings)
        self.reranker = reranker or Reranker(settings)
        self._dense_search = dense_search

    def _bm25_ranking(self, query: str, top_n: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_n]
        return [
            f"{self.pages[i].doc_id}:{self.pages[i].page_num}"
            for i in order
            if scores[i] > 0
        ]

    def _dense_ranking(self, query: str, top_n: int) -> list[str]:
        qvec = self.encoder.encode([query])[0]
        if self._dense_search is not None:
            return [key for key, _ in self._dense_search(qvec, top_n)]
        # in-memory fallback: encode corpus lazily (small corpora only)
        if not hasattr(self, "_corpus_vecs"):
            self._corpus_vecs = self.encoder.encode([p.text[:2000] for p in self.pages])
        sims = self._corpus_vecs @ qvec
        order = np.argsort(sims)[::-1][:top_n]
        return [f"{self.pages[i].doc_id}:{self.pages[i].page_num}" for i in order]

    def search(
        self, query: str, top_k: int = 5, fuse_n: int = 20, rerank_n: int = 10
    ) -> list[tuple[TextPage, float]]:
        fused = rrf_fuse(
            [self._bm25_ranking(query, fuse_n), self._dense_ranking(query, fuse_n)]
        )[:rerank_n]
        candidates = [self.by_key[k] for k in fused if k in self.by_key]
        if not candidates:
            return []
        scores = self.reranker.rerank(query, [p.text[:4000] for p in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
