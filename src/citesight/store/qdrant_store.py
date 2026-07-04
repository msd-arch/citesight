"""Qdrant page store: native multi-vector (MaxSim) with two-stage search.

Design notes (why two-stage):
  Patch embeddings are expensive at scale — a page is ~700-800 x 128-dim
  vectors, so exact MaxSim against every stored page is O(pages * tokens^2).
  We store two named vectors per page:
    * "mean"    - a single mean-pooled vector, HNSW-indexed, used as a fast
                  prefetch to shortlist candidates;
    * "colqwen" - the full multi-vector, HNSW disabled (m=0), used only to
                  MaxSim-rescore the shortlist.
  This keeps recall high while making query cost ~O(prefetch_k * tokens^2).

Storage cost is logged per page (bytes/page); enable POOLING_ENABLED to cut
the multi-vector footprint by ~pool_factor at a measured retrieval-quality
tradeoff (see eval).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from citesight.config.settings import Settings

logger = logging.getLogger(__name__)

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


@dataclass
class PagePoint:
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: str
    page_num: int
    image_path: str
    multivector: np.ndarray  # (n_tokens, dim)


@dataclass
class PageHit:
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: str
    page_num: int
    image_path: str
    score: float


def point_id_for(doc_id: str, page_num: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{page_num}"))


class QdrantPageStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.qdrant_mode == "embedded":
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    # ------------------------------------------------------------ collection
    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "colqwen": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    # rescoring-only vector: no ANN index needed
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
                "mean": models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            },
        )
        logger.info("created collection %s (dim=%d)", self.collection, dim)

    # ---------------------------------------------------------------- upsert
    def upsert_pages(self, pages: Sequence[PagePoint], batch_size: int = 8) -> None:
        total_bytes = 0
        for start in range(0, len(pages), batch_size):
            batch = pages[start : start + batch_size]
            points = []
            for p in batch:
                mv = p.multivector.astype(np.float32)
                total_bytes += mv.nbytes
                points.append(
                    models.PointStruct(
                        id=point_id_for(p.doc_id, p.page_num),
                        vector={
                            "colqwen": mv.tolist(),
                            "mean": mv.mean(axis=0).tolist(),
                        },
                        payload={
                            "doc_id": p.doc_id,
                            "ticker": p.ticker,
                            "filing_type": p.filing_type,
                            "filing_date": p.filing_date,
                            "page_num": p.page_num,
                            "image_path": p.image_path,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points)
        if pages:
            logger.info(
                "upserted %d pages, multivector storage: %.1f KB/page (%.2f MB total)",
                len(pages),
                total_bytes / len(pages) / 1024,
                total_bytes / 1e6,
            )

    # ---------------------------------------------------------------- search
    def search(
        self,
        query_multivector: np.ndarray,
        top_k: int = 5,
        prefetch_k: int = 50,
        ticker: str | None = None,
        filing_type: str | None = None,
    ) -> list[PageHit]:
        qmv = query_multivector.astype(np.float32)
        conditions = []
        if ticker:
            conditions.append(
                models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))
            )
        if filing_type:
            conditions.append(
                models.FieldCondition(
                    key="filing_type", match=models.MatchValue(value=filing_type)
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=models.Prefetch(
                query=qmv.mean(axis=0).tolist(),
                using="mean",
                filter=query_filter,
                limit=prefetch_k,
            ),
            query=qmv.tolist(),
            using="colqwen",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            PageHit(
                doc_id=pt.payload["doc_id"],
                ticker=pt.payload["ticker"],
                filing_type=pt.payload["filing_type"],
                filing_date=pt.payload["filing_date"],
                page_num=pt.payload["page_num"],
                image_path=pt.payload["image_path"],
                score=pt.score,
            )
            for pt in result.points
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()
