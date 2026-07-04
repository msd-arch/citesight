"""Adapts the hybrid text retriever to the agent's PageRef interface."""
from __future__ import annotations

import logging

from citesight.agent.state import PageRef
from citesight.config.settings import Settings
from citesight.models.text_retrieval import HybridTextRetriever, TextPage
from citesight.store.manifest import Manifest

logger = logging.getLogger(__name__)


class HybridPageSearch:
    """Builds the hybrid index from manifest page texts; returns PageRefs."""

    def __init__(self, settings: Settings, manifest: Manifest) -> None:
        self.settings = settings
        self._docs = {d.doc_id: d for d in manifest.list_documents()}
        rows = manifest.get_page_texts()
        self._retriever = HybridTextRetriever(
            settings,
            [TextPage(doc_id=d, page_num=p, text=t) for d, p, t in rows],
        )
        logger.info("hybrid index built over %d pages", len(rows))

    def search(self, query: str, top_k: int = 5) -> list[PageRef]:
        refs: list[PageRef] = []
        for page, score in self._retriever.search(query, top_k=top_k):
            doc = self._docs.get(page.doc_id)
            if doc is None:
                continue
            refs.append(
                PageRef(
                    doc_id=page.doc_id,
                    ticker=doc.ticker,
                    filing_type=doc.filing_type,
                    filing_date=doc.filing_date,
                    page_num=page.page_num,
                    image_path=str(
                        self.settings.pages_dir / page.doc_id / f"{page.page_num:04d}.png"
                    ),
                    score=score,
                )
            )
        return refs
