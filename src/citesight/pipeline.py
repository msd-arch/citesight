"""Phase 2 pipeline: retrieve top-k page images -> VLM answers with citations.

No agent loop yet — that arrives in Phase 3 (LangGraph). This is the straight
retrieval->answer path the agent's reasoner node will reuse.
"""
from __future__ import annotations

import logging
from pathlib import Path

from citesight.config.settings import Settings
from citesight.models.retriever import ColQwenRetriever, load_page_image
from citesight.models.vlm import QwenVlAnswerer
from citesight.schemas import AskResult, Citation, Claim
from citesight.store.qdrant_store import QdrantPageStore

logger = logging.getLogger(__name__)


def ask(
    question: str,
    settings: Settings,
    retriever: ColQwenRetriever | None = None,
    store: QdrantPageStore | None = None,
    answerer: QwenVlAnswerer | None = None,
    top_k: int = 3,
    ticker: str | None = None,
    filing_type: str | None = None,
) -> AskResult:
    """Retrieve top-k pages for the question and answer from the page images."""
    retriever = retriever or ColQwenRetriever(settings)
    store = store or QdrantPageStore(settings)
    answerer = answerer or QwenVlAnswerer(settings)

    hits = store.search(
        retriever.embed_query(question),
        top_k=top_k,
        ticker=ticker,
        filing_type=filing_type,
    )
    if not hits:
        return AskResult(
            question=question,
            answer="No relevant pages found in the index.",
            claims=[],
            citations=[],
            retrieved_pages=0,
            model_id=answerer.model_id,
        )

    images = [load_page_image(Path(h.image_path)) for h in hits]
    result = answerer.answer(question, images)

    claims = [
        Claim(
            text=c["text"],
            page_num=hits[c["page"] - 1].page_num,
            doc_id=hits[c["page"] - 1].doc_id,
            evidence=c["evidence"],
        )
        for c in result["claims"]
    ]
    citations = [
        Citation(
            doc_id=h.doc_id,
            ticker=h.ticker,
            filing_type=h.filing_type,
            filing_date=h.filing_date,
            page_num=h.page_num,
            image_path=h.image_path,
            score=h.score,
        )
        for h in hits
    ]
    return AskResult(
        question=question,
        answer=result["answer"],
        claims=claims,
        citations=citations,
        retrieved_pages=len(hits),
        model_id=answerer.model_id,
    )
