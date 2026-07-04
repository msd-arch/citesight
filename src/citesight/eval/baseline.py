"""Naive text-chunk RAG baseline: extracted text -> chunks -> bge-m3 -> LLM answer.

The comparison system the brief requires: every CiteSight headline number is
reported against this. Deliberately simple — paragraph chunks, dense-only
retrieval, text-only answering (no page images, no reranker, no agent loop).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from citesight.models.llm import ChatLLM
from citesight.models.text_retrieval import TextEncoder

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    doc_id: str
    page_num: int
    text: str


def chunk_pages(
    page_texts: list[tuple[str, int, str]], target_chars: int = 700
) -> list[Chunk]:
    """Paragraph-ish chunks keyed back to their source page."""
    chunks: list[Chunk] = []
    for doc_id, page_num, text in page_texts:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paragraphs:
            if buf and len(buf) + len(p) > target_chars:
                chunks.append(Chunk(doc_id, page_num, buf))
                buf = p
            else:
                buf = f"{buf}\n{p}" if buf else p
        if buf:
            chunks.append(Chunk(doc_id, page_num, buf))
    return chunks


class NaiveRagBaseline:
    def __init__(self, encoder: TextEncoder, chunks: list[Chunk]) -> None:
        self.encoder = encoder
        self.chunks = chunks
        self._vecs: np.ndarray | None = None

    def _ensure_encoded(self) -> None:
        if self._vecs is None:
            logger.info("baseline: encoding %d chunks", len(self.chunks))
            self._vecs = self.encoder.encode([c.text[:1500] for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 8, ticker: str | None = None) -> list[Chunk]:
        self._ensure_encoded()
        qvec = self.encoder.encode([query])[0]
        sims = self._vecs @ qvec
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if ticker and not c.doc_id.upper().startswith(ticker.upper() + "_"):
                continue
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def page_ranking(self, query: str, ticker: str | None = None, top_n: int = 20
                     ) -> list[tuple[str, int]]:
        """Page-level ranking (dedup chunks by page, best-chunk order)."""
        seen: list[tuple[str, int]] = []
        for c in self.retrieve(query, top_k=top_n * 3, ticker=ticker):
            key = (c.doc_id, c.page_num)
            if key not in seen:
                seen.append(key)
            if len(seen) >= top_n:
                break
        return seen

    def answer(self, llm: ChatLLM, query: str, ticker: str | None = None) -> str:
        chunks = self.retrieve(query, top_k=6, ticker=ticker)
        context = "\n\n---\n\n".join(
            f"[{c.doc_id} p.{c.page_num}]\n{c.text[:1200]}" for c in chunks
        )
        return llm.complete(
            "Answer strictly from the provided filing excerpts. Be concise. "
            "If the excerpts do not contain the answer, say so.",
            f"Excerpts:\n{context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=256,
        )
