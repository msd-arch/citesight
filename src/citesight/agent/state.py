"""Agent graph state (LangGraph)."""
from __future__ import annotations

from typing import Any, TypedDict


class PageRef(TypedDict):
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: str
    page_num: int
    image_path: str
    score: float


class AgentState(TypedDict, total=False):
    query: str
    plan: dict[str, Any]  # router output: query_type, ticker, filing_type, top_k, text_heavy
    filters: dict[str, Any]
    retrieved_pages: list[PageRef]
    draft_answer: str
    claims: list[dict[str, Any]]  # {text, page (1-based into retrieved), evidence}
    verdicts: list[dict[str, Any]]
    region_hints: list[str]
    grounded: bool
    reformulated_query: str | None
    attempts: int
    answer: str
    citations: list[dict[str, Any]]
    cost: dict[str, Any]  # per-node elapsed_ms etc.
