"""Pydantic models shared across pipeline boundaries."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One factual statement in the answer, tied to a supporting page."""

    text: str
    page_num: int = Field(description="1-based page number within the cited document")
    doc_id: str
    evidence: str = Field(
        default="", description="Verbatim snippet from the page supporting the claim"
    )


class Citation(BaseModel):
    doc_id: str
    ticker: str
    filing_type: str
    filing_date: str
    page_num: int
    image_path: str
    score: float


class AskResult(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    citations: list[Citation]
    retrieved_pages: int
    model_id: str
