"""Golden dataset loading and gold-page resolution.

Items live in eval/golden/*.jsonl. Gold pages can be pinned explicitly
(`gold_doc_id` + `gold_pages`) or — because page numbering depends on the
render — defined portably via `gold_text_patterns`: a page is gold when its
extracted text contains ALL patterns (case-insensitive). Resolution happens
against the manifest's page texts at eval time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GoldenItem(BaseModel):
    id: str
    ticker: str
    query: str
    reference_answer: str
    type: str = "factoid"  # factoid | synthesis | table-math | multi-doc-comparison
    text_heavy: bool = False
    gold_doc_id: str | None = None
    gold_pages: list[int] = Field(default_factory=list)
    gold_text_patterns: list[str] = Field(default_factory=list)


def load_golden(golden_dir: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for path in sorted(golden_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(GoldenItem.model_validate(json.loads(line)))
    logger.info("loaded %d golden items from %s", len(items), golden_dir)
    return items


def resolve_gold_pages(
    item: GoldenItem, page_texts: list[tuple[str, int, str]]
) -> set[tuple[str, int]]:
    """Resolve an item's gold pages against the indexed corpus."""
    gold: set[tuple[str, int]] = set()
    if item.gold_doc_id and item.gold_pages:
        gold |= {(item.gold_doc_id, p) for p in item.gold_pages}
    if item.gold_text_patterns:
        patterns = [p.lower() for p in item.gold_text_patterns]
        for doc_id, page_num, text in page_texts:
            if not doc_id.upper().startswith(item.ticker.upper() + "_"):
                continue
            lowered = text.lower()
            if all(p in lowered for p in patterns):
                gold.add((doc_id, page_num))
    return gold
