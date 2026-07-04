"""Discriminative canary checks: catch silently-broken models BEFORE long runs.

Shape checks are not enough — a wrong transformers version once loaded ColQwen
with dropped adapter weights and produced correctly-shaped random embeddings
(see constraints.md). These checks assert *behavior*.
"""
from __future__ import annotations

import logging

from citesight.config.settings import Settings

logger = logging.getLogger(__name__)


def _make_page(lines: list[str]):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    y = 60
    for line in lines:
        d.text((60, y), line, fill="black")
        y += 40
    return img


SALES_PAGE = [
    "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
    "Total net sales    $ 391,035    $ 383,285",
    "Products           $ 294,866    $ 298,085",
]
RISK_PAGE = [
    "Item 1A. Risk Factors",
    "The Company's business can be impacted by political events,",
    "trade disputes, war, terrorism, and natural disasters.",
]


def retriever_canary(settings: Settings) -> bool:
    """ColQwen must rank the matching synthetic page clearly above the other."""
    from citesight.models.retriever import ColQwenRetriever

    r = ColQwenRetriever(settings)
    embs = r.embed_pages([_make_page(SALES_PAGE), _make_page(RISK_PAGE)])
    s_sales = r.score(r.embed_query("What was total net sales?"), embs)
    s_risk = r.score(r.embed_query("What risk factors does the company face?"), embs)
    ok = s_sales[0] > s_sales[1] and s_risk[1] > s_risk[0]
    logger.info(
        "retriever canary: sales q -> (%.2f, %.2f); risk q -> (%.2f, %.2f); %s",
        s_sales[0], s_sales[1], s_risk[0], s_risk[1], "PASS" if ok else "FAIL",
    )
    if settings.sequential_models:
        r.unload()
    return ok


def vlm_canary(settings: Settings) -> bool:
    """VLM must read a value off a synthetic page and cite it."""
    from citesight.models.vlm import QwenVlAnswerer

    a = QwenVlAnswerer(settings)
    result = a.answer(
        "What are total net sales?", [_make_page(SALES_PAGE)], max_new_tokens=96
    )
    ok = "391" in (result["answer"] + str(result["claims"]))
    logger.info("vlm canary: answer=%r %s", result["answer"][:100], "PASS" if ok else "FAIL")
    if settings.sequential_models:
        a.unload()
    return ok


def run_canaries(settings: Settings, include_vlm: bool = True) -> bool:
    ok = retriever_canary(settings)
    if include_vlm:
        ok = vlm_canary(settings) and ok
    return ok
