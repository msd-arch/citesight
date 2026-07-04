"""Eval harness unit tests: metrics, golden resolution, baseline chunking, judge prompt."""
from pathlib import Path

from citesight.eval.baseline import chunk_pages
from citesight.eval.golden import GoldenItem, load_golden, resolve_gold_pages
from citesight.eval.judge import _section
from citesight.eval.metrics import mrr, ndcg_at_k, percentile, recall_at_k

G = {("D", 2), ("D", 5)}


def test_ndcg_perfect_and_miss():
    assert ndcg_at_k([("D", 2), ("D", 5), ("D", 9)], G, k=5) == 1.0
    assert ndcg_at_k([("D", 9), ("D", 8)], G, k=5) == 0.0
    partial = ndcg_at_k([("D", 9), ("D", 2)], G, k=5)
    assert 0.0 < partial < 1.0


def test_recall_and_mrr():
    assert recall_at_k([("D", 2), ("D", 9)], G, k=2) == 0.5
    assert mrr([("D", 9), ("D", 5)], G) == 0.5
    assert mrr([("X", 1)], G) == 0.0
    assert mrr([], G) == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 95) == 0.0


def test_golden_pattern_resolution():
    item = GoldenItem(
        id="x", ticker="AAPL", query="q", reference_answer="a",
        gold_text_patterns=["item 1a", "risk factors"],
    )
    pages = [
        ("AAPL_10-K_1", 9, "Item 1A. Risk Factors — summary of risks"),
        ("AAPL_10-K_1", 2, "Item 1. Business"),
        ("MSFT_10-K_9", 4, "Item 1A. Risk Factors"),  # wrong ticker
    ]
    assert resolve_gold_pages(item, pages) == {("AAPL_10-K_1", 9)}


def test_golden_explicit_pages_union_patterns():
    item = GoldenItem(
        id="x", ticker="AAPL", query="q", reference_answer="a",
        gold_doc_id="AAPL_10-K_1", gold_pages=[1],
        gold_text_patterns=["nasdaq"],
    )
    pages = [("AAPL_10-K_1", 3, "listed on Nasdaq")]
    assert resolve_gold_pages(item, pages) == {("AAPL_10-K_1", 1), ("AAPL_10-K_1", 3)}


def test_seed_golden_file_is_valid():
    items = load_golden(Path("eval/golden"))
    assert len(items) >= 20
    assert all(i.gold_text_patterns or i.gold_pages for i in items)
    assert {i.ticker for i in items} == {"AAPL", "MSFT", "NVDA"}
    assert len({i.id for i in items}) == len(items)  # unique ids


def test_chunk_pages_keys_back_to_source_page():
    pages = [("D", 1, "para one\n\npara two\n\n" + "x" * 900), ("D", 2, "short")]
    chunks = chunk_pages(pages, target_chars=100)
    assert all(c.doc_id == "D" for c in chunks)
    assert {c.page_num for c in chunks} == {1, 2}
    assert len([c for c in chunks if c.page_num == 1]) >= 2


def test_judge_prompt_sections_load():
    correctness = _section("correctness")
    citation = _section("citation")
    assert "{question}" in correctness and '"score"' in correctness
    assert "{claim}" in citation and '"supported"' in citation
    assert "{{" not in correctness  # unescaped for direct .replace use
