"""Qdrant store tests using the embedded engine with tiny random multivectors."""
import numpy as np
import pytest

from citesight.config.settings import Settings
from citesight.store.qdrant_store import PagePoint, QdrantPageStore

DIM = 8


@pytest.fixture
def store(tmp_path):
    settings = Settings(
        qdrant_mode="embedded",
        data_dir=tmp_path / "data",
        sec_user_agent="t",
        _env_file=None,
    )
    s = QdrantPageStore(settings)
    yield s
    s.close()


def _page(doc_id, page_num, ticker, vec):
    return PagePoint(
        doc_id=doc_id,
        ticker=ticker,
        filing_type="10-K",
        filing_date="2024-11-01",
        page_num=page_num,
        image_path=f"data/pages/{doc_id}/{page_num:04d}.png",
        multivector=vec,
    )


def test_upsert_search_and_filter(store):
    rng = np.random.default_rng(7)
    # doc A pages point one way, doc B the other
    base_a = rng.normal(size=(1, DIM)).astype(np.float32)
    base_b = -base_a
    pages = []
    for i in range(1, 4):
        pages.append(_page("AAPL_10-K_x", i, "AAPL", base_a + 0.01 * rng.normal(size=(5, DIM)).astype(np.float32)))
        pages.append(_page("MSFT_10-K_y", i, "MSFT", base_b + 0.01 * rng.normal(size=(5, DIM)).astype(np.float32)))

    store.ensure_collection(dim=DIM)
    store.upsert_pages(pages)
    assert store.count() == 6

    query = base_a + 0.01 * rng.normal(size=(3, DIM)).astype(np.float32)
    hits = store.search(query, top_k=3)
    assert len(hits) == 3
    assert all(h.ticker == "AAPL" for h in hits)
    assert hits[0].score >= hits[-1].score

    msft_hits = store.search(query, top_k=3, ticker="MSFT")
    assert msft_hits and all(h.ticker == "MSFT" for h in msft_hits)


def test_upsert_is_idempotent(store):
    vec = np.ones((4, DIM), dtype=np.float32)
    store.ensure_collection(dim=DIM)
    store.upsert_pages([_page("D", 1, "AAPL", vec)])
    store.upsert_pages([_page("D", 1, "AAPL", vec)])  # same deterministic id
    assert store.count() == 1
