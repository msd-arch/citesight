"""ask() pipeline wiring with stubbed retriever/store/answerer — no model loads."""
import numpy as np
import pytest
from PIL import Image

from citesight.config.settings import Settings
from citesight.pipeline import ask
from citesight.store.qdrant_store import PageHit


class StubRetriever:
    def embed_query(self, text):
        return np.ones((3, 8), dtype=np.float32)


class StubStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, qvec, top_k, ticker=None, filing_type=None):
        return self._hits[:top_k]


class StubAnswerer:
    model_id = "stub-vlm"

    def __init__(self, result):
        self._result = result

    def answer(self, question, page_images, extra_context=""):
        assert all(isinstance(i, Image.Image) for i in page_images)
        return self._result


@pytest.fixture
def hits(tmp_path):
    out = []
    for i in (1, 2):
        p = tmp_path / f"{i:04d}.png"
        Image.new("RGB", (50, 50), "white").save(p)
        out.append(
            PageHit(
                doc_id="AAPL_10-K_x",
                ticker="AAPL",
                filing_type="10-K",
                filing_date="2025-10-31",
                page_num=i,
                image_path=str(p),
                score=10.0 - i,
            )
        )
    return out


def test_ask_maps_claim_pages_to_doc_pages(hits, tmp_path):
    settings = Settings(sec_user_agent="t", data_dir=tmp_path, _env_file=None)
    answerer = StubAnswerer(
        {
            "answer": "Listed on Nasdaq.",
            "claims": [{"text": "Listed on Nasdaq", "page": 2, "evidence": "Nasdaq"}],
        }
    )
    result = ask(
        "Where is AAPL listed?",
        settings,
        retriever=StubRetriever(),
        store=StubStore(hits),
        answerer=answerer,
        top_k=2,
    )
    assert result.answer == "Listed on Nasdaq."
    # claim's "page 2" = second retrieved image -> real page_num 2 of the doc
    assert result.claims[0].page_num == hits[1].page_num
    assert result.claims[0].doc_id == "AAPL_10-K_x"
    assert len(result.citations) == 2
    assert result.citations[0].score > result.citations[1].score


def test_ask_empty_index(tmp_path):
    settings = Settings(sec_user_agent="t", data_dir=tmp_path, _env_file=None)
    result = ask(
        "Anything?",
        settings,
        retriever=StubRetriever(),
        store=StubStore([]),
        answerer=StubAnswerer({"answer": "", "claims": []}),
    )
    assert result.retrieved_pages == 0
    assert result.claims == []
