"""Hybrid retrieval unit tests: RRF, BM25+dense fusion, rerank — no model loads."""
import numpy as np
import pytest

from citesight.config.settings import Settings
from citesight.models.text_retrieval import (
    HybridTextRetriever,
    TextPage,
    rrf_fuse,
    tokenize,
)


def test_rrf_prefers_items_ranked_high_in_both():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[0] in ("a", "b")
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("c") > fused.index("a")


def test_rrf_single_list_passthrough():
    assert rrf_fuse([["x", "y"]]) == ["x", "y"]


def test_tokenize():
    assert tokenize("Total net sales: $391,035!") == [
        "total", "net", "sales", "391", "035",
    ]


PAGES = [
    TextPage("D1", 1, "Total net sales were $391,035 million for fiscal 2024."),
    TextPage("D1", 2, "Risk factors include supply chain disruption and competition."),
    TextPage("D1", 3, "The Board of Directors declared a quarterly dividend."),
]


class StubEncoder:
    """Deterministic 'dense' vectors from token overlap with a tiny vocab."""

    VOCAB = ["sales", "risk", "dividend", "supply", "net"]

    def encode(self, texts, batch_size=8):
        out = []
        for t in texts:
            toks = tokenize(t)
            v = np.array([float(w in toks) for w in self.VOCAB], dtype=np.float32)
            norm = np.linalg.norm(v)
            out.append(v / norm if norm else v)
        return np.stack(out)


class StubReranker:
    """Scores by token overlap — enough to test ordering plumbing."""

    def rerank(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


@pytest.fixture
def retriever(tmp_path):
    settings = Settings(sec_user_agent="t", data_dir=tmp_path, _env_file=None)
    return HybridTextRetriever(
        settings, PAGES, encoder=StubEncoder(), reranker=StubReranker()
    )


def test_hybrid_ranks_relevant_page_first(retriever):
    results = retriever.search("What were total net sales?", top_k=2)
    assert results[0][0].page_num == 1
    results = retriever.search("What risk factors exist for supply?", top_k=2)
    assert results[0][0].page_num == 2


def test_hybrid_empty_query_no_crash(retriever):
    assert retriever.search("zzz qqq xxx", top_k=3) is not None


def test_manifest_page_texts_roundtrip(tmp_path):
    from citesight.store.manifest import Manifest

    m = Manifest(tmp_path / "m.db")
    m.add_page_texts("D1", ["page one text", "page two text"])
    m.add_page_texts("D1", ["page one text v2", "page two text"])  # idempotent replace
    rows = m.get_page_texts()
    assert rows == [("D1", 1, "page one text v2"), ("D1", 2, "page two text")]
    m.close()
