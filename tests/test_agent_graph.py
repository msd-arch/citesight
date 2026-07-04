"""Agent graph tests with fake LLM + stub retriever/store/VLM: no model loads.

Covers: full happy path, the grounded-else-re-retrieve conditional loop with
budget expansion, checkpointing, and graceful degradation on router failure.
"""
import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image

from citesight.agent.graph import AgentDeps, build_graph
from citesight.config.settings import Settings
from citesight.models.llm import FakeLLM
from citesight.observability.tracing import Tracer
from citesight.store.qdrant_store import PageHit

ROUTER_JSON = (
    '{"query_type": "factoid", "ticker": "AAPL", "filing_type": "10-K", '
    '"top_k": 3, "text_heavy": false}'
)
VERDICT_FAIL = (
    '{"verdicts": [{"claim_index": 0, "supported": false, "reason": "mismatch", '
    '"region_hint": ""}], "grounded": false, '
    '"reformulated_query": "total net sales income statement"}'
)
VERDICT_PASS = (
    '{"verdicts": [{"claim_index": 0, "supported": true, "reason": "matches", '
    '"region_hint": "income statement, Total net sales row"}], '
    '"grounded": true, "reformulated_query": null}'
)
COMPOSE_JSON = '{"answer": "Final grounded answer.", "used_claim_indices": [0]}'


class StubRetriever:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return np.ones((2, 4), dtype=np.float32)

    def unload(self):
        pass


class StubStore:
    def __init__(self, hits):
        self.hits = hits
        self.top_ks = []

    def search(self, qvec, top_k, ticker=None, filing_type=None):
        self.top_ks.append(top_k)
        return self.hits[:top_k]


class StubAnswerer:
    model_id = "stub-vlm"

    def answer(self, question, page_images, extra_context="", max_new_tokens=384):
        return {
            "answer": "Draft answer.",
            "claims": [{"text": "claim A", "page": 1, "evidence": "evidence A"}],
        }

    def unload(self):
        pass


@pytest.fixture
def hits(tmp_path):
    p = tmp_path / "0001.png"
    Image.new("RGB", (40, 40), "white").save(p)
    return [
        PageHit(
            doc_id="AAPL_10-K_x", ticker="AAPL", filing_type="10-K",
            filing_date="2025-10-31", page_num=n, image_path=str(p), score=10.0 - n,
        )
        for n in (1, 2, 3, 4, 5)
    ]


def _deps(tmp_path, hits, llm):
    settings = Settings(
        sec_user_agent="t", data_dir=tmp_path, agent_max_attempts=2, _env_file=None
    )
    return AgentDeps(
        settings=settings,
        retriever=StubRetriever(),
        store=StubStore(hits),
        answerer=StubAnswerer(),
        llm=llm,
        tracer=Tracer(),
    )


def test_happy_path_grounded_first_try(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_JSON, VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    graph = build_graph(deps)
    state = graph.invoke({"query": "What were total net sales?"})
    assert state["grounded"] is True
    assert state["attempts"] == 1
    assert state["answer"] == "Final grounded answer."
    assert state["citations"][0]["region_hint"].startswith("income statement")
    assert deps.store.top_ks == [3]


def test_reretrieve_loop_on_failed_verification(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_JSON, VERDICT_FAIL, VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    graph = build_graph(deps)
    state = graph.invoke({"query": "What were total net sales?"})
    assert state["attempts"] == 2  # retrieved twice
    assert deps.store.top_ks == [3, 5]  # budget expanded on retry
    # retry used the verifier's reformulated query
    assert deps.retriever.queries[1] == "total net sales income statement"
    assert state["grounded"] is True


def test_gives_up_after_max_attempts(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_JSON, VERDICT_FAIL, VERDICT_FAIL, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    graph = build_graph(deps)
    state = graph.invoke({"query": "What were total net sales?"})
    assert state["attempts"] == 2
    assert state["grounded"] is False
    assert state["answer"]  # composer still produces an answer


def test_router_failure_degrades_to_default_plan(tmp_path, hits):
    llm = FakeLLM(script=["THIS IS NOT JSON", "STILL NOT JSON", VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    graph = build_graph(deps)
    state = graph.invoke({"query": "q"})
    assert state["plan"]["query_type"] == "factoid"  # default plan
    assert state["answer"] == "Final grounded answer."


def test_checkpointing_records_state(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_JSON, VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"query": "q"}, config=config)
    snapshot = graph.get_state(config)
    assert snapshot.values["answer"] == "Final grounded answer."
    assert snapshot.values["grounded"] is True


ROUTER_TEXT_HEAVY = (
    '{"query_type": "synthesis", "ticker": "AAPL", "filing_type": "10-K", '
    '"top_k": 3, "text_heavy": true}'
)


class StubHybrid:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        return [
            {
                "doc_id": h.doc_id, "ticker": h.ticker, "filing_type": h.filing_type,
                "filing_date": h.filing_date, "page_num": h.page_num,
                "image_path": h.image_path, "score": h.score,
            }
            for h in self.hits[:top_k]
        ]


def test_text_heavy_routes_through_hybrid(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_TEXT_HEAVY, VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)
    deps.hybrid = StubHybrid(hits)
    graph = build_graph(deps)
    state = graph.invoke({"query": "Summarize the risk factors."})
    assert deps.hybrid.queries == ["Summarize the risk factors."]
    assert deps.retriever.queries == []  # visual path not used
    assert state["answer"] == "Final grounded answer."


def test_text_heavy_without_hybrid_falls_back_to_visual(tmp_path, hits):
    llm = FakeLLM(script=[ROUTER_TEXT_HEAVY, VERDICT_PASS, COMPOSE_JSON])
    deps = _deps(tmp_path, hits, llm)  # hybrid=None
    graph = build_graph(deps)
    state = graph.invoke({"query": "Summarize the risk factors."})
    assert deps.retriever.queries  # visual path used
    assert state["answer"] == "Final grounded answer."


def test_empty_index_short_circuits(tmp_path):
    llm = FakeLLM(script=[ROUTER_JSON])
    deps = _deps(tmp_path, [], llm)
    graph = build_graph(deps)
    state = graph.invoke({"query": "q"})
    assert state["grounded"] is False
    assert "could not find" in state["answer"]
