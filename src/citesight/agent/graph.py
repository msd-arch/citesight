"""LangGraph agent: router -> retrieve -> reason -> verify -> (loop|compose).

Explicit state machine with checkpointing; the grounded-else-re-retrieve loop is
a conditional edge, not an ad-hoc while loop. Every node emits a tracer span.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from citesight.agent.prompt_loader import load_prompt
from citesight.agent.state import AgentState, PageRef
from citesight.config.settings import Settings
from citesight.models.llm import ChatLLM
from citesight.models.retriever import ColQwenRetriever, load_page_image
from citesight.models.vlm import QwenVlAnswerer
from citesight.observability.tracing import Tracer
from citesight.store.qdrant_store import QdrantPageStore

logger = logging.getLogger(__name__)

VALID_QUERY_TYPES = {"factoid", "synthesis", "table-math", "multi-doc-comparison"}


@dataclass
class AgentDeps:
    settings: Settings
    retriever: ColQwenRetriever
    store: QdrantPageStore
    answerer: QwenVlAnswerer
    llm: ChatLLM
    tracer: Tracer
    hybrid: Any = None  # HybridPageSearch | None — text-heavy retrieval path


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.d = deps

    # ------------------------------------------------------------- router
    def route(self, state: AgentState) -> dict:
        with self.d.tracer.span("router", input=state["query"]) as span:
            try:
                plan = self.d.llm.complete_json(
                    "You are a precise query router. Reply with only JSON.",
                    load_prompt("router", question=state["query"]),
                )
            except Exception as exc:  # graceful degradation: default plan
                logger.warning("router failed (%s); using default plan", exc)
                plan = {}
            if plan.get("query_type") not in VALID_QUERY_TYPES:
                plan["query_type"] = "factoid"
            plan.setdefault("top_k", 3)
            plan["top_k"] = max(1, min(int(plan.get("top_k") or 3), 10))
            filters = {
                "ticker": plan.get("ticker") or None,
                "filing_type": plan.get("filing_type") or None,
            }
            span.output = plan
        return {"plan": plan, "filters": filters, "attempts": 0}

    # ----------------------------------------------------------- retriever
    def retrieve(self, state: AgentState) -> dict:
        attempts = state.get("attempts", 0) + 1
        query = state.get("reformulated_query") or state["query"]
        # expand the budget on each retry
        top_k = state["plan"]["top_k"] + 2 * (attempts - 1)
        use_hybrid = bool(state["plan"].get("text_heavy")) and self.d.hybrid is not None
        with self.d.tracer.span(
            "retriever", input=query, attempt=attempts, top_k=top_k,
            path="hybrid" if use_hybrid else "visual",
        ) as span:
            if use_hybrid:
                pages = self.d.hybrid.search(query, top_k=top_k)
            else:
                if self.d.settings.sequential_models and attempts > 1:
                    self.d.answerer.unload()  # make room before reloading the retriever
                hits = self.d.store.search(
                    self.d.retriever.embed_query(query),
                    top_k=top_k,
                    ticker=state["filters"].get("ticker"),
                    filing_type=state["filters"].get("filing_type"),
                )
                pages = [
                    PageRef(
                        doc_id=h.doc_id,
                        ticker=h.ticker,
                        filing_type=h.filing_type,
                        filing_date=h.filing_date,
                        page_num=h.page_num,
                        image_path=h.image_path,
                        score=h.score,
                    )
                    for h in hits
                ]
            span.output = [f"{p['doc_id']} p.{p['page_num']}" for p in pages]
        return {"retrieved_pages": pages, "attempts": attempts}

    # ------------------------------------------------------------ reasoner
    def reason(self, state: AgentState) -> dict:
        pages = state["retrieved_pages"]
        with self.d.tracer.span("reasoner", input=state["query"], n_pages=len(pages)) as span:
            if not pages:
                span.output = "no pages retrieved"
                return {"draft_answer": "", "claims": []}
            if self.d.settings.sequential_models:
                self.d.retriever.unload()  # one big model resident at a time
            # cap prefill cost: only the best pages go to the VLM (prefix slice,
            # so claim page indices still map into retrieved_pages)
            pages = pages[: self.d.settings.vlm_max_pages]
            images = [load_page_image(Path(p["image_path"])) for p in pages]
            result = self.d.answerer.answer(
                state["query"],
                images,
                max_new_tokens=self.d.settings.vlm_max_new_tokens,
            )
            span.output = {"answer": result["answer"], "n_claims": len(result["claims"])}
        return {"draft_answer": result["answer"], "claims": result["claims"]}

    # ------------------------------------------------------------ verifier
    def verify(self, state: AgentState) -> dict:
        claims = state.get("claims", [])
        with self.d.tracer.span("verifier", input=len(claims)) as span:
            if not claims:
                span.output = {"grounded": False, "reason": "no claims produced"}
                return {"grounded": False, "verdicts": [], "region_hints": [],
                        "reformulated_query": None}
            claims_block = "\n".join(
                f"{i}. claim: {c['text']}\n   evidence: {c.get('evidence') or '(none)'}"
                f"   [page image {c['page']}]"
                for i, c in enumerate(claims)
            )
            try:
                out = self.d.llm.complete_json(
                    "You are a strict grounding verifier. Reply with only JSON.",
                    load_prompt("verifier", question=state["query"],
                                claims_block=claims_block),
                )
            except Exception as exc:
                logger.warning("verifier failed (%s); passing through ungrounded", exc)
                out = {"verdicts": [], "grounded": False, "reformulated_query": None}
            verdicts = out.get("verdicts") or []
            grounded = bool(out.get("grounded")) and len(verdicts) == len(claims)
            span.output = {"grounded": grounded, "n_verdicts": len(verdicts)}
        return {
            "grounded": grounded,
            "verdicts": verdicts,
            "region_hints": [v.get("region_hint", "") for v in verdicts],
            "reformulated_query": out.get("reformulated_query") or None,
        }

    # ------------------------------------------------------------ composer
    def compose(self, state: AgentState) -> dict:
        claims = state.get("claims", [])
        verdicts = state.get("verdicts", [])
        with self.d.tracer.span("composer", input=state.get("draft_answer", "")) as span:
            if not claims:
                answer = (
                    "I could not find evidence for this question in the indexed "
                    "filings after retrying retrieval."
                )
                span.output = answer
                return {"answer": answer, "citations": []}
            claims_block = "\n".join(
                f"{i}. {c['text']} [page image {c['page']}]" for i, c in enumerate(claims)
            )
            verdicts_block = "\n".join(
                f"{v.get('claim_index')}: supported={v.get('supported')} "
                f"({v.get('reason', '')})"
                for v in verdicts
            ) or "(verifier produced no verdicts)"
            try:
                out = self.d.llm.complete_json(
                    "You compose final grounded answers. Reply with only JSON.",
                    load_prompt(
                        "composer",
                        question=state["query"],
                        draft_answer=state.get("draft_answer", ""),
                        claims_block=claims_block,
                        verdicts_block=verdicts_block,
                    ),
                )
                answer = str(out.get("answer") or state.get("draft_answer", ""))
                used = out.get("used_claim_indices")
                if not isinstance(used, list):
                    used = list(range(len(claims)))
            except Exception as exc:
                logger.warning("composer failed (%s); using draft answer", exc)
                answer, used = state.get("draft_answer", ""), list(range(len(claims)))

            pages = state["retrieved_pages"]
            hints = state.get("region_hints", [])
            citations = []
            for i in used:
                if not (0 <= i < len(claims)):
                    continue
                c = claims[i]
                page = pages[c["page"] - 1] if 0 < c["page"] <= len(pages) else None
                if page is None:
                    continue
                citations.append(
                    {
                        **page,
                        "claim": c["text"],
                        "evidence": c.get("evidence", ""),
                        "region_hint": hints[i] if i < len(hints) else "",
                    }
                )
            span.output = {"answer": answer, "n_citations": len(citations)}
        return {"answer": answer, "citations": citations}


def should_retry(state: AgentState, max_attempts: int) -> str:
    if not state.get("grounded") and state.get("attempts", 0) < max_attempts:
        return "retrieve"
    return "compose"


def build_graph(deps: AgentDeps, checkpointer: Any = None):
    nodes = AgentNodes(deps)
    g = StateGraph(AgentState)
    g.add_node("route", nodes.route)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("reason", nodes.reason)
    g.add_node("verify", nodes.verify)
    g.add_node("compose", nodes.compose)

    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", "verify")
    g.add_conditional_edges(
        "verify",
        lambda s: should_retry(s, deps.settings.agent_max_attempts),
        {"retrieve": "retrieve", "compose": "compose"},
    )
    g.add_edge("compose", END)
    return g.compile(checkpointer=checkpointer)
