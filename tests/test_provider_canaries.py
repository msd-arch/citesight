"""Discriminative per-provider canaries: catch silent quality regressions when a
model/endpoint is swapped. These hit live free-tier endpoints, so they only run
when the corresponding key (or local Ollama) is available; CI skips them.

Shape checks are not enough — Phase 1 proved a model can load, emit the right
shapes, and still be garbage. Each canary asserts *behavior*.
"""
import os

import httpx
import pytest

from citesight.agent.prompt_loader import load_prompt
from citesight.models.llm import PROVIDERS, OpenAICompatLLM

ROUTER_SYSTEM = "You are a precise query router. Reply with only JSON."
VERIFIER_SYSTEM = "You are a strict grounding verifier. Reply with only JSON."

SUPPORTED_CLAIM = (
    "0. claim: Total net sales were $391,035 million in fiscal 2024\n"
    "   evidence: Total net sales $ 391,035   [page image 1]"
)
CONTRADICTED_CLAIM = (
    "0. claim: Total net sales were $500,000 million in fiscal 2024\n"
    "   evidence: Total net sales $ 391,035   [page image 1]"
)


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except httpx.TransportError:
        return False


def _llm(provider: str) -> OpenAICompatLLM:
    spec = PROVIDERS[provider]
    key = os.environ.get(spec.api_key_env, "not-needed") if spec.api_key_env else "not-needed"
    return OpenAICompatLLM(spec.base_url, key, spec.default_model, requests_per_min=10)


def _canary(llm: OpenAICompatLLM) -> None:
    # 1. router discriminates factoid vs multi-doc-comparison
    factoid = llm.complete_json(
        ROUTER_SYSTEM,
        load_prompt("router", question="What was Apple's total revenue in fiscal 2024?"),
    )
    assert factoid["query_type"] in ("factoid", "table-math")
    assert factoid.get("ticker") == "AAPL"
    multi = llm.complete_json(
        ROUTER_SYSTEM,
        load_prompt(
            "router",
            question="Compare the risk factors in Apple's and Microsoft's latest 10-Ks.",
        ),
    )
    assert multi["query_type"] == "multi-doc-comparison"

    # 2. verifier discriminates supported vs contradicted numeric claims
    ok = llm.complete_json(
        VERIFIER_SYSTEM,
        load_prompt("verifier", question="What were total net sales?",
                    claims_block=SUPPORTED_CLAIM),
    )
    assert ok["grounded"] is True
    bad = llm.complete_json(
        VERIFIER_SYSTEM,
        load_prompt("verifier", question="What were total net sales?",
                    claims_block=CONTRADICTED_CLAIM),
    )
    assert bad["grounded"] is False


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_canary():
    _canary(_llm("groq"))


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_canary():
    _canary(_llm("gemini"))


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
def test_ollama_canary():
    _canary(_llm("ollama"))
