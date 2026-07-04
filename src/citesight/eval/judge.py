"""LLM-as-judge for the eval harness (versioned prompt: agent/prompts/judge_v1.md).

Uses the JUDGE role from make_llm — configurable separately from the agent LLM
so e.g. the app runs on Groq while judging runs on Gemini/Ollama. Judge runs
should enable the on-disk LLM cache to stay under free-tier caps.
"""
from __future__ import annotations

import logging
from pathlib import Path

from citesight.models.llm import ChatLLM

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "v1"
_PROMPT_PATH = (
    Path(__file__).parent.parent / "agent" / "prompts" / f"judge_{JUDGE_PROMPT_VERSION}.md"
)


def _section(name: str) -> str:
    body = _PROMPT_PATH.read_text(encoding="utf-8").split("---", 1)[1]
    part = body.split(f"## {name}", 1)[1]
    part = part.split("\n## ", 1)[0]
    return part.strip().replace("{{", "{").replace("}}", "}")


def judge_correctness(
    llm: ChatLLM, question: str, reference: str, answer: str
) -> tuple[float, str]:
    prompt = (
        _section("correctness")
        .replace("{question}", question)
        .replace("{reference}", reference)
        .replace("{answer}", answer or "(no answer)")
    )
    try:
        out = llm.complete_json("You are a strict eval judge. Reply with only JSON.", prompt)
        return float(out.get("score", 0.0)), str(out.get("reason", ""))
    except Exception as exc:
        logger.warning("correctness judge failed: %s", exc)
        return 0.0, f"judge error: {exc}"


def judge_citation(
    llm: ChatLLM, claim: str, page_text: str
) -> tuple[bool, str]:
    prompt = (
        _section("citation")
        .replace("{claim}", claim)
        .replace("{page_text}", page_text[:4000] or "(no text)")
    )
    try:
        out = llm.complete_json("You are a strict eval judge. Reply with only JSON.", prompt)
        return bool(out.get("supported")), str(out.get("reason", ""))
    except Exception as exc:
        logger.warning("citation judge failed: %s", exc)
        return False, f"judge error: {exc}"
