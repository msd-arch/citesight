# CiteSight

Agentic multimodal RAG over visually complex SEC filings (10-K / 10-Q / 8-K), using
**visual document retrieval** (ColQwen2.5 page-image embeddings — no OCR) with
region-level page citations gated by a verification agent.

> Status: **Phase 1** (ingestion + visual index) complete. Phases 2-8 (VLM answering,
> LangGraph agent, hybrid retrieval, eval harness, ops, API/MCP/frontend) in progress.

## Architecture (Phase 1 slice)

```
EDGAR (data.sec.gov) ──> page images (PyMuPDF, 150 DPI)
                              │
                     ColQwen2.5 (colpali-engine)
                     per-token multi-vectors
                              │
                 Qdrant (multi-vector, MaxSim)
        two-stage search: mean-vector prefetch ──> MaxSim rescore
```

## Quickstart

```bash
pip install uv
uv sync --all-extras
cp .env.example .env       # set SEC_USER_AGENT (contact email required by EDGAR)

# index 2 Apple filings (use --max-pages on CPU-only machines)
uv run citesight ingest --ticker AAPL --types 10-K --limit 2

# raw MaxSim retrieval
uv run citesight search "What was Apple's total net sales?" --top-k 5
uv run citesight status
```

Qdrant runs **embedded** (on-disk, no Docker) by default; set `QDRANT_MODE=server` and
`docker compose -f docker/docker-compose.yml up` to use a Qdrant server instead.

## LLM providers (no paid APIs)

The agent LLM (router / verifier / composer) and the eval judge are **separately
configurable** and speak the OpenAI chat-completions protocol. Set
`AGENT_LLM_PROVIDER` / `JUDGE_LLM_PROVIDER` (+ optional `*_LLM_MODEL`):

| Provider | Endpoint | Key env | Default model | Notes |
|---|---|---|---|---|
| `groq` | `api.groq.com/openai/v1` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | fast free tier; app default |
| `gemini` | `generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `gemini-2.0-flash` | generous free volume; judge default |
| `ollama` | `localhost:11434/v1` | — | `qwen2.5:7b-instruct` | local, keyless, no rate limits |
| `openai_compatible` | `OPENAI_COMPATIBLE_BASE_URL` | `OPENAI_COMPATIBLE_API_KEY` | `OPENAI_COMPATIBLE_MODEL` | any OpenAI-protocol endpoint |
| `self_vlm` | in-process | — | the loaded Qwen2.5-VL | zero extra deps; text-only reasoning through the VLM |
| `fake` | — | — | deterministic | CI / offline tests |

Free-tier protection: request pacing (`LLM_REQUESTS_PER_MIN`), exponential backoff
honoring `retry-after` on 429s, and an on-disk response cache (`LLM_CACHE_ENABLED`,
always on for eval runs) so a full eval stays under daily token caps. Each provider
has a **discriminative canary test** (`tests/test_provider_canaries.py`) so a model
swap that silently degrades routing/judging fails loudly, not just shape-checks.

## Tracing

`TRACING=langfuse | local | off` behind one interface: `langfuse` exports to Langfuse
Cloud (free Hobby tier, `LANGFUSE_*` keys); `local` (default) writes JSONL spans to
`data/traces/`; `off` for CI. **Eval runs are always forced to the local exporter** to
protect the free-tier monthly quota — Langfuse Cloud is reserved for the interactive path.

See [constraints.md](constraints.md) for environment adaptations and version-sensitive quirks.

## Results / cost tables

Populated in Phase 5 (eval harness: CiteSight vs naive text-RAG baseline) and
Phase 6 (cost per query before/after tiered routing + caching + pooling).
