# constraints.md — version-sensitive quirks & environment adaptations

Documenting every deviation from the master build prompt and why.

## Environment adaptations (dev machine: Windows 11, no GPU, no Docker)

| Spec item | Adaptation | Why |
|---|---|---|
| Qdrant "via Docker in dev" | `QDRANT_MODE=embedded` (qdrant-client local on-disk engine) is the dev default; `docker/docker-compose.yml` provided for server mode | Docker is not installed on the dev machine. Embedded mode implements the same multi-vector (MaxSim) + prefetch Query API in-process, so all store code is identical; switch with `QDRANT_MODE=server`. |
| `pdf2image` for page rendering | PyMuPDF (`fitz`) for both PDF and EDGAR HTML | pdf2image requires poppler system binaries (painful on Windows). PyMuPDF is a self-contained headless renderer and also paginates reflowable HTML (`fitz.open(..., filetype="html")` + `doc.layout()`). HTML fidelity is basic-CSS-level; `render_document()` is the single seam where a Chromium print-to-PDF renderer can be swapped in later if table fidelity needs it. |
| vLLM serving option | `VLM_BACKEND=vllm` flag exists but vLLM **does not support Windows** | On this machine the flag must stay `transformers`. Use vLLM only on a Linux GPU host. |
| Indexing on a 24GB GPU | Device auto-detection (`cuda`/`mps`/`cpu`); dev machine runs CPU | No NVIDIA GPU present. CPU embedding of a 3B ColQwen is slow (~minutes/page batch), so `citesight ingest --max-pages N` caps pages for local acceptance checks. Full-corpus indexing should run on the GPU box. |
| `flash_attention_2` | Enabled only when `flash_attn` is importable AND device is CUDA | Not installable on Windows/CPU. |
| CPU VLM inference benchmark (i7-8650U, 15.7GB RAM, NVMe SSD) | bf16 kept as CPU default; slim profile for live runs: `VLM_MAX_PAGES=2 VLM_MAX_NEW_TOKENS=192 VLM_MAX_VISUAL_TOKENS=400` | Measured on Qwen2.5-VL-3B: **bf16 decode 0.21 tok/s** (CPU-bound; this pre-AVX512 CPU emulates bf16), **fp32 decode 0.03 tok/s** (15GB model > RAM, page-bound). Both answer a 1-small-image question correctly in ~4.5 min. Full-page agent rounds are ~30 min each on this machine — fine for a one-shot acceptance, not for iteration. Interactive latency requires the GPU box (7B/4-bit CUDA). |
| Model residency on 16GB RAM | `CPU_DTYPE=bfloat16` (default) + `SEQUENTIAL_MODELS=true`: the agent graph unloads the retriever before VLM reasoning and vice-versa | **Incident:** the first live agent run held ColQwen (3B) + Qwen2.5-VL (3B) simultaneously in fp32 = ~30GB of weights on a 15.7GB machine. The process spent 15h swap-thrashing at ~10% CPU without finishing one generation. bf16 halves each model to ~7.5GB; sequential residency keeps peak under ~10GB. Reload from local HF cache costs ~1-2 min per swap — noise compared to paging. `VLM_MAX_PAGES=3` and `VLM_MAX_NEW_TOKENS=384` cap prefill/decode cost. |
| 4-bit bitsandbytes | Gated on CUDA | bitsandbytes has no CPU path. |

## Version-sensitive quirks

- **transformers must be `>=4.49,<4.52` (CONFIRMED BREAKAGE)**: transformers 4.52 refactored
  Qwen2.5-VL so the text stack moved from `model.layers.*` to `model.language_model.layers.*`.
  The `vidore/colqwen2.5-v0.2` LoRA adapter (and `embed_tokens`) use the old key names, so under
  transformers 5.12.1 the adapter weights were **silently dropped** (load report showed all
  `lora_*` keys UNEXPECTED/MISSING). The model still loaded and produced correctly-shaped
  embeddings — but retrieval was random: a "total net sales" query scored a risk-factors page
  *above* the income-statement page (15.34 vs 15.05). Caught by a discriminative smoke test,
  not by shape checks. Pinning `transformers<4.52` resolves colpali-engine to 0.3.10 and fixes
  adapter loading. Lesson: after any transformers/colpali upgrade, rerun the discriminative
  check before trusting the index.

- `colpali-engine` >= 0.3.1 pinned (`<0.4`): `ColQwen2_5` / `ColQwen2_5_Processor` live in
  `colpali_engine.models`; `HierarchicalTokenPooler` in
  `colpali_engine.compression.token_pooling`. Exact resolved versions are pinned in `uv.lock`.
- ColQwen model outputs are **padded** per batch — embeddings must be trimmed with the
  processor's `attention_mask` before storing, or padding tokens pollute MaxSim scores
  (handled in `models/retriever.py`).
- Qdrant multivector: the `colqwen` named vector uses `MultiVectorComparator.MAX_SIM` with
  `hnsw_config.m=0` (rescoring-only; building an ANN index over per-token vectors is wasted
  work). Prefetch runs on the single `mean` vector. Requires qdrant-client >= 1.10 Query API.
- `langfuse` resolved to **4.13.0**: the v2 API (`client.trace()`, `trace.span()`) is gone in
  v3+/v4 — the SDK is OTel-based (`start_span`, `span.update/end`, `auth_check`). Our
  `LangfuseTracer` targets the v3+/v4 API; anything copied from older Langfuse+LangGraph
  tutorials will not work against this version.
- `qwen-vl-utils` (0.0.14) imports `torchvision` at module import but does **not** declare it
  as a dependency — declare `torchvision` explicitly (it also silently disappears when a
  transformers/torch downgrade re-resolves the tree, which is how we hit it).
- EDGAR: `company_tickers.json` and `/Archives/...` live on `www.sec.gov`, while the
  submissions API lives on `data.sec.gov`. Both require the User-Agent header; missing it
  returns 403 (we retry with backoff, but a real UA is mandatory).

## Actual resolved versions (2026-07-02, `uv.lock` is authoritative)

- Python 3.11.9
- colpali-engine **0.3.10** (downgraded from 0.3.17 by the transformers pin; ColQwen2_5 /
  ColQwen2_5_Processor / HierarchicalTokenPooler all present)
- transformers **4.51.3** (pinned `<4.52` — see confirmed breakage above)
- torch 2.6.0 (CPU wheels on this machine), peft 0.15.2
- qdrant-client 1.18.0, pymupdf 1.28.0, pillow 12.3.0, httpx 0.28.1, pydantic 2.13.4
