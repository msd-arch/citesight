"""Single typed settings module for CiteSight (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal[
    "groq", "gemini", "ollama", "openai_compatible", "self_vlm", "fake"
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- SEC EDGAR ---
    # EDGAR requires a descriptive User-Agent with a contact email.
    sec_user_agent: str | None = None
    edgar_max_requests_per_sec: float = 8.0  # SEC allows <=10; stay under

    # --- Storage / paths ---
    data_dir: Path = Path("data")

    # --- Qdrant ---
    # "embedded" runs qdrant-client's local on-disk engine (no Docker needed);
    # "server" talks to a running Qdrant instance (docker/docker-compose.yml).
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "citesight_pages"

    # --- Visual retriever (ColQwen2.5) ---
    retriever_model_id: str = "vidore/colqwen2.5-v0.2"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    # bf16 halves CPU-resident model memory; float32 only for numerical debugging
    cpu_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    # On <32GB-RAM machines keep only ONE big model resident at a time
    # (the agent graph unloads retriever/VLM around each other).
    sequential_models: bool = True
    quantize_4bit: bool = False  # bitsandbytes; CUDA only
    embed_batch_size: int = 2
    # Token pooling (HierarchicalTokenPooler) trades retrieval quality for a
    # ~pool_factor reduction in stored patch embeddings. Benchmarked in eval.
    pooling_enabled: bool = False
    pool_factor: int = 3

    # --- Page rendering ---
    render_dpi: int = 150
    # Cap the longest image edge so ColQwen2.5 stays <=768 visual patches.
    max_image_edge: int = 1540

    # --- VLM answerer (Phase 2+) ---
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_variant: Literal["3b", "7b", "32b"] = "7b"
    vlm_backend: Literal["transformers", "vllm"] = "transformers"
    vlm_quantize_4bit: bool = True  # default for <24GB VRAM
    vlm_max_pages: int = 3  # cap page images per VLM call (prefill cost)
    vlm_max_new_tokens: int = 384
    vlm_max_visual_tokens: int = 768  # per-image patch budget (prefill cost)

    # --- Agent LLM / judge (Phase 3+, provider-agnostic; NO paid APIs) ---
    # All providers speak the OpenAI chat-completions API. The agent LLM
    # (router/verifier/composer) and the eval judge are configured separately
    # so e.g. the app can run on Groq while the judge runs on Gemini/Ollama.
    # LLM_PROVIDER (if set) is a convenience fallback applied to BOTH roles
    # unless the role-specific variable overrides it.
    llm_provider: LlmProvider | None = None
    agent_llm_provider: LlmProvider = "groq"
    agent_llm_model: str | None = None  # None -> provider default
    judge_llm_provider: LlmProvider = "gemini"
    judge_llm_model: str | None = None
    # openai_compatible provider: fully env-configured endpoint
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None
    openai_compatible_api_key_env: str = "OPENAI_COMPATIBLE_API_KEY"
    # free-tier protection
    llm_requests_per_min: float = 25.0
    llm_cache_enabled: bool = False  # on-disk response cache (eval turns this on)
    llm_max_retries: int = 5

    # --- Agent graph ---
    agent_max_attempts: int = 2  # re-retrieval loop budget

    # --- Tracing ---
    # langfuse -> Langfuse Cloud (keys via LANGFUSE_* env, free Hobby tier);
    # local -> JSONL under data/traces (default: offline dev / eval);
    # off -> no-op (CI).
    tracing: Literal["langfuse", "local", "off"] = "local"

    # --- Eval harness ---
    eval_golden_dir: Path = Path("eval/golden")
    eval_reports_dir: Path = Path("eval/reports")
    eval_top_k: int = 5

    # --- Derived paths ---
    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.db"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm_cache"

    @property
    def checkpoints_path(self) -> Path:
        return self.data_dir / "checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    # Load .env into os.environ too: provider API keys (GROQ_API_KEY, ...) are
    # read from the environment by make_llm and the langfuse SDK, not via
    # pydantic fields.
    from dotenv import load_dotenv

    load_dotenv()
    return Settings()
