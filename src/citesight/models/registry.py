"""Lazy model registry with device/dtype auto-detection.

Nothing outside the models/ package imports transformers or colpali_engine
directly — model loading is isolated here and in the model wrappers.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, cpu_dtype: str = "bfloat16") -> "Any":
    import torch

    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    # CPU: bf16 halves resident memory (a 3B model: ~15GB fp32 -> ~7.5GB).
    # On sub-32GB machines that is the difference between computing and
    # swap-thrashing (see constraints.md).
    return torch.bfloat16 if cpu_dtype == "bfloat16" else torch.float32


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


class ModelRegistry:
    """Caches loaded models so each is loaded at most once per process."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._models:
            logger.info("loading model: %s", key)
            self._models[key] = loader()
        return self._models[key]

    def unload(self, key: str) -> None:
        """Free a model's memory (reloaded from disk cache on next use)."""
        import gc

        if self._models.pop(key, None) is not None:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release VRAM, not just RAM
            except ImportError:
                pass
            logger.info("unloaded model: %s", key)

    def unload_prefixed(self, prefix: str) -> None:
        for key in [k for k in self._models if k.startswith(prefix)]:
            self.unload(key)


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
