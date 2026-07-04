"""Visual retriever: ColQwen2.5 late-interaction page/query embeddings.

Exposes plain numpy multi-vectors (n_tokens, dim) so the rest of the codebase
never touches torch/transformers types.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from citesight.config.settings import Settings
from citesight.models.registry import (
    flash_attention_available,
    get_registry,
    resolve_device,
    resolve_dtype,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

MultiVector = np.ndarray  # shape (n_tokens, dim), float32


class ColQwenRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        loaded = get_registry().get_or_load(
            f"retriever:{self.settings.retriever_model_id}", self._load
        )
        self._model, self._processor = loaded

    def _load(self):
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        device = resolve_device(self.settings.device)
        dtype = resolve_dtype(device, self.settings.cpu_dtype)
        kwargs: dict = {"torch_dtype": dtype, "device_map": device}

        if self.settings.quantize_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        if device == "cuda" and flash_attention_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        model = ColQwen2_5.from_pretrained(
            self.settings.retriever_model_id, **kwargs
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(
            self.settings.retriever_model_id
        )
        logger.info(
            "retriever loaded: model=%s device=%s dtype=%s quant4bit=%s",
            self.settings.retriever_model_id,
            device,
            dtype,
            self.settings.quantize_4bit and device == "cuda",
        )
        return model, processor

    def unload(self) -> None:
        """Free model memory (sequential-residency mode on low-RAM machines)."""
        get_registry().unload(f"retriever:{self.settings.retriever_model_id}")
        self._model = None
        self._processor = None

    # ----------------------------------------------------------------- embed
    def embed_pages(self, images: Sequence["Image.Image"]) -> list[MultiVector]:
        """Embed page images into per-token multi-vectors (padding trimmed)."""
        import torch

        self._ensure_loaded()
        out: list[MultiVector] = []
        bs = self.settings.embed_batch_size
        for start in range(0, len(images), bs):
            batch = list(images[start : start + bs])
            inputs = self._processor.process_images(batch).to(self._model.device)
            with torch.no_grad():
                embs = self._model(**inputs)  # (B, seq, dim), padded
            mask = inputs["attention_mask"].bool()
            for i in range(embs.shape[0]):
                vec = embs[i][mask[i]].to(torch.float32).cpu().numpy()
                out.append(np.ascontiguousarray(vec))
            logger.info("embedded pages %d-%d / %d", start + 1, start + len(batch), len(images))
        return out

    def embed_query(self, text: str) -> MultiVector:
        import torch

        self._ensure_loaded()
        inputs = self._processor.process_queries([text]).to(self._model.device)
        with torch.no_grad():
            embs = self._model(**inputs)
        mask = inputs["attention_mask"].bool()
        return np.ascontiguousarray(embs[0][mask[0]].to(torch.float32).cpu().numpy())

    # ----------------------------------------------------------------- score
    def score(self, query: MultiVector, pages: Sequence[MultiVector]) -> list[float]:
        """In-memory MaxSim scoring (smoke-test path; Qdrant is production)."""
        import torch

        self._ensure_loaded()
        scores = self._processor.score_multi_vector(
            [torch.from_numpy(query)], [torch.from_numpy(p) for p in pages]
        )
        return scores[0].tolist()

    # ------------------------------------------------------------------ pool
    def pool_embeddings(self, embeddings: Sequence[MultiVector]) -> list[MultiVector]:
        """Hierarchical token pooling to cut storage ~pool_factor x."""
        import torch
        from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

        pooler = HierarchicalTokenPooler()
        pooled = pooler.pool_embeddings(
            [torch.from_numpy(e) for e in embeddings],
            pool_factor=self.settings.pool_factor,
            padding=False,
        )
        return [np.ascontiguousarray(p.to(torch.float32).numpy()) for p in pooled]


def load_page_image(path: Path) -> "Image.Image":
    from PIL import Image

    return Image.open(path).convert("RGB")
