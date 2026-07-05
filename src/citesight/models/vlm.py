"""VLM answerer: Qwen2.5-VL page-grounded answering with per-claim evidence.

Backends: HF transformers (default; 4-bit on CUDA) or vLLM (Linux-only, VRAM
permitting). Variant flag maps 3b/7b/32b to the corresponding Instruct model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from citesight.config.settings import Settings
from citesight.utils.json_parsing import extract_json, salvage_truncated_json
from citesight.models.registry import (
    flash_attention_available,
    get_registry,
    resolve_device,
    resolve_dtype,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

VARIANT_MODEL_IDS = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}

PROMPT_VERSION = "v2"
PROMPT_PATH = (
    Path(__file__).parent.parent / "agent" / "prompts" / f"vlm_answer_{PROMPT_VERSION}.md"
)


class VlmAnswer(dict):
    """{'answer': str, 'claims': [{'text','page','evidence'}]} with parse metadata."""


def parse_vlm_answer(raw_text: str, n_pages: int) -> dict:
    """Parse + sanitize the VLM's JSON: clamp page indices, drop malformed claims.

    Falls back to salvaging truncated output (generation hitting max_new_tokens
    mid-JSON is a common failure mode on tight token budgets).
    """
    try:
        data = extract_json(raw_text)
    except ValueError:
        data = salvage_truncated_json(raw_text)
        logger.warning("VLM JSON was truncated; salvaged partial result")
    answer = str(data.get("answer", "")).strip()
    claims = []
    for c in data.get("claims", []) or []:
        try:
            page = int(c["page"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= page <= n_pages):
            continue
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        claims.append(
            {"text": text, "page": page, "evidence": str(c.get("evidence", "")).strip()}
        )
    return {"answer": answer, "claims": claims}


def build_prompt(question: str, n_pages: int, extra_context: str = "") -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8").split("---", 1)[1].strip()
    pages = "\n".join(f"- Page image {i + 1}" for i in range(n_pages))
    return (
        template.replace("{n_pages}", str(n_pages))
        .replace("{pages}", pages)
        .replace("{question}", question)
        .replace(
            "{extra_context}",
            f"Additional context: {extra_context}" if extra_context else "",
        )
        .replace("{{", "{")
        .replace("}}", "}")
    )


class QwenVlAnswerer:
    """Page-grounded answering. `answer()` returns parsed answer + per-claim evidence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = VARIANT_MODEL_IDS[settings.vlm_variant]
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.settings.vlm_backend == "vllm":
            raise NotImplementedError(
                "vLLM backend is Linux-only and wired up in deployment; "
                "use VLM_BACKEND=transformers here (see constraints.md)."
            )
        self._model, self._processor = get_registry().get_or_load(
            f"vlm:{self.model_id}", self._load
        )

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = resolve_device(self.settings.device)
        dtype = resolve_dtype(device, self.settings.cpu_dtype)
        kwargs: dict = {"torch_dtype": dtype, "device_map": device}
        quantized = False
        if self.settings.vlm_quantize_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            quantized = True
        if device == "cuda" and flash_attention_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        ).eval()
        # cap visual tokens per image to keep prefill tractable
        processor = AutoProcessor.from_pretrained(
            self.model_id, max_pixels=self.settings.vlm_max_visual_tokens * 28 * 28
        )
        logger.info(
            "vlm loaded: model=%s device=%s dtype=%s quant4bit=%s",
            self.model_id, device, dtype, quantized,
        )
        return model, processor

    def unload(self) -> None:
        """Free model memory (sequential-residency mode on low-RAM machines)."""
        get_registry().unload(f"vlm:{self.model_id}")
        self._model = None
        self._processor = None

    # ---------------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        page_images: Sequence["Image.Image"],
        extra_context: str = "",
        max_new_tokens: int = 512,
    ) -> dict:
        """Returns {'answer': str, 'claims': [{'text','page','evidence'}], 'raw': str}."""
        import torch
        from qwen_vl_utils import process_vision_info

        self._ensure_loaded()
        prompt = build_prompt(question, len(page_images), extra_context)
        # Cap image resolution HERE (in the message), so qwen_vl_utils'
        # process_vision_info downscales each page to the visual-token budget.
        # The processor-level max_pixels does NOT reliably propagate to the image
        # processor on transformers 4.51, which let full-res pages (~3M px, ~12k
        # patches) into the vision tower and OOM'd its attention on a T4.
        max_px = self.settings.vlm_max_visual_tokens * 28 * 28
        messages = [
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image", "image": img, "max_pixels": max_px}
                        for img in page_images
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        logger.info("vlm raw output (%d chars): %s", len(raw), raw[:300])
        try:
            parsed = parse_vlm_answer(raw, n_pages=len(page_images))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("VLM output was not valid JSON: %s", exc)
            parsed = {"answer": raw.strip(), "claims": []}
        parsed["raw"] = raw
        return parsed
