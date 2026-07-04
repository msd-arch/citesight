"""Load versioned prompts from agent/prompts/*.md and fill placeholders."""
from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, version: str = "v1", **placeholders: str) -> str:
    body = (
        (PROMPTS_DIR / f"{name}_{version}.md")
        .read_text(encoding="utf-8")
        .split("---", 1)[1]
        .strip()
    )
    for key, value in placeholders.items():
        body = body.replace("{" + key + "}", str(value))
    return body.replace("{{", "{").replace("}}", "}")
