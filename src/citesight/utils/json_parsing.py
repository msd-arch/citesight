"""Robust JSON extraction from LLM/VLM text output (code fences, prefix chatter)."""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output (handles code fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in model output: {text[:200]!r}")


def salvage_truncated_json(text: str) -> dict:
    """Best-effort recovery of a JSON object whose generation was cut off.

    Walks the text tracking container/string state, then tries progressively
    shorter prefixes that end on a completed element, appending the closers
    for whatever containers are still open. Recovers complete claims from a
    truncated claims array; an element cut mid-string is dropped.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)(?:```|$)", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object to salvage")

    stack: list[str] = []
    in_string = False
    escape = False
    candidates: list[tuple[int, str]] = []  # (end_pos, closers_needed)
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                candidates.append(
                    (i, "".join("}" if c == "{" else "]" for c in reversed(stack)))
                )
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return json.loads(text[start : i + 1])
            candidates.append(
                (i, "".join("}" if c == "{" else "]" for c in reversed(stack)))
            )

    for pos, closers in reversed(candidates):
        try:
            return json.loads(text[start : pos + 1] + closers)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unsalvageable truncated JSON: {text[:200]!r}")
