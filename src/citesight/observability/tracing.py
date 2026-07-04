"""Tracing behind a thin interface: langfuse | local (JSONL) | off.

The interactive app path can use Langfuse Cloud (free Hobby tier; keys via
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST). Eval-harness runs
should use the LOCAL exporter to protect the free-tier monthly quota — pass
purpose="eval" so this is enforced by default. CI runs with TRACING=off.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from citesight.config.settings import Settings

logger = logging.getLogger(__name__)


class Span:
    def __init__(self, name: str, trace_id: str) -> None:
        self.name = name
        self.trace_id = trace_id
        self.input: Any = None
        self.output: Any = None
        self.metadata: dict = {}
        self.started = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started) * 1000


class Tracer:
    """No-op base tracer (tracing=off)."""

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex[:16]

    @contextmanager
    def span(self, name: str, input: Any = None, **metadata: Any) -> Iterator[Span]:
        s = Span(name, self.trace_id)
        s.input = input
        s.metadata = metadata
        try:
            yield s
        finally:
            self._export(s)

    def _export(self, span: Span) -> None:  # no-op
        pass

    def flush(self) -> None:
        pass


class LocalTracer(Tracer):
    """Appends spans as JSON lines under data/traces/<date>.jsonl."""

    def __init__(self, traces_dir: Path) -> None:
        super().__init__()
        traces_dir.mkdir(parents=True, exist_ok=True)
        self._path = traces_dir / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"

    def _export(self, span: Span) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": span.trace_id,
            "span": span.name,
            "elapsed_ms": round(span.elapsed_ms, 1),
            "input": _safe(span.input),
            "output": _safe(span.output),
            "metadata": _safe(span.metadata),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class LangfuseTracer(Tracer):
    """Langfuse Cloud exporter (v3+/v4 OTel SDK; reads LANGFUSE_* env vars).

    All node spans are children of one root span so a query shows up as a
    single trace in the Langfuse UI.
    """

    def __init__(self) -> None:
        super().__init__()
        from langfuse import Langfuse

        self._lf = Langfuse()
        if not self._lf.auth_check():
            raise RuntimeError("langfuse auth_check failed (check LANGFUSE_* keys)")
        self._root = self._lf.start_span(name="citesight")
        self.trace_id = self._root.trace_id

    def _export(self, span: Span) -> None:
        try:
            child = self._root.start_span(name=span.name, input=_safe(span.input))
            child.update(
                output=_safe(span.output),
                metadata={**_safe(span.metadata), "elapsed_ms": round(span.elapsed_ms, 1)},
            )
            child.end()
        except Exception as exc:  # tracing must never break the app
            logger.warning("langfuse export failed: %s", exc)

    def flush(self) -> None:
        try:
            self._root.end()
            self._lf.flush()
        except Exception as exc:
            logger.warning("langfuse flush failed: %s", exc)


def _safe(obj: Any, limit: int = 4000) -> Any:
    try:
        text = json.dumps(obj, default=str)
    except (TypeError, ValueError):
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return json.loads(text) if text.startswith(("{", "[", '"')) else obj


def get_tracer(settings: Settings, purpose: str = "app") -> Tracer:
    """purpose='eval' is forced onto the local exporter to protect quota."""
    mode = settings.tracing
    if purpose == "eval" and mode == "langfuse":
        logger.info("eval traces forced to local exporter (quota protection)")
        mode = "local"
    if mode == "off":
        return Tracer()
    if mode == "langfuse":
        try:
            return LangfuseTracer()
        except Exception as exc:
            logger.warning("langfuse unavailable (%s); falling back to local", exc)
            mode = "local"
    return LocalTracer(settings.traces_dir)
