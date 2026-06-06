# SPDX-License-Identifier: Apache-2.0
"""Structured JSON logging for sndr-platform.

Every log line is a single JSON object with stable keys:

    ts, level, service, version, logger, event, trace_id, ...

This is the format expected by ELK, Datadog, Loki, and similar log
aggregation systems. No prose log messages — every emission is structured.

Engineering principle: never use ``print()``. Always use a logger obtained
from ``logging.getLogger("sndr.<module>")``.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ContextVar lets us inject trace_id without threading it through every
# function signature. Set by middleware/dispatcher; read by the formatter.
_trace_id_var: ContextVar[str | None] = ContextVar("sndr_trace_id", default=None)


def current_trace_id() -> str | None:
    """Return the trace_id for the current execution context, if set."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None) -> Any:
    """Set the trace_id for the current execution context.

    Returns a token that can be passed to ``_trace_id_var.reset(token)`` to
    restore the previous value (used in context managers).
    """
    return _trace_id_var.set(trace_id)


class SndrJsonFormatter(logging.Formatter):
    """Minimal JSON formatter (no external dependencies).

    Single-line JSON output. Stable field order for human-readability:
        ts, level, service, version, logger, event, trace_id, ...extras
    """

    # These fields are present on every logging.LogRecord; we either rename
    # them or use them as the source for our structured fields.
    _STD_FIELDS = (
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message",
    )

    def __init__(self, service: str = "sndr") -> None:
        super().__init__()
        self.service = service
        # Imported lazily to avoid a circular import.
        from sndr.version import __version__
        self.version = __version__

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
            timespec="milliseconds",
        ).replace("+00:00", "Z")

        # The "event" name comes from the first positional arg to log methods:
        #     log.info("patch.applied", extra={...})
        event = record.getMessage()

        out: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "service": self.service,
            "version": self.version,
            "logger": record.name,
            "event": event,
        }

        if (trace_id := current_trace_id()) is not None:
            out["trace_id"] = trace_id

        # Merge extras from logger.X(event, extra={...})
        for key, value in record.__dict__.items():
            if key in self._STD_FIELDS or key.startswith("_"):
                continue
            if key in out:  # do not let extras overwrite stable fields
                continue
            try:
                # Coerce to JSON-safe types.
                json.dumps(value)
                out[key] = value
            except (TypeError, ValueError):
                out[key] = repr(value)

        # Exception info, if present
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)

        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO", service: str = "sndr") -> None:
    """Install SndrJsonFormatter on the root logger.

    Idempotent. Safe to call multiple times; subsequent calls update level
    only.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Find or install a single StreamHandler with our formatter.
    has_handler = False
    for h in root.handlers:
        if isinstance(h.formatter, SndrJsonFormatter):
            h.setLevel(level)
            has_handler = True
            break

    if not has_handler:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(SndrJsonFormatter(service=service))
        handler.setLevel(level)
        root.addHandler(handler)


__all__ = [
    "SndrJsonFormatter",
    "configure_logging",
    "current_trace_id",
    "set_trace_id",
]
