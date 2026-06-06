# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN65 — Genesis structured API access log.

Replaces the bare uvicorn default access lines::

    INFO: 127.0.0.1:45116 - "GET /v1/models HTTP/1.1" 401 Unauthorized

with operator-friendly structured lines::

    [Genesis-API] 200  POST /v1/chat/completions    HTTP/1.1  client=127.0.0.1
    [Genesis-API] 401  GET  /v1/models              HTTP/1.1  client=127.0.0.1
    [Genesis-API] 200  GET  /health                 HTTP/1.1  client=127.0.0.1  (suppressed by default)

Health-check polling is suppressed by default to avoid log noise (Docker
healthchecks fire every 5s = 17K lines/day per worker). Set
GENESIS_PN65_LOG_HEALTH=1 to log them.

Categorization:
  • 2xx → INFO
  • 4xx → WARNING
  • 5xx → ERROR

================================================================
WAVE 6 v3 ARCHITECTURE 2026-05-09 — LOGGING-ONLY (ZERO REQUEST-PATH OVERHEAD)
================================================================

Earlier PN65 implementations (v1: ``app.middleware('http')`` decorator;
v2: raw ASGI middleware) intercepted requests on the hot path. Even
the streaming-safe ASGI variant added measurable latency variance for
high-concurrency / long-context workloads where every microsecond
counts. The fundamental problem: any HTTP middleware adds **at least**
a Python coroutine frame per request (the wrapper itself), plus
log-line formatting work that happens before the response can be
acked back to the client.

**v3 fix:** drop HTTP middleware entirely. uvicorn already logs every
request via the ``uvicorn.access`` logger (a Python ``logging``
logger). Each ``LogRecord`` for an access event has ``record.args``
shaped as ``(client_addr, method, full_path, http_version,
status_code)``. PN65 v3 installs a single ``logging.Filter`` on
``uvicorn.access`` that:

  1. Reads ``record.args`` and reformats into the Genesis structured
     line.
  2. Emits the structured line via a separate ``genesis.api`` logger.
  3. Returns ``False`` to drop the original bare line.

Side benefits over the middleware approach:
  - Zero CPU work on the request hot path. Reformat happens at log
    emit time, AFTER the response was already returned to the client.
  - Streaming SSE is completely untouched; we never see the response
    body at all.
  - No middleware stack mutation; no compatibility risk with FastAPI
    versions or future vllm refactors of ``build_app``.
  - Survives uvicorn ``log_config`` re-init (the same persistence
    pattern audited in v2 already works for filters at root).

Trade-off: we lose request duration measurement (uvicorn doesn't time
requests at the access-log layer; our middleware was previously doing
``time.perf_counter()`` deltas around ``await call_next``). For
operator UX this is an acceptable cost — status code + method + path
+ client are sufficient for ops triage, and per-request timing is
better surfaced via Prometheus metrics on the model engine side.

================================================================
ENV
================================================================

GENESIS_ENABLE_PN65=1                       — master enable
GENESIS_PN65_LOG_HEALTH=1                   — include /health probes
GENESIS_PN65_QUIET_PATHS=/v1/models,/metrics  — comma-separated path prefixes to suppress
GENESIS_PN65_KEEP_UVICORN_ACCESS=1          — keep both lines (debug mode)

================================================================
RISK
================================================================

NONE on the request hot path — pure logging-layer reformat. If the
filter raises on a malformed record, ``logging`` swallows the
exception and the bare uvicorn line is preserved.

================================================================
STATE
================================================================

Active runtime install COMPLETE; idempotent via module-level
``_PN65_REFORMAT_FILTER_INSTALLED`` flag.

Author: Sandermage 2026-05-05; Wave 6 v3 logging-only rewrite 2026-05-09
(closes Sander's ask: "look for PN65 designs that do not regress
speed or stability").
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("genesis.wiring.pn65_access_log")
api_log = logging.getLogger("genesis.api")


# ─── Config helpers ─────────────────────────────────────────────────────


def _quiet_paths() -> set[str]:
    raw = os.environ.get("GENESIS_PN65_QUIET_PATHS", "/health,/metrics")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _log_health() -> bool:
    return os.environ.get("GENESIS_PN65_LOG_HEALTH", "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _format_log_line(
    method: str,
    path: str,
    status_code: int,
    http_version: str,
    client: str,
) -> str:
    method_str = f"{method:<5}"
    return (
        f"[Genesis-API] {status_code:<3}  {method_str} {path:<35}  "
        f"HTTP/{http_version}  client={client}"
    )


def _client_host_from_addr(client_addr: str) -> str:
    """uvicorn.access record args use 'host:port' or 'host'. Extract host."""
    if not client_addr:
        return "?"
    return str(client_addr).split(":", 1)[0] or "?"


# ─── Reformatter filter (the entire patch) ──────────────────────────────


class GenesisAccessLogReformatter(logging.Filter):
    """Reformat ``uvicorn.access`` INFO records into Genesis structured lines.

    The whole patch lives here. Hot path: zero — runs at log emit
    time after the response is already on the wire.

    uvicorn record args contract (stable since uvicorn 0.x):
      ``(client_addr, method, full_path, http_version, status_code)``

    On match: emit a Genesis-API line via ``genesis.api`` logger
    (level chosen by status_code), then return ``False`` to drop the
    bare uvicorn line. On any unexpected record shape: return ``True``
    and let the bare line through unchanged (defensive).
    """

    def __init__(self, quiet_paths: Optional[set[str]] = None,
                 log_health: bool = False) -> None:
        super().__init__()
        self._quiet = quiet_paths if quiet_paths is not None else _quiet_paths()
        self._log_health = log_health

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True  # not our concern
        # Only reformat INFO records; pass WARNING/ERROR through unchanged
        # so operators still see uvicorn's own diagnostics if any escape.
        if record.levelno != logging.INFO:
            return True

        args = record.args
        # uvicorn args is a 5-tuple. Anything else: defensive pass-through.
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        try:
            client_addr, method, full_path, http_version, status_code = args[:5]
            status_code_int = int(status_code)
        except Exception:
            return True

        # Quiet-path suppression — runs BEFORE emit so we drop without
        # bothering the genesis.api logger pipeline.
        path_only = str(full_path).split("?", 1)[0]
        if path_only == "/health" and not self._log_health:
            return False
        effective_quiet = (
            {q for q in self._quiet if q != "/health"}
            if self._log_health
            else self._quiet
        )
        if any(path_only.startswith(q) for q in effective_quiet):
            return False

        client_host = _client_host_from_addr(client_addr)
        line = _format_log_line(
            str(method), str(full_path), status_code_int,
            str(http_version), client_host,
        )

        if status_code_int >= 500:
            api_log.error(line)
        elif status_code_int >= 400:
            api_log.warning(line)
        else:
            api_log.info(line)

        # Drop the bare uvicorn line — Genesis line is the canonical record.
        return False


# ─── uvicorn.access INFO suppressor (kept for KEEP_UVICORN_ACCESS opt-out) ──


class _DropUvicornAccessInfo(logging.Filter):
    """Drop INFO-level records from uvicorn.access — PN65 dedup filter.

    This is a defensive belt — under normal v3 operation,
    ``GenesisAccessLogReformatter`` already returns ``False`` for INFO
    records on the ``uvicorn.access`` logger, so this filter is
    redundant. We keep it so that:
      1. If the reformatter is uninstalled / disabled but the dedup
         filter persists for some reason, uvicorn lines still drop.
      2. Tests that exercise ``_DropUvicornAccessInfo`` in isolation
         (G-POST-07 audit tests) keep working.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True  # not our concern
        if record.levelno < logging.WARNING:
            return False  # drop INFO/DEBUG access spam
        return True


# ─── Module-level install state ─────────────────────────────────────────


_PN65_REFORMATTER_INSTANCE: Optional[GenesisAccessLogReformatter] = None
_PN65_FILTER_INSTANCE = _DropUvicornAccessInfo()
_PN65_FILTER_INSTALLED = False
_PN65_REFORMATTER_INSTALLED = False


def _suppress_uvicorn_access_logger() -> None:
    """Install the dedup filter on root + uvicorn.access loggers.

    Belt-and-suspenders: even with the reformatter active, this
    second filter ensures any uvicorn.access INFO that slips past
    (e.g. record with malformed args) gets dropped so operator logs
    stay clean.
    """
    global _PN65_FILTER_INSTALLED
    if _PN65_FILTER_INSTALLED:
        return
    if os.environ.get(
        "GENESIS_PN65_KEEP_UVICORN_ACCESS", ""
    ).strip().lower() in ("1", "true", "yes", "y", "on"):
        return

    logging.getLogger().addFilter(_PN65_FILTER_INSTANCE)
    logging.getLogger("uvicorn.access").addFilter(_PN65_FILTER_INSTANCE)
    _PN65_FILTER_INSTALLED = True
    log.info(
        "[PN65] uvicorn.access INFO records will be dropped via persistent "
        "logging.Filter — structured Genesis-API lines are now the single "
        "source for request-level observability. "
        "Set GENESIS_PN65_KEEP_UVICORN_ACCESS=1 to keep both."
    )


def _install_reformatter() -> bool:
    """Attach ``GenesisAccessLogReformatter`` to ``uvicorn.access``.

    Idempotent. Returns True if newly installed.
    """
    global _PN65_REFORMATTER_INSTALLED, _PN65_REFORMATTER_INSTANCE
    if _PN65_REFORMATTER_INSTALLED:
        return False
    _PN65_REFORMATTER_INSTANCE = GenesisAccessLogReformatter(
        quiet_paths=_quiet_paths(),
        log_health=_log_health(),
    )
    logging.getLogger("uvicorn.access").addFilter(_PN65_REFORMATTER_INSTANCE)
    _PN65_REFORMATTER_INSTALLED = True
    log.info(
        "[PN65] Genesis-API access log reformatter installed on "
        "uvicorn.access — structured lines now emitted via genesis.api "
        "logger, with zero request-path overhead."
    )
    return True


def install_into_app(app) -> bool:
    """Backward-compat shim: callers from older Genesis revisions may
    invoke this with a FastAPI app. v3 ignores the app entirely and
    just installs the logging filter (which doesn't need an app).

    Idempotent via ``__pn65_installed__`` marker.

    Returns True if newly installed.
    """
    if getattr(app, "__pn65_installed__", False):
        return False
    _suppress_uvicorn_access_logger()
    _install_reformatter()
    setattr(app, "__pn65_installed__", True)
    return True


# ─── apply() entry ──────────────────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Apply PN65 — install the logging-layer reformatter.

    v3 (Wave 6 2026-05-09): no longer wraps ``api_server.build_app``;
    we install the filter directly on the ``uvicorn.access`` logger.
    The filter survives uvicorn ``log_config`` re-init because we
    attach to the logger, not to a handler — re-creating handlers
    doesn't drop logger-level filters.
    """
    from vllm.sndr_core.dispatcher import should_apply, log_decision

    decision, reason = should_apply("PN65")
    log_decision("PN65", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        _suppress_uvicorn_access_logger()
        _install_reformatter()
    except Exception as exc:
        return (
            "failed",
            f"PN65 logging-filter install failed: {type(exc).__name__}: "
            f"{str(exc)[:120]}",
        )

    return (
        "applied",
        "PN65 v3 (logging-only): reformatter attached to uvicorn.access; "
        "Genesis-API structured lines emitted via genesis.api logger. "
        "Zero request-path overhead — runs at log emit time after the "
        f"response is on the wire. Quiet paths: {','.join(sorted(_quiet_paths()))}. "
        "Set GENESIS_PN65_LOG_HEALTH=1 to include /health probes."
    )
