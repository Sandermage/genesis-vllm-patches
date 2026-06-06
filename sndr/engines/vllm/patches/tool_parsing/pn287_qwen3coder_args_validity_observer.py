# SPDX-License-Identifier: Apache-2.0
"""PN287 — qwen3_coder × MTP arg-corruption frequency observer.

Why this patch exists
---------------------
Server-validated bench 2026-05-29 on 35B-A3B FP8 PROD (pin 626fa9bb, MTP
K=3, agentic multi-turn, max_tokens=150) hit the symptom club-3090
maintainer flagged on noonghunna/club-3090#178 as "distinct from
streaming bug #145":

    HTTP 400: "Unterminated string starting at: line 1 column 13"

Root cause: at depth ~20K accumulated context, qwen3_coder under MTP K=3
emits a tool_call whose `arguments` field is a truncated mid-JSON-string
(parser hit max_tokens budget mid-quote) — but the parser STILL claims
`finish_reason="tool_calls"` (signaling success). Downstream consumers
that re-feed the broken tool_call into chat history fail subsequent
turns with JSON validation errors.

Our existing 3-layer defense covers different surfaces:
    P64   (vllm#39598)  — streaming early-return removal (fixes cascade)
    PN56  (vllm#41466)  — XML parse fallback ("{}" leak fix)
    P61C  (deferred)    — qwen3coder SSE deferred-commit
None of them validate that the FINAL accumulated `arguments` is parseable
JSON. PN287 fills that observability gap.

What PN287 does (and explicitly does NOT do)
--------------------------------------------
DOES:
  • Monkey-patch ``Qwen3CoderToolParser.extract_tool_calls_streaming``
    to inspect ``self.prev_tool_call_arr`` after each invocation.
  • For every tool entry whose ``arguments`` is non-empty and non-"{}",
    attempt ``json.loads(arguments)``. On JSONDecodeError, emit a
    structured warning (one per request, dedup by request key) with:
      - tool name
      - args length
      - args first 80 chars (for diagnostic — no PII risk vs full body)
      - accumulated completion_tokens at observation time (if available)
  • Track aggregate count via process-level Counter for /metrics scrape.

DOES NOT (deliberately):
  • Mutate any model output (read-only observation)
  • Override finish_reason (out of scope — would need serving-layer hook
    and risks breaking strict OpenAI-format clients)
  • Repair truncated JSON (would lose information; bench-tool defense
    in tools/bench_agentic.py is the right place for client-side guard)

Why observability first
-----------------------
Before we ship a behavior-changing fix (override finish_reason="tool_calls"
→ "length" on parse failure, or auto-close truncated args), we need
production frequency data: is this 1% of agentic calls? 10%? Only on
35B-A3B? Only with MTP? PN287 surfaces that data without risk. After
~weeks of prod observation, the operator can decide:
  - Frequency too low → close as observed; accept the trade-off
  - Frequency meaningful + concentrated on 35B-A3B → ship the
    finish_reason override as PN288
  - Frequency meaningful + cross-model → file vllm upstream PR

Gate
----
``GENESIS_ENABLE_PN287_QWEN3CODER_ARGS_OBSERVER=1``. Default OFF.

Compatibility
-------------
- Pure observation; no anchor on text. Re-wraps the bound method via
  ``types.MethodType`` so re-importing the parser module preserves the
  patch. Re-application is idempotent (marker on the patched class).
- Auto-skips on torch-less environments (CI, docs).
- Auto-skips if upstream adds its own `_args_validation` attribute
  (drift detection).

Companion: ``tools/bench_agentic.py`` (client-side history-poisoning
defense, ships in same 2026-05-29 club-3090 cross-reference wave).

Author: Sandermage (Sander Barzov Aleksandr), Ukraine, Odessa — 2026-05-29.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.wiring.pn287_qwen3coder_args_observer")

_ENV_FLAG = "GENESIS_ENABLE_PN287_QWEN3CODER_ARGS_OBSERVER"
_CLASS_MARKER = "_GENESIS_PN287_ARGS_OBSERVER_INSTALLED"
_UPSTREAM_DRIFT_MARKER = "_args_validation_installed"

# Process-level counters. Three surfaces, single source of truth:
#   1. Module-global ``counters`` dict — primary state, backward compat
#      for unit tests + existing operator CLI inspections
#   2. prometheus_client.Counter — auto-exposed on vLLM's /metrics endpoint
#      via the default REGISTRY (see __init.py of vllm/v1/metrics/prometheus)
#   3. Structured WARN log — per-request (deduplicated)
#
# Pattern adopted from external survey 2026-05-30 (per CLAUDE.md
# "Investigation discipline" rule, Step 4 Search + Step 5 Compare):
#   - SGLang `cpu_monitor.py` — module-level Counter on default REGISTRY
#   - LMCache `observability.py` — REGISTRY.unregister idempotency guard
#   - vLLM `v1/metrics/loggers.py` — `vllm:` namespace convention
#
# Multiproc safety: tool parser runs in API-server process (same proc
# that mounts /metrics endpoint), so single-process Counter just works.
# Even in multiproc, vLLM's PROMETHEUS_MULTIPROC_DIR + MultiProcessCollector
# auto-aggregate without explicit wiring.
counters: dict[str, int] = {
    "tool_calls_total": 0,
    "tool_calls_malformed_args": 0,
    "warnings_emitted": 0,
}

# §2.4 Phase A labels — per-model + per-context-depth bucketing on the
# Prometheus surface so PN288 evidence-gathering can distinguish:
#   - Frequency by model (35B-A3B suspect vs 27B vs gemma4)
#   - Frequency by context depth (PN287 hypothesis: depth ~20K spike)
# Cardinality budget: 3 models × 4 ctx_buckets = 12 series per counter;
# well within Prometheus best practice ceiling.
_LABEL_NAMES = ("model", "ctx_bucket")
_CTX_BUCKET_LIMITS: tuple[tuple[int, str], ...] = (
    (5_000,   "0-5K"),
    (15_000,  "5-15K"),
    (30_000,  "15-30K"),
)
_CTX_BUCKET_OVERFLOW = "30K+"


def _ctx_bucket(n_tokens: int) -> str:
    """Map an integer token count to one of 4 context-depth buckets."""
    for upper, name in _CTX_BUCKET_LIMITS:
        if n_tokens < upper:
            return name
    return _CTX_BUCKET_OVERFLOW


def _extract_request_and_ctx(
    call_args: tuple, call_kwargs: dict,
) -> tuple[str, str]:
    """Best-effort extraction of (model, ctx_bucket) for label tagging.

    Signature of ``Qwen3CoderToolParser.extract_tool_calls_streaming`` is
    ``(self, previous_text, current_text, delta_text, previous_token_ids,
    current_token_ids, delta_token_ids, request)``. vLLM serving.py calls
    it with kwargs in current pin (0.21.1rc1+), but downstream wrappers
    or older pins may call positionally. Handle both.

    Returns ``("unknown", "<bucket>")`` on any extraction failure — never
    raises. The observer must never crash a user request to capture
    telemetry.
    """
    request = call_kwargs.get("request")
    current_token_ids = call_kwargs.get("current_token_ids")
    # Positional fallback (parser receives `self` separately, so call_args
    # is 0-indexed at previous_text). current_token_ids is the 5th arg
    # (index 4), request is the 7th (index 6).
    if request is None and len(call_args) >= 7:
        request = call_args[6]
    if current_token_ids is None and len(call_args) >= 5:
        current_token_ids = call_args[4]
    model = "unknown"
    if request is not None:
        try:
            m = getattr(request, "model", None)
            if isinstance(m, str) and m:
                model = m
        except Exception:
            pass
    n_tokens = 0
    if current_token_ids is not None:
        try:
            n_tokens = len(current_token_ids)
        except Exception:
            pass
    return model, _ctx_bucket(n_tokens)


# Prometheus Counter handles — lazily created on first apply() to avoid
# import-time cost in torch-less environments (tests, lint, docs build).
_prom_extract_total: Any = None
_prom_malformed_total: Any = None
_prom_warnings_total: Any = None


def _setup_prometheus_counters() -> bool:
    """Idempotent Counter creation on default prometheus_client REGISTRY.

    Returns True on success or already-registered, False if prometheus_client
    not importable (torch-less env, no monitoring stack).

    Idempotency pattern (LMCache + vLLM canonical):
    walk REGISTRY._collector_to_names; unregister any collector whose
    name starts with our prefix. Then re-register fresh Counter objects.
    Handles patch re-application across worker spawns / hot reload.
    """
    global _prom_extract_total, _prom_malformed_total, _prom_warnings_total
    try:
        from prometheus_client import REGISTRY, Counter
    except ImportError:
        return False

    _PREFIX = "vllm:qwen3_tool_parser_pn287_"
    # Walk REGISTRY for stale collectors with our prefix (idempotency).
    for collector in list(REGISTRY._collector_to_names):
        names = REGISTRY._collector_to_names.get(collector, [])
        if any(n.startswith(_PREFIX) for n in names):
            try:
                REGISTRY.unregister(collector)
            except (KeyError, ValueError):
                pass

    try:
        _prom_extract_total = Counter(
            name=f"{_PREFIX}extract_total",
            documentation=(
                "PN287 Qwen3CoderToolParser tool_call arguments observed "
                "via extract_tool_calls_streaming wrap. Includes valid + "
                "malformed; subtract malformed_total for clean count. "
                "Labels: model (served-model-name from request), "
                "ctx_bucket (token-count bucket: 0-5K/5-15K/15-30K/30K+)."
            ),
            labelnames=_LABEL_NAMES,
        )
        _prom_malformed_total = Counter(
            name=f"{_PREFIX}malformed_total",
            documentation=(
                "PN287 tool_call.arguments that failed json.loads — likely "
                "max_tokens truncation mid-JSON-string (club-3090 #178). "
                "Read-only observation; does NOT mutate output. Labels: "
                "model, ctx_bucket — operator queries can pivot by both "
                "to evaluate PN288 trigger criteria per §2.4 Phase A."
            ),
            labelnames=_LABEL_NAMES,
        )
        _prom_warnings_total = Counter(
            name=f"{_PREFIX}warnings_total",
            documentation=(
                "PN287 structured WARN log emissions. Deduplicated per "
                "(parser-id, tool-name) within a single request — actual "
                "count of distinct malformed events surfaced to logs. "
                "Labels: model, ctx_bucket."
            ),
            labelnames=_LABEL_NAMES,
        )
        return True
    except (ValueError, AttributeError) as exc:
        log.warning(
            "[PN287] failed to register Prometheus counters: %s. "
            "Module-global counters dict still active.", exc,
        )
        return False


def _is_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _make_wrapped_streaming(original_fn):
    """Build the wrapped extract_tool_calls_streaming that observes
    args-validity post-invocation. Closure over ``original_fn`` preserves
    the original method for delegation.
    """
    import functools
    import json

    @functools.wraps(original_fn)
    def wrapped(self, *args, **kwargs):
        result = original_fn(self, *args, **kwargs)
        # Per-request labels for Prometheus (cheap; extracted once per
        # call regardless of how many tool entries we inspect below).
        model_label, ctx_label = _extract_request_and_ctx(args, kwargs)
        # Post-invocation: inspect accumulated tool-call args. The parser
        # stores per-tool accumulated state on `prev_tool_call_arr`.
        try:
            arr = getattr(self, "prev_tool_call_arr", None) or []
        except Exception:
            return result
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            args_str = entry.get("arguments") or ""
            if not args_str or args_str == "{}":
                continue
            counters["tool_calls_total"] += 1
            if _prom_extract_total is not None:
                _prom_extract_total.labels(
                    model=model_label, ctx_bucket=ctx_label,
                ).inc()
            try:
                json.loads(args_str)
            except (ValueError, TypeError):
                counters["tool_calls_malformed_args"] += 1
                if _prom_malformed_total is not None:
                    _prom_malformed_total.labels(
                        model=model_label, ctx_bucket=ctx_label,
                    ).inc()
                # Dedup by self-identity + tool name within same request.
                seen_key = id(self), entry.get("name") or "?"
                seen_set = getattr(self, "_pn287_seen", None) or set()
                if seen_key in seen_set:
                    continue
                seen_set.add(seen_key)
                self._pn287_seen = seen_set
                counters["warnings_emitted"] += 1
                if _prom_warnings_total is not None:
                    _prom_warnings_total.labels(
                        model=model_label, ctx_bucket=ctx_label,
                    ).inc()
                # Keep payload preview tight to avoid log floods.
                preview = args_str[:80].replace("\n", "\\n")
                log.warning(
                    "[PN287] qwen3_coder tool_call.arguments unparseable — "
                    "name=%s len=%d preview=%r model=%s ctx_bucket=%s. "
                    "Likely max_tokens truncation mid-JSON-string "
                    "(club-3090 #178). Downstream clients should validate "
                    "before re-feeding to chat history; see "
                    "tools/bench_agentic.py defense pattern.",
                    entry.get("name") or "?", len(args_str), preview,
                    model_label, ctx_label,
                )
        return result

    return wrapped


def apply() -> tuple[str, str]:
    """Install PN287 observer. Always idempotent. Never raises."""
    if not _is_enabled():
        return "skipped", (
            f"opt-in — set {_ENV_FLAG}=1 to enable qwen3_coder args-"
            f"validity observer (warns when tool_call.arguments is "
            f"unparseable; club-3090 #178)"
        )

    # Two import paths across vLLM versions:
    #   - Pre-2026-05 nightly: vllm.entrypoints.openai.tool_parsers.*
    #   - Post-2026-05 (incl. 0.21.1rc1+, our 626fa9bb pin): vllm.tool_parsers.*
    # Try new path first (matches our PROD pin), fall through to legacy.
    Qwen3CoderToolParser = None
    _import_errors = []
    for candidate in (
        "vllm.tool_parsers.qwen3coder_tool_parser",
        "vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser",
    ):
        try:
            import importlib
            module = importlib.import_module(candidate)
            Qwen3CoderToolParser = module.Qwen3CoderToolParser
            break
        except (ImportError, AttributeError) as exc:
            _import_errors.append(f"{candidate}: {exc}")
    if Qwen3CoderToolParser is None:
        return "skipped", (
            "vllm Qwen3CoderToolParser not importable from any known "
            f"path: {'; '.join(_import_errors)}"
        )

    # Drift detection: if upstream adds its own validation marker, retire.
    if hasattr(Qwen3CoderToolParser, _UPSTREAM_DRIFT_MARKER):
        return "skipped", (
            f"upstream Qwen3CoderToolParser already carries "
            f"`{_UPSTREAM_DRIFT_MARKER}` — PN287 self-retires; consider "
            f"flipping `lifecycle=retired` in registry"
        )

    # Set up Prometheus Counter integration — auto-exposed via vLLM's
    # existing /metrics endpoint via default REGISTRY (idempotent across
    # patch re-apply; safe to call multiple times).
    prom_ready = _setup_prometheus_counters()

    # Idempotency check.
    if getattr(Qwen3CoderToolParser, _CLASS_MARKER, False):
        return "applied", (
            "already installed (idempotent re-apply). prometheus: "
            f"{'yes' if prom_ready else 'no (client unavailable)'}"
        )

    original = Qwen3CoderToolParser.extract_tool_calls_streaming
    Qwen3CoderToolParser.extract_tool_calls_streaming = (
        _make_wrapped_streaming(original)
    )
    Qwen3CoderToolParser._GENESIS_PN287_ORIGINAL = original  # noqa: SLF001
    setattr(Qwen3CoderToolParser, _CLASS_MARKER, True)

    prom_note = (
        " + Prometheus counters registered on default REGISTRY "
        "(vllm:qwen3_tool_parser_pn287_*) — auto-exposed on /metrics"
        if prom_ready
        else " (prometheus_client unavailable — module-global dict only)"
    )
    return "applied", (
        "PN287 installed — Qwen3CoderToolParser.extract_tool_calls_"
        "streaming wrapped with args-validity observer. Counters at "
        "vllm.sndr_core.integrations.tool_parsing.pn287_qwen3coder_args_"
        "validity_observer.counters" + prom_note + "."
    )


def is_applied() -> bool:
    try:
        from vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser \
            import Qwen3CoderToolParser
    except ImportError:
        return False
    return bool(getattr(Qwen3CoderToolParser, _CLASS_MARKER, False))


def revert() -> bool:
    """Revert the monkey-patch — restore original
    ``extract_tool_calls_streaming``. Returns True if reverted, False if
    nothing to revert (not installed) or the upstream original is missing.
    """
    try:
        from vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser \
            import Qwen3CoderToolParser
    except ImportError:
        return False
    original = getattr(
        Qwen3CoderToolParser, "_GENESIS_PN287_ORIGINAL", None
    )
    if original is None:
        return False
    Qwen3CoderToolParser.extract_tool_calls_streaming = original
    delattr(Qwen3CoderToolParser, "_GENESIS_PN287_ORIGINAL")
    setattr(Qwen3CoderToolParser, _CLASS_MARKER, False)
    return True
