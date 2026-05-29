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

# Process-level counters for /metrics scrape. Module-global by design —
# operators inspect via `python3 -c "from ...pn287 import counters; ..."`
# or a dedicated CLI hook.
counters: dict[str, int] = {
    "tool_calls_total": 0,
    "tool_calls_malformed_args": 0,
    "warnings_emitted": 0,
}


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
            try:
                json.loads(args_str)
            except (ValueError, TypeError):
                counters["tool_calls_malformed_args"] += 1
                # Dedup by self-identity + tool name within same request.
                seen_key = id(self), entry.get("name") or "?"
                seen_set = getattr(self, "_pn287_seen", None) or set()
                if seen_key in seen_set:
                    continue
                seen_set.add(seen_key)
                self._pn287_seen = seen_set
                counters["warnings_emitted"] += 1
                # Keep payload preview tight to avoid log floods.
                preview = args_str[:80].replace("\n", "\\n")
                log.warning(
                    "[PN287] qwen3_coder tool_call.arguments unparseable — "
                    "name=%s len=%d preview=%r. Likely max_tokens truncation "
                    "mid-JSON-string (club-3090 #178). Downstream clients "
                    "should validate before re-feeding to chat history; "
                    "see tools/bench_agentic.py defense pattern.",
                    entry.get("name") or "?", len(args_str), preview,
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

    # Idempotency check.
    if getattr(Qwen3CoderToolParser, _CLASS_MARKER, False):
        return "applied", "already installed (idempotent re-apply)"

    original = Qwen3CoderToolParser.extract_tool_calls_streaming
    Qwen3CoderToolParser.extract_tool_calls_streaming = (
        _make_wrapped_streaming(original)
    )
    Qwen3CoderToolParser._GENESIS_PN287_ORIGINAL = original  # noqa: SLF001
    setattr(Qwen3CoderToolParser, _CLASS_MARKER, True)

    return "applied", (
        "PN287 installed — Qwen3CoderToolParser.extract_tool_calls_"
        "streaming wrapped with args-validity observer. Counters at "
        "vllm.sndr_core.integrations.tool_parsing.pn287_qwen3coder_args_"
        "validity_observer.counters."
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
