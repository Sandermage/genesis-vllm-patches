# SPDX-License-Identifier: Apache-2.0
"""PN72 — coarse first-token frequency filter for vllm ngram drafts (numpy-only, offline-testable).

**Algorithm (as actually implemented):** post-filter vllm's already-produced
drafts by checking that the FIRST draft token has been observed at least
`MIN_OBSERVATIONS` times in the recent context window. The full
suffix→continuation conditioning from llama.cpp is NOT replicated — we only
count first-token occurrences. This is a coarse safety net, not a true
ngram-frequency filter. (Audit A-05 honesty fix 2026-05-06: docstring
hierarchy made explicit.)

Inspired by llama.cpp's `draft_min_sample_size_*` + `draft_min_percent_*`
heuristic in `common/ngram-cache.cpp`. Their approach: emit a draft only
when the candidate continuation has been observed N≥4 times AND is the
dominant choice ≥75% of the time. This rejects spurious matches that vllm's
"longest-match" ngram drafter happily accepts (e.g., 2-token suffix `<<` in
Qwen3-coder chat templates that triggers tool-call corruption per
noonghunna club-3090 #16 / Genesis v7.13 BREAKTHROUGH report).

We can NOT replicate llama.cpp's full ngram cache (their cache is built
during prefill via `common_ngram_cache_update` — invasive integration).
But we CAN approximate the dominant signal cheaply: post-filter vllm's
already-produced drafts by checking that the predicted FIRST draft token
appears at least N times in the recent window of context. If not, the
match was likely spurious — reject the draft (return empty).

Tradeoff: coarser than full ngram-frequency check (we look at token
frequency, not full-suffix-frequency). But:
  - O(window_size) per request — fast (default window=1024)
  - Conservative — never invents drafts, only rejects weak ones
  - Composes additively with P70 hardcoded `prompt_lookup_min=8`
  - No torch / numba / GPU dependency — pure numpy, testable offline

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference: noonghunna/club-3090#16, Genesis v7.13 BREAKTHROUGH.
"""
from __future__ import annotations

import os

import numpy as np


# ─── Defaults (mirror llama.cpp lax-mode for ngram_max=4) ─────────────


_DEFAULT_MIN_OBS = 4         # llama.cpp draft_min_sample_size_lax[3]
_DEFAULT_WINDOW = 1024       # locality bound — recent context only

_ENV_ENABLE = "GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER"
_ENV_MIN_OBS = "GENESIS_PN72_MIN_OBSERVATIONS"
_ENV_WINDOW = "GENESIS_PN72_FREQUENCY_WINDOW"


# ─── Env-aware getters (defensive against bad input) ──────────────────


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    """Read an int env var. Falls back to default on missing/invalid/<min."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        if n < min_value:
            return default
        return n
    except (TypeError, ValueError):
        return default


def is_enabled() -> bool:
    """Master gate — does the operator want PN72 active?"""
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_min_observations() -> int:
    """Minimum count of first-draft-token in window for accept."""
    return _env_int(_ENV_MIN_OBS, _DEFAULT_MIN_OBS, min_value=1)


def get_frequency_window() -> int:
    """Window size (in tokens) over which we count first-token frequency."""
    return _env_int(_ENV_WINDOW, _DEFAULT_WINDOW, min_value=1)


# ─── Decision primitive ───────────────────────────────────────────────


def should_accept_draft(
    context: np.ndarray,
    first_draft_token: int,
    window: int = _DEFAULT_WINDOW,
    min_obs: int = _DEFAULT_MIN_OBS,
) -> bool:
    """Return True iff `first_draft_token` appears >= `min_obs` times in the
    last `window` tokens of `context`.

    Defensive: never raises on weird inputs (empty context, window > len, etc.)
    """
    if min_obs <= 0:
        # Filter disabled — accept everything (used by tests + ops escape hatch)
        return True
    n = int(context.shape[0]) if hasattr(context, "shape") else len(context)
    if n <= 0:
        return False
    # Window clamped to actual context size — never index out of range
    win = min(window, n)
    recent = context[-win:]
    # `==` on numpy gives bool array; `count_nonzero` is the fast count
    count = int(np.count_nonzero(recent == first_draft_token))
    return count >= min_obs


# ─── Batch wrapper (matches NgramProposer.propose return shape) ───────


def filter_drafts_by_frequency(
    drafts: list[list[int]],
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    window: int = _DEFAULT_WINDOW,
    min_obs: int = _DEFAULT_MIN_OBS,
) -> list[list[int]]:
    """Post-filter a batch of ngram drafts by per-request first-token frequency.

    Args:
        drafts: list of draft token lists (one per request)
        num_tokens_no_spec: 1-D array, valid tokens count per request
        token_ids_cpu: 2-D array (batch, max_model_len) of context tokens
        window: how far back to look
        min_obs: minimum count to accept

    Returns: new list of drafts (empty list = rejected). Inputs unmodified.
    """
    out: list[list[int]] = []
    rows = token_ids_cpu.shape[0] if token_ids_cpu.ndim == 2 else 0
    cols = token_ids_cpu.shape[1] if token_ids_cpu.ndim == 2 else 0
    for idx, draft in enumerate(drafts):
        # Empty draft — nothing to filter, fast pass-through
        if not draft:
            out.append([])
            continue
        # Resolve context length defensively (clamp negative, clamp > cols)
        if idx < len(num_tokens_no_spec):
            ntok = int(num_tokens_no_spec[idx])
        else:
            ntok = 0
        ntok = max(0, min(ntok, cols))
        if ntok == 0 or idx >= rows:
            # No context to verify against → treat as low-confidence, reject
            out.append([])
            continue
        ctx = token_ids_cpu[idx, :ntok]
        if should_accept_draft(ctx, draft[0], window=window, min_obs=min_obs):
            # Defensive copy — caller may not expect us to return shared refs
            out.append(list(draft))
        else:
            out.append([])
    return out


# ─── Diagnostic counters (for `genesis doctor` / metrics endpoints) ────


_STATS = {
    "filter_calls": 0,
    "drafts_in": 0,
    "drafts_kept": 0,
    "drafts_rejected": 0,
    "errors": 0,
}


def get_stats() -> dict[str, int]:
    """Return current counters (copy)."""
    return dict(_STATS)


def reset_stats() -> None:
    """Reset counters (for tests)."""
    for k in _STATS:
        _STATS[k] = 0


def record_filter_call(in_count: int, kept: int, errors: int = 0) -> None:
    """Bookkeeping helper — wiring layer calls this after each filter pass."""
    _STATS["filter_calls"] += 1
    _STATS["drafts_in"] += in_count
    _STATS["drafts_kept"] += kept
    _STATS["drafts_rejected"] += in_count - kept
    if errors:
        _STATS["errors"] += errors


# ─── Convenience: read env once + apply (used by wiring) ──────────────


def filter_with_env(
    drafts: list[list[int]],
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
) -> list[list[int]]:
    """Read tunables from env and apply. Wrapper for the wiring layer."""
    return filter_drafts_by_frequency(
        drafts, num_tokens_no_spec, token_ids_cpu,
        window=get_frequency_window(),
        min_obs=get_min_observations(),
    )


__all__ = [
    "should_accept_draft",
    "filter_drafts_by_frequency",
    "filter_with_env",
    "is_enabled",
    "get_min_observations",
    "get_frequency_window",
    "get_stats",
    "reset_stats",
    "record_filter_call",
]
