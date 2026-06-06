# SPDX-License-Identifier: Apache-2.0
"""SNDR_EAGLE3_AUX_HIDDEN_001 — Genesis-original EAGLE-3 model-side prep.

EAGLE-3 (arXiv 2503.01840) is the latest in the EAGLE speculative-decode
family. Like EAGLE-1/-2, it relies on auxiliary hidden states from the
target model: a drafter is a small auto-regressive model that consumes
a FUSION of {input embeddings, last hidden state, and AUX hidden states
sampled from intermediate layers of the target}.

vLLM upstream landed EAGLE-3 in V2 ModelRunner via:
  - PR #35029 (initial V2 support, no CUDA graph)
  - PR #35040 (CUDA graph support)
  - PRs #36658 / #39450 / #37512 (Qwen3.5 / Gemma4 / MiniMax-M2)
  - PR #43132 (Qwen3 — still open as of 2026-06)
  - PR #42143 (norm_before_fc config plumbing)

A trained Qwen3.6 EAGLE-3 drafter checkpoint does NOT exist publicly as
of 2026-06-03. When one lands, the wire-up is <1 day (G4_71-style
backend reroute + KV spec config + YAML preset). This patch prepares the
TARGET-MODEL side so we can land the drafter wire-up the moment a
checkpoint exists.

What this patch does (preparation only — no drafter, no spec-decode)
--------------------------------------------------------------------

1. Provides `register_aux_hidden_state_hooks(model, layer_ids)` — a
   safe utility that hooks the requested intermediate layers of any
   Qwen3.x / Llama-family target model and captures their output hidden
   states into a list stored as `model._sndr_eagle3_aux_hidden_states`.

2. Provides `pop_aux_hidden_states(model)` — drains the captured states
   into a stacked tensor `(num_layers, B, S, D)` and clears the buffer.

3. Provides `EAGLE3_AUX_LAYER_IDS_ENV` parsing — reads the comma-
   separated env var `GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS` to override
   the layer-ids list at boot.

4. The apply() function is a NO-OP for the target-model forward path
   when no drafter is wired. The hooks register lazily — only when an
   operator explicitly calls `register_aux_hidden_state_hooks()` from
   a future drafter init code path.

By design: with `default_on=False` AND no caller invoking the helper,
this patch has ZERO runtime cost on the target model. It's a documented,
unit-tested API surface waiting for the drafter to come online.

Operator usage (when checkpoint exists)
---------------------------------------

In the drafter wire-up (future patch, e.g. SNDR_EAGLE3_DRAFTER_001):

    from vllm.sndr_core.integrations.spec_decode.sndr_eagle3_aux_hidden_001 import (
        register_aux_hidden_state_hooks,
        pop_aux_hidden_states,
    )

    # At target-model init:
    register_aux_hidden_state_hooks(
        target_model,
        layer_ids=[0, 8, 16, 24, 31],   # last + early + mid samples
    )

    # In the drafter's propose() inside each step:
    aux = pop_aux_hidden_states(target_model)
    drafter_input = fuse(input_embeds, target_last_hidden, aux)
    ...

Empirical layer-id selection
----------------------------

EAGLE-3 paper (Tab 2) recommends sampling the LAST layer + 3 EARLY
layers (1, 4, 7 on 32-layer models). The exact indices for Qwen3.6
hybrid (30 GDN + 11 attention layers = 41 total) need empirical
determination at checkpoint-ingest time. The env var
`GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS=0,4,8,12,...,40` lets operators
override without re-patching.

References
----------

- Paper: https://arxiv.org/abs/2503.01840
- vLLM PR #35029: V2 ModelRunner support (merged 2026-02-21)
- vLLM PR #43132: Qwen3 support (open 2026-06)
- Genesis G4_71-G4_76: existing drafter routing patterns (reusable
  template for the future drafter wire-up).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.2.0+ Phase 7 EAGLE-3 model-side preparation (no behavior
change without explicit caller; documented API for future drafter wire-up).
"""
from __future__ import annotations

import logging
import os
from threading import RLock
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch  # noqa: F401
    from torch.utils.hooks import RemovableHandle

log = logging.getLogger("genesis.eagle3.aux_hidden")


_ENV_FLAG = "GENESIS_ENABLE_SNDR_EAGLE3_AUX_HIDDEN_001"
_ENV_LAYER_IDS = "GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS"
_GENESIS_MARKER = "_genesis_sndr_eagle3_001_applied"
_AUX_STATES_ATTR = "_sndr_eagle3_aux_hidden_states"
_HANDLES_ATTR = "_sndr_eagle3_aux_handles"


__all__ = [
    "EAGLE3_AUX_LAYER_IDS_ENV",
    "is_enabled",
    "parse_layer_ids_from_env",
    "register_aux_hidden_state_hooks",
    "pop_aux_hidden_states",
    "clear_aux_hidden_state_hooks",
    "apply",
]


EAGLE3_AUX_LAYER_IDS_ENV = _ENV_LAYER_IDS


def is_enabled() -> bool:
    """True when the operator opted in via env flag."""
    return os.environ.get(_ENV_FLAG, "").strip() in ("1", "true", "True")


def parse_layer_ids_from_env() -> list[int]:
    """Parse `GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS=0,4,8,12,...` into a list.

    Returns an empty list when the env is unset or malformed. Invalid
    entries (non-int, negative) are logged and skipped.
    """
    raw = os.environ.get(_ENV_LAYER_IDS, "").strip()
    if not raw:
        return []
    ids: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            log.warning(
                "[SNDR_EAGLE3] %s contains non-int entry %r — skipped",
                _ENV_LAYER_IDS, tok,
            )
            continue
        if v < 0:
            log.warning(
                "[SNDR_EAGLE3] %s contains negative index %d — skipped",
                _ENV_LAYER_IDS, v,
            )
            continue
        ids.append(v)
    return ids


_HOOK_LOCK = RLock()


def _resolve_layers(model: Any):
    """Resolve the iterable of decoder layers on a vLLM target model.

    Different model classes expose layers under different attribute
    paths. We try the canonical hot-paths in priority order and return
    the first one that's indexable.
    """
    # Common vLLM path: model.model.layers
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers
    # Direct: model.layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    # Llama variant: model.model.decoder.layers
    if inner is not None:
        decoder = getattr(inner, "decoder", None)
        if decoder is not None:
            layers = getattr(decoder, "layers", None)
            if layers is not None:
                return layers
    return None


def register_aux_hidden_state_hooks(
    model: Any,
    layer_ids: Optional[list[int]] = None,
) -> int:
    """Register forward hooks on `model`'s intermediate layers so their
    output hidden states are captured into `model._sndr_eagle3_aux_hidden_states`.

    Args:
        model: target-model instance (vLLM Qwen3.x / Llama-class model)
        layer_ids: indices of layers to hook. If None, reads from
            `GENESIS_SNDR_EAGLE3_AUX_LAYER_IDS` env. If still empty,
            no-ops (returns 0).

    Returns:
        Number of hooks successfully registered.

    Idempotent: if already registered, returns the previous count
    without re-registering.
    """
    if layer_ids is None:
        layer_ids = parse_layer_ids_from_env()
    if not layer_ids:
        log.info(
            "[SNDR_EAGLE3] no layer_ids — register_aux_hidden_state_hooks "
            "is a no-op (set %s=... to enable)", _ENV_LAYER_IDS,
        )
        return 0

    with _HOOK_LOCK:
        # Idempotent — if already registered, don't double-hook.
        existing = getattr(model, _HANDLES_ATTR, None)
        if existing is not None and len(existing) > 0:
            log.debug(
                "[SNDR_EAGLE3] hooks already registered (%d) — skipping",
                len(existing),
            )
            return len(existing)

        layers = _resolve_layers(model)
        if layers is None:
            log.warning(
                "[SNDR_EAGLE3] could not resolve layers attribute on %s — "
                "no hooks registered", type(model).__name__,
            )
            return 0

        # Initialize the aux states buffer on the model
        setattr(model, _AUX_STATES_ATTR, [])
        handles: list[Any] = []

        for idx in layer_ids:
            if idx < 0:
                continue
            try:
                layer = layers[idx]
            except (IndexError, KeyError) as e:
                log.warning(
                    "[SNDR_EAGLE3] layer index %d out of range (%s) — "
                    "skipped", idx, e,
                )
                continue
            handle = layer.register_forward_hook(_make_capture_hook(model))
            handles.append(handle)

        setattr(model, _HANDLES_ATTR, handles)
        log.info(
            "[SNDR_EAGLE3] registered %d aux-hidden-state hooks on %s "
            "(layer ids: %s)",
            len(handles), type(model).__name__, layer_ids,
        )
        return len(handles)


def _make_capture_hook(model: Any):
    """Build a forward hook that appends the layer's output[0] to the
    model's aux-states buffer. Returns the hook callable."""
    def _hook(_module, _input, output):
        # output is typically a tuple (hidden_states, ...) for vLLM
        # decoder layers. We capture the first element.
        if isinstance(output, tuple):
            captured = output[0]
        else:
            captured = output
        try:
            getattr(model, _AUX_STATES_ATTR).append(captured)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("[SNDR_EAGLE3] capture failed: %s", e)
        # Return original output unchanged — the hook is observer-only
        return None  # forward-hook return semantics: None → keep output

    return _hook


def pop_aux_hidden_states(model: Any) -> Optional["torch.Tensor"]:
    """Drain the captured aux states into a stacked tensor + clear buffer.

    Returns shape `(num_layers_hooked, B, S, D)` when buffer non-empty,
    else None.

    Pop semantics: after this call, the model's buffer is empty. The
    next forward pass starts fresh.
    """
    states_list = getattr(model, _AUX_STATES_ATTR, None)
    if not states_list:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    with _HOOK_LOCK:
        # Re-check after lock (another thread may have popped)
        states_list = getattr(model, _AUX_STATES_ATTR, None)
        if not states_list:
            return None
        try:
            stacked = torch.stack(states_list, dim=0)
        except Exception as e:
            log.warning("[SNDR_EAGLE3] stack failed: %s", e)
            stacked = None
        # Clear buffer for next forward
        setattr(model, _AUX_STATES_ATTR, [])
    return stacked


def clear_aux_hidden_state_hooks(model: Any) -> int:
    """Remove all registered hooks from `model`. Returns count removed.

    Idempotent. Safe to call on a model that never had hooks.
    """
    with _HOOK_LOCK:
        handles = getattr(model, _HANDLES_ATTR, None)
        if not handles:
            return 0
        count = 0
        for h in handles:
            try:
                h.remove()
                count += 1
            except Exception as e:
                log.warning("[SNDR_EAGLE3] hook remove failed: %s", e)
        setattr(model, _HANDLES_ATTR, [])
        setattr(model, _AUX_STATES_ATTR, [])
    log.info("[SNDR_EAGLE3] removed %d aux-hidden-state hooks", count)
    return count


def apply() -> tuple[str, str]:
    """Apply marker — no-op until a drafter actually calls
    `register_aux_hidden_state_hooks()`.

    Returns a `(status, reason)` 2-tuple — required by both the legacy
    @register_patch wrapper AND the spec-driven orchestrator path
    (SNDR_APPLY_VIA_SPECS=1) which unpacks as
    `status, reason = mod.apply()`.

    v11.3.0 bug fix: pre-fix this returned a dict, which silently broke
    the spec-driven path with a TypeError on tuple-unpack. Other
    apply_modules in the repo (PN12, PN79_V2_*, PN116, etc.) all return
    2-tuples — fixed for consistency. Idempotent.
    """
    # Idempotent marker check
    if globals().get(_GENESIS_MARKER):
        return (
            "skipped",
            "SNDR_EAGLE3_AUX_HIDDEN_001 already applied (idempotent)",
        )
    globals()[_GENESIS_MARKER] = True
    return (
        "applied",
        "SNDR_EAGLE3_AUX_HIDDEN_001 model-side prep ready; "
        "register_aux_hidden_state_hooks() awaits drafter wire-up",
    )
