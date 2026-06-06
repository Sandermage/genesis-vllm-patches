# SPDX-License-Identifier: Apache-2.0
"""G4_73 — Skip drafter.dummy_run during profile_run (PN262-D pragmatic).

================================================================
PROBLEM (PN262-A/B verdict)
================================================================

After G4_71 (drafter impl→FlashAttn) and G4_72 (drafter spec→native),
K=2 boot crashes inside ``determine_available_memory → profile_run``::

    gpu_model_runner.py:5794   self.drafter.dummy_run(...)
      → llm_base_proposer.py:1494   self.model(**kwargs)
        → gemma4_mtp.py:550   Gemma4MTPAttention.forward
          → attention.py:746  unified_attention_with_output
            → get_attention_context(layer_name) → kv_cache    (wrong shape)
              → self.impl.forward(kv_cache=...)
                → flash_attn.py:744  key_cache, value_cache = kv_cache.unbind(0)
                ValueError: too many values to unpack (expected 2)

PN262 fail-fast trace::

    shape=(8192, 8, 256)  stride=(2048, 256, 1)  bf16
    contig=True ndim=3   kv_sharing_target=None  VLLM_KV_CACHE_LAYOUT='<unset>'

A/B with PN259c OFF was bit-identical, so PN259c is not the culprit.

The (8192, 8, 256) layout is what
``TurboQuantAttentionBackend.get_kv_cache_shape()`` returns
(``num_blocks * block_size, num_kv_heads, head_dim`` packed-flat).
At profile_run time, drafter's KV cache placeholder is sized by the
GROUP's backend, which is still TQ — even though G4_71 forced
FlashAttn at the per-layer impl level. The profile-time grouping
happens BEFORE the proper ``initialize_attn_backend`` sub-grouping
pass that would split layers by per-layer ``get_attn_backend()``.

================================================================
FIX (PRAGMATIC, MINIMAL)
================================================================

Skip drafter's profile dummy run via a thread-local flag:

  1. Wrap ``GPUModelRunner._dummy_run``: when ``is_profile=True``,
     set ``_thread_state.in_profile = True`` for the duration of the
     call.
  2. Wrap ``SpecDecodeBaseProposer.dummy_run``: when flag is set,
     log + return immediately without invoking ``self.model(**kwargs)``.

What this gives us:

  * profile_run completes (memory estimate omits drafter — drafter is
    small relative to the 31B target, so the estimate is still useful).
  * Engine proceeds to the real ``initialize_kv_cache(kv_cache_config)``
    which calls ``initialize_attn_backend`` → ``get_attn_backends_for_group``
    (gpu_model_runner.py:6470) — and THAT function sub-groups layers by
    per-layer ``attn_backend = layers[layer_name].get_attn_backend()``,
    so drafter (FlashAttn impl, G4_71) and target (TQ impl) end up in
    DIFFERENT ``AttentionGroup``s with correct per-group backend.
  * Runtime kv_cache allocation should then produce the right shape
    for drafter.

What this does NOT do:

  * Does NOT fix profile-time grouping itself (architectural issue with
    upstream profile path that uses minimal_config built BEFORE per-layer
    backend sub-grouping). If profile-time accurate memory estimate for
    drafter is needed in a future iteration, G4_74 will need to wrap the
    grouping path itself.
  * Does NOT change FlashAttn or TQ kernels.
  * Does NOT affect non-profile dummy runs (cudagraph capture etc.) —
    only the ``is_profile=True`` path is gated.

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP=1   (opt-in)

================================================================
GATE OUTCOMES
================================================================

A. K=2 boots cleanly past profile_run.
   Runtime first prompt succeeds → grouping at runtime IS correct.
   → G4_73 is the right minimal fix.

B. K=2 boots past profile_run, but runtime first prompt still crashes
   with PN262 shape mismatch.
   → grouping is broken at runtime too. Need G4_74 = explicit drafter
   group split in ``get_attn_backends_for_group`` or in
   ``kv_cache_config.kv_cache_groups`` construction.

C. K=2 still crashes during profile_run despite G4_73.
   → drafter.dummy_run is called via a different code path that the
   wrap doesn't cover. Need to inspect stack and broaden the gate.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("genesis.spec_decode.g4_73_drafter_profile_skip")

GENESIS_G4_73_MARKER = (
    "Genesis G4_73 Skip drafter.dummy_run during profile_run "
    "(PN262-D pragmatic — unblock profile path with TQ-grouped drafter)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP"
_APPLIED = False
_ORIGINAL_DUMMY_RUN = None
_ORIGINAL_DRAFTER_DUMMY_RUN = None
_thread_state = threading.local()
_SKIP_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _is_in_profile() -> bool:
    return getattr(_thread_state, "in_profile", False)


def _set_in_profile(value: bool) -> None:
    _thread_state.in_profile = value


def apply() -> tuple[str, str]:
    """Wrap GPUModelRunner._dummy_run + SpecDecodeBaseProposer.dummy_run."""
    global _APPLIED, _ORIGINAL_DUMMY_RUN, _ORIGINAL_DRAFTER_DUMMY_RUN

    if not _env_enabled():
        return "skipped", (
            f"G4_73 disabled (set {_ENV_ENABLE}=1 to skip drafter.dummy_run "
            "during profile_run and unblock the K=2 boot for the "
            "TQ-grouped drafter case)"
        )

    if _APPLIED:
        return "applied", "G4_73 already installed (idempotent)"

    log.warning("[G4_73] apply() entered — beginning import phase")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_73] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_73] SKIP: SpecDecodeBaseProposer not importable: %s", e)
        return "skipped", f"SpecDecodeBaseProposer not importable: {e!r}"

    if not hasattr(GPUModelRunner, "_dummy_run"):
        return "skipped", "GPUModelRunner._dummy_run missing on this pin"
    if not hasattr(SpecDecodeBaseProposer, "dummy_run"):
        return "skipped", "SpecDecodeBaseProposer.dummy_run missing on this pin"

    original_dummy_run = GPUModelRunner._dummy_run
    if getattr(original_dummy_run, "_genesis_g4_73_wrapped", False):
        _APPLIED = True
        return "applied", "GPUModelRunner._dummy_run already wrapped"
    _ORIGINAL_DUMMY_RUN = original_dummy_run

    original_drafter_dummy_run = SpecDecodeBaseProposer.dummy_run
    if getattr(original_drafter_dummy_run, "_genesis_g4_73_wrapped", False):
        _APPLIED = True
        return "applied", "SpecDecodeBaseProposer.dummy_run already wrapped"
    _ORIGINAL_DRAFTER_DUMMY_RUN = original_drafter_dummy_run

    def _wrapped_main_dummy_run(self, *args, **kwargs):
        # is_profile may arrive positionally or by keyword; the upstream
        # signature places it as a kw-only parameter, so we read kwargs.
        is_profile = bool(kwargs.get("is_profile", False))
        if is_profile:
            _set_in_profile(True)
        try:
            return original_dummy_run(self, *args, **kwargs)
        finally:
            if is_profile:
                _set_in_profile(False)

    _wrapped_main_dummy_run._genesis_g4_73_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner._dummy_run = _wrapped_main_dummy_run  # type: ignore[method-assign]

    def _wrapped_drafter_dummy_run(self, *args, **kwargs):
        if _is_in_profile():
            _SKIP_COUNT[0] += 1
            if _SKIP_COUNT[0] <= 4:
                log.warning(
                    "[G4_73] skipping drafter.dummy_run during profile_run "
                    "(args=%r kwargs.keys=%s) — avoids profile-time TQ-grouped "
                    "shape mismatch on drafter (skip #%d)",
                    [type(a).__name__ for a in args[:3]],
                    list(kwargs.keys()),
                    _SKIP_COUNT[0],
                )
            elif _SKIP_COUNT[0] == 5:
                log.warning("[G4_73] further drafter-profile-skip logs suppressed (> 4)")
            return None
        return original_drafter_dummy_run(self, *args, **kwargs)

    _wrapped_drafter_dummy_run._genesis_g4_73_wrapped = True  # type: ignore[attr-defined]
    SpecDecodeBaseProposer.dummy_run = _wrapped_drafter_dummy_run  # type: ignore[method-assign]

    _APPLIED = True
    log.warning(
        "[G4_73] INSTALLED: GPUModelRunner._dummy_run + "
        "SpecDecodeBaseProposer.dummy_run wrapped; drafter dummy_run will "
        "be skipped when GPUModelRunner._dummy_run is called with "
        "is_profile=True."
    )
    return "applied", (
        "G4_73 installed: drafter.dummy_run is skipped during "
        "GPUModelRunner._dummy_run(is_profile=True) to bypass profile-time "
        "TQ-grouped shape mismatch."
    )


def is_applied() -> bool:
    return _APPLIED


def skip_count() -> int:
    return _SKIP_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_DUMMY_RUN, _ORIGINAL_DRAFTER_DUMMY_RUN
    if not _APPLIED:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        if _ORIGINAL_DUMMY_RUN is not None:
            GPUModelRunner._dummy_run = _ORIGINAL_DUMMY_RUN  # type: ignore[method-assign]
    except ImportError:
        return False
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        if _ORIGINAL_DRAFTER_DUMMY_RUN is not None:
            SpecDecodeBaseProposer.dummy_run = _ORIGINAL_DRAFTER_DUMMY_RUN  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_DUMMY_RUN = None
    _ORIGINAL_DRAFTER_DUMMY_RUN = None
    return True


__all__ = [
    "GENESIS_G4_73_MARKER",
    "apply",
    "is_applied",
    "skip_count",
    "revert",
]
