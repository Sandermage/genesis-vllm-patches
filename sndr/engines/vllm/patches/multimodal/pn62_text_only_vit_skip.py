# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN62 — text-only ViT scratch skip (Wave 6, real hook).

Source: apnar club-3090#51 NVFP4 boot failure 2026-05-04. After PN61
auto-sets language_model_only=True, vLLM still reserves ViT-tower
scratch memory inside `profile_run` (multimodal encoder profiling
branch). On a single 32 GB card this 3-5 GiB reservation collides with
the model's 25 GiB load, leaving only ~0.67 GiB for KV cache → boot
fails::

    ValueError: To serve at least one request with the models's max seq len
    (96000), 3.89 GiB KV cache is needed, which is larger than the available
    KV cache memory (0.67 GiB).

================================================================
GENESIS APPROACH (Wave 6 real hook 2026-05-09)
================================================================

vLLM dev93 already has the right knob:
``MultiModalConfig.skip_mm_profiling: bool = False`` (see
``vllm/config/multimodal.py:183``). When set to True before
``profile_run()`` runs, the entire ``if self.supports_mm_inputs:``
branch in ``GPUModelRunner.profile_run`` short-circuits via the
``if mm_config and mm_config.skip_mm_profiling: return`` early-out at
``gpu_model_runner.py:5879-5882``. That includes:

  * ``self._get_mm_dummy_batch(...)`` — encoder dummy-batch allocation
  * ``self.model.embed_multimodal(...)`` — ViT forward (the 3-5 GiB hog)
  * ``sanity_check_mm_encoder_outputs(...)``
  * The ``self.encoder_cache[...]`` writes

PN62's job: detect "this server is text-only" (operator passed
``--language-model-only`` AND ``limit_mm_per_prompt`` all-zero) and
flip ``model_config.multimodal_config.skip_mm_profiling = True``
*before* profile_run runs. After that, vLLM's own native code path
saves the memory.

This is a clean **runtime config flip + safety wrapper around
profile_run** — no text-patch into vllm source needed because the
official knob already exists upstream.

================================================================
RUNTIME OBSERVABLE DIFFERENCE (no-stubs rule)
================================================================

When PN62 OFF (env unset):
  * profile_run executes encoder profiling branch
  * 3-5 GiB ViT scratch reserved on multimodal-capable models
  * Boot may fail on 24-32 GB cards as KV cache shrinks

When PN62 ON + text-only detected:
  * skip_mm_profiling flipped to True before profile_run
  * Encoder profiling branch returns early
  * 3-5 GiB GPU memory remains available for KV cache
  * Boot succeeds

When PN62 ON + multimodal usage detected (some mm_limit > 0):
  * Wrapper falls through to vanilla — no behavior change
  * NULL impact on production multimodal serving

================================================================
ENV
================================================================

GENESIS_ENABLE_PN62=1

Companion: GENESIS_PN62_DEBUG=1 (logs each skip event with bytes-saved
estimate; default OFF to avoid log noise).

================================================================
RISK
================================================================

LOW — wrapper checks two operator-set fields (mm_limits_all_zero,
language_model_only) BEFORE flipping. If either is False, the wrapper
falls through. NULL on multimodal-disabled models that have no ViT
branch in profile_run anyway.

Idempotent — wrapper detects prior wrapping via ``__pn62_wrapped__``
marker.

The flag mutation is surgical: PN62 only flips ``skip_mm_profiling``
from False → True (never the reverse), so it cannot accidentally
re-enable profiling that the operator explicitly disabled.

Author: Sandermage 2026-05-05; Wave 6 real hook 2026-05-09.
Backport reference: apnar club-3090#51 KV-cache-cliff after lang-only fallback.
Sister patch: PN35 (text-only inputs_embeds skip, vllm#35975 merged upstream).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn62_text_only_vit_skip")


def _is_text_only_mode(self) -> bool:
    """Return True iff the runner is in 'text-only' regime where ViT
    profiling should be skipped.

    Conditions:
      1. Operator passed ``--language-model-only`` (vllm sets
         ``model_config.language_model_only = True``)
      2. ``limit_mm_per_prompt`` is empty OR all values == 0
    """
    mc = getattr(self, "model_config", None)
    if mc is None:
        vc = getattr(self, "vllm_config", None)
        if vc is not None:
            mc = getattr(vc, "model_config", None)
    if mc is None:
        return False

    lmo = getattr(mc, "language_model_only", False)
    if not lmo:
        return False

    mm_config = getattr(mc, "multimodal_config", None)
    if mm_config is None:
        return True

    mm_limits = getattr(mm_config, "limit_per_prompt", None)
    if mm_limits is None:
        mm_limits = getattr(mc, "limit_mm_per_prompt", None)
    if mm_limits is None or not mm_limits:
        return True
    if hasattr(mm_limits, "items"):
        return all(v == 0 for v in mm_limits.values())
    return True


def _flip_skip_flag(self) -> tuple[bool, str]:
    """Set ``skip_mm_profiling = True`` on the runner's multimodal_config.

    Returns ``(flipped, reason)``. ``flipped=True`` only when the flag
    actually transitioned False → True (idempotent).
    """
    mc = getattr(self, "model_config", None)
    if mc is None:
        vc = getattr(self, "vllm_config", None)
        if vc is not None:
            mc = getattr(vc, "model_config", None)
    if mc is None:
        return False, "no model_config on runner"

    mm_config = getattr(mc, "multimodal_config", None)
    if mm_config is None:
        return False, "no multimodal_config — model is not mm-capable, NULL"

    if not hasattr(mm_config, "skip_mm_profiling"):
        return False, (
            "multimodal_config.skip_mm_profiling absent on this vllm pin "
            "(< dev93?) — PN62 NULL"
        )

    if getattr(mm_config, "skip_mm_profiling", False):
        return False, "already True (operator pre-set or PN62 idempotent)"

    try:
        mm_config.skip_mm_profiling = True
    except Exception as exc:
        return False, f"flip failed: {exc!r}"
    return True, "flipped False→True"


def _wrap_profile_run(original_profile_run):
    """Decorate profile_run with text-only ViT-scratch skip guard.

    Strategy: detect text-only mode → flip
    ``mm_config.skip_mm_profiling = True`` BEFORE calling the original,
    so vllm's native short-circuit in profile_run (line ~5879 of
    gpu_model_runner.py) takes effect.
    """

    def wrapped(self, *args, **kwargs):
        debug = os.environ.get("GENESIS_PN62_DEBUG", "") == "1"
        if _is_text_only_mode(self):
            flipped, reason = _flip_skip_flag(self)
            if flipped:
                if debug:
                    log.info(
                        "[PN62 text-only ViT skip] profile_run text-only "
                        "regime detected — skip_mm_profiling %s; vllm will "
                        "now short-circuit encoder profiling, freeing "
                        "~3-5 GiB on NVFP4 / VL stacks.",
                        reason,
                    )
            else:
                if debug:
                    log.info(
                        "[PN62 text-only ViT skip] profile_run text-only "
                        "detected but flip not actionable: %s",
                        reason,
                    )
                # mark on runner for downstream observability / tests
                try:
                    setattr(self, "_pn62_skip_vit_scratch", True)
                except Exception:
                    pass
        return original_profile_run(self, *args, **kwargs)

    wrapped.__wrapped__ = original_profile_run
    wrapped.__pn62_wrapped__ = True
    return wrapped


def apply() -> tuple[str, str]:
    """Apply PN62 — install class-rebind wrapper around profile_run."""
    from sndr.dispatcher import should_apply, log_decision

    decision, reason = should_apply("PN62")
    log_decision("PN62", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _Runner
    except Exception:
        return (
            "skipped",
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner not importable "
            "on this pin — PN62 NULL",
        )

    if not hasattr(_Runner, "profile_run"):
        return (
            "skipped",
            "GPUModelRunner.profile_run not present on this pin — "
            "PN62 NULL (encoder-skip needs newer vllm)",
        )

    # Verify the official knob is present on this pin before wrapping
    try:
        from vllm.config.multimodal import MultiModalConfig as _MMC

        if not hasattr(_MMC, "skip_mm_profiling"):
            return (
                "skipped",
                "MultiModalConfig.skip_mm_profiling absent on this vllm pin "
                "— PN62 NULL (real hook needs dev93+)",
            )
    except Exception:
        return (
            "skipped",
            "vllm.config.multimodal.MultiModalConfig not importable — "
            "PN62 NULL",
        )

    if getattr(_Runner.profile_run, "__pn62_wrapped__", False):
        return "applied", "PN62 already wrapped profile_run (idempotent)"

    _Runner.profile_run = _wrap_profile_run(_Runner.profile_run)
    return (
        "applied",
        "PN62 wrapped GPUModelRunner.profile_run — when "
        "--language-model-only + mm_limits_all_zero, sets "
        "multimodal_config.skip_mm_profiling=True before profile_run, "
        "so vLLM's native encoder-skip short-circuit fires. Saves "
        "~3-5 GiB ViT scratch on qwen3_vl + NVFP4 single-card boot. "
        "Sister to PN35 (text-only inputs_embeds skip, vllm#35975 merged)."
    )
