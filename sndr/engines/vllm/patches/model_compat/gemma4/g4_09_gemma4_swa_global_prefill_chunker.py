# SPDX-License-Identifier: Apache-2.0
"""G4_09 — chunked prefill across SWA→global attention boundary (closes #39914).

================================================================
WHAT IT FIXES
================================================================

vllm-project/vllm#39914 (OPEN, 15 comments, 6 months stale): Gemma 4's
**interleaved attention** (5 sliding_attention layers per 1 full_attention
layer for 31B; 5:1 pattern for 26B-A4B) plus **dual RoPE** (standard
RoPE on sliding, p-RoPE on global for 256K extension) causes a hard
engine hang during prefill when prompt length exceeds ~4096 tokens.

The bug manifests specifically at the **sliding → global attention
transition**: the scheduler/router stops processing and the worker
loop deadlocks. The TODO-author found that token-by-token incremental
prefill works fine up to 200K+; large-batch prefill at the boundary
hangs.

Confirmed reproducers (from issue comments):
  * @lucianommartins (Gemma 4 author) tagged; ack but no upstream PR yet
  * @subnet-dev — same on A100-80G + vLLM v0.19.1
  * @mcr-ksh — Jetson Thor same symptom

================================================================
APPROACH
================================================================

Force ``max_num_batched_tokens`` to be **conservative** when Gemma 4 +
prefill > 4K is detected — clamp it to a value that keeps each scheduling
step under the apparent hang threshold. The result is effectively
**multiple small prefill batches** that each cross the boundary
cleanly, instead of one large batch that deadlocks.

vLLM v1 already supports chunked prefill (``--enable-chunked-prefill``)
— this patch enforces a **conservative chunk size** when the target
is Gemma 4. Empirical sweep from #39914 comments suggests 2048
tokens-per-chunk is safe.

We install the chunk-size override at apply time:

  1. Detect Gemma 4 model
  2. Read current ``vllm_config.scheduler_config.max_num_batched_tokens``
  3. If > 2048, clamp to 2048 with a log warning
  4. Also force ``enable_chunked_prefill = True`` if not set

================================================================
WHY THIS IS A WORKAROUND, NOT A FIX
================================================================

The real fix is in vLLM's scheduler — it should handle the SWA→global
transition without scheduling > 4K tokens through it in a single step.
That requires deeper engineering (boundary-aware scheduling decisions).
This patch is a **safe-by-default conservative chunking** that bypasses
the bug without touching scheduler code.

When upstream fixes #39914 properly, we set ``superseded_by`` and
flip default_on to False.

================================================================
SAFETY MODEL
================================================================

* default_on: True (small perf cost — chunked prefill is slightly
  slower than batch prefill at small batch sizes, but no hang)
* env_flag: GENESIS_ENABLE_G4_09_GEMMA4_SWA_PREFILL_CHUNKER
* applies_to: architecture == gemma4
* conflicts_with: none
* superseded_by: when vllm#39914 merges

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/39914 (15 comments,
    OPEN as of 2026-05-17)
"""
from __future__ import annotations

import logging

from ._gemma4_detect import env_truthy, is_gemma4_arch

log = logging.getLogger("genesis.gemma4.g4_09_swa_prefill_chunker")

GENESIS_G4_09_MARKER = (
    "Genesis G4_09 gemma4 SWA→global prefill chunker v1 "
    "(workaround for vllm#39914 engine hang at prefill > 4K)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_09_GEMMA4_SWA_PREFILL_CHUNKER"
_ENV_CHUNK_SIZE = "GENESIS_G4_09_CHUNK_SIZE"  # tunable override; audit P0 follow-up

# Default chunk size budget per scheduling step.
# Audit GEMMA4_PATCH_OPTIMIZATION_PLAN_2026-05-17_RU notes:
#   "G4_09 stable but limit 2048 may cut throughput; provide profiles
#    2048 / 4096 / 8192".
# 2048 is the empirically-safe minimum from issue #39914 reproducer
# comments. 4096 has been tested on dev371 without hang. 8192 is the
# aggressive ceiling — operator must validate per checkpoint.
_DEFAULT_CHUNK_SIZE_TOKENS = 2048

# Hard ceiling — anything above this approaches the hang threshold per
# #39914 measurements (~4096 was reported as triggering on FP8_BLOCK;
# 8192 was confirmed safe on AWQ checkpoints).
_MAX_CHUNK_SIZE_TOKENS = 8192


def _resolve_chunk_size() -> int:
    """Read GENESIS_G4_09_CHUNK_SIZE env, clamp to safe range."""
    import os
    raw = os.environ.get(_ENV_CHUNK_SIZE, "").strip()
    if not raw:
        return _DEFAULT_CHUNK_SIZE_TOKENS
    try:
        v = int(raw)
        if v < 512:
            log.warning("[G4_09] chunk size %d too small (min 512); using default %d",
                        v, _DEFAULT_CHUNK_SIZE_TOKENS)
            return _DEFAULT_CHUNK_SIZE_TOKENS
        if v > _MAX_CHUNK_SIZE_TOKENS:
            log.warning("[G4_09] chunk size %d exceeds safe ceiling %d; clamping",
                        v, _MAX_CHUNK_SIZE_TOKENS)
            return _MAX_CHUNK_SIZE_TOKENS
        return v
    except ValueError:
        log.warning("[G4_09] invalid %s=%r; using default %d",
                    _ENV_CHUNK_SIZE, raw, _DEFAULT_CHUNK_SIZE_TOKENS)
        return _DEFAULT_CHUNK_SIZE_TOKENS

_APPLIED = False
_ORIGINAL_VERIFY = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def apply() -> tuple[str, str]:
    """Install a Gemma4Config wrapper that clamps prefill chunk size."""
    global _APPLIED, _ORIGINAL_VERIFY

    if not _env_enabled():
        return "skipped", (
            f"G4_09 disabled (set {_ENV_ENABLE}=1 to enforce conservative "
            "prefill chunking on Gemma 4 — workaround for vllm#39914 hang)"
        )

    if _APPLIED:
        return "applied", "G4_09 already installed (idempotent)"

    # We wrap Gemma4Config.verify_and_update_config (or the equivalent)
    # to clamp scheduler max_num_batched_tokens.
    # dev371+ moved the verify_and_update_config wrappers from
    # vllm.model_executor.models.gemma4 to vllm.model_executor.models.config.
    # Search both locations so the patch survives the move.
    _candidate_modules: list[tuple[str, object]] = []
    try:
        from vllm.model_executor.models import config as _g4_cfg_mod
        _candidate_modules.append(
            ("vllm.model_executor.models.config", _g4_cfg_mod)
        )
    except ImportError:
        pass
    try:
        from vllm.model_executor.models import gemma4 as _g4_legacy_mod
        _candidate_modules.append(
            ("vllm.model_executor.models.gemma4", _g4_legacy_mod)
        )
    except ImportError:
        pass

    if not _candidate_modules:
        return "skipped", (
            "Neither vllm.model_executor.models.config nor .gemma4 "
            "importable; G4_09 is no-op on this pin"
        )

    target_cls = None
    for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
        for mod_name, mod in _candidate_modules:
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "verify_and_update_config"):
                target_cls = cls
                break
        if target_cls is not None:
            break
    if target_cls is None:
        return "skipped", (
            "No Gemma4Config-like class with verify_and_update_config found "
            f"in {[m for m, _ in _candidate_modules]}; G4_09 is no-op on this pin"
        )

    original = target_cls.verify_and_update_config
    if getattr(original, "_genesis_g4_09_wrapped", False):
        _APPLIED = True
        return "applied", "G4_09 already wrapped (idempotent)"
    _ORIGINAL_VERIFY = original

    # Resolve operator-configured chunk size (default 2048; tunable via
    # GENESIS_G4_09_CHUNK_SIZE env per audit P0 follow-up). Baked at
    # apply() time so per-step calls don't reread env.
    chunk_size = _resolve_chunk_size()

    def _genesis_g4_09_wrapped_verify(vllm_config):
        result = original(vllm_config)
        try:
            sc = getattr(vllm_config, "scheduler_config", None)
            mc = getattr(vllm_config, "model_config", None)
            if sc is not None and mc is not None and is_gemma4_arch(mc):
                current = getattr(sc, "max_num_batched_tokens", None)
                if current is None or current > chunk_size:
                    log.warning(
                        "[G4_09] clamping scheduler.max_num_batched_tokens "
                        "%s → %d (Gemma 4 + #39914 workaround)",
                        current, chunk_size,
                    )
                    sc.max_num_batched_tokens = chunk_size
                # Force chunked prefill on (its absence triggers single-batch path)
                if hasattr(sc, "enable_chunked_prefill"):
                    if not sc.enable_chunked_prefill:
                        log.warning(
                            "[G4_09] enabling chunked-prefill (was off; required "
                            "for the workaround to be effective)"
                        )
                        sc.enable_chunked_prefill = True
        except Exception as e:  # noqa: BLE001
            log.warning("[G4_09] scheduler clamp failed: %r; leaving config as-is", e)
        return result

    _genesis_g4_09_wrapped_verify._genesis_g4_09_wrapped = True
    _genesis_g4_09_wrapped_verify.__wrapped__ = original

    def _classmethod_shim(cls, vllm_config):
        return _genesis_g4_09_wrapped_verify(vllm_config)
    _classmethod_shim._genesis_g4_09_wrapped = True
    target_cls.verify_and_update_config = classmethod(_classmethod_shim)
    _APPLIED = True
    log.info(
        "[G4_09] installed: Gemma 4 will use max_num_batched_tokens ≤ %d "
        "and enable_chunked_prefill=True to avoid #39914 SWA→global hang. "
        "Override via %s env (range 512..%d).",
        chunk_size, _ENV_CHUNK_SIZE, _MAX_CHUNK_SIZE_TOKENS,
    )
    return "applied", (
        f"G4_09 installed: Gemma 4 prefill chunked to ≤{chunk_size} tokens "
        f"per scheduling step (#39914 workaround). Tune via {_ENV_CHUNK_SIZE}=<512..{_MAX_CHUNK_SIZE_TOKENS}>."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_VERIFY
    if not _APPLIED or _ORIGINAL_VERIFY is None:
        return False
    _modules = []
    try:
        from vllm.model_executor.models import config as _m
        _modules.append(_m)
    except ImportError:
        pass
    try:
        from vllm.model_executor.models import gemma4 as _m
        _modules.append(_m)
    except ImportError:
        pass
    for _g4_mod in _modules:
        for cls_name in ("Gemma4Config", "Gemma4TextConfig", "Gemma4ForConditionalGenerationConfig"):
            cls = getattr(_g4_mod, cls_name, None)
            if cls is not None and getattr(cls.verify_and_update_config, "_genesis_g4_09_wrapped", False):
                cls.verify_and_update_config = _ORIGINAL_VERIFY  # type: ignore[assignment]
                _APPLIED = False
                return True
    return False


__all__ = ["GENESIS_G4_09_MARKER", "apply", "is_applied", "revert"]
