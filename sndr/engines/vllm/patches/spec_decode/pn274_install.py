# SPDX-License-Identifier: Apache-2.0
"""PN274 — install SpecDecode safety guard at config boot.

Wires ``safety_guard.evaluate_from_config(vllm_config)`` into vLLM's
boot path BEFORE workers spawn:

  - Wraps ``EngineArgs.create_engine_config(...)`` (returns VllmConfig).
  - After the original call, runs the guard.
  - If guard.allowed = False: sets ``vllm_config.speculative_config =
    None`` IN PLACE and logs the operator-facing reason.

Default-on: this install runs automatically at ``sndr``
import. Escape hatch:

  GENESIS_DISABLE_SPEC_DECODE_SAFETY_GUARD=1

When set, the guard is not installed at all — vLLM proceeds with the
operator's requested speculative_config unchanged.

Adapter / functional-unknown override (per-spec-decode-pair, not
per-install):

  GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER=1
  GENESIS_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN=1

Behavior change matrix:

  Qwen (no provider match)      -> ALLOW    speculative_config unchanged
  Gemma4 + native bf16 KV       -> ALLOW    speculative_config unchanged
  Gemma4 + TQ KV (FUNCTIONAL_UNVERIFIED)
                                -> DENY     speculative_config = None
                                            unless BOTH override envs set

Note: this is a SAFETY net, not a feature. Operators who want MTP on
quantized Gemma4 must explicitly accept the runtime risk via both
envs above; the canonical default reflects PN272's empirical finding
that runtime acceptance with the K/V bridge is currently 0%.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.spec_decode.pn274_install")

GENESIS_PN274_MARKER = "Genesis PN274 SpecDecode safety guard install"

# P1 naming migration: bare suffix; resolver handles SNDR_/GENESIS_.
_ENV_DISABLE = "DISABLE_SPEC_DECODE_SAFETY_GUARD"
_APPLIED = False
_ORIGINAL_CREATE_ENGINE_CONFIG = None


def _env_disabled() -> bool:
    # Absolute import: the v12 move to sndr/engines/vllm/patches/ added a
    # nesting level, so the old `...env` relative path no longer reaches
    # the platform-level sndr/env.py module.
    from sndr.env import get_sndr_env_bool
    return get_sndr_env_bool(_ENV_DISABLE)


def apply() -> tuple[str, str]:
    """Install the guard. Returns (status, reason)."""
    global _APPLIED, _ORIGINAL_CREATE_ENGINE_CONFIG

    if _env_disabled():
        return "skipped", f"PN274 guard disabled via SNDR_{_ENV_DISABLE}=1"
    if _APPLIED:
        return "applied", "PN274 guard already installed"

    log.warning("[PN274] installing SpecDecode safety guard at "
                "EngineArgs.create_engine_config")

    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception as e:  # noqa: BLE001
        log.warning("[PN274] SKIP: EngineArgs not importable: %s", e)
        return "skipped", f"EngineArgs import failed: {e!r}"

    if not hasattr(EngineArgs, "create_engine_config"):
        return "skipped", "EngineArgs.create_engine_config missing"

    original = EngineArgs.create_engine_config
    if getattr(original, "_genesis_pn274_wrapped", False):
        _APPLIED = True
        return "applied", "create_engine_config already wrapped"
    _ORIGINAL_CREATE_ENGINE_CONFIG = original

    def _wrapped(self, *args, **kwargs):
        vllm_config = original(self, *args, **kwargs)
        try:
            _apply_guard(vllm_config)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[PN274] guard pass failed (model boot continues with "
                "ORIGINAL speculative_config; this is the safe fallback): "
                "%s", e,
            )
        return vllm_config

    _wrapped._genesis_pn274_wrapped = True  # type: ignore[attr-defined]
    EngineArgs.create_engine_config = _wrapped  # type: ignore[method-assign]

    _APPLIED = True
    log.warning(
        "[PN274] INSTALLED — guard will evaluate each create_engine_config "
        "result and may disable speculative_config when contract is "
        "ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED or UNSUPPORTED. "
        "Escape: SNDR_%s=1.", _ENV_DISABLE,
    )
    return "applied", "PN274 installed"


def _apply_guard(vllm_config) -> None:
    """Run safety_guard.evaluate_from_config and possibly disable MTP."""
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    if spec_cfg is None:
        # No MTP requested — nothing to guard.
        return

    # Lazy import to keep this module torch-less at install time.
    from .safety_guard import evaluate_from_config

    decision = evaluate_from_config(vllm_config)

    model_name = "<unknown>"
    try:
        mc = getattr(vllm_config, "model_config", None)
        hf = getattr(mc, "hf_config", None) if mc is not None else None
        model_name = (
            getattr(hf, "model_type", None)
            or getattr(mc, "model", None)
            or "<unknown>"
        )
    except Exception:
        pass

    verdict_str = decision.overall_verdict.value
    log.warning(
        "[SpecDecodeGuard] model=%s verdict=%s allowed=%s reason=%r",
        model_name, verdict_str, decision.allowed, decision.reason,
    )

    if decision.allowed:
        return

    # DENY: disable MTP in place. Workers receive the modified config
    # via the standard pickle/IPC path.
    vllm_config.speculative_config = None
    # Per-verdict override hint — KERNEL_*_MISMATCH is non-overridable
    # because the consumer kernel would misread bytes, so no env flips
    # are safe; operator must change the backend/layout contract.
    if "KERNEL_STORAGE_DTYPE_MISMATCH" in verdict_str or (
            "KERNEL_LAYOUT_CONTRACT_MISMATCH" in verdict_str):
        log.warning(
            "[SpecDecodeGuard] ACTION=disable_mtp — speculative_config "
            "has been set to None. This verdict is NON-OVERRIDABLE: the "
            "consumer kernel would misread cache bytes. To research, "
            "change the engine backend or KV storage layout so the "
            "drafter kernel and target storage agree on (dtype, layout)."
        )
    elif verdict_str == "UNSUPPORTED":
        log.warning(
            "[SpecDecodeGuard] ACTION=disable_mtp — speculative_config "
            "has been set to None. Verdict UNSUPPORTED is not "
            "overridable; the contract is fundamentally incompatible."
        )
    else:
        log.warning(
            "[SpecDecodeGuard] ACTION=disable_mtp — speculative_config "
            "has been set to None. To override, set BOTH "
            "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER=1 and "
            "SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN=1 in your "
            "container env, then restart. (GENESIS_* aliases also "
            "work, with a deprecation warning.)"
        )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_CREATE_ENGINE_CONFIG
    if not _APPLIED or _ORIGINAL_CREATE_ENGINE_CONFIG is None:
        return False
    try:
        from vllm.engine.arg_utils import EngineArgs
        EngineArgs.create_engine_config = _ORIGINAL_CREATE_ENGINE_CONFIG  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_CREATE_ENGINE_CONFIG = None
    return True


__all__ = ["GENESIS_PN274_MARKER", "apply", "is_applied", "revert"]
