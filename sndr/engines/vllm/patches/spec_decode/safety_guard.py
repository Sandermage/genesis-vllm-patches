# SPDX-License-Identifier: Apache-2.0
"""safety_guard — production policy for SpecDecode KV sharing.

PN274 deliverable (skeleton only — not yet auto-installed).

Library-grade policy module. Provides:

  evaluate(runner) -> GuardDecision

  where GuardDecision encodes:
    - per-pair verdicts (Verdict from kv_contract)
    - overall worst-case verdict
    - whether MTP should be allowed to proceed
    - reason strings

Policy (conservative, default for production):

  EXACT_COPY                                -> ALLOW
  GQA_REPEAT / LAYOUT_ADAPTER               -> ALLOW iff env opt-in
  COMPOSITE_ADAPTER                         -> ALLOW iff env opt-in
  DEQUANT_REQUIRED                          -> DENY (bridge not implemented)
  ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED ->
      DENY by default; ALLOW iff a SECOND env opt-in is also set
  UNSUPPORTED                               -> DENY

Env flags:
  GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER=1
      Permits LAYOUT/GQA/COMPOSITE/DEQUANT verdicts to proceed.
  GENESIS_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN=1
      Permits ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED verdict to
      proceed. Independent of the first env — operators have to
      consciously accept the "no acceptance guarantee" risk.

This module does NOT yet patch anything. It is a building block for a
future patch (PN274) that wires this guard into the spec-decode boot
path to flip ``vllm_config.speculative_config = None`` when the
verdict is a DENY.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from .kv_contract import (
    KVContract,
    Verdict,
    compare_contracts,
    extract_contract,
)
from .mapping.base import LayerMapping
from .mapping.registry import find_provider

log = logging.getLogger("genesis.spec_decode.safety_guard")


# P1 naming migration (2026-05-20):
#   Canonical names are SNDR_*; GENESIS_* aliases continue to work
#   via get_sndr_env() helper with a deprecation warning on each
#   GENESIS_-only resolution.
_ENV_ALLOW_ADAPTER = "ALLOW_SPEC_DECODE_KV_ADAPTER"
_ENV_ALLOW_FUNCTIONAL_UNKNOWN = "ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN"


def _env_flag(name: str) -> bool:
    # Read with SNDR_/GENESIS_ alias semantics.
    from ...env import get_sndr_env_bool
    return get_sndr_env_bool(name)


def _canonical(name: str) -> str:
    """Render canonical env name (SNDR_<name>) for operator log lines."""
    return f"SNDR_{name}"


@dataclass
class PairAssessment:
    drafter_idx: int
    target_full_prefix: str
    drafter_contract: KVContract
    target_contract: KVContract | None
    verdict: Verdict
    divergences: list[str]
    hints: dict[str, Any]


@dataclass
class GuardDecision:
    """Final policy outcome."""
    overall_verdict: Verdict
    allowed: bool
    reason: str
    per_pair: list[PairAssessment]


def evaluate(runner: Any,
             *, require_functional_gate: bool = True) -> GuardDecision:
    """Compute the GuardDecision for one runner.

    Does NOT modify state. Returns the assessment; the caller decides
    what to do (typically: deny -> set vllm_config.speculative_config
    = None; allow -> proceed).
    """
    provider = find_provider(runner)
    if provider is None:
        # No spec-decode K/V sharing concern for this model.
        return GuardDecision(
            overall_verdict=Verdict.EXACT_COPY,
            allowed=True,
            reason="no MappingProvider matched (model uses no kv-sharing "
                   "spec-decode path)",
            per_pair=[],
        )

    log.warning(
        "[safety_guard] using provider=%s", provider.name,
    )

    mappings: list[LayerMapping] = []
    try:
        mappings = provider.get_mapping(runner)
    except Exception as _e:  # noqa: BLE001
        log.warning("[safety_guard] mapping failed: %s — DENY", _e)
        return GuardDecision(
            overall_verdict=Verdict.UNSUPPORTED,
            allowed=False,
            reason=f"mapping provider raised: {_e!r}",
            per_pair=[],
        )

    if not mappings:
        return GuardDecision(
            overall_verdict=Verdict.UNSUPPORTED,
            allowed=False,
            reason=f"provider {provider.name} returned empty mapping",
            per_pair=[],
        )

    assessments: list[PairAssessment] = []
    flags: set[Verdict] = set()
    for m in mappings:
        dst_contract = extract_contract(
            m.drafter_self_attn,
            layer_full_name=f"drafter[{m.drafter_idx}]",
        )
        src_contract = (
            extract_contract(m.target_self_attn, m.target_full_prefix)
            if m.target_self_attn is not None else None
        )
        if src_contract is None:
            verdict = Verdict.UNSUPPORTED
            divergences = ["target self_attn not resolvable"]
            hints: dict[str, Any] = {}
        else:
            verdict, divergences, hints = compare_contracts(
                src_contract, dst_contract,
                require_functional_gate=require_functional_gate,
            )
        flags.add(verdict)
        assessments.append(PairAssessment(
            drafter_idx=m.drafter_idx,
            target_full_prefix=m.target_full_prefix,
            drafter_contract=dst_contract,
            target_contract=src_contract,
            verdict=verdict,
            divergences=divergences,
            hints=hints,
        ))

    # Aggregate worst-case
    overall = _aggregate(flags)

    # Policy decision
    allow_adapter = _env_flag(_ENV_ALLOW_ADAPTER)
    allow_unknown = _env_flag(_ENV_ALLOW_FUNCTIONAL_UNKNOWN)

    if overall == Verdict.EXACT_COPY:
        return GuardDecision(
            overall_verdict=overall,
            allowed=True,
            reason="all pairs EXACT_COPY",
            per_pair=assessments,
        )
    if overall == Verdict.UNSUPPORTED:
        return GuardDecision(
            overall_verdict=overall,
            allowed=False,
            reason="at least one pair is UNSUPPORTED",
            per_pair=assessments,
        )
    if overall == Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED:
        return GuardDecision(
            overall_verdict=overall,
            allowed=(allow_unknown and allow_adapter),
            reason=(
                f"adapter-required AND no functional gate; allow only if "
                f"BOTH {_canonical(_ENV_ALLOW_ADAPTER)}=1 AND "
                f"{_canonical(_ENV_ALLOW_FUNCTIONAL_UNKNOWN)}=1 "
                f"(currently adapter={allow_adapter} "
                f"functional={allow_unknown})"
            ),
            per_pair=assessments,
        )
    # Structural-adapter classes
    return GuardDecision(
        overall_verdict=overall,
        allowed=allow_adapter,
        reason=(
            f"adapter required ({overall.value}); allow only if "
            f"{_canonical(_ENV_ALLOW_ADAPTER)}=1 (currently {allow_adapter})"
        ),
        per_pair=assessments,
    )


def _aggregate(flags: set[Verdict]) -> Verdict:
    if Verdict.UNSUPPORTED in flags:
        return Verdict.UNSUPPORTED
    if Verdict.KERNEL_STORAGE_DTYPE_MISMATCH in flags:
        return Verdict.KERNEL_STORAGE_DTYPE_MISMATCH
    if Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH in flags:
        return Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH
    if Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED in flags:
        return Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED
    if Verdict.DEQUANT_REQUIRED in flags:
        return Verdict.DEQUANT_REQUIRED
    if Verdict.COMPOSITE_ADAPTER in flags:
        return Verdict.COMPOSITE_ADAPTER
    has_layout = Verdict.LAYOUT_ADAPTER in flags
    has_gqa = Verdict.GQA_REPEAT in flags
    if has_layout and has_gqa:
        return Verdict.COMPOSITE_ADAPTER
    if has_layout:
        return Verdict.LAYOUT_ADAPTER
    if has_gqa:
        return Verdict.GQA_REPEAT
    return Verdict.EXACT_COPY


def evaluate_from_config(vllm_config: Any) -> GuardDecision:
    """Config-only evaluation, run BEFORE workers spawn.

    Used by PN274's hook on ``EngineArgs.create_engine_config`` so the
    guard can deny MTP before drafter weights load.

    Policy:
      - No provider matches at config time:
          ALLOW (model doesn't use kv-sharing spec-decode path).
      - Provider matches + verdict EXACT_COPY:
          ALLOW.
      - Provider matches + verdict FUNCTIONALLY_VALIDATED
        (bench-receipt match):
          ALLOW iff GENESIS_ALLOW_SPEC_DECODE_KV_ADAPTER=1.
          FUNCTIONAL_UNKNOWN env not required.
      - Provider matches + non-EXACT verdict (FUNCTIONAL_UNVERIFIED,
        adapter classes, UNSUPPORTED):
          require explicit env opt-ins; otherwise DENY.
    """
    provider = find_provider_for_config(vllm_config)
    if provider is None:
        return GuardDecision(
            overall_verdict=Verdict.EXACT_COPY,
            allowed=True,
            reason="no MappingProvider matched at config time (model is "
                   "outside the kv-sharing safety guard scope)",
            per_pair=[],
        )

    log.warning("[SpecDecodeGuard] provider=%s matched at config time",
                provider.name)
    try:
        verdict, reason = provider.evaluate_from_config(vllm_config)
    except Exception as _e:  # noqa: BLE001
        log.warning("[SpecDecodeGuard] provider %s.evaluate_from_config "
                    "raised %s — DENY", provider.name, _e)
        return GuardDecision(
            overall_verdict=Verdict.UNSUPPORTED,
            allowed=False,
            reason=f"provider {provider.name} raised: {_e!r}",
            per_pair=[],
        )

    # If verdict is adapter-required, try to find a matching
    # functional_artifact. If found, promote to FUNCTIONALLY_VALIDATED.
    if verdict in (Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED,
                   Verdict.LAYOUT_ADAPTER, Verdict.GQA_REPEAT,
                   Verdict.COMPOSITE_ADAPTER):
        art = _try_match_artifact(provider, vllm_config)
        if art is not None:
            verdict = Verdict.FUNCTIONALLY_VALIDATED
            reason = (
                f"matched artifact={art.profile!r} "
                f"config_hash={art.config_hash} "
                f"decision={art.decision!r} "
                f"allowed_workloads={art.allowed_workloads}. "
                f"Original verdict: {reason}"
            )

    allow_adapter = _env_flag(_ENV_ALLOW_ADAPTER)
    allow_unknown = _env_flag(_ENV_ALLOW_FUNCTIONAL_UNKNOWN)

    if verdict == Verdict.EXACT_COPY:
        allowed = True
        decision_reason = f"{provider.name}: EXACT_COPY — {reason}"
    elif verdict == Verdict.FUNCTIONALLY_VALIDATED:
        # Bench-receipt promoted. Requires structural opt-in env only.
        allowed = allow_adapter
        decision_reason = (
            f"{provider.name}: FUNCTIONALLY_VALIDATED — {reason} "
            f"(allow requires {_canonical(_ENV_ALLOW_ADAPTER)}=1; "
            f"adapter={allow_adapter}; FUNCTIONAL_UNKNOWN not needed)"
        )
    elif verdict == Verdict.UNSUPPORTED:
        allowed = False
        decision_reason = f"{provider.name}: UNSUPPORTED — {reason}"
    elif verdict in (Verdict.KERNEL_STORAGE_DTYPE_MISMATCH,
                     Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH):
        # Kernel will MISREAD bytes — denial is non-overridable.
        # Operator must change the backend/layout, not just opt in.
        allowed = False
        decision_reason = (
            f"{provider.name}: {verdict.value} — {reason} (NON-OVERRIDABLE; "
            f"fix the backend/storage contract, do not opt in)"
        )
    elif verdict == Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED:
        allowed = allow_adapter and allow_unknown
        decision_reason = (
            f"{provider.name}: {verdict.value} — {reason} "
            f"(allow requires BOTH {_canonical(_ENV_ALLOW_ADAPTER)}=1 and "
            f"{_canonical(_ENV_ALLOW_FUNCTIONAL_UNKNOWN)}=1; "
            f"adapter={allow_adapter} functional={allow_unknown})"
        )
    else:
        allowed = allow_adapter
        decision_reason = (
            f"{provider.name}: {verdict.value} — {reason} "
            f"(allow requires {_canonical(_ENV_ALLOW_ADAPTER)}=1; "
            f"adapter={allow_adapter})"
        )

    return GuardDecision(
        overall_verdict=verdict,
        allowed=allowed,
        reason=decision_reason,
        per_pair=[],
    )


def find_provider_for_config(vllm_config: Any):
    """Find the first provider whose .supports_config(vllm_config) is True.

    Separate from runtime ``mapping.registry.find_provider(runner)``:
    config-time path takes a VllmConfig, not a runner.
    """
    from .mapping.registry import PROVIDERS
    for p in PROVIDERS:
        try:
            if p.supports_config(vllm_config):
                return p
        except Exception as _e:
            log.warning("[SpecDecodeGuard] %s.supports_config raised: %s",
                        p.name, _e)
    return None


def _try_match_artifact(provider: Any, vllm_config: Any):
    """If a matching FunctionalArtifact exists for this config, return it.

    Provider must expose ``artifact_lookup_keys(vllm_config) ->
    (model_id, profile, config_hash)`` for this to fire. Returns None
    on any lookup failure (no silent enable).
    """
    try:
        from .functional_artifact import find_matching
    except ImportError:
        return None
    if not hasattr(provider, "artifact_lookup_keys"):
        return None
    try:
        keys = provider.artifact_lookup_keys(vllm_config)
        if not keys:
            log.warning(
                "[SpecDecodeGuard] artifact_lookup_keys returned None — "
                "live config did not match any known profile shape",
            )
            return None
        model_id, profile, config_hash = keys
        log.warning(
            "[SpecDecodeGuard] artifact lookup keys: model_id=%r "
            "profile=%r config_hash=%r",
            model_id, profile, config_hash,
        )
        result = find_matching(
            model_id=model_id, profile=profile, config_hash=config_hash,
        )
        if result is None:
            log.warning(
                "[SpecDecodeGuard] no matching artifact for "
                "(%s, %s, %s)", model_id, profile, config_hash,
            )
        return result
    except Exception as _e:  # noqa: BLE001
        log.warning("[SpecDecodeGuard] artifact lookup failed: %s", _e)
        return None


__all__ = [
    "PairAssessment",
    "GuardDecision",
    "evaluate",
    "evaluate_from_config",
    "find_provider_for_config",
]
