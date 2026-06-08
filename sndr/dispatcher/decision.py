# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — should_apply() decision logic.

Layer-2/Layer-3 gates that determine whether a patch should run on the
current vllm + model + workload. Used by every wiring patch's apply()
function as the first gate.

Wiring usage:

    from sndr.dispatcher import should_apply, log_decision

    decision, reason = should_apply("P60")
    if not decision:
        log_decision("P60", decision, reason)
        return "skipped", reason

Migration history:
  - Original location: vllm/_genesis/dispatcher.py (Stage 0).
  - Stage 3 (CURRENT): split into dispatcher/decision.py.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .registry import PATCH_REGISTRY  # noqa: F401 (re-exported)

# v11.3.0 hot-path optimization: hoist per-decision imports to module
# level. should_apply() is called for every patch at boot AND from
# inside many patch apply()/runtime hooks. Cumulative ~200+ patches ×
# 4 processes × per-call attr-lookup = ~10ms saved at boot.
from sndr.env import is_disabled as _env_is_disabled
from sndr.env import is_enabled as _env_is_enabled

log = logging.getLogger("genesis.dispatcher")


def _live_registry() -> dict[str, dict[str, Any]]:
    """Resolve registry via the canonical SNDR Core dispatcher package.

    PR38 cleanup (2026-05-08): `sndr.dispatcher` re-exports
    `PATCH_REGISTRY` from `.registry` at package level. Tests now do
    `monkeypatch.setattr(sndr.dispatcher, "PATCH_REGISTRY", fake)`
    and this reader sees the patched attribute on the same package
    module — same identity, monkey-patch propagates.
    """
    from sndr import dispatcher as _canonical
    return _canonical.PATCH_REGISTRY


# ─── Layer 2: model-aware applies_to gate ─────────────────────────────────

def _check_applies_to(
    patch_id: str, meta: dict[str, Any]
) -> tuple[bool, str]:
    """Layer 2 model-compatibility gate.

    Reads `meta["applies_to"]` (a dict mapping profile-key → list of allowed
    values), looks up the live model profile via `model_detect.get_model_profile()`,
    and returns (compatible, reason).

    Profile keys recognized: 'model_class', 'quant_format', 'kv_cache_dtype',
    'is_moe', 'is_hybrid', 'is_turboquant'. Any key whose actual value is None
    (detector couldn't resolve) is treated as compatible (conservative — let
    the patch apply and have its own call-site guards decide).

    Returns:
        (True, reason)  — model matches all applies_to constraints (or none
                          declared, or model couldn't be resolved)
        (False, reason) — explicit incompatibility: actual_value not in
                          allowed set for at least one key
    """
    applies_to = meta.get("applies_to")
    if not applies_to:
        return True, "no applies_to declared (model-class agnostic)"

    try:
        from sndr.engines.vllm.detection.model_detect import get_model_profile
        profile = get_model_profile()
    except Exception as e:
        return True, f"model_detect probe failed ({e}) — conservative apply"

    if not profile.get("resolved", False):
        return True, "model profile unresolved — conservative apply"

    # Map profile keys onto applies_to keys (some applies_to use boolean
    # aliases like is_moe / is_hybrid / is_turboquant for readability).
    key_aliases = {
        "is_moe": "moe",
        "is_hybrid": "hybrid",
        "is_turboquant": "turboquant",
    }

    # Build a flat profile dict that combines model_detect output with
    # boolean-alias mapping so the new predicates evaluator can read both
    # "is_turboquant" (used in applies_to) and "turboquant" (model_detect
    # native key) interchangeably.
    flat_profile: dict[str, Any] = dict(profile)
    for applies_key, profile_key in key_aliases.items():
        if profile_key in profile and applies_key not in flat_profile:
            flat_profile[applies_key] = profile[profile_key]

    # ─── Path A: richer predicate DSL (compat/predicates) ────────────────
    # Detect compound keys: all_of / any_of / not / none_of. If present,
    # delegate to the new evaluator. This lets new patches use the richer
    # syntax while old ones keep working unchanged.
    compound_keys = ("all_of", "any_of", "not", "none_of")
    if any(k in applies_to for k in compound_keys):
        try:
            from sndr.compat.predicates import evaluate
            ok, reason = evaluate(applies_to, flat_profile)
            return ok, ("applies_to satisfied" if ok
                        else f"MODEL-COMPAT: {reason}")
        except Exception as e:
            log.warning(
                "[Genesis dispatcher] %s: predicate evaluator raised (%s) — "
                "conservative apply. Check applies_to syntax.",
                patch_id, e,
            )
            return True, f"predicate evaluator error ({e}) — conservative apply"

    # ─── Path B: legacy flat-dict applies_to (backward compatible) ───────
    # Also pull version-related keys out and check via version_check.
    version_keys = (
        "vllm_version_range", "torch_version_min", "triton_version_min",
        "cuda_runtime_min", "nvidia_driver_min", "python_version_min",
        "compute_capability_min", "compute_capability_max",
    )
    version_constraints = {k: v for k, v in applies_to.items() if k in version_keys}
    profile_constraints = {k: v for k, v in applies_to.items() if k not in version_keys}

    # Version range checks
    if version_constraints:
        try:
            from sndr.compat.version_check import (
                check_version_constraints,
            )
            v_ok, v_results = check_version_constraints(version_constraints)
            if not v_ok:
                failed = [r for r in v_results if r.matched is False]
                if failed:
                    return False, f"VERSION: {failed[0].reason}"
                return False, "VERSION: constraint violation"
        except Exception as e:
            log.debug("[Genesis dispatcher] %s: version_check failed (%s) — "
                      "conservative apply", patch_id, e)

    # Legacy profile gates
    for key, allowed in profile_constraints.items():
        profile_key = key_aliases.get(key, key)

        # ─── Architecture list gates (model_arch / architectures) ────────
        # Added 2026-06-05: previously these keys were silently ignored
        # (profile.get("model_arch") = None → continue) which meant 56
        # patches (1 P103 + 55 Gemma4) had their declared registry gate
        # silently bypassed. Runtime self-guards inside apply() saved
        # correctness, but the dispatcher's `patches doctor` reported
        # them as "applies_to satisfied" misleadingly.
        #
        # Safe semantics:
        #   - If `allowed` contains "*"  → registry gate ALWAYS passes
        #     (operator intent: "defer to runtime guard"). This matches
        #     the existing Gemma4 G4_* convention.
        #   - Else: at least one architecture from profile["architectures"]
        #     must be in `allowed`.
        if key in ("model_arch", "architectures"):
            if not isinstance(allowed, (list, tuple, set)):
                allowed = [allowed]
            if "*" in allowed:
                continue
            archs = profile.get("architectures", []) or []
            if not archs:
                continue  # unresolved → conservative
            if not any(a in allowed for a in archs):
                return False, (
                    f"MODEL-COMPAT: {key}={archs!r} not in {list(allowed)!r}"
                )
            continue  # at least one match

        actual = profile.get(profile_key)
        if actual is None:
            continue  # detector couldn't resolve → conservative
        if not isinstance(allowed, (list, tuple, set)):
            allowed = [allowed]
        if actual not in allowed:
            return False, (
                f"MODEL-COMPAT: {key}={actual!r} not in {list(allowed)!r}"
            )
    return True, "applies_to satisfied"


# ─── Single-call gate ─────────────────────────────────────────────────────



def _check_tier_gate(meta: dict[str, Any]) -> Optional[tuple[bool, str]]:
    """Phase 4 (F-010-012 audit fix, 2026-05-08): structured tier gate.

    ``tier="engine"`` patches require BOTH:

      - ``vllm.sndr_engine`` commercial package present
      - License key (env ``SNDR_ENGINE_LICENSE_KEY`` OR ``~/.sndr/license.json``)
      - sndr_engine major version matches sndr_core

    Previous gate (Stage 5, pre-PR38) only checked ``import sndr_engine``.
    The sndr_engine repo lives alongside sndr_core in this codebase so
    the import always succeeded — no real boundary. The structured
    check below adds license + version verification so the boundary
    becomes the policy choke point rather than a code-presence check.

    ``SNDR_ENABLE_TIER_OVERRIDE=1`` still forces community-only (skips
    engine patches even if licensed) — useful for CI and pure-community
    deployments.

    Returns ``None`` when the gate passes (community tier or eligible
    engine tier); returns ``(False, reason)`` when an engine patch
    fails the eligibility probe.
    """
    tier = meta.get("tier", "community")
    if tier == "engine":
        from sndr.license import check_engine_tier_eligible
        result = check_engine_tier_eligible()
        if not result.eligible:
            return False, f"tier=engine: {result.reason}"
    return None


def _resolve_env_state(
    meta: dict[str, Any],
) -> tuple[bool, bool, Optional[str], Optional[str]]:
    """Resolve the env-flag state for a patch.

    F-008 fix (2026-05-07): registry entries store ``env_flag`` in
    full-prefix form (e.g. ``"GENESIS_ENABLE_P58_..."``), but
    ``env.is_enabled`` expects a *bare* flag name and applies
    ``SNDR_↔GENESIS_`` alias internally (``SNDR_ENABLE_X`` wins,
    ``GENESIS_ENABLE_X`` is the legacy fallback). The old
    ``os.environ.get(env_flag)`` literal-lookup ignored the alias, so
    an operator who set ``SNDR_ENABLE_P58_...=1`` saw the patch stay
    skipped. Strip the canonical prefix before delegating to
    ``env.is_enabled``.

    Returns ``(env_truthy, env_disabled, bare_flag, env_flag)``.
    When the patch has no ``env_flag`` declared, returns
    ``(False, False, None, None)`` — caller proceeds to the default
    strict-opt-in branch.
    """
    env_flag = meta.get("env_flag")
    if not env_flag:
        return False, False, None, None

    bare_flag = env_flag
    for _prefix in ("SNDR_ENABLE_", "GENESIS_ENABLE_"):
        if bare_flag.startswith(_prefix):
            bare_flag = bare_flag[len(_prefix):]
            break

    # v11.3.0: use module-level imports (was per-call).
    env_truthy = _env_is_enabled(bare_flag)
    env_disabled = _env_is_disabled(bare_flag)
    return env_truthy, env_disabled, bare_flag, env_flag


def _check_disable_gate(
    patch_id: str,
    env_flag: Optional[str],
    bare_flag: Optional[str],
    env_truthy: bool,
    env_disabled: bool,
) -> Optional[tuple[bool, str]]:
    """F-2026-05-14: explicit operator opt-out via
    ``SNDR_DISABLE_<bare>=1`` / ``GENESIS_DISABLE_<bare>=1``.

    Before this gate, the only knobs ``should_apply()`` consulted
    were the ENABLE variants. For ``default_on=True`` patches that
    meant the community had no way to A/B-test the patch's
    contribution (or temporarily disable a regressing patch) without
    editing ``registry.py`` — the very workflow operators use during
    bench validation. ``env.is_disabled()`` has existed for a while
    but was never wired here.

    Precedence: DISABLE wins over ENABLE when both are set. Intent-
    clear opt-out semantics are what operators expect from a kill-
    switch — "I said disable, I meant disable, even if some other
    env still says enable." A WARNING is emitted on the conflict so
    the contradiction is visible.

    Returns ``None`` when the operator did not request DISABLE;
    returns ``(False, reason)`` when an opt-out env var is set.
    """
    if not env_disabled:
        return None
    if env_truthy:
        log.warning(
            "[Genesis dispatcher] %s: both ENABLE and DISABLE env "
            "flags set for %s — DISABLE wins. Drop one of the env "
            "vars to clear the conflict.",
            patch_id, bare_flag,
        )
    return False, (
        f"explicitly disabled by operator (SNDR_DISABLE_{bare_flag}=1 "
        f"or GENESIS_DISABLE_{bare_flag}=1). Drop the env var to "
        f"re-engage."
    )


def _resolve_env_override(
    patch_id: str, meta: dict[str, Any], env_flag: Optional[str],
) -> tuple[bool, str]:
    """Env-flag truthy path: operator override always applies (subject
    to anchor presence). ``applies_to`` is informational under env
    override; ``config_detect`` is consulted for the reason string but
    cannot block the apply.
    """
    # Layer 2 applies_to is informational under env-override
    compat, compat_reason = _check_applies_to(patch_id, meta)
    if not compat:
        log.warning(
            "[Genesis dispatcher] %s: env OVERRIDE applies_to mismatch — "
            "%s. Proceeding because operator set %s=1.",
            patch_id, compat_reason, env_flag,
        )
    # Still consult config_detect to PRINT the recommendation as info
    try:
        from sndr.engines.vllm.detection.config_detect import recommend
        verdict, reason = recommend(patch_id)
        if verdict == "apply":
            return True, f"opt-in env + config recommends apply: {reason}"
        elif verdict == "neutral":
            return True, "opt-in env (config: neutral)"
        else:
            return True, (
                f"opt-in env OVERRIDE (config recommends {verdict}: "
                f"{reason}) — proceeding because operator forced it"
            )
    except Exception as e:
        return True, f"opt-in env (config_detect probe failed: {e})"


def _is_legacy_default_on_mode() -> bool:
    """Backward-compat escape hatch — ``GENESIS_LEGACY_DEFAULT_ON=1``
    reverts to pre-2026-05-17 auto-apply semantics."""
    return os.environ.get("GENESIS_LEGACY_DEFAULT_ON", "").strip().lower() in (
        "1", "true", "yes",
    )


def _resolve_legacy_default_on(
    patch_id: str, meta: dict[str, Any], env_flag: Optional[str],
) -> tuple[bool, str]:
    """Legacy path — preserved for backward compat with the
    pre-2026-05-17 auto-apply behaviour. Opt-in patches stay skipped;
    ``default_on=True`` patches enforce ``applies_to`` as a Layer-2
    HARD skip and then consult ``config_detect``.
    """
    if not meta.get("default_on", False):
        if meta.get("deprecated", False):
            return False, (
                f"opt-in only AND empirically deprecated — "
                f"keeping skip; set {env_flag}=1 only for diagnostics"
            )
        return False, f"opt-in only — set {env_flag}=1 to engage"

    # default_on=True: enforce applies_to as Layer 2 HARD skip.
    compat, compat_reason = _check_applies_to(patch_id, meta)
    if not compat:
        log.warning(
            "[Genesis dispatcher] %s HARD-SKIP — %s. Patch designed for "
            "a different model class; skipping to avoid overhead. Set "
            "%s=1 to force-apply if you know what you are doing.",
            patch_id, compat_reason, env_flag,
        )
        return False, compat_reason

    # default_on=True patches still consult config_detect
    try:
        from sndr.engines.vllm.detection.config_detect import recommend
        verdict, reason = recommend(patch_id)
        return (
            verdict in ("apply", "neutral"),
            f"config_detect: {verdict}:{reason}",
        )
    except Exception as e:
        return False, f"config_detect failed: {e}"


def _resolve_strict_opt_in(
    meta: dict[str, Any], env_flag: Optional[str],
) -> tuple[bool, str]:
    """STRICT OPT-IN POLICY (Sander directive 2026-05-17).

    Patches activate ONLY when explicitly listed in operator's config
    via ``env_flag``. ``default_on=True`` becomes INFORMATIONAL (used
    by recommendation docs + bench profiles), it does NOT trigger
    auto-apply anymore.

    Rationale: prior behaviour caused operator surprise — patches
    active in production whose env flag was never set in the launch
    script. With 200+ patches in registry, implicit activation made
    it hard to reason about which patches were live. Strict opt-in
    makes the active stack exactly the set of env flags in the
    config — no hidden auto-applies.
    """
    if meta.get("deprecated", False):
        return False, (
            f"strict opt-in + deprecated — set {env_flag}=1 only for "
            f"diagnostics (or GENESIS_LEGACY_DEFAULT_ON=1 for old behaviour)"
        )
    if meta.get("default_on", False):
        return False, (
            f"strict opt-in: patch has default_on=True (informational) but "
            f"env_flag={env_flag} unset. Add {env_flag}=1 to launch config "
            f"to engage. Set GENESIS_LEGACY_DEFAULT_ON=1 to revert to "
            f"pre-2026-05-17 auto-apply semantics."
        )
    return False, (
        f"strict opt-in — set {env_flag}=1 in config to engage "
        f"(GENESIS_LEGACY_DEFAULT_ON=1 reverts to pre-2026-05-17 semantics)"
    )


def should_apply(patch_id: str) -> tuple[bool, str]:
    """Unified gate: returns (apply_decision, reason).

    Combines:
      - env-flag check (``GENESIS_ENABLE_P<patch>=1`` opt-in)
      - ``applies_to`` model-compatibility hard-skip (Layer 2, opt-in
        patches respect env override; default_on patches honor it
        strictly)
      - ``config_detect.recommend(patch_id)`` (model+config-aware
        decision)

    The decision rule:

      1. If env flag is truthy AND patch is opt-in (default_on=False)
         → apply, operator override wins over applies_to (logged as
         override).
      2. If env flag is unset/falsy AND patch is ``default_on=False``
         → skip (opt-in)
      3. If applies_to declared and actual model profile mismatches
         → hard-skip with WARNING-class reason ("MODEL-COMPAT: ...").
         For default_on=True patches this kicks in unconditionally;
         for env-truthy opt-ins it's logged but apply proceeds (override).
      4. Otherwise consult ``config_detect.recommend()``:
         - "skip:..." → don't apply
         - "redundant:..." → don't apply
         - "deprecated:..." → don't apply
         - "neutral" / "apply" → apply

    M.1.1.T1.B restructure (2026-05-27): the original 175-LOC body is
    split into private named helpers below. Decision order, reason
    strings (byte-identical), and the public ``(bool, str)`` contract
    are preserved; the
    ``tests/unit/dispatcher/fixtures/decision_no_env.json`` snapshot
    is the byte-identity guard.

    Returns:
        (True, reason) — patch should apply
        (False, reason) — patch should skip, with human-readable reason
    """
    meta = _live_registry().get(patch_id)
    if meta is None:
        return False, f"unknown patch_id {patch_id!r}"

    tier_decision = _check_tier_gate(meta)
    if tier_decision is not None:
        return tier_decision

    env_truthy, env_disabled, bare_flag, env_flag = _resolve_env_state(meta)

    disable_decision = _check_disable_gate(
        patch_id, env_flag, bare_flag, env_truthy, env_disabled,
    )
    if disable_decision is not None:
        return disable_decision

    if env_truthy:
        return _resolve_env_override(patch_id, meta, env_flag)

    if _is_legacy_default_on_mode():
        return _resolve_legacy_default_on(patch_id, meta, env_flag)

    return _resolve_strict_opt_in(meta, env_flag)


# ─── Decision logging ─────────────────────────────────────────────────────

# Module-level cache of decisions made this boot, for matrix dump.
_DECISIONS: list[dict[str, Any]] = []


def log_decision(patch_id: str, applied: bool, reason: str) -> None:
    """Log + record a patch decision for the boot-time matrix dump.

    Single condensed line per patch. Operator can see all decisions at boot
    via `Genesis Dispatcher v2 decisions:` log block (called from apply_all).
    """
    meta = _live_registry().get(patch_id, {})
    title = meta.get("title", patch_id)
    status = "APPLY" if applied else "SKIP "
    log.info(
        "[Genesis Dispatcher] %s %s — %s | %s",
        status, patch_id, title, reason[:120],
    )
    _DECISIONS.append({
        "patch_id": patch_id,
        "title": title,
        "applied": applied,
        "reason": reason,
        "env_flag": meta.get("env_flag", ""),
        "credit": meta.get("credit", ""),
        "upstream_pr": meta.get("upstream_pr"),
    })


