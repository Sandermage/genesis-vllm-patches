# SPDX-License-Identifier: Apache-2.0
"""Patch-plan resolver — Phase B of the patch-attribution integration plan.

The resolver answers the question "given a composed ModelConfig and a
policy name, which env-flag entries should the runtime actually
include?". It does *not* override the dispatcher's runtime decision
(``should_apply()``); it is a *read-only* explanation layer used by
``sndr patches plan --explain`` and (in Phase C) ``sndr compose
render --policy``.

Contract:

  PatchPlan
    .policy      "compat" | "safe" | "minimal"
    .included    tuple of PatchDecision (env_flag stays in the runtime)
    .excluded    tuple of PatchDecision (env_flag dropped, with reason)
    .warnings    tuple of strings (advisory only; never blocking here)

  PatchDecision per entry carries patch_id, env_flag, value, role,
  reason, note, bench_evidence — everything a CLI explainer needs.

Policy semantics:

  compat  : pass-through. Every flag in cfg.genesis_env with value != "0"
            is included. Excluded set holds only operator-disabled
            (value == "0") entries — so compose render can still tell
            "operator put this here on purpose" from "patch never was on".

  safe    : compat + drop role == "no_op".
            Operator-visible payoff: shorter env block, fewer red
            herrings in diagnose runs. Conservative — won't drop
            anything that *might* be doing work.

  minimal : safe + drop role in {"suspected_regression", "unknown"}.
            For advanced operators who curated attribution and want a
            lean stack. Unknowns are dropped because by definition
            we cannot defend their inclusion in a minimal preset.

See `docs/_internal/PATCH_ATTRIBUTION_COMPOSE_GENERATOR_INTEGRATION_PLAN_2026-05-16_RU.md`
§ 6 for the algorithm and § 5.4 for the data shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid runtime cycle (schema imports nothing from us)
    from .schema import ModelConfig, PatchAttribution


_VALID_POLICIES: tuple[str, ...] = ("compat", "safe", "minimal")


def _build_env_flag_to_patch_id() -> dict[str, tuple[str, ...]]:
    """Return reverse map env_flag → sorted tuple of patch IDs.

    Most env flags map 1:1 to a single patch ID, but A-19-exempt
    "tightly coupled subpatch" families share a flag — flipping
    ``GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL`` enables BOTH
    P67 and P67b together; the same with PN40 + PN40-classifier.
    Storing a tuple keeps the family visible to conflict-detection
    and avoids "last write wins" patch dropouts in the reverse map.

    Result is alphabetically sorted so resolution is deterministic
    (the primary ID surfaced by ``_patch_id_from_env_flag()`` is
    always the alphabetical first, regardless of registry dict
    iteration order).
    """
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    families: dict[str, list[str]] = {}
    for pid, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag") if isinstance(meta, dict) else None
        if isinstance(flag, str) and flag:
            families.setdefault(flag, []).append(pid)
    return {flag: tuple(sorted(pids)) for flag, pids in families.items()}


_ENV_FLAG_TO_PATCH_ID: dict[str, tuple[str, ...]] | None = None


_GENESIS_PREFIX_RE = None  # lazy-initialised, see _strip_genesis_prefix


def _strip_genesis_prefix(env_flag: str) -> str:
    """Strip ``GENESIS_(ENABLE|DISABLE)_`` prefix → bare suffix.

    Synthetic fallback for env flags that don't appear in the live
    registry reverse index (test fixtures, custom operator overrides).
    Returns the input unchanged when the prefix isn't present.
    """
    global _GENESIS_PREFIX_RE
    if _GENESIS_PREFIX_RE is None:
        import re
        _GENESIS_PREFIX_RE = re.compile(r"^GENESIS_(?:ENABLE|DISABLE)_")
    m = _GENESIS_PREFIX_RE.match(env_flag)
    return env_flag[m.end():] if m else env_flag


def _patch_family_for_env_flag(env_flag: str) -> tuple[str, ...]:
    """Resolve env_flag → full family of patch IDs (sorted).

    Most flags have a one-element family. A-19 tight-coupled
    subpatches (P67 + P67b, PN40 + PN40-classifier) share an
    env_flag — both flip together — and the resolver needs to see
    every family member for conflict detection.
    """
    global _ENV_FLAG_TO_PATCH_ID
    if _ENV_FLAG_TO_PATCH_ID is None:
        try:
            _ENV_FLAG_TO_PATCH_ID = _build_env_flag_to_patch_id()
        except Exception:
            _ENV_FLAG_TO_PATCH_ID = {}
    hit = _ENV_FLAG_TO_PATCH_ID.get(env_flag)
    if hit is not None:
        return hit
    return (_strip_genesis_prefix(env_flag),)


def _patch_id_from_env_flag(env_flag: str) -> str:
    """Return the primary patch ID for an env flag.

    For single-member families this is just the patch ID. For A-19
    families it's the alphabetical first (P67 before P67b, PN40
    before PN40-classifier) — chosen so resolution is deterministic
    and operator-facing output is stable across registry edits.
    """
    return _patch_family_for_env_flag(env_flag)[0]


@dataclass(frozen=True)
class PatchDecision:
    """One env-flag entry's include/exclude decision under a policy.

    `patch_id` is best-effort: it's derived from the env-flag name when
    the flag follows the canonical ``GENESIS_(ENABLE|DISABLE)_<PID>``
    pattern, else the raw flag name. Role-related fields default to
    "unknown" / empty string when the cfg has no attribution for this
    patch.
    """
    patch_id: str
    env_flag: str
    value: str
    decision: str           # "include" | "exclude"
    role: str               # PatchAttribution.role or "unknown"
    reason: str             # human-readable rationale
    note: str = ""          # attribution.note (verbatim)
    bench_evidence: str = ""  # attribution.bench_evidence (verbatim)


@dataclass(frozen=True)
class PatchPlan:
    """Result of `resolve_patch_plan()` — what's in/out under a policy.

    ``included`` / ``excluded`` only carry decisions for *patch toggle*
    keys (``GENESIS_ENABLE_*`` / ``GENESIS_DISABLE_*``). Other
    ``GENESIS_*`` env vars in cfg.genesis_env (parameter knobs like
    ``GENESIS_PN95_CONFIG_KEY``, ``GENESIS_BUFFER_MODE``,
    ``GENESIS_PROFILE_RUN_CAP_M``) are kept as-is in ``passthrough``
    and merged into ``env`` regardless of policy. Filtering parameter
    values would break the patches they configure.
    """
    policy: str
    included: tuple[PatchDecision, ...] = ()
    excluded: tuple[PatchDecision, ...] = ()
    warnings: tuple[str, ...] = ()
    passthrough: dict[str, str] = field(default_factory=dict)

    @property
    def env(self) -> dict[str, str]:
        """Drop-in replacement for ``cfg.genesis_env`` after filtering.

        Merges policy-included toggle flags with the verbatim passthrough
        parameter keys. Order-preserving: included toggles first (the
        order they appeared in cfg.genesis_env), passthrough keys after."""
        out: dict[str, str] = {d.env_flag: d.value for d in self.included}
        out.update(self.passthrough)
        return out


# ─── Helpers ─────────────────────────────────────────────────────────────


def _attribution_for(
    attribution: dict[str, "PatchAttribution"],
    env_flag: str,
) -> tuple[str, "PatchAttribution | None"]:
    """Return (primary_patch_id, attribution_entry_or_None).

    Resolution rule for A-19 family attribution:

      1. Check attribution keyed by the family primary (alphabetical
         first member). This is the dominant case — operators usually
         attribute the canonical patch ID that surfaces in plan output.
      2. If primary has no attribution, fall back to any non-primary
         family member that does have an entry. Lets operators key
         attribution by the operator-facing main patch (e.g. "PN40"
         when family[0] happens to be a sub-id alphabetically first).
      3. Otherwise return (primary, None) → role defaults to "unknown".

    The surfaced patch_id is always the primary so PatchDecision stays
    deterministic across operators and registry edits — only the
    metadata source falls back through the family.
    """
    family = _patch_family_for_env_flag(env_flag)
    primary = family[0]
    attr = attribution.get(primary)
    if attr is not None:
        return primary, attr
    for pid in family[1:]:
        attr = attribution.get(pid)
        if attr is not None:
            return primary, attr
    return primary, None


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _is_toggle_flag(env_flag: str) -> bool:
    """Classify a GENESIS_* env key as a patch toggle vs a parameter.

    Toggle keys start with ``GENESIS_ENABLE_`` or ``GENESIS_DISABLE_``
    and pair an enable verb with a P-style patch ID. Parameter keys
    like ``GENESIS_BUFFER_MODE``, ``GENESIS_PN95_TICK_EVERY``,
    ``GENESIS_PROFILE_RUN_CAP_M`` configure how a patch behaves once
    it's active — filtering them by policy would break the dependent
    patch silently. They pass through ``plan.env`` untouched.

    Conservative match: must start with the canonical
    ``GENESIS_(ENABLE|DISABLE)_`` prefix. Bare ``GENESIS_<X>`` keys
    are always treated as parameters (passthrough).
    """
    return env_flag.startswith("GENESIS_ENABLE_") or env_flag.startswith(
        "GENESIS_DISABLE_"
    )


# ─── Main resolver ───────────────────────────────────────────────────────


def resolve_patch_plan(
    cfg: "ModelConfig",
    policy: str = "compat",
) -> PatchPlan:
    """Compute the patch plan for a composed ``ModelConfig``.

    Args:
        cfg: A V1 ModelConfig produced by compose() (or built by hand
            in tests). The resolver reads ``cfg.genesis_env`` for the
            current canonical env-flag set and ``cfg.patches_attribution``
            for the role/note/evidence metadata.
        policy: One of "compat", "safe", "minimal". See module docstring.

    Returns:
        A frozen ``PatchPlan`` with included/excluded decisions and an
        empty warnings tuple (Phase B is read-only; warnings reserved
        for future delivery-path checks in Phase C).

    Raises:
        ValueError: when ``policy`` is not in ``_VALID_POLICIES``.
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"policy={policy!r} not in {_VALID_POLICIES} — "
            f"resolver supports the documented modes only"
        )

    attribution = getattr(cfg, "patches_attribution", {}) or {}
    genesis_env = getattr(cfg, "genesis_env", {}) or {}

    included: list[PatchDecision] = []
    excluded: list[PatchDecision] = []
    passthrough: dict[str, str] = {}

    for env_flag, raw_value in genesis_env.items():
        value = str(raw_value)
        # Parameter keys (GENESIS_BUFFER_MODE, GENESIS_PN95_TICK_EVERY,
        # GENESIS_PROFILE_RUN_CAP_M, P67_NUM_KV_SPLITS, etc.) pass
        # through unconditionally — their values configure how an
        # already-decided patch behaves, not whether the patch fires.
        if not _is_toggle_flag(env_flag):
            passthrough[env_flag] = value
            continue
        pid, attr = _attribution_for(attribution, env_flag)
        role = attr.role if attr is not None else "unknown"
        note = (attr.note or "") if attr is not None else ""
        bench = (attr.bench_evidence or "") if attr is not None else ""

        # First filter: operator-disabled (value=="0") is always excluded
        # under every policy. The decision surfaces in `.excluded` so a
        # downstream diff can show "operator turned this off here".
        if not _is_truthy(value):
            excluded.append(PatchDecision(
                patch_id=pid, env_flag=env_flag, value=value,
                decision="exclude", role=role,
                reason="operator-disabled in genesis_env (value != truthy)",
                note=note, bench_evidence=bench,
            ))
            continue

        # Policy filters — applied in order of aggressiveness.
        drop_for: str | None = None
        if policy in ("safe", "minimal") and role == "no_op":
            drop_for = "role='no_op' (declared inactive on this config)"
        elif policy == "minimal" and role == "suspected_regression":
            drop_for = "role='suspected_regression' (minimal policy)"
        elif policy == "minimal" and role == "unknown":
            drop_for = "role='unknown' (no attribution; minimal policy)"

        if drop_for is not None:
            excluded.append(PatchDecision(
                patch_id=pid, env_flag=env_flag, value=value,
                decision="exclude", role=role,
                reason=drop_for, note=note, bench_evidence=bench,
            ))
        else:
            included.append(PatchDecision(
                patch_id=pid, env_flag=env_flag, value=value,
                decision="include", role=role,
                reason=f"role={role!r} kept under policy={policy!r}",
                note=note, bench_evidence=bench,
            ))

    warnings = (
        _detect_candidate_when_warnings(included, attribution, cfg)
        + _detect_conflict_warnings(included)
    )

    return PatchPlan(
        policy=policy,
        included=tuple(included),
        excluded=tuple(excluded),
        warnings=warnings,
        passthrough=passthrough,
    )


# ─── candidate_when predicate evaluator ──────────────────────────────────


# Supported predicate suffixes — pure operator-facing ergonomics on top
# of a flat ``{key: expected}`` map. Suffix selects the comparison
# operator; suffix-less keys default to list-membership when expected
# is list, else equality.
_PREDICATE_SUFFIXES: tuple[str, ...] = ("_gte", "_lte", "_eq")


# Map from predicate base name (with suffix stripped) to a callable that
# extracts the comparable value from ``cfg``. Operator-authored
# candidate_when keys are intentionally limited to fields a deployment
# operator can reason about — model_class / cuda_capability stay on
# registry.applies_to (which the runtime dispatcher consults), not here.
def _cfg_value_for_predicate(cfg, base_key: str):
    if base_key == "max_num_seqs":
        return getattr(cfg, "max_num_seqs", None)
    if base_key == "max_model_len":
        return getattr(cfg, "max_model_len", None)
    if base_key == "n_gpus":
        hw = getattr(cfg, "hardware", None)
        return getattr(hw, "n_gpus", None) if hw is not None else None
    if base_key == "tool_call_parser":
        return getattr(cfg, "tool_call_parser", None)
    if base_key == "reasoning_parser":
        return getattr(cfg, "reasoning_parser", None)
    if base_key == "kv_cache_dtype":
        return getattr(cfg, "kv_cache_dtype", None)
    if base_key == "quantization":
        return getattr(cfg, "quantization", None)
    if base_key == "dtype":
        return getattr(cfg, "dtype", None)
    return _UNKNOWN_KEY


_UNKNOWN_KEY = object()
# Sentinel — distinguishes "key unknown to resolver" from "key known but
# value is None". The unknown case produces a forward-compat warning
# but the resolver doesn't fail closed (operators may author new
# predicate names that the resolver hasn't been updated to recognise).


def _evaluate_candidate_when(
    cw: dict,
    cfg,
) -> tuple[bool, str]:
    """Return ``(matches, mismatch_reason)`` for a candidate_when dict.

    Conjunctive: every key in cw must satisfy its predicate. Empty /
    None cw matches everything. The mismatch_reason names the FIRST
    failing predicate so the warning is concrete; if multiple fail
    only the first surfaces (cheap, deterministic).

    Unknown predicate keys are surfaced as ``KEY=<unknown>`` and the
    resolver treats them as a mismatch — the warning surfaces them
    so operators see the typo / forward-compat gap; behaviour stays
    safe (warning, never silently include).
    """
    if not cw:
        return True, ""
    for key, expected in cw.items():
        base = key
        op = "list_or_eq"
        for suffix in _PREDICATE_SUFFIXES:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                op = suffix[1:]  # "gte" / "lte" / "eq"
                break

        actual = _cfg_value_for_predicate(cfg, base)
        if actual is _UNKNOWN_KEY:
            return False, (
                f"unknown predicate key {key!r} (resolver doesn't know "
                f"how to evaluate it against the current cfg)"
            )

        ok: bool
        if op == "gte":
            ok = actual is not None and actual >= expected
        elif op == "lte":
            ok = actual is not None and actual <= expected
        elif op == "eq":
            ok = actual == expected
        else:
            # list_or_eq: bare key with list-valued expected → membership;
            # bare key with scalar expected → equality (rare but supported
            # for symmetry with applies_to-style predicates).
            if isinstance(expected, list):
                ok = actual in expected
            else:
                ok = actual == expected
        if not ok:
            return False, (
                f"{key}={actual!r} doesn't satisfy candidate_when "
                f"(wanted {expected!r})"
            )
    return True, ""


def _detect_candidate_when_warnings(
    included: list[PatchDecision],
    attribution: dict[str, "PatchAttribution"],
    cfg,
) -> tuple[str, ...]:
    """For every included decision with an attribution carrying
    ``candidate_when``, evaluate the predicate against cfg and append
    one warning per mismatch.

    Warnings are advisory — they do NOT move the patch to excluded.
    Rationale (intentional design):

      * candidate_when is operator-authored hint metadata. Filtering
        on it could surprise an operator who knows the patch helps
        despite the predicate (e.g. a defensive fix that doesn't need
        the conditions it was originally measured under).
      * Future iteration may add a ``mode: filter`` field to escalate
        to exclusion. Phase D ships warning-only so the rollout is
        observable before any exclusion behaviour ships.
    """
    if not attribution:
        return ()
    warnings: list[str] = []
    for d in included:
        attr = attribution.get(d.patch_id)
        if attr is None:
            # Decision's primary patch_id may be the A-19 alphabetical
            # first; attribution might sit on a family sibling.
            family = _patch_family_for_env_flag(d.env_flag)
            for pid in family:
                if pid in attribution:
                    attr = attribution[pid]
                    break
        if attr is None or not attr.candidate_when:
            continue
        matches, reason = _evaluate_candidate_when(attr.candidate_when, cfg)
        if not matches:
            warnings.append(
                f"candidate_when: {d.patch_id} ({d.env_flag}) — {reason}. "
                f"Patch stays included under policy={d.reason.split(' ')[1]} "
                f"but may be a no-op at runtime."
            )
    return tuple(warnings)


def _detect_conflict_warnings(
    included: list[PatchDecision],
) -> tuple[str, ...]:
    """Surface ``conflicts_with`` violations among the included set.

    Reads PATCH_REGISTRY[<pid>].conflicts_with for every included
    patch. When two members of ``included`` declare each other (or one
    declares the other), append one canonical "A ⨯ B" warning. The
    pair is sorted alphabetically so the (A→B) and (B→A) views of
    the same conflict produce only one warning string, not two.

    Advisory only — does not move patches between included / excluded.
    Resolver is read-only; the dispatcher remains the runtime gate.
    """
    if len(included) < 2:
        return ()
    try:
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    except Exception:
        # Resolver stays usable even when registry can't load.
        return ()

    # Expand each included decision into its full family (A-19 subpatches
    # share env_flag → the decision's patch_id is the primary, but both
    # members are runtime-active and either can trigger a conflict).
    included_family_ids: set[str] = set()
    decision_families: list[tuple[PatchDecision, tuple[str, ...]]] = []
    for d in included:
        family = _patch_family_for_env_flag(d.env_flag)
        included_family_ids.update(family)
        decision_families.append((d, family))

    seen_pairs: set[tuple[str, str]] = set()
    warnings: list[str] = []
    for d, family in decision_families:
        for pid in family:
            meta = PATCH_REGISTRY.get(pid)
            if not isinstance(meta, dict):
                continue
            conflicts = meta.get("conflicts_with") or []
            if isinstance(conflicts, str):
                conflicts = [conflicts]
            for other in conflicts:
                if other not in included_family_ids:
                    continue
                if other == pid:
                    continue  # self-conflict declarations are bogus, skip
                pair = tuple(sorted((pid, other)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                warnings.append(
                    f"conflict: {pair[0]} ⨯ {pair[1]} — both included "
                    f"but registry declares them incompatible. Drop one "
                    f"or override via profile.patches_delta.disable."
                )
    return tuple(warnings)


__all__ = ["PatchDecision", "PatchPlan", "resolve_patch_plan"]
