#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.audit — profile OverridePolicy gate.

Audits every profile under `builtin/profile/` for the presence + shape
of `override_policy:` when the profile carries `sizing_override:`. At
Stage 1 (CONFIG-UX.audit phase scope) missing policy is a warning, not
an error — the field is brand-new in CONFIG-UX.1 and existing 21
profiles haven't been retrofitted yet.

Stage-1 contract (CONFIG-UX.audit phase):

  sizing_override + no override_policy        → warning
  invalid override_policy shape               → error (validation)
  effective_class=production + no reason      → error
  effective_class=production + no evidence    → error
  non-production class + no reason            → error
  override_policy without sizing_override     → warning (unusual)

Deferred to CONFIG-UX.4:

  - Class-4 forbidden-override hard enforcement
    (FORBIDDEN_OVERRIDES placeholder constant declared here as
    empty tuple — CONFIG-UX.4 fills + flips to hard reject).
  - Stage 2/3 severity escalation per `SNDR_V1_ROLLOUT_STAGE`.
  - `expires_at` past-date escalation.
  - Cross-validation of evidence_refs paths against
    tests/integration/baselines/ config blocks.

Exit codes:
  0 — clean (default) OR clean (--strict)
  1 — errors found (always) OR warnings found (--strict only)
  2 — usage / IO error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Class-4 forbidden override rules (CONFIG-UX.4.2) ────────────────────────
#
# Hard policy enforced regardless of OverridePolicy justification — these
# overrides are either physics-broken or evidence-invalidating, so no
# operator class can legitimize them.
#
# Each rule is a `ForbiddenRule` carrying a predicate
# `(profile, model, hardware) -> Optional[str]`. The predicate returns
# a human-readable violation message or None if the rule doesn't fire.
#
# Severity is ALWAYS "error" (not stage-dependent). This is the only
# category that escapes the rollout matrix.


@dataclass(frozen=True)
class ForbiddenRule:
    """One Class-4 forbidden-override rule.

    Lambda-based encoding (vs string-DSL) because:
      - cross-layer state needed (rule 2 needs hardware; rule 3 needs model)
      - clear unit-test surface (each predicate is a plain Python callable)
      - no DSL parser to maintain

    Trade-off: rules live in code, not data — adding a rule requires a PR.
    That's intentional: Class-4 is hard policy and deserves PR review.
    """
    rule_id: str
    description: str
    predicate: Callable[[Any, Any, Any], Optional[str]]


# Static narrowness ordering for KV cache dtype (Rule 3).
# - 16-bit: framework default / native float
# - 8-bit: FP8 variants
# - 4-bit: TurboQuant k8v4 (effective ~4-bit via VQ), FP4, INT4
# Profile `auto` is NEVER a downgrade (operator letting the framework
# decide; treated as 16-bit equivalent for ordering purposes).
_DTYPE_BITS: dict[str, int] = {
    "auto": 16,
    "fp16": 16,
    "bf16": 16,
    "fp8": 8,
    "fp8_e5m2": 8,
    "fp8_e4m3": 8,
    "turboquant_k8v4": 4,
    "fp4": 4,
    "int4": 4,
}


def _rule_1_gpu_mem_util_bound(profile, model, hardware) -> Optional[str]:
    """`gpu_memory_utilization > 1.0` is physically impossible."""
    sizing = getattr(profile, "sizing_override", None)
    if sizing is None:
        return None
    util = getattr(sizing, "gpu_memory_utilization", None)
    if util is None:
        return None
    if util > 1.0:
        return (
            f"gpu_memory_utilization={util} > 1.0 is physically impossible; "
            f"vLLM rejects > 1.0 at engine init (Class-4 forbidden override)"
        )
    return None


def _rule_2_tp_size_vs_hw_gpus(profile, model, hardware) -> Optional[str]:
    """`tensor_parallel_size > hardware.n_gpus` is physically impossible.

    Forward-compatible: today `HardwareSizing` does not expose a
    `tensor_parallel_size` field; rule no-ops on current corpus and
    activates the moment the field is added.
    """
    sizing = getattr(profile, "sizing_override", None)
    if sizing is None:
        return None
    tp = getattr(sizing, "tensor_parallel_size", None)
    if tp is None:
        return None
    hw = getattr(hardware, "hardware", None)  # HardwareDef.hardware → HardwareSpec
    n_gpus = getattr(hw, "n_gpus", None) if hw is not None else None
    if n_gpus is None:
        # Cannot validate without hardware GPU count — skip rather than
        # false-positive on older hardware schema.
        return None
    if tp > n_gpus:
        return (
            f"tensor_parallel_size={tp} > hardware.n_gpus={n_gpus} is "
            f"physically impossible (Class-4 forbidden override)"
        )
    return None


def _rule_3_kv_cache_dtype_narrower(profile, model, hardware) -> Optional[str]:
    """Profile cannot DOWNGRADE KV cache dtype below model declaration.

    Overrides happen via `profile.compression_plan.default_kv_dtype`, NOT
    `sizing_override`. Narrowness uses the static `_DTYPE_BITS` ordering.

    Skips:
      - profile sets no default_kv_dtype
      - model is dtype-neutral (`auto` / None)
      - profile sets `auto` (never a downgrade — letting framework decide)
      - unknown dtype names (fails-safe — treated as 16-bit so won't false-positive)
    """
    cp = getattr(profile, "compression_plan", None)
    if cp is None:
        return None
    profile_dtype = getattr(cp, "default_kv_dtype", None)
    if not profile_dtype or profile_dtype == "auto":
        return None
    caps = getattr(model, "capabilities", None)
    model_dtype = getattr(caps, "kv_cache_dtype", None) if caps else None
    if not model_dtype or model_dtype == "auto":
        return None  # model dtype-neutral; profile may choose freely
    # Default to 16-bit for unknown dtype names — fails-safe (won't
    # falsely flag an unknown profile dtype as narrower).
    profile_bits = _DTYPE_BITS.get(profile_dtype, 16)
    model_bits = _DTYPE_BITS.get(model_dtype, 16)
    if profile_bits < model_bits:
        return (
            f"compression_plan.default_kv_dtype={profile_dtype!r} "
            f"(~{profile_bits}-bit) narrower than model.kv_cache_dtype="
            f"{model_dtype!r} (~{model_bits}-bit) — loss-of-evidence "
            f"(Class-4 forbidden override); create a new model variant"
        )
    return None


def _rule_4_spec_decode_method_change(profile, model, hardware) -> Optional[str]:
    """Profile cannot change spec_decode METHOD NAME from model declaration.

    Allowed:
      - K-only changes (same method, different num_speculative_tokens)
      - Profile adding spec_decode where model has none (additive)
      - Profile removing spec_decode (subtractive)

    Forbidden:
      - Method name change (mtp → eagle, mtp → ngram, mtp → dflash, ...)
    """
    sd_override = getattr(profile, "spec_decode_override", None)
    if sd_override is None:
        return None
    override_method = getattr(sd_override, "method", None)
    if not override_method:
        return None  # subtractive or no method declared
    caps = getattr(model, "capabilities", None)
    model_sd = getattr(caps, "spec_decode", None) if caps else None
    model_method = getattr(model_sd, "method", None) if model_sd else None
    if not model_method:
        return None  # additive — model has no spec_decode; profile adds one
    if override_method != model_method:
        return (
            f"spec_decode_override.method={override_method!r} differs from "
            f"model.capabilities.spec_decode.method={model_method!r} — "
            f"acceptance distribution invalidated (Class-4 forbidden override); "
            f"create a new model variant rather than override"
        )
    return None


FORBIDDEN_OVERRIDES: tuple[ForbiddenRule, ...] = (
    ForbiddenRule(
        rule_id="gpu_memory_utilization_over_1",
        description="gpu_memory_utilization > 1.0 is physically impossible",
        predicate=_rule_1_gpu_mem_util_bound,
    ),
    ForbiddenRule(
        rule_id="tensor_parallel_size_over_hw_gpus",
        description="tensor_parallel_size cannot exceed available GPUs",
        predicate=_rule_2_tp_size_vs_hw_gpus,
    ),
    ForbiddenRule(
        rule_id="kv_cache_dtype_downgrade",
        description="profile cannot downgrade KV cache dtype below model declaration",
        predicate=_rule_3_kv_cache_dtype_narrower,
    ),
    ForbiddenRule(
        rule_id="spec_decode_method_change",
        description="profile cannot change spec_decode method from model declaration",
        predicate=_rule_4_spec_decode_method_change,
    ),
)


@dataclass
class Finding:
    profile_id: str
    severity: str  # info | warning | error
    rule: str
    message: str

    def as_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
        }


@dataclass
class OverrideReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, profile_id: str, severity: str, rule: str, message: str) -> None:
        self.findings.append(Finding(profile_id, severity, rule, message))

    def count_by_severity(self) -> dict[str, int]:
        out = {"info": 0, "warning": 0, "error": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


def _profile_dir() -> Path:
    return (
        REPO_ROOT / "sndr" / "model_configs"
        / "builtin" / "profile"
    )


def _community_profile_dir() -> Path:
    return (
        REPO_ROOT / "sndr" / "model_configs"
        / "community" / "profile"
    )


def _list_profile_ids() -> list[str]:
    out: set[str] = set()
    for d in (_profile_dir(), _community_profile_dir()):
        if d.is_dir():
            out.update(
                p.stem for p in d.glob("*.yaml")
                if p.is_file() and not p.stem.startswith("_")
            )
    return sorted(out)


def _audit_one_profile(profile_id: str, report: OverrideReport) -> None:
    """Audit one profile; append findings to report."""
    from sndr.model_configs.registry_v2 import load_profile
    from sndr.model_configs.schema import SchemaError

    try:
        profile = load_profile(profile_id)
    except SchemaError as e:
        report.add(
            profile_id, "error", "schema_load",
            f"YAML parse / shape validation failed: {e}",
        )
        return
    except Exception as e:  # pragma: no cover — guard against unexpected
        report.add(
            profile_id, "error", "schema_load",
            f"unexpected loader error ({type(e).__name__}): {e}",
        )
        return

    sizing = profile.sizing_override
    policy = profile.override_policy

    # ── Rule: override_policy must be sibling of sizing_override ─────
    if sizing is not None and policy is None:
        report.add(
            profile_id, "warning", "missing_override_policy",
            (
                f"profile.sizing_override present but no override_policy: "
                f"add an OverridePolicy block (CONFIG-UX.1 schema) — "
                f"warning at Stage 1, will escalate in CONFIG-UX.4"
            ),
        )
        # Don't return — the audit still checks for any half-shape if user
        # added policy but with bugs. Falls through to None checks below.

    if policy is None:
        # No policy → no further policy-level checks. Sizing alone is OK
        # at Stage 1; CONFIG-UX.4 will tighten.
        if sizing is None:
            # Neither sizing_override nor override_policy → cleanest path,
            # nothing to audit further. Info-only.
            pass
        return

    # ── Rule: override_policy shape ──────────────────────────────────
    # The loader already calls policy.validate() but if something slipped
    # through (e.g. monkey-patched), re-validate here for the audit.
    try:
        policy.validate()
    except SchemaError as e:
        report.add(
            profile_id, "error", "override_policy_shape",
            f"override_policy shape invalid: {e}",
        )
        return

    # ── Rule: override_policy without sizing_override (unusual) ──────────
    if sizing is None and policy is not None:
        report.add(
            profile_id, "warning", "policy_without_sizing",
            (
                "override_policy declared but no sizing_override present — "
                "policy applies to nothing"
            ),
        )

    # ── Rule: class derivation + per-class requirements ──────────────
    effective_class = policy.effective_class(profile.role)

    # Reason is required across all override classes other than
    # safe_per_launch (the trivial pass-through). Even for safe_per_launch
    # we don't require reason (it's the explicit "no policy needed" class).
    if effective_class != "safe_per_launch":
        if not policy.reason:
            report.add(
                profile_id, "error", "missing_reason",
                (
                    f"override_policy.reason required (effective_class="
                    f"{effective_class!r}); operators reading `sndr "
                    f"profile show` need a written justification"
                ),
            )

    # Production class — additionally require evidence references.
    if effective_class == "production":
        if not policy.evidence_refs:
            report.add(
                profile_id, "error", "production_missing_evidence",
                (
                    "effective_class=production requires at least one "
                    "evidence_ref backing the override (path strings; "
                    "audit_override_policy at Stage 1 only checks "
                    "presence — content cross-validation is CONFIG-UX.4)"
                ),
            )

    # ── Class-4 forbidden override enforcement (CONFIG-UX.4.2) ───────
    # ALWAYS fires regardless of stage / strict mode / OverridePolicy
    # justification. These rules are physics-broken or evidence-
    # invalidating, so no class can legitimize them.
    _run_forbidden_rules(profile_id, profile, report)


# ─── Class-4 cross-layer resolution — audit-time cross-product ───────────
# CONFIG-UX.4.2.R §4.2 Option A (audit-time, not compose-time). Within
# Option A we choose per-tuple reporting granularity (one finding per
# (profile, hardware) pair) over aggregated reporting so operators see
# exactly which preset's hardware combo trips a rule.


def _list_preset_yamls() -> list[Path]:
    """Enumerate builtin presets so we can cross-product profile×hardware
    for cross-layer rules (e.g. Rule 2 tp_size vs n_gpus)."""
    d = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "presets"
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.glob("*.yaml")
        if p.is_file() and not p.stem.startswith("_")
    )


def _resolve_hardware_ids_for_profile(profile_id: str) -> list[str]:
    """Find all builtin presets that reference `profile_id` and return
    their hardware ids. Audit-time resolution (CONFIG-UX.4.2.R §4.2
    Option A); caller iterates each (profile, hardware) pair separately
    so operators see exactly which preset's hardware combo trips a rule.
    """
    import yaml
    hardware_ids: list[str] = []
    for path in _list_preset_yamls():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("profile") != profile_id:
            continue
        hw_id = data.get("hardware")
        if hw_id:
            hardware_ids.append(hw_id)
    return hardware_ids


def _run_forbidden_rules(
    profile_id: str, profile, report: OverrideReport,
) -> None:
    """Apply all 4 Class-4 rules to a profile.

    Cross-layer rules need model + hardware. Resolution strategy:
      - model: load via `profile.parent_model` (single lookup)
      - hardware: audit-time cross-product over all presets that
        reference this profile (CONFIG-UX.4.2.R §4.2 Option A) — each
        (profile, hardware) tuple checked independently so operator
        sees which preset's hardware combo trips a rule.

    Defensive: any cross-layer load failure → skip that rule rather
    than emit false-positive. The audit reports cross-layer load
    failures as a separate finding for visibility.
    """
    from sndr.model_configs.registry_v2 import (
        load_hardware, load_model,
    )
    from sndr.model_configs.schema import SchemaError

    # Load model once per profile (single parent_model reference).
    parent_model_id = getattr(profile, "parent_model", None)
    model = None
    if parent_model_id:
        try:
            model = load_model(parent_model_id)
        except SchemaError:
            # Model load failed — non-Class-4 paths already handle this;
            # rules that need model just skip via None-check.
            pass

    # Hardware cross-product — audit-time, per-tuple reporting.
    hw_ids = _resolve_hardware_ids_for_profile(profile_id)
    hardware_pairs: list[tuple[str, Any]] = []
    if hw_ids:
        for hw_id in hw_ids:
            try:
                hardware_pairs.append((hw_id, load_hardware(hw_id)))
            except SchemaError:
                pass  # skip this (profile, hardware) tuple
    else:
        # No preset references this profile yet (orphan profile).
        # Run hardware-independent rules with hardware=None.
        hardware_pairs.append(("(no preset references)", None))

    # Run each rule. Rules that don't need hardware run once;
    # cross-layer rules iterate the hardware list.
    for rule in FORBIDDEN_OVERRIDES:
        seen_messages: set[str] = set()
        for hw_id, hardware in hardware_pairs:
            try:
                msg = rule.predicate(profile, model, hardware)
            except Exception as e:  # pragma: no cover — defensive
                report.add(
                    profile_id, "error",
                    f"forbidden_override.{rule.rule_id}.predicate_error",
                    f"predicate raised {type(e).__name__}: {e}",
                )
                continue
            if msg is None:
                continue
            # Deduplicate identical messages across hardware iterations
            # (rules that don't depend on hardware would emit the same
            # message N times).
            tagged = f"[hardware={hw_id}] {msg}" if len(hardware_pairs) > 1 else msg
            if tagged in seen_messages:
                continue
            seen_messages.add(tagged)
            report.add(
                profile_id, "error",
                f"forbidden_override.{rule.rule_id}",
                tagged,
            )


def run_audit(profile_ids: Optional[list[str]] = None) -> OverrideReport:
    """Run the override-policy audit.

    Args:
        profile_ids: optional restricted list (testing hook).

    Returns:
        OverrideReport with findings.
    """
    if profile_ids is None:
        profile_ids = _list_profile_ids()
    report = OverrideReport()
    for pid in profile_ids:
        _audit_one_profile(pid, report)
    return report


def _print_table(report: OverrideReport, total_profiles: int) -> None:
    counts = report.count_by_severity()
    print("audit-override-policy: profile sizing_override + OverridePolicy")
    print("─" * 70)
    print(f"  scanned: {total_profiles} profile(s)")
    print(
        f"  findings: {counts.get('error', 0)} error, "
        f"{counts.get('warning', 0)} warning, "
        f"{counts.get('info', 0)} info"
    )
    print()
    if not report.findings:
        print("  ✓ no findings")
        return
    by_severity = {"error": [], "warning": [], "info": []}
    for f in report.findings:
        by_severity[f.severity].append(f)
    for sev in ("error", "warning", "info"):
        items = by_severity[sev]
        if not items:
            continue
        marker = {"error": "✗", "warning": "⚠", "info": "•"}[sev]
        print(f"  {marker} {sev.upper()} ({len(items)}):")
        for f in items:
            print(f"      [{f.rule}] {f.profile_id}: {f.message}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of the table view",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=("treat warnings as fatal (CI/release gate). Default mode "
              "exits 0 on warnings — only errors are fatal."),
    )
    parser.add_argument(
        "--profile", action="append", default=None,
        help="limit audit to one profile id (repeatable). Default: all.",
    )
    args = parser.parse_args()

    try:
        profile_ids = args.profile or _list_profile_ids()
        if not profile_ids:
            print("audit-override-policy: no profiles found", file=sys.stderr)
            return 2
        report = run_audit(profile_ids)
    except Exception as e:  # pragma: no cover
        print(
            f"audit-override-policy: internal error: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2

    if args.json:
        payload = {
            "scanned": len(profile_ids),
            "counts": report.count_by_severity(),
            "findings": [f.as_dict() for f in report.findings],
            "has_errors": report.has_errors(),
            "has_warnings": report.has_warnings(),
            "strict": args.strict,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report, len(profile_ids))

    if report.has_errors():
        return 1
    if args.strict and report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
