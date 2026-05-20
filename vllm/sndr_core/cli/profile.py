# SPDX-License-Identifier: Apache-2.0
"""V2 layered config — `sndr profile` subcommand (Phase 4, P1).

Subcommands surface the V2 ProfileDef layer:

  sndr profile list [--model <id>]
      List every ProfileDef under `builtin/profile/*.yaml`. With --model,
      filter to profiles whose `parent_model` matches.

  sndr profile show <id>
      Print the resolved ProfileDef: parent model, patches delta
      (enable/disable/override), sizing override, promotion contract.

  sndr profile diff <id>
      Show what would change vs the canonical parent ModelDef.patches —
      a preview of the patches matrix after compose(model, hw, profile).

Read-only. Does not run any patch or modify any file. Promotion CLI
(`sndr profile new/promote/validate`) ships in Phase 5 community SDK.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from . import _io


__all__ = [
    "add_argparser",
    "run_list",
    "run_show",
    "run_diff",
    "run_validate",
    "validate_profile",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "profile",
        help="V2 profile layer — list/show/diff/validate ProfileDef definitions.",
        description=(
            "Inspect V2 ProfileDef layer (model_configs/builtin/profile/*.yaml). "
            "Sister command of `sndr hardware` and `sndr model` (V2 layered config)."
        ),
    )
    sub = p.add_subparsers(dest="profile_cmd", required=True)

    p_list = sub.add_parser("list", help="List ProfileDef ids; optionally filter by parent model.")
    p_list.add_argument("--model", default=None,
                        help="Filter to profiles targeting this parent_model id.")
    p_list.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_list.set_defaults(func=run_list)

    p_show = sub.add_parser("show",
                            help="Print resolved ProfileDef (delta, sizing override, promotion).")
    p_show.add_argument("profile_id", help="profile id (e.g. 'wave9-balanced')")
    p_show.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_show.set_defaults(func=run_show)

    p_diff = sub.add_parser("diff",
                            help="Show patches matrix delta vs parent ModelDef.patches.")
    p_diff.add_argument("profile_id", help="profile id")
    p_diff.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    p_diff.set_defaults(func=run_diff)

    p_validate = sub.add_parser(
        "validate",
        help="Validate ProfileDef schema, parent linkage, artifact reference, "
             "and routing contract. Mirrors `sndr patches doctor` exit-code "
             "and JSON conventions.",
    )
    p_validate.add_argument(
        "profile_id", nargs="?", default=None,
        help="profile id to validate; omit to validate ALL builtin profiles.",
    )
    p_validate.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any ERROR is found (default: exit 0 unless tooling "
             "failure). WARNINGs never affect exit code.",
    )
    p_validate.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    p_validate.set_defaults(func=run_validate)


def _profile_summary(profile_id: str) -> dict:
    from vllm.sndr_core.model_configs.registry_v2 import load_profile
    p = load_profile(profile_id)
    delta = p.patches_delta
    sz = p.sizing_override
    return {
        "id": p.id,
        "parent_model": p.parent_model,
        "status": p.status,
        "created": p.created,
        "delta_enable_count": len(delta.enable),
        "delta_disable_count": len(delta.disable),
        "delta_override_count": len(delta.override),
        "has_sizing_override": sz is not None,
        "promote_to": p.promotion.promote_to if p.promotion else None,
    }


# ─── list

def run_list(args: argparse.Namespace) -> int:
    from vllm.sndr_core.model_configs.registry_v2 import list_profiles
    from vllm.sndr_core.model_configs.schema import SchemaError

    ids = list_profiles(parent_model=args.model)
    summaries: list[dict] = []
    errors: list[tuple[str, str]] = []
    for pid in ids:
        try:
            summaries.append(_profile_summary(pid))
        except (SchemaError, Exception) as e:
            errors.append((pid, f"{type(e).__name__}: {e}"))

    if args.json:
        out = {
            "filter_model": args.model,
            "profiles": summaries,
            "errors": errors,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 1 if errors else 0

    title = "sndr profile list — V2 ProfileDef registry"
    if args.model:
        title += f"  (filter: parent_model={args.model})"
    print(title)
    print("─" * 60)
    if not summaries and not errors:
        msg = "  (no V2 profile files found"
        if args.model:
            msg += f" with parent_model={args.model!r}"
        msg += ")"
        print(msg)
        return 0
    for s in summaries:
        sz_marker = " sizing-override" if s["has_sizing_override"] else ""
        print(f"  {s['id']}")
        print(f"      parent: {s['parent_model']}  status: {s['status']}  "
              f"delta: +{s['delta_enable_count']}/-{s['delta_disable_count']}"
              f"/~{s['delta_override_count']}{sz_marker}")
    if errors:
        print()
        print("  Errors loading these IDs:")
        for pid, msg in errors:
            print(f"    {pid}: {msg}")
    print()
    print(f"  Total: {len(summaries)} profiles"
          + (f" ({len(errors)} errors)" if errors else ""))
    return 1 if errors else 0


# ─── show

def run_show(args: argparse.Namespace) -> int:
    from vllm.sndr_core.model_configs.registry_v2 import load_profile
    from vllm.sndr_core.model_configs.schema import SchemaError

    try:
        p = load_profile(args.profile_id)
    except SchemaError as e:
        _io.warn(f"profile id {args.profile_id!r}: {e}")
        return 2

    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(p), indent=2, sort_keys=True, default=str))
        return 0

    print(f"sndr profile show '{p.id}'")
    print("─" * 60)
    print(f"  parent_model:  {p.parent_model}")
    print(f"  maintainer:    {p.maintainer}")
    print(f"  status:        {p.status}")
    print(f"  created:       {p.created}")
    print()
    d = p.patches_delta
    print("  Patches delta:")
    if d.enable:
        print(f"    enable ({len(d.enable)}):")
        for k, v in sorted(d.enable.items()):
            print(f"      + {k} = {v!r}")
    if d.disable:
        print(f"    disable ({len(d.disable)}):")
        for k in sorted(d.disable):
            print(f"      - {k}")
    if d.override:
        print(f"    override ({len(d.override)}):")
        for k, v in sorted(d.override.items()):
            print(f"      ~ {k} = {v!r}")
    if not (d.enable or d.disable or d.override):
        print("    (empty — uses parent model.patches as-is)")
    print()
    sz = p.sizing_override
    if sz is not None:
        print("  Sizing override (operator tuning for (model × hardware) pair):")
        print(f"    max_model_len:            {sz.max_model_len}")
        print(f"    gpu_memory_utilization:   {sz.gpu_memory_utilization}")
        print(f"    max_num_seqs:             {sz.max_num_seqs}")
        print(f"    max_num_batched_tokens:   {sz.max_num_batched_tokens}")
        print(f"    enable_chunked_prefill:   {sz.enable_chunked_prefill}")
        print(f"    enforce_eager:            {sz.enforce_eager}")
        print(f"    disable_custom_all_reduce:{sz.disable_custom_all_reduce}")
    else:
        print("  Sizing override: none (uses hardware.sizing defaults)")
    print()
    promo = p.promotion
    if promo is not None:
        print("  Promotion:")
        print(f"    promote_to: {promo.promote_to}")
        if promo.validation_required:
            print(f"    validation_required ({len(promo.validation_required)}):")
            for v in promo.validation_required:
                print(f"      • {v}")
    return 0


# ─── diff

def run_diff(args: argparse.Namespace) -> int:
    """Show what the patches matrix looks like AFTER apply_patches_delta
    is run on the parent model's canonical patches. This is the
    same delta the composer applies in compose()."""
    from vllm.sndr_core.model_configs.compose import apply_patches_delta
    from vllm.sndr_core.model_configs.registry_v2 import load_model, load_profile
    from vllm.sndr_core.model_configs.schema import SchemaError

    try:
        p = load_profile(args.profile_id)
        m = load_model(p.parent_model)
    except SchemaError as e:
        _io.warn(f"profile {args.profile_id!r} diff failed: {e}")
        return 2

    canonical = dict(m.patches)
    merged = apply_patches_delta(canonical, p.patches_delta)

    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []
    changed: list[tuple[str, str, str]] = []

    canonical_keys = set(canonical.keys())
    merged_keys = set(merged.keys())
    for k in sorted(merged_keys - canonical_keys):
        added.append((k, merged[k]))
    for k in sorted(canonical_keys - merged_keys):
        removed.append((k, canonical[k]))
    for k in sorted(canonical_keys & merged_keys):
        if canonical[k] != merged[k]:
            changed.append((k, canonical[k], merged[k]))

    if args.json:
        out = {
            "profile_id": p.id,
            "parent_model": p.parent_model,
            "canonical_count": len(canonical),
            "merged_count": len(merged),
            "added": [{"key": k, "value": v} for k, v in added],
            "removed": [{"key": k, "value": v} for k, v in removed],
            "changed": [
                {"key": k, "canonical": cv, "merged": mv}
                for k, cv, mv in changed
            ],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"sndr profile diff '{p.id}' vs '{p.parent_model}'")
    print("─" * 60)
    print(f"  canonical patches: {len(canonical)}")
    print(f"  merged patches:    {len(merged)}")
    print(f"  delta: +{len(added)} / -{len(removed)} / ~{len(changed)}")
    print()
    if added:
        print("  Added (profile enable on top of canonical):")
        for k, v in added:
            print(f"    + {k} = {v!r}")
    if removed:
        print("  Removed (profile disable):")
        for k, v in removed:
            print(f"    - {k}  (canonical was {v!r})")
    if changed:
        print("  Changed (profile override):")
        for k, cv, mv in changed:
            print(f"    ~ {k}: {cv!r} → {mv!r}")
    if not (added or removed or changed):
        print("  (no delta — profile matches canonical model.patches)")
    return 0


# ─── validate ───────────────────────────────────────────────────────────


# Severity levels for validation issues.
_SEV_ERROR = "ERROR"
_SEV_WARNING = "WARNING"
_SEV_INFO = "INFO"


def _artifacts_dir():
    """Path to the spec_decode functional artifacts directory."""
    import pathlib
    here = pathlib.Path(__file__).resolve()
    # cli/ → sndr_core/ → integrations/spec_decode/artifacts/
    return here.parent.parent / "integrations" / "spec_decode" / "artifacts"


def _read_artifact(artifact_id: str):
    """Return parsed artifact dict or None if file missing / malformed.

    Returns:
        (data, error_msg). data is None on failure; error_msg is a short
        human-readable description of the failure mode.
    """
    import pathlib
    path = _artifacts_dir() / f"{artifact_id}.json"
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"json parse error: {e}"
    except OSError as e:
        return None, f"read error: {e}"


def validate_profile(profile_id: str) -> tuple[list[dict], str]:
    """Run the 11 P1.4 checks against a single profile.

    Returns:
        (issues, status). ``issues`` is a list of
        ``{check, severity, message}`` dicts. ``status`` is one of:
          * ``ok``      — no errors, no warnings
          * ``warn``    — warnings only
          * ``failed``  — at least one error
          * ``unloadable`` — profile YAML could not be loaded at all
                           (special case; nothing else was checked)
    """
    from vllm.sndr_core.model_configs.compose import (
        _check_compression_kv_dtype_compat,
    )
    from vllm.sndr_core.model_configs.registry_v2 import (
        load_model, load_profile,
    )
    from vllm.sndr_core.model_configs.schema import SchemaError
    from vllm.sndr_core.model_configs.schema_v2 import PROFILE_ROLES

    issues: list[dict] = []

    def emit(check: str, severity: str, message: str) -> None:
        issues.append({
            "check": check,
            "severity": severity,
            "message": message,
        })

    # Check 1: load + schema validate
    try:
        profile = load_profile(profile_id)
        profile.validate()
    except (SchemaError, FileNotFoundError) as e:
        emit("01_schema_load", _SEV_ERROR,
             f"profile failed to load/validate: {e}")
        return issues, "unloadable"

    # Check 2: parent_model exists + loads
    try:
        model = load_model(profile.parent_model)
        model.validate()
    except (SchemaError, FileNotFoundError) as e:
        emit("02_parent_model", _SEV_ERROR,
             f"parent_model={profile.parent_model!r} does not load: {e}")
        # Without a parent we cannot run the compatibility / role-vs-model checks.
        return issues, "failed"

    # Check 3: role valid (enum was enforced by ProfileDef.validate, but
    # re-assert as a separate check_id so JSON consumers can see it).
    if profile.role is not None and profile.role not in PROFILE_ROLES:
        emit("03_role_enum", _SEV_ERROR,
             f"role={profile.role!r} not in {PROFILE_ROLES}")

    # Check 4: spec_decode_override valid (re-assert; schema already validated).
    if profile.spec_decode_override is not None:
        try:
            profile.spec_decode_override.validate()
        except SchemaError as e:
            emit("04_spec_decode_override", _SEV_ERROR,
                 f"spec_decode_override invalid: {e}")

    # Check 5: compression_plan compatible with parent (Δ vs P1.2b semantics).
    try:
        _check_compression_kv_dtype_compat(model, profile)
    except SchemaError as e:
        emit("05_compression_dtype", _SEV_ERROR, str(e))

    # Checks 6 + 7 + 8 + 9: validation artifact + workload intersection.
    artifact_data = None
    if profile.validation is None:
        if profile.role == "structured":
            emit("06_artifact_present", _SEV_WARNING,
                 "structured profile without validation block — runtime "
                 "router has no artifact_id to look up; tool_json requests "
                 "will fall back to default")
        # role=default / None / gateway with no validation is normal.
    else:
        # Check 6: artifact JSON file exists + parses.
        artifact_data, err = _read_artifact(profile.validation.artifact_id)
        if artifact_data is None:
            emit("06_artifact_present", _SEV_ERROR,
                 f"validation.artifact_id={profile.validation.artifact_id!r}: "
                 f"{err}")
        else:
            # Check 7: config_hash matches.
            actual_hash = artifact_data.get("config_hash")
            if actual_hash != profile.validation.config_hash:
                emit("07_config_hash", _SEV_ERROR,
                     f"validation.config_hash={profile.validation.config_hash!r} "
                     f"does not match artifact.config_hash={actual_hash!r}")

            # Check 8: effective_workloads = intersection.
            allowed = set(artifact_data.get("allowed_workloads") or [])
            intended = (
                set(profile.routing.intended_workloads)
                if profile.routing is not None
                else set()
            )
            effective = intended & allowed
            denied = intended - allowed
            if denied:
                emit("08_intended_workloads", _SEV_WARNING,
                     f"intended_workloads {sorted(denied)} not present in "
                     f"artifact.allowed_workloads {sorted(allowed)}; "
                     f"router will deny these classes")

            # Check 9: structured profile must have non-empty effective_workloads.
            if profile.role == "structured" and not effective:
                emit("09_structured_effective_nonempty", _SEV_ERROR,
                     f"role=structured but effective_workloads is empty "
                     f"(intended={sorted(intended)} ∩ allowed={sorted(allowed)} "
                     f"= {{}}). Structured runtime would receive no traffic.")

            # Check 11: artifact decision ≠ denied / KERNEL_STORAGE_DTYPE_MISMATCH.
            decision = artifact_data.get("decision", "")
            if decision == "denied":
                emit("11_artifact_verdict", _SEV_ERROR,
                     f"validation artifact decision={decision!r} — "
                     f"profile is referencing a denied artifact")
            elif "MISMATCH" in str(decision).upper() or \
                    "UNSUPPORTED" in str(decision).upper():
                emit("11_artifact_verdict", _SEV_ERROR,
                     f"validation artifact decision={decision!r} signals "
                     f"a non-overridable contract failure")

    # Check 10: role=default must NOT carry spec_decode/compression/routing/validation.
    if profile.role == "default":
        for field_name in ("spec_decode_override", "compression_plan",
                           "backend_plan", "routing", "validation"):
            val = getattr(profile, field_name)
            if val is not None:
                emit("10_default_clean", _SEV_ERROR,
                     f"role=default but {field_name} is set; default-role "
                     f"profiles must leave all runtime-role blocks unset "
                     f"to preserve broad workload safety")

    # Roll up status.
    if any(i["severity"] == _SEV_ERROR for i in issues):
        return issues, "failed"
    if any(i["severity"] == _SEV_WARNING for i in issues):
        return issues, "warn"
    return issues, "ok"


def run_validate(args: argparse.Namespace) -> int:
    """Validate one profile (when ``profile_id`` provided) or every
    builtin profile (when omitted).

    Exit codes:
      * 0 — all profiles validate OK (or only WARNINGs and --strict not set)
      * 1 — at least one profile has ERRORs and --strict was provided
      * 2 — tooling failure (registry unloadable, etc.)
    """
    from vllm.sndr_core.model_configs.registry_v2 import list_profiles

    try:
        if args.profile_id is not None:
            targets = [args.profile_id]
        else:
            targets = list_profiles()
    except Exception as e:  # noqa: BLE001
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            _io.error(f"could not list profiles: {e}")
        return 2

    per_profile: list[dict] = []
    for pid in targets:
        issues, status = validate_profile(pid)
        per_profile.append({
            "profile_id": pid,
            "status": status,
            "issues": issues,
        })

    total_errors = sum(
        1 for entry in per_profile
        for i in entry["issues"] if i["severity"] == _SEV_ERROR
    )
    total_warnings = sum(
        1 for entry in per_profile
        for i in entry["issues"] if i["severity"] == _SEV_WARNING
    )
    ok_count = sum(1 for e in per_profile if e["status"] == "ok")

    if args.json:
        print(json.dumps({
            "profiles_checked": len(per_profile),
            "ok": ok_count,
            "errors": total_errors,
            "warnings": total_warnings,
            "results": per_profile,
        }, indent=2, sort_keys=True))
    else:
        _io.banner(
            "sndr profile validate",
            f"{len(per_profile)} profile(s) checked",
        )
        for entry in per_profile:
            pid = entry["profile_id"]
            issues = entry["issues"]
            status = entry["status"]
            if not issues:
                _io.success(f"{pid}")
                continue
            # Status header
            label = {
                "ok": _io.success,
                "warn": _io.warn,
                "failed": _io.error,
                "unloadable": _io.error,
            }[status]
            label(f"{pid}  [{status}]")
            for i in issues:
                sev = i["severity"]
                msg = f"  [{i['check']}] {i['message']}"
                if sev == _SEV_ERROR:
                    _io.error(msg)
                elif sev == _SEV_WARNING:
                    _io.warn(msg)
                else:
                    _io.info(msg)
        print()
        _io.info(
            f"summary: ok={ok_count}  errors={total_errors}  "
            f"warnings={total_warnings}  total={len(per_profile)}"
        )

    if args.strict and total_errors > 0:
        return 1
    return 0
