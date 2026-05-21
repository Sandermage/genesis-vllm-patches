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
    "run_render_launchers",
    "render_profile_launcher",
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

    p_render = sub.add_parser(
        "render-launchers",
        help="Render a bash launcher derived from a V2 ProfileDef.",
        description=(
            "Generate a bash launcher script from the composed V1 "
            "ModelConfig (model + hardware + profile). Output goes to "
            "stdout by default (--dry-run). Use --output DIR to write a "
            "file; --force is required to overwrite existing files."
        ),
    )
    p_render.add_argument("profile_id", help="profile id to render")
    p_render.add_argument(
        "--hardware", default=None,
        help="HardwareDef id; if omitted, auto-pick the first known "
             "hardware satisfying the parent model's `requires`.",
    )
    p_render.add_argument(
        "--dry-run", action="store_true",
        help="Print the rendered script to stdout. This is the default "
             "when --output is not given.",
    )
    p_render.add_argument(
        "--output", default=None,
        help="Write to <output_dir>/start_<profile_id>.sh. Implies "
             "non-dry-run unless --dry-run is also passed.",
    )
    p_render.add_argument(
        "--force", action="store_true",
        help="Allow overwriting an existing output file.",
    )
    p_render.set_defaults(func=run_render_launchers)


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


# ─── render-launchers ───────────────────────────────────────────────────


# P1.8 (2026-05-21): the source-of-truth backend mapping moved to
# compose.py:BACKEND_PLAN_EMISSION_MAP. Same data, but now also used
# for compose-time env emission via render_backend_env(), not just
# render-launchers consistency. This module imports it for the
# consistency check so both layers reference the same dict (no parallel
# maps that can drift). Re-exported here as _BACKEND_PLAN_MAP for
# back-compat with the existing P1.5 test surface.
from vllm.sndr_core.model_configs.compose import (
    BACKEND_PLAN_EMISSION_MAP as _BACKEND_PLAN_MAP,
)


def _validate_backend_plan_consistency(profile, genesis_env: dict) -> None:
    """Verify that profile.backend_plan declarations match the env in
    the composed genesis_env. Raises SchemaError on:
      * unknown (field, value) pair not in the mapping table
      * mapped env var that is not its expected value in genesis_env
        (i.e. profile declared a backend but the corresponding env
        is missing or has the wrong value — silent mismatch)

    Backend values mapped to None are CLI-arg or config-time concerns
    and are not env-checked here. Multi-env mappings (e.g. drafter_kv_sharing
    emits BOTH SNDR canonical AND GENESIS legacy alias) check every entry.
    """
    from vllm.sndr_core.model_configs.schema import SchemaError

    bp = profile.backend_plan
    if bp is None:
        return
    for field_name in (
        "target_default", "target_native_layers",
        "drafter_sliding", "drafter_full",
        "drafter_kv_sharing",
    ):
        value = getattr(bp, field_name)
        if value is None:
            continue
        key = (field_name, value)
        if key not in _BACKEND_PLAN_MAP:
            raise SchemaError(
                f"profile.backend_plan.{field_name}={value!r}: not in the "
                f"supported backend mapping table. Adding a new value "
                f"requires extending BACKEND_PLAN_EMISSION_MAP in "
                f"compose.py with the env mapping AND a test."
            )
        expected_envs = _BACKEND_PLAN_MAP[key]
        if expected_envs is None:
            continue
        for env_name, expected_value in expected_envs.items():
            observed = genesis_env.get(env_name)
            if observed != expected_value:
                raise SchemaError(
                    f"profile.backend_plan.{field_name}={value!r} requires "
                    f"{env_name}={expected_value} in composed genesis_env, "
                    f"but observed {observed!r}. Either compose did not "
                    f"emit it (P1.8 regression) or operator override in "
                    f"patches_delta blocked it."
                )


# Subset of canonical envs the byte-equivalence gate cares about. The
# render path does NOT auto-add these; they must arrive via compose
# (patches_delta.enable or model.patches or compression_plan emission
# or backend_plan emission). This list is the set the render-launchers
# smoke gate scans for when the profile is structured-role.
#
# P1.8 added the two G4_76 disable envs (value "0") via the
# drafter_kv_sharing: physical declaration. Without them the Gemma4
# mapping provider's artifact_lookup_keys() returns None and the
# safety guard denies MTP at boot.
#
# P1.9 (2026-05-21 control-A) adds the output-correctness envs that
# the hand-written launcher used but the first V2-rendered launcher
# missed. Without P65 + G4_68 + G4_70C the engine can boot and pass
# the guard while producing corrupt unicode due to wrong TQ+MTP
# cudagraph / split-allocator state.
_STRUCTURED_REQUIRED_ENVS = (
    "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
    "GENESIS_ENABLE_G4_68_TQ_SPEC_CG_DOWNGRADE_OVERLAY",
    "GENESIS_ENABLE_G4_70_PN259C_ROUTE_B",
    "GENESIS_ENABLE_G4_71B_DRAFTER_SLIDING_TRITON",
    "GENESIS_ENABLE_G4_75_DRAFTER_HEAD512_TRITON",
    "GENESIS_ENABLE_G4_71_DRAFTER_NATIVE_BACKEND",  # P1.9 (= "0")
    "GENESIS_ENABLE_G4_72_DRAFTER_NATIVE_SPEC",     # P1.9 (= "0")
    "GENESIS_ENABLE_G4_73_DRAFTER_PROFILE_SKIP",    # P1.9 (= "0")
    "GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT",      # P1.9 (= "0")
    "GENESIS_ENABLE_G4_78_DRAFTER_TARGET_KV_BRIDGE",  # P1.9 (= "0")
    "SNDR_G4_TQ_FORCE_SKIP_LAYERS",
    "GENESIS_G4_TQ_FORCE_SKIP_LAYERS",
    "SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER",
    "SNDR_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING",     # P1.8 (= "0")
    "GENESIS_ENABLE_G4_76_DISABLE_DRAFTER_KV_SHARING",  # P1.8 legacy alias (= "0")
)


# Observability envs that must NOT be in the rendered launcher by
# default (operator-decision per the opt-in plan). If any of these is
# present in cfg.genesis_env at render time, it means an operator
# explicitly added it via patches_delta — that's allowed and we keep
# it. But the renderer does not auto-emit them.
_OBSERVABILITY_OPTIN_ENVS = (
    "SNDR_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC",
    "GENESIS_ENABLE_SPEC_DECODE_ACCEPTANCE_METRIC",
    "PROMETHEUS_MULTIPROC_DIR",
    "SNDR_PROMETHEUS_MULTIPROC_CLEAN",
    "SNDR_SPEC_DECODE_PROFILE_LABEL",
)


def _format_env_flags(env: dict[str, str]) -> str:
    """Render a dict of env vars as docker `-e KEY=value \\` lines,
    sorted by key for deterministic output."""
    lines = []
    for k in sorted(env):
        v = env[k]
        lines.append(f"  -e {k}={v} \\")
    return "\n".join(lines)


def _pick_default_hardware(model):
    """Auto-pick the first hardware that satisfies model.requires.
    Returns the HardwareDef or raises SchemaError if no hardware fits.
    """
    from vllm.sndr_core.model_configs.registry_v2 import (
        list_hardware, load_hardware,
    )
    from vllm.sndr_core.model_configs.schema import SchemaError

    req = model.requires
    candidates = []
    for hw_id in list_hardware():
        hw = load_hardware(hw_id)
        total_vram = hw.hardware.min_vram_per_gpu_mib * hw.hardware.n_gpus
        if total_vram < req.min_total_vram_mib:
            continue
        if hw.hardware.n_gpus < req.min_gpu_count:
            continue
        candidates.append(hw)
    if not candidates:
        raise SchemaError(
            f"no HardwareDef satisfies model {model.id!r}'s requires "
            f"(min_total_vram_mib={req.min_total_vram_mib}, "
            f"min_gpu_count={req.min_gpu_count}); pass --hardware <id> "
            f"explicitly."
        )
    return candidates[0]


def render_profile_launcher(
    profile_id: str, hardware_id: str | None = None,
) -> str:
    """Compose (model + hw + profile) and render a bash launcher script.

    Args:
        profile_id: V2 ProfileDef id (e.g. ``gemma4-tq-mtp-structured-k4``).
        hardware_id: optional HardwareDef id; if None, auto-picks the
            first hardware satisfying the parent model's requires.

    Returns:
        Bash script source as a single string.

    Raises:
        SchemaError on invalid backend_plan / unloadable profile / no
        compatible hardware / etc.
    """
    from datetime import datetime, timezone

    from vllm.sndr_core.model_configs.compose import compose
    from vllm.sndr_core.model_configs.registry_v2 import (
        load_hardware, load_model, load_profile,
    )

    profile = load_profile(profile_id)
    profile.validate()
    model = load_model(profile.parent_model)
    model.validate()
    if hardware_id is None:
        hw = _pick_default_hardware(model)
    else:
        hw = load_hardware(hardware_id)
        hw.validate()

    cfg = compose(model, hw, profile)

    # Strict backend_plan consistency check (raises SchemaError on
    # unknown values or missing envs).
    _validate_backend_plan_consistency(profile, cfg.genesis_env)

    role = profile.role or "tuning"
    has_spec_decode = cfg.spec_decode is not None
    has_compression = profile.compression_plan is not None and (
        profile.compression_plan.native_source_layers
    )
    has_overlay = any(
        k.startswith("GENESIS_ENABLE_G4_60") and v == "1"
        for k, v in cfg.genesis_env.items()
    )

    # Build the speculative-config CLI arg if profile sets spec_decode.
    spec_decode_arg = ""
    spec_decode_k_default = ""
    if has_spec_decode:
        spec_json = cfg.spec_decode.to_vllm_arg()
        spec_decode_arg = f"  --speculative-config '{spec_json}' \\\n"
        spec_decode_k_default = str(cfg.spec_decode.num_speculative_tokens)

    # Attention backend (vLLM CLI flag). Defaults to None (engine auto)
    # unless backend_plan.target_default is set.
    attn_backend_arg = ""
    if profile.backend_plan is not None and profile.backend_plan.target_default:
        attn_backend_arg = (
            f"  --attention-backend {profile.backend_plan.target_default} \\\n"
        )

    # PR42637 overlay mounts (8 files), only when any G4_60* env is set.
    overlay_mounts = ""
    if has_overlay:
        overlay_mounts = """\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/turboquant_attn.py:${TGT}/v1/attention/backends/turboquant_attn.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/triton_turboquant_decode.py:${TGT}/v1/attention/ops/triton_turboquant_decode.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/triton_turboquant_store.py:${TGT}/v1/attention/ops/triton_turboquant_store.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/turboquant_config.py:${TGT}/model_executor/layers/quantization/turboquant/config.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/kv_cache_interface.py:${TGT}/v1/kv_cache_interface.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/kv_cache_utils.py:${TGT}/v1/core/kv_cache_utils.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/single_type_kv_cache_manager.py:${TGT}/v1/core/single_type_kv_cache_manager.py:ro \\
  -v ${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637/block_pool.py:${TGT}/v1/core/block_pool.py:ro \\
"""

    # Validation receipt comment block.
    validation_line = "validation: none"
    if profile.validation is not None:
        validation_line = (
            f"validation: artifact_id={profile.validation.artifact_id} "
            f"config_hash={profile.validation.config_hash}"
        )

    # Header comment block (timestamped, traceable).
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"#!/bin/bash\n"
        f"# Generated by `sndr profile render-launchers {profile_id}` at {now}.\n"
        f"# DO NOT edit manually — re-render from the V2 profile YAML.\n"
        f"#\n"
        f"# Source profile:    {profile_id}\n"
        f"# Composed key:      {cfg.key}\n"
        f"# Role:              {role}\n"
        f"# Parent model:      {profile.parent_model}\n"
        f"# Hardware:          {hw.id}\n"
        f"# {validation_line}\n"
        f"# spec_decode:       {'MTP K=' + spec_decode_k_default if has_spec_decode else 'OFF'}\n"
        f"# compression_plan:  {'native_source_layers=' + str(profile.compression_plan.native_source_layers) if has_compression else 'none'}\n"
        f"# PR42637 overlay:   {'mounted' if has_overlay else 'not needed'}\n"
        f"#\n"
        f"# Observability envs (PN282/PN283) are NOT in this launcher.\n"
        f"# They are operator opt-in per\n"
        f"# docs/_internal/PN282_PN283_PRODUCTION_LAUNCHER_OPT_IN_PLAN_2026-05-20.md\n"
    )

    # Genesis env flags (sorted for determinism).
    genesis_env_lines = _format_env_flags(cfg.genesis_env)
    # System env flags (e.g. PYTORCH_*, VLLM_*).
    system_env_lines = _format_env_flags(cfg.system_env)

    # Per-role port. Single-role-per-launcher → fixed port matches the
    # gateway README quick-start convention. Override at deploy time
    # by editing the rendered file (this is a starting template).
    port = 8101 if role == "default" else 8102

    container_name = f"vllm-{profile_id}-k${{K}}" if has_spec_decode else \
                     f"vllm-{profile_id}"

    # Inner run.sh — what the docker entrypoint executes.
    inner_run = []
    inner_run.append('#!/bin/bash')
    inner_run.append('set -e')
    inner_run.append('echo "=== Install vllm-sndr-core ==="')
    inner_run.append('pip install -e ${GENESIS_REPO} --no-deps --quiet 2>&1 | tail -2')
    if has_spec_decode and "GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE" in cfg.genesis_env \
            and cfg.genesis_env.get("GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE") == "1":
        inner_run.append('echo "=== Clear PN248 trace ==="')
        inner_run.append('rm -f /tmp/genesis_pn248_acceptance_trace.log')
    inner_run.append('')
    inner_run.append(f'echo "=== Launch {profile_id} role={role}" K=${{K:-(none)}}')
    inner_run.append(f'exec vllm serve {cfg.model_path} \\')
    inner_run.append(f'  --served-model-name {cfg.served_model_name} \\')
    inner_run.append(f'  --tensor-parallel-size {hw.hardware.n_gpus} \\')
    inner_run.append('  --disable-custom-all-reduce \\')
    inner_run.append(f'  --dtype {cfg.dtype} \\')
    inner_run.append(f'  --kv-cache-dtype {cfg.kv_cache_dtype} \\')
    if attn_backend_arg:
        inner_run.append(attn_backend_arg.rstrip(' \\\n') + ' \\')
    inner_run.append(f'  --max-model-len {cfg.max_model_len} \\')
    inner_run.append(f'  --max-num-seqs {cfg.max_num_seqs} \\')
    inner_run.append(f'  --max-num-batched-tokens {cfg.max_num_batched_tokens} \\')
    if cfg.enable_chunked_prefill:
        inner_run.append('  --enable-chunked-prefill \\')
    if cfg.trust_remote_code:
        inner_run.append('  --trust-remote-code \\')
    if spec_decode_arg:
        inner_run.append(spec_decode_arg.rstrip(' \\\n') + ' \\')
    inner_run.append(f'  --gpu-memory-utilization {cfg.gpu_memory_utilization} \\')
    inner_run.append(f'  --api-key {cfg.api_key} \\')
    inner_run.append(f'  --host {cfg.host} --port {port} \\')
    inner_run.append('  --disable-log-stats')

    inner = "\n".join(inner_run)

    # Outer script.
    k_default = spec_decode_k_default or "0"
    script = f"""{header}
set -e
K="${{1:-{k_default}}}"
CONTAINER="{container_name}"
PORT={port}

docker rm -f "$CONTAINER" 2>/dev/null || true

GENESIS_REPO=/home/sander/genesis-vllm-patches
TGT=/usr/local/lib/python3.12/dist-packages/vllm
IMAGE="vllm/vllm-openai:nightly"

LAUNCHER_DIR=/tmp/{profile_id}_launcher
mkdir -p "$LAUNCHER_DIR"
cat > "$LAUNCHER_DIR/run.sh" <<INNER_EOF
{inner}
INNER_EOF
chmod +x "$LAUNCHER_DIR/run.sh"

docker run -d --name "$CONTAINER" \\
  --gpus all --ipc=host -p ${{PORT}}:${{PORT}} \\
  --entrypoint "$LAUNCHER_DIR/run.sh" \\
{system_env_lines}
{genesis_env_lines}
  -v "$LAUNCHER_DIR":"$LAUNCHER_DIR":ro \\
  -v ${{GENESIS_REPO}}:${{GENESIS_REPO}}:rw \\
  -v /nfs/genesis/models:/models:ro \\
  -v ${{GENESIS_REPO}}/vllm/sndr_core:${{TGT}}/sndr_core:ro \\
{overlay_mounts}  ${{IMAGE}}

echo "$CONTAINER on port $PORT ({profile_id}, role={role})"
"""
    return script


def run_render_launchers(args: argparse.Namespace) -> int:
    """Handler for `sndr profile render-launchers`.

    Exit codes:
      0  success (rendered to stdout or wrote file)
      1  output target already exists and --force not provided
      2  schema / backend_plan inconsistency / tooling failure
    """
    import pathlib

    from vllm.sndr_core.model_configs.schema import SchemaError

    try:
        script = render_profile_launcher(args.profile_id, args.hardware)
    except SchemaError as e:
        if args.json if hasattr(args, "json") else False:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            _io.error(f"render failed: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        _io.error(f"render tooling failure: {type(e).__name__}: {e}")
        return 2

    # Decide between stdout and file output.
    # Default: --dry-run is implicit when --output is not given.
    if args.output is None or args.dry_run:
        print(script)
        return 0

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"start_{args.profile_id}.sh"
    if out_path.exists() and not args.force:
        _io.error(
            f"{out_path} already exists; pass --force to overwrite."
        )
        return 1
    out_path.write_text(script)
    out_path.chmod(0o755)
    _io.success(f"wrote {out_path}")
    return 0
