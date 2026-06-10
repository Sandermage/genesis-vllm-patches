#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.5.1 — generate `build/config_catalog/config_catalog.json`.

Builds a deterministic derived catalog from the V2 YAML tree + public
baseline JSONs. Operator-locked architectural principle: **catalog is
a derived API, not a new source of truth**. This script reads source
YAMLs (typed via `registry_v2` loaders) and emits one row per
operator-visible artifact (preset / profile / model / hardware /
baseline). No SQLite in .5.1; deferred. No CLI in .5.1; deferred.

Redaction discipline (operator-locked):
  - `sndr_private/...` paths replaced with `{redacted: true, ...}` markers
  - Local absolute paths (`/Users/...`, `/home/...`) replaced same way
  - Private evidence_refs raw values never serialised — `redacted: true`
    marker emitted in their place

Usage:
  python3 scripts/generate_config_catalog.py            # write to build/...
  python3 scripts/generate_config_catalog.py --stdout   # emit to stdout
  python3 scripts/generate_config_catalog.py --check    # exit 1 if drift
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_PATH = REPO_ROOT / "build" / "config_catalog" / "config_catalog.json"


# ─── Helpers ───────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _last_git_commit_for(path: Path) -> Optional[str]:
    """Return short SHA of the last commit touching `path`. Defensive:
    returns None if git is unavailable or path is untracked."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(path)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _redact_evidence_ref(ref_dict: dict) -> dict:
    """Replace a private/local evidence ref with a redacted marker.

    Public-safe forms (verbatim): repo-relative paths under
    `tests/`, `docs/`, `vllm/`, `evidence/`, ...; `external://...`
    schemes.
    """
    from sndr.model_configs.catalog_schema import (
        is_private_visibility, is_redactable_path,
    )
    visibility = ref_dict.get("visibility")
    path = ref_dict.get("path", "")
    if is_private_visibility(visibility) or is_redactable_path(path):
        return {
            "type": ref_dict.get("type", "unknown"),
            "redacted": True,
            "visibility": "private",
            "note": "private evidence — not exposed in generated catalog",
        }
    # Public — return verbatim (only the fields the catalog cares about)
    out = {"type": ref_dict.get("type", "unknown"), "path": path}
    if visibility:
        out["visibility"] = visibility
    if ref_dict.get("note"):
        out["note"] = ref_dict["note"]
    return out


# ─── Per-row-type builders ─────────────────────────────────────────────────


def _build_preset_row(alias: str, generated_at: str) -> dict:
    """Build PresetRow dict from preset YAML + composed runtime."""
    from sndr.model_configs.registry_v2 import (
        _alias_dir, load_alias, load_preset_def,
    )
    from sndr.model_configs.schema import dump_yaml

    yaml_path = _alias_dir() / f"{alias}.yaml"
    pd = load_preset_def(alias)
    # Compose to get composed_sha256
    cfg = load_alias(alias)
    composed_yaml = dump_yaml(cfg)

    card = pd.card
    # Build evidence_refs with redaction
    ev_refs: list[dict] = []
    if card:
        for ref in card.evidence_refs:
            ref_dict = dataclasses.asdict(ref)
            ev_refs.append(_redact_evidence_ref(ref_dict))

    return {
        "schema_version": 1,
        "row_type": "preset",
        "id": alias,
        "source_path": str(yaml_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256_file(yaml_path),
        "status": card.status if card else None,
        "family": card.routing_family if card else None,
        "tags": [card.audience] if (card and card.audience) else [],
        "updated_from_git_commit": _last_git_commit_for(yaml_path),
        "generated_at": generated_at,
        # Type-specific
        "model_id": pd.model,
        "hardware_id": pd.hardware,
        "profile_id": pd.profile,
        "composed_key": cfg.key,
        "composed_sha256": _sha256_string(composed_yaml),
        "has_card": pd.has_card(),
        "card_title": card.title if card else None,
        "card_status": card.status if card else None,
        "card_audience": card.audience if card else None,
        "card_mode": card.mode if card else None,
        "card_workload_allow": list(card.workload_allow) if card else [],
        "card_workload_deny": list(card.workload_deny) if card else [],
        "card_K": card.K if card else None,
        "card_routing_family": card.routing_family if card else None,
        "card_default_for_family": bool(card.default_for_family) if card else False,
        "card_fallback_preset": card.fallback_preset if card else None,
        "card_primary_metric_kind": (
            card.primary_metric.kind if (card and card.primary_metric) else None
        ),
        "card_primary_metric_value": (
            card.primary_metric.value if (card and card.primary_metric) else None
        ),
        "card_evidence_visibility": card.evidence_visibility if card else None,
        "card_evidence_ref_count": len(ev_refs),
        "card_evidence_refs": ev_refs,
    }


def _build_profile_row(profile_id: str, generated_at: str,
                       class4_clean_set: set[str]) -> dict:
    """Build ProfileRow dict from profile YAML + override_policy + class4 verdict.

    `class4_clean_set` is precomputed once (operator-locked § acceptance
    criterion: every profile in catalog reports its Class-4 cleanliness).
    """
    from sndr.model_configs.registry_v2 import (
        _builtin_dir, load_profile,
    )

    yaml_path = _builtin_dir("profile") / f"{profile_id}.yaml"
    profile = load_profile(profile_id)

    sizing = profile.sizing_override
    policy = profile.override_policy
    patches = profile.patches_delta

    # Build override evidence_refs with redaction (string list, not EvidenceRef dataclass)
    override_ev_count = (
        len(policy.evidence_refs) if (policy and policy.evidence_refs) else 0
    )

    return {
        "schema_version": 1,
        "row_type": "profile",
        "id": profile_id,
        "source_path": str(yaml_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256_file(yaml_path),
        "status": profile.status,
        "family": None,  # profile family derived via parent_model
        "tags": [profile.role] if profile.role else [],
        "updated_from_git_commit": _last_git_commit_for(yaml_path),
        "generated_at": generated_at,
        # Type-specific
        "parent_model": profile.parent_model,
        "role": profile.role,
        "sizing_max_model_len": sizing.max_model_len if sizing else None,
        "sizing_max_num_seqs": sizing.max_num_seqs if sizing else None,
        "sizing_max_num_batched_tokens": sizing.max_num_batched_tokens if sizing else None,
        "sizing_gpu_memory_utilization": sizing.gpu_memory_utilization if sizing else None,
        "sizing_enable_chunked_prefill": sizing.enable_chunked_prefill if sizing else None,
        "sizing_enforce_eager": sizing.enforce_eager if sizing else None,
        "has_override_policy": policy is not None,
        "override_class": policy.override_class if policy else None,
        "override_reason": policy.reason if policy else None,
        "override_evidence_ref_count": override_ev_count,
        "override_evidence_visibility": policy.evidence_visibility if policy else None,
        "override_expires_at": policy.expires_at if policy else None,
        "override_allowed_to_exceed_hardware_default": (
            policy.allowed_to_exceed_hardware_default if policy else False
        ),
        "class4_clean": profile_id in class4_clean_set,
        "patches_enable_count": len(patches.enable) if patches else 0,
        "patches_disable_count": len(patches.disable) if patches else 0,
        "patches_override_count": len(patches.override) if patches else 0,
    }


def _build_model_row(model_id: str, generated_at: str) -> dict:
    """Build ModelRow dict from model YAML."""
    from sndr.model_configs.registry_v2 import (
        _builtin_dir, load_model,
    )

    yaml_path = _builtin_dir("model") / f"{model_id}.yaml"
    model = load_model(model_id)
    caps = model.capabilities
    versions = model.versions

    sd = caps.spec_decode if caps else None

    # Count enabled patches (genesis_env keys with values that look like enabled flags)
    enabled_count = 0
    for v in (model.patches or {}).values():
        if str(v) in ("1", "true", "True"):
            enabled_count += 1

    return {
        "schema_version": 1,
        "row_type": "model",
        "id": model_id,
        "source_path": str(yaml_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256_file(yaml_path),
        "status": None,
        "family": None,
        "tags": [],
        "updated_from_git_commit": _last_git_commit_for(yaml_path),
        "generated_at": generated_at,
        # Type-specific
        "title": model.title,
        "quantization": model.quantization,
        "kv_cache_dtype": caps.kv_cache_dtype if caps else None,
        "spec_decode_method": sd.method if sd else None,
        "spec_decode_K": sd.num_speculative_tokens if sd else None,
        "enable_auto_tool_choice": caps.enable_auto_tool_choice if caps else None,
        "tool_call_parser": caps.tool_call_parser if caps else None,
        "vllm_pin_required": versions.vllm_pin_required if versions else None,
        "genesis_pin_min": versions.genesis_pin_min if versions else None,
        "enabled_patches_count": enabled_count,
    }


def _build_hardware_row(hardware_id: str, generated_at: str) -> dict:
    """Build HardwareRow dict from hardware YAML."""
    from sndr.model_configs.registry_v2 import (
        _builtin_dir, load_hardware,
    )

    yaml_path = _builtin_dir("hardware") / f"{hardware_id}.yaml"
    hw = load_hardware(hardware_id)
    spec = hw.hardware
    sizing = hw.sizing
    runtime = hw.runtime

    return {
        "schema_version": 1,
        "row_type": "hardware",
        "id": hardware_id,
        "source_path": str(yaml_path.relative_to(REPO_ROOT)),
        "source_sha256": _sha256_file(yaml_path),
        "status": None,
        "family": None,
        "tags": [],
        "updated_from_git_commit": _last_git_commit_for(yaml_path),
        "generated_at": generated_at,
        # Type-specific
        "title": hw.title,
        "n_gpus": spec.n_gpus if spec else 0,
        "gpu_match_keys": list(spec.gpu_match_keys) if spec else [],
        "min_vram_per_gpu_mib": spec.min_vram_per_gpu_mib if spec else None,
        "cuda_capability_min": (
            list(spec.cuda_capability_min) if (spec and spec.cuda_capability_min) else None
        ),
        "sizing_max_model_len": sizing.max_model_len if sizing else None,
        "sizing_max_num_seqs": sizing.max_num_seqs if sizing else None,
        "sizing_gpu_memory_utilization": sizing.gpu_memory_utilization if sizing else None,
        "runtime_default": runtime.default if runtime else None,
    }


def _build_baseline_row(
    baseline_path: Path, generated_at: str,
    model_ids: list[str], preset_evidence_index: dict[str, list[str]],
) -> dict:
    """Build BaselineRow dict from baseline JSON + corpus match.

    `preset_evidence_index` maps `tests/integration/baselines/<file>.json` →
    list of preset_ids citing that path in card.evidence_refs.
    """
    with baseline_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data.get("config", {}) if isinstance(data.get("config"), dict) else {}

    raw_model = str(data.get("model", ""))

    # Match quality determination
    matched_models: list[str] = []
    if raw_model:
        for mid in model_ids:
            # `qwen3.6-27b` matches `qwen3.6-27b-int4-autoround-tq-k8v4` family
            if mid.startswith(raw_model):
                matched_models.append(mid)

    rel_baseline = str(baseline_path.relative_to(REPO_ROOT))
    matched_presets = preset_evidence_index.get(rel_baseline, [])

    if matched_presets:
        match_quality = "exact_preset"
    elif len(matched_models) == 1:
        match_quality = "model_only"
    elif len(matched_models) > 1:
        match_quality = "family_only"
    else:
        match_quality = "none"

    vllm_pin = data.get("vllm_version")
    if isinstance(vllm_pin, dict):
        vllm_pin = vllm_pin.get("system_fingerprint")

    return {
        "schema_version": 1,
        "row_type": "baseline",
        "id": baseline_path.stem,
        "source_path": rel_baseline,
        "source_sha256": _sha256_file(baseline_path),
        "status": None,
        "family": raw_model or None,
        "tags": [],
        "updated_from_git_commit": _last_git_commit_for(baseline_path),
        "generated_at": generated_at,
        # Type-specific
        "bench_model": raw_model,
        "bench_vllm_pin": vllm_pin if isinstance(vllm_pin, str) else None,
        "bench_ctx": cfg.get("ctx"),
        "bench_max_tokens": cfg.get("max_tokens"),
        "bench_prompts_set": cfg.get("prompts_set"),
        "bench_runs": cfg.get("runs"),
        "match_quality": match_quality,
        "matched_model_ids": matched_models,
        "matched_preset_ids": matched_presets,
    }


# ─── Catalog assembly ──────────────────────────────────────────────────────


def build_catalog(*, generated_at: Optional[str] = None) -> list[dict]:
    """Assemble the full catalog as a list of row dicts.

    Deterministic ordering: rows sorted by `(row_type, id)`.
    `generated_at` defaults to current UTC ISO timestamp; tests can pin
    a fixed value to compare regenerated catalogs without timestamp churn.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    from sndr.model_configs.registry_v2 import (
        _alias_dir, _builtin_dir, list_hardware, list_models, list_profiles,
    )

    # Suppress the V1/card-less DeprecationWarning chatter during full corpus scan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Pre-compute Class-4 cleanliness per profile (one audit pass)
        class4_clean_set = _compute_class4_clean_set()

        # Pre-compute preset_path → preset_id index for baseline match
        preset_evidence_index = _compute_preset_evidence_index()

        rows: list[dict] = []

        # Presets (sorted by id)
        for path in sorted(_alias_dir().glob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            try:
                rows.append(_build_preset_row(path.stem, generated_at))
            except Exception as e:  # pragma: no cover — defensive
                print(
                    f"WARN: failed to build preset row for {path.stem}: {e}",
                    file=sys.stderr,
                )

        # Profiles
        for pid in sorted(list_profiles()):
            try:
                rows.append(_build_profile_row(pid, generated_at, class4_clean_set))
            except Exception as e:
                print(
                    f"WARN: failed to build profile row for {pid}: {e}",
                    file=sys.stderr,
                )

        # Models
        model_ids = sorted(list_models())
        for mid in model_ids:
            try:
                rows.append(_build_model_row(mid, generated_at))
            except Exception as e:
                print(
                    f"WARN: failed to build model row for {mid}: {e}",
                    file=sys.stderr,
                )

        # Hardware
        for hid in sorted(list_hardware()):
            try:
                rows.append(_build_hardware_row(hid, generated_at))
            except Exception as e:
                print(
                    f"WARN: failed to build hardware row for {hid}: {e}",
                    file=sys.stderr,
                )

        # Baselines
        baselines_dir = REPO_ROOT / "tests" / "integration" / "baselines"
        if baselines_dir.is_dir():
            for path in sorted(baselines_dir.glob("*.json")):
                try:
                    rows.append(_build_baseline_row(
                        path, generated_at,
                        model_ids=model_ids,
                        preset_evidence_index=preset_evidence_index,
                    ))
                except Exception as e:
                    print(
                        f"WARN: failed to build baseline row for {path.name}: {e}",
                        file=sys.stderr,
                    )

    return rows


def _compute_class4_clean_set() -> set[str]:
    """Run audit_override_policy.run_audit() once; return profile_ids
    with zero `forbidden_override.*` findings.

    Logic: enumerate ALL builtin profile ids, then subtract profiles
    that have any Class-4 violation in the audit report. Profiles with
    zero findings (the clean ones) appear in the returned set.
    """
    import importlib.util
    from sndr.model_configs.registry_v2 import list_profiles

    script = REPO_ROOT / "scripts" / "audit_override_policy.py"
    spec = importlib.util.spec_from_file_location("_aopaudit_class4_helper", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_aopaudit_class4_helper"] = mod
    spec.loader.exec_module(mod)

    report = mod.run_audit()
    violating: set[str] = {
        f.profile_id for f in report.findings
        if f.rule.startswith("forbidden_override.")
    }
    all_profiles = set(list_profiles())
    return all_profiles - violating


def _compute_preset_evidence_index() -> dict[str, list[str]]:
    """Map evidence path → preset_ids citing that path in their card.

    Used by baseline rows to detect `exact_preset` match quality.
    """
    from sndr.model_configs.registry_v2 import _alias_dir, load_preset_def
    out: dict[str, list[str]] = {}
    for path in sorted(_alias_dir().glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        try:
            pd = load_preset_def(path.stem)
        except Exception:
            continue
        if not pd.has_card():
            continue
        for ref in pd.card.evidence_refs:
            out.setdefault(ref.path, []).append(path.stem)
    return out


# ─── Serialisation ──────────────────────────────────────────────────────────


def serialise_catalog(rows: list[dict]) -> str:
    """Render catalog as deterministic JSON.

    Two-key sort by (row_type, id) for stable diff; pretty-printed
    for human readability; trailing newline so check-in writes a
    POSIX-clean file.
    """
    rows_sorted = sorted(rows, key=lambda r: (r["row_type"], r["id"]))
    payload = {
        "schema_version": 1,
        "row_count": len(rows_sorted),
        "rows": rows_sorted,
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def serialise_for_drift(rows: list[dict]) -> str:
    """Like `serialise_catalog` but with `generated_at` and
    `updated_from_git_commit` stripped per row — used by `--check` to
    detect content drift independent of build timestamp / git churn."""
    rows_stripped = []
    for r in rows:
        copy = dict(r)
        copy.pop("generated_at", None)
        copy.pop("updated_from_git_commit", None)
        rows_stripped.append(copy)
    return serialise_catalog(rows_stripped)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="emit JSON to stdout instead of writing to build/...",
    )
    parser.add_argument(
        "--check", action="store_true",
        help=(
            "Verify drift-stripped content is deterministic — runs generator "
            "twice and asserts byte-equality of drift-stripped output. Exit 1 "
            "on mismatch. Does NOT require a committed JSON artifact (CONFIG-UX.5.1 "
            "scope: no committed catalog yet)."
        ),
    )
    args = parser.parse_args()

    try:
        rows = build_catalog()
    except Exception as e:
        print(f"generate-config-catalog: internal error: {e}", file=sys.stderr)
        return 2

    if args.check:
        # Deterministic regeneration check: rebuild and compare drift-stripped output
        rows2 = build_catalog()
        s1 = serialise_for_drift(rows)
        s2 = serialise_for_drift(rows2)
        if s1 != s2:
            print(
                "generate-config-catalog --check: non-deterministic output "
                "(content differs between two consecutive runs)",
                file=sys.stderr,
            )
            return 1
        print(f"generate-config-catalog --check: deterministic ({len(rows)} rows)")
        return 0

    payload = serialise_catalog(rows)
    if args.stdout:
        sys.stdout.write(payload)
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    print(
        f"generate-config-catalog: wrote {len(rows)} rows to "
        f"{OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
