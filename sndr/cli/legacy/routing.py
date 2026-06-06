# SPDX-License-Identifier: Apache-2.0
"""`sndr routing-table` — emit the canonical workload-gate routing
table for external consumers (aggregators, proxies, gateways).

Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT (2026-05-23): Option D —
two-preset deployment + caller-side routing. The K choice (K=1 vs
K=4) is a per-request decision, not a per-engine setting. This CLI
publishes the rule table; the aggregator consumes it at startup and
applies the rules per incoming request.

Emit-only contract. No engine surgery, no runtime hooks. The
``sndr_core`` repo is authoritative for the table content (rules
follow bench evidence from B1.1 / B1.2 / B2 / B3 / B4 verdicts);
external consumers must NOT mutate it.

Schema lives at ``vllm/sndr_core/cli/routing_schema.json`` (v1
frozen). Versioning policy: consumers must warn-and-degrade-
gracefully on unknown schema versions, not crash.

Usage:

  python3 -m vllm.sndr_core.cli routing-table --json
  python3 -m vllm.sndr_core.cli routing-table --json --out /tmp/routing.json
  python3 -m vllm.sndr_core.cli routing-table --validate  # schema check
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

from . import _io


__all__ = [
    "add_argparser",
    "compute_routing_table",
    "run_routing_table",
    "SCHEMA_VERSION",
]


SCHEMA_VERSION = 1
SHORT_THRESHOLD_TOKENS = 256
DEFAULT_K = 1

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = Path(__file__).parent / "routing_schema.json"


# ─── Model-family mapping ───────────────────────────────────────────────

# Maps the ModelDef id (preset.model field) to the routing-axis
# family label. New models default to "unknown" so the aggregator's
# fallback (no K=4 override on unknown families) keeps them safe.
_MODEL_FAMILY: dict[str, str] = {
    "gemma-4-26b-a4b-it-awq":               "gemma4_moe_26b_a4b",
    "gemma-4-26b-a4b-it-awq-experimental":  "gemma4_moe_26b_a4b",
    "gemma-4-31b-it-awq":                   "gemma4_dense_31b",
    "gemma-4-31b-it-awq-mtp-n8-code":       "gemma4_dense_31b",
    "qwen3.6-27b":                          "qwen3_6_27b",
    "qwen3.6-27b-int4":                     "qwen3_6_27b",
    "qwen3.6-27b-fp8":                      "qwen3_6_27b",
    "qwen3.6-27b-dflash":                   "qwen3_6_27b",
    "qwen3.6-35b-a3b":                      "qwen3_6_35b",
    "qwen3.6-35b-a3b-fp8":                  "qwen3_6_35b",
}


def _classify_model_family(model_id: str) -> str:
    """Return canonical family label for a ModelDef id, or 'unknown'."""
    return _MODEL_FAMILY.get(model_id, "unknown")


# ─── Preset discovery ───────────────────────────────────────────────────


def _load_preset_yaml(path: Path) -> dict[str, Any]:
    """Minimal preset YAML loader: top-level key=value pairs only.
    Preset aliases are 3-line files (model / hardware / profile), so
    we sidestep importing pyyaml and parse what we need by hand.
    """
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _list_preset_paths() -> list[Path]:
    presets_dir = _REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "presets"
    return sorted(p for p in presets_dir.glob("*.yaml") if p.is_file())


def _discover_presets() -> list[dict[str, Any]]:
    """Walk every builtin preset alias, resolve its parent profile +
    parent model, return one normalized dict per preset.

    Defensive: a missing profile or model yields an entry with
    spec_decode_K=DEFAULT_K and role=None — never crashes the emit.
    """
    from vllm.sndr_core.model_configs.registry_v2 import (
        load_profile, load_model,
    )
    from vllm.sndr_core.model_configs.schema import SchemaError

    rows: list[dict[str, Any]] = []
    for preset_path in _list_preset_paths():
        preset_key = preset_path.stem
        fields = _load_preset_yaml(preset_path)
        model_id = fields.get("model", "")
        profile_id = fields.get("profile", "")

        # Resolve spec_decode K — profile override takes precedence,
        # then ModelDef default, then K=1.
        spec_K = DEFAULT_K
        role: str | None = None
        intended_workloads: list[str] = []
        max_num_seqs = 1
        served_model_name: str | None = None

        if profile_id:
            try:
                prof = load_profile(profile_id)
                role = getattr(prof, "role", None)
                if prof.spec_decode_override is not None:
                    spec_K = int(prof.spec_decode_override.num_speculative_tokens)
                if prof.routing is not None:
                    intended_workloads = list(prof.routing.intended_workloads)
                if prof.sizing_override is not None:
                    max_num_seqs = int(getattr(prof.sizing_override, "max_num_seqs", 1) or 1)
            except (SchemaError, FileNotFoundError):
                pass

        if model_id:
            try:
                m = load_model(model_id)
                if spec_K == DEFAULT_K and getattr(m, "capabilities", None):
                    sd = getattr(m.capabilities, "spec_decode", None)
                    if sd is not None:
                        spec_K = int(sd.num_speculative_tokens)
                served_model_name = getattr(m, "served_model_name", None)
            except (SchemaError, FileNotFoundError):
                pass

        family = _classify_model_family(model_id)

        rows.append({
            "preset_key":         preset_key,
            "model":              model_id,
            "served_model_name":  served_model_name,
            "model_family":       family,
            "spec_decode_K":      spec_K,
            "max_num_seqs":       max_num_seqs,
            "role":               role,
            "intended_workloads": intended_workloads,
            "default_for_family": False,  # filled in below
        })

    # Mark default_for_family per family: the K=1 preset with the
    # smallest max_num_seqs and role=default wins (the canonical
    # "broad workload safe" choice from WORKLOAD-GATE-POLICY.UPDATE §3).
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["model_family"], []).append(row)

    for family, family_rows in by_family.items():
        candidates = [
            r for r in family_rows
            if r["spec_decode_K"] == 1
            and r["role"] in (None, "default")
            and r["max_num_seqs"] <= 2  # single-stream-class
        ]
        if not candidates:
            # Family has no K=1 default-role low-conc preset — happens
            # for Qwen presets where K decisions weren't touched by G4
            # work. Skip default_for_family flagging entirely; the
            # aggregator will fall back to V1 model-config for these.
            continue
        # Deterministic tiebreak: alphabetical preset_key.
        candidates.sort(key=lambda r: r["preset_key"])
        candidates[0]["default_for_family"] = True

    return rows


# ─── Routing rules ──────────────────────────────────────────────────────


def _build_routing_rules(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hand-curated rules from WORKLOAD-GATE-POLICY.UPDATE §4 +
    IMPLEMENT.R §4. Order matters: multi-conc rules MUST precede
    single-stream rules for the same workload axis.

    A rule is emitted only if its target preset_key exists in the
    discovered preset list — keeps the table self-consistent if a
    preset is ever renamed/retired.
    """
    preset_keys = {r["preset_key"] for r in presets}

    candidate_rules: list[dict[str, Any]] = [
        # ── 26B-A4B MoE — multi-conc structured (B4 measured) ──
        {
            "model_family": "gemma4_moe_26b_a4b",
            "when": {
                "workload_class":         ["structured_json", "tool_call"],
                "expected_output_length": ["short"],
                "concurrency_mode":       ["multi_conc"],
            },
            "preset_key": "prod-gemma4-26b-multiconc",
            "evidence": (
                "B4 (2026-05-23): K=4 conc=8 Mode A 235.9 TPS vs K=1 209.1 TPS "
                "(+12.8%); Mode B xgrammar 271.4 TPS / schema_rate 100%. See "
                "sndr_private/runs/g4_26b_a4b_structured_multiconc_2026-05-23/."
            ),
            "evidence_tag": "measured",
        },
        # ── 26B-A4B MoE — single-stream structured (B2 measured) ──
        {
            "model_family": "gemma4_moe_26b_a4b",
            "when": {
                "workload_class":         ["structured_json", "tool_call"],
                "expected_output_length": ["short"],
                "concurrency_mode":       ["single_stream"],
            },
            "preset_key": "prod-gemma4-26b-mtp-k4",
            "evidence": (
                "B2 (2026-05-23): K=4 short-structured mean elapsed 443 ms "
                "vs K=1 720 ms (-38%); parse_rate 10/10 vs 9/10. See "
                "sndr_private/runs/g4_26b_a4b_K1_K4_single_2026-05-23/."
            ),
            "evidence_tag": "measured",
        },
        # ── 31B dense — single-stream structured (B1.2 measured) ──
        {
            "model_family": "gemma4_dense_31b",
            "when": {
                "workload_class":         ["structured_json", "tool_call"],
                "concurrency_mode":       ["single_stream"],
            },
            "preset_key": "prod-gemma4-31b-tq-mtp-structured-k4",
            "evidence": (
                "B1.2 (2026-05-23): K=4 structured-JSON mean k=2.79, "
                "+12% elapsed (Mode A excl. warmup). β'-A artifact "
                "config_hash 71c874d7ffedae04 allowed_workloads = "
                "[tool_json, structured_count]. See "
                "sndr_private/runs/g4_betaA_K1_K4_tooljson_2026-05-23/."
            ),
            "evidence_tag": "measured",
        },
    ]

    return [r for r in candidate_rules if r["preset_key"] in preset_keys]


# ─── Coverage gaps ──────────────────────────────────────────────────────


def _build_coverage_gaps(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explicit known-untested cells. Shipping with these gaps is
    operator-authorized (§9.1); making them explicit (rather than
    silently letting requests fall to the default) gives consumers
    a foothold to surface the gap to operators.
    """
    families_present = {r["model_family"] for r in presets}
    gaps: list[dict[str, Any]] = []

    if "gemma4_dense_31b" in families_present:
        gaps.append({
            "model_family": "gemma4_dense_31b",
            "missing_cell": (
                "workload_class=structured_json/tool_call AND "
                "concurrency_mode=multi_conc"
            ),
            "fallback_preset": "prod-gemma4-31b-tq-default",
            "next_phase": "7.G4.31B.MULTICONC (Q2 from WORKLOAD-GATE-POLICY.R §8)",
        })
        gaps.append({
            "model_family": "gemma4_dense_31b",
            "missing_cell": (
                "concurrency_mode=multi_conc for ANY workload "
                "(no prod-gemma4-31b-tq-multiconc preset exists)"
            ),
            "fallback_preset": "prod-gemma4-31b-tq-default",
            "next_phase": "7.G4.31B.MULTICONC.PRE (preset creation precursor)",
        })

    # Tool-call workload as a standalone class — currently inferred
    # from structured_json similarity, not measured directly.
    if "gemma4_moe_26b_a4b" in families_present or "gemma4_dense_31b" in families_present:
        gaps.append({
            "model_family": "all",
            "missing_cell": (
                "workload_class=tool_call (real vLLM tool-call envelope, "
                "tools[] field non-empty) — currently inferred from "
                "structured_json.short evidence"
            ),
            "fallback_preset": "(same rule as structured_json — inferred allow)",
            "next_phase": "7.G4.TOOL_CALL_GATE (Q3 from WORKLOAD-GATE-POLICY.R §8)",
        })

    return gaps


# ─── Static spec blocks ─────────────────────────────────────────────────


def _workload_class_detection() -> dict[str, Any]:
    """Language-neutral workload detection spec. The aggregator
    implements this; the spec ships in JSON to keep the contract
    portable to non-Python consumers."""
    return {
        "evaluation_order": [
            "tool_call",
            "structured_json",
            "summarization",
            "code_gen",
            "free_chat",
        ],
        "classes": {
            "tool_call": {
                "trigger": "request.body.tools is a non-empty array",
                "priority": 100,
            },
            "structured_json": {
                "trigger": (
                    "request.body.response_format.type IN "
                    "[json_object, json_schema]"
                ),
                "priority": 90,
                "subtype": {
                    "short": (
                        f"max_tokens <= {SHORT_THRESHOLD_TOKENS} "
                        "OR caller-hint expected_length=short"
                    ),
                    "long": (
                        f"max_tokens > {SHORT_THRESHOLD_TOKENS} "
                        "AND not flagged short"
                    ),
                    "priority_note": (
                        "Caller hint expected_length wins over the "
                        "max_tokens-derived bucket. See length_detection."
                    ),
                },
            },
            "summarization": {
                "trigger": (
                    "input_tokens > 2 * max_tokens "
                    "AND no response_format AND no tools"
                ),
                "priority": 50,
            },
            "code_gen": {
                "trigger": (
                    "caller-hint category=code (no auto-detection in v1; "
                    "opt-in only)"
                ),
                "priority": 40,
            },
            "free_chat": {
                "trigger": "default — no other class matched",
                "priority": 0,
            },
        },
    }


def _concurrency_mode_detection() -> dict[str, Any]:
    return {
        "single_stream": "in_flight_request_count <= 1 on the target upstream",
        "multi_conc":    "in_flight_request_count >= 2 on the target upstream",
        "owner": (
            "Consumer (aggregator/proxy). This repo does not expose an "
            "in-flight counter."
        ),
        "fallback": "single_stream (conservative — never silently enables K=4 on uncertain conc).",
    }


def _length_detection() -> dict[str, Any]:
    return {
        "short_threshold_tokens": SHORT_THRESHOLD_TOKENS,
        "caller_hint_wins": True,
        "buckets": {
            "short":  f"max_tokens <= {SHORT_THRESHOLD_TOKENS}",
            "medium": f"{SHORT_THRESHOLD_TOKENS} < max_tokens <= 512 (unused in v1 rules; reserved)",
            "long":   "max_tokens > 512",
        },
    }


def _fallback_policy() -> dict[str, Any]:
    return {
        "no_rule_matched":       "Use the preset with default_for_family=true in the requested model_family. Effective K=1.",
        "model_family_unknown":  "Use the model's natural preset (V1 model-config lookup). Do NOT apply K=4 overrides. Effective K=1.",
        "classification_failed": "Treat workload_class as free_chat → falls through to no_rule_matched.",
        "default_K":             DEFAULT_K,
    }


# ─── Public library entry point ─────────────────────────────────────────


def compute_routing_table(
    *,
    vllm_pin: str | None = None,
    version_string: str | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Build the full routing-table dict (no I/O).

    Library callers can use this directly; the CLI is a thin wrapper.
    """
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    if version_string is None:
        version_string = "vllm-sndr-core/genesis"

    presets = _discover_presets()
    rules = _build_routing_rules(presets)
    gaps = _build_coverage_gaps(presets)

    # Suppress default_for_family for families that have no routing
    # rules in our table. Without a rule set, there's nothing for the
    # consumer to fall back FROM, so a default would be misleading.
    # Qwen / unknown families therefore have no default flag — the
    # aggregator's documented fallback ("model_family unknown → V1
    # model-config lookup") handles them correctly.
    families_with_rules = {r["model_family"] for r in rules}
    for p in presets:
        if p["default_for_family"] and p["model_family"] not in families_with_rules:
            p["default_for_family"] = False

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by":   version_string,
        "generated_at":   now.isoformat(timespec="seconds"),
        "vllm_pin":       vllm_pin,
        "presets":        presets,
        "routing_rules":  rules,
        "workload_class_detection": _workload_class_detection(),
        "concurrency_mode_detection": _concurrency_mode_detection(),
        "length_detection": _length_detection(),
        "fallback":       _fallback_policy(),
        "coverage_gaps":  gaps,
    }


# ─── Schema validation ──────────────────────────────────────────────────


def load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_against_schema(table: dict[str, Any]) -> list[str]:
    """Best-effort structural validation (no jsonschema dependency).
    Returns a list of error strings; empty list = valid.

    Checks the schema-v1 invariants the aggregator depends on:
      - schema_version == 1
      - required top-level keys present
      - presets entries have required fields
      - routing_rules.preset_key references exist in presets
      - exactly one default_for_family per family that has at least one default candidate
      - length_detection.short_threshold_tokens == 256
      - fallback.default_K == 1
    """
    errors: list[str] = []
    schema = load_schema()

    # Top-level required keys.
    for key in schema.get("required", []):
        if key not in table:
            errors.append(f"missing required top-level key: {key}")

    # schema_version invariant.
    if table.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{table.get('schema_version')!r}"
        )

    # Preset structure + reference integrity.
    preset_keys: set[str] = set()
    for i, p in enumerate(table.get("presets", [])):
        for required in ("preset_key", "model", "model_family",
                         "spec_decode_K", "max_num_seqs", "role",
                         "intended_workloads", "default_for_family"):
            if required not in p:
                errors.append(f"presets[{i}] missing field: {required}")
        if "preset_key" in p:
            preset_keys.add(p["preset_key"])

    # Routing-rule target presets must exist.
    for i, r in enumerate(table.get("routing_rules", [])):
        if r.get("preset_key") not in preset_keys:
            errors.append(
                f"routing_rules[{i}].preset_key={r.get('preset_key')!r} "
                f"references unknown preset"
            )
        if not r.get("evidence"):
            errors.append(f"routing_rules[{i}] has empty evidence string")

    # default_for_family — at most one per family.
    family_default_count: dict[str, int] = {}
    for p in table.get("presets", []):
        if p.get("default_for_family"):
            fam = p.get("model_family", "")
            family_default_count[fam] = family_default_count.get(fam, 0) + 1
    for fam, count in family_default_count.items():
        if count > 1:
            errors.append(f"model_family={fam!r} has {count} default_for_family presets; max 1")

    # Length-detection threshold frozen.
    ld = table.get("length_detection", {})
    if ld.get("short_threshold_tokens") != SHORT_THRESHOLD_TOKENS:
        errors.append(
            f"length_detection.short_threshold_tokens must be "
            f"{SHORT_THRESHOLD_TOKENS}, got {ld.get('short_threshold_tokens')!r}"
        )

    # Fallback K frozen.
    fb = table.get("fallback", {})
    if fb.get("default_K") != DEFAULT_K:
        errors.append(
            f"fallback.default_K must be {DEFAULT_K}, got {fb.get('default_K')!r}"
        )

    return errors


# ─── CLI ────────────────────────────────────────────────────────────────


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "routing-table",
        help="Emit the workload-gate routing table for external "
             "callers (aggregators, proxies).",
        description=(
            "Phase 7.G4.WORKLOAD-GATE-POLICY.IMPLEMENT emit surface. "
            "Computes the routing-table contract from existing preset "
            "metadata + bench-evidenced rules and emits JSON (schema "
            "v1). Read by external consumers at startup; cache for the "
            "process lifetime, restart to refresh."
        ),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit JSON (default; here for symmetry with other sndr CLIs).")
    p.add_argument("--out", default=None,
                   help="Write to FILE instead of stdout.")
    p.add_argument("--validate", action="store_true",
                   help="Run structural validation against the v1 schema "
                        "and exit non-zero if any errors found.")
    p.add_argument("--vllm-pin", default=None,
                   help="Optional vllm pin string to stamp into the output.")
    p.set_defaults(func=run_routing_table)


def run_routing_table(opts: argparse.Namespace) -> int:
    table = compute_routing_table(vllm_pin=opts.vllm_pin)
    if opts.validate:
        errors = validate_against_schema(table)
        if errors:
            for e in errors:
                _io.error(e)
            return 1
        _io.success("routing-table validates against schema v1")
        return 0
    payload = json.dumps(table, indent=2)
    if opts.out:
        Path(opts.out).write_text(payload + "\n", encoding="utf-8")
        _io.success(f"wrote {opts.out}")
    else:
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()
    return 0
