#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 hardware-schema gate — `make audit-launch-coverage`.

The V1→V2 migration of `model_configs/builtin/` split each monolithic
preset (`*.yaml`) into a (model, hardware, profile) triplet plus an
alias preset. During that migration, several canonical mount slots and
system_env keys were silently dropped from the V2 hardware files —
discovered by the operator in Entry 22 investigation:

  • `/root/.triton/cache` mount → Triton kernel cache became ephemeral,
    +30-60s recompile penalty on every container restart.
  • `/plugin` overlay → Genesis vLLM plugin unavailable.

(The legacy `vllm/sndr_core` compat overlay was retired in v12 — the rig
now loads patches via the modern `sndr.plugin:register` entry point, and
the canonical source is mounted at `dist-packages/sndr`.)

This gate freezes the canonical mount + env schema as a static
contract. Every V2 hardware YAML under `model_configs/builtin/hardware/`
must declare a mount entry for each REQUIRED container_path and an
env entry for each REQUIRED system_env key. Adding a new V2 hardware
file with missing slots will fail this gate at PR time, before the
operator pays the runtime cost.

Schema definitions live in the constants below — extend them when
new canonical slots become mandatory.

Exit codes:
  0 — every V2 hardware YAML satisfies the canonical schema
  1 — at least one V2 hardware file is missing required slots
  2 — internal error (YAML parse, etc.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HW_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


# ─── Canonical mount schema ──────────────────────────────────────────
#
# Ground truth: scripts/launch/_archive/superseded_by_model_configs/
# start_35b_fp8_PROD.sh + the 8 real V1 YAMLs. Both reference the same
# 5 mount slots (per-config tag differs, but container_path + access
# mode are invariant).
#
# Each entry: (container_path, mode, description, required_unless)
# `required_unless` is a free-form note when the slot is conditionally
# optional (e.g. plugin not needed when no plugin code is shipped).
#
# Adding to this list is a deliberate operator decision — the migration
# of V1 to V2 cannot drop one of these slots and still claim
# launch-parity.

@dataclass(frozen=True)
class MountSlot:
    container_path: str
    mode: str                  # "ro" | "rw"
    description: str
    host_var: str = ""         # canonical ${...} placeholder for auto-completer
    required_unless: str = ""  # operator-visible escape hatch


REQUIRED_MOUNTS: tuple[MountSlot, ...] = (
    MountSlot(
        "/models", "ro",
        "model checkpoints — read-only volume",
        host_var="${models_dir}",
    ),
    MountSlot(
        "/root/.cache/huggingface", "ro",
        "HuggingFace cache — read-only volume",
        host_var="${hf_cache}",
    ),
    MountSlot(
        "/root/.triton/cache", "rw",
        "Triton kernel cache — must persist or pay +30-60s recompile on restart",
        host_var="${triton_cache}",
    ),
    MountSlot(
        "/root/.cache/vllm/torch_compile_cache", "rw",
        "torch.compile cache — must persist or pay recompile on restart",
        host_var="${compile_cache}",
    ),
    MountSlot(
        "/plugin", "ro",
        "Genesis vLLM plugin source — REQUIRED for plugin loading",
        host_var="${plugin_src}",
        required_unless="plugin code is not shipped with this hardware profile",
    ),
)


# ─── Canonical env defaults (used by auto-completer) ─────────────────
#
# Frozen canonical values for each REQUIRED env key. The auto-completer
# (`scripts/config_v2_complete.py`) uses these when a hardware YAML is
# missing a required env key — it injects the canonical value rather
# than asking the operator to fill in a blank.
#
# Values reflect the V1 PROD reference + project policy:
#   - PYTORCH_CUDA_ALLOC_CONF: production tuning (expandable_segments + 256 MiB max_split)
#   - OMP_NUM_THREADS: 1 (avoid CPU oversubscription with TP>1)
#   - CUDA_DEVICE_MAX_CONNECTIONS: 8 (spec-decode K+1 verify pattern optimum)
#   - TRITON_CACHE_DIR: must match the /root/.triton/cache mount path
#   - VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1 (project default for long-context profiles)
#   - VLLM_WORKER_MULTIPROC_METHOD: spawn (REQUIRED for CUDA reinit when TP>1)
#   - VLLM_NO_USAGE_STATS: 1 (project privacy policy)

ENV_DEFAULTS: dict[str, str] = {
    "PYTORCH_CUDA_ALLOC_CONF": "'expandable_segments:True,max_split_size_mb:256'",
    "OMP_NUM_THREADS": "'1'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "'8'",
    "TRITON_CACHE_DIR": "'/root/.triton/cache'",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "'1'",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_NO_USAGE_STATS": "'1'",
}


# ─── Env-VALUE invariants (E24) ───────────────────────────────────────
#
# Beyond "key is present" (E22), some env keys carry STRUCTURAL invariants:
# their value must match a specific other field. The most important is
# `TRITON_CACHE_DIR` — its value must equal the container_path of the
# Triton mount, otherwise Triton writes to a different directory than
# the one mounted and the cache becomes ephemeral despite the mount
# being present.
#
# `ENV_VALUE_LINKS[key]` says: the value of `key` MUST equal the
# container_path of the referenced mount slot (matched by container_path).
#
# `ENV_VALUE_LITERALS[key]` says: the value of `key` MUST equal this
# exact literal — used for keys with no flexibility (e.g.
# VLLM_WORKER_MULTIPROC_METHOD must be `spawn` for TP>1 correctness).
#
# Both checks tolerate YAML's quoting variants: '1', "1", 1 are all
# treated equivalent. Comparison is on the trimmed string form.

ENV_VALUE_LINKS: dict[str, str] = {
    # When TRITON_CACHE_DIR is set, it MUST equal the Triton mount path.
    "TRITON_CACHE_DIR": "/root/.triton/cache",
}

ENV_VALUE_LITERALS: dict[str, frozenset[str]] = {
    # `spawn` is structurally required for CUDA reinit across TP workers.
    # `fork` causes silent CUDA-context corruption.
    "VLLM_WORKER_MULTIPROC_METHOD": frozenset({"spawn"}),
    # Telemetry-off is project policy (privacy + offline boots).
    "VLLM_NO_USAGE_STATS": frozenset({"1", "true", "True"}),
    # Long-context contract — '1' is the only sensible value when this
    # knob exists (key is required by E22; if you set '0', drop the key).
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": frozenset({"1", "true", "True"}),
}


def _normalize_env_value(v) -> str:
    """Coerce a YAML-loaded env value to a stripped string for
    comparison. Tolerates int, bool, and quote-style variations."""
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v).strip().strip("'").strip('"')


# ─── Required system_env keys ────────────────────────────────────────
#
# Subset of the V1 17-key system_env that is functionally REQUIRED
# (not just a perf knob). Missing one of these typically means the
# container won't start correctly, will use wrong defaults, or will
# silently corrupt cache layout.
#
# The remaining V1 env keys (VLLM_USE_FLASHINFER_*, VLLM_MARLIN_*, etc.)
# are perf tunables — informational drift, not gating drift.

REQUIRED_ENV_KEYS: frozenset[str] = frozenset({
    # Allocator + threading correctness:
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    # Cache layout — must match the mount's container_path:
    "TRITON_CACHE_DIR",
    # Long-context contract — must be set to use the project's max_model_len:
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    # Multi-process worker correctness (TP>1):
    "VLLM_WORKER_MULTIPROC_METHOD",
    # Telemetry off (privacy + offline boots — project policy):
    "VLLM_NO_USAGE_STATS",
})


# ─── Audit result types ───────────────────────────────────────────────


@dataclass
class HardwareAudit:
    yaml_path: Path
    hardware_id: str
    mount_container_paths: set[str] = field(default_factory=set)
    env_keys: set[str] = field(default_factory=set)
    missing_mounts: list[str] = field(default_factory=list)
    missing_envs: list[str] = field(default_factory=list)
    # E24: env value violations — (key, got_value, expected_description)
    env_value_violations: list[tuple[str, str, str]] = field(default_factory=list)
    parse_error: str = ""

    @property
    def passed(self) -> bool:
        return (
            not self.parse_error
            and not self.missing_mounts
            and not self.missing_envs
            and not self.env_value_violations
        )


# ─── YAML walking ─────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


_MOUNT_PATH_RE = re.compile(r"^\s*\$\{[^}]+\}:(?P<container>[^:]+)(?::(?P<mode>ro|rw))?\s*$")


def _extract_mount_container_paths(mounts: list) -> set[str]:
    """Parse `${host}:/container/path[:mode]` strings; return the set of
    container paths declared. Unknown shapes are skipped silently — the
    gate is a coverage check, not a shape validator."""
    out: set[str] = set()
    for entry in mounts or []:
        if not isinstance(entry, str):
            continue
        s = entry.strip().strip('"').strip("'")
        # Strip a trailing comment if any (YAML loader usually strips it,
        # but be defensive).
        if "#" in s and ":" in s:
            s = s.split("#", 1)[0].strip()
        m = _MOUNT_PATH_RE.match(s)
        if m:
            out.add(m.group("container"))
            continue
        # Fallback: split on `:` and take field [1] if present.
        parts = s.split(":")
        if len(parts) >= 2:
            out.add(parts[1].strip())
    return out


def _walk_to(data: dict, dotted: str):
    cur = data
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def audit_one_hardware_yaml(path: Path) -> HardwareAudit:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return HardwareAudit(
            yaml_path=path,
            hardware_id="?",
            parse_error=f"YAML parse error: {e}",
        )

    hardware_id = data.get("id", path.stem)
    # Mounts live under `runtime.docker.mounts:` in the V2 hardware schema.
    mounts = _walk_to(data, "runtime.docker.mounts") or []
    mount_paths = _extract_mount_container_paths(mounts)
    env_block = data.get("system_env") or {}
    env_keys = set(env_block.keys()) if isinstance(env_block, dict) else set()

    missing_mounts = [
        slot.container_path
        for slot in REQUIRED_MOUNTS
        if slot.container_path not in mount_paths
    ]
    missing_envs = sorted(REQUIRED_ENV_KEYS - env_keys)

    # E24: value-level invariants. Only check keys that are actually
    # present — missing keys are reported via `missing_envs` already.
    env_value_violations: list[tuple[str, str, str]] = []
    if isinstance(env_block, dict):
        for key, expected_path in ENV_VALUE_LINKS.items():
            if key not in env_block:
                continue
            got = _normalize_env_value(env_block[key])
            if got != expected_path:
                env_value_violations.append(
                    (key, got,
                     f"must equal {expected_path!r} "
                     f"(linked to mount container_path)")
                )
        for key, accepted in ENV_VALUE_LITERALS.items():
            if key not in env_block:
                continue
            got = _normalize_env_value(env_block[key])
            if got not in accepted:
                env_value_violations.append(
                    (key, got,
                     f"must be one of {sorted(accepted)}")
                )

    return HardwareAudit(
        yaml_path=path,
        hardware_id=hardware_id,
        mount_container_paths=mount_paths,
        env_keys=env_keys,
        missing_mounts=missing_mounts,
        missing_envs=missing_envs,
        env_value_violations=env_value_violations,
    )


def audit_launch_coverage(hw_dir: Path = HW_DIR) -> list[HardwareAudit]:
    if not hw_dir.is_dir():
        return []
    return [audit_one_hardware_yaml(p) for p in sorted(hw_dir.glob("*.yaml"))]


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(results: list[HardwareAudit]) -> str:
    lines = []
    lines.append(f"audit-launch-coverage: {len(results)} V2 hardware YAML(s)")
    lines.append("─" * 70)
    for r in results:
        sym = "✓" if r.passed else "✗"
        lines.append(f"  {sym} {r.hardware_id}")
        if r.parse_error:
            lines.append(f"      ! {r.parse_error}")
        if r.missing_mounts:
            lines.append(
                f"      missing mounts ({len(r.missing_mounts)}):"
            )
            for cp in r.missing_mounts:
                slot = next(s for s in REQUIRED_MOUNTS if s.container_path == cp)
                tail = (
                    f"  — unless {slot.required_unless}"
                    if slot.required_unless else ""
                )
                lines.append(f"        - {cp} ({slot.mode}) — {slot.description}{tail}")
        if r.missing_envs:
            lines.append(
                f"      missing env keys: {r.missing_envs}"
            )
        if r.env_value_violations:
            lines.append(
                f"      env value violations ({len(r.env_value_violations)}):"
            )
            for key, got, why in r.env_value_violations:
                lines.append(f"        - {key}={got!r} — {why}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(
        f"  {passed}/{len(results)} hardware file(s) cover the canonical "
        f"launch schema "
        f"({len(REQUIRED_MOUNTS)} required mounts, "
        f"{len(REQUIRED_ENV_KEYS)} required env keys)"
    )
    if any(not r.passed for r in results):
        lines.append("")
        lines.append(
            "  ✗ Fix: add the missing mount/env entries to the YAML. "
            "See audit_launch_coverage.py:REQUIRED_MOUNTS for the canonical schema."
        )
    return "\n".join(lines)


def _render_json(results: list[HardwareAudit]) -> str:
    payload = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "required_mounts": [
            {
                "container_path": s.container_path,
                "mode": s.mode,
                "description": s.description,
                "required_unless": s.required_unless or None,
            }
            for s in REQUIRED_MOUNTS
        ],
        "required_env_keys": sorted(REQUIRED_ENV_KEYS),
        "env_value_links": ENV_VALUE_LINKS,
        "env_value_literals": {
            k: sorted(v) for k, v in ENV_VALUE_LITERALS.items()
        },
        "results": [
            {
                "yaml": _yaml_label(r.yaml_path),
                "hardware_id": r.hardware_id,
                "passed": r.passed,
                "missing_mounts": r.missing_mounts,
                "missing_envs": r.missing_envs,
                "env_value_violations": [
                    {"key": k, "got": g, "expected": w}
                    for (k, g, w) in r.env_value_violations
                ],
                "parse_error": r.parse_error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _yaml_label(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument("--hw-dir", default=None,
                    help="Override V2 hardware YAML directory.")
    args = ap.parse_args()

    hw_dir = Path(args.hw_dir) if args.hw_dir else HW_DIR
    results = audit_launch_coverage(hw_dir)

    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
