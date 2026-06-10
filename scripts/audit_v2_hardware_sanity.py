#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 hardware sanity gate.

Catches typos / nonsensical values in V2 hardware YAML numeric fields
that the V1→V2 migration could plausibly mangle:

  • `hardware.cuda_capability_min` — must be a `[major, minor]` pair
    of small non-negative ints (CUDA SM versions: 6.0..12.0 range).
  • `hardware.n_gpus` — positive int (1..16 sanity bound).
  • `hardware.min_vram_per_gpu_mib` — positive int, ≥ 8000 (8 GiB
    minimum; anything smaller can't run a useful Qwen-class model).
  • `sizing.gpu_memory_utilization` — float in (0.0, 1.0]. Genesis
    rejects 0 (degenerate) and > 1.0 (impossible).
  • `sizing.max_num_seqs` — positive int.
  • `sizing.max_num_batched_tokens` — positive int.
  • Cross-field: `gpu_memory_utilization * min_vram_per_gpu_mib`
    must yield ≥ 4 GiB usable KV budget (lower = won't fit any model).

Exit codes:
  0 — every V2 hardware YAML passes all sanity checks
  1 — at least one violation
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


# Bounds. Adjust here if a new GPU class lands.
CUDA_MAJOR_RANGE = (6, 12)     # SM 6.0 (Pascal) .. SM 12.0 (future Hopper++)
CUDA_MINOR_RANGE = (0, 9)
N_GPUS_RANGE     = (1, 16)
MIN_VRAM_MIB_MIN = 8_000        # 8 GiB lower bound
GMU_RANGE        = (0.0, 1.0)   # exclusive 0, inclusive 1.0
MAX_NUM_SEQS_MIN = 1
MAX_BATCH_TOK_MIN = 256
MIN_USABLE_VRAM_MIB = 4_000     # cross-field: gmu * vram_mib >= 4 GiB


@dataclass
class SanityCheck:
    path: Path
    hardware_id: str
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _check_int(label: str, val, lo: int, hi: int,
               violations: list[str]) -> None:
    if not isinstance(val, int) or isinstance(val, bool):
        violations.append(f"{label}={val!r} not int")
        return
    if val < lo or val > hi:
        violations.append(f"{label}={val} out of [{lo}..{hi}]")


def _check_int_min(label: str, val, lo: int,
                   violations: list[str]) -> None:
    if not isinstance(val, int) or isinstance(val, bool):
        violations.append(f"{label}={val!r} not int")
        return
    if val < lo:
        violations.append(f"{label}={val} < {lo}")


def check_one_hardware(path: Path) -> SanityCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return SanityCheck(
            path=path, hardware_id="?",
            error=f"YAML parse error: {e}",
        )

    hw = data.get("hardware") or {}
    sz = data.get("sizing") or {}
    hardware_id = data.get("id", path.stem)
    v: list[str] = []

    # cuda_capability_min: [major, minor]
    cc = hw.get("cuda_capability_min")
    if not isinstance(cc, list) or len(cc) != 2:
        v.append(f"cuda_capability_min={cc!r} not [major, minor] list")
    else:
        _check_int(
            "cuda_capability_min[0]", cc[0],
            CUDA_MAJOR_RANGE[0], CUDA_MAJOR_RANGE[1], v,
        )
        _check_int(
            "cuda_capability_min[1]", cc[1],
            CUDA_MINOR_RANGE[0], CUDA_MINOR_RANGE[1], v,
        )

    # n_gpus
    n_gpus = hw.get("n_gpus")
    _check_int("n_gpus", n_gpus, N_GPUS_RANGE[0], N_GPUS_RANGE[1], v)

    # min_vram_per_gpu_mib
    vram = hw.get("min_vram_per_gpu_mib")
    _check_int_min("min_vram_per_gpu_mib", vram, MIN_VRAM_MIB_MIN, v)

    # gpu_memory_utilization (0.0, 1.0]
    gmu = sz.get("gpu_memory_utilization")
    if not isinstance(gmu, (int, float)) or isinstance(gmu, bool):
        v.append(f"gpu_memory_utilization={gmu!r} not numeric")
    else:
        gmu_f = float(gmu)
        if gmu_f <= GMU_RANGE[0] or gmu_f > GMU_RANGE[1]:
            v.append(
                f"gpu_memory_utilization={gmu_f} outside ({GMU_RANGE[0]}, {GMU_RANGE[1]}]"
            )

    # max_num_seqs
    _check_int_min(
        "sizing.max_num_seqs",
        sz.get("max_num_seqs"), MAX_NUM_SEQS_MIN, v,
    )

    # max_num_batched_tokens
    _check_int_min(
        "sizing.max_num_batched_tokens",
        sz.get("max_num_batched_tokens"), MAX_BATCH_TOK_MIN, v,
    )

    # Cross-field: usable VRAM after utilization
    if (
        isinstance(vram, int)
        and isinstance(gmu, (int, float))
        and not isinstance(gmu, bool)
    ):
        usable = float(gmu) * vram
        if usable < MIN_USABLE_VRAM_MIB:
            v.append(
                f"usable VRAM = gmu({gmu}) * vram({vram} MiB) = "
                f"{usable:.0f} MiB < {MIN_USABLE_VRAM_MIB}"
            )

    return SanityCheck(
        path=path, hardware_id=hardware_id, violations=v,
    )


def audit_v2_hardware_sanity(
    hw_dir: Path = HARDWARE_DIR,
) -> list[SanityCheck]:
    if not hw_dir.is_dir():
        return []
    return [check_one_hardware(p) for p in sorted(hw_dir.glob("*.yaml"))]


def _render_text(results: list[SanityCheck]) -> str:
    lines = [
        f"audit-v2-hardware-sanity: {len(results)} V2 hardware YAML(s)",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.hardware_id}: {r.error}")
            continue
        lines.append(f"  {sym} {r.hardware_id}")
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} hardware files clean")
    return "\n".join(lines)


def _render_json(results: list[SanityCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "bounds": {
            "cuda_major":  list(CUDA_MAJOR_RANGE),
            "cuda_minor":  list(CUDA_MINOR_RANGE),
            "n_gpus":      list(N_GPUS_RANGE),
            "vram_mib_min": MIN_VRAM_MIB_MIN,
            "gmu_range":   list(GMU_RANGE),
            "min_usable_vram_mib": MIN_USABLE_VRAM_MIB,
        },
        "results": [
            {
                "hardware_id": r.hardware_id,
                "path": _rel(r.path),
                "passed": r.passed,
                "violations": r.violations,
                "error": r.error or None,
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit_v2_hardware_sanity()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
