# SPDX-License-Identifier: Apache-2.0
"""Hardware-fit / requirements report for a model × hardware pairing.

This answers the operator question "can this model run on this rig?" using
ONLY the validated facts the V2 catalog actually carries:

  * model.requires  — min GPU count, min CUDA capability, min total VRAM,
                       rig-architecture blocklist (hard requirements, validated)
  * hardware.spec   — GPU count, per-GPU VRAM floor, CUDA capability
  * hardware.sizing — gpu_memory_utilization (the share vLLM is allowed to use)

The compatibility verdict is a HARD check (GPU count / CUDA capability /
arch blocklist) — those are deterministic from the catalog.

VRAM is reported as INFORMATIONAL only. ``min_vram_per_gpu_mib`` in the
hardware def is a conservative *matching floor*, not the card's real capacity,
so a precise "fits in X MiB" verdict would be fake-precision (a validated PROD
pairing can show a floor below the model minimum yet run fine on the real
24 GB cards). We surface both numbers and let the operator judge, rather than
emit a misleading red/green. See the user rule: no fake-precision estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FitCheck:
    id: str
    title: str
    ok: bool
    severity: str  # "ok" | "info" | "warning" | "blocked"
    detail: str


@dataclass(frozen=True)
class FitReport:
    model_id: str
    hardware_id: str
    model_title: str
    hardware_title: str
    compatible: bool
    checks: tuple[FitCheck, ...]
    vram: dict
    notes: tuple[str, ...] = field(default_factory=tuple)


def _cap_str(cap: Optional[tuple]) -> str:
    if not cap:
        return "any"
    return f"{cap[0]}.{cap[1]}"


def _cap_ge(have: Optional[tuple], need: Optional[tuple]) -> bool:
    """True if `have` CUDA capability is at least `need` (lexicographic)."""
    if not need:
        return True
    if not have:
        # Hardware does not declare a capability; cannot prove it meets the
        # requirement, so treat as a soft pass (informational elsewhere).
        return True
    return tuple(have) >= tuple(need)


def estimate_fit(*, model_id: str, hardware_id: str) -> FitReport:
    """Build a read-only hardware-fit report for a model × hardware pairing.

    Raises whatever ``load_model`` / ``load_hardware`` raise for unknown ids
    (the HTTP layer maps that to a 404).
    """
    from vllm.sndr_core.model_configs.registry_v2 import load_hardware, load_model

    model = load_model(model_id)
    hw = load_hardware(hardware_id)

    req = model.requires
    spec = hw.hardware
    util = hw.sizing.gpu_memory_utilization

    checks: list[FitCheck] = []

    # --- GPU count (hard) ---------------------------------------------------
    gpu_ok = spec.n_gpus >= req.min_gpu_count
    checks.append(
        FitCheck(
            id="gpu_count",
            title="GPU count",
            ok=gpu_ok,
            severity="ok" if gpu_ok else "blocked",
            detail=(
                f"rig has {spec.n_gpus} GPU(s); model requires "
                f"{req.min_gpu_count}"
                + ("" if gpu_ok else " — not enough GPUs")
            ),
        )
    )

    # --- CUDA capability (hard, soft-pass when undeclared) ------------------
    have_cap = spec.cuda_capability_min
    need_cap = req.min_cuda_capability
    cap_ok = _cap_ge(have_cap, need_cap)
    cap_undeclared = bool(need_cap) and not have_cap
    checks.append(
        FitCheck(
            id="cuda",
            title="CUDA capability",
            ok=cap_ok,
            severity="info" if cap_undeclared else ("ok" if cap_ok else "blocked"),
            detail=(
                f"rig {_cap_str(have_cap)} vs model minimum {_cap_str(need_cap)}"
                + (
                    " — rig capability undeclared, cannot verify"
                    if cap_undeclared
                    else ("" if cap_ok else " — below model minimum")
                )
            ),
        )
    )

    # --- Architecture blocklist (hard) -------------------------------------
    blocked_arch = hardware_id in req.rig_arch_blocklist
    if req.rig_arch_blocklist:
        checks.append(
            FitCheck(
                id="arch_blocklist",
                title="Rig architecture",
                ok=not blocked_arch,
                severity="blocked" if blocked_arch else "ok",
                detail=(
                    f"this rig is on the model's blocklist: "
                    f"{', '.join(req.rig_arch_blocklist)}"
                    if blocked_arch
                    else f"rig not blocklisted ({len(req.rig_arch_blocklist)} entr"
                    f"{'y' if len(req.rig_arch_blocklist) == 1 else 'ies'})"
                ),
            )
        )

    # --- VRAM (informational only) -----------------------------------------
    rig_floor = int(spec.n_gpus * spec.min_vram_per_gpu_mib * util)
    model_min = req.min_total_vram_mib
    vram_headroom = rig_floor - model_min
    vram_ok = model_min == 0 or rig_floor >= model_min
    checks.append(
        FitCheck(
            id="vram_floor",
            title="VRAM (informational)",
            ok=vram_ok,
            severity="info" if vram_ok else "warning",
            detail=(
                f"model minimum {model_min:,} MiB vs rig usable floor "
                f"{rig_floor:,} MiB "
                f"({spec.n_gpus}×{spec.min_vram_per_gpu_mib:,} MiB × "
                f"{util:.0%} util)"
                + (
                    ""
                    if vram_ok
                    else " — floor below minimum; the matching floor is "
                    "conservative, real card VRAM may still be sufficient"
                )
            ),
        )
    )

    # Compatibility = all HARD checks pass. The VRAM check is informational
    # and never flips the verdict (its floor is a conservative threshold).
    hard = [c for c in checks if c.id != "vram_floor"]
    compatible = all(c.ok for c in hard)

    notes: list[str] = [
        "Compatibility is judged on GPU count, CUDA capability and the rig "
        "blocklist — the deterministic requirements in the catalog.",
        "VRAM is informational: min_vram_per_gpu_mib is a conservative matching "
        "floor, not the card's real capacity, so it does not flip the verdict.",
    ]
    if not compatible:
        notes.append(
            "Blocked checks must be resolved (more GPUs / a newer GPU "
            "architecture) before this pairing can launch."
        )

    vram = {
        "model_min_mib": model_min,
        "rig_floor_mib": rig_floor,
        "headroom_mib": vram_headroom,
        "n_gpus": spec.n_gpus,
        "vram_per_gpu_mib": spec.min_vram_per_gpu_mib,
        "gpu_memory_utilization": util,
        "kv_cache_dtype": model.capabilities.kv_cache_dtype or "auto",
    }

    return FitReport(
        model_id=model_id,
        hardware_id=hardware_id,
        model_title=model.title,
        hardware_title=hw.title,
        compatible=compatible,
        checks=tuple(checks),
        vram=vram,
        notes=tuple(notes),
    )
