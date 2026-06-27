# SPDX-License-Identifier: Apache-2.0
"""Preset hardware fit-check — "can this rig run this preset" BEFORE boot.

This is the Genesis analogue of club-3090's ``scripts/preflight.sh``
``preflight_compose_hardware`` (noonghunna/club-3090@master), adapted to our
YAML-preset model. club-3090 reads the per-compose header trailers
(``# Requires-min-vram-gb`` / ``# Tensor-parallel`` / ``# Requires-min-gpu-count``
/ ``# Requires-sm``) via a bash regex and projects VRAM/GPU-count/SM fit
against ``nvidia-smi`` before ``docker compose up``. Here the same projection
runs against:

  - the typed ``card.hardware_fit`` envelope (preferred — the operator-declared
    requirement, see ``preset_schema.HardwareFit``), OR
  - the composed hardware definition (``cfg.hardware``) as a fallback for
    presets that don't yet declare a fit block.

Extends club-3090's check:

  - ENGINE-PIN dimension. club-3090's trailers stop at VRAM/GPU/SM; we also
    project the validated vLLM pin (``engine_pin`` / ``vllm_pin_required``)
    against the rig's pinned image so a pin-drifted rig is caught pre-boot,
    not at a cryptic engine-init crash.
  - Structured verdict object. The check returns a typed
    :class:`FitReport` (per-dimension :class:`FitCheck` rows) so the CLI can
    render text OR JSON and CI can assert on it — club-3090's is print-only.

Side effects (``nvidia-smi``) live behind :class:`RigProbe` so the fit logic
is unit-testable with a fake rig (the club-3090 ``CLUB3090_FAKE_GPUS`` pattern).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ─── Rig model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectedGpu:
    """One GPU as seen on the rig (or a synthetic --fake-gpus entry)."""
    index: int
    name: str
    vram_mib: int
    compute_cap: Optional[tuple[int, int]]  # (major, minor); None if unknown


@dataclass(frozen=True)
class Rig:
    """The hardware the preset is being projected against."""
    gpus: list[DetectedGpu]
    source: str  # "nvidia-smi" | "rig:<hardware_id>" | "fake"

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def min_vram_gb(self) -> Optional[int]:
        """Smallest card's VRAM in whole GB (floor) — the binding constraint
        for a TP launch (the engine sizes the per-card pool to the weakest
        card). None when no GPUs."""
        if not self.gpus:
            return None
        return min(g.vram_mib for g in self.gpus) // 1024

    @property
    def min_compute_cap(self) -> Optional[tuple[int, int]]:
        caps = [g.compute_cap for g in self.gpus if g.compute_cap is not None]
        return min(caps) if caps else None


# ─── Required envelope (resolved from the preset) ───────────────────────────


@dataclass(frozen=True)
class RequiredEnvelope:
    """What the preset needs from a rig. Built from ``card.hardware_fit`` when
    present, else from the composed hardware definition."""
    requires_min_vram_gb: Optional[int]
    requires_min_gpu_count: Optional[int]
    tensor_parallel: Optional[int]
    requires_min_cuda_capability: Optional[tuple[int, int]]
    engine_pin: Optional[str]
    source: str  # "card.hardware_fit" | "composed_hardware"


def resolve_required_envelope(cfg, preset_def) -> RequiredEnvelope:
    """Resolve the requirement envelope for a preset.

    Prefers the typed ``card.hardware_fit`` block (the operator-declared,
    audit-cross-validated envelope). Falls back to the composed hardware
    when the preset has no fit block, so ``sndr preflight`` still works on
    un-annotated presets (parity with club-3090, which always reads the
    compose body).
    """
    card = getattr(preset_def, "card", None)
    fit = getattr(card, "hardware_fit", None) if card is not None else None
    hw = getattr(cfg, "hardware", None)
    pin = getattr(cfg, "vllm_pin_required", None)

    if fit is not None:
        cc = fit.requires_min_cuda_capability
        return RequiredEnvelope(
            requires_min_vram_gb=fit.requires_min_vram_gb,
            requires_min_gpu_count=fit.requires_min_gpu_count,
            tensor_parallel=fit.tensor_parallel,
            requires_min_cuda_capability=tuple(cc) if cc is not None else None,
            engine_pin=fit.engine_pin or pin,
            source="card.hardware_fit",
        )

    # Fallback: derive from the composed hardware definition.
    n_gpus = int(getattr(hw, "n_gpus", 0) or 0) if hw is not None else 0
    per_gpu_mib = int(getattr(hw, "min_vram_per_gpu_mib", 0) or 0) if hw is not None else 0
    cc_hw = getattr(hw, "cuda_capability_min", None) if hw is not None else None
    return RequiredEnvelope(
        requires_min_vram_gb=(per_gpu_mib // 1024) if per_gpu_mib else None,
        requires_min_gpu_count=n_gpus or None,
        tensor_parallel=n_gpus or None,
        requires_min_cuda_capability=tuple(cc_hw) if cc_hw is not None else None,
        engine_pin=pin,
        source="composed_hardware",
    )


# ─── Verdict model ──────────────────────────────────────────────────────────


@dataclass
class FitCheck:
    """One dimension of the fit projection."""
    dimension: str          # "gpu_count" | "vram" | "cuda_capability" | "engine_pin"
    status: str             # "pass" | "fail" | "warn" | "skip"
    required: str           # human-readable requirement
    detected: str           # human-readable observed value
    message: str            # one-line explanation


@dataclass
class FitReport:
    preset_id: str
    rig_source: str
    envelope_source: str
    checks: list[FitCheck] = field(default_factory=list)

    def add(self, dimension, status, required, detected, message) -> None:
        self.checks.append(FitCheck(dimension, status, required, detected, message))

    @property
    def can_run(self) -> bool:
        """True if no hard FAILs. WARN/SKIP do not block (operator may know
        better — e.g. a sub-floor card with tuned gpu-memory-utilization,
        exactly club-3090's TP>=2 soft-VRAM stance)."""
        return not any(c.status == "fail" for c in self.checks)

    @property
    def verdict(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "CANNOT RUN"
        if any(c.status == "warn" for c in self.checks):
            return "RUNNABLE (with warnings)"
        return "CAN RUN"


# ─── The projection (pure — no I/O) ─────────────────────────────────────────


def evaluate_fit(preset_id: str, env: RequiredEnvelope, rig: Rig) -> FitReport:
    """Project the required envelope against a rig. Pure function.

    Mirrors club-3090 preflight_compose_hardware semantics:
      - GPU count below the TP requirement is a hard FAIL (TP can't init).
      - SM/compute-capability below the floor is a hard FAIL (kernels gated).
      - VRAM below the floor is a WARN for multi-GPU TP rigs (tuned mem-util
        may still fit) but a FAIL for single-GPU (no headroom to trade).
      - Engine pin mismatch is a WARN (re-tag or re-pin; not a hard block).
    """
    report = FitReport(
        preset_id=preset_id,
        rig_source=rig.source,
        envelope_source=env.source,
    )

    # ── GPU count ──
    need_gpus = env.requires_min_gpu_count or env.tensor_parallel
    if need_gpus is not None:
        if rig.gpu_count >= need_gpus:
            report.add(
                "gpu_count", "pass",
                f">= {need_gpus} GPU(s)", f"{rig.gpu_count} GPU(s)",
                f"rig has {rig.gpu_count} GPU(s); preset needs {need_gpus} for "
                f"TP={env.tensor_parallel or need_gpus}",
            )
        else:
            report.add(
                "gpu_count", "fail",
                f">= {need_gpus} GPU(s)", f"{rig.gpu_count} GPU(s)",
                f"only {rig.gpu_count} GPU(s) visible; TP="
                f"{env.tensor_parallel or need_gpus} needs {need_gpus} — vLLM "
                f"will fail at init (pick a lower-TP preset or add a GPU)",
            )
    else:
        report.add(
            "gpu_count", "skip", "unspecified", f"{rig.gpu_count} GPU(s)",
            "preset declares no GPU-count requirement",
        )

    # ── VRAM floor (per card) ──
    if env.requires_min_vram_gb is not None:
        rig_vram = rig.min_vram_gb
        if rig_vram is None:
            report.add(
                "vram", "skip",
                f">= {env.requires_min_vram_gb} GB/GPU", "unknown",
                "no GPUs detected to measure VRAM",
            )
        elif rig_vram >= env.requires_min_vram_gb:
            report.add(
                "vram", "pass",
                f">= {env.requires_min_vram_gb} GB/GPU", f"{rig_vram} GB/GPU",
                f"smallest card has {rig_vram} GB; preset floor is "
                f"{env.requires_min_vram_gb} GB",
            )
        else:
            # Single-GPU → FAIL (no second card to absorb the deficit).
            # Multi-GPU TP → WARN (club-3090's tuned-mem-util escape hatch).
            multi_gpu = (env.tensor_parallel or 1) >= 2 and rig.gpu_count >= 2
            status = "warn" if multi_gpu else "fail"
            tail = (
                "TP>=2 rigs may still fit with tuned gpu-memory-utilization / "
                "smaller KV — proceed with care"
                if multi_gpu else
                "single card has no headroom to trade — this preset will OOM"
            )
            report.add(
                "vram", status,
                f">= {env.requires_min_vram_gb} GB/GPU", f"{rig_vram} GB/GPU",
                f"smallest card has {rig_vram} GB, below the "
                f"{env.requires_min_vram_gb} GB floor; {tail}",
            )
    else:
        report.add(
            "vram", "skip", "unspecified",
            f"{rig.min_vram_gb} GB/GPU" if rig.min_vram_gb else "unknown",
            "preset declares no VRAM floor",
        )

    # ── CUDA capability (SM floor) ──
    if env.requires_min_cuda_capability is not None:
        rig_cc = rig.min_compute_cap
        need = env.requires_min_cuda_capability
        if rig_cc is None:
            report.add(
                "cuda_capability", "skip",
                f"sm_{need[0]}.{need[1]}+", "unknown",
                "could not read compute capability for the rig",
            )
        elif rig_cc >= need:
            report.add(
                "cuda_capability", "pass",
                f"sm_{need[0]}.{need[1]}+", f"sm_{rig_cc[0]}.{rig_cc[1]}",
                f"rig is sm_{rig_cc[0]}.{rig_cc[1]}; preset floor is "
                f"sm_{need[0]}.{need[1]}",
            )
        else:
            report.add(
                "cuda_capability", "fail",
                f"sm_{need[0]}.{need[1]}+", f"sm_{rig_cc[0]}.{rig_cc[1]}",
                f"rig is sm_{rig_cc[0]}.{rig_cc[1]}, below the required "
                f"sm_{need[0]}.{need[1]} — SM-gated kernels won't compile",
            )
    else:
        report.add(
            "cuda_capability", "skip", "unspecified",
            (f"sm_{rig.min_compute_cap[0]}.{rig.min_compute_cap[1]}"
             if rig.min_compute_cap else "unknown"),
            "preset declares no compute-capability floor",
        )

    # ── Projected VRAM (byte-level — the Genesis kv_projector extension) ──
    # Additive: this row is appended by ``add_projection_check`` AFTER the
    # envelope rows when the preset's model carries a byte-level shape. The
    # envelope checks above answer "clears the declared floor?"; this row
    # answers "given THIS ctx/kv-format/seqs/tp, what's the per-card GB and
    # will it OOM?". It is wired by the CLI / route, not here, so ``evaluate_fit``
    # stays pure and shape-free (a preset without a shape simply skips the row).

    # ── Engine pin (the Genesis extension beyond club-3090's trailers) ──
    if env.engine_pin is not None:
        report.add(
            "engine_pin", "warn" if rig.source == "nvidia-smi" else "skip",
            env.engine_pin,
            "rig image not inspected"
            if rig.source == "nvidia-smi" else "n/a (offline rig)",
            f"preset was validated on vLLM {env.engine_pin}; verify the rig's "
            f"pinned image matches before boot (`docker images vllm/vllm-openai`)"
            if rig.source == "nvidia-smi" else
            f"preset validated on vLLM {env.engine_pin} (no live image to check)",
        )
    else:
        report.add(
            "engine_pin", "skip", "unspecified", "n/a",
            "preset declares no engine pin",
        )

    return report


# ─── Rig probes (side effects live here) ────────────────────────────────────


class RigProbe:
    """Detect the live rig via ``nvidia-smi``. Subclass/override for tests.

    ``timeout`` bounds the ``nvidia-smi`` subprocess so a wedged driver / busy
    GPU (e.g. mid-load of a 35B engine) can never block the caller longer than
    the cap — on timeout we degrade to an empty rig (the fit-check then SKIPs
    the rig-dependent dimensions rather than hanging). Default 5s keeps the GUI
    pre-launch probe snappy; the CLI may pass a larger value if it wants more
    patience on a cold rig.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def detect(self) -> Rig:
        try:
            res = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return Rig(gpus=[], source="nvidia-smi")
        if res.returncode != 0:
            return Rig(gpus=[], source="nvidia-smi")
        return Rig(gpus=parse_nvidia_smi_csv(res.stdout), source="nvidia-smi")


def parse_nvidia_smi_csv(text: str) -> list[DetectedGpu]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,compute_cap
    --format=csv,noheader,nounits`` output into DetectedGpu rows."""
    gpus: list[DetectedGpu] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        try:
            vram_mib = int(float(parts[2]))
        except ValueError:
            vram_mib = 0
        cc = _parse_compute_cap(parts[3])
        gpus.append(DetectedGpu(index=idx, name=name, vram_mib=vram_mib, compute_cap=cc))
    return gpus


def _parse_compute_cap(s: str) -> Optional[tuple[int, int]]:
    """Parse a compute-capability string like '8.6' into (8, 6)."""
    s = s.strip()
    if not s:
        return None
    if "." in s:
        major, _, minor = s.partition(".")
    else:
        major, minor = s, "0"
    try:
        return (int(major), int(minor))
    except ValueError:
        return None


def rig_from_hardware_def(hw_def, source: str) -> Rig:
    """Build a synthetic Rig from a builtin hardware definition (for the
    ``--rig <hardware_id>`` flag — projects a preset against a DIFFERENT
    declared rig than the one it composes onto)."""
    hw = hw_def.hardware
    n = int(hw.n_gpus or 0)
    per_gpu_mib = int(hw.min_vram_per_gpu_mib or 0)
    cc = tuple(hw.cuda_capability_min) if hw.cuda_capability_min else None
    name = (hw.gpu_match_keys[0] if hw.gpu_match_keys else "gpu")
    gpus = [
        DetectedGpu(index=i, name=name, vram_mib=per_gpu_mib, compute_cap=cc)
        for i in range(n)
    ]
    return Rig(gpus=gpus, source=source)


def rig_from_fake_spec(spec: str) -> Rig:
    """Parse a ``--fake-gpus`` spec (club-3090 ``CLUB3090_FAKE_GPUS`` style):

        "name:vram_mib:cc;name:vram_mib:cc"   e.g.
        "RTX 3090:24576:8.6"       (one 3090)
        "RTX A5000:24564:8.6;RTX A5000:24564:8.6"  (two A5000)
    """
    gpus: list[DetectedGpu] = []
    for i, entry in enumerate(spec.split(";")):
        entry = entry.strip()
        if not entry:
            continue
        fields = [f.strip() for f in entry.split(":")]
        name = fields[0] if fields else f"gpu{i}"
        vram_mib = int(float(fields[1])) if len(fields) > 1 and fields[1] else 0
        cc = _parse_compute_cap(fields[2]) if len(fields) > 2 else None
        gpus.append(DetectedGpu(index=i, name=name, vram_mib=vram_mib, compute_cap=cc))
    return Rig(gpus=gpus, source="fake")


# ─── Projected-VRAM row (byte-level, additive over the envelope) ─────────────


def add_projection_check(report: FitReport, projection, *, vram_is_measured: bool = True) -> None:
    """Append a ``projected_vram`` :class:`FitCheck` row to ``report`` from a
    :class:`kv_projector.Projection`.

    Strictly additive: the envelope rows produced by :func:`evaluate_fit` are
    untouched. The projection row's status maps the byte-level verdict onto the
    envelope's pass/warn/fail vocabulary so ``FitReport.verdict`` stays coherent:

      - PASS  → "pass"  (fits with headroom)
      - TIGHT → "warn"  (boots, but vLLM caps the KV pool — does NOT block)
      - FAIL  → "fail"  (fixed footprint alone overflows — won't boot)

    A TIGHT projection deliberately does NOT flip ``can_run`` to False (it boots),
    matching the sub-floor-VRAM WARN stance the envelope already takes.

    ``vram_is_measured`` distinguishes a real per-card VRAM (live nvidia-smi,
    --fake-gpus, --card — the card's PHYSICAL size) from a builtin hardware
    definition's DECLARED ``min_vram_per_gpu_mib`` floor (intentionally
    conservative, e.g. 22000 MiB for a 24564 MiB A5000). A FAIL projected
    against a declared floor is downgraded to WARN — the floor is a lower bound,
    not the card's real capacity, so it must not hard-block a config that runs
    on the physical card. A FAIL against a measured card IS a hard block (a real
    byte-level OOM, stronger than the declared-floor envelope).
    """
    verdict = projection.verdict
    if verdict == "FAIL" and not vram_is_measured:
        status = "warn"
    else:
        status = {"PASS": "pass", "TIGHT": "warn", "FAIL": "fail"}.get(verdict, "skip")
    prov = " [PROVISIONAL calibration]" if projection.provisional else ""
    required = f"<= {projection.budget_gib:.1f} GiB/card (util {projection.mem_util})"
    detected = (
        f"{projection.total_gib:.1f} GiB/card "
        f"(weights {projection.weights_gib:.1f} + KV {projection.kv_pool_actual_gib:.1f} "
        f"+ overhead {projection.fixed_gib - projection.weights_gib:.1f})"
    )
    if projection.verdict == "PASS":
        msg = (
            f"projected {projection.total_gib:.1f} GiB/card at ctx="
            f"{projection.ctx:,} / seqs={projection.max_num_seqs} / "
            f"{projection.kv_format} — {projection.headroom_gib:.1f} GiB headroom{prov}"
        )
    elif projection.verdict == "TIGHT":
        msg = (
            f"projected {projection.total_gib:.1f} GiB/card — fixed footprint "
            f"leaves only {projection.available_for_kv_gib:.1f} GiB for KV; vLLM "
            f"will cap the pool (effective concurrency may drop below "
            f"{projection.max_num_seqs} at ctx={projection.ctx:,}){prov}"
        )
    elif projection.verdict == "FAIL":
        msg = (
            f"projected {projection.fixed_gib:.1f} GiB fixed footprint exceeds the "
            f"{projection.budget_gib:.1f} GiB budget at ctx={projection.ctx:,} — "
            f"vLLM will OOM at boot; lower max_model_len or kv-format{prov}"
        )
    else:  # pragma: no cover — defensive
        msg = f"projection verdict {projection.verdict!r}{prov}"

    report.add("projected_vram", status, required, detected, msg)


def resolve_model_shape(preset_def):
    """Resolve the byte-level ``ModelShape`` for a preset def (the I/O boundary).

    Returns ``None`` when the model declares no shape block (the projection row
    is then skipped — envelope-only, exactly as before). Kept here so both the
    CLI command and the HTTP route share one resolver and never diverge.
    """
    model_id = getattr(preset_def, "model", None)
    if not model_id:
        return None
    try:
        from sndr.model_configs.registry_v2 import load_model
        model_def = load_model(model_id)
    except Exception:  # noqa: BLE001 — missing/broken model → skip the row
        return None
    caps = getattr(model_def, "capabilities", None)
    return getattr(caps, "shape", None) if caps else None


__all__ = [
    "DetectedGpu", "Rig", "RequiredEnvelope", "FitCheck", "FitReport",
    "resolve_required_envelope", "evaluate_fit",
    "RigProbe", "parse_nvidia_smi_csv",
    "rig_from_hardware_def", "rig_from_fake_spec",
    "add_projection_check", "resolve_model_shape",
]
