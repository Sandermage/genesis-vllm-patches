# SPDX-License-Identifier: Apache-2.0
"""Byte-level KV / VRAM projector — "given THIS ctx / kv-format / max-num-seqs /
tp, what's my ACTUAL per-card GB and will it OOM?".

This is the Genesis analogue of club-3090's ``tools/kv-calc.py`` (the math
STRUCTURE is borrowed — per-card weights + growing-KV pool + recurrent state +
activation peak + cudagraph overhead + drafter, with the vLLM "KV pool fills the
budget" capping behavior and a PASS/TIGHT/FAIL verdict). The per-family
calibration coefficients are OUR OWN, derived from OUR measured reality on
2× A5000 24 GB, vLLM pin dev424 — NOT copied from club-3090.

Why this module exists
──────────────────────
``preflight_fit.py`` answers the ENVELOPE question ("does my rig clear the
declared min-VRAM floor + SM + GPU count?"). It does NOT answer the BYTE-LEVEL
question. Before this module, Genesis had zero byte-level fit math: no
``kv_pool_per_card`` / ``solve_max_ctx`` / ``project`` anywhere. ``kv_calc.py``
(the tool) and ``memory_estimator.py`` read ``config.json`` + safetensors from
DISK — they are I/O-bound and can't run from a typed preset alone. This module
is PURE and I/O-free: it drives entirely off the ``ModelShape`` dims declared on
the preset (``capabilities.shape`` in schema_v2) plus the rig VRAM, so the GUI /
CLI can project a fit before anything touches the host.

The math (per card, after TP split)
────────────────────────────────────
    total_gb = weights_gb               # resident weights ÷ TP
             + kv_pool_gb              # growing KV pool (attention layers only)
             + recurrent_state_gb      # GDN/Mamba fixed state (hybrid models)
             + activation_gb           # prefill activation peak (ctx-linear)
             + cudagraph_overhead_gb   # capture + workspace
             + drafter_gb              # MTP / DFlash adder

vLLM sizes the KV pool to fill ``mem_util × VRAM`` minus the fixed components,
so when the *requested* pool (max_ctx × max_num_seqs) exceeds the available
slack the verdict is TIGHT (vLLM caps the pool — effective concurrency drops),
not FAIL. FAIL is reserved for "the fixed footprint alone leaves no room for
even one max_ctx sequence" (a real refuse-to-boot).

Calibration (see docs/KV_PROJECTOR.md for the full derivation)
──────────────────────────────────────────────────────────────
- **35B-A3B FP8 TQ-k8v4 — CALIBRATED (strong anchor).** The dev424 PN403
  investigation captured LIVE engine telemetry at max-model-len=280000, TP=2,
  k8v4, mem-util=0.9: ``kv_cache_size_tokens=388620``, ``num_gpu_blocks=161``.
  The 10-full-attention × 2-KV-head × 128-head_dim × k8v4 per-token formula
  reproduces a 388,672-token pool capacity = 0.01% off the live 388,620. The
  fixed-footprint coefficients (activation, recurrent, overhead, MTP) are tuned
  so the projector's available-for-KV at that operating point matches the live
  pool within ~1%.
- **27B int4 AutoRound TQ-k8v4 — PROVISIONAL.** No captured ``num_gpu_blocks``
  anchor exists; only the docs/HARDWARE.md residency table + context formula.
  ``project()`` flags this model's ``Projection.provisional=True`` so callers
  surface the lower confidence honestly. The framework is identical; only the
  confidence label differs.

Public surface
──────────────
``kv_pool_per_card_bytes(shape, kv_format, ctx, max_num_seqs, tp, mtp_n=0)``
``weights_per_card_bytes(shape, tp)``
``recurrent_state_per_card_bytes(shape, max_num_seqs, tp)``
``activation_peak_per_card_bytes(shape, kv_format, ctx, tp)``
``solve_max_ctx(shape, kv_format, max_num_seqs, tp, mem_util, vram_gb, ...)``
``project(preset, rig) -> Projection``
``fit_verdict(projection) -> "PASS" | "TIGHT" | "FAIL"``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ─── Units ──────────────────────────────────────────────────────────────────

_GIB = 1024 ** 3


# ─── KV format → bytes per element ──────────────────────────────────────────
# One element = one head-dim slot of one KV head. K and V counted separately at
# the call sites (×2 factor) for models that store them independently. Our
# formats: bf16/fp16 = 2 B, fp8 = 1 B, TurboQuant k8v4 = 0.75 B (8-bit K +
# 4-bit V, averaged — Genesis turbo-quant KV), int8 = 1 B, int4 = 0.5 B.
KV_FORMAT_BYTES: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "auto": 2.0,             # conservative: assume full-precision KV
    "fp8": 1.0,
    "fp8_e5m2": 1.0,
    "fp8_e4m3": 1.0,
    "int8": 1.0,
    "turboquant_k8v4": 0.75,
    "tq_k8v4": 0.75,         # alias used by product_api kv_math
    "k8v4": 0.75,           # short alias
    "int4": 0.5,
}

#: Default KV format when a preset declares ``kv_cache_dtype: null``. vLLM keeps
#: full-precision (bf16) KV in that case, so budgeting must assume the heaviest.
DEFAULT_KV_FORMAT = "fp16"


def kv_format_bytes(kv_format: Optional[str]) -> float:
    """Bytes per KV element for ``kv_format`` (falls back to bf16 = 2 B)."""
    if not kv_format:
        return KV_FORMAT_BYTES[DEFAULT_KV_FORMAT]
    return KV_FORMAT_BYTES.get(str(kv_format).lower(), KV_FORMAT_BYTES[DEFAULT_KV_FORMAT])


# ─── Calibration coefficients (OUR measured reality — see module docstring) ──
#
# Cudagraph + workspace overhead per card (GiB). Roughly linear in mem_util
# (higher util → more capture sizes retained) plus a per-extra-rank NCCL bump.
# Anchored so the 35B PN403 fixed-footprint reconstruction lands on the live
# 388,620-token pool capacity at TP=2 / util 0.9.
_OVERHEAD_BASE_GIB = 0.5
_OVERHEAD_PER_UTIL_GIB = 1.0
_OVERHEAD_PER_EXTRA_RANK_GIB = 0.3

# GDN / Mamba recurrent state — bytes per (recurrent-layer × hidden-unit ×
# stream), bf16-ish state with PN59 streaming-GDN keeping it compact. Small.
_RECUR_COEF_BYTES = 320.0

# Activation peak per (recurrent-layer × token), pre-TP. GDN block-state
# materialization (PerfMamba O(γ·D·N·L) form) scales linearly with context.
# CALIBRATED on the 35B PN403 anchor (see docs/KV_PROJECTOR.md): solving the
# fixed-footprint reconstruction at 280K gives 363.76 B/layer/token.
_ACTIVATION_COEF_BYTES = 363.76

# MTP / spec-decode draft resident footprint per card (GiB), per MTP layer.
# PN348 shares the draft backbone (embed_tokens + lm_head) with the target, so
# the resident adder is ~1 GiB/rank for our 1-MTP-layer Qwen3.6 configs.
_MTP_GIB_PER_LAYER = 1.0

# Verdict band: how far the requested KV pool may exceed the available slack
# before the verdict drops from PASS to TIGHT (vLLM caps the pool). 5% slack
# absorbs the estimator's own noise without false-TIGHTing a config that boots.
_TIGHT_BAND = 1.05

#: Documented estimator error band (GiB). The fixed-footprint coefficients are
#: anchored to a single live point per model; treat predictions as ±this.
FIT_BAND_GIB = 1.5


# ─── Rig + projection result objects ────────────────────────────────────────


@dataclass(frozen=True)
class ProjectorRig:
    """The hardware a projection is run against. Pure data — no probing here
    (the live ``nvidia-smi`` probe lives in ``preflight_fit.RigProbe``)."""
    vram_gib_per_card: float
    gpu_count: int
    name: str = "gpu"


@dataclass(frozen=True)
class Projection:
    """Per-card byte-level VRAM projection for one (preset, rig, operating-point)."""
    preset_id: str
    kv_format: str
    ctx: int
    max_num_seqs: int
    tp: int
    mem_util: float
    vram_gib: float

    weights_gib: float
    kv_pool_requested_gib: float    # what max_ctx × max_num_seqs WANTS
    kv_pool_actual_gib: float       # what fits after the fixed footprint (vLLM cap)
    recurrent_state_gib: float
    activation_gib: float
    cudagraph_overhead_gib: float
    drafter_gib: float

    fixed_gib: float                # everything except the growing KV pool
    budget_gib: float               # mem_util × vram
    total_gib: float                # fixed + actual KV pool
    headroom_gib: float             # budget − total
    available_for_kv_gib: float     # budget − fixed (the pool's ceiling)

    verdict: str                    # "PASS" | "TIGHT" | "FAIL"
    provisional: bool               # True when the model's calibration is provisional
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def utilization(self) -> float:
        return self.total_gib / self.budget_gib if self.budget_gib > 0 else 0.0


# ─── Shape helpers ──────────────────────────────────────────────────────────


def _has_byte_math(shape) -> bool:
    """True when the shape carries enough dims to do byte-level KV math."""
    if shape is None:
        return False
    return all(
        getattr(shape, name, None) is not None
        for name in ("num_attention_layers", "num_kv_heads", "head_dim")
    )


def _is_provisional(shape) -> bool:
    """A shape is calibration-PROVISIONAL unless it was anchored to live engine
    telemetry. We mark that with ``num_experts`` presence as a proxy ONLY for
    the MoE 35B (the one with a live PN403 anchor); the dense 27B has no live
    block capture, so it is provisional. The real signal is the explicit list
    of calibrated model fingerprints below — anything not in it is provisional.
    """
    return _shape_fingerprint(shape) not in _CALIBRATED_FINGERPRINTS


def _shape_fingerprint(shape) -> tuple:
    """A stable identity for a shape, used to look up calibration confidence."""
    if shape is None:
        return ()
    return (
        getattr(shape, "num_hidden_layers", None),
        getattr(shape, "num_attention_layers", None),
        getattr(shape, "num_kv_heads", None),
        getattr(shape, "head_dim", None),
        getattr(shape, "hidden_size", None),
        getattr(shape, "num_experts", None),
    )


# Models whose fixed-footprint calibration is anchored to LIVE engine telemetry.
# The 35B-A3B (40 layers, 10 attn, 2 KV heads, head_dim 128, hidden 2048, 256
# experts) is the dev424 PN403 anchor. Add a fingerprint here only when a model
# gains a live num_gpu_blocks / kv_cache_size_tokens capture.
_CALIBRATED_FINGERPRINTS: set[tuple] = {
    (40, 10, 2, 128, 2048, 256),   # qwen3.6-35b-a3b-fp8 (PN403 live anchor)
}


# ─── Per-component byte math (pure) ─────────────────────────────────────────


def weights_per_card_bytes(shape, tp: int) -> int:
    """Resident weight bytes per card after TP shard.

    Prefers the declared ``weights_total_gib`` (the measured on-disk == resident
    footprint, exact for fp8/int4 packed weights). Falls back to a
    params×bits estimate only when total is absent. For MoE the FULL expert set
    is resident and sharded across TP ranks (dense in VRAM, sparse in compute).
    """
    tp = max(int(tp), 1)
    total_gib = getattr(shape, "weights_total_gib", None) if shape else None
    if total_gib:
        return int(float(total_gib) * _GIB) // tp
    # Estimate fallback: hidden_size is not enough to recover params, so without
    # a measured total we return 0 and let project() flag a low-confidence note.
    return 0


def kv_pool_per_card_bytes(
    shape,
    kv_format: Optional[str],
    ctx: int,
    max_num_seqs: int,
    tp: int,
    *,
    mtp_n: int = 0,
) -> int:
    """Growing KV-pool bytes per card.

    Only the full-attention layers grow KV (hybrid GDN/Mamba layers carry a
    fixed-size recurrent state — see ``recurrent_state_per_card_bytes``). K and
    V stored independently → ×2. GQA-aware via ``num_kv_heads``. Sharded by TP.

    The MTP draft tokens extend the effective per-sequence context slightly
    (``mtp_n`` extra slots per step), modeled as ``mtp_n × 32`` extra tokens.
    """
    if not _has_byte_math(shape):
        return 0
    n_attn = int(shape.num_attention_layers)
    n_kv = int(shape.num_kv_heads)
    head_dim = int(shape.head_dim)
    bpe = kv_format_bytes(kv_format)
    tp = max(int(tp), 1)

    per_token_per_card = (n_attn * n_kv * head_dim * 2 * bpe) / tp
    effective_ctx = int(ctx) + int(mtp_n) * 32
    return int(per_token_per_card * effective_ctx * max(int(max_num_seqs), 1))


def recurrent_state_per_card_bytes(shape, max_num_seqs: int, tp: int) -> int:
    """GDN / Mamba recurrent-state bytes per card.

    Fixed-size per running stream (NOT context-linear). Replicated per rank
    (recurrent state is per-stream, not sharded like dense weights), so this is
    divided by TP only because the layer set itself is split across ranks for
    our hybrid configs. Zero for pure-dense models (no recurrent layers).
    """
    if shape is None:
        return 0
    n_recur = getattr(shape, "num_recurrent_layers", None)
    hidden = getattr(shape, "hidden_size", None)
    if not n_recur or not hidden:
        return 0
    tp = max(int(tp), 1)
    per_stream = int(n_recur) * _RECUR_COEF_BYTES * int(hidden) / tp
    return int(per_stream * max(int(max_num_seqs), 1))


def activation_peak_per_card_bytes(shape, kv_format: Optional[str], ctx: int, tp: int) -> int:
    """Prefill activation-peak bytes per card.

    For hybrid GDN/Mamba models the peak comes from the GDN block-state
    materialization (PerfMamba O(γ·D·N·L) form), which is linear in context.
    Coefficient calibrated on the 35B PN403 anchor. For a pure-dense model with
    no recurrent layers we fall back to the attention-layer count so the term is
    never zero (a dense forward still has activation).
    """
    if shape is None:
        return 0
    n_recur = getattr(shape, "num_recurrent_layers", None) or 0
    n_attn = getattr(shape, "num_attention_layers", None) or 0
    # Layers that drive the ctx-linear activation peak. Hybrid → GDN layers
    # dominate; pure-dense → the attention layers carry it.
    n_act_layers = int(n_recur) if n_recur else int(n_attn)
    if n_act_layers <= 0:
        return 0
    tp = max(int(tp), 1)
    return int(_ACTIVATION_COEF_BYTES * n_act_layers * int(ctx) / tp)


def cudagraph_overhead_per_card_bytes(mem_util: float, tp: int) -> int:
    """vLLM cudagraph-capture + workspace overhead per card (bytes).

    Linear-ish in mem_util (more capture sizes retained at higher util) with a
    per-extra-rank NCCL-workspace bump. Calibrated within the 35B fixed-footprint
    reconstruction.
    """
    tp = max(int(tp), 1)
    gib = (
        _OVERHEAD_BASE_GIB
        + _OVERHEAD_PER_UTIL_GIB * float(mem_util)
        + _OVERHEAD_PER_EXTRA_RANK_GIB * (tp - 1)
    )
    return int(gib * _GIB)


def drafter_per_card_bytes(shape, *, mtp: bool, tp: int, extra_drafter_gib: float = 0.0) -> int:
    """Resident MTP / DFlash drafter bytes per card.

    Built-in MTP (shared backbone via PN348) adds ~1 GiB/rank per MTP layer.
    ``extra_drafter_gib`` is an explicit external-drafter weight (e.g. DFlash),
    sharded by TP.
    """
    tp = max(int(tp), 1)
    gib = 0.0
    if mtp and shape is not None:
        n_mtp = getattr(shape, "mtp_num_layers", None) or 0
        gib += _MTP_GIB_PER_LAYER * int(n_mtp)
    if extra_drafter_gib > 0:
        gib += float(extra_drafter_gib) / tp
    return int(gib * _GIB)


# ─── Aggregate projection ───────────────────────────────────────────────────


def project_from_shape(
    shape,
    *,
    preset_id: str,
    kv_format: Optional[str],
    ctx: int,
    max_num_seqs: int,
    tp: int,
    mem_util: float,
    vram_gib: float,
    mtp: bool = False,
    mtp_n: int = 0,
    extra_drafter_gib: float = 0.0,
) -> Projection:
    """Core projection from a raw ``ModelShape`` (the I/O-free heart).

    ``project()`` is the preset-driven convenience wrapper around this.
    """
    tp = max(int(tp), 1)
    ctx = max(int(ctx), 1)
    max_num_seqs = max(int(max_num_seqs), 1)
    kv_fmt = (kv_format or DEFAULT_KV_FORMAT)

    weights_b = weights_per_card_bytes(shape, tp)
    kv_req_b = kv_pool_per_card_bytes(
        shape, kv_fmt, ctx, max_num_seqs, tp, mtp_n=mtp_n if mtp else 0,
    )
    recurrent_b = recurrent_state_per_card_bytes(shape, max_num_seqs, tp)
    activation_b = activation_peak_per_card_bytes(shape, kv_fmt, ctx, tp)
    overhead_b = cudagraph_overhead_per_card_bytes(mem_util, tp)
    drafter_b = drafter_per_card_bytes(
        shape, mtp=mtp, tp=tp, extra_drafter_gib=extra_drafter_gib,
    )

    fixed_b = weights_b + recurrent_b + activation_b + overhead_b + drafter_b
    budget_b = mem_util * vram_gib * _GIB
    available_for_kv_b = max(0.0, budget_b - fixed_b)
    # vLLM caps the growing pool to the available slack (PagedAttention).
    kv_actual_b = min(kv_req_b, available_for_kv_b)
    total_b = fixed_b + kv_actual_b

    notes: list[str] = []

    if weights_b == 0:
        notes.append(
            "weights_total_gib not declared on this model's shape — weight "
            "footprint is 0; projection under-counts. Add weights_total_gib."
        )

    # Verdict (club-3090 structure, OUR thresholds):
    #   FAIL  — the fixed footprint leaves no room for even ONE max_ctx
    #           sequence's growing KV (vLLM's boot pre-check refuses to start).
    #   TIGHT — the requested pool exceeds the available slack: vLLM caps it,
    #           so effective concurrency at full max_ctx drops below max_num_seqs.
    #   PASS  — requested pool fits with room to spare.
    per_seq_kv_b = kv_req_b / max_num_seqs
    # A single sequence must fit; cap the floor at 1 GiB so KV-light hybrid/MoE
    # configs (tiny growing pool) are not false-FAILed, and dense long-KV
    # configs still need real room.
    min_kv_b = max(0.01 * _GIB, min(1.0 * _GIB, per_seq_kv_b))

    if available_for_kv_b < min_kv_b:
        verdict = "FAIL"
        notes.append(
            f"fixed footprint ({fixed_b / _GIB:.1f} GiB) leaves only "
            f"{available_for_kv_b / _GIB:.2f} GiB for KV — below the "
            f"{min_kv_b / _GIB:.2f} GiB one max_ctx={ctx:,} sequence needs; "
            "vLLM will refuse to boot. Lower max_ctx, drop the drafter, raise "
            "mem_util, or add a card."
        )
    elif kv_req_b > available_for_kv_b * _TIGHT_BAND:
        verdict = "TIGHT"
        notes.append(
            f"requested KV pool ({kv_req_b / _GIB:.1f} GiB) > available "
            f"({available_for_kv_b / _GIB:.1f} GiB) — vLLM will cap the pool; "
            f"effective concurrency may be below max_num_seqs={max_num_seqs} at "
            f"full max_ctx={ctx:,}."
        )
    else:
        verdict = "PASS"

    provisional = _is_provisional(shape)
    if provisional and _has_byte_math(shape):
        notes.append(
            "calibration PROVISIONAL for this model (no live engine "
            "telemetry anchor) — treat per-card GB as ±1.5 GiB."
        )

    return Projection(
        preset_id=preset_id,
        kv_format=kv_fmt,
        ctx=ctx,
        max_num_seqs=max_num_seqs,
        tp=tp,
        mem_util=mem_util,
        vram_gib=vram_gib,
        weights_gib=weights_b / _GIB,
        kv_pool_requested_gib=kv_req_b / _GIB,
        kv_pool_actual_gib=kv_actual_b / _GIB,
        recurrent_state_gib=recurrent_b / _GIB,
        activation_gib=activation_b / _GIB,
        cudagraph_overhead_gib=overhead_b / _GIB,
        drafter_gib=drafter_b / _GIB,
        fixed_gib=fixed_b / _GIB,
        budget_gib=budget_b / _GIB,
        total_gib=total_b / _GIB,
        headroom_gib=(budget_b - total_b) / _GIB,
        available_for_kv_gib=available_for_kv_b / _GIB,
        verdict=verdict,
        provisional=provisional,
        notes=tuple(notes),
    )


def fit_verdict(projection: Projection) -> str:
    """Return the PASS / TIGHT / FAIL verdict carried by a projection."""
    return projection.verdict


def solve_max_ctx(
    shape,
    *,
    kv_format: Optional[str],
    max_num_seqs: int,
    tp: int,
    mem_util: float,
    vram_gib: float,
    mtp: bool = False,
    mtp_n: int = 0,
    extra_drafter_gib: float = 0.0,
    ctx_cap: int = 1_048_576,
    step: int = 1024,
) -> int:
    """Largest ``max_ctx`` (rounded to ``step``) that keeps the verdict at
    PASS or TIGHT. Binary search — monotone because growing-KV is strictly
    increasing in ctx while the fixed footprint grows only sub-linearly, so the
    PASS/TIGHT region is a prefix ``[0, max_ctx]``.

    Returns 0 when even the smallest context FAILs (fixed footprint alone blows
    the budget).
    """
    lo, hi = step, int(ctx_cap)
    best = 0
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        if mid <= 0:
            break
        p = project_from_shape(
            shape,
            preset_id="<solve>",
            kv_format=kv_format,
            ctx=mid,
            max_num_seqs=max_num_seqs,
            tp=tp,
            mem_util=mem_util,
            vram_gib=vram_gib,
            mtp=mtp,
            mtp_n=mtp_n,
            extra_drafter_gib=extra_drafter_gib,
        )
        if p.verdict in ("PASS", "TIGHT"):
            best = mid
            lo = mid + step
        else:
            hi = mid - step
    return best


# ─── Preset-driven convenience wrapper ──────────────────────────────────────


def _resolve_operating_point(preset) -> dict:
    """Pull (ctx, max_num_seqs, tp, mem_util, kv_format, mtp, mtp_n) off a
    composed ModelConfig (``load_alias`` output). Tolerant of missing fields."""
    caps = getattr(preset, "capabilities", None)
    spec = getattr(caps, "spec_decode", None) if caps else getattr(preset, "spec_decode", None)
    mtp = bool(spec and getattr(spec, "method", None) == "mtp")
    mtp_n = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0

    hw = getattr(preset, "hardware", None)
    n_gpus = int(getattr(hw, "n_gpus", 1) or 1) if hw else 1

    kv_format = getattr(preset, "kv_cache_dtype", None)
    if not kv_format and caps is not None:
        kv_format = getattr(caps, "kv_cache_dtype", None)

    return {
        "ctx": int(getattr(preset, "max_model_len", 0) or 0),
        "max_num_seqs": int(getattr(preset, "max_num_seqs", 1) or 1),
        "tp": max(n_gpus, 1),
        "mem_util": float(getattr(preset, "gpu_memory_utilization", 0.9) or 0.9),
        "kv_format": kv_format,
        "mtp": mtp,
        "mtp_n": mtp_n,
    }


def _shape_of(preset):
    """Best-effort extraction of the ModelShape from a composed preset."""
    caps = getattr(preset, "capabilities", None)
    shape = getattr(caps, "shape", None) if caps else None
    if shape is None:
        shape = getattr(preset, "shape", None)
    return shape


def project(
    preset,
    rig: ProjectorRig,
    *,
    shape=None,
    ctx: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    kv_format: Optional[str] = None,
    preset_id: Optional[str] = None,
) -> Projection:
    """Project a composed preset against a rig at its declared (or overridden)
    operating point. ``ctx`` / ``max_num_seqs`` / ``kv_format`` override the
    preset's declared values when supplied.

    ``shape`` may be passed explicitly (the composed V1 ``ModelConfig`` drops
    the V2 ``capabilities.shape`` block, so a caller that loaded the ModelDef
    separately passes the ``ModelShape`` here). When omitted, ``project`` tries
    to read it off the preset.

    Raises ``ValueError`` when no ``shape`` is resolvable (the projector cannot
    do byte math without dims) — callers should fall back to the envelope-only
    check in that case.
    """
    if shape is None:
        shape = _shape_of(preset)
    if not _has_byte_math(shape):
        raise ValueError(
            "preset's model declares no byte-level shape (capabilities.shape "
            "with num_attention_layers/num_kv_heads/head_dim) — cannot project "
            "byte-level VRAM; use the envelope fit-check instead."
        )

    op = _resolve_operating_point(preset)
    return project_from_shape(
        shape,
        preset_id=(preset_id or getattr(preset, "key", None) or "<preset>"),
        kv_format=(kv_format if kv_format is not None else op["kv_format"]),
        ctx=(ctx if ctx is not None else op["ctx"]),
        max_num_seqs=(max_num_seqs if max_num_seqs is not None else op["max_num_seqs"]),
        tp=op["tp"],
        mem_util=op["mem_util"],
        vram_gib=rig.vram_gib_per_card,
        mtp=op["mtp"],
        mtp_n=op["mtp_n"],
    )


__all__ = [
    "KV_FORMAT_BYTES",
    "DEFAULT_KV_FORMAT",
    "FIT_BAND_GIB",
    "kv_format_bytes",
    "ProjectorRig",
    "Projection",
    "weights_per_card_bytes",
    "kv_pool_per_card_bytes",
    "recurrent_state_per_card_bytes",
    "activation_peak_per_card_bytes",
    "cudagraph_overhead_per_card_bytes",
    "drafter_per_card_bytes",
    "project_from_shape",
    "project",
    "fit_verdict",
    "solve_max_ctx",
]
