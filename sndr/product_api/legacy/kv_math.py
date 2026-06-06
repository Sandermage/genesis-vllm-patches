# SPDX-License-Identifier: Apache-2.0
"""KV-cache / VRAM / max-context calculator for the GUI's fit planner.

A typed, testable model of where VRAM goes when serving a transformer:

    VRAM/GPU ≈ weights/TP + KV(context, concurrency)/TP + fixed overhead

with the standard KV formula ``2·layers·kv_heads·head_dim·dtype_bytes`` per
token (GQA-aware via ``num_kv_heads``). For MoE models the *full* expert weight
set is resident (dense in VRAM, sparse in compute) and is sharded by tensor
parallelism — getting this right is what keeps long-context MoE fit honest.

This is the engine behind the GUI's interactive fit panel. It is deliberately
dims-driven (no hidden model assumptions): the caller supplies a
:class:`ModelArch` (from the curated registry or the GUI's editable fields), and
the overhead term can be *calibrated* against a real measured VRAM point — e.g.
the live total from host discovery — so predictions track reality instead of a
guessed constant.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

_MIB = 1024 * 1024

# KV dtype → bytes per element. The OpenAI/vLLM names operators actually use.
KV_DTYPE_BYTES: dict[str, float] = {
    "fp16": 2.0, "bf16": 2.0, "float16": 2.0,
    "fp8": 1.0, "fp8_e5m2": 1.0, "fp8_e4m3": 1.0, "int8": 1.0,
    "tq_k8v4": 0.75,  # Genesis turbo-quant KV (8-bit K + 4-bit V ≈ 0.75 B/elem)
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelArch:
    """Architecture dims needed to size a model. GQA via ``num_kv_heads``.

    ``weights_bytes_total`` (when known — e.g. read from the model files on the
    host) is the *exact* resident weight size and overrides the params×bits
    estimate. ``max_context`` is the model's native window (informational)."""

    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    params_b: float          # total parameters, in billions (estimate fallback)
    weight_bits: float       # quant bit-width of the weights (4/8/16…)
    is_moe: bool = False
    active_params_b: Optional[float] = None  # informational (compute, not VRAM)
    weights_bytes_total: Optional[int] = None  # exact resident weight bytes, if measured
    max_context: Optional[int] = None          # native context window
    # Sliding-window attention (Gemma, some Qwen): local layers cap their KV at
    # ``sliding_window`` tokens; only ``global_layers`` hold the full context.
    sliding_window: Optional[int] = None
    global_layers: Optional[int] = None
    source: str = "curated"   # "curated" | "host-config"


def _kv_token_layers(arch: ModelArch, context: int) -> int:
    """Effective per-token KV "layer-tokens" at a context length, accounting for
    sliding-window attention. Without a window this is ``num_layers·context``."""
    if arch.sliding_window and arch.global_layers is not None:
        glob = max(0, min(arch.num_layers, arch.global_layers))
        local = arch.num_layers - glob
        return glob * context + local * min(context, arch.sliding_window)
    return arch.num_layers * context


def kv_bytes_per_token(arch: ModelArch, *, kv_bytes: float) -> int:
    """Bytes of KV cache one token occupies across all layers (K and V)."""
    return int(2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * kv_bytes)


def weights_bytes(arch: ModelArch) -> int:
    """Total resident weight bytes. Exact when measured from the host, else the
    params×bits estimate. For MoE this is the *full* expert set."""
    if arch.weights_bytes_total and arch.weights_bytes_total > 0:
        return int(arch.weights_bytes_total)
    return int(arch.params_b * 1e9 * arch.weight_bits / 8)


def estimate(
    arch: ModelArch,
    *,
    context: int,
    concurrency: int = 1,
    tp: int = 1,
    kv_bytes: float = 1.0,
    gpu_count: int = 1,
    gpu_vram_mib: int = 24564,
    util: float = 0.90,
    overhead_mib: float = 1500.0,
) -> dict[str, Any]:
    """Per-GPU VRAM breakdown + fit verdict + achievable max context."""
    tp = max(1, int(tp))
    concurrency = max(1, int(concurrency))
    context = max(1, int(context))

    weights_per_gpu = weights_bytes(arch) / tp
    kv_bpt = kv_bytes_per_token(arch, kv_bytes=kv_bytes)  # per all-layers token (legacy field)
    per_layer_token = 2 * arch.num_kv_heads * arch.head_dim * kv_bytes  # K+V, one layer, one token
    kv_total = per_layer_token * _kv_token_layers(arch, context) * concurrency
    kv_per_gpu = kv_total / tp
    overhead_b = max(0.0, overhead_mib) * _MIB

    total_per_gpu = weights_per_gpu + kv_per_gpu + overhead_b
    budget_per_gpu = gpu_vram_mib * _MIB * util

    # Max context: free budget after weights+overhead. With sliding-window
    # attention the KV→context relation is piecewise-linear (flatter past the
    # window), so solve both regimes and take the larger valid root.
    free_for_kv = budget_per_gpu - weights_per_gpu - overhead_b
    max_context = 0
    if free_for_kv > 0 and per_layer_token > 0:
        unit = per_layer_token * concurrency / tp  # bytes per (layer-token)
        budget_layer_tokens = free_for_kv / unit
        glob = arch.num_layers
        local = 0
        window = arch.sliding_window or 0
        if window and arch.global_layers is not None:
            glob = max(0, min(arch.num_layers, arch.global_layers))
            local = arch.num_layers - glob
        # Regime A: ctx <= window → all layers count fully.
        ctx_a = budget_layer_tokens / arch.num_layers if arch.num_layers else 0
        if not window or ctx_a <= window:
            max_context = int(ctx_a)
        else:
            # Regime B: ctx > window → local layers pinned at window.
            denom = glob if glob else arch.num_layers
            max_context = int((budget_layer_tokens - local * window) / denom) if denom else 0

    mib = lambda b: round(b / _MIB)  # noqa: E731
    return {
        "model": arch.name,
        "weights_per_gpu_mib": mib(weights_per_gpu),
        "kv_per_gpu_mib": mib(kv_per_gpu),
        "kv_total_mib": mib(kv_total),
        "overhead_mib": round(overhead_mib),
        "total_per_gpu_mib": mib(total_per_gpu),
        "budget_per_gpu_mib": mib(budget_per_gpu),
        "headroom_mib": mib(budget_per_gpu - total_per_gpu),
        "fits": total_per_gpu <= budget_per_gpu,
        "max_context": max(0, max_context),
        "kv_bytes_per_token": kv_bpt,
        "tp": tp,
        "concurrency": concurrency,
        "context": context,
    }


def fit_envelope(
    arch: ModelArch,
    *,
    contexts: list[int],
    concurrencies: list[int],
    kv_bytes: float,
    tp: int,
    gpu_vram_mib: int,
    util: float,
    overhead_mib: float,
) -> list[list[dict[str, Any]]]:
    """A (concurrency × context) grid of fit verdicts + headroom — the operating
    envelope. Row = concurrency, col = context. Drives the planner heatmap."""
    grid: list[list[dict[str, Any]]] = []
    for conc in concurrencies:
        row: list[dict[str, Any]] = []
        for ctx in contexts:
            e = estimate(arch, context=ctx, concurrency=conc, tp=tp, kv_bytes=kv_bytes,
                         gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead_mib)
            row.append({"context": ctx, "concurrency": conc, "fits": e["fits"],
                        "headroom_mib": e["headroom_mib"], "total_per_gpu_mib": e["total_per_gpu_mib"]})
        grid.append(row)
    return grid


def recommend(
    arch: ModelArch,
    *,
    target_context: int,
    target_concurrency: int,
    tp: int,
    gpu_vram_mib: int,
    util: float,
    overhead_mib: float,
    dtypes: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Rank KV dtypes by *fidelity* (bytes/elem) at the target operating point,
    flagging which fit. The recommended pick is the highest-fidelity KV that
    still fits — operators get "use fp8 here", not a wall of numbers."""
    table = dtypes or KV_DTYPE_BYTES
    # De-dup by byte size, keep the canonical name per size, fidelity desc.
    seen: dict[float, str] = {}
    for name, b in table.items():
        seen.setdefault(b, name)
    options = sorted(seen.items(), key=lambda kv: -kv[0])
    out: list[dict[str, Any]] = []
    for b, name in options:
        e = estimate(arch, context=target_context, concurrency=target_concurrency, tp=tp,
                     kv_bytes=b, gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead_mib)
        out.append({"kv_dtype": name, "kv_bytes": b, "fits": e["fits"],
                    "headroom_mib": e["headroom_mib"], "max_context": e["max_context"],
                    "total_per_gpu_mib": e["total_per_gpu_mib"], "recommended": False})
    # Recommended = the highest-fidelity option that fits at the target.
    for row in out:
        if row["fits"]:
            row["recommended"] = True
            break
    return out


def calibrate_overhead(
    arch: ModelArch,
    *,
    measured_total_mib: float,
    context: int,
    concurrency: int = 1,
    tp: int = 1,
    kv_bytes: float = 1.0,
) -> float:
    """Back out the fixed overhead (MiB) from a real measured VRAM total.

    ``overhead = measured − weights/TP − KV/TP``. Clamped at 0. Lets predictions
    track a live point (e.g. discovery's residency) instead of a guess.
    """
    weights_per_gpu = weights_bytes(arch) / tp
    kv_per_gpu = kv_bytes_per_token(arch, kv_bytes=kv_bytes) * max(1, context) * max(1, concurrency) / max(1, tp)
    overhead_b = measured_total_mib * _MIB - weights_per_gpu - kv_per_gpu
    return max(0.0, overhead_b / _MIB)


# ── Curated starter registry (public architecture facts; GUI-editable) ──────
# These are reasonable, editable defaults for the families the project serves.
# The calculator is dims-driven — operators tweak any field in the GUI, and a
# future step can auto-pull exact dims from each model's config.json over SSH.
def known_models() -> dict[str, ModelArch]:
    return {
        "qwen3.6-27b-int4": ModelArch("Qwen3.6-27B INT4", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4),
        "qwen3.6-35b-a3b-fp8": ModelArch("Qwen3.6-35B-A3B FP8 (MoE)", num_layers=64, num_kv_heads=8, head_dim=128, params_b=35.0, weight_bits=8, is_moe=True, active_params_b=3.0),
        "gemma-4-31b-awq": ModelArch("Gemma 4 31B AWQ", num_layers=62, num_kv_heads=8, head_dim=128, params_b=31.0, weight_bits=4),
        "gemma-4-26b-a4b-awq": ModelArch("Gemma 4 26B-A4B AWQ (MoE)", num_layers=48, num_kv_heads=8, head_dim=128, params_b=26.0, weight_bits=4, is_moe=True, active_params_b=4.0),
    }


def arch_from_dict(data: dict[str, Any]) -> ModelArch:
    """Build a ModelArch from a GUI payload, falling back to a registry entry.

    When the GUI passes dims read from the host's ``config.json`` (plus an exact
    ``weights_bytes_total`` from the model files), the calculator sizes the model
    from reality instead of the curated estimate."""
    base = known_models().get(str(data.get("model_id") or ""))
    if base is None:
        base = ModelArch(
            name=str(data.get("name") or "custom"),
            num_layers=int(data.get("num_layers") or 64),
            num_kv_heads=int(data.get("num_kv_heads") or 8),
            head_dim=int(data.get("head_dim") or 128),
            params_b=float(data.get("params_b") or 27.0),
            weight_bits=float(data.get("weight_bits") or 4),
            is_moe=bool(data.get("is_moe")),
            active_params_b=data.get("active_params_b"),
            weights_bytes_total=(int(data["weights_bytes_total"]) if data.get("weights_bytes_total") else None),
            max_context=(int(data["max_context"]) if data.get("max_context") else None),
            source=str(data.get("source") or "curated"),
        )
    int_fields = ("num_layers", "num_kv_heads", "head_dim", "weights_bytes_total", "max_context", "sliding_window", "global_layers")
    float_fields = ("params_b", "weight_bits")
    overrides: dict[str, Any] = {}
    for k in int_fields:
        if data.get(k) is not None:
            overrides[k] = int(data[k])
    for k in float_fields:
        if data.get(k) is not None:
            overrides[k] = float(data[k])
    if data.get("source"):
        overrides["source"] = str(data["source"])
    return replace(base, **overrides) if overrides else base


__all__ = [
    "KV_DTYPE_BYTES", "ModelArch", "arch_from_dict", "calibrate_overhead",
    "estimate", "fit_envelope", "known_models", "kv_bytes_per_token",
    "recommend", "weights_bytes",
]
