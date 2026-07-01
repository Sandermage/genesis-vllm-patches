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
from typing import Any

_MIB = 1024 * 1024

# KV dtype → bytes per element. The OpenAI/vLLM names operators actually use.
# Exact bytes/element include quant-scale overhead (calibrated against measured
# per-card VRAM — cross-checked with the club-3090 kv-calc byte table).
KV_DTYPE_BYTES: dict[str, float] = {
    "fp16": 2.0, "bf16": 2.0, "float16": 2.0,
    "fp8": 1.0, "fp8_e5m2": 1.0, "fp8_e4m3": 1.0, "int8": 1.0,
    "int8_per_token_head": 1.01,  # int8 + per-token-head fp16 scale (compressed-tensors weights)
    "tq_k8v4": 0.75,  # Genesis turbo-quant KV (8-bit K + 4-bit V ≈ 0.75 B/elem)
    "q4_0": 0.5625,   # 4-bit + per-group fp16 scale (0.5 + 0.0625)
    "int4": 0.5,
    "turboquant_3bit_nc": 0.425,  # TQ3: 3-bit + QJL overhead (0.375 + 0.05); Genesis-only
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
    active_params_b: float | None = None  # informational (compute, not VRAM)
    weights_bytes_total: int | None = None  # exact resident weight bytes, if measured
    max_context: int | None = None          # native context window
    # Sliding-window attention (Gemma, some Qwen): local layers cap their KV at
    # ``sliding_window`` tokens; only ``global_layers`` hold the full context.
    sliding_window: int | None = None
    global_layers: int | None = None
    # Hybrid (GDN/Mamba/DeltaNet) + MoE-hybrid models grow KV in ONLY their
    # full-attention layers; the recurrent layers hold a fixed per-sequence state
    # (weights/activation, not KV). ``attn_layers`` is that KV-growing count
    # (None → all ``num_layers`` grow, the dense default). Without this a hybrid
    # like Qwen3.6 (16 attn of 64) over-estimates KV ~4×.
    attn_layers: int | None = None
    source: str = "curated"   # "curated" | "host-config"


def _kv_growing_layers(arch: ModelArch) -> int:
    """Number of layers whose KV grows with context (attention layers). Hybrid
    models set ``attn_layers``; dense models grow all ``num_layers``."""
    if arch.attn_layers is not None and arch.attn_layers > 0:
        return min(arch.num_layers, arch.attn_layers)
    return arch.num_layers


def _kv_token_layers(arch: ModelArch, context: int) -> int:
    """Effective per-token KV "layer-tokens" at a context length, accounting for
    both hybrid (only attn layers grow) and sliding-window attention."""
    n = _kv_growing_layers(arch)
    if arch.sliding_window and arch.global_layers is not None:
        glob = max(0, min(n, arch.global_layers))
        local = n - glob
        return glob * context + local * min(context, arch.sliding_window)
    return n * context


def kv_bytes_per_token(arch: ModelArch, *, kv_bytes: float) -> int:
    """Bytes of KV cache one token occupies across the KV-growing layers (K+V)."""
    return int(2 * _kv_growing_layers(arch) * arch.num_kv_heads * arch.head_dim * kv_bytes)


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
        n_grow = _kv_growing_layers(arch)  # only these layers grow KV (hybrid-aware)
        glob = n_grow
        local = 0
        window = arch.sliding_window or 0
        if window and arch.global_layers is not None:
            glob = max(0, min(n_grow, arch.global_layers))
            local = n_grow - glob
        # Regime A: ctx <= window → all KV-growing layers count fully.
        ctx_a = budget_layer_tokens / n_grow if n_grow else 0
        if not window or ctx_a <= window:
            max_context = int(ctx_a)
        else:
            # Regime B: ctx > window → local layers pinned at window.
            denom = glob if glob else n_grow
            max_context = int((budget_layer_tokens - local * window) / denom) if denom else 0

    # Three-way verdict (vLLM reality, not a binary fits/doesn't). vLLM boots even
    # when the requested KV pool exceeds free VRAM — it CAPS the pool (TIGHT:
    # effective concurrency/context reduced) instead of failing. FAIL is reserved
    # for when the fixed weights+overhead can't even fit a minimal boot pool. The
    # boot floor is per-sequence, not a flat 1 GB (a flat floor false-FAILs
    # KV-light MoE/SWA models).
    _gib = 1024 * _MIB
    fixed_per_gpu = weights_per_gpu + overhead_b
    min_kv_per_gpu = max(0.01 * _gib, min(1.0 * _gib, kv_per_gpu / concurrency))
    if fixed_per_gpu + min_kv_per_gpu > budget_per_gpu:
        verdict = "fail"
    elif total_per_gpu > budget_per_gpu:
        verdict = "tight"
    else:
        verdict = "pass"
    kv_pool_capped_mib = (
        round(max(0.0, budget_per_gpu - fixed_per_gpu) / _MIB) if verdict == "tight" else None
    )

    warnings: list[str] = []
    # Valid-TP: tensor-parallel width must divide the KV heads (GQA) — vLLM errors otherwise.
    if tp > 1 and arch.num_kv_heads % tp != 0:
        warnings.append(
            f"TP={tp} does not divide {arch.num_kv_heads} KV heads — invalid tensor-parallel width"
        )
    # Cliff 2: hybrid GDN forward OOMs at ~50-60K single-prompt tokens on one 24GB
    # card regardless of mem-util, for non-fp16 KV — a hard ceiling a byte budget misses.
    if arch.attn_layers is not None and tp == 1 and context > 50000 and kv_bytes < 2.0:
        warnings.append(
            "Cliff 2: hybrid GDN may OOM past ~50-60K single-prompt tokens on one card "
            "(kv≠fp16), regardless of budget — split across GPUs or use fp16 KV"
        )

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
        "verdict": verdict,                       # "pass" | "tight" | "fail"
        "kv_pool_capped_mib": kv_pool_capped_mib,  # capped KV pool when TIGHT
        "warnings": warnings,
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
    dtypes: dict[str, float] | None = None,
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
        # Qwen3.6 hybrid (qwen3-next-hybrid): only the full-attention layers grow
        # KV (the rest are GDN/DeltaNet recurrent). GQA num_kv_heads=4, head_dim=256.
        "qwen3.6-27b-int4": ModelArch("Qwen3.6-27B INT4", num_layers=64, attn_layers=16, num_kv_heads=4, head_dim=256, params_b=27.0, weight_bits=4),
        "qwen3.6-35b-a3b-fp8": ModelArch("Qwen3.6-35B-A3B FP8 (MoE)", num_layers=40, attn_layers=10, num_kv_heads=4, head_dim=256, params_b=35.0, weight_bits=8, is_moe=True, active_params_b=3.0),
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


# ─── Grounded cross-rig reference (MEASURED numbers only, each cited) ─────────
# The planner estimates VRAM/context from first principles; it deliberately does
# NOT predict throughput (topology/SM/kernel interactions make that unreliable).
# Instead we surface REAL measured TPS points so an operator can sanity-check the
# throughput their config is likely to see. Sources: our own canonical
# genesis_bench_suite runs, and the noonghunna/club-3090 BENCHMARKS.md
# (public, cross-rig 3090/4090/5090 data). No fabricated multipliers.
#   `alt_kind` labels the SECOND number honestly — the two rig families measure
#   different axes: our genesis rows report single-stream + concurrency-aggregate
#   ("concN"); the club-3090 rows report narration + code single-stream ("code").
#   Never conflate them into one "1/N" — the GUI shows the label.
MEASURED_REFERENCE: list[dict[str, Any]] = [
    # model, hardware, tp, link, tps_single, tps_multi, alt_kind, context_k, source
    {"model": "Qwen3.6-35B-A3B FP8 TQ k8v4 (MTP K=5)", "hardware": "2× A5000 24GB", "tp": 2, "link": "pcie",
     "tps_single": 240.6, "tps_multi": 644, "alt_kind": "conc8", "context_k": 280, "source": "genesis bench dev672 2026-07-01"},
    {"model": "Qwen3.6-27B INT4 TQ k8v4 (MTP K=3)", "hardware": "2× A5000 24GB", "tp": 2, "link": "pcie",
     "tps_single": 120, "tps_multi": 292, "alt_kind": "conc4", "context_k": 280, "source": "genesis bench"},
    # ── club-3090 cross-rig (narration / code single-stream, alt_kind=code) ──
    {"model": "Qwen3.6-35B-A3B FP8", "hardware": "2× 3090 24GB", "tp": 2, "link": "pcie",
     "tps_single": 178.5, "tps_multi": 173.7, "alt_kind": "code", "context_k": 262, "source": "club-3090 qwen-35b-a3b-dual (7.81× conc separately)"},
    {"model": "Qwen3.6-27B FP8", "hardware": "2× 3090 24GB", "tp": 2, "link": "pcie",
     "tps_single": 70, "tps_multi": 90, "alt_kind": "code", "context_k": 262, "source": "club-3090 dual.yml baseline"},
    {"model": "Qwen3.6-27B INT4 int8-PTH KV", "hardware": "4× 3090 PCIe", "tp": 4, "link": "pcie",
     "tps_single": 92.5, "tps_multi": 121.9, "alt_kind": "code", "context_k": 262, "source": "club-3090 multi-fast v0.24.0 (int8-PTH = +57%/+65% vs fp8 KV, verify 7/7)"},
    {"model": "Qwen3.6-27B FP8 int8-PTH KV", "hardware": "2× 3090 24GB", "tp": 2, "link": "pcie",
     "tps_single": 82.0, "tps_multi": 104.6, "alt_kind": "code", "context_k": 262, "source": "club-3090 dual-max v0.24.0"},
    {"model": "Qwen3.6-27B W8A8 int8-PTH (experimental)", "hardware": "2× 3090 24GB", "tp": 2, "link": "pcie",
     "tps_single": 76.2, "tps_multi": 96.4, "alt_kind": "code", "context_k": 262, "source": "club-3090 v0.24.0 EXPERIMENTAL — prefill +17-51% vs fp8, TTFT 122ms, quality tied (activation-quant ~free)"},
    {"model": "Qwen3.6-27B (NVLink baseline)", "hardware": "2× 3090 NVLink", "tp": 2, "link": "nvlink",
     "tps_single": 67.2, "tps_multi": 86.1, "alt_kind": "code", "context_k": 262, "source": "club-3090 dual.yml NVLink = +56-60% vs SAME rig PHB/no-P2P (42.7/53.5)"},
    {"model": "Qwen3.6-27B DFlash (NVLink)", "hardware": "2× 3090 NVLink", "tp": 2, "link": "nvlink",
     "tps_single": 103, "tps_multi": 167, "alt_kind": "code", "context_k": 188, "source": "club-3090 dual-nvlink-dflash-noviz"},
    {"model": "Qwen3.6-27B DFlash", "hardware": "1× 3090 24GB", "tp": 1, "link": "single",
     "tps_single": 50.2, "tps_multi": 99.7, "alt_kind": "code", "context_k": 160, "source": "club-3090 single-card DEFAULT (ships 102K)"},
    {"model": "Qwen3.6-27B FP8", "hardware": "2× 5090 PCIe5", "tp": 2, "link": "pcie",
     "tps_single": 153.4, "tps_multi": 196.9, "alt_kind": "code", "context_k": 262, "source": "club-3090 dual.yml 5090 (8-pack quality 110/150)"},
    {"model": "Gemma-4 31B bf16 MTP", "hardware": "2× 3090 24GB", "tp": 2, "link": "pcie",
     "tps_single": 118.8, "tps_multi": 154.3, "alt_kind": "code", "context_k": 131, "source": "club-3090 gemma-bf16-mtp"},
    {"model": "Gemma-4 31B int8-mtp", "hardware": "1× 5090 32GB", "tp": 1, "link": "single",
     "tps_single": 159.7, "tps_multi": 215.1, "alt_kind": "code", "context_k": 131, "source": "club-3090 5090 single-card"},
]

# Single-card "escape hatch" lanes for when TP=2 won't fit (or you only have one
# card). Real llama.cpp / ik-llama configs with measured decode TPS + max ctx.
_SINGLE_CARD: dict[str, list[dict[str, Any]]] = {
    "qwen3.6-27b": [
        {"engine": "ik-llama", "config": "iq4ks-two-stage", "tps_single": 59.4, "tps_code": 97.8, "context_k": 200,
         "vram_gb": 22.3, "note": "code-optimized two-stage; single 3090", "source": "club-3090 BENCHMARKS.md"},
        {"engine": "ik-llama", "config": "iq4ks-mtp", "tps_single": 59.7, "tps_code": 68.8, "context_k": 200,
         "vram_gb": 22.3, "note": "MTP draft; 1109 tok/s prefill", "source": "club-3090"},
        {"engine": "llama.cpp", "config": "mtp Q4_K_M", "tps_single": 50.3, "tps_code": 58.9, "context_k": 200,
         "vram_gb": 22.3, "note": "mainline llama.cpp; 1025 tok/s prefill", "source": "club-3090"},
        {"engine": "beellama", "config": "dflash", "tps_single": 50.2, "tps_code": 99.7, "context_k": 160,
         "vram_gb": 23.0, "note": "PROMOTED single-card DEFAULT (ships 102K ctx); fastest single-card code path", "source": "club-3090 v0.24.0"},
    ],
    # Gemma-4 (26B-A4B / 31B): a 24GB single-card vLLM OOMs on SM86 (Ampere) —
    # the honest single-card path is a 32GB card (5090) or a llama.cpp GGUF.
    "gemma-4": [
        {"engine": "vllm", "config": "int8-mtp", "tps_single": 159.7, "tps_code": 215.1, "context_k": 131,
         "vram_gb": 32, "note": "needs a 32GB card (5090); on 24GB single-card vLLM OOMs (SM86) — fall back to a llama.cpp GGUF Q4 there", "source": "club-3090"},
    ],
}
# alias the registry keys to the escape-hatch table
_SINGLE_CARD_ALIASES = {
    "qwen3.6-27b-int4": "qwen3.6-27b", "qwen3.6-27b-int4-autoround": "qwen3.6-27b",
    "qwen3.6-27b": "qwen3.6-27b",
    "gemma-4-31b-awq": "gemma-4", "gemma-4-26b-a4b-awq": "gemma-4",
    "gemma-4-31b": "gemma-4", "gemma-4-26b": "gemma-4", "gemma-4": "gemma-4",
}


def single_card_alternatives(model: str) -> list[dict[str, Any]]:
    """Real single-card escape-hatch configs for a model (empty if none known).

    Surfaced when TP=2 does not fit — a 24GB card can still run 27B via
    llama.cpp / ik-llama at 200K context. 35B / Gemma-4 need a 32GB card
    (5090) or the llama.cpp fallback; 24GB single-card vLLM OOMs (SM86)."""
    key = _SINGLE_CARD_ALIASES.get(str(model).lower().strip())
    return [dict(a) for a in _SINGLE_CARD.get(key or "", [])]


def topology_note(tp: int) -> dict[str, Any] | None:
    """Informational NVLink-vs-PCIe throughput note for multi-GPU (TP≥2).

    This is a CITED throughput observation, NOT applied to the VRAM/context
    estimate — interconnect changes tokens/s, not the byte budget."""
    if int(tp) < 2:
        return None
    return {
        "level": "info",
        "applies_to_estimate": False,
        "text": (
            "TP≥2 throughput scales with interconnect. Biggest lift vs a "
            "no-P2P board (PHB): NVLink measured ~+56–60% (club-3090 27B "
            "dual.yml 67/86 NVLink vs 43/54 PHB, same rig). Vs P2P-capable "
            "PCIe the gap is smaller (~+15%). Patched-P2P (no NVLink) recovers "
            "~+10–14%. Affects tokens/s only, not the VRAM budget."
        ),
        "source": "club-3090 27B dual.yml NVLink-vs-PHB toggle (measured, 2026-06-26)",
    }


__all__ = [
    "KV_DTYPE_BYTES", "MEASURED_REFERENCE", "ModelArch", "arch_from_dict",
    "calibrate_overhead", "estimate", "fit_envelope", "known_models",
    "kv_bytes_per_token", "recommend", "single_card_alternatives",
    "topology_note", "weights_bytes",
]
