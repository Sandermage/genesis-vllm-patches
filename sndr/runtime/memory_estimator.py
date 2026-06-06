# SPDX-License-Identifier: Apache-2.0
"""Static VRAM budget estimator — pre-launch breakdown.

T1.3 / production roadmap §18.3 (Phase 1). Companion to the runtime
`memory_metrics.py` which collects ACTUAL bytes from live buffer pools;
this module computes EXPECTED bytes from a model_config preset before
vLLM boots, so operators can reason about "will this fit in 24 GiB"
without an A/B-launch-and-pray loop.

Five components feed into the estimate:

  1. **Model weights** (after TP shard) — read from `config.json` +
     `safetensors` index, divided by `tensor_parallel_size`.
  2. **KV cache** — analytic formula:
     ``2 × n_layers × n_kv_heads × head_dim × max_len × bytes_per_kv_elem``
     divided by `tp_size`. `bytes_per_kv_elem` honors fp8 (1) / bf16 (2)
     / fp32 (4).
  3. **Activations / scratch** — heuristic floor scaled by hidden_size +
     batched-token count. Phase 1 = constant 2 GiB upper bound until
     Marlin repack estimator (§17.1) lands; flagged with a `confidence`
     marker so the CLI can render "± estimate" honestly.
  4. **CUDA-graph reserve** — heuristic 0.4 GiB * (n_seqs / 2) clamped
     to 0.4–1.5 GiB. Real vLLM allocates a bucket per max_num_seqs.
  5. **Marlin repack scratch** — only if quant=Marlin family. Computed
     from intermediate_size + n_layers when present, falls back to
     1.5 GiB conservative default.

Exposed surface
───────────────
``estimate_for_config(cfg) -> MemoryEstimate``
    The high-level entry point — pass a `ModelConfig` (model_configs
    schema), get a MemoryEstimate dataclass with per-component bytes
    + warnings. Used by the CLI.

``read_model_shape(model_path) -> ModelShape``
    Best-effort reader of `config.json` for the keys we need
    (n_layers, n_heads, n_kv_heads, head_dim, hidden_size,
    intermediate_size). Returns None values when missing.

Author: Sandermage(Sander)-Barzov Aleksandr.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.memory_estimator")


# Bytes per element for each tensor dtype the estimator recognizes.
# `auto` and `unknown` fall back to fp16 (2 bytes) — conservative for
# memory accounting (most quantized models have fp16 KV cache).
_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1,
    "int4": 1,  # packed nibble — 0.5 bytes effective; round up to 1
    "auto": 2,
    "unknown": 2,
}


def _dtype_bytes(name: Optional[str]) -> int:
    if not name:
        return 2
    return _DTYPE_BYTES.get(str(name).lower(), 2)


def _humanize(n: int) -> str:
    """Bytes → human-readable GiB/MiB."""
    n = int(n)
    for unit, power in [("KiB", 10), ("MiB", 20), ("GiB", 30)]:
        if n < (1 << (power + 10)):
            if power == 10:
                return f"{n / 1024:.1f} KiB"
            if power == 20:
                return f"{n / (1 << 20):.0f} MiB"
            return f"{n / (1 << 30):.2f} GiB"
    return f"{n / (1 << 30):.2f} GiB"


# ─── Model shape readers ────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelShape:
    """Subset of `config.json` fields required for KV + weight estimates.

    Any field can be None when the source doesn't expose it (e.g. some
    HF configs omit `head_dim` and we derive it as
    `hidden_size / num_attention_heads`). Estimators degrade
    gracefully: each component returns 0 when its inputs are missing,
    and the warning list surfaces "couldn't estimate X" so the CLI can
    say so honestly instead of pretending.
    """
    model_path: str
    n_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    quant_method: Optional[str] = None
    raw_config: dict[str, Any] = field(default_factory=dict)
    weights_size_bytes: Optional[int] = None  # sum of safetensors


def _scan_safetensors_size(model_dir: Path) -> Optional[int]:
    """Sum the byte size of every `*.safetensors` shard in `model_dir`.

    Doesn't open the files — uses `Path.stat().st_size`. This gives the
    on-disk footprint, which matches the in-memory weight footprint to
    within a few MiB for fp16/bf16/fp8 (no compression). For int4
    AutoRound the on-disk size IS the in-memory size (already packed).
    """
    if not model_dir.is_dir():
        return None
    total = 0
    found = False
    for f in model_dir.rglob("*.safetensors"):
        try:
            total += f.stat().st_size
            found = True
        except OSError:
            continue
    return total if found else None


def read_model_shape(model_path: str) -> ModelShape:
    """Read `config.json` from `model_path` and extract the fields the
    estimator needs. Returns `ModelShape` with None fields when the
    source doesn't have them — never raises."""
    p = Path(model_path).expanduser()
    config_path = p / "config.json"
    raw: dict[str, Any] = {}
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning(
                "[memory_estimator] failed to read %s: %s", config_path, e
            )
    # Some HF configs nest the actual params under `text_config`
    # (multimodal models, qwen3_5/3_6 hybrid). Merge so flat lookups work.
    text_cfg = raw.get("text_config") if isinstance(raw, dict) else None
    if isinstance(text_cfg, dict):
        merged = dict(raw)
        for k, v in text_cfg.items():
            merged.setdefault(k, v)
        raw = merged

    n_layers = raw.get("num_hidden_layers")
    hidden = raw.get("hidden_size")
    n_heads = raw.get("num_attention_heads")
    n_kv_heads = (
        raw.get("num_key_value_heads")
        or raw.get("num_kv_heads")
        or n_heads
    )
    head_dim = raw.get("head_dim")
    if not head_dim and hidden and n_heads:
        try:
            head_dim = int(hidden) // int(n_heads)
        except (TypeError, ValueError, ZeroDivisionError):
            head_dim = None
    intermediate = raw.get("intermediate_size")

    quant_method: Optional[str] = None
    qcfg = raw.get("quantization_config")
    if isinstance(qcfg, dict):
        quant_method = (
            qcfg.get("quant_method") or qcfg.get("method")
        )

    weights_bytes = _scan_safetensors_size(p)

    return ModelShape(
        model_path=str(p),
        n_layers=int(n_layers) if isinstance(n_layers, int) else None,
        hidden_size=int(hidden) if isinstance(hidden, int) else None,
        n_heads=int(n_heads) if isinstance(n_heads, int) else None,
        n_kv_heads=int(n_kv_heads) if isinstance(n_kv_heads, int) else None,
        head_dim=int(head_dim) if isinstance(head_dim, int) else None,
        intermediate_size=(int(intermediate)
                           if isinstance(intermediate, int) else None),
        quant_method=str(quant_method) if quant_method else None,
        raw_config=raw,
        weights_size_bytes=weights_bytes,
    )


# ─── Per-component estimators ───────────────────────────────────────────


def estimate_weights(shape: ModelShape, *, tp_size: int = 1) -> int:
    """Return per-GPU weight bytes after tensor-parallel shard.

    Phase 1 strategy: trust the safetensors total. TP shards weights
    along the hidden dim (most layers) so dividing by `tp_size` is the
    right first-order approximation — within a few hundred MiB of
    actual depending on whether expert weights are sharded across the
    whole TP group or replicated.
    """
    if not shape.weights_size_bytes:
        return 0
    tp = max(int(tp_size), 1)
    return int(shape.weights_size_bytes) // tp


def estimate_kv_cache(
    shape: ModelShape,
    *,
    max_model_len: int,
    max_num_seqs: int = 1,
    kv_dtype: Optional[str] = None,
    tp_size: int = 1,
) -> int:
    """Bytes per GPU for KV cache at `max_model_len` × `max_num_seqs`.

    Formula:
      ``2 × n_layers × n_kv_heads × head_dim × max_len × max_seqs ×
        dtype_bytes / tp_size``

    The factor of 2 covers K + V. When `n_kv_heads` is missing (some
    Qwen3 hybrid configs only declare `num_attention_heads`), falls
    back to that value — over-estimates for GQA models, which is fine
    for budgeting (operator sees a conservative ceiling).
    """
    if not all((shape.n_layers, shape.head_dim, max_model_len)):
        return 0
    n_layers = shape.n_layers
    head_dim = shape.head_dim
    n_kv = shape.n_kv_heads or shape.n_heads or 0
    if n_kv == 0:
        return 0
    elem_bytes = _dtype_bytes(kv_dtype)
    tp = max(int(tp_size), 1)

    total = (
        2 * int(n_layers) * int(n_kv) * int(head_dim)
        * int(max_model_len) * max(int(max_num_seqs), 1) * elem_bytes
    )
    return total // tp


def estimate_activations(
    shape: ModelShape,
    *,
    max_num_batched_tokens: int = 4096,
) -> int:
    """Heuristic activation/scratch budget.

    Phase 1: ``hidden_size × max_num_batched_tokens × 2 (fp16) × 8 (depth)``
    — covers attention scratch + MLP intermediates for a single layer
    held live at once. Capped at 2 GiB so unusually-wide models don't
    skew the total. Real value is workload-dependent; a Marlin repack
    estimator (§17.1) will replace this with a precise bound later.
    """
    if not shape.hidden_size:
        return 512 * (1 << 20)  # 512 MiB conservative floor
    raw = (
        int(shape.hidden_size)
        * max(int(max_num_batched_tokens), 1)
        * 2  # fp16 activation
        * 8  # depth multiplier
    )
    return min(raw, 2 * (1 << 30))


def estimate_cuda_graph_reserve(*, max_num_seqs: int = 2) -> int:
    """Approximate vLLM CUDA-graph capture reserve.

    vLLM captures a graph per max_num_seqs bucket; each bucket holds
    ~200 MiB. Phase 1 uses a flat band: 0.4 GiB at low concurrency,
    ramping to 1.5 GiB as max_num_seqs grows, capped at 1.5 GiB.
    """
    bands = [
        (1, 400 * (1 << 20)),
        (4, 600 * (1 << 20)),
        (8, 900 * (1 << 20)),
        (16, 1200 * (1 << 20)),
    ]
    n = max(int(max_num_seqs), 1)
    last = bands[-1][1]
    for cap, val in bands:
        if n <= cap:
            return val
    return last


def estimate_marlin_scratch(shape: ModelShape) -> int:
    """Marlin repack peak scratch (T1.9 / audit §17.1).

    Replaces the Phase-1 hidden×intermediate heuristic with a
    weights-relative bound documented by the audit:

      ``scratch_peak ≈ max(largest_layer_weight_bytes × 1.5,
                            64 MiB per layer)``

    Rationale: GPTQ / AWQ Marlin repack stages the largest single
    layer's quantized tensor into a fp16 / bf16 buffer for the kernel
    pre-computation (scales, zero-points, packed indices). The peak
    is layer-bounded — repack runs serially per layer — so we don't
    multiply by `n_layers`. Empirically club-3090 #60 OOM bisected to
    ≈1.4× weight footprint of the largest MoE expert block.

    Falls back to a 1.5 GiB conservative default when shape data is
    missing (no config.json, or HF cache lookup), so the estimator
    never silently under-counts.

    Only fires when quant_method indicates Marlin family
    (`marlin`, `gptq`, `awq`, `auto_round`).
    """
    qm = (shape.quant_method or "").lower()
    is_marlin = any(
        m in qm
        for m in ("marlin", "gptq", "awq", "auto_round")
    )
    if not is_marlin:
        return 0

    # Per-layer ceiling: hidden × intermediate × 2 (fp16 staging)
    # represents the largest dense block (gate/up_proj fused). For MoE
    # models this is per-expert; vLLM repacks experts serially so the
    # peak is bounded by ONE expert's footprint. We add an explicit
    # 64 MiB floor because tiny models still incur fixed overhead
    # (scales, zero-points, index buffers) that don't scale with
    # weights.
    MIN_SCRATCH_BYTES = 64 * (1 << 20)

    # Strategy A: use measured weights if available — most accurate.
    # T1.9 (audit §17.1): when weights_size_bytes is known, the audit's
    # `weights × 1.5×` rule applies to the LARGEST LAYER (not the whole
    # model — repack is serial per layer). Approximate the largest
    # layer as `weights / n_layers × 2` (× 2 = expert vs dense margin).
    if (shape.weights_size_bytes
            and shape.n_layers
            and shape.n_layers > 0):
        per_layer_estimate = (
            int(shape.weights_size_bytes) // int(shape.n_layers)
        )
        # 1.5× the weights footprint of one layer is the audit's
        # documented bound. Multiply by 2 to cover MoE expert spread
        # (router weights repack alongside experts).
        scratch = int(per_layer_estimate * 1.5 * 2)
        return max(scratch, MIN_SCRATCH_BYTES)

    # Strategy B: hidden × intermediate fallback — works without
    # safetensors, sufficient for dense (non-MoE) models.
    if shape.hidden_size and shape.intermediate_size:
        raw = (
            int(shape.hidden_size)
            * int(shape.intermediate_size)
            * 2  # fp16 staging
        )
        scratch = int(raw * 1.2)
        return max(scratch, MIN_SCRATCH_BYTES)

    # Strategy C: nothing known — use a conservative 1.5 GiB ceiling.
    return int(1.5 * (1 << 30))


def marlin_scratch_warns(
    shape: ModelShape,
    *,
    free_vram_bytes: int,
    weights_bytes: int,
) -> tuple[bool, str]:
    """Companion check: does the Marlin repack peak fit alongside
    already-allocated weights?

    Used by `sndr memory explain` and `sndr launch` preflight to warn
    operators when the repack peak + committed weights would exceed
    the free VRAM ceiling. Returns (warn, message). When `warn` is
    True, `message` is a single human-readable sentence pointing at
    the specific overage.

    Per the audit (§17.1), this is the check missing in club-3090 #60:
    nightly bumps that swap vllm/torch silently change the repack
    backend's working set; without this check, operators discover the
    OOM only when EngineCore crashes mid-warmup.
    """
    scratch = estimate_marlin_scratch(shape)
    if scratch == 0:
        return False, ""
    needed = scratch + weights_bytes
    if needed <= free_vram_bytes:
        return False, ""
    overage = needed - free_vram_bytes
    overage_gib = overage / (1 << 30)
    return True, (
        f"Marlin repack peak ({scratch / (1 << 30):.2f} GiB) + "
        f"committed weights ({weights_bytes / (1 << 30):.2f} GiB) "
        f"exceeds free VRAM ({free_vram_bytes / (1 << 30):.2f} GiB) "
        f"by {overage_gib:.2f} GiB. Lower gpu_memory_utilization or "
        "switch to a non-Marlin quant_method."
    )


# ─── Aggregate estimate ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryComponent:
    """One row in the budget table."""
    name: str
    bytes_: int
    notes: str = ""
    confidence: str = "high"  # "high" | "medium" | "low"

    @property
    def human(self) -> str:
        return _humanize(self.bytes_)


@dataclass(frozen=True)
class MemoryEstimate:
    """Full budget breakdown for one preset on one GPU."""
    preset_key: str
    model_path: str
    gpu_count: int
    gpu_vram_bytes: int  # per-GPU capacity (operator-supplied)
    components: tuple[MemoryComponent, ...]
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    shape: Optional[ModelShape] = None

    @property
    def total_bytes(self) -> int:
        # Marlin scratch is transient (only during weight load), not committed.
        # Excluded from "committed" total but flagged in warnings.
        return sum(
            c.bytes_ for c in self.components
            if c.name != "Marlin repack scratch (peak)"
        )

    @property
    def utilization(self) -> float:
        if self.gpu_vram_bytes <= 0:
            return 0.0
        return self.total_bytes / self.gpu_vram_bytes


# Default GPU vram lookup by name — used when the operator doesn't
# specify a concrete GPU. These match vendor spec in bytes (× 1024^3).
_GPU_VRAM_DEFAULTS: dict[str, int] = {
    "RTX A5000": 24 * (1 << 30),
    "RTX 3090": 24 * (1 << 30),
    "RTX 4090": 24 * (1 << 30),
    "RTX 6000 Ada": 48 * (1 << 30),
    "RTX 6000 Pro Blackwell": 96 * (1 << 30),
    "A100 40GB": 40 * (1 << 30),
    "A100 80GB": 80 * (1 << 30),
    "H100 80GB": 80 * (1 << 30),
    "H200 141GB": 141 * (1 << 30),
}

DEFAULT_GPU_VRAM_BYTES = 24 * (1 << 30)  # A5000 default


def lookup_gpu_vram(gpu_name: Optional[str]) -> int:
    """Return per-GPU VRAM in bytes for `gpu_name`, falling back to
    24 GiB. Fuzzy-matches partial names ('A5000' → 'RTX A5000').
    """
    if not gpu_name:
        return DEFAULT_GPU_VRAM_BYTES
    name = gpu_name.lower()
    for full, bytes_ in _GPU_VRAM_DEFAULTS.items():
        if full.lower() == name:
            return bytes_
    # Substring match (e.g. operator says "a5000")
    for full, bytes_ in _GPU_VRAM_DEFAULTS.items():
        if full.lower() in name or name in full.lower():
            return bytes_
    return DEFAULT_GPU_VRAM_BYTES


def estimate_for_config(cfg: Any) -> MemoryEstimate:
    """High-level entry: build a `MemoryEstimate` for one ModelConfig.

    Reads:
      - `cfg.model_path` for the on-disk layout
      - `cfg.hardware.gpu_match_keys[0]` for the GPU model
      - `cfg.hardware.n_gpus` for tensor parallelism
      - `cfg.max_model_len`, `cfg.max_num_seqs`, `cfg.max_num_batched_tokens`
      - `cfg.kv_cache_dtype`
    """
    warnings: list[str] = []
    recommendations: list[str] = []

    model_path = getattr(cfg, "model_path", "") or ""
    shape = read_model_shape(model_path) if model_path else None

    hw = getattr(cfg, "hardware", None)
    n_gpus = int(getattr(hw, "n_gpus", 1) or 1) if hw else 1
    gpu_match_keys = list(getattr(hw, "gpu_match_keys", []) or []) if hw else []
    primary_gpu = gpu_match_keys[0] if gpu_match_keys else None
    per_gpu_vram = (
        int(getattr(hw, "min_vram_per_gpu_mib", 0) or 0) * (1 << 20)
        if hw else 0
    )
    if per_gpu_vram == 0:
        per_gpu_vram = lookup_gpu_vram(primary_gpu)

    tp_size = max(n_gpus, 1)
    max_len = int(getattr(cfg, "max_model_len", 0) or 0)
    max_seqs = int(getattr(cfg, "max_num_seqs", 1) or 1)
    max_batch_tok = int(getattr(cfg, "max_num_batched_tokens", 4096) or 4096)
    kv_dtype = getattr(cfg, "kv_cache_dtype", None)

    components: list[MemoryComponent] = []

    if shape and shape.weights_size_bytes:
        weight_bytes = estimate_weights(shape, tp_size=tp_size)
        components.append(MemoryComponent(
            "Model weights (after TP shard)",
            weight_bytes,
            notes=(f"on-disk total {_humanize(shape.weights_size_bytes)} "
                   f"÷ TP={tp_size}"),
            confidence="high",
        ))
    else:
        components.append(MemoryComponent(
            "Model weights (after TP shard)",
            0,
            notes="(safetensors not found — operator-supplied path "
                  "may be HF-cached or remote)",
            confidence="low",
        ))
        warnings.append(
            f"Model path {model_path!r} has no readable safetensors; "
            "weight estimate is 0 — provide a local checkout for "
            "accurate budgeting."
        )

    if shape and shape.n_layers and shape.head_dim:
        kv_bytes = estimate_kv_cache(
            shape,
            max_model_len=max_len,
            max_num_seqs=max_seqs,
            kv_dtype=kv_dtype,
            tp_size=tp_size,
        )
        components.append(MemoryComponent(
            f"KV cache ({max_len // 1024}K ctx × {max_seqs} seq, "
            f"{kv_dtype or 'fp16'})",
            kv_bytes,
            notes=(f"2 × {shape.n_layers}L × "
                   f"{shape.n_kv_heads or shape.n_heads}h × "
                   f"{shape.head_dim}d × {max_len} ÷ TP={tp_size}"),
            confidence="high",
        ))
    else:
        components.append(MemoryComponent(
            "KV cache",
            0,
            notes="(model shape unavailable — config.json missing fields)",
            confidence="low",
        ))
        warnings.append(
            "KV cache estimate is 0 — config.json missing num_hidden_layers / "
            "head_dim. Add them or estimator can't budget KV."
        )

    components.append(MemoryComponent(
        "Activations / scratch (heuristic)",
        estimate_activations(
            shape or ModelShape(model_path=model_path),
            max_num_batched_tokens=max_batch_tok,
        ),
        notes=f"hidden × {max_batch_tok} × fp16 × depth (capped 2 GiB)",
        confidence="medium",
    ))

    components.append(MemoryComponent(
        "CUDA-graph reserve",
        estimate_cuda_graph_reserve(max_num_seqs=max_seqs),
        notes=f"per-bucket capture cost (max_num_seqs={max_seqs})",
        confidence="medium",
    ))

    if shape:
        marlin_bytes = estimate_marlin_scratch(shape)
        if marlin_bytes > 0:
            components.append(MemoryComponent(
                "Marlin repack scratch (peak)",
                marlin_bytes,
                notes="transient — released after warmup",
                confidence="medium",
            ))

            # T1.9 (audit §17.1): warn when peak repack + weights exceed
            # the per-GPU cap. This was the silent OOM in club-3090 #60.
            committed_weights = next(
                (c.bytes_ for c in components
                 if c.name.startswith("Model weights")),
                0,
            )
            warn, msg = marlin_scratch_warns(
                shape,
                free_vram_bytes=per_gpu_vram - estimate_activations(
                    shape, max_num_batched_tokens=max_batch_tok,
                ) - estimate_cuda_graph_reserve(max_num_seqs=max_seqs),
                weights_bytes=committed_weights,
            )
            if warn:
                warnings.append(msg)

    # Compose recommendations based on utilization.
    estimate = MemoryEstimate(
        preset_key=getattr(cfg, "key", "(unknown)"),
        model_path=model_path,
        gpu_count=n_gpus,
        gpu_vram_bytes=per_gpu_vram,
        components=tuple(components),
        warnings=tuple(warnings),
        recommendations=(),
        shape=shape,
    )

    util = estimate.utilization
    if util > 0.95:
        recommendations.append(
            f"⚠ Very tight budget ({util * 100:.0f}% of {_humanize(per_gpu_vram)}). "
            "Consider lowering max_model_len or enabling fp8 KV cache."
        )
    elif util > 0.85:
        recommendations.append(
            f"Budget at {util * 100:.0f}% — leave room for fragmentation; "
            "drop gpu_memory_utilization to 0.88 if first-boot OOMs occur."
        )
    elif util > 0.0 and util < 0.6:
        recommendations.append(
            f"Budget only {util * 100:.0f}% utilized — you can raise "
            "max_model_len or max_num_seqs for more throughput."
        )

    # Build the final estimate object with recommendations populated.
    return MemoryEstimate(
        preset_key=estimate.preset_key,
        model_path=estimate.model_path,
        gpu_count=estimate.gpu_count,
        gpu_vram_bytes=estimate.gpu_vram_bytes,
        components=estimate.components,
        warnings=estimate.warnings,
        recommendations=tuple(recommendations),
        shape=estimate.shape,
    )


def render_waterfall(estimate: MemoryEstimate, *, use_color: bool = False) -> str:
    """Render the budget breakdown as a human-readable waterfall string.

    `use_color` enables ANSI escape codes — the CLI passes `True` for
    interactive TTY output, `False` for piped/captured output.
    """
    GREEN = "\033[32m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    RED = "\033[31m" if use_color else ""
    DIM = "\033[2m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    lines: list[str] = []
    lines.append(f"Memory budget for preset: {estimate.preset_key}")
    primary = (estimate.shape.model_path if estimate.shape
               else estimate.model_path) or "(no model path)"
    lines.append(f"Model: {primary}")
    lines.append(
        f"GPU count: {estimate.gpu_count} × "
        f"{_humanize(estimate.gpu_vram_bytes)} per GPU"
    )
    lines.append("─" * 60)

    width = max((len(c.name) for c in estimate.components), default=20)
    width = max(width, 30)

    for c in estimate.components:
        marker = ""
        if c.confidence == "low":
            marker = f"  {DIM}(low confidence){RESET}"
        elif c.confidence == "medium":
            marker = f"  {DIM}(±25%){RESET}"
        lines.append(
            f"  {c.name.ljust(width)}  {c.human:>10}{marker}"
        )

    lines.append("─" * 60)

    util_pct = estimate.utilization * 100
    color = GREEN if util_pct < 85 else (YELLOW if util_pct < 95 else RED)
    lines.append(
        f"  Subtotal (committed)        {_humanize(estimate.total_bytes):>10}"
        f"  /  {_humanize(estimate.gpu_vram_bytes)}  "
        f"({color}{util_pct:.0f}%{RESET})"
    )

    headroom = max(estimate.gpu_vram_bytes - estimate.total_bytes, 0)
    lines.append(f"  Headroom for fragmentation  {_humanize(headroom):>10}")

    if estimate.warnings:
        lines.append("─" * 60)
        for w in estimate.warnings:
            lines.append(f"{YELLOW}⚠ {w}{RESET}")

    if estimate.recommendations:
        lines.append("─" * 60)
        for r in estimate.recommendations:
            lines.append(f"  {r}")

    return "\n".join(lines)


__all__ = [
    "ModelShape",
    "MemoryComponent",
    "MemoryEstimate",
    "DEFAULT_GPU_VRAM_BYTES",
    "lookup_gpu_vram",
    "read_model_shape",
    "estimate_weights",
    "estimate_kv_cache",
    "estimate_activations",
    "estimate_cuda_graph_reserve",
    "estimate_marlin_scratch",
    "marlin_scratch_warns",
    "estimate_for_config",
    "render_waterfall",
]
