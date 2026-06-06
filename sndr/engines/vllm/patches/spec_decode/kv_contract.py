# SPDX-License-Identifier: Apache-2.0
"""kv_contract — SpecDecode KV sharing compatibility model.

Library code. Does NOT monkey-patch anything. Two things live here:

  (1) ``KVContract`` — a frozen dataclass that captures the shape,
      layout, dtype, attention numerics, RoPE state, quantization,
      and backend of one Attention module. Built by
      ``extract_contract`` from a live vLLM Attention instance plus
      its parent self_attn module.

  (2) ``Verdict`` (enum) + ``compare_contracts(src, dst)`` — produces
      a structured compatibility verdict between two contracts, with
      a list of human-readable divergences.

The verdict vocabulary intentionally includes a state for
"structurally compatible with adapter but acceptance unverified" so
that downstream code (safety_guard, bridge factory) can refuse to
silently enable MTP on configurations that pass shape checks but
have not been runtime-validated.

Provenance: extracted from
``integrations/spec_decode/pn271_kv_contract_audit.py``
2026-05-20 per architectural directive (PN273). Relocated from
``integrations/gemma4/`` 2026-05-21 (Phase 3 bucket 1).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("genesis.spec_decode.kv_contract")


# ----------------------- Verdict vocabulary -----------------------

class Verdict(str, enum.Enum):
    """Compatibility class between source (target layer) and
    destination (drafter layer) Attention contracts.

    Ordering (worst to best for "is functional speedup expected"):

      UNSUPPORTED:
        Fundamentally incompatible — different head_size, divergent
        attention numerics (scale ratio, RoPE base, soft_cap),
        impossible to bridge.

      ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED:
        Shapes and numerics CAN be adapted (any combination of LAYOUT,
        GQA, DEQUANT), but the bridge has not produced a non-zero
        runtime acceptance gate signal for THIS exact model+pin
        combination. Safe to ALLOW only with explicit operator
        opt-in.

      DEQUANT_REQUIRED:
        Source cache is quantized (turboquant_*, fp8, etc.); dest
        expects native bf16/fp16. Bridge must dequantize on read.
        More expensive but mechanically possible.

      LAYOUT_ADAPTER:
        Source HND vs dest NHD (or vice versa). Single axis swap.
        No numeric change.

      GQA_REPEAT:
        Source has fewer kv_heads than dest by an integer factor.
        repeat_interleave on the heads axis.

      COMPOSITE_ADAPTER:
        Two or more of {LAYOUT, GQA, DEQUANT} required at the same
        time. Bridge composes the adapters in order.

      EXACT_COPY:
        Bit-identical shape/layout/dtype/numerics. Drop-in.
    """
    EXACT_COPY = "EXACT_COPY"
    LAYOUT_ADAPTER = "LAYOUT_ADAPTER"
    GQA_REPEAT = "GQA_REPEAT"
    DEQUANT_REQUIRED = "DEQUANT_REQUIRED"
    COMPOSITE_ADAPTER = "COMPOSITE_ADAPTER"
    ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED = (
        "ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED"
    )
    # NEW (PN271b 2026-05-20) — caught a blind spot exposed by β:
    # source storage was bf16 native, but drafter's impl was the
    # TurboQuant kernel (expects TQ-packed bytes). Pre-existing
    # shape/layout/dtype-real audit said EXACT_COPY because real
    # tensor dtype matched. The consumer-kernel-vs-source-storage
    # contract is a separate axis that must be checked explicitly.
    KERNEL_STORAGE_DTYPE_MISMATCH = "KERNEL_STORAGE_DTYPE_MISMATCH"
    KERNEL_LAYOUT_CONTRACT_MISMATCH = "KERNEL_LAYOUT_CONTRACT_MISMATCH"
    # NEW (C4 2026-05-20) — a matching functional_artifact has been
    # found. The contract is structurally non-trivial but a bench
    # receipt proves it produces measured non-regressive (or net-
    # positive on declared allowed_workloads) performance. Caller can
    # admit this verdict with just the structural opt-in env, no
    # FUNCTIONAL_UNKNOWN.
    FUNCTIONALLY_VALIDATED = "FUNCTIONALLY_VALIDATED"
    UNSUPPORTED = "UNSUPPORTED"


# ----------------------- KVContract dataclass -----------------------

@dataclass(frozen=True)
class KVContract:
    """One Attention module's K/V contract.

    Built by ``extract_contract``. Frozen — comparisons are
    side-effect free and cacheable.
    """
    # Identity
    layer_full_name: str
    self_attn_class: str
    inner_attn_class: str

    # Shape
    num_kv_heads: int | None
    num_heads: int | None
    head_size: int | None

    # KV cache
    kv_cache_shape: tuple[int, ...] | None
    kv_cache_layout: str  # 'HND' / 'NHD' / 'unknown'
    kv_cache_dtype_real: str | None  # actual tensor dtype
    kv_cache_dtype_decl: str | None  # declared (e.g., 'auto', 'turboquant_4bit_nc')
    block_size: int | None
    sliding_window: int | None

    # Attention numerics
    scale: float | None
    logits_soft_cap: float | None

    # Q / K normalization weight norms (None if module absent)
    q_norm_weight_norm: float | None
    k_norm_weight_norm: float | None

    # RoPE
    rope_class: str | None
    rope_base: float | None
    rope_max_position_embeddings: int | None
    rope_rotary_dim: int | None

    # Backend
    impl_class: str | None
    attn_backend_class: str | None
    kv_sharing_target_layer_name: str | None

    # Quantization
    quant_kind: str | None  # 'native' / 'turboquant' / 'fp8' / ...

    # Consumer-kernel contract (PN271b — what the impl class assumes
    # about the cache it is going to read/write). Independent of what
    # is physically stored — the mismatch is exactly the blind spot
    # that β exposed.
    kernel_expects_quantized: bool | None = None
    kernel_expected_layout: str | None = None  # 'HND' / 'NHD' / 'unknown'

    # Projections present (True/False per name)
    projections_present: dict[str, bool] = field(default_factory=dict)


# ----------------------- Helpers -----------------------

def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        v = getattr(obj, name, default)
        if v == "<absent>":
            return default
        return v
    except Exception:
        return default


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _weight_norm(mod: Any) -> float | None:
    try:
        w = getattr(mod, "weight", None)
        if w is None:
            return None
        return float(w.float().norm().item())
    except Exception:
        return None


def _classify_layout(
    shape: tuple[int, ...] | None,
    dtype: Any | None = None,
) -> str:
    """Classify a KV cache shape into a layout family.

    Pure shape (+ optional dtype) inspection — no backend handle. Used
    by ``extract_contract`` to populate ``KVContract.kv_cache_layout``.

    K.1.R.R.3 (2026-05-29): extended to recognize TurboQuant's 4-dim
    packed layout (``TQ_PACKED``) and 3-dim Mamba state tensors
    (``MAMBA_STATE``). Previously these fell through to ``"unknown"``,
    masking real configuration in audit reports.

    Disambiguation note (HND vs NHD): when ``shape[0] == 2`` and
    ``shape[1] == 2`` (a 2-block warmup cache), the shape alone
    cannot distinguish layouts. This routine picks ``HND``, matching
    the pre-#42095 convention. ``layout_introspect.block_dim_of``
    with a backend handle is unambiguous (sentinel num_blocks).
    """
    if shape is None or len(shape) < 2:
        return "unknown"

    # TurboQuant packed: 4-D + uint8 + slot_size dim > 2.
    # K and V share a single byte slot per (block, position, head);
    # there is no leading 2-axis to split.
    if len(shape) == 4 and int(shape[3]) > 2:
        try:
            import torch
            if dtype == torch.uint8:
                return "TQ_PACKED"
        except ImportError:
            pass

    # Mamba state: 3-D, num_blocks-first, downstream dims encode
    # the per-state shape spec (conv state, temporal, SSM).
    if len(shape) == 3:
        return "MAMBA_STATE"

    # HND / NHD via the dim-with-size-2 convention.
    if int(shape[0]) == 2:
        return "HND"
    if int(shape[1]) == 2:
        return "NHD"
    return "unknown"


def _classify_quant(decl: str | None) -> str:
    if decl is None:
        return "native"
    s = str(decl).lower()
    if "turboquant" in s:
        return "turboquant"
    if "fp8" in s:
        return "fp8"
    if "quant" in s and s != "auto":
        return "quantized"
    return "native"


def _kernel_expects_quantized(impl_class: str | None) -> bool | None:
    """Decide from impl_class name whether the consumer kernel will
    INTERPRET cache bytes as quantized.

    None = unknown (impl_class missing). Used by the contract check
    only when both sides resolve to True/False.
    """
    if impl_class is None:
        return None
    s = impl_class.lower()
    if "turboquant" in s:
        return True
    if "fp8" in s:
        return True
    # Native impls.
    if "flashattn" in s or "flashattention" in s:
        return False
    if "tritonattn" in s or "tritonattention" in s:
        return False
    return None


def _kernel_expected_layout(impl_class: str | None) -> str | None:
    """Decide from impl_class name the layout the kernel expects.

    None = unknown. Used by the contract check only when both sides
    resolve to a concrete layout.
    """
    if impl_class is None:
        return None
    s = impl_class.lower()
    if "flashattn" in s or "flashattention" in s:
        return "HND"
    if "tritonattn" in s or "tritonattention" in s:
        return "NHD"
    if "turboquant" in s:
        # TQ overlay uses NHD per PR #42637.
        return "NHD"
    return None


def _walk_attn_kv_cache(inner_attn: Any) -> tuple[
        tuple[int, ...] | None, str | None, Any | None]:
    """Pull live kv_cache shape+dtype from bound Attention module.

    Returns (shape, dtype_str, dtype). All three are None if absent.

    K.1.R.R.3 (2026-05-29): also returns the raw torch dtype object
    so callers (specifically ``_classify_layout``) can match against
    ``torch.uint8`` without re-resolving the live tensor.
    """
    try:
        kv = getattr(inner_attn, "kv_cache", None)
        if kv is None:
            return None, None, None
        return tuple(kv.shape), str(kv.dtype), kv.dtype
    except Exception:
        return None, None, None


def extract_contract(
    self_attn: Any,
    layer_full_name: str | None = None,
) -> KVContract:
    """Build a KVContract from a live self_attn module (Gemma4-style:
    has .attn = vllm.Attention).

    Works for any model where the inner Attention layer exposes
    standard attributes (num_kv_heads, scale, kv_sharing_target_layer_name).
    """
    inner = getattr(self_attn, "attn", None)
    self_attn_class = type(self_attn).__qualname__
    inner_attn_class = (
        type(inner).__qualname__ if inner is not None else "<None>"
    )

    # Shape
    num_kv_heads = _to_int(_safe_attr(inner, "num_kv_heads"))
    num_heads = _to_int(_safe_attr(inner, "num_heads"))
    head_size = _to_int(
        _safe_attr(inner, "head_size") or _safe_attr(inner, "head_dim")
    )

    # KV cache
    kv_shape, kv_dtype_real, kv_dtype_obj = _walk_attn_kv_cache(inner)
    kv_layout = _classify_layout(kv_shape, kv_dtype_obj)
    kv_dtype_decl = _safe_attr(inner, "kv_cache_dtype")
    block_size = None
    if kv_shape is not None and len(kv_shape) >= 3:
        # NHD: (num_blocks, 2, block_size, kv_heads, head_size)
        # HND: (2, num_blocks, block_size, kv_heads, head_size)
        block_size = int(kv_shape[2])
    sliding_window = _to_int(_safe_attr(inner, "sliding_window"))

    # Numerics
    scale = _to_float(_safe_attr(inner, "scale"))
    logits_soft_cap = _to_float(_safe_attr(inner, "logits_soft_cap"))

    # RoPE
    rope = getattr(self_attn, "rotary_emb", None)
    rope_class = type(rope).__qualname__ if rope is not None else None
    rope_base = _to_float(_safe_attr(rope, "base"))
    rope_max_pos = _to_int(_safe_attr(rope, "max_position_embeddings"))
    rope_rotary_dim = _to_int(_safe_attr(rope, "rotary_dim"))

    # Q / K norm weight norms
    q_norm_n = _weight_norm(getattr(self_attn, "q_norm", None))
    k_norm_n = _weight_norm(getattr(self_attn, "k_norm", None))

    # Backend
    impl = _safe_attr(inner, "impl")
    impl_class = (
        type(impl).__qualname__
        if impl is not None and impl is not False else None
    )
    attn_backend = _safe_attr(inner, "attn_backend")
    attn_backend_class = (
        attn_backend.__qualname__
        if (attn_backend is not None and hasattr(attn_backend, "__qualname__"))
        else (type(attn_backend).__qualname__
              if attn_backend is not None else None)
    )

    kv_sharing = _safe_attr(inner, "kv_sharing_target_layer_name")
    if kv_sharing in ("<absent>", "None"):
        kv_sharing = None

    quant_kind = _classify_quant(kv_dtype_decl)

    kernel_expects_quantized = _kernel_expects_quantized(impl_class)
    kernel_expected_layout = _kernel_expected_layout(impl_class)

    projections_present = {
        name: getattr(self_attn, name, None) is not None
        for name in ("q_proj", "k_proj", "v_proj", "qkv_proj", "kv_proj",
                     "o_proj")
    }

    return KVContract(
        layer_full_name=layer_full_name or "<unknown>",
        self_attn_class=self_attn_class,
        inner_attn_class=inner_attn_class,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        head_size=head_size,
        kv_cache_shape=kv_shape,
        kv_cache_layout=kv_layout,
        kv_cache_dtype_real=kv_dtype_real,
        kv_cache_dtype_decl=kv_dtype_decl,
        block_size=block_size,
        sliding_window=sliding_window,
        scale=scale,
        logits_soft_cap=logits_soft_cap,
        q_norm_weight_norm=q_norm_n,
        k_norm_weight_norm=k_norm_n,
        rope_class=rope_class,
        rope_base=rope_base,
        rope_max_position_embeddings=rope_max_pos,
        rope_rotary_dim=rope_rotary_dim,
        impl_class=impl_class,
        attn_backend_class=attn_backend_class,
        kv_sharing_target_layer_name=kv_sharing,
        quant_kind=quant_kind,
        kernel_expects_quantized=kernel_expects_quantized,
        kernel_expected_layout=kernel_expected_layout,
        projections_present=projections_present,
    )


# ----------------------- Comparison -----------------------

def compare_contracts(
    src: KVContract, dst: KVContract,
    *,
    scale_tolerance: float = 0.01,
    require_functional_gate: bool = True,
) -> tuple[Verdict, list[str], dict[str, Any]]:
    """Compare a target-side contract (src) against a drafter-side
    contract (dst). Returns (verdict, divergences, adapter_hints).

    adapter_hints carries actionable info, e.g.:
      {"gqa_repeat": 4, "src_layout": "NHD", "dst_layout": "HND",
       "block_size_src": 64, "block_size_dst": 16}

    When require_functional_gate=True (default for production), any
    non-EXACT verdict is downgraded to
    ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED unless the caller has
    flagged the (src,dst) pair as functionally validated.
    """
    divergences: list[str] = []
    flags: set[Verdict] = set()
    hints: dict[str, Any] = {}

    # head_size — fundamental
    if (src.head_size is not None and dst.head_size is not None
            and src.head_size != dst.head_size):
        divergences.append(
            f"head_size mismatch src={src.head_size} dst={dst.head_size}"
        )
        flags.add(Verdict.UNSUPPORTED)

    # num_kv_heads — GQA acceptable if integer divisor
    if (src.num_kv_heads is not None and dst.num_kv_heads is not None):
        if src.num_kv_heads == dst.num_kv_heads:
            pass
        elif (dst.num_kv_heads > src.num_kv_heads
              and dst.num_kv_heads % src.num_kv_heads == 0):
            repeat = dst.num_kv_heads // src.num_kv_heads
            divergences.append(
                f"num_kv_heads src={src.num_kv_heads} dst={dst.num_kv_heads} "
                f"(GQA repeat={repeat})"
            )
            flags.add(Verdict.GQA_REPEAT)
            hints["gqa_repeat"] = repeat
        else:
            divergences.append(
                f"num_kv_heads not a divisor: src={src.num_kv_heads} "
                f"dst={dst.num_kv_heads}"
            )
            flags.add(Verdict.UNSUPPORTED)

    # Scale — tolerate within 1%
    if (src.scale is not None and dst.scale is not None
            and src.scale != 0 and dst.scale != 0):
        ratio = dst.scale / src.scale
        if not (1.0 - scale_tolerance <= ratio <= 1.0 + scale_tolerance):
            expected = (1.0 / math.sqrt(dst.head_size)
                        if dst.head_size else None)
            divergences.append(
                f"scale mismatch src={src.scale} dst={dst.scale} "
                f"(ratio={ratio:.4f}; 1/sqrt(head)={expected})"
            )
            flags.add(Verdict.UNSUPPORTED)

    # Soft-cap
    if src.logits_soft_cap != dst.logits_soft_cap:
        divergences.append(
            f"logits_soft_cap src={src.logits_soft_cap} "
            f"dst={dst.logits_soft_cap}"
        )
        flags.add(Verdict.UNSUPPORTED)

    # RoPE base
    if (src.rope_base is not None and dst.rope_base is not None
            and src.rope_base != dst.rope_base):
        divergences.append(
            f"rope_base src={src.rope_base} dst={dst.rope_base}"
        )
        flags.add(Verdict.UNSUPPORTED)

    # Layout
    if (src.kv_cache_layout and dst.kv_cache_layout
            and src.kv_cache_layout != dst.kv_cache_layout
            and "unknown" not in (src.kv_cache_layout, dst.kv_cache_layout)):
        divergences.append(
            f"kv_cache layout src={src.kv_cache_layout} "
            f"dst={dst.kv_cache_layout}"
        )
        flags.add(Verdict.LAYOUT_ADAPTER)
        hints["src_layout"] = src.kv_cache_layout
        hints["dst_layout"] = dst.kv_cache_layout

    # Quantization
    if (src.quant_kind and src.quant_kind != "native"
            and dst.quant_kind == "native"):
        divergences.append(
            f"src quantized ({src.quant_kind}); dst native ({dst.quant_kind})"
        )
        flags.add(Verdict.DEQUANT_REQUIRED)

    # (PN271b) Consumer-kernel-vs-source-storage contract.
    # If the destination's kernel will INTERPRET reads as quantized
    # but the source's declared dtype is native (or vice versa),
    # the bytes will be misread. Same risk applies for layout
    # expectations.
    if (dst.kernel_expects_quantized is True
            and src.quant_kind == "native"):
        divergences.append(
            f"KERNEL_STORAGE_DTYPE_MISMATCH: dst.impl={dst.impl_class!r} "
            f"expects quantized bytes; src.quant_kind={src.quant_kind} "
            f"(declared dtype={src.kv_cache_dtype_decl!r})"
        )
        flags.add(Verdict.KERNEL_STORAGE_DTYPE_MISMATCH)
    elif (dst.kernel_expects_quantized is False
            and src.quant_kind not in (None, "native")):
        divergences.append(
            f"KERNEL_STORAGE_DTYPE_MISMATCH: dst.impl={dst.impl_class!r} "
            f"expects native bytes; src.quant_kind={src.quant_kind} "
            f"(declared dtype={src.kv_cache_dtype_decl!r})"
        )
        flags.add(Verdict.KERNEL_STORAGE_DTYPE_MISMATCH)

    if (dst.kernel_expected_layout is not None
            and src.kv_cache_layout not in ("unknown", None)
            and dst.kernel_expected_layout != src.kv_cache_layout):
        divergences.append(
            f"KERNEL_LAYOUT_CONTRACT_MISMATCH: dst.impl={dst.impl_class!r} "
            f"expects {dst.kernel_expected_layout}; src.kv_cache_layout="
            f"{src.kv_cache_layout}"
        )
        flags.add(Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH)

    # Block size hints (always include if known; bridge needs it)
    if src.block_size is not None:
        hints["src_block_size"] = src.block_size
    if dst.block_size is not None:
        hints["dst_block_size"] = dst.block_size

    # Aggregate. Kernel/storage mismatches are NOT mere "adapter
    # required" — they imply the consumer will misread bytes. Promote
    # to UNSUPPORTED at the structural level so the safety guard
    # blocks them by default. Operators can still override via the
    # FUNCTIONAL_UNKNOWN env if they accept the risk.
    if Verdict.UNSUPPORTED in flags:
        verdict = Verdict.UNSUPPORTED
    elif (Verdict.KERNEL_STORAGE_DTYPE_MISMATCH in flags
            or Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH in flags):
        # Conservative: the consumer kernel will misread bytes.
        verdict = (Verdict.KERNEL_STORAGE_DTYPE_MISMATCH
                   if Verdict.KERNEL_STORAGE_DTYPE_MISMATCH in flags
                   else Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH)
        # Still expose composite if other classes also present
        other_adapter = [
            f for f in flags
            if f not in (Verdict.KERNEL_STORAGE_DTYPE_MISMATCH,
                         Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH,
                         Verdict.UNSUPPORTED)
        ]
        if other_adapter:
            hints["kernel_mismatch_with_adapters"] = [
                f.value for f in other_adapter
            ]
    elif len([f for f in flags if f != Verdict.UNSUPPORTED]) >= 2:
        verdict = Verdict.COMPOSITE_ADAPTER
    elif Verdict.DEQUANT_REQUIRED in flags:
        verdict = Verdict.DEQUANT_REQUIRED
    elif Verdict.LAYOUT_ADAPTER in flags:
        verdict = Verdict.LAYOUT_ADAPTER
    elif Verdict.GQA_REPEAT in flags:
        verdict = Verdict.GQA_REPEAT
    else:
        verdict = Verdict.EXACT_COPY

    # Functional-gate downgrade.
    # In production, a "structurally satisfiable" verdict does NOT
    # imply non-zero runtime acceptance. Without a recorded
    # functional gate, downgrade to FUNCTIONAL_UNVERIFIED so callers
    # must explicitly opt in.
    # KERNEL_*_MISMATCH verdicts and UNSUPPORTED are NOT eligible
    # for the functional-unverified path — they're structurally
    # broken, not "unproven", and override env should not silently
    # admit them as if they were merely uncertain.
    if (require_functional_gate
            and verdict not in (Verdict.EXACT_COPY, Verdict.UNSUPPORTED,
                                Verdict.KERNEL_STORAGE_DTYPE_MISMATCH,
                                Verdict.KERNEL_LAYOUT_CONTRACT_MISMATCH)):
        hints["pre_functional_gate_verdict"] = verdict.value
        verdict = Verdict.ADAPTER_STRUCTURAL_OK_FUNCTIONAL_UNVERIFIED

    return verdict, divergences, hints


__all__ = [
    "Verdict",
    "KVContract",
    "extract_contract",
    "compare_contracts",
]
