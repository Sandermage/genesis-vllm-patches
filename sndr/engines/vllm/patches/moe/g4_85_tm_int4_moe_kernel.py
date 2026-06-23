# SPDX-License-Identifier: Apache-2.0
"""G4_85 — TurboMind int4 grouped-MoE kernel replacing the slow CUDA-core moe_wna16.

At TP>1 for int4-MoE models where Marlin is structurally rejected
(`intermediate_per_partition % max(64, group_size) != 0`, detected by G4_84),
vLLM falls back to `moe_wna16_gemm` (CUDA-core, memory-bound). This patch routes
those layers through TurboMind's tensor-core `sm80_16816` int4 grouped-MoE GEMM
(see `third_party/tm_int4_moe`), which is 3-6x faster on SM80/86 and numerically
faithful (0.036% rel-err vs FP16, proven on the rig).

LIVE TARGET (re-targeted 2026-06-23). The path that actually carries
Marlin-ineligible compressed-tensors int4 MoE on this rig is
``CompressedTensorsWNA16MoEMethod.apply`` (verified by a 26B live boot:
``Using CompressedTensorsWNA16MoEMethod`` + ``fused_moe.py:1074 ... E=128,N=352
... int4_w4a16``), NOT ``moe_wna16.MoeWNA16Method``. G4_85 monkey-patches
``CompressedTensorsWNA16MoEMethod.apply`` — the byte-identical 7-arg signature
``(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)``.

Pipeline per MoE layer (built/cached on first apply from the layer's int4
weights, dequantized to fp16): gate (topk from vLLM) -> w1w3 grouped int4 GEMM
-> SwiGLU -> w2 grouped int4 GEMM -> combine.

Gating (all must hold): env flag GENESIS_ENABLE_G4_85 (default OFF, experimental)
AND ``is_moe_model()`` (dense models never dispatch fused_moe) AND, per-layer at
apply time, the G4_84 ``marlin_moe_marginal()`` detector — so the TurboMind op
only ever runs where Marlin was structurally rejected and vLLM would otherwise
take the slow CUDA-core moe_wna16 path. With GENESIS_G4_85_VALIDATE=1 the first
apply of each layer also runs the original method and logs the rel-err.

The TurboMind op is the `genesis_tm.TmInt4MoE` torch custom class built from
`third_party/tm_int4_moe/torch_ext` (JIT-compiled on first use).

Fail-open: ANY exception (including a wrong compressed-tensors weight attribute
name on a vLLM build whose layout we could not verify offline) degrades to the
original ``CompressedTensorsWNA16MoEMethod.apply`` — never a crash, never a
numeric change unless the TurboMind path completes cleanly.

Author: deep TurboMind-int4-MoE port (Genesis), 2026-06-22.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("genesis.g4_85")

GENESIS_G4_85_MARKER = "G4_85_TM_INT4_MOE"
_ENV = "GENESIS_ENABLE_G4_85"
_VALIDATE_ENV = "GENESIS_G4_85_VALIDATE"

_orig_apply = None  # saved CompressedTensorsWNA16MoEMethod.apply for revert + validation


def _enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _resolve_wna16_method():
    """Import the LIVE compressed-tensors WNA16 MoE method class.

    This is the method that actually carries Marlin-ineligible int4 MoE
    on the rig (``Using CompressedTensorsWNA16MoEMethod`` at boot). Returns
    the class, or raises (caller treats as a runtime gap / skip).

    The class moved across vLLM versions. On the dev148 pin
    ``compressed_tensors_moe`` is a PACKAGE and the class lives in its
    ``compressed_tensors_moe_wna16`` submodule (verified live 2026-06-23 on
    ``vllm/vllm-openai:nightly-b4c80ec0f`` — the package root does NOT re-export
    it, so the old flat lookup raised AttributeError and the patch silently
    self-skipped). Older trees expose a flat ``compressed_tensors_moe_wna16``
    module or the class at the package root. Probe the known locations in order.
    """
    import importlib

    base = "vllm.model_executor.layers.quantization.compressed_tensors"
    candidates = (
        # dev148: compressed_tensors_moe is a package; class in the wna16 submodule
        f"{base}.compressed_tensors_moe.compressed_tensors_moe_wna16",
        # older: flat module sibling of the package
        f"{base}.compressed_tensors_moe_wna16",
        # fallback: class re-exported at the package/module root
        f"{base}.compressed_tensors_moe",
    )
    last_exc = None
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            return getattr(mod, "CompressedTensorsWNA16MoEMethod")
        except (ImportError, AttributeError) as exc:  # noqa: PERF203
            last_exc = exc
            continue
    raise ImportError(
        "CompressedTensorsWNA16MoEMethod not found in any known location "
        f"{candidates}: {last_exc}"
    )


# --------------------------------------------------------------------------- #
# weight dequant (symmetric int4, compressed-tensors pack-quantized, uint8 2/byte)
# --------------------------------------------------------------------------- #
def _dequant_wna16(qweight, scale, group_size):
    """(E, N, K//2) uint8 + (E, N, K//g) fp16 -> (E, K, N) fp16, input-major.

    moe_wna16 decodes the symmetric int4 as an UNSIGNED nibble with zero-point 8:
    w = (nibble - 8) * scale, packed 2-per-byte along K (low nibble first).
    Verified against vLLM fused_experts_impl on the rig: this decode gives
    rel-err 0.05 (the dequant->requant residual) vs 1.32 for the signed decode.
    The (K, N) result is the input-major layout TmInt4MoE expects.
    """
    import torch

    E, N, Kh = qweight.shape
    K = Kh * 2
    b = qweight.to(torch.int32)
    q = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(E, N, K)
    q = q - 8  # unsigned nibble [0,15], zero-point 8 -> signed value
    g = K // scale.shape[-1]
    s = scale.to(torch.float16).repeat_interleave(g, dim=-1)  # (E, N, K)
    return (q.to(torch.float16) * s).transpose(1, 2).contiguous()  # (E, K, N) input-major


def _build_routing(topk_ids, topk_weights, num_experts):
    """topk_ids/topk_weights (M, top_k) -> f2n, offsets, gate, slot order (GPU)."""
    import torch

    M, TK = topk_ids.shape
    flat_e = topk_ids.reshape(-1).to(torch.int64)               # (M*TK,)
    flat_t = torch.arange(M, device=topk_ids.device).repeat_interleave(TK)
    flat_g = topk_weights.reshape(-1).to(torch.float32)
    order = torch.argsort(flat_e, stable=True)                  # group by expert
    f2n = flat_t[order].to(torch.int32)                         # (R,) source token
    gate = flat_g[order]                                        # (R,)
    counts = torch.bincount(flat_e, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    return f2n, offsets, gate


class _LayerOps:
    """Cached TurboMind ops + routing scratch for one MoE layer."""

    def __init__(self, op13, op2, num_experts, inter):
        self.op13 = op13
        self.op2 = op2
        self.E = num_experts
        self.I = inter


_EXT_LOADED = False


def _ensure_ext():
    """JIT-build + load the TurboMind torch extension (genesis_tm.TmInt4MoE).

    Expects the vendored tree at $GENESIS_TM_INT4_MOE_DIR (default
    /opt/genesis/tm_int4_moe) with its engine objects pre-built in build/. The
    extension links those objects against libtorch (see torch_ext/build_ext.py).
    """
    global _EXT_LOADED
    if _EXT_LOADED:
        return
    import glob

    import torch
    from torch.utils.cpp_extension import load

    work = os.environ.get("GENESIS_TM_INT4_MOE_DIR", "/opt/genesis/tm_int4_moe")
    objs = [o for o in glob.glob(f"{work}/build/*.o")
            if "test_gemm_v2" not in o and "_test_" not in o]
    if not objs:
        raise RuntimeError(f"G4_85: no pre-built engine objects under {work}/build")
    flags = ["-arch=sm_86", "-std=c++17", "-DENABLE_BF16", "-DFMT_HEADER_ONLY",
             "--expt-relaxed-constexpr", "--extended-lambda",
             "-include", "cuda_fp16.h", "-include", "cuda_bf16.h",
             f"-I{work}", f"-I{work}/third_party/fmt/include",
             f"-I{work}/third_party/moodycamel"]
    torch.zeros(1, device="cuda")
    load(name="genesis_tm", sources=[f"{work}/torch_ext/tm_moe_op.cu"],
         extra_cuda_cflags=flags,
         extra_cflags=["-std=c++17", "-DFMT_HEADER_ONLY", f"-I{work}",
                       f"-I{work}/third_party/fmt/include"],
         extra_ldflags=[*objs, "-lcublas", "-lcublasLt",
                        "-L/usr/local/cuda/lib64/stubs", "-lcuda"],
         is_python_module=False, verbose=False)
    _EXT_LOADED = True
    logger.info("[G4_85] TurboMind torch extension loaded (%d objects)", len(objs))


def _layer_attr(layer, *names):
    """First present attribute among ``names`` (compressed-tensors layouts
    vary across vLLM builds; we could not introspect the class offline, so
    we probe the known names and let the fail-open path catch a miss)."""
    for name in names:
        value = getattr(layer, name, None)
        if value is not None:
            return value
    raise AttributeError(
        f"G4_85: none of {names!r} present on layer "
        f"(have: {[a for a in dir(layer) if 'weight' in a or 'scale' in a]})"
    )


def _build_layer_ops(layer, group_size):
    import torch

    _ensure_ext()
    # CompressedTensorsWNA16MoEMethod.create_weights registers the packed
    # int4 expert weights as ``w13_weight_packed`` / ``w2_weight_packed`` and
    # the group scales as ``w13_weight_scale`` / ``w2_weight_scale``. We also
    # probe the moe_wna16 (``*_qweight`` / ``*_scale``) names so the same op
    # builds if a future build relabels — a miss falls through to fail-open.
    w13_packed = _layer_attr(layer, "w13_weight_packed", "w13_qweight")
    w2_packed = _layer_attr(layer, "w2_weight_packed", "w2_qweight")
    w13_scale = _layer_attr(layer, "w13_weight_scale", "w13_scale")
    w2_scale = _layer_attr(layer, "w2_weight_scale", "w2_scale")
    w13 = _dequant_wna16(w13_packed, w13_scale, group_size)  # (E,K,2I)
    w2 = _dequant_wna16(w2_packed, w2_scale, group_size)     # (E,I,K)
    op13 = torch.classes.genesis_tm.TmInt4MoE(w13, group_size)
    op2 = torch.classes.genesis_tm.TmInt4MoE(w2, group_size)
    return _LayerOps(op13, op2, w13.shape[0], w2.shape[1])


def _tm_moe_forward(ops: "_LayerOps", x, topk_weights, topk_ids):
    import torch
    import torch.nn.functional as F

    M, K = x.shape
    f2n, offsets, gate = _build_routing(topk_ids, topk_weights, ops.E)
    R = f2n.shape[0]
    ident = torch.arange(R, dtype=torch.int32, device=x.device)
    de = ops.op13.forward_w1w3(x.contiguous(), f2n, offsets)      # (R, 2I)
    inter = (F.silu(de[:, : ops.I].float()) * de[:, ops.I:].float()).half()
    oe = ops.op2.forward_w1w3(inter, ident, offsets)             # (R, K)
    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    out.index_add_(0, f2n.long(), gate[:, None] * oe.float())
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# patched apply
# --------------------------------------------------------------------------- #
def _marlin_marginal(intermediate_per_partition, group_size):
    """Reuse G4_84's Marlin-ineligibility detector (single source of truth).

    Falls back to a local mirror if the G4_84 module is not importable, so
    the per-layer gate never crashes the apply path.
    """
    try:
        from .g4_84_moe_geometry_advisor import marlin_moe_marginal
        return marlin_moe_marginal(intermediate_per_partition, group_size)
    except Exception:  # noqa: BLE001
        divisor = max(64, group_size if group_size and group_size > 0 else 64)
        return (intermediate_per_partition % divisor) != 0


def _genesis_apply(self, layer, x, topk_weights, topk_ids,
                   shared_experts=None, shared_experts_input=None):
    """Replacement for CompressedTensorsWNA16MoEMethod.apply (exact signature).

    Falls back to the original on any error, when shared_experts is requested,
    or when this layer's geometry is Marlin-ELIGIBLE (G4_85 must only fire on
    the Marlin-ineligible int4 MoE path the slow CUDA-core kernel carries).
    Fail-open: a wrong weight attribute name -> AttributeError -> original.
    """
    # Only handle the plain routed-experts case; defer anything exotic.
    if shared_experts is not None:
        return _orig_apply(self, layer, x, topk_weights, topk_ids,
                           shared_experts, shared_experts_input)
    try:
        group_size = getattr(getattr(self, "quant_config", None), "group_size", None) \
            or getattr(self, "group_size", 32)

        # Per-layer Marlin-ineligibility gate (G4_84 detector). Only fire
        # where Marlin was structurally rejected — i.e. exactly where vLLM
        # would otherwise take the slow CUDA-core moe_wna16 path. If the
        # geometry IS Marlin-eligible, defer to the original (Marlin) method.
        inter = getattr(layer, "intermediate_size_per_partition", None)
        if inter is None:
            w2p = getattr(layer, "w2_weight_packed", None)
            if w2p is None:
                w2p = getattr(layer, "w2_qweight", None)
            # w2 is (E, hidden, inter//2) packed -> inter = last_dim * 2.
            inter = (w2p.shape[-1] * 2) if w2p is not None else None
        if inter is not None and not _marlin_marginal(int(inter), int(group_size)):
            return _orig_apply(self, layer, x, topk_weights, topk_ids,
                               shared_experts, shared_experts_input)

        ops = getattr(layer, "_g4_85_ops", None)
        if ops is None:
            ops = _build_layer_ops(layer, group_size)
            layer._g4_85_ops = ops
            logger.info("[G4_85] built TurboMind int4 MoE ops (E=%d I=%d g=%d)",
                        ops.E, ops.I, group_size)

        x2 = x.view(-1, x.shape[-1])
        out = _tm_moe_forward(ops, x2, topk_weights, topk_ids)

        if os.environ.get(_VALIDATE_ENV) == "1" and not getattr(layer, "_g4_85_val", False):
            layer._g4_85_val = True
            ref = _orig_apply(self, layer, x, topk_weights, topk_ids,
                              shared_experts, shared_experts_input).view(-1, x.shape[-1])
            rel = ((out.float() - ref.float()).abs().mean()
                   / ref.float().abs().mean().clamp_min(1e-6)).item()
            logger.warning("[G4_85][VALIDATE] reldiff vs original = %.5f (E=%d M=%d)",
                           rel, ops.E, x2.shape[0])
        return out.view_as(x)
    except Exception as e:  # noqa: BLE001
        logger.warning("[G4_85] fell back to original WNA16 MoE apply: %r", e)
        return _orig_apply(self, layer, x, topk_weights, topk_ids,
                           shared_experts, shared_experts_input)


def apply() -> tuple[str, str]:
    """Monkey-patch CompressedTensorsWNA16MoEMethod.apply when enabled.

    Returns ``(status, reason)`` per the Genesis apply contract — status in
    {applied, skipped, failed}. default OFF: a no-op ``skipped`` unless
    ``GENESIS_ENABLE_G4_85=1``. Additionally gated on ``is_moe_model()`` so
    dense models never install the hook. Fail-open is preserved at the
    per-layer ``_genesis_apply`` level.
    """
    global _orig_apply
    if not _enabled():
        return ("skipped", "disabled via GENESIS_ENABLE_G4_85 (default OFF)")

    # P52 MoE-dispatch gate: dense models never dispatch fused_moe, the hook
    # would be dead weight. Best-effort — proceed if the probe is unavailable.
    try:
        from sndr.engines.vllm.detection.model_detect import is_moe_model
        if not is_moe_model():
            return ("skipped", "P52 dispatch: model has no MoE layers")
    except Exception as e:  # noqa: BLE001
        logger.debug("[G4_85] model_detect probe failed (proceeding): %s", e)

    try:
        method = _resolve_wna16_method()
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if any(m in msg for m in ("torch", "triton", "flashinfer", "vllm")):
            return ("skipped",
                    f"runtime not present on this host ({e}) — patch would "
                    "apply on a vllm-equipped server")
        logger.warning("[G4_85] CompressedTensorsWNA16MoEMethod not found: %s", e)
        return ("failed", f"CompressedTensorsWNA16MoEMethod not resolvable: {e}")

    if _orig_apply is not None:
        return ("skipped", "already patched (idempotent)")
    _orig_apply = method.apply
    method.apply = _genesis_apply
    logger.info("[G4_85] patched %s.apply -> TurboMind int4 MoE", method.__name__)
    return ("applied",
            "CompressedTensorsWNA16MoEMethod.apply re-routed to TurboMind "
            "int4 grouped-MoE (fires only on Marlin-ineligible int4 MoE "
            "layers; fail-open to the original)")


def is_applied() -> bool:
    return _orig_apply is not None


def revert() -> bool:
    global _orig_apply
    if _orig_apply is None:
        return False
    try:
        method = _resolve_wna16_method()
    except Exception:  # noqa: BLE001
        return False
    method.apply = _orig_apply
    _orig_apply = None
    return True
