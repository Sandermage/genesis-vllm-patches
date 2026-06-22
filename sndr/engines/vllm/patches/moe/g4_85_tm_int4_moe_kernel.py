# SPDX-License-Identifier: Apache-2.0
"""G4_85 — TurboMind int4 grouped-MoE kernel replacing the slow CUDA-core moe_wna16.

At TP>1 for int4-MoE models where Marlin is structurally rejected
(`intermediate_per_partition % max(64, group_size) != 0`, detected by G4_84),
vLLM falls back to `moe_wna16_gemm` (CUDA-core, memory-bound). This patch routes
those layers through TurboMind's tensor-core `sm80_16816` int4 grouped-MoE GEMM
(see `third_party/tm_int4_moe`), which is 3-6x faster on SM80/86 and numerically
faithful (0.036% rel-err vs FP16, proven on the rig).

Pipeline per MoE layer (built/cached on first apply from the layer's int4
weights, dequantized to fp16): gate (topk from vLLM) -> w1w3 grouped int4 GEMM
-> SwiGLU -> w2 grouped int4 GEMM -> combine.

Gating: env flag GENESIS_ENABLE_G4_85 (default off, build-gated/experimental) AND
the G4_84 marlin-ineligible detector. With GENESIS_G4_85_VALIDATE=1 the first
apply of each layer also runs the original moe_wna16 and logs the rel-err.

The TurboMind op is the `genesis_tm.TmInt4MoE` torch custom class built from
`third_party/tm_int4_moe/torch_ext` (JIT-compiled on first use).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("genesis.g4_85")

GENESIS_G4_85_MARKER = "G4_85_TM_INT4_MOE"
_ENV = "GENESIS_ENABLE_G4_85"
_VALIDATE_ENV = "GENESIS_G4_85_VALIDATE"

_orig_apply = None  # saved moe_wna16 MoEMethod.apply for revert + validation


def _enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


# --------------------------------------------------------------------------- #
# weight dequant (symmetric int4, compressed-tensors pack-quantized, uint8 2/byte)
# --------------------------------------------------------------------------- #
def _dequant_wna16(qweight, scale, group_size):
    """(E, N, K//2) uint8 + (E, N, K//g) fp16 -> (E, K, N) fp16, input-major.

    Symmetric int4: w = q_signed * scale, q packed 2-per-byte along K (low nibble
    first). The (K, N) result is the input-major layout TmInt4MoE expects.
    """
    import torch

    E, N, Kh = qweight.shape
    K = Kh * 2
    b = qweight.to(torch.int32)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    # interleave low/high nibble back to (E, N, K)
    q = torch.stack([lo, hi], dim=-1).reshape(E, N, K)
    q = torch.where(q >= 8, q - 16, q)  # unsigned nibble -> signed int4 [-8,7]
    g = K // scale.shape[-1]
    s = scale.to(torch.float16).repeat_interleave(g, dim=-1)  # (E, N, K)
    w = (q.to(torch.float16) * s)  # (E, N, K)
    return w.transpose(1, 2).contiguous()  # (E, K, N) input-major


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


def _build_layer_ops(layer, group_size):
    import torch

    _ensure_ext()
    w13 = _dequant_wna16(layer.w13_qweight, layer.w13_scale, group_size)  # (E,K,2I)
    w2 = _dequant_wna16(layer.w2_qweight, layer.w2_scale, group_size)     # (E,I,K)
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
def _genesis_apply(self, layer, x, topk_weights, topk_ids,
                   shared_experts=None, shared_experts_input=None):
    """Replacement for moe_wna16 MoEMethod.apply (exact signature). Falls back to
    the original on any error or when shared_experts is requested (fail-open)."""
    # Only handle the plain routed-experts case; defer anything exotic.
    if shared_experts is not None:
        return _orig_apply(self, layer, x, topk_weights, topk_ids,
                           shared_experts, shared_experts_input)
    try:
        group_size = getattr(getattr(self, "quant_config", None), "group_size", None) \
            or getattr(self, "group_size", 32)

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
            logger.warning("[G4_85][VALIDATE] reldiff vs moe_wna16 = %.5f (E=%d M=%d)",
                           rel, ops.E, x2.shape[0])
        return out.view_as(x)
    except Exception as e:  # noqa: BLE001
        logger.warning("[G4_85] fell back to moe_wna16: %r", e)
        return _orig_apply(self, layer, x, topk_weights, topk_ids,
                           shared_experts, shared_experts_input)


def apply() -> bool:
    """Monkey-patch moe_wna16 MoEMethod.apply when enabled."""
    global _orig_apply
    if not _enabled():
        return False
    try:
        from vllm.model_executor.layers.quantization import moe_wna16
        method = moe_wna16.MoeWNA16Method if hasattr(moe_wna16, "MoeWNA16Method") \
            else moe_wna16.MoEMethod
    except Exception as e:  # noqa: BLE001
        logger.warning("[G4_85] moe_wna16 method not found: %s", e)
        return False
    if _orig_apply is not None:
        return True
    _orig_apply = method.apply
    method.apply = _genesis_apply
    logger.info("[G4_85] patched %s.apply -> TurboMind int4 MoE", method.__name__)
    return True


def is_applied() -> bool:
    return _orig_apply is not None


def revert() -> bool:
    global _orig_apply
    if _orig_apply is None:
        return False
    from vllm.model_executor.layers.quantization import moe_wna16
    method = moe_wna16.MoeWNA16Method if hasattr(moe_wna16, "MoeWNA16Method") \
        else moe_wna16.MoEMethod
    method.apply = _orig_apply
    _orig_apply = None
    return True
