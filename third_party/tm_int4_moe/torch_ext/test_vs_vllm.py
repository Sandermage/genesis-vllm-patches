# SPDX-License-Identifier: Apache-2.0
"""Validate the G4_85 dequant + TmInt4MoE pipeline against vLLM's actual moe_wna16.

Standalone (no model launch): generate symmetric int4 weights in vLLM's wna16
layout, run BOTH vLLM fused_experts_impl (-> moe_wna16 CUDA kernel at decode)
and the TurboMind pipeline (dequant -> two int4 GEMMs -> SwiGLU -> combine), and
compare. A small rel-err proves the dequant nibble/sign format + routing +
pipeline reproduce moe_wna16, gating the live integration.

Run inside the vLLM image with the extension prebuilt:
  docker run --rm --gpus all -v <vendor>:/work ... \
    bash -c "bash /work/torch_ext/build_probe.sh >/dev/null 2>&1 ; python3 -u /work/torch_ext/test_vs_vllm.py"
(build_probe.sh leaves the engine objects + genesis_tm.so loadable.)
"""
import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

WORK = os.environ.get("TM_WORK", "/work")


def _load_ext():
    objs = [o for o in glob.glob(f"{WORK}/build/*.o")
            if "test_gemm_v2" not in o and "_test_" not in o]
    flags = ["-arch=sm_86", "-std=c++17", "-DENABLE_BF16", "-DFMT_HEADER_ONLY",
             "--expt-relaxed-constexpr", "--extended-lambda",
             "-include", "cuda_fp16.h", "-include", "cuda_bf16.h",
             f"-I{WORK}", f"-I{WORK}/third_party/fmt/include",
             f"-I{WORK}/third_party/moodycamel"]
    torch.zeros(1, device="cuda")
    load(name="genesis_tm", sources=[f"{WORK}/torch_ext/tm_moe_op.cu"],
         extra_cuda_cflags=flags,
         extra_cflags=["-std=c++17", "-DFMT_HEADER_ONLY", f"-I{WORK}",
                       f"-I{WORK}/third_party/fmt/include"],
         extra_ldflags=[*objs, "-lcublas", "-lcublasLt",
                        "-L/usr/local/cuda/lib64/stubs", "-lcuda"],
         is_python_module=False, verbose=False)


def _dequant_wna16(qweight, scale, group_size):
    """(E,N,K//2) uint8 + (E,N,K//g) -> (E,K,N) fp16. Symmetric, low-nibble-first."""
    E, N, Kh = qweight.shape
    K = Kh * 2
    b = qweight.to(torch.int32)
    q = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(E, N, K)
    mode = os.environ.get("TM_DQ_MODE", "off8")
    if mode == "off8":
        q = q - 8                                  # unsigned nibble, zero-point 8
    else:
        q = torch.where(q >= 8, q - 16, q)         # two's-complement signed int4
    g = K // scale.shape[-1]
    s = scale.to(torch.float16).repeat_interleave(g, dim=-1)
    return (q.to(torch.float16) * s).transpose(1, 2).contiguous()  # (E,K,N)


def main():
    _load_ext()
    torch.ops.genesis_tm.tm_probe()  # establish the persistent TurboMind context early
    torch.manual_seed(0)
    E, K, I, G, M, TOPK = 8, 2816, 704, 32, 16, 8
    dev = "cuda"

    # symmetric int4 weights in vLLM wna16 layout: qweight (E, N, K//2) uint8
    def make(N, Kdim):
        q = torch.randint(-8, 8, (E, N, Kdim), device=dev)              # signed int4
        scale = (torch.rand(E, N, Kdim // G, device=dev) * 0.02 + 0.01).half()
        nib = (q & 0xF).to(torch.uint8).reshape(E, N, Kdim // 2, 2)
        packed = (nib[..., 0] | (nib[..., 1] << 4)).contiguous()        # (E,N,K//2) uint8
        return packed, scale, q

    w13_q, w13_s, _ = make(2 * I, K)     # gate-up: N=2I, K=hidden
    w2_q, w2_s, _ = make(K, I)           # down:    N=hidden, K=inter

    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    ti = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(torch.int32).to(dev)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1).to(torch.float32)

    # --- vLLM moe_wna16 (the competitor / ground truth) ---
    out_vllm = fused_experts_impl(
        x, w13_q, w2_q, tw, ti, activation="silu", use_int4_w4a16=True,
        w1_scale=w13_s, w2_scale=w2_s, block_shape=[0, G], global_num_experts=E)

    # --- TurboMind pipeline on the dequantized weights ---
    w13 = _dequant_wna16(w13_q, w13_s, G)     # (E,K,2I)
    w2 = _dequant_wna16(w2_q, w2_s, G)        # (E,I,K)
    op13 = torch.classes.genesis_tm.TmInt4MoE(w13, G)
    op2 = torch.classes.genesis_tm.TmInt4MoE(w2, G)

    flat_e = ti.reshape(-1).to(torch.int64)
    flat_t = torch.arange(M, device=dev).repeat_interleave(TOPK)
    order = torch.argsort(flat_e, stable=True)
    f2n = flat_t[order].to(torch.int32)
    gate = tw.reshape(-1)[order]
    offs = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offs[1:] = torch.bincount(flat_e, minlength=E).cumsum(0).to(torch.int32)
    R = f2n.shape[0]
    ident = torch.arange(R, dtype=torch.int32, device=dev)

    de = op13.forward_w1w3(x, f2n, offs)
    inter = (F.silu(de[:, :I].float()) * de[:, I:].float()).half()
    oe = op2.forward_w1w3(inter, ident, offs)
    out_tm = torch.zeros(M, K, dtype=torch.float32, device=dev)
    out_tm.index_add_(0, f2n.long(), gate[:, None] * oe.float())

    rel = ((out_tm - out_vllm.float()).abs().mean()
           / out_vllm.float().abs().mean().clamp_min(1e-6)).item()
    print(f"[vs_vllm] M={M} E={E} reldiff(TurboMind vs moe_wna16) = {rel:.5f}  "
          f"finite={bool(torch.isfinite(out_tm).all())}")
    print(f"[vs_vllm] out_vllm[0,:4]={out_vllm[0,:4].tolist()}")
    print(f"[vs_vllm] out_tm  [0,:4]={out_tm[0,:4].half().tolist()}")


if __name__ == "__main__":
    main()
