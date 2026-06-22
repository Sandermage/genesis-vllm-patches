# SPDX-License-Identifier: Apache-2.0
"""Build + load the Genesis TurboMind int4-MoE torch extension on the rig.

Links the pre-built vendored engine objects (/work/build/*.o, minus the test
main) into a torch C++ extension and exercises the probe op. Run inside the
vLLM image after the engine objects are compiled (see build_probe.sh).
"""
import glob
import os

import torch
from torch.utils.cpp_extension import load

WORK = os.environ.get("TM_WORK", "/work")

# Engine objects, minus anything carrying main() or test-only cublas deps.
objs = [
    o for o in glob.glob(f"{WORK}/build/*.o")
    if "test_gemm_v2" not in o and "_test_" not in o
]
print(f"[build_ext] linking {len(objs)} engine objects")

cuda_flags = [
    "-arch=sm_86", "-std=c++17", "-DENABLE_BF16", "-DFMT_HEADER_ONLY",
    "--expt-relaxed-constexpr", "--extended-lambda",
    "-include", "cuda_fp16.h", "-include", "cuda_bf16.h",
    f"-I{WORK}", f"-I{WORK}/third_party/fmt/include", f"-I{WORK}/third_party/moodycamel",
]
cpp_flags = ["-std=c++17", "-DFMT_HEADER_ONLY", f"-I{WORK}", f"-I{WORK}/third_party/fmt/include"]

torch.zeros(1, device="cuda")  # force torch to init the CUDA context first

# This torch build's load() has no extra_objects kwarg; pass the pre-built
# engine objects positionally through the linker via extra_ldflags.
# is_python_module=False: ops are registered via TORCH_LIBRARY static init, not
# a pybind module, so torch just dlopen()s the .so and the ops appear on torch.ops.
load(
    name="genesis_tm",
    sources=[f"{WORK}/torch_ext/tm_moe_op.cu"],
    extra_cuda_cflags=cuda_flags,
    extra_cflags=cpp_flags,
    extra_ldflags=[*objs, "-lcublas", "-lcublasLt",
                   "-L/usr/local/cuda/lib64/stubs", "-lcuda"],
    is_python_module=False,
    verbose=True,
)
print("[build_ext] extension LOADED OK")
print("[build_ext] tm_probe() =>", torch.ops.genesis_tm.tm_probe())

# --- correctness sweep: raw w1w3 grouped int4 GEMM vs fp16 reference ----------
def run_case(E, K, N, G, M, TOPK, mean):
    torch.manual_seed(0)
    W = (torch.randn(E, K, N, device="cuda", dtype=torch.float16) * 0.1 + mean)  # (E,K,N)
    op = torch.classes.genesis_tm.TmInt4MoE(W, G)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    ti = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)])
    f2n, slot_e, counts = [], [], [0] * E
    for e in range(E):
        for t in range(M):
            if e in ti[t].tolist():
                f2n.append(t); slot_e.append(e); counts[e] += 1
    offs = [0]
    for e in range(E):
        offs.append(offs[-1] + counts[e])
    f2n_t = torch.tensor(f2n, dtype=torch.int32, device="cuda")
    off_t = torch.tensor(offs, dtype=torch.int32, device="cuda")
    out = op.forward_w1w3(x, f2n_t, off_t)
    dq = torch.stack([op.get_dequant(e) for e in range(E)])  # (E,K,N) int4-reconstructed
    ref_orig = torch.empty_like(out)
    ref_dq = torch.empty_like(out)
    for r in range(len(f2n)):
        ref_orig[r] = (x[f2n[r]].float() @ W[slot_e[r]].float()).half()
        ref_dq[r] = (x[f2n[r]].float() @ dq[slot_e[r]].float()).half()

    def rd(a, b):
        return ((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)).item()

    orig = torch.stack([op.get_orig(e) for e in range(E)])  # (E,K,N) weight quantizer saw
    gemm_err = rd(out, ref_dq)          # op int4 GEMM vs GEMM-on-dequant -> kernel correctness
    quant_w = rd(dq, W)                 # int4-dequant weight vs original weight -> quant error
    copy_err = rd(orig, W)              # weight quantizer saw vs torch W -> Copy correctness
    print(f"[test] E={E:3d} K={K:5d} N={N:5d} mean={mean:.1f} R={len(f2n):4d} "
          f"copy_err(orig vs W)={copy_err:.4f}  GEMM_err={gemm_err:.4f}  quant_err(dq vs W)={quant_w:.4f}")
    if mean == 1.0 and E == 1:
        print("   W[0,0,:6] =", W[0, 0, :6].tolist())
        print("   dq[0,0,:6]=", dq[0, 0, :6].tolist())
        print("   W[0,:6,0] =", W[0, :6, 0].tolist())
        print("   dq[0,:6,0]=", dq[0, :6, 0].tolist())

# GEMM_err (op vs GEMM-on-int4-dequant) is the KERNEL correctness metric -> ~0.
# quant_err (dequant vs original) is the int4-g32 accuracy: tight-spread weights
# (mean=1.0) -> ~0.007, wide zero-mean N(0,0.1) -> ~0.087. With the quantizer
# zero-point clamp removed (quantization.cu:426), the dequant->requant weight-prep
# is robust for ALL distributions, including all-positive groups.
run_case(8, 2816, 1408, 32, 64, 8, 0.0)   # zero-mean: GEMM_err~0, quant_err~0.087
run_case(1, 2816, 1408, 32, 16, 1, 1.0)   # all-positive: GEMM_err~0, quant_err~0.007 (fix verified)
