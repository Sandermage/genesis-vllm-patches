# tm_int4_moe — vendored TurboMind sm80_16816 int4 grouped-MoE GEMM

Phase 2 of the 26B MoE speed work (see
`docs/superpowers/specs/2026-06-22-turbomind-int4-moe-port.md`).

## What this is
A minimal vendor of TurboMind's (LMDeploy) hand-written `sm80_16816`
tensor-core int4 grouped-MoE GEMM kernel, to be JIT-compiled (via
`torch.utils.cpp_extension`) and wired into vLLM as a custom op that
replaces the slow CUDA-core `moe_wna16_gemm` for Marlin-ineligible int4
MoE shapes (Gemma-4-26B-A4B at TP=2: N=352%64≠0, group_size=32).

## Source & license
- Upstream: **`InternLM/lmdeploy`** @ `main` (HEAD 3c00e811), path
  `src/turbomind/kernels/gemm/`.
- License: **Apache-2.0** (same as Genesis). Vendored files retain their
  original SPDX headers. This is a vendored subset, not a fork.

## Why TurboMind (verified)
The only tensor-core int4-MoE kernel that (a) runs on SM86, (b) **tolerates
K=352** (its 16×8×32 MMA tile + offline weight repack, where Marlin's hard
`thread_k∈{64,128}` rejects), (c) **supports group_size=32** (confirmed in
`gemm/kernel/sm80_16816_4.cu:82` — `register_u4_g(integral_constant<int,32>)`),
(d) is **CUTLASS-free** (hand-written multistage). Measured +134-220% over
FP16 on M=1-16 decode, +19% over Marlin (A100, arXiv:2508.15601) — pending
A5000 re-bench.

## Vendint status (Phase 0)
- [x] full `gemm/` (104 files), `gemm/arch/sm80`, `gemm/kernel/sm80_16816_{4,8,16}`
- [x] `kernels/core/` (header-only) + high-level `core/` headers (data_type,
      check, logger, allocator, tensor, ...) + `attention/quantization.h`
- [x] structure preserves original `src/turbomind/...` include paths (build `-I.`)
- [x] fmt: apt `libfmt-dev` (preferred) OR vendored `third_party/fmt` header-only
- [x] **ALL 3 tensor-core kernels COMPILE on SM86** — proven 2026-06-22 in the
      vLLM image: `sm80_16816_{4,8,16}.cu` → 19.5/2.7/5.8 MB objects. See
      `build_kernels.sh` for the exact recipe. Critical flags found build-driven:
      `-arch=sm_86 -DENABLE_BF16 --expt-relaxed-constexpr -include cuda_fp16.h
      -include cuda_bf16.h -I.`. **The core::-dependency risk is RESOLVED** —
      gemm.cu pulls only check.h+logger.h; the kernel headers need bf16 includes,
      not a Tensor/Allocator shim.
- [x] Phase 0 cont.: full multi-TU build (`test/buildrun.sh`, 54 TUs) → `tm_test`
      links + runs on SM86. **252 sm80 kernels register** (file-probe verified;
      sm70/75 correctly arch-skipped). cuBLAS reference (`test/reference.cu`) wired.
- [x] **Phase 1 PROVEN on rig (2026-06-22, 2×A5000 SM86)** — TurboMind
      `sm80_16816` int4 g32 grouped-MoE kernel dispatches + runs **correctly** for
      the exact Gemma-4-26B-A4B geometry (E=128, top_k=8, hidden=2816,
      inter=704, group_size=32, `f16` act). **rel-err vs FP16 reference (same
      dequantized weights) = 0.000356 mean / 1.60 max, ~0 outliers** → the int4
      tensor-core math is numerically faithful; quantization itself adds ~1.88%
      (inherent to int4 g32, NOT the kernel). **Zero-point fusion verified**:
      `fuse_scales_and_zeros` (packV=0x141) decodes `x*s + (-zp*s)` correctly —
      this **resolves spec risk #1** (the zero-point format) empirically.
      - **Critical learning for Phase 2**: MoE expert weights MUST call
        `LinearWeight::set_grouped(true)` before `prepare()` so `GetConverters`
        returns the *grouped* u4 layout (`order_b` col-major). The dense u4 layout
        has the opposite `order_b` → "No feasible kernel" for the grouped GEMM.
- [ ] Phase 2: torch.ops custom op + offline weight-prep (grouped layout) +
      swap moe_wna16 + rig A/B (speed vs moe_wna16, accuracy vs FP16)

## Build feasibility — PROVEN
`cpp_extension.load_inline` JIT-compiled+ran a CUDA op under `-arch=sm_86`
in the vLLM image (nvcc 13.0, torch 2.11) on the rig 2026-06-22. The full
kernel is multi-TU so it builds as a static lib + cpp_extension binding,
not a single-file JIT.

## Next (Phase 0 → 1 → 2)
See the spec's scaffold plan. Phase 0: finish vendint + shim + standalone
sm_86 build. Phase 1: cuBLAS reference + weight-repack byte-test (the
zero-point format is the critical risk). Phase 2: torch.ops custom op +
swap moe_wna16 + rig A/B vs moe_wna16 (speed) and FP16 (accuracy).
