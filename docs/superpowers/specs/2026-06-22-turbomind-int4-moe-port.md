# TurboMind sm80_16816 int4 grouped-MoE GEMM → vLLM custom op (Phase 2)

**Date:** 2026-06-22
**Goal:** replace vLLM's slow CUDA-core `moe_wna16_gemm` with TurboMind's
tensor-core `sm80_16816` int4 grouped-MoE GEMM for Gemma-4-26B-A4B on
2×A5000 (SM86, TP=2), where Marlin is structurally rejected (N=352 %
64 ≠ 0). Source studied: `InternLM/lmdeploy@main` (HEAD 3c00e811).

## Why (from the Phase-1 audit, §80-81 journal)
26B int4 MoE at TP=2 falls back to `moe_wna16_gemm` — CUDA-core, memory-
bound, ~1.5-1.85× slower than tensor-core. Marlin can't help (352%64≠0
group-straddle, verified). TurboMind's hand-written `sm80_16816` is the
ONE tensor-core int4-MoE kernel that **tolerates K=352** (its 16×8×32
MMA tile + offline weight repack pad to the tile, where Marlin's hard
`thread_k∈{64,128}` cannot) and supports **group_size=32** (our
checkpoint). Measured +134-220% over FP16 on M=1-16 decode, +19% over
Marlin (A100, arXiv:2508.15601) — needs A5000 re-bench.

## Two blockers — RESOLVED
1. **g32 supported** (NOT a blocker): `kernel/sm80_16816_4.cu:81-82`
   instantiates `register_u4_g(integral_constant<int,32>{})` alongside 128.
   `GroupSizeV` is a template param, `CHUNK_K=lcm(GroupSizeV,CTA_K)` handles
   32. No re-quant to g128 needed.
2. **CUTLASS not needed**: the sm80 path is hand-written multistage. The
   only `cutlass`/`cute` references on the sm80 path are inspiration-URL
   comments; the CUTLASS link target is `gemm2_sm90` only.

K=352 is the OUTPUT dim N (not K) — safe via partial-tiles (`align.y=1`,
`cdiv(N,CTA_N)`). Real constraint is `K % CTA_K == 0`, CTA_K∈{32,64,128};
Gemma hidden/inter are multiples of 128 → pass.

## Build feasibility — PROVEN on rig (2026-06-22)
`torch.utils.cpp_extension.load_inline` JIT-compiled a CUDA op under
`-arch=sm_86` in the vLLM image (nvcc 13.0, torch 2.11) and ran it. So
the overlay-JIT pattern (a Python patch that compiles the vendored kernel,
mirroring our Triton-JIT kernels) is viable. Caveat: the full TurboMind
kernel is multi-TU `.cu` + template registration, so NOT a single-file
JIT — use a thin static lib (`libtm_int4_moe.a`) + `cpp_extension` with
`extra_objects`, OR `sources=[3 kernels, gemm.cu, convert/cast, registry]`.

## Gemm::Run API (verbatim, gemm.h:23-39)
```cpp
int Run(const Operation&, float alpha,
        const void* A, const MatrixLayout& Adesc,   // activations (m,k) RowMajor
        const void* U, const MatrixLayout& Udesc,   // input-quant scales (nullptr for us)
        const void* B, const MatrixLayout& Bdesc,   // packed int4 weights
        const void* V, const MatrixLayout& Vdesc,   // fused (scale, -zero*scale) uint32
        float beta,
        const void* C, const MatrixLayout& Cdesc,   // = D, beta=0
        void* D, const MatrixLayout& Ddesc,         // output
        const Workspace&, cudaStream_t);
```
POD args (desc.h/types.h): `Operation{dispatch=kDefault, epilogue=kGatedSilu
(w1w3)/kNone(w2), quant_a={kNone,0}, quant_b={kK,32}, batch_dim=0}`;
`MatrixLayout{type, order, rows, cols, ld, pack, num=E, offsets, idxs}`;
`Workspace{barriers/partials/tensormaps/flags}` (kBarriersSize=1<<20,
kPartialsSize=32<<20).

Grouped-MoE: `desc_A.num=E`, `desc_A.offsets=[E+1] prefix-sums`,
`desc_A.idxs=f2n` (scatter, w1w3 only). `offsets!=nullptr → kBlocked`.

## MoE path
`invokeMoeGate_V2(...)` (moe_utils_v2.h:17) builds f2n/f2E/en2f/offsets/
scales from gate logits in one call. w1w3: `Forward(input, w1w3, f2n,
offsets, inter)` epilogue kGatedSilu. w2: `Forward(inter, w2, {}, offsets,
out)` epilogue kNone. Combine: `invokeMoeCombine(...)` by en2f/scales.

## Weight repack (CRITICAL risk — linear_weight.cc::prepare, convert_v3.cu)
Input (our compressed-tensors g32): packed int4 (K,N) + scales (K/32,N) +
int zeros (K/32,N). Pipeline:
1. `extend_to_u16`: int4 → uint16.
2. transpose to (N,K) if RowMajor.
3. `conv_w->Convert`: repack to SM80 16×8×32 MMA lane-layout (pack=0x122).
4. `fuse_scales_and_zeros` (cast.cu:135): `vf[2i]=scale; vf[2i+1]=-zero*scale`
   packed as uint32 (two half) → FMA dequant `x*s + (-zero*s)`.
**RISK:** our zero is an INT in int4-domain (`w=s*(w_q-zp)`); TurboMind
wants `-zp*s` in half AND decodes int4 as **unsigned 0..15** (lop3 magic).
If our `w_q` is signed (-8..7), a +8 shift must fold into zero-point.
MITIGATION: byte-exact unit test on one expert (Phase 1.6); fallback =
dequant our weights to FP16 and re-quantize via their `QuantizeGroupwise`
(guaranteed format, one offline pass).

## Vendor set (~25-30 files, ~9-10k LOC) — see agent report §5
gemm/: gemm.h gemm.cu desc.h types.h kernel.{h,cu} kernel_impl.h
registry.{h,cu} context.{h,cu} dispatch_cache.{h,cu} convert.{h,cuh}
convert_v3.cu cast.{h,cu} unpack.cu gemm_universal.h mainloop_sm80_v2.h
iterator{,_sm80}.h scheduler_sm70.cuh cta_map.h thread_group_map.h
tiled_mma.h operand.h transform.h epilogue.h smem_copy.h predicate.h
cp_async.h matrix_ptr.h thread_map.h arch.h format.h utils.h
moe_utils_v2.{h,cu} gpu_metric.{h,cu}
gemm/arch/: mma_sm80.h operand_sm80_s16816.h config_sm80_s16816.h smem_copy_sm80.h
gemm/kernel/: sm80_16816_{4,8,16}.cu
attention/: quantization.h (cvt_f16x8_u4 only)
core/ (header-only ~60KB): array{,_ops}.h common.h data_type.h
floating_point.h layout.h math.h meta.h mma.h pipe_iter.h smem.h
sub_byte_ptr.h sync.h thread_map.h + check.h logger.h (replace with shims)
STUB OUT: sm70/75/90 registry bodies, tuner/ (force DispatchPolicy::kDefault),
cublas.cu, sm100, LlamaLinear/moe_ffn_layer (engine-coupled — don't vendor).

## Reference/test (reusable)
`test/reference.cu:61` = cuBLAS gemm on **dequantized** FP16 = golden.
`testbed_v3.h:326` = QuantizeGroupwise → prepare → Run → Compare (rel-err,
not bit-exact). `test_gemm_v2.cc:64-76` ALREADY has E=128/top_k=8 — just
set Gemma dims + group_size=32.

## Project adaptation — Genesis overlay
- Vendor → `third_party/tm_int4_moe/` (kernel sources + mini-CMakeLists).
- New patch `g4_85_tm_int4_moe_kernel.py` (sndr/.../moe/): JIT-build via
  cpp_extension on first apply, monkey-patch `CompressedTensorsWNA16MoEMethod.
  apply` / `invoke_fused_moe_wna16_cuda_kernel` to route Marlin-ineligible
  int4 MoE through the TM op. Gated by G4_84's `marlin_moe_marginal()`
  detector (only fires where moe_wna16 would). default_off (experimental,
  build-gated) until rig-validated.
- Offline weight-prep utility: compressed-tensors → TM-packed, cached
  next to the model on the rig.
- A/B harness: `tools/triton_gemm_sweep.py` + G4_84 config-provider table.

## Scaffold plan (order)
**Phase 0 (1-2d):** vendor set → shim out core::Tensor/Context + stub
sm70/75/90 + drop tuner → mini-CMake → `libtm_int4_moe.a` builds under
sm_86, no CUTLASS, no link errors.
**Phase 1 (2-3d, TDD): ✅ DONE 2026-06-22 (rig, 2×A5000 SM86).** Built the
full multi-TU `test_gemm_v2` (`test/buildrun.sh`, 54 TUs) — links + runs;
252 sm80 kernels register (file-probe verified). Full grouped E=128/top_k=8
with real offsets/f2n at the Gemma-4-26B-A4B geometry (hidden=2816,
inter=704, g32, f16 act) dispatches + runs **correct**: rel-err vs the FP16
dequant reference = **0.000356 mean / 1.60 max, ~0 outliers**; quantization
alone adds ~1.88% (int4 g32, not the kernel). **Risk #1 (zero-point format)
RESOLVED** — `fuse_scales_and_zeros` (packV=0x141) decodes `x*s + (-zp*s)`
faithfully. **Key fix surfaced** (carry to Phase 2): MoE expert weights need
`LinearWeight::set_grouped(true)` before `prepare()` so `GetConverters`
yields the *grouped* u4 layout (`order_b` col-major) the grouped kernel
expects — the dense layout's opposite `order_b` causes "No feasible kernel".
**Phase 2 (2-3d):** torch.ops custom op (build MatrixLayout/Operation/
Workspace from data_ptr), offline weight-prep, swap moe_wna16 in FusedMoE,
rig smoke+bench+tool-call on current pin (vllm-pin policy), A/B vs moe_wna16
(speed) + FP16 (accuracy).

## Risks (ranked)
1. Zero-point format (int-zp vs -zp*s half + signed/unsigned int4 shift) —
   mitigate with Phase-1.6 unit test + re-quantize fallback.
2. core:: de-coupling — Gemm::Run is clean but neighbour .cu pull core::Tensor.
3. K % CTA_K — verify Gemma hidden/inter % 32 (almost certainly yes).
