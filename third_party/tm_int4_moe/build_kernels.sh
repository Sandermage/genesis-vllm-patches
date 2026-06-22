#!/bin/bash
# Build TurboMind sm80_16816 int4 MoE tensor-core kernels standalone (SM86).
# PROVEN 2026-06-22 in vllm/vllm-openai:nightly (nvcc 13.0, torch 2.11):
# all three kernels compile to object files on a 2x A5000 rig.
#
# Run inside the vLLM image with internet (apt fmt):
#   docker run --rm --network host --entrypoint bash \
#     -v $(pwd):/work:ro vllm/vllm-openai:nightly -c "bash /work/build_kernels.sh"
set -e

# fmt: prefer apt (resolves cleanly). Fallback = vendored header-only fmt:
#   add  -DFMT_HEADER_ONLY -Ithird_party/fmt/include  and skip the apt line.
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libfmt-dev >/dev/null 2>&1

# Key flags discovered build-driven:
#  -DENABLE_BF16           : TurboMind bf16 MMA path uses nv_bfloat16
#  -include cuda_fp16/bf16 : the kernel headers assume these are already pulled
#  --expt-relaxed-constexpr: constexpr in __device__ helpers
#  -I.                     : TurboMind uses absolute "src/turbomind/..." includes
FLAGS="-arch=sm_86 -std=c++17 -DENABLE_BF16 --expt-relaxed-constexpr \
  -include cuda_fp16.h -include cuda_bf16.h -I."

mkdir -p build

# Engine TUs (Gemm::Run + registry + converters + MoE gate). PROVEN: all 13
# objects compile and link into libtm_int4_moe.a (40 MB) on SM86, 2026-06-22.
for src in gemm registry dispatch_cache gpu_metric kernel \
           convert_v3 unpack cast context moe_utils_v2; do
  echo "compiling ${src}.cu ..."
  nvcc $FLAGS -c "src/turbomind/kernels/gemm/${src}.cu" -o "build/${src}.o"
done
# The 3 tensor-core int4-MoE kernels
for k in 4 8 16; do
  echo "compiling sm80_16816_${k}.cu ..."
  nvcc $FLAGS -c "src/turbomind/kernels/gemm/kernel/sm80_16816_${k}.cu" \
    -o "build/sm80_16816_${k}.o"
done
ar rcs build/libtm_int4_moe.a build/*.o
echo "OK — libtm_int4_moe.a: $(stat -c%s build/libtm_int4_moe.a) bytes, 13 objects (SM86)."

# NEXT — Phase 1: cuBLAS reference (test/reference.cu) + a thin testbed calling
#   Gemm::Run directly; weight-repack byte-test (zero-point format = #1 risk).
# Phase 2: torch.ops custom op (build MatrixLayout/Operation/Workspace from
#   data_ptr) + offline weight-prep + swap moe_wna16 in FusedMoE + rig A/B.
