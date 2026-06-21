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
for k in 4 8 16; do
  echo "compiling sm80_16816_${k}.cu ..."
  nvcc $FLAGS -c "src/turbomind/kernels/gemm/kernel/sm80_16816_${k}.cu" \
    -o "build/sm80_16816_${k}.o"
done
echo "OK — 3 tensor-core int4-MoE kernels built for SM86."

# NEXT (Phase 0 cont.): compile gemm.cu + convert/cast + registry + moe_utils_v2,
# stub sm70/75/90 registry bodies + drop tuner, then link into libtm_int4_moe.a.
# Phase 1: cuBLAS reference + weight-repack byte-test. Phase 2: torch.ops op.
