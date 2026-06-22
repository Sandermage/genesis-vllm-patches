#!/bin/bash
# Compile the vendored TurboMind engine objects (no test harness) and build the
# torch extension probe. Run inside vllm/vllm-openai:nightly with the vendored
# tree mounted at /work and --gpus all.
set -e
cd /work
find src third_party -name "._*" -delete 2>/dev/null || true
mkdir -p build

# -Xcompiler -fPIC: the objects are linked into a shared object (torch ext).
F="-arch=sm_86 -std=c++17 -DENABLE_BF16 -DFMT_HEADER_ONLY --expt-relaxed-constexpr \
   --extended-lambda -Xcompiler -fPIC -include cuda_fp16.h -include cuda_bf16.h -I. \
   -Ithird_party/fmt/include -Ithird_party/moodycamel"

# Engine TUs only: drop the whole test/ harness (main + cublas-reference) and
# the disabled sm90_64n32 / anomaly_handler.
SRCS=$(find src/turbomind -name "*.cu" -o -name "*.cc" \
  | grep -vE "/test/|sm90_64n32|anomaly_handler|gemm_bench|test_logger|test_core|test_scope|test_data_format|test_moe_utils")

n=0; fails=""
for f in $SRCS; do
  o=build/$(echo "$f" | tr '/' '_' | sed 's/\.\(cu\|cc\)$/.o/')
  [ -f "$o" ] && continue   # reuse already-compiled (fPIC) object
  if nvcc $F -c "$f" -o "$o" 2>/dev/null; then n=$((n+1)); else fails="$fails $(basename $f)"; fi
done
echo "compiled: $n / $(echo "$SRCS" | wc -w)  FAILED:$fails"

python3 -u /work/torch_ext/build_ext.py
