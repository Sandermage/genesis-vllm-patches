# Genesis vLLM Patches — Installation Guide

Step-by-step setup for running Genesis-patched vLLM on NVIDIA Ampere (validated on 2× RTX A5000) for Qwen3.6-class long-context inference.

---

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches

# 2. Pull a recent vLLM nightly image
docker pull vllm/vllm-openai:nightly

# 3. Download Qwen3.6-35B-A3B-FP8 weights to /nfs/genesis/models (or adapt path)
huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir /nfs/genesis/models/Qwen3.6-35B-A3B-FP8

# 4. Run with our example compose
docker compose -f docker-compose.example.yml up -d

# 5. Watch boot (5-8 min for cold compile cache; 1-2 min warm)
docker logs -f vllm-genesis | grep -E "Genesis|HEALTHY|Started"

# 6. Health check
curl http://localhost:8000/health -H "Authorization: Bearer genesis-local"
```

---

## Hardware requirements

### Tested configurations

| Hardware | Validation status | Notes |
|---|---|---|
| 2× RTX A5000 24GB (Ampere SM 8.6) | **Primary** — full v7.52 stack tested (driver 580.126.09 / CUDA 13.0 / vLLM dev212+g8cd174fa3) | Default config targets this |
| 1× RTX 3090 24GB | Cross-validated by [@noonghunna](https://github.com/noonghunna/qwen36-27b-single-3090) | Same SM 8.6 family |
| 2× RTX 3090 24GB | Cross-validated by [@noonghunna](https://github.com/noonghunna/qwen36-dual-3090) | TP=2 PCIe Gen4 (no NVLink) |

### Minimum requirements

- **GPU**: NVIDIA Ampere SM 8.0+ (A100, A5000, A6000, RTX 3090/3090Ti, A40)
- **VRAM**: 24GB per GPU minimum (48GB total for default Qwen3.6-35B-A3B-FP8)
- **CUDA**: **13.0** (current vLLM nightly ships with PyTorch 2.11+cu130)
- **Driver**: **NVIDIA ≥ 580.126.09 REQUIRED** as of v7.48 (2026-04-27). Driver 570 still loads but PyTorch falls into compat mode → ~3× slower decode. Install via `apt install nvidia-driver-580-server` on Ubuntu 24.04, then reboot. See [`scripts/launch/README.md`](scripts/launch/README.md) for the full version matrix.
- **System RAM**: 64GB+ (model weights need to be paged in)
- **Disk**: ~40GB for FP8 model weights, +10GB for vLLM compile cache

### Other architectures (best-effort, no first-class support)

- AMD ROCm: patches graceful-skip on platform mismatch (don't crash). Untested.
- Intel XPU: same.
- Hopper / Blackwell: patches detect SM and skip Ampere-specific code (e.g., Marlin FP8 weight-only path is unnecessary on Hopper which has native FP8). Use upstream vLLM directly.

---

## Step-by-step setup

### 1. Repository layout

```
genesis-vllm-patches/
├── README.md                          # Overview + changelog
├── INSTALL.md                         # This file
├── MODELS.md                          # Supported models + selection guide
├── QUICKSTART.md                      # Original (still valid)
├── CREDITS.md                         # Attributions
├── docker-compose.example.yml         # Default (Qwen3.6-35B-A3B-FP8 MTP)
├── docker-compose.qwen3-5-dense.yml   # Qwen3.6-27B dense variant
├── docker-compose.gemma4-26b-moe.yml  # Gemma 4 (experimental)
├── docker-compose.integration*.yml    # Integration test variants
├── vllm/_genesis/                     # The Genesis package (bind-mounted into container)
│   ├── dispatcher.py                  # Patch registry + dispatch logic
│   ├── prealloc_budget.py             # P73 central resolver
│   ├── kernels/                       # Custom Triton kernels + helpers
│   ├── wiring/                        # Text-patch definitions (one file per patch)
│   ├── patches/apply_all.py           # Patch orchestrator (called at container start)
│   └── tests/                         # Patch unit + integration tests
├── external_probe/                    # Pre-Genesis startup probes (tolist bypass etc.)
└── benchmarks/                        # Bench scripts (genesis_bench_v3.py etc, now in scripts/)
```

### 2. Container architecture

The Genesis approach: **bind-mount our `_genesis/` package into a stock vLLM image**, run `apply_all.py` at container start to text-patch upstream files, then `exec vllm serve`.

This means:
- No need to fork or rebuild vLLM
- Patches apply transparently — visible to operator via boot logs
- New vLLM nightly versions can be tried without recompiling — pull image, restart container, observe drift markers
- `_genesis/` is the only thing under version control we ship

### 3. Pre-flight checks

```bash
# Verify GPU + driver
nvidia-smi
# Look for: Driver Version >= 580.126.09 (REQUIRED), CUDA Version: 13.0, all GPUs visible, ECC OK

# Verify NCCL (for TP>1)
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi -L

# Verify model weights
ls -lh /nfs/genesis/models/Qwen3.6-35B-A3B-FP8
# Expect: ~40GB across multiple safetensors shards + config.json + tokenizer files
```

### 4. First boot (cold compile cache)

First run takes 5-8 minutes for torch.compile + cudagraph capture. Subsequent runs (warm cache) take 1-2 minutes.

```bash
docker compose -f docker-compose.example.yml up -d
docker logs -f vllm-genesis 2>&1 | grep -E "Genesis|Capturing CUDA|Loading|Started server"
```

Expected log progression:
1. Genesis dispatcher prints applied/skipped patches (~10 sec)
2. Model weights load from disk (~30-60 sec)
3. torch.compile + Inductor pass (~2-4 min cold, ~30 sec warm)
4. CUDA graph capture (~30-60 sec)
5. `Started server process` — ready for requests

### 5. Verify patches applied correctly

```bash
docker logs vllm-genesis 2>&1 | grep "Genesis Dispatcher"
# Expect: 30+ APPLY lines, ~10 SKIP lines (opt-in patches not enabled, or platform mismatch)

docker logs vllm-genesis 2>&1 | grep "applied" | wc -l
# Expect: 35-37 active patches
```

### 6. Smoke test

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer genesis-local" \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "prompt":"Explain Triton in one paragraph:",
    "max_tokens":100,
    "temperature":0.0
  }'
```

Expect ~127 tok/s (MTP default).

---

## Bare-metal install (without Docker)

This path installs vLLM + Genesis directly on the host (Ubuntu 24.04 / Debian 12 / RHEL 9). Use this if you don't want containers, are on a system without Docker GPU support, or want to develop / iterate on patches without the container R/W layer trap (see [`CONFIGURATION.md`](CONFIGURATION.md) "Container R/W layer note").

**Trade-offs vs Docker:**

- ✅ No `docker compose down/up` cycle — just restart the Python process
- ✅ Source-level edits to `_genesis/` apply on next process restart (no bind-mount needed)
- ✅ Easier to debug with `pdb` / `py-spy` / `nsys` — no container PID translation
- ❌ You manage the Python environment, NVIDIA driver, CUDA, Triton, PyTorch versions yourself
- ❌ vLLM's CI builds and tests primarily on the official Docker image; bare-metal is your responsibility to keep in sync
- ❌ Patches text-modify files in your `site-packages/vllm/` — you must back up before patching, and `pip install --upgrade vllm` will silently undo Genesis (re-apply afterwards)

### Bare-metal prerequisites

```bash
# 1. NVIDIA driver — see "Hardware requirements" above
nvidia-smi  # must show driver ≥ 580.126.09

# 2. CUDA toolkit (for nvcc, Triton compilation)
sudo apt install cuda-toolkit-13-0
# or use NVIDIA's official .run installer
nvcc --version  # must show release 13.0

# 3. Python 3.12 (vLLM nightly requirement as of 2026-04-27)
sudo apt install python3.12 python3.12-venv python3.12-dev
python3.12 --version  # 3.12.x

# 4. System libs vLLM needs
sudo apt install build-essential pkg-config libssl-dev \
                  libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
                  libjpeg-dev libpng-dev libsndfile1
```

### 1. Create dedicated Python environment

**Use a venv or conda env. Do NOT install into system Python** — vLLM's dependency tree (PyTorch, Triton, FlashInfer, xformers) will conflict with your distro's packages.

```bash
# Pick a location with at least 30 GB free for the env
mkdir -p ~/vllm-genesis && cd ~/vllm-genesis

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### 2. Install vLLM nightly

Genesis is pinned to a specific vLLM nightly. Find the SHA / version that matches our [`Production baseline`](README.md#production-baseline) — currently `0.19.2rc1.dev205+g07351e088`.

```bash
# Option A — install from a specific nightly wheel (recommended if you can match)
pip install --pre vllm==0.19.2rc1.dev205+g07351e088 \
  --extra-index-url https://wheels.vllm.ai/nightly

# Option B — install from source at a specific commit
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 07351e088  # match the SHA from Production baseline
pip install -e . --no-build-isolation
cd ..

# Option C — install latest nightly (may drift from our anchors;
# `apply_all` will skip patches whose anchors no longer match — observable in startup log)
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

PyTorch / Triton / FlashInfer should be pulled in as vLLM dependencies. Verify:

```bash
python3 -c "
import vllm, torch, triton
print(f'vllm {vllm.__version__}')
print(f'torch {torch.__version__} cuda={torch.version.cuda}')
print(f'triton {triton.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'cuda devices: {torch.cuda.device_count()}')
"
# Expect:
# vllm 0.19.2rc1.dev205+g07351e088
# torch 2.11.0+cu130 cuda=13.0
# triton 3.6.0
# cuda available: True
# cuda devices: 2  (or however many GPUs you have)
```

### 3. Install Genesis package into vLLM's `site-packages`

Genesis is a Python module at `vllm/_genesis/`. We need to drop it into the **same `vllm/` package directory** that vLLM itself installed.

```bash
# Find where vLLM lives
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "vLLM installed at: $VLLM_DIR"
# Example: /home/sander/vllm-genesis/.venv/lib/python3.12/site-packages/vllm

# Clone the Genesis patch repo
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches

# Symlink _genesis/ into vLLM's package directory
# (symlink, not copy — so `git pull` updates the live install in place)
ln -s "$(pwd)/vllm/_genesis" "$VLLM_DIR/_genesis"

# Verify import works
python3 -c "from vllm import _genesis; print(_genesis.__file__)"
# Expect: .../site-packages/vllm/_genesis/__init__.py
```

If you prefer **copy over symlink** (e.g., for a frozen production install):

```bash
cp -r vllm/_genesis "$VLLM_DIR/_genesis"
# After updating the patcher repo: rsync -a vllm/_genesis/ "$VLLM_DIR/_genesis/"
```

### 4. Install Genesis runtime extras

Genesis needs a few extra packages at startup (`pandas`, `scipy`, `xxhash` for prefix-cache hash, optionally `arctic-inference` for Suffix Decoding P75):

```bash
pip install pandas scipy xxhash
# Optional — only if you plan to use P75 (suffix decoding):
pip install arctic-inference
```

Install the Genesis vLLM plugin (auto-loads via vLLM entry point):

```bash
cd genesis-vllm-patches
pip install --no-deps -e ./genesis_vllm_plugin
```

### 5. Apply external probes (recommended)

Genesis depends on two startup probes that text-patch upstream vLLM:

```bash
# These run once per Python process startup; they are idempotent
python3 external_probe/patch_tolist_cudagraph.py
python3 external_probe/patch_40074_iooo.py
```

These can also be run once after install — they modify files in `$VLLM_DIR` directly.

### 6. Run apply_all (text-patches vLLM source)

```bash
cd genesis-vllm-patches  # or anywhere — the module is now importable

# Set patch enable flags FIRST (env vars are read at apply time):
export GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1
export GENESIS_ENABLE_P67B=1
export GENESIS_ENABLE_P82=1
export GENESIS_P82_THRESHOLD_SINGLE=0.3
export GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1
export GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1
export GENESIS_BUFFER_MODE=shared
# (See CONFIGURATION.md for the full list)

# Apply patches (text-modifies $VLLM_DIR/v1/sample/rejection_sampler.py etc.)
python3 -m vllm._genesis.patches.apply_all
# Watch for: "Genesis Dispatcher" matrix output, [P82] applied, etc.
```

**Important:** patches are idempotent — running `apply_all` twice is safe. They include drift detectors that gracefully SKIP if upstream changes the anchor.

To **reverse all patches** (e.g., before `pip install --upgrade vllm`):

```bash
# The simplest way: reinstall vLLM from scratch
pip install --force-reinstall --no-deps vllm
# Or restore from backup if you made one before applying
```

### 7. Launch vLLM

```bash
# Production-equivalent invocation (Qwen3.6-35B-A3B-FP8, TP=2, MTP K=3, TurboQuant k8v4):
vllm serve --model /path/to/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.91 \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_k8v4 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --dtype float16 \
  --disable-custom-all-reduce \
  --language-model-only \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --api-key genesis-local \
  --served-model-name qwen3.6-35b-a3b \
  --host 0.0.0.0 --port 8000 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --async-scheduling \
  --performance-mode interactivity \
  --no-scheduler-reserve-full-isl \
  --prefix-caching-hash-algo xxhash \
  --disable-log-stats
```

For convenience, save the env vars + serve command into a launch script (modeled on [`scripts/launch/`](scripts/launch/)):

```bash
cat > ~/run-genesis.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
source ~/vllm-genesis/.venv/bin/activate

# Patch enable flags
export GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1
export GENESIS_ENABLE_P82=1 GENESIS_P82_THRESHOLD_SINGLE=0.3
export GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1
export GENESIS_BUFFER_MODE=shared
# (... full list — see CONFIGURATION.md)

# Apply patches (idempotent)
python3 -m vllm._genesis.patches.apply_all

# Launch
exec vllm serve --model "${MODEL_PATH:-/path/to/Qwen3.6-35B-A3B-FP8}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.91 \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_k8v4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --async-scheduling --performance-mode interactivity \
  --api-key "${VLLM_API_KEY:-genesis-local}" \
  --port "${PORT:-8000}"
EOF
chmod +x ~/run-genesis.sh
~/run-genesis.sh
```

### 8. Run as a systemd service (production)

```bash
sudo tee /etc/systemd/system/genesis-vllm.service > /dev/null << 'EOF'
[Unit]
Description=Genesis vLLM Patches Production Server
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=sander
Group=sander
WorkingDirectory=/home/sander/vllm-genesis
Environment="MODEL_PATH=/path/to/Qwen3.6-35B-A3B-FP8"
Environment="VLLM_API_KEY=YOUR_KEY_HERE"
ExecStart=/home/sander/run-genesis.sh
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity
LimitSTACK=67108864

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now genesis-vllm
sudo systemctl status genesis-vllm
journalctl -u genesis-vllm -f
```

### 9. Updating Genesis on bare-metal

```bash
# Pull latest patches from git
cd ~/vllm-genesis/genesis-vllm-patches
git pull origin main

# If you symlinked _genesis/, the live install is already updated — just restart vLLM:
sudo systemctl restart genesis-vllm

# If you copied (not symlinked), re-sync first:
rsync -a vllm/_genesis/ "$(python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')/_genesis/"
sudo systemctl restart genesis-vllm
```

### 10. Updating vLLM on bare-metal (rare, careful)

When upstream vLLM ships a new nightly that includes patches Genesis backports, our drift markers will detect it and SKIP those patches automatically. To upgrade:

```bash
# 1. Save current state
pip freeze > ~/vllm-pre-upgrade.txt

# 2. Upgrade vLLM
pip install --upgrade --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly

# 3. Re-apply Genesis (it's idempotent + drift-aware)
python3 -m vllm._genesis.patches.apply_all
# Watch the dispatcher matrix — newly-merged-upstream patches will show:
#   PXX | SKIP | <title> | upstream may have absorbed this fix
# That's correct — drop the corresponding GENESIS_ENABLE_PXX=1 flag from your env.

# 4. Restart
sudo systemctl restart genesis-vllm
```

### 11. Bare-metal troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named 'vllm._genesis'` | Symlink missing or wrong path | Re-run step 3 (`ln -s` into `$VLLM_DIR`) |
| Boot hangs on `Capturing CUDA graphs` | Driver mismatch (570 instead of 580) or stale Triton cache | `apt install nvidia-driver-580-server`, reboot. `rm -rf ~/.triton/cache/*` |
| `apply_all` reports `required_anchor_missing` for many patches | vLLM nightly drifted from Genesis pin | Pin to the SHA in [`Production baseline`](README.md#production-baseline), or accept that some patches will skip (read each SKIP reason) |
| Patches re-apply on every restart and accumulate | You're running `apply_all` from multiple processes simultaneously | Add a lockfile, or run apply_all once at boot before launching workers |
| `pip install --upgrade vllm` silently undid Genesis | Expected — `pip` reinstalls vLLM's own files cleanly | Re-run step 6 (apply_all) after upgrade |

For more troubleshooting see the [`Troubleshooting`](#troubleshooting) section below — most container-side issues apply equally to bare-metal.

---

## Environment variables — full reference

### Genesis enable/disable flags (opt-in patches)

All Genesis patches are opt-in by default. Set the matching env var to `1` to enable.

| Env var | Patch | What it does |
|---|---|---|
| `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD` | P56 | Spec-decode safe-path guard (deprecated workaround) |
| `GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE` | P57 | Capture-safe buffer expansion for spec-decode (experimental) |
| `GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX` | P58 | Async-scheduler `[-1]` placeholder fix (root cause for #40831) |
| `GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY` | P59 | Backport of vllm#39055 (qwen3 reasoning embedded tool_call). **Currently superseded by upstream PR #35687 in our pin — keep disabled** |
| `GENESIS_ENABLE_P60_GDN_NGRAM_FIX` | P60 | GDN+ngram SSM state recovery (Phase 1) |
| `GENESIS_ENABLE_P60B_TRITON_KERNEL` | P60b | GDN+ngram conv state Triton kernel offset (Phase 2) |
| `GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL` | P61 | Qwen3 multi-tool first-occurrence (vs LAST in upstream) |
| `GENESIS_ENABLE_P61B_STREAMING_OVERLAP` | P61b | Streaming partial-tag overlap guard |
| `GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING` | P62 | Reasoning-aware grammar acceptance + spec-token validation |
| `GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY` | P63 | **DEPRECATED** — kept only for archival diagnostics |
| `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING` | P64 | qwen3coder streaming early-return fix (vllm#39598 backport) |
| `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE` | P65 | Cudagraph downgrade for spec-decode (workaround; replaced by P67/P67b) |
| `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER` | P66 | Filter cudagraph_capture_sizes by spec-decode divisibility |
| `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL` | P67/P67b | TurboQuant multi-query kernel for spec-decode K+1 verify (proper fix for #40880, replaces P65) |
| `GENESIS_ENABLE_P68_AUTO_FORCE_TOOL` | P68 | Auto-upgrade `tool_choice=auto → required` for long-ctx tool calls |
| `GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER` | P69 | Append format reminder to last user msg on long-ctx |
| `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM` | P70 | Auto-bump ngram `prompt_lookup_min ≥ 8` |
| `GENESIS_ENABLE_P71_BLOCK_VERIFY` | P71 | Block-verify rejection sampler (Sun 2024) — opt-in experimental |
| `GENESIS_ENABLE_P72_PROFILE_RUN_CAP` | P72 | Cap profile_run M to unblock `--max-num-batched-tokens > 4096` |
| `GENESIS_ENABLE_P74_CHUNK_CLAMP` | P74 | Auto chunk-clamp via `long_prefill_token_threshold` (P72 companion) |
| `GENESIS_ENABLE_P75_SUFFIX_DECODING` | P75 | Auto-swap `method=ngram → method=suffix` (Arctic Inference) |
| `GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K` | P77 | Adaptive ngram K controller (EMA + hysteresis + auto-disable) |

### Genesis tunable parameters

| Env var | Default | Description |
|---|---|---|
| `GENESIS_TQ_MAX_MODEL_LEN` | (auto-detect) | TurboQuant max sequence length for buffer sizing. Set to `262144` for full 256K context. |
| `GENESIS_PROFILE_RUN_CAP_M` | `4096` | P72: cap M passed to `_dummy_run` during profile_run |
| `GENESIS_PREALLOC_TOKEN_BUDGET` | (uses scheduler config) | P73 central budget for all prealloc patches. Set to `4096` to keep prefill chunks safe with batched=8192 |
| `GENESIS_GDN_MAX_BATCHED_TOKENS` | `4096` | P28: GDN core_attn_out prealloc size (back-compat — superseded by P73) |
| `GENESIS_MOE_MAX_BATCHED_TOKENS` | `4096` | P37: MoE intermediate cache size (back-compat) |
| `GENESIS_TQ_MAX_BATCHED_TOKENS` | `4096` | P26/P44: TurboQuant prealloc fallback (back-compat) |
| `GENESIS_P67_USE_UPSTREAM` | `1` | P67: use upstream `triton_turboquant_decode` instead of our v7.22 kernel (drift-free) |
| `GENESIS_P67_NUM_KV_SPLITS` | `32` | P67: multi-CTA parallelism for upstream path |
| `GENESIS_P67_MAX_PRIOR_LEN` | `4096` | P67: threshold above which fall through to upstream (drift safety) |
| `GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS` | `8000` | P68/P69: trigger threshold for long-ctx tool adherence patches |
| `GENESIS_P77_STEPS` | `0,1,3,5` | P77: discrete K choices for adaptive controller (0 = disable spec) |
| `GENESIS_P77_EMA_ALPHA` | `0.2` | P77: EMA smoothing factor |
| `GENESIS_P77_DISABLE_THRESHOLD` | `0.30` | P77: accept rate below → drop to K=0 (no-spec) |
| `GENESIS_P77_PROBE_INTERVAL` | `100` | P77: every N batches, force K>0 to retest acceptance |
| `GENESIS_P75_TREE_DEPTH` | `24` | P75: suffix tree max depth |
| `GENESIS_P75_SPEC_FACTOR` | `2.0` | P75: max draft length factor |
| `GENESIS_P75_MIN_PROB` | `0.10` | P75: branch probability threshold for emission |
| `GENESIS_P75_CACHE_REQS` | `10000` | P75: cross-request suffix-tree cache size |

### Standard vLLM env (for reference)

| Env var | Value we use | Why |
|---|---|---|
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Accurate memory profiling — needed for GMU > 0.88 safely |
| `VLLM_NO_USAGE_STATS` | `1` | No telemetry to vLLM project |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | Faster sampler kernel (no perf degradation in our config) |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | `1` | Use fused MoE top-k kernel |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `high` | TF32 path for non-attention matmul |
| `VLLM_LOGGING_LEVEL` | `WARNING` | Silence noise |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Cleaner per-worker process isolation |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | ~2% gain on Ampere Marlin reductions |
| `VLLM_MOE_USE_DEEP_GEMM` / `VLLM_USE_DEEP_GEMM` | `0` | Hopper-only kernel path; force off on Ampere |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `0` | Not stable with TurboQuant; leave off |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow `--max-model-len 262144` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512` | Better fragmentation behavior under long-context dynamic shapes |
| `NCCL_P2P_DISABLE` | `1` | A5000 doesn't have NVLink → P2P over PCIe is unreliable, use staged copy instead |
| `OMP_NUM_THREADS` | `1` | Tight OMP usage; numba/Triton handles their own threading |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | Improves multi-stream overlap |

---

## Common operational scenarios

### Scenario 1: Free-form chat workload (default)

Use MTP. No env tweaks needed.

```yaml
# docker-compose.example.yml (snippet)
command:
  - "exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8
     --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'
     ..."
```

Expected: 127 tok/s mean.

### Scenario 2: Tool-call / agentic-heavy workload

Enable P75 (Suffix Decoding) — best results.

```yaml
environment:
  GENESIS_ENABLE_P75_SUFFIX_DECODING: "1"
  # also need arctic-inference installed in container:
  # add `arctic-inference` to pip install line in entrypoint
command:
  - "... --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":3}' ..."
  # P75 auto-swaps method=ngram → method=suffix
```

Expected: 99 tok/s mean, peak 175 on highly repetitive batches.

### Scenario 3: Need batched_tokens > 4096 (large prefill batches)

Enable P72 + P74 together.

```yaml
environment:
  GENESIS_ENABLE_P72_PROFILE_RUN_CAP: "1"
  GENESIS_PROFILE_RUN_CAP_M: "4096"
  GENESIS_ENABLE_P74_CHUNK_CLAMP: "1"
  GENESIS_PREALLOC_TOKEN_BUDGET: "4096"
command:
  - "... --max-num-batched-tokens 8192 ..."
```

Long-context up to 252K tokens verified safe with this combo.

### Scenario 4: Ngram-only deployment (no MTP available)

Enable P77 adaptive controller.

```yaml
environment:
  GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K: "1"
  GENESIS_P77_DISABLE_THRESHOLD: "0.30"  # drop to K=0 if accept < 30%
command:
  - "... --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_min\":8}' ..."
```

P77 will auto-tune K={0,1,3,5} per acceptance rate; drops to K=0 (no-spec mode, ~150 tok/s) on free-form text where ngram contributes nothing.

---

## Troubleshooting

### Boot fails with `cudaErrorStreamCaptureInvalidated`

You probably enabled spec-decode without P67/P67b. Either:
- Enable `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1` (proper fix), or
- Enable `GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=1` (workaround — disables FULL CG, costs ~30% throughput)

### Boot fails with `RuntimeError: tensor a (65536) must match tensor b (16*s72)`

You set `--max-num-batched-tokens > 4096` without P72. Enable `GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1`.

### Long-context (180K+) crashes with `setStorage out of bounds`

You set batched > 4096 with P72 but not P74. Enable `GENESIS_ENABLE_P74_CHUNK_CLAMP=1` + `GENESIS_PREALLOC_TOKEN_BUDGET=4096`.

### `Worker proc died unexpectedly` during cudagraph capture

Likely OOM. Lower `--gpu-memory-utilization` from current to 0.88.

### Tool-call cascades / `<tool_call>\n<tool_call>...` repeats

Make sure these are enabled:
- `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1` (root cause for #40880)
- `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1` (filter weak ngram drafts)
- `GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1` (streaming early-return fix)

### Empty `tool_calls` in response

Enable `GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1` (FIRST occurrence vs upstream LAST) and `GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1`.

### Container stops cleanly responding mid-generation

Check `docker logs --tail 200` for the actual exception. Common cause: model arch mismatch — confirm `--model` path points to a Qwen3.5 MoE variant. For dense Qwen3.6-27B use `docker-compose.qwen3-5-dense.yml`.

---

## Updating to new vLLM nightly

Genesis patches use **drift-marker** detection: if upstream introduces equivalent code, the patch refuses to apply (returns SKIPPED) and logs the marker that triggered the skip. To update:

```bash
docker pull vllm/vllm-openai:nightly
docker compose down
docker compose up -d
docker logs vllm-genesis 2>&1 | grep -E "drift marker|skipped"
```

If a patch you rely on now skips with "drift marker found", congrats — upstream absorbed the fix. You can delete the env enable flag for that patch (or leave it; the SKIP is harmless).

---

## Where to next

- See [README.md](README.md) for changelog + benchmark history
- See [MODELS.md](MODELS.md) for supported models + how to choose
- See [QUICKSTART.md](QUICKSTART.md) for the original quick-start guide
- See `vllm/_genesis/wiring/patch_*.py` for individual patch source — each file has a detailed docstring explaining the bug it fixes
