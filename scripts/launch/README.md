# Genesis vLLM — Launch Scripts

Production-tested launch scripts for Genesis Patches. **4 PROD-ready configs**
covering Qwen3.6-35B-A3B-FP8 (1 variant) and Qwen3.6-27B-int4-Lorbus
(3 variants). Each ships in **two flavors**:

- `start_*.sh` — **Docker** (recommended for reproducibility; bind-mounts
  Genesis patches into stock `vllm/vllm-openai:nightly`)
- `bare_metal_*.sh` — **Native** (assumes vLLM installed via pip on the host;
  symlinks Genesis `_genesis` package into the existing vllm install)

Plus 3 utility scripts and an `_archive/` of historical / research arms.

---

## Quick reference

### TP=2 (validated PROD on 2× RTX A5000)

| Model | Config | Docker | Bare metal |
|---|---|---|---|
| **35B-A3B-FP8** PROD (TQ k8v4 + MTP K=3 + PN8) | `--max-model-len 320000` | [`start_35b_fp8_PROD.sh`](start_35b_fp8_PROD.sh) | [`bare_metal_35b_fp8_PROD.sh`](bare_metal_35b_fp8_PROD.sh) |
| **27B-INT4 Lorbus** short-ctx (no TQ, fp8_e5m2) | `--max-model-len 131072` `--max-num-seqs 4` | [`start_27b_int4_no_TQ_short.sh`](start_27b_int4_no_TQ_short.sh) | [`bare_metal_27b_int4_no_TQ_short.sh`](bare_metal_27b_int4_no_TQ_short.sh) |
| **27B-INT4 Lorbus** long-ctx 256K (no TQ) | `--max-model-len 280000` `--max-num-seqs 2` `--gpu-mem-util 0.90` | [`start_27b_int4_no_TQ_long_256K.sh`](start_27b_int4_no_TQ_long_256K.sh) | [`bare_metal_27b_int4_no_TQ_long_256K.sh`](bare_metal_27b_int4_no_TQ_long_256K.sh) |
| **27B-INT4 Lorbus** TQ k8v4 (hybrid + P98) | TQ packed slot + 256K capable | [`start_27b_int4_TQ_k8v4.sh`](start_27b_int4_TQ_k8v4.sh) | [`bare_metal_27b_int4_TQ_k8v4.sh`](bare_metal_27b_int4_TQ_k8v4.sh) |

### TP=1 single-card (⚠️ EXPERIMENTAL — NOT TESTED by maintainer)

These are the TP=1 derivatives of the four PROD configs above. Same Genesis env
flags + `--tensor-parallel-size 1`, otherwise identical. Sander runs 2× A5000
so these have NOT been benched / stress-tested end-to-end. Each script carries
a prominent EXPERIMENTAL warning header and per-card sizing notes.

If you run one and it works, please share results via [GitHub Discussions](https://github.com/Sandermage/genesis-vllm-patches/discussions) — we'll fold confirmed configs back into the main table and drop the experimental tag for that card class.

| Model | Card class fit | Docker | Bare metal |
|---|---|---|---|
| **35B-A3B-FP8** (~35 GB weights) | 48 GB+ cards (A6000, 6000 Ada, L40, RTX PRO 5000/6000 Blackwell, A100, H100, B200) | [`start_35b_fp8_PROD_single_card.sh`](start_35b_fp8_PROD_single_card.sh) | [`bare_metal_35b_fp8_PROD_single_card.sh`](bare_metal_35b_fp8_PROD_single_card.sh) |
| **27B-INT4 Lorbus** short-ctx (~14 GB weights) | 24 GB+ cards (3090, 4090, 5090, A5000, etc.) | [`start_27b_int4_no_TQ_short_single_card.sh`](start_27b_int4_no_TQ_short_single_card.sh) | [`bare_metal_27b_int4_no_TQ_short_single_card.sh`](bare_metal_27b_int4_no_TQ_short_single_card.sh) |
| **27B-INT4 Lorbus** long-ctx 256K | 24 GB+ (tighter — may need to lower max-model-len on 24 GB) | [`start_27b_int4_no_TQ_long_256K_single_card.sh`](start_27b_int4_no_TQ_long_256K_single_card.sh) | [`bare_metal_27b_int4_no_TQ_long_256K_single_card.sh`](bare_metal_27b_int4_no_TQ_long_256K_single_card.sh) |
| **27B-INT4 Lorbus** TQ k8v4 + P98 | 24 GB+ | [`start_27b_int4_TQ_k8v4_single_card.sh`](start_27b_int4_TQ_k8v4_single_card.sh) | [`bare_metal_27b_int4_TQ_k8v4_single_card.sh`](bare_metal_27b_int4_TQ_k8v4_single_card.sh) |

**Empirical numbers** (2× RTX A5000 24 GB, vLLM nightly pin `8cd174fa3`,
N=500 stress over 200 min continuous):

| Variant | wall_TPS | CV | tool-call | max stable ctx |
|---|---:|---:|:---:|:---:|
| 35B PROD | **183.05** | 8.92% | 3/4 (case 4 = max_tokens artifact, NOT regression) | 128K (256K+ needs longer timeout) |
| 27B no-TQ short | **89.23** | 9.97% | 4/4 | 8K (OOMs at 16K with util=0.95) |
| 27B no-TQ long-ctx | ~80 (estimate) | n/a | 4/4 (with `--max-tokens 1500`) | **256K verified** (262 104 tokens, 311 s prefill) |
| 27B TQ k8v4 | **90.49** | 10.08% | 4/4 | 256K capable; +1.9% over fp8_e5m2 (Welch p=0.067 NS) |

---

## Tested environment

- vLLM `0.20.1rc1.dev16+g7a1eb8ac2` (image `vllm/vllm-openai:nightly`,
  pin commit `8cd174fa3`)
- PyTorch `2.11.0+cu130`, Triton `3.6.0`, CUDA `13.0`
- **NVIDIA driver `≥580.126.09`** (REQUIRED — driver 570 puts PyTorch in
  compat fallback ≈ 3× slower decode)
- 2× RTX A5000 (Ampere SM 8.6), Ubuntu 24.04 + kernel 6.8

Docker users: `vllm/vllm-openai:nightly` ships everything pre-installed; you
only need NVIDIA Container Toolkit on the host.

Bare-metal users: install vLLM via `pip install vllm`, `pip install
flashinfer-python`, then point `GENESIS_REPO` env to your git clone.

---

## Quick start (Docker)

```bash
# 1. Adjust env paths to match your system
export MODELS_DIR=/path/to/your/models       # required (must contain the model dir)
export GENESIS_REPO=$HOME/genesis-vllm-patches
export HF_CACHE=$HOME/.cache/huggingface
export VLLM_CACHE_BASE=$HOME/.cache/genesis_vllm
export CONTAINER_NAME=vllm-genesis

# 2. Pick a script and launch
./scripts/launch/start_35b_fp8_PROD.sh

# 3. Wait ~3-5 min for cold compile cache (subsequent boots ~1-2 min)
docker logs -f $CONTAINER_NAME

# 4. Health check
curl http://localhost:8000/v1/models -H "Authorization: Bearer genesis-local"

# 5. Run the benchmark suite
python3 tools/genesis_bench_suite.py --quick --host 127.0.0.1
```

## Quick start (bare metal)

```bash
# Prereq: vLLM installed natively
pip install vllm flashinfer-python

# 1. Clone genesis-vllm-patches
git clone https://github.com/Sandermage/genesis-vllm-patches ~/genesis-vllm-patches

# 2. Set GENESIS_REPO + model path
export GENESIS_REPO=$HOME/genesis-vllm-patches
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B-FP8

# 3. Launch
./scripts/launch/bare_metal_35b_fp8_PROD.sh
# (script symlinks Genesis _genesis into the installed vllm package
#  on first run, then exec vllm serve ...)

# 4. Same health check + bench as Docker workflow
```

## VM / Proxmox deployment

The Docker scripts work identically inside a Proxmox VM as long as:

- The VM has a GPU passed through (`pcie=1`, `multifunction=on`,
  `x-vga=on` for primary GPU)
- The host kernel has VFIO + IOMMU enabled (`intel_iommu=on` /
  `amd_iommu=on` in GRUB)
- The VM runs Ubuntu 22.04+ or any distro with NVIDIA driver ≥ 580.126.09
- Inside the VM you install Docker + NVIDIA Container Toolkit, then run
  `start_*.sh` exactly as on bare metal

For fine-grained NUMA / IRQ pinning details on Proxmox, see
[`../docs/BENCHMARK_GUIDE.md`](../../docs/BENCHMARK_GUIDE.md#scenario-3-proxmox-vm--ubuntu-vm).

## WSL2

Bare-metal scripts work on WSL2 if `nvidia-smi` is functional inside the
Linux side. Docker-in-WSL2 also works but adds one virtualization layer
(slightly higher TTFT). Native is recommended for benchmark accuracy.

---

## Utility scripts

| Script | Purpose |
|---|---|
| [`preflight_check.sh`](preflight_check.sh) | Validate host before first launch (driver, CUDA, container toolkit, GPU memory headroom). |
| [`snapshot_pre_arm.sh`](snapshot_pre_arm.sh) | Capture full server state (running container env, GPU usage, repo HEAD, git status) into `docs/_internal/snapshots/<timestamp>_<arm_name>/`. Use before any swap to enable rollback / forensics. |
| [`nsight_profile_capture.sh`](nsight_profile_capture.sh) | Drive Nsight Systems profile capture against a running container (requires `nsys` on host; install via `apt install nsight-systems-2025.6.3` after adding NVIDIA CUDA repo). |

---

## Customization

The Docker scripts hardcode `MODELS_DIR=/nfs/genesis/models` and similar paths
because they were extracted from Sander's homelab. **Override via env or edit
the file** — there's no clever templating.

Common overrides:

```bash
# Most-common edits at the top of any start_*.sh:
- v /nfs/genesis/models:/models:ro                  # → your model dir
- e CONTAINER_NAME=vllm-server-mtp-test             # → name you prefer
- e PORT=8000                                       # → port if 8000 occupied

# Inside the launch CMD:
--gpu-memory-utilization 0.90    # raise/lower based on VRAM headroom
--max-model-len 320000           # match your VRAM budget (lower = less KV pool)
--max-num-seqs 2                 # raise for batch workloads
--tensor-parallel-size 2         # match your GPU count
```

For per-GPU recommendations (which patches to enable) see the
[per-GPU table in the main README](../../README.md#per-gpu-recommendations) and
the auto-detection at boot via `vllm/_genesis/gpu_profile.py`.

---

## Sub-folders

- `_archive/research/` — Phase 1 A/B test arms (v786 series), Phase 2/3
  research scripts (v788, v789, v791, v791c). Kept for forensic reference;
  do NOT use for fresh deployments — config drift, often missing
  required env vars.
- `_archive/historical/` — old PROD baselines (v759, v775) and bisect arms
  (v755-v757), plus generic templates from earlier iterations
  (`start_mtp.sh`, `start_ngram.sh`, `start_suffix.sh`, etc).

If you need to reproduce a specific finding from `feedback_*.md` notes in
the memory index, the relevant launch script lives in `_archive/`.

---

## Legend

- `PROD` in a filename = the variant Sander runs in his daily-driver
  homelab; tool-call validated, stress-tested ≥ 200 min, well-known number.
- `_archive/` = historical or experimental; not maintained.

For benchmarks see [`../../docs/BENCHMARK_GUIDE.md`](../../docs/BENCHMARK_GUIDE.md)
and the unified suite at [`../../tools/genesis_bench_suite.py`](../../tools/genesis_bench_suite.py).
