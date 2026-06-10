# Genesis vLLM — Launch Scripts

> **Wave 10 update (2026-05-15)** — the legacy per-config `start_*.sh` /
> `bare_metal_*.sh` shell scripts have been superseded by the unified
> V2 preset launcher (`sndr launch <preset>`). The shell scripts are
> archived under [`_archive/superseded_by_model_configs/`](_archive/superseded_by_model_configs/) <!-- audit-links: allow -->  (historical path; subdir removed in post-Wave-10 cleanup).
> The canonical operator UX is now:
>
> ```bash
> sndr launch prod-qwen3.6-35b-balanced                  # 35B latency
> sndr launch prod-qwen3.6-35b-multiconc        # 35B throughput (multi-conc)
> sndr launch prod-qwen3.6-35b-dflash           # 35B DFlash N=3 (single-stream)
> sndr launch prod-qwen3.6-35b-dflash-multiconc # 35B DFlash multi-conc
> sndr launch prod-qwen3.6-27b-tq-k8v4               # 27B + TurboQuant k8v4 (latency)
> sndr launch prod-qwen3.6-27b-tq-multiconc     # 27B + TQ k8v4 multi-conc
> sndr launch prod-qwen3.6-27b-dflash           # 27B DFlash N=5 (single-stream)
> sndr launch prod-qwen3.6-27b-dflash-multiconc # 27B DFlash multi-conc
> ```
>
> Each preset resolves to a (model, hardware, profile) triplet under
> [`sndr/model_configs/builtin/`](../../sndr/model_configs/builtin/).
> See `sndr launch --help` for `--dry-run` and other flags.

Historical context (pre-Wave-10, kept for archeology):

Originally Genesis shipped **4 PROD-ready configs** covering
Qwen3.6-35B-A3B-FP8 (1 variant) and Qwen3.6-27B-int4-Lorbus (3 variants),
each in two flavors:

- `start_*.sh` — Docker (bind-mounted Genesis into `vllm/vllm-openai:nightly`)
- `bare_metal_*.sh` — Native (assumes vLLM via pip + symlink `sndr_core`)

After Wave 10 these scripts moved to
[`_archive/superseded_by_model_configs/`](_archive/superseded_by_model_configs/) <!-- audit-links: allow -->  (historical path; subdir removed in post-Wave-10 cleanup).
The TP=2 + TP=1 hardware bench tables below remain accurate; the
"`start_*.sh`" / "`bare_metal_*.sh`" filename column has been replaced
by the canonical V2 preset alias.

---

## Quick reference

### TP=2 (validated PROD on 2× RTX A5000)

| Model | Config | V2 preset (canonical) |
|---|---|---|
| **35B-A3B-FP8** PROD (TQ k8v4 + MTP K=3, latency) | `--max-model-len 280000 --max-num-seqs 2` | `sndr launch prod-qwen3.6-35b-balanced` |
| **35B-A3B-FP8** multi-conc throughput | `--max-num-seqs 8` (3.21x scaling) | `sndr launch prod-qwen3.6-35b-multiconc` |
| **27B-INT4 Lorbus** TQ k8v4 (hybrid + P98) | TQ packed slot + 262K capable | `sndr launch prod-qwen3.6-27b-tq-k8v4` |
| **27B-INT4 Lorbus** TQ k8v4 multi-conc | 3.89x scaling at conc=8 | `sndr launch prod-qwen3.6-27b-tq-multiconc` |

### TP=1 single-card (⚠️ EXPERIMENTAL — NOT TESTED by maintainer)

These are the TP=1 derivatives of the four PROD configs above. Same Genesis env
flags + `--tensor-parallel-size 1`, otherwise identical. Sander runs 2× A5000
so these have NOT been benched / stress-tested end-to-end. Each script carries
a prominent EXPERIMENTAL warning header and per-card sizing notes.

If you run one and it works, please share results via [GitHub Discussions](https://github.com/Sandermage/genesis-vllm-patches/discussions) — we'll fold confirmed configs back into the main table and drop the experimental tag for that card class.

| Model | Card class fit | Recommendation |
|---|---|---|
| **35B-A3B-FP8** (~35 GB weights) | 48 GB+ cards (A6000, 6000 Ada, L40, RTX PRO 5000/6000 Blackwell, A100, H100, B200) | Compose a TP=1 hardware preset under `vllm/sndr_core/model_configs/builtin/hardware/` and a matching profile under `builtin/profile/`. Use `sndr launch <your-preset>` once registered. |
| **27B-INT4 Lorbus** (~14 GB weights) | 24 GB+ cards (3090, 4090, 5090, A5000, etc.) | Same — single-card hardware preset + 27B profile + alias. |

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
[`../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) (former BENCHMARK_GUIDE.md content consolidated 2026-05-16).

## WSL2

Bare-metal scripts work on WSL2 if `nvidia-smi` is functional inside the
Linux side. Docker-in-WSL2 also works but adds one virtualization layer
(slightly higher TTFT). Native is recommended for benchmark accuracy.

---

## Utility scripts

| Script | Purpose |
|---|---|
| [`preflight_check.sh`](preflight_check.sh) | Validate host before first launch (driver, CUDA, container toolkit, GPU memory headroom). |

> The earlier `snapshot_pre_arm.sh` and `nsight_profile_capture.sh`
> utilities were retired in v11 (Etap 6.3, audit 2026-05-12). The
> snapshot flow is now operator-driven (capture `docker inspect`,
> `nvidia-smi`, `git rev-parse HEAD` manually before a swap). Nsight
> profiling stays in the operator's local toolbox — invoke
> `nsys profile --output … docker exec <container> python …` against a
> running container.

---

## Customization

Archived Docker scripts may contain old homelab paths. Active launches should
prefer `sndr launch <config-key>` or a model config with explicit mounts.

Common overrides:

```bash
# Most-common edits at the top of any archived start_*.sh:
- v ${HOME}/models:/models:ro                       # → your model dir
- e CONTAINER_NAME=vllm-server                      # → name you prefer
- e PORT=8000                                       # → port if 8000 occupied

# Inside the launch CMD:
--gpu-memory-utilization 0.90    # raise/lower based on VRAM headroom
--max-model-len 320000           # match your VRAM budget (lower = less KV pool)
--max-num-seqs 2                 # raise for batch workloads
--tensor-parallel-size 2         # match your GPU count
```

For per-GPU recommendations (which patches to enable) see the
[Supported GPU Classes in docs/HARDWARE.md](../../docs/HARDWARE.md#supported-gpu-classes) and
the auto-detection at boot via `vllm/sndr_core/runtime/gpu_profile.py`.

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

For benchmarks see [`../../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md)
(former BENCHMARK_GUIDE.md content consolidated 2026-05-16)
and the unified suite at [`../../tools/genesis_bench_suite.py`](../../tools/genesis_bench_suite.py).
