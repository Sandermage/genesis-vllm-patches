# Quickstart — from clone to running vLLM in 5 minutes

Get a Genesis-patched vLLM server up on a single machine, end-to-end.
Covers the three entry points (`install.sh`, the `sndr` CLI, direct
docker compose) and the first-30-minute acceptance walkthrough.

> Stack as of 2026-05-22:
> Genesis `v12.0.0` (226 PATCH_REGISTRY entries) ·
> vLLM `0.20.2rc1.dev371+gbf610c2f5` ·
> Reference rig: 2× RTX A5000 24 GB · driver ≥ 580.126 · CUDA 13.

## 1. Prerequisites

| Requirement | How to verify |
| --- | --- |
| NVIDIA GPU with ≥ 24 GiB (any 2× config preferred) | `nvidia-smi` lists your GPU |
| Driver ≥ 580.126 (CUDA 13 capable) | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| Docker + nvidia-container-toolkit *(docker path)* | `docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu24.04 nvidia-smi` |
| Python ≥ 3.10 *(bare-metal path)* | `python3 --version` |
| ≥ 80 GiB free disk | `df -h ~/` |

For sizing guidance against specific GPUs (3090, 4090, 5090, A6000,
H100, mixed rigs) see [`HARDWARE.md`](HARDWARE.md). For the full
model lineup and chosen-default rationale see [`MODELS.md`](MODELS.md).

## 2. Install

Pick the path that matches your environment.

### 2a. One-command bootstrap (recommended)

```bash
curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash
```

The installer:

1. Detects OS / Python / GPU / vLLM presence / disk budget.
2. Resolves the Genesis pin (default: latest stable tag; pass
   `-s -- --pin dev` to track the dev branch).
3. Clones the repo into `~/.sndr/` (or `$SNDR_HOME`; legacy
   `$GENESIS_HOME` honoured for back-compat).
4. `pip install -e tools/genesis_vllm_plugin` so vLLM auto-loads
   Genesis via the `vllm.general_plugins` entry point.
5. Picks a preset for your (GPU × workload) combo and writes the
   matching launch script.
6. Runs a 60-second smoke test via `sndr verify --quick`.

Non-interactive flavour for CI:

```bash
curl -sSL .../install.sh | bash -s -- --workload tool_agent -y
```

Full `install.sh` flag matrix and troubleshooting:
[`INSTALL.md`](INSTALL.md).

### 2b. Clone manually

```bash
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
pip install -e tools/genesis_vllm_plugin
```

This gives you the `sndr` CLI without the auto-detected preset; you
pick the preset yourself in the next step.

## 3. Pick a preset

```bash
sndr model-config list           # browse builtin + V2 presets
sndr config diff prod-35b        # see the resolved YAML for one preset
sndr patches plan prod-35b       # preview which patches will apply
```

The 12 builtin configs are auto-inventoried in
[`CONFIGS_AUTO.md`](CONFIGS_AUTO.md). Pick by hardware shape:

| Hardware | Preset (V1 key / V2 alias) | Notes |
| --- | --- | --- |
| 2× RTX A5000 24 GB | `a5000-2x-35b-prod` / `prod-35b` | Flagship — Qwen3.6-35B-A3B-FP8, ~216 TPS sustained. |
| 2× RTX A5000 multi-conc | `a5000-2x-35b-prod-multiconc` / `prod-35b-multiconc` | `max_num_seqs=8`, aggregate ~675 TPS. |
| 2× 24 GB (3090 / 4090 / A5000) | `a5000-2x-27b-int4-tq-k8v4` / `prod-27b-tq` | Lorbus 27B int4 + TurboQuant k8v4 (long context). |
| 2× 24 GB long-context | `a5000-2x-27b-int4-long-ctx` / `long-ctx-27b` | Same model, `--max-model-len 320000`. |
| 1× RTX A5000 / 3090 | `a5000-1x-27b-int4-tested` | TP=1, 32K context. |
| Single 3090 (community) | `single-3090-dense-cpu-offload-example` | CPU-offload preview. |

For other rig shapes see [`HARDWARE.md`](HARDWARE.md). To add your
own, see [`MODELS.md` § Adding a model recipe](MODELS.md).

## 4. Launch

```bash
sndr launch prod-35b               # V2 alias
# or
sndr launch a5000-2x-35b-prod      # V1 monolithic key
```

The launcher:

1. Runs preflight (mounts, GPU, pin, quant arg coherence).
2. Renders the per-preset launch script.
3. Boots vLLM with Genesis env exports.
4. Streams the structured Genesis boot summary on stdout.

Dry-run first if you want to inspect the rendered command:

```bash
sndr launch prod-35b --dry-run
```

First boot takes 2–5 minutes (Triton kernel JIT, CUDA graph capture).
Warm restarts are ~30–90 seconds.

### Capture an existing running container

If you have a hand-tuned container running and want to lift its
config into Genesis:

```bash
sndr model-config new my-rig --from-running vllm-test-container
```

The docker-inspect-based captor (audit C2 closure, 2026-05-16) reads
the Entrypoint + Cmd + Env + Mounts and reverse-engineers a
`ModelConfig` YAML you can edit, validate, and launch.

### Other deployment runtimes

`sndr model-config render <key> --runtime <name>` emits the
artefact for the runtime you need:

| Runtime | What it emits |
| --- | --- |
| `docker` *(default)* | `docker run …` shell script. |
| `bare_metal` | `python3 -m venv` + `vllm serve …` shell script. |
| `podman` | docker render with podman binary / GPU-flag substitution. |
| `kubernetes` | Single-stream Deployment + Service + ConfigMap manifest. |
| `lxc_proxmox` | Runnable Proxmox VE bootstrap script (audit C4 closure). |

The k8s + Proxmox lifecycles also have full `sndr service install /
start / stop / status / logs / uninstall` symmetry as of audit C3
(2026-05-16).

## 5. Day 1 — first 30 minutes of acceptance

Six checks. Each has a clear pass signal — if anything looks off, the
linked doc has the fix.

### 5.1 Verify hardware + software — `sndr doctor`

```bash
sndr doctor
```

Checks: GPU type / count / VRAM, driver / CUDA version, Python / torch /
vllm versions, NCCL availability, plugin registration, applied patch
manifest.

**Pass signal:** all sections green; "0 issues found" at the end.

**Common failures:**

- `vllm version mismatch (got X, expected 0.20.2rc1.dev371+gbf610c2f5)`
  → re-run installer with `--pin <pin>` to align, or
  `pip install vllm==<your-pin>` and accept the drift warning.
- `NCCL P2P_DISABLE recommended on consumer Ampere` → set
  `NCCL_P2P_DISABLE=1` in your launch env (already in builtin
  presets, but a hand-rolled script may miss it).

Doc: [`CLI_REFERENCE.md` → `sndr doctor`](CLI_REFERENCE.md).

### 5.2 Run smoke test — `sndr verify --quick`

```bash
sndr verify --quick
```

Loads a tiny model, fires 10 inferences, exits. Catches "obvious
broken" states — bad CUDA driver, broken vllm install, missing
weights, plugin failed to register.

**Pass signal:** `10/10 inferences successful` → `verify PASSED`.

**Common failures:**

- Plugin not registered → `pip install -e tools/genesis_vllm_plugin`.
- Model not found → ensure `~/.sndr/models/` points where weights
  live, or pull from HuggingFace by passing the repo id directly.
- OOM at 8 GiB load → check no other process is holding the GPU
  (`nvidia-smi` should show < 1 GiB used).

### 5.3 Browse available presets — `sndr model-config list`

```bash
sndr model-config list
```

Already covered in step 3 above — re-run after a `sndr install` to
see which preset the auto-detect picked for your rig. Cross-check
against the inventory in [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md).

### 5.4 Preflight the chosen preset

```bash
sndr launch prod-35b --preflight-only
```

Validates env vars, no conflicting Genesis patches enabled,
quantization args coherent, disk + VRAM budget vs preset
requirements.

**Pass signal:** `preflight PASSED`. Failures name the exact
mismatch (e.g. `quant mismatch: model declares auto_round but env
says compressed-tensors`).

Time: ~3 seconds, no GPU load.

### 5.5 Boot — `sndr launch <preset>`

```bash
sndr launch prod-35b
```

**Pass signal:** `docker logs <container> | grep "Application
startup complete"` → API ready at `http://localhost:8000/v1/models`.

Smoke chat:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer genesis-local" \
    -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role":"user","content":"Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0
    }'
```

**Common failures** are catalogued in [`FAQ.md`](FAQ.md): boot loop,
patches reapplied to a writable layer ("R/W layer trap"), OOM at
long context. The cliffs catalogue lives in the same file.

### 5.6 Verify against reference metrics — `sndr model-config verify`

```bash
sndr model-config verify prod-35b
```

Runs a 5-dimension benchmark (short_gen, long_gen, tool_call,
stability, concurrent) via `tools/genesis_bench_suite.py`, compares
to the preset's `reference_metrics` and reports whether your rig
matches the validated baseline (within typical CV noise).

**Pass signal:** Δ < 5% on TPS, tool quality matches, stability CV
within reference + 1σ. Results land in `~/.sndr/bench-results/`.

Time: ~5–10 minutes. Methodology + canonical Wave 10 numbers in
[`BENCHMARKS.md`](BENCHMARKS.md).

## 6. Stopping cleanly

Always use `sndr service stop <preset>` or, when you launched via
docker compose by hand, `docker compose down`. Plain `docker stop` +
`docker start` recycles the same writable layer; Genesis text-patches
applied to that layer fail to re-apply (anchors don't match) and the
next boot reports `[FAIL]` for every patch in the manifest.

Recovery from a stuck "R/W layer trap":

```bash
docker compose -f <your-compose>.yml down
docker compose -f <your-compose>.yml up -d
```

## 7. What's next

| Topic | Where |
| --- | --- |
| Tune Genesis env flags (P67 splits, P82 threshold, ...) | [`CONFIGURATION.md`](CONFIGURATION.md) |
| Browse the patch system + dispatcher | [`PATCHES.md`](PATCHES.md) |
| Fix common OOM patterns + named cliffs | [`FAQ.md`](FAQ.md) |
| Add a custom model preset | [`MODELS.md`](MODELS.md) |
| Author a new patch | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Compare your rig to validated baselines | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Look up a specific `sndr` command | [`CLI_REFERENCE.md`](CLI_REFERENCE.md) |
| Recover from a regression | [`FAQ.md` § Rollback playbook](FAQ.md) |

## If something broke

1. `sndr doctor` — most issues are environment drift; re-run.
2. `docker logs <container>` — last 200 lines for the actual error.
3. [`FAQ.md`](FAQ.md) — common issues, cliffs, rollback playbook.
4. Open an issue with `sndr doctor --json` output attached.
