# Day 1 Checklist — first 30 minutes after install

After running `curl -sSL .../install.sh | bash`, walk through these 6 steps to
make sure your Genesis vLLM stack is healthy and you know what to do next.
Each step has a clear pass/fail signal — if anything looks off, the
referenced doc has the fix.

## 0. Prerequisites checklist

| Required | How to verify |
|---|---|
| NVIDIA GPU (A5000/3090/4090/H100/etc.) | `nvidia-smi` shows your GPU |
| Driver ≥ 580.x (CUDA 13.0 capable) | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| Docker + nvidia-container-toolkit (Docker path) | `docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu24.04 nvidia-smi` |
| Python ≥ 3.10 (bare-metal path) | `python3 --version` |
| ≥ 50 GiB free disk for model weights | `df -h ~/` |

If any check fails, see [INSTALL.md → Prerequisites](INSTALL.md#prerequisites)
before running `install.sh`.

## 1. ✅ Verify hardware/software stack — `sndr doctor`

```bash
sndr doctor
```

What it checks: GPU type/count/VRAM, driver/CUDA version, Python/torch/vllm
versions, NCCL availability, plugin registration, applied patch manifest.

**Pass signal:** all 6 sections green; "0 issues found" at the end.

**Common fail patterns:**
- `vllm version mismatch (got X, expected 0.20.2rc1.dev9+g01d4d1ad3)` —
  re-run installer with `--pin <pin>` to align, OR pin via
  `pip install vllm==<your-current-pin>` and accept the drift warning
- `NCCL P2P_DISABLE recommended on consumer Ampere` — set
  `NCCL_P2P_DISABLE=1` in your launch env (already in builtin configs)

Time: ~5 sec. Doc: [docs/COMMANDS.md#genesis-doctor](COMMANDS.md#genesis-doctor)

## 2. ✅ Run smoke test — `sndr verify --quick`

```bash
sndr verify --quick
```

Loads a tiny model, fires 10 inferences, exits. Catches "obvious broken"
states — bad CUDA driver, broken vllm install, missing model weights path,
plugin failed to register.

**Pass signal:** "10/10 inferences successful" + final "verify PASSED".

**Common fail patterns:**
- Plugin not registered → `pip install -e ~/.genesis/tools/genesis_vllm_plugin`
- Model not found → ensure `~/.genesis/models/` symlinks to where weights live
- Out-of-memory at 8 GiB load → check no other process holding GPU
  (`nvidia-smi` should show < 1 GiB used)

Time: ~60 sec. Doc: [docs/COMMANDS.md#genesis-verify](COMMANDS.md#genesis-verify)

## 3. ✅ Browse available model configs — `sndr model-config list`

```bash
sndr model-config list
```

Or equivalently: `sndr config list`.

Shows the 8 builtin configs (a5000-2x-35b-prod, a5000-2x-27b-int4-tq-k8v4,
etc.) with their reference TPS, tool quality, and tier (stable/tested).

**Pick the right config for your hardware:**
- 2× A5000 (24 GiB each) → `a5000-2x-35b-prod` (flagship, 196 TPS)
- 2× 3090 / 4090 → see [HARDWARE.md → cross-rig configs](HARDWARE.md#cross-rig-configs)
- 1× A5000 / 3090 → `a5000-1x-27b-int4-tested`
- Long-context (128K+) → `a5000-2x-27b-int4-long-ctx`

Doc: [docs/MODELS.md](MODELS.md) for choosing the right model;
[docs/MODEL_CONFIG_LAUNCHER.md](MODEL_CONFIG_LAUNCHER.md) for the config
schema.

## 4. ✅ Preflight your chosen config — `--preflight`

```bash
sndr launch <key> --preflight-only
```

Validates: env vars set correctly, no conflicting Genesis patches enabled,
quantization args coherent (NVFP4/AutoRound/compressed-tensors detection),
disk + VRAM budget vs config requirements.

**Pass signal:** "preflight PASSED". Failures are clearly named (e.g.
`quant mismatch: model declares auto_round but env says compressed-tensors`).

Time: ~3 sec, no GPU load. Doc: [docs/COMMANDS.md#genesis-preflight](COMMANDS.md#genesis-preflight)

## 5. ✅ Boot — `sndr launch <key>`

```bash
sndr launch a5000-2x-35b-prod
```

Builds container, mounts model weights, starts vLLM server. First boot
takes 2-5 min (Triton kernel compile, CUDA graph capture, etc.). Subsequent
boots ~30-90 sec (warm cache).

**Pass signal:** `docker logs <container> | grep "Application startup complete"`.
API ready at `http://localhost:8000/v1/models`.

**Common fail patterns:** see [docs/OOM_RECIPES.md](OOM_RECIPES.md) and
[docs/CLIFFS.md](CLIFFS.md) for OOM mitigation. If your config errors at
load with `cudaErrorIllegalAddress` on GDN — that's the well-known Cliff 2b;
PN59 streaming-GDN driver is the fix (env opt-in).

## 6. ✅ Verify against reference metrics — `sndr model-config verify <key>`

```bash
sndr model-config verify a5000-2x-35b-prod
```

Runs a 5-dimension benchmark (short_gen, long_gen, tool_call, stability,
concurrent), compares to the config's `reference_metrics`, and tells you
whether your rig matches the validated baseline (within typical CV noise).

**Pass signal:** delta < 5% on TPS, tool quality matches, stability CV
within reference + 1σ. Saves results to `~/.genesis/bench/`.

Time: ~5-10 min depending on bench depth. Doc:
[docs/BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

---

## Done — what's next?

| If you want to... | Read |
|---|---|
| Tune Genesis env flags (P67 splits, P82 threshold, etc.) | [docs/CONFIGURATION.md](CONFIGURATION.md) |
| Understand the patch system + dispatcher | [docs/PATCHES.md](PATCHES.md) |
| Fix common OOM patterns (Cliff 1, 2a, 2b) | [docs/OOM_RECIPES.md](OOM_RECIPES.md) + [CLIFFS.md](CLIFFS.md) |
| Add a custom model config | [docs/MODEL_CONFIG_LAUNCHER.md](MODEL_CONFIG_LAUNCHER.md) → community-test lifecycle |
| Author a new patch | [docs/CONTRIBUTING.md](CONTRIBUTING.md) + [PLUGINS.md](PLUGINS.md) |
| Compare your rig to community validators | [README.md → Cross-rig validators](../README.md#cross-rig-validators) |

## If something broke

1. `sndr doctor` (re-run — most issues are environment drift)
2. Check `docker logs <container>` last 200 lines
3. Search [docs/FAQ.md](FAQ.md) — common issues addressed
4. [docs/CLIFFS.md](CLIFFS.md) — known failure modes with named fixes
5. Open an issue with `sndr doctor --json` output attached
