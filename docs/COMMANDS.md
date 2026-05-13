# Genesis vLLM Patches — command reference

> Single-page command cheatsheet for Genesis. Every command shown here ships
> in the repo on the listed pin. If you copy-paste this into a runbook, also
> grab `docs/CONFIGURATION.md` for the env-flag matrix.

The README intentionally keeps only the highest-leverage three or four commands
in the body — this file is the long-form catalogue. Sections are roughly
ordered by "first day on a new rig" → "weekly maintenance" → "deep
diagnostic".

---

## 1. Install · update · uninstall

| Command | What it does | When to run |
|:---|:---|:---|
| `curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh \| bash` | Detect OS / Python / GPU / vLLM, clone Genesis to `~/.genesis`, install plugin, write a tailored launch script, run smoke test | First-time install |
| `curl -sSL .../install.sh \| bash -s -- --workload tool_agent -y` | Same, fully non-interactive, picks tool-agent preset | CI / scripted bring-up |
| `curl -sSL .../install.sh \| bash -s -- --pin v7.72` | Pin Genesis to a specific tag instead of `stable` | Reproducible deploys |
| `curl -sSL .../install.sh \| bash -s -- --pin dev` | Use dev branch tip (mutable, latest fixes) | Tracking pre-release |
| `bash ~/.genesis/install.sh -h` | Print all installer flags + env overrides | Anytime |
| `curl -sSL .../install.sh \| bash -s -- --uninstall` | Remove plugin + symlink, leave source tree | Clean rollback |
| `cd ~/.genesis && git pull && git checkout <ref>` | Pin a different ref by hand | Custom workflows |

**Genesis bin shim** (after install): `genesis <subcommand>` is a thin
wrapper for `python3 -m vllm.sndr_core.compat.cli`. Both forms work.

---

## 2. First-day diagnostic

```bash
# Full system diagnostic — vendor / chip / vLLM pin / torch / triton / driver
# / patch matrix / known-issues. Always start here on a new box.
sndr doctor

# Same, but JSON output for grafana / pipelines
sndr doctor --json

# Auto-pick a preset matching this rig (gpu × n_gpus × workload) and write
# the launch script to ~/.genesis/launch/start_<gpu>_<n>_<workload>.sh
genesis preset auto

# Browse all available presets without applying
genesis preset list
genesis preset show rtx_a5000_2_balanced --script

# Quick smoke test (no model load, ~5 seconds)
sndr verify --quick

# Full smoke test (boots vLLM, sends 10 prompts, checks tool-call) — ~3 minutes
sndr verify
```

---

## 3. Booting vLLM with Genesis

The repo ships **6 reference launch scripts** in [`scripts/`](../scripts/). Each
runs `vllm serve` in Docker with `GENESIS_*` env flags pre-populated for that
config. See the table in the root README for which to pick.

```bash
# Daily driver — Qwen3.6-35B-A3B-FP8 + MTP K=3 + TQ k8v4, highest TPS
bash scripts/start_35b_fp8_PROD.sh

# 27B + TQ k8v4 hybrid GDN — good for long-ctx without OOM
bash scripts/start_27b_int4_TQ_k8v4.sh

# 27B + fp8_e5m2 + 256K context — RAG-style long-ctx
bash scripts/start_27b_int4_fp8_e5m2_long_256K.sh

# 27B + fp8_e5m2 short-context — high-TPS no-spec-decode
bash scripts/start_27b_int4_fp8_e5m2_short.sh

# Research drafter — DFlash K=5 on 35B
bash scripts/start_35b_fp8_DFLASH.sh

# Research drafter — DFlash K=5 on hybrid 27B
bash scripts/start_27b_int4_DFLASH.sh
```

Each script ends with a `docker logs -f` tail. Cold compile cache takes
~3-5 min on first boot, then ~1-2 min on warm restarts.

---

## 4. Per-patch interrogation

```bash
# Long-form explanation of one patch (env flag, applies-to, deps, A/B numbers)
genesis explain PN59
genesis explain P67
genesis explain PN65

# All patches in one category
genesis categories --category spec_decode
genesis categories --category gdn
genesis categories --category memory

# Lifecycle audit — patches near retirement, broken anchors, stale markers
genesis lifecycle-audit

# Schema validator — PATCH_REGISTRY entries follow the right schema
python3 -m vllm.sndr_core.compat.cli validate-schema

# Per-patch deep test
python3 -m pytest vllm/sndr_core/tests/test_pn59_streaming_gdn.py -v
python3 -m pytest vllm/sndr_core/tests/test_p67_kernel.py -v
```

---

## 5. Models · downloads · launch scripts

```bash
# List curated models (with HF id + recommended preset)
genesis list-models

# Download model (SHA-verified, resumable, idempotent)
./scripts/fetch_models.sh Lorbus/Qwen3.6-27B-int4-AutoRound ~/models

# Genesis-bundled HF puller — same idea via the CLI
genesis pull qwen3.6-27b-int4 --to ~/models

# Show launch script for a curated model (writes to stdout, doesn't run)
genesis recipe qwen3.6-27b-int4 --workload long_context
```

---

## 6. Benchmarks

Every bench lands a markdown table you can paste into GitHub issues / PRs.

```bash
# Comprehensive bench — README-ready output, 6 stages
GENESIS_MODEL=qwen3.6-35b-a3b python3 tests/bench/comprehensive_bench.py
GENESIS_MODEL=qwen3.6-27b      python3 tests/bench/comprehensive_bench.py

# 7-stage smoke test — server live + tool-call + SSE + thinking + needle
ENDPOINT=http://127.0.0.1:8000 \
MODEL=qwen3.6-27b \
  ./scripts/verify-full.sh

# Auto-binary-search for max stable --max-model-len (no OOM, 5-min sustained)
./scripts/probe_max_ctx.sh --start 16384 --max 320000

# A/B bench between two GENESIS_ENABLE_* configs
GENESIS_AB_BASELINE='GENESIS_ENABLE_PN59=0' \
GENESIS_AB_TREATMENT='GENESIS_ENABLE_PN59=1' \
  python3 tests/bench/ab_bench.py --runs 5

# Quick latency probe — 10 prompts, P50 / P95 / P99
python3 tests/bench/latency_probe.py --endpoint http://127.0.0.1:8000

# Needle-in-haystack at 4 depths (1K / 10K / 51K / 92K)
python3 tests/bench/needle_ladder.py --max-ctx 92000

# MoE Triton config naming + staging helper (NOT autotuner — see file header)
GPU_OVERRIDE=NVIDIA_GeForce_RTX_3090 ./scripts/moe_lookup_helper.sh
```

---

## 7. Diagnostics during a live run

```bash
# Tail the structured boot summary (replaces scattered uvicorn lines)
docker logs vllm-server 2>&1 | grep -A 200 'structured boot summary'

# Enable the structured API access log (PN65, opt-in)
#  Sample line:  [Genesis-API] 200  POST /v1/chat/completions
#                34ms  prompt=46t  completion=400t  tools=1  client=127.0.0.1
docker run -e GENESIS_ENABLE_PN65=1 ...

# Dump per-patch decision matrix at boot (which patches APPLY / SKIP / FAIL)
sndr doctor --patches

# Preflight quant-arg validator (catches club-3090#51 NVFP4 boot OOM)
genesis preflight --quantization auto_round \
  --model /models/Qwen3.6-27B-int4-AutoRound

# View VRAM steady-state (5-second sample) — handy for cliff debugging
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 5

# Dump all GENESIS_* env vars currently set in the container
docker exec vllm-server env | grep '^GENESIS_'
```

---

## 8. Maintenance

```bash
# Pre-commit hook install (lint, schema check, no-leaks scan)
bash scripts/git/install.sh

# Rebuild compile cache — necessary after vLLM pin bump
rm -rf ~/.cache/vllm/* && sndr verify

# Test suite (1858 tests, ~12 sec on macOS, ~25 sec on Linux)
python3 -m pytest vllm/sndr_core/tests/ --no-header -q

# Test one file with verbose output
python3 -m pytest vllm/sndr_core/tests/test_pn65_access_log.py -v

# Lint without fixing — CI-style check
python3 -m pyflakes vllm/sndr_core/

# Patch-registry → docs/PATCHES.md sync gate
python3 -m pytest vllm/sndr_core/tests/test_patches_md_sync.py -v

# Apply text-patches in-place on the current vLLM install (bare-metal flow)
python3 -m vllm.sndr_core.apply
```

---

## 9. Telemetry · plugins · advanced

```bash
# Telemetry summary (boot count, patches applied per session, no PII)
genesis telemetry

# List loaded community plugins (third-party Genesis patches)
genesis plugins list

# Update the rolling channel (dev / stable / a specific tag)
genesis update_channel --set dev

# Migrate from a previous Genesis pin (rewrites launch scripts to current schema)
sndr migrate --from v7.65 --to v7.72
```

---

## 10. Cleanup · uninstall

```bash
# Remove plugin + symlink, leave source tree (safe)
bash ~/.genesis/install.sh --uninstall

# Full wipe (after --uninstall above)
rm -rf ~/.genesis

# Revert text-patches inside the vllm install (re-install clean)
pip uninstall vllm && pip install vllm

# Drop compile cache
rm -rf ~/.cache/vllm/* ~/.cache/torch/*
```

---

## Cross-reference

- **All env flags** — [docs/CONFIGURATION.md](CONFIGURATION.md)
- **Per-patch detail** — [docs/PATCHES.md](PATCHES.md)
- **Hardware envelope** — [docs/HARDWARE.md](HARDWARE.md)
- **Cliffs (OOM patterns)** — [docs/CLIFFS.md](CLIFFS.md)
- **Bench reproduction** — [docs/BENCHMARKS.md](BENCHMARKS.md)
- **Models registry** — [docs/MODELS.md](MODELS.md)
- **Authoring a community patch** — [docs/PLUGINS.md](PLUGINS.md)
- **Per-release notes** — [CHANGELOG.md](../CHANGELOG.md)
- **Engineering log (per-commit)** — [CHANGELOG.md](../CHANGELOG.md) (root is single source — audit 2026-05-11)

---

*Last refreshed: 2026-05-05 (v7.72 sprint). Run `sndr doctor` if a command listed here errors out — Genesis is rapidly evolving and the pin may have drifted.*
