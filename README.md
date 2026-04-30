<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches — The Sea-Born Neural Beacon" width="780">
</p>

# Genesis vLLM Patches

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — Qwen3.6-class inference on consumer NVIDIA Ampere with TurboQuant k8v4 KV cache, MTP K=3 spec-decode, tool-calling, and 256K-class context.**

> **The fastest way in:** `python3 -m vllm._genesis.compat.cli init` —
> Genesis detects your hardware, picks a model that fits, and writes a
> tailored launch script. See [Quick start](#quick-start) for the manual path.

> **Status:** v7.63.x (2026-04-30). Production stack runs 24/7 on 2× RTX A5000 with **Qwen3.6-27B-int4-AutoRound** (v794 PROD baseline, **103.3 TPS @ 280K context**, +17% vs prior FP8 KV baseline). Also supports Qwen3.6-35B-A3B-FP8 (v759/v789 lineage). Cross-rig validated on community RTX 3090 / 4090 / 5090 / H20 / R6000 Blackwell / 8× A4000 deployments via [@noonghunna](https://github.com/noonghunna), [@thc1006](https://github.com/thc1006), [@Quentin-M](https://github.com/Quentin-M), [@MidasMining](https://github.com/MidasMining), [@jhsmith409](https://github.com/jhsmith409), [@webcodes-cz](https://github.com/webcodes-cz) and others.

> **🆕 What's new in v7.63.x — Genesis Compat Layer (8 phases shipped):**
>
> ```bash
> # 🆕 Unified CLI dispatcher — one entry point, 13 subcommands
> python3 -m vllm._genesis.compat.cli                  # show all subcommands
> python3 -m vllm._genesis.compat.cli doctor
> python3 -m vllm._genesis.compat.cli explain PN14
> python3 -m vllm._genesis.compat.cli categories --category spec_decode
>
> # Legacy per-module form continues to work (backwards-compat):
> python3 -m vllm._genesis.compat.doctor
>
> # Per-patch detail — applies_to + lifecycle + upstream PR + decision today
> python3 -m vllm._genesis.compat.explain PN14
>
> # First-run wizard (detect hw → pick model → generate launch script)
> python3 -m vllm._genesis.compat.init_wizard
>
> # Browse curated model registry
> python3 -m vllm._genesis.compat.models.list_cli
>
> # One-shot model download + tailored launch script
> python3 -m vllm._genesis.compat.models.pull qwen3_6_27b_int4_autoround
>
> # Lifecycle audit (CI-ready, exit 1 on unknown state)
> python3 -m vllm._genesis.compat.lifecycle_audit_cli --quiet
>
> # PATCH_REGISTRY schema validation (catches typos before commit)
> python3 -m vllm._genesis.compat.schema_validator
>
> # Install pre-commit hook (runs schema + dispatcher + lifecycle on commit)
> bash scripts/git/install.sh
>
> # Browse patches by category (no disk move — logical grouping)
> python3 -m vllm._genesis.compat.categories
> python3 -m vllm._genesis.compat.categories --category spec_decode
>
> # Migration runbook for a planned vllm pin bump
> python3 -m vllm._genesis.compat.migrate /path/to/upstream-vllm-clone \
>     --out runbook.md
>
> # Recipe system — capture/share/replay launch configurations
> python3 -m vllm._genesis.compat.recipes save my-prod --from-container vllm-server
> python3 -m vllm._genesis.compat.recipes list
> python3 -m vllm._genesis.compat.recipes load my-prod --out start.sh
>
> # Override HF id when pulling (e.g. Lorbus vs Intel quant variant)
> python3 -m vllm._genesis.compat.models.pull qwen3_6_27b_int4_autoround \
>     --hf-id-override Lorbus/Qwen3.6-27B-int4-AutoRound
>
> # Community plugin discovery (opt-in via GENESIS_ALLOW_PLUGINS=1)
> # See docs/PLUGINS.md for how to author + ship a community patch.
> GENESIS_ALLOW_PLUGINS=1 python3 -m vllm._genesis.compat.plugins list
>
> # Opt-in anonymized telemetry (default OFF; no PII; local-first)
> python3 -m vllm._genesis.compat.telemetry status
> GENESIS_ENABLE_TELEMETRY=1 python3 -m vllm._genesis.compat.telemetry show
>
> # Update channel — apt-style stable/beta/dev (24h cached; uses GitHub API)
> python3 -m vllm._genesis.compat.update_channel status
> python3 -m vllm._genesis.compat.update_channel check
> python3 -m vllm._genesis.compat.update_channel channel set beta
>
> # Adopt a recipe shared by another operator (HTTPS-only, schema-validated)
> python3 -m vllm._genesis.compat.cli recipe adopt \
>     https://gist.githubusercontent.com/.../v794-prod.json my-prod
>
> # A/B compare two saved recipes (community Q&A workflow)
> python3 -m vllm._genesis.compat.cli recipe diff my-prod community-prod
> python3 -m vllm._genesis.compat.cli recipe diff a b --json    # CI / scripts
>
>
> # Self-test (operator sanity check after `git pull` / pin bump)
> python3 -m vllm._genesis.compat.cli self-test
> python3 -m vllm._genesis.compat.cli self-test --json    # CI-friendly
>
> # Genesis Benchmark Suite (decode TPOT, wall TPS, tool-call quality, stress)
> python3 -m vllm._genesis.compat.cli bench --quick
> python3 -m vllm._genesis.compat.cli bench --mode standard --ctx 8k
> python3 -m vllm._genesis.compat.cli bench --compare a.json b.json
> ```
>
> Plus: richer `applies_to` predicate DSL (AND/OR/NOT) for patch gating, version-range matching (vllm/torch/cuda/triton/driver), patch lifecycle state machine, A3/D2 dependency/conflict validator, reference benchmark fingerprints, B2 shared `result_to_wiring_status` helper, **JSON schema for PATCH_REGISTRY**, **pre-commit hook for contributors**, **Quentin-M P67b cudaErrorIllegalAddress fix cherry-picked**, **canonical `__version__` constant**, **GitHub Actions CI on every push/PR**, **`genesis self-test` structural sanity check**. **467 unit tests, all green. No PROD config changes — backward-compatible, opt-in.** See [Genesis Compat Layer](#genesis-compat-layer) below.

[![CI](https://github.com/Sandermage/genesis-vllm-patches/actions/workflows/test.yml/badge.svg)](https://github.com/Sandermage/genesis-vllm-patches/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![35B-A3B-FP8](https://img.shields.io/badge/35B--A3B--FP8-171.8_TPS-brightgreen.svg)](#reference-configs)
[![27B-INT4](https://img.shields.io/badge/27B--INT4-103.3_TPS-green.svg)](#reference-configs)
[![Long context](https://img.shields.io/badge/27B-256K_verified-blue.svg)](#reference-configs)
[![Patches](https://img.shields.io/badge/PATCH__REGISTRY-50_entries-orange.svg)](PATCHES.md)
[![Tests](https://img.shields.io/badge/tests-1351_pass-brightgreen.svg)](.github/workflows/test.yml)

---

## Headline numbers

Measured on 2× RTX A5000 24 GB (Ampere SM 8.6), driver 580.126.09, vLLM
pin `0.20.1rc1.dev16+g7a1eb8ac2`, captured 2026-04-30 via
`tools/genesis_bench_suite.py --mode standard` (25 runs × 5 prompts ×
1024 max-tokens, decode-only TPOT methodology, 8K context). Both
fingerprints are committed under
[`vllm/_genesis/compat/fingerprints/`](vllm/_genesis/compat/fingerprints/).

| Workload | Model | Wall TPS | Decode TPOT | CV | Tool-call | Source |
|---|---|---:|---:|---:|:---:|---|
| Production short-ctx | Qwen3.6-35B-A3B-FP8 + MTP K=3 + TQ k8v4 | **171.80** | 5.71 ms | 5.89 % | **7 / 7** | [`rtx_a5000_x2_qwen3_6_35b_a3b_fp8_v775.json`](vllm/_genesis/compat/fingerprints/rtx_a5000_x2_qwen3_6_35b_a3b_fp8_v775.json) |
| Production long-ctx | Qwen3.6-27B-int4-AutoRound + MTP K=3 + TQ k8v4 + 280K | **103.34** | 9.36 ms | 4.87 % | **7 / 7** | [`rtx_a5000_x2_qwen3_6_27b_int4_v794.json`](vllm/_genesis/compat/fingerprints/rtx_a5000_x2_qwen3_6_27b_int4_v794.json) |

Stability stress (30 iter × 5 prompts = 150 samples) — both models
**0 HTTP failures**, **0 NaN/inf incidence**. Long-context probe on
27B Lorbus config validates 256K context (262 104-token prompt;
~121 s prefill at 280K max-model-len). VRAM headroom 1.9 GB on 35B
and 2.6 GB on 27B with `--gpu-memory-utilization 0.90` (room to
raise to 0.93 for context spread).

---

## Documentation map

| If you are... | Start with |
|---|---|
| New to Genesis | [QUICKSTART.md](QUICKSTART.md) → [docs/GLOSSARY.md](docs/GLOSSARY.md) → [docs/FAQ.md](docs/FAQ.md) |
| Sizing your hardware | [docs/HARDWARE.md](docs/HARDWARE.md) — VRAM budget, GPU classes, NVLink notes |
| Adding your own model | [docs/CONFIGS.md](docs/CONFIGS.md) — step-by-step recipe with worked example |
| Hitting a weird bug | [docs/CLIFFS.md](docs/CLIFFS.md) — known performance/correctness cliffs |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) — how to add patches, scripts, doc PRs |
| Patch catalog | [PATCHES.md](PATCHES.md) — full list with metadata |
| Install + boot | [INSTALL.md](INSTALL.md) — pinned vLLM commits + docker-compose |
| Detailed config | [CONFIGURATION.md](CONFIGURATION.md) — env-flag reference |
| Bench reproducibility | [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) — 5-environment guide |

Per-launch examples in [scripts/](scripts/) — each `start_<model>_<kv>_<mode>.sh` is a working starting point.

---

## Table of contents

1. [Quick start](#quick-start) — get running in 60 s, Docker, or bare metal
2. [What's new in v7.63.x — Genesis Compat Layer](#whats-new-in-v763x--genesis-compat-layer)
3. [Genesis Compat Layer](#genesis-compat-layer) — doctor / init / models / recipes / plugins / telemetry / channels / self-test / bench
4. [What's new in v7.62.x — 36-hour session timeline](#whats-new-in-v762x--36-hour-session-timeline)
5. [Reference configs](#reference-configs) — PROD launch scripts + fingerprints
6. [Genesis Benchmark Suite](#genesis-benchmark-suite) — `tools/genesis_bench_suite.py` + JSON schema
7. [Patch catalog](#patch-catalog) — all 50 entries
8. [Per-GPU recommendations](#per-gpu-recommendations) — `[REC]` / `[OFF]` predicates
9. [Empirical findings — what worked, what didn't](#empirical-findings--what-worked-what-didnt)
10. [Hardware and operating envelope](#hardware-and-operating-envelope)
11. [How patches work](#how-patches-work) — text-anchor framework
12. [Acknowledgments](#acknowledgments) — community contributors
13. [License / disclaimer](#license--disclaimer)

For deeper dives:

- [`CHANGELOG.md`](CHANGELOG.md) — public release log
- [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) — engineering log
- [`PATCHES.md`](PATCHES.md) — per-patch decision table
- [`MODELS.md`](MODELS.md) — supported model registry
- [`INSTALL.md`](INSTALL.md) — bare-metal install guide
- [`QUICKSTART.md`](QUICKSTART.md) — Docker quickstart (EN + RU)
- [`CONFIGURATION.md`](CONFIGURATION.md) — env vars and tunables
- [`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md) — 5-environment bench guide
- [`docs/SELF_TEST.md`](docs/SELF_TEST.md) — `genesis self-test` reference
- [`docs/PLUGINS.md`](docs/PLUGINS.md) — community plugin authoring
- [`CREDITS.md`](CREDITS.md) — per-patch upstream credits

---

## What's new in v7.64 — empirical patch validation + DFlash + new docs (2026-05-01)

This release captures the empirical validation cycle done after v7.63.x and ships a complete documentation overhaul.

### Big fixes shipped

- **P67 generalized to non-power-of-2 GQA.** The TurboQuant multi-query Triton kernel previously required `HEADS_PER_KV` to be a power of two; on Qwen3.6-27B (GQA=24/4=6) the kernel failed to compile and fell through to the broken upstream path, producing `<tool_call>` cascades under FULL cudagraph. Now `BLOCK_QH = triton.next_power_of_2(HEADS_PER_KV)` with a `lane_valid` mask. Result on 27B: 0/5 tool-call → **7/7 tool-call** with TQ k8v4 + FULL_AND_PIECEWISE. See [docs/CLIFFS.md#cliff-4](docs/CLIFFS.md) Cliff 4.
- **PN17 + PN19 wiring kwargs fix.** Two memory-savings patches were silently failing on apply due to a missing `applied_message`/`patch_name` kwarg in `result_to_wiring_status()`. Both now apply cleanly. PN17 frees 50-100 MiB on long-context FA2; PN19 frees 200-500 MiB during model load.
- **TextPatcher hardening discussion.** Cliff 8 (anchor drift on pin bumps) documented; partial-apply warnings counter planned. See [docs/CLIFFS.md#cliff-8](docs/CLIFFS.md).

### New launch scripts (per workload)

Each script is a self-contained working starting point. Pick by KV cache + workload:

| Script | KV | Spec-decode | Best for |
|---|---|---|---|
| `start_27b_int4_TQ_k8v4.sh` | turboquant_k8v4 | MTP K=3 | High-concurrency / long-ctx (5× KV pool) |
| `start_27b_int4_TQ_k8v4_NGRAM.sh` | turboquant_k8v4 | ngram | Tool-use heavy with strict prompt_lookup_min=8 |
| `start_27b_int4_DFLASH.sh` | auto (fp16) | DFlash N=5 | Coding agents (135 TPS code workload) |
| `start_27b_int4_fp8_e5m2_short.sh` | fp8_e5m2 | MTP K=3 | Short-to-mid context, simpler stack |
| `start_27b_int4_fp8_e5m2_long_256K.sh` | fp8_e5m2 | MTP K=3 | Long context (validated 256K on 2× A5000) |
| `start_35b_fp8_PROD.sh` | turboquant_k8v4 | MTP K=3 | 35B-A3B-FP8 production (186 TPS reference) |
| `start_35b_fp8_DFLASH.sh` | auto (bfloat16) | DFlash N=4 | 35B coding agents |

### Empirical validation matrix (2× RTX A5000, n≥5 per row)

| Config | Tool-call | Wall TPS @ 256t | Wall TPS @ 512t | Notes |
|---|---|---|---|---|
| 27B + TQ k8v4 + 8 patches (Group A+B baked) | 7/7 | 100.7 | 95.3 | +9.2% vs original baseline 87.3 |
| 27B + TQ k8v4 + ngram strict | 7/7 | — | 27 | Empirically slow on prose; tool-use only |
| 27B + DFlash N=5 | 7/7 | 60-86 | 86 (prose) / **135 (code)** | DFlash excels code-heavy workloads |
| 35B-A3B-FP8 + 3 patches (P103+PN17+PN19) | 7/7 | 201 | 183.9 | Within CV of 186 PROD reference (NS) |
| 35B-A3B-FP8 + 5 patches A+B set | 7/7 | 197 | 179 | **Regresses −3.9%** — keep OFF on 35B |

Key finding: the 5 defensive backports (PN9 / PN12 / PN13 / PN14 / P94) help 27B (+9% TPS) but regress 35B FP8 (−4%). They are baked into 27B default, kept OFF on 35B.

### DFlash speculative decoding — first ship

Both Qwen3.6-27B and Qwen3.6-35B-A3B now have working DFlash launch scripts using z-lab drafts:

- `z-lab/Qwen3.6-27B-DFlash` (3.3 GB BF16) — drafter for INT4 main
- `z-lab/Qwen3.6-35B-A3B-DFlash` (905 MB BF16) — drafter for FP8 MoE main

Both are gated on HuggingFace — accept license then `huggingface-cli login`. DFlash trades prose throughput (slower than MTP on prose) for code-completion throughput (135 TPS on 27B vs noonghunna's 3090 quote of 128). Recommended `num_speculative_tokens=4` per z-lab discussion #2.

DFlash on 24 GB cards is currently capped at ~80K context due to draft model adding ~2-3 GB per GPU; cliffs documented at [docs/CLIFFS.md#cliff-7](docs/CLIFFS.md). DFlash + 200K context requires upstream PR #40898 (SWA) backport — tracked for next release.

### Documentation overhaul

Six new public docs ship in this release:

- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** — terms (TPS, KV, MTP, GQA, GDN, FA2, CUDA Graph, etc) for newcomers.
- **[docs/HARDWARE.md](docs/HARDWARE.md)** — VRAM budgets per model, GPU class support, NVLink discussion, PSU/cooling notes.
- **[docs/FAQ.md](docs/FAQ.md)** — 18 common questions with direct answers.
- **[docs/CONFIGS.md](docs/CONFIGS.md)** — adding your own model recipe (with full Llama-3 70B walkthrough showing generic patches work outside Qwen3-family).
- **[docs/CLIFFS.md](docs/CLIFFS.md)** — 8 known performance/correctness cliffs catalogued with mechanism + impact + fix + refs.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — patch authoring guide, code style, PR template, security rules.

### Cross-engine research absorbed

- **SGLang has DFlash + SWA knob** (`speculative-dflash-draft-window-size`) we don't yet have. Backport candidate as future P-N21.
- **SGLang DDTree** (issue #22887) — tree-attention extension claiming +2.13× over vanilla DFlash on Qwen3-30B-MoE; tracking only, not yet merged in SGLang.
- **vLLM 24h activity audit**: PR #40898 (SWA), PR #40849 (FP8 draft inheritance, already in our PN8), PR #39419 (local argmax TP, +9-30% on TP=2), PR #41268 (max_split_size_mb, already in our PN19), PR #41306 (MoE backend regression on v0.20+ for non-FP8 — config-only mitigation `--moe-backend=triton`).
- **noonghunna fork PRs #12 + #13** (anchor drift fixes for P101 and PN12 on dev205+) — empirically not needed on our exact pin, but the underlying class bug (silent sub-patch skip on `required=False`) deserves a TextPatcher hardening pass.

### Privacy / repo hygiene

- `.gitignore` hardened: `_internal/`, `snapshots/`, `*.bak`, `._*` (macOS AppleDouble), `*.json`, `*.log`.
- `tools/` sanitized: hardcoded server IP `192.168.1.10` → `localhost`.

---

## What's new in v7.63.x — Genesis Compat Layer

Released 2026-04-30, building on the v7.62.x sprint. **Phase 1 of the
compat overhaul** — turns Genesis from "a custom patcher running on
Sander's machine" into a discoverable, self-documenting, hardware-aware
product.

> See [`CHANGELOG.md`](CHANGELOG.md) for the public release log,
> [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) for the
> per-commit engineering log (2300+ lines), and
> [`vllm/_genesis/compat/fingerprints/`](vllm/_genesis/compat/fingerprints/)
> for blessed reference numbers per model × hardware × patch-set.

### TL;DR

| Feature | Command | What it does |
|---|---|---|
| Doctor | `python3 -m vllm._genesis.compat.doctor` | Single-command diagnostic — hw + sw + model + patches + lifecycle + validator. Outputs human-readable report or JSON. |
| Init wizard | `python3 -m vllm._genesis.compat.init_wizard` | Interactive: detect hardware → recommend model → workload pick → generate launch script |
| Model browser | `python3 -m vllm._genesis.compat.models.list_cli` | Show curated model registry (5 models so far: 27B/35B INT4 + 35B FP8 + experimental + planned 80B) |
| Model puller | `python3 -m vllm._genesis.compat.models.pull <key>` | HF download + verify + tailored launch script (auto-engages right Genesis patches for hw + quant combo) |
| A3/D2 validator | runs at every boot | Catches `requires_patches`/`conflicts_with` violations + unknown patch IDs (caught 2 real prod-config issues at first run) |

### Engineering changes

- **Richer `applies_to` predicate DSL** in `vllm/_genesis/compat/predicates.py`: AND/OR/NOT trees over hardware + model + version + quant. Solves "INT4 alone doesn't need this, but INT4+TurboQuant does". Backwards-compatible with all 48 existing flat-dict entries.
- **Version-range matching** in `vllm/_genesis/compat/version_check.py`: declare `vllm_version_range`, `torch_version_min`, `triton_version_min`, `cuda_runtime_min`, `nvidia_driver_min`, `compute_capability_min/max` per patch. Validator at boot enforces.
- **Patch lifecycle states** in `vllm/_genesis/compat/lifecycle.py`: `experimental` / `stable` / `deprecated` / `research` / `community` / `retired`. Code removal requires prior `lifecycle: retired`. Doctor surfaces deprecation `superseded_by` actionably.
- **Reference fingerprints** in `vllm/_genesis/compat/fingerprints/`: blessed benchmark numbers per (hardware × model × patch_set). First entry: `rtx_a5000_x2_qwen3_6_27b_int4_v794.json` (103.3 TPS, CV 4.9%).
- **B2 shared `result_to_wiring_status` helper** in `vllm/_genesis/wiring/text_patch.py`: DRY across 5 PN-family wiring modules. Caught silent-bug class where SKIPPED reported as APPLIED in boot logs.
- **D1 CI drift watcher** in `tools/check_upstream_drift.py` + `.github/workflows/upstream_drift_watcher.yml`: daily check that text-patch anchors still match upstream main HEAD; reports newly-merged upstream PRs that allow Genesis self-retirement.

### What stays the same (no PROD impact)

- Existing launch scripts in `scripts/launch/` work unchanged.
- All 48 PATCH_REGISTRY entries keep their env flags, default-OFF semantics, and behavior.
- `vllm._genesis.gpu_profile`, `model_detect`, `config_detect` legacy import paths still work (re-exported from `compat/`).

### Test coverage

| Suite | Tests | Subject |
|---|---:|---|
| `tests/compat/test_predicates.py` | 27 | applies_to DSL evaluator |
| `tests/compat/test_version_check.py` | 20 | vllm/torch/cuda version-range gating |
| `tests/compat/test_lifecycle.py` | 18 | lifecycle state machine |
| `tests/compat/test_models_registry.py` | 13 | curated model registry |
| `tests/compat/test_doctor_smoke.py` | 6 | doctor section smoke tests |
| `tests/compat/test_categories.py` | 21 | categories index + CLI |
| `tests/compat/test_explain.py` | 24 | per-patch explain tool |
| `tests/compat/test_recipes.py` | 36 | recipe save/load/share + diff |
| `tests/compat/test_recipe_adopt.py` | 14 | recipe adopt URL (HTTPS-only) |
| `tests/compat/test_plugins.py` | 18 | community plugin discovery |
| `tests/compat/test_telemetry.py` | 23 | opt-in anonymized telemetry |
| `tests/compat/test_update_channel.py` | 18 | apt-style stable/beta/dev |
| `tests/compat/test_schema_validator.py` | 15 | PATCH_REGISTRY schema |
| `tests/compat/test_self_test.py` | 17 | structural sanity check |
| `tests/compat/test_bench.py` | 11 | unified-CLI bench shim |
| `tests/compat/test_cli.py` | 14 | unified CLI dispatcher |
| `tests/test_dispatcher_validator.py` (A3/D2) | 24 | dependency / conflict graph |
| `tests/test_pn14_tq_decode_oob_clamp.py` | 13 | TQ grouped-decode OOB clamp |
| `tests/test_pn16_lazy_reasoner.py` | 41 | lazy reasoner harness |
| `tests/test_bench_ablation.py` (D3) | 11 | per-patch ablation orchestrator |
| `tests/test_wiring_status_helper.py` (B2) | 10 | shared result_to_wiring_status |
| `tests/test_version.py` | 5 | canonical __version__ constant |
| `tests/test_ci_workflow.py` | 6 | GitHub Actions gate contract |
| **Full session suite** | **467 tests** | — |

---

## Genesis Compat Layer

The `vllm/_genesis/compat/` module is the **single home for everything Genesis needs to know about the environment** and the surface operators interact with for setup + diagnosis + maintenance.

### Module map

```text
vllm/_genesis/compat/
├── doctor.py                # `python3 -m vllm._genesis.compat.doctor`
├── init_wizard.py           # `python3 -m vllm._genesis.compat.init_wizard`
├── version_check.py         # vllm/torch/cuda/triton/driver range matching
├── predicates.py            # AND/OR/NOT applies_to evaluator
├── lifecycle.py             # patch lifecycle state machine
├── gpu_profile.py           # re-export shim (legacy path still works)
├── model_detect.py          # re-export shim
├── config_detect.py         # re-export shim
├── models/
│   ├── registry.py          # SUPPORTED_MODELS dict
│   ├── pull.py              # HF download + verify + launch script gen
│   └── list_cli.py          # `python3 -m vllm._genesis.compat.models.list_cli`
└── fingerprints/            # reference benchmark JSONs per (hw × model × patches)
    ├── rtx_a5000_x2_qwen3_6_27b_int4_v794.json    # 27B Lorbus + TQ k8v4
    └── rtx_a5000_x2_qwen3_6_35b_a3b_fp8_v775.json # 35B-A3B + TQ k8v4 (NEW 2026-04-30)
```

### Doctor output (sample)

```text
========================================================================
Genesis doctor — system diagnostic
========================================================================

[1/6] Hardware
  GPU 0: NVIDIA RTX A5000      sm_86  VRAM 24.0 GB
  GPU 1: NVIDIA RTX A5000      sm_86  VRAM 24.0 GB

[2/6] Software
  vllm:          0.20.1rc1.dev16+g7a1eb8ac2
    commit:      7a1eb8ac2
  torch:         2.5.1+cu124
  triton:        3.1.0
  cuda runtime:  12.4
  nvidia driver: 580.126.09
  python:        3.12.7

[3/6] Model profile
  model_class:   qwen3_5
  is_hybrid:     True
  is_moe:        True
  is_turboquant: True
  quant_format:  autoround_int4

[4/6] Patch registry decisions
  total: 48, APPLY: 27, SKIP: 21
  Applied (27):
    ✓ P58      Async-scheduler -1 placeholder fix
    ✓ P60      GDN+ngram state recovery (Phase 1: SSM pre-copy)
    ...
    ✓ PN14     TQ decode IOOB safe_page_idx clamp (vllm#40074)

[5/6] Lifecycle audit
  stable: 44
  deprecated: 4

[6/6] Validator
  ✓ clean — no validator issues

========================================================================
Recommendations
========================================================================
  [OK] no issues detected. System is healthy.
========================================================================
```

### Models registry

Curated, tested-and-validated models with known-good launch configs and expected performance fingerprints:

| Key | HF id | Size | Quant | Status |
|---|---|---:|---|---|
| `qwen3_6_27b_int4_autoround` | Intel/Qwen3.6-27B-A3B-int4-AutoRound | 14.2 GB | autoround_int4 | **PROD** |
| `qwen3_6_35b_a3b_fp8` | Qwen/Qwen3.6-35B-A3B-FP8 | 38.0 GB | fp8 | SUPPORTED |
| `qwen3_6_35b_a3b_int4_autoround` | Intel/Qwen3.6-35B-A3B-int4-AutoRound | 18.5 GB | autoround_int4 | SUPPORTED |
| `qwen3_6_27b_fp8_lmhead_fp8` | inferRouter/Qwen3.6-27B-FP8-lmhead-fp8 | ~20 GB | fp8 + FP8 lm_head | EXPERIMENTAL |
| `qwen3_next_80b_awq` | Qwen/Qwen3-Next-80B-AWQ | 40.0 GB | awq_int4 | PLANNED Q3'26 |

Each entry declares: HF id, file SHAs, size, quant, model_class, hybrid/MoE flags, min VRAM per TP rank, tested hardware classes, **tested launch configs** (vllm pin + env flags + command-line args + recommended Genesis patches), expected metrics, license, gating, known quirks, lifecycle status.

### Richer applies_to DSL — the "INT4+TurboQuant" example

Old (legacy, still works):
```python
"P67": {
    "applies_to": {"is_turboquant": [True]},
}
```

New compound forms:
```python
"P67": {
    "applies_to": {
        "all_of": [
            {"is_turboquant": True},
            {"any_of": [
                {"quant_format": "fp8"},
                {"quant_format": "autoround_int4"},
                {"quant_format": "int4_w4a16"},
            ]},
            {"compute_capability_min": [8, 6]},
            {"vllm_version_range": [">=0.20.0", "<0.21.0"]},
        ],
    },
}
```

Solves the user's scenario:
- Pure INT4 (no TQ) → `all_of` fails → SKIP ✓
- INT4 + TurboQuant → all conditions met → APPLY ✓
- FP8 + TurboQuant → all conditions met → APPLY ✓
- INT4 on sm_75 (older Tesla) → `compute_capability_min` fails → SKIP ✓
- vllm 0.22 → `vllm_version_range` fails → SKIP ✓

### Lifecycle states

| State | Meaning | Doctor behavior |
|---|---|---|
| `experimental` | new, may break across releases | warns operator |
| `stable` | proven, default for the indicated workload | normal apply path |
| `deprecated` | superseded; declares `superseded_by` + `removal_planned` | warns + actionable suggestion |
| `research` | kept as reference for future hardware/configs | listed but no warning |
| `community` | contributed via plugin entry-point (Phase 5+) | warns (origin disclosure) |
| `retired` | no longer applied; code may exist only for archeology | hard-skip; requires `--allow-retired` |

State transitions are forward-only on the public timeline. **Code removal requires prior `lifecycle: retired`** for at least one release — guards against accidental deletion.

### Phase roadmap

- ✅ **Phase 1** (2026-04-30, this release) — compat module + doctor + models + version_check + predicates + lifecycle + fingerprints
- 🚧 **Phase 2** — refactor `wiring/patch_*.py` into category subdirs (`spec_decode/`, `kv_cache/`, `kernel_perf/`, …) for better navigation
- 📋 **Phase 3** — auto-update channel system (`stable`/`beta`/`dev`) + migration runbook generator
- 📋 **Phase 4** — pre-commit hook + JSON schema + `genesis explain <patch>` + Prometheus metrics endpoint
- 📋 **Phase 5** — plugin entry-points for community patches + opt-in telemetry + `genesis recipe` system

Internal v8.x arena allocator design lives in maintainer-only `docs/_internal/` (gitignored).

---

---

## What's new in v7.62.x — 36-hour session timeline

This release captures a focused 36-hour optimization sprint (2026-04-28 to 2026-04-29) that turned into a much bigger story than expected. We measured, disproved, fixed, and shipped — all empirically. The TL;DR up front, then the chronological narrative below.

### TL;DR — what changed

**Configuration & validation:**
- Real PROD baseline recalibrated: **183.05 TPS** on 35B-A3B-FP8 (CV 8.92%, N=500). The previously-tracked 162 TPS was a *one-off measurement using older methodology* — the same stack measured properly is 13% higher.
- 27B Lorbus INT4 stable at **TRUE 256K context** (262 104 tokens) via `v791b` config (util 0.90, max-num-seqs 2, max-num-batched-tokens 2048). Previously OOMed at 16K with the aggressive `v771b` PROD config — was config-aggressiveness, not a model limit.
- TurboQuant k8v4 unlocked on hybrid GDN models (Qwen3.6-27B Lorbus) via the **P4 + P98** combo. First time TQ runs on hybrid in Genesis. Tool-call 4/4 clean.

**New patches and infrastructure:**
- `PN8` — MTP/draft online-quant propagation (vllm#40849 backport). Frees ~1 GiB VRAM per GPU on FP8 + MTP.
- `PN9` — independent drafter attention backend (vllm#39930 self-retired upstream merged in our pin).
- `PN11` — GDN a/b contiguity defensive fix (vllm#41142, **by [Quentin Machu](https://github.com/Quentin-M)** via fork PR — first community-contributed Genesis patch).
- `P40` signature drift fix — backport of vllm#40792 brought into the current pin (was crashing with `_fwd_kernel_stage2 missing 3 args`).
- `P4` drift markers added for vllm#41123 self-retirement.

**Tooling:**
- `tools/genesis_bench_suite.py` — flagship community-grade benchmark suite. 7 phases, configurable contexts (1K → 512K), Welch's t-test compare. Outputs JSON + Markdown.
- `vllm/_genesis/gpu_profile.py` — per-GPU patch recommendation engine. 18-card datasheet (Ampere consumer + workstation + Ada + Hopper + **all 5 Blackwell PRO workstation cards**). Auto-prints `[REC]/[OFF]` per patch at boot.
- `docs/BENCHMARK_GUIDE.md` — 5-environment run guide (bare metal / Docker / Proxmox / WSL2 / RunPod).

**Disproven (do NOT enable on Ampere consumer):**
- P83 + P84 + P85 (prefix-cache "cake-and-eat" stack) → -29% TPS regression. Even with the supposedly-root-cause P84.
- P40 default-on (only ~+1% on Ampere consumer; needs L2 ≥ 24 MB to deliver — recommended for 4090, 5090, H100, Blackwell).

### Chronological narrative

The 36-hour sprint covered: 5-agent research sweep → 4-arm prefix-cache
A/B (P83+P84+P85 disproven, -29 % regression) → P40 signature-drift fix
(rebased onto current pin, not significant on Ampere consumer) → PN8
PROMOTED (~1 GiB VRAM saved per GPU) → INT8-gs128 boot crash isolated to
upstream `torch.compile + qwen3_next.linear_attn @custom_op` (workaround:
use Lorbus INT4 v771b) → **256K context unlocked on 27B Lorbus** via
the looser v791b config (was config-aggressiveness, not a model limit) →
**TurboQuant k8v4 unlocked on hybrid GDN** via `P4 + P98` (first time
TQ runs on hybrid in Genesis, tool-call clean) → Quentin Machu's PN11
landed (first community-contributed Genesis patch) → GPU profile
recommendation system (16-card datasheet + per-patch predicates) →
flagship benchmark suite shipped.

The full per-event log (timestamps, env settings, A/B numbers, p-values,
follow-up TODOs) lives in
[`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md). Two day-1
findings turned into shipping patches (PN8, PN11), one into a root-cause
memory note (INT8-gs128 boot crash on `torch.compile + qwen3_next.linear_attn
@custom_op`, workaround: use Lorbus INT4).

## Quick start

Three on-ramps, depending on how much you want to do yourself.

### Path A — interactive wizard (easiest, 60 seconds)

The wizard reads your hardware, recommends a model that fits, asks
about workload (tool-call vs throughput vs long-context), and writes
a personalized launch script with the right Genesis patches engaged:

```bash
git clone https://github.com/Sandermage/genesis-vllm-patches ~/genesis-vllm-patches
cd ~/genesis-vllm-patches
python3 -m vllm._genesis.compat.cli init
```

Sample session: 1× RTX 4090 → recommends `qwen3_6_27b_int4_autoround`,
asks workload, prints `start_27b_4090.sh`, optionally pulls weights
via `huggingface-cli`, and shows the next-step command. From zero to
running container: ~20 minutes (mostly model download).

### Path B — Docker Compose (recommended for production)

```bash
# 1. Set env paths
export MODELS_DIR=/path/to/your/models
export GENESIS_REPO=$HOME/genesis-vllm-patches
export HF_CACHE=$HOME/.cache/huggingface

# 2. Pick your config and launch
git clone https://github.com/Sandermage/genesis-vllm-patches "$GENESIS_REPO"
cd "$GENESIS_REPO"
./scripts/launch/start_35b_fp8_PROD.sh

# 3. Wait ~3-5 min for cold compile cache (subsequent boots ~1-2 min)
docker logs -f vllm-server-mtp-test

# 4. Health check
curl http://localhost:8000/v1/models -H "Authorization: Bearer genesis-local"

# 5. Run a quick benchmark
python3 tools/genesis_bench_suite.py --quick --host 127.0.0.1
```

### Path C — bare metal (no Docker)

```bash
# Prereq
pip install vllm flashinfer-python

# Clone + launch
git clone https://github.com/Sandermage/genesis-vllm-patches ~/genesis-vllm-patches
export GENESIS_REPO=$HOME/genesis-vllm-patches
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B-FP8

./scripts/launch/bare_metal_35b_fp8_PROD.sh
# (the script symlinks Genesis _genesis into the installed vllm package
#  on first run, then exec vllm serve ...)
```

For VM (Proxmox), WSL2, or RunPod see [`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md).
For step-by-step bare-metal install see [`INSTALL.md`](INSTALL.md).
For per-config explanations see [`QUICKSTART.md`](QUICKSTART.md).

---

## Reference configs

Four PROD-ready configs ship in [`scripts/launch/`](scripts/launch/) —
each in two flavors (Docker `start_*.sh` and bare-metal
`bare_metal_*.sh`).

**Reference fingerprints** — every PROD config has a JSON fingerprint
under [`vllm/_genesis/compat/fingerprints/`](vllm/_genesis/compat/fingerprints/)
with full hardware × software × config × measured metrics. The bench
suite can compare a fresh run against the fingerprint:

```bash
# Re-run the bench, compare against the v775 35B fingerprint
python3 tools/genesis_bench_suite.py --mode standard --runs 25 \
    --max-tokens 1024 --ctx 8K --probe-output-length \
    --name my-run --out /tmp/my-run.json
python3 tools/genesis_bench_suite.py --compare \
    vllm/_genesis/compat/fingerprints/rtx_a5000_x2_qwen3_6_35b_a3b_fp8_v775.json \
    /tmp/my-run.json
```

A regression alarm fires if `wall_TPS` drops below 85 % of the
fingerprint median or `decode_TPOT_ms` exceeds 120 % — thresholds are
declared inside each fingerprint and enforced by the comparator.

### 1. 35B-A3B-FP8 PROD (the daily driver)

```
Model:               Qwen3.6-35B-A3B-FP8
KV cache:            turboquant_k8v4
Spec-decode:         MTP K=3
Max context:         320 000
gpu-mem-util:        0.90
max-num-seqs:        2
max-num-batched-tk:  4 096
Tensor parallel:     2

Genesis env (PN8 + standard stack):
  GENESIS_ENABLE_P58/60/60b/61/61b/62/64/66/67/68/69/70/72/74=1
  GENESIS_ENABLE_P81=1, P82=1, P82_THRESHOLD_SINGLE=0.3
  GENESIS_ENABLE_P99=1, P101=1, P37=1
  GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1   ⭐ NEW
  GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1       ⭐ defensive

Empirical:           wall_TPS 183.05, CV 8.92%, N=500 stress (200 min)
                     tool-call 3/4 (case 4 is max_tokens artifact, not regression)
                     GPU memory 22.66 GB after PN8 (-1 GiB vs v775)
                     128K context stable; 256K+ needs longer probe timeout
```

Files: [`scripts/launch/start_35b_fp8_PROD.sh`](scripts/launch/start_35b_fp8_PROD.sh) · [`bare_metal_35b_fp8_PROD.sh`](scripts/launch/bare_metal_35b_fp8_PROD.sh)

### 2. 27B-INT4-Lorbus short-ctx (high TPS for chat ≤8K)

```
Model:               Qwen3.6-27B-int4-AutoRound (community standard)
KV cache:            fp8_e5m2 (NOT TQ — Lorbus routes via AllSpark, no benefit)
Spec-decode:         MTP K=3
Max context:         131 072
gpu-mem-util:        0.95          ← aggressive — OK for ≤8K prompts
max-num-seqs:        4
max-num-batched-tk:  8 192
Prefix caching:      OFF           ← REQUIRED (DS conv state crash if ON)

Genesis env:
  GENESIS_ENABLE_P58/60/60b/61/61b/62/64/66/74=1
  GENESIS_ENABLE_P67=1, P83=1, P85=1, P87=1, P91=1
  GENESIS_ENABLE_P99=1, P100=1, P101=1
  GENESIS_ENABLE_PN11=1
  P82=0  (not yet swept on INT4)

Empirical:           wall_TPS 89.23, CV 9.97%, N=500 stress
                     tool-call 4/4 with max_tokens=1500
                     OOM at ≥16K prompts (config-aggressiveness, see #3)
```

Files: [`scripts/launch/start_27b_int4_no_TQ_short.sh`](scripts/launch/start_27b_int4_no_TQ_short.sh) · [`bare_metal_27b_int4_no_TQ_short.sh`](scripts/launch/bare_metal_27b_int4_no_TQ_short.sh)

### 3. 27B-INT4-Lorbus long-ctx 256K (RAG / agentic / long-doc)

```
Model:               Qwen3.6-27B-int4-AutoRound
KV cache:            fp8_e5m2
Spec-decode:         MTP K=3
Max context:         280 000       ← 192K and 256K both verified
gpu-mem-util:        0.90          ← reduced from 0.95 to free compile-time headroom
max-num-seqs:        2             ← halved (KV pool footprint)
max-num-batched-tk:  2 048         ← smaller chunked-prefill chunks

Same Genesis env as variant 2 plus PN8.

Empirical:           ~80 TPS at 1K (slightly lower than short-ctx variant)
                     128K stable: 53 s prefill, 1258 t/s
                     256K stable: 311 s prefill, 845 t/s
                     262 104 actual prompt tokens at TRUE 256K
                     GPU memory 19.60 GB after load (vs 22.69 GB short-ctx)
```

Files: [`scripts/launch/start_27b_int4_no_TQ_long_256K.sh`](scripts/launch/start_27b_int4_no_TQ_long_256K.sh) · [`bare_metal_27b_int4_no_TQ_long_256K.sh`](scripts/launch/bare_metal_27b_int4_no_TQ_long_256K.sh)

### 4. 27B-INT4-Lorbus + TurboQuant k8v4 (hybrid TQ)

```
Model:               Qwen3.6-27B-int4-AutoRound
KV cache:            turboquant_k8v4   ← unlocked via P4 + P98
Spec-decode:         MTP K=3
Max context:         280 000
gpu-mem-util:        0.90
max-num-seqs:        2
max-num-batched-tk:  2 048

Same Genesis env as variant 3 plus:
  GENESIS_ENABLE_P98=1   ⭐ REQUIRED — WorkspaceManager revert (vllm#40941 lock fix)

Empirical:           wall_TPS 90.49 (+1.9% NS vs no-TQ, Welch p=0.067)
                     tool-call 4/4 clean
                     256K capable (probe pending)
```

Files: [`scripts/launch/start_27b_int4_TQ_k8v4.sh`](scripts/launch/start_27b_int4_TQ_k8v4.sh) · [`bare_metal_27b_int4_TQ_k8v4.sh`](scripts/launch/bare_metal_27b_int4_TQ_k8v4.sh)

### Single-card variants (⚠️ EXPERIMENTAL — NOT TESTED by maintainer)

Each of the 4 PROD configs above ships a TP=1 single-card derivative:

- [`start_35b_fp8_PROD_single_card.sh`](scripts/launch/start_35b_fp8_PROD_single_card.sh) — 35B-A3B-FP8 needs ≥48 GB single card (A6000, 6000 Ada, L40, RTX PRO 5000 Blackwell 48 GB, RTX PRO 6000 Blackwell 96 GB, A100, H100, B200)
- [`start_27b_int4_no_TQ_short_single_card.sh`](scripts/launch/start_27b_int4_no_TQ_short_single_card.sh) — fits 24 GB+ (3090, 4090, 5090, A5000, RTX PRO 4000 Blackwell 24 GB, etc.)
- [`start_27b_int4_no_TQ_long_256K_single_card.sh`](scripts/launch/start_27b_int4_no_TQ_long_256K_single_card.sh)
- [`start_27b_int4_TQ_k8v4_single_card.sh`](scripts/launch/start_27b_int4_TQ_k8v4_single_card.sh)

Plus matching `bare_metal_*_single_card.sh` for native (non-Docker) runs.

Each script has a prominent header warning marking it as **EXPERIMENTAL · NOT TESTED**, with hardware-class sizing notes. Sander runs 2× A5000 — these have not been benched end-to-end. **If you run one and it works, please share results via [GitHub Discussions](https://github.com/Sandermage/genesis-vllm-patches/discussions)** — confirmed configs get folded back and the EXPERIMENTAL tag dropped for that card class.

### Deferred — 27B-INT8-Minachist

Both Minachist `Qwen3.6-27B-INT8-AutoRound` (group_size=-1, AllSpark path) and `Qwen3.6-27B-INT8-gs128` (group_size=128, Marlin path) currently boot-crash on `torch._dynamo.exc.Unsupported: infer_schema` in `qwen3_next.py:408 self.linear_attn(...)`. **Crash is independent of Genesis patches** — reproduces with `v764d` minimal config. Wait for pin bump past v0.20.2 or upstream fix.

---

## Genesis Benchmark Suite

The flagship community-grade benchmark is **[`tools/genesis_bench_suite.py`](tools/genesis_bench_suite.py)** — a single self-contained Python script (816 LOC, stdlib + `requests` only, no scipy, no extra deps) that runs the full battery in one command.

### What it measures

Seven phases, configurable scope, JSON + Markdown output:

1. **Server discovery + GPU profile** — auto-detects card class via nvidia-smi, looks up bandwidth from the static datasheet
2. **Tool-call quality** — 8 cases (4 prompts × thinking on/off + 1 negative case to check the model can refuse)
3. **Decode-only TPOT bench** — N runs × M prompts × K decode tokens. Methodology adopted from [thc1006](https://github.com/thc1006)'s `bench_v3_clean_ab.py`. Strips TTFT + queue + scheduler from raw decode rate.
4. **Multi-turn TTFT** — 5 sequential same-prefix requests, smell test for cache benefit
5. **Stability stress** — configurable N iterations × M prompts. Tracks CV and any failures over a long run.
6. **Context window probe** — selectable max via `--ctx`: 1K / 4K / 8K / 16K / 32K / 64K / 128K / 256K / 512K / `all`. Stops at first OOM and reports the max stable.
7. **Output** — `<name>.json` (machine-readable, full per-trial data) + `<name>.md` (human summary)

### Run it

```bash
# Quick smoke test (~5 min)
python3 tools/genesis_bench_suite.py --quick

# Standard run (25 min) — bench + tool-call + multi-turn TTFT + ctx probe
python3 tools/genesis_bench_suite.py --mode standard --ctx 8K

# Full battery — pick the largest ctx your card can hold
python3 tools/genesis_bench_suite.py --mode full --ctx 256K

# Compare two arms (Welch's t-test on per-prompt decode TPOT)
python3 tools/genesis_bench_suite.py --compare run_A.json run_B.json
```

The default targets `localhost:8000` with API key `genesis-local`. Override with `--host`, `--port`, `--api-key`, `--model`. See `--help` for the full flag list.

### Sample output

```
========================================================================
Genesis Benchmark Suite — standard_2026-04-29T13-17-22Z
Mode: standard  ctx max: 8K  runs: 25  stress: 30
========================================================================

[0/7] Server discovery...
      reachable: True (HTTP 200)
      model: qwen3.6-35b-a3b
      local GPUs:
        GPU 0: NVIDIA RTX A5000  VRAM 22663/24564 MiB
        GPU 1: NVIDIA RTX A5000  VRAM 22663/24564 MiB

[1/7] Tool-call quality (8 cases)...
      4/7 positive cases passed (negative case scored separately)

[2/7] Decode bench (25 runs × 5 prompts × 1024)...
      wall_TPS = 184.471  CV 0.0873
      decode_TPOT_ms = 5.2953  CV 0.0824
      TTFT_ms = 159.50  CV 0.1382

[3/7] Multi-turn TTFT (5 turns)...
      turn 1: 171.3ms
      turn 2: 161.1ms
      turn 3: 155.5ms
      turn 4: 206.5ms
      turn 5: 164.5ms

[4/7] Stability stress (30 iters × 5 prompts)...
      duration 1217.4s  failures 0
      wall_TPS 184.81  CV 0.0899

[5/7] Context probe (max 8K)...
      max stable: 8K
========================================================================
```

### Run guide

For detailed step-by-step instructions covering bare metal, Docker, Proxmox VM, WSL2, and RunPod see **[`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md)**.

The guide also documents:

- How to interpret each metric (wall_TPS vs decode_TPOT_ms; CV; what "tool-call 3/4" actually means)
- Common failure modes (`ConnectionRefused` = vllm not yet booted; `FAIL_HTTP_500` at long ctx = OOM)
- Sharing protocol via [GitHub Discussions](https://github.com/Sandermage/genesis-vllm-patches/discussions) — the project is interested in community baselines on diverse hardware
- Privacy: bench does NOT phone home, NOT upload, NOT collect telemetry. All data stays local.

### Validation scripts (different purpose)

The bench suite measures **performance**. For **correctness validation** (apply matrix, smoke tests, pytest) use:

- [`validate_unit.sh`](scripts/validate_unit.sh) — CPU-only Python pytest in transient Docker container (~30 sec)
- [`validate_integration.sh`](scripts/validate_integration.sh) — GPU pytest + container health + chat-completion smoke + diagnostic probes
- [`scripts/run_validation_suite.sh <model_tag>`](scripts/run_validation_suite.sh) — universal per-model validation runner with new model tags (`qwen3_6_35b_fp8`, `qwen3_6_27b_int4_short`, `qwen3_6_27b_int4_long`, `qwen3_6_27b_int4_TQ`)

---

## Patch catalog

Genesis ships **48 runtime patches** in the dispatcher's `PATCH_REGISTRY` (the schema-validated, lifecycle-tracked set). Each patch is opt-in via env var. See [`PATCHES.md`](PATCHES.md) for the canonical full reference; the table below highlights the most operator-facing ones.

### TurboQuant + KV cache (foundational)

| ID | Title | Default | Notes |
|---|---|:---:|---|
| P3 | TQ BF16→FP8 cast (Ampere fix) | ON | Required for FP8 on SM 8.6 |
| P4 | TQ hybrid model support | ON | Removes the hybrid rejection (Qwen3.6-27B etc) |
| P5 / P5b | KV cache page size unification | ON | LCM-pad for hybrid; v1 baseline |
| P6 | TQ-aware attention page size | ON | Aligns block size to TQ packed slot |
| P22 | TQ shared dequant prealloc | (auto-skipped) | PR #40655 merged upstream — auto-no-op |
| P36 | TQ shared decode buffers | (auto-skipped) | PR #40798 merged — auto-no-op |
| P38 | TQ continuation-prefill workspace | ON | Persistent K_full/V_full buffers |
| **P40** | **TQ k8v4 GQA grouping kernel** | **OFF** | **Recommend ON for 4090/5090/H100/Blackwell only** |
| P81 | FP8 block_scaled M ≤ 8 decode tune | ON | Targets MTP K=3 verify M=4 |
| **P98** | **TQ WorkspaceManager revert** | **conditional** | **REQUIRED for TQ on hybrid models** |
| P99 | WorkspaceManager.get_simultaneous memoize | ON | ~5× speedup per call, perf hotfix |
| P101 | TQ continuation 64-token slicing | ON | Selective backport of vllm#41123 |

### Spec-decode (MTP K=3)

| ID | Title | Default |
|---|---|:---:|
| P56/P57 | Spec-decode capture-safe buffers | OFF |
| P58 | Async-scheduler placeholder fix (vllm#40768) | ON |
| P63 | MTP GDN state recovery | OFF |
| P65 | TQ spec-decode CG downgrade | OFF |
| P66 | CUDA graph size divisibility filter | ON |
| P67 / P67b | Multi-query verify kernel for K+1 | ON |
| P70 | Auto-strict ngram (clean_rate ceiling fix) | ON |
| P77 | Adaptive ngram K controller (SGLang port) | OFF |
| P79b/c/d | Async × spec-decode race fixes | OFF |
| P82 | SGLang threshold_single OR-clause | ON (35B) |
| **PN8** | **MTP draft online-quant propagation** | **ON for FP8 + MTP** |
| **PN9** | **Independent drafter attention backend** | **(self-retired in pin)** |
| **P102** | **Unified spec-decode metadata (TRT-LLM-inspired)** | **OFF (Phase 1 assertion-only mode)** |

### Hybrid GDN / Mamba

| ID | Title | Default |
|---|---|:---:|
| P7 / P7b | GDN dual-stream custom_op | ON |
| P28 | GDN core_attn_out prealloc | ON |
| P34 | Mamba zero-collapse deadlock guard | ON |
| P39 / P39a | FLA chunk_scaled_dot_kkt buffer pool | ON |
| P46 | GDN gating buffers | ON |
| P60 / P60b | GDN+ngram state recovery (vllm#40738) | ON |
| **P103** | **FLA Cliff 2 chunked fwd_h+fwd_o orchestrator** | **ON for long-ctx single-card** |
| **PN11** | **GDN a/b contiguity (vllm#41142, by Quentin Machu)** | **ON (defensive)** |

### Tool-call + reasoning parsers

| ID | Title | Default |
|---|---|:---:|
| P12 | Tool-call reasoning extraction | ON |
| P15 | Qwen3 None-null normalization | ON |
| P27 | Reasoning-before-think parser fallback | ON |
| P59 | Qwen3 reasoning tool-call recovery (vllm#39055) | ON |
| P61 | Multi-tool first-occurrence (vllm#40783) | ON |
| P61b | Streaming overlap guard slice | ON |
| P62 | Reasoning-aware grammar (vllm#36138, sfbemerk) | ON |
| **P64** | **Streaming tool-call early-return + Quentin's IndexError fix** | **ON** |
| P68 / P69 | Auto force tool / long-ctx tool reminder | ON |
| P74 | Chunk clamp | ON |

### Marlin / quantization

| ID | Title | Default |
|---|---|:---:|
| P17/P18 | Marlin tuning | ON |
| P23 | Marlin FP32_REDUCE auto-disable on SM 8.6 | ON |
| P24 | MoE tuning | ON |
| P31 | Router softmax | ON |
| P37 | MoE intermediate cache pool | conditional |
| P87 | Marlin sub-tile pad-on-load (vllm#40361) | ON |
| P91 | AutoRound row-parallel cdiv (vllm#39460) | ON |
| P93 | AllSpark bypass for INT8 W8A16 | OFF |
| P95 | Marlin TP cudagraph cap | OFF |

### Cache / scheduler / memory utilities

| ID | Title | Default |
|---|---|:---:|
| P14 | KV cache block table guard | ON |
| P26 | TQ prefill output prealloc | (auto-skipped if upstream merged) |
| P44 | TQ mixed-attention out buffer | ON |
| P72 | Profile-run M cap (unblocks max-num-batched-tokens > 4096 on MoE) | ON |
| P75 | Suffix decoding enable (Arctic Inference port, vllm#25784) | OFF |
| P78 | TQ `.tolist()` cudagraph capture guard (Apache-2.0 from @noonghunna) | OFF |
| P79c | Stale `spec_token_ids` cleanup for unscheduled requests | OFF |

### Pre-Genesis startup probes (`external_probe/`)

Genesis ships **three small probes** that run BEFORE the main `apply_all` orchestrator. They patch crashes that would otherwise prevent the engine from booting under our exact stack — fixed first, then Genesis applies on top of a working engine:

| Probe | Purpose | Status |
|---|---|---|
| [`external_probe/patch_tolist_cudagraph.py`](external_probe/patch_tolist_cudagraph.py) | `.tolist()` cudagraph-capture guards in `turboquant_attn.py` (forward + `_prefill_attention`). Adapted from [@noonghunna](https://github.com/noonghunna)'s `patch_tolist_cudagraph.py` (Apache-2.0, full attribution in CREDITS.md). | runs every boot |
| [`external_probe/patch_40074_iooo.py`](external_probe/patch_40074_iooo.py) | 5-line backport of [vllm#40074](https://github.com/vllm-project/vllm/pull/40074) — IOOB (index-out-of-bounds) clamp on Triton block-table pointer arithmetic. Fixes [vllm#39998](https://github.com/vllm-project/vllm/issues/39998), possibly [#40831](https://github.com/vllm-project/vllm/issues/40831). | runs every boot |
| [`external_probe/patch_pr40798_backport.py`](external_probe/patch_pr40798_backport.py) | Backport of [vllm#40798](https://github.com/vllm-project/vllm/pull/40798) — TurboQuant share decode scratch workspace across layers (4 file changes). Hypothesized side-effect fix for vllm#40831 token loops via stable workspace pointer. | runs every boot |

These are referenced by Docker launch scripts via `python3 /external_probe/patch_*.py` calls before `python3 -m vllm._genesis.patches.apply_all`. If you fork the repo or copy launch scripts, **make sure `external_probe/` exists at the path the script mounts** (`-v $GENESIS_REPO/external_probe:/external_probe:ro`). The bare-metal scripts call them via `python3 $GENESIS_REPO/external_probe/patch_*.py` instead.

### P102 — Unified spec-decode metadata (TRT-LLM-inspired architecture)

A library file rather than a wiring text-patch — lives at [`vllm/_genesis/spec_meta.py`](vllm/_genesis/spec_meta.py). Per Sander's 2026-04-28 architectural study of `tensorrt_llm/_torch/speculative/interface.py`:

> Current Genesis approach: each spec-decode-aware patch (P67/P67b/P78/P98/P99) re-derives "are we in spec-decode? cudagraph? warmup?" from local hints. There is NO single source of truth, so v756 regression broke specifically because P67 fired on chunked-prefill batches when spec-decode was OFF.

**Solution:** single `GenesisSpecMeta` dataclass holds the full spec-decode + cudagraph + warmup state for the current step. Set by V1 `GPUModelRunner.execute_model` prelude. Consumed by predicate functions like `should_dispatch_p67(...)` that replace scattered inline checks.

The TRT-LLM canonical idiom we ported (from `dflash.py:269`):

```python
is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
```

This single-source warmup-vs-replay distinction is what bites our P67 hot path. Status: opt-in via `GENESIS_ENABLE_P102=1` (default OFF — Phase 1 assertion-only mode where predicates RECORD their decision and disagreement logs WARNING but doesn't change behavior). Phase 2 / 3 land in subsequent releases as we migrate the existing inline checks over.

### Disproven / not recommended

| ID | Why |
|---|---|
| P83 + P84 + P85 | Empirically -29% TPS regression with prefix-cache on current pin (this session) |
| P104 | -16.2% via L2 cache thrashing on 32+ layer transformer (prior session) |
| P105 | Variance noise on dequant kernel (prior session) |

For full per-patch documentation, opt-in env names, and credit lines see [PATCHES.md](PATCHES.md) and [CREDITS.md](CREDITS.md).

---

## Per-GPU recommendations

Genesis auto-detects your GPU at boot via [`vllm/_genesis/gpu_profile.py`](vllm/_genesis/gpu_profile.py) and prints `[REC]` (recommended) / `[OFF]` (not recommended) per patch in the apply log. The static datasheet covers 18 GPU classes:

| GPU class | Bandwidth | L2 | Regime | P40 | P67 | P82 | PN8 | P83/84/85 |
|---|---:|---:|---|:---:|:---:|:---:|:---:|:---:|
| RTX 3060 / 3070 / 3080 | 360-760 GB/s | 3-5 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX 3090 | 936 GB/s | 6 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX A4000 | 448 GB/s | 4 MB | bandwidth | OFF | REC | REC | REC | OFF |
| **RTX A5000 (Sander's PROD)** | **768 GB/s** | **4 MB** | **bandwidth** | **OFF** | **REC** | **REC** | **REC** | **OFF** |
| RTX A6000 | 768 GB/s | 6 MB | bandwidth | OFF | REC | REC | REC | OFF |
| RTX 4070 | 504 GB/s | 36 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 4080 | 716 GB/s | 64 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 4090 | 1008 GB/s | 72 MB | mixed | **REC** | REC | REC | REC | OFF |
| L40 / L40S | 864 GB/s | 96 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 6000 Ada | 960 GB/s | 96 MB | mixed | REC | REC | REC | REC | OFF |
| RTX 5080 | 960 GB/s | 64 MB | mixed | REC | REC | REC | REC | OFF |
| **RTX 5090** | **1792 GB/s** | **88 MB** | **compute** | **REC** | REC | REC | REC | OFF |
| **RTX PRO 4000 Blackwell** (24 GB) | 672 GB/s | 24 MB | mixed | REC | REC | REC | REC | OFF |
| RTX PRO 4500 Blackwell (32 GB) | 896 GB/s | 32 MB | mixed | REC | REC | REC | REC | OFF |
| RTX PRO 5000 Blackwell (48 GB) | 1344 GB/s | 64 MB | compute | REC | REC | REC | REC | OFF |
| **RTX PRO 6000 Blackwell** (96 GB) | 1792 GB/s | 88 MB | compute | REC | REC | REC | REC | OFF |
| RTX PRO 6000 Blackwell Max-Q | 1792 GB/s | 88 MB | compute | REC | REC | REC | REC | OFF |
| A100 80GB | 2039 GB/s | 40 MB | compute | REC | REC | REC | REC | OFF |
| H100 / H200 | 3350-4800 GB/s | 50 MB | compute | REC | REC | REC | REC | OFF |
| B200 | 8000 GB/s | 80 MB+ | compute | REC | REC | REC | REC | OFF |

**Reading the table:** `REC` = empirically expected to deliver gain on this card class; `OFF` = either neutral or known-regressive. Auto-detection runs at every boot — operator pastes the `export GENESIS_ENABLE_*=1` lines from the printed recommendations into the launch script.

If your card isn't listed, Genesis falls back to "unknown regime — no recommendations" and you get the dispatcher's default behavior. Open a [GitHub Discussion](https://github.com/Sandermage/genesis-vllm-patches/discussions) with `nvidia-smi --query-gpu=name,memory.total --format=csv` output and we'll add it to the next release.

---

## Empirical findings — what worked, what didn't

Operators value honesty about both. From this session and prior:

### What worked (empirically validated, default ON or REC)

- **TurboQuant k8v4 KV cache** — measured **+11.3% TPS** on 35B-A3B-FP8 (Sander 2026-04-29 A/B vs `--kv-cache-dtype auto`). NOT just memory savings — the packed-slot layout helps L2 locality even on Ampere consumer.
- **MTP K=3 spec-decode** — accepts ~50-70% of speculative tokens depending on workload; net positive on every model class tested.
- **PN8 (online-quant draft propagation)** — saves ~1 GiB VRAM per GPU on FP8 + MTP. Throughput-neutral. **PROMOTE in any FP8 + MTP target.**
- **P82 (SGLang threshold_single OR-clause)** — cross-rig +12% on Sander's A5000 + FP8, +10.5% on noonghunna's 3090 + INT4. Two SM 8.6 datapoints, two quant paths.
- **P67 / P67b (multi-query verify kernel)** — +25-35% on spec-decode K+1 verify across all GPU classes tested.
- **256K context on 27B Lorbus via lighter config** — util 0.90 / max-num-seqs 2 / max-num-batched-tokens 2048. Was config-aggressiveness, not a model limit.
- **TurboQuant on hybrid GDN (P4 + P98)** — first time we got TQ k8v4 working on Lorbus 27B. Tool-call 4/4 clean.

### What didn't work (do NOT enable on Ampere consumer)

- **Prefix caching on 35B + MTP** — `--enable-prefix-caching` triggers a -29% TPS regression that **none of P83 / P84 / P85 mitigate**. Tested 4-arm A/B (this session). The supposed root-cause patch P84 didn't help. Conclusion: cache machinery overhead is real and the patches we have don't recover it.
- **P40 default-on for Ampere consumer (4 MB L2)** — fix landed (signature drift), kernel runs correctly, but +1.14% NS (Welch p=0.284). The +27% from upstream is real on Hopper-class L2 (50+ MB) but doesn't materialize when you're already memory-bandwidth-bound.
- **P104 (L2 persistence cache thrashing)** — -16.2% on 32+ layer transformer with KV >> L2. Each layer's pin evicts the previous. Architecture mismatch. Removed.
- **P105 (lmdeploy num_stages=3 hint on dequant kernel)** — variance noise (-1.4% retest after proper registration). Cross-engine kernel hints from prefill contexts don't apply to dequant utility kernels.
- **Triton 3.6 `disallow_acc_multi_buffer + loop_unroll_factor=2`** — -6.5% on small-BLOCK_M split-M kernel. Generic compiler hints don't transfer; always per-kernel A/B.
- **Aggressive `gpu-memory-utilization=0.95` + 4 streams + 8K batch on 27B** — OOMs at ≥16K context. Use 0.85-0.90 + 2 streams + 2K batch for long-ctx; the v791b config validates 256K stable.

### Special cases

- **Tool-call 3/4 on 35B** — case 4 (`think=true` + complex prompt + small `max_tokens`) is a **harness budget artifact, not a quality regression.** The model writes a long reasoning chain inside `<think>` and runs out of tokens before emitting the tool_call. Raise `max_tokens` from 300 to 1500 and you get 4/4. The bench suite uses 1024 by default; the test harness was upgraded to 1500.
- **TQ k8v4 on Lorbus 27B-INT4** — works (P4 + P98), tool-call 4/4 clean, but only +1.9% NS over fp8_e5m2. The +11.3% TQ bonus seen on 35B-A3B-FP8 doesn't reproduce because Lorbus routes through `AllSparkLinearKernel`, not Marlin / compressed-tensors. Different memory traffic profile. Both variants ship — no strict winner.

For the full chronological derivations see the memory notes referenced in each section above.

---

## Hardware and operating envelope

### Tested PROD environment

- 2× RTX A5000 24 GB (Ampere SM 8.6)
- Ubuntu 24.04, kernel 6.8
- NVIDIA driver `≥580.126.09` — **REQUIRED**, driver 570 puts PyTorch into a compatibility fallback ≈ 3× slower decode
- vLLM nightly pin `8cd174fa3` (image `vllm/vllm-openai:nightly`)
- PyTorch 2.11.0 + CUDA 13.0
- Triton 3.6.0, FlashInfer (MoE FP8 disabled per `VLLM_USE_FLASHINFER_MOE_FP8=0`)
- Genesis patch tree at the commit you cloned

### Cross-rig validated

- noonghunna's dual-3090 cluster (cross-rig +10.5% on P82, INT4)
- thc1006's 2×3090 cluster (provided the bench methodology + decode-only TPOT design)
- Quentin Machu's setup (P64 IndexError reproduced + fixed)

### Pin management

We hold pin at `8cd174fa3` deliberately. **Do NOT bump unless you've read [`docs/_internal/research_vllm_main_nightly_audit_20260429.md`](docs/_internal/) first** (gitignored, ask Sander) — there are several merged PRs in main that would break our patches simultaneously:

- vllm#40941 (TurboQuant share-dequant) — caused our prior 200→167 PROD regression, mitigated via P98/P99
- vllm#40860 (DSV4 mega-merge, +16k LOC) — breaks anchors for ~13 patches; ~2.5h of careful re-anchoring needed
- vllm#41184 (FusedMoE inversion, 108 files) — mass `intermediate_size_per_partition` → `intermediate_size` rename

The plan is to wait for v0.20.2 stable and do a single full TDD-gated re-anchor pass.

### Software you also need

- Docker 24+ with NVIDIA Container Toolkit (for the `start_*.sh` flow)
- Python 3.10+ + `requests` (for the bench suite)
- `gh` CLI (optional; for community sharing)
- `nsys` 2025.6+ (optional; for performance profiling — install via NVIDIA CUDA repo: `sudo apt install nsight-systems-2025.6.3` after adding `cuda-keyring`)

---

## How patches work

Genesis patches are **runtime modifications** — the engine loads stock vLLM, then `python3 -m vllm._genesis.patches.apply_all` modifies in-memory bytecode + monkey-patches Python objects to install the fixes. Three layers:

### 1. The dispatcher ([`vllm/_genesis/dispatcher.py`](vllm/_genesis/dispatcher.py))

Single source of truth — `PATCH_REGISTRY` dict with metadata for each patch (env flag, default state, applies-to predicate, credit line, upstream PR ref). Auto-evaluates per-patch predicates against detected model + GPU + spec-decode method, prints the apply matrix at boot.

### 2. The wiring layer ([`vllm/_genesis/wiring/`](vllm/_genesis/wiring/))

One file per patch (e.g., `patch_82_sglang_acceptance_threshold.py`). Each defines:
- `apply()` — the function `apply_all` calls
- `_make_patcher()` — for text-patches, defines anchor strings + replacement strings
- `UPSTREAM_DRIFT_MARKERS` — strings whose presence in upstream source triggers automatic self-retirement (when the upstream PR finally merges, our patch goes idempotent without operator intervention)

### 3. The GPU profile ([`vllm/_genesis/gpu_profile.py`](vllm/_genesis/gpu_profile.py))

18-card datasheet with bandwidth + L2 + SM count + compute capability + regime classification (bandwidth-bound / mixed / compute-bound). Per-patch predicates evaluate against the detected card. Prints `[REC]/[OFF]` recommendations at boot.

### Container R/W layer caveat

If you ran `docker compose stop` then `docker compose start`, the container's R/W layer **preserves the patched files** from the previous boot. Monolithic re-application can fail on anchor drift because the file is already mutated. Always use `docker compose down && docker compose up -d` (full container recreate) when re-applying patches. The text-patcher has idempotency markers, but the cleanest invariant is "fresh container per Genesis version".

---

## Acknowledgments

Genesis stands on the shoulders of the upstream vLLM project + the open-source community. This list captures the authors whose work directly powers Genesis. For the canonical full credit reference see [CREDITS.md](CREDITS.md).

### Community contributors (this release)

- **[Quentin Machu (@Quentin-M)](https://github.com/Quentin-M)** — first community-contributed Genesis patch. Diagnosed the IndexError in `chat_completion_stream_generator` final delta + landed the fix as P64 sub-patch E in [his fork](https://github.com/Quentin-M/genesis-vllm-patches/commit/09688b1d). Excellent root-cause writeup + minimal correct fix. Thank you Quentin.
- **[thc1006](https://github.com/thc1006)** — `bench_v3_clean_ab.py` decode-only TPOT methodology (the foundation of our bench suite). Also originated the prefix-cache OFF discovery (vllm#38182 adverse interaction).
- **[@noonghunna](https://github.com/noonghunna)** — community lead, two downstream forks (`qwen36-dual-3090`, `qwen36-27b-single-3090`), cross-rig validation on multiple rigs, contributed apples-to-apples baseline data for Sander's PRs.

### Upstream PR authors we backport

| Genesis patch | vllm PR | Author |
|---|---|---|
| P58 | [#40768](https://github.com/vllm-project/vllm/pull/40768) | z1ying |
| P59 | [#39055](https://github.com/vllm-project/vllm/pull/39055) | ZenoAFfectionate |
| P60, P60b | [#40738](https://github.com/vllm-project/vllm/pull/40738) | Thomas Parnell ([@tdoublep](https://github.com/tdoublep)) |
| P61, P61b | [#40783](https://github.com/vllm-project/vllm/pull/40783) | ExtReMLapin |
| P62 | [#36138](https://github.com/vllm-project/vllm/pull/36138) | sfbemerk |
| P64 | [#39598](https://github.com/vllm-project/vllm/pull/39598) | kotori-yan |
| P71 | [#40819](https://github.com/vllm-project/vllm/pull/40819) | Z. Golpayegani + gemini-code-assist (review) |
| P77 | SGLang `adaptive_spec_params.py` (Apache-2.0 port) | SGLang team + Nightjar paper authors |
| P81 | [#40925](https://github.com/vllm-project/vllm/pull/40925) | tonyliu312 |
| P82 | SGLang `speculative_sampling.cuh` ([SGLang](https://github.com/sgl-project/sglang)) | SGLang team |
| P86 | [#40876](https://github.com/vllm-project/vllm/pull/40876) | aaronagent |
| P87 | [#40361](https://github.com/vllm-project/vllm/pull/40361) | vLLM contributor |
| P91 | [#39460](https://github.com/vllm-project/vllm/pull/39460) | vLLM contributor |
| P94 | [#41043](https://github.com/vllm-project/vllm/pull/41043) | wangluochao902 |
| P100 | [#41127](https://github.com/vllm-project/vllm/pull/41127) | vLLM contributor |
| P101 | [#41123](https://github.com/vllm-project/vllm/pull/41123) (selective) | cderinbogaz |
| **PN8** | **[#40849](https://github.com/vllm-project/vllm/pull/40849)** | **[@bhoomit](https://github.com/bhoomit)** |
| **PN9** | **[#39930](https://github.com/vllm-project/vllm/pull/39930)** | **MatthewBonanni (merged upstream — auto-self-retires)** |
| **PN11** | **[#41142](https://github.com/vllm-project/vllm/pull/41142)** | **Yeuvoir (upstream PR) + [Quentin Machu](https://github.com/Quentin-M) (Genesis backport)** |

### Issue reporters whose investigation informed our patches

- **noonghunna** — vllm#40807 / #40831 (TQ + spec-decode + chunked-prefill `tolist()` crash; degenerate token loop)
- **SongXiaoMao** — vllm#40756 (MTP IMA on long sequences, sibling-bug)
- **bhaktatejas922** — vllm#39273 (GDN + ngram corruption original report)
- **cicirori (yinghui)** — vllm#34650 (MTP + reasoning + structured output `</think>` detection)
- **uOnePiece** + **@Angazenn** — vllm#38182 root-cause analysis (informed P83)
- **JartX** — vllm#39931 (broader hybrid TurboQuant work)

### vLLM core maintainers we depend on

The entire upstream vLLM team. Our patches are runtime modifications of their code; without their engine there is no Genesis. See [vLLM's CONTRIBUTORS](https://github.com/vllm-project/vllm/graphs/contributors).

### Special thanks

- Sander's homelab (2× RTX A5000 + Proxmox infra) for taking the pounding of 36+ hours of A/B testing without a single hardware crash
- The community that's reproduced findings cross-rig and reported back

---

## License / disclaimer

Apache-2.0 (matches vLLM upstream — see [LICENSE](LICENSE)).

Genesis is **NOT affiliated with or endorsed by** vLLM, NVIDIA, Alibaba, or any model author. It's a community downstream project. Patches are AS-IS — they work on the tested stack and are documented; no warranty implied.

Don't blindly enable everything. **Read the patch description, check the recommendation for your card, and bench in your environment** before promoting to production. The bench suite makes this fast — 25 minutes for a full standard run.

For commercial support / consulting / collaboration:

- **Author:** Sandermage (Sander) Barzov Aleksandr — Ukraine, Odessa
- **Repo:** https://github.com/Sandermage/genesis-vllm-patches
- **Discussions:** https://github.com/Sandermage/genesis-vllm-patches/discussions
- **License:** [Apache-2.0](LICENSE)

If you find Genesis useful and want to support continued development, see [SPONSORS.md](SPONSORS.md).

---

*Genesis vLLM Patches — empirical, attribution-rich, AS-IS. Built nights and weekends with the community.*
