<p align="center">
  <img src="assets/logo.png" alt="SNDR Core Engine — Genesis vLLM Patches" width="780">
</p>

# SNDR Core Engine

> **Genesis vLLM Patches** — runtime patch-overlay for vLLM on consumer NVIDIA
> Ampere / Ada / Blackwell.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM pin](https://img.shields.io/badge/vllm-0.23.1rc1.dev424+g3f5a1e173-orange.svg)](https://github.com/vllm-project/vllm)
[![Patches](https://img.shields.io/badge/registry-321%20patches-green.svg)](docs/PATCHES.md)
[![SNDR Core](https://img.shields.io/badge/SNDR%20Core-v12.0.0-blue.svg)](CHANGELOG.md)
[![GPU](https://img.shields.io/badge/GPU-RTX%203090%20%7C%204090%20%7C%205090%20%7C%20A5000%20%7C%20H20%20%7C%20R6000-purple.svg)](docs/HARDWARE.md)

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — Qwen3.6-class
inference on consumer NVIDIA Ampere / Ada / Blackwell with TurboQuant k8v4 KV
cache, MTP K=5 spec-decode, tool-calling, and 256K-class context. 321 patches
across ~23 families. Apache 2.0.**

---

> 🚀 **New here?** → [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — who it's for, what you get, and the one install line.
> 🧠 **New to local AI?** → [`docs/LOCAL_AI_PRIMER.md`](docs/LOCAL_AI_PRIMER.md) — GPUs, engines, MoE, and quants in plain English.
> 📖 **Hit an unfamiliar term** (TPS · KV · MTP · TurboQuant · GDN)? → [`docs/GLOSSARY.md`](docs/GLOSSARY.md).
> 💸 **Self-host or cloud?** → [`docs/COMPARISONS.md`](docs/COMPARISONS.md) — the cost-crossover trade.

## What it is

A **drop-in runtime patcher** for vLLM. It pins to a specific vLLM nightly
commit and applies 321 small, surgical changes — text edits at known anchors,
class-rebind wrappers, and FastAPI middleware — that together turn an
out-of-the-box vLLM into a production-grade Qwen3.6 inference server on
*consumer* NVIDIA hardware (3090, 4090, 5090, A5000, A6000, …) where vLLM
upstream mostly targets datacenter SKUs.

It is **not** a fork of vLLM, a quantizer, a new inference engine, or a
training framework. Patches retire automatically when upstream merges the
underlying fix.

## How it works

**The overlay / apply model.** Genesis never edits vLLM on disk. At every
process start the plugin registers via vLLM's `vllm.general_plugins` entry
point (loaded in the main process, the engine, and every worker rank) and
the dispatcher walks `PATCH_REGISTRY`. Each patch declares an `applies_to`
version range and an apply method — a **text edit at a unique source
anchor**, a **class-rebind wrapper**, or **FastAPI middleware**. Patches
whose anchors match and whose range covers the live pin apply; the rest
print `[SKIP — applies_to mismatch]` and no-op. The result is an in-memory
overlay: the same wheel, transformed at boot, with a structured apply
summary (`applied=N skipped=M failed=0`) and an audit trail. Nothing is
written to the vLLM package tree.

**Patch families.** The 321 entries group into ~23 canonical families. The
largest: `attention.turboquant` (k8v4 KV-cache quant), `spec_decode` (MTP /
ngram speculative decoding), `attention.gdn` (hybrid Gated-DeltaNet linear
attention), `gemma4` (Gemma-4 enablement), `kv_cache`, `compile_safety`,
`worker`, `serving`, `tool_parsing`, and `moe`. The full table is
[`docs/PATCHES.md`](docs/PATCHES.md) (curated) +
[`docs/PATCHES_AUTO.md`](docs/PATCHES_AUTO.md) (generated from the registry).

**Pin lifecycle.** Genesis pins to one canonical vLLM nightly at a time,
plus an optional previous pin held for rollback during validation — at most
two ("≤2-pin policy"). A bump happens only on an explicit instruction
naming the target pin; there are **no proactive pulls**. The candidate is
validated before promotion (anchor-drift resolved, the `bump-preflight`
gate clean, boot-smoke + tokenizer-fingerprint + canonical bench), then the
old 2-back pin is dropped. Current: `dev424` (`3f5a1e173`); rollback:
`dev301` (`04c2a8dea`). See [`docs/PIN_BUMP_PLAYBOOK.md`](docs/PIN_BUMP_PLAYBOOK.md)
(canonical) + [`docs/ANCHOR_SOT.md`](docs/ANCHOR_SOT.md).

**Model catalog (current registry).**

| Model | Quant | KV cache | Spec-decode | Status |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B-FP8 | FP8 dense MoE | TurboQuant k8v4 | MTP K=5 | ✅ PROD (default) |
| Qwen3.6-27B-int4-AutoRound | INT4 AutoRound (hybrid GDN+Mamba) | TurboQuant k8v4 | MTP K=5 | ✅ PROD |
| Gemma-4-31B | INT4 / kv-auto | TurboQuant or uniform fp16 | MTP K=3 (separate drafter) | ⚙️ boots + patches apply; serving needs MM-budget config |
| DiffusionGemma-26B-A4B-FP8 | FP8-dynamic block-diffusion MoE | TP=2 | — | ✅ serving at TP=2 |

Per-model deep-dives + the V2 layered config system:
[`docs/MODELS.md`](docs/MODELS.md). Hardware envelope:
[`docs/HARDWARE.md`](docs/HARDWARE.md).

**Launching.** Boot any model through a preset — the launcher resolves the
preset, runs preflight, and renders the `docker run` (or podman / bare-metal
/ k8s) command for you with the correct pin, mounts, and env:

```bash
sndr launch prod-qwen3.6-35b-balanced            # boot a preset
sndr launch prod-qwen3.6-35b-balanced --dry-run  # inspect the rendered command, no boot
```

Full operator manual: [`docs/USAGE.md`](docs/USAGE.md).

## Headline numbers (v12.0.0 current registry)

Reference rig: **2× RTX A5000 24 GB** (Ampere SM 8.6), driver 580.142,
CUDA 13.0.2, MTP K=5 + TurboQuant k8v4, TP=2.

| Model | Stock vLLM | Genesis (v12.0.0) | Δ |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 (single-conc, K=5) | ~157 t/s | **239.7 t/s** | +53 % |
| Qwen3.6-35B-A3B-FP8 (8-way multi-conc, K=3) | n/a | **~675 t/s agg** | 3.21× scaling |
| Qwen3.6-27B-int4-AutoRound (single-conc, K=5) | ~87 t/s | **127.4 t/s** | +46 % |
| Tool-call clean rate (35B / 27B) | 2–6 / 10 | **7/7 · 8/8** | qualitative |

256K context hardware-verified on both models. Full methodology, historical
comparisons, and per-rig reproduction recipes:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

![Sustained TPS — Genesis vs stock](assets/charts/tps_genesis_vs_stock.png)

> **Current pin (2026-06-25):** the vLLM pin is `0.23.1rc1.dev424+g3f5a1e173`
> (image `vllm/vllm-openai:nightly-3f5a1e173…`, commit `3f5a1e173`, +123 commits
> over dev301). `dev301` (`0.23.1rc1.dev301+g04c2a8dea`, commit `04c2a8dea`)
> is retained as the previous / rollback pin per the ≤2-pin policy; `dev148` is
> **dropped**. The dev301→dev424 bump was the first to dogfood the anchor-SOT
> bump tooling (`make bump-preflight` + `retire_impact`) — see
> [`docs/PIN_BUMP_PLAYBOOK.md`](docs/PIN_BUMP_PLAYBOOK.md) (canonical) and
> [`docs/ANCHOR_SOT.md`](docs/ANCHOR_SOT.md). The per-model bench table below is
> the validated dev148 K=5 re-tune cycle (still the canonical sustained-bench
> evidence; decode carried forward across the dev301 and dev424 bumps with no
> regression).

### Validated rig baseline — 2026-06-19 (measured on pin `0.23.1rc1.dev148+gb4c80ec0f`)

Full model-cycle re-test on the reference 2× A5000 rig after the MTP K=3→K=5 re-tune. These are
the canonical sustained-bench numbers; the pin has since bumped dev148 → dev301 → **dev424**
(current) with the decode results carried forward (no regression — anchor regen confirmed at each
bump). Each model boots the Genesis apply pipeline, applies its patch set, and is benchmarked /
smoke-tested live (`tools/genesis_bench_suite.py`, single-stream warm sweep). The 35B / 27B
single-stream rows are the K=5 re-tune numbers; Gemma stays K=3 (its separate drafter is optimal
at K=3).

| Model | Quant / KV | Patches | Decode TPS | Tool-call | Status |
| --- | --- | ---: | ---: | :---: | --- |
| Qwen3.6-35B-A3B-FP8 | FP8 dense · TQ k8v4 · MTP K=5 | 95 | **239.7** (CV 4.9 %) | 7/7 | ✅ serving — +15.8 % vs K=3 |
| Qwen3.6-27B-int4-AutoRound | INT4 AutoRound · TQ k8v4 · MTP K=5 | 93 | **127.4** (CV 8.3 %) | 7/7 | ✅ serving — +8.2 % vs K=3 |
| Gemma-4-31B | INT4 · TQ k8v4 · MTP K=3 | 81 | — | — | ⚙️ boots + patches apply; serving needs MM-budget config (multimodal-bidirectional × spec-decode) |
| DiffusionGemma-26B-A4B-FP8 | FP8-dynamic · block-diffusion · TP=2 | 45 | coherent | — | ✅ **serving at TP=2** — `PN-FP8MOE-KPAD` (Marlin N=352) + `G4_26` (TP-vocab soft-embed); enforce-eager · max-num-seqs 2 · gpu-util 0.80 |

The 35B and 27B clear their historical peak band — the K=5 re-tune lifts single-stream decode
to 239.7 / 127.4 t/s (+15.8 % / +8.2 % vs K=3) within CV → the v12 platform carries **no decode
regression**. `PN-FP8MOE-KPAD` (backport of open vLLM
PR [#45703](https://github.com/vllm-project/vllm/pull/45703), model-agnostic Marlin-MoE
intermediate-pad) plus `G4_26` (backport of [#45774](https://github.com/vllm-project/vllm/pull/45774),
DiffusionGemma TP>1 vocab-sharded soft-embed all-gather) make
**DiffusionGemma the first block-diffusion FP8-MoE checkpoint to boot AND serve coherently
at TP=2 on consumer Ampere** without a kernel rebuild — validated 2026-06-17 (clears the
Marlin N=352 thread-tile crash, then the `probs @ embed_weight` `[131072,2816]` TP-vocab
shape mismatch; the coherent generation confirms the soft-embed all-gather yields correct
TP=2 output).

## Pick your path

| You have | Start here |
| --- | --- |
| **1× consumer card** (3090 / 4090 / 5090 / A5000) | [`docs/SINGLE_CARD.md`](docs/SINGLE_CARD.md) |
| **2× cards** (TP=2 — the reference topology) | [`docs/HARDWARE.md`](docs/HARDWARE.md) + [`docs/MODELS.md`](docs/MODELS.md) |
| **A model not in the catalog** | [`docs/MODELS.md`](docs/MODELS.md) (add-a-model + the V2 config system) |
| **Brand-new / weighing self-host vs cloud** | [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) · [`docs/COMPARISONS.md`](docs/COMPARISONS.md) |

## Quick install

```bash
curl -sSL https://raw.githubusercontent.com/Sandermage/sndr_core_engine/main/install.sh | bash
```

The installer detects your OS / Python / GPU / vLLM presence, clones into
`~/.sndr/`, installs the plugin, writes a tailored launch script, and runs a
60-second smoke test. Five-minute walk-through and Day-1 acceptance steps:
[`docs/QUICKSTART.md`](docs/QUICKSTART.md).

To pick a different vLLM pin, workload, or non-interactive flag set:
[`docs/INSTALL.md`](docs/INSTALL.md).

## Documentation map

| If you want to... | Read |
| --- | --- |
| One-page operator manual (installer → launcher → configs → patches) | [`docs/USAGE.md`](docs/USAGE.md) |
| Install + first boot | [`docs/INSTALL.md`](docs/INSTALL.md) → [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| Browse `sndr` commands | [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) |
| Pick a model + hardware combo | [`docs/MODELS.md`](docs/MODELS.md) + [`docs/HARDWARE.md`](docs/HARDWARE.md) |
| Tune an env-var flag | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| Browse the patch catalogue + compatibility matrix | [`docs/PATCHES.md`](docs/PATCHES.md) |
| Diagnose an OOM, cliff, or boot failure | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| Roll a broken release back | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| See current bench numbers + reproduce | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Author a patch or community plugin | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |
| Sponsorship / hardware loan / business invoicing | [`docs/SPONSORS.md`](docs/SPONSORS.md) |
| Disclose a security issue | [`SECURITY.md`](SECURITY.md) |

Full docs index: [`docs/README.md`](docs/README.md).

## Repository structure

The layout separates the shippable engine from the maintainer tooling and
vendored third-party code, so the published wheel stays small and the apply
pipeline stays auditable.

| Path | What it is |
| --- | --- |
| [`sndr/`](sndr) | The engine. The `PATCH_REGISTRY` + dispatcher, the apply pipeline (text-anchor / class-rebind / middleware patchers), per-engine patch sets (`sndr/engines/vllm/...`), the V2 layered model-config system, the universal launcher, the CLI (`sndr`/`genesis`), and the read-only product API the GUI consumes. This is the only tree the Apache wheel ships. |
| [`gui/`](gui) | The control center — a desktop/web front-end (`gui/web`, `gui/desktop`) that drives the `sndr` product API: launch presets, inspect the live apply summary, browse the patch catalogue, run benches, and manage remote hosts. Built static assets are served by the product API. |
| [`tests/`](tests) | The pytest suite (13k+ collected). Unit tests per subsystem under `tests/unit/...`, contract/bundle/proof tests, and the load-bearing CI gate. Excluded from the wheel. |
| [`docs/`](docs) | All public documentation (USAGE, INSTALL, MODELS, HARDWARE, PATCHES, BENCHMARKS, the pin-bump playbook, anchor SOT, …). `docs/README.md` is the index. |
| [`scripts/`](scripts) + [`tools/`](tools) | Maintainer tooling — the audit gates (`make gates`), doc-sync / link / attribution / drift checkers, anchor-SOT regeneration, bench harnesses, and pin-bump preflight. Not shipped in the wheel. |
| [`third_party/`](third_party) | Vendored upstream kernel source (a curated subset of TurboMind's int4 grouped-MoE GEMM, used by the experimental `G4_85` MoE kernel patch). See [`third_party/tm_int4_moe/README.md`](third_party/tm_int4_moe/README.md) for provenance + license. |
| [`compose/`](compose) | Reference `docker-compose` files for the canonical prod presets (35B / 27B, single- and multi-concurrency, long-context). |
| [`benchmarks/`](benchmarks) + [`evidence/`](evidence) | Bench harness/data and per-patch proof artefacts (`evidence/patch_proof/`) plus the A/B validation evidence the registry cites for default-on/off decisions. |
| [`schemas/`](schemas) + [`plugins/`](plugins) + [`assets/`](assets) + [`release/`](release) | JSON schemas (patch-entry, config), community plugin samples, README/chart/logo assets, and release artefacts (SBOM, constraints). |
| `pyproject.toml` | Single source of truth for packaging **and** all tool config — `[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.mypy]`, and the setuptools package layout. |
| `Makefile` | The maintainer entry point: `make gates` (CI gates), `make test`, `make docs`, `make gui-build`, pin-bump preflight, audits. |

## Contributing

Bug reports, new patches with empirical evidence, new model recipes, and
cross-rig bench reports are all welcome. The full workflow (anchor
conventions, lifecycle ratchet, pin-bump playbook, PR template) is in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). Security disclosures go
through [`SECURITY.md`](SECURITY.md).

## Ecosystem / Related

- **[vLLM](https://github.com/vllm-project/vllm)** — the upstream engine SNDR
  Core patches. Genesis is an overlay, not a fork; patches retire as upstream
  merges the underlying fix.
- **[club-3090](https://github.com/noonghunna/club-3090)** — community,
  multi-engine (vLLM · llama.cpp · ik_llama) serving recipes for consumer GPUs.
  Complementary to this repo and cross-references Genesis in its
  `TQ3_MTP_GENESIS.md`. **Where SNDR Core fits:** it is the deep, single-stack
  vLLM patch engine; club-3090 is the broad multi-engine recipe hub. If you
  want the widest engine/model menu, start there; if you want the fastest,
  most-patched vLLM path on Ampere/Ada/Blackwell, you're in the right place.

## Credits + license

Apache-2.0 (see [`LICENSE`](LICENSE)). Per-patch attribution and upstream
PR linkage in [`docs/CREDITS.md`](docs/CREDITS.md).

Author: Sandermage (Aleksandr Barzov), Odessa, Ukraine.
Sponsorship channels (voluntary, no obligations) and hardware-loan
contact: [`docs/SPONSORS.md`](docs/SPONSORS.md).
