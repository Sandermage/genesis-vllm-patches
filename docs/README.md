# `docs/` — Documentation Map

Public documentation for Genesis vLLM Patches. The top-level
`README.md` covers the headline project description; this folder
holds every operator-facing and contributor-facing reference.
Internal planning notes live in a gitignored sibling directory and
never ship publicly.

## Start here

| If you want to... | Read |
| --- | --- |
| **Brand-new** — orient, then clone to first token | [`GETTING_STARTED.md`](GETTING_STARTED.md) |
| New to local AI itself (hardware / engines / quants, plain English) | [`LOCAL_AI_PRIMER.md`](LOCAL_AI_PRIMER.md) |
| Weigh self-host vs a cloud API (cost crossover) | [`COMPARISONS.md`](COMPARISONS.md) |
| Single-page operator manual covering all four layers (installer / launcher / configs / patches) | [`USAGE.md`](USAGE.md) |
| Install Genesis end-to-end | [`INSTALL.md`](INSTALL.md) → [`QUICKSTART.md`](QUICKSTART.md) |
| Get running in 5 minutes + Day 1 acceptance | [`QUICKSTART.md`](QUICKSTART.md) |
| Browse all `sndr` commands | [`CLI_REFERENCE.md`](CLI_REFERENCE.md) |
| Pick a model + hardware combo | [`MODELS.md`](MODELS.md) + [`HARDWARE.md`](HARDWARE.md) |
| Run on a single 3090 / 4090 (consumer GPU) | [`SINGLE_CARD.md`](SINGLE_CARD.md) |
| Tune an env-var flag | [`CONFIGURATION.md`](CONFIGURATION.md) |
| Add a model recipe or contribute a community config | [`MODELS.md`](MODELS.md) |
| Write a new patch or community plugin | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Diagnose an OOM, cliff, or boot failure | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| Roll a broken release back | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| See current bench numbers + reproduce them | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Browse the patch catalogue + compatibility matrix | [`PATCHES.md`](PATCHES.md) |
| Read the technical design appendices (PN95, GDN, ...) | [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md) |

## File catalogue (30 markdown files)

### Onboarding & concepts

| Doc | Purpose |
| --- | --- |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Two-minute orientation for newcomers — who it's for, what you get, the one install line, and where to go next. |
| [`LOCAL_AI_PRIMER.md`](LOCAL_AI_PRIMER.md) | Plain-English explainer of local AI — GPU/VRAM, inference engine, model size & MoE, quantization — and how SNDR Core fits. |
| [`COMPARISONS.md`](COMPARISONS.md) | Honest self-host-vs-cloud trade and the cost-crossover shape. |

### Installation & quickstart

| Doc | Purpose |
| --- | --- |
| [`USAGE.md`](USAGE.md) | Single-page operator manual that threads through all four layers (installer → launcher → configs → patches) plus the production-readiness bench-proof workflow. |
| [`INSTALL.md`](INSTALL.md) | Full installer walkthrough — `install.sh` flags, preflight checks, troubleshooting. |
| [`QUICKSTART.md`](QUICKSTART.md) | 5-minute setup path plus the 6-step Day-1 acceptance walkthrough. |

### Command + configuration reference

| Doc | Purpose |
| --- | --- |
| [`CLI_REFERENCE.md`](CLI_REFERENCE.md) | Complete `sndr` CLI surface, ordered by operator workflow. Cheatsheet at the top + per-subcommand reference with stability badges + detailed `sndr self-test` section. |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Every Genesis env var — defaults, ranges, which patch reads it. |
| [`MODEL_CONFIG_LAUNCHER.md`](MODEL_CONFIG_LAUNCHER.md) | `sndr model-config` schema + launcher commands. |

### Models, hardware, configs

| Doc | Purpose |
| --- | --- |
| [`MODELS.md`](MODELS.md) | Tested models (Qwen3.6 lineup) + tested alternatives + adding your own model + the V2 layered config system + community config submission pipeline. |
| [`CONFIGS.md`](CONFIGS.md) | Narrative "I want to add a model" recipe. |
| [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md) | Auto-generated full config inventory (regenerated from `model_configs/builtin/*.yaml`). |
| [`HARDWARE.md`](HARDWARE.md) | Tested GPU envelope (A5000, 3090, 4090, 5090, H100, ...) + cross-rig validators. |
| [`SINGLE_CARD.md`](SINGLE_CARD.md) | Single-card (3090/4090) operator guide — honest cliff story, `sndr preflight`, and the cliff-immune escape hatches (llama.cpp MTP, ik_llama two-stage) for consumer GPUs. Credits club-3090. |

### Patches + dispatcher

| Doc | Purpose |
| --- | --- |
| [`PATCHES.md`](PATCHES.md) | Curated narrative reference + compatibility matrix per patch per model + patch_plan resolver (`--policy compat \| safe \| minimal`). |
| [`PATCHES_AUTO.md`](PATCHES_AUTO.md) | Auto-generated full patch table from `dispatcher/registry.py`. |
| [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md) | Technical design appendices: PN95 tier-aware KV cache · GDN kernel fusion roadmap · Qwen3 reasoning/content streaming contract · `Genesis → sndr_core` v11 rename. |
| [`RELEASE_POLICY.md`](RELEASE_POLICY.md) | Which patch-proof mode gates a public release (`require-static` today + two stricter ratchets). |

### Pin management

| Doc | Purpose |
| --- | --- |
| [`PIN_BUMP_PLAYBOOK.md`](PIN_BUMP_PLAYBOOK.md) | **Canonical** end-to-end vLLM pin-bump procedure: candidate extraction → preflight verdicts → fix-drifts → iron-rule-#11 deep-diff → throwaway-container smoke → tokenizer-fingerprint gate → canonical bench → promotion → tag rotation. |
| [`ANCHOR_SOT.md`](ANCHOR_SOT.md) | The per-pin anchor source-of-truth + bump tooling manual: `sndr/engines/vllm/pins/<pin>/` manifests, `make rebuild-pin` / `audit-pin` / `bump-preflight` / `summarize-rej`, and what the bump-gate detects (silent perf-landmine class). |
| [`guides/PIN_UPGRADE.md`](guides/PIN_UPGRADE.md) | Short pin-**policy** summary (≤2-pin, no-proactive-pull, validate-before-promote) + universal launcher template. Points to the playbook + anchor-SOT manual. |

### Benchmarks + troubleshooting

| Doc | Purpose |
| --- | --- |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Canonical PROD numbers (v12.0.0 current registry) + methodology + 5 scenarios + bench command reference + result-sharing rules + interpretation of every metric. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Quick triage → named cliffs (1–8) → OOM recipes → operational cookbook (10 named scenarios) → rollback playbook (R-001…R-008). |
| [`FAQ.md`](FAQ.md) | Common operator questions (registry size, default-on subset, LoRA, streaming, `--from-running`, k8s/proxmox lifecycle). |

### Contributing + glossary

| Doc | Purpose |
| --- | --- |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to author a new patch + community plugin paths (pip-installable and in-repo SDK) + project map (where every script / module / test lives). |
| [`GLOSSARY.md`](GLOSSARY.md) | Term definitions — TQ, MTP, GDN, FA2, A3B, Marlin, ... |
| [`ANNOUNCEMENT_TEMPLATE.md`](ANNOUNCEMENT_TEMPLATE.md) | Reusable skeleton for a "we shipped X" post + r/LocalLLaMA / Show HN / X variants. |

### Credits

| Doc | Purpose |
| --- | --- |
| [`CREDITS.md`](CREDITS.md) | Per-patch attribution + upstream-PR linkage. |
| [`SPONSORS.md`](SPONSORS.md) | Hardware + compute sponsors. |

## Subdirectories

| Folder | What's inside |
| --- | --- |
| [`img/`](img/) | Diagrams referenced from the narrative docs (DFlash vs MTP, patch impact, per-config perf). |

The previous `reference/`, `security/`, `upstream/`, and
`upstream_refs/` subdirectories were retired in the 2026-05-16
consolidation pass:

- Historical experiment / deferred-work logs from 2026-04-27
  (`reference/`) were preserved in git history and removed from
  the public surface — they were operator-irrelevant snapshots.
- The Ed25519 trust-anchor ceremony (`security/`) was an internal
  artefact for a future license-tier that does not gate any
  public-core functionality.
- `upstream/STABLE_PROMOTION_CHECKLIST.md` was merged into
  [`CONTRIBUTING.md` § "Promoting a patch to `lifecycle=stable`"](CONTRIBUTING.md).
- `upstream/UPSTREAM_WATCHLIST.yaml` (consumed by
  `scripts/audit_upstream_watchlist.py`) moved to
  [`tools/upstream_watchlist.yaml`](../tools/upstream_watchlist.yaml)
  because it is a data file, not narrative documentation.
- `upstream/VLLM_PR_DECISION.md` was a historical 2026-04-26
  decision document preserved in git history.
- `upstream_refs/` (frozen vLLM source snapshots used as
  text-patch anchor references) was maintainer-side reference
  material with no operator-facing value; moved off the public
  tree entirely. Contributors authoring a new text-patch should
  pull upstream code directly via
  `git show vllm-project/vllm:<path>@<sha>` instead of reading
  pre-saved snapshots.

## Top-level repo docs (one folder up)

| Doc | Purpose |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, quick install, hardware tested, architecture. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Per-version changelog (technical, deep — single source of truth). |
| [`../SECURITY.md`](../SECURITY.md) | Security policy + private disclosure email. |
| [`../LICENSE`](../LICENSE) | Apache-2.0. |

Contributor onboarding lives at [`CONTRIBUTING.md`](CONTRIBUTING.md) (this `docs/` folder, listed above).

## Auto-generated content

Two files are regenerated from canonical sources by CI gates:

- [`PATCHES_AUTO.md`](PATCHES_AUTO.md) —
  `python3 scripts/generate_patches_md.py`
- [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md) —
  `python3 scripts/generate_configs_md.py`

Do not edit these by hand — the `--check` mode of each generator
gates pull requests. The matching narrative files
([`PATCHES.md`](PATCHES.md), [`CONFIGS.md`](CONFIGS.md)) capture
the explanations that don't fit in a machine-readable table.

## Consolidation history

The previous structure had 38 markdown files. After the 2026-05-16
consolidation pass, sibling topics were merged into single
operator-facing references so the surface matches industry-standard
docs/ layouts (vllm-project/vllm, sgl-project/sglang,
huggingface/transformers all keep ≤ 15 narrative .md files in
their main docs tree). The current 26 files map to the previous
38 as follows:

- `QUICKSTART.md` ← `QUICKSTART.md` (rewritten) + `DAY_1_CHECKLIST.md`.
- `CLI_REFERENCE.md` ← `CLI_REFERENCE.md` + `COMMANDS.md` + `SELF_TEST.md`.
- `MODELS.md` ← `MODELS.md` + `CONFIGS_FOR_COMMUNITY.md` + `CONFIG_SYSTEM_V2.md`.
- `PATCHES.md` ← `PATCHES.md` + `COMPATIBILITY.md` + `PATCH_PLAN.md`.
- `PATCH_DESIGNS.md` ← `GDN_KERNEL_FUSION_DESIGN.md` + `PATH_C_TIER_AWARE_KV_CACHE.md` + `REASONING_CONTENT_CONTRACT.md` + `MIGRATION_V11_RENAME.md`.
- `CONTRIBUTING.md` ← `CONTRIBUTING.md` + `PLUGINS.md` + `COMMUNITY_PATCHES.md` + `PROJECT_MAP.md`.
- `BENCHMARKS.md` ← `BENCHMARKS.md` + `BENCHMARK_GUIDE.md`.
- `TROUBLESHOOTING.md` (new) ← `CLIFFS.md` + `OOM_RECIPES.md` + `COOKBOOK.md` + `ROLLBACK_PLAYBOOK.md`.

Plus 4 dated Russian-language audit files removed from
`docs/upstream/` (preserved in git history).
