# `docs/` — Documentation Map

Public documentation for Genesis vLLM Patches. The top-level `README.md`
covers the headline project description; this folder holds every
operator-facing and contributor-facing reference. Internal planning
notes live in a gitignored sibling directory and never ship publicly.

## Start here

| If you want to... | Read |
| --- | --- |
| Install Genesis end-to-end | [`INSTALL.md`](INSTALL.md) → [`QUICKSTART.md`](QUICKSTART.md) |
| Get running in 5 minutes | [`QUICKSTART.md`](QUICKSTART.md) |
| Walk the first-30-minutes checklist | [`DAY_1_CHECKLIST.md`](DAY_1_CHECKLIST.md) |
| Browse all `sndr` commands | [`CLI_REFERENCE.md`](CLI_REFERENCE.md) (full) · [`COMMANDS.md`](COMMANDS.md) (cheatsheet) |
| Pick a model + hardware combo | [`MODELS.md`](MODELS.md) + [`HARDWARE.md`](HARDWARE.md) |
| Look up a single command | [`CLI_REFERENCE.md`](CLI_REFERENCE.md) |
| Tune an env-var flag | [`CONFIGURATION.md`](CONFIGURATION.md) |
| Add a new model recipe | [`CONFIGS.md`](CONFIGS.md) → [`CONFIGS_FOR_COMMUNITY.md`](CONFIGS_FOR_COMMUNITY.md) |
| Write a new patch | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Ship a community patch (pip-installable) | [`PLUGINS.md`](PLUGINS.md) |
| Ship a community patch (in-repo SDK) | [`COMMUNITY_PATCHES.md`](COMMUNITY_PATCHES.md) |
| Diagnose an OOM / cliff | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) → [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| Roll a broken release back | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |

## Reference catalogue

### Installation & quickstart

| Doc | Purpose |
| --- | --- |
| [`INSTALL.md`](INSTALL.md) | Full installer walkthrough — `install.sh` flags, preflight checks, troubleshooting. |
| [`QUICKSTART.md`](QUICKSTART.md) | 5-minute path: clone → install → launch → smoke test. |
| [`DAY_1_CHECKLIST.md`](DAY_1_CHECKLIST.md) | 6 acceptance steps for a fresh install — what good output looks like. |
| [`SELF_TEST.md`](SELF_TEST.md) | `sndr self-test` — structural sanity check after a `git pull` or pin bump. |
| [`MIGRATION_V11_RENAME.md`](MIGRATION_V11_RENAME.md) | `Genesis → sndr_core` package rename appendix (2026-05-08). |

### Command + configuration reference

| Doc | Purpose |
| --- | --- |
| [`CLI_REFERENCE.md`](CLI_REFERENCE.md) | Complete `sndr` CLI surface grouped by operator workflow. Stability badges per subcommand. |
| [`COMMANDS.md`](COMMANDS.md) | One-page cheatsheet — the top 30 commands ordered "first day on rig → weekly maintenance → deep diagnostic". |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Every Genesis env var — what it does, default, valid range, which patch reads it. |
| [`CONFIG_SYSTEM_V2.md`](CONFIG_SYSTEM_V2.md) | V2 layered model-configuration architecture (model × hardware × profile × preset). |
| [`MODEL_CONFIG_LAUNCHER.md`](MODEL_CONFIG_LAUNCHER.md) | `sndr model-config` schema + launcher commands. |

### Models, hardware, recipes

| Doc | Purpose |
| --- | --- |
| [`MODELS.md`](MODELS.md) | Tested models (Qwen3.6 lineup) + why the defaults were chosen. |
| [`HARDWARE.md`](HARDWARE.md) | Tested GPU envelope (A5000, 3090, 4090, 5090, H100, ...) + cross-rig validators. |
| [`COMPATIBILITY.md`](PATCHES.md) | Patch × model × hardware enable/disable matrix. |
| [`CONFIGS.md`](CONFIGS.md) | Narrative guide — "I want to add my own model" recipe. |
| [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md) | Auto-generated full config inventory (regenerated from `model_configs/builtin/*.yaml`). |
| [`CONFIGS_FOR_COMMUNITY.md`](CONFIGS_FOR_COMMUNITY.md) | Community config submission path + validation rules. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Recipe-style how-tos for production scenarios (symptom → root → workaround → fix). |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Targeted fixes for the most common OOM patterns. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Named behavioural cliffs (OOM patterns, regressions) — the catalogue. |

### Patches + dispatcher

| Doc | Purpose |
| --- | --- |
| [`PATCHES.md`](PATCHES.md) | Curated narrative reference for every patch — what / why / status / env-flag / credit. |
| [`PATCHES_AUTO.md`](PATCHES_AUTO.md) | Auto-generated full patch table from `dispatcher/registry.py`. |
| [`PATCH_PLAN.md`](PATCHES.md) | Resolver policy — `compat` / `safe` / `minimal` semantics. |
| [`PATH_C_TIER_AWARE_KV_CACHE.md`](PATH_C_TIER_AWARE_KV_CACHE.md) | PN95 tier-aware KV cache design. |
| [`GDN_KERNEL_FUSION_DESIGN.md`](GDN_KERNEL_FUSION_DESIGN.md) | GDN kernel fusion deep-dive. |
| [`REASONING_CONTENT_CONTRACT.md`](REASONING_CONTENT_CONTRACT.md) | Qwen3 streaming reasoning-content semantics. |
| [`RELEASE_POLICY.md`](RELEASE_POLICY.md) | Which patch-proof mode gates a public release (`require-static` today). |

### Benchmarks + verification

| Doc | Purpose |
| --- | --- |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Canonical PROD bench numbers (Wave 10, 2026-05-15). |
| [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md) | `sndr bench` methodology + reproduction recipes. |

### Contributing

| Doc | Purpose |
| --- | --- |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to author a new patch — anchors, lifecycle, PR workflow. |
| [`PLUGINS.md`](PLUGINS.md) | Ship a patch as a separate pip-installable plugin (entry-points). |
| [`COMMUNITY_PATCHES.md`](COMMUNITY_PATCHES.md) | In-repo community SDK — `sndr community new-patch` workflow. |
| [`PROJECT_MAP.md`](PROJECT_MAP.md) | Where every script / module / test lives. |
| [`GLOSSARY.md`](GLOSSARY.md) | Term definitions — TQ, MTP, GDN, FA2, A3B, Marlin, ... |

### Operations

| Doc | Purpose |
| --- | --- |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Production rollback procedure when a patch regresses. |
| [`FAQ.md`](FAQ.md) | Common operator questions (registry size, default-on subset, LoRA, streaming, ...). |

### Credits

| Doc | Purpose |
| --- | --- |
| [`CREDITS.md`](CREDITS.md) | Per-patch attribution + upstream-PR linkage. |
| [`SPONSORS.md`](SPONSORS.md) | Hardware + compute sponsors. |

## Subdirectories

| Folder | What's inside |
| --- | --- |
| [`reference/`](reference/) | Long-form deferred-PR notes + cross-rig validation logs (DEFERRED_P50_DEPLOY, V758/V759 historical points). |
| [`security/`](security/) | Ed25519 trust-anchor ceremony for release artifact signing. |
| [`upstream/`](upstream/) | vLLM upstream watchlist + PR decision logs + production roadmap. |
| [`upstream_refs/`](upstream_refs/) | Frozen upstream source snapshots used as anchor references for text patches. |
| [`img/`](img/) | Diagrams referenced from the narrative docs (DFlash vs MTP, patch impact, per-config perf). |

## Top-level repo docs (one folder up)

| Doc | Purpose |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, quick install, hardware tested, architecture. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Per-version changelog (technical, deep — single source of truth). |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Top-level contributor onboarding (links here). |
| [`../LICENSE`](../LICENSE) | Apache-2.0. |

## Auto-generated content

Two files are regenerated from canonical sources by CI gates:

- [`PATCHES_AUTO.md`](PATCHES_AUTO.md) — `python3 scripts/generate_patches_md.py`
- [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md) — `python3 scripts/generate_configs_md.py`

Do not edit these by hand — the `--check` mode of each generator
gates pull requests. The matching narrative files
([`PATCHES.md`](PATCHES.md), [`CONFIGS.md`](CONFIGS.md)) capture
the explanations that don't fit in a machine-readable table.
