# `docs/` — Documentation Map

This folder holds extended documentation that doesn't belong in the top-level READMEs.

## Operator reference pages

| Page | Purpose |
|---|---|
| [`SELF_TEST.md`](SELF_TEST.md) | `genesis self-test` — structural sanity check after `git pull` / pin bump |
| [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md) | `genesis bench` methodology and reproduction recipes |
| [`PLUGINS.md`](PLUGINS.md) | Authoring and shipping a community plugin patch |

## Subdirectories

| Folder | What's inside |
|---|---|
| `_internal/sprint_reports/` (gitignored) | Time-stamped engineering sprint reports (Russian) — internal-only per memory rule `feedback_no_internal_docs_in_public`. Local on Sander's machine; not shipped publicly. |
| [`reference/`](reference/) | Long-form technical references (memory architecture, bot setup, etc.). Stable docs that don't get updated per-sprint. |
| [`upstream/`](upstream/) | Drafts and decisions related to upstream vLLM PRs (review notes, decision logs). |

## Top-level docs (one folder up)

| Doc | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview, quick install, hardware tested, architecture |
| [`../INSTALL.md`](../INSTALL.md) | Detailed installation instructions |
| [`../QUICKSTART.md`](../QUICKSTART.md) | Get running in 5 minutes |
| [`../CONFIGURATION.md`](../CONFIGURATION.md) | Every env var documented |
| [`../PATCHES.md`](../PATCHES.md) | All 48 PATCH_REGISTRY patches × metadata × credits |
| [`../CREDITS.md`](../CREDITS.md) | Comprehensive attribution log |
| [`../MODELS.md`](../MODELS.md) | Tested model configurations |
| [`../SPONSORS.md`](../SPONSORS.md) | Hardware / time sponsors |
| [`../vllm/_genesis/CHANGELOG.md`](../vllm/_genesis/CHANGELOG.md) | Per-version changelog (technical, deep) |
