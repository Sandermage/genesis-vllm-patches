# Changelog

All notable changes to **Genesis vLLM Patches** are tracked here.

This is the public-facing release log. The exhaustive engineering log
(per-commit, per-patch decisions, per-A/B numbers) lives in
[`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) — 2300+ lines.

The project uses [Semantic-ish Versioning](https://semver.org/) keyed
to internal sprints (`v7.62.x` etc). Until a 1.0 cut, expect
breaking changes only when an upstream vLLM PR replaces a Genesis
patch and the patch retires accordingly — those are flagged
loud-and-clear in the per-release notes.

---

## [Unreleased] — `v7.63.x` series

> 50 commits ahead of `origin/main` at time of writing. Local-only
> until Sander explicitly green-lights a GitHub push. Run on PROD
> (server `192.168.1.10`, 2× RTX A5000) since 2026-04-29.

### Added

- **Genesis Compat Layer** (`vllm/_genesis/compat/`) — discovery and
  diagnostics package + 16-subcommand unified CLI:
  - `genesis doctor` — single-shot diagnostic (hardware + software +
    model + patches + lifecycle + dispatcher validator). Emits text
    or JSON.
  - `genesis init` — interactive first-run wizard (detect hardware →
    pick model → workload preference → generate launch script).
  - `genesis explain <patch>` — per-patch deep-dive
    (applies_to predicate, lifecycle state, upstream PR, recommendation).
  - `genesis list-models` / `genesis pull <key>` — curated 5-model
    registry; `pull` downloads the weights and writes a tailored
    launch script that engages the right Genesis patches for the
    hardware × quant combination.
  - `genesis lifecycle-audit` — CI-ready check that every entry in
    `PATCH_REGISTRY` has a known lifecycle state (exit 1 on
    `experimentl` / `retried` typos).
  - `genesis validate-schema` — shape-validates `PATCH_REGISTRY`
    (env-flag prefix, required fields, `applies_to` predicate
    well-formedness, dependency graph).
  - `genesis categories` — browse patches by category.
  - `genesis migrate <vllm-clone> --out runbook.md` — pin-bump
    runbook generator: scans your upstream vLLM checkout, flags every
    Genesis text-anchor that drifted, suggests retirement candidates.
  - `genesis recipe save / load / share / diff / adopt` — capture
    launch configurations, share them by URL, A/B them.
  - `genesis plugins` — community plugin entry-points
    (`GENESIS_ALLOW_PLUGINS=1` opt-in; HTTPS-validated).
  - `genesis telemetry` — opt-in anonymized stats; default OFF.
  - `genesis update-channel status / check / set` — apt-style
    stable / beta / dev channels.
  - `genesis self-test` — operator-facing structural sanity check
    (post-`git pull` / pin-bump). Same gate CI runs.
  - `genesis bench` — wraps the full benchmark suite under a unified
    entry point.
- **`applies_to` predicate DSL** with AND / OR / NOT trees, version-
  range matching for vllm / torch / cuda / triton / driver / compute
  capability. Backwards-compatible with all 50 existing flat-dict
  registry entries.
- **Patch lifecycle state machine** — `experimental` / `stable` /
  `deprecated` / `research` / `community` / `retired`. Code removal
  blocked until lifecycle state is `retired`.
- **A3/D2 dispatcher validator** — boots fail loud on
  `requires_patches` / `conflicts_with` referencing unknown patch
  IDs. Caught two real PROD-config issues at first run.
- **Reference benchmark fingerprints** in
  `vllm/_genesis/compat/fingerprints/` — blessed numbers per
  hardware × model × patch-set; bench tool can compare a fresh
  run against a fingerprint.
- **D1 CI upstream drift watcher** — daily GitHub Actions cron
  diffs Genesis text-anchors against `vllm-project/vllm@main` and
  flags newly-merged PRs that allow Genesis self-retirement.
- **JSON-Schema for `PATCH_REGISTRY`** at `schemas/patch_registry.json`
  + `genesis validate-schema`.
- **Pre-commit hook** at `scripts/git/pre-commit` — runs schema +
  A3/D2 + lifecycle audit + self-test before every commit. Install
  via `bash scripts/git/install.sh`.
- **Genesis CI workflow** (`.github/workflows/test.yml`) — 492-test
  scoped CI gate on Python 3.10 + 3.12. The full session test
  surface is **1351 tests / 70 skipped / 0 failed** as of this
  release.
- **Canonical `__version__` constant** at
  `vllm/_genesis/__version__.py` with `__commit__` and `__channel__`.
- **`scripts/git/`** — pre-commit hook + installer.
- **`docs/upstream_refs/`** — historical upstream PR diff studies
  (moved out of the root `reference/` directory by Phase 2.2).

### Patches added

| Patch | What | Status |
|---|---|---|
| `PN14` | TQ decode IOOB safe_page_idx clamp (vllm#40074 backport) | opt-in, validated |
| `PN16` | Lazy-reasoner request hook (Genesis-original) | opt-in, validated |
| `PN17` | FA2 softmax_lse runtime clamp (Genesis Issue #11 fix) | opt-in, validated |
| `PN19` | Scoped `max_split_size_mb` during model load (vllm#41268 backport) | opt-in, validated |

### Patches retired / annotated

- `P5` auto-retire when JartX vllm#39931 merges (TurboQuant hybrid)
- `P82` drift markers for vllm#40819
- `P94` superseded-on-merge by vllm#41043
- `P98` deliberate inverse of merged vllm#40941 (still required on
  hybrid GDN + TQ k8v4 path; documented why)

### Repository structure

- **Phase 2.1 wiring reorg** — `vllm/_genesis/wiring/` regrouped
  into 9 category subdirectories
  (`spec_decode/`, `structured_output/`, `kv_cache/`, `kernels/`,
  `hybrid/`, `middleware/`, `perf_hotfix/`, `compile_safety/`,
  `legacy/`). Layout-agnostic resolution via `rglob` + computed
  dotted paths. No callsite churn.
- **Phase 2.2 root cleanup** — 29 → 21 root entries.
  `docker-compose.*.yml` × 7 moved to `compose/`,
  `validate_*.sh` × 2 moved to `scripts/`, upstream PR diff studies
  moved to `docs/upstream_refs/`. Doc cross-references updated
  (35+ replacements across README / INSTALL / QUICKSTART / MODELS /
  BENCHMARK_GUIDE).

### Bench upgrades (Genesis Benchmark Suite v2)

`tools/genesis_bench_suite.py` now produces a single rich JSON per
run with the following sections:

- `engine` — vLLM version, system fingerprint, Genesis self-test
  summary, applied patches list.
- `tool_call` — 8-case quality matrix (4 cities × 2 thinking modes;
  positive vs. negative cases scored separately).
- `decode_bench` — N runs × M prompts × max_tokens; per-prompt
  detail + aggregate `wall_TPS`, `decode_TPOT_ms`, `TTFT_ms` with
  median, mean, stddev, CV.
- `multi_turn` — N-turn TTFT with conversation context growing
  per turn.
- `stress` — stability stress: SHA1 drift, NaN sentinel scan,
  repetition detection, TPOT trend, **`STABILITY_VERDICT`**.
- `output_length` — generation capacity probe at 1K..16K target
  lengths, with per-probe VRAM tracking (`vram_before_mib`,
  `vram_after_mib`, `vram_delta_per_gpu_mib`,
  `vram_delta_total_mib`, `verdict`).
- `accept_rate` — Prometheus `/metrics` scrape for spec-decode
  counters (returns gracefully when `--disable-log-stats` is set).
- `vllm_version` — parsed `system_fingerprint` (`vllm_version`,
  `tp`, `commit`).
- `genesis_state` — `--quiet --json` self-test invocation result.

Two new CLI flags:

- `--probe-output-length` — engages section 7 of the run.
- `--scheme http|https` — picks transport for `_build_url()`.
- `--arm-name <name>` — alias for `--name` (A/B compare ergonomics).
- `--compare a.json b.json --compare-out delta.json` — Welch
  t-test + per-percentile delta JSON.

### Docs

- New: `MODELS.md`, `QUICKSTART.md`, `INSTALL.md` (Docker + bare-metal),
  `CONFIGURATION.md`, `docs/BENCHMARK_GUIDE.md`,
  `docs/SELF_TEST.md`, `docs/PLUGINS.md`, `docs/reference/` (operator
  reference per release), `docs/upstream_refs/` (upstream PR diff
  studies), `assets/README.md` (brand asset placement).
- Refactored: `README.md` cross-references updated for Phase 2.1 +
  Phase 2.2 layout.
- This file: `CHANGELOG.md` (root) — concise public release log.

### Tests

The exhaustive (out-of-CI) session test surface is now **1351 / 1391
collected (97% pass rate, 70 skip, 0 fail)**. The 121 drifted
out-of-CI tests rescued in commit `a3a8c8d` covered:

- `test_platform_matrix.py` — 66 tests rewritten for the new
  snapshot-at-load `guards.is_*` constants (no more `cache_clear()`
  calls — those functions are no longer `@functools.cache`).
- `test_v7_14_15_audit.py` — Python 3.13 dataclass introspection
  via `spec_from_file_location` now needs the module in
  `sys.modules` before `exec_module`. Added the bind.
- `test_p51_tq_active.py` — registry shape changed
  (`num_k_buffers` + `num_v_buffers` → unified `total_buffers`);
  logger renamed to `genesis.dequant_buffer`.
- `test_wiring_patch_8.py` — P8 `Issue #5` post-apply import probe
  caused over-defensive skip when `kv_cache_utils.py` returned
  `SKIPPED upstream_merged`. Now the scheduler.py sub-patch carves
  out that case explicitly (helper IS in the file → import will
  succeed).
- `test_p58_async_placeholder_fix.py` — single `SCHED_DRAFT_OLD`
  anchor was split into Site A / Site B in the 2026-04-28 P62-compat
  refactor. Test updated.
- `test_p59_qwen3_reasoning_tool_call_recovery.py` — `RETURN_THINK_OLD`
  similarly split into MONOLITH / MODULAR. Test updated.
- `test_wiring_runtime_rebind.py` — fake `TQAttentionImpl` was
  missing the `_init_turboquant_buffers` sentinel that the upstream-
  drift detector probes for. Added.
- `test_config_detect.py` / `test_model_detect.py` — guarded with
  `pytest.mark.skipif` when `vllm.config` not importable (CPU-only
  / no-vllm envs). Run normally in the integration container.
- Added an autouse `conftest.py` fixture that wipes the central
  `prealloc_budget._CACHED` before/after every test — prevents
  cross-test pollution.

### Production validation

- **27B Lorbus + TQ k8v4 + MTP K=3 + 280K context**: PROD baseline
  on 2× RTX A5000. Reference fingerprint at
  `vllm/_genesis/compat/fingerprints/rtx_a5000_x2_qwen3_6_27b_int4_v794.json`.
- **35B-A3B-FP8 + MTP K=3 + 320K context**: validated reference;
  fingerprint pending (this release).
- **Cross-rig validated**: contributors on RTX 3090 / 4090 / 5090 /
  H20 / R6000 Pro Blackwell / 8× A4000 — see `CREDITS.md`.

---

## Earlier history

For everything before this release, see
[`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md). It tracks
every commit on the engineering side back to v7.0 (the start of the
modular `_genesis/` package).
