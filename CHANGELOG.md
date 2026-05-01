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

## [Unreleased] — `v7.65` series

> Pin: `0.20.1rc1.dev16+g7a1eb8ac2` (committed 2026-04-28).
> Series builds on top of v7.64 release and adds 4 new patches +
> infrastructure hardening, all opt-in OFF by default. No
> regression in PROD bench; pushed to `dev` branch only (main promotion
> deferred until cross-rig validation).

### Added — new patches

- **PN21 — DFlash SWA partial backport** (`vllm#40898`, opt-in OFF).
  Two of three sub-files of jianc99's PR backported: `algos.py`
  (preserve `layer_types` / `use_sliding_window` / `sliding_window` /
  `max_window_layers` from speculators-format checkpoint into HF
  config) + `dflash.py` (force `causal=True` for sliding-window
  layer attention metadata). The `qwen3_dflash.py` model-class
  changes (7+ sub-patches across `Attention.__init__` / `DecoderLayer.__init__`
  / Model class) NOT backported — too fragile for text-patch.
  Empirical on 35B-A3B-FP8-DFlash 160K (3-run sweep):
  `5-6/7` tool-call with PN21 ON vs `7/7` baseline OFF.
  Without the model-side changes, config preserves SWA but the model
  still constructs full attention → metadata/compute mismatch shifts
  spec acceptance. **Default OFF, NOT enabled in any launch script**
  until upstream merges or full manual model-class backport.

- **PN25 — SiluAndMul.forward_native opaque-op pool** (Genesis-original,
  opt-in OFF). Sister-patch to PN12; complement, not replacement.
  PN12 patches `forward_cuda` (eager dispatch) but
  `custom_ops=["none"]` (V1 default for `aot_compile_fullgraph`)
  routes through `forward_native` which Inductor inlines and lowers
  to `empty_strided_cuda(...)` — completely bypassing PN12's pool.
  Reported by noonghunna in `club-3090#16` (VolandBerlioz Reddit + ampersandru
  cross-rig: RTX 3090 24 GB + Lorbus 27B + OpenCode 29K-token prefill
  OOMs at 137.6 MiB). PN25 registers `genesis::silu_and_mul_pooled`
  as `torch.library.custom_op` with `mutates_args=()` and
  `device_types=("cuda",)`. Inductor treats opaque ops as no-inline.
  Body acquires from `FFNIntermediateCache` pool (same one PN12
  uses) and dispatches to `torch.ops._C.silu_and_mul`.
  PN12 + PN25 patch DIFFERENT methods so anchors never collide;
  pool is shared singleton. Recommended pairing for any
  inductor-heavy config; standalone use covers single-path setups.

- **PN26 — TQ unified perf pack** (Genesis-original combining 3
  upstream OPEN PRs from jasonkim8652, opt-in OFF):
  - **Taken from #41418** (centroids prebake): pre-baked Lloyd-Max
    centroid tables for `(d=128, bits=4 / 8 / 3)` — covers our PROD
    presets `turboquant_4bit_nc` / `turboquant_k8v4` / `turboquant_3bit_nc`.
    Empirical on live container: `(128, 8)` `0.018ms` vs solver
    `4583.9ms` = **259,812× speedup** on cold boot.
  - **Genesis defensive addition** vs upstream: at first use, runs
    `prebaked == solver` self-check for `(128, 4)`. On drift > 1e-3
    (real Lloyd-Max algorithm change upstream), auto-disables
    prebake and falls through to runtime solver with a WARNING.
    On 1e-6 drift (round-noise from int/1e10 encoding), logs INFO
    and keeps prebake. Threshold gates against silent staleness.
  - **Taken from #41422 (scaffold-only)**: sparse V tile-skip kernel
    modification. Author validated AMD MI300X only; NVIDIA Ampere
    correctness needs empirical confirmation. Ships as scaffold
    gated by `GENESIS_ENABLE_PN26_SPARSE_V=1` sub-flag; actual
    kernel wiring deferred to next iteration.
  - **Dropped from #41414**: head_dim power-of-2 padding. Qwen3.6
    head_dim=128 is already pow-2; the patch would add a runtime
    branch that is dead code on our model.

- **PN27 — Revert MoERunnerInterface PluggableLayer** (`vllm#41440`
  backport, proactive scaffold, opt-in OFF). Reverts vllm#35178
  (commit `b55b2652`, merged 2026-04-30) which made
  `MoERunnerInterface` inherit from `PluggableLayer` for OOT support.
  Issue #41306 reports v0.20 MoE perf regression: Mixtral-8x7B
  TPOT +21%, TTFT +59%, throughput -19% (8× H200). bnellnm (vLLM core)
  confirmed `--moe-backend=triton` restores v0.19 perf.
  **Our pin `g7a1eb8ac2` was committed 2026-04-28 — 2 days BEFORE
  #35178 merged.** So we are accidentally pre-#35178 and NOT
  vulnerable. PN27 is a **proactive scaffold**: when we eventually
  pin-bump past `b55b2652` BEFORE upstream's #41440 (or equivalent)
  merges, all 3 sub-patches engage and revert to pre-regression
  behavior. On our current pin, all sub-patches SKIP cleanly.

### Added — infrastructure

- **Cliff 8 hardening** (`apply_all.py`) — new
  `PatchStats.partial_apply_warnings` property surfaces skipped
  patches whose reason indicates real anchor drift / ambiguous-anchor /
  required-anchor-missing — distinct from benign skips
  (opt-in OFF, upstream-merged, platform mismatch, deferred,
  redundant). Boot summary line now appends
  `N ⚠️  partial-apply warning(s)` when count is non-zero, plus
  per-warning WARNING-level lines that name each patch + reason.
  Promised to noonghunna in `club-3090` discussion #19. First
  detection in PROD: PN9 self-retire on 27B PROD boot
  (`'spec_cfg.attention_backend' present in llm_base_proposer.py`)
  — manually verified PR #39930 + DFlashProposer `use_non_causal=True`
  is full superset of our partial backport; self-retire correct.

### Changed

- **A2 — P68/P69 long-context threshold default 8000 → 50000 chars**
  (Issue #9). Old 8000-char default (~2K tokens) was too aggressive —
  triggered P68 force-tool-choice and P69 explicit-format-reminder on
  routine IDE-agent flows that are NOT genuinely long-context. New
  50000-char default (~12.5K tokens) keeps the behavior for genuine
  long histories. Code default updated; 6 active launch scripts
  updated to override `8000 → 50000` explicitly.

- **CLIFFS.md PN19 H100-only flag** (Cliff 1 mech A section).
  noonghunna 2026-05-01 confirmed PN19 costs ~120 MiB KV pool on a
  24 GB single-3090 (vs documented 200-500 MiB win on H100). Disable
  PN19 on 24 GB consumer cards (3090, 4090, A5000) running long
  context. Same lesson as P104 L2 persistence — generic allocator
  hints don't survive GPU class boundaries.

### Verified (no regression)

- **#41190 stress test** — TP=2 + spec-decode + first-request
  `cudaErrorIllegalAddress` reported by their RTX 6000 Ada / AWQ /
  WIP-PR-#40898-build setup. Stress-tested on our 35B DFlash 160K
  (TP=2 + DFlash spec K=3): 5 concurrent + 30 sequential rapid-fire
  chat completions. **ZERO `cudaError`**, zero `illegal memory
  access`, zero `watchdog` events. Differences:
  they used QuantTrio AWQ (online-quant), we use FP8 (offline);
  their pin built off PR #40898 head (WIP), our pin on main.
  Possibly P58 (async scheduler placeholder) or P60 (GDN+ngram)
  defends against the codepath.

- **#41306 MoE regression** — verified via runtime probe that our
  installed `MoERunnerInterface.__bases__ == (<class 'abc.ABC'>,)`
  (no PluggableLayer inheritance). Our pin pre-dates #35178 by 2
  days; we are NOT vulnerable. PN27 scaffold ready when we pin-bump.

### PN26b sparse-V kernel — major iteration (v5, 2026-05-01)

Comprehensive deep-dive on Genesis-original sparse-V Triton kernel based
on 4-agent research synthesis (skip-rate observability + per-row vote +
memory profiling + 14-day community scan).

**v5 design** (lean dispatcher + tuning + observability):

- **Lean dispatcher** (no per-call GPU↔CPU sync; v1's `.item()` per call
  caused -16% short-ctx + -22% long-ctx regression — REJECTED).
- **Configurable launch params** baked at apply() time: BLOCK_KV (4/8/16),
  num_warps (1/2/4/8), num_stages.
- **`tl.range()` pipelining hint** (P67 v7.50 pattern, Triton compiler
  cp.async overlap with prior-iter MMA on Ampere).
- **Cache modifier `.cg`** on K/V dequant raw loads (L2 streaming).
- **Sink-token protection** (StreamingLLM finding — first 4 KV positions
  never skipped).
- **Skip-rate observability** (NEW): per-CTA atomic int64 counters,
  constexpr-DCE'd to zero overhead when disabled, `~50-100 ns` per CTA
  at epilogue when enabled. Periodic logging every 500 calls so
  operator sees real skip rate without cross-process IPC.
- **BLASST adaptive threshold scaffold** (`λ = scale_factor / ctx_len`)
  ready in code; default OFF until skip-rate data informs which mode
  is better.

**Empirical sweep on 35B FP8 PROD (TQ k8v4 + MTP K=3, 2× A5000 SM86)**:

| BLOCK_KV | num_warps | mean | max | CV |
|---|---|---|---|---|
| OFF (baseline) | — | 175.41 | 185.15 | 4.20% |
| 8 | 1 | 178.33 | 187.67 | 3.78% |
| 8 | 2 | 180.36 | 190.24 | 4.70% |
| 16 | 2 | 178.35 | 190.74 | 3.26% |
| 8 | 4 | 183.11 | 202.38 | 5.26% |
| 8 | 8 | 181.24 | 196.60 | 5.78% |
| **4** | **4** | **184.89** | 194.56 | 4.63% |
| 4 | 8 | 177.40 | 191.97 | 5.79% |

Winner: **BLOCK_KV=4, num_warps=4** (baked as kernel default).

**Final 35B PROD A/B (apples to apples, 100t output)**:

| Config              | tool-call | mean   | min   | max    | CV    |
|---------------------|-----------|--------|-------|--------|-------|
| Baseline (OFF)      | 7/7       | 175.41 | 158.71| 185.15 | 4.20% |
| **PN26b v5**        | **7/7**   | **182.30** | 153.53 | **212.24** | 7.02% |
| Δ                   | match     | **+3.9%** | -3.3% | **+14.7%** ⭐ | +2.82pp |

The `212 max` exceeds the historical 35B PROD ceiling reference (171-204
TPS quoted from earlier sessions). Tool-call quality preserved (7/7).
Sustained 50-request load: 0 errors, p50=181, p90=197, p99=211. VRAM
delta +142 MiB (acceptable, no leak).

**Caveat**: skip rate at threshold=0.005 is empirically very low on our
short-output workload (most TPS gain comes from kernel restructuring,
not the skip itself). Skip-rate counter scaffold ships so future
operators can data-drive their threshold tuning. Long-context (>16K
input) deeper sweep deferred to next session — needs sustained-context
workload to characterize properly.

### Bench results — `v7.65` PROD eligibility

35B FP8 DFlash 160K (TP=2 + DFlash spec K=3 + PN22+PN23+PN24):

- 44 patches applied / 0 failed / 0 partial-apply warnings
- prose 256t mean **125.07 TPS, CV 3.07%**
- tool-call 5-7/7 (variance band)

27B Lorbus INT4 PROD (TQ k8v4 + MTP K=3 + 8 baked patches):

- 54 patches applied / 0 failed / 1 partial-apply warning (PN9
  self-retire — verified correct, upstream is strict superset)
- tool-call **7/7**
- prose 256t mean **88.39 TPS, CV 2.59%**
- code  512t mean **104.25 TPS, CV 0.20%**

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
