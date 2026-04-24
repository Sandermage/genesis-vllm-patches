# Genesis `_genesis/` Package Changelog

## v7.9.0 — 2026-04-24 (runtime architecture-dispatch detection)

**Defense-in-depth layer 2: detect which patches need to fire before work begins.**

### Added

- `model_detect.py` — cached `get_model_profile()` returns `(moe, hybrid, turboquant)`
  - `is_moe_model()` — Qwen3-MoE / Mixtral / DeepSeek / Gemma-4-MoE / architecture + model_type heuristics
  - `is_hybrid_model()` — Qwen3-Next `layer_types`, Mamba, GDN, SSM detection
  - `is_turboquant_active()` — config-level `kv_cache_dtype` check (layer-level is P51 in `dequant_buffer.py`)
  - `log_skip(patch, reason)` — uniform single-line dispatch log format
  - `clear_for_tests()` — cache reset for unit tests
  - Conservative fallback: unknown config → True for all flags (patches still apply, their own guards decide)

- **P51 — TQ-active runtime detection** in `kernels/dequant_buffer.py::ensure_turboquant_buffers`
  - Reads `impl.kv_cache_dtype`; early-returns with single log if non-TurboQuant
  - Saves ~516 MiB / rank on FP16-KV + `auto` deployments where TQ text-patches graceful-skip but preallocs would fire
  - `_p51_logged` flag avoids log spam across all model layers (one log per impl)

- **P52 — MoE-active dispatch gate** wired into `wiring/patch_{24,31,37}_*.py`
  - Skips P24 (MoE num_warps overlay), P31 (grouped-topk fp32 upcast), P37 (intermediate-cache pool) on dense models
  - Single log line per skipped patch at apply time; no runtime overhead thereafter

- **P53 — Hybrid-active dispatch gate** wired into `wiring/patch_{28,34,39,46}_*.py`
  - Skips P28 (GDN core-attn rewire), P34 (Mamba zero-collapse guard), P39a (FLA kkt pool), P46 (GDN gating pool) on pure-attention models
  - All targets still graceful-skip without P53 (their text-patch anchors wouldn't match), but the dispatch log now explains *why*

- `tests/test_model_detect.py` — 19 tests covering MoE detection across architectures, hybrid detection, TQ detection, conservative fallback, caching, log helper
- `tests/test_p51_tq_active.py` — 8 tests covering fp8/auto/fp16 skip, single-log-per-impl, legacy-impl backward compat, TQ-active passthrough

### Changed

- `kernels/dequant_buffer.py::ensure_turboquant_buffers` now early-returns on non-TQ impls before any config resolution work
- Wiring apply() docstrings updated to reference P52/P53 gates where applicable
- Root `README.md` rewritten for v7.9 with compatibility matrix, installation guide, patch roster, upstream tracking

### Upstream correspondence

Re-audit of `vllm-project/vllm` since 2026-04-24 surfaced:
- **#40807** (OPEN) — TurboQuant + spec-decode capture crash; reporter namechecks Sander's Patch 23. Our P44 aligns.
- **#40792** (OPEN) — TQ k8v4 GQA head grouping; may supersede our P40. Diff + bench pending.
- **#40798** (OPEN) — TQ scratch workspace across layers; superset of #40655+#40706. May conflict with P28 anchor.
- **#40794** (MERGED 2026-04-24) — MoE unpad routed output; smoke test on Qwen3.6-35B-A3B pending.
- **#40420** (OPEN) — TQ continuation-prefill OOM at 185k; adding ≥150k regression to integration gate.

No PR posted upstream without explicit user approval (per `feedback_no_push_without_explicit_approval`).

---

## v7.8.5 — 2026-04-24 (cross-quantization validation)

Validated v7.8 on three configurations: FP8 prod / AWQ 4-bit / FP16-KV 32k.

**Results**: 28 applied / 0 failed across all three. 3× 256k stable on FP8 + AWQ. AWQ frees ~9 GiB/rank → 2.5× KV capacity (1.099M → 2.787M tokens). Speed: AWQ 1-4% slower than FP8 (4-bit dequant cost on SM 8.6). Linear degradation unchanged: `1/tgs ≈ 0.007 + 2.4e-5 × ctx`.

**Finding**: TQ preallocated buffers waste ~516 MiB/rank on FP16-KV deployments where TQ is inactive — led to P51 in v7.9.

## v7.8.0 — 2026-04-24 (interface guards + middleware)

### Added

- **P49 — interface contract validation** (`interface_guard.py`, ~240 lines)
  - `GenesisInterfaceMismatch` exception
  - `validate_impl(impl, required_attrs, required_methods, optional_attrs, role)` helper
  - `validate_method_signature(method, expected_params)` — catches renamed params
  - `assert_shape_compat(t, expected, msg)` — runtime shape drift detection
  - `describe_impl(impl)` — diagnostic snapshot
  - `ANY` sentinel — presence-only check (used for Triton `@triton.jit` kernels that aren't `callable()` in Python sense)
  - Wired into P22, P38, P39a as pre-flight guards (defense layer 1)

- **P50 — ASGI `ResponseCacheMiddleware`** (`middleware/response_cache_middleware.py`, ~280 lines)
  - Drop-in ASGI middleware for any FastAPI/Starlette app (target: cliproxyapi:8330)
  - Deterministic cache key (JSON `sort_keys=True`)
  - `stream=True` + sampled requests (`temp>0`, `top_p<1`, `top_k>1`) NOT cached by default
  - Graceful degradation on cache errors (silent miss)
  - `x-genesis-cache: HIT|MISS` header for diagnostics

- 18 tests in `test_interface_guard.py` (validate, sig, shape, describe)
- 25 tests in `test_response_cache_middleware.py` (key extraction, ASGI flow, error handling)

### Fixed

- P39a initial false-positive: Triton `@triton.jit` `chunk_scaled_dot_kkt_fwd_kernel` isn't Python-callable. Switched to `required_attrs={...: ANY}` (presence check) instead of `required_methods` (callable check). The guard correctly caught the edge case — API usage corrected.

### Tests

Full unit suite: 605 passed / 8 skipped / 0 failed.

---

## v7.0.0-dev — 2026-04-24

**Major architectural shift**: migrate from monolithic text-replacement overlay (`patch_genesis_unified.py`, ~3000 LOC) to modular professional package.

### Added

- `vllm/_genesis/` package structure (upstream-compatible namespace)
- `guards.py` — canonical vendor/chip/model/dependency detection
  - Vendor identity: `is_nvidia_cuda()`, `is_amd_rocm()`, `is_intel_xpu()`, `is_cpu_only()`
  - NVIDIA compute capability: `get_compute_capability()`, `is_sm_at_least(major, minor)`, arch predicates (`is_ampere_consumer()`, `is_hopper()`, `is_blackwell()`, etc.)
  - AMD architecture: `is_rocm_cdna2()`, `is_rocm_cdna3()`, `is_rocm_rdna()` via `_GCN_ARCH` parsing
  - Dependency versions: `get_torch_version()`, `get_transformers_version()`, `get_vllm_version_tuple()`, `is_transformers_v5_plus()`, `is_torch_211_plus()`
  - Model architecture: `is_model_arch(cfg, arch_name)`, family helpers (`is_qwen3_family`, `is_deepseek_v3`, etc.)
  - Backend detection: `has_turboquant_support()`, `is_marlin_selected()`, `is_flash_attn_backend()`
  - Path resolution: `vllm_install_root()`, `resolve_vllm_file()` — replaces hardcoded `/usr/local/lib/python3.12/` paths (works on any Python version, Mac/Linux/Docker slim)
  - Diagnostic: `platform_summary()` returns full JSON-serializable platform info

- `prealloc.py` — `GenesisPreallocBuffer` framework
  - Class-level registry for shared tensor allocation
  - `get_or_create(namespace, shape, dtype, device, zero_init)` — fresh or cached
  - `slice_to(buf, n, dim)` — pointer-stable view (CUDA graph safe)
  - `get_registry_info()` — diagnostic JSON of all allocations
  - `clear_for_tests()` — test helper (warns if called outside pytest)

- `kernels/router_softmax.py` — **Patch 31** implemented
  - Drop-in replacement for `torch.softmax` in MoE routers
  - Fp32-upcast intermediate prevents bf16 mantissa collision
  - Fixes non-deterministic top-k routing on Qwen3-MoE (pre-SM90)
  - `router_softmax()` and `router_softmax_preserving_mask()` variants
  - Platform-universal: CUDA / ROCm / XPU / CPU all supported

- `kernels/dequant_buffer.py` — **Patch 22** skeleton (Phase 2 target)
  - `TurboQuantBufferManager` class with platform guard
  - Designed for profiler-visible KV buffer pre-allocation

- `kernels/gdn_dual_stream.py` — **Patch 7** skeleton (Phase 2 target)
  - `DualStreamDispatcher` with platform-aware fallback
  - NVIDIA parallel, ROCm HIP attempt, XPU/CPU sequential

- `kernels/marlin_tuning.py` — **Patch 17/18** skeleton (Phase 2 target)
  - Per-SM optimal `block_size_m` auto-selection
  - Env overrides: `VLLM_MARLIN_MOE_BLOCK_SIZE_M`, `_NUM_WARPS`, `_NUM_STAGES`

- `kernels/fp8_dispatcher.py` — **Patch 1/2** skeleton (Phase 2 target)
  - `requires_marlin_fp8_fallback()` — SM<8.9 detection
  - Per-arch routing logic

- `patches/apply_all.py` — new orchestrator replacing monolithic patcher
  - Decorator-based patch registration (`@register_patch("P31 ...")`)
  - `PatchStats` with counts and per-patch details
  - CLI entrypoint: `python3 -m vllm._genesis.patches.apply_all`
  - Exit codes: 0 success / 1 patch failure / 2 setup error
  - Stub registration for Patch 31 (full implementation Phase 2)

- `patches/upstream_compat.py` — upstream PR marker registry
  - Central tracking of all upstream fixes Genesis mirrors
  - Used by Layer 3 (upstream merge) defensive checks
  - Coverage: #39016, #39391, #39953, #40060, #40105, #40159, #40172, #40194, #40384, #40572, #40633, #38479

- `tests/conftest.py` — pytest fixtures
  - `cuda_available`, `rocm_available`, `nvidia_cuda_available` platform fixtures
  - `reset_genesis_prealloc` — clear registry before/after test
  - `deterministic_seed` — torch.manual_seed(42)
  - Custom markers: `cuda_required`, `rocm_required`, `gpu_required`, `slow`
  - Auto-skip GPU tests on CPU-only hosts

- `tests/test_guards.py` — comprehensive guards test coverage
  - TestVendorIdentity (6 tests)
  - TestComputeCapability (5 tests)
  - TestDependencyVersions (4 tests)
  - TestModelArchDetection (4 tests)
  - TestBackendDetection (2 tests)
  - TestPathResolution (3 tests)
  - TestPlatformSummary (2 tests)

- `tests/test_prealloc.py` — `GenesisPreallocBuffer` test coverage
  - TestGetOrCreate (7 tests)
  - TestSliceTo (6 tests)
  - TestRegistryInfo (3 tests)
  - TestPointerStability (2 tests) — CRITICAL for CUDA graph
  - TestClearForTests (2 tests)
  - TestCUDABehavior (2 tests)

- `tests/test_router_softmax.py` — Patch 31 TDD test suite
  - TestRouterSoftmaxDeterminism (3 tests)
  - TestRouterSoftmaxDtypePreservation (5 tests, parametrized)
  - TestRouterSoftmaxMathematicalCorrectness (5 tests)
  - TestRouterSoftmaxPlatformSafety (4 tests)
  - TestRouterSoftmaxEdgeCases (3 tests)
  - TestRouterSoftmaxPerformanceCUDA (1 test, CUDA-gated)

- `README.md` — package documentation with usage, testing, migration status
- `CHANGELOG.md` — this file

### Design decisions (why this structure)

1. **Why `vllm/_genesis/` namespace**: placed inside vllm's package layout so installation via overlay mount works without PYTHONPATH manipulation. Leading underscore marks it as "private" (Genesis-specific, not upstream API).

2. **Why separate `kernels/` and `patches/`**: clean separation between WHAT the code does (kernels) and HOW it integrates (patches). When we submit upstream PRs, we submit kernels/ directly — patches/ is just the bridging overlay.

3. **Why TDD discipline**: matches user's `CLAUDE.md` explicit requirement: "Test-first for new functionality." Also mandatory for Patch 28 (GDN prealloc) to prevent repeating Patch 19's revert (−30% throughput, 188× stdev).

4. **Why `@functools.cache` on guards**: NVML probe and vllm.platforms queries are ~1ms. Cached after first call (~50ns). At 20+ patches × startup = 20ms vs 1μs difference.

5. **Why `vllm_install_root()` helper**: replaces hardcoded `/usr/local/lib/python3.12/dist-packages/` (breaks on Mac, venv, Python 3.13 coming 2027, Docker slim images). `vllm.__file__` is canonical universal.

### Not yet done (Phase 2 target)

- Full monkey-patch glue from `kernels/` to upstream vllm modules (current v5.14.1 does text-replacement; v7.0 will use function-level monkey-patching via `patches/apply_all.py`)
- Remaining kernel implementations: `dequant_buffer.py`, `gdn_dual_stream.py`, `marlin_tuning.py`, `fp8_dispatcher.py`
- Test suites for the 4 remaining kernels
- Integration platform matrix tests (`test_platform_matrix.py`)
- Migration of Patches 1-25 from monolithic `patch_genesis_unified.py` to per-patch modular entries

## Late v7.0-dev additions (2026-04-24, session 2)

### New patches wired

- **P7** — GDN dual-stream in_proj parallelism. Text-patch on
  `model_executor/layers/mamba/gdn_linear_attn.py:544-545` replacing the
  serial `in_proj_qkvz` + `in_proj_ba` calls with a
  `DualStreamDispatcher.maybe_parallel(...)` call. Platform-safe:
  sequential fallback on CPU / XPU; true parallelism on CUDA SM ≥ 8.0.
- **P12** — Qwen3 `<tool_call>` as implicit reasoning end (ADDITIVE scope
  to avoid conflict with P27). Adds `_tool_call_token_id`,
  `is_reasoning_end`, `is_reasoning_end_streaming`,
  `extract_content_ids` methods to `Qwen3ReasoningParser`.
- **P24** — Per-SM auto-select for Marlin MoE `num_warps` and
  `num_stages`. Ampere A5000 (SM 8.6) → warps=4, stages=3 measured
  optimum. Env `VLLM_MARLIN_MOE_NUM_WARPS`, `_NUM_STAGES` still override.
- **P26** — TurboQuant prefill output prealloc. Helper
  `TurboQuantBufferManager.get_or_create_prefill_output` +
  `layer._tq_prefill_output` attach. Kernel text-patch deferred to A/B
  benchmark.
- **P27** — Qwen3 reasoning parser BEFORE-THINK fallback. Captures text
  before `<think>` (previously dropped) and routes it to `content` in
  both streaming and non-streaming paths. Coexists with P12.
- **P28** — GDN `core_attn_out` prealloc via `GdnCoreAttnManager`.
  Correct P19 redo: allocation is profiler-visible via the manager's
  first `get_or_create` on the max-sized buffer, not lazy in forward.
  Text-patch on `gdn_linear_attn.py:569-575` with unique anchor
  (includes the preceding #28182 comment) so `forward_xpu`'s identical
  line is untouched.
- **P29** — Verified the qwen3coder tool parser already contains
  bounded-index guards in the v7.0 baseline (lines 609-616, 659-666,
  436-438). Registration is a no-op on the current image; re-emits if a
  future vLLM upgrade regresses.
- **P32 / P33** — `_cu_2` and `synth_seq_lens` preallocs bundled with
  P22. `TurboQuantBufferManager.get_or_create_cu_2` +
  `get_or_create_synth_seq_lens`; attached to the layer inside
  `ensure_turboquant_buffers`.
- **P5b** — Scaffolding for the future pad-smaller-to-max KV
  unification. `kernels/page_size_padded.py` helpers
  (`is_p5b_enabled`, `compute_real_page_size_bytes`, `clamp_to_real_shape`)
  behind `GENESIS_ENABLE_P5B=1`. Kernel text-patch intentionally not
  shipped.

### Infrastructure

- `benchmarks/harness/` — Part 11.1 pre-deploy gate runner:
  - `gsm8k_regression`, `quality_harness`, `long_context_oom`,
    `tgs_decode`, `offline_api_parity`, `cuda_graph_recapture`,
    `run_all`.
  - Standard JSON report format, P0/P1 tiering, aggregated `summary.json`.
  - Dataset stubs in `benchmarks/data/`.
- `docs/RUNBOOK.md` — steady-state ops, diagnostic probes, blue/green
  deploy, rollback, known gotchas.

### Patch registry size

- Session start: 16 registered patches.
- Session end: **23 registered patches** (+P7, +P12, +P24, +P26, +P27,
  +P29, +P32/P33, +P28, +P5b).

### Compatibility

- Python: 3.10+ (uses modern type hints and `from __future__ import annotations`)
- PyTorch: 2.10+ (compatible with 2.11 upgrade in v0.20.0)
- Transformers: v5.0+ (compatible with vLLM v0.19.1+ requirement)
- vLLM: 0.19+ (tested against 0.19.2rc1.dev8, targeting 0.20.0)

### Author

Sandermage(Sander)-Barzov Aleksandr — Ukraine, Odessa
