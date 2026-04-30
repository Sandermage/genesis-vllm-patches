# Genesis — `vllm/_genesis/` package

**Modular drop-in architecture for Genesis vLLM patches — v7.0 → v7.63.x (current).**

> Current release: **v7.63.x** (2026-04-30) — Phase 1 of the **Compat Layer overhaul**: doctor CLI, model registry, version-range matching, richer predicates DSL, lifecycle states.
> Full feature changelog: [CHANGELOG.md](CHANGELOG.md). User-facing setup: [../../QUICKSTART.md](../../QUICKSTART.md). Top-level README: [../../README.md](../../README.md).

Replaces monolithic `patch_genesis_unified.py` (v5.14.1) with a clean package structure that:
- Works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU with graceful skip (philosophy: **МЫ ЧИНИМ, НЕ ЛОМАЕМ**)
- Follows TDD discipline (tests first, implementation second) — **800+ tests across the package**
- Is upstream-ready — kernels can be submitted as vLLM PRs directly
- Self-documents via `genesis doctor` and the curated model registry

## Package layout (v7.63.x)

```text
vllm/_genesis/
├── __init__.py              Public API entry
├── dispatcher.py            PATCH_REGISTRY (48 entries) + A3/D2 validator
├── guards.py                Canonical vendor/chip/model/dep detection
├── prealloc.py              GenesisPreallocBuffer framework
│
├── compat/                  🆕 v7.63.x — Unified compat / UX / diagnostic layer
│   ├── doctor.py            🆕 `python3 -m vllm._genesis.compat.doctor`
│   ├── init_wizard.py       🆕 `python3 -m vllm._genesis.compat.init_wizard`
│   ├── version_check.py     🆕 vllm/torch/cuda/triton/driver range matching
│   ├── predicates.py        🆕 AND/OR/NOT applies_to evaluator
│   ├── lifecycle.py         🆕 patch lifecycle state machine
│   ├── gpu_profile.py       Re-export shim (legacy import path preserved)
│   ├── model_detect.py      Re-export shim
│   ├── config_detect.py     Re-export shim
│   ├── models/
│   │   ├── registry.py      🆕 SUPPORTED_MODELS dict (5 entries)
│   │   ├── pull.py          🆕 HF download + verify + launch script gen
│   │   └── list_cli.py      🆕 `python3 -m vllm._genesis.compat.models.list_cli`
│   └── fingerprints/        🆕 Reference benchmark JSONs
│       └── rtx_a5000_x2_qwen3_6_27b_int4_v794.json
│
├── kernels/                 Genesis-original Triton kernels
│   ├── ffn_intermediate_cache.py   PN12 — Cliff 1 fix
│   ├── p67_multi_query_kernel.py   P67 — TQ K+1 verify
│   ├── block_verify_sampler.py     P71 — Sun 2024 ICLR (A4-hardened)
│   ├── router_softmax.py           P31
│   ├── dequant_buffer.py           P22/P26
│   ├── gdn_dual_stream.py          P7
│   ├── gdn_core_attn_manager.py    P28
│   ├── fla_kkt_buffer.py           P39a
│   └── ...
│
├── wiring/                  Text-patch wiring (72 modules in 9 dirs — Phase 2.1)
│   ├── text_patch.py        TextPatcher framework + B2 result_to_wiring_status
│   ├── rebind.py            runtime class-method rebind helpers
│   ├── spec_decode/         22 — P56-P79c, P82-83, P86, P94, PN8-9
│   ├── structured_output/    6 — P59, P61/61b, P62, P64, P68/69
│   ├── perf_hotfix/          4 — P98-101
│   ├── compile_safety/       3 — P72, P74, P78
│   ├── kv_cache/             2 — P84-85
│   ├── kernels/              4 — P81, P87, P91, PN14
│   ├── hybrid/               5 — P95, P103, PN11-13
│   ├── middleware/           1 — PN16 lazy_reasoner
│   └── legacy/              25 — P1-P55 (pre-PATCH_REGISTRY series,
│                                  apply_all.py dry-run only)
│
├── middleware/              Request-level pre-engine logic
│   ├── lazy_reasoner.py            PN16 — hybrid policy (variants 1+3+5)
│   ├── long_ctx_tool_adherence.py  P68/P69
│   └── response_cache_middleware.py
│
├── patches/                 Orchestration + upstream tracking
│   ├── apply_all.py         Boot-time orchestrator
│   └── upstream_compat.py   PR marker registry (auto-retire on merge)
│
└── tests/                   pytest TDD suite — full session = 456 green
    ├── conftest.py
    ├── compat/              🆕 v7.63.x test directory
    │   ├── test_predicates.py        27 tests
    │   ├── test_version_check.py     20 tests
    │   ├── test_lifecycle.py         18 tests
    │   ├── test_models_registry.py   13 tests
    │   ├── test_doctor_smoke.py       6 tests
    │   ├── test_categories.py        21 tests
    │   ├── test_explain.py           24 tests
    │   ├── test_recipes.py           36 tests
    │   ├── test_recipe_adopt.py      14 tests
    │   ├── test_plugins.py           18 tests
    │   ├── test_plugin_example.py    13 tests
    │   ├── test_telemetry.py         23 tests
    │   ├── test_update_channel.py    18 tests
    │   ├── test_schema_validator.py  15 tests
    │   ├── test_self_test.py         17 tests
    │   ├── test_bench.py             11 tests
    │   └── test_cli.py               14 tests
    ├── test_dispatcher_validator.py  A3/D2 — 24 tests
    ├── test_pn14_tq_decode_oob_clamp.py   13 tests
    ├── test_pn16_lazy_reasoner.py    41 tests
    ├── test_bench_ablation.py        D3 — 11 tests
    ├── test_wiring_status_helper.py  B2 — 10 tests
    ├── test_version.py                5 tests
    ├── test_ci_workflow.py            6 tests
    └── ... (legacy suites, all green)
```

## v7.63.x quick start (operator)

```bash
# 1. Diagnostic — what's my system, what would Genesis do?
python3 -m vllm._genesis.compat.doctor

# 2. Browse curated model registry
python3 -m vllm._genesis.compat.models.list_cli

# 3. First-run wizard (interactive)
python3 -m vllm._genesis.compat.init_wizard

# 4. Or, direct: download a model + get tailored launch script
python3 -m vllm._genesis.compat.models.pull qwen3_6_27b_int4_autoround \
    --workload long_ctx_tool_call --tp 2

# 5. Run your bench, optionally compare against reference fingerprint
python3 tools/genesis_bench_suite.py --quick --ablate-against \
    vllm/_genesis/compat/fingerprints/rtx_a5000_x2_qwen3_6_27b_int4_v794.json
```

## Design principles

### 1. МЫ ЧИНИМ, НЕ ЛОМАЕМ (We fix, we don't break)
Every patch uses a 5-layer defensive guard: file exists → idempotency marker → upstream merge check → vendor/chip compat → model/backend arch.
If any layer fails, patch returns `True` (success = skipped cleanly), never raises.

### 2. Three-source truth tracking
Verify each patch against three vLLM sources simultaneously:
- Release tag (e.g. `v0.20.0`)
- `main` branch HEAD
- `nightly` docker image

Patch ready for deploy only when all three are green.

### 3. TDD discipline
For each new kernel module:
```
1. Write test → run → see RED (ImportError or assertion failure)
2. Implement minimal code → run → see GREEN
3. Refactor keeping GREEN
```

### 4. Canonical vendor detection
No duplication of detection logic across patches. All helpers live in `guards.py`:
- `is_nvidia_cuda()` — strict NVIDIA (not ROCm trap)
- `is_sm_at_least(major, minor)` — compute capability gate
- `is_rocm_cdna3()` — MI300X/MI325X detection
- `is_model_arch(cfg, "Qwen3")` — architecture match
- `has_turboquant_support(cache_dtype)` — backend gate

## Usage

### As container entrypoint

```yaml
# docker-compose.yml
services:
  vllm-server:
    image: vllm/vllm-openai:nightly
    volumes:
      - ./vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
    entrypoint: [
      "/bin/bash", "-c",
      "python3 -m vllm._genesis.patches.apply_all && exec vllm serve \"$@\"",
      "--"
    ]
```

### Standalone for testing

```bash
# Apply all patches (dry inspection)
python3 -m vllm._genesis.patches.apply_all

# Expected output format:
# [INFO:genesis.apply_all] Genesis Unified Patch v7.0 — Ampere FP8 + TQ + ...
# [INFO:genesis.apply_all] [Genesis P31] Target file at /path/to/file.py ready for monkey-patch
# [INFO:genesis.apply_all] Genesis Results: 1 applied, 0 skipped, 0 failed
```

### From Python code

```python
from vllm._genesis.guards import is_nvidia_cuda, is_ampere_consumer
from vllm._genesis.kernels.router_softmax import router_softmax
from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

# Platform-aware code
if is_nvidia_cuda() and is_ampere_consumer():
    print("Running on A5000-class GPU")

# Drop-in replacement
weights = router_softmax(gating_output)  # instead of torch.softmax(...)

# Safe pre-allocation
buf = GPB.get_or_create(
    namespace="my_kernel_scratch",
    shape=(4, 128),
    dtype=torch.bfloat16,
    device="cuda",
)
slice_view = GPB.slice_to(buf, 2)  # view, pointer-stable
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
cd /Users/sander/Documents/Visual Studio Code/genesis-vllm-patches
PYTHONPATH=. pytest vllm/_genesis/tests/ -v

# With coverage
PYTHONPATH=. pytest vllm/_genesis/tests/ -v --cov=vllm._genesis --cov-report=term-missing

# Only CPU tests (skip GPU-required)
PYTHONPATH=. pytest vllm/_genesis/tests/ -v -m 'not cuda_required'
```

## Migration status (as of 2026-04-24)

| Module | Status | Patch # | Priority |
|--------|--------|---------|----------|
| `guards.py` | ✅ Complete | — | P0 foundation |
| `prealloc.py` | ✅ Complete | — | P0 foundation |
| `kernels/router_softmax.py` | ✅ Complete | P31 | P0 — self-contained universal win |
| `kernels/dequant_buffer.py` | 🚧 Skeleton | P22 | P1 — high value, TQ prod user |
| `kernels/gdn_dual_stream.py` | 🚧 Skeleton | P7 | P1 — +8% decode |
| `kernels/marlin_tuning.py` | 🚧 Skeleton | P17/P18 | P1 — +1.2% tuned |
| `kernels/fp8_dispatcher.py` | 🚧 Skeleton | P1/P2 | P1 — Ampere viability |
| `patches/apply_all.py` | ✅ Complete | — | P0 orchestration |
| `patches/upstream_compat.py` | ✅ Complete | — | P0 PR tracking |
| `tests/conftest.py` | ✅ Complete | — | P0 test infra |
| `tests/test_guards.py` | ✅ Complete | — | P0 |
| `tests/test_prealloc.py` | ✅ Complete | — | P0 |
| `tests/test_router_softmax.py` | ✅ Complete | P31 | P0 |
| `tests/test_dequant_buffer.py` | ⏳ Pending | P22 | Phase 2 |
| `tests/test_gdn_dual_stream.py` | ⏳ Pending | P7 | Phase 2 |
| `tests/test_marlin_tuning.py` | ⏳ Pending | P17/18 | Phase 2 |
| `tests/test_fp8_dispatcher.py` | ⏳ Pending | P1/2 | Phase 2 |
| `tests/test_platform_matrix.py` | ⏳ Pending | — | Phase 2 |

## Upstream attribution

Genesis v7.0 draws on and credits prior work:

- **DeepSeek-V3 team** (vLLM contributors) — fp32 router upcast pattern (`deepseek_v2.py:345`), basis for Patch 31
- **@JartX** — TurboQuant author, `JartX/vllm#11` FP16 rotation (Patch 20 prerequisite)
- **@jhsmith409** — endorsed Genesis Ampere investigation, pre-approved Patch 22 approach
- **@ZJY0516** — hybrid prefix cache design clarifications
- **@vibhavagarwal5** — collaborative guidance on related PR scopes
- **@youkaichao** — memory profiler invariants documentation
- **vLLM core team** (@WoosukKwon, @zhuohan123, @robertgshaw2-redhat, @bnellnm) — responsive community, educational codebase

Full PR attribution in each kernel module docstring.

## Author

**Sandermage(Sander)-Barzov Aleksandr**
Ukraine, Odessa
GitHub: [@Sandermage](https://github.com/Sandermage)
Project: [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

---

*Part of Genesis vLLM Master Plan v7.0 (2026-04-24)*
*Canonical reference: `Genesis_Doc/common/GENESIS_VLLM_MASTER_PLAN_v7.0_20260424.md`*
