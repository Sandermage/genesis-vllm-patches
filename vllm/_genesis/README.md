# Genesis — `vllm/_genesis/` package

**Modular drop-in architecture for Genesis vLLM patches — v7.0 → v7.11 (current).**

> Current release: v7.11 (28 default + 5 opt-in patches incl. P56 spec-decode workaround).
> Full feature changelog: [CHANGELOG.md](CHANGELOG.md). User-facing setup: [../QUICKSTART.md](../QUICKSTART.md).

Replaces monolithic `patch_genesis_unified.py` (v5.14.1) with a clean package structure that:
- Works on NVIDIA CUDA / AMD ROCm / Intel XPU / CPU with graceful skip (philosophy: **МЫ ЧИНИМ, НЕ ЛОМАЕМ**)
- Follows TDD discipline (tests first, implementation second)
- Is upstream-ready — kernels can be submitted as vLLM PRs directly

## Package layout

```
vllm/_genesis/
├── __init__.py              Public API entry
├── guards.py                Canonical vendor/chip/model/dep detection
├── prealloc.py              GenesisPreallocBuffer framework
├── kernels/                 Professional drop-in replacements
│   ├── router_softmax.py    ✅ Patch 31 (implemented + tests)
│   ├── dequant_buffer.py    🚧 Patch 22 skeleton (Phase 2 target)
│   ├── gdn_dual_stream.py   🚧 Patch 7 skeleton (Phase 2 target)
│   ├── marlin_tuning.py     🚧 Patch 17/18 skeleton (Phase 2 target)
│   └── fp8_dispatcher.py    🚧 Patch 1/2 skeleton (Phase 2 target)
├── patches/                 Thin bridges to upstream (legacy compat)
│   ├── apply_all.py         Orchestration entrypoint
│   └── upstream_compat.py   PR marker registry
└── tests/                   pytest TDD suite
    ├── conftest.py
    ├── test_guards.py       ✅ Complete
    ├── test_prealloc.py     ✅ Complete
    └── test_router_softmax.py ✅ Complete
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
