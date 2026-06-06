# SGLang Engine Adapter

**Status**: Functional adapter. Detects a live SGLang install, normalizes
pins, and discovers per-pin manifests (`pins/`) + community patches
(`patches/`). No pins/patches ship yet — those light up as they are added,
with no adapter change. Runtime introspection (`get_runtime_config` /
`get_model_profile`) lands with the first ported SGLang pin.

## Purpose

This directory holds the SGLang engine adapter for `sndr-platform`. It:

1. **Proves** the `EngineAdapter` ABC accommodates a non-vLLM engine with no
   changes to engine-agnostic code — a real, registered second engine
   (`list_engines()` → `['sglang', 'vllm']`).
2. **Reserves + structures** the namespace `sndr.engines.sglang.*` (adapter,
   `pins/`, `patches/`) so future manifests/patches land here cleanly.
3. **Document** the porting guide for the eventual contributor.

## How to start porting

1. Create `sndr/engines/sglang/adapter.py` with `class SglangEngine(EngineAdapter)`.
2. Implement the required ABC methods:
   - `detect_version()` — read sglang's `__version__`
   - `install_root()` — locate the sglang package directory
   - `resolve_file()` — given a relative path, return the absolute path within sglang
   - `is_pin_supported(pin)` — check `sndr/engines/sglang/pins/<pin>/` exists
   - `list_supported_pins()` — enumerate the pins/ directory
   - `get_runtime_config()` — read sglang's live config
   - `get_model_profile()` — map sglang model state to `ModelProfile`
   - `list_patches()` — enumerate `sndr/engines/sglang/patches/`
3. Update `sndr/engines/__init__.py` to import and register `SglangEngine`.
4. Create `sndr/engines/sglang/pins/<first_pin>/manifest.yaml` by running
   `sndr manifest generate --engine sglang --pin <version>`.
5. Port your first patch as proof-of-concept. Recommended starting point:
   a "reasoning" patch (low engine coupling, easy to verify).
6. Add integration test in `tests/integration/pin_matrix/sglang/`.
7. Write `docs/concepts/SGLANG_ADAPTER.md` and an ADR (`ADR-XXX-sglang-adoption.md`).

## What NOT to port directly from vLLM patches

vLLM patches target specific upstream files (e.g.
`v1/attention/ops/triton_turboquant_store.py`). These files do **not** exist
in sglang — sglang has its own attention kernels and runtime architecture.

Concepts that **may** be portable (require careful rework):
- Reasoning / tool-call parsing (model-level, engine-light)
- MoE routing logic (if sglang exposes a similar hook point)
- Quantization handling (if file structure aligns)

Concepts that are **not** portable:
- `vllm.config` introspection — sglang has different config classes
- `vllm.v1.*` paths — sglang has its own runtime layout
- TurboQuant kernel patches — TQ is specific to vllm's KV cache shape

## Reference

See:
- `sndr/engines/base.py` — the EngineAdapter ABC contract
- `sndr/engines/vllm/adapter.py` — VllmEngine as a reference implementation
- `docs/concepts/ENGINES.md` — high-level engine model
- `docs/guides/ENGINE_ADAPTER.md` — adapter authoring guide
