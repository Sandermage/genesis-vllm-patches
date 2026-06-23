# `sndr/extras/tools/` — package-internal tools

This directory holds tools that **sndr_core imports at runtime**. Moving
them inside the package makes `sndr_core` self-contained — no Python
imports cross the package boundary into the repo-root `tools/` directory.

## Wave 10 refactor (2026-05-15)

Before Wave 10, `vllm/sndr_core/compat/bench.py` and `vllm/sndr_core/cli/bench.py`
imported files from `<repo-root>/tools/`, creating an inverse dependency
where the package depended on the repo layout. Wave 10 moved the runtime-
required files here and updated the path search order to prefer the
package-internal location, with the repo-root paths kept as legacy
fallbacks for older checkouts.

## Inventory

| File | Imported by | Purpose |
|---|---|---|
| `genesis_bench_suite.py` | `sndr/compat/bench.py:_locate_bench_module()` | Main bench tool (1425 LOC). Loaded as a module via `importlib.util.spec_from_file_location` — keeps it runnable both standalone (`python3 …/genesis_bench_suite.py …`) and through the unified CLI (`python3 -m sndr.compat.cli bench …`). |
| `bench_methodology.yaml` | `sndr/cli/legacy/bench.py:load_methodology()` | Phase 6 release-tier bench methodology contract. `bench-validate` reads it to verify result JSONs carry the mandatory fields. |

## Files that STAY in the repo-root `tools/`

These are **operator-facing** tools that ship with the source checkout but
are NOT imported by `sndr_core` at runtime:

- `restart_*_multiconc.sh` — production launch scripts (operator runs them)
- `audit_yaml_vs_runtime.sh` — drift audit shell
- `long_ctx_smoke.sh`, `soak.sh`, `memory_observability.sh` — operator
  smoke/stress shells
- `openai_smoke.py`, `progressive_context_probe.py` — operator-facing probes
- `bench_decode_tpot_clean_ab.py`, `multi_conc_bench.py` — standalone bench
  building blocks (no sndr_core import; operators may call them directly)
- `check_upstream_drift.py`, `kv_calc.py`, `license_keygen.py` — utility
  scripts referenced only from docs/CLI help strings
- `genesis_vllm_plugin/` — separate pip-installable plugin package (its own
  `pyproject.toml`); back-compat re-export shim for the v7.x era when the
  plugin lived outside sndr_core (canonical entry point is now
  `sndr.plugin:register`)
- `examples/`, `external_probe/`, `memory_explain_calibration/`, `policies/` —
  examples + isolated helper artefacts

## Policy

When adding a new tool, choose its home by asking:

1. Is it **imported at Python level from inside `sndr/`**? → place here.
2. Is it **referenced only as a string/docstring/install hint** from sndr_core? → root `tools/` is fine.
3. Is it **operator-facing** (shell script, smoke probe, launch helper)? → root `tools/`.
