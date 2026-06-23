# genesis-vllm-plugin (legacy back-compat shim)

Thin compatibility shim that wires the Genesis `sndr` package into vLLM
via the official plugin API (`vllm.general_plugins` entry-point group).
When the entry-point is registered in a vLLM process, vLLM's
`load_general_plugins()` calls `register()` automatically at process
start — in **every** process (main, engine core, each worker TP rank) —
which is the right place for Genesis to rebind upstream vLLM attributes
to its own integrations.

> v12 + UNIFIED ROOT BUG fix (2026-06-22): both this subdir's
> `pyproject.toml` and the repo-root `pyproject.toml` now register the
> SAME canonical target, `sndr.plugin:register`. The shim's
> `genesis_v7/__init__.py` simply delegates to `sndr.plugin.register`
> (single apply path — no divergent logic). The previous
> `genesis_v7:register` shim target applied no Genesis runtime patch in
> the serving process, which is the bug this fix closes.

## Why the entry-point is REQUIRED (two-process boundary)

The container boot runs `python3 -m sndr.apply ; exec vllm serve` as
**two** processes:

1. `python3 -m sndr.apply` (subprocess) — applies the patch stack, then
   exits. **Text-patches** (edits to vLLM source files on disk) persist.
   But **runtime monkey-patches** (`SomeClass.method = wrapper`, e.g.
   g4_85) live only in that subprocess's memory and are LOST when it
   exits.
2. `exec vllm serve` — a brand-new Python process. It inherits the
   on-disk text-patches but NONE of the runtime monkey-patches.

The ONLY supported way to re-apply runtime monkey-patches **inside** the
serving process is vLLM's plugin system: `load_general_plugins()` calls
every `vllm.general_plugins` entry-point at engine + worker init. That
entry-point exists only if the `sndr` package is pip-installed **with its
`.dist-info` / `.egg-info` metadata**. A bare bind-mount of `sndr/` makes
the package importable but registers no entry-point — so the in-process
plugin never loads.

## Install

PROD (recommended): bake the `sndr` wheel into the image at build time —
its root `pyproject.toml` already registers the entry-point, so no
boot-time install runs:

```bash
pip install -e /path/to/genesis-vllm-patches    # repo ROOT
# Registers: vllm.general_plugins -> genesis_v7 = sndr.plugin:register
```

DEV (editable, opt-in): the launch renderer bind-mounts the sndr repo
ROOT at `/plugin` and, under `SNDR_DEV_INSTALL_PLUGIN=1`, pip-installs it
editable at boot so the entry-point registers for that run.

This legacy subdir is kept only for v7.x operators who already pip-
installed the standalone plugin package; installing it now also registers
`sndr.plugin:register`:

```bash
pip install -e /path/to/tools/genesis_vllm_plugin    # legacy subdir
```

## Verify

```bash
python3 -c "
from importlib.metadata import entry_points
for ep in entry_points(group='vllm.general_plugins'):
    print(f'{ep.name:20}  {ep.value}')
"
# Expected (either install path): genesis_v7  sndr.plugin:register
```

## What it does

Upon `register()` being called (delegated to `sndr.plugin.register`):

1. Runs `sndr.apply.run()` over the patch registry.
2. Per platform-guard, each patch either:
   - applies a text-level patch to vLLM source files, or
   - monkey-patches upstream vLLM attributes to Genesis kernels.
3. Returns; vLLM continues its startup.

Each patch is idempotent per-process and graceful on unsupported
platforms (wrong SM tier / wrong quant / drift marker present → the
patch self-skips with a clear reason).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
