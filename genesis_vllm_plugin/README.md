# genesis-vllm-plugin

Thin shim that wires the Genesis v7.0 `vllm._genesis/` package into vLLM via
the official plugin API (`vllm.general_plugins` entry point group). When
installed in a vLLM process, vLLM calls `genesis_v7.register()` automatically
at process start — in **every** process (main, engine core, each worker TP
rank) — which is the right place for Genesis to rebind upstream vLLM
attributes to its own kernels.

## Install

```bash
# Inside the vLLM container:
pip install -e /path/to/genesis_vllm_plugin
```

## Verify

```bash
python3 -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.general_plugins')
for ep in eps:
    print(f'{ep.name:20}  {ep.value}')
"
# Expected: genesis_v7  genesis_v7:register
```

## What it does

Upon `register()` being called:

1. Imports `vllm._genesis.patches.apply_all` and runs it with `apply=True`.
2. The orchestrator walks the patch registry and, per-platform-guard, either:
   - Applies text-level patches to vLLM source files (e.g. P4 arg_utils fix), or
   - Monkey-patches upstream vLLM attributes to Genesis kernels (e.g. P31).
3. Returns; vLLM continues its startup.

Each patch is idempotent per-process and graceful on unsupported platforms.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
