# genesis-vllm-plugin (legacy shim)

Thin compatibility shim that wires the Genesis `vllm.sndr_core` package
into vLLM via the official plugin API (`vllm.general_plugins` entry
point group). When installed in a vLLM process, vLLM calls
`genesis_v7.register()` automatically at process start — in **every**
process (main, engine core, each worker TP rank) — which is the right
place for Genesis to rebind upstream vLLM attributes to its own
integrations.

> Audit 2026-05-14 P1-6: this README was rewritten to match the v11
> reality. The shim's `genesis_v7/__init__.py` no longer imports the
> retired `vllm._genesis` namespace — it pulls from
> `vllm.sndr_core.apply.orchestrator` directly. The canonical
> contributor-facing entry point is now
> `vllm.sndr_core.plugin:register`, which is the same callable
> registered by the in-package wheel (`pip install vllm-sndr-core`).
> This standalone tool is kept only as a back-compat shim for
> operators who already have it installed.

## Install

```bash
# Inside the vLLM container (only when the in-package plugin is not used):
pip install -e /path/to/tools/genesis_vllm_plugin
```

Or, preferred for new operators, install the in-package plugin:

```bash
pip install -e /path/to/genesis-vllm-patches
# This already registers:
#   vllm.general_plugins → genesis_v7 = vllm.sndr_core.plugin:register
```

## Verify

```bash
python3 -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.general_plugins')
for ep in eps:
    print(f'{ep.name:20}  {ep.value}')
"
# Expected: genesis_v7  genesis_v7:register   (legacy shim)
#        OR genesis_v7  vllm.sndr_core.plugin:register   (in-package)
```

## What it does

Upon `register()` being called:

1. Imports `vllm.sndr_core.apply.orchestrator` and runs it.
2. The orchestrator walks the patch registry and, per-platform-guard,
   either:
   - Applies text-level patches to vLLM source files (e.g.
     P67 TurboQuant multi-query), or
   - Monkey-patches upstream vLLM attributes to Genesis kernels (e.g.
     P31 / PN59 / PN204).
3. Returns; vLLM continues its startup.

Each patch is idempotent per-process and graceful on unsupported
platforms (a wrong SM tier / wrong quant / drift marker present →
patch self-skips with a clear reason).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
