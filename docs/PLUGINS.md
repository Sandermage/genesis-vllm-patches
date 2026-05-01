# Genesis plugins — community-shipped patches via entry-points

Genesis supports third-party patches without forking the core repo.
Ship your patch as an installable Python package with a single
entry-point declaration; Genesis auto-discovers + registers it at boot.

> **Reference example:** see [`examples/genesis-plugin-hello-world/`](../examples/genesis-plugin-hello-world/) for a working scaffold you can copy. The example is tested in CI (`tests/compat/test_plugin_example.py`); if the contract drifts, those tests catch it before the docs become misleading.

## Quick start

### 1. Create a plugin package

```text
my-genesis-plugin/
├── pyproject.toml
└── my_genesis_plugin/
    ├── __init__.py
    └── patch.py
```

### 2. `pyproject.toml`

```toml
[project]
name = "my-genesis-plugin"
version = "0.1.0"
dependencies = []  # Genesis is the only requirement; users install it separately

[project.entry-points."vllm_genesis_patches"]
my_patch = "my_genesis_plugin.patch:get_patch_metadata"
```

The entry-point group is **`vllm_genesis_patches`** (no dots, per
PyPA convention). Each entry-point's value points to a callable.

### 3. `my_genesis_plugin/patch.py`

```python
"""Example community plugin for Genesis."""
from __future__ import annotations


def get_patch_metadata():
    """Return the patch metadata dict (or list of dicts for multiple)."""
    return {
        "patch_id": "MY_PATCH",                          # required
        "title":    "My community patch — does X",        # required
        "env_flag": "GENESIS_ENABLE_MY_PATCH",            # required
        "default_on": False,                              # required

        # lifecycle is auto-tagged "community" — your value is ignored,
        # so don't bother claiming "stable". Genesis enforces provenance.

        "category": "spec_decode",                        # optional
        "credit":   "What + why my patch fixes; reference any upstream PRs.",
        "community_credit": "@your_handle on GitHub",     # required for community

        # Optional: gate on hardware / model / version
        "applies_to": {
            "all_of": [
                {"is_turboquant": True},
                {"compute_capability_min": [8, 6]},
            ],
        },

        # Optional: declare dependencies / conflicts on other patches
        "requires_patches": [],
        "conflicts_with": [],
    }
```

### 4. Install + enable

```bash
pip install -e my-genesis-plugin/        # editable install
export GENESIS_ALLOW_PLUGINS=1            # opt-in (off by default)
export GENESIS_ENABLE_MY_PATCH=1          # operator opts into your patch

# Boot Genesis as usual; your patch shows up:
python3 -m vllm._genesis.compat.plugins list
python3 -m vllm._genesis.compat.doctor
```

## What Genesis enforces

When a plugin is discovered, Genesis runs the same validation pipeline
core patches go through:

1. **Schema validation** against `schemas/patch_entry.schema.json`
2. **Lifecycle force**: `lifecycle: community` always (provenance signal)
3. **Collision check**: rejects plugin if `patch_id` clashes with a
   core registry entry
4. **Origin stamping**: `_plugin_origin = "<entry-point>:<value>"` so
   doctor + explain show provenance
5. **Auto-strip**: `community_credit` field auto-filled from entry-point
   if you don't provide one

Bad plugin → SKIPPED, not crashed. One broken plugin can't break
discovery of others.

## Opt-in security model

**Plugin discovery is OFF by default**. Set `GENESIS_ALLOW_PLUGINS=1`
to enable. This is a hard gate — Genesis loads zero foreign code
without explicit operator consent.

When the gate is closed, every plugin-related CLI shows:

```
Plugin discovery is OFF — set GENESIS_ALLOW_PLUGINS=1 to enable.
```

## Returning multiple patches per package

A single entry-point callable may return a list of dicts to ship
multiple patches:

```python
def get_patch_metadata():
    return [
        {"patch_id": "MY_A", "title": "...", "env_flag": "GENESIS_ENABLE_MY_A",
         "default_on": False, "community_credit": "..."},
        {"patch_id": "MY_B", "title": "...", "env_flag": "GENESIS_ENABLE_MY_B",
         "default_on": False, "community_credit": "..."},
    ]
```

## Implementation: where the patch actually applies

Declare an `apply_callable` field in your plugin metadata. Genesis
will call it during boot when:
1. `GENESIS_ALLOW_PLUGINS=1` (master gate)
2. Your plugin's `env_flag` is set to a truthy value
3. `apply_all` is invoked with `apply=True` (production path)

Two ways to provide the callable:

**1. As a string** (preferred — entry-point style, lazy-loaded):

```python
# In get_patch_metadata():
{
    ...
    "apply_callable": "my_genesis_plugin.apply:apply",
}

# In my_genesis_plugin/apply.py:
def apply():
    """Apply the patch. Returns (status, reason)."""
    # Use vllm._genesis.wiring.text_patch.TextPatcher for text patches,
    # or directly modify modules / classes via setattr.
    return "applied", "MY_PATCH applied: did the thing"
```

**2. As a Python callable** (advanced — for plugins that compose
metadata at runtime):

```python
def my_apply():
    return "applied", "MY_PATCH applied"

def get_patch_metadata():
    return {
        ...
        "apply_callable": my_apply,  # callable directly
    }
```

### Return value contract

`apply_callable` should return `(status, reason)` where status is one of:

- `"applied"` — patch successfully engaged
- `"skipped"` — patch determined it shouldn't apply on this system
  (e.g. wrong vllm pin, missing kernel, etc.)
- `"failed"` — patch tried to apply but hit an unrecoverable error

The `reason` is a free-text string surfaced in the dispatcher matrix
+ `genesis doctor` output. Make it diagnostic.

If your callable returns a plain string, Genesis treats it as
`("applied", <string>)`. If it returns `None`, treated as applied
with a generic message. Returning anything else (e.g. an int) is
treated as success-with-warning.

### Error isolation

If `apply_callable` raises any exception, Genesis catches it, logs the
traceback, and reports `("failed", "<ExceptionClass>: <message>")` —
the error never propagates out of `apply_all`. One bad plugin can't
take down the engine.

### What if I don't declare apply_callable?

Then your plugin is **metadata-only**: it shows up in `genesis doctor`,
`genesis lifecycle-audit`, and the patch registry, but Genesis won't
run any code on your behalf. This is useful for:

- Documenting community-known issues without shipping a fix
- Declaring environment requirements (env_flag becomes a sentinel)
- Tracking community-contributed metadata before the code is ready

## CLI

```bash
# List discovered plugins (when gate is open)
python3 -m vllm._genesis.compat.plugins list

# Show one plugin's full metadata
python3 -m vllm._genesis.compat.plugins show MY_PATCH

# Validate plugins without booting Genesis
python3 -m vllm._genesis.compat.plugins validate
```

## Author etiquette

- Open an issue at <https://github.com/Sandermage/genesis-vllm-patches/issues>
  describing your plugin's purpose before publishing
- Use a unique `patch_id` namespace (`MY_HANDLE_X`) to avoid future
  collisions with core patches
- Document hardware tested on
- If your patch supersedes a core patch, mention it in `community_credit`

## Future (Phase 5c)

Coming in a later release:

- `apply_callable` declaration — plugin's own apply function
- Opt-in telemetry: anonymized success-stack reporting
- Plugin marketplace docs / curated index
