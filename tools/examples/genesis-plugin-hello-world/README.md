# genesis-plugin-hello-world

Reference Genesis plugin demonstrating the `vllm_genesis_patches`
entry-point + metadata contract end-to-end. The patch itself is a no-op
so this example focuses purely on the plumbing third-party authors need
to follow.

## Install

From this directory (or anywhere with `pip install -e <path>`):

```bash
pip install -e .
```

## Enable

```bash
# Allow Genesis to load community plugins (off by default).
export GENESIS_ALLOW_PLUGINS=1

# Opt into THIS plugin specifically (each plugin has its own flag).
export GENESIS_ENABLE_HELLO_WORLD=1
```

## Verify

```bash
# Lists all discovered plugins, with lifecycle/origin annotations.
python3 -m vllm._genesis.compat.cli plugins list
```

Expected entry: `HELLO_WORLD` (lifecycle: `community`, origin from this
package's entry-point).

## What this example demonstrates

- `pyproject.toml` with the `vllm_genesis_patches` entry-point group
- `get_patch_metadata()` returning the required fields (`patch_id`,
  `title`, `env_flag`, `default_on`, `community_credit`) plus
  optional fields (`apply_callable`, `category`, `credit`)
- `apply()` returning the canonical `(status, reason)` tuple
- Schema-clean metadata (passes `vllm._genesis.compat.schema_validator`)
- End-to-end discovery + validation pipeline (covered by
  `vllm/_genesis/tests/compat/test_plugin_example.py`)

## What it does NOT demonstrate

This example is intentionally minimal. For real plugins you will likely
also want:

- `applies_to` predicate (gate on hardware / model / vLLM version)
- `requires_patches` / `conflicts_with` declarations
- Real patching logic (text-patch via `wiring/text_patch.py`, or runtime
  monkey-patch — see core Genesis patches in `vllm/_genesis/wiring/` for
  worked examples)

See `docs/PLUGINS.md` at the repo root for the full plugin guide.
