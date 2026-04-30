# Genesis hello-world plugin

Reference implementation showing how to ship a community Genesis patch
via setuptools entry-points. Useful as a copy-paste starting point for
real plugins.

## What this plugin does

Nothing useful in production — it just logs a message when invoked.
The point is to demonstrate the **plugin contract** end-to-end:

1. `pyproject.toml` declares the entry-point under
   `vllm_genesis_patches`
2. `plugin.py:get_patch_metadata` returns the patch metadata dict
3. `plugin.py:apply` is what runs when the patch is engaged

## Install

```bash
# From the Genesis repo root:
pip install -e examples/genesis-plugin-hello-world/

# Verify Python can import the package:
python3 -c "import genesis_plugin_hello_world; print('ok')"

# Verify the entry-point registers:
python3 -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm_genesis_patches')
for ep in eps:
    print(f'{ep.name} → {ep.value}')
"
```

## Enable

Two opt-ins are required:

```bash
# 1. Master gate — Genesis only loads any plugin when this is set.
export GENESIS_ALLOW_PLUGINS=1

# 2. Per-plugin gate — operator must opt into THIS specific patch.
export GENESIS_ENABLE_HELLO_WORLD=1
```

## Verify

```bash
# List discovered plugins (should show HELLO_WORLD)
python3 -m vllm._genesis.compat.cli plugins list

# Show details
python3 -m vllm._genesis.compat.cli plugins show HELLO_WORLD

# Validate plugin schema
python3 -m vllm._genesis.compat.cli plugins validate

# Doctor surfaces it under the lifecycle=community bucket
python3 -m vllm._genesis.compat.cli doctor
```

## Plugin contract reference

See `docs/PLUGINS.md` in the main Genesis repo for the full plugin
authoring guide:

- Schema fields (required + optional)
- `applies_to` predicate DSL (AND / OR / NOT / NONE_OF)
- Lifecycle states + collision detection
- `apply_callable` return-value contract
- Error isolation guarantees
- Multi-patch packages (return `list[dict]`)
- Author etiquette (unique patch_id namespacing)

## Uninstall

```bash
pip uninstall genesis-plugin-hello-world

# Or temporarily disable without uninstalling:
unset GENESIS_ALLOW_PLUGINS
```

## Make this a real plugin

To ship a real Genesis plugin, fork this directory and:

1. Rename the package directory + module
2. Update `pyproject.toml`:
   - `name`, `description`, `authors`
   - Entry-point name + path
3. Replace `plugin.py:apply()` with your actual patching code:
   - text-patch via `vllm._genesis.wiring.text_patch.TextPatcher`
   - or runtime setattr / monkey-patch
   - or middleware injection
4. Update `community_credit` field with your handle
5. Pick a unique `patch_id` namespace (e.g. `MY_HANDLE_X`) to avoid
   future collisions with core patches
6. Add tests
7. Open an issue on the main Genesis repo describing what your plugin does

## License

Apache-2.0 (matches Genesis core). Adjust to taste in your own plugin.
