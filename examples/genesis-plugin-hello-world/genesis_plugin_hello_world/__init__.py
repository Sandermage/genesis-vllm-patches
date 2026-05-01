"""Reference Genesis plugin — minimal end-to-end example.

See `plugin.py` for the metadata + apply contract that real plugins
must follow. See repo `docs/PLUGINS.md` for the full plugin guide.
"""
from genesis_plugin_hello_world.plugin import apply, get_patch_metadata

__all__ = ["apply", "get_patch_metadata"]
