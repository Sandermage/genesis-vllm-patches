# SPDX-License-Identifier: Apache-2.0
"""Plugin discovery layer.

External wheels (notably ``sndr-engine``) register patches and other extensions
with sndr-platform via setuptools entry points. This package provides the
loader that discovers, validates, and exposes those plugins to the engine
adapters.

See ``docs/guides/COMMERCIAL_TIER.md`` for the full plugin author guide.
"""
from sndr.plugins.loader import discover_engine_patches, get_plugin_info

__all__ = ["discover_engine_patches", "get_plugin_info"]
