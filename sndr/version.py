# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for sndr-platform version.

The version follows semantic versioning (semver.org):
    MAJOR.MINOR.PATCH

  - MAJOR: breaking changes to public API or removal of features
  - MINOR: new features, backward compatible
  - PATCH: bug fixes only

External consumers may import __version__ from this module.

Do NOT bump this file manually; release tooling updates it during the
publish workflow (see .github/workflows/release.yml).
"""
from __future__ import annotations

__version__: str = "12.0.0.dev0"

# Major version is the wire/contract version for license tokens and
# engine adapter ABC compatibility. Customers' license tokens carry an
# engine_major field that must match this value at boot.
__version_major__: int = 12

# Build commit SHA — populated by release pipeline (git rev-parse HEAD).
# In dev builds this is "dev"; in published wheels this is the actual SHA.
__commit__: str = "dev"

# ---------------------------------------------------------------------------
# Backward-compatibility aliases preserved for v12.x.
#
# Tests, telemetry, and downstream tooling have historically read constants
# named ``GENESIS_VERSION`` and ``SNDR_CORE_VERSION`` from the legacy
# ``vllm.sndr_core.version`` module. We surface them here so the shim at
# the old location can re-export the SAME identifiers via ``import *``.
#
# Remove in v13.0.
# ---------------------------------------------------------------------------
SNDR_CORE_VERSION = __version__
GENESIS_VERSION = __version__


__all__ = [
    "__version__",
    "__version_major__",
    "__commit__",
    "SNDR_CORE_VERSION",
    "GENESIS_VERSION",
]
