# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.version.

Canonical location: :mod:`sndr.version`.

Tests, telemetry, and downstream tooling have historically read constants
named ``GENESIS_VERSION`` and ``SNDR_CORE_VERSION`` from this module. They
remain available as aliases for ``sndr.version.__version__``.

Will be removed in v13.0.
"""
from sndr.version import (  # noqa: F401
    GENESIS_VERSION,
    SNDR_CORE_VERSION,
    __commit__,
    __version__,
    __version_major__,
)
from sndr.version import __all__  # noqa: F401
