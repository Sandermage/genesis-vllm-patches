# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: vllm.sndr_core.env.

Canonical location: :mod:`sndr.env`.

We import explicit names rather than using ``from sndr.env import *`` because
``sndr.env`` carries an ``__all__`` that intentionally limits the public API,
but legacy callers historically accessed several module-level helpers and
constants beyond ``__all__`` (e.g. ``get_sndr_env_bool``, ``GENESIS_*``
constants). The wide import below preserves backward compatibility.

Will be removed in v13.0.
"""
from sndr.env import *  # noqa: F401,F403
# Full back-compat: mirror the canonical module's entire namespace (names
# outside __all__ + private helpers) so legacy imports keep resolving.
import sndr.env as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

# Force-import non-``__all__`` symbols that legacy callers rely on.
from sndr.env import (  # noqa: F401
    get_sndr_env_bool,
)
try:
    from sndr.env import __all__  # noqa: F401
except ImportError:
    pass
