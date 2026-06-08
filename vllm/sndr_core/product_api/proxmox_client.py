# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

Canonical location: ``sndr.product_api.legacy.proxmox_client``.
"""
from sndr.product_api.legacy.proxmox_client import *  # noqa: F401,F403
import sndr.product_api.legacy.proxmox_client as _sndr_src  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_sndr_src).items() if not _k.startswith("__")})
del _sndr_src

try:
    from sndr.product_api.legacy.proxmox_client import __all__  # noqa: F401
except ImportError:
    pass
