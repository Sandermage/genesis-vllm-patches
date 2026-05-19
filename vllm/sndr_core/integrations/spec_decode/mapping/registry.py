# SPDX-License-Identifier: Apache-2.0
"""mapping/registry — model-arch -> MappingProvider lookup.

A tiny registry. Future providers (EAGLE, MTP variants, custom)
register themselves here. ``find_provider(runner)`` returns the first
provider whose ``.supports(runner)`` is True; ``None`` if none match
(in which case spec-decode K/V sharing semantics are simply not
applicable to that model).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import MappingProvider
from .gemma4 import Gemma4MappingProvider

log = logging.getLogger("genesis.spec_decode.mapping.registry")


#: First-match-wins registry. Order matters — most-specific first.
PROVIDERS: list[MappingProvider] = [
    Gemma4MappingProvider(),
]


def register(provider: MappingProvider, *, prepend: bool = False) -> None:
    """Add a provider. ``prepend=True`` makes it the most-specific."""
    if prepend:
        PROVIDERS.insert(0, provider)
    else:
        PROVIDERS.append(provider)


def find_provider(runner: Any) -> MappingProvider | None:
    """Return the first provider whose .supports(runner) is True."""
    for p in PROVIDERS:
        try:
            if p.supports(runner):
                return p
        except Exception as _e:
            log.warning("[mapping.registry] %s.supports failed: %s",
                        p.name, _e)
    return None


__all__ = ["PROVIDERS", "register", "find_provider"]
