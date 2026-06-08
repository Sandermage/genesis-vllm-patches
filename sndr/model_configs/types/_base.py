# SPDX-License-Identifier: Apache-2.0
"""Shared base types for ``model_configs.types`` modules — M.5.1.

Single source of truth for :class:`SchemaError`. Every sub-component
dataclass in the ``types/`` package imports it from here, and
``schema.py`` re-exports it under its historical name so existing
``from sndr.model_configs.schema import SchemaError`` and
``isinstance(e, SchemaError)`` checks continue to resolve to the same
class object across the M.5.1 boundary.
"""
from __future__ import annotations


class SchemaError(ValueError):
    """Raised when a ModelConfig (or sub-component) fails validation."""
