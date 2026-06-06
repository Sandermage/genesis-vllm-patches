# SPDX-License-Identifier: Apache-2.0
"""SGLang engine adapter.

A functional :class:`~sndr.engines.base.EngineAdapter` for SGLang (community
tier), registered in the engine registry. It detects a live SGLang install,
normalizes pins, and discovers per-pin manifests + community patches under
this package. Runtime introspection lands with the first ported SGLang pin.

Adding SGLang coverage:
  1. Validate a pin, generate its manifest under ``pins/<pin>/manifest.yaml``
     (tools/manifest_gen.py) — ``list_supported_pins()`` picks it up.
  2. Port the first patch into ``patches/`` — ``list_patches()`` picks it up.
  3. Add an integration test.
"""
from sndr.engines.sglang.adapter import SglangEngine

__all__ = ["SglangEngine"]
