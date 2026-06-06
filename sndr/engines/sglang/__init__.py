# SPDX-License-Identifier: Apache-2.0
"""SGLang engine adapter — skeleton.

The adapter is registered in the engine registry but its methods raise
:exc:`EngineNotInstalledError` until a real port lands. See
``sndr/engines/sglang/README.md`` for the porting guide.

Activating sglang in v13+:
  1. Implement adapter methods (see README)
  2. Generate first manifest (tools/manifest_gen.py)
  3. Port first patch as proof-of-concept
  4. Add integration test
  5. Update SUPPORTED_LOCALES if Russian sglang docs needed
"""
from sndr.engines.sglang.adapter import SglangEngine

__all__ = ["SglangEngine"]
