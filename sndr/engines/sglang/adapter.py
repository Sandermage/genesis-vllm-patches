# SPDX-License-Identifier: Apache-2.0
"""SGLang engine adapter — skeleton implementation.

This is a NON-FUNCTIONAL skeleton intended for v13+ contributors. The methods
raise EngineNotInstalledError to make it clear that sglang support is not
yet ported. See ``sndr/engines/sglang/README.md`` for the porting guide.

Why we ship this skeleton in v12:

  1. **Architecture validation**: proving that EngineAdapter accommodates a
     non-vllm engine without changes confirms the abstraction is correct.
  2. **Namespace reservation**: makes ``sndr.engines.sglang.*`` an addressable
     location for future patches.
  3. **Documentation**: the porting guide tells future contributors exactly
     how to fill in this skeleton.

To activate this adapter, fill in the methods, register it in
``sndr/engines/__init__.py``, generate manifests, and port the first patch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sndr.engines.base import EngineAdapter, ModelProfile
from sndr.exceptions import EngineNotInstalledError


class SglangEngine(EngineAdapter):
    """Skeleton adapter for SGLang.

    All methods raise :exc:`EngineNotInstalledError` until a real porting
    effort fills them in. This is intentional — we do not want partial
    implementations leaking into production.
    """

    name = "sglang"

    def detect_version(self) -> str:
        raise EngineNotInstalledError(
            "SGLang adapter is a skeleton in v12. Port required for activation. "
            "See sndr/engines/sglang/README.md.",
            engine="sglang",
        )

    def install_root(self) -> Path | None:
        return None

    def resolve_file(self, relative_path: str) -> Path | None:
        return None

    def is_pin_supported(self, pin: str | None) -> bool:
        return False

    def list_supported_pins(self) -> tuple[str, ...]:
        return ()

    def get_runtime_config(self) -> dict[str, Any] | None:
        return None

    def get_model_profile(self) -> ModelProfile | None:
        return None

    def list_patches(self) -> list[Any]:
        return []


__all__ = ["SglangEngine"]
