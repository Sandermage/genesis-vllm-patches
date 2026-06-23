# SPDX-License-Identifier: Apache-2.0
"""vLLM engine adapter.

The adapter delegates to the canonical ``sndr`` engine helpers (detection /
patching / profile resolution) so production behaviour is unchanged across
the v12 relocation. The pre-v12 ``vllm/sndr_core`` shim tree is gone — every
symbol is imported from the ``sndr`` package directly; nothing references or
re-exports through ``vllm/sndr_core`` any more.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sndr.engines.base import EngineAdapter, ModelProfile
from sndr.exceptions import EngineNotInstalledError

log = logging.getLogger("sndr.engines.vllm")


class VllmEngine(EngineAdapter):
    """Engine adapter for vLLM (community tier).

    Phase 1 implementation delegates to sndr helpers via try/except
    imports. Phases 2-4 progressively move the actual logic here.
    """

    name = "vllm"

    # -----------------------------------------------------------------
    # Required EngineAdapter API
    # -----------------------------------------------------------------

    def detect_version(self) -> str:
        """Return the installed vLLM version string.

        Tries vllm.__version__ first, falls back to legacy guards helper.
        """
        try:
            import vllm
            return str(vllm.__version__)
        except ImportError as e:
            raise EngineNotInstalledError(
                "vllm package is not installed",
                engine="vllm",
                cause=str(e),
            ) from e

    def install_root(self) -> Path | None:
        """Return the directory containing the vllm package."""
        try:
            import vllm
            return Path(vllm.__file__).parent
        except ImportError:
            return None

    def resolve_file(self, relative_path: str) -> Path | None:
        """Resolve a path relative to the vllm install root.

        Phase 1 delegates to sndr.engines.vllm.detection.guards.resolve_vllm_file
        for backward compatibility. Phase 3 will move this logic here.
        """
        # Try new path first (Phase 2+)
        try:
            from sndr.engines.vllm.detection.guards import resolve_vllm_file
            result = resolve_vllm_file(relative_path)
            if result:
                return Path(result)
        except ImportError:
            pass

        # Fallback to legacy path during Phase 1 transition
        try:
            from sndr.engines.vllm.detection.guards import resolve_vllm_file as legacy
            result = legacy(relative_path)
            return Path(result) if result else None
        except ImportError:
            # No detection helpers available; compute manually.
            root = self.install_root()
            if root is None:
                return None
            candidate = root / relative_path
            return candidate if candidate.exists() else None

    def is_pin_supported(self, pin: str | None) -> bool:
        """Return True if a manifest exists for this pin.

        Phase 1: always returns True (manifests not yet generated).
        Phase 7 will check engines/vllm/pins/<pin>/manifest.yaml exists.
        """
        if pin is None:
            return False
        # TODO Phase 7: check manifest existence
        return True

    def list_supported_pins(self) -> tuple[str, ...]:
        """List all pins with manifests.

        Phase 1: returns empty tuple (no manifests generated yet).
        Phase 7 will enumerate engines/vllm/pins/ subdirectories.
        """
        # TODO Phase 7: scan engines/vllm/pins/
        return ()

    def get_runtime_config(self) -> dict[str, Any] | None:
        """Return vllm's current vllm_config if available."""
        try:
            from vllm.config import get_current_vllm_config
            return get_current_vllm_config()
        except Exception:
            return None

    def get_model_profile(self) -> ModelProfile | None:
        """Return normalized model profile.

        Phase 1: delegates to legacy model_detect helpers.
        Phase 3 will use sndr/engines/vllm/detection/ directly.
        """
        cfg = self.get_runtime_config()
        if cfg is None:
            return None

        try:
            from sndr.engines.vllm.detection import model_detect
            mp = model_detect.get_model_profile()
            return ModelProfile(
                architectures=tuple(mp.get("architectures") or ()),
                model_class=str(mp.get("model_class") or "unknown"),
                quant_format=str(mp.get("quant_format") or "unknown"),
                kv_cache_dtype=str(mp.get("kv_cache_dtype") or "auto"),
                is_moe=bool(mp.get("moe", False)),
                is_hybrid=bool(mp.get("hybrid", False)),
                is_turboquant=bool(mp.get("turboquant", False)),
                extra={"resolved": mp.get("resolved", False)},
            )
        except Exception as e:
            log.warning("Failed to read model profile via legacy: %s", e)
            return None

    def list_patches(self) -> list[Any]:
        """Return all patches available for vLLM.

        Phase 1: delegates to legacy PATCH_REGISTRY.
        Phase 4 will use sndr.engines.vllm.patches.registry.
        """
        try:
            from sndr.dispatcher.registry import PATCH_REGISTRY
            return list(PATCH_REGISTRY.items())
        except ImportError:
            return []

    # -----------------------------------------------------------------
    # Pin normalization
    # -----------------------------------------------------------------

    def _normalize_pin(self, version: str) -> str:
        """Map a full vllm version string to a pin manifest directory name.

        Examples:
            "0.22.1rc1.dev195+gda1daf40b" -> "0.22.1_da1daf40b"
            "0.21.1rc0+g626fa9bba566"      -> "0.21.1_626fa9bba"
        """
        # Best-effort heuristic: extract major.minor.patch + short SHA
        import re
        match = re.match(
            r"(\d+\.\d+\.\d+)(?:rc\d+)?(?:\.dev\d+)?\+g([0-9a-f]{6,})",
            version,
        )
        if match:
            base, sha = match.group(1), match.group(2)
            return f"{base}_{sha[:9]}"
        return version


__all__ = ["VllmEngine"]
