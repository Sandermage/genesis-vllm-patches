# SPDX-License-Identifier: Apache-2.0
"""SGLang engine adapter.

A functional second-engine implementation of the :class:`EngineAdapter`
contract, mirroring the vLLM adapter's structure. It proves the multi-engine
abstraction with a real, non-vllm engine:

  - version detection + pin normalization (``import sglang``),
  - install-root resolution + relative file resolution,
  - per-pin manifest discovery (``engines/sglang/pins/<pin>/manifest.yaml``),
  - community patch discovery (``engines/sglang/patches/``).

Runtime introspection (``get_runtime_config`` / ``get_model_profile``) returns
None for now: SGLang exposes no engine-agnostic global config accessor, so
those land alongside the first ported SGLang pin. Everything else is live —
the adapter detects a real SGLang install and reports its pins/patches, or
degrades gracefully (None / empty) when SGLang is absent.

No SGLang pins or patches ship yet, so ``list_supported_pins()`` and
``list_patches()`` return empty until the first manifest/patch is added under
this package — at which point they light up with no adapter change.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from sndr.engines.base import EngineAdapter, ModelProfile
from sndr.exceptions import EngineNotInstalledError

log = logging.getLogger("sndr.engines.sglang")

_HERE = Path(__file__).resolve().parent
# major.minor.patch (+ optional rcN / .devN) + short git sha → "base_sha9"
_PIN_RE = re.compile(r"(\d+\.\d+\.\d+)(?:rc\d+)?(?:\.dev\d+)?\+g([0-9a-f]{6,})")


class SglangEngine(EngineAdapter):
    """Engine adapter for SGLang (community tier)."""

    name = "sglang"

    # -----------------------------------------------------------------
    # Required EngineAdapter API
    # -----------------------------------------------------------------

    def detect_version(self) -> str:
        """Return the installed SGLang version string."""
        try:
            import sglang
        except ImportError as e:
            raise EngineNotInstalledError(
                "sglang package is not installed",
                engine="sglang",
                cause=str(e),
            ) from e
        return str(getattr(sglang, "__version__", "unknown"))

    def install_root(self) -> Path | None:
        """Return the directory containing the sglang package, or None."""
        try:
            import sglang
        except ImportError:
            return None
        file = getattr(sglang, "__file__", None)
        return Path(file).parent if file else None

    def resolve_file(self, relative_path: str) -> Path | None:
        """Resolve a path relative to the sglang install root (or None)."""
        root = self.install_root()
        if root is None:
            return None
        candidate = (root / relative_path).resolve()
        return candidate if candidate.exists() else None

    def is_pin_supported(self, pin: str | None) -> bool:
        """True if ``engines/sglang/pins/<pin>/manifest.yaml`` exists."""
        if not pin:
            return False
        return (self._pins_dir() / pin / "manifest.yaml").is_file()

    def list_supported_pins(self) -> tuple[str, ...]:
        """All pins with a manifest under ``engines/sglang/pins/``."""
        pins_dir = self._pins_dir()
        if not pins_dir.is_dir():
            return ()
        return tuple(sorted(
            p.name for p in pins_dir.iterdir()
            if p.is_dir() and (p / "manifest.yaml").is_file()
        ))

    def get_runtime_config(self) -> dict[str, Any] | None:
        """SGLang has no engine-agnostic global runtime-config accessor yet;
        live introspection lands with the first ported SGLang pin."""
        return None

    def get_model_profile(self) -> ModelProfile | None:
        """No standardized SGLang model-profile source yet (see above)."""
        return None

    def list_patches(self) -> list[Any]:
        """Community SGLang patches under ``engines/sglang/patches/``.

        Returns the module stems discovered there (none yet). The directory
        is enumerated so the surface lights up as soon as the first patch is
        added, with no adapter change.
        """
        patches_dir = self._patches_dir()
        if not patches_dir.is_dir():
            return []
        return [
            f.stem
            for f in sorted(patches_dir.rglob("*.py"))
            if not f.name.startswith("_")
        ]

    # -----------------------------------------------------------------
    # Pin normalization + private helpers
    # -----------------------------------------------------------------

    def _normalize_pin(self, version: str) -> str:
        """Map a full sglang version into a pin manifest directory name,
        e.g. ``0.4.6rc1.dev10+gabcdef123`` → ``0.4.6_abcdef123``."""
        match = _PIN_RE.match(version)
        if match:
            base, sha = match.group(1), match.group(2)
            return f"{base}_{sha[:9]}"
        return version

    def _pins_dir(self) -> Path:
        return _HERE / "pins"

    def _patches_dir(self) -> Path:
        return _HERE / "patches"


__all__ = ["SglangEngine"]
