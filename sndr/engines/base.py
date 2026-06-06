# SPDX-License-Identifier: Apache-2.0
"""EngineAdapter ABC — the contract every engine implementation must satisfy.

This is the **foundation of multi-engine support**. Adding a new engine
(sglang, TensorRT-LLM, etc.) means implementing this contract; the rest of
sndr's orchestration code is engine-agnostic.

Design principle: keep the contract minimal. Only methods that **must** be
called from engine-agnostic layers (kernel, dispatcher, apply, product_api)
go here. Engine-specific helpers stay private to the adapter package.

Lifecycle:

    1. Construction: EngineAdapter(config=SndrConfig)
    2. Bootstrap: adapter.bootstrap()
        - detect_version()
        - load_pin_manifest()
        - verify_manifest_md5sums()
        - discover_patches()
    3. Steady state: adapter.list_patches(), adapter.get_runtime_config(), etc.
    4. Shutdown: adapter.shutdown() (optional, for graceful cleanup)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sndr.config import SndrConfig


@dataclass(frozen=True)
class EngineInfo:
    """Engine-level metadata reported by the adapter."""

    name: str
    """Engine identifier (e.g. "vllm", "sglang")."""

    version: str
    """Full version string (e.g. "0.22.1rc1.dev195+gda1daf40b")."""

    install_root: Path | None
    """Filesystem path where the engine package is installed."""

    pin: str | None
    """Normalized pin identifier matching a manifest directory name."""

    supported: bool
    """True if this version is in the supported list for this sndr release."""


@dataclass(frozen=True)
class ModelProfile:
    """A snapshot of the currently-loaded model's properties.

    Different engines expose model metadata in different ways. The adapter
    normalizes them into this engine-agnostic shape.
    """

    architectures: tuple[str, ...]
    """HF config architectures list (e.g. ("Qwen3_5MoeForConditionalGeneration",))."""

    model_class: str
    """Normalized family identifier (e.g. "qwen3_5_moe")."""

    quant_format: str
    """Normalized quantization (e.g. "autoround_int4", "fp8", "fp16")."""

    kv_cache_dtype: str
    """KV cache storage format (e.g. "auto", "fp8", "turboquant_k8v4")."""

    is_moe: bool
    is_hybrid: bool
    is_turboquant: bool

    extra: dict[str, Any] = field(default_factory=dict)
    """Engine-specific extras (do not rely on these in agnostic code)."""


class EngineAdapter(ABC):
    """Abstract base class for engine adapters.

    Each engine (vllm, sglang, ...) provides a concrete subclass that wires
    engine-specific behavior into the engine-agnostic orchestrator.
    """

    name: str
    """Engine identifier. Subclass must override."""

    def __init__(self, config: "SndrConfig") -> None:
        self.config = config
        self._info: EngineInfo | None = None
        self._bootstrapped = False

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    @abstractmethod
    def detect_version(self) -> str:
        """Return the installed engine's full version string.

        Raises:
            EngineNotInstalledError: If the engine package is not importable.
            EngineVersionMismatchError: If the version cannot be parsed.
        """

    @abstractmethod
    def install_root(self) -> Path | None:
        """Return filesystem path of the engine's installation directory.

        Returns None if the engine package is not installed (e.g. running
        adapter for inspection without the engine present).
        """

    @abstractmethod
    def resolve_file(self, relative_path: str) -> Path | None:
        """Resolve a path relative to the engine's install root.

        Args:
            relative_path: Path relative to the engine package root, e.g.
                "v1/attention/ops/triton_turboquant_store.py".

        Returns:
            Absolute Path if the file exists; None otherwise.
        """

    def bootstrap(self) -> None:
        """Initialize the adapter and apply patches.

        Idempotent. Called once at sndr.init().
        """
        if self._bootstrapped:
            return

        version = self.detect_version()
        pin = self.config.engine_pin or self._normalize_pin(version)

        self._info = EngineInfo(
            name=self.name,
            version=version,
            install_root=self.install_root(),
            pin=pin,
            supported=self.is_pin_supported(pin),
        )

        self._on_bootstrap()
        self._bootstrapped = True

    def _on_bootstrap(self) -> None:
        """Subclass hook for additional bootstrap steps.

        Default implementation is a no-op. Subclasses override to:
          - Load per-pin manifest
          - Verify manifest md5sums against live files
          - Apply text patches
          - Install runtime hooks
        """

    def shutdown(self) -> None:
        """Optional graceful shutdown hook.

        Default implementation is a no-op. Subclasses override to release
        file handles, close connections, etc.
        """

    # ---------------------------------------------------------------------
    # Metadata and introspection
    # ---------------------------------------------------------------------

    @property
    def info(self) -> EngineInfo:
        """Return cached EngineInfo. Requires bootstrap() to have run."""
        if self._info is None:
            raise RuntimeError(
                f"EngineAdapter({self.name}) has not been bootstrapped. "
                "Call .bootstrap() before .info."
            )
        return self._info

    @abstractmethod
    def is_pin_supported(self, pin: str | None) -> bool:
        """Return True if the given pin has a manifest in this adapter."""

    @abstractmethod
    def list_supported_pins(self) -> tuple[str, ...]:
        """Return all pins for which this adapter has manifests."""

    # ---------------------------------------------------------------------
    # Runtime introspection
    # ---------------------------------------------------------------------

    @abstractmethod
    def get_runtime_config(self) -> dict[str, Any] | None:
        """Return engine's live runtime config (model_config, parallel_config, ...).

        Returns None if the engine has not been initialized yet.
        """

    @abstractmethod
    def get_model_profile(self) -> ModelProfile | None:
        """Return a snapshot of the currently-loaded model.

        Returns None if no model is loaded.
        """

    # ---------------------------------------------------------------------
    # Patch system
    # ---------------------------------------------------------------------

    @abstractmethod
    def list_patches(self) -> list[Any]:
        """Return all patches available for this engine.

        This includes community patches (in sndr/engines/<name>/patches/)
        AND engine-tier patches (via entry_points discovery if sndr_engine
        wheel is installed and license is valid).
        """

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _normalize_pin(self, version: str) -> str:
        """Normalize a raw version string into a manifest directory name.

        Default implementation keeps the raw version. Subclasses override
        to map e.g. "0.22.1rc1.dev195+gda1daf40b" → "0.22.1_da1daf40b".
        """
        return version


__all__ = ["EngineAdapter", "EngineInfo", "ModelProfile"]
