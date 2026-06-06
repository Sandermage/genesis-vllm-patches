# SPDX-License-Identifier: Apache-2.0
"""Engine service — queries engine adapters for status, info, capabilities."""
from __future__ import annotations

from sndr.engines import get_engine, list_engines
from sndr.exceptions import EngineNotInstalledError, EngineUnsupportedError
from sndr.product_api.schemas.engines import EngineDetail, EngineSummary

# Display names for known engines.
_DISPLAY_NAMES = {
    "vllm": "vLLM",
    "sglang": "SGLang",
}


def list_engine_summaries() -> list[EngineSummary]:
    """Return summary of every registered engine."""
    summaries: list[EngineSummary] = []
    for name in list_engines():
        summaries.append(_build_summary(name))
    return summaries


def get_engine_detail(name: str) -> EngineDetail:
    """Return full detail for one engine.

    Raises:
        EngineUnsupportedError: If name is not a registered engine.
        EngineNotInstalledError: If the engine package is not importable.
    """
    summary = _build_summary(name)
    EngineCls = get_engine(name)

    # Try to instantiate; if it fails, return summary-level info only.
    try:
        from sndr.config import SndrConfig
        # Best-effort: build a minimal config; engine may not be installed.
        config = SndrConfig.from_env() if name == "vllm" else _minimal_config(name)
        engine = EngineCls(config=config)
        supported_pins = list(engine.list_supported_pins())
        install_root = str(engine.install_root()) if engine.install_root() else None
        patches = engine.list_patches()
        # Heuristic counts by tier (community/engine).
        community = sum(1 for _, meta in patches if isinstance(meta, dict) and meta.get("tier", "community") == "community")
        engine_count = sum(1 for _, meta in patches if isinstance(meta, dict) and meta.get("tier") == "engine")
    except (EngineNotInstalledError, Exception):  # noqa: BLE001
        supported_pins = []
        install_root = None
        community = 0
        engine_count = 0

    return EngineDetail(
        **summary.model_dump(),
        supported_pins=supported_pins,
        patch_count_community=community,
        patch_count_engine=engine_count,
        install_root=install_root,
        capabilities={
            "multi_pin": bool(supported_pins),
            "drift_detection": bool(supported_pins),
            "license_gated_patches": True,
        },
    )


def _build_summary(name: str) -> EngineSummary:
    """Build a summary entry for one engine."""
    notes: list[str] = []
    active = True
    version: str | None = None
    pin: str | None = None

    try:
        EngineCls = get_engine(name)
        # Try to detect version, but do not fail summary if engine not installed.
        try:
            from sndr.config import SndrConfig
            config = SndrConfig.from_env() if name == "vllm" else _minimal_config(name)
            engine = EngineCls(config=config)
            version = engine.detect_version()
            pin = engine._normalize_pin(version)  # type: ignore[attr-defined]
        except EngineNotInstalledError:
            notes.append("Engine package is not installed.")
            active = False
        except Exception as e:  # noqa: BLE001
            notes.append(f"Detection failed: {e}")
    except EngineUnsupportedError:
        active = False
        notes.append("Engine adapter not registered.")

    return EngineSummary(
        name=name,
        display_name=_DISPLAY_NAMES.get(name, name),
        active=active,
        version=version,
        pin=pin,
        container_count=0,  # filled by container service in routes layer
        notes=notes,
    )


def _minimal_config(engine_name: str):
    """Build a minimal SndrConfig for inspection-only calls."""
    from sndr.config import SndrConfig  # local import to avoid cycles
    import os
    # We intentionally do not modify os.environ; we synthesize a config object.
    return SndrConfig(
        engine=engine_name,  # type: ignore[arg-type]
        engine_pin=None,
        sndr_home=SndrConfig.from_env().sndr_home,
        config_path=None,
        strict_drift=False,
        strict_apply=False,
        strict_deps=False,
        audit_on_apply=False,
        log_level="WARNING",
        otel_endpoint=None,
    )


__all__ = ["get_engine_detail", "list_engine_summaries"]
