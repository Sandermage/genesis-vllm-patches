# SPDX-License-Identifier: Apache-2.0
"""Pin service — manifest-aware pin operations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from sndr.engines import get_engine
from sndr.exceptions import EngineUnsupportedError, PinManifestMissingError
from sndr.product_api.schemas.pins import PinManifestSummary, PinSummary


def list_pins(engine_name: str) -> list[PinSummary]:
    """List every pin with a manifest for the given engine."""
    try:
        EngineCls = get_engine(engine_name)
    except EngineUnsupportedError:
        return []

    # Locate the pins directory for this engine adapter.
    # We do not instantiate the engine (it may not be installed) — we read
    # the manifest files directly from the filesystem.
    module_path = Path(EngineCls.__module__.replace(".", "/")).parent
    pins_dir = (Path(__file__).parent.parent.parent / "engines" / engine_name / "pins").resolve()
    if not pins_dir.is_dir():
        return []

    summaries: list[PinSummary] = []
    for pin_dir in sorted(pins_dir.iterdir()):
        if not pin_dir.is_dir():
            continue
        manifest_path = pin_dir / "manifest.yaml"
        if not manifest_path.is_file():
            continue
        try:
            data = yaml.safe_load(manifest_path.read_text())
        except yaml.YAMLError:
            continue

        summaries.append(PinSummary(
            pin=pin_dir.name,
            status="staging",  # promotion state — TODO Phase 7
            full_version=data.get("pin", pin_dir.name),
            upstream_sha=data.get("upstream_sha"),
            generated_at=_parse_iso(data.get("generated_at")),
            has_manifest=True,
            has_drift=False,  # TODO Phase 7 drift check
        ))
    return summaries


def get_pin_manifest_summary(engine_name: str, pin: str) -> PinManifestSummary:
    """Return summary of one pin's manifest.

    Raises:
        PinManifestMissingError: If no manifest exists for this pin.
    """
    pins_dir = (Path(__file__).parent.parent.parent / "engines" / engine_name / "pins").resolve()
    manifest_path = pins_dir / pin / "manifest.yaml"
    if not manifest_path.is_file():
        raise PinManifestMissingError(
            f"No manifest for {engine_name}/{pin}",
            engine=engine_name,
            pin=pin,
        )

    data = yaml.safe_load(manifest_path.read_text())
    files = data.get("files", {})
    anchor_count = sum(
        len(file_data.get("anchors", {}))
        for file_data in files.values()
    )
    patch_ids: set[str] = set()
    for file_data in files.values():
        for anchor in file_data.get("anchors", {}).values():
            patch_ids.update(anchor.get("used_by_patches", []))

    return PinManifestSummary(
        pin=pin,
        file_count=len(files),
        anchor_count=anchor_count,
        patch_count=len(patch_ids),
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


__all__ = ["get_pin_manifest_summary", "list_pins"]
