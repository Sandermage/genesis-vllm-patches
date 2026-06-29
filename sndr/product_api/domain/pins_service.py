# SPDX-License-Identifier: Apache-2.0
"""Pin service — manifest-aware pin operations."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml

from sndr.engines import get_engine
from sndr.exceptions import EngineUnsupportedError, PinManifestMissingError
from sndr.product_api.schemas.pins import PinManifestSummary, PinSummary


def list_pins(engine_name: str) -> list[PinSummary]:
    """List every committed pin for the engine with REAL promotion status + drift.

    A pin is included if it ships ANY recognized artifact — a legacy
    ``manifest.yaml`` OR the current ``anchors.json`` + ``drift.rej.json`` pair.
    (The current pins ship only the latter, so the old manifest-only listing hid
    the live pins entirely while fabricating ``staging``/no-drift for the rest.)

    Status is derived from the canonical ``vllm_pin_required`` the model configs
    declare (``updater.supported_pins`` — most-common first = current):

    * ``current``  — full version == the declared canonical pin
    * ``previous`` — full version is another declared (supported) pin
    * ``staging``  — committed on disk but not declared current/previous

    ``has_drift`` reads the pin's ``drift.rej.json`` (a non-empty
    ``genuine_anchor_drift`` = real anchor drift the operator must review). Reads
    are defensive: a missing/corrupt artifact degrades that field rather than
    dropping the pin or raising.
    """
    try:
        # Validate the engine is registered; the adapter class itself is
        # not needed — pins are read straight from the filesystem below.
        get_engine(engine_name)
    except EngineUnsupportedError:
        return []

    # Locate the pins directory for this engine adapter. We do not instantiate
    # the engine (it may not be installed) — we read the artifacts directly.
    pins_dir = (Path(__file__).parent.parent.parent / "engines" / engine_name / "pins").resolve()
    if not pins_dir.is_dir():
        return []

    canonical, supported = _declared_pins(engine_name)

    summaries: list[PinSummary] = []
    for pin_dir in sorted(pins_dir.iterdir()):
        if not pin_dir.is_dir():
            continue
        manifest = _read_yaml(pin_dir / "manifest.yaml")
        anchors = _read_json(pin_dir / "anchors.json")
        drift = _read_json(pin_dir / "drift.rej.json")
        if manifest is None and anchors is None and drift is None:
            continue  # no recognized pin artifact

        full_version = (
            (manifest or {}).get("full_version")
            or (manifest or {}).get("pin")
            or (drift or {}).get("pin")
            or pin_dir.name
        )
        summaries.append(PinSummary(
            pin=pin_dir.name,
            status=_status_for(full_version, canonical, supported),
            full_version=full_version,
            upstream_sha=(manifest or {}).get("upstream_sha") or _sha_from_version(full_version),
            generated_at=_parse_iso(
                (manifest or {}).get("generated_at") or (anchors or {}).get("generated_at")
            ),
            has_manifest=manifest is not None,
            has_drift=bool((drift or {}).get("genuine_anchor_drift")),
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


def _read_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


_SHA_RE = re.compile(r"\+g([0-9a-fA-F]+)")


def _sha_from_version(version: str | None) -> str | None:
    """Pull the upstream short SHA out of a ``...+g<sha>`` version string."""
    if not version:
        return None
    match = _SHA_RE.search(version)
    return match.group(1) if match else None


def _declared_pins(engine_name: str) -> tuple[str | None, set[str]]:
    """``(canonical_current, {supported_pins})`` from the engine's declared
    ``vllm_pin_required`` values (most-common first = current). Falls back to
    ``(None, set())`` for non-vllm engines or when the legacy updater can't be
    read — status then degrades to ``staging`` rather than crashing the listing."""
    if engine_name != "vllm":
        return None, set()
    try:
        from sndr.product_api.legacy.updater import supported_pins
        pins = [p for p in supported_pins() if p and p != "null"]
    except Exception:  # noqa: BLE001 — best-effort; never sink the listing
        return None, set()
    return (pins[0] if pins else None), set(pins)


def _status_for(full_version: str, canonical: str | None, supported: set[str]) -> str:
    if canonical and full_version == canonical:
        return "current"
    if full_version in supported:
        return "previous"
    return "staging"


__all__ = ["get_pin_manifest_summary", "list_pins"]
