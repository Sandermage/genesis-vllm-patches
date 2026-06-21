# SPDX-License-Identifier: Apache-2.0
"""Read-only status for the per-pin anchor source-of-truth manifests.

Surfaces the anchor-SoT subsystem (``sndr/engines/vllm/pins/<pin>/anchors.json``)
to the GUI: which pins have a generated manifest, its pin versions, file/patch/
anchor counts, schema validity, which manifest is ACTIVE for the running vLLM,
and — for the active one — live drift (anchors verified against the installed
vLLM source via the canonical ``verify_manifest_against_source``).

Purely read-only: it loads + summarises existing manifests and reuses the
engine's own validators. It never generates or mutates a manifest, so it stays
decoupled from the (still-evolving) generation pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _running_vllm() -> Optional[str]:
    try:
        import vllm  # type: ignore

        return getattr(vllm, "__version__", None)
    except Exception:  # noqa: BLE001 - vllm may be absent outside the engine image
        return None


def _pins_dir() -> Optional[Path]:
    try:
        import sndr.engines.vllm as ev  # type: ignore

        d = Path(ev.__file__).resolve().parent / "pins"
        return d if d.is_dir() else None
    except Exception:  # noqa: BLE001
        return None


def _counts(manifest: dict) -> dict[str, int]:
    files = manifest.get("files") or {}
    patches = anchors = 0
    for entry in files.values():
        for patch in ((entry or {}).get("patches") or {}).values():
            patches += 1
            anchors += len((patch or {}).get("anchors") or {})
    return {"files": len(files), "patches": patches, "anchors": anchors}


def _check_drift(manifest: dict, *, limit: int) -> dict[str, Any]:
    """Verify the manifest's recorded md5s/anchors against the live (pristine)
    vLLM source installed in this process. drift_count 0 = manifest matches
    reality. Best-effort: any failure reports checked=False, never raises."""
    try:
        from sndr.engines.vllm.wiring.anchor_manifest import verify_manifest_against_source
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "reason": f"verifier unavailable: {exc}"}
    try:
        import vllm  # type: ignore

        vllm_dir = Path(vllm.__file__).resolve().parent
    except Exception:  # noqa: BLE001
        return {"checked": False, "reason": "vllm not importable"}

    def _loader(rel_path: str) -> Optional[str]:
        try:
            return (vllm_dir / rel_path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None

    try:
        errors = verify_manifest_against_source(manifest, _loader)
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "reason": f"verify failed: {exc}"}
    return {
        "checked": True,
        "in_sync": not errors,
        "drift_count": len(errors),
        "details": errors[:limit],
        "truncated": len(errors) > limit,
    }


def manifest_status(*, drift: bool = True, drift_limit: int = 50) -> dict[str, Any]:
    """Summarise every per-pin anchor manifest on disk, mark the one active for the
    running vLLM, and (optionally) report live drift for the active manifest."""
    from sndr.engines.vllm.wiring.anchor_manifest import validate_manifest_schema

    running = _running_vllm()
    pins_dir = _pins_dir()
    manifests: list[dict[str, Any]] = []
    active_manifest: Optional[dict] = None

    if pins_dir is not None:
        for pin_dir in sorted(pins_dir.iterdir()):
            f = pin_dir / "anchors.json"
            if not f.is_file():
                continue
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                manifests.append({"pin_dir": pin_dir.name, "error": f"unreadable: {exc}"})
                continue
            pins = m.get("pins") or {}
            schema_errors = validate_manifest_schema(m)
            is_active = bool(running and pins.get("vllm") == running)
            manifests.append({
                "pin_dir": pin_dir.name,
                "vllm": pins.get("vllm"),
                "genesis": pins.get("genesis"),
                "generated_at": m.get("generated_at"),
                "generated_by": m.get("generated_by"),
                "manifest_version": m.get("manifest_version"),
                "schema_valid": not schema_errors,
                "schema_errors": schema_errors[:10],
                "active": is_active,
                **_counts(m),
            })
            if is_active:
                active_manifest = m

    out: dict[str, Any] = {
        "available": pins_dir is not None,
        "running_vllm": running,
        "manifest_count": len(manifests),
        "manifests": manifests,
    }
    if not drift:
        out["drift"] = {"checked": False, "reason": "drift check disabled"}
    elif active_manifest is None:
        reason = "no manifest matches the running vLLM pin" if running else "running vLLM version unknown"
        out["drift"] = {"checked": False, "reason": reason}
    else:
        out["drift"] = _check_drift(active_manifest, limit=drift_limit)
    return out
