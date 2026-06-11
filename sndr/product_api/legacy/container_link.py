# SPDX-License-Identifier: Apache-2.0
"""Link a RUNNING container back to the preset/config that defines it, and detect
drift between the two.

This is the "everything is connected" layer: when you look at a running engine
container, the GUI can tell you which preset/config it came from and whether its
live runtime (image + GENESIS_* patch flags) still matches that config — so an
edit "here" (the container) is understood against the source "there" (the config).

Two link paths, best-effort and read-only:
  1. an explicit ``sndr.preset`` container label (authoritative, written by future
     GUI launches);
  2. a name match against each preset's resolved ``container_name`` (covers
     containers launched today via start-scripts, which carry no label yet).

The pure helpers (:func:`resolve_preset`, :func:`compute_drift`, :func:`parse_env`)
are unit-tested without touching the registry; the registry wiring lives in the
cached :func:`source_report`.
"""
from __future__ import annotations

import functools
import re
import threading
import time
from typing import Any, Callable, Optional

_PRESET_LABEL = "sndr.preset"
# A live Genesis patch flag in a container's env (GENESIS_ENABLE_P82, PN95_…).
_PATCH_ENV_RE = re.compile(r"^(GENESIS_|PN\d)")
_TRUTHY = {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def _sndr_enable_patch_flags() -> frozenset[str]:
    """The registry's ``SNDR_ENABLE_*`` patch flags (e.g. PN282, PN283).

    A handful of patches use the ``SNDR_ENABLE_`` prefix instead of
    ``GENESIS_ENABLE_``. Those must be recognised as live patch flags, but the
    same-prefix daemon gates ``SNDR_ENABLE_APPLY`` / ``SNDR_ENABLE_EXEC`` are
    NOT patches and must never appear in the live patch overlay. Matching the
    real registry env_flag set (rather than a prefix guess) keeps both right and
    stays correct if more SNDR_ENABLE_* flags are added either way.
    """
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except Exception:  # noqa: BLE001 — registry optional in some contexts
        return frozenset()
    return frozenset(
        flag for entry in PATCH_REGISTRY.values()
        if isinstance((flag := entry.get("env_flag")), str)
        and flag.startswith("SNDR_ENABLE_")
    )


def _is_patch_flag(key: str) -> bool:
    """Whether a container env var name is a Genesis patch flag."""
    return bool(_PATCH_ENV_RE.match(key)) or key in _sndr_enable_patch_flags()

_INDEX_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_INDEX_TTL = 120.0
_INDEX_LOCK = threading.Lock()


# ─── pure helpers ──────────────────────────────────────────────────────


def parse_env(env_list: Optional[list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in env_list or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            out[k] = v
    return out


def build_preset_index(list_presets: Callable[[], list[str]],
                       container_name_of: Callable[[str], Optional[str]]) -> dict[str, str]:
    """Map a container_name → preset_id for every preset that declares one."""
    index: dict[str, str] = {}
    for pid in list_presets():
        try:
            cn = container_name_of(pid)
        except Exception:
            continue
        if cn:
            index[cn] = pid
    return index


def resolve_preset(name: str, labels: Optional[dict[str, str]],
                   index: dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    """Return (preset_id, linked_by) where linked_by is "label" | "name" | None."""
    label = (labels or {}).get(_PRESET_LABEL)
    if label:
        return str(label), "label"
    if name in index:
        return index[name], "name"
    return None, None


def live_patches(inspect: dict[str, Any]) -> list[dict[str, str]]:
    """Genesis patch flags that are actually ON in the running container's env.

    Answers "which patches are LIVE right now" — connecting the engine container
    to the patch registry, independent of what the preset declares."""
    env = parse_env((inspect.get("Config") or {}).get("Env"))
    out = [{"flag": k, "value": v} for k, v in env.items()
           if _is_patch_flag(k) and str(v).strip().lower() in _TRUTHY]
    return sorted(out, key=lambda d: d["flag"])


def reconcile_patches(expected_env: dict[str, str], inspect: dict[str, Any]) -> dict[str, list[str]]:
    """Compare the patches the config DECLARES against what's LIVE in the engine.

    - in_sync: config wants on AND engine has on
    - missing: config wants on BUT engine has off/absent (a silent feature regression)
    - extra:   on in the engine BUT not declared by the config (drift the other way)"""
    declared = {k for k in (expected_env or {}) if _is_patch_flag(k)}
    declared_on = {k for k in declared if str((expected_env or {})[k]).strip().lower() in _TRUTHY}
    live = {p["flag"] for p in live_patches(inspect)}
    return {
        "in_sync": sorted(declared_on & live),
        "missing": sorted(declared_on - live),
        "extra": sorted(live - declared),
    }


def _same_image_ref(expected: str, actual: str) -> bool:
    """Whether two image refs denote the same image for drift purposes.

    A tag ref (``repo:nightly-abc``) and a digest ref (``repo@sha256:…``) to the
    same repo are NOT string-comparable — the catalog may pin by digest while the
    launcher deploys the equivalent tag. Treat same-repo as a match when either
    side is digest-pinned, so we never raise a phantom 'image' drift on a pure
    tag-vs-digest representation difference (genuine repo/tag changes still flag,
    and functional drift is caught by the GENESIS_* patch diff regardless)."""
    if expected == actual:
        return True
    exp_repo = expected.split("@", 1)[0].rsplit(":", 1)[0]
    act_repo = actual.split("@", 1)[0].rsplit(":", 1)[0]
    if exp_repo == act_repo and ("@sha256:" in expected or "@sha256:" in actual):
        return True
    return False


def compute_drift(expected_image: str, expected_env: dict[str, str],
                  inspect: dict[str, Any]) -> list[dict[str, Any]]:
    """Diff a running container's runtime against its preset's declared config.

    Surfaces image mismatch and every GENESIS_*/PN* flag that is missing or
    differs from what the config declares — i.e. config drift."""
    cfg = inspect.get("Config") or {}
    actual_image = str(cfg.get("Image") or "")
    actual_env = parse_env(cfg.get("Env"))
    drift: list[dict[str, Any]] = []
    if expected_image and actual_image and not _same_image_ref(expected_image, actual_image):
        drift.append({"field": "image", "expected": expected_image, "actual": actual_image, "kind": "image"})
    for key, exp in (expected_env or {}).items():
        act = actual_env.get(key)
        if act is None:
            drift.append({"field": key, "expected": str(exp), "actual": None, "kind": "missing"})
        elif str(act) != str(exp):
            drift.append({"field": key, "expected": str(exp), "actual": str(act), "kind": "changed"})
    return drift


# ─── registry wiring (cached) ──────────────────────────────────────────


def _preset_index() -> dict[str, str]:
    with _INDEX_LOCK:
        cached = _INDEX_CACHE["data"]
        if cached is not None and (time.time() - _INDEX_CACHE["ts"]) < _INDEX_TTL:
            return cached
    from . import deployment, presets

    def container_name_of(pid: str) -> Optional[str]:
        return deployment._parameters(deployment._resolve_cfg(pid)).get("container_name")  # noqa: SLF001

    try:
        index = build_preset_index(lambda: list(presets.list_presets()), container_name_of)
    except Exception:
        index = {}
    with _INDEX_LOCK:
        _INDEX_CACHE["data"] = index
        _INDEX_CACHE["ts"] = time.time()
    return index


def invalidate_preset_index() -> None:
    with _INDEX_LOCK:
        _INDEX_CACHE["data"] = None


def source_report(name: str, inspect: dict[str, Any]) -> dict[str, Any]:
    """Resolve a container's source preset + drift, using THIS daemon's catalog."""
    labels = (inspect.get("Config") or {}).get("Labels") or {}
    preset_id, linked_by = resolve_preset(name, labels, _preset_index())
    patches = live_patches(inspect)
    report: dict[str, Any] = {
        "container": name, "preset_id": preset_id, "linked_by": linked_by,
        "preset_title": None, "drift": [], "drift_count": 0,
        "live_patches": patches, "live_patch_count": len(patches),
        # Identity stamped on the container by the launcher (no engine api-key
        # needed): served model, pin, role. None when the engine predates the
        # labels or was launched by hand.
        "served_model": labels.get("sndr.served-model") or None,
        "pin": labels.get("sndr.pin") or None,
        "role": labels.get("sndr.role") or None,
    }
    if not preset_id:
        return report
    try:
        from . import deployment
        cfg = deployment._resolve_cfg(preset_id)  # noqa: SLF001
        params = deployment._parameters(cfg)  # noqa: SLF001
        expected_env = deployment._genesis_env(cfg)  # noqa: SLF001
        report["preset_title"] = getattr(cfg, "title", None) or preset_id
        drift = compute_drift(str(params.get("image") or ""), expected_env, inspect)
        report["drift"] = drift
        report["drift_count"] = len(drift)
        report["patch_sync"] = reconcile_patches(expected_env, inspect)
    except Exception as exc:  # unknown/foreign preset — link without drift
        report["error"] = str(exc)
    return report


__all__ = [
    "parse_env", "build_preset_index", "resolve_preset", "compute_drift",
    "live_patches", "reconcile_patches", "source_report", "invalidate_preset_index",
]
