# SPDX-License-Identifier: Apache-2.0
"""Daemon bridge to the spec-decode request router + functional artifacts.

Exposes, through the admin/control plane (the daemon), the SAME workload→profile
classification the data-plane gateway uses, plus each profile's bench-validated
per-workload economics. The GUI and any client therefore share ONE source of
truth for "how is this chat request classified, and what is this backend good
at" — instead of the knowledge living only inside the (often un-deployed)
gateway. Read-only; never mutates the engine.

Graceful degradation: the routing brain lives in sndr_core's ``spec_decode``
integration. If it — or its artifacts — is not importable (the GUI ships with
sndr_core, but the engine/patch layer may be absent in a given deployment), every
entry point returns an ``{"available": False, "reason": ...}`` envelope rather
than raising, so the GUI degrades to a clean "routing unavailable" state.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional


def _modules() -> "tuple[Any, Any]":
    """Lazily import the router + artifact modules; (None, None) if absent."""
    try:
        from vllm.sndr_core.integrations.spec_decode import (  # noqa: PLC0415
            functional_artifact,
            request_router,
        )
        return request_router, functional_artifact
    except Exception:  # noqa: BLE001 — engine/patch layer may be absent
        return None, None


def available() -> bool:
    rr, fa = _modules()
    return rr is not None and fa is not None


def _artifact_dirs(fa: Any) -> list[Path]:
    dirs: list[Path] = []
    shipped = getattr(fa, "_ARTIFACTS_DIR", None)
    if shipped and Path(shipped).is_dir():
        dirs.append(Path(shipped))
    extra = (os.environ.get("SNDR_SPEC_DECODE_ARTIFACTS_DIR")
             or os.environ.get("GENESIS_SPEC_DECODE_ARTIFACTS_DIR") or "").strip()
    if extra and Path(extra).is_dir():
        dirs.append(Path(extra))
    return dirs


def _k_of(art: Any) -> Optional[int]:
    m = re.search(r"k(\d+)$", art.profile or "")
    if m:
        return int(m.group(1))
    kv = art.kv_plan or {}
    val = kv.get("mtp_k") or kv.get("num_speculative_tokens")
    return int(val) if isinstance(val, (int, float)) else None


def _summary(art: Any) -> dict[str, Any]:
    m = art.metrics or {}
    return {
        "profile": art.profile,
        "model_id": art.model_id,
        "decision": art.decision,
        "k": _k_of(art),
        "allowed_workloads": list(art.allowed_workloads),
        "denied_workloads": list(art.denied_workloads),
        "workload_classes": list(art.workload_classes),
        "delta_tps_per_class": m.get("delta_tps_per_class", {}),
        "profile_tps_per_class": m.get("profile_tps_per_class", {}),
        "baseline_tps_per_class": m.get("baseline_tps_per_class", {}),
        "profile_delta_global": m.get("profile_delta_global"),
        "acceptance_mean": (m.get("acceptance") or {}).get("mean"),
        "vram_free_mib_min": m.get("vram_free_mib_min"),
        "vllm_pin": art.vllm_pin,
        "notes": art.notes,
    }


def _iter_artifacts(fa: Any):
    seen: set[str] = set()
    for d in _artifact_dirs(fa):
        for path in sorted(d.glob("*.json")):
            try:
                art = fa.read(path)
            except Exception:  # noqa: BLE001 — skip malformed
                continue
            if art.profile in seen:
                continue
            seen.add(art.profile)
            yield art


def list_artifacts() -> dict[str, Any]:
    """All known bench-validated profiles with their per-workload economics."""
    rr, fa = _modules()
    if fa is None:
        return {"available": False, "reason": "spec_decode integration not importable", "artifacts": []}
    return {"available": True, "artifacts": [_summary(a) for a in _iter_artifacts(fa)]}


def _load_artifact(fa: Any, profile: str) -> Optional[Any]:
    for art in _iter_artifacts(fa):
        if art.profile == profile:
            return art
    return None


def active_profile() -> dict[str, Any]:
    """The profile the operator considers live.

    Source priority: explicit ``SNDR_ACTIVE_PROFILE`` env (the operator knows
    what is actually running) → the sole validated profile → none.
    """
    rr, fa = _modules()
    if fa is None:
        return {"available": False, "reason": "spec_decode integration not importable"}
    arts = list_artifacts()["artifacts"]
    env = (os.environ.get("SNDR_ACTIVE_PROFILE") or "").strip()
    if env:
        match = next((a for a in arts if a["profile"] == env), None)
        return {"available": True, "profile": env, "source": "env",
                "artifact": match, "candidates": [a["profile"] for a in arts]}
    if len(arts) == 1:
        return {"available": True, "profile": arts[0]["profile"], "source": "sole-artifact",
                "artifact": arts[0], "candidates": [arts[0]["profile"]]}
    return {"available": True, "profile": None, "source": "unset",
            "artifact": None, "candidates": [a["profile"] for a in arts]}


def classify(*, signals: dict[str, Any], profile: Optional[str] = None) -> dict[str, Any]:
    """Run the gateway's exact router over explicit request signals.

    ``signals`` carries the OpenAI-shaped hints — ``response_format``,
    ``tool_choice`` and/or ``workload_class`` — and returns the resolved profile,
    whether it was accepted (vs a conservative fallback) and the bench-measured
    TPS delta the classified workload would see on that profile.
    """
    rr, fa = _modules()
    if rr is None:
        return {"available": False, "reason": "spec_decode integration not importable"}
    prof = profile or active_profile().get("profile")
    art = _load_artifact(fa, prof) if prof else None

    req: dict[str, Any] = {}
    if signals.get("response_format"):
        req["response_format"] = signals["response_format"]
    if signals.get("tool_choice"):
        req["tool_choice"] = signals["tool_choice"]
    if signals.get("workload_class"):
        req["extra_body"] = {"workload_class": signals["workload_class"]}

    sel = rr.select_profile(request=req, artifact=art, fallback_profile="default (MTP off)")
    expected = None
    if art and sel.workload_class:
        expected = (art.metrics.get("delta_tps_per_class", {}) or {}).get(sel.workload_class)
    return {
        "available": True,
        "profile": sel.profile,
        "signal": sel.signal,
        "workload_class": sel.workload_class,
        "accepted": sel.accepted,
        "reason": sel.reason,
        "expected_delta_tps": expected,
        "active_profile": prof,
    }
