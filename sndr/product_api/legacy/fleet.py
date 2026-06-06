# SPDX-License-Identifier: Apache-2.0
"""Fleet overview — one summary across all registered GPU hosts.

The hybrid fleet model's *width* layer: instead of switching the GUI between
hosts one at a time, this fans out (concurrently, over SSH) to every registered
engine host and returns a compact per-host summary — reachability, the running
container(s) + vLLM version + served model(s), GPUs, and the count of live
Genesis patches. The operator sees the whole fleet at a glance, then drills into
one host's card for detail.

Read-only: it runs the same fixed discovery commands as :func:`ssh_client.discover_host`
plus a live engine probe; it mutates nothing. ``discover`` / ``probe`` are
injectable so the aggregation is unit-testable without live SSH.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional


def _ssh_target(profile: Any) -> dict[str, Any]:
    ssh_target = getattr(profile, "ssh_target", "") or ""
    user = getattr(profile, "ssh_user", "") or (ssh_target.split("@", 1)[0] if "@" in ssh_target else None)
    return {
        "host": getattr(profile, "host", ""),
        "port": getattr(profile, "ssh_port", 22) or 22,
        "user": user,
        "auth_method": getattr(profile, "ssh_auth", "agent") or "agent",
        "key_path": getattr(profile, "ssh_key_path", "") or "",
        "secret_id": f"ssh:{getattr(profile, 'id', '')}",
    }


def _is_engine_host(profile: Any) -> bool:
    return (getattr(profile, "transport", "") == "ssh") or bool(getattr(profile, "ssh_user", ""))


def summarize_host(
    profile: Any,
    *,
    discover: Callable[[dict[str, Any]], dict[str, Any]],
    probe: Callable[..., dict[str, Any]],
    resolve_key: Optional[Callable[[Any], Optional[str]]] = None,
) -> dict[str, Any]:
    """Discover one host and fold it into a compact fleet-summary row."""
    base = {
        "id": getattr(profile, "id", ""), "label": getattr(profile, "label", ""),
        "host": getattr(profile, "host", ""), "role": getattr(profile, "role", ""),
        "ssh_ok": False, "engines": [], "gpus": [], "gpu_count": 0,
        "arch": getattr(profile, "gpu_arch", "") or "", "interconnect": None,
        "active_patches": 0, "models": [], "vllm_version": None, "error": None,
    }
    try:
        disco = discover(_ssh_target(profile))
    except Exception as exc:  # noqa: BLE001 - one host failing must not break the fleet
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    base["ssh_ok"] = bool(disco.get("available")) and disco.get("error") is None
    if disco.get("error") and not disco.get("engines"):
        base["error"] = disco.get("error")

    key = resolve_key(profile) if resolve_key else None
    engines: list[dict[str, Any]] = []
    models: list[str] = []
    patches = 0
    for e in disco.get("engines", []) or []:
        port = e.get("host_port")
        reachable, version, e_models = False, None, []
        if port:
            try:
                pr = probe(getattr(profile, "host", ""), port, api_key=key or None)
                reachable = bool(pr.get("reachable"))
                version = pr.get("version")
                e_models = pr.get("models", []) or []
            except Exception:  # noqa: BLE001 - probe failure ≠ discovery failure
                pass
        flags = e.get("genesis_flags") or []
        patches += len(flags)
        for m in e_models:
            if m not in models:
                models.append(m)
        engines.append({
            "container": e.get("container"), "port": port, "reachable": reachable,
            "version": version, "models": e_models, "patches": len(flags),
        })
        if version and not base["vllm_version"]:
            base["vllm_version"] = version
    base["engines"] = engines
    base["models"] = models
    base["active_patches"] = patches
    base["gpus"] = [
        {"name": g.get("name"), "memory_total_mib": g.get("memory_total_mib"),
         "arch": g.get("arch"), "utilization": g.get("utilization")}
        for g in disco.get("gpus", []) or []
    ]
    base["gpu_count"] = len(base["gpus"])
    if base["gpus"] and not base["arch"]:
        base["arch"] = base["gpus"][0].get("arch") or ""
    ic = disco.get("interconnect")
    if isinstance(ic, dict):
        base["interconnect"] = "NVLink" if ic.get("has_nvlink") else ic.get("worst_link")
    return base


def collect_fleet_overview(
    profiles: list[Any],
    *,
    discover: Callable[[dict[str, Any]], dict[str, Any]],
    probe: Callable[..., dict[str, Any]],
    resolve_key: Optional[Callable[[Any], Optional[str]]] = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Concurrently summarise every engine host in the registry (read-only)."""
    targets = [p for p in profiles if _is_engine_host(p)]
    if not targets:
        return []

    def one(p: Any) -> dict[str, Any]:
        return summarize_host(p, discover=discover, probe=probe, resolve_key=resolve_key)

    workers = max(1, min(max_workers, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, targets))


__all__ = ["collect_fleet_overview", "summarize_host"]
