# SPDX-License-Identifier: Apache-2.0
"""Bridge a live vLLM engine's served model to the SNDR V2 catalog.

``engine_client`` already detects *what* a running engine serves (the ``/v1/models``
ids). This module adds the *meaning*: it resolves each served-model id to its
catalog ``ModelDef`` — inherent capabilities (attention arch, tool/reasoning
parsers, spec-decode, KV dtype), hardware requirements, the required vLLM pin —
and the presets that run it, plus the richer per-model metadata vLLM reports
(``max_model_len``, ``root``). Read-only; every catalog lookup is best-effort so a
catalog hiccup never breaks model detection.
"""
from __future__ import annotations

import json
import urllib.error
from typing import Any, Optional

from . import engine_client


def _candidate_keys(md: Any) -> list[tuple[int, str]]:
    """Match keys for one ModelDef, by priority (lower = stronger match).

    A vLLM server reports either its ``--served-model-name`` (priority 0) or, when
    that flag is absent, the ``--model`` path (priority 1, by full path or
    basename). The catalog id is the weakest fallback (priority 2)."""
    keys: list[tuple[int, str]] = []
    served = (md.served_model_name or "").strip()
    if served:
        keys.append((0, served.lower()))
    path = (md.model_path or "").strip().rstrip("/")
    if path:
        keys.append((1, path.lower()))
        keys.append((1, path.split("/")[-1].lower()))
    if md.id:
        keys.append((2, md.id.lower()))
    return keys


def match_catalog_model(served_id: str) -> Optional[dict[str, Any]]:
    """Resolve a live served-model id to its catalog ModelDef + the presets that
    run it. Returns ``None`` when nothing in the V2 catalog serves that id."""
    sid = (served_id or "").strip().lower()
    if not sid:
        return None

    from sndr.model_configs import registry_v2

    # Collect every catalog model that could serve this id, with its best priority.
    matched: list[tuple[int, str, Any]] = []
    for model_id in registry_v2.list_models():
        try:
            md = registry_v2.load_model(model_id)
        except Exception:  # noqa: BLE001 - a single bad model must not break detection
            continue
        best: Optional[int] = None
        for prio, key in _candidate_keys(md):
            if key == sid and (best is None or prio < best):
                best = prio
        if best is not None:
            matched.append((best, model_id, md))

    if not matched:
        return None

    matched.sort(key=lambda item: (item[0], item[1]))
    prio, model_id, md = matched[0]
    serving_ids = {mid for _, mid, _ in matched}

    presets: list[dict[str, Any]] = []
    try:
        from .presets import list_presets

        presets = [
            {"id": rec.id, "hardware": rec.hardware}
            for rec in list_presets().presets
            if rec.model in serving_ids
        ]
    except Exception:  # noqa: BLE001 - presets are enrichment, not load-bearing
        presets = []

    caps = md.capabilities
    # The catalog's validated sampling for this model (cross-referenced against
    # club-3090's canonical defaults), surfaced so the GUI can offer "apply
    # recommended" — only the sampling keys, never the rest of the gen config.
    ogc = md.override_generation_config or {}
    recommended = {k: ogc[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty") if k in ogc}
    return {
        "model_id": md.id,
        "title": md.title,
        "served_model_name": md.served_model_name,
        "match_kind": ("served_model_name", "model_path", "id")[prio],
        "quantization": md.quantization,
        "dtype": md.dtype,
        "recommended_sampling": recommended or None,
        "capabilities": {
            "attention_arch": caps.attention_arch,
            "tool_call_parser": caps.tool_call_parser,
            "reasoning_parser": caps.reasoning_parser,
            "spec_decode": bool(caps.spec_decode),
            "kv_cache_dtype": caps.kv_cache_dtype,
        },
        "requires": {
            "min_total_vram_mib": md.requires.min_total_vram_mib,
            "min_gpu_count": md.requires.min_gpu_count,
        },
        "vllm_pin_required": md.versions.vllm_pin_required,
        "presets": presets,
    }


def _vllm_model_meta(base_url: str, *, timeout: float, api_key: Optional[str]) -> dict[str, dict[str, Any]]:
    """Per-served-id ``max_model_len`` / ``root`` from the engine's ``/v1/models``."""
    out: dict[str, dict[str, Any]] = {}
    try:
        _, body = engine_client._get(f"{base_url}/models", timeout=timeout, api_key=api_key)
        for entry in json.loads(body).get("data", []):
            if isinstance(entry, dict) and entry.get("id"):
                out[entry["id"]] = {
                    "max_model_len": entry.get("max_model_len"),
                    "root": entry.get("root"),
                }
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        pass
    return out


def _port_from_base_url(base_url: Optional[str]) -> Optional[int]:
    """The port from an engine base_url (``http://host:8102/v1`` → 8102), or None."""
    if not base_url:
        return None
    try:
        import urllib.parse
        return urllib.parse.urlparse(base_url).port
    except (ValueError, TypeError):
        return None


def discover_engine(
    *,
    timeout: float = 2.5,
    profiles: Optional[list[Any]] = None,
    key_for: Optional[Any] = None,
) -> dict[str, Any]:
    """Find a running engine that actually serves models, so the GUI auto-connects
    instead of blindly probing the daemon's localhost:8000.

    Order: the configured/local engine first, then each registered host's declared
    engine endpoint (``host`` + ``engine_port`` + its stored key via ``key_for``).
    Returns the enriched detail of the first hit with ``host`` / ``port`` /
    ``host_id`` set so the caller knows where to connect; falls back to the local
    (unreachable) result. ``profiles`` / ``key_for`` are injected by the route
    (they own the host-profile + key plumbing) — kept as params so this stays
    pure and unit-testable."""
    local = engine_model_detail(timeout=timeout)
    if local.get("reachable") and local.get("models"):
        # The configured engine may be on a non-default port (SNDR_OPENAI_BASE_URL,
        # e.g. :8102) — surface it so a chat stuck on :8000 can adopt the real port.
        return {**local, "port": _port_from_base_url(local.get("base_url")), "host_id": None}

    for prof in profiles or []:
        key = None
        if key_for is not None:
            try:
                key = key_for(prof)
            except Exception:  # noqa: BLE001 - a bad key resolver must not abort discovery
                key = None
        detail = engine_model_detail(prof.host, port=prof.engine_port, api_key=key, timeout=timeout)
        if detail.get("reachable") and detail.get("models"):
            return {**detail, "port": prof.engine_port, "host_id": prof.id}

    return {**local, "port": None, "host_id": None}


def engine_model_detail(
    host: Optional[str] = None,
    *,
    port: Optional[int] = None,
    timeout: float = 3.0,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Detect the running engine's served model(s) and bridge each to the catalog.

    Returns ``{reachable, host, version, models: [{id, max_model_len, root,
    catalog}]}`` where ``catalog`` is the :func:`match_catalog_model` payload or
    ``None``. Degrades gracefully (``reachable=False``, ``models=[]``) when no
    engine is listening."""
    status = engine_client.engine_status(host, port=port, timeout=timeout, api_key=api_key)
    result: dict[str, Any] = {
        "reachable": status.get("reachable", False),
        "host": status.get("host"),
        "base_url": status.get("base_url"),
        "version": status.get("version"),
        "error": status.get("error"),
        "models": [],
    }
    served_ids = status.get("models") or []
    if not status.get("reachable") or not served_ids:
        return result

    meta = _vllm_model_meta(status["base_url"], timeout=timeout, api_key=engine_client._resolve_api_key(api_key))
    result["models"] = [
        {
            "id": served,
            "max_model_len": meta.get(served, {}).get("max_model_len"),
            "root": meta.get(served, {}).get("root"),
            "catalog": match_catalog_model(served),
        }
        for served in served_ids
    ]
    return result
