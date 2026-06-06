# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Hub model search.

Queries the public Hub API (``https://huggingface.co/api/models``) so the GUI can
browse and search models to download. stdlib ``urllib`` only; fixed host (no
SSRF); the search term is URL-encoded.
"""
from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from typing import Any

_HF_MODELS_API = "https://huggingface.co/api/models"
_HF_MODEL_API = "https://huggingface.co/api/models/"


def ssl_context() -> ssl.SSLContext:
    """A verifying TLS context that uses certifi's CA bundle when available.

    Some hosts (notably macOS python.org builds) lack a usable system CA bundle;
    falling back to certifi keeps certificate verification ON rather than
    insecurely disabling it."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def search_models(query: str, *, limit: int = 20, pipeline: str = "text-generation", timeout: float = 8.0) -> list[dict[str, Any]]:
    """Search the Hub, returning a distilled list sorted by downloads."""
    params: dict[str, Any] = {"limit": max(1, min(50, int(limit))), "sort": "downloads", "direction": -1}
    cleaned = (query or "").strip()
    if cleaned:
        params["search"] = cleaned
    if pipeline:
        params["filter"] = pipeline
    url = f"{_HF_MODELS_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "sndr-gui"})
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:  # noqa: S310 - fixed HF host
        data = json.loads(response.read().decode("utf-8"))
    results: list[dict[str, Any]] = []
    for entry in data if isinstance(data, list) else []:
        model_id = entry.get("id") or entry.get("modelId")
        if not model_id:
            continue
        results.append(
            {
                "id": model_id,
                "downloads": entry.get("downloads"),
                "likes": entry.get("likes"),
                "pipeline_tag": entry.get("pipeline_tag"),
                "gated": bool(entry.get("gated")),
                "tags": [tag for tag in (entry.get("tags") or []) if isinstance(tag, str)][:6],
            }
        )
    return results[: int(limit)]


__all__ = ["search_models"]
