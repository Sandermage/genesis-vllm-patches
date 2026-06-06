# SPDX-License-Identifier: Apache-2.0
"""Read-only model cache-status report for the GUI Models screen.

Reports, per V2 model, the declared ``model_path`` and whether that path exists
**on the API daemon host**. This is honest about its scope:

  * In Server-Web mode the daemon runs on the GPU host, so the check reflects
    the real checkpoint cache there.
  * In dev / Remote-Desktop mode the daemon runs on the operator machine, where
    container-side paths like ``/models/...`` will not exist — reported as
    absent, not guessed as present.

No download, no subprocess, no network. Existence is a plain ``os.path.isdir``
plus a best-effort size when present and cheap to compute.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCacheEntry:
    model_id: str
    model_path: str
    present: bool
    size_mib: int | None  # best-effort, only when present and cheaply known


@dataclass(frozen=True)
class ModelCacheReport:
    host: str
    total: int
    present_count: int
    models: tuple[ModelCacheEntry, ...] = field(default_factory=tuple)


def _dir_size_mib(path: str, *, cap_entries: int = 20000) -> int | None:
    """Best-effort directory size in MiB; bounded so it never stalls the API."""
    total = 0
    seen = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                seen += 1
                if seen > cap_entries:
                    return None  # too large to size cheaply; report unknown
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return None
    return total // (1024 * 1024)


def collect_model_cache_report() -> ModelCacheReport:
    """Build the read-only model cache-status report (daemon-host scoped)."""
    from vllm.sndr_core.model_configs.registry_v2 import list_models, load_model

    entries: list[ModelCacheEntry] = []
    for model_id in list_models():
        try:
            model = load_model(model_id)
            model_path = str(model.model_path or "")
        except Exception:
            model_path = ""
        present = bool(model_path) and os.path.isdir(model_path)
        size = _dir_size_mib(model_path) if present else None
        entries.append(
            ModelCacheEntry(
                model_id=model_id,
                model_path=model_path,
                present=present,
                size_mib=size,
            )
        )

    try:
        host = socket.gethostname()
    except Exception:
        host = "daemon-host"

    return ModelCacheReport(
        host=host,
        total=len(entries),
        present_count=sum(1 for entry in entries if entry.present),
        models=tuple(entries),
    )
