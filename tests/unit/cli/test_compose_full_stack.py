# SPDX-License-Identifier: Apache-2.0
"""v12 UX R3 — ``compose/docker-compose.full.yml`` parses and wires the full
product on one network: the engine service AND the product-API + GUI daemon.

The CLI ``sndr up`` is the ergonomic front; this compose file is the declarative
equivalent so ``docker compose -f compose/docker-compose.full.yml up`` also
brings up the whole product. These tests assert only the *shape* (it parses,
defines both services, shares one network, exposes the daemon on 8765) — they
do not pull images or start containers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_COMPOSE = Path(__file__).resolve().parents[3] / "compose" / "docker-compose.full.yml"


def _load() -> dict:
    assert _COMPOSE.exists(), f"missing compose file: {_COMPOSE}"
    with _COMPOSE.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_parses():
    doc = _load()
    assert isinstance(doc, dict)
    assert "services" in doc


def test_defines_engine_and_daemon_services():
    doc = _load()
    services = doc["services"]
    # An engine service (vLLM) AND a daemon/web service (product-API + GUI).
    names = set(services)
    assert any("engine" in n or "vllm" in n for n in names), f"no engine service in {names}"
    assert any("daemon" in n or "web" in n or "gui" in n for n in names), f"no daemon/web service in {names}"


def test_services_share_one_network():
    doc = _load()
    services = doc["services"]
    nets_per_service = []
    for spec in services.values():
        nets = spec.get("networks")
        # networks may be a list or a mapping; normalise to a set of names.
        if isinstance(nets, dict):
            nets_per_service.append(set(nets))
        elif isinstance(nets, list):
            nets_per_service.append(set(nets))
        else:
            nets_per_service.append(set())
    # Every service that declares a network must share at least one common one.
    declaring = [n for n in nets_per_service if n]
    assert declaring, "no service declares a network"
    common = set.intersection(*declaring)
    assert common, f"services do not share a network: {nets_per_service}"
    # The shared network is declared at top level.
    assert "networks" in doc
    assert common & set(doc["networks"]), "shared network is not declared at top level"


def test_daemon_service_exposes_8765():
    doc = _load()
    services = doc["services"]
    daemon = next(
        spec for name, spec in services.items()
        if "daemon" in name or "web" in name or "gui" in name
    )
    ports = daemon.get("ports") or []
    flat = " ".join(str(p) for p in ports)
    assert "8765" in flat, f"daemon service does not expose 8765: {ports}"


def test_daemon_service_runs_the_gui_api_server():
    # The daemon service must launch the SAME Product API server the gui-api path
    # uses (run_server / http_app), not a bespoke server.
    doc = _load()
    services = doc["services"]
    daemon = next(
        spec for name, spec in services.items()
        if "daemon" in name or "web" in name or "gui" in name
    )
    blob = repr(daemon)
    assert "http_app" in blob or "run_server" in blob or "gui-api" in blob, (
        "daemon service must reuse the existing gui-api/run_server entrypoint"
    )


def test_no_hardcoded_lan_ip():
    # Same hygiene the rest of the repo enforces: no operator LAN IP baked in.
    text = _COMPOSE.read_text(encoding="utf-8")
    import re

    # 192.168.x / 10.x private ranges (the homelab IPs the repo bans).
    assert not re.search(r"\b192\.168\.\d{1,3}\.\d{1,3}\b", text), "hardcoded 192.168 IP"
    assert not re.search(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text), "hardcoded 10.x IP"
