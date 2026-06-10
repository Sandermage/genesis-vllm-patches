#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 network + port consistency gate.

For each V2 hardware YAML, validates the docker runtime networking
block:

  • `host_port`      ∈ [1024, 65535]   (privileged ports blocked)
  • `container_port` ∈ [1024, 65535]
  • `shm_size`       matches docker size format: `<int>[bkmgBKMG]?`
                     (k/m/g are decimal; bytes default)
  • `network`        non-empty string (docker network name)

A typo here (port = 8000.0, shm_size = '8' without unit, port < 1024
requiring root) breaks container startup. Catches operator typos at
PR time, not at production launch.

Exit codes:
  0 — every hardware YAML's network block is valid
  1 — at least one violation
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


PORT_RANGE = (1024, 65535)
SHM_SIZE_RE = re.compile(r"^\d+[bBkKmMgG]?$")


@dataclass
class NetPortCheck:
    path: Path
    hardware_id: str
    host_port: object = None
    container_port: object = None
    shm_size: object = None
    network: object = None
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _check_port(label: str, val, violations: list[str]) -> None:
    if not isinstance(val, int) or isinstance(val, bool):
        violations.append(f"{label}={val!r} not int")
        return
    if val < PORT_RANGE[0] or val > PORT_RANGE[1]:
        violations.append(
            f"{label}={val} outside [{PORT_RANGE[0]}..{PORT_RANGE[1]}]"
        )


def check_one_hardware(path: Path) -> NetPortCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return NetPortCheck(path=path, hardware_id="?",
                           error=f"YAML parse error: {e}")
    hardware_id = data.get("id", path.stem)
    rt = (data.get("runtime") or {}).get("docker") or {}
    r = NetPortCheck(
        path=path, hardware_id=hardware_id,
        host_port=rt.get("host_port"),
        container_port=rt.get("container_port"),
        shm_size=rt.get("shm_size"),
        network=rt.get("network"),
    )

    _check_port("host_port",      r.host_port,      r.violations)
    _check_port("container_port", r.container_port, r.violations)

    if not isinstance(r.shm_size, str) or not SHM_SIZE_RE.match(r.shm_size):
        r.violations.append(
            f"shm_size={r.shm_size!r} not docker size format "
            f"({SHM_SIZE_RE.pattern!r})"
        )

    if not isinstance(r.network, str) or not r.network.strip():
        r.violations.append("network missing or empty")
    return r


def audit_v2_network_port_consistency(
    hw_dir: Path = HARDWARE_DIR,
) -> list[NetPortCheck]:
    if not hw_dir.is_dir():
        return []
    return [check_one_hardware(p) for p in sorted(hw_dir.glob("*.yaml"))]


def _render_text(results: list[NetPortCheck]) -> str:
    lines = [
        f"audit-v2-network-port-consistency: {len(results)} hardware YAML(s)",
        f"  port range: [{PORT_RANGE[0]}..{PORT_RANGE[1]}]",
        f"  shm_size regex: {SHM_SIZE_RE.pattern}",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.hardware_id}: {r.error}")
            continue
        lines.append(
            f"  {sym} {r.hardware_id:36s}  "
            f"ports={r.host_port}:{r.container_port}  "
            f"shm={r.shm_size}  net={r.network!r}"
        )
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} hardware files clean")
    return "\n".join(lines)


def _render_json(results: list[NetPortCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "port_range": list(PORT_RANGE),
        "shm_size_regex": SHM_SIZE_RE.pattern,
        "results": [
            {
                "hardware_id": r.hardware_id,
                "path": _rel(r.path),
                "host_port": r.host_port,
                "container_port": r.container_port,
                "shm_size": r.shm_size,
                "network": r.network,
                "passed": r.passed,
                "violations": r.violations,
                "error": r.error or None,
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit_v2_network_port_consistency()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
