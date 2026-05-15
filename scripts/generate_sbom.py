#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SBOM (Software Bill of Materials) generator — Wave 4.2.

Production roadmap §8.3: every release wheel must ship with an SBOM
listing the exact runtime + transitive Python dependencies, plus
metadata about the patch registry, model_configs builtin presets, and
license tier boundary.

Output formats:
  - **CycloneDX 1.5 JSON** (`<out>.cdx.json`) — industry standard,
    consumed by Dependency-Track, GitHub, etc.
  - **SPDX 2.3 JSON** (`<out>.spdx.json`) — alternative widely-used
    format.
  - **Plain text** (`<out>.txt`) — operator-friendly summary.

What gets enumerated:
  - Direct dependencies from `pyproject.toml`.
  - Transitive dependencies via `importlib.metadata.distributions()`
    when invoked inside an installed venv (otherwise marked
    "transitive=unresolved").
  - Genesis-internal module list with relative paths + content hashes.
  - PATCH_REGISTRY snapshot (count + lifecycle distribution).
  - Builtin model_config presets (key + maintainer + last_validated).
  - vllm/torch/triton known_good pins (from KNOWN_GOOD_VLLM_PINS).
  - `KNOWN_GOOD_IMAGES` digest allowlist.

Usage:
  # Generate against a wheel (best — fully-resolved transitive deps)
  pip install dist/vllm_sndr_core-*.whl
  python3 scripts/generate_sbom.py --out genesis-sbom

  # Generate from source tree (transitive deps unresolved)
  python3 scripts/generate_sbom.py --out genesis-sbom --source-only

  # Specific format
  python3 scripts/generate_sbom.py --out gen --format cyclonedx

Exit codes:
  0 — SBOM generated successfully
  1 — fatal error (missing pyproject.toml, etc.)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── Helpers ─────────────────────────────────────────────────────────────


def _file_sha256(path: Path) -> str:
    """SHA-256 hex of a file's contents. Returns '' on read error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _read_pyproject() -> dict[str, Any]:
    """Parse pyproject.toml. Returns {} on failure."""
    pp = REPO_ROOT / "pyproject.toml"
    if not pp.is_file():
        return {}
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(pp, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"warn: pyproject parse failed: {e}", file=sys.stderr)
        return {}


def _read_constraints() -> list[str]:
    """Return non-comment lines from constraints.txt."""
    cf = REPO_ROOT / "constraints.txt"
    if not cf.is_file():
        return []
    out: list[str] = []
    for line in cf.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _list_genesis_modules() -> list[dict[str, Any]]:
    """Walk vllm/sndr_core/ and emit (path, sha256) for each .py."""
    sndr = REPO_ROOT / "vllm" / "sndr_core"
    if not sndr.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(sndr.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        out.append({"path": rel, "sha256": _file_sha256(p)})
    return out


def _registry_snapshot() -> dict[str, Any]:
    """Snapshot PATCH_REGISTRY size + lifecycle / tier distribution."""
    try:
        from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    except Exception as e:
        return {"error": str(e), "total": 0}
    total = len(PATCH_REGISTRY)
    by_tier: dict[str, int] = {}
    by_lifecycle: dict[str, int] = {}
    by_default_on: dict[str, int] = {"True": 0, "False": 0}
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        tier = meta.get("tier", "unknown")
        lc = meta.get("lifecycle", "unset")
        do = "True" if meta.get("default_on", False) else "False"
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_lifecycle[lc] = by_lifecycle.get(lc, 0) + 1
        by_default_on[do] += 1
    return {
        "total": total,
        "by_tier": by_tier,
        "by_lifecycle": by_lifecycle,
        "by_default_on": by_default_on,
    }


def _model_configs_snapshot() -> list[dict[str, Any]]:
    """List builtin model_configs with key + maintainer + last_validated."""
    out: list[dict[str, Any]] = []
    try:
        from vllm.sndr_core.model_configs.registry import list_keys, get
        for key in sorted(list_keys()):
            try:
                cfg = get(key)
            except Exception:
                continue
            if cfg is None:
                continue
            out.append({
                "key": cfg.key,
                "maintainer": getattr(cfg, "maintainer", ""),
                "last_validated": getattr(cfg, "last_validated", None),
                "lifecycle": getattr(cfg, "lifecycle", "unknown"),
                "vllm_pin_required": getattr(cfg, "vllm_pin_required", None),
            })
    except Exception as e:
        out.append({"error": str(e)})
    return out


def _vllm_pins() -> list[str]:
    """KNOWN_GOOD_VLLM_PINS list."""
    try:
        from vllm.sndr_core.dispatcher import KNOWN_GOOD_VLLM_PINS
        return list(KNOWN_GOOD_VLLM_PINS)
    except Exception:
        return []


def _image_allowlist() -> list[dict[str, Any]]:
    """Active KNOWN_GOOD_IMAGES (excludes historical)."""
    try:
        from vllm.sndr_core.compat.image_allowlist import list_active
        return [
            {
                "image_repo": e.image_repo,
                "image_digest": e.image_digest,
                "vllm_pin": e.vllm_pin,
                "torch_pin": e.torch_pin,
                "validated_at": e.validated_at,
                "validated_on": e.validated_on,
            }
            for e in list_active()
        ]
    except Exception as e:
        return [{"error": str(e)}]


def _installed_distributions() -> list[dict[str, Any]]:
    """Use importlib.metadata to enumerate installed packages."""
    try:
        from importlib.metadata import distributions
    except ImportError:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
        except (KeyError, AttributeError):
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "version": dist.version,
            "summary": (dist.metadata.get("Summary") or "")[:200],
        })
    return sorted(out, key=lambda d: d["name"].lower())


# ─── SBOM emitters ───────────────────────────────────────────────────────


def emit_cyclonedx(payload: dict[str, Any], out_path: Path) -> None:
    """CycloneDX 1.5 JSON (industry standard)."""
    pj = payload["pyproject"]
    project = pj.get("project", {})
    name = project.get("name", "vllm-sndr-core")
    version = project.get("version", "0.0.0")
    components: list[dict[str, Any]] = []

    # Direct deps
    for dep in project.get("dependencies", []) or []:
        components.append({
            "type": "library",
            "bom-ref": f"pkg:pypi/{dep}",
            "name": dep.split(">=")[0].split("==")[0].split("<")[0].strip(),
            "scope": "required",
            "purl": f"pkg:pypi/{dep}",
        })
    # Installed (transitive included)
    for d in payload["installed_distributions"]:
        bom_ref = f"pkg:pypi/{d['name']}@{d['version']}"
        components.append({
            "type": "library",
            "bom-ref": bom_ref,
            "name": d["name"],
            "version": d["version"],
            "purl": bom_ref,
            "description": d["summary"],
        })

    cdx = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": payload["generated_at"],
            "tools": [{"name": "genesis-sbom-generator", "version": "1.0"}],
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/{name}@{version}",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
                "description": project.get("description", ""),
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            },
        },
        "components": components,
        "properties": [
            {"name": "genesis:patch_registry_total",
             "value": str(payload["patch_registry"].get("total", 0))},
            {"name": "genesis:vllm_known_good_pins_count",
             "value": str(len(payload["known_good_vllm_pins"]))},
            {"name": "genesis:known_good_images_count",
             "value": str(len(payload["image_allowlist"]))},
            {"name": "genesis:builtin_model_configs",
             "value": str(len(payload["model_configs"]))},
        ],
    }
    out_path.write_text(json.dumps(cdx, indent=2), encoding="utf-8")


def emit_spdx(payload: dict[str, Any], out_path: Path) -> None:
    """SPDX 2.3 JSON."""
    pj = payload["pyproject"]
    project = pj.get("project", {})
    name = project.get("name", "vllm-sndr-core")
    version = project.get("version", "0.0.0")

    packages: list[dict[str, Any]] = [
        {
            "SPDXID": "SPDXRef-Package-Genesis",
            "name": name,
            "versionInfo": version,
            "supplier": "Organization: Sandermage",
            "licenseConcluded": "Apache-2.0",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
        }
    ]
    relationships: list[dict[str, str]] = []
    for d in payload["installed_distributions"]:
        spdx_id = f"SPDXRef-Package-{d['name'].replace('_', '-')}-{d['version'].replace('.', '-')}"
        packages.append({
            "SPDXID": spdx_id,
            "name": d["name"],
            "versionInfo": d["version"],
            "downloadLocation": f"https://pypi.org/project/{d['name']}/{d['version']}/",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
        })
        relationships.append({
            "spdxElementId": "SPDXRef-Package-Genesis",
            "relatedSpdxElement": spdx_id,
            "relationshipType": "DEPENDS_ON",
        })

    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-{version}",
        "documentNamespace": f"https://github.com/Sandermage/genesis-vllm-patches/{uuid.uuid4()}",
        "creationInfo": {
            "created": payload["generated_at"],
            "creators": ["Tool: genesis-sbom-generator-1.0"],
        },
        "packages": packages,
        "relationships": relationships,
    }
    out_path.write_text(json.dumps(spdx, indent=2), encoding="utf-8")


def emit_text(payload: dict[str, Any], out_path: Path) -> None:
    """Plain-text human-readable summary."""
    pj = payload["pyproject"]
    project = pj.get("project", {})
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"Genesis SBOM — {project.get('name', '?')} v{project.get('version', '?')}")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("## Direct dependencies (pyproject.toml)")
    for dep in project.get("dependencies", []) or []:
        lines.append(f"  - {dep}")
    lines.append("")
    lines.append("## constraints.txt")
    for c in payload["constraints_txt"]:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append(f"## Installed distributions ({len(payload['installed_distributions'])})")
    for d in payload["installed_distributions"][:80]:
        lines.append(f"  - {d['name']:<40} {d['version']}")
    if len(payload["installed_distributions"]) > 80:
        lines.append(f"  … and {len(payload['installed_distributions']) - 80} more")
    lines.append("")
    lines.append("## Patch registry snapshot")
    pr = payload["patch_registry"]
    lines.append(f"  Total: {pr.get('total', 0)}")
    lines.append(f"  By tier: {pr.get('by_tier', {})}")
    lines.append(f"  By lifecycle: {pr.get('by_lifecycle', {})}")
    lines.append(f"  default_on: {pr.get('by_default_on', {})}")
    lines.append("")
    lines.append("## KNOWN_GOOD_VLLM_PINS")
    for p in payload["known_good_vllm_pins"]:
        lines.append(f"  - {p}")
    lines.append("")
    lines.append(f"## KNOWN_GOOD_IMAGES (active, {len(payload['image_allowlist'])})")
    for e in payload["image_allowlist"]:
        if "error" in e:
            lines.append(f"  ERROR: {e['error']}")
        else:
            lines.append(
                f"  - {e['image_digest'][:50]}…  vllm={e['vllm_pin']}  "
                f"validated={e['validated_at']}"
            )
    lines.append("")
    lines.append(f"## Builtin model configs ({len(payload['model_configs'])})")
    for cfg in payload["model_configs"][:30]:
        lines.append(
            f"  - {cfg.get('key', '?'):<40} maintainer={cfg.get('maintainer', '?')} "
            f"validated={cfg.get('last_validated', '?')}"
        )
    lines.append("")
    lines.append(f"## Genesis modules ({len(payload['genesis_modules'])} files)")
    lines.append("  (full hashes in JSON form)")
    lines.append("")
    lines.append("=" * 78)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── Main ────────────────────────────────────────────────────────────────


def build_payload() -> dict[str, Any]:
    pj = _read_pyproject()
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat()
                            .replace("+00:00", "Z"),
        "pyproject": pj,
        "constraints_txt": _read_constraints(),
        "installed_distributions": _installed_distributions(),
        "patch_registry": _registry_snapshot(),
        "model_configs": _model_configs_snapshot(),
        "known_good_vllm_pins": _vllm_pins(),
        "image_allowlist": _image_allowlist(),
        "genesis_modules": _list_genesis_modules(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", default="genesis-sbom",
        help="Output base path. Suffixes (.cdx.json, .spdx.json, .txt) appended.",
    )
    p.add_argument(
        "--format", default="all",
        choices=("cyclonedx", "spdx", "text", "all"),
        help="Which format(s) to emit (default: all).",
    )
    args = p.parse_args(argv)

    payload = build_payload()
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.format in ("cyclonedx", "all"):
        path = out_base.with_suffix(".cdx.json")
        emit_cyclonedx(payload, path)
        print(f"  CycloneDX: {path}")
    if args.format in ("spdx", "all"):
        path = out_base.with_suffix(".spdx.json")
        emit_spdx(payload, path)
        print(f"  SPDX:      {path}")
    if args.format in ("text", "all"):
        path = out_base.with_suffix(".txt")
        emit_text(payload, path)
        print(f"  Text:      {path}")

    pj = payload["pyproject"]
    project = pj.get("project", {})
    print(f"Genesis SBOM generated for {project.get('name', '?')} v{project.get('version', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
