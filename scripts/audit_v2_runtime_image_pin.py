#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 runtime image-pin gate.

For each V2 hardware YAML, `runtime.docker.image` must be non-empty
string and `runtime.docker.image_digest` must match a canonical
docker digest format:

    <repo>@sha256:<64-hex>

Floating tags (`:nightly`, `:latest`) survive in the `image` field
for human readability, but `image_digest` is what production launchers
should pin to (reproducibility). Missing or malformed digest = no pin =
release-time drift if upstream re-tags the floating reference.

Exit codes:
  0 — every hardware YAML's image_digest matches canonical format
  1 — at least one missing / malformed
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


# Format: <repo>@sha256:<64-hex>
DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9./_-]*@sha256:[0-9a-f]{64}$")


@dataclass
class ImagePinCheck:
    path: Path
    hardware_id: str
    image: Optional[str] = None
    image_digest: Optional[str] = None
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_one_hardware(path: Path) -> ImagePinCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return ImagePinCheck(path=path, hardware_id="?",
                            error=f"YAML parse error: {e}")
    hardware_id = data.get("id", path.stem)
    rt = (data.get("runtime") or {}).get("docker") or {}
    image = rt.get("image")
    digest = rt.get("image_digest")

    r = ImagePinCheck(
        path=path, hardware_id=hardware_id,
        image=image if isinstance(image, str) else None,
        image_digest=digest if isinstance(digest, str) else None,
    )
    if not isinstance(image, str) or not image.strip():
        r.violations.append("runtime.docker.image missing or empty")
    if not isinstance(digest, str) or not digest.strip():
        r.violations.append("runtime.docker.image_digest missing or empty")
    elif not DIGEST_RE.match(digest):
        r.violations.append(
            f"image_digest={digest!r} does not match {DIGEST_RE.pattern!r}"
        )
    return r


def audit_v2_runtime_image_pin(
    hw_dir: Path = HARDWARE_DIR,
) -> list[ImagePinCheck]:
    if not hw_dir.is_dir():
        return []
    return [check_one_hardware(p) for p in sorted(hw_dir.glob("*.yaml"))]


def _render_text(results: list[ImagePinCheck]) -> str:
    lines = [
        f"audit-v2-runtime-image-pin: {len(results)} hardware YAML(s)",
        f"  digest regex: {DIGEST_RE.pattern}",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.hardware_id}: {r.error}")
            continue
        digest_short = (r.image_digest or "")[:64] + "…" if r.image_digest else "<none>"
        lines.append(f"  {sym} {r.hardware_id:36s}  image={r.image!r}")
        lines.append(f"      digest={digest_short}")
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} hardware files have valid image_digest")
    return "\n".join(lines)


def _render_json(results: list[ImagePinCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "digest_regex": DIGEST_RE.pattern,
        "results": [
            {
                "hardware_id": r.hardware_id,
                "path": _rel(r.path),
                "image": r.image,
                "image_digest": r.image_digest,
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
    results = audit_v2_runtime_image_pin()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
