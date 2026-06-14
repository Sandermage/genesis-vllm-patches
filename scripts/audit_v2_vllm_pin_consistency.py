#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 vllm-pin consistency gate.

For every V2 `kind: model` YAML that names a baseline JSON via
`versions.reference_metrics_ref`, the model's `versions.vllm_pin_required`
must match the baseline's recorded vllm version. Otherwise the bench
claims live at a different pin than the model declares — the F-018
class of drift that the V1 yaml comment in `a5000-2x-35b-prod.yaml`
explicitly called out.

Baseline JSON schema variant tolerance: vllm_version may live at the
top level (`vllm_version: "..."`), inside an object
(`vllm_version: {parsed: {vllm_version: "..."}}`), or under config
(`config.vllm_pin`). We walk known paths in order.

Exit codes:
  0 — every model with a baseline ref agrees on the vllm pin
  1 — at least one model.vllm_pin_required ≠ baseline.vllm_version
  2 — internal error (baseline file missing / unreadable)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"
HARDWARE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"

# The canonical PROD rig: every model in the fleet renders on this hardware on
# the live server, so the model's declared pin MUST match the image that
# hardware actually ships — otherwise launching on PROD silently runs an
# unvalidated pin (the dev259→dev491 drift the reference_metrics_ref baseline
# check missed because every model has reference_metrics_ref: null).
PROD_HARDWARE_ID = "a5000-2x-24gbvram-16cpu-128gbram"


def _short_sha(value: Optional[str]) -> Optional[str]:
    """Extract the git short-SHA from a vllm pin or docker image tag.

    Handles both `0.22.1rc1.dev491+g1033ffac2` (pin, after `+g`) and
    `vllm/vllm-openai:nightly-1033ffac2` / `nightly-1033ffac2d66` (image tag,
    after `nightly-`). Returns the first 8 hex chars for a stable compare
    (setuptools_scm uses 8, some tags carry 12+; 8 is the common prefix).
    """
    if not value:
        return None
    import re
    m = re.search(r"\+g([0-9a-f]{7,})", value) or re.search(
        r"nightly-([0-9a-f]{7,})", value
    )
    if not m:
        return None
    return m.group(1)[:8]


@dataclass
class PinCheck:
    path: Path
    model_id: str
    yaml_pin: Optional[str] = None
    baseline_ref: Optional[str] = None
    baseline_pin: Optional[str] = None
    skipped_reason: str = ""    # populated when there's no baseline to compare
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.skipped_reason:
            return True   # vacuously — no claim to check
        return self.yaml_pin == self.baseline_pin


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _walk_string(d, path: list[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, str) else None


def _extract_baseline_vllm_version(d: dict) -> Optional[str]:
    """Try known schema variants in order. Return first non-None hit."""
    for path in (
        ["vllm_version"],
        ["vllm_pin"],
        ["vllm_version", "parsed", "vllm_version"],
        ["parsed", "vllm_version"],
        ["config", "vllm_pin"],
        ["headline", "vllm_pin"],
        ["summary", "vllm_pin"],
    ):
        v = _walk_string(d, path)
        if v:
            return v
    return None


def check_one_model(path: Path) -> PinCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return PinCheck(
            path=path, model_id="?",
            error=f"YAML parse error: {e}",
        )
    model_id = data.get("id", path.stem)
    versions = data.get("versions") or {}
    yaml_pin = versions.get("vllm_pin_required")
    ref = versions.get("reference_metrics_ref")

    r = PinCheck(
        path=path,
        model_id=model_id,
        yaml_pin=yaml_pin,
        baseline_ref=ref,
    )

    if not ref:
        r.skipped_reason = "no reference_metrics_ref — no baseline to compare"
        return r

    ref_path = (REPO_ROOT / str(ref)).resolve()
    if not ref_path.is_file():
        r.error = f"baseline file not found: {ref}"
        return r

    try:
        bdata = json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        r.error = f"baseline parse error: {e}"
        return r

    baseline_pin = _extract_baseline_vllm_version(bdata)
    if not baseline_pin:
        r.error = "baseline file has no recognized vllm version field"
        return r
    r.baseline_pin = baseline_pin
    return r


def audit_v2_vllm_pin_consistency(
    model_dir: Path = MODEL_DIR,
) -> list[PinCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


# ─── Baseline-independent check: model pin SHA vs PROD hardware image SHA ────


@dataclass
class PinImageCheck:
    model_id: str
    path: Path
    model_pin: Optional[str] = None
    model_sha: Optional[str] = None
    image_sha: Optional[str] = None
    held: bool = False          # explicit versions.pin_hold: true exemption
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.held:
            return True         # operator-acknowledged intentional hold
        if self.model_sha is None or self.image_sha is None:
            return True         # nothing comparable (no pin / unknown image)
        return self.model_sha == self.image_sha


def _prod_image_sha(hardware_dir: Path = HARDWARE_DIR) -> Optional[str]:
    """Short-SHA of the docker image the canonical PROD rig actually ships."""
    hw = hardware_dir / f"{PROD_HARDWARE_ID}.yaml"
    if not hw.is_file():
        return None
    try:
        data = _load_yaml(hw)
    except Exception:
        return None
    image = (
        (((data.get("runtime") or {}).get("docker") or {}).get("image"))
        or ""
    )
    return _short_sha(image)


def audit_pin_vs_image_drift(
    model_dir: Path = MODEL_DIR,
    hardware_dir: Path = HARDWARE_DIR,
) -> list[PinImageCheck]:
    """Every model's declared vllm_pin_required must share the SHA of the
    image the PROD hardware ships (unless versions.pin_hold: true). This is
    baseline-independent — it fires even when reference_metrics_ref is null,
    closing the gap that let the dev491 hardware image bump pass CI while the
    model pins still declared dev259."""
    image_sha = _prod_image_sha(hardware_dir)
    out: list[PinImageCheck] = []
    for path in sorted(model_dir.glob("*.yaml")):
        try:
            data = _load_yaml(path)
        except Exception as e:
            out.append(PinImageCheck(model_id=path.stem, path=path,
                                     error=f"YAML parse error: {e}"))
            continue
        versions = data.get("versions") or {}
        pin = versions.get("vllm_pin_required")
        out.append(PinImageCheck(
            model_id=data.get("id", path.stem),
            path=path,
            model_pin=pin,
            model_sha=_short_sha(pin),
            image_sha=image_sha,
            held=bool(versions.get("pin_hold")),
        ))
    return out


def _render_pin_image(results: list[PinImageCheck], image_sha: Optional[str]) -> str:
    lines = [
        f"audit-pin-vs-image-drift: {len(results)} model YAML(s) vs "
        f"{PROD_HARDWARE_ID} image sha={image_sha}",
        "─" * 70,
    ]
    for r in sorted(results, key=lambda x: (x.passed, x.model_id)):
        if r.error:
            lines.append(f"  ✗ {r.model_id}: {r.error}")
        elif r.held:
            lines.append(f"  · {r.model_id:40s} [pin_hold] pin={r.model_pin}")
        elif r.passed:
            lines.append(f"  ✓ {r.model_id:40s} sha={r.model_sha} == image")
        else:
            lines.append(
                f"  ✗ {r.model_id:40s} pin sha={r.model_sha} != "
                f"PROD image sha={r.image_sha}  (declared pin behind the "
                f"image this rig ships — bump vllm_pin_required after "
                f"validating, or set versions.pin_hold: true)"
            )
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {len(results) - failed}/{len(results)} models agree with the PROD image SHA")
    return "\n".join(lines)


def _render_text(results: list[PinCheck]) -> str:
    lines = [f"audit-v2-vllm-pin-consistency: {len(results)} model YAML(s)",
             "─" * 70]
    checked = skipped = 0
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        if r.skipped_reason:
            skipped += 1
            lines.append(
                f"  · {r.model_id:36s} [skipped] {r.skipped_reason}"
            )
            continue
        checked += 1
        if r.passed:
            lines.append(
                f"  {sym} {r.model_id:36s} pin={r.yaml_pin} == baseline"
            )
        else:
            lines.append(
                f"  {sym} {r.model_id:36s} "
                f"yaml={r.yaml_pin!r} vs baseline={r.baseline_pin!r}"
            )
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(
        f"  {passed}/{len(results)} models pass  "
        f"({checked} compared, {skipped} skipped — no baseline)"
    )
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either bump model.versions.vllm_pin_required to "
            "match the baseline, or re-snap baseline against the current pin."
        )
    return "\n".join(lines)


def _render_json(results: list[PinCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "yaml_pin": r.yaml_pin,
                "baseline_ref": r.baseline_ref,
                "baseline_pin": r.baseline_pin,
                "skipped_reason": r.skipped_reason or None,
                "passed": r.passed,
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
    results = audit_v2_vllm_pin_consistency()
    image_results = audit_pin_vs_image_drift()
    image_sha = _prod_image_sha()
    if args.json:
        combined = json.loads(_render_json(results))
        combined["pin_vs_image"] = {
            "prod_hardware": PROD_HARDWARE_ID,
            "image_sha": image_sha,
            "failed": sum(1 for r in image_results if not r.passed),
            "models": [
                {
                    "model_id": r.model_id,
                    "path": _rel(r.path),
                    "model_pin": r.model_pin,
                    "model_sha": r.model_sha,
                    "image_sha": r.image_sha,
                    "held": r.held,
                    "passed": r.passed,
                    "error": r.error or None,
                }
                for r in image_results
            ],
        }
        print(json.dumps(combined, indent=2, sort_keys=True))
    else:
        print(_render_text(results))
        print()
        print(_render_pin_image(image_results, image_sha))
    baseline_ok = all(r.passed for r in results)
    image_ok = all(r.passed for r in image_results)
    return 0 if (baseline_ok and image_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
