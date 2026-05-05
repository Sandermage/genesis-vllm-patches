#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D1 — Genesis upstream drift watcher.

Periodically (run from CI) checks two things against an upstream vllm
checkout:

  1. **Anchor drift.** For every Genesis wiring patch in
     `vllm/_genesis/wiring/patch_*.py`, attempt to construct its
     `_make_patcher()` and verify that all required anchors are still
     present in the upstream source files they target. If an anchor no
     longer matches, upstream has refactored that region — we need to
     re-derive the anchor before the next pin bump.

  2. **Upstream-merged markers.** For every entry in
     `vllm/_genesis/patches/upstream_compat.py::UPSTREAM_MARKERS`,
     check whether its `marker` string now appears in the upstream
     source. A new match means the upstream PR has merged → the
     corresponding Genesis patch should self-retire on the next pin
     bump.

Usage:
    python3 tools/check_upstream_drift.py /path/to/upstream-vllm-clone

Exit code:
    0 — no drift, all anchors still match
    1 — anchor drift detected (operator action required)
    2 — invocation error (clone path missing, etc.)

Output:
    JSON report on stdout for machine consumption + readable summary on
    stderr. The JSON shape:

    {
      "checked_at": "2026-04-29T22:30:00Z",
      "upstream_path": "/tmp/upstream-vllm",
      "upstream_head_sha": "abc123...",
      "anchors": {
        "PN14": {"file": "v1/.../triton_turboquant_decode.py",
                 "matches": True, "anchor_count": 1},
        "PN16": {"file": "entrypoints/.../serving.py",
                 "matches": False, "anchor_count": 0,
                 "drift": "anchor not found"},
        ...
      },
      "merged_markers": [
        {"key": "PR_40074_tq_decode_oob_clamp", "marker": "safe_page_idx",
         "file": "v1/.../triton_turboquant_decode.py", "merged_now": True},
        ...
      ],
      "summary": {
        "total_anchors": 18, "drifted": 0,
        "total_markers": 24, "newly_merged": 1,
      }
    }

D1 design constraint: read-only against upstream clone. Never mutate.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ─── Path setup so we can import Genesis from a repo checkout ──────────


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── Helpers ────────────────────────────────────────────────────────────


def _git_head_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out[:12]
    except Exception:
        return "unknown"


def _list_wiring_modules() -> list[str]:
    """Find all patch_*.py wiring modules."""
    wiring_dir = REPO_ROOT / "vllm" / "_genesis" / "wiring"
    out = []
    for f in sorted(wiring_dir.glob("patch_*.py")):
        out.append(f"vllm._genesis.wiring.{f.stem}")
    return out


def _check_one_patch(module_name: str, upstream_root: Path) -> dict:
    """Import a wiring module, monkey-patch resolve_vllm_file to point at
    the upstream clone, build the patcher, and verify all anchors present.
    """
    result: dict = {
        "module": module_name, "patch_name": None, "file": None,
        "matches": None, "anchor_count": 0, "drift": None,
    }

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        result["drift"] = f"import failed: {e}"
        return result

    # Find _make_patcher in the module
    make_patcher = getattr(mod, "_make_patcher", None)
    if make_patcher is None:
        result["drift"] = "module has no _make_patcher() — non-text-patch wiring"
        return result

    # Some wiring modules have parameterized _make_patcher (e.g. PN9 takes
    # `backend`, P77 takes `threshold`). For drift checking we just want
    # the static anchors — pass `None` or a sensible default for any
    # required positional arg.
    import inspect
    try:
        sig = inspect.signature(make_patcher)
        kwargs: dict = {}
        for pname, p in sig.parameters.items():
            if p.default is not inspect._empty:
                continue
            # Required positional — guess by annotation
            ann = p.annotation
            if ann is inspect._empty:
                kwargs[pname] = None
            elif "int" in str(ann):
                kwargs[pname] = 0
            elif "bool" in str(ann):
                kwargs[pname] = False
            elif "float" in str(ann):
                kwargs[pname] = 0.0
            elif "str" in str(ann):
                kwargs[pname] = None
            else:
                kwargs[pname] = None
    except (TypeError, ValueError):
        kwargs = {}

    # Monkey-patch resolve_vllm_file in the module's namespace to point
    # at the upstream clone.
    def _fake_resolve(rel: str) -> str | None:
        candidate = upstream_root / "vllm" / rel
        if candidate.is_file():
            return str(candidate)
        return None

    orig_resolve = getattr(mod, "resolve_vllm_file", None)
    orig_install_root = getattr(mod, "vllm_install_root", None)
    try:
        if orig_resolve is not None:
            mod.resolve_vllm_file = _fake_resolve
        if orig_install_root is not None:
            mod.vllm_install_root = lambda: str(upstream_root)

        patcher = make_patcher(**kwargs) if kwargs else make_patcher()
        if patcher is None:
            result["drift"] = "_make_patcher returned None — target file not found"
            return result

        result["patch_name"] = patcher.patch_name
        target = Path(patcher.target_file)
        if not target.is_file():
            result["drift"] = f"target file does not exist in upstream: {target}"
            return result
        try:
            content = target.read_text()
        except Exception as e:
            result["drift"] = f"target read failed: {e}"
            return result

        result["file"] = str(target.relative_to(upstream_root))

        # Check each sub-patch's anchor
        sub_patches = getattr(patcher, "sub_patches", []) or []
        if not sub_patches:
            result["drift"] = "patcher declared no sub_patches"
            return result

        all_match = True
        match_counts: list[int] = []
        for sp in sub_patches:
            anchor = getattr(sp, "anchor", None)
            required = getattr(sp, "required", True)
            if not anchor:
                continue
            count = content.count(anchor)
            match_counts.append(count)
            # An anchor must appear EXACTLY ONCE (so the patch unambiguously
            # knows where to apply); multiple matches = ambiguous, zero = drift.
            if count == 0 and required:
                all_match = False
            elif count > 1:
                all_match = False

        result["matches"] = all_match
        result["anchor_count"] = sum(match_counts)
        if not all_match:
            zero = [i for i, c in enumerate(match_counts) if c == 0]
            many = [i for i, c in enumerate(match_counts) if c > 1]
            parts = []
            if zero:
                parts.append(f"{len(zero)} anchor(s) absent (drift)")
            if many:
                parts.append(f"{len(many)} anchor(s) match >1 (ambiguous)")
            result["drift"] = "; ".join(parts)
    except Exception as e:
        result["drift"] = f"exception: {e}\n{traceback.format_exc()}"
        result["matches"] = False
    finally:
        if orig_resolve is not None:
            mod.resolve_vllm_file = orig_resolve
        if orig_install_root is not None:
            mod.vllm_install_root = orig_install_root

    return result


def _check_markers(upstream_root: Path) -> list[dict]:
    """Walk UPSTREAM_MARKERS and check each marker string against the
    upstream source. A match where `verified_in_main_*` is False means
    the upstream PR has just merged."""
    try:
        from vllm._genesis.patches.upstream_compat import UPSTREAM_MARKERS
    except Exception as e:
        return [{"error": f"upstream_compat import failed: {e}"}]

    results: list[dict] = []
    for key, info in UPSTREAM_MARKERS.items():
        # Resolve files (single or multiple)
        files = info.get("files") or ([info["file"]] if "file" in info else [])
        marker = info.get("marker")
        # Some entries have marker_decode / marker_store split shapes
        markers = []
        if marker:
            markers.append(marker)
        for k in ("marker_decode", "marker_store"):
            if k in info:
                markers.append(info[k])
        if not markers or not files:
            continue

        any_match_per_marker = []
        for m in markers:
            found_in: list[str] = []
            for rel in files:
                # Normalize: some entries pass full module path, some short
                target = upstream_root / "vllm" / rel
                if not target.is_file():
                    continue
                try:
                    content = target.read_text()
                except Exception:
                    continue
                if m in content:
                    found_in.append(rel)
            any_match_per_marker.append({
                "marker": m, "found_in": found_in,
            })

        # Was the marker recorded as "merged" already? Find the latest
        # verified_in_main_* flag
        verified_keys = [k for k in info if k.startswith("verified_in_main_")]
        already_known = any(info.get(k, False) for k in verified_keys)

        currently_present = any(m["found_in"] for m in any_match_per_marker)
        results.append({
            "key": key,
            "files": files,
            "marker_results": any_match_per_marker,
            "currently_present": currently_present,
            "already_known_merged": already_known,
            "newly_merged": currently_present and not already_known,
        })

    return results


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_upstream_drift.py <upstream-vllm-clone-path>",
              file=sys.stderr)
        return 2

    upstream = Path(argv[1]).resolve()
    if not upstream.is_dir():
        print(f"upstream path is not a directory: {upstream}", file=sys.stderr)
        return 2
    if not (upstream / "vllm").is_dir():
        print(f"no vllm/ subdir in {upstream}", file=sys.stderr)
        return 2

    head_sha = _git_head_sha(upstream)

    # Anchor drift checks
    anchors_report: dict[str, dict] = {}
    for module_name in _list_wiring_modules():
        # Use a short patch-id label keyed by module name (or whatever it
        # exposes). We strip 'vllm._genesis.wiring.' prefix for readability.
        label = module_name.replace("vllm._genesis.wiring.", "")
        anchors_report[label] = _check_one_patch(module_name, upstream)

    # Upstream-merged marker checks
    markers_report = _check_markers(upstream)

    # Compute summary
    text_patch_results = [
        v for v in anchors_report.values()
        if v.get("matches") is not None
    ]
    drifted = [v for v in text_patch_results if v["matches"] is False]
    newly_merged = [m for m in markers_report if m.get("newly_merged")]

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_path": str(upstream),
        "upstream_head_sha": head_sha,
        "anchors": anchors_report,
        "merged_markers": markers_report,
        "summary": {
            "total_text_patches": len(text_patch_results),
            "drifted_anchors": len(drifted),
            "total_markers": len(markers_report),
            "newly_merged_markers": len(newly_merged),
        },
    }

    # JSON to stdout
    print(json.dumps(report, indent=2, default=str))

    # Human summary to stderr
    print("=" * 64, file=sys.stderr)
    print(f"Genesis drift report — upstream HEAD {head_sha}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"Text-patches checked: {len(text_patch_results)}", file=sys.stderr)
    print(f"Anchor drift detected: {len(drifted)}", file=sys.stderr)
    if drifted:
        for d in drifted:
            print(f"  ⚠ {d['module']}: {d['drift']}", file=sys.stderr)
    print(f"Upstream markers checked: {len(markers_report)}", file=sys.stderr)
    print(f"Newly-merged upstream PRs: {len(newly_merged)}", file=sys.stderr)
    if newly_merged:
        for m in newly_merged:
            print(f"  ✓ {m['key']} appears in upstream now — Genesis patch "
                  f"will self-retire on next pin bump", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    # Exit code: 1 on anchor drift, 0 otherwise.
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
