#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D1 — Genesis upstream anchor-drift watcher (runtime-parity redesign).

Checks every Genesis text-patch / import-wiring against an upstream vllm
checkout and reports which patches need re-anchoring before the next pin
bump. The 2026-06-13 rewrite makes the STATIC check match RUNTIME reality
(``TextPatcher.apply()``): the old tool false-positived on every applied
patch (post-apply the anchor is gone) and false-negatived whole patch
classes (inline-builder + import-wiring patches it never looked at).

What "runtime parity" means
---------------------------
``TextPatcher.apply()`` short-circuits BEFORE the anchor scan with three
gates the old tool ignored. We replicate them so a patch the engine would
legitimately skip is never reported as drift:

  * Layer 2 — marker idempotency. If the patcher's ``marker`` is already
    in the file, apply() returns IDEMPOTENT. → status ``already_applied``.
  * Layer 3 — ``upstream_drift_markers``. If any is present, apply()
    returns SKIPPED (upstream merged the fix). → status ``upstream_merged``
    (retire, not drift).
  * Version gate — the dispatcher
    (``decision._check_applies_to`` → ``version_check.check_version_constraints``)
    skips a patch whose ``applies_to.vllm_version_range`` excludes the
    running pin. We evaluate the SAME constraint against the TREE's pin.
    → status ``version_gated_skip`` (excluded from the drift count).

This is exactly why ~164 patches "skip" at boot — they are version-gated,
NOT drifted.

Pristine-tree requirement
--------------------------
The check is only meaningful against a PRISTINE upstream clone (pre-apply
anchors present, no Genesis markers). Running against the deployed/patched
tree is structurally guaranteed to false-positive (each applied patch has
already deleted its own anchor and written its marker). The tool now
REFUSES (exit 2) to run against a tree that carries Genesis wiring markers.

Patch-class coverage (false-negatives closed)
----------------------------------------------
  1. Module-level ``_make_patcher()`` — the classic text-patch wiring.
  2. Inline-builder patches (e.g. PN347) that build their ``TextPatcher``
     inside ``apply()``. They opt in via ``_make_patcher_for_drift()``.
  3. Import-wiring monkey-patches (e.g. PN287) whose real drift vector is
     the engine moving/renaming the parser class module (exactly what
     vllm#45413 / #45588 did). They opt in via a ``_parser_targets()``-style
     accessor; if NO candidate module/class resolves under the tree the
     patch reports ``import_drift``.

Exit semantics
--------------
  0 — no genuine drift.
  1 — genuine drift detected: any anchor with status in
      {anchor_drift, import_drift}. (operator action required)
  2 — invocation error: bad path, patched tree, or --expect-pin mismatch.

Whitespace-only / needs-fixture / unbuildable cases go to a separate
NON-BLOCKING warnings bucket and never drive a non-zero exit.

Usage
-----
    python3 tools/check_upstream_drift.py <pristine-vllm-clone> \
        [--expect-pin <ver>] [--json] [--quiet]

``--expect-pin <ver>`` is read against ``<root>/vllm/_version.py`` (then
``version.py``). When the tree declares a concrete pin it is a GUARD —
a mismatch exits 2 so the gate can never silently drift-check the wrong
pin/tree. When the tree declares no concrete pin (a fresh ``git clone``
whose vcs-generated ``_version.py`` is absent), ``--expect-pin`` doubles
as the operator-asserted gating pin so version-gating still runs.

D1 design constraint: read-only against upstream clone. Never mutate.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ─── Path setup so we can import Genesis from a repo checkout ──────────

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── Status taxonomy ───────────────────────────────────────────────────
#
# Only DRIFT_STATUSES drive the non-zero (drift) exit code. Everything
# else is either clean or a non-blocking warning.

STATUS_OK = "ok"                          # anchor present exactly once
STATUS_ALREADY_APPLIED = "already_applied"  # marker already present (Layer 2)
STATUS_UPSTREAM_MERGED = "upstream_merged"  # drift marker present (Layer 3)
STATUS_VERSION_GATED = "version_gated_skip"  # applies_to excludes tree pin
STATUS_RETIRED = "retired_skip"           # lifecycle=retired — does not load
STATUS_ANCHOR_DRIFT = "anchor_drift"      # genuine: required anchor absent
STATUS_IMPORT_DRIFT = "import_drift"      # genuine: import target unresolvable
# Non-blocking warnings:
STATUS_AMBIGUOUS = "ambiguous_anchor"     # anchor matches >1 (needs tightening)
STATUS_NEEDS_FIXTURE = "needs_fixture"    # could not build patcher / no anchors
STATUS_TARGET_MISSING = "target_file_missing"  # file absent at this pin

# The ONLY statuses that mean "operator must re-anchor before pin bump".
DRIFT_STATUSES = frozenset({STATUS_ANCHOR_DRIFT, STATUS_IMPORT_DRIFT})

# Canonical Genesis wiring-marker prefix written by TextPatcher.apply()
# Layer 6 (`# [Genesis wiring marker: <marker>]`). Its presence in target
# files is the unambiguous signature of a PATCHED (not pristine) tree.
GENESIS_WIRING_MARKER_PROBE = "[Genesis wiring marker:"


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


def read_tree_pin(tree_root: Path) -> str | None:
    """Read the vllm ``__version__`` declared in the tree itself.

    Looks at ``<root>/vllm/_version.py`` first (the vcs-versioning file
    that carries the concrete pin in an installed/deployed tree), then
    ``<root>/vllm/version.py``. Returns None when no concrete version can
    be parsed (e.g. a fresh `git clone` whose `_version.py` is not
    generated yet) — callers treat None as "pin not detectable".
    """
    for rel in ("vllm/_version.py", "vllm/version.py"):
        f = tree_root / rel
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Parse `__version__ = version = '...'` style assignments without
        # importing (the file may reference build-time names we lack).
        for m in re.finditer(
            r"^\s*(?:__version__|version)\s*=.*?=\s*(['\"][^'\"]+['\"])",
            text, re.MULTILINE,
        ):
            try:
                val = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            if isinstance(val, str) and val and val != "dev":
                return val
        # Fallback: a plain `__version__ = "..."`.
        for m in re.finditer(
            r"^\s*__version__\s*=\s*(['\"][^'\"]+['\"])", text, re.MULTILINE,
        ):
            try:
                val = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            if isinstance(val, str) and val and val != "dev":
                return val
    return None


def tree_carries_genesis_markers(tree_root: Path, sample_limit: int = 400) -> bool:
    """True iff the tree appears to be PATCHED (carries Genesis markers).

    Walks the vllm/ subtree looking for the wiring-marker signature. Stops
    at the first hit — this is a guard, not a census. Bounded by
    ``sample_limit`` scanned files so an enormous tree never stalls the
    gate; the marker, when present, sits on line 1 of every patched file
    so a partial scan is sufficient in practice.
    """
    vllm_dir = tree_root / "vllm"
    scanned = 0
    for path in vllm_dir.rglob("*.py"):
        scanned += 1
        if scanned > sample_limit:
            break
        try:
            # Marker is always the FIRST line; read a small head only.
            with open(path, encoding="utf-8", errors="ignore") as fh:
                head = fh.read(256)
        except OSError:
            continue
        if GENESIS_WIRING_MARKER_PROBE in head:
            return True
    return False


# ─── Spec enumeration ───────────────────────────────────────────────────


def _iter_specs():
    """Yield every PatchSpec that has an on-disk ``apply_module``.

    Unlike the pre-rewrite tool we do NOT pre-filter on
    ``implementation_status`` here — the per-module discovery step decides
    whether a module is checkable (has a builder / import-target accessor)
    and the version gate handles "doesn't apply at this pin". Filtering on
    status up front was one source of false-negatives (it dropped patches
    whose status string didn't match a hardcoded set)."""
    try:
        from sndr.dispatcher.spec import iter_patch_specs
    except Exception as exc:  # pragma: no cover — defensive
        print(f"check_upstream_drift: iter_patch_specs unavailable: {exc}",
              file=sys.stderr)
        return
    for spec in iter_patch_specs():
        if getattr(spec, "apply_module", None):
            yield spec


# ─── Runtime-parity gate ─────────────────────────────────────────────────


def _version_gated_out(spec, tree_pin: str | None) -> tuple[bool, str]:
    """True iff the patch's applies_to.vllm_version_range EXCLUDES tree_pin.

    Mirrors the dispatcher gate
    (decision._check_applies_to → version_check.check_version_constraints)
    but evaluates against the TREE's pin rather than the host's. When the
    tree pin is unknown OR no range is declared we DO NOT gate (conservative
    — let the anchor scan run, same posture as the engine when detection is
    imperfect).
    """
    if not tree_pin:
        return False, "tree pin unknown — not version-gated"
    applies_to = getattr(spec, "applies_to", None) or {}
    vrange = applies_to.get("vllm_version_range")
    if not vrange:
        return False, "no vllm_version_range declared"
    try:
        from sndr.compat.version_check import (
            VersionProfile,
            check_version_constraints,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return False, f"version_check unavailable: {exc}"
    profile = VersionProfile(vllm=tree_pin)
    ok, _results = check_version_constraints(
        {"vllm_version_range": vrange}, profile=profile,
    )
    # ok=True → pin satisfies the range → patch applies → NOT gated out.
    return (not ok), (
        f"vllm_version_range {vrange!r} {'excludes' if not ok else 'includes'} "
        f"{tree_pin}"
    )


def _marker_present(content: str, patcher) -> bool:
    marker = getattr(patcher, "marker", None)
    return bool(marker) and marker in content


def _drift_marker_present(content: str, patcher) -> str | None:
    for m in getattr(patcher, "upstream_drift_markers", []) or []:
        if m in content:
            return m
    return None


# ─── Per-patcher anchor check (text-patch + inline-builder classes) ──────


def check_patcher_anchors(patcher, tree_root: Path) -> dict[str, Any]:
    """Check one TextPatcher's anchors against the tree, with full runtime
    parity (marker idempotency + upstream-drift markers).

    Returns a result dict with at least ``status`` and ``file`` keys. The
    status is one of the STATUS_* constants. Genuine drift only ever yields
    ``anchor_drift``; everything else is clean or a warning.
    """
    result: dict[str, Any] = {
        "patch_name": getattr(patcher, "patch_name", None),
        "file": None,
        "status": None,
        "anchor_count": 0,
        "detail": None,
    }

    target = Path(patcher.target_file)
    if not target.is_file():
        result["status"] = STATUS_TARGET_MISSING
        result["detail"] = f"target file absent in tree: {target}"
        return result

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        result["status"] = STATUS_NEEDS_FIXTURE
        result["detail"] = f"target read failed: {e}"
        return result

    try:
        result["file"] = str(target.relative_to(tree_root))
    except ValueError:
        result["file"] = str(target)

    # ── Runtime parity Layer 2: marker idempotency. ──────────────────────
    # On a pristine tree the marker is absent; if it IS present the tree
    # is (partially) patched and apply() would short-circuit IDEMPOTENT.
    if _marker_present(content, patcher):
        result["status"] = STATUS_ALREADY_APPLIED
        result["detail"] = "patcher marker already present (apply() idempotent)"
        return result

    # ── Runtime parity Layer 3: upstream-merged drift markers. ───────────
    dm = _drift_marker_present(content, patcher)
    if dm is not None:
        result["status"] = STATUS_UPSTREAM_MERGED
        result["detail"] = f"upstream_drift_marker {dm!r} present — retire candidate"
        return result

    # ── Anchor scan (Layer 5 parity: required anchors must appear once). ─
    sub_patches = getattr(patcher, "sub_patches", []) or []
    if not sub_patches:
        result["status"] = STATUS_NEEDS_FIXTURE
        result["detail"] = "patcher declares no sub_patches"
        return result

    zero_required: list[str] = []
    ambiguous: list[str] = []
    total = 0
    for sp in sub_patches:
        anchor = getattr(sp, "anchor", None)
        required = getattr(sp, "required", False)
        if not anchor:
            continue
        # Per-sub upstream-merge markers (TextPatch.upstream_merged_markers)
        # — apply() soft-skips these subs; not drift.
        sub_markers = getattr(sp, "upstream_merged_markers", []) or []
        if any(um in content for um in sub_markers):
            continue
        count = content.count(anchor)
        total += count
        if count == 0:
            if required:
                zero_required.append(getattr(sp, "name", "?"))
        elif count > 1:
            ambiguous.append(getattr(sp, "name", "?"))

    result["anchor_count"] = total

    if zero_required:
        result["status"] = STATUS_ANCHOR_DRIFT
        result["detail"] = (
            f"required anchor(s) absent: {', '.join(zero_required)} — "
            "upstream refactored this region; re-derive anchor"
        )
        return result
    if ambiguous:
        # Ambiguity is a tightening task, not a pin-blocking drift.
        result["status"] = STATUS_AMBIGUOUS
        result["detail"] = (
            f"anchor(s) match >1: {', '.join(ambiguous)} — tighten anchor"
        )
        return result

    result["status"] = STATUS_OK
    return result


# ─── Per-module discovery ────────────────────────────────────────────────


# _build_patcher_for_module moved to the shared anchor_discovery module (Phase 1
# extraction, 2026-06-21) so the drift-checker and the per-pin manifest
# generator share ONE enumerator (design requirement R1 — single source of
# "what to anchor"). Imported here to preserve this tool's existing call sites.
from sndr.engines.vllm.anchor_discovery import (  # noqa: E402
    _build_patcher_for_module,
)


# ─── Import-wiring check (PN287 class) ───────────────────────────────────


def check_import_wiring(mod, tree_root: Path) -> dict[str, Any] | None:
    """Resolve an import-wiring patch's target classes under the tree.

    Convention (PN287): the module exposes ``_parser_targets()`` returning
    tuples ``(label, class_name, candidate_modules, factory)``. For each
    target we attempt to import the class from the FIRST resolvable
    candidate module under the upstream tree. Outcome:

      * at least one target resolves → ``ok`` (the engine moving a parser
        module to its OTHER candidate path is exactly the drift this guards;
        as long as one candidate still resolves the monkey-patch wires up).
      * a resolvable class already carries the patch's upstream-merge
        marker (``_UPSTREAM_DRIFT_MARKER`` attr) → ``upstream_merged``.
      * NO target resolves from ANY candidate → ``import_drift`` (the real
        drift vector — engine renamed/removed the class module; vllm#45413
        / #45588 class).

    Returns None when the module is not an import-wiring patch (no
    ``_parser_targets``), so the caller can fall through to other checks.

    Resolution is performed against the TREE, not the host's installed
    vllm: each candidate dotted path is mapped to a file under
    ``<tree>/vllm/...`` and the class presence is checked by AST scan (no
    import side-effects, no host-vllm contamination).
    """
    targets_fn = getattr(mod, "_parser_targets", None)
    if targets_fn is None:
        return None
    try:
        targets = targets_fn()
    except Exception as e:  # noqa: BLE001
        return {
            "status": STATUS_NEEDS_FIXTURE,
            "detail": f"_parser_targets() raised: {e}",
        }

    drift_marker_attr = getattr(mod, "_UPSTREAM_DRIFT_MARKER", None)

    resolved_labels: list[str] = []
    merged_labels: list[str] = []
    tried: list[str] = []
    for target in targets:
        # Support both the 4-tuple (label, class_name, candidates, factory)
        # PN287 convention and the 3-tuple (label, class_name, candidates)
        # PN392 convention — only the first three fields drive resolution;
        # the factory (if present) is irrelevant to a static drift check.
        if not isinstance(target, (tuple, list)) or len(target) < 3:
            continue
        label, class_name, candidates = target[0], target[1], target[2]
        for dotted in candidates:
            rel = dotted.replace(".", "/")
            # `vllm.x.y` → tree/vllm/x/y.py (or package __init__.py).
            file_py = tree_root / (rel + ".py")
            file_pkg = tree_root / rel / "__init__.py"
            src_path = (
                file_py if file_py.is_file()
                else file_pkg if file_pkg.is_file()
                else None
            )
            tried.append(dotted)
            if src_path is None:
                continue
            if _class_defined_in_file(src_path, class_name):
                resolved_labels.append(label)
                if drift_marker_attr and _attr_assigned_in_class(
                    src_path, class_name, drift_marker_attr,
                ):
                    merged_labels.append(label)
                break  # first resolvable candidate wins for this target

    if merged_labels and not resolved_labels:
        # Unreachable (merged implies resolved) — kept for clarity.
        pass

    if resolved_labels:
        if merged_labels and len(merged_labels) == len(resolved_labels):
            return {
                "status": STATUS_UPSTREAM_MERGED,
                "detail": (
                    f"upstream class(es) {', '.join(merged_labels)} carry "
                    f"{drift_marker_attr!r} — patch self-retires"
                ),
            }
        return {
            "status": STATUS_OK,
            "detail": f"import target(s) resolved: {', '.join(resolved_labels)}",
        }

    return {
        "status": STATUS_IMPORT_DRIFT,
        "detail": (
            "no import target resolved from any candidate path "
            f"({', '.join(sorted(set(tried)))}) — engine moved/renamed the "
            "parser class module; re-wire import targets"
        ),
    }


def _class_defined_in_file(path: Path, class_name: str) -> bool:
    """True iff ``class_name`` is defined (class def) in the file. AST-based
    so we never import upstream code (no side-effects, no host-vllm clash)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
    return False


def _attr_assigned_in_class(path: Path, class_name: str, attr: str) -> bool:
    """True iff ``attr`` is assigned as a class-body attribute of
    ``class_name`` in the file (upstream-merge marker detection)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == attr:
                            return True
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name,
                ) and stmt.target.id == attr:
                    return True
    return False


# ─── One patch ───────────────────────────────────────────────────────────


def _check_one_spec(spec, tree_root: Path, tree_pin: str | None) -> dict[str, Any]:
    """Resolve and check a single patch spec end-to-end."""
    result: dict[str, Any] = {
        "module": spec.apply_module,
        "status": None,
        "file": None,
        "detail": None,
    }

    # Retired/deprecated patches do NOT load at runtime
    # (registry_metadata: retired lifecycle short-circuits to blocked).
    # Their anchors being gone is expected and never blocks a pin bump —
    # classify as a non-blocking skip to match runtime reality.
    lifecycle = str(getattr(spec, "lifecycle", "") or "").lower()
    impl_status = str(getattr(spec, "implementation_status", "") or "").lower()
    if lifecycle in ("retired", "deprecated") or impl_status in (
        "retired", "deprecated",
    ):
        result["status"] = STATUS_RETIRED
        result["detail"] = (
            f"lifecycle={lifecycle or impl_status} — does not load at runtime; "
            "anchor state not pin-blocking"
        )
        return result

    # Version gate (cheap, no import) — matches the dispatcher, which skips
    # out-of-range patches before touching the file.
    gated, reason = _version_gated_out(spec, tree_pin)
    if gated:
        result["status"] = STATUS_VERSION_GATED
        result["detail"] = reason
        return result

    try:
        mod = importlib.import_module(spec.apply_module)
    except Exception as e:  # noqa: BLE001
        result["status"] = STATUS_NEEDS_FIXTURE
        result["detail"] = f"import failed: {e}"
        return result

    # Import-wiring patches (PN287 class) — checked by class resolution.
    if getattr(mod, "_parser_targets", None) is not None:
        wiring = check_import_wiring(mod, tree_root)
        if wiring is not None:
            result.update(wiring)
            return result

    # Text-patch / inline-builder patches — build patcher, scan anchors.
    builder_attrs = ("_make_patcher", "_make_patcher_for_drift")
    if not any(getattr(mod, a, None) for a in builder_attrs):
        result["status"] = STATUS_NEEDS_FIXTURE
        result["detail"] = (
            "non-text-patch wiring (no _make_patcher / "
            "_make_patcher_for_drift / _parser_targets) — not statically "
            "checkable; covered by runtime self-test"
        )
        return result

    # Point Genesis path resolution at the TREE for the duration of the
    # build. We patch guards.vllm_install_root ONCE — resolve_vllm_file
    # dispatches through it (see guards.resolve_vllm_file), so every wiring
    # module's resolution flows to the tree regardless of how it imported
    # the helper. Far more robust than per-module attribute patching.
    from sndr.engines.vllm.detection import guards
    orig_root = guards.vllm_install_root
    try:
        guards.vllm_install_root = lambda: str(tree_root / "vllm")
        patcher, note = _build_patcher_for_module(mod)
        if patcher is None:
            result["status"] = STATUS_NEEDS_FIXTURE
            result["detail"] = note
            return result
        try:
            anchor_result = check_patcher_anchors(patcher, tree_root)
        except Exception as e:  # noqa: BLE001
            result["status"] = STATUS_NEEDS_FIXTURE
            result["detail"] = f"anchor check raised: {e}\n{traceback.format_exc()}"
            return result
        result.update(anchor_result)
        result["module"] = spec.apply_module
        return result
    finally:
        guards.vllm_install_root = orig_root


# ─── Upstream-merged marker scan (unchanged contract) ────────────────────


def _check_markers(tree_root: Path) -> list[dict]:
    """Walk UPSTREAM_MARKERS and check each marker string against the tree.
    A match where the marker was not previously recorded as merged means
    the upstream PR just landed → the Genesis patch should self-retire."""
    try:
        from sndr.engines.vllm.upstream_compat import UPSTREAM_MARKERS
    except Exception as e:
        return [{"error": f"upstream_compat import failed: {e}"}]

    results: list[dict] = []
    for key, info in UPSTREAM_MARKERS.items():
        files = info.get("files") or ([info["file"]] if "file" in info else [])
        markers: list[str] = []
        if info.get("marker"):
            markers.append(info["marker"])
        for k in ("marker_decode", "marker_store"):
            if k in info:
                markers.append(info[k])
        if not markers or not files:
            continue

        per_marker = []
        for m in markers:
            found_in: list[str] = []
            for rel in files:
                target = tree_root / "vllm" / rel
                if not target.is_file():
                    continue
                try:
                    content = target.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if m in content:
                    found_in.append(rel)
            per_marker.append({"marker": m, "found_in": found_in})

        verified_keys = [k for k in info if k.startswith("verified_in_main_")]
        already_known = any(info.get(k, False) for k in verified_keys)
        currently_present = any(pm["found_in"] for pm in per_marker)
        results.append({
            "key": key,
            "files": files,
            "marker_results": per_marker,
            "currently_present": currently_present,
            "already_known_merged": already_known,
            "newly_merged": currently_present and not already_known,
        })
    return results


# ─── Orchestration ───────────────────────────────────────────────────────


class PristineGuardError(RuntimeError):
    """Raised when the target tree is not a pristine upstream clone."""


def run_drift_check(
    tree_root: Path,
    *,
    expect_pin: str | None = None,
    enforce_pristine: bool = True,
) -> dict[str, Any]:
    """Run the full drift check against ``tree_root`` (the dir CONTAINING
    ``vllm/``). Returns the report dict.

    Raises ``PristineGuardError`` when ``enforce_pristine`` and the tree
    carries Genesis wiring markers, or when ``expect_pin`` is given and the
    tree's detected pin mismatches. The CLI maps these to exit 2.
    """
    tree_root = Path(tree_root).resolve()

    detected_pin = read_tree_pin(tree_root)
    # The pin used for version-gating. When the tree declares a concrete
    # version we use it; --expect-pin then acts as a guard (mismatch → exit
    # 2). When the tree declares no pin (typical for a fresh `git clone`
    # whose vcs-generated _version.py is absent), --expect-pin doubles as
    # the operator-asserted pin so version-gating still runs correctly.
    tree_pin = detected_pin

    if expect_pin is not None:
        if detected_pin is not None and detected_pin != expect_pin:
            raise PristineGuardError(
                f"pin mismatch: --expect-pin {expect_pin} but tree declares "
                f"{detected_pin}. Refusing to drift-check the wrong pin/tree."
            )
        # Tree declares no pin OR declares the same pin — trust the operator
        # assertion as the gating pin.
        tree_pin = expect_pin

    if enforce_pristine and tree_carries_genesis_markers(tree_root):
        raise PristineGuardError(
            f"{tree_root} is a PATCHED tree (carries Genesis wiring markers), "
            "not a pristine upstream clone. Anchor drift is structurally "
            "meaningless against a patched tree: every applied patch has "
            "already deleted its own anchor and written its marker, so every "
            "applied patch would false-positive. Point this tool at a "
            "PRISTINE upstream vllm clone (pre-apply anchors present, no "
            "Genesis markers)."
        )

    anchors_report: dict[str, dict] = {}
    for spec in _iter_specs():
        anchors_report[spec.patch_id] = _check_one_spec(spec, tree_root, tree_pin)

    markers_report = _check_markers(tree_root)

    # Tally by status.
    by_status: dict[str, int] = {}
    for r in anchors_report.values():
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    drifted = [
        pid for pid, r in anchors_report.items() if r["status"] in DRIFT_STATUSES
    ]
    warnings = [
        pid for pid, r in anchors_report.items()
        if r["status"] in (STATUS_AMBIGUOUS, STATUS_NEEDS_FIXTURE,
                            STATUS_TARGET_MISSING)
    ]
    newly_merged = [m for m in markers_report if m.get("newly_merged")]

    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree_path": str(tree_root),
        "tree_pin": tree_pin,
        "tree_head_sha": _git_head_sha(tree_root),
        "anchors": anchors_report,
        "merged_markers": markers_report,
        "summary": {
            "total_patches": len(anchors_report),
            "by_status": by_status,
            "drifted": drifted,
            "drifted_count": len(drifted),
            "warnings": warnings,
            "warnings_count": len(warnings),
            "total_markers": len(markers_report),
            "newly_merged_markers": len(newly_merged),
        },
    }


# ─── CLI ─────────────────────────────────────────────────────────────────


def _print_human_summary(report: dict, *, file=sys.stderr) -> None:
    s = report["summary"]
    p = print
    p("=" * 68, file=file)
    p(f"Genesis drift report — tree {report['tree_head_sha']} "
      f"(pin {report['tree_pin']})", file=file)
    p("=" * 68, file=file)
    p(f"Patches checked: {s['total_patches']}", file=file)
    p("Status breakdown:", file=file)
    for status, n in sorted(s["by_status"].items()):
        p(f"  {status:24s} {n}", file=file)
    p("", file=file)
    p(f"GENUINE DRIFT (action required): {s['drifted_count']}", file=file)
    for pid in s["drifted"]:
        r = report["anchors"][pid]
        p(f"  ⚠ {pid}: {r['status']} — {r.get('detail')}", file=file)
    if s["warnings_count"]:
        p(f"Warnings (non-blocking): {s['warnings_count']}", file=file)
        for pid in s["warnings"]:
            r = report["anchors"][pid]
            p(f"  · {pid}: {r['status']} — {r.get('detail')}", file=file)
    p(f"Upstream markers checked: {s['total_markers']} "
      f"(newly merged: {s['newly_merged_markers']})", file=file)
    for m in report["merged_markers"]:
        if m.get("newly_merged"):
            p(f"  ✓ {m['key']} now in upstream — Genesis patch self-retires "
              "on next pin bump", file=file)
    p("=" * 68, file=file)


def main(argv: list[str]) -> int:
    args = argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0

    expect_pin: str | None = None
    want_json = False
    quiet = False
    positionals: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--expect-pin":
            i += 1
            if i >= len(args):
                print("--expect-pin requires a version argument", file=sys.stderr)
                return 2
            expect_pin = args[i]
        elif a.startswith("--expect-pin="):
            expect_pin = a.split("=", 1)[1]
        elif a == "--json":
            want_json = True
        elif a == "--quiet":
            quiet = True
        elif a.startswith("-"):
            print(f"unknown option: {a}", file=sys.stderr)
            return 2
        else:
            positionals.append(a)
        i += 1

    if len(positionals) != 1:
        print("usage: check_upstream_drift.py <pristine-vllm-clone> "
              "[--expect-pin VER] [--json] [--quiet]", file=sys.stderr)
        return 2

    tree_root = Path(positionals[0]).resolve()
    if not tree_root.is_dir():
        print(f"tree path is not a directory: {tree_root}", file=sys.stderr)
        return 2
    if not (tree_root / "vllm").is_dir():
        print(f"no vllm/ subdir in {tree_root}", file=sys.stderr)
        return 2

    try:
        report = run_drift_check(tree_root, expect_pin=expect_pin)
    except PristineGuardError as e:
        print(f"check_upstream_drift: {e}", file=sys.stderr)
        return 2

    if want_json:
        print(json.dumps(report, indent=2, default=str))
    if not quiet:
        _print_human_summary(report)

    return 1 if report["summary"]["drifted_count"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
