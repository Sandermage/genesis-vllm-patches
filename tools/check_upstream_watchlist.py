#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the ``sweep:`` section of ``tools/upstream_watchlist.yaml``.

Added 2026-06-11 with the 50-PR upstream sweep (see
docs/superpowers/journal/2026-06-11-pr-sweep-50-roadmap.md). The
watchlist file carries TWO top-level sections:

  * ``watch:``  — legacy entries (upstream/status/action/since/notes),
                  validated by ``scripts/audit_upstream_watchlist.py``
                  (``make audit-upstream-watchlist``). Untouched here.
  * ``sweep:``  — one row per deep-studied upstream PR that needs
                  merge-event bookkeeping. Validated by THIS script
                  (``make watchlist-check``).

``sweep:`` row schema (all keys required):

  pr            int  — upstream vllm PR number
  genesis_patch str  — existing Genesis patch id(s) tied to the PR,
                       ``planned: <id>`` for a not-yet-vendored patch,
                       or ``watch-only`` when no code is involved
  trigger       str  — what to do when the PR merges into a pin:
                       retire-on-merge   (deep-diff, then retire vendor)
                       reanchor-on-merge (anchors break; re-derive)
                       review-on-merge   (re-read; no automatic action)
  note          str  — non-empty context (duplicates, clusters, plans)

Registry-binding check (``--check-registry``)
---------------------------------------------
Schema validation alone lets a sweep entry go STALE: a ``reanchor-on-merge``
row names ``PN346`` as the patch that drifts when vllm#46384 lands, but if
PN346 is later renamed / retired / its required anchors made ``required=False``,
the binding between the watchlist and the mechanical drift detector silently
breaks — the incoming-drift watch no longer points at a live, drift-detectable
patch, and the next bump would NOT surface PN346 in ``genuine_anchor_drift``.

``--check-registry`` closes that gap. For every sweep row whose
``genesis_patch`` names CONCRETE patch ids (i.e. not ``planned: …`` /
``watch-only``):

  * the id must EXIST in the live ``PATCH_REGISTRY`` (catches rename/removal);
  * a ``reanchor-on-merge`` / ``retire-on-merge`` target must NOT already be
    retired (a "re-anchor / retire on merge" intent contradicts an already-dead
    patch — the entry is stale and should move to ``review-on-merge`` or drop);
  * a ``reanchor-on-merge`` target must carry at least one ``required=True``
    anchor sub-patch. This is the load-bearing robustness invariant: the
    mechanical drift detector (anchor-SOT ``build_manifest``) only routes an
    absent REQUIRED anchor to ``genuine_anchor_drift``. An absent OPTIONAL
    anchor SOFT-SKIPS (status ``optional_absent``) and the patch still reports
    ``ok`` — so a reanchor-on-merge entry whose patch has only ``required=False``
    anchors would silently no-op exactly the incoming drift the entry exists to
    catch (the PN340/PN129 class).

Exit codes: 0 = clean, 2 = schema error or missing/unreadable file,
            3 = registry-binding error (``--check-registry`` only).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCHLIST = REPO_ROOT / "tools" / "upstream_watchlist.yaml"

REQUIRED_KEYS = ("pr", "genesis_patch", "trigger", "note")
ALLOWED_TRIGGERS = frozenset({
    "retire-on-merge",
    "reanchor-on-merge",
    "review-on-merge",
})

# Triggers that name a CONCRETE, live patch whose drift/retire must remain
# mechanically detectable. ``review-on-merge`` is advisory (re-read on merge,
# no automatic action) so it is NOT registry-bound — a ``review-on-merge`` row
# may legitimately reference a patch that no longer applies.
_BINDING_TRIGGERS = frozenset({"retire-on-merge", "reanchor-on-merge"})
# A reanchor-on-merge target MUST carry a required anchor (see module docstring).
_NEEDS_REQUIRED_ANCHOR_TRIGGERS = frozenset({"reanchor-on-merge"})

# Optional per-row ``detection:`` field — declares HOW a reanchor-on-merge entry
# is caught when the PR lands, so the binding check can verify a real detection
# path exists rather than assuming the anchor-SOT manifest. Values:
#   anchor (default) — a required=True anchor; the per-pin build_manifest routes
#                      the absent anchor to genuine_anchor_drift.
#   test             — a dedicated drift test imports the patch's anchor
#                      constants and FAILS LOUDLY on drift (the PN340/PN341
#                      chain tests). Waives the required-anchor invariant.
#   manual           — operator reconciliation documented in ``note`` (the
#                      G4_60E shared-function fold). Waives required-anchor.
_ALLOWED_DETECTION = frozenset({"anchor", "test", "manual"})
_NON_ANCHOR_DETECTION = frozenset({"test", "manual"})


def parse_genesis_patch_ids(genesis_patch: str) -> list[str]:
    """Extract the CONCRETE patch ids named by a sweep row's ``genesis_patch``.

    The field carries three forms (see module docstring):
      * concrete ids, comma-separated — ``"PN346, PN346B"`` -> ["PN346","PN346B"]
      * ``planned: <id>``             — not yet vendored, returns []
      * ``watch-only``                — no code involved, returns []

    Only concrete (already-vendored) ids are registry-bound; ``planned:`` and
    ``watch-only`` deliberately return [] so they are skipped by the binding
    check (there is no live patch to bind to yet).
    """
    gp = (genesis_patch or "").strip()
    if not gp or gp == "watch-only" or gp.lower().startswith("planned:"):
        return []
    return [tok.strip() for tok in gp.split(",") if tok.strip()]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed (`pip install pyyaml`)",
              file=sys.stderr)
        raise SystemExit(2)
    if not path.is_file():
        print(f"ERROR: watchlist not found at {path}", file=sys.stderr)
        raise SystemExit(2)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sweep(path: Path) -> list[dict[str, Any]]:
    """Return the raw ``sweep:`` rows (no validation)."""
    data = _load_yaml(path)
    rows = data.get("sweep")
    return rows if isinstance(rows, list) else []


def validate_rows(rows: list[Any]) -> list[str]:
    """Return a list of schema errors; empty list = clean."""
    errors: list[str] = []
    seen_prs: set[int] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row #{idx}: must be a mapping")
            continue
        for key in REQUIRED_KEYS:
            if key not in row:
                errors.append(f"row #{idx}: missing required key `{key}`")
        pr = row.get("pr")
        if not isinstance(pr, int) or isinstance(pr, bool):
            errors.append(f"row #{idx}: `pr` must be an int "
                          f"(got {pr!r})")
        else:
            if pr in seen_prs:
                errors.append(f"row #{idx}: duplicate pr {pr}")
            seen_prs.add(pr)
        trigger = row.get("trigger")
        if "trigger" in row and trigger not in ALLOWED_TRIGGERS:
            errors.append(
                f"row #{idx} (pr={pr!r}): trigger={trigger!r} must be "
                f"one of {sorted(ALLOWED_TRIGGERS)}"
            )
        for key in ("genesis_patch", "note"):
            val = row.get(key)
            if key in row and (not isinstance(val, str) or not val.strip()):
                errors.append(
                    f"row #{idx} (pr={pr!r}): `{key}` must be a "
                    f"non-empty string"
                )
        # `detection:` is OPTIONAL (defaults to "anchor"); when present it must
        # be a known value so a typo can't silently waive the required-anchor
        # binding invariant.
        if "detection" in row:
            det = row.get("detection")
            if det not in _ALLOWED_DETECTION:
                errors.append(
                    f"row #{idx} (pr={pr!r}): detection={det!r} must be one of "
                    f"{sorted(_ALLOWED_DETECTION)}"
                )
    return errors


def check_registry_binding(
    rows: list[dict[str, Any]],
    *,
    registry_ids: set[str],
    retired_ids: set[str],
    has_required_anchor: set[str],
) -> list[str]:
    """Return registry-binding errors for the sweep rows; [] = clean.

    PURE + injectable (tests pass synthetic id sets; the CLI derives them from
    the live registry via ``load_registry_state``). Checks every CONCRETE-id
    sweep row against three invariants (see module docstring):

      1. existence  — a named id must be in ``registry_ids`` (rename/removal).
      2. not-retired — a retire/reanchor-on-merge target must NOT be in
         ``retired_ids`` (the intent contradicts an already-dead patch).
      3. required-anchor — a reanchor-on-merge target must be in
         ``has_required_anchor`` (else its incoming drift soft-skips, never
         surfacing as genuine_anchor_drift — the PN340/PN129 class).
    """
    errors: list[str] = []
    for row in rows:
        trigger = row.get("trigger")
        if trigger not in _BINDING_TRIGGERS:
            continue
        pr = row.get("pr")
        detection = str(row.get("detection") or "anchor").strip().lower()
        if detection not in _ALLOWED_DETECTION:
            errors.append(
                f"pr={pr} trigger={trigger}: detection={detection!r} must be "
                f"one of {sorted(_ALLOWED_DETECTION)}")
            detection = "anchor"  # continue with the strictest interpretation
        for pid in parse_genesis_patch_ids(row.get("genesis_patch", "")):
            if pid not in registry_ids:
                errors.append(
                    f"pr={pr} trigger={trigger}: genesis_patch {pid!r} not in "
                    f"PATCH_REGISTRY (renamed/removed?) — the watchlist binding "
                    f"is stale; update genesis_patch or drop the row"
                )
                continue
            if pid in retired_ids:
                errors.append(
                    f"pr={pr} trigger={trigger}: genesis_patch {pid!r} is "
                    f"already RETIRED — a {trigger} intent contradicts a dead "
                    f"patch; move to review-on-merge or drop the row"
                )
                continue
            # A reanchor-on-merge target must have SOME declared detection path:
            # either a required=True anchor (the manifest catches it) OR an
            # explicit detection=test/manual saying how else it is caught. An
            # anchor-detection reanchor target with NO required anchor would
            # soft-skip its incoming drift (status optional_absent), never
            # surfacing as genuine_anchor_drift — the PN340/PN341 class.
            if (trigger in _NEEDS_REQUIRED_ANCHOR_TRIGGERS
                    and detection not in _NON_ANCHOR_DETECTION
                    and pid not in has_required_anchor):
                errors.append(
                    f"pr={pr} trigger=reanchor-on-merge: genesis_patch {pid!r} "
                    f"carries NO required=True anchor and no detection: override "
                    f"— its incoming drift would soft-skip (status "
                    f"optional_absent) and never surface as genuine_anchor_drift, "
                    f"so reanchor-on-merge cannot be caught mechanically. Either "
                    f"make the load-bearing anchor required=True, OR declare "
                    f"detection: test|manual on the row documenting how the "
                    f"drift IS caught (e.g. a constant-import drift test)"
                )
    return errors


def _module_source(apply_module: Optional[str]) -> Optional[str]:
    """Read a patch's apply-module source as text WITHOUT importing it.

    Resolves the dotted module to a file via importlib's spec (no import, no
    torch/vLLM side effects). Mirrors retire_impact._read_module_source so the
    required-anchor scan runs host-side (no rig, no installed vLLM)."""
    if not apply_module:
        return None
    try:
        spec = importlib.util.find_spec(apply_module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        with open(spec.origin, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


_REQUIRED_TRUE_RE = re.compile(r"required\s*=\s*True\b")


def load_registry_state() -> tuple[set[str], set[str], set[str]]:
    """Derive (registry_ids, retired_ids, has_required_anchor) from the live
    dispatcher registry — host-runnable (no installed vLLM required).

    ``has_required_anchor`` is a STATIC source scan for a ``required=True``
    occurrence in the patch's apply-module (the same posture retire_impact uses
    to read anchor source host-side): the per-pin manifest's required/optional
    split is what decides genuine_anchor_drift vs optional_absent, and a
    ``required=True`` sub-patch is the thing that surfaces an absent anchor as
    genuine drift. The byte-level patcher build needs the rig, so the source
    scan is the host-side proxy.
    """
    from sndr.dispatcher.spec import iter_patch_specs

    registry_ids: set[str] = set()
    retired_ids: set[str] = set()
    has_required_anchor: set[str] = set()
    for spec in iter_patch_specs():
        pid = spec.patch_id
        registry_ids.add(pid)
        lifecycle = str(getattr(spec, "lifecycle", "") or "").lower()
        impl = str(getattr(spec, "implementation_status", "") or "").lower()
        if lifecycle in ("retired", "deprecated") or impl in (
            "retired", "deprecated",
        ):
            retired_ids.add(pid)
        src = _module_source(getattr(spec, "apply_module", None))
        if src and _REQUIRED_TRUE_RE.search(src):
            has_required_anchor.add(pid)
    return registry_ids, retired_ids, has_required_anchor


def _render_text(rows: list[dict[str, Any]]) -> str:
    by_trigger: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_trigger.setdefault(r["trigger"], []).append(r)
    out = ["Upstream sweep watchlist", "=" * 60]
    for trig in sorted(ALLOWED_TRIGGERS):
        items = by_trigger.get(trig, [])
        if not items:
            continue
        out.append(f"\n[{trig}] {len(items)} rows")
        for r in sorted(items, key=lambda x: x["pr"]):
            out.append(f"  - vllm#{r['pr']:<7} -> {r['genesis_patch']}")
    out.append("")
    out.append(f"Total: {len(rows)} rows")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", type=Path,
                        default=DEFAULT_WATCHLIST,
                        help="Path to upstream_watchlist.yaml")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")
    parser.add_argument(
        "--check-registry", action="store_true",
        help="Also verify each concrete genesis_patch id is live + "
             "drift-detectable (exists, not retired, reanchor targets carry a "
             "required anchor). Exit 3 on a binding error.")
    args = parser.parse_args(argv)

    try:
        rows = load_sweep(args.watchlist)
    except SystemExit as exc:
        return int(exc.code or 2)

    if not rows:
        print(f"ERROR: no `sweep:` rows found in {args.watchlist}",
              file=sys.stderr)
        return 2

    errors = validate_rows(rows)
    if errors:
        if args.json:
            print(json.dumps({"schema_errors": errors}, indent=2))
        else:
            print("Schema errors:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        return 2

    binding_errors: list[str] = []
    if args.check_registry:
        try:
            registry_ids, retired_ids, has_required_anchor = (
                load_registry_state())
        except Exception as exc:  # noqa: BLE001 — registry unavailable
            print(f"ERROR: --check-registry needs the dispatcher registry "
                  f"importable: {exc}", file=sys.stderr)
            return 2
        binding_errors = check_registry_binding(
            rows, registry_ids=registry_ids, retired_ids=retired_ids,
            has_required_anchor=has_required_anchor)

    if args.json:
        print(json.dumps(
            {"rows": rows, "schema_errors": [],
             "binding_errors": binding_errors},
            indent=2, sort_keys=False, default=str))
    else:
        print(_render_text(rows))
        if args.check_registry and binding_errors:
            print("\nRegistry-binding errors (stale watchlist <-> live "
                  "registry):", file=sys.stderr)
            for e in binding_errors:
                print(f"  - {e}", file=sys.stderr)
        elif args.check_registry:
            print("\nRegistry binding: OK — every concrete genesis_patch id is "
                  "live and drift-detectable.")
    return 3 if binding_errors else 0


if __name__ == "__main__":
    sys.exit(main())
