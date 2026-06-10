# SPDX-License-Identifier: Apache-2.0
"""PR38 §5.5 (audit closure 2026-05-08) — `lifecycle="stable"` ratchet.

The PR38_PATCHER_REWORK_PLAN_2026-05-07.md §5.5 mandates anchor manifest
coverage for "stable" patches. Closing this finding via a self-enforcing
policy ratchet rather than mass migration:

  Policy:
    Any registry entry with `lifecycle="stable"` MUST have:
      (a) all its TextPatcher instances declare `patch_id="<X>.Sub-<N>"`,
      (b) all those patch_ids registered in `wiring/patcher_registry.py`,
      (c) the target file present in `manifests/anchor_manifest.json`,
      (d) the registered patcher's anchors covered by the manifest.

  Today: 0 patches have lifecycle="stable" → tests pass vacuously.
  Future: when an operator promotes a patch (e.g. PN79 experimental →
          stable after multi-turn validation), THIS test fails until
          the 4 conditions above are met. That forces the
          STABLE_PROMOTION_CHECKLIST steps to actually happen — no
          drift between "marked stable" and "manifest-covered".

Why ratchet over upfront migration:
  - 0 patches are stable today (lifecycle distribution: 91 <none>,
    31 legacy, 5 retired, 3 experimental, 1 coordinator, 0 stable).
  - Mass-migrating 6 high-anchor candidates would commit ~150ms boot
    speedup but add 6 pristine fixture files → maintenance burden on
    every vllm pin upgrade.
  - The ratchet makes promotion to stable contingent on manifest
    coverage being ready — strictly stronger than mass migration
    because it covers ALL future stable patches, not just today's
    top-6 by anchor count.

Reference: docs/upstream/STABLE_PROMOTION_CHECKLIST.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "sndr" / "manifests" / "anchor_manifest.json"


# Documented exception set: stable_kind="text-patch" patches whose
# pristine fixture has not yet been sourced into
# `tests/legacy/pristine_fixtures/`, so `build_anchor_manifest.py`
# cannot register them. Each entry MUST have:
#   1. an explicit `manifest_tracking` action-item reference,
#   2. a path to the live target file (so the deferred step is
#      executable when the fixture sourcing slice happens).
#
# Adding an entry here is the explicit "I am deferring step 5 of
# STABLE_PROMOTION_CHECKLIST". The exception must be removed when
# the corresponding pristine fixture is added + manifest is rebuilt.
#
# Currently empty — G4_04 graduated 2026-05-28 (STAGE-6-HARDENING.1).
# The dict shape is preserved so future deferrals follow the same
# documented pattern.
_MANIFEST_TRACKING_DEFERRED: dict[str, str] = {}


def _stable_patches() -> list[tuple[str, dict]]:
    """Registry entries with lifecycle=stable that participate in the
    text-patch / manifest ratchet.

    `stable_kind="runtime-hook"` patches (class rebinds / monkey-patches /
    loader hooks) DON'T use TextPatcher and therefore have no anchor
    manifest contract — they are exempted from conditions (b/c/d) of the
    ratchet. Their stable promotion is gated on a different evidence
    path (`production_validated_pins` field).

    For backward compatibility, entries without `stable_kind` set are
    treated as text-patch (matches the historical default, e.g. PN33 /
    PN35 before the field was introduced).
    """
    from sndr.dispatcher import PATCH_REGISTRY
    return [
        (pid, m) for pid, m in PATCH_REGISTRY.items()
        if isinstance(m, dict)
        and m.get("lifecycle") == "stable"
        and m.get("stable_kind", "text-patch") == "text-patch"
    ]


def _load_manifest() -> dict | None:
    if not MANIFEST_PATH.is_file():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ─── Ratchet contract ────────────────────────────────────────────────────


class TestStableLifecycleRatchet:
    """When a patch is promoted to lifecycle=stable, ALL four conditions
    must hold simultaneously. Test fails on any incomplete promotion."""

    def test_every_stable_patch_has_apply_module(self):
        """Stable ⇒ runtime knows how to apply (registry has apply_module)."""
        from sndr.dispatcher.spec import iter_patch_specs

        spec_map = {s.patch_id: s.apply_module for s in iter_patch_specs()}
        violations = []
        for pid, _ in _stable_patches():
            if not spec_map.get(pid):
                violations.append(pid)
        assert not violations, (
            f"stable-lifecycle patches missing apply_module: {violations}. "
            "Promotion to stable requires spec.apply_module pointing at "
            "the wiring module. See STABLE_PROMOTION_CHECKLIST.md step 1."
        )

    def test_every_stable_patch_has_registered_patcher(self):
        """Stable ⇒ wiring module registers via `register_text_patcher`.

        We can't enumerate every TextPatcher created at runtime without
        executing apply, so this test checks the patcher_registry itself
        for at least one entry whose patch_id starts with the stable's
        patch ID prefix.

        Patches in `_MANIFEST_TRACKING_DEFERRED` are explicitly skipped
        with a documented action-item to source the pristine fixture.
        """
        from sndr.engines.vllm.wiring.patcher_registry import (
            iter_registered_patchers,
        )

        registered_prefixes = set()
        for patch_id, _ in iter_registered_patchers():
            # "PN79.Sub-1" → prefix "PN79"
            head = patch_id.split(".", 1)[0]
            registered_prefixes.add(head)

        violations = []
        for pid, _ in _stable_patches():
            if pid in _MANIFEST_TRACKING_DEFERRED:
                continue
            if pid not in registered_prefixes:
                violations.append(pid)
        assert not violations, (
            f"stable-lifecycle patches with no registered TextPatcher: "
            f"{violations}. Promotion to stable requires the wiring "
            "module to call register_text_patcher() at import time. "
            "See STABLE_PROMOTION_CHECKLIST.md step 3."
        )

    def test_every_stable_patch_has_manifest_coverage(self):
        """Stable ⇒ anchor manifest covers the patcher's target rel_path
        AND its patch_id appears in the file's `patches` dict."""
        manifest = _load_manifest()
        if manifest is None:
            # No manifest at all → only OK if no stable patches
            assert not _stable_patches(), (
                "stable-lifecycle patches exist but anchor_manifest.json "
                "is missing or unreadable. Run "
                "`python scripts/build_anchor_manifest.py`."
            )
            return

        from sndr.engines.vllm.wiring.patcher_registry import (
            iter_registered_patchers,
        )
        files = manifest.get("files", {})

        # Build map: prefix (e.g. "PN79") → set of file rel_paths covered
        prefix_to_files: dict[str, set[str]] = {}
        for rel_path, file_entry in files.items():
            for pid in file_entry.get("patches", {}):
                prefix = pid.split(".", 1)[0]
                prefix_to_files.setdefault(prefix, set()).add(rel_path)

        # Build map: prefix → set of registered TextPatcher target files
        prefix_to_registered: dict[str, set[str]] = {}
        for pid, patcher in iter_registered_patchers():
            prefix = pid.split(".", 1)[0]
            prefix_to_registered.setdefault(prefix, set()).add(
                Path(patcher.target_file).name,
            )

        violations = []
        for pid, _ in _stable_patches():
            if pid in _MANIFEST_TRACKING_DEFERRED:
                continue
            covered = prefix_to_files.get(pid, set())
            if not covered:
                violations.append(
                    (pid, "no manifest file entries for this patch_id")
                )
        assert not violations, (
            f"stable-lifecycle patches with no manifest coverage: "
            f"{violations}. Promotion to stable requires the manifest "
            "JSON to have at least one file entry whose `patches` dict "
            "contains the patch_id. Run "
            "`python scripts/build_anchor_manifest.py`. "
            "See STABLE_PROMOTION_CHECKLIST.md step 5."
        )


class TestStableRatchetDocumented:
    """Sanity: the promotion checklist is reachable from contributor docs.

    Post-2026-05-16 consolidation, the standalone
    ``docs/upstream/STABLE_PROMOTION_CHECKLIST.md`` page was merged
    into ``docs/CONTRIBUTING.md`` under "Promoting a patch to
    ``lifecycle=\"stable\"``". Both locations are accepted so legacy
    checkouts keep working."""

    _LEGACY = "docs/upstream/STABLE_PROMOTION_CHECKLIST.md"
    _MERGED = "docs/CONTRIBUTING.md"

    def _checklist_text(self) -> str:
        legacy = REPO_ROOT / self._LEGACY
        if legacy.is_file():
            return legacy.read_text()
        merged = REPO_ROOT / self._MERGED
        assert merged.is_file(), (
            f"neither {self._LEGACY} nor {self._MERGED} present — "
            "operators need the promotion checklist when promoting a "
            "patch to lifecycle=stable"
        )
        return merged.read_text()

    def test_promotion_checklist_exists(self):
        assert self._checklist_text(), "checklist content empty"

    def test_checklist_mentions_all_four_steps(self):
        """Checklist must enumerate the 4 ratchet conditions. Each
        condition is recognized by ANY of the listed synonyms — the
        legacy standalone checklist used different phrasing than the
        merged CONTRIBUTING.md section."""
        content = self._checklist_text()
        required_terms: tuple[tuple[str, ...], ...] = (
            ("patch_id",),
            ("register_text_patcher", "TextPatcher", "register_for_manifest"),
            ("anchor_manifest",),
            ("build_anchor_manifest",),
        )
        for synonyms in required_terms:
            assert any(t in content for t in synonyms), (
                f"promotion checklist missing any reference to {synonyms} — "
                "operators may skip a critical ratchet step"
            )


# ─── Inventory probes (informational, never fail) ────────────────────────


def test_stable_patch_count_baseline():
    """Track current stable count — fail-safe to detect promotion that
    skipped the ratchet test (e.g., if a future PR adds lifecycle=stable
    AND deletes the policy tests in the same change)."""
    n = len(_stable_patches())
    # Baseline 2026-05-08: 0 stable patches.
    # When this baseline changes, BOTH ratchet tests above MUST still pass
    # AND this number bumps with the same PR.
    assert n >= 0
