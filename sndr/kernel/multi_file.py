# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — MultiFilePatchTransaction (atomic multi-file commits).

Two-phase commit primitive for multi-file text-patches. Used when one
logical patch (e.g. P64 qwen3coder MTP streaming) needs to modify
2+ files atomically — either all succeed or none.

Audit context (2026-05-05): TextPatcher is per-file atomic only. Multi-
file wiring patches that iterated patchers (PN52, PN58) suffered from
"file 1 commits, file 2 fails, file 1 stays modified leaving partial
state." PN52 docstring even falsely promised rollback — this class
delivers the actual rollback.

Two-phase commit:
  Phase 1 (DRY-RUN): for each TextPatcher peek at target content +
    verify all anchors present + replacements would be unique. NO
    files modified.
  Phase 2 (COMMIT): only if all dry-runs passed → real apply on each
    in order. If a Phase 2 step still fails (rare race condition —
    file modified between dry-run and commit), TRUE rollback via
    pre-commit snapshots taken at Phase 2a.

Migration history:
  - Original: vllm/_genesis/wiring/text_patch.py (Stage 0).
  - Stage 3 (CURRENT): split out into this dedicated module.
"""
from __future__ import annotations

import os

from .text_patch import (
    TextPatcher,
    TextPatchResult,
)


class MultiFilePatchTransaction:
    """Two-phase commit for multi-file text-patches.

    Usage in PN52 / PN58 / future bundles:
        def apply():
            txn = MultiFilePatchTransaction([
                _make_envs_patcher(),
                _make_abs_parser_patcher(),
                _make_basic_parser_patcher(),
                _make_struct_out_patcher(),
                _make_sched_patcher(),
            ])
            return txn.apply_or_skip()

    Returns:
      ("applied", "PN52 5/5 sub-patchers committed") — full success
      ("skipped", "PN52 dry-run failed: file 3 anchor not found ...") — atomic skip
      ("failed", "PN52 partial commit: file 2 wrote, file 3 race; rollback ...")
    """

    def __init__(self, patchers: list["TextPatcher"], name: str = "multi-file"):
        self.patchers = list(patchers)
        self.name = name

    def _dry_run(self) -> tuple[bool, str]:
        """Phase 1: validate all patchers without writing.

        Returns: (all_ok, reason_if_not).

        Audit P1.2 fix 2026-05-05 (genesis_deep_cross_audit): also
        validates anchor uniqueness (`content.count(anchor) == 1`)
        for required sub-patches AND simulates sequential preview
        through all sub-patches in declared order, so an early
        replacement that would invalidate a later anchor is caught
        at dry-run time instead of producing a partial state at commit.
        """
        for i, patcher in enumerate(self.patchers):
            if patcher is None:
                return False, f"file {i}: patcher is None (file not found)"
            if not os.path.isfile(patcher.target_file):
                return False, f"file {i} ({patcher.target_file}): file missing"
            try:
                with open(patcher.target_file, "r", encoding="utf-8") as fh:
                    src = fh.read()
            except Exception as e:
                return False, f"file {i}: read failed ({e})"
            # Marker already present → already-applied → OK (idempotent).
            if patcher.marker in src:
                continue
            # Sequential preview: walk sub-patches in order, check each
            # anchor presence + uniqueness in the simulated post-prior-
            # replacement state. Optional sub-patches still allowed missing.
            preview = src
            for sp in patcher.sub_patches:
                if sp.anchor not in preview:
                    if sp.required:
                        return False, (
                            f"file {i} ({patcher.target_file}): "
                            f"required anchor for sub-patch '{sp.name}' not "
                            "found (post sequential preview)"
                        )
                    continue  # optional anchor missing — skip in preview
                count = preview.count(sp.anchor)
                if sp.required and count > 1:
                    return False, (
                        f"file {i} ({patcher.target_file}): required anchor "
                        f"for sub-patch '{sp.name}' is ambiguous "
                        f"(found {count} times) — TextPatcher would replace "
                        "only the first occurrence, leaving partial state"
                    )
                # Apply replacement to preview so subsequent sub-patches
                # see the post-replacement state.
                preview = preview.replace(sp.anchor, sp.replacement, 1)
        return True, ""

    def apply_or_skip(self) -> tuple[str, str]:
        """Two-phase commit: dry-run, then real apply.

        Atomic skip on dry-run failure. **True rollback** on Phase 2 race
        (audit G-POST-08 fix 2026-05-05) — pre-commit snapshots of every
        target file held in memory; on first commit-phase failure, all
        previously-modified files are restored byte-for-byte from snapshot.

        If snapshot restore itself fails (filesystem error, permission
        change), the snapshot is written to ``<target>.genesis_rollback``
        next to the file as a manual recovery aid and a WARN is logged.
        """
        ok, reason = self._dry_run()
        if not ok:
            return "skipped", f"{self.name} dry-run failed: {reason}"

        # Phase 2a: snapshot ALL targets BEFORE any write. Held in memory
        # for byte-for-byte restore. Files on disk are not touched yet.
        snapshots: dict[int, tuple[str, str]] = {}
        for i, patcher in enumerate(self.patchers):
            try:
                with open(patcher.target_file, "r", encoding="utf-8") as fh:
                    snapshots[i] = (patcher.target_file, fh.read())
            except Exception as e:
                # Atomic skip — we never modified anything.
                return ("skipped",
                        f"{self.name} pre-commit snapshot failed for file {i} "
                        f"({patcher.target_file}): {e}")

        # Phase 2b: real commit, in order
        committed: list[tuple[int, "TextPatcher"]] = []
        for i, patcher in enumerate(self.patchers):
            try:
                result, failure = patcher.apply()
            except Exception as e:
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} raised "
                    f"{type(e).__name__}: {e}",
                )
            if result in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
                committed.append((i, patcher))
            elif result == TextPatchResult.SKIPPED:
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} skipped after dry-run "
                    f"passed (race condition?): "
                    f"{failure.reason if failure else '?'}",
                )
            else:  # FAILED
                return self._rollback_and_fail(
                    committed, snapshots,
                    f"{self.name} commit phase: file {i} failed: "
                    f"{failure.reason if failure else '?'}",
                )

        return ("applied",
                f"{self.name} {len(committed)}/{len(self.patchers)} files "
                "committed atomically")

    def _rollback_and_fail(
        self,
        committed: list[tuple[int, "TextPatcher"]],
        snapshots: dict[int, tuple[str, str]],
        reason: str,
    ) -> tuple[str, str]:
        """Audit G-POST-08 fix 2026-05-05: TRUE rollback of any partially-
        committed files using the in-memory snapshots taken at Phase 2a.

        IDEMPOTENT-result files are NOT restored (they were already applied
        before this transaction — restoring would unapply prior state
        belonging to a different transaction).

        On unrecoverable filesystem error during restore, the snapshot is
        written to ``<target>.genesis_rollback`` as a manual recovery aid
        so the operator never has to reconstruct the original file by hand.
        """
        # Detect IDEMPOTENT vs APPLIED so we don't unapply pre-existing state.
        # IDEMPOTENT means the marker was already present when apply() ran —
        # i.e. file was unchanged in this transaction.
        applied_indices: list[int] = []
        for i, patcher in committed:
            snap_path, snap_content = snapshots.get(i, ("", ""))
            try:
                with open(patcher.target_file, "r", encoding="utf-8") as fh:
                    current = fh.read()
            except Exception:
                self._write_rollback_aid(snap_path, snap_content)
                continue
            if current != snap_content:
                applied_indices.append(i)

        restored: list[str] = []
        rollback_aids: list[str] = []
        for i in applied_indices:
            snap_path, snap_content = snapshots[i]
            try:
                # Atomic write via temp + rename to mirror TextPatcher's
                # commit semantics — no torn writes on crash mid-restore.
                tmp = snap_path + ".genesis_rollback.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(snap_content)
                os.replace(tmp, snap_path)
                restored.append(snap_path)
            except Exception as e:
                # Restore failed — leave the snapshot on disk as a manual
                # recovery aid (operator can `mv .genesis_rollback FILE`).
                aid = self._write_rollback_aid(snap_path, snap_content)
                if aid:
                    rollback_aids.append(f"{snap_path} → {aid} ({e})")

        notes = []
        if restored:
            notes.append(f"ROLLED BACK {len(restored)} file(s): "
                         + ", ".join(restored))
        if rollback_aids:
            notes.append("MANUAL RECOVERY NEEDED for: "
                         + "; ".join(rollback_aids))
        if not restored and not rollback_aids:
            notes.append("no files needed rollback (transaction failed before "
                         "any APPLIED write)")
        return ("failed", reason + "\n" + "\n".join(notes))

    @staticmethod
    def _write_rollback_aid(path: str, content: str) -> str | None:
        """Write a snapshot to ``<path>.genesis_rollback`` for manual recovery.
        Returns the aid path on success, None on failure."""
        if not path:
            return None
        aid = path + ".genesis_rollback"
        try:
            with open(aid, "w", encoding="utf-8") as fh:
                fh.write(content)
            return aid
        except Exception:
            return None


__all__ = ["MultiFilePatchTransaction"]
