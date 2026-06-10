# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — TextPatcher (canonical home for per-file text patching).

Per-file text-patching primitive. Used by every SNDR Core / Genesis
patch wiring to apply anchor→replacement edits to one target file
in the installed vllm tree.

Why text-patches (vs pure monkey-patch)
----------------------------------------
Some upstream code sites are not cleanly monkey-patchable:

  - Raises inside a method body (e.g. `arg_utils.py` `if model_config.is_hybrid:
    raise NotImplementedError(...)`) — bypassing requires re-defining
    ~50 lines of upstream logic that drifts between vLLM versions.
  - Compile-time Triton kernel literals (e.g. `BLOCK_KV=4, num_warps=1`
    as immediate args to `@triton.jit`) — baked into kernel compilation,
    cannot be overridden at call site.
  - Control-flow that depends on local variable state only available
    inside the original function (no clean rebind point).

For those, surgical text-replacement at plugin-register time is
pragmatic: small, targeted, verifiable, far less invasive than full
method rewrite.

Migration history:
  - Original: vllm/_genesis/wiring/text_patch.py (833 LOC monolith).
  - Stage 3 (CURRENT): split into core/{text_patch.py, multi_file.py,
    manifest_cache.py}. The legacy `wiring/text_patch.py` becomes a
    thin re-export shim for back-compat.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .manifest import (
    cached_load_manifest,
    derive_rel_path_from_target,
    md5_bytes,
)

log = logging.getLogger("genesis.wiring.text_patch")


def _is_path_writable(path: str) -> bool:
    """Reliable writability check for text-patch preflight.

    v11.3.0 FATAL BUG FIX. Replaces `os.access(path, os.W_OK)` which
    has a Python-stdlib documented quirk: it ignores CAP_DAC_OVERRIDE
    (root's blanket file-permission override). When the container runs
    as root and upstream vllm files are owned by uid != 0 (typical:
    pip install runs as uid 1000 under nightly-image build), the
    legacy os.access returned False — the patcher then SKIPPED with
    "read_only_mount" despite the kernel being perfectly happy to
    write. Caused ~50% of our text-patches (P67, PN12, PN95, etc.) to
    silently no-op in production for ~weeks.

    The fix: actually open the file for read+write (no actual write
    happens — open(r+b) just verifies the OS would permit it). This
    matches what subsequent write would experience.

    Returns True if the file can be opened with r+b mode (writable),
    False otherwise (true read-only filesystem or permission denied).
    """
    try:
        f = open(path, "r+b")
        f.close()
        return True
    except (OSError, PermissionError):
        return False


class TextPatchResult(Enum):
    APPLIED = "applied"        # File modified in this call.
    IDEMPOTENT = "idempotent"  # Marker already present, nothing to do.
    SKIPPED = "skipped"        # Anchor drift / not applicable — not an error.
    FAILED = "failed"          # Unexpected condition, patch safety violated.


@dataclass(frozen=True)
class TextPatchFailure:
    """Why a patch ended in non-APPLIED/IDEMPOTENT state."""
    reason: str
    detail: str = ""


@dataclass
class TextPatch:
    """A single anchor→replacement edit.

    Attributes:
      name: Short identifier for logs.
      anchor: Exact substring that must appear in the file (pre-patch).
      replacement: What to substitute for `anchor`.
      required: If True, failure of this sub-patch aborts the parent group.
                If False (default), sibling sub-patches still run.

    Stage 8 (per-sub drift, 2026-05-07):
      upstream_merged_markers: Optional list of strings that, if present
        anywhere in the target file, indicate that THIS sub-patch's fix
        was merged upstream. The sub will silently skip while sibling
        subs in the same TextPatcher continue. Differs from
        TextPatcher.upstream_drift_markers (patcher-level — whole patch
        skipped). Use per-sub when a multi-anchor patch has SOME of its
        anchors absorbed by upstream while OTHERS remain valid (common
        when upstream cherry-picks part of a multi-anchor backport).
      on_upstream_merge: Behavior when an upstream marker matches:
        "skip_silently" (default) — no-op this sub, log INFO, siblings continue.
        "warn"                    — same as skip_silently but log WARNING.
        "abort_bundle"            — entire TextPatcher returns SKIPPED.
                                    Inside MultiFilePatchTransaction
                                    this triggers transaction rollback.
    """
    name: str
    anchor: str
    replacement: str
    required: bool = False
    # ── Stage 8 (per-sub drift) ────────────────────────────────────────
    upstream_merged_markers: list[str] = field(default_factory=list)
    on_upstream_merge: str = "skip_silently"
        # Use Literal["skip_silently","warn","abort_bundle"] when py3.12+
        # baseline is established (currently py3.10 minimum).


@dataclass
class TextPatcher:
    """Apply a sequence of anchor→replacement edits to one target file.

    Attributes:
      patch_name: Stable human-readable identifier (e.g. "P4 TQ hybrid").
      target_file: Absolute path to file to patch.
      marker: Unique string that, once present in the file, indicates this
              patch has been applied. Used for idempotency.
      sub_patches: Ordered list of TextPatch edits. Applied in sequence.
      upstream_drift_markers: If any of these strings are present in the
              file, consider this patch OBSOLETE (upstream merged a fix).
              Returns SKIPPED with a clear message.
      patch_id: Optional machine-readable patch identifier matching the
              key used in the Site Map anchor manifest (e.g. "PN79.Sub-1").
              When non-None AND the manifest is available AND the file's
              md5 matches the manifest's pristine record, apply() bypasses
              the O(N×M) Layer 5 anchor scan and uses the manifest's
              pre-computed byte offsets for O(1) lookup. None (default)
              keeps the patcher on the legacy path.
    """
    patch_name: str
    target_file: str
    marker: str
    sub_patches: list[TextPatch]
    upstream_drift_markers: list[str] = field(default_factory=list)
    patch_id: Optional[str] = None
    # Populated by apply() with the names of sub-patches whose anchor matched
    # and whose replacement was actually written. Empty until apply() runs.
    # Callers building detail strings should read this list rather than
    # hardcoding the union of sub_patches (which produces misleading reports
    # when sub-patches soft-skip with required=False).
    applied_sub_patches: list[str] = field(default_factory=list)

    def _try_apply_via_manifest(
        self, content: str
    ) -> Optional[tuple[str, list[str]]]:
        """Phase 3 fast-path (P2.1, 2026-05-07): bypass Layer 5 anchor scan
        when an anchor offset manifest is available and pristine matches.

        Returns:
          (modified_content, applied_sub_patch_names) — manifest path
              succeeded, all required anchors splice-applied. Caller skips
              Layer 5 and proceeds directly to Layer 6 (marker prepend).
          None — manifest abstained (any of 7 gates failed). Caller falls
              back to Layer 5 legacy O(N×M) anchor scan.

        7 abstain gates (in order):
          G1: GENESIS_NO_PATCH_CACHE / SNDR_NO_PATCH_CACHE env set
          G2: self.patch_id is None (not opt-in to manifest)
          G3: Manifest unavailable (absent/corrupt/pin mismatch)
          G4: target_file rel_path not under vllm tree
          G4b: rel_path missing from manifest.files
          G5: pristine md5 mismatch (file modified vs manifest baseline)
          G6: self.patch_id missing from file's patches dict
          G7: any required sub_patch's anchor missing OR anchor_md5 mismatch

        On success: splices applied in REVERSE byte_offset order so earlier
        offsets stay valid throughout sequential mutations.
        """
        # G1: env opt-out
        try:
            from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
            if genesis_no_patch_cache():
                return None
        except Exception:
            return None

        # G2: must have machine-readable patch_id
        if not self.patch_id:
            return None

        # G3: manifest available?
        manifest = cached_load_manifest()
        if manifest is None:
            return None

        # G4: derive rel_path
        rel_path = derive_rel_path_from_target(self.target_file)
        if rel_path is None:
            return None

        # G4b: file in manifest?
        files = manifest.get("files", {})
        file_entry = files.get(rel_path)
        if file_entry is None:
            return None

        # G5: pristine md5 match?
        content_bytes = content.encode("utf-8")
        actual_md5 = md5_bytes(content_bytes)
        expected_md5 = file_entry.get("md5_pristine")
        if actual_md5 != expected_md5:
            log.info(
                "[%s] manifest pristine md5 mismatch (%s vs %s) — "
                "fall back to legacy",
                self.patch_name, actual_md5[:8], (expected_md5 or "?")[:8],
            )
            return None

        # G6: patch_id covered?
        patch_entry = file_entry.get("patches", {}).get(self.patch_id)
        if patch_entry is None:
            return None

        # G7: every required sub_patch covered + anchor_md5 sanity check
        cached_anchors = patch_entry.get("anchors", {})
        splices: list[dict] = []
        applied_names: list[str] = []
        for sp in self.sub_patches:
            a = cached_anchors.get(sp.name)
            if a is None:
                if sp.required:
                    return None
                continue  # optional — same as legacy "soft skip"
            offset = a.get("byte_offset")
            length = a.get("byte_length")
            anchor_md5 = a.get("anchor_md5")
            if (not isinstance(offset, int) or not isinstance(length, int)
                    or not isinstance(anchor_md5, str)):
                return None
            if offset < 0 or offset + length > len(content_bytes):
                return None
            slice_md5 = md5_bytes(content_bytes[offset:offset + length])
            if slice_md5 != anchor_md5:
                log.info(
                    "[%s/%s] manifest anchor_md5 mismatch — anchor moved? "
                    "fall back to legacy",
                    self.patch_name, sp.name,
                )
                return None
            splices.append({
                "byte_offset": offset,
                "byte_length": length,
                "replacement": sp.replacement,
                "name": sp.name,
            })
            applied_names.append(sp.name)

        if not splices:
            # Either no sub_patches OR all optional+missing — let caller
            # fall through to Layer 5 which has the SKIPPED reason logic.
            return None

        # All gates passed. Splice in REVERSE offset order so earlier
        # offsets stay valid for ALL splices.
        splices.sort(key=lambda s: -s["byte_offset"])
        for s in splices:
            content_bytes = (
                content_bytes[:s["byte_offset"]]
                + s["replacement"].encode("utf-8")
                + content_bytes[s["byte_offset"] + s["byte_length"]:]
            )

        log.info(
            "[%s] manifest fast-path applied %d sub-patches: %s",
            self.patch_name, len(applied_names), ", ".join(applied_names),
        )
        return content_bytes.decode("utf-8"), applied_names

    def apply(self) -> tuple[TextPatchResult, Optional[TextPatchFailure]]:
        """Execute the patch. Returns (result, failure_info_if_not_ok).

        NEVER raises — returns SKIPPED/FAILED on any issue.

        8-layer apply path:
          Layer 0: file_cache fast-path (P2.2 — mtime+size+marker)
          Layer 1: file existence + readability
          Layer 2: marker idempotency check
          Layer 3: upstream drift markers (patch obsolete?)
          Layer 4: writability preflight (T1.5 — read-only mount guard)
          Layer 4.5: manifest fast-path (P2.1 — byte-offset splices)
          Layer 5: legacy O(N×M) anchor scan + replace
          Layer 6: marker prepend
          Layer 7: write + verify (re-read marker presence)
        """
        # Layer 0: persistent file cache fast-path (P2.2, 2026-05-07).
        # Skip full Layer 1+2+3 for already-patched files on warm restart.
        # Single os.stat() check (~10μs) replaces ~160μs of disk read +
        # marker scan when (mtime_ns, size_bytes, marker) match cache.
        try:
            from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
            from sndr.engines.vllm.wiring.file_cache import (
                is_marker_cached_present,
            )
            if not genesis_no_patch_cache():
                if is_marker_cached_present(self.target_file, self.marker):
                    log.debug(
                        "[%s] Layer 0 file_cache HIT — IDEMPOTENT without I/O",
                        self.patch_name,
                    )
                    return TextPatchResult.IDEMPOTENT, None
        except Exception as e:
            # Cache layer must NEVER fail apply().
            log.debug(
                "[%s] Layer 0 file_cache exception: %s — falling through",
                self.patch_name, e,
            )

        # Layer 1: file must exist and be readable.
        if not os.path.isfile(self.target_file):
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="target_file_missing",
                detail=f"{self.target_file} not found",
            )

        try:
            with open(self.target_file) as f:
                content = f.read()
        except (OSError, PermissionError) as e:
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="read_error", detail=str(e),
            )

        # Layer 2: idempotency — already applied?
        if self.marker in content:
            log.debug(
                "[%s] marker %r already present — idempotent skip",
                self.patch_name, self.marker,
            )
            # P2.2: warm Layer 0 cache so next boot's Layer 0 hits.
            try:
                from sndr.engines.vllm.wiring.file_cache import record_apply_result
                record_apply_result(
                    self.target_file, self.marker,
                    post_apply_content=content,
                )
            except Exception:
                pass  # cache write must never fail apply()
            return TextPatchResult.IDEMPOTENT, None

        # Layer 3: upstream merged?
        for m in self.upstream_drift_markers:
            if m in content:
                log.info(
                    "[%s] upstream marker %r detected — patch obsolete, skip",
                    self.patch_name, m,
                )
                return TextPatchResult.SKIPPED, TextPatchFailure(
                    reason="upstream_merged",
                    detail=f"marker {m!r} present",
                )

        # Layer 4 (T1.5 / audit §17.4): writability preflight.
        # Catches club-3090 #47 — operator bind-mounts SNDR Core tree
        # read-only into a container, text-patcher silently no-ops on
        # write. Detect it BEFORE the splice work so the failure mode is
        # a structured SKIPPED with a clear remediation hint, not a
        # late "write_error: permission denied" after Layer 5 burned
        # cycles computing replacements that can never land.
        #
        # Marker check has already passed (Layer 2), so we know the file
        # is unpatched — meaning a write WILL be attempted. os.access
        # returns False both for true read-only mounts AND for the file
        # being missing-but-parent-readable; Layer 1 already ruled out
        # missing files, so a False here is unambiguous.
        # v11.3.0 FATAL BUG FIX: `os.access(W_OK)` does NOT honor
        # CAP_DAC_OVERRIDE (root's blanket file-permission override).
        # When the container runs as root and the upstream vllm files
        # are owned by uid != 0 (typical: pip install runs as uid 1000),
        # os.access(W_OK) returns False even though the kernel would
        # successfully write. This caused ~50% of our text-patches to
        # silently skip with "read_only_mount" in production despite
        # the filesystem being fully writable.
        # See: https://docs.python.org/3/library/os.html#os.access
        #   "Some operating systems may set this flag on dirs that are
        #    not writable, leading to false negatives." Python docs
        #   recommend an actual open(rw) probe instead.
        if not _is_path_writable(self.target_file):
            log.warning(
                "[%s] target %s is not writable — patch cannot apply. "
                "Common cause: read-only bind mount in container. "
                "Workaround: use overlay mount or rebind read-write.",
                self.patch_name, self.target_file,
            )
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="read_only_mount",
                detail=(
                    f"{self.target_file} not writable. If running in a "
                    "container, the SNDR Core tree was bind-mounted "
                    "read-only. Workaround: use overlay mount, or rebind "
                    "the path read-write. See "
                    "docs/INSTALL.md#read-only-mount-overlay for the "
                    "recommended pattern."
                ),
            )

        # Layer 4.5 (P2.1 Phase 3): manifest fast-path — replaces Layer 5's
        # O(N×M) scan with O(1) lookup + O(64) anchor_md5 sanity check.
        # Falls through to Layer 5 if manifest abstains.
        manifest_result = self._try_apply_via_manifest(content)
        if manifest_result is not None:
            modified, applied_patches = manifest_result
        else:
            # Layer 5: validate ALL anchors before applying ANY. Legacy
            # O(N×M) full-scan path — ground truth.
            layer5_result = self._apply_layer5_legacy(content)
            first = layer5_result[0]
            if isinstance(first, TextPatchResult):
                return layer5_result  # type: ignore[return-value]
            modified, applied_patches = layer5_result

        # Layer 6: prepend marker comment so future runs see IDEMPOTENT.
        # NOTE: keeps "Genesis wiring marker" prefix for back-compat with
        # files patched by pre-v8 versions (Q2 mixed-branding decision).
        marker_line = f"# [Genesis wiring marker: {self.marker}]\n"
        if not modified.startswith(marker_line):
            modified = marker_line + modified

        # Layer 7: write + verify.
        try:
            with open(self.target_file, "w") as f:
                f.write(modified)
        except (OSError, PermissionError) as e:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="write_error", detail=str(e),
            )

        try:
            with open(self.target_file) as f:
                reread = f.read()
        except (OSError, PermissionError) as e:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="reread_error", detail=str(e),
            )

        if self.marker not in reread:
            return TextPatchResult.FAILED, TextPatchFailure(
                reason="marker_not_persisted",
                detail="file write succeeded but marker absent on re-read",
            )

        # P2.2: record apply result in persistent cache so next boot's
        # Layer 0 can short-circuit Layer 1+2 entirely.
        try:
            from sndr.engines.vllm.wiring.file_cache import record_apply_result
            record_apply_result(
                self.target_file, self.marker, post_apply_content=reread,
            )
        except Exception:
            pass  # cache write must never fail apply()

        self.applied_sub_patches = list(applied_patches)
        log.info(
            "[%s] applied %d sub-patches: %s",
            self.patch_name, len(applied_patches), ", ".join(applied_patches),
        )
        return TextPatchResult.APPLIED, None

    def _apply_layer5_legacy(self, content: str):
        """Layer 5 legacy path — O(N×M) anchor scan + sequential replace.

        Returns one of:
          (TextPatchResult.SKIPPED, TextPatchFailure(...))  — when:
            - required anchor missing
            - ambiguous anchor (count > 1)
            - all sub_patches missed
          (modified_str, applied_names_list)  — on success.

        Caller (apply()) discriminates via isinstance check on first
        element — list[str] = success, TextPatchResult enum = skip/fail.
        """
        modified = content
        applied_patches: list[str] = []

        for sp in self.sub_patches:
            # ── Stage 8 (per-sub drift, 2026-05-07): check this sub's
            # individual upstream merge markers BEFORE the anchor scan.
            # Differs from Layer 3 (patcher-level `upstream_drift_markers`)
            # which kills the whole patch — here ONLY this sub no-ops,
            # siblings continue. Used when upstream cherry-picks part of
            # a multi-anchor backport.
            sub_drift_match = next(
                (um for um in sp.upstream_merged_markers if um in modified),
                None,
            )
            if sub_drift_match is not None:
                if sp.on_upstream_merge == "abort_bundle":
                    return TextPatchResult.SKIPPED, TextPatchFailure(
                        reason="sub_upstream_merged_abort_bundle",
                        detail=(
                            f"sub-patch {sp.name!r}: upstream-merge marker "
                            f"{sub_drift_match!r} fired with "
                            f"on_upstream_merge=abort_bundle — patcher aborts"
                        ),
                    )
                log_fn = log.warning if sp.on_upstream_merge == "warn" else log.info
                log_fn(
                    "[%s/%s] upstream-merged (marker %r) — sibling "
                    "sub-patches continue",
                    self.patch_name, sp.name, sub_drift_match,
                )
                continue  # sibling subs continue with current `modified`

            if sp.anchor not in modified:
                if sp.required:
                    return TextPatchResult.SKIPPED, TextPatchFailure(
                        reason="required_anchor_missing",
                        detail=f"sub-patch {sp.name!r}: anchor not found in file",
                    )
                log.info(
                    "[%s/%s] anchor not found — soft skip (sibling patches continue)",
                    self.patch_name, sp.name,
                )
                continue

            if modified.count(sp.anchor) != 1:
                return TextPatchResult.SKIPPED, TextPatchFailure(
                    reason="ambiguous_anchor",
                    detail=(
                        f"sub-patch {sp.name!r}: anchor appears "
                        f"{modified.count(sp.anchor)} times (expected 1)"
                    ),
                )

            modified = modified.replace(sp.anchor, sp.replacement, 1)
            applied_patches.append(sp.name)

        if not applied_patches:
            return TextPatchResult.SKIPPED, TextPatchFailure(
                reason="no_applicable_sub_patches",
                detail="every sub-patch anchor absent — file may be post-upstream-fix",
            )

        return modified, applied_patches


# ─────────────────────────────────────────────────────────────────────────
# Shared wiring-result mapper.
#
# Most wiring `apply()` functions follow the same skeleton:
#   1. Check dispatcher should_apply
#   2. Check vllm_install_root + resolve target
#   3. Run patcher.apply()
#   4. Translate (TextPatchResult, TextPatchFailure | None) into the
#      wiring contract: ("applied" | "skipped" | "failed", reason: str)
#
# Step 4 was duplicated across ~25 wiring modules with subtle drift —
# some forgot to handle SKIPPED / IDEMPOTENT and silently reported
# "applied" when the file was actually unchanged (caught by PN14 TDD
# 2026-04-29). This helper centralizes the mapping.
# ─────────────────────────────────────────────────────────────────────────


def result_to_wiring_status(
    result: TextPatchResult,
    failure: TextPatchFailure | None,
    *,
    applied_message: str,
    patch_name: str,
) -> tuple[str, str]:
    """Translate a (TextPatchResult, TextPatchFailure) pair into the
    wiring `apply()` return contract: (status, reason).

    Returns:
      (status, reason) where status ∈ {"applied", "skipped", "failed"}.
        APPLIED → ("applied", applied_message)
        IDEMPOTENT → ("skipped", f"{patch_name}: already applied (marker present)")
        SKIPPED → ("skipped", f"{patch_name}: <reason> — <detail>")
        FAILED → ("failed", f"{patch_name}: <reason> (<detail>)")
    """
    if result == TextPatchResult.APPLIED:
        return "applied", applied_message
    if result == TextPatchResult.IDEMPOTENT:
        return "skipped", f"{patch_name}: already applied (marker present)"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown_skip"
        detail = failure.detail if failure and failure.detail else None
        msg = f"{patch_name}: {reason}"
        if detail:
            msg += f" — {detail}"
        return "skipped", msg
    # TextPatchResult.FAILED
    reason = failure.reason if failure else "unknown"
    detail = failure.detail if failure and failure.detail else ""
    return "failed", f"{patch_name}: {reason} ({detail})"


def marker_present_in_target(patcher: "TextPatcher") -> bool:
    """True iff the patcher's idempotency ``marker`` is already present in its
    ``target_file`` — i.e. the patch has been applied.

    Best-effort idempotency probe for ``is_applied()`` hooks: returns False if
    the marker is empty or the target file is missing/unreadable, so callers
    never raise from a probe.
    """
    marker = getattr(patcher, "marker", "") or ""
    target = getattr(patcher, "target_file", None)
    if not marker or not target:
        return False
    try:
        with open(target, "r", encoding="utf-8", errors="ignore") as fh:
            return marker in fh.read()
    except OSError:
        return False


# PR38 cleanup (2026-05-08): tests historically imported a few helpers
# from the legacy `vllm/_genesis/wiring/text_patch.py` monolith that
# Stage 3 split into siblings (`multi_file.py`, `manifest_cache.py`).
# Lazy `__getattr__` re-exports them on demand so
# `from sndr.kernel.text_patch import MultiFilePatchTransaction`
# keeps working without creating a circular import (multi_file.py
# imports TextPatcher from this module at load time).
_LAZY_REEXPORTS = {
    "MultiFilePatchTransaction": ("sndr.kernel.multi_file", "MultiFilePatchTransaction"),
    "_reset_manifest_cache_for_tests": ("sndr.kernel.manifest", "_reset_manifest_cache_for_tests"),
    "cached_load_manifest": ("sndr.kernel.manifest", "cached_load_manifest"),
    "derive_rel_path_from_target": ("sndr.kernel.manifest", "derive_rel_path_from_target"),
    # Legacy private-name aliases (single underscore prefix in pre-Stage-3
    # text_patch.py monolith). Map to the public Stage-3-renamed names.
    "_cached_load_manifest": ("sndr.kernel.manifest", "cached_load_manifest"),
    "_derive_rel_path_from_target": ("sndr.kernel.manifest", "derive_rel_path_from_target"),
}


def __getattr__(name):
    target = _LAZY_REEXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'sndr.kernel.text_patch' has no attribute {name!r}"
        )
    import importlib
    mod = importlib.import_module(target[0])
    return getattr(mod, target[1])


# Last 4 names resolve via the module-level `__getattr__` lazy proxy
# above (see `_LAZY_REEXPORTS`). They are real runtime attributes but
# have no top-level assignment, so ruff F822 needs to be silenced
# per-line. The lazy proxy is required to avoid a circular import
# with `core.multi_file` / `core.manifest_cache`.
__all__ = [
    "TextPatch",
    "TextPatchFailure",
    "TextPatchResult",
    "TextPatcher",
    "result_to_wiring_status",
    "MultiFilePatchTransaction",          # noqa: F822 — lazy via __getattr__
    "_reset_manifest_cache_for_tests",    # noqa: F822 — lazy via __getattr__
    "cached_load_manifest",               # noqa: F822 — lazy via __getattr__
    "derive_rel_path_from_target",        # noqa: F822 — lazy via __getattr__
]
