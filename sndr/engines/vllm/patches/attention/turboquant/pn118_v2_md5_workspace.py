# SPDX-License-Identifier: Apache-2.0
"""PN118 v2 — md5+full-file PoC of the PN119 reference pattern (workspace.py scope).

Companion to ``pn118_tq_workspace_fallback.py`` (the anchor-based
original). The original PN118 patches **two** files via 4 text
anchors:

  - ``v1/worker/workspace.py``                  (2 anchors)
  - ``v1/attention/backends/turboquant_attn.py`` (2 anchors)

This v2 PoC validates the PN119 single-file md5 + full-file
replacement pattern against **only** the workspace.py target. The
original PN118 retains coverage of turboquant_attn.py via its
anchors. PN118 self-detects v2's Genesis marker on workspace.py and
skips re-anchoring there once v2 has run — the two patches compose,
they do not conflict.

================================================================
SCOPE CORRECTION FROM v11.1.0 SPEC
================================================================

The v11.1.0 enterprise-design spec assumed pn118 patches a single
file at ``v1/attention/ops/workspace.py``. Reality, surfaced during
Track B.1 scout: pn118 patches ``v1/worker/workspace.py`` AND
``v1/attention/backends/turboquant_attn.py`` (4 anchors total, not
8 as earlier prose suggested). This v2 PoC is scoped to workspace.py
only — multi-file md5 replacement is a separate v11.2.0+ refactor
(not yet validated against pn79's 35-anchor / 3-file case either).

================================================================
PATTERN (matches PN119)
================================================================

    PN118_V2_MD5_WORKSPACE_PRE_PATCH_MD5 = "<32-char hex>"
    PN118_V2_MD5_WORKSPACE_POST_PATCH_CONTENT = '''<full file post-patch>'''

    def apply():
        target = resolve_vllm_file("v1/worker/workspace.py")
        if _GENESIS_PN118_V2_WORKSPACE_MARKER in target.read_text():
            return _skipped("already applied")
        if _file_md5(target) != PN118_V2_MD5_WORKSPACE_PRE_PATCH_MD5:
            return _skipped("md5 mismatch — upstream drifted")
        target.write_text(PN118_V2_MD5_WORKSPACE_POST_PATCH_CONTENT)
        return _applied()

================================================================
STATUS
================================================================

Default OFF; opt-in via ``GENESIS_ENABLE_PN118_V2_MD5_WORKSPACE=1``.
Composes with the original PN118 (not a replacement) — both can be
enabled simultaneously without conflict, because:

  - v2 owns workspace.py exclusively (writes Genesis marker)
  - PN118 detects the marker on workspace.py and skips its 2 anchors
    on that file, but still applies its other 2 anchors on
    turboquant_attn.py

v11.1.0 Phase 6 P3.1 closeout PoC.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn118_v2_md5_workspace")


_GENESIS_PN118_V2_WORKSPACE_MARKER = (
    "# Genesis PN118 v2 (md5+full-file PoC) marker — DO NOT REMOVE"
)


# Md5 hash of the upstream ``v1/worker/workspace.py`` at PROD pin
# 0.20.2rc1.dev338+gbf0d2dc6d. If a future pin bump changes the file
# this constant must be regenerated alongside the bundled post-patch
# fixture. On drift the v2 path self-skips via md5 mismatch and the
# original PN118 (anchor-based) continues to cover the same target.
PN118_V2_MD5_WORKSPACE_PRE_PATCH_MD5 = "439f0c086cc50f467960b6e610bdf803"


# Bundled fixtures (referenced by tests + by apply path).
# v12.x moved this module deeper; walk up to the repo root that holds
# tests/unit/integrations instead of a fixed parents[N].
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "tests" / "unit" / "integrations").is_dir()),
    Path(__file__).resolve().parents[5],
)
_TESTS_FIXTURE_DIR = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "integrations"
    / "attention"
    / "turboquant"
    / "fixtures"
)
_POST_PATCH_FIXTURE = (
    _TESTS_FIXTURE_DIR / "pn118_v2_md5_workspace_post_patch.py.txt"
)
_PRE_PATCH_FIXTURE = (
    _TESTS_FIXTURE_DIR / "pn118_v2_md5_workspace_pre_patch.py.txt"
)


def _load_post_patch_content() -> str:
    """Load the post-patch file content. Bundled as test fixture so the
    patch module itself stays small; the fixture is read at module
    import. Marker is appended if not already present (defense in
    depth — the rig-generated fixture already contains it)."""
    if not _POST_PATCH_FIXTURE.is_file():
        return ""
    content = _POST_PATCH_FIXTURE.read_text()
    if _GENESIS_PN118_V2_WORKSPACE_MARKER not in content:
        sep = "" if content.endswith("\n") else "\n"
        content = content + sep + _GENESIS_PN118_V2_WORKSPACE_MARKER + "\n"
    return content


PN118_V2_MD5_WORKSPACE_POST_PATCH_CONTENT = _load_post_patch_content()


@dataclass(frozen=True)
class _Result:
    status: str  # "applied" | "skipped" | "failed"
    reason: str = ""


def _applied(reason: str = "") -> _Result:
    return _Result(
        status="applied",
        reason=reason or "PN118 v2 md5+full-file replacement applied",
    )


def _skipped(reason: str) -> _Result:
    return _Result(status="skipped", reason=reason)


def _failed(reason: str) -> _Result:
    return _Result(status="failed", reason=reason)


def _file_md5(path: Path) -> str:
    """Compute md5 of a file's bytes. Matches stdlib hashlib.md5."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _do_apply(target: Path) -> _Result:
    """Core apply logic — separated from apply() for testability.

    Order of checks:
      1. Target exists (skip if not)
      2. Marker present (skip — idempotent re-entry)
      3. md5 matches PRE_PATCH_MD5 (skip on mismatch — upstream drift)
      4. Post-patch content available (skip if fixture missing)
      5. Write post-patch content to target → applied
    """
    if not target.is_file():
        return _skipped(f"target not found: {target}")
    try:
        current_text = target.read_text()
    except OSError as e:
        return _failed(f"read failed: {e}")
    if _GENESIS_PN118_V2_WORKSPACE_MARKER in current_text:
        return _skipped("already applied (marker present)")
    current_md5 = _file_md5(target)
    if current_md5 != PN118_V2_MD5_WORKSPACE_PRE_PATCH_MD5:
        return _skipped(
            f"md5 mismatch (got {current_md5}, expected "
            f"{PN118_V2_MD5_WORKSPACE_PRE_PATCH_MD5}) — upstream drifted "
            "from PoC baseline. The original PN118 (anchor-based) will "
            "continue to attempt patching this file on its own."
        )
    if not PN118_V2_MD5_WORKSPACE_POST_PATCH_CONTENT:
        return _skipped(
            "post-patch fixture missing — regenerate from rig "
            "(tests/unit/integrations/attention/turboquant/fixtures/"
            "pn118_v2_md5_workspace_post_patch.py.txt)"
        )
    try:
        target.write_text(PN118_V2_MD5_WORKSPACE_POST_PATCH_CONTENT)
    except OSError as e:
        return _failed(f"write failed: {e}")
    return _applied()


def apply() -> tuple[str, str]:
    """Dispatcher entry point.

    Returns (status, reason) tuple in the wiring convention used by
    _wiring_text_patch in apply/_state.py.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN118_V2_MD5_WORKSPACE")
    log_decision("PN118_V2_MD5_WORKSPACE", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target_str = resolve_vllm_file("v1/worker/workspace.py")
    if target_str is None:
        return "skipped", "v1/worker/workspace.py not found"
    target = Path(target_str)
    if not target.is_file():
        return "skipped", f"target not a file: {target}"

    result = _do_apply(target)
    log.info(
        "[PN118_V2_MD5_WORKSPACE] %s — %s", result.status, result.reason
    )
    return result.status, result.reason


def is_applied() -> bool:
    """True iff the Genesis v2 marker is present in the workspace.py target."""
    if vllm_install_root() is None:
        return False
    target_str = resolve_vllm_file("v1/worker/workspace.py")
    if target_str is None:
        return False
    target = Path(target_str)
    if not target.is_file():
        return False
    try:
        return _GENESIS_PN118_V2_WORKSPACE_MARKER in target.read_text()
    except OSError:
        return False
