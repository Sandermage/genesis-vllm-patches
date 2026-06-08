# SPDX-License-Identifier: Apache-2.0
"""PN118 v2 — md5+full-file PoC (turboquant_attn.py scope).

Sibling to ``pn118_v2_md5_workspace.py`` which shipped in v11.1.0
covering pn118's other target file (``v1/worker/workspace.py``).
Together the two v2 patches replace pn118's anchor-based coverage of
its full 2-file scope with md5+full-file replacements — one v2 patch
per target file. Original anchor-based PN118 still ships and composes
with both v2 patches via Genesis markers.

================================================================
WHY THIS PoC EXISTS (v11.2.0 continuation of P3.1 closeout)
================================================================

During the v11.1.0 closeout (workspace.py PoC), Track B.1 scout
discovered that pn118 patches **two** files. The v11.1.0 release
landed v2 PoC for workspace.py only. This v11.2.0 patch closes the
second file.

Drift finding from rig (2026-06-02): the upstream
``v1/attention/backends/turboquant_attn.py`` has drifted from pn118's
anchor baseline. Of pn118's 2 anchor pairs for this file:

  - ``TQ_ANCHOR_INIT_OLD`` — NOT FOUND in current upstream
    (pn118 silently no-ops on this anchor at the current pin)
  - ``TQ_ANCHOR_DECODE_OLD`` — still matches; pn118 applies it cleanly

This silent partial-apply is exactly the failure mode the md5+full-
file pattern is designed to prevent. The v2 patch:

  - Pin baseline (PRE_PATCH_MD5): current pin
    ``0.21.1rc0+g626fa9bba5`` upstream content
  - Post-patch fixture: pre-patch + DECODE anchor applied + marker
    (the INIT anchor was already drifted before the PoC was authored,
     so there's no "what INIT-applied would look like" reference —
     the v2 PoC documents the current state of pn118's coverage)
  - Status on this pin: ``applied`` when md5 matches → DECODE-equivalent
    full file content written; INIT remains unpatched (matches pn118's
    real behavior on this pin); marker added so future apply() calls
    skip cleanly
  - Status on future pin bumps: ``skipped`` (md5 mismatch) — operator
    must regenerate fixtures from the new rig pin before v2 can apply

================================================================
PATTERN (matches PN119 + workspace.py sibling)
================================================================

    PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5 = "<32-char hex>"
    PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT = '''<full file post-patch>'''

    def apply():
        target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
        if _GENESIS_PN118_V2_TQ_ATTN_MARKER in target.read_text():
            return _skipped("already applied")
        if _file_md5(target) != PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5:
            return _skipped("md5 mismatch — upstream drifted")
        target.write_text(PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT)
        return _applied()

================================================================
STATUS
================================================================

Default OFF; opt-in via ``GENESIS_ENABLE_PN118_V2_MD5_TURBOQUANT_ATTN=1``.
Composes with the original PN118 + the workspace.py sibling — all
three can be enabled simultaneously:

  - PN118_V2_MD5_WORKSPACE     owns ``v1/worker/workspace.py``
  - PN118_V2_MD5_TURBOQUANT_ATTN owns ``v1/attention/backends/turboquant_attn.py``
  - PN118 (original) detects both markers and skips its 4 anchors
    on both files; effectively becomes a no-op when both v2 patches
    are on (which is the intended end-state for pn118's eventual
    full retirement once md5 pattern is validated across a couple
    of pin bumps)

When all three patches are off + only pn118 is on: legacy behavior,
partial-apply silently degrades on the drifted INIT anchor.
When all three patches are on: v2 patches own both files fully,
pn118 no-ops via marker detection, INIT anchor's silent drift becomes
explicit (operator-visible via boot log + md5 verification).

v11.2.0 Phase 6 P3.1 continuation (continues v11.1.0 PoC).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn118_v2_md5_turboquant_attn")


_GENESIS_PN118_V2_TQ_ATTN_MARKER = (
    "# Genesis PN118 v2 (md5+full-file PoC, turboquant_attn.py scope) marker"
    " — DO NOT REMOVE"
)


# Md5 hash of the upstream ``v1/attention/backends/turboquant_attn.py``
# at PROD pin ``0.21.1rc0+g626fa9bba5``. If a future pin bump changes
# the file this constant must be regenerated alongside the bundled
# post-patch fixture. On drift the v2 path self-skips via md5 mismatch
# and the original PN118 (anchor-based) continues to attempt patching.
PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5 = "8ee234caac59bf099d717e56d7cfa00c"


# Bundled fixtures (referenced by tests + by apply path).
# v12.x moved this module deeper; walk up to the repo root that holds
# tests/unit/integrations instead of a fixed parents[N] (which now points
# at the sndr/ package, leaving the fixture unreadable -> empty md5).
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
    _TESTS_FIXTURE_DIR / "pn118_v2_md5_turboquant_attn_post_patch.py.txt"
)
_PRE_PATCH_FIXTURE = (
    _TESTS_FIXTURE_DIR / "pn118_v2_md5_turboquant_attn_pre_patch.py.txt"
)


def _load_post_patch_content() -> str:
    """Load post-patch file content. Bundled as test fixture so the
    patch module stays small; the fixture is read at module import.
    Marker is appended if not already present (defense in depth —
    the rig-generated fixture already contains it)."""
    if not _POST_PATCH_FIXTURE.is_file():
        return ""
    content = _POST_PATCH_FIXTURE.read_text()
    if _GENESIS_PN118_V2_TQ_ATTN_MARKER not in content:
        sep = "" if content.endswith("\n") else "\n"
        content = content + sep + _GENESIS_PN118_V2_TQ_ATTN_MARKER + "\n"
    return content


PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT = _load_post_patch_content()


@dataclass(frozen=True)
class _Result:
    status: str  # "applied" | "skipped" | "failed"
    reason: str = ""


def _applied(reason: str = "") -> _Result:
    return _Result(
        status="applied",
        reason=reason or "PN118 v2 md5+full-file replacement applied (turboquant_attn.py)",
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
    if _GENESIS_PN118_V2_TQ_ATTN_MARKER in current_text:
        return _skipped("already applied (marker present)")
    current_md5 = _file_md5(target)
    if current_md5 != PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5:
        return _skipped(
            f"md5 mismatch (got {current_md5}, expected "
            f"{PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5}) — upstream drifted "
            "from PoC baseline. The original PN118 (anchor-based) will "
            "continue to attempt patching this file on its own."
        )
    if not PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT:
        return _skipped(
            "post-patch fixture missing — regenerate from rig "
            "(tests/unit/integrations/attention/turboquant/fixtures/"
            "pn118_v2_md5_turboquant_attn_post_patch.py.txt)"
        )
    try:
        target.write_text(PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT)
    except OSError as e:
        return _failed(f"write failed: {e}")
    return _applied()


def apply() -> tuple[str, str]:
    """Dispatcher entry point.

    Returns (status, reason) tuple in the wiring convention used by
    _wiring_text_patch in apply/_state.py.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN118_V2_MD5_TURBOQUANT_ATTN")
    log_decision("PN118_V2_MD5_TURBOQUANT_ATTN", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target_str = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target_str is None:
        return "skipped", "v1/attention/backends/turboquant_attn.py not found"
    target = Path(target_str)
    if not target.is_file():
        return "skipped", f"target not a file: {target}"

    result = _do_apply(target)
    log.info(
        "[PN118_V2_MD5_TURBOQUANT_ATTN] %s — %s", result.status, result.reason
    )
    return result.status, result.reason


def is_applied() -> bool:
    """True iff the Genesis v2 marker is present in the turboquant_attn.py target."""
    if vllm_install_root() is None:
        return False
    target_str = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target_str is None:
        return False
    target = Path(target_str)
    if not target.is_file():
        return False
    try:
        return _GENESIS_PN118_V2_TQ_ATTN_MARKER in target.read_text()
    except OSError:
        return False
