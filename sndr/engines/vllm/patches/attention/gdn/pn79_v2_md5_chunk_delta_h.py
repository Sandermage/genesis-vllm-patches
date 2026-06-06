# SPDX-License-Identifier: Apache-2.0
"""PN79 v2 — md5+full-file PoC (chunk_delta_h.py scope).

Sibling 2 of pn79's multi-file md5 conversion. Companion to
``pn79_v2_md5_chunk.py`` (sibling 1). Together the two v2 patches
cover pn79's remaining-in-upstream targets via md5+full-file
replacements — one v2 patch per target file. Original anchor-based
PN79 still ships and composes with both v2 patches via Genesis markers
(post-v2 markers prevent pn79 from re-anchoring on the same files).

================================================================
WHY THIS PoC EXISTS (v11.2.0 continuation of P3.1 closeout)
================================================================

PN79 (``pn79_inplace_ssm_state``) originally targeted four files; the
model-side targets (``gdn_linear_attn.py``, ``olmo_hybrid.py``) have
drifted out of upstream entirely (file split / file removed). The v2
PoC scope is therefore the two FLA ops files. This module covers
``chunk_delta_h.py`` — drift finding from rig (2026-06-03): 3 of pn79's
4 anchors for this file apply cleanly on current pin, but
``ANCHOR_2B_KERNEL_SIG`` does not match upstream (silent no-op).

This silent partial-apply is the failure mode the md5+full-file pattern
is designed to prevent: md5 guards against ANY drift, not just the
per-anchor view.

================================================================
PATTERN (matches PN119 + PN118 v2 + PN79 v2 chunk.py siblings)
================================================================

    PN79_V2_MD5_CHUNK_DELTA_H_PRE_PATCH_MD5 = "<32-char hex>"
    PN79_V2_MD5_CHUNK_DELTA_H_POST_PATCH_CONTENT = '''<full file post-patch>'''

    def apply():
        target = resolve_vllm_file(
            "model_executor/layers/fla/ops/chunk_delta_h.py"
        )
        if _GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER in target.read_text():
            return _skipped("already applied")
        if _file_md5(target) != PN79_V2_MD5_CHUNK_DELTA_H_PRE_PATCH_MD5:
            return _skipped("md5 mismatch — upstream drifted")
        target.write_text(PN79_V2_MD5_CHUNK_DELTA_H_POST_PATCH_CONTENT)
        return _applied()

================================================================
STATUS
================================================================

Default OFF; opt-in via ``GENESIS_ENABLE_PN79_V2_MD5_CHUNK_DELTA_H=1``.
Composes with the original PN79 + the chunk.py sibling — all three can
be enabled simultaneously:

  - PN79_V2_MD5_CHUNK         owns ``model_executor/layers/fla/ops/chunk.py``
  - PN79_V2_MD5_CHUNK_DELTA_H owns ``model_executor/layers/fla/ops/chunk_delta_h.py``
  - PN79 (original) detects both markers and skips its 7+4 anchors on
    those files; effectively becomes a no-op on the FLA ops scope when
    both v2 patches are on (which is the intended end-state for pn79's
    eventual full retirement once md5 pattern is validated across a
    couple of pin bumps)

When v2 patch is off + only PN79 is on: legacy behavior, partial-apply
silently degrades on the drifted ANCHOR_2B_KERNEL_SIG.
When both v2 patches are on: v2 owns both FLA ops files fully, PN79
no-ops via marker detection on those files, drift becomes explicit
(operator-visible via boot log + md5 verification).

v11.2.0 Phase 6 P3.1 continuation (continues v11.1.0 PoC).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn79_v2_md5_chunk_delta_h")


_GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER = (
    "# Genesis PN79 v2 (md5+full-file PoC, chunk_delta_h.py scope) marker"
    " — DO NOT REMOVE"
)


# Md5 hash of the upstream ``model_executor/layers/fla/ops/chunk_delta_h.py``
# at PROD pin ``0.20.2rc1.dev338+gbf0d2dc6d``. If a future pin bump
# changes the file this constant must be regenerated alongside the
# bundled post-patch fixture. On drift the v2 path self-skips via md5
# mismatch and the original PN79 (anchor-based) continues to attempt
# patching.
PN79_V2_MD5_CHUNK_DELTA_H_PRE_PATCH_MD5 = "71b7a5017e8cb4c08617c19f5b5f7d4b"


# Bundled fixtures (referenced by tests + by apply path).
_TESTS_FIXTURE_DIR = (
    Path(__file__).resolve().parents[5]
    / "tests"
    / "unit"
    / "integrations"
    / "attention"
    / "gdn"
    / "fixtures"
)
_POST_PATCH_FIXTURE = (
    _TESTS_FIXTURE_DIR / "pn79_v2_md5_chunk_delta_h_post_patch.py.txt"
)
_PRE_PATCH_FIXTURE = (
    _TESTS_FIXTURE_DIR / "pn79_v2_md5_chunk_delta_h_pre_patch.py.txt"
)


def _load_post_patch_content() -> str:
    """Load post-patch file content. Bundled as test fixture so the
    patch module stays small; the fixture is read at module import.
    Marker is appended if not already present (defense in depth —
    the rig-generated fixture already contains it)."""
    if not _POST_PATCH_FIXTURE.is_file():
        return ""
    content = _POST_PATCH_FIXTURE.read_text()
    if _GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER not in content:
        sep = "" if content.endswith("\n") else "\n"
        content = content + sep + _GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER + "\n"
    return content


PN79_V2_MD5_CHUNK_DELTA_H_POST_PATCH_CONTENT = _load_post_patch_content()


@dataclass(frozen=True)
class _Result:
    status: str  # "applied" | "skipped" | "failed"
    reason: str = ""


def _applied(reason: str = "") -> _Result:
    return _Result(
        status="applied",
        reason=reason
        or "PN79 v2 md5+full-file replacement applied (chunk_delta_h.py)",
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
    if _GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER in current_text:
        return _skipped("already applied (marker present)")
    current_md5 = _file_md5(target)
    if current_md5 != PN79_V2_MD5_CHUNK_DELTA_H_PRE_PATCH_MD5:
        return _skipped(
            f"md5 mismatch (got {current_md5}, expected "
            f"{PN79_V2_MD5_CHUNK_DELTA_H_PRE_PATCH_MD5}) — upstream drifted "
            "from PoC baseline. The original PN79 (anchor-based) will "
            "continue to attempt patching this file on its own."
        )
    if not PN79_V2_MD5_CHUNK_DELTA_H_POST_PATCH_CONTENT:
        return _skipped(
            "post-patch fixture missing — regenerate from rig "
            "(tests/unit/integrations/attention/gdn/fixtures/"
            "pn79_v2_md5_chunk_delta_h_post_patch.py.txt)"
        )
    try:
        target.write_text(PN79_V2_MD5_CHUNK_DELTA_H_POST_PATCH_CONTENT)
    except OSError as e:
        return _failed(f"write failed: {e}")
    return _applied()


def apply() -> tuple[str, str]:
    """Dispatcher entry point.

    Returns (status, reason) tuple in the wiring convention used by
    _wiring_text_patch in apply/_state.py.
    """
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN79_V2_MD5_CHUNK_DELTA_H")
    log_decision("PN79_V2_MD5_CHUNK_DELTA_H", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target_str = resolve_vllm_file(
        "model_executor/layers/fla/ops/chunk_delta_h.py"
    )
    if target_str is None:
        return (
            "skipped",
            "model_executor/layers/fla/ops/chunk_delta_h.py not found",
        )
    target = Path(target_str)
    if not target.is_file():
        return "skipped", f"target not a file: {target}"

    result = _do_apply(target)
    log.info(
        "[PN79_V2_MD5_CHUNK_DELTA_H] %s — %s", result.status, result.reason
    )
    return result.status, result.reason


def is_applied() -> bool:
    """True iff the Genesis v2 marker is present in the chunk_delta_h.py target."""
    if vllm_install_root() is None:
        return False
    target_str = resolve_vllm_file(
        "model_executor/layers/fla/ops/chunk_delta_h.py"
    )
    if target_str is None:
        return False
    target = Path(target_str)
    if not target.is_file():
        return False
    try:
        return _GENESIS_PN79_V2_CHUNK_DELTA_H_MARKER in target.read_text()
    except OSError:
        return False
