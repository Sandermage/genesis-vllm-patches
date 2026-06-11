# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard for the v11.3.0 bug class — integration
modules registering POOL_* names with the wrong pool type.

The bug class
-------------

PersistentBufferRegistry exposes two pool types:

  - BufferPool — free-list acquire/release semantics (transient buffers
    with explicit lifecycle, caller knows when done).

  - PersistentSlicePool — grow-in-place + slice-on-acquire (hot-path
    buffers with "max-so-far" sizing + CUDA-graph capture requirements).

A pool NAME is locked to a TYPE on first registration. Subsequent
get_pool() / get_slice_pool() calls on the same name with a different
type raise ValueError.

Pre-v11.3.0 bug: 4 integration modules (PN12, P46, P36, P39a) called
get_pool() in their ensure_pool_registered() helpers, but their storage
classes (FFNIntermediateCache, GdnGatingBufferManager,
TurboQuantBufferManager, FlaKktBufferManager) used get_slice_pool()
internally. Whichever ran first won, and the other raised ValueError.

Root cause: ensure_pool_registered() docstrings said "operator
visibility only", but the wrong-type-registration broke the storage
class's real allocation path.

This file scans every Genesis integration module for the bug pattern:
calls to PersistentBufferRegistry().get_pool(POOL_*_NAME) in a function
named ensure_pool_registered(). The default in v11.3.0+ is to use
get_slice_pool() unless the storage truly uses BufferPool's
acquire/release semantics — but we don't have any such case yet.

If a new patch ships with the same wrong pattern, this test fails fast
in CI.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.3.0 bug-class regression guard
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INTEGRATIONS_DIR = (
    REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
)
KERNELS_DIR = REPO_ROOT / "vllm" / "sndr_core" / "kernels"

# Regex that matches the buggy pattern. We look for the full chain
# `PersistentBufferRegistry().get_pool(SOMETHING)` inside a file that
# also defines `def ensure_pool_registered`. The function-scoped match
# is approximate (we don't full-parse Python) but the cost of a false
# positive (some non-helper `get_pool` call) is low: just whitelist
# the file with a docstring explanation.
_BUGGY_PATTERN = re.compile(
    r"PersistentBufferRegistry\(\)\s*\.\s*get_pool\s*\(",
    re.MULTILINE,
)
_HELPER_FUNC_PATTERN = re.compile(
    r"^def ensure_pool_registered\s*\(", re.MULTILINE,
)

# Files that are documented to use BufferPool semantics (acquire +
# release with free-list reuse). Empty at v11.3.0 — no integration
# module currently uses BufferPool in production.
#
# When adding a future allocator that genuinely uses BufferPool (e.g.
# a transient scratch with explicit caller release), add its path here
# with a comment justifying the BufferPool choice over PersistentSlicePool.
_KNOWN_BUFFER_POOL_HELPERS: frozenset[str] = frozenset({
    # No allowlist entries at v11.3.0.
})


def _scan_for_buggy_helpers() -> list[tuple[Path, str]]:
    """Walk integration + kernels dirs, return (path, source_snippet)
    for every file that has BOTH `def ensure_pool_registered` AND a
    `PersistentBufferRegistry().get_pool(...)` call."""
    candidates: list[tuple[Path, str]] = []
    for root in (INTEGRATIONS_DIR, KERNELS_DIR):
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if not _HELPER_FUNC_PATTERN.search(src):
                continue
            if _BUGGY_PATTERN.search(src):
                candidates.append((p, src))
    return candidates


def test_no_integration_module_misuses_get_pool_in_registration_helper():
    """Every `def ensure_pool_registered` must call `.get_slice_pool(`
    not `.get_pool(`. Allowlist via _KNOWN_BUFFER_POOL_HELPERS when a
    helper genuinely needs BufferPool semantics."""
    candidates = _scan_for_buggy_helpers()
    repo_rel = lambda p: str(p.relative_to(REPO_ROOT))  # noqa: E731
    offenders = [
        (p, src) for (p, src) in candidates
        if repo_rel(p) not in _KNOWN_BUFFER_POOL_HELPERS
    ]
    if offenders:
        lines = []
        for (p, _) in offenders:
            lines.append(f"  - {repo_rel(p)}")
        raise AssertionError(
            "v11.3.0 bug-class regression: the following integration "
            "modules use `PersistentBufferRegistry().get_pool(...)` "
            "in a `def ensure_pool_registered` helper.\n\n"
            "Storage classes that back these pools use PersistentSlicePool "
            "(get_slice_pool), so the wrong-type registration breaks the "
            "downstream lookup with ValueError.\n\n"
            "Fix: change `.get_pool(` to `.get_slice_pool(` in the helper. "
            "If the storage class genuinely uses BufferPool's "
            "acquire/release semantics, allowlist the file path in "
            "_KNOWN_BUFFER_POOL_HELPERS with a comment explaining why.\n\n"
            f"Offending files ({len(offenders)}):\n" + "\n".join(lines)
        )


def test_known_good_helpers_pass_the_scan():
    """Positive sanity — the 4 fixed v11.3.0 helpers (PN12, P46, P36, P39a)
    all call get_slice_pool, not get_pool."""
    fixed_helpers = [
        REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "kernels"
        / "pn12_ffn_intermediate_pool.py",
        REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "attention" / "gdn"
        / "p46_gdn_gating_buffers.py",
        # P36 retired 2026-06-11 (preflight residual triage §3) — module
        # moved to _archive/; helper code unchanged, scan still applies.
        REPO_ROOT / "sndr" / "engines" / "vllm" / "_archive"
        / "p36_tq_shared_decode_buffers.py",
        REPO_ROOT / "sndr" / "engines" / "vllm" / "patches" / "attention" / "gdn"
        / "p39a_fla_kkt_buffer.py",
    ]
    for p in fixed_helpers:
        assert p.is_file(), f"expected fixed helper missing: {p}"
        src = p.read_text(encoding="utf-8")
        assert "get_slice_pool(" in src, (
            f"{p.name} doesn't call get_slice_pool() — was it un-fixed?"
        )
        # The helper function (ensure_pool_registered) specifically
        # should not call get_pool(SOMETHING_constant).
        # Allow get_pool to appear in unrelated contexts (e.g. comments
        # explaining the v11.3.0 fix) but not in an active code call.
        # Heuristic: find a line matching `PersistentBufferRegistry().get_pool(`
        # that's NOT in a comment.
        lines = src.split("\n")
        offending = []
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if "PersistentBufferRegistry()" in line and (
                ".get_pool(" in line
                and ".get_slice_pool(" not in line
            ):
                offending.append((i, line.rstrip()))
        assert not offending, (
            f"{p.name} still has buggy get_pool() call(s): {offending}"
        )
