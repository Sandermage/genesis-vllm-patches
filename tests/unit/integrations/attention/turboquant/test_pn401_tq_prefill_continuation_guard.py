# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN401 — TurboQuant prefill continuation guard (vllm#46461).

Genesis backport+improvement of OPEN vllm#46461 ([Bugfix][TurboQuant] guard
the flash_attn prefill fast path against silently dropping cached prefix
K/V on co-batched continuation requests), authored against live dev424
(pin ``0.23.1rc1.dev424+g3f5a1e173``).

The bug
-------
``_prefill_attention`` takes a flash_attn fast path on
``attn_metadata.max_query_len == attn_metadata.max_seq_len``, claiming "no
request has prior cached KV". That is INSUFFICIENT: a long first-chunk
prefill can inflate ``max_query_len`` to equal ``max_seq_len`` while the
SAME batch carries shorter continuation requests (``q_len < seq_len`` —
a non-first chunked-prefill chunk, no APC required). For those
continuations the fast path passes ``cu_seqlens_k = query_start_loc``, so
flash_attn attends only to the current chunk's raw K/V and silently drops
the cached prefix → hallucination on the continued request.

The fix
-------
Compute a host-side continuation check on the CPU mirror tensors BEFORE
the fast path and gate it with ``and not _has_continuation``. When any
continuation exists, fall through to the per-request branch that reads
cached K/V correctly.

OUR version over the raw PR (iron rule #10)
-------------------------------------------
1. Conservative None-mirror fall-safe: if EITHER CPU mirror is None we
   treat ``_has_continuation=True`` (skip the fast path) rather than the
   PR's implicit ``False`` (take the unsafe fast path). A missing mirror
   must never silently re-enable the buggy path.
2. Length-mismatch hardening: if the CPU tensors are shape-inconsistent
   (``len(qsl) < len(seq_lens)+1``) we also fall to the safe path.

This patch is a correctness fix that must be ON for ALL TurboQuant
hardware regardless of P101 (opt-in perf) / PN116 (HW-gated) state, so it
gets its own always-on lifecycle and is NOT folded into either. It
composes with P101 (continuation slicing, disjoint anchor) and PN116
(prefill max_seq fallback, disjoint anchor); PN399 touches the decode
path (disjoint).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.attention.turboquant import (
    pn401_tq_prefill_continuation_guard as pn401,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_LIVE = _FIXTURES / "pn401_live_anchor_region.txt"

_ENV_FLAG = "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD"

_DRIFT_MARKERS = [
    "_has_continuation",
    "[Genesis PN401",
]


# ─────────────────────────────────────────────────────────────────────
# 1. ANCHOR-PRESENCE on the frozen live-dev424 fixture
# ─────────────────────────────────────────────────────────────────────


def test_anchor_old_present_and_unique_in_live_region():
    """The pristine dev424 fast-path gate (3 comment lines + the `if`) is a
    unique substring of the frozen live region — guards against silent
    anchor drift."""
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    assert pn401.PN401_FASTPATH_OLD in live, "PN401 anchor_old absent in live"
    assert live.count(pn401.PN401_FASTPATH_OLD) == 1, "PN401 anchor not unique"


def test_pr_literal_has_continuation_absent_in_pristine():
    """The PR's `_has_continuation` literal must be ABSENT in pristine dev424
    (pre-merge state) — proves we did not silently no-op against an already
    merged tree."""
    live = _FIXTURE_LIVE.read_text(encoding="utf-8")
    assert "_has_continuation" not in live


def test_replacement_keeps_fast_path_and_adds_guard():
    """The replacement must still contain the original fast-path `if` (we
    only ADD `and not _has_continuation`), and must compute the guard from
    the CPU mirror tensors."""
    new = pn401.PN401_FASTPATH_NEW
    assert "and not _has_continuation" in new
    assert "attn_metadata.query_start_loc_cpu" in new
    assert "attn_metadata.seq_lens_cpu" in new
    # The original gate predicate is preserved inside the new `if` (the `if`
    # is wrapped across lines to stay within line-length limits, so assert
    # each component is present rather than the single-line spelling).
    assert "_HAS_FLASH_ATTN" in new
    assert "attn_metadata.max_query_len == attn_metadata.max_seq_len" in new
    # Our improvement: conservative None fall-safe is present.
    assert "is None" in new


# ─────────────────────────────────────────────────────────────────────
# Helpers / fixtures for apply tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def env_pn401_on(monkeypatch):
    monkeypatch.setenv(_ENV_FLAG, "1")
    monkeypatch.setenv("SNDR_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD", "1")
    yield


def _redirect_resolver(monkeypatch, tq_path: Path | None):
    def fake_resolve(rel: str):
        if "turboquant_attn" in rel:
            return str(tq_path) if tq_path is not None else None
        return None
    monkeypatch.setattr(pn401, "resolve_vllm_file", fake_resolve)


@pytest.fixture
def synthetic_target(tmp_path):
    tq = tmp_path / "turboquant_attn.py"
    tq.write_text(_FIXTURE_LIVE.read_text(encoding="utf-8"), encoding="utf-8")
    return tq


# ─────────────────────────────────────────────────────────────────────
# 2. APPLY -> APPLIED on a synthetic file built from the live region
# ─────────────────────────────────────────────────────────────────────


def test_apply_applied_drift_markers_present(env_pn401_on, monkeypatch, synthetic_target):
    tq = synthetic_target
    _redirect_resolver(monkeypatch, tq)

    status, reason = pn401.apply()
    assert status == "applied", reason

    text = tq.read_text(encoding="utf-8")
    for marker in _DRIFT_MARKERS:
        assert marker in text, f"drift marker {marker!r} absent post-apply"
    assert pn401.GENESIS_PN401_MARKER in text


# ─────────────────────────────────────────────────────────────────────
# 3. BEHAVIOURAL REPRO — execute the patched body against synthetic metadata
#    (the core TDD: prove the bug is fixed, not just that text changed)
# ─────────────────────────────────────────────────────────────────────


class _MetaStub:
    """Minimal TurboQuantMetadata stand-in exposing only the fields the
    fast-path gate reads."""

    def __init__(self, max_query_len, max_seq_len, qsl_cpu, seq_lens_cpu):
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.query_start_loc_cpu = qsl_cpu
        self.seq_lens_cpu = seq_lens_cpu


def _eval_gate(meta):
    """Evaluate the PATCHED fast-path gate predicate against a metadata stub.

    Reproduces exactly the guard the patch inserts: builds `_has_continuation`
    from the CPU mirrors with our None/length fall-safe, then returns the
    final boolean `_HAS_FLASH_ATTN and max_query_len==max_seq_len and not
    _has_continuation`. _HAS_FLASH_ATTN is assumed True (the bug only exists
    when flash_attn is available)."""
    return pn401.eval_fast_path_taken(meta, has_flash_attn=True)


def test_fast_path_skipped_when_continuation_present():
    """REPRO of vllm#46461: max_query_len == max_seq_len BUT one request is a
    continuation (q_len < seq_len). The unpatched gate would take the fast
    path (and drop the cached prefix); the patched gate must SKIP it."""
    # Two requests. Req-0 is a fresh full prefill (q_len==seq_len==512 →
    # inflates max_query_len to 512 == max_seq_len). Req-1 is a continuation
    # chunk: q_len=64 but seq_len=512 (448 cached tokens).
    qsl_cpu = [0, 512, 576]          # query_start_loc: cumulative query lens
    seq_lens_cpu = [512, 512]        # full sequence lengths per request
    meta = _MetaStub(
        max_query_len=512, max_seq_len=512,
        qsl_cpu=qsl_cpu, seq_lens_cpu=seq_lens_cpu,
    )
    # Unpatched predicate: just `max_query_len == max_seq_len` → True (BUG).
    assert meta.max_query_len == meta.max_seq_len
    # Patched: continuation present → fast path NOT taken.
    assert _eval_gate(meta) is False


def test_fast_path_taken_all_first_chunk():
    """Common case: every request is a first-chunk prefill (q_len == seq_len).
    No continuation → the fast path is preserved (no perf regression)."""
    qsl_cpu = [0, 256, 512]
    seq_lens_cpu = [256, 256]
    meta = _MetaStub(
        max_query_len=256, max_seq_len=256,
        qsl_cpu=qsl_cpu, seq_lens_cpu=seq_lens_cpu,
    )
    assert _eval_gate(meta) is True


def test_fast_path_not_taken_when_max_query_lt_max_seq():
    """When max_query_len != max_seq_len the gate is False regardless of the
    continuation check (the original short-circuit is preserved)."""
    qsl_cpu = [0, 64]
    seq_lens_cpu = [512]
    meta = _MetaStub(
        max_query_len=64, max_seq_len=512,
        qsl_cpu=qsl_cpu, seq_lens_cpu=seq_lens_cpu,
    )
    assert _eval_gate(meta) is False


def test_none_mirror_falls_safe():
    """OUR divergence from the raw PR: if query_start_loc_cpu is None we
    conservatively treat _has_continuation=True → fast path SKIPPED. The raw
    PR would take the (unsafe) fast path here."""
    meta = _MetaStub(
        max_query_len=512, max_seq_len=512,
        qsl_cpu=None, seq_lens_cpu=[512],
    )
    assert _eval_gate(meta) is False


def test_none_seq_lens_mirror_falls_safe():
    meta = _MetaStub(
        max_query_len=512, max_seq_len=512,
        qsl_cpu=[0, 512], seq_lens_cpu=None,
    )
    assert _eval_gate(meta) is False


def test_length_mismatch_falls_safe():
    """OUR hardening: if the CPU tensors are shape-inconsistent
    (len(qsl) < len(seq_lens)+1) we fall to the safe path."""
    # 2 requests in seq_lens but qsl only has 2 entries (needs 3).
    meta = _MetaStub(
        max_query_len=512, max_seq_len=512,
        qsl_cpu=[0, 512], seq_lens_cpu=[512, 512],
    )
    assert _eval_gate(meta) is False


# ─────────────────────────────────────────────────────────────────────
# 4. IDEMPOTENCY + state
# ─────────────────────────────────────────────────────────────────────


def test_apply_is_idempotent(env_pn401_on, monkeypatch, synthetic_target):
    tq = synthetic_target
    _redirect_resolver(monkeypatch, tq)

    status1, _ = pn401.apply()
    assert status1 == "applied"
    first = tq.read_text(encoding="utf-8")

    status2, reason2 = pn401.apply()
    # Marker present → idempotent skip (result_to_wiring_status maps
    # IDEMPOTENT -> "skipped").
    assert status2 == "skipped", reason2
    assert "already applied" in reason2.lower() or "marker present" in reason2.lower()
    assert tq.read_text(encoding="utf-8") == first  # byte-stable


def test_is_applied_reflects_state(env_pn401_on, monkeypatch, synthetic_target):
    tq = synthetic_target
    _redirect_resolver(monkeypatch, tq)
    assert pn401.is_applied() is False
    pn401.apply()
    assert pn401.is_applied() is True


def test_disabled_by_default(monkeypatch, synthetic_target):
    """With the env flag UNSET, PN401 is OFF in the dispatcher unless the
    YAML/operator enables it (default_on=False in the registry)."""
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    monkeypatch.delenv("SNDR_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD", raising=False)
    tq = synthetic_target
    _redirect_resolver(monkeypatch, tq)
    status, reason = pn401.apply()
    assert status == "skipped", reason
    assert pn401.GENESIS_PN401_MARKER not in tq.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 5. REGISTRY contract
# ─────────────────────────────────────────────────────────────────────


def test_registry_pn401_contract():
    from sndr.dispatcher.registry import PATCH_REGISTRY

    assert "PN401" in PATCH_REGISTRY, "PN401 missing from registry"
    meta = PATCH_REGISTRY["PN401"]
    assert meta["env_flag"] == _ENV_FLAG
    assert meta["family"] == "attention.turboquant"
    assert meta["category"] == "correctness"
    assert meta["lifecycle"] == "experimental"
    assert meta["upstream_pr"] == 46461
    assert meta["applies_to"]["is_turboquant"] is True
    assert meta["applies_to"]["vllm_version_range"] == (">=0.23.0", "<0.24.0")
    # Composes with the sibling TQ prefill patches (disjoint anchors).
    composes = set(meta.get("composes_with", []))
    assert {"P101", "PN116"}.issubset(composes)


def test_env_flag_registered():
    from sndr.env import Flags
    assert hasattr(Flags, "PN401_TQ_PREFILL_CONTINUATION_GUARD")
    assert (
        Flags.PN401_TQ_PREFILL_CONTINUATION_GUARD
        == "PN401_TQ_PREFILL_CONTINUATION_GUARD"
    )
