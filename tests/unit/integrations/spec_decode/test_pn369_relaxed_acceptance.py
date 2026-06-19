# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN369 — relaxed acceptance for MTP spec-decode (torch-less).

Pure-Python tests: env parsing (defaults / garbage / clamps), anchor-
replacement structure for the three sub-patches on rejection_sampler.py,
apply() decision tree, P71 wiring threading, registry/Flags consistency.

The torch / Triton runtime tests (mask vs loop reference, block-verify
tail-extension cases, kernel ON/OFF equivalence, Triton-PyTorch parity)
live in `test_pn369_relaxed_acceptance_torch.py` — the tests/conftest.py
AST scan auto-skips any file importing torch on torch-less hosts, so the
split keeps THIS group running on CPU-only dev rigs and in torch-less CI.

Run via:
  pytest tests/unit/integrations/spec_decode/test_pn369_relaxed_acceptance.py -v
"""
from __future__ import annotations

import pytest

from sndr.engines.vllm.patches.spec_decode.pn369_relaxed_acceptance import (
    GENESIS_PN369_MARKER,
    PN369_BODY_NEW,
    PN369_BODY_OLD,
    PN369_DEFAULT_DELTA,
    PN369_DEFAULT_TOPK,
    PN369_LAUNCH_NEW,
    PN369_LAUNCH_OLD,
    PN369_SIG_NEW,
    PN369_SIG_OLD,
    apply,
    is_pn369_runtime_enabled,
    read_relaxed_delta,
    read_relaxed_topk,
)


# ─── env parsing ────────────────────────────────────────────────────────


def test_runtime_enabled_default_off(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", raising=False)
    assert is_pn369_runtime_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", " on "])
def test_runtime_enabled_truthy(monkeypatch, val):
    monkeypatch.setenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", val)
    assert is_pn369_runtime_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "off", ""])
def test_runtime_enabled_falsy(monkeypatch, val):
    monkeypatch.setenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", val)
    assert is_pn369_runtime_enabled() is False


def test_topk_default_when_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_PN369_RELAXED_TOPK", raising=False)
    assert read_relaxed_topk() == PN369_DEFAULT_TOPK == 4


def test_topk_default_when_garbage(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_TOPK", "four")
    assert read_relaxed_topk() == PN369_DEFAULT_TOPK


def test_topk_valid(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_TOPK", "8")
    assert read_relaxed_topk() == 8


def test_topk_clamped(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_TOPK", "0")
    assert read_relaxed_topk() == 1
    monkeypatch.setenv("GENESIS_PN369_RELAXED_TOPK", "100")
    assert read_relaxed_topk() == 32


def test_delta_default_when_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_PN369_RELAXED_DELTA", raising=False)
    assert read_relaxed_delta() == PN369_DEFAULT_DELTA == 0.2


def test_delta_default_when_garbage(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_DELTA", "tiny")
    assert read_relaxed_delta() == PN369_DEFAULT_DELTA


def test_delta_valid(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_DELTA", "0.35")
    assert read_relaxed_delta() == pytest.approx(0.35)


def test_delta_clamped(monkeypatch):
    monkeypatch.setenv("GENESIS_PN369_RELAXED_DELTA", "-0.5")
    assert read_relaxed_delta() == 0.0
    monkeypatch.setenv("GENESIS_PN369_RELAXED_DELTA", "1.5")
    assert read_relaxed_delta() == 1.0


# ─── anchor / replacement structure ─────────────────────────────────────


def test_sig_replacement_preserves_anchor_lines():
    """SIG replacement keeps every original signature line (append-only)."""
    for line in PN369_SIG_OLD.splitlines():
        assert line in PN369_SIG_NEW.splitlines(), f"missing line: {line!r}"
    assert "genesis_pn369_relaxed_ok_ptr=None" in PN369_SIG_NEW
    assert "GENESIS_PN369_RELAXED: tl.constexpr = False" in PN369_SIG_NEW


def test_body_replacement_keeps_if_accepted_block_intact():
    """BODY replacement ends with the unmodified upstream block."""
    assert PN369_BODY_NEW.endswith(PN369_BODY_OLD)
    assert "[Genesis PN369]" in PN369_BODY_NEW


def test_body_replacement_guards_synthetic_and_off():
    """Relaxed OR-compose is dead code unless RELAXED constexpr is True
    and the kernel is not in synthetic mode."""
    assert "if GENESIS_PN369_RELAXED:" in PN369_BODY_NEW
    assert "if not SYNTHETIC_MODE:" in PN369_BODY_NEW
    assert "if not accepted:" in PN369_BODY_NEW


def test_body_anchor_after_strict_accept_line():
    """The body anchor must NOT include the strict-accept assignment —
    P82 replaces that region; disjointness keeps both apply-order safe."""
    assert "draft_prob > 0" not in PN369_BODY_OLD
    assert "NOTE(woosuk)" not in PN369_BODY_OLD


def test_launch_replacement_preserves_anchor_and_threads_args():
    assert PN369_LAUNCH_NEW.endswith(
        "        genesis_pn369_relaxed_ok_ptr=_genesis_pn369_relaxed_ok,\n"
        "        GENESIS_PN369_RELAXED=_genesis_pn369_relaxed_ok is not None,\n"
        "    )\n"
    )
    # Original launch kwargs preserved.
    assert "NO_DRAFT_PROBS=draft_probs is None," in PN369_LAUNCH_NEW
    assert "SYNTHETIC_MODE=synthetic_mode," in PN369_LAUNCH_NEW
    # Mask only computed outside synthetic mode, with fail-safe fallback.
    assert "if not synthetic_mode:" in PN369_LAUNCH_NEW
    assert "except Exception" in PN369_LAUNCH_NEW
    assert "compute_relaxed_ok_mask" in PN369_LAUNCH_NEW


def test_launch_anchor_contains_original_launch_block():
    """Every line of the original launch block survives in the
    replacement (the kernel call itself is preserved verbatim)."""
    for line in PN369_LAUNCH_OLD.splitlines():
        assert line in PN369_LAUNCH_NEW.splitlines(), f"missing line: {line!r}"


def test_marker_mentions_pn369():
    assert "PN369" in GENESIS_PN369_MARKER


def test_no_baked_values_in_replacements():
    """topk/delta are runtime knobs — they must NOT be baked into the
    injected text (unlike P82's baked threshold)."""
    for text in (PN369_SIG_NEW, PN369_BODY_NEW, PN369_LAUNCH_NEW):
        assert "RELAXED_TOPK" not in text
        assert "RELAXED_DELTA" not in text


# ─── apply() decision tree ──────────────────────────────────────────────


def test_apply_skipped_when_env_unset(monkeypatch):
    # apply() delegates to the consolidated P71+PN369 module since 2026-06-19;
    # with BOTH features' enable flags unset the consolidated apply() skips.
    for f in (
        "GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE",
        "SNDR_ENABLE_PN369_RELAXED_ACCEPTANCE",
        "GENESIS_ENABLE_P71_BLOCK_VERIFY",
        "SNDR_ENABLE_P71_BLOCK_VERIFY",
    ):
        monkeypatch.delenv(f, raising=False)
    status, reason = apply()
    assert status == "skipped", f"expected skip, got {status}: {reason}"


def test_apply_skipped_when_env_zero(monkeypatch):
    # Consolidated apply() needs BOTH features off to skip; P71 stays unset.
    monkeypatch.delenv("GENESIS_ENABLE_P71_BLOCK_VERIFY", raising=False)
    monkeypatch.delenv("SNDR_ENABLE_P71_BLOCK_VERIFY", raising=False)
    monkeypatch.setenv("GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE", "0")
    status, reason = apply()
    assert status == "skipped", f"expected skip, got {status}: {reason}"


# ─── P71 wiring threading ───────────────────────────────────────────────


def test_p71_wiring_threads_relaxed_ok():
    from sndr.engines.vllm.patches.spec_decode.p71_block_verify import (
        GENESIS_P71_MARKER,
        P71_NEW,
        P71_OLD,
    )
    assert "compute_relaxed_ok_mask" in P71_NEW
    assert "relaxed_ok=_genesis_pn369_relaxed_ok," in P71_NEW
    # Marker bumped so the new text re-bakes on a clean container fs.
    assert "v7.43" in GENESIS_P71_MARKER
    assert "v7.42" not in GENESIS_P71_MARKER
    # Anchor head/tail still intact (sanity for the injected structure).
    assert P71_NEW.startswith(P71_OLD.split("\n")[0])
    assert P71_NEW.endswith("    recovered_token_ids = sample_recovered_tokens(\n")


# ─── registry / Flags consistency ───────────────────────────────────────


def test_registry_entry_consolidated_into_p71():
    """2026-06-19: PN369 was consolidated INTO the P71 registry entry (both
    text-patch rejection_sampler.py at disjoint regions). PN369 is no longer
    a standalone registry id; its enable flag is retained as an
    env_flag_alias on P71, and P71's apply_module points at the consolidated
    wiring module."""
    from sndr.dispatcher import PATCH_REGISTRY
    assert "PN369" not in PATCH_REGISTRY
    meta = PATCH_REGISTRY["P71"]
    assert meta["family"] == "spec_decode"
    assert meta["env_flag"] == "GENESIS_ENABLE_P71_BLOCK_VERIFY"
    assert "GENESIS_ENABLE_PN369_RELAXED_ACCEPTANCE" in meta["env_flag_aliases"]
    assert meta["default_on"] is False
    assert meta["apply_module"] == (
        "sndr.engines.vllm.patches.spec_decode."
        "p71_pn369_rejection_sampler_consolidated"
    )


def test_env_flags_class_has_pn369():
    from sndr.env import Flags
    assert Flags.PN369_RELAXED_ACCEPTANCE == "PN369_RELAXED_ACCEPTANCE"
