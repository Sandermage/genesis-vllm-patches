# SPDX-License-Identifier: Apache-2.0
"""TDD for PN80 — LoRA tensorizer device kwarg backport (vllm#41845).

Single anchor patch. Tests verify:
  - Module imports cleanly
  - Anchor strings well-formed (OLD vs NEW differ, NEW has device=device)
  - Round-trip idempotency on tmp_path fixture
  - Registry entry properly gated
  - apply() returns 'skipped' when env disabled
  - Drift marker absent from pristine
"""
from __future__ import annotations

import pytest


def _wiring():
    from sndr.engines.vllm._archive import pn80_lora_tensorizer_device as M
    return M


class TestPN80AnchorContent:

    def test_OLD_has_TensorDeserializer_call(self):
        m = _wiring()
        assert "TensorDeserializer(" in m.ANCHOR_OLD
        assert "lora_tensor_path" in m.ANCHOR_OLD
        assert "dtype=tensorizer_config.dtype" in m.ANCHOR_OLD
        assert "**tensorizer_args.deserialization_kwargs" in m.ANCHOR_OLD

    def test_OLD_does_NOT_have_device_kwarg(self):
        """Pristine code lacks device= line."""
        m = _wiring()
        assert "device=device" not in m.ANCHOR_OLD

    def test_NEW_inserts_device_kwarg(self):
        m = _wiring()
        assert "device=device,\n" in m.ANCHOR_NEW
        # Must come BEFORE **deserialization_kwargs
        idx_device = m.ANCHOR_NEW.find("device=device,")
        idx_kwargs = m.ANCHOR_NEW.find("**tensorizer_args.deserialization_kwargs")
        assert 0 < idx_device < idx_kwargs

    def test_OLD_NEW_differ(self):
        m = _wiring()
        assert m.ANCHOR_OLD != m.ANCHOR_NEW
        # Sanity: difference is exactly the device=device line
        # (NEW has 1 more line than OLD)
        old_lines = m.ANCHOR_OLD.count("\n")
        new_lines = m.ANCHOR_NEW.count("\n")
        assert new_lines == old_lines + 1


class TestPN80RoundTrip:

    def test_anchor_idempotent_apply(self, tmp_path):
        """Apply OLD anchor on a tmp file, second apply must be IDEMPOTENT."""
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        m = _wiring()

        target = tmp_path / "lora_model.py"
        target.write_text("# header\n" + m.ANCHOR_OLD + "\n# tail\n")

        patcher = TextPatcher(
            patch_name="PN80 test",
            target_file=str(target),
            marker=m.GENESIS_PN80_MARKER,
            sub_patches=[
                TextPatch(name="lora_device", anchor=m.ANCHOR_OLD,
                          replacement=m.ANCHOR_NEW, required=True),
            ],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        body = target.read_text()
        assert "device=device" in body
        assert m.GENESIS_PN80_MARKER in body

        # Second apply — must be IDEMPOTENT (marker present)
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT


class TestPN80Registry:

    def test_PN80_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN80" in PATCH_REGISTRY

    def test_PN80_default_off(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN80"]["default_on"] is False

    def test_PN80_env_flag(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert (PATCH_REGISTRY["PN80"]["env_flag"]
                == "GENESIS_ENABLE_PN80_LORA_TENSORIZER_DEVICE")

    def test_PN80_credit_mentions_pr_41845(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "41845" in PATCH_REGISTRY["PN80"]["credit"]


class TestPN80ApplyContract:

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN80_LORA_TENSORIZER_DEVICE",
                           raising=False)
        m = _wiring()
        status, reason = m.apply()
        assert status == "skipped"
        assert "off" in reason.lower() or "opt-in" in reason.lower()


class TestPN80AgainstPristine:
    """Verify anchor matches actual main HEAD pristine source."""

    def test_anchor_unique_in_committed_pristine(self):
        """If we have a committed lora_model.py pristine fixture, verify
        anchor matches uniquely. Otherwise — skip (fixture not committed
        for non-PN79 paths yet)."""
        from pathlib import Path
        m = _wiring()
        # Only run if fixture exists
        fixture = (Path(__file__).resolve().parents[4] / "tests" / "legacy" / "pristine_fixtures"
                   / "lora_model.py")
        if not fixture.is_file():
            pytest.skip("pristine_fixtures/lora_model.py not committed yet")
        src = fixture.read_text()
        assert src.count(m.ANCHOR_OLD) == 1, (
            "PN80 anchor must appear exactly once in pristine"
        )
        assert m.ANCHOR_NEW not in src, (
            "PN80 NEW must not appear in pristine (idempotency guard)"
        )
