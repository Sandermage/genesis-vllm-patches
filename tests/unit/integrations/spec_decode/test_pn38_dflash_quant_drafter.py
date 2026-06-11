# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for PN38 — DFlash drafter quantization (PR #40425
backport) after the 2026-06-11 Site B retirement.

Contract under test (preflight residual triage action plan §1b):

  * Site B TextPatch (``pN38_b_pass_quant_config``) is no longer
    registered in ``sub_patches``: pin 0.22.1rc1.dev259+g303916e93
    plumbs quant_config natively (pristine qwen3_dflash.py:228 init +
    :252 decoder-layer kwarg). Re-applying B would inject a duplicate
    ``quant_config=`` keyword argument → SyntaxError on model import.
  * The B anchor/replacement constants stay in the module for
    git-history reference (P78 Site A convention).
  * ``apply()`` carries an upstream-presence guard: BOTH native Site B
    lines must be present in the target, else a loud skip with an
    explicit reason — never a partial A/C/D application on a pin that
    lacks native quant_config plumbing.
  * A/C/D anchors are each unique (count == 1) against the pristine
    pin tree when it is available on this machine.

No torch / CUDA dependency required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PRISTINE_QWEN3_DFLASH = Path(
    "/private/tmp/candidate_pin_current/vllm/model_executor/models/qwen3_dflash.py"
)


def _import_patch():
    from sndr.engines.vllm.patches.spec_decode import (
        pn38_dflash_quant_drafter as p,
    )
    return p


# ─── Site B retirement: registration shape ─────────────────────────────


class TestSiteBRetired:
    def test_sub_patches_are_a_c_d_only(self, monkeypatch, tmp_path):
        p = _import_patch()
        target = tmp_path / "qwen3_dflash.py"
        target.write_text("# stand-in\n", encoding="utf-8")
        monkeypatch.setattr(p, "resolve_vllm_file", lambda rel: target)
        patcher = p._make_patcher()
        assert patcher is not None
        names = [sp.name for sp in patcher.sub_patches]
        assert "pN38_b_pass_quant_config" not in names
        assert names == [
            "pN38_a_qkv_proj_call",
            "pN38_c_conditional_fused_kv",
            "pN38_d_quant_fallback",
        ]

    def test_site_b_constants_kept_for_git_history(self):
        # P78 Site A convention: constants stay in the module as
        # commented history even though not registered in sub_patches.
        p = _import_patch()
        assert isinstance(p.PN38_B_ANCHOR, str) and p.PN38_B_ANCHOR
        assert isinstance(p.PN38_B_REPLACEMENT, str) and p.PN38_B_REPLACEMENT

    def test_native_guard_constants_exist(self):
        p = _import_patch()
        assert "get_draft_quant_config(vllm_config)" in p.PN38_NATIVE_QUANT_INIT
        assert "quant_config=self.quant_config," in p.PN38_NATIVE_QUANT_KWARG


# ─── Upstream-presence guard helper ─────────────────────────────────────


class TestNativeSiteBStatus:
    def test_both_lines_present_passes(self):
        p = _import_patch()
        content = p.PN38_NATIVE_QUANT_INIT + p.PN38_NATIVE_QUANT_KWARG
        ok, reason = p._native_site_b_status(content)
        assert ok is True
        assert reason

    def test_missing_init_line_fails(self):
        p = _import_patch()
        ok, reason = p._native_site_b_status(p.PN38_NATIVE_QUANT_KWARG)
        assert ok is False
        assert "get_draft_quant_config" in reason

    def test_missing_kwarg_line_fails(self):
        p = _import_patch()
        ok, reason = p._native_site_b_status(p.PN38_NATIVE_QUANT_INIT)
        assert ok is False
        assert "quant_config=self.quant_config" in reason

    def test_empty_content_fails_with_both_named(self):
        p = _import_patch()
        ok, reason = p._native_site_b_status("")
        assert ok is False
        assert "get_draft_quant_config" in reason
        assert "quant_config=self.quant_config" in reason


# ─── apply() end-to-end on synthetic target ─────────────────────────────


def _force_enabled(monkeypatch, p, tmp_path, target):
    import sndr.dispatcher as dispatcher

    monkeypatch.setattr(
        dispatcher, "should_apply", lambda patch_id: (True, "test-forced")
    )
    monkeypatch.setattr(
        dispatcher, "log_decision", lambda *a, **kw: None
    )
    monkeypatch.setattr(p, "vllm_install_root", lambda: str(tmp_path))
    monkeypatch.setattr(p, "resolve_vllm_file", lambda rel: target)


def _synthetic_target_content(p, *, with_native: bool) -> str:
    parts = ["# synthetic qwen3_dflash.py stand-in\n"]
    if with_native:
        parts += [p.PN38_NATIVE_QUANT_INIT, p.PN38_NATIVE_QUANT_KWARG]
    parts += [p.PN38_A_ANCHOR, p.PN38_C_ANCHOR, p.PN38_D_ANCHOR]
    return "".join(parts)


class TestApplyGuard:
    def test_apply_skips_loudly_when_native_lines_absent(
        self, monkeypatch, tmp_path
    ):
        p = _import_patch()
        target = tmp_path / "qwen3_dflash.py"
        target.write_text(
            _synthetic_target_content(p, with_native=False), encoding="utf-8"
        )
        _force_enabled(monkeypatch, p, tmp_path, target)
        status, reason = p.apply()
        assert status == "skipped"
        # Loud, explicit reason: names the guard and the missing lines.
        assert "upstream-presence guard" in reason
        assert "get_draft_quant_config" in reason
        # Target untouched — no partial A/C/D application.
        post = target.read_text(encoding="utf-8")
        assert "[Genesis PN38" not in post

    def test_apply_succeeds_with_native_lines(self, monkeypatch, tmp_path):
        p = _import_patch()
        target = tmp_path / "qwen3_dflash.py"
        target.write_text(
            _synthetic_target_content(p, with_native=True), encoding="utf-8"
        )
        _force_enabled(monkeypatch, p, tmp_path, target)
        status, reason = p.apply()
        assert status == "applied", reason
        post = target.read_text(encoding="utf-8")
        assert "[Genesis PN38 Site A]" in post
        assert "[Genesis PN38 Site C]" in post
        assert "[Genesis PN38 Site D]" in post
        # The retired Site B must NOT have injected a second kwarg:
        # exactly the one native occurrence survives.
        assert post.count("quant_config=self.quant_config,") == 1

    def test_second_apply_is_idempotent(self, monkeypatch, tmp_path):
        p = _import_patch()
        target = tmp_path / "qwen3_dflash.py"
        target.write_text(
            _synthetic_target_content(p, with_native=True), encoding="utf-8"
        )
        _force_enabled(monkeypatch, p, tmp_path, target)
        status, _ = p.apply()
        assert status == "applied"
        status2, reason2 = p.apply()
        assert status2 == "skipped"
        assert "already applied" in reason2


# ─── Pristine-tree evidence (skips when pin tree absent) ────────────────


@pytest.mark.skipif(
    not PRISTINE_QWEN3_DFLASH.is_file(),
    reason="pristine candidate pin tree not present on this machine",
)
class TestPristineAnchors:
    def _src(self) -> str:
        return PRISTINE_QWEN3_DFLASH.read_text(encoding="utf-8")

    def test_a_c_d_anchors_unique(self):
        p = _import_patch()
        src = self._src()
        for name in ("PN38_A_ANCHOR", "PN38_C_ANCHOR", "PN38_D_ANCHOR"):
            assert src.count(getattr(p, name)) == 1, name

    def test_retired_b_anchor_is_dead(self):
        # Upstream reordered the decoder-layer kwargs and passes
        # quant_config natively — the old B anchor must not match.
        p = _import_patch()
        assert self._src().count(p.PN38_B_ANCHOR) == 0

    def test_native_site_b_lines_present_and_unique(self):
        p = _import_patch()
        src = self._src()
        assert src.count(p.PN38_NATIVE_QUANT_INIT) == 1
        assert src.count(p.PN38_NATIVE_QUANT_KWARG) == 1
        ok, _ = p._native_site_b_status(src)
        assert ok is True
