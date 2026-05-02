# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P103 — FLA Cliff 2 chunked fwd_h+fwd_o orchestrator.

CPU-only smoke tests — verify dispatcher metadata, wiring import, env-gate
behaviour without actually invoking the GPU kernels. Numerical correctness
test (which requires GPU + triton) is in a separate gpu_test_p103.py.
"""
from __future__ import annotations

import unittest.mock as mock



def test_p103_in_dispatcher():
    """P103 must be registered in PATCH_REGISTRY with the expected schema."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P103" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["P103"]
    assert meta["env_flag"] == "GENESIS_ENABLE_P103"
    assert meta["default_on"] is False
    assert meta["category"] == "memory_hotfix"
    assert "Cliff 2" in meta["title"]
    assert "fwd_h" in meta["credit"] or "fwd_o" in meta["credit"]


def test_p103_wiring_module_imports():
    """The wiring module must import cleanly (no syntax errors)."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert hasattr(p103, "apply")
    assert hasattr(p103, "is_applied")
    assert hasattr(p103, "should_apply")


def test_p103_apply_register_in_apply_all():
    """P103 must have a wrapper function registered via @register_patch."""
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_103_fla_cliff2_chunked")


def test_p103_should_apply_off_by_default(monkeypatch):
    """Without GENESIS_ENABLE_P103=1, should_apply() must return False."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    monkeypatch.delenv("GENESIS_ENABLE_P103", raising=False)
    assert p103.should_apply() is False


def test_p103_should_apply_recognizes_truthy_env(monkeypatch):
    """should_apply() must accept all truthy env values."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    # Mock platform checks since this test runs CPU-only
    with mock.patch.object(p103, "is_nvidia_cuda", return_value=True), \
         mock.patch.object(p103, "is_sm_at_least", return_value=True):
        for v in ("1", "true", "yes", "on", "True", "YES"):
            monkeypatch.setenv("GENESIS_ENABLE_P103", v)
            assert p103.should_apply() is True, f"{v!r} should activate P103"
        for v in ("0", "", "off", "no", "False"):
            monkeypatch.setenv("GENESIS_ENABLE_P103", v)
            assert p103.should_apply() is False, f"{v!r} should NOT activate P103"


def test_p103_apply_fails_soft_when_module_missing(monkeypatch):
    """If FLA module is unavailable, apply() must return ('skipped', ...)
    not raise."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    with mock.patch.object(p103, "is_nvidia_cuda", return_value=True), \
         mock.patch.object(p103, "is_sm_at_least", return_value=True):
        # Simulate FLA missing
        import importlib
        original_import = importlib.import_module

        def _fail_for_chunk_module(name, *a, **kw):
            if name == p103._TARGET_MODULE:
                raise ImportError("simulated: chunk module not available")
            return original_import(name, *a, **kw)

        with mock.patch.object(importlib, "import_module",
                               side_effect=_fail_for_chunk_module):
            status, reason = p103.apply()
            assert status == "skipped"
            assert "FLA module" in reason or "not available" in reason


def test_p103_marker_attr_consistent():
    """The wrapper marker attribute name must match between apply and is_applied."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert p103._GENESIS_P103_MARKER_ATTR == "_genesis_p103_chunked_wrap"


def test_p103_max_t_env_default():
    """MAX_T defaults to 16384 when env unset, rounded down to FLA_CHUNK_SIZE multiple."""
    # We can't easily test the actual wrapper without FLA loaded, but we
    # can verify the default value is in the code (defensive sanity).
    import inspect
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    src = inspect.getsource(p103._make_chunked_wrapper)
    assert '"16384"' in src
    assert "GENESIS_FLA_FWD_H_MAX_T" in src
    # rounding to FLA_CHUNK_SIZE multiple
    assert "_MAX_T // fla_chunk_size" in src or "// fla_chunk_size) * fla_chunk_size" in src


def test_p103_kda_path_not_covered_documented():
    """The patch deliberately doesn't cover kda.py path; this should be
    documented in the wiring module docstring."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert "KDA" in p103.__doc__ or "kda" in p103.__doc__.lower()
