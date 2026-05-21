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
    from vllm.sndr_core.dispatcher import PATCH_REGISTRY
    assert "P103" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["P103"]
    assert meta["env_flag"] == "GENESIS_ENABLE_P103"
    assert meta["default_on"] is False
    assert meta["category"] == "memory_hotfix"
    assert "Cliff 2" in meta["title"]
    assert "fwd_h" in meta["credit"] or "fwd_o" in meta["credit"]


def test_p103_wiring_module_imports():
    """The wiring module must import cleanly (no syntax errors)."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    assert hasattr(p103, "apply")
    assert hasattr(p103, "is_applied")
    assert hasattr(p103, "should_apply")


def test_p103_apply_register_in_apply_all():
    """P103 must have a wrapper function registered via @register_patch."""
    from vllm.sndr_core.apply import apply_all
    assert hasattr(apply_all, "apply_patch_103_fla_cliff2_chunked")


def test_p103_should_apply_off_by_default(monkeypatch):
    """Without GENESIS_ENABLE_P103=1, should_apply() must return False."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    monkeypatch.delenv("GENESIS_ENABLE_P103", raising=False)
    assert p103.should_apply() is False


def test_p103_should_apply_recognizes_truthy_env(monkeypatch):
    """should_apply() must accept all truthy env values."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
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
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
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
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    assert p103._GENESIS_P103_MARKER_ATTR == "_genesis_p103_chunked_wrap"


def test_p103_max_t_env_default():
    """MAX_T defaults to 16384 when env unset, rounded down to FLA_CHUNK_SIZE multiple."""
    # We can't easily test the actual wrapper without FLA loaded, but we
    # can verify the default value is in the code (defensive sanity).
    import inspect
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    src = inspect.getsource(p103._make_chunked_wrapper)
    assert '"16384"' in src
    assert "GENESIS_FLA_FWD_H_MAX_T" in src
    # rounding to FLA_CHUNK_SIZE multiple
    assert "_MAX_T // fla_chunk_size" in src or "// fla_chunk_size) * fla_chunk_size" in src


def test_p103_kda_path_not_covered_documented():
    """The patch deliberately doesn't cover kda.py path; this should be
    documented in the wiring module docstring."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    assert "KDA" in p103.__doc__ or "kda" in p103.__doc__.lower()


# ─────────────────────────────────────────────────────────────────
# v7.69 — self-install at module-import time (club-3090#19 finding 2)
# ─────────────────────────────────────────────────────────────────


def test_p103_self_install_helper_exists():
    """v7.69: `_genesis_p103_install_at_import` must be exposed at module
    top so the text-patched chunk.py can `from ... import` it."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    assert hasattr(p103, "_genesis_p103_install_at_import"), (
        "v7.69 helper missing — text-patched chunk.py won't be able to "
        "import the install function"
    )


def test_p103_self_install_helper_signature():
    """Helper must accept a single dict-like (module globals)."""
    import inspect
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    sig = inspect.signature(p103._genesis_p103_install_at_import)
    params = list(sig.parameters.values())
    assert len(params) == 1, (
        f"helper must take exactly one arg (module globals), got "
        f"{len(params)}: {[p.name for p in params]}"
    )


def test_p103_self_install_returns_false_when_env_off(monkeypatch):
    """env-off path: helper must short-circuit cleanly (no side effects)."""
    monkeypatch.delenv("GENESIS_ENABLE_P103", raising=False)
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    fake_globals = {"chunk_gated_delta_rule_fwd": lambda *a, **kw: None}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is False
    # globals must be untouched
    assert fake_globals["chunk_gated_delta_rule_fwd"].__name__ != (
        "chunk_gated_delta_rule_fwd"
    ) or not hasattr(
        fake_globals["chunk_gated_delta_rule_fwd"],
        "_genesis_p103_chunked_wrap",
    )


def test_p103_self_install_no_op_if_already_wrapped(monkeypatch):
    """Idempotency: helper called twice on same module dict returns
    True both times, second call is no-op."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    # Pre-mark the function as already wrapped
    def fake_fn():
        pass
    fake_fn._genesis_p103_chunked_wrap = True
    fake_globals = {"chunk_gated_delta_rule_fwd": fake_fn}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is True
    # Same wrapper still in place
    assert fake_globals["chunk_gated_delta_rule_fwd"] is fake_fn


def test_p103_self_install_returns_false_on_missing_deps(monkeypatch):
    """Soft failure: missing closure dep must NOT raise — return False."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    # Provide chunk_gated_delta_rule_fwd but NO closure deps — helper
    # should fall through softly.
    fake_globals = {"chunk_gated_delta_rule_fwd": lambda *a, **kw: None}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is False, (
        "missing closure deps must produce False, not raise — chunk.py "
        "import must always succeed"
    )


def test_p103_self_install_succeeds_with_mock_chunk_globals(monkeypatch):
    """Full happy path: helper installs wrapper into a synthetic chunk.py
    globals dict that has all expected symbols."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    def orig_fwd(q, k, v, g, beta, scale, initial_state, output_final_state,
                 cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
        return None
    fake_globals = {
        "chunk_gated_delta_rule_fwd": orig_fwd,
        "chunk_local_cumsum": lambda **kw: None,
        "chunk_scaled_dot_kkt_fwd": lambda **kw: None,
        "solve_tril": lambda **kw: None,
        "recompute_w_u_fwd": lambda **kw: None,
        "chunk_gated_delta_rule_fwd_h": lambda **kw: None,
        "chunk_fwd_o": lambda **kw: None,
        "FLA_CHUNK_SIZE": 64,
        "SUPPRESS_LEVEL": 0,
    }
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is True

    new_fn = fake_globals["chunk_gated_delta_rule_fwd"]
    assert new_fn is not orig_fwd
    assert getattr(new_fn, "_genesis_p103_chunked_wrap", False) is True


def test_p103_text_patch_block_includes_env_check_and_helper_call():
    """The text-patch block appended to chunk.py must contain BOTH the
    env-flag check AND the call to _genesis_p103_install_at_import.

    Regression guard: if someone refactors the block string and forgets
    the env check, P103 would always fire even without opt-in.
    """
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    block = p103._P103_SELF_INSTALL_BLOCK
    assert "GENESIS_ENABLE_P103" in block, "env-flag check missing"
    assert "_genesis_p103_install_at_import" in block, "helper import missing"
    assert "globals()" in block, (
        "must call install with globals() — needs the chunk.py module dict"
    )
    assert "try:" in block and "except" in block, (
        "must wrap in try/except so chunk.py import survives any failure"
    )


def test_p103_text_patch_anchor_matches_real_chunk_py_pattern():
    """Anchor must match the EXACT end-of-file pattern of vllm's chunk.py
    (chunk_gated_delta_rule's final `return o, final_state` block).
    """
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    anchor = p103._P103_SELF_INSTALL_ANCHOR
    # Anchor must end with the high-level function's return
    assert "return o, final_state" in anchor
    # And include the autograd Function.apply call (so we don't anchor
    # on a generic `return o, final_state` elsewhere in the file)
    assert "ChunkGatedDeltaRuleFunction.apply" in anchor


def test_p103_self_install_text_patcher_builds_with_specific_drift_marker():
    """Drift markers on the chunk.py text-patch must NOT include generic
    prefixes that could collide with other patches in the same file."""
    import os
    import tempfile

    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    import vllm.sndr_core.detection.guards as guards

    with tempfile.TemporaryDirectory() as td:
        ops_dir = os.path.join(
            td, "model_executor", "layers", "fla", "ops"
        )
        os.makedirs(ops_dir)
        with open(os.path.join(ops_dir, "chunk.py"), "w") as f:
            f.write("# placeholder\n")

        orig = guards.vllm_install_root
        guards.vllm_install_root = lambda: td
        try:
            patcher = p103._make_self_install_text_patcher()
        finally:
            guards.vllm_install_root = orig

    assert patcher is not None

    # Drift markers must be specific (no generic '[Genesis P103' that
    # would collide with future patches' insertions in chunk.py).
    for m in patcher.upstream_drift_markers:
        # Must include version/feature qualifier
        assert "v7.69" in m or "self-install" in m, (
            f"drift marker {m!r} too generic — risk of collision with "
            f"sibling patches"
        )


def test_p103_apply_attempts_text_patch_first(monkeypatch):
    """apply() in v7.69 must call _make_self_install_text_patcher BEFORE
    the legacy setattr step — text-patch is the durable mechanism that
    survives `exec vllm serve` + worker spawn."""
    import inspect
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    src = inspect.getsource(p103.apply)
    # text-patch step must appear before the importlib.import_module call
    text_patch_pos = src.find("_make_self_install_text_patcher")
    setattr_pos = src.find("setattr(chunk_mod, _FN_NAME, wrapper)")
    assert text_patch_pos > 0, "text-patch step missing"
    assert setattr_pos > 0, "setattr step missing"
    assert text_patch_pos < setattr_pos, (
        "text-patch step must run BEFORE setattr step (text-patch is the "
        "durable mechanism; setattr is defense-in-depth for current process)"
    )


def test_p103_module_docstring_explains_v7_69_install_model():
    """v7.69 install-model rationale must be documented at module top."""
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103
    doc = p103.__doc__ or ""
    assert "exec vllm serve" in doc, (
        "module docstring must explain why v7.69 was needed (entrypoint "
        "exec pattern)"
    )
    assert "self-install" in doc.lower() or "module-import" in doc.lower()


# ─────────────────────────────────────────────────────────────────
# P2.4b — ABI-forward-compat (dev371 `core_attn_out` kwarg drift)
# ─────────────────────────────────────────────────────────────────
#
# Q35-TQ HALT 2026-05-21 traced to a closure built by
# `_make_chunked_wrapper` whose explicit signature pre-dated the
# dev338→dev371 addition of `core_attn_out` to
# `chunk_gated_delta_rule_fwd`. Upstream callers on dev371 pass
# `core_attn_out=<tensor>` → wrapper raised TypeError. The fix below
# accepts arbitrary `**kwargs` and bypasses the chunked rewrite when
# the caller supplies an output buffer (P103 has no semantics for
# caller-owned outputs).
#
# These tests build the wrapper directly via `_make_chunked_wrapper`
# with lightweight fakes (no CUDA, no torch tensors). They exercise
# the closure behavior, not the full apply() flow.


class _FakeShape(tuple):
    """Tuple-with-int-subscript so `q.shape[1]` works on a non-tensor."""

    def __new__(cls, *vals):
        return super().__new__(cls, vals)


class _FakeQ:
    """Minimal q-stub: needs `.shape` (tuple) and `__getitem__` for slicing.

    The wrapper reads `q.shape[1]` to decide hot-path vs chunked path; it
    does NOT actually slice when the hot-path returns. So __getitem__ can
    raise — we never reach it in the tests below.
    """

    def __init__(self, t: int):
        self.shape = _FakeShape(1, t, 1, 1)


def _build_wrapper(original_fwd, max_t_env: str = "16384"):
    """Build a P103 chunked_fwd closure with fake deps + a real original_fwd.

    The chunked-loop branch is never reached by these tests (hot-path
    returns first, or core_attn_out bypass returns first), so the inner
    closure deps can be no-op lambdas.
    """
    import os
    import unittest.mock as _mock
    from vllm.sndr_core.integrations.attention.gdn import p103_fla_cliff2_chunked as p103

    _noop = lambda **kw: None  # noqa: E731

    with _mock.patch.dict(os.environ, {"GENESIS_FLA_FWD_H_MAX_T": max_t_env}):
        wrapper = p103._make_chunked_wrapper(
            original_fwd=original_fwd,
            chunk_local_cumsum=_noop,
            chunk_scaled_dot_kkt_fwd=_noop,
            solve_tril=_noop,
            recompute_w_u_fwd=_noop,
            chunk_gated_delta_rule_fwd_h=_noop,
            chunk_fwd_o_callable=_noop,
            fla_chunk_size=64,
            suppress_level=0,
        )
    return wrapper


def test_p103_wrapper_accepts_core_attn_out_none_no_error():
    """dev371 callers pass `core_attn_out=None` on the default path.

    The wrapper must NOT raise TypeError on the new kwarg, even when
    its value is None. Hot-path (T=1) fallthrough preserves the old
    11-positional contract to `original_fwd`.
    """
    recorded = {}

    def fake_original(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "ORIG_RESULT"

    wrapper = _build_wrapper(fake_original)
    q = _FakeQ(t=1)  # T=1 → hot-path fallthrough
    result = wrapper(
        q=q, k=None, v=None, g=None, beta=None, scale=1.0,
        initial_state=None, output_final_state=False,
        cu_seqlens=None,
        core_attn_out=None,
    )
    assert result == "ORIG_RESULT"
    # core_attn_out=None must reach original_fwd via **kwargs so upstream
    # can either default-allocate or skip allocation per its own logic.
    assert recorded["kwargs"].get("core_attn_out") is None
    assert "core_attn_out" in recorded["kwargs"]


def test_p103_wrapper_bypasses_when_core_attn_out_tensor_provided():
    """When the caller supplies a non-None `core_attn_out` buffer, the
    wrapper must call `original_fwd` directly and forward the kwarg —
    even if shape would normally trigger the chunked-loop branch.

    P103 does not own caller-owned output buffer semantics.
    """
    recorded = {}
    sentinel_buffer = object()  # not None — looks like a preallocated tensor

    def fake_original(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "ORIG_RESULT"

    wrapper = _build_wrapper(fake_original, max_t_env="64")  # small MAX_T
    # T=128 > MAX_T(64) AND cu_seqlens=None → would normally enter chunked
    # branch. But core_attn_out!=None must bypass.
    q = _FakeQ(t=128)
    result = wrapper(
        q=q, k="K", v="V", g="G", beta="B", scale=2.5,
        initial_state="S", output_final_state=True,
        cu_seqlens=None,
        core_attn_out=sentinel_buffer,
    )
    assert result == "ORIG_RESULT"
    assert recorded["kwargs"].get("core_attn_out") is sentinel_buffer
    # And the positional arg layout reaches original_fwd unchanged
    args = recorded["args"]
    assert args[0] is q
    assert args[1] == "K"
    assert args[5] == 2.5  # scale


def test_p103_wrapper_forwards_unknown_future_kwarg_on_hot_path():
    """Forward-compat: an unrelated kwarg added by a future upstream
    must traverse the hot fallthrough without TypeError and reach
    `original_fwd` via **kwargs."""
    recorded = {}

    def fake_original(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return None

    wrapper = _build_wrapper(fake_original)
    q = _FakeQ(t=1)  # hot-path
    wrapper(
        q=q, k=None, v=None, g=None, beta=None, scale=1.0,
        initial_state=None, output_final_state=False,
        cu_seqlens=None,
        future_param_2027="abc",
    )
    assert recorded["kwargs"].get("future_param_2027") == "abc"


def test_p103_wrapper_preserves_old_signature_call():
    """Backward-compat: a dev338-style caller (no kwargs beyond the
    explicit ones) must continue to work identically. Hot-path
    fallthrough forwards positional args."""
    recorded = {}

    def fake_original(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "OK"

    wrapper = _build_wrapper(fake_original)
    q = _FakeQ(t=1)
    result = wrapper(
        q=q, k="K", v="V", g="G", beta="B", scale=0.125,
        initial_state="S", output_final_state=True,
        cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    )
    assert result == "OK"
    # No extraneous kwargs leaked into original_fwd
    assert recorded["kwargs"] == {}


def test_p103_wrapper_signature_accepts_var_keyword():
    """The closure signature must include **kwargs (or equivalent).

    Regression guard: catches a future refactor that re-introduces an
    explicit-only signature and reopens the dev371 ABI gap.
    """
    import inspect

    recorded = {}

    def fake_original(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return None

    wrapper = _build_wrapper(fake_original)
    sig = inspect.signature(wrapper)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    assert has_var_kw, (
        f"wrapper signature must accept **kwargs for ABI-forward-compat; "
        f"got {sig}"
    )
