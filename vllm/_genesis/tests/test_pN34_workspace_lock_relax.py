# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN34 — WorkspaceManager runtime lock relaxation.

Companion to PN33 — same bug class (workspace under-counted) but on
the runtime decode path. Direct port of noonghunna's club-3090 setup-
time sidecar patch_workspace_lock_disable.py promoted into Genesis.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Port credit: noonghunna club-3090 (commit 2b5ab4d).
"""
from __future__ import annotations



def test_pn34_wiring_imports():
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN34_MARKER")
    assert hasattr(mod, "PN34_ANCHOR")
    assert hasattr(mod, "PN34_REPLACEMENT")


def test_pn34_dispatcher_registry():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PN34" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN34"]
    assert e["env_flag"] == "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX"
    assert e["default_on"] is False  # opt-in (relaxes strict assertion)
    assert e["requires_patches"] == ["PN33"]


def test_pn34_anchor_targets_strict_assertion():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        PN34_ANCHOR,
    )
    assert "if self._locked:" in PN34_ANCHOR
    assert "raise AssertionError" in PN34_ANCHOR
    assert "Workspace is locked" in PN34_ANCHOR
    assert "Workspace growth is not allowed after locking" in PN34_ANCHOR


def test_pn34_replacement_relaxes_to_warn_plus_grow():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        PN34_REPLACEMENT,
    )
    # Relaxation: still detects locked state but warns instead of asserts
    assert "if self._locked:" in PN34_REPLACEMENT
    # No AssertionError raise in replacement (regression guard —
    # would re-introduce the strict failure mode)
    assert "raise AssertionError" not in PN34_REPLACEMENT
    # One-shot warning at WARNING level (visible in default log levels)
    assert "logger.warning" in PN34_REPLACEMENT
    # Module-level flag prevents log spam
    assert "_GENESIS_PN34_WORKSPACE_LOCK_WARNED" in PN34_REPLACEMENT
    assert "global _GENESIS_PN34_WORKSPACE_LOCK_WARNED" in PN34_REPLACEMENT
    # Defensive NameError guard for first call
    assert "NameError" in PN34_REPLACEMENT


def test_pn34_skips_when_env_off_and_config_neutral(monkeypatch):
    """Without env flag AND with config-detect returning neutral, PN34 skips."""
    monkeypatch.delenv("GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX", raising=False)
    monkeypatch.delenv("GENESIS_DISABLE_PN34", raising=False)
    # Override config-detect to return neutral (non-TQ config)
    import vllm._genesis.config_detect as cd
    monkeypatch.setattr(cd, "recommend", lambda pid: ("neutral", "no TQ KV"))
    from vllm._genesis import dispatcher
    decision, reason = dispatcher.should_apply("PN34")
    assert decision is False
    assert "opt-in" in reason.lower()


def test_pn34_register_in_apply_all():
    from vllm._genesis.patches.apply_all import (
        PATCH_REGISTRY as APPLY_REGISTRY,
    )
    names = [name for name, _ in APPLY_REGISTRY]
    pn34 = [n for n in names if "PN34" in n]
    assert len(pn34) == 1, (
        f"PN34 not registered in apply_all, names: {names[:5]}"
    )


def test_pn34_marker_unique():
    from vllm._genesis.wiring.perf_hotfix.patch_N34_workspace_lock_runtime_relax import (
        GENESIS_PN34_MARKER,
    )
    assert "PN34" in GENESIS_PN34_MARKER
    assert "v7.68" in GENESIS_PN34_MARKER


def test_pn34_documents_pn33_companion_relationship():
    """Source must document that PN34 is companion to PN33 — both
    address the same root cause (workspace under-counted) but at
    different layers (boot vs runtime decode)."""
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "PN33" in src
    assert "companion" in src.lower()
    assert "boot" in src.lower() or "_dummy_sampler_run" in src
    assert "runtime decode" in src.lower() or "_decode_attention" in src


def test_pn34_documents_noonghunna_credit():
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "noonghunna" in src
    assert "club-3090" in src
    assert "patch_workspace_lock_disable" in src


def test_pn34_documents_upstream_retirement_path():
    """Source must reference the upstream PR that obsoletes PN34."""
    import inspect
    from vllm._genesis.wiring.perf_hotfix import (
        patch_N34_workspace_lock_runtime_relax as mod,
    )
    src = inspect.getsource(mod)
    assert "40706" in src  # vllm#40706 = TQ scratch dedup


# ── Auto-enable via config-detect (new behavior) ─────────────────────────


def test_pn34_config_detect_recommends_apply_for_tq_spec(monkeypatch):
    """config_detect recommends 'apply' when TQ KV + spec-decode + WorkspaceManager."""
    from vllm._genesis import config_detect

    fake_profile = {
        "kv_cache_dtype": "turboquant_4bit_nc",
        "spec_decode_enabled": True,
        "workspace_manager_present": True,
        "max_num_seqs": 4,
        "pr40798_active": False,
        "pr40792_active": False,
        "pr40384_active": False,
        "pr40074_active": False,
        "cudagraph_capture_active": True,
    }
    monkeypatch.setattr(config_detect, "_CACHED_PROFILE", None)

    with monkeypatch.context() as mp:
        mp.setattr(config_detect, "_try_get_vllm_config", lambda: object())
        mp.setattr(config_detect, "_probe_scheduler", lambda cfg: {})
        mp.setattr(config_detect, "_probe_spec_decode", lambda cfg: {
            "spec_decode_enabled": True,
        })
        mp.setattr(config_detect, "_probe_compilation", lambda cfg: {
            "cudagraph_capture_active": True,
        })
        mp.setattr(config_detect, "_probe_cache", lambda cfg: {
            "kv_cache_dtype": "turboquant_4bit_nc",
        })
        mp.setattr(config_detect, "_probe_pr40384_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40074_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40798_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40792_active", lambda: False)
        mp.setattr(config_detect, "_probe_workspace_manager_present", lambda: True)

        profile = config_detect.get_runtime_profile()

    verdict, reason = config_detect.recommend("PN34")
    assert verdict == "apply", f"Expected 'apply', got {verdict!r}: {reason}"
    assert "TQ KV" in reason or "turboquant" in reason.lower() or "spec-decode" in reason


def test_pn34_config_detect_skips_without_spec_decode(monkeypatch):
    """config_detect skips PN34 when TQ KV but no spec-decode."""
    from vllm._genesis import config_detect

    monkeypatch.setattr(config_detect, "_CACHED_PROFILE", None)

    with monkeypatch.context() as mp:
        mp.setattr(config_detect, "_try_get_vllm_config", lambda: object())
        mp.setattr(config_detect, "_probe_scheduler", lambda cfg: {})
        mp.setattr(config_detect, "_probe_spec_decode", lambda cfg: {
            "spec_decode_enabled": False,
        })
        mp.setattr(config_detect, "_probe_compilation", lambda cfg: {
            "cudagraph_capture_active": False,
        })
        mp.setattr(config_detect, "_probe_cache", lambda cfg: {
            "kv_cache_dtype": "turboquant_4bit_nc",
        })
        mp.setattr(config_detect, "_probe_pr40384_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40074_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40798_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40792_active", lambda: False)
        mp.setattr(config_detect, "_probe_workspace_manager_present", lambda: True)

        config_detect.get_runtime_profile()

    verdict, _ = config_detect.recommend("PN34")
    assert verdict == "skip", f"Expected 'skip' without spec-decode, got {verdict!r}"


def test_pn34_config_detect_skips_without_tq(monkeypatch):
    """config_detect skips PN34 when spec-decode but no TQ KV."""
    from vllm._genesis import config_detect

    monkeypatch.setattr(config_detect, "_CACHED_PROFILE", None)

    with monkeypatch.context() as mp:
        mp.setattr(config_detect, "_try_get_vllm_config", lambda: object())
        mp.setattr(config_detect, "_probe_scheduler", lambda cfg: {})
        mp.setattr(config_detect, "_probe_spec_decode", lambda cfg: {
            "spec_decode_enabled": True,
        })
        mp.setattr(config_detect, "_probe_compilation", lambda cfg: {
            "cudagraph_capture_active": True,
        })
        mp.setattr(config_detect, "_probe_cache", lambda cfg: {
            "kv_cache_dtype": "bfloat16",
        })
        mp.setattr(config_detect, "_probe_pr40384_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40074_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40798_active", lambda: False)
        mp.setattr(config_detect, "_probe_pr40792_active", lambda: False)
        mp.setattr(config_detect, "_probe_workspace_manager_present", lambda: True)

        config_detect.get_runtime_profile()

    verdict, _ = config_detect.recommend("PN34")
    assert verdict == "skip", f"Expected 'skip' without TQ, got {verdict!r}"


def test_pn34_disable_env_suppresses_auto_enable(monkeypatch):
    """GENESIS_DISABLE_PN34=1 suppresses config-detect auto-enable."""
    monkeypatch.delenv("GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX", raising=False)
    monkeypatch.setenv("GENESIS_DISABLE_PN34", "1")

    from vllm._genesis import dispatcher
    decision, reason = dispatcher.should_apply("PN34")
    assert decision is False
    assert "PN34" in reason


def test_pn34_tq_kv_argv_probe_detects_turboquant(monkeypatch):
    """_probe_tq_kv_from_argv reads --kv-cache-dtype from sys.argv."""
    import sys
    monkeypatch.setattr(sys, "argv", [
        "apply_all",
        "--kv-cache-dtype", "turboquant_4bit_nc",
        "--speculative-config", '{"method":"qwen3_next_mtp","num_speculative_tokens":4}',
    ])
    from vllm._genesis.config_detect import _probe_tq_kv_from_argv
    result = _probe_tq_kv_from_argv()
    assert result == "turboquant_4bit_nc"


def test_pn34_tq_kv_argv_probe_returns_none_for_bfloat16(monkeypatch, tmp_path):
    """_probe_tq_kv_from_argv returns None when no --kv-cache-dtype in argv or proc."""
    import sys, builtins
    monkeypatch.setattr(sys, "argv", ["apply_all", "--dtype", "bfloat16"])
    # Stub out /proc/1/cmdline so it doesn't bleed in from the container launch args
    original_open = builtins.open
    def patched_open(path, *args, **kwargs):
        if str(path) == "/proc/1/cmdline":
            import io
            return io.BytesIO(b"python3\x00apply_all\x00--dtype\x00bfloat16\x00")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", patched_open)
    from vllm._genesis.config_detect import _probe_tq_kv_from_argv
    result = _probe_tq_kv_from_argv()
    assert result is None


def test_pn34_auto_enable_via_dispatcher_when_config_recommends(monkeypatch):
    """dispatcher.should_apply auto-enables PN34 when config_detect returns 'apply'."""
    monkeypatch.delenv("GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX", raising=False)
    monkeypatch.delenv("GENESIS_DISABLE_PN34", raising=False)

    from vllm._genesis import dispatcher
    import vllm._genesis.config_detect as cd

    monkeypatch.setattr(cd, "recommend", lambda pid: ("apply", "TQ KV + spec-decode") if pid == "PN34" else ("neutral", ""))

    decision, reason = dispatcher.should_apply("PN34")
    assert decision is True
    assert "config-detect auto-enable" in reason
