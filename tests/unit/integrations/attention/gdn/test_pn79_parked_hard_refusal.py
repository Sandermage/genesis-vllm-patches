# SPDX-License-Identifier: Apache-2.0
"""PN79 is PARKED after a reproduced PROD CUDA illegal-memory-access on the
first 8K chunked prefill. Its 18 required anchors still apply cleanly, so a
single GENESIS_ENABLE_PN79_INPLACE_SSM_STATE=1 in a launcher would silently
re-introduce the crash. apply() must hard-refuse unless an explicit override
naming the PROD IMA is set (deep-audit 2026-06-14 #4).
"""
from __future__ import annotations

import sndr.engines.vllm.patches.attention.gdn.pn79_inplace_ssm_state as m
from sndr.dispatcher.decision import should_apply


def test_enabling_pn79_is_reached_by_dispatcher(monkeypatch):
    # Sanity: with the env flag set, the dispatcher would otherwise APPLY —
    # so the hard-refusal below is what actually blocks it, not the gate.
    monkeypatch.setenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", "1")
    monkeypatch.delenv("GENESIS_LEGACY_DEFAULT_ON", raising=False)
    decision, _ = should_apply("PN79")
    assert decision is True


def test_enabled_without_force_is_refused(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", "1")
    monkeypatch.delenv(
        "GENESIS_PN79_FORCE_REENABLE_DESPITE_PROD_IMA", raising=False
    )
    status, reason = m.apply()
    assert status == "skipped"
    assert "PARKED" in reason
    assert "PROD IMA" in reason
    assert "GENESIS_PN79_FORCE_REENABLE_DESPITE_PROD_IMA" in reason


def test_force_override_passes_the_parked_gate(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", "1")
    monkeypatch.setenv(
        "GENESIS_PN79_FORCE_REENABLE_DESPITE_PROD_IMA", "1"
    )
    status, reason = m.apply()
    # On a host without a live vllm install the run proceeds PAST the parked
    # gate and skips later (vllm install not discoverable) — proving the
    # override let it through rather than the parked gate stopping it.
    assert status in ("skipped", "applied", "failed")
    assert "PARKED" not in reason
