# SPDX-License-Identifier: Apache-2.0
"""log_decision() must be idempotent by patch_id so the boot apply-matrix
holds exactly one record per patch — even when both the spec-driven loop
(_apply_spec_module) and a patch's own apply() log the same decision.

Regression guard for the deep-audit #3 finding: spec-only patches were
invisible to get_apply_matrix() because _apply_spec_module never recorded
their decision; adding that record must not double-count self-logging
patches.
"""
from __future__ import annotations

from sndr.dispatcher import decision as D
from sndr.dispatcher.reporting import get_apply_matrix


def _records_for(pid):
    return [d for d in get_apply_matrix() if d["patch_id"] == pid]


def test_repeated_log_decision_keeps_one_record_last_wins():
    pid = "PN252"  # a real registry id (so title/meta resolve)
    D.log_decision(pid, False, "first: gate skip")
    n_after_first = len(_records_for(pid))
    D.log_decision(pid, True, "second: applied")
    recs = _records_for(pid)
    assert n_after_first == 1
    assert len(recs) == 1, f"expected 1 record, got {len(recs)}"
    assert recs[0]["applied"] is True
    assert recs[0]["reason"] == "second: applied"


def test_distinct_patch_ids_each_get_a_record():
    D.log_decision("PN252", True, "a")
    D.log_decision("PN517", False, "b")
    assert len(_records_for("PN252")) == 1
    assert len(_records_for("PN517")) == 1


def test_apply_spec_module_records_decision_for_skipped_patch():
    """A disabled spec-only patch must still appear in the apply matrix as
    a SKIP — proving _apply_spec_module records the gate decision."""
    from sndr.apply import orchestrator
    from sndr.apply._state import PatchStats

    # PN517 is default_on=False + strict-opt-in: with no env flag it gates
    # to skip. Drive it through _apply_spec_module and assert it is recorded.
    from sndr.dispatcher.spec import iter_patch_specs

    pn517 = next((s for s in iter_patch_specs() if s.patch_id == "PN517"), None)
    assert pn517 is not None, "PN517 spec must exist"

    stats = PatchStats()
    status = orchestrator._apply_spec_module(pn517, stats)
    assert status == "skipped"
    recs = _records_for("PN517")
    assert len(recs) == 1
    assert recs[0]["applied"] is False
