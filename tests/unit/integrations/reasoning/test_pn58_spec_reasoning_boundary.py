# SPDX-License-Identifier: Apache-2.0
"""TDD for PN58 — spec-decode reasoning boundary validation (vllm#40962)."""
from __future__ import annotations

import pytest


def _wiring():
    from sndr.engines.vllm.patches.reasoning import pn58_spec_reasoning_boundary as M
    return M


# Verbatim tail of pristine envs.py (pin 0.22.1rc1.dev259+g303916e93,
# lines 1997-2013 of /private/tmp/candidate_pin_current/vllm/envs.py,
# extracted 2026-06-11 for the PN58 envs re-anchor). Documents WHY the
# old anchor died: the LoRA dual-stream comment grew from 1 line to 3,
# and 3 new entries (VLLM_USE_SPINLOOP_EXT, VLLM_GPU_NIC_PCIE_MAPPING,
# VLLM_NIC_SELECTION_VARS) landed between it and the closing brace.
PRISTINE_ENVS_TAIL = (
    '    # Whether to enable dual cuda streams for LoRA computation\n'
    '    # (used by both BaseLinearLayerWithLoRA and FusedMoEWithLoRA to\n'
    '    # overlap the base layer compute with the LoRA fast path).\n'
    '    "VLLM_LORA_ENABLE_DUAL_STREAM": lambda: bool(\n'
    '        int(os.getenv("VLLM_LORA_ENABLE_DUAL_STREAM", "0"))\n'
    '    ),\n'
    '    # If set to 1, use Python spinloop extension to poll in a more efficient\n'
    '    # way when using the mp backend.\n'
    '    "VLLM_USE_SPINLOOP_EXT": lambda: bool(int(os.getenv("VLLM_USE_SPINLOOP_EXT", "0"))),\n'
    '    # Comma-separated GPU_BDF=NIC_BDF pairs for RDMA NIC selection.\n'
    '    # Must be set together with VLLM_NIC_SELECTION_VARS.\n'
    '    "VLLM_GPU_NIC_PCIE_MAPPING": lambda: os.getenv("VLLM_GPU_NIC_PCIE_MAPPING", ""),\n'
    '    # Comma-separated list of env vars to set from the GPU-NIC mapping.\n'
    '    # Each entry is VAR_NAME or VAR_NAME:<suffix> (suffix appended to\n'
    '    # RDMA device name). Must be set together with VLLM_GPU_NIC_PCIE_MAPPING.\n'
    '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n'
    '}\n'
)


def test_anchor_envs_matches_pristine_tail_exactly_once():
    """2026-06-11 re-anchor: ENVS_OLD must match the pristine dict tail.

    A dead envs anchor blocks ALL 5 PN58 files — the apply path uses
    MultiFilePatchTransaction (validate-all-then-write-all), so phase-1
    dry-run failure on envs.py vetoes the whole transaction atomically.
    """
    M = _wiring()
    assert PRISTINE_ENVS_TAIL.count(M.ENVS_OLD) == 1, (
        "ENVS_OLD does not match the pristine envs.py tail exactly once — "
        "re-anchor against the current pin (see PRISTINE_ENVS_TAIL note)"
    )


def test_anchor_envs_is_comment_churn_resistant():
    """The previous anchor died because an upstream COMMENT grew. The
    minimal tail anchor must contain no comment lines at all."""
    M = _wiring()
    for line in M.ENVS_OLD.splitlines():
        assert not line.lstrip().startswith("#"), (
            f"ENVS_OLD contains comment line {line!r} — comment churn "
            "killed the previous anchor; keep the anchor comment-free"
        )
    assert M.ENVS_OLD.endswith("\n}")


def test_anchor_envs_new_preserves_anchor_and_adds_flag():
    M = _wiring()
    # NEW must keep the anchor's own content (entry line + closing brace)
    # so the rest of the dict is untouched, inserting only between them.
    anchor_head = M.ENVS_OLD[: -len("}")]
    assert M.ENVS_NEW.startswith(anchor_head)
    assert M.ENVS_NEW.endswith("}")
    assert "VLLM_SPEC_REASONING_BOUNDARY_VALIDATION" in M.ENVS_NEW
    # Replacement must NOT appear in pristine (idempotency false-fire guard).
    assert M.ENVS_NEW not in PRISTINE_ENVS_TAIL


def test_anchor_abs_parser_targets_extract_content_ids():
    M = _wiring()
    assert "@abstractmethod" in M.ABS_PARSER_OLD
    assert "extract_content_ids" in M.ABS_PARSER_OLD
    assert "find_reasoning_end_index" in M.ABS_PARSER_NEW
    assert "may_have_reasoning_end_in_delta" in M.ABS_PARSER_NEW


def test_anchor_basic_parser_overrides_methods():
    M = _wiring()
    assert "is_reasoning_end_streaming" in M.BASIC_PARSER_OLD
    assert "delta_ids.index(end_token_id)" in M.BASIC_PARSER_NEW


def test_anchor_struct_out_inserts_helper():
    M = _wiring()
    assert "class StructuredOutputManager:" in M.STRUCT_OUT_OLD
    assert "validate_spec_tokens_with_reasoning_boundary" in M.STRUCT_OUT_NEW


def test_anchor_sched_replaces_should_advance_block():
    M = _wiring()
    assert "should_advance(request)" in M.SCHED_VALIDATE_OLD
    assert "should_advance" in M.SCHED_VALIDATE_NEW
    assert "_pn58_advanced_with_boundary" in M.SCHED_VALIDATE_NEW
    assert "validate_spec_tokens_with_reasoning_boundary" in M.SCHED_VALIDATE_NEW


def test_replacements_carry_pn58_marker():
    M = _wiring()
    for name, new in [
        ("ENVS_NEW", M.ENVS_NEW),
        ("ABS_PARSER_NEW", M.ABS_PARSER_NEW),
        ("BASIC_PARSER_NEW", M.BASIC_PARSER_NEW),
        ("STRUCT_OUT_NEW", M.STRUCT_OUT_NEW),
        ("SCHED_VALIDATE_NEW", M.SCHED_VALIDATE_NEW),
        ("SCHED_IMPORT_NEW", M.SCHED_IMPORT_NEW),
    ]:
        assert "PN58" in new, f"{name} missing PN58 marker"


def test_idempotent_envs(tmp_path):
    from sndr.kernel.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "envs.py"
    target.write_text("# header\n" + M.ENVS_OLD + "\n")
    patcher = TextPatcher(
        patch_name="PN58 envs test",
        target_file=str(target),
        marker=M.GENESIS_PN58_MARKER + " (envs)",
        sub_patches=[TextPatch(name="pn58_envs",
                                anchor=M.ENVS_OLD,
                                replacement=M.ENVS_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "VLLM_SPEC_REASONING_BOUNDARY_VALIDATION" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_idempotent_sched_validate_block(tmp_path):
    from sndr.kernel.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "scheduler.py"
    target.write_text("# header\n" + M.SCHED_VALIDATE_OLD + "\n# tail\n")
    patcher = TextPatcher(
        patch_name="PN58 sched test",
        target_file=str(target),
        marker=M.GENESIS_PN58_MARKER + " (sched)",
        sub_patches=[TextPatch(name="pn58_sched_validate",
                                anchor=M.SCHED_VALIDATE_OLD,
                                replacement=M.SCHED_VALIDATE_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_mutex_with_p62_skips_when_p62_active(monkeypatch):
    """Apply check must SKIP cleanly when P62 active."""
    monkeypatch.setenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", "1")
    monkeypatch.setenv("GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING", "1")
    from sndr.engines.vllm.patches.reasoning import pn58_spec_reasoning_boundary as M
    status, reason = M.apply()
    assert status == "skipped"
    assert "P62" in reason
    assert "MUTUAL" in reason.upper() or "mutual" in reason.lower() or "exclusive" in reason.lower()


def test_env_flag_default_off(monkeypatch):
    from sndr.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", raising=False)
    decision, _ = should_apply("PN58")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from sndr.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY", "1")
    decision, _ = should_apply("PN58")
    assert decision is True


def test_registry_entry_complete():
    from sndr.dispatcher import PATCH_REGISTRY
    assert "PN58" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN58"]
    assert meta["upstream_pr"] == 40962
    assert "P62" in meta.get("conflicts_with", [])


def test_apply_all_registers_pn58():
    from sndr.apply import apply_all
    assert hasattr(apply_all, "apply_patch_N58_spec_reasoning_boundary")
