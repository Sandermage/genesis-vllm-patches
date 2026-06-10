# SPDX-License-Identifier: Apache-2.0
"""TDD for PN79 — in-place SSM state for GDN chunk prefill (vllm#41824 backport).

State 2026-05-07: Sub-1 (chunk.py) + Sub-3 (gdn_linear_attn.py) anchors are
fully implemented. Sub-2 (chunk_delta_h.py Triton kernel) is intentionally
deferred — `_make_chunk_delta_h_patcher()` returns None which causes
MultiFilePatchTransaction to atomic-skip until next dedicated session.

These tests verify:
  1. Module imports cleanly + apply contract
  2. Sub-1 anchors (4): 1A drop input_guard, 1B fwd sig, 1C internal call,
     1D forward rewrite
  3. Sub-3 anchor (1): 3C gather/scatter elimination
  4. Drift markers configured per-patcher and absent from OLD anchors
  5. Per-anchor TextPatcher idempotency on tmp_path fixtures (Sub-1 anchors
     applied serially against a composite fixture that mirrors pristine
     chunk.py contents from vllm 0.20.2rc1.dev9+g01d4d1ad3)
  6. PATCH_REGISTRY entry has conflicts_with: [PN59, PN54]
  7. Apply returns 'skipped' both when env disabled AND when env enabled
     (because Sub-2 patcher returns None — atomic-skip safety net)
"""
from __future__ import annotations

import pytest


def _wiring():
    from sndr.engines.vllm.patches.attention.gdn import pn79_inplace_ssm_state as M
    return M


# ─────────────────────────────────────────────────────────────────────────
# 1. Module + apply contract
# ─────────────────────────────────────────────────────────────────────────


class TestModuleContract:

    def test_module_importable(self):
        m = _wiring()
        assert m is not None

    def test_apply_function_exists(self):
        m = _wiring()
        assert callable(m.apply)

    def test_marker_defined(self):
        m = _wiring()
        assert "PN79" in m.GENESIS_PN79_MARKER
        assert "41824" in m.GENESIS_PN79_MARKER


# ─────────────────────────────────────────────────────────────────────────
# 2. Sub-1 anchors (chunk.py)
# ─────────────────────────────────────────────────────────────────────────


class TestSub1Anchors:
    """4 sub-anchors: 1A import drop, 1B fwd sig, 1C internal call, 1D forward."""

    def test_1A_drops_input_guard_import(self):
        m = _wiring()
        assert "input_guard" in m.ANCHOR_1A_IMPORT_OLD
        assert "input_guard" not in m.ANCHOR_1A_IMPORT_NEW
        # Must keep the other two utils imports
        assert "FLA_CHUNK_SIZE" in m.ANCHOR_1A_IMPORT_NEW
        assert "SUPPRESS_LEVEL" in m.ANCHOR_1A_IMPORT_NEW

    def test_1B_fwd_signature_adds_ssm_state_indices(self):
        m = _wiring()
        assert m.ANCHOR_1B_FWD_SIG_OLD != m.ANCHOR_1B_FWD_SIG_NEW
        # OLD must NOT have the new params (false-fire guard)
        assert "ssm_state_indices" not in m.ANCHOR_1B_FWD_SIG_OLD
        assert "has_initial_state" not in m.ANCHOR_1B_FWD_SIG_OLD
        # NEW must have both new params
        assert "ssm_state_indices: torch.Tensor | None = None" in m.ANCHOR_1B_FWD_SIG_NEW
        assert "has_initial_state: torch.Tensor | None = None" in m.ANCHOR_1B_FWD_SIG_NEW
        # Both must end in `):` and continue into `g = chunk_local_cumsum(`
        assert "g = chunk_local_cumsum(" in m.ANCHOR_1B_FWD_SIG_OLD
        assert "g = chunk_local_cumsum(" in m.ANCHOR_1B_FWD_SIG_NEW

    def test_1C_internal_call_passes_new_kwargs(self):
        m = _wiring()
        assert m.ANCHOR_1C_FWD_INTERNAL_OLD != m.ANCHOR_1C_FWD_INTERNAL_NEW
        # NEW passes ssm_state_indices and has_initial_state
        assert "ssm_state_indices=ssm_state_indices" in m.ANCHOR_1C_FWD_INTERNAL_NEW
        assert "has_initial_state=has_initial_state" in m.ANCHOR_1C_FWD_INTERNAL_NEW
        # OLD must NOT (false-fire guard)
        assert "ssm_state_indices=" not in m.ANCHOR_1C_FWD_INTERNAL_OLD
        # Both keep the chunk_fwd_o continuation marker
        assert "chunk_fwd_o(" in m.ANCHOR_1C_FWD_INTERNAL_OLD
        assert "chunk_fwd_o(" in m.ANCHOR_1C_FWD_INTERNAL_NEW

    def test_1D_forward_drops_input_guard_decorator(self):
        m = _wiring()
        # Discriminate decorator-line ("    @input_guard\n") from prose
        # mention in a comment ("# instead of @input_guard.").
        def _has_decorator(body: str) -> bool:
            return any(
                ln.lstrip().startswith("@input_guard")
                for ln in body.splitlines()
                if not ln.lstrip().startswith("#")
            )
        assert _has_decorator(m.ANCHOR_1D_FORWARD_OLD)
        assert not _has_decorator(m.ANCHOR_1D_FORWARD_NEW)

    def test_1D_forward_adds_accelerator_device_index_context(self):
        m = _wiring()
        # NEW wraps the call in torch.accelerator.device_index(...) context
        assert "torch.accelerator.device_index" in m.ANCHOR_1D_FORWARD_NEW
        assert "torch.accelerator.device_index" not in m.ANCHOR_1D_FORWARD_OLD

    def test_1D_forward_adds_manual_contiguous_block(self):
        m = _wiring()
        # NEW must call .contiguous() manually on q,k,v,g,beta (replacing @input_guard)
        for tensor in ("q.contiguous()", "k.contiguous()", "v.contiguous()",
                       "g.contiguous()", "beta.contiguous()"):
            assert tensor in m.ANCHOR_1D_FORWARD_NEW, f"1D NEW missing {tensor}"

    def test_1D_forward_skips_initial_state_contiguous_when_indices_given(self):
        m = _wiring()
        # The whole point: when ssm_state_indices is given, do NOT
        # re-materialize the entire initial_state via .contiguous().
        # Pristine source (OLD) has unconditional initial_state.contiguous()
        # via @input_guard. NEW guards on `ssm_state_indices is None`.
        assert "if ssm_state_indices is None and initial_state is not None:" in \
            m.ANCHOR_1D_FORWARD_NEW

    def test_1E_sig_high_level_api_adds_kwargs(self):
        m = _wiring()
        old, new = m.ANCHOR_1E_SIG_OLD, m.ANCHOR_1E_SIG_NEW
        # OLD: 4-space indent (top-level fn) + use_qk_l2norm_in_kernel: bool = False
        assert "    use_qk_l2norm_in_kernel: bool = False,\n):" in old
        # OLD has NO new kwargs (false-fire guard)
        assert "ssm_state_indices" not in old
        # NEW adds both kwargs at 4-space indent
        assert "    ssm_state_indices: torch.Tensor | None = None,\n" in new
        assert "    has_initial_state: torch.Tensor | None = None,\n" in new

    def test_1E_validation_gates_on_ssm_state_indices(self):
        m = _wiring()
        old, new = m.ANCHOR_1E_VAL_OLD, m.ANCHOR_1E_VAL_NEW
        # Pristine has unconditional shape mismatch ValueError
        assert "initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1" in old
        # NEW restructures to 4 lines + adds `and ssm_state_indices is None`
        assert "and ssm_state_indices is None\n" in new
        assert "and initial_state.shape[0] != len(cu_seqlens) - 1\n" in new

    def test_1E_apply_call_passes_new_args_to_function(self):
        m = _wiring()
        old, new = m.ANCHOR_1E_APPLY_CALL_OLD, m.ANCHOR_1E_APPLY_CALL_NEW
        # Pristine ChunkGatedDeltaRuleFunction.apply call ends with use_qk_l2norm
        assert "        use_qk_l2norm_in_kernel,\n    )" in old
        assert "ssm_state_indices" not in old
        # NEW adds 2 trailing args before `)\n    return o, final_state`
        assert "        ssm_state_indices,\n" in new
        assert "        has_initial_state,\n" in new
        assert "    return o, final_state\n" in new


# ─────────────────────────────────────────────────────────────────────────
# 3. Sub-3 anchor (gdn_linear_attn.py)
# ─────────────────────────────────────────────────────────────────────────


class TestSub3Anchor:
    """Three anchors:
       3A — forward_cuda (FlashInfer fallback, gather/scatter remains
            because fi kernel can't read ssm_state in-place)
       3B — forward_native (passthrough kwargs to fla_chunk_gated_delta_rule)
       3C — _forward_core gather/scatter elimination (THE WIN SITE)
    """

    def test_3A_forward_cuda_OLD_uses_fi_kernel(self):
        m = _wiring()
        old = m.ANCHOR_3A_FORWARD_CUDA_OLD
        # Pristine returns fi_chunk_gated_delta_rule (FlashInfer)
        assert "return fi_chunk_gated_delta_rule(" in old
        # No PN79 kwargs yet
        assert "ssm_state_indices" not in old
        assert "has_initial_state" not in old

    def test_3A_forward_cuda_NEW_adds_gather_scatter_fallback(self):
        m = _wiring()
        new = m.ANCHOR_3A_FORWARD_CUDA_NEW
        # Signature gets two new kwargs
        assert "ssm_state_indices: torch.Tensor | None = None," in new
        assert "has_initial_state: torch.Tensor | None = None," in new
        # New if-branch does the gather + zero-fill + call + scatter
        assert "if ssm_state_indices is not None:" in new
        assert "gathered_initial = initial_state[ssm_state_indices].contiguous()" in new
        assert "gathered_initial[~has_initial_state, ...] = 0" in new
        assert "initial_state[ssm_state_indices] = final_state.to(initial_state.dtype)" in new
        # Original fi_chunk call kept as fallback path
        assert "return fi_chunk_gated_delta_rule(" in new

    def test_3B_forward_native_OLD_uses_fla_kernel(self):
        m = _wiring()
        old = m.ANCHOR_3B_FORWARD_NATIVE_OLD
        assert "return fla_chunk_gated_delta_rule(" in old
        assert "ssm_state_indices" not in old

    def test_3B_forward_native_NEW_passes_kwargs_through(self):
        m = _wiring()
        new = m.ANCHOR_3B_FORWARD_NATIVE_NEW
        # Sig adds two kwargs
        assert "ssm_state_indices: torch.Tensor | None = None," in new
        assert "has_initial_state: torch.Tensor | None = None," in new
        # Body forwards both to fla_chunk_gated_delta_rule
        assert "ssm_state_indices=ssm_state_indices," in new
        assert "has_initial_state=has_initial_state," in new

    def test_3C_OLD_has_full_gather_scatter_pattern(self):
        m = _wiring()
        a = m.ANCHOR_3C_GATHER_SCATTER_OLD
        # Pristine has these 4 lines: assert + gather + assert + zero-fill
        assert "assert non_spec_state_indices_tensor is not None" in a
        assert "ssm_state[non_spec_state_indices_tensor].contiguous()" in a
        assert "assert has_initial_state is not None" in a
        assert "initial_state[~has_initial_state, ...] = 0" in a
        # And scatter at end
        assert "ssm_state[non_spec_state_indices_tensor] = last_recurrent_state" in a

    def test_3C_NEW_eliminates_gather(self):
        m = _wiring()
        a = m.ANCHOR_3C_GATHER_SCATTER_NEW
        # Gather (ssm_state[indices].contiguous()) gone
        assert "ssm_state[non_spec_state_indices_tensor].contiguous()" not in a
        # Zero-fill of initial_state[~has_initial_state] gone
        assert "initial_state[~has_initial_state" not in a

    def test_3C_NEW_eliminates_scatter(self):
        m = _wiring()
        a = m.ANCHOR_3C_GATHER_SCATTER_NEW
        # No `ssm_state[indices] = last_recurrent_state...`
        assert "ssm_state[non_spec_state_indices_tensor] = last_recurrent_state" not in a

    def test_3C_NEW_passes_indices_and_mask_to_kernel(self):
        m = _wiring()
        a = m.ANCHOR_3C_GATHER_SCATTER_NEW
        # The kernel gets ssm_state in-place + indices + mask kwargs
        assert "initial_state=ssm_state," in a
        assert "ssm_state_indices=non_spec_state_indices_tensor," in a
        assert "has_initial_state=has_initial_state," in a


# ─────────────────────────────────────────────────────────────────────────
# 3b. Sub-2 anchors (chunk_delta_h.py — Triton kernel + Python wrapper)
# ─────────────────────────────────────────────────────────────────────────


class TestSub2Anchors:
    """7 sub-anchors against pristine vllm 0.20.2rc1.dev9+g01d4d1ad3:
       2A heuristics dict, 2B kernel signature, 2C kernel main flow,
       2D kernel epilogue, 2E wrapper signature, 2F wrapper body strides,
       2G wrapper kernel-call kwargs.
    """

    def test_2A_heuristics_dict_adds_two_lambdas(self):
        m = _wiring()
        old = m.ANCHOR_2A_HEURISTICS_OLD
        new = m.ANCHOR_2A_HEURISTICS_NEW
        # OLD ends with IS_VARLEN; NEW adds two more lambdas
        assert '"IS_VARLEN":' in old
        assert "IS_CONTINUOUS_BATCHING" not in old
        assert "HAS_INITIAL_STATE_MASK" not in old
        assert '"IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None' in new
        assert '"HAS_INITIAL_STATE_MASK": lambda args: args["has_initial_state"] is not None' in new

    def test_2B_kernel_signature_adds_params_strides_constexpr(self):
        m = _wiring()
        old = m.ANCHOR_2B_KERNEL_SIG_OLD
        new = m.ANCHOR_2B_KERNEL_SIG_NEW
        # Pristine has chunk_offsets followed by T (no ssm_state_indices)
        assert "chunk_offsets,\n    T,\n" in old or "chunk_offsets,\n    ssm_state_indices" not in old
        # NEW has new positional params
        assert "ssm_state_indices,\n    has_initial_state,\n" in new
        # Four new stride constexpr params
        for s in ("stride_init_state_token", "stride_final_state_token",
                  "stride_indices_seq", "stride_has_initial_state"):
            assert f"{s}: tl.constexpr," in new, f"2B NEW missing stride {s}"
        # Two new constexpr flags
        assert "IS_CONTINUOUS_BATCHING: tl.constexpr," in new
        assert "HAS_INITIAL_STATE_MASK: tl.constexpr," in new

    def test_2C_kernel_main_flow_restructures_initial_state_load(self):
        m = _wiring()
        old = m.ANCHOR_2C_KERNEL_MAIN_OLD
        new = m.ANCHOR_2C_KERNEL_MAIN_NEW
        # Pristine has both unconditional offset assigns
        assert "h0 = h0 + i_nh * V * K" in old
        assert "ht = ht + i_nh * V * K" in old
        # NEW introduces should_load + IS_CONTINUOUS_BATCHING branch
        assert "should_load = True" in new
        assert "if IS_CONTINUOUS_BATCHING:" in new
        assert "state_idx = tl.load(ssm_state_indices" in new
        # NEW wraps loads in `if should_load:`
        assert "if should_load:" in new
        # NEW removes the `ht = ht + i_nh * V * K` from this region
        # (it moves to anchor 2D — the epilogue)
        assert "ht = ht + i_nh * V * K" not in new

    def test_2D_kernel_epilogue_adds_ht_offset_branch(self):
        m = _wiring()
        old = m.ANCHOR_2D_KERNEL_EPILOGUE_OLD
        new = m.ANCHOR_2D_KERNEL_EPILOGUE_NEW
        # Pristine epilogue immediately constructs p_ht — no ht offset
        assert "# epilogue\n    if STORE_FINAL_STATE:\n        p_ht = " in old
        # NEW conditionally offsets ht via state_idx OR i_nh
        assert "if IS_CONTINUOUS_BATCHING:" in new
        assert "ht = ht + state_idx * stride_final_state_token + i_h * V * K" in new
        assert "ht = ht + i_nh * V * K" in new   # else branch

    def test_2E_wrapper_signature_adds_two_kwargs(self):
        m = _wiring()
        old = m.ANCHOR_2E_WRAPPER_SIG_OLD
        new = m.ANCHOR_2E_WRAPPER_SIG_NEW
        # Pristine signature ends right after chunk_offsets + ) ->
        assert "chunk_offsets: torch.Tensor | None = None,\n) -> tuple[torch.Tensor, torch.Tensor]:" in old
        # NEW adds two params
        assert "ssm_state_indices: torch.Tensor | None = None,\n    has_initial_state: torch.Tensor | None = None,\n) -> tuple[torch.Tensor, torch.Tensor]:" in new

    def test_2F_wrapper_body_adds_strides_block(self):
        m = _wiring()
        old = m.ANCHOR_2F_WRAPPER_BODY_OLD
        new = m.ANCHOR_2F_WRAPPER_BODY_NEW
        # Pristine has just final_state init + h alloc
        assert "h = k.new_empty(B, NT, H, V, K)" in old
        assert "if ssm_state_indices is not None:" not in old
        # NEW has the if/else stride block AND keeps h alloc
        assert "if ssm_state_indices is not None:" in new
        assert "stride_indices_seq = ssm_state_indices.stride(0)" in new
        assert "stride_init_state_token = initial_state.stride(0)" in new
        assert "final_state = initial_state if output_final_state else None" in new
        # NEW preserves the v_new line (kept by diff)
        assert "v_new = torch.empty_like(u) if save_new_value else None" in new

    def test_2G_wrapper_kernel_call_passes_new_args(self):
        m = _wiring()
        old = m.ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD
        new = m.ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW
        # Pristine kernel-call kwargs end with BT=BT,
        assert "BT=BT,\n    )" in old
        # NEW passes ssm_state_indices, has_initial_state, 4 strides
        assert "ssm_state_indices=ssm_state_indices," in new
        assert "has_initial_state=has_initial_state," in new
        for s in ("stride_init_state_token", "stride_final_state_token",
                  "stride_indices_seq", "stride_has_initial_state"):
            assert f"{s}={s}," in new, f"2G NEW missing kwarg {s}={s},"


# ─────────────────────────────────────────────────────────────────────────
# 4. Drift markers — configured per-patcher AND absent from OLD anchors
# ─────────────────────────────────────────────────────────────────────────


class TestDriftMarkers:

    def test_chunk_patcher_drift_markers_configured(self):
        m = _wiring()
        # On Mac vllm is not installed → resolve_vllm_file returns None,
        # so patchers cannot be constructed for direct introspection.
        # Source-string check: each declared drift marker appears as a
        # list element in the wiring source.
        import inspect
        src = inspect.getsource(m)
        # Sub-1 patcher drift markers
        assert '"ssm_state_indices"' in src
        assert '"has_initial_state"' in src
        assert '"torch.accelerator.device_index"' in src
        # Sub-2 patcher drift markers (kernel-level)
        assert '"IS_CONTINUOUS_BATCHING"' in src
        assert '"HAS_INITIAL_STATE_MASK"' in src
        assert '"stride_init_state_token"' in src
        # Sub-3 patcher: no drift markers (see docstring rationale —
        # the candidate marker false-fires on pristine decode-path lines).
        # Verify the empty list is declared:
        assert "upstream_drift_markers=[]," in src

    def test_drift_markers_absent_from_pristine_OLD_anchors(self):
        """If a marker appears in pristine OLD, the upstream-drift detection
        will false-fire — patcher would skip with status='upstream-merged'
        before even checking the OLD anchor."""
        m = _wiring()
        markers_chunk = ("ssm_state_indices", "has_initial_state",
                         "torch.accelerator.device_index")
        for marker in markers_chunk:
            for anchor_name in ("ANCHOR_1A_IMPORT_OLD",
                                "ANCHOR_1B_FWD_SIG_OLD",
                                "ANCHOR_1C_FWD_INTERNAL_OLD",
                                "ANCHOR_1D_FORWARD_OLD",
                                "ANCHOR_1E_SIG_OLD",
                                "ANCHOR_1E_VAL_OLD",
                                "ANCHOR_1E_APPLY_CALL_OLD"):
                anchor = getattr(m, anchor_name)
                assert marker not in anchor, (
                    f"chunk drift marker '{marker}' found in pristine "
                    f"anchor '{anchor_name}' — would false-fire drift "
                    f"detection"
                )

    def test_kernel_drift_markers_absent_from_Sub2_OLD_anchors(self):
        """Sub-2 (kernel) drift markers — IS_CONTINUOUS_BATCHING and
        HAS_INITIAL_STATE_MASK — must not appear in pristine OLD."""
        m = _wiring()
        markers_kernel = ("IS_CONTINUOUS_BATCHING", "HAS_INITIAL_STATE_MASK",
                          "ssm_state_indices", "has_initial_state")
        for marker in markers_kernel:
            for anchor_name in ("ANCHOR_2A_HEURISTICS_OLD",
                                "ANCHOR_2B_KERNEL_SIG_OLD",
                                "ANCHOR_2C_KERNEL_MAIN_OLD",
                                "ANCHOR_2D_KERNEL_EPILOGUE_OLD",
                                "ANCHOR_2E_WRAPPER_SIG_OLD",
                                "ANCHOR_2F_WRAPPER_BODY_OLD",
                                "ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD"):
                anchor = getattr(m, anchor_name)
                assert marker not in anchor, (
                    f"kernel drift marker '{marker}' found in pristine "
                    f"anchor '{anchor_name}' — would false-fire drift detection"
                )

    def test_drift_marker_present_in_NEW_anchors_positive_control(self):
        """Inverse check: NEW anchors MUST contain the markers. If they
        don't, the patch is incomplete — the sentinel never lands."""
        m = _wiring()
        # ssm_state_indices: appears in 1B sig, 1C call, 1D body
        for anchor_name in ("ANCHOR_1B_FWD_SIG_NEW",
                            "ANCHOR_1C_FWD_INTERNAL_NEW",
                            "ANCHOR_1D_FORWARD_NEW"):
            anchor = getattr(m, anchor_name)
            assert "ssm_state_indices" in anchor, \
                f"NEW anchor '{anchor_name}' missing 'ssm_state_indices' marker"
        # accelerator marker only in 1D
        assert "torch.accelerator.device_index" in m.ANCHOR_1D_FORWARD_NEW


# ─────────────────────────────────────────────────────────────────────────
# 5. Per-anchor TextPatcher idempotency (Sub-1 + Sub-3)
# ─────────────────────────────────────────────────────────────────────────


class TestTextPatcherIdempotency:
    """Verify each anchor pair survives TextPatcher round-trip:
    OLD → apply → patched (contains marker) → 2nd apply → IDEMPOTENT (no-op)."""

    @pytest.mark.parametrize("name,old_attr,new_attr", [
        ("1A_import",       "ANCHOR_1A_IMPORT_OLD",       "ANCHOR_1A_IMPORT_NEW"),
        ("1B_fwd_sig",      "ANCHOR_1B_FWD_SIG_OLD",      "ANCHOR_1B_FWD_SIG_NEW"),
        ("1C_internal_call","ANCHOR_1C_FWD_INTERNAL_OLD", "ANCHOR_1C_FWD_INTERNAL_NEW"),
        ("1D_forward",      "ANCHOR_1D_FORWARD_OLD",      "ANCHOR_1D_FORWARD_NEW"),
        ("1E_sig",          "ANCHOR_1E_SIG_OLD",          "ANCHOR_1E_SIG_NEW"),
        ("1E_val",          "ANCHOR_1E_VAL_OLD",          "ANCHOR_1E_VAL_NEW"),
        ("1E_apply_call",   "ANCHOR_1E_APPLY_CALL_OLD",   "ANCHOR_1E_APPLY_CALL_NEW"),
        ("2A_heuristics",   "ANCHOR_2A_HEURISTICS_OLD",   "ANCHOR_2A_HEURISTICS_NEW"),
        ("2B_kernel_sig",   "ANCHOR_2B_KERNEL_SIG_OLD",   "ANCHOR_2B_KERNEL_SIG_NEW"),
        ("2C_kernel_main",  "ANCHOR_2C_KERNEL_MAIN_OLD",  "ANCHOR_2C_KERNEL_MAIN_NEW"),
        ("2D_kernel_epi",   "ANCHOR_2D_KERNEL_EPILOGUE_OLD", "ANCHOR_2D_KERNEL_EPILOGUE_NEW"),
        ("2E_wrapper_sig",  "ANCHOR_2E_WRAPPER_SIG_OLD",  "ANCHOR_2E_WRAPPER_SIG_NEW"),
        ("2F_wrapper_body", "ANCHOR_2F_WRAPPER_BODY_OLD", "ANCHOR_2F_WRAPPER_BODY_NEW"),
        ("2G_wrapper_call", "ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD", "ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW"),
        ("3A_forward_cuda", "ANCHOR_3A_FORWARD_CUDA_OLD",        "ANCHOR_3A_FORWARD_CUDA_NEW"),
        ("3B_forward_native","ANCHOR_3B_FORWARD_NATIVE_OLD",     "ANCHOR_3B_FORWARD_NATIVE_NEW"),
        ("3C_gather_scatter","ANCHOR_3C_GATHER_SCATTER_OLD","ANCHOR_3C_GATHER_SCATTER_NEW"),
        ("4A_olmo_forward_core","ANCHOR_4A_OLMO_FORWARD_CORE_OLD","ANCHOR_4A_OLMO_FORWARD_CORE_NEW"),
    ])
    def test_anchor_round_trip(self, tmp_path, name, old_attr, new_attr):
        from sndr.kernel.text_patch import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        m = _wiring()
        old = getattr(m, old_attr)
        new = getattr(m, new_attr)

        target = tmp_path / f"pn79_{name}.py"
        target.write_text("# header\n" + old + "\n# tail\n")
        patcher = TextPatcher(
            patch_name=f"PN79_{name}",
            target_file=str(target),
            marker=m.GENESIS_PN79_MARKER,
            sub_patches=[
                TextPatch(name=f"pn79_{name}", anchor=old,
                          replacement=new, required=True),
            ],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED, f"{name} 1st apply must succeed"
        body1 = target.read_text()
        # Marker present (TextPatcher inserts marker comment + replacement)
        assert m.GENESIS_PN79_MARKER in body1, f"{name} marker missing after apply"
        # Second apply must be a no-op
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT, \
            f"{name} 2nd apply must be IDEMPOTENT (got {r2})"
        assert target.read_text() == body1, f"{name} 2nd apply mutated body"


# ─────────────────────────────────────────────────────────────────────────
# 6. apply() contract
# ─────────────────────────────────────────────────────────────────────────


class TestApplyContract:
    """apply() should:
       - SKIP cleanly when env disabled (default OFF gate)
       - SKIP cleanly when env enabled (because Sub-2 patcher is None →
         MultiFilePatchTransaction atomic-skip safety net)
       In NEITHER case may apply() raise, even if vllm is not installed."""

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE",
                           raising=False)
        m = _wiring()
        status, reason = m.apply()
        assert status == "skipped"
        assert "off" in reason.lower() or "opt-in" in reason.lower()

    def test_apply_skipped_when_env_enabled_because_sub2_pending(
            self, monkeypatch):
        """Until Sub-2 chunk_delta_h.py anchors land, apply() must atomic-
        skip even with env=1. This protects PROD: partial apply would
        crash boot."""
        monkeypatch.setenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", "1")
        m = _wiring()
        status, reason = m.apply()
        assert status == "skipped"
        # Reason must mention the actual blocker (not "default OFF").
        # Either: vllm not discoverable on this machine, OR Sub-2 None.
        low = reason.lower()
        assert any(s in low for s in (
            "vllm install root not discoverable",
            "patcher is none",
            "dry-run",
            "dry_run",
            "atomic",
        )), f"unexpected skip reason: {reason!r}"


# ─────────────────────────────────────────────────────────────────────────
# 7. Registry entry — conflicts_with [PN59, PN54]
# ─────────────────────────────────────────────────────────────────────────


class TestRegistryEntry:

    def test_PN79_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN79" in PATCH_REGISTRY

    def test_PN79_default_off(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN79"]["default_on"] is False

    def test_PN79_env_flag(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert (PATCH_REGISTRY["PN79"]["env_flag"]
                == "GENESIS_ENABLE_PN79_INPLACE_SSM_STATE")

    def test_PN79_lifecycle_experimental(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN79"]["lifecycle"] == "experimental"

    def test_PN79_applies_only_to_hybrid(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN79"]["applies_to"] == {"is_hybrid": [True]}

    def test_PN79_credit_mentions_pr_41824(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "41824" in PATCH_REGISTRY["PN79"]["credit"]

    def test_PN79_conflicts_with_PN59_and_PN54(self):
        from sndr.dispatcher import PATCH_REGISTRY
        cw = PATCH_REGISTRY["PN79"].get("conflicts_with", [])
        assert "PN59" in cw, "PN79 must declare conflicts_with PN59"
        assert "PN54" in cw, "PN79 must declare conflicts_with PN54"
