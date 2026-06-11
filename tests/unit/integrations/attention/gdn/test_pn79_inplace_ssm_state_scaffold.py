# SPDX-License-Identifier: Apache-2.0
"""TDD for PN79 — in-place SSM state for GDN chunk prefill (vllm#41824 backport).

State 2026-06-10 (K.2 re-anchor on pin 0.22.1rc1.dev259+g303916e93):
all four sub-patchers are implemented against the post-#44700 upstream
layout — Sub-1 chunk.py (8 anchors), Sub-2 chunk_delta_h.py (7 anchors),
Sub-3 qwen_gdn_linear_attn.py (2 anchors, backend-gated), Sub-4
olmo_gdn_linear_attn.py (1 anchor).

These tests verify:
  1. Module imports cleanly + apply contract
  2. Sub-1 anchors (8): 1B fwd sig, 1C internal call, 1D decorator drop,
     1D forward sig + manual contiguity, 1D forward inner call,
     1E_SIG/1E_VAL/1E_APPLY_CALL high-level API
  3. Sub-3 anchors (2): 3B forward_native passthrough, 3C backend-gated
     prefill in-place state (gather/scatter kept verbatim for
     flashinfer/cutedsl backends)
  4. Sub-2 anchors (7): kernel heuristics/signature/main/epilogue +
     wrapper signature/body/call
  5. Drift markers configured per-patcher and absent from OLD anchors
  6. Per-anchor TextPatcher idempotency on tmp_path fixtures
  7. PATCH_REGISTRY entry has conflicts_with: [PN59, PN54]
  8. apply() returns 'skipped' both when env disabled AND when env
     enabled on a vllm-less dev machine (no install root)
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
    """8 sub-anchors: 1B fwd sig, 1C internal call, 1D decorator /
    forward sig+contiguity / forward inner call, 1E_SIG/VAL/APPLY_CALL."""

    def test_1B_fwd_signature_adds_ssm_state_indices(self):
        m = _wiring()
        assert m.ANCHOR_1B_FWD_SIG_OLD != m.ANCHOR_1B_FWD_SIG_NEW
        # OLD must NOT have the new params (false-fire guard)
        assert "ssm_state_indices" not in m.ANCHOR_1B_FWD_SIG_OLD
        assert "has_initial_state" not in m.ANCHOR_1B_FWD_SIG_OLD
        # NEW must have both new params, BEFORE core_attn_out (upstream order)
        assert "ssm_state_indices: torch.Tensor | None = None" in m.ANCHOR_1B_FWD_SIG_NEW
        assert "has_initial_state: torch.Tensor | None = None" in m.ANCHOR_1B_FWD_SIG_NEW
        assert m.ANCHOR_1B_FWD_SIG_NEW.index("ssm_state_indices") < \
            m.ANCHOR_1B_FWD_SIG_NEW.index("core_attn_out")
        # Both anchors carry the #44700 core_attn_out param (post-rebase
        # pre-image) — this is what distinguishes the K.2 anchors from
        # the stale pre-#44700 ones.
        assert "core_attn_out: torch.Tensor | None = None" in m.ANCHOR_1B_FWD_SIG_OLD

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

    def test_1D_decorator_drops_input_guard(self):
        m = _wiring()
        # Discriminate decorator-line ("    @input_guard\n") from prose
        # mention in a comment ("# ... @input_guard dropped ...").
        def _has_decorator(body: str) -> bool:
            return any(
                ln.lstrip().startswith("@input_guard")
                for ln in body.splitlines()
                if not ln.lstrip().startswith("#")
            )
        assert _has_decorator(m.ANCHOR_1D_DECORATOR_OLD)
        assert not _has_decorator(m.ANCHOR_1D_DECORATOR_NEW)
        # @staticmethod and custom_fwd survive
        assert "@staticmethod" in m.ANCHOR_1D_DECORATOR_NEW
        assert "@torch.amp.custom_fwd" in m.ANCHOR_1D_DECORATOR_NEW

    def test_1D_forward_skips_accelerator_device_index_context(self):
        """K.2 deliberate deviation from upstream #41824: the
        torch.accelerator.device_index wrapper is NOT ported (Genesis TP
        workers are single-device per process — the context is a no-op,
        and skipping it keeps the forward body at original indentation).
        The string remains configured as an upstream-merge drift marker.
        """
        m = _wiring()
        assert "torch.accelerator.device_index" not in m.ANCHOR_1D_FORWARD_SIG_NEW
        assert "torch.accelerator.device_index" not in m.ANCHOR_1D_FORWARD_CALL_NEW

    def test_1D_forward_adds_manual_contiguous_block(self):
        m = _wiring()
        # NEW must call .contiguous() manually on q,k,v,g,beta (replacing @input_guard)
        for tensor in ("q.contiguous()", "k.contiguous()", "v.contiguous()",
                       "g.contiguous()", "beta.contiguous()"):
            assert tensor in m.ANCHOR_1D_FORWARD_SIG_NEW, \
                f"1D_FORWARD_SIG NEW missing {tensor}"

    def test_1D_forward_skips_initial_state_contiguous_when_indices_given(self):
        m = _wiring()
        # The whole point: when ssm_state_indices is given, do NOT
        # re-materialize the entire initial_state via .contiguous().
        assert "if ssm_state_indices is None and initial_state is not None:" in \
            m.ANCHOR_1D_FORWARD_SIG_NEW

    def test_1D_forward_inner_call_passes_kwargs_before_core_attn_out(self):
        m = _wiring()
        old, new = m.ANCHOR_1D_FORWARD_CALL_OLD, m.ANCHOR_1D_FORWARD_CALL_NEW
        assert "ssm_state_indices" not in old
        assert "ssm_state_indices=ssm_state_indices," in new
        assert "has_initial_state=has_initial_state," in new
        # Upstream kwarg order: new kwargs sit before core_attn_out
        assert new.index("ssm_state_indices=") < new.index("core_attn_out=")

    def test_1E_sig_high_level_api_adds_kwargs(self):
        m = _wiring()
        old, new = m.ANCHOR_1E_SIG_OLD, m.ANCHOR_1E_SIG_NEW
        # OLD: 4-space indent (top-level fn) + use_qk_l2norm_in_kernel
        assert "    use_qk_l2norm_in_kernel: bool = False,\n" in old
        # OLD has NO new kwargs (false-fire guard)
        assert "ssm_state_indices" not in old
        # NEW adds both kwargs at 4-space indent
        assert "    ssm_state_indices: torch.Tensor | None = None,\n" in new
        assert "    has_initial_state: torch.Tensor | None = None,\n" in new
        # Docstring opener tail pins the anchor to the high-level API
        assert 'r"""' in old and 'r"""' in new

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
        # Pristine .apply() trailing args end with core_attn_out (#44700)
        assert "        use_qk_l2norm_in_kernel,\n        core_attn_out,\n    )" in old
        assert "ssm_state_indices" not in old
        # NEW adds 2 positional args BETWEEN use_qk_l2norm and core_attn_out
        # (must mirror forward's parameter order exactly)
        assert "        ssm_state_indices,\n" in new
        assert "        has_initial_state,\n" in new
        assert new.index("use_qk_l2norm_in_kernel,") \
            < new.index("ssm_state_indices,") \
            < new.index("has_initial_state,") \
            < new.index("core_attn_out,")
        assert "    return o, final_state\n" in new


# ─────────────────────────────────────────────────────────────────────────
# 3. Sub-3 anchors (qwen_gdn_linear_attn.py)
# ─────────────────────────────────────────────────────────────────────────


class TestSub3Anchors:
    """Two anchors:
       3B — forward_native (passthrough kwargs to fla_chunk_gated_delta_rule)
       3C — _forward_core prefill block, backend-gated in-place state
            (THE WIN SITE). flashinfer/cutedsl keep upstream gather/scatter.
    """

    def test_3B_forward_native_OLD_uses_fla_kernel(self):
        m = _wiring()
        old = m.ANCHOR_3B_FORWARD_NATIVE_OLD
        assert "return fla_chunk_gated_delta_rule(" in old
        assert "ssm_state_indices" not in old
        # Post-#44700 pre-image: core_attn_out threaded through
        assert "core_attn_out=core_attn_out," in old

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
        a = m.ANCHOR_3C_PREFILL_INPLACE_OLD
        # Pristine prefill block: asserts + gather + zero-fill
        assert "assert prefill_state_indices is not None" in a
        assert "initial_state = ssm_state[prefill_state_indices]" in a
        assert "initial_state[~prefill_has_initial_state, ...] = 0" in a
        # And scatter at end
        assert "ssm_state[prefill_state_indices] = last_recurrent_state" in a
        # No gate yet
        assert "_pn79_inplace" not in a

    def test_3C_NEW_gates_on_triton_backend(self):
        """K.2 design: in-place kwargs are passed ONLY when
        self.gdn_prefill_backend == "triton". The attribute name and the
        Literal value were verified on live pin g303916e93
        (_resolve_gdn_prefill_backend returns
        Literal["triton", "flashinfer", "cutedsl"])."""
        m = _wiring()
        a = m.ANCHOR_3C_PREFILL_INPLACE_NEW
        assert '_pn79_inplace = self.gdn_prefill_backend == "triton"' in a
        assert "**_pn79_kwargs," in a
        # Triton branch: pool passed directly + in-place kwargs
        assert "initial_state = ssm_state\n" in a
        assert '"ssm_state_indices": prefill_state_indices,' in a
        assert '"has_initial_state": prefill_has_initial_state,' in a

    def test_3C_NEW_keeps_upstream_gather_scatter_for_other_backends(self):
        """flashinfer/cutedsl kernels do not accept the in-place kwargs —
        the else branch must keep the upstream gather/scatter verbatim,
        and the scatter must stay (conditionally) for that path."""
        m = _wiring()
        a = m.ANCHOR_3C_PREFILL_INPLACE_NEW
        assert "initial_state = ssm_state[prefill_state_indices]" in a
        assert "initial_state[~prefill_has_initial_state, ...] = 0" in a
        assert "_pn79_kwargs = {}" in a
        assert "if not _pn79_inplace:" in a
        assert "ssm_state[prefill_state_indices] = last_recurrent_state.to(" in a

    def test_3C_NEW_does_not_touch_forward_cuda(self):
        """The old 3A forward_cuda anchor is retired: with the call-site
        gate, the FlashInfer path stays upstream-identical. Guard against
        accidental resurrection."""
        m = _wiring()
        assert not hasattr(m, "ANCHOR_3A_FORWARD_CUDA_OLD")
        assert "fi_chunk_gated_delta_rule" not in m.ANCHOR_3C_PREFILL_INPLACE_NEW


# ─────────────────────────────────────────────────────────────────────────
# 3b. Sub-4 anchor (olmo_gdn_linear_attn.py)
# ─────────────────────────────────────────────────────────────────────────


class TestSub4Anchor:

    def test_4A_OLD_has_gather_scatter(self):
        m = _wiring()
        a = m.ANCHOR_4A_OLMO_PREFILL_OLD
        assert "ssm_state[non_spec_state_indices_tensor].contiguous()" in a
        assert "initial_state[~has_initial_state, ...] = 0" in a
        assert "ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(" in a

    def test_4A_NEW_eliminates_gather_scatter_unconditionally(self):
        """Olmo always uses the FLA Triton kernel (free-function call,
        no backend dispatch) — upstream-verbatim elimination, no gate."""
        m = _wiring()
        a = m.ANCHOR_4A_OLMO_PREFILL_NEW
        assert "ssm_state[non_spec_state_indices_tensor].contiguous()" not in a
        assert "initial_state[~has_initial_state" not in a
        assert "initial_state=ssm_state," in a
        assert "ssm_state_indices=non_spec_state_indices_tensor," in a
        assert "has_initial_state=has_initial_state," in a
        # Scatter gone (kernel stores in place)
        assert "= last_recurrent_state.to(" not in a


# ─────────────────────────────────────────────────────────────────────────
# 4. Sub-2 anchors (chunk_delta_h.py — Triton kernel + Python wrapper)
# ─────────────────────────────────────────────────────────────────────────


class TestSub2Anchors:
    """7 sub-anchors against pristine vllm 0.22.1rc1.dev259+g303916e93:
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
        # K.2 pre-image carries USE_G/USE_GK and terminal USE_EXP2
        assert "USE_G: tl.constexpr," in old
        assert "USE_EXP2: tl.constexpr,\n):" in old
        assert "ssm_state_indices" not in old
        # NEW has new positional params
        assert "ssm_state_indices,\n    has_initial_state,\n" in new
        # Four new stride constexpr params
        for s in ("stride_init_state_token", "stride_final_state_token",
                  "stride_indices_seq", "stride_has_initial_state"):
            assert f"{s}: tl.constexpr," in new, f"2B NEW missing stride {s}"
        # Two new constexpr flags, placed before USE_EXP2 (upstream order)
        assert "IS_CONTINUOUS_BATCHING: tl.constexpr," in new
        assert "HAS_INITIAL_STATE_MASK: tl.constexpr," in new
        assert new.index("IS_CONTINUOUS_BATCHING") < new.index("USE_EXP2")

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
        # K.2 pre-image: use_exp2 terminal kwarg before the return arrow
        assert "use_exp2: bool = False,\n) -> tuple[torch.Tensor, torch.Tensor]:" in old
        assert "ssm_state_indices" not in old
        # NEW adds two params BEFORE use_exp2 (upstream order)
        assert ("ssm_state_indices: torch.Tensor | None = None,\n"
                "    has_initial_state: torch.Tensor | None = None,\n"
                "    use_exp2: bool = False,") in new

    def test_2F_wrapper_body_adds_strides_block(self):
        m = _wiring()
        old = m.ANCHOR_2F_WRAPPER_BODY_OLD
        new = m.ANCHOR_2F_WRAPPER_BODY_NEW
        # K.2 anchor covers ONLY the fp32 final_state allocation (the
        # h allocation above it belongs to PN106's h-pool territory)
        assert "final_state = (" in old
        assert "k.new_empty(N, H, V, K, dtype=torch.float32)" in old
        assert "if ssm_state_indices is not None:" not in old
        assert "h = k.new_empty(B, NT, H, V, K)" not in old
        # NEW has the if/else stride block; pool doubles as final_state
        assert "if ssm_state_indices is not None:" in new
        assert "stride_indices_seq = ssm_state_indices.stride(0)" in new
        assert "stride_init_state_token = initial_state.stride(0)" in new
        assert "final_state = initial_state if output_final_state else None" in new
        # else branch keeps the upstream fp32 allocation
        assert "k.new_empty(N, H, V, K, dtype=torch.float32)" in new

    def test_2G_wrapper_kernel_call_passes_new_args(self):
        m = _wiring()
        old = m.ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD
        new = m.ANCHOR_2G_WRAPPER_KERNEL_CALL_NEW
        # K.2 pre-image kernel-call kwargs end with USE_EXP2
        assert "BT=BT,\n        USE_EXP2=use_exp2,\n    )" in old
        assert "ssm_state_indices" not in old
        # NEW passes ssm_state_indices, has_initial_state, 4 strides
        assert "ssm_state_indices=ssm_state_indices," in new
        assert "has_initial_state=has_initial_state," in new
        for s in ("stride_init_state_token", "stride_final_state_token",
                  "stride_indices_seq", "stride_has_initial_state"):
            assert f"{s}={s}," in new, f"2G NEW missing kwarg {s}={s},"


# ─────────────────────────────────────────────────────────────────────────
# 5. Drift markers — configured per-patcher AND absent from OLD anchors
# ─────────────────────────────────────────────────────────────────────────


_SUB1_OLD_ANCHOR_NAMES = (
    "ANCHOR_1B_FWD_SIG_OLD",
    "ANCHOR_1C_FWD_INTERNAL_OLD",
    "ANCHOR_1D_DECORATOR_OLD",
    "ANCHOR_1D_FORWARD_SIG_OLD",
    "ANCHOR_1D_FORWARD_CALL_OLD",
    "ANCHOR_1E_SIG_OLD",
    "ANCHOR_1E_VAL_OLD",
    "ANCHOR_1E_APPLY_CALL_OLD",
)

_SUB2_OLD_ANCHOR_NAMES = (
    "ANCHOR_2A_HEURISTICS_OLD",
    "ANCHOR_2B_KERNEL_SIG_OLD",
    "ANCHOR_2C_KERNEL_MAIN_OLD",
    "ANCHOR_2D_KERNEL_EPILOGUE_OLD",
    "ANCHOR_2E_WRAPPER_SIG_OLD",
    "ANCHOR_2F_WRAPPER_BODY_OLD",
    "ANCHOR_2G_WRAPPER_KERNEL_CALL_OLD",
)


class TestDriftMarkers:

    def test_chunk_patcher_drift_markers_configured(self):
        m = _wiring()
        # On Mac vllm is not installed → resolve_vllm_file returns None,
        # so patchers cannot be constructed for direct introspection.
        # Source-string check: each declared drift marker appears as a
        # list element in the wiring source.
        import inspect
        src = inspect.getsource(m)
        # Sub-1 patcher: only the strictly-upstream-only marker remains.
        # Self-collision lint (triage plan §6 2026-06-11): the former
        # entries (ssm_state_indices, has_initial_state,
        # IS_CONTINUOUS_BATCHING, HAS_INITIAL_STATE_MASK,
        # stride_init_state_token) are baked verbatim by PN79's own
        # replacements and were removed — they could false-skip the patch
        # as "upstream_merged" on residue (PN369 class).
        assert '"torch.accelerator.device_index"' in src
        # The removed names must NOT reappear as drift-marker list entries
        # (exact list-element spelling; mentions in comments are fine):
        for removed in ("ssm_state_indices", "has_initial_state",
                        "IS_CONTINUOUS_BATCHING", "HAS_INITIAL_STATE_MASK",
                        "stride_init_state_token"):
            assert f'            "{removed}",\n' not in src, (
                f"former self-colliding drift marker {removed!r} "
                "reintroduced as a list entry"
            )
        # Sub-2/Sub-3/Sub-4 patchers: no drift markers (see docstring
        # rationale — candidate markers either false-fire on pristine
        # decode-path lines or collide with our own replacements).
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
            for anchor_name in _SUB1_OLD_ANCHOR_NAMES:
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
            for anchor_name in _SUB2_OLD_ANCHOR_NAMES:
                anchor = getattr(m, anchor_name)
                assert marker not in anchor, (
                    f"kernel drift marker '{marker}' found in pristine "
                    f"anchor '{anchor_name}' — would false-fire drift detection"
                )

    def test_drift_marker_present_in_NEW_anchors_positive_control(self):
        """Inverse check: the ssm_state_indices sentinel MUST land in the
        NEW anchors. (torch.accelerator.device_index deliberately does
        NOT land — K.2 skips the wrapper; it remains configured purely
        as an upstream-merge detector.)"""
        m = _wiring()
        for anchor_name in ("ANCHOR_1B_FWD_SIG_NEW",
                            "ANCHOR_1C_FWD_INTERNAL_NEW",
                            "ANCHOR_1D_FORWARD_SIG_NEW",
                            "ANCHOR_1D_FORWARD_CALL_NEW"):
            anchor = getattr(m, anchor_name)
            assert "ssm_state_indices" in anchor, \
                f"NEW anchor '{anchor_name}' missing 'ssm_state_indices' marker"


# ─────────────────────────────────────────────────────────────────────────
# 6. Per-anchor TextPatcher idempotency (all 18 anchors)
# ─────────────────────────────────────────────────────────────────────────


class TestTextPatcherIdempotency:
    """Verify each anchor pair survives TextPatcher round-trip:
    OLD → apply → patched (contains marker) → 2nd apply → IDEMPOTENT (no-op)."""

    @pytest.mark.parametrize("name,old_attr,new_attr", [
        ("1B_fwd_sig",      "ANCHOR_1B_FWD_SIG_OLD",      "ANCHOR_1B_FWD_SIG_NEW"),
        ("1C_internal_call","ANCHOR_1C_FWD_INTERNAL_OLD", "ANCHOR_1C_FWD_INTERNAL_NEW"),
        ("1D_decorator",    "ANCHOR_1D_DECORATOR_OLD",    "ANCHOR_1D_DECORATOR_NEW"),
        ("1D_forward_sig",  "ANCHOR_1D_FORWARD_SIG_OLD",  "ANCHOR_1D_FORWARD_SIG_NEW"),
        ("1D_forward_call", "ANCHOR_1D_FORWARD_CALL_OLD", "ANCHOR_1D_FORWARD_CALL_NEW"),
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
        ("3B_forward_native","ANCHOR_3B_FORWARD_NATIVE_OLD",     "ANCHOR_3B_FORWARD_NATIVE_NEW"),
        ("3C_prefill_inplace","ANCHOR_3C_PREFILL_INPLACE_OLD","ANCHOR_3C_PREFILL_INPLACE_NEW"),
        ("4A_olmo_prefill", "ANCHOR_4A_OLMO_PREFILL_OLD","ANCHOR_4A_OLMO_PREFILL_NEW"),
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
# 7. apply() contract
# ─────────────────────────────────────────────────────────────────────────


class TestApplyContract:
    """apply() should:
       - SKIP cleanly when env disabled (default OFF gate)
       - SKIP cleanly when env enabled but vllm is not installed
         (dev machine) — MultiFilePatchTransaction atomic-skip safety net
       In NEITHER case may apply() raise."""

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE",
                           raising=False)
        m = _wiring()
        status, reason = m.apply()
        assert status == "skipped"
        assert "off" in reason.lower() or "opt-in" in reason.lower()

    def test_apply_skipped_when_env_enabled_without_vllm_install(
            self, monkeypatch):
        """On a vllm-less dev machine apply() must skip cleanly with the
        actual blocker as reason (not 'default OFF'). On a machine WITH
        vllm installed the transaction either applies fully or
        atomic-skips — never half-applies."""
        monkeypatch.setenv("GENESIS_ENABLE_PN79_INPLACE_SSM_STATE", "1")
        m = _wiring()
        status, reason = m.apply()
        assert status in ("skipped", "applied")
        if status == "skipped":
            low = reason.lower()
            assert any(s in low for s in (
                "vllm install root not discoverable",
                "patcher is none",
                "dry-run",
                "dry_run",
                "atomic",
                "anchor",
            )), f"unexpected skip reason: {reason!r}"


# ─────────────────────────────────────────────────────────────────────────
# 8. Registry entry — conflicts_with [PN59, PN54]
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

    def test_PN79_credit_mentions_k2_reanchor(self):
        from sndr.dispatcher import PATCH_REGISTRY
        credit = PATCH_REGISTRY["PN79"]["credit"]
        assert "g303916e93" in credit
        assert "2026-06-10" in credit

    def test_PN79_conflicts_with_PN59_and_PN54(self):
        from sndr.dispatcher import PATCH_REGISTRY
        cw = PATCH_REGISTRY["PN79"].get("conflicts_with", [])
        assert "PN59" in cw, "PN79 must declare conflicts_with PN59"
        assert "PN54" in cw, "PN79 must declare conflicts_with PN54"
