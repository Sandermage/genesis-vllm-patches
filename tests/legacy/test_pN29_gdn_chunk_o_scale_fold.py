# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PN29 — GDN chunk_o scale-fold (vllm#41446 pattern (c) backport).

2026-06-19: PN29 was CONSOLIDATED with PN298 into one wiring module
(`pn29_pn298_chunk_o_consolidated`) — both patch the same engine file
`model_executor/layers/fla/ops/chunk_o.py` at disjoint regions. The merged
registry entry keeps the id "PN298"; PN29's env flag
(`GENESIS_ENABLE_PN29_GDN_SCALE_FOLD`) is retained as a recognized alias and
still independently gates the `pn29_scale_fold` sub-patch. These tests were
repointed to the consolidated module while PRESERVING their assertions about
the PN29 anchor, replacement, marker, and env-gating.

Test contract:
1. Anchor text matches exact upstream code (line 137 in chunk_o.py)
2. Replacement preserves the math (1 fewer fp32 multiply per inner iter)
3. Marker comment present for drift detection
4. ENV opt-in default OFF
5. Numerical drift bounded for typical attention scales

Mathematical identity:
    b_o * scale + dot * scale  ==  (b_o + dot) * scale     (distributive)

In fp32, this can differ by ≤1 ULP per element. For attention with
scale = 1/sqrt(d_head) (small constant), both forms are numerically
equivalent within rounding error.

Reference: vllm#41446 (zobinHuang, MI300X GDN optimization, pattern (c)).
Hardware-agnostic — Triton compiler can't auto-fuse this pattern across
the `b_o = b_o * scale + dot * scale` boundary, so explicit fold = guaranteed
1 fewer fp32 mul per `chunk_fwd_kernel_o` inner iter.
"""
from __future__ import annotations


_CONSOLIDATED = (
    "sndr.engines.vllm.patches.attention.gdn.pn29_pn298_chunk_o_consolidated"
)


def test_pn29_wiring_imports():
    """PN29 wiring (now consolidated) imports cleanly and exposes the
    PN29 marker + apply()."""
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    assert hasattr(mod, "apply")
    assert hasattr(mod, "GENESIS_PN29_MARKER")


def test_pn29_dispatcher_registry():
    """PN29 was consolidated into the PN298 registry entry; its env flag
    survives as a recognized alias so existing YAMLs keep working."""
    from sndr.dispatcher import PATCH_REGISTRY
    # PN29 no longer a standalone entry — merged into PN298.
    assert "PN29" not in PATCH_REGISTRY
    assert "PN298" in PATCH_REGISTRY
    e = PATCH_REGISTRY["PN298"]
    assert e["default_on"] is False
    # PN29's vllm#41446 backport provenance is carried on the merged entry.
    assert e["upstream_pr"] == 41446
    # Both flags recognized: PN298 primary + PN29 alias.
    assert e["env_flag"] == "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS"
    assert "GENESIS_ENABLE_PN29_GDN_SCALE_FOLD" in e.get("env_flag_aliases", [])
    # The merged entry must NOT carry requires_patches=['PN296'] at the
    # entry level (that would over-gate the version-agnostic PN29 scale-fold;
    # the PN296 precondition lives inside the pn298 sub-patch's injected code).
    assert not e.get("requires_patches")


def test_pn29_skips_when_env_off(monkeypatch):
    """When the PN29 env flag is OFF (and PN298 OFF), the consolidated
    apply() returns 'skipped' for the scale-fold sub-patch."""
    monkeypatch.delenv("GENESIS_ENABLE_PN29_GDN_SCALE_FOLD", raising=False)
    monkeypatch.delenv("SNDR_ENABLE_PN29_GDN_SCALE_FOLD", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS", raising=False)
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    status, reason = mod.apply()
    assert status == "skipped"
    assert "default OFF" in reason or "GENESIS_ENABLE_PN29_GDN_SCALE_FOLD" in reason


def test_pn29_scale_fold_gate_independent_of_pn298(monkeypatch):
    """The PN29 scale-fold sub-patch is gated by ITS OWN flag, independent
    of PN298 — and is NOT transitively gated on PN296."""
    monkeypatch.delenv("SNDR_ENABLE_PN29_GDN_SCALE_FOLD", raising=False)
    monkeypatch.delenv("SNDR_DISABLE_PN29_GDN_SCALE_FOLD", raising=False)
    monkeypatch.delenv("GENESIS_DISABLE_PN29_GDN_SCALE_FOLD", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_PN296_ARCH_PROFILE_INIT", raising=False)
    monkeypatch.setenv("GENESIS_ENABLE_PN29_GDN_SCALE_FOLD", "1")
    monkeypatch.delenv("GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS", raising=False)
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    # PN29 gate fires even with PN296 unset (no transitive PN296 dependency).
    assert mod._pn29_enabled() is True
    assert mod._pn298_enabled() is False


def test_pn29_anchor_text_matches_upstream():
    """PN29 anchor matches exact upstream chunk_o.py:137 line (verbatim in
    the consolidated module)."""
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    PN29_ANCHOR, PN29_REPLACEMENT = mod.PN29_ANCHOR, mod.PN29_REPLACEMENT
    # Anchor: the EXACT current upstream line
    assert "b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale" in PN29_ANCHOR
    # Replacement: scale-fold form
    assert "b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale" in PN29_REPLACEMENT
    # Drift marker
    assert "Genesis PN29" in PN29_REPLACEMENT
    assert "vllm#41446" in PN29_REPLACEMENT


def test_pn29_marker_string_unique():
    """PN29 marker is non-trivial for drift detection (preserved verbatim
    on the consolidated module)."""
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    GENESIS_PN29_MARKER = mod.GENESIS_PN29_MARKER
    assert "PN29" in GENESIS_PN29_MARKER
    assert len(GENESIS_PN29_MARKER) > 30


def test_pn29_register_in_apply_all():
    """PN29 boot-log label remains registered via @register_patch in
    apply_all (now delegating to the consolidated module).

    Match the PN29 label precisely — a bare ``"PN29" in n`` substring test
    would also match the sibling ``"PN298 ..."`` label (PN29 ⊂ PN298)."""
    import sndr.apply._per_patch_dispatch  # noqa: F401  — trigger registration
    from sndr.apply import PATCH_REGISTRY as APPLY_REGISTRY
    names = [name for name, _ in APPLY_REGISTRY]
    pn29 = [n for n in names if n.startswith("PN29 ")]
    assert len(pn29) == 1, f"PN29 not registered, names: {names[:5]}"


# ─────────────────────────────────────────────────────────────────
# Numerical equivalence tests (pure Python, no Triton needed)
# ─────────────────────────────────────────────────────────────────


def test_pn29_numerical_equivalence_typical_attention():
    """Scale-fold preserves attention output within 1 ULP for typical inputs.

    Attention scale = 1/sqrt(d_head) ~= 0.088 for d=128. Magnitudes:
    - b_o accumulator: ~[-10, 10] after several tl.dot accumulations
    - dot result: ~[-100, 100] after Q@K
    - scale * (b_o + dot): ~[-10, 10] expected output range
    """
    import torch

    torch.manual_seed(42)
    BT, BV = 64, 128
    scale = 1.0 / (128 ** 0.5)
    b_o = torch.randn(BT, BV, dtype=torch.float32) * 5.0
    dot = torch.randn(BT, BV, dtype=torch.float32) * 50.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale

    # Max abs diff per element
    max_abs_diff = (original - folded).abs().max().item()
    # Relative diff
    rel_diff = (original - folded).abs().max() / (original.abs().max() + 1e-10)

    # IEEE 754 fp32: (a*c)+(b*c) vs (a+b)*c can differ by 1-2 ULPs.
    # For our magnitudes (output ~50, scale ~0.088): 1 ULP ~= 6e-6.
    assert max_abs_diff < 1e-4, f"max_abs_diff = {max_abs_diff} too large"
    assert rel_diff.item() < 1e-5, f"rel_diff = {rel_diff.item()} too large"


def test_pn29_numerical_equivalence_extreme_scale():
    """Extreme scale (small d_head, e.g. MLA) — drift still bounded."""
    import torch

    torch.manual_seed(0)
    BT, BV = 64, 64
    scale = 1.0 / (32 ** 0.5)  # d_head=32
    b_o = torch.randn(BT, BV, dtype=torch.float32) * 100.0
    dot = torch.randn(BT, BV, dtype=torch.float32) * 1000.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale

    rel_diff = (original - folded).abs().max() / (original.abs().max() + 1e-10)
    assert rel_diff.item() < 1e-5, (
        f"Extreme magnitudes: rel_diff = {rel_diff.item()}"
    )


def test_pn29_numerical_equivalence_zero_scale():
    """scale=0 edge case (degenerate): both forms produce zero."""
    import torch

    torch.manual_seed(7)
    b_o = torch.randn(64, 128, dtype=torch.float32)
    dot = torch.randn(64, 128, dtype=torch.float32)
    scale = 0.0

    original = b_o * scale + dot * scale
    folded = (b_o + dot) * scale
    assert torch.equal(original, folded)
    assert original.abs().max().item() == 0.0


def test_pn29_idempotency_via_marker():
    """Re-applying PN29 doesn't double-patch (marker check)."""
    import importlib
    mod = importlib.import_module(_CONSOLIDATED)
    GENESIS_PN29_MARKER = mod.GENESIS_PN29_MARKER
    # The TextPatcher uses the marker comment to detect already-applied state.
    # Re-application should be no-op. (Tested in TextPatcher integration; here
    # we just verify the marker has the canonical form.)
    assert GENESIS_PN29_MARKER.startswith("Genesis PN29")
