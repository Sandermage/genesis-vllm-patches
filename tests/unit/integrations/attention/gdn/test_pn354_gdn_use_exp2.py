# SPDX-License-Identifier: Apache-2.0
"""TDD for PN354 — GDN chunked-prefill exp2 gate decay (vllm#43195 pattern).

Anchor-content + idempotency + env-gate contract, following the PN54
test pattern. Live-pin anchor uniqueness is verified server-side via
``tools/_tmp_pn354_check.py`` (counts each OLD anchor exactly once in
the running container's FLA tree).
"""
from __future__ import annotations


def _wiring():
    from sndr.engines.vllm.patches.attention.gdn import pn354_gdn_use_exp2 as M
    return M


# ── anchor content sanity ────────────────────────────────────────────────

def test_chunk_o_anchors_target_the_two_exp_sites():
    M = _wiring()
    assert "b_o = b_o * exp(b_g)[:, None]" in M.CHUNK_O_EXP_OLD
    assert "b_A = b_A * exp(b_g[:, None] - b_g[None, :])" in M.CHUNK_O_EXP_OLD
    # Dual-branch replacement keeps the exp fallback verbatim
    assert "if USE_EXP2:" in M.CHUNK_O_EXP_NEW
    assert "b_o = b_o * exp2(b_g)[:, None]" in M.CHUNK_O_EXP_NEW
    assert "b_o = b_o * exp(b_g)[:, None]" in M.CHUNK_O_EXP_NEW
    # Signature gains the constexpr; wrapper gains default-False kwarg
    assert "USE_EXP2: tl.constexpr," in M.CHUNK_O_SIG_NEW
    assert "USE_EXP2" not in M.CHUNK_O_SIG_OLD
    assert "use_exp2: bool = False," in M.CHUNK_O_WRAP_SIG_NEW
    assert "USE_EXP2=use_exp2," in M.CHUNK_O_LAUNCH_NEW
    assert M.CHUNK_O_IMPORT_OLD == "from .op import exp\n"
    assert M.CHUNK_O_IMPORT_NEW == "from .op import exp, exp2\n"


def test_kkt_anchors_target_the_exp_site():
    M = _wiring()
    assert "b_A = b_A * exp(b_g_diff)" in M.KKT_EXP_OLD
    assert "b_A = b_A * exp2(b_g_diff)" in M.KKT_EXP_NEW
    assert "b_A = b_A * exp(b_g_diff)" in M.KKT_EXP_NEW  # fallback kept
    assert "USE_EXP2: tl.constexpr," in M.KKT_SIG_NEW
    assert "use_exp2: bool = False," in M.KKT_WRAP_SIG_NEW
    assert "USE_EXP2=use_exp2," in M.KKT_LAUNCH_NEW


def test_wy_fast_uses_raw_tl_exp2():
    # wy_fast.py does NOT import from .op — the new branch must use
    # tl.exp2 (raw), not the .op alias.
    M = _wiring()
    assert "b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))" in M.WY_EXP_OLD
    assert "tl.exp2(tl.load(p_g, boundary_check=(0,)))" in M.WY_EXP_NEW
    assert "from .op import" not in M.WY_EXP_NEW
    assert "USE_EXP2: tl.constexpr," in M.WY_SIG_NEW
    assert "use_exp2: bool = False," in M.WY_WRAP_SIG_NEW
    assert "USE_EXP2=use_exp2," in M.WY_LAUNCH_NEW


def test_chunk_dispatcher_prescale_and_conditional_kwargs():
    M = _wiring()
    # Pre-scale engages ONLY under the runtime flag, after the cumsum
    assert "g = chunk_local_cumsum(" in M.CHUNK_CUMSUM_OLD
    assert "if _GENESIS_PN354_USE_EXP2:" in M.CHUNK_CUMSUM_NEW
    assert "g = g * _GENESIS_PN354_RCP_LN2" in M.CHUNK_CUMSUM_NEW
    # Anchor discipline: must NOT overlap the PN59 dispatch text above
    assert "PN59" not in M.CHUNK_CUMSUM_OLD
    assert "pass  # logger import failed" not in M.CHUNK_CUMSUM_OLD
    # Env read ONCE at module import scope, not per-call
    assert "GENESIS_ENABLE_PN354_GDN_USE_EXP2" in M.CHUNK_IMPORTS_NEW
    assert "RCP_LN2" in M.CHUNK_IMPORTS_NEW
    # All four consumer calls thread the conditional kwargs splat
    for new in (
        M.CHUNK_KKT_CALL_NEW,
        M.CHUNK_WY_CALL_NEW,
        M.CHUNK_FWD_H_CALL_NEW,
        M.CHUNK_FWD_O_CALL_NEW,
    ):
        assert "**_GENESIS_PN354_KW," in new
    # Flag-off bit-identity: the splat dict is EMPTY when env unset —
    # no use_exp2 kwarg is passed at all
    assert (
        '_GENESIS_PN354_KW = {"use_exp2": True} if _GENESIS_PN354_USE_EXP2 else {}'
        in M.CHUNK_IMPORTS_NEW
    )


def test_replacements_carry_pn354_marker():
    M = _wiring()
    for name in (
        "CHUNK_O_EXP_NEW", "KKT_EXP_NEW", "WY_EXP_NEW",
        "CHUNK_IMPORTS_NEW", "CHUNK_CUMSUM_NEW",
    ):
        assert "PN354" in getattr(M, name), f"{name} missing PN354 marker"


def test_old_anchors_clean_of_new_text():
    """Every OLD anchor must be free of PN354 text so the drift markers
    ('use_exp2' / 'USE_EXP2' / 'RCP_LN2' appearing upstream) can never
    be confused with our own anchors."""
    M = _wiring()
    for name in dir(M):
        if name.endswith("_OLD"):
            old = getattr(M, name)
            assert "use_exp2" not in old, f"{name} contains use_exp2"
            assert "USE_EXP2" not in old, f"{name} contains USE_EXP2"
            assert "RCP_LN2" not in old, f"{name} contains RCP_LN2"
            assert "PN354" not in old, f"{name} contains PN354"


# ── idempotency (PN54 pattern) ───────────────────────────────────────────

def test_idempotent_apply(tmp_path):
    from sndr.kernel.text_patch import TextPatch, TextPatcher, TextPatchResult
    M = _wiring()

    cases = [
        ("chunk_o.py", M.GENESIS_PN354_MARKER_CHUNK_O, [
            ("imp", M.CHUNK_O_IMPORT_OLD, M.CHUNK_O_IMPORT_NEW),
            ("sig", M.CHUNK_O_SIG_OLD, M.CHUNK_O_SIG_NEW),
            ("exp", M.CHUNK_O_EXP_OLD, M.CHUNK_O_EXP_NEW),
            ("wrap", M.CHUNK_O_WRAP_SIG_OLD, M.CHUNK_O_WRAP_SIG_NEW),
            ("launch", M.CHUNK_O_LAUNCH_OLD, M.CHUNK_O_LAUNCH_NEW),
        ]),
        ("chunk_scaled_dot_kkt.py", M.GENESIS_PN354_MARKER_KKT, [
            ("imp", M.KKT_IMPORT_OLD, M.KKT_IMPORT_NEW),
            ("sig", M.KKT_SIG_OLD, M.KKT_SIG_NEW),
            ("exp", M.KKT_EXP_OLD, M.KKT_EXP_NEW),
            ("wrap", M.KKT_WRAP_SIG_OLD, M.KKT_WRAP_SIG_NEW),
            ("launch", M.KKT_LAUNCH_OLD, M.KKT_LAUNCH_NEW),
        ]),
        ("wy_fast.py", M.GENESIS_PN354_MARKER_WY, [
            ("sig", M.WY_SIG_OLD, M.WY_SIG_NEW),
            ("exp", M.WY_EXP_OLD, M.WY_EXP_NEW),
            ("wrap", M.WY_WRAP_SIG_OLD, M.WY_WRAP_SIG_NEW),
            ("launch", M.WY_LAUNCH_OLD, M.WY_LAUNCH_NEW),
        ]),
        ("chunk.py", M.GENESIS_PN354_MARKER_CHUNK, [
            ("flag", M.CHUNK_IMPORTS_OLD, M.CHUNK_IMPORTS_NEW),
            ("prescale", M.CHUNK_CUMSUM_OLD, M.CHUNK_CUMSUM_NEW),
            ("kkt", M.CHUNK_KKT_CALL_OLD, M.CHUNK_KKT_CALL_NEW),
            ("wy", M.CHUNK_WY_CALL_OLD, M.CHUNK_WY_CALL_NEW),
            ("fwdh", M.CHUNK_FWD_H_CALL_OLD, M.CHUNK_FWD_H_CALL_NEW),
            ("fwdo", M.CHUNK_FWD_O_CALL_OLD, M.CHUNK_FWD_O_CALL_NEW),
        ]),
    ]
    for fname, marker, subs in cases:
        target = tmp_path / fname
        target.write_text(
            "# header\n" + "\n".join(old for _, old, _ in subs) + "\n# tail\n"
        )
        patcher = TextPatcher(
            patch_name=fname,
            target_file=str(target),
            marker=marker,
            sub_patches=[
                TextPatch(name=f"pn354_{n}", anchor=old, replacement=new,
                          required=True)
                for n, old, new in subs
            ],
        )
        r1, f1 = patcher.apply()
        assert r1 == TextPatchResult.APPLIED, f"{fname} 1st apply: {f1}"
        body1 = target.read_text()
        assert "PN354" in body1
        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT, f"{fname} 2nd apply"
        assert target.read_text() == body1


def test_markers_are_pairwise_non_substrings():
    """Per-file markers must not be substrings of each other — the
    Layer 2 idempotency check is a plain `marker in content` scan."""
    M = _wiring()
    markers = [
        M.GENESIS_PN354_MARKER_CHUNK_O,
        M.GENESIS_PN354_MARKER_KKT,
        M.GENESIS_PN354_MARKER_WY,
        M.GENESIS_PN354_MARKER_CHUNK,
    ]
    assert len(set(markers)) == 4
    for a in markers:
        for b in markers:
            if a is not b:
                assert a not in b, f"marker collision: {a!r} ⊂ {b!r}"


# ── env gate contract ────────────────────────────────────────────────────

def test_env_flag_default_off(monkeypatch):
    from sndr.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN354_GDN_USE_EXP2", raising=False)
    decision, reason = should_apply("PN354")
    assert decision is False
    assert "opt-in" in reason.lower() or "off" in reason.lower()


def test_apply_skips_without_env(monkeypatch):
    M = _wiring()
    monkeypatch.delenv("GENESIS_ENABLE_PN354_GDN_USE_EXP2", raising=False)
    status, detail = M.apply()
    assert status == "skipped"
    assert "GENESIS_ENABLE_PN354_GDN_USE_EXP2" in detail


def test_driver_reads_same_flag_once_at_import():
    """The PN59 streaming driver must gate on the SAME env flag, read at
    module scope (not per-call), and use vllm's RCP_LN2 with the exact
    fp32 fallback constant. Source-text check (no import — the driver
    needs torch, absent in the torch-less unit-test env)."""
    import pathlib

    import sndr

    src = (
        pathlib.Path(sndr.__file__).parent
        / "engines" / "vllm" / "kernels_legacy" / "streaming_gdn_driver.py"
    ).read_text(encoding="utf-8")
    assert "GENESIS_ENABLE_PN354_GDN_USE_EXP2" in src
    assert "_PN354_USE_EXP2" in src
    assert "1.4426950216" in src  # vllm RCP_LN2 fp32 value (fallback)
    # Both pipeline paths thread the conditional kwargs
    assert src.count("**_PN354_KW") >= 8, (
        "expected >=8 threaded call sites (4 vanilla + 4 windowed)"
    )
    # Pre-scale present in both paths
    assert src.count("* _PN354_RCP_LN2") == 2
