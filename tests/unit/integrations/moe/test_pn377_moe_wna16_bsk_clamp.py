# SPDX-License-Identifier: Apache-2.0
"""PN377 — moe_wna16 BLOCK_SIZE_K legality clamp (vendor of OPEN vllm#44563).

Contract pinned here (TDD, written before the implementation):
  1. RED proof: the pristine pin-form heuristic returns kernel-illegal
     BLOCK_SIZE_K for group_size=32 (#36008 reporter decode shape:
     BLOCK_SIZE_K=512, ratio 16) — the deterministic warmup abort PN377
     exists to fix, and ``check_block_config_legality`` must flag it.
  2. apply() on the pin-form target installs the clamp BEFORE the
     ``_ensure_block_size_k_divisible`` step; the patched text compiles;
     a second apply() is idempotent (marker short-circuit).
  3. PR #44563's CPU-only invariant test, ported: the post-patch
     heuristic keeps ``BLOCK_SIZE_K // group_size`` in {1, 2, 4, 8}
     across the PR's full sweep (group_size x size_k x size_n x
     num_valid_tokens x num_experts), the reporter decode shape, and
     leaves an already-tuned config untouched.
  4. Genesis extra: boot-time legality assert for the actual model grid
     (loud ERROR log + apply() message annotation instead of a cryptic
     warmup abort). Grid discovery is best-effort: unavailable config
     skips the check quietly; a kernel-illegal grid (e.g. size_k not
     divisible by group_size) fires the loud error even with the clamp
     applied.
  5. apply() self-skips on #44563's merged form via drift markers, and
     the merged form is verified already-legal (drift-skip is safe).
  6. Drift markers never collide with PN377's own emitted text or its
     Layer-6 marker line (tools/lint_drift_markers.py contract).
  7. Pristine pin invariants (opportunistic, skipped without the pin
     tree): anchor unique, drift markers absent, embedded fixture is
     byte-identical to the pin segment, and P24 anchors survive a PN377
     splice (and vice versa) — same-file non-collision proof.

Fixture note: PIN_HEURISTIC_SRC is the byte-exact
``_ensure_block_size_k_divisible`` + ``get_moe_wna16_block_config``
segment of ``model_executor/layers/fused_moe/fused_moe.py`` (lines
1079-1190 at pin g303916e93 / 0.22.1rc1.dev259, segment md5
051ca07439ae4a89f6ca3d72cb08ce4e). Both functions are pure Python at
this pin (no torch / no triton), so the ported PR test executes them
directly via the module's own ast-extraction helper — the
exec-patched-text technique from the 2026-06-11 roadmap (Theme C).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.engines.vllm.patches.moe import p24_moe_tune as p24
from sndr.engines.vllm.patches.moe import pn377_moe_wna16_bsk_clamp as m

# ── Fixtures ─────────────────────────────────────────────────────────

PIN_HEURISTIC_SRC = '''def _ensure_block_size_k_divisible(
    size_k: int, block_size_k: int, group_size: int
) -> int:
    """Ensure block_size_k is a divisor of size_k and divisible by group_size.

    This ensures BLOCK_SIZE_K compatibility with MoeWNA16 CUDA kernel which
    requires size_k % BLOCK_SIZE_K == 0 and BLOCK_SIZE_K % group_size == 0.

    Args:
        size_k: The size_k dimension that must be divisible by result.
        block_size_k: Preferred block size (will be adjusted if needed).
        group_size: The result must be divisible by this.

    Returns:
        A valid BLOCK_SIZE_K that divides size_k and is divisible by group_size.
    """
    # Fast path: already valid
    if size_k % block_size_k == 0 and block_size_k % group_size == 0:
        return block_size_k

    # Find the largest value that:
    # 1. Divides size_k (size_k % candidate == 0)
    # 2. Is divisible by group_size (candidate % group_size == 0)
    # 3. Is <= block_size_k (prefer smaller values close to block_size_k)
    #
    # Strategy: Search from min(block_size_k, size_k) down to group_size,
    # stepping by group_size to ensure divisibility by group_size
    max_search = min(block_size_k, size_k)
    start = (max_search // group_size) * group_size
    for candidate in range(start, group_size - 1, -group_size):
        if size_k % candidate == 0:
            return candidate

    # Fallback: if group_size divides size_k, use it
    # This should always be true with correct group_size configuration
    if size_k % group_size == 0:
        return group_size

    # This should not happen with correct group_size, but ensure divisibility
    return size_k


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        # Ensure BLOCK_SIZE_K is a divisor of size_k for CUDA kernel compatibility
        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}

'''

# The single line both the PR and PN377 anchor in front of.
_DIVIS_COMMENT = (
    "        # Ensure BLOCK_SIZE_K is a divisor of size_k"
    " for CUDA kernel compatibility\n"
)

# #44563's exact added lines (reconstruction of the merged form).
UPSTREAM_44563_CLAMP = (
    "        # CUDA moe_wna16_gemm only supports BLOCK_SIZE_K // group_size in\n"
    "        # {1, 2, 4, 8}; the heuristic above can overshoot (e.g. 512 // 32 = 16).\n"
    "        # Clamp to at most 8 groups per block row before enforcing divisibility.\n"
    "        max_block_size_k = group_size * 8\n"
    "        if block_size_k > max_block_size_k:\n"
    "            block_size_k = max_block_size_k\n"
    "\n"
)

MERGED_HEURISTIC_SRC = PIN_HEURISTIC_SRC.replace(
    _DIVIS_COMMENT, UPSTREAM_44563_CLAMP + _DIVIS_COMMENT
)

PIN_FILE = Path(
    "/private/tmp/candidate_pin_current/vllm/model_executor/layers/"
    "fused_moe/fused_moe.py"
)

# PR #44563's CPU sweep dimensions (test_moe_wna16_block_config.py).
GROUP_SIZES = (32, 64, 128)
SIZE_KS = (512, 1024, 2048, 4096)
SIZE_NS = (512, 1024, 2048)
NUM_VALID_TOKENS = (1, 8, 16, 64, 256, 4096)
NUM_EXPERTS = (8, 128, 256)
TOP_K = 8

# Reporter decode shape from #36008 / the PR (gate_up gemm of a
# group_size=32 GPTQ 4-bit MoE in single-token decode).
REPORTER_SHAPE = dict(
    config={},
    use_moe_wna16_cuda=True,
    num_valid_tokens=8,
    size_k=2048,
    size_n=1024,
    num_experts=256,
    group_size=32,
    real_top_k=8,
    block_size_m=1,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _write_fake(tmp_path, text):
    fake = tmp_path / "fused_moe.py"
    fake.write_text(text, encoding="utf-8")
    return fake


def _install_fake(tmp_path, monkeypatch, text):
    fake = _write_fake(tmp_path, text)
    monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: str(fake))
    return fake


def _heuristic_from_text(tmp_path, text, name="heuristic_only.py"):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    fn = m.load_block_config_heuristic(str(f))
    assert fn is not None, "heuristic functions not extractable from fixture"
    return fn


# ── 1. RED proof: the bug exists in the pin form ─────────────────────


class TestPinFormIsIllegal:
    def test_reporter_decode_shape_overshoots_to_ratio_16(self, tmp_path):
        """Unpatched pin heuristic returns BLOCK_SIZE_K=512 for the
        reporter's group_size=32 decode shape — ratio 16, which makes
        moe_wna16_gemm abort ('BLOCK_SIZE_K // group_size must be one
        of [1, 2, 4, 8]')."""
        fn = _heuristic_from_text(tmp_path, PIN_HEURISTIC_SRC)
        cfg = fn(**REPORTER_SHAPE)
        assert cfg["BLOCK_SIZE_K"] == 512
        assert cfg["BLOCK_SIZE_K"] // 32 == 16  # the literal #36008 crash

    def test_legality_check_flags_pin_form(self, tmp_path):
        fn = _heuristic_from_text(tmp_path, PIN_HEURISTIC_SRC)
        violations = m.check_block_config_legality(
            fn,
            group_size=32,
            gemm_shapes=[(2048, 1024)],
            num_experts_candidates=[256, 1024],
            top_k=8,
        )
        assert violations, "pin-form gs=32 grid must be flagged illegal"
        assert any("BLOCK_SIZE_K=512" in v for v in violations)


# ── 2. Patcher shape ─────────────────────────────────────────────────


class TestPatcherShape:
    def test_patcher_built_with_required_clamp_sub(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        patcher = m._make_patcher()
        assert patcher is not None
        names = [sp.name for sp in patcher.sub_patches]
        assert names == ["pn377_bsk_clamp"]
        assert patcher.sub_patches[0].required is True
        assert "PN377" in patcher.marker

    def test_module_tracks_pr_44563(self):
        assert "44563" in (m.__doc__ or "")
        assert "PN377" in m.GENESIS_PN377_MARKER

    def test_patcher_none_when_target_missing(self, monkeypatch):
        monkeypatch.setattr(m, "resolve_vllm_file", lambda rel: None)
        assert m._make_patcher() is None


# ── 3. Apply semantics + the ported PR CPU test ──────────────────────


@pytest.fixture(scope="class")
def patched_heuristic(request, tmp_path_factory):
    """Apply PN377 once to a pin-form fake, return the post-patch
    heuristic (exec-patched-text technique)."""
    tmp_path = tmp_path_factory.mktemp("pn377_sweep")
    fake = _write_fake(tmp_path, PIN_HEURISTIC_SRC)
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setattr(m, "resolve_vllm_file", lambda rel: str(fake))
    status, reason = m.apply()
    assert status == "applied", reason
    return m.load_block_config_heuristic(str(fake))


class TestApply:
    def test_apply_installs_clamp_before_divisibility_step(
        self, tmp_path, monkeypatch
    ):
        fake = _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        status, reason = m.apply()
        assert status == "applied", reason
        out = fake.read_text(encoding="utf-8")
        assert "_g_pn377_bsk_cap" in out
        # The divisibility step survives, exactly once, AFTER the clamp.
        assert out.count(_DIVIS_COMMENT) == 1
        assert out.index("_g_pn377_bsk_cap") < out.index(_DIVIS_COMMENT)
        compile(out, str(fake), "exec")

    def test_second_apply_is_idempotent(self, tmp_path, monkeypatch):
        _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        first, _ = m.apply()
        assert first == "applied"
        second, reason = m.apply()
        assert second == "skipped"
        assert "already applied" in reason

    def test_install_disable_env_skips(self, tmp_path, monkeypatch):
        fake = _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        monkeypatch.setenv("GENESIS_ENABLE_PN377_MOE_WNA16_BSK_CLAMP", "0")
        status, reason = m.apply()
        assert status == "skipped"
        assert fake.read_text(encoding="utf-8") == PIN_HEURISTIC_SRC

    def test_self_skips_on_44563_merged_form(self, tmp_path, monkeypatch):
        fake = _install_fake(tmp_path, monkeypatch, MERGED_HEURISTIC_SRC)
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason
        assert fake.read_text(encoding="utf-8") == MERGED_HEURISTIC_SRC


class TestPortedPrSweep:
    """Port of tests/kernels/moe/test_moe_wna16_block_config.py from
    vllm#44563, executed against the post-patch heuristic."""

    @pytest.mark.parametrize("group_size", GROUP_SIZES)
    @pytest.mark.parametrize("size_k", SIZE_KS)
    @pytest.mark.parametrize("size_n", SIZE_NS)
    @pytest.mark.parametrize("num_valid_tokens", NUM_VALID_TOKENS)
    @pytest.mark.parametrize("num_experts", NUM_EXPERTS)
    def test_groups_per_block_row_is_legal(
        self,
        patched_heuristic,
        group_size,
        size_k,
        size_n,
        num_valid_tokens,
        num_experts,
    ):
        block_size_m = min(16, max(1, num_valid_tokens // TOP_K))
        cfg = patched_heuristic(
            config={},
            use_moe_wna16_cuda=True,
            num_valid_tokens=num_valid_tokens,
            size_k=size_k,
            size_n=size_n,
            num_experts=num_experts,
            group_size=group_size,
            real_top_k=TOP_K,
            block_size_m=block_size_m,
        )
        block_size_k = cfg["BLOCK_SIZE_K"]
        # The moe_wna16_gemm kernel requires all three of these.
        assert block_size_k % group_size == 0
        assert size_k % block_size_k == 0
        assert block_size_k // group_size in (1, 2, 4, 8), (
            f"BLOCK_SIZE_K // group_size = {block_size_k // group_size} "
            f"(BLOCK_SIZE_K={block_size_k}, group_size={group_size})"
        )

    def test_reporter_decode_shape_is_legal_post_patch(self, patched_heuristic):
        cfg = patched_heuristic(**REPORTER_SHAPE)
        assert cfg["BLOCK_SIZE_K"] // 32 in (1, 2, 4, 8)

    def test_tuned_config_is_left_untouched(self, patched_heuristic):
        shape = dict(REPORTER_SHAPE)
        shape["config"] = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}
        assert patched_heuristic(**shape) == {}

    def test_merged_form_is_already_legal(self, tmp_path):
        """Drift-skip safety: #44563's merged form passes the same
        legality sweep, so self-skipping on it loses nothing."""
        fn = _heuristic_from_text(tmp_path, MERGED_HEURISTIC_SRC)
        violations = m.check_block_config_legality(
            fn,
            group_size=32,
            gemm_shapes=[(2048, 1024), (768, 2048)],
            num_experts_candidates=[256, 1024, 2048],
            top_k=8,
        )
        assert violations == []


# ── 4. Genesis extra: boot-time legality assert ──────────────────────


class TestBootLegalityCheck:
    def test_grid_unavailable_skips_quietly(self, tmp_path, monkeypatch):
        fake = _write_fake(tmp_path, PIN_HEURISTIC_SRC)
        monkeypatch.setattr(m, "_model_wna16_grid", lambda: None)
        ok, detail = m.run_boot_legality_check(str(fake))
        assert ok is None
        assert detail  # human-readable skip reason

    def test_heuristic_not_extractable_skips_quietly(self, tmp_path):
        fake = _write_fake(tmp_path, "x = 1\n")
        ok, detail = m.run_boot_legality_check(str(fake))
        assert ok is None

    def test_legal_grid_passes_post_patch(self, tmp_path, monkeypatch):
        fake = _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        status, _ = m.apply()
        assert status == "applied"
        monkeypatch.setattr(
            m,
            "_model_wna16_grid",
            lambda: {
                "group_size": 32,
                "gemm_shapes": [(2048, 1024), (768, 2048)],
                "num_experts_candidates": [256, 1024, 2048],
                "top_k": 8,
            },
        )
        ok, detail = m.run_boot_legality_check(str(fake))
        assert ok is True

    def test_illegal_grid_fires_loud_error(self, tmp_path, monkeypatch, caplog):
        """size_k=1000 is not divisible by group_size=32 — no legal
        BLOCK_SIZE_K exists, even with the clamp. The boot check must
        produce the loud, actionable ERROR instead of letting warmup
        abort with the cryptic kernel RuntimeError."""
        fake = _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        status, _ = m.apply()
        assert status == "applied"
        monkeypatch.setattr(
            m,
            "_model_wna16_grid",
            lambda: {
                "group_size": 32,
                "gemm_shapes": [(1000, 512)],
                "num_experts_candidates": [8],
                "top_k": 8,
            },
        )
        with caplog.at_level("ERROR"):
            ok, detail = m.run_boot_legality_check(str(fake))
        assert ok is False
        assert "BLOCK_SIZE_K" in detail
        assert any(
            "BOOT LEGALITY CHECK FAILED" in r.message for r in caplog.records
        )

    def test_apply_message_annotated_on_illegal_grid(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        monkeypatch.setattr(
            m,
            "_model_wna16_grid",
            lambda: {
                "group_size": 32,
                "gemm_shapes": [(1000, 512)],
                "num_experts_candidates": [8],
                "top_k": 8,
            },
        )
        status, reason = m.apply()
        assert status == "applied"  # the text patch itself succeeded
        assert "BOOT LEGALITY CHECK FAILED" in reason


# ── 5. Lint contract (tools/lint_drift_markers.py) ───────────────────


class TestDriftMarkerSelfCollision:
    def test_markers_not_substring_of_own_emitted_text(
        self, tmp_path, monkeypatch
    ):
        _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        patcher = m._make_patcher()
        marker_line = f"# [Genesis wiring marker: {patcher.marker}]\n"
        for dm in patcher.upstream_drift_markers:
            for sp in patcher.sub_patches:
                assert dm not in sp.replacement, (
                    f"drift marker {dm!r} collides with {sp.name} "
                    f"replacement — would false-fire Layer 3 on our own text"
                )
            assert dm not in marker_line

    def test_every_marker_matches_44563_merged_form(
        self, tmp_path, monkeypatch
    ):
        """Each drift marker must be a substring of the merged-form
        fixture — a marker that can never fire is the PN367-v1 bug
        class."""
        _install_fake(tmp_path, monkeypatch, PIN_HEURISTIC_SRC)
        patcher = m._make_patcher()
        assert patcher.upstream_drift_markers
        for dm in patcher.upstream_drift_markers:
            assert dm in MERGED_HEURISTIC_SRC, (
                f"drift marker {dm!r} does not occur in #44563's merged "
                f"form — it could never fire"
            )


# ── 6. Pristine pin invariants (opportunistic) ───────────────────────


@pytest.mark.skipif(
    not PIN_FILE.is_file(),
    reason="pristine pin tree not present on this machine",
)
class TestAgainstPristinePin:
    def test_embedded_fixture_matches_pin_segment(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        assert PIN_HEURISTIC_SRC in src, (
            "embedded fixture drifted from the pin segment — re-extract "
            "lines 1079-1190 of fused_moe.py"
        )

    def test_anchor_unique_and_markers_absent(self):
        src = PIN_FILE.read_text(encoding="utf-8")
        assert src.count(m.PN377_CLAMP_OLD) == 1
        assert m.PN377_CLAMP_NEW not in src
        for dm in m._UPSTREAM_DRIFT_MARKERS:
            assert dm not in src

    def test_p24_anchors_survive_pn377_splice(self):
        """Same-file non-collision: applying PN377's splice must leave
        both P24 anchors intact (count==1), and applying both P24
        splices must leave PN377's anchor intact."""
        src = PIN_FILE.read_text(encoding="utf-8")
        # PN377 first, then P24 anchors still match.
        after_pn377 = src.replace(m.PN377_CLAMP_OLD, m.PN377_CLAMP_NEW)
        assert after_pn377.count(p24._OLD_FP8_CFG) == 1
        assert after_pn377.count(p24._OLD_GEN_CFG) == 1
        # P24 first, then PN377's anchor still matches.
        after_p24 = src.replace(p24._OLD_FP8_CFG, p24._NEW_FP8_CFG).replace(
            p24._OLD_GEN_CFG, p24._NEW_GEN_CFG
        )
        assert after_p24.count(m.PN377_CLAMP_OLD) == 1
