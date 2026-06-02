# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN118 v2 — md5+full-file PoC (turboquant_attn.py scope).

Sibling to ``test_pn118_v2_md5_workspace.py`` which covers the other
pn118 target file. Together the two v2 patches replace pn118's
anchor-based coverage of its full 2-file scope with md5+full-file
replacements — one v2 patch per target.

Drift finding documented in the patch module: pn118's
``TQ_ANCHOR_INIT_OLD`` does not match current upstream at our pin,
so pn118 silently no-ops on that anchor. The md5+full-file pattern
prevents this silent partial-apply — md5 guards against ANY drift,
not just the per-anchor view.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

FIXTURE_PRE = (
    Path(__file__).parent
    / "fixtures"
    / "pn118_v2_md5_turboquant_attn_pre_patch.py.txt"
)
FIXTURE_POST = (
    Path(__file__).parent
    / "fixtures"
    / "pn118_v2_md5_turboquant_attn_post_patch.py.txt"
)


def _expected_pre_md5() -> str:
    return hashlib.md5(FIXTURE_PRE.read_bytes()).hexdigest()


def test_pn118_v2_md5_turboquant_attn_constant_matches_fixture():
    """The module's PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5 constant must
    equal the md5 of the bundled pre-patch fixture (rig-extracted
    upstream turboquant_attn.py at our PROD pin)."""
    from vllm.sndr_core.integrations.attention.turboquant.pn118_v2_md5_turboquant_attn import (  # noqa: E501
        PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5,
    )
    assert PN118_V2_MD5_TQ_ATTN_PRE_PATCH_MD5 == _expected_pre_md5()


def test_pn118_v2_md5_turboquant_attn_helper_computes_correct_hash():
    """_file_md5() returns the same hash as stdlib hashlib.md5."""
    from vllm.sndr_core.integrations.attention.turboquant.pn118_v2_md5_turboquant_attn import (  # noqa: E501
        _file_md5,
    )
    assert _file_md5(FIXTURE_PRE) == _expected_pre_md5()


def test_pn118_v2_md5_turboquant_attn_apply_skips_when_md5_mismatches(tmp_path):
    """When target md5 does not match PRE_PATCH_MD5, _do_apply() returns
    skipped (no write, target unchanged)."""
    from vllm.sndr_core.integrations.attention.turboquant import (
        pn118_v2_md5_turboquant_attn,
    )

    target = tmp_path / "turboquant_attn.py"
    original = "# not the real turboquant_attn file\n"
    target.write_text(original)

    result = pn118_v2_md5_turboquant_attn._do_apply(target)
    assert result.status == "skipped"
    assert "md5 mismatch" in result.reason.lower()
    assert target.read_text() == original


def test_pn118_v2_md5_turboquant_attn_apply_writes_post_patch_when_md5_matches(tmp_path):
    """When target md5 matches PRE_PATCH_MD5, _do_apply() writes
    POST_PATCH_CONTENT + marker, returns applied."""
    from vllm.sndr_core.integrations.attention.turboquant import (
        pn118_v2_md5_turboquant_attn,
    )

    target = tmp_path / "turboquant_attn.py"
    target.write_bytes(FIXTURE_PRE.read_bytes())

    result = pn118_v2_md5_turboquant_attn._do_apply(target)
    assert result.status == "applied"
    after = target.read_text()
    assert pn118_v2_md5_turboquant_attn._GENESIS_PN118_V2_TQ_ATTN_MARKER in after


def test_pn118_v2_md5_turboquant_attn_apply_idempotent_via_marker(tmp_path):
    """Second _do_apply() against an already-patched file returns
    skipped(already_applied) via marker detection — does not re-write."""
    from vllm.sndr_core.integrations.attention.turboquant import (
        pn118_v2_md5_turboquant_attn,
    )

    target = tmp_path / "turboquant_attn.py"
    target.write_text(
        pn118_v2_md5_turboquant_attn.PN118_V2_MD5_TQ_ATTN_POST_PATCH_CONTENT
    )
    pre_apply_text = target.read_text()

    result = pn118_v2_md5_turboquant_attn._do_apply(target)
    assert result.status == "skipped"
    assert "already applied" in result.reason.lower()
    assert target.read_text() == pre_apply_text
