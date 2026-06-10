# SPDX-License-Identifier: Apache-2.0
"""TDD for Layer 4 read-only mount preflight — T1.5 / audit §17.4.

Closes club-3090 #47: when an operator bind-mounts the SNDR Core tree
read-only into a vllm container, text-patcher previously silently
no-op'd because the eventual write phase raised PermissionError, which
got mapped to a generic FAILED with reason='write_error'. The new
Layer 4 preflight catches it BEFORE the splice work, surfaces a
structured SKIPPED with reason='read_only_mount', and includes
remediation hints.
"""
from __future__ import annotations

import os
import stat

import pytest


@pytest.fixture
def fake_source(tmp_path):
    """Create a writable fake source file (default state)."""
    path = tmp_path / "fake_module.py"
    path.write_text(
        "# fake module\n"
        "def foo():\n"
        "    raise NotImplementedError('no hybrid')\n"
    )
    return str(path)


def _build_patcher(target: str, marker: str = "RO_TEST_MARKER"):
    from sndr.kernel.text_patch import TextPatch, TextPatcher
    return TextPatcher(
        patch_name="ro-test",
        target_file=target,
        marker=marker,
        sub_patches=[
            TextPatch(
                name="edit_foo",
                anchor="    raise NotImplementedError('no hybrid')\n",
                replacement="    return 42\n",
                required=True,
            ),
        ],
    )


class TestLayer4ReadOnlyMount:
    def test_writable_file_applies_normally(self, fake_source):
        from sndr.kernel.text_patch import TextPatchResult
        p = _build_patcher(fake_source)
        result, failure = p.apply()
        assert result == TextPatchResult.APPLIED
        assert failure is None

    def test_read_only_file_returns_skipped_with_structured_reason(
        self, fake_source
    ):
        from sndr.kernel.text_patch import TextPatchResult
        # chmod 0o444 doesn't enforce read-only for the root user (and
        # CI containers run as root); skip when we'd be a no-op probe.
        if os.geteuid() == 0:
            pytest.skip("root bypasses chmod; read-only test requires unprivileged user")
        # Strip write bits (chmod 0o444 — read-only for everyone)
        os.chmod(fake_source, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            p = _build_patcher(fake_source)
            result, failure = p.apply()
            assert result == TextPatchResult.SKIPPED, (
                "Layer 4 must surface a SKIPPED, not a late FAILED"
            )
            assert failure is not None
            assert failure.reason == "read_only_mount"
            # Detail must mention the file path so operators can act on it
            assert fake_source in failure.detail
            # Detail must include a remediation hint
            assert ("overlay" in failure.detail.lower()
                    or "rebind" in failure.detail.lower())
        finally:
            # Restore so pytest can clean up tmp_path
            os.chmod(fake_source, stat.S_IRUSR | stat.S_IWUSR)

    def test_idempotent_short_circuits_before_writability(
        self, fake_source
    ):
        """If marker is already present, Layer 2 returns IDEMPOTENT
        BEFORE Layer 4 fires — read-only mount with already-patched
        file is fine, not an error."""
        from sndr.kernel.text_patch import TextPatchResult
        # Pre-patch the file so marker is present
        marker = "PRE_PATCHED_MARKER"
        with open(fake_source, "w") as f:
            f.write(
                f"# [Genesis wiring marker: {marker}]\n"
                "# fake module\n"
                "def foo():\n"
                "    return 42\n"
            )
        # Now make it read-only
        os.chmod(fake_source, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            p = _build_patcher(fake_source, marker=marker)
            result, failure = p.apply()
            assert result == TextPatchResult.IDEMPOTENT
            assert failure is None
        finally:
            os.chmod(fake_source, stat.S_IRUSR | stat.S_IWUSR)

    def test_missing_file_returns_skipped_target_missing_not_ro(
        self, tmp_path
    ):
        """Layer 1 must catch missing file before Layer 4 fires."""
        from sndr.kernel.text_patch import TextPatchResult
        p = _build_patcher(str(tmp_path / "does_not_exist.py"))
        result, failure = p.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure is not None
        # Must surface the missing-file reason, not a misleading
        # writability message.
        assert failure.reason == "target_file_missing"
