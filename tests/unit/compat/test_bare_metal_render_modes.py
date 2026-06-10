# SPDX-License-Identifier: Apache-2.0
"""S0.3 audit closure (2026-05-08 noonghunna P2-2): tests for bare-metal
render modes (wheel / dev / dev_legacy).

Audit finding: previous bare-metal renderer always emitted
``pip install --quiet -e {plugin_src} 2>/dev/null || true`` which silently
masked install failures and was unsafe for production paths. Wave 7 / S0.3
introduces three explicit modes so production paths fail fast on missing
wheel and dev paths surface install errors.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.compat.model_config_cli import _render_bare_metal
from sndr.model_configs.registry import get as get_config


# Phase 10 (2026-06-01): V1 sunset — module fixture loads V1
# `a5000-2x-35b-prod` because render content is mode-driven not config-
# driven (any cfg works); when the V1 file retires the entire module
# skips. Migrating to V2 alias is feasible (V2 composes a V1 ModelConfig)
# but defers — keeping V1 fixture preserves byte-identical pre-sunset
# behavior for the mode contract.
_V1_35B_BARE = (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
                / "model_configs" / "builtin" / "a5000-2x-35b-prod.yaml")
pytestmark = pytest.mark.skipif(
    not _V1_35B_BARE.is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)"
           " — bare-metal render mode tests are V1-bound and skip",
)


@pytest.fixture(scope="module")
def cfg():
    """Pick any builtin config — render content is mode-driven, not config-
    driven, so any cfg is fine for these tests."""
    cfg = get_config("a5000-2x-35b-prod")
    assert cfg is not None
    return cfg


def _executable_pip_install_lines(rendered: str) -> list[str]:
    """Return non-comment, non-echo, non-operator-help lines that actually
    run ``pip install``. Skips echoed instructions like
    ``echo "Install the wheel:  pip install ..."``."""
    out = []
    for line in rendered.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "pip install" not in stripped:
            continue
        # Skip echo / operator-help lines (they are instructional, not executable)
        if stripped.startswith("echo "):
            continue
        out.append(stripped)
    return out


class TestWheelMode:
    def test_wheel_mode_does_not_emit_pip_install(self, cfg):
        out = _render_bare_metal(cfg, mode="wheel")
        runners = _executable_pip_install_lines(out)
        assert runners == [], (
            "wheel mode must NOT execute pip install — operator pre-installs "
            f"the wheel. Found executable: {runners}"
        )

    def test_wheel_mode_no_silent_or_true(self, cfg):
        out = _render_bare_metal(cfg, mode="wheel")
        # Wheel mode wraps the import in a `||` block but `|| true` (the
        # silent-failure pattern) must not appear anywhere.
        assert "|| true" not in out, (
            "wheel mode must not silently mask errors with `|| true`"
        )

    def test_wheel_mode_imports_sndr_core_as_smoke(self, cfg):
        out = _render_bare_metal(cfg, mode="wheel")
        assert "import sndr" in out, (
            "wheel mode must verify the wheel is importable"
        )

    def test_wheel_mode_fails_fast_on_missing_wheel(self, cfg):
        out = _render_bare_metal(cfg, mode="wheel")
        # Hard-fail path must be present
        assert "exit 1" in out
        assert "pip install vllm-sndr-core" in out, (
            "error message must point operator at fix"
        )

    def test_wheel_mode_header_reflects_mode(self, cfg):
        out = _render_bare_metal(cfg, mode="wheel")
        assert "--mode wheel" in out


class TestDevMode:
    def test_dev_mode_emits_pip_install_e(self, cfg):
        out = _render_bare_metal(cfg, mode="dev")
        assert "pip install --quiet -e" in out

    def test_dev_mode_no_silent_or_true(self, cfg):
        out = _render_bare_metal(cfg, mode="dev")
        # Per audit closure: install errors must be visible in dev mode too.
        # Only `dev_legacy` is allowed to silence them.
        # Check that the line with `pip install -e` does NOT contain `|| true`.
        for line in out.splitlines():
            if "pip install" in line and "-e" in line:
                assert "|| true" not in line, (
                    f"dev mode must NOT mask install errors: {line!r}"
                )

    def test_dev_mode_sets_pythonpath(self, cfg):
        out = _render_bare_metal(cfg, mode="dev")
        assert 'export PYTHONPATH=' in out

    def test_dev_mode_header_reflects_mode(self, cfg):
        out = _render_bare_metal(cfg, mode="dev")
        assert "--mode dev" in out


class TestDevLegacyMode:
    def test_dev_legacy_keeps_silent_or_true(self, cfg):
        """Legacy mode is the OLD behaviour — kept for back-compat. Test
        captures the contract that we have NOT changed it."""
        out = _render_bare_metal(cfg, mode="dev_legacy")
        # The pip install line silently fails to keep historical behavior
        assert "pip install --quiet -e" in out
        assert "|| true" in out

    def test_dev_legacy_emits_deprecation_warn(self, cfg):
        out = _render_bare_metal(cfg, mode="dev_legacy")
        assert "deprecated" in out.lower(), (
            "dev_legacy must self-warn so operators migrate"
        )


class TestModeValidation:
    def test_unknown_mode_raises_valueerror(self, cfg):
        with pytest.raises(ValueError, match="unknown bare-metal render mode"):
            _render_bare_metal(cfg, mode="totally-bogus")

    def test_default_mode_is_wheel(self, cfg):
        """Default (production-safe) when caller doesn't specify mode."""
        default_out = _render_bare_metal(cfg)
        wheel_out = _render_bare_metal(cfg, mode="wheel")
        assert default_out == wheel_out


class TestAllModesShareScaffold:
    """All three modes must produce a runnable bash script with the
    same boot scaffolding (shebang, set -euo pipefail, venv check)."""

    @pytest.mark.parametrize("mode", ["wheel", "dev", "dev_legacy"])
    def test_shebang_present(self, cfg, mode):
        out = _render_bare_metal(cfg, mode=mode)
        assert out.startswith("#!/usr/bin/env bash\n")

    @pytest.mark.parametrize("mode", ["wheel", "dev", "dev_legacy"])
    def test_set_strict(self, cfg, mode):
        out = _render_bare_metal(cfg, mode=mode)
        assert "set -euo pipefail" in out

    @pytest.mark.parametrize("mode", ["wheel", "dev", "dev_legacy"])
    def test_venv_check(self, cfg, mode):
        out = _render_bare_metal(cfg, mode=mode)
        assert "VENV=" in out
        assert "venv not found" in out

    @pytest.mark.parametrize("mode", ["wheel", "dev", "dev_legacy"])
    def test_vllm_serve_exec(self, cfg, mode):
        out = _render_bare_metal(cfg, mode=mode)
        assert "exec vllm serve" in out
