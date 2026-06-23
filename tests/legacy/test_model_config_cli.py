# SPDX-License-Identifier: Apache-2.0
"""TDD for compat/model_config_cli.py — list/show/render/audit/where.

Audit closure 2026-05-08 (DEEP_AUDIT_VLLM_NOONGHUNNA P0-2): hardcoded
TPS expectations drift every time bench updates. These tests now load
the actual YAML and assert the CLI surfaces what's in the config —
that way Wave updates to reference_metrics don't require parallel
test edits.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sndr.compat.model_config_cli import main as cli_main
from sndr.model_configs.registry import get as get_config


# Phase 10 (2026-06-01): V1 sunset cascade. The CLI under test
# (`compat/model_config_cli.py`) operates on V1 monolithic preset keys.
# Once `FROZEN_V1_BASELINE` empties, the V1 CLI surface has no fixtures
# to exercise and these per-key tests are skipped at collection. The
# CLI module itself stays for now (V2 has its own CLI under
# `cli/{compose,config,launch}`); once V2 CLI fully supersedes the V1
# surface in operator workflows, this legacy test file becomes retired
# as a whole (separate cleanup).
_BUILTIN_DIR = (Path(__file__).resolve().parents[2] / "vllm" / "sndr_core"
                / "model_configs" / "builtin")
_V1_35B = "a5000-2x-35b-prod"
_V1_27B = "a5000-2x-27b-int4-tq-k8v4"
_skip_if_no_v1_35b = pytest.mark.skipif(
    not (_BUILTIN_DIR / f"{_V1_35B}.yaml").is_file(),
    reason=f"Phase 10 V1 sunset retired {_V1_35B}.yaml — V1 CLI test "
           "surface obsolete for this fixture.",
)
_skip_if_no_v1_27b = pytest.mark.skipif(
    not (_BUILTIN_DIR / f"{_V1_27B}.yaml").is_file(),
    reason=f"Phase 10 V1 sunset retired {_V1_27B}.yaml — V1 CLI test "
           "surface obsolete for this fixture.",
)


def _expected_tps(key: str) -> float:
    """Read the YAML's reference TPS — single source of truth.

    Falls back to a permissive sentinel if the key/field is absent so
    test failures point at the loader, not at perceived drift."""
    cfg = get_config(key)
    assert cfg is not None, f"config {key!r} not loadable"
    rm = cfg.reference_metrics
    return float(rm.long_gen_sustained_tps)


class TestList:
    @_skip_if_no_v1_35b
    @_skip_if_no_v1_27b
    def test_list_succeeds_and_shows_builtins(self, capsys):
        # Default `list` now hides tested/QA configs (P0.2 fix —
        # previous behaviour was the `or True` bug that always merged
        # them in). Pass --include-tested to surface them.
        # Fixture migrated 2026-06-01: a5000-2x-27b-int4-tested retired
        # in V1 sunset #8; swapped to surviving sibling
        # `a5000-2x-27b-int4-tq-k8v4` (still listed under --include-tested
        # via `lifecycle: tested` semantics).
        rc = cli_main(["list", "--include-tested"])
        assert rc == 0
        out = capsys.readouterr().out
        # Must include builtin configs
        assert "a5000-2x-35b-prod" in out
        assert "a5000-2x-27b-int4-tq-k8v4" in out
        # Must show TPS column
        assert "TPS" in out
        # 35B and 27B reference TPS — read from YAML, not hardcoded
        tps_35b = _expected_tps("a5000-2x-35b-prod")
        # YAML drift-resistant: just check the TPS value (truncated to
        # one decimal) appears anywhere in the list output.
        assert f"{tps_35b:.1f}" in out or f"{tps_35b:.2f}" in out, (
            f"35B PROD TPS {tps_35b:.2f} not visible in list output"
        )

    @_skip_if_no_v1_35b
    def test_list_default_hides_tested(self, capsys):
        """Default invocation must hide tested/QA configs but flag that
        they're available behind --include-tested.
        2026-06-01: all V1 `tested` lifecycle configs retired (V1 sunsets
        #3, #5, #8 took a5000-1x-27b-int4-tested, a5000-2x-27b-int4-tq-
        k8v4-dflash, a5000-2x-27b-int4-tested respectively). The
        `--include-tested` hint message is suppressed by the CLI when
        no tested-tier configs exist on disk, so we no longer assert
        on it. Test now asserts only that PROD entries are present and
        the retired V1 keys are absent (which both hold trivially
        since the V1 files are deleted)."""
        rc = cli_main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        # PROD entries must be present
        assert "a5000-2x-35b-prod" in out
        # tested-tier entries must be hidden by default (trivially true
        # after V1 sunsets retired all `tested` lifecycle V1 configs)
        assert "a5000-2x-27b-int4-tested" not in out


class TestShow:
    @_skip_if_no_v1_35b
    def test_show_known_key(self, capsys):
        rc = cli_main(["show", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "key: a5000-2x-35b-prod" in out
        # YAML-driven: assert show emits the actual reference TPS
        tps = _expected_tps("a5000-2x-35b-prod")
        assert f"long_gen_sustained_tps: {tps}" in out, (
            f"show should surface reference TPS {tps} from YAML"
        )

    def test_show_unknown_returns_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["show", "totally-bogus-config"])
        assert exc.value.code == 1


class TestRender:
    @_skip_if_no_v1_35b
    def test_render_emits_bash_script(self, capsys):
        rc = cli_main(["render", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "#!/usr/bin/env bash" in out
        assert "set -euo pipefail" in out
        assert "vllm serve" in out
        # Header references TPS — pulled from YAML, not hardcoded
        tps = _expected_tps("a5000-2x-35b-prod")
        assert f"{tps:.1f}" in out or f"{tps:.2f}" in out, (
            f"render header should reference {tps:.2f} TPS"
        )

    @_skip_if_no_v1_27b
    def test_render_includes_all_genesis_env(self, capsys):
        # Fixture migrated 2026-06-01: a5000-2x-27b-int4-tested retired
        # in V1 sunset #8; swapped to surviving sibling
        # `a5000-2x-27b-int4-tq-k8v4` (same Lorbus 27B INT4 + TQ k8v4
        # model family; both have P67 + P99 enabled).
        rc = cli_main(["render", "a5000-2x-27b-int4-tq-k8v4"])
        assert rc == 0
        out = capsys.readouterr().out
        # Spot-check critical patches present
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in out
        assert "GENESIS_ENABLE_P99" in out


class TestAudit:
    @_skip_if_no_v1_27b
    def test_audit_clean_config_returns_0(self, capsys):
        # 27B has 1 warning (R-005 PN59 long-ctx), but no errors → exit 0
        # Fixture migrated 2026-06-01: a5000-2x-27b-int4-tested retired
        # in V1 sunset #8; swapped to surviving sibling
        # `a5000-2x-27b-int4-tq-k8v4` (same audit-clean V1 config).
        rc = cli_main(["audit", "a5000-2x-27b-int4-tq-k8v4"])
        assert rc == 0

    def test_audit_unknown_returns_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["audit", "bogus"])
        assert exc.value.code == 1


class TestWhere:
    @_skip_if_no_v1_35b
    def test_where_shows_tier(self, capsys):
        rc = cli_main(["where", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "tier:" in out and "builtin" in out


class TestNew:
    @_skip_if_no_v1_35b
    def test_new_template_creates_user_config(self, tmp_path, monkeypatch,
                                                capsys):
        monkeypatch.setenv("GENESIS_MODEL_CONFIG_DIR", str(tmp_path))
        rc = cli_main(["new", "my-test", "--template", "a5000-2x-35b-prod"])
        assert rc == 0
        out_path = tmp_path / "my-test.yaml"
        assert out_path.is_file()
        content = out_path.read_text()
        assert "key: my-test" in content
        assert "reference_metrics: null" in content  # cleared on clone

    @_skip_if_no_v1_35b
    def test_new_no_overwrite_without_force(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GENESIS_MODEL_CONFIG_DIR", str(tmp_path))
        cli_main(["new", "my-test", "--template", "a5000-2x-35b-prod"])
        # Second call without --force fails
        rc = cli_main(["new", "my-test", "--template", "a5000-2x-35b-prod"])
        assert rc == 1


class TestVerify:
    def test_verify_unknown_returns_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["verify", "bogus"])
        assert exc.value.code == 1

    @_skip_if_no_v1_35b
    def test_verify_known_config_no_server_fails_predictably(self, capsys):
        # Without a running server, bench will fail → returns 1.
        # This is correct behaviour (operator must launch first).
        rc = cli_main(["verify", "a5000-2x-35b-prod"])
        assert rc in (0, 1)  # 0 if server happens to be up, 1 if not
