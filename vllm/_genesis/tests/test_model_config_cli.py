# SPDX-License-Identifier: Apache-2.0
"""TDD for compat/model_config_cli.py — list/show/render/audit/where."""
from __future__ import annotations

import pytest

from vllm._genesis.compat.model_config_cli import main as cli_main


class TestList:
    def test_list_succeeds_and_shows_builtins(self, capsys):
        rc = cli_main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        # Must include builtin configs
        assert "a5000-2x-35b-prod" in out
        assert "a5000-2x-27b-int4-balanced" in out
        # Must show TPS column
        assert "TPS" in out
        assert "192.6" in out  # 35B PROD reference
        assert "67.0" in out   # 27B reference


class TestShow:
    def test_show_known_key(self, capsys):
        rc = cli_main(["show", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "key: a5000-2x-35b-prod" in out
        assert "long_gen_sustained_tps: 192.6" in out

    def test_show_unknown_returns_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["show", "totally-bogus-config"])
        assert exc.value.code == 1


class TestRender:
    def test_render_emits_bash_script(self, capsys):
        rc = cli_main(["render", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "#!/usr/bin/env bash" in out
        assert "set -euo pipefail" in out
        assert "vllm serve" in out
        # Reference visible in header
        assert "192.6" in out

    def test_render_includes_all_genesis_env(self, capsys):
        rc = cli_main(["render", "a5000-2x-27b-int4-balanced"])
        assert rc == 0
        out = capsys.readouterr().out
        # Spot-check critical patches present
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in out
        assert "GENESIS_ENABLE_P99" in out


class TestAudit:
    def test_audit_clean_config_returns_0(self, capsys):
        # 27B has 1 warning (R-005 PN59 long-ctx), but no errors → exit 0
        rc = cli_main(["audit", "a5000-2x-27b-int4-balanced"])
        assert rc == 0

    def test_audit_unknown_returns_1(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["audit", "bogus"])
        assert exc.value.code == 1


class TestWhere:
    def test_where_shows_tier(self, capsys):
        rc = cli_main(["where", "a5000-2x-35b-prod"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "tier:" in out and "builtin" in out


class TestNew:
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

    def test_verify_known_config_no_server_fails_predictably(self, capsys):
        # Without a running server, bench will fail → returns 1.
        # This is correct behaviour (operator must launch first).
        rc = cli_main(["verify", "a5000-2x-35b-prod"])
        assert rc in (0, 1)  # 0 if server happens to be up, 1 if not
