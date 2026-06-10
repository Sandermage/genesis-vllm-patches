# SPDX-License-Identifier: Apache-2.0
"""Phase C — `sndr compose render --policy` opt-in YAML filter.

Phase B already exposed the resolver in `sndr patches plan`. Phase C
wires the same resolver into compose rendering so operators can
generate compose files where the `environment:` block is the
policy-filtered env, not the raw model.patches matrix.

Behaviour matrix:

  no --policy             render_compose_yaml unchanged (legacy
                          default — every operator's existing flow
                          keeps working byte-for-byte).
  --policy compat         resolver runs, passthrough parameters
                          preserved, included toggles unchanged
                          (compat policy passes everything truthy
                          through). Header gets plan summary.
  --policy safe           drops role=='no_op' toggles from env block.
  --policy minimal        also drops suspected_regression/unknown.

The header always carries an explicit policy line so operators
reading a compose file 6 weeks later still know how it was generated.

The tests call ``render_compose_yaml`` directly with
``host_paths={...stub...}`` (the convention used by
``test_compose_render.py``). The CLI-flag wiring is covered separately
by ``TestCliFlagWiring`` which mocks ``_load_host_paths``.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from sndr.cli.legacy.compose import render_compose_yaml
from sndr.model_configs.registry_v2 import load_alias


_HOST_PATHS_STUB = {
    "models_dir": "/srv/models",
    "hf_cache": "/srv/hf",
    "triton_cache": "/srv/triton",
    "compile_cache": "/srv/compile",
    "genesis_src": "/srv/genesis",
    "plugin_src": "/srv/plugin",
}


@pytest.fixture(scope="module")
def cfg_prod_35b():
    return load_alias("prod-qwen3.6-35b-balanced")


def _parse(out: str) -> dict:
    return yaml.safe_load(out)


def _service_env(parsed: dict) -> dict[str, str]:
    services = parsed["services"]
    assert len(services) == 1
    svc = next(iter(services.values()))
    raw = svc.get("environment") or {}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if "=" in item:
                k, _, v = item.partition("=")
                out[k] = v
        return out
    return dict(raw)


# ─── Backwards compat: no policy → unchanged output ──────────────────────


class TestBackwardsCompat:
    def test_render_without_policy_has_no_plan_header(self, cfg_prod_35b):
        out = render_compose_yaml(cfg_prod_35b, host_paths=_HOST_PATHS_STUB)
        assert "Patch policy:" not in out
        # Body still parses cleanly.
        assert _parse(out)["services"]

    def test_render_without_policy_env_matches_cfg(self, cfg_prod_35b):
        out = render_compose_yaml(cfg_prod_35b, host_paths=_HOST_PATHS_STUB)
        env = _service_env(_parse(out))
        # Every cfg.genesis_env key survives the legacy unfiltered path.
        for k in cfg_prod_35b.genesis_env:
            assert k in env, f"legacy render dropped {k!r}"


# ─── --policy compat: passthrough preserved + header ─────────────────────


class TestCompatPolicy:
    def test_compat_emits_plan_header(self, cfg_prod_35b):
        out = render_compose_yaml(
            cfg_prod_35b, host_paths=_HOST_PATHS_STUB, policy="compat",
        )
        assert "Patch policy: compat" in out
        for kw in ("included:", "excluded:", "passthrough:"):
            assert kw in out

    def test_compat_keeps_non_toggle_parameters(self, cfg_prod_35b):
        """Parameter env vars (GENESIS_PN95_CONFIG_KEY etc.) must
        survive compat. Without them, dependent patches silently noop."""
        out = render_compose_yaml(
            cfg_prod_35b, host_paths=_HOST_PATHS_STUB, policy="compat",
        )
        env = _service_env(_parse(out))
        assert "GENESIS_PN95_CONFIG_KEY" in env
        assert "GENESIS_BUFFER_MODE" in env


# ─── --policy safe ───────────────────────────────────────────────────────


class TestSafePolicy:
    def test_safe_renders_cleanly_with_header(self, cfg_prod_35b):
        out = render_compose_yaml(
            cfg_prod_35b, host_paths=_HOST_PATHS_STUB, policy="safe",
        )
        assert "Patch policy: safe" in out
        assert _parse(out)["services"]


# ─── --policy minimal: parameters survive, unknown toggles drop ──────────


class TestMinimalPolicy:
    def test_minimal_drops_unknown_toggles(self, cfg_prod_35b):
        out = render_compose_yaml(
            cfg_prod_35b, host_paths=_HOST_PATHS_STUB, policy="minimal",
        )
        env = _service_env(_parse(out))
        # Parameter keys MUST survive minimal — they configure surviving patches.
        assert "GENESIS_PN95_CONFIG_KEY" in env, (
            "minimal must not drop GENESIS_PN95_CONFIG_KEY — "
            "PN95 silently no-ops without it"
        )
        # After the 2026-05-16 attribution backfill, the 35B model
        # declares load_bearing/defensive/optional_perf for ~10 patches
        # → some toggles legitimately survive minimal. The invariant
        # we still want to assert: no role='unknown' toggle leaks.
        toggle_keys = [k for k in env if k.startswith("GENESIS_ENABLE_")]
        # An unknown-role toggle (no attribution backfill yet) — make
        # sure it's gone. PN71_THINKING_TAG_NORMALIZE is one of the
        # un-attributed enables in qwen3.6-35b-a3b-fp8.yaml.
        assert not any(
            "PN71_THINKING_TAG_NORMALIZE" in k for k in toggle_keys
        ), f"unknown-role toggle leaked under minimal: {toggle_keys}"
        # And at least one attributed toggle should be present (proves
        # the resolver isn't accidentally dropping everything).
        assert any(
            "P67_TQ_MULTI_QUERY_KERNEL" in k for k in toggle_keys
        ), (
            f"load_bearing P67 missing under minimal — backfill regression? "
            f"toggles={toggle_keys}"
        )

    def test_minimal_keeps_passthrough_count_in_header(self, cfg_prod_35b):
        out = render_compose_yaml(
            cfg_prod_35b, host_paths=_HOST_PATHS_STUB, policy="minimal",
        )
        # The passthrough count must be >= 1 — otherwise the resolver
        # accidentally classified parameters as toggles.
        for line in out.splitlines():
            if "passthrough:" in line:
                # Format: "#   passthrough: <N> parameter(s)"
                count = int(line.strip().split(":")[1].split()[0])
                assert count >= 1, (
                    f"passthrough count is 0 — parameter keys mis-classified"
                )
                break
        else:
            pytest.fail("passthrough count line missing from header")


# ─── CLI flag wiring: --policy passes through to render_compose_yaml ─────


class TestCliFlagWiring:
    def test_cli_invokes_render_with_policy(self):
        """Verify the argparse plumbing calls render_compose_yaml with
        ``policy=<value>`` when --policy is set on the command line.

        Mock _load_host_paths so the test never depends on a real
        host.yaml file."""
        import io
        from contextlib import redirect_stdout
        from sndr.cli.legacy import cli_main

        captured = {}

        def _stub_render(cfg, host_paths=None, *, policy=None):
            captured["policy"] = policy
            captured["key"] = cfg.key
            # Return enough YAML for the CLI not to crash.
            return "# stub\nservices:\n  vllm-server:\n    image: x\n"

        with patch(
            "sndr.cli.legacy.compose.render_compose_yaml",
            side_effect=_stub_render,
        ):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    rc = cli_main([
                        "compose", "render", "prod-qwen3.6-35b-balanced", "--policy", "safe",
                    ])
            except SystemExit as e:
                rc = int(e.code) if isinstance(e.code, int) else 2

        assert rc == 0, f"unexpected rc={rc}"
        assert captured["policy"] == "safe"
        assert "qwen3" in captured["key"]

    def test_cli_invalid_policy_rejected(self):
        import io
        from contextlib import redirect_stdout
        from sndr.cli.legacy import cli_main

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = cli_main([
                    "compose", "render", "prod-qwen3.6-35b-balanced", "--policy", "bogus",
                ])
        except SystemExit as e:
            rc = int(e.code) if isinstance(e.code, int) else 2
        assert rc != 0
