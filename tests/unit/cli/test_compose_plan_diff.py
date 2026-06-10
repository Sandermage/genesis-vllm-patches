# SPDX-License-Identifier: Apache-2.0
"""`sndr compose plan-diff` — A/B between two policies.

Operator workflow: before flipping a real launch from
``--policy compat`` to ``--policy minimal``, see exactly which toggle
flags get dropped. Reduces "I changed one flag and now the bench
moved by 5%" surprise.

Shape:

  sndr compose plan-diff <preset> --from compat --to minimal
  sndr compose plan-diff <preset> --from compat --to safe --json

Comparison axes:

  newly_excluded   toggles that were included under `from`, excluded under `to`
  newly_included   toggles that flipped the other way (rare; happens only if
                   attribution role changed between policies — usually empty)
  passthrough_diff parameter keys added / removed (should be empty since
                   passthrough is policy-independent)
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from sndr.cli.legacy import cli_main


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli_main(argv)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    return rc, buf.getvalue()


# ─── JSON shape ──────────────────────────────────────────────────────────


class TestJsonShape:
    def test_diff_compat_to_minimal_returns_structured_payload(self):
        rc, out = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "compat", "--to", "minimal", "--json",
        ])
        assert rc == 0, out[:500]
        payload = json.loads(out)
        assert payload["preset"] == "prod-qwen3.6-35b-balanced"
        assert payload["from_policy"] == "compat"
        assert payload["to_policy"] == "minimal"
        diff = payload["diff"]
        for key in ("newly_excluded", "newly_included", "unchanged_included",
                    "unchanged_excluded", "passthrough_diff"):
            assert key in diff

    def test_diff_minimal_drops_unknown_role_toggles_vs_compat(self):
        """Real semantic: under compat every truthy toggle survives;
        under minimal everything role='unknown' (or no_op/regression)
        gets dropped. The diff's newly_excluded list must therefore be
        non-empty on prod-qwen3.6-35b-balanced (where most patches lack attribution)."""
        rc, out = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "compat", "--to", "minimal", "--json",
        ])
        assert rc == 0
        payload = json.loads(out)
        newly_excluded = payload["diff"]["newly_excluded"]
        assert len(newly_excluded) > 0
        # Every newly_excluded entry must carry the patch_id, env_flag,
        # and the role that caused the drop.
        for entry in newly_excluded:
            assert "patch_id" in entry
            assert "env_flag" in entry
            assert "role" in entry

    def test_compat_to_compat_yields_empty_newly_diff(self):
        """Same policy on both sides → no toggles cross the boundary."""
        rc, out = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "compat", "--to", "compat", "--json",
        ])
        assert rc == 0
        diff = json.loads(out)["diff"]
        assert diff["newly_excluded"] == []
        assert diff["newly_included"] == []
        assert diff["passthrough_diff"] == {"added": [], "removed": []}


# ─── Human renderer ──────────────────────────────────────────────────────


class TestHumanRenderer:
    def test_human_mode_header_carries_policies_and_counts(self):
        rc, out = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "compat", "--to", "minimal",
        ])
        assert rc == 0, out[:500]
        assert "compat" in out
        assert "minimal" in out
        # Banner-style summary lines.
        for kw in ("newly excluded", "newly included"):
            assert kw in out.lower()


# ─── Invalid policy ──────────────────────────────────────────────────────


class TestInvalidPolicy:
    def test_invalid_from_policy_rejected(self):
        rc, _ = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "bogus", "--to", "compat",
        ])
        assert rc != 0

    def test_invalid_to_policy_rejected(self):
        rc, _ = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-35b-balanced",
            "--from", "compat", "--to", "bogus",
        ])
        assert rc != 0


# ─── V1 / V2 preset resolution ───────────────────────────────────────────


class TestPresetResolution:
    def test_v2_alias_works(self):
        rc, out = _run_cli([
            "compose", "plan-diff", "prod-qwen3.6-27b-tq-k8v4",
            "--from", "compat", "--to", "safe", "--json",
        ])
        assert rc == 0, out[:500]
        payload = json.loads(out)
        assert payload["preset"] == "prod-qwen3.6-27b-tq-k8v4"

    def test_unknown_preset_returns_error(self):
        rc, _ = _run_cli([
            "compose", "plan-diff", "this-preset-does-not-exist",
            "--from", "compat", "--to", "minimal",
        ])
        assert rc != 0
