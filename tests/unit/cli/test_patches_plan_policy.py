# SPDX-License-Identifier: Apache-2.0
"""Phase B CLI tests — `sndr patches plan` gains `--policy` + `--explain`.

The existing `plan` subcommand simulates the dispatcher's APPLY/SKIP
decision per registry entry against a preset's env. Phase B adds two
complementary flags:

  --policy compat|safe|minimal   filters genesis_env through the
                                  patch_plan resolver before reporting
  --explain                       includes role/note/bench_evidence from
                                  patches_attribution in the JSON output

Default (no flag) behaviour stays unchanged — operators don't see new
columns until they ask for them.

Tests run the CLI in-process by invoking `sndr_core.cli.run()` so we
get realistic exit codes and JSON output without subprocess overhead.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from sndr.cli.legacy import cli_main as cli_run


# ─── In-process CLI runner ───────────────────────────────────────────────


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run `sndr ...` in-process, capture stdout, return (rc, stdout).

    argparse exits via SystemExit on usage errors (e.g. invalid
    --policy); catch and translate to a regular return code so tests
    can assert non-zero exits without exception bookkeeping.
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli_run(argv)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    return rc, buf.getvalue()


# ─── Default policy must stay backwards compatible ───────────────────────


class TestDefaultPolicy:
    def test_plan_without_policy_runs_simulator(self):
        """No --policy → existing dispatcher-simulator behaviour."""
        rc, out = _run_cli([
            "patches", "plan", "--preset", "prod-qwen3.6-35b-balanced", "--json",
        ])
        assert rc in (0, 2), f"unexpected rc={rc}: {out[:500]}"
        payload = json.loads(out)
        # The simulator shape carries `apply`/`skip`/`errors`.
        assert "apply" in payload and "skip" in payload
        # The resolver shape (`policy`/`included`/`excluded`) must NOT
        # appear unless the operator opts in.
        assert "included" not in payload


# ─── --policy + --explain surfaces resolver output ───────────────────────


class TestExplicitPolicy:
    @pytest.mark.parametrize("policy", ["compat", "safe", "minimal"])
    def test_policy_emits_resolver_shape(self, policy):
        rc, out = _run_cli([
            "patches", "plan", "--preset", "prod-qwen3.6-35b-balanced",
            "--policy", policy, "--json",
        ])
        assert rc in (0, 2), f"unexpected rc={rc}: {out[:500]}"
        payload = json.loads(out)
        # Resolver shape carries the dedicated block.
        assert payload["resolver"]["policy"] == policy
        assert "included" in payload["resolver"]
        assert "excluded" in payload["resolver"]

    def test_explain_adds_role_and_note_fields(self):
        rc, out = _run_cli([
            "patches", "plan", "--preset", "prod-qwen3.6-35b-balanced",
            "--policy", "compat", "--explain", "--json",
        ])
        assert rc in (0, 2), f"unexpected rc={rc}: {out[:500]}"
        payload = json.loads(out)
        # The 35B preset seeded PN204 / PN134 attributions in Phase A.
        # Whichever one ends up included or excluded should carry role +
        # source metadata.
        all_decisions = (
            payload["resolver"]["included"] + payload["resolver"]["excluded"]
        )
        keyed = {d["patch_id"]: d for d in all_decisions}
        assert "PN204" in keyed, f"PN204 missing from resolver output"
        d = keyed["PN204"]
        assert d["role"] == "optional_perf"
        assert "689" in d["bench_evidence"] or "TPS" in d["bench_evidence"]


# ─── Minimal policy filters the seeded 35B example correctly ─────────────


class TestMinimalDropsKnownNoOps:
    def test_minimal_excludes_suspected_regression_when_truthy(self):
        """The 35B model YAML keeps PN134 commented out, so it never
        appears in genesis_env truthy. Use --policy minimal + a preset
        whose env has truthy entries; we assert the resolver path runs
        cleanly and emits an `env` block keyed by env_flag."""
        rc, out = _run_cli([
            "patches", "plan", "--preset", "prod-qwen3.6-35b-balanced",
            "--policy", "minimal", "--json",
        ])
        assert rc in (0, 2)
        payload = json.loads(out)
        env = payload["resolver"]["env"]
        assert isinstance(env, dict)
        # All toggle-keyed entries (GENESIS_ENABLE_* / GENESIS_DISABLE_*)
        # that survived minimal filtering must be truthy. Parameter
        # keys (GENESIS_BUFFER_MODE, GENESIS_PROFILE_RUN_CAP_M, …) can
        # legitimately carry "0" — they're not toggle flags, the value
        # configures behaviour rather than gating it.
        for k, v in env.items():
            if k.startswith("GENESIS_ENABLE_") or k.startswith("GENESIS_DISABLE_"):
                assert str(v).strip().lower() not in ("0", "false", ""), (
                    f"toggle {k}={v!r} leaked into minimal env"
                )


# ─── Invalid policy rejected by argparse ─────────────────────────────────


class TestInvalidPolicy:
    def test_bogus_policy_rejected(self):
        rc, out = _run_cli([
            "patches", "plan", "--preset", "prod-qwen3.6-35b-balanced",
            "--policy", "bogus",
        ])
        assert rc != 0
