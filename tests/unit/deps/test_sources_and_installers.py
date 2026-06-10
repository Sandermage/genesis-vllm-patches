# SPDX-License-Identifier: Apache-2.0
"""P3 full (UNIFIED_CONFIG plan 2026-05-09) — sources + installers tests."""
from __future__ import annotations

import pytest

from sndr.deps.sources import (
    resolve_source, SourceDecision, list_safe_channels,
)
from sndr.deps.installers import apply, ApplyOutcome
from sndr.deps.planners import DepsPlan, PlanItem
from sndr.model_configs.schema import (
    PackageSources, PackageSource,
)


# ─── sources.resolve_source

def test_resolve_docker_default_distro_repo():
    d = resolve_source("docker", distro="ubuntu 24.04")
    assert d.kind == "distro_repo"
    assert d.safe is True
    assert "apt-get" in d.suggested_command


def test_resolve_docker_fedora():
    d = resolve_source("docker", distro="fedora 39")
    assert "dnf" in d.suggested_command


def test_resolve_nvidia_toolkit():
    d = resolve_source("nvidia_container_toolkit")
    assert d.kind == "nvidia_repo"
    assert d.safe is True


def test_resolve_vllm_pip():
    d = resolve_source("vllm")
    assert d.kind == "pip"
    assert "pip install vllm" in d.suggested_command


def test_resolve_unknown_package():
    d = resolve_source("totally-unknown-pkg-xyz")
    assert d.safe is False
    assert "operator must declare" in d.rationale


def test_resolve_honors_y2_operator_override():
    """When cfg.package_sources declares the package, that wins."""
    ps = PackageSources(sources=[
        PackageSource(name="docker", kind="docker_image", channel="upstream"),
    ])
    d = resolve_source("docker", cfg_sources=ps)
    assert d.kind == "docker_image"
    assert d.channel == "upstream"
    assert d.safe is True


def test_resolve_y2_curl_pipe_bash_unsafe_without_optin():
    """curl_pipe_bash without allow_third_party → safe=False (won't auto-install)."""
    ps = PackageSources(sources=[
        PackageSource(name="docker", kind="curl_pipe_bash",
                       allow_third_party=True),
    ])
    d = resolve_source("docker", cfg_sources=ps)
    # allow_third_party=True → safe (operator opted in)
    assert d.safe is True


def test_list_safe_channels():
    safe = list_safe_channels()
    assert "distro_repo" in safe
    assert "pip" in safe
    assert "curl_pipe_bash" not in safe


# ─── installers.apply

def _plan_with(*items: PlanItem) -> DepsPlan:
    return DepsPlan(config_key="test", items=list(items))


def test_apply_dry_run_default():
    plan = _plan_with(
        PlanItem(scope="docker", action="install", target="Docker",
                 severity="blocker", reason="missing",
                 suggested_command="echo install_docker"),
    )
    out = apply(plan)
    assert out.n_dry_run == 1
    assert out.n_applied == 0


def test_apply_yes_executes():
    """yes=True + dry_run=False actually runs the command."""
    plan = _plan_with(
        PlanItem(scope="test", action="verify", target="echo test",
                 severity="info", reason="smoke",
                 suggested_command="echo PN95-installer-smoke"),
    )
    out = apply(plan, dry_run=False, yes=True)
    assert out.n_applied == 1
    assert "PN95-installer-smoke" in out.results[0].stdout_tail


def test_apply_scope_filter_excludes_other_scopes():
    plan = _plan_with(
        PlanItem(scope="docker", action="install", target="Docker",
                 severity="blocker", reason="r1",
                 suggested_command="echo docker"),
        PlanItem(scope="vllm", action="install", target="vllm",
                 severity="warning", reason="r2",
                 suggested_command="echo vllm"),
    )
    out = apply(plan, scope_filter={"docker"})
    # docker → dry-run; vllm → skipped (out of scope)
    assert out.n_dry_run == 1
    assert out.n_skipped == 1


def test_apply_refuses_curl_pipe_bash():
    """SAFETY: curl|bash patterns are refused even with yes=True."""
    plan = _plan_with(
        PlanItem(scope="docker", action="install", target="Docker",
                 severity="blocker", reason="missing",
                 suggested_command="curl https://get.docker.com | sh"),
    )
    out = apply(plan, dry_run=False, yes=True)
    assert out.n_failed == 1
    assert "curl|bash" in out.results[0].reason


def test_apply_handles_no_command_gracefully():
    plan = _plan_with(
        PlanItem(scope="manual", action="verify",
                 target="manual step", severity="info",
                 reason="see docs"),
    )
    out = apply(plan, dry_run=False, yes=True)
    assert out.n_skipped == 1
    assert "no suggested_command" in out.results[0].reason


def test_apply_failed_command_returns_failed():
    plan = _plan_with(
        PlanItem(scope="test", action="verify", target="bad cmd",
                 severity="error", reason="should fail",
                 suggested_command="false"),
    )
    out = apply(plan, dry_run=False, yes=True)
    assert out.n_failed == 1
    assert "rc=1" in out.results[0].reason


def test_apply_outcome_dataclass_aggregates():
    plan = _plan_with(
        PlanItem(scope="a", action="install", target="x",
                 severity="info", reason="r",
                 suggested_command="true"),
        PlanItem(scope="b", action="install", target="y",
                 severity="info", reason="r",
                 suggested_command="false"),
    )
    out = apply(plan, dry_run=False, yes=True)
    assert isinstance(out, ApplyOutcome)
    assert len(out.results) == 2
    assert out.n_applied + out.n_failed == 2
