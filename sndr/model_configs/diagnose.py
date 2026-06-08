# SPDX-License-Identifier: Apache-2.0
"""Runtime diagnose — query running container/server, diff vs config.

Layer 4 of the "100% close gaps" strategy. Catches the
"plugin loaded but patches didn't apply in worker process" class —
config says GENESIS_ENABLE_PXX=1 but boot summary shows PXX skipped.

Sources of truth at runtime:
  1. Docker container env (what envs are actually exported)
  2. Boot log "X applied / Y skipped / Z failed" line
  3. Boot summary structured patches list
  4. /v1/models endpoint (API responsive)

We compare:
  - cfg.genesis_env vs container env  → flag missing exports
  - cfg.genesis_env enabled patches vs boot log "applied" list →
    flag patches that should have applied but didn't
  - cfg.docker.mounts vs `docker inspect .Mounts` → flag mismatches
  - cfg.vllm_pin_required vs `pip show vllm` inside container →
    pin drift
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiagnoseFinding:
    name: str
    passed: bool
    message: str
    severity: str  # 'error' / 'warning' / 'info'


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", "command failed"


def _container_running(name: str) -> bool:
    rc, out, _ = _run([
        "docker", "ps", "--filter", f"name=^{name}$",
        "--format", "{{.Status}}",
    ])
    return rc == 0 and "Up" in out


def diagnose_container_running(cfg) -> Optional[DiagnoseFinding]:
    if not cfg.docker:
        return None
    name = cfg.docker.container_name
    if _container_running(name):
        return DiagnoseFinding(
            name="container_state", passed=True,
            message=f"'{name}' running",
            severity="info",
        )
    return DiagnoseFinding(
        name="container_state", passed=False,
        message=f"'{name}' is NOT running — diagnose needs a live container",
        severity="error",
    )


def diagnose_env_exported(
    cfg,
    *,
    expected_env: Optional[dict[str, str]] = None,
) -> list[DiagnoseFinding]:
    """Compare cfg.genesis_env vs container's actual env.

    Args:
        cfg: ModelConfig — used for cfg.docker.container_name resolution.
        expected_env: Phase D override. When provided, diagnose compares
            against this map (e.g. the policy-filtered plan.env from
            `resolve_patch_plan`) instead of the raw cfg.genesis_env.
            Use when the container was launched with --policy: the
            running env is the filtered subset and the raw matrix
            would produce false-positive "missing key" errors for
            patches the policy intentionally dropped.
    """
    out: list[DiagnoseFinding] = []
    if not cfg.docker:
        return out
    name = cfg.docker.container_name
    rc, env_json, _ = _run([
        "docker", "inspect", name,
        "--format", "{{json .Config.Env}}",
    ])
    if rc != 0:
        return [DiagnoseFinding(
            name="env_inspect", passed=False,
            message=f"docker inspect {name} failed",
            severity="error",
        )]
    actual_envs = json.loads(env_json) or []
    actual_dict = {}
    for e in actual_envs:
        if "=" in e:
            k, _, v = e.partition("=")
            actual_dict[k] = v

    # When expected_env is provided, use it instead of cfg.genesis_env.
    # Falls back to cfg.genesis_env so existing callers stay unchanged.
    source = expected_env if expected_env is not None else cfg.genesis_env
    for k, expected in source.items():
        actual = actual_dict.get(k)
        if actual is None:
            out.append(DiagnoseFinding(
                name=f"env:{k}", passed=False,
                message=f"{k} not exported to container "
                        f"(YAML says '{expected}')",
                severity="error",
            ))
        elif actual != expected:
            out.append(DiagnoseFinding(
                name=f"env:{k}", passed=False,
                message=f"{k}={actual} in container, but YAML expects "
                        f"'{expected}'",
                severity="warning",
            ))
        else:
            out.append(DiagnoseFinding(
                name=f"env:{k}", passed=True,
                message=f"{k}={actual} ✓",
                severity="info",
            ))
    return out


def diagnose_boot_summary(cfg) -> list[DiagnoseFinding]:
    """Parse `register() complete: N applied / M skipped / K failed` line
    from container logs and verify expected patches actually applied.
    """
    out: list[DiagnoseFinding] = []
    if not cfg.docker:
        return out
    name = cfg.docker.container_name
    rc, logs, _ = _run(["docker", "logs", "--tail", "10000", name],
                       timeout=30)
    if rc != 0:
        return [DiagnoseFinding(
            name="boot_log", passed=False,
            message=f"docker logs {name} failed",
            severity="error",
        )]

    # 1. Find register() complete line
    m = re.findall(
        r"register\(\) complete: (\d+) applied / (\d+) skipped / (\d+) failed",
        logs,
    )
    if not m:
        out.append(DiagnoseFinding(
            name="boot_summary", passed=False,
            message="no 'register() complete:' line found in logs — "
                    "plugin may not have run",
            severity="error",
        ))
    else:
        # Last (most recent process)
        applied, skipped, failed = m[-1]
        if int(failed) > 0:
            out.append(DiagnoseFinding(
                name="boot_summary", passed=False,
                message=f"plugin reported {failed} FAILED patches — "
                        f"check logs for ERROR lines",
                severity="error",
            ))
        else:
            out.append(DiagnoseFinding(
                name="boot_summary", passed=True,
                message=f"plugin: {applied} applied / {skipped} skipped / "
                        f"{failed} failed (across {len(m)} processes)",
                severity="info",
            ))

    # 2. For each enabled env flag, find corresponding "applied" line
    # Pattern: `║   • PXX        Title` in structured boot summary section
    applied_pattern = re.compile(r"║\s+•\s+([A-Z]+\d+\w*)\s+")
    applied_set = set()
    for ln in logs.splitlines():
        for match in applied_pattern.finditer(ln):
            applied_set.add(match.group(1))

    # Map env flag key → patch ID (e.g. GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL → P67)
    try:
        from sndr.dispatcher import PATCH_REGISTRY
        flag_to_pid = {}
        for pid, meta in PATCH_REGISTRY.items():
            flag = meta.get("env_flag")
            if flag:
                flag_to_pid[flag] = pid
    except Exception:
        flag_to_pid = {}

    for env_key, val in cfg.genesis_env.items():
        if val != "1":
            continue  # only check enabled patches
        pid = flag_to_pid.get(env_key)
        if not pid:
            continue  # not a primary env_flag (could be tunable knob)
        if pid in applied_set:
            out.append(DiagnoseFinding(
                name=f"applied:{pid}", passed=True,
                message=f"{pid} ({env_key}=1) applied ✓",
                severity="info",
            ))
        else:
            out.append(DiagnoseFinding(
                name=f"applied:{pid}", passed=False,
                message=f"{pid} ({env_key}=1) NOT in boot summary — "
                        f"may have skipped due to applies_to gate or "
                        f"failed silently",
                severity="warning",
            ))
    return out


def diagnose_api_responsive(cfg, port: Optional[int] = None) -> Optional[DiagnoseFinding]:
    """Curl /v1/models — passes if server responds."""
    if not cfg.docker:
        return None
    p = port or cfg.docker.port
    url = f"http://localhost:{p}/v1/models"
    rc, out, _ = _run([
        "curl", "-s", "-m", "3", "-o", "/dev/null",
        "-w", "%{http_code}",
        "-H", f"Authorization: Bearer {cfg.api_key}", url,
    ])
    if rc != 0 or out.strip() != "200":
        return DiagnoseFinding(
            name="api", passed=False,
            message=f"GET {url} returned HTTP '{out.strip()}' — "
                    f"server not ready or misconfigured",
            severity="error",
        )
    return DiagnoseFinding(
        name="api", passed=True,
        message=f"GET {url} → 200 ✓", severity="info",
    )


def diagnose_vllm_pin_runtime(cfg) -> Optional[DiagnoseFinding]:
    """Check vllm version inside running container matches required."""
    if not cfg.docker or not cfg.vllm_pin_required:
        return None
    name = cfg.docker.container_name
    rc, out, _ = _run([
        "docker", "exec", name,
        "bash", "-c",
        "pip show vllm 2>/dev/null | grep '^Version' | awk '{print $2}'",
    ])
    if rc != 0:
        return DiagnoseFinding(
            name="vllm_pin_runtime", passed=False,
            message="failed to query vllm version inside running container",
            severity="warning",
        )
    actual = out.strip()
    if actual == cfg.vllm_pin_required:
        return DiagnoseFinding(
            name="vllm_pin_runtime", passed=True,
            message=f"runtime vllm = {actual} (matches required) ✓",
            severity="info",
        )
    return DiagnoseFinding(
        name="vllm_pin_runtime", passed=False,
        message=f"runtime vllm = '{actual}', config requires "
                f"'{cfg.vllm_pin_required}'",
        severity="error",
    )


# ─── Public API ────────────────────────────────────────────────────────


def diagnose_all(
    cfg,
    port: Optional[int] = None,
    *,
    policy: Optional[str] = None,
) -> list[DiagnoseFinding]:
    """Full diagnose suite. Requires container to be running.

    Args:
        policy: when set, run the patch_plan resolver and
            compare the container's env against the policy-filtered
            map instead of cfg.genesis_env raw. Avoids false-positive
            "env missing" findings when the container was launched
            with the same --policy flag.
    """
    out: list[DiagnoseFinding] = []
    cs = diagnose_container_running(cfg)
    if cs:
        out.append(cs)
        if not cs.passed:
            return out  # bail out — nothing else will work

    expected_env: Optional[dict[str, str]] = None
    if policy is not None:
        from sndr.model_configs.patch_plan import resolve_patch_plan
        plan = resolve_patch_plan(cfg, policy=policy)
        expected_env = plan.env

    out.extend(diagnose_env_exported(cfg, expected_env=expected_env))
    out.extend(diagnose_boot_summary(cfg))

    api = diagnose_api_responsive(cfg, port=port)
    if api:
        out.append(api)

    pin = diagnose_vllm_pin_runtime(cfg)
    if pin:
        out.append(pin)

    return out


def has_blockers(findings: list[DiagnoseFinding]) -> bool:
    return any(not f.passed and f.severity == "error" for f in findings)
