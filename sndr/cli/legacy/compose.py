# SPDX-License-Identifier: Apache-2.0
"""S3.1 (UNIFIED_CONFIG plan 2026-05-09; audit P3-1 closure 2026-05-12):
`sndr compose render/up/down/logs` — docker-compose renderer and thin
wrapper over the `docker compose` CLI.

Why
---
Previously Genesis generated ONLY a bash launch-script via
`ModelConfig.to_launch_script()`. Community feedback (issue X-COMPOSE):
operators integrate Genesis into an existing compose stack and want
a ready-made `docker-compose.yml` with all patches and envs,
rather than manually translating a bash script into compose.

`sndr compose render <preset>` — reversible renderer:

  • Takes ModelConfig from the registry.
  • Emits `docker-compose.yml` with the correct image, container_name,
    ports, environment (genesis_env + system_env + patch knobs),
    volumes (mounts with host.yaml resolution), command (vllm serve
    flags). Uses yaml.safe_dump for correct escapes
    (avoiding string concatenation footguns).
  • Idempotent: re-rendering with the same input produces the same output.

`sndr compose up/down/logs` — thin wrapper over `docker compose -f
<rendered> up/down/logs`, for operator convenience. If the operator
integrates the output into their own stack, these commands need not be used.

Test contract — `tests/unit/cli/test_compose_render.py`:

  • Render canonical 27B PROD config → the result contains the expected
    image / container / ports / env / volumes / command.
  • yaml.safe_load(render) returns dict (proves it's parseable).
  • Hermetic: does not require docker to be installed.
  • render --output path writes a file with the same content.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import _io


__all__ = [
    "add_argparser", "render_compose_yaml",
    "run_compose_render", "run_compose_up",
    "run_compose_down", "run_compose_logs",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "compose",
        help="docker-compose renderer + thin wrapper (audit P3-1).",
        description=(
            "Render a preset as a ready-made docker-compose.yml, or "
            "start/stop/fetch logs via docker compose. "
            "An alternative to `sndr launch` for operators integrating "
            "Genesis into an existing compose stack."
        ),
    )
    sub = p.add_subparsers(dest="compose_cmd", required=True)

    # render
    p_render = sub.add_parser(
        "render", help="Render preset → docker-compose.yml to stdout/file",
    )
    p_render.add_argument("config", help="preset key")
    p_render.add_argument(
        "-o", "--output", default=None,
        help="Write to file instead of stdout.",
    )
    p_render.add_argument(
        "--policy",
        choices=("compat", "safe", "minimal"),
        default=None,
        help=(
            "filter the rendered environment "
            "block through the patch_plan resolver. compat keeps every "
            "truthy toggle, safe drops role='no_op', minimal also "
            "drops suspected_regression/unknown. Non-toggle GENESIS_* "
            "parameter keys always pass through. Omit the flag to keep "
            "legacy render output (every model.patches entry, "
            "unfiltered)."
        ),
    )
    p_render.set_defaults(func=run_compose_render)

    # plan-diff (Phase D ext — A/B between policies)
    p_diff = sub.add_parser(
        "plan-diff",
        help="Compare patch_plan resolver output between two policies.",
        description=(
            "Renders the patch_plan twice for the same preset and "
            "reports which toggle flags are newly included / newly "
            "excluded under --to compared with --from. Read-only — "
            "doesn't render any YAML; useful before switching a real "
            "launch from one policy to another."
        ),
    )
    p_diff.add_argument("config", help="preset key (V1 or V2 alias)")
    p_diff.add_argument(
        "--from", dest="from_policy", required=True,
        choices=("compat", "safe", "minimal"),
        help="Baseline policy to diff against.",
    )
    p_diff.add_argument(
        "--to", dest="to_policy", required=True,
        choices=("compat", "safe", "minimal"),
        help="Target policy.",
    )
    p_diff.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON output.",
    )
    p_diff.set_defaults(func=run_compose_plan_diff)

    # up
    p_up = sub.add_parser("up", help="docker compose up -d (renders inline)")
    p_up.add_argument("config", help="preset key")
    p_up.add_argument(
        "--detach", "-d", action="store_true", default=True,
        help="detached mode (default true)",
    )
    p_up.set_defaults(func=run_compose_up)

    # down
    p_down = sub.add_parser("down", help="docker compose down")
    p_down.add_argument("config", help="preset key")
    p_down.set_defaults(func=run_compose_down)

    # logs
    p_logs = sub.add_parser("logs", help="docker compose logs -f")
    p_logs.add_argument("config", help="preset key")
    p_logs.add_argument(
        "-n", "--lines", default="100",
        help="how many lines to tail (default 100)",
    )
    p_logs.add_argument(
        "-f", "--follow", action="store_true",
        help="follow log output",
    )
    p_logs.set_defaults(func=run_compose_logs)


def _resolve(key: str):
    """Accept either a V1 monolithic preset key or a V2 alias.

    switched from V1-only `registry.get()` to
    the same V1+V2 helper memory.py / patches.py already use, so
    `sndr compose render prod-qwen3.6-35b-balanced` works alongside the legacy
    `sndr compose render a5000-2x-35b-prod`.
    """
    try:
        from sndr.cli.legacy.memory import _resolve_preset_v1_or_v2
        return _resolve_preset_v1_or_v2(key)
    except Exception as e:
        _io.error(f"unknown preset key {key!r} ({e})")
        return None


# ──── Render ───────────────────────────────────────────────────────────


def _load_host_paths() -> dict[str, str]:
    """Read host.yaml and return the substitution table.

    `HostConfig` exposes the operator-tunable paths under the `paths`
    attribute (model_configs/host.py). The previous reference to
    `symbolic_mounts` matched no attribute and was silently swallowed
    by the bare except, leaving compose render with an empty table
    and producing literal `${var}` mount strings.
    """
    try:
        from sndr.model_configs.host import load_host_config
    except ImportError:
        return {}
    try:
        hc = load_host_config()
    except Exception:
        return {}
    if hc is None:
        return {}
    return dict(getattr(hc, "paths", {}) or {})


# strict substitution. Previously unresolved
# `${var}` placeholders silently passed through → Docker mount attempted
# to use the literal `${unknown}` as hostpath → cryptic boot failure.
# Now we raise on any remaining unresolved placeholders.
#
# Pattern is intentionally `${var}` only (not bare `$var`) — the schema
# contract in model_configs/schema.py uses brace form exclusively, so
# matching `$FOO` would generate false unresolved-placeholder alerts on
# shell-style env references that pass through to the runtime.
import re as _re  # noqa: E402  — section-local import after explanatory docstring
_UNRESOLVED_PLACEHOLDER_RE = _re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_mount(mount_spec: str, host_paths: dict[str, str]) -> str:
    """Applies host.yaml substitution to `${var}:/container_path:mode`.

    Etap 2.2: after substitution, verifies that no unresolved
    `${var}` placeholders remain. If any remain — `ValueError` with a hint
    about which variables need to be declared in host.yaml.
    """
    out = mount_spec
    for var, path in host_paths.items():
        out = out.replace(f"${{{var}}}", path)
        out = out.replace(f"${var}", path)
    unresolved = _UNRESOLVED_PLACEHOLDER_RE.findall(out)
    if unresolved:
        raise ValueError(
            f"unresolved mount placeholder(s) {unresolved!r} in {mount_spec!r}; "
            "add this variable to host.yaml `symbolic_mounts` or pass "
            "an explicit `host_paths` dict to the renderer."
        )
    return out


def _container_command(cfg) -> list[str]:
    """Delegates to canonical `build_runtime_command`.

    Etap 2.1 (audit 2026-05-12): previously compose/quadlet/k8s had
    independent command builders that diverged from bare-metal. Now
    they all go through `sndr.model_configs.runtime_command`.
    """
    from sndr.model_configs.runtime_command import (
        build_runtime_command,
    )
    return build_runtime_command(cfg).argv


def render_compose_yaml(
    cfg,
    host_paths: Optional[dict[str, str]] = None,
    *,
    policy: Optional[str] = None,
) -> str:
    """Renders ModelConfig into docker-compose.yml.

    Args:
        cfg: ModelConfig from registry.
        host_paths: optional substitution table for `${var}` in mounts.
            None → attempt to read host.yaml.
        policy: when set, filter cfg.genesis_env through the
            patch_plan resolver before rendering. ``None`` keeps the
            legacy unfiltered behaviour byte-for-byte. Valid values:
            ``compat``, ``safe``, ``minimal`` (see patch_plan.py).
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "compose render requires `pyyaml` — `pip install pyyaml`"
        ) from e

    if host_paths is None:
        host_paths = _load_host_paths()

    docker = cfg.docker
    if docker is None:
        raise ValueError(
            f"preset {cfg.key!r} has no docker block — compose "
            "requires a container image / name / port. Add a "
            "DockerConfig to the model_config YAML."
        )

    container_name = docker.container_name or f"sndr-{cfg.key}"
    image = docker.effective_image_ref()
    host_port = docker.effective_host_port()
    container_port = docker.effective_container_port()

    # Environment: combine system_env, genesis_env. All values as
    # strings — compose requires string-valued env vars.
    #
    # if `policy` is set, run cfg.genesis_env
    # through the patch_plan resolver. Policy-filtered toggles +
    # passthrough parameters together form the new genesis_env block.
    # When policy is None the env block matches the legacy unfiltered
    # render byte-for-byte.
    plan = None
    if policy is not None:
        from sndr.model_configs.patch_plan import (
            resolve_patch_plan,
        )
        plan = resolve_patch_plan(cfg, policy=policy)
        genesis_env_to_render = plan.env
    else:
        genesis_env_to_render = cfg.genesis_env

    env: dict[str, str] = {}
    for k, v in cfg.system_env.items():
        env[str(k)] = str(v)
    for k, v in genesis_env_to_render.items():
        env[str(k)] = str(v)
    # VLLM_API_KEY is rendered as a compose
    # interpolation reference `${VLLM_API_KEY:?...}`, not a literal value.
    # Docker Compose will resolve it from shell env / `.env` file at
    # `compose up` time. The literal no longer appears in YAML → no leak
    # of the key via a file in /tmp.
    if cfg.api_key:
        env.setdefault(
            "VLLM_API_KEY",
            "${VLLM_API_KEY:?VLLM_API_KEY env required — "
            "export in shell or .env file before `docker compose up`}",
        )

    volumes_resolved = [
        _resolve_mount(m, host_paths)
        for m in (docker.mounts or [])
    ]

    service: dict[str, Any] = {
        "image": image,
        "container_name": container_name,
        "restart": "unless-stopped",
        "ports": [f"{host_port}:{container_port}"],
        "environment": env,
        "command": _container_command(cfg),
        # Provenance labels — let the GUI link a RUNNING container back to the
        # preset/config that defines it (and detect drift). Pure metadata, no
        # runtime effect. `sndr.preset` is resolvable via the V2 catalog.
        "labels": {"sndr.managed": "true", "sndr.preset": str(cfg.key)},
    }
    if volumes_resolved:
        service["volumes"] = volumes_resolved
    if docker.shm_size:
        service["shm_size"] = docker.shm_size
    if docker.network:
        service["networks"] = [docker.network]
    # GPU access — Docker Compose Spec deploy.resources.reservations.devices.
    service["deploy"] = {
        "resources": {
            "reservations": {
                "devices": [{
                    "driver": "nvidia",
                    "count": cfg.hardware.n_gpus,
                    "capabilities": ["gpu"],
                }],
            },
        },
    }

    compose: dict[str, Any] = {
        "services": {"vllm-server": service},
    }
    if docker.network:
        compose["networks"] = {docker.network: {"external": True}}

    plan_header = ""
    if plan is not None:
        plan_header = (
            f"#\n"
            f"# Patch policy: {plan.policy}\n"
            f"#   included: {len(plan.included)} toggle(s)\n"
            f"#   excluded: {len(plan.excluded)} toggle(s)\n"
            f"#   passthrough: {len(plan.passthrough)} parameter(s)\n"
            f"#   regenerate: "
            f"sndr compose render {cfg.key} --policy {plan.policy}\n"
            f"#   inspect:    "
            f"sndr patches plan --preset {cfg.key} --policy {plan.policy} --explain\n"
        )
        if plan.warnings:
            plan_header += f"#   warnings ({len(plan.warnings)}):\n"
            for w in plan.warnings:
                plan_header += f"#     ⚠ {w}\n"

    header = (
        f"# Generated by `sndr compose render {cfg.key}` — "
        f"DO NOT edit by hand.\n"
        f"# Re-run `sndr compose render {cfg.key}` to refresh.\n"
        f"# Source preset: {cfg.key} ({cfg.title})\n"
        f"# Maintainer: {cfg.maintainer}\n"
        + plan_header +
        f"#\n"
        f"# Secrets (Etap 0.4 hardening): VLLM_API_KEY is NOT written into the YAML.\n"
        f"# Instead of a literal, `${{VLLM_API_KEY:?...}}` is used — compose\n"
        f"# will pull the value from shell env or a `.env` file next to this YAML.\n"
        f"# Launch:\n"
        f"#   VLLM_API_KEY=mykey docker compose -f docker-compose.yml up -d\n"
        f"# OR place `VLLM_API_KEY=mykey` into `.env` (chmod 0600!)\n"
        f"#\n"
        f"# Usage via sndr:\n"
        f"#   sndr compose up {cfg.key}      # equivalent to docker compose up -d\n"
        f"#   sndr compose logs {cfg.key} -f\n"
        f"#   sndr compose down {cfg.key}\n"
        f"#\n"
    )
    body = yaml.safe_dump(compose, sort_keys=False, default_flow_style=False)
    return header + body


def run_compose_plan_diff(args: argparse.Namespace) -> int:
    """A/B between two patch_plan policies for the same preset.

    Read-only — never renders any compose YAML, never touches the
    runtime. Returns a structured diff of:
      - toggles newly excluded under --to
      - toggles newly included under --to
      - parameter passthrough additions / removals (should always be
        empty in practice — passthrough is policy-independent — but
        surfaced anyway for completeness)

    Useful before flipping a real launch from one policy to another:
    the operator sees the exact env-flag delta they'd be signing up
    for, with role attribution attached for triage.
    """
    import json as _json
    from sndr.model_configs.patch_plan import resolve_patch_plan

    cfg = _resolve(args.config)
    if cfg is None:
        return 2

    try:
        plan_from = resolve_patch_plan(cfg, policy=args.from_policy)
        plan_to = resolve_patch_plan(cfg, policy=args.to_policy)
    except ValueError as e:
        _io.error(str(e))
        return 2

    def _decision_summary(d) -> dict:
        return {
            "patch_id": d.patch_id,
            "env_flag": d.env_flag,
            "value": d.value,
            "role": d.role,
            "reason": d.reason,
        }

    from_included = {d.env_flag: d for d in plan_from.included}
    from_excluded = {d.env_flag: d for d in plan_from.excluded}
    to_included = {d.env_flag: d for d in plan_to.included}
    to_excluded = {d.env_flag: d for d in plan_to.excluded}

    # Cross-set deltas — what crossed the boundary in either direction.
    newly_excluded = [
        _decision_summary(to_excluded[k])
        for k in to_excluded
        if k in from_included
    ]
    newly_included = [
        _decision_summary(to_included[k])
        for k in to_included
        if k in from_excluded
    ]
    unchanged_included = [
        _decision_summary(to_included[k])
        for k in to_included
        if k in from_included
    ]
    unchanged_excluded = [
        _decision_summary(to_excluded[k])
        for k in to_excluded
        if k in from_excluded
    ]
    pt_from_keys = set(plan_from.passthrough)
    pt_to_keys = set(plan_to.passthrough)
    passthrough_diff = {
        "added": sorted(pt_to_keys - pt_from_keys),
        "removed": sorted(pt_from_keys - pt_to_keys),
    }

    payload = {
        "preset": args.config,
        "from_policy": args.from_policy,
        "to_policy": args.to_policy,
        "diff": {
            "newly_excluded": newly_excluded,
            "newly_included": newly_included,
            "unchanged_included": unchanged_included,
            "unchanged_excluded": unchanged_excluded,
            "passthrough_diff": passthrough_diff,
        },
    }

    if args.json:
        print(_json.dumps(payload, indent=2, default=str))
        return 0

    _io.banner(
        f"Plan diff: {args.config}",
        f"{args.from_policy} → {args.to_policy}",
    )
    _io.info(f"  newly excluded: {len(newly_excluded)} toggle(s)")
    _io.info(f"  newly included: {len(newly_included)} toggle(s)")
    _io.info(f"  unchanged included: {len(unchanged_included)}")
    _io.info(f"  unchanged excluded: {len(unchanged_excluded)}")
    if passthrough_diff["added"] or passthrough_diff["removed"]:
        _io.info(
            f"  passthrough: +{len(passthrough_diff['added'])} / "
            f"-{len(passthrough_diff['removed'])}"
        )
    if newly_excluded:
        _io.info("")
        _io.info(f"  ⊘ Newly excluded under '{args.to_policy}':")
        for d in newly_excluded[:25]:
            _io.info(
                f"    - {d['patch_id']:<10} role={d['role']:<22} {d['env_flag']}"
            )
        if len(newly_excluded) > 25:
            _io.info(f"    … and {len(newly_excluded) - 25} more "
                     f"(use --json for full list)")
    if newly_included:
        _io.info("")
        _io.info(f"  + Newly included under '{args.to_policy}':")
        for d in newly_included[:10]:
            _io.info(
                f"    + {d['patch_id']:<10} role={d['role']:<22} {d['env_flag']}"
            )
    return 0


def run_compose_render(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    policy = getattr(args, "policy", None)
    out = render_compose_yaml(cfg, policy=policy)
    if args.output:
        Path(args.output).write_text(out)
        _io.info(f"wrote {args.output} ({len(out)} bytes)")
    else:
        print(out)
    return 0


# ──── docker compose wrappers ─────────────────────────────────────────


def _write_temp_compose(cfg) -> Path:
    """Renders compose into a temp dir and returns the path.

    Etap 0.4 (audit 2026-05-12): tempdir is always `0o700` — even if it
    already exists (mkdir(mode=...) does not change the mode of an existing dir).
    The rendered YAML is `0o600`. Defense-in-depth on a multi-user host.
    """
    import tempfile
    out = render_compose_yaml(cfg)
    tmpdir = Path(tempfile.gettempdir()) / "sndr-compose"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.chmod(tmpdir, 0o700)
    path = tmpdir / f"docker-compose.{cfg.key}.yml"
    path.write_text(out)
    os.chmod(path, 0o600)
    return path


def _docker_compose(*args, dry_run: bool = False) -> int:
    if shutil.which("docker") is None:
        _io.error("docker not on PATH")
        return 1
    cmd = ["docker", "compose"] + list(args)
    if dry_run:
        _io.info(f"[dry-run] {' '.join(cmd)}")
        return 0
    r = subprocess.run(cmd)
    return r.returncode


def run_compose_up(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    path = _write_temp_compose(cfg)
    _io.info(f"using compose file: {path}")
    return _docker_compose("-f", str(path), "up", "-d")


def run_compose_down(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    path = _write_temp_compose(cfg)
    return _docker_compose("-f", str(path), "down")


def run_compose_logs(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    path = _write_temp_compose(cfg)
    extra = ["--tail", args.lines]
    if args.follow:
        extra.append("-f")
    return _docker_compose("-f", str(path), "logs", *extra)
