# SPDX-License-Identifier: Apache-2.0
"""SNDR Core CLI — `sndr launch <config>` first-run + day-to-day driver.

Composes existing infrastructure (no manual flag-by-flag rebuild):

  - `model_configs.registry.get(key)` — load a preset YAML by key
  - `cfg.to_launch_script(host_paths=...)` — schema-driven renderer
    (handles system_env, genesis_env, docker mounts with symbolic refs,
    vllm serve flags, and bare-metal vs docker dispatch)
  - `apply.run(apply=True)` — patcher run BEFORE handing off to vllm
  - `subprocess` to exec the rendered bash script

In dry-run mode, prints the script to stdout. Useful for diagnostic +
CI smoke + integration into external orchestrators (k8s, systemd).

F-004 / L-01 fix (audit 2026-05-07): previously this module rebuilt
the vllm serve cmd flag-by-flag with `_build_vllm_serve_args`, which
duplicated the schema's `to_launch_script` and missed env exports +
docker mounts entirely. Now the CLI is a thin driver around the
schema renderer.

L-02 fix: previously read non-existent `cfg.env` field. The schema
has `system_env` (CUDA_VISIBLE_DEVICES etc.) and `genesis_env`
(GENESIS_ENABLE_* flags), both rendered into the script by
`to_launch_script`.

L-03 fix: non-interactive mode no longer silently defaults to the
alphabetically-first config — fails with a helpful error instead.

Usage:
  sndr launch <config_key>           # render + apply patches + exec
  sndr launch --dry-run <config_key> # print the bash script, don't exec
  sndr launch                         # interactive picker (TTY only)
  sndr launch --port 8001 <key>      # override config's port
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Any

from . import _io


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "launch",
        help="Launch vLLM with a SNDR Core preset.",
        description="`sndr launch [config_key] [opts]` — driver around vllm serve.",
    )
    p.add_argument("config_key", nargs="?", default=None,
                   help="model_config key (e.g. a5000-2x-27b-int4-tq-k8v4). "
                        "Required when stdin is not a TTY (-y / non-interactive).")
    p.add_argument("--dry-run", action="store_true",
                   help="Render the launch script to stdout without exec.")
    p.add_argument("--port", type=int, default=None,
                   help="Override config's port (default: keep config value).")
    p.add_argument("--skip-apply", action="store_true",
                   help="Skip the patcher apply phase. Use only when patches "
                        "are already applied via a separate flow.")
    p.add_argument("-y", "--non-interactive", action="store_true",
                   help="Accept defaults without prompting; fails if "
                        "config_key is omitted (no alphabetical fallback).")
    p.add_argument(
        "--strict-image",
        choices=("on", "off", "auto"),
        default="auto",
        help="Image digest enforcement (T1.6 / audit §7.4). 'on': "
             "refuse to launch when local image digest doesn't match "
             "the preset's docker.image_digest. 'off': skip entirely. "
             "'auto' (default): enforce only when the preset declares "
             "an image_digest field.",
    )
    # Stage-out CLI gaps (audit 2026-05-12, post-MASTER_REMEDIATION_PLAN).
    # Each flag below short-circuits before exec so the operator can
    # split a launch into discrete steps (preflight → pull → check-deps
    # → exec) for CI pipelines / runbooks that need the granularity.
    p.add_argument(
        "--preflight-only", action="store_true",
        help="Run the preflight gate (constraints + image digest + "
             "mount resolution) and exit without invoking vllm. "
             "Useful in CI to fail fast before the slow apply step.",
    )
    p.add_argument(
        "--pull", action="store_true",
        help="`docker pull` the preset's image before exec. Only "
             "meaningful for docker-backed configs (no-op for "
             "bare-metal). Exits non-zero if pull fails.",
    )
    p.add_argument(
        "--check-deps", action="store_true",
        help="Run `sndr deps inspect` against the preset before exec "
             "and abort if any required dependency is missing. "
             "Equivalent to `sndr deps inspect <key>` followed by launch.",
    )
    p.set_defaults(func=run_launch)


def _list_available_configs() -> list[str]:
    """Return sorted list of available model_config keys."""
    try:
        from vllm.sndr_core.model_configs.registry import list_keys
        return sorted(list_keys())
    except Exception:
        return []


def _resolve_config(key: str | None, non_interactive: bool):
    """Find a model_config by key or prompt the operator. Returns (cfg, key).

    L-03 fix: non-interactive mode requires an explicit key (no
    alphabetical default) so CI / unattended scripts fail loud rather
    than booting whatever happens to sort first.
    """
    available = _list_available_configs()
    if not available:
        _io.fatal("no model_configs available — run `sndr install` first", 2)

    if key is None:
        if non_interactive or not sys.stdin.isatty():
            _io.error(
                "no config_key provided and stdin is not a TTY.\n"
                "    Available: " + ", ".join(available[:6])
                + (" …" if len(available) > 6 else "")
                + "\n    Re-run as: sndr launch <key>"
            )
            sys.exit(2)
        _io.info("Available presets:")
        for i, k in enumerate(available, 1):
            print(f"    [{i}] {k}")
        ans = _io.prompt("Choose preset (number or key)", default="1",
                         non_interactive=False)
        try:
            idx = int(ans) - 1
            key = available[idx]
        except (ValueError, IndexError):
            key = ans  # treat as literal key

    # V2 alias resolution path: if `<key>.yaml` exists under
    # `builtin/presets/`, treat it as a triplet pointer and compose.
    # Falls through to legacy V1 registry on miss so existing presets
    # keep working unchanged.
    try:
        from vllm.sndr_core.model_configs.registry_v2 import (
            load_alias as _v2_load_alias,
        )
        from vllm.sndr_core.model_configs.schema import SchemaError as _SchemaError
        try:
            cfg = _v2_load_alias(key)
            return cfg, key
        except _SchemaError:
            pass  # Not a V2 alias; try V1 registry below.
    except ImportError:
        pass

    try:
        from vllm.sndr_core.model_configs.registry import get as get_config
        cfg = get_config(key)
        if cfg is None:
            raise RuntimeError(f"key {key!r} not in V1 registry")
    except Exception as e:
        _io.fatal(f"config {key!r} not found ({e})", 2)
    return cfg, key


def _emit_unresolved_mount_diagnostics(
    script: str,
    host_paths: dict[str, str] | None,
) -> None:
    """P1-6 fix (audit 2026-05-08): scan the rendered launch script for
    `${var}` placeholders that survived `resolve_symbolic_mounts(strict=False)`
    and surface them on stderr so dry-run callers know the rendered
    script is NOT immediately runnable.

    The dry-run preserves placeholders by design (preview should always
    succeed even on machines missing some host.yaml entries), but
    operators have repeatedly missed this and tried to `bash <(sndr
    launch --dry-run …)` straight into a broken docker run. Make the
    gap impossible to miss.
    """
    import re
    placeholders = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", script)))
    if not placeholders:
        return
    known = set((host_paths or {}).keys())
    missing = [p for p in placeholders if p not in known]
    if not missing:
        return
    _io.warn("\nUNRESOLVED MOUNTS — host.yaml is missing entries for:")
    for var in missing:
        _io.info(f"  ${{{var}}}")
    _io.info(
        "\nThe rendered script above is a PREVIEW. Live `sndr launch` "
        "(without --dry-run) would refuse these placeholders.\n"
        "Fix:  edit ~/.sndr/host.yaml (or ~/.genesis/host.yaml legacy) "
        "and add the missing keys.\n"
        "Or:   set them in the environment before re-running."
    )


def _emit_preflight_render_diagnostic(
    cfg: Any,
    host_paths: dict[str, str] | None,
    error: Exception,
) -> None:
    """Replace a fatal `to_launch_script` traceback with a structured
    preflight finding when the caller asked for `--preflight-only`.

    The point of preflight is to *list what's missing*. A SchemaError on
    `resolve_symbolic_mounts: unknown variable 'models_dir'` therefore
    becomes a single actionable diagnostic listing every unresolved
    mount variable, plus the env knobs an operator can set without
    editing host.yaml.
    """
    import re
    from vllm.sndr_core.model_configs.host import _ENV_OVERRIDES

    known = set((host_paths or {}).keys())
    mounts = list(getattr(cfg.docker, "mounts", []) or []) if getattr(cfg, "docker", None) else []
    placeholders: set[str] = set()
    for m in mounts:
        for hit in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(m)):
            placeholders.add(hit)
    missing = sorted(p for p in placeholders if p not in known)

    _io.warn(f"preflight: render failed — {type(error).__name__}: {error}")
    _io.info("")
    _io.info("Mount variables that need a value (from host.yaml or env):")
    if not missing:
        _io.info("  (no unresolved mount placeholders — root cause is elsewhere; see error above)")
    else:
        for var in missing:
            envs = _ENV_OVERRIDES.get(var, ())
            hint = ", ".join(envs) if envs else "(no env alias — set via host.yaml)"
            _io.info(f"  ${{{var}}}  →  env: {hint}")
    _io.info("")
    _io.info("Fix:")
    _io.info("  1. Edit ~/.sndr/host.yaml (or ~/.genesis/host.yaml) and add the missing keys, OR")
    _io.info("  2. Export the env variables listed above before re-running.")
    _io.info("  3. Then re-run `sndr launch <preset> --preflight-only` to confirm.")


def _load_host_paths() -> dict[str, str] | None:
    """Best-effort load of host.yaml symbolic mount mapping. Returns
    None if not configured — the renderer will then probe defaults."""
    try:
        from vllm.sndr_core.model_configs.host import load_host_config
        hc = load_host_config()
        return dict(hc.paths) if hc and hc.paths else None
    except Exception:
        return None


def _maybe_override_port(cfg: Any, port: int | None) -> None:
    """Apply `--port` override directly on the config object so
    `to_launch_script` picks it up."""
    if port is None:
        return
    if hasattr(cfg, "docker") and cfg.docker:
        cfg.docker.port = port
    if hasattr(cfg, "port"):
        cfg.port = port


def _verify_image_digest(cfg: Any, mode: str) -> int:
    """T1.6 / audit §7.4: verify local docker image matches the
    `image_digest` field declared on the preset.

    Args:
      cfg: ModelConfig (must have a `docker` block; otherwise no-op).
      mode: 'on' | 'off' | 'auto' (from --strict-image flag).

    Returns:
      0 — verification passed, OR skipped per mode policy.
      non-zero — strict mismatch; caller must abort the launch.
    """
    if mode == "off":
        return 0
    docker = getattr(cfg, "docker", None)
    if docker is None:
        return 0  # Bare-metal launch — no image to verify
    expected = getattr(docker, "image_digest", None)
    if not expected:
        if mode == "on":
            _io.error(
                "--strict-image=on but preset has no docker.image_digest. "
                "Either add the digest to the preset YAML or run with "
                "--strict-image=auto."
            )
            return 2
        return 0  # auto + no digest declared → fall through

    # auto + digest declared, OR strict-image=on + digest declared
    import shutil
    if not shutil.which("docker"):
        _io.warn(
            "image_digest declared but `docker` not on PATH — "
            "cannot verify. Skipping in --strict-image=auto."
        )
        return 0 if mode == "auto" else 2

    image_ref = getattr(docker, "image", "")
    try:
        res = subprocess.run(
            ["docker", "inspect", "--format", "{{json .RepoDigests}}", image_ref],
            capture_output=True, text=True, timeout=8,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _io.warn(f"docker inspect failed ({e}) — skipping digest check")
        return 0 if mode == "auto" else 2

    if res.returncode != 0:
        _io.warn(
            f"docker inspect returned non-zero for {image_ref}: "
            f"{res.stderr.strip() or '(no stderr)'} — skipping digest check"
        )
        return 0 if mode == "auto" else 2

    import json
    try:
        local_digests = json.loads(res.stdout.strip()) or []
    except json.JSONDecodeError:
        local_digests = []

    if not isinstance(local_digests, list):
        local_digests = []

    if expected in local_digests:
        _io.success(f"image digest verified: {expected}")
        return 0

    _io.error(
        f"IMAGE DIGEST MISMATCH:\n"
        f"  expected: {expected}\n"
        f"  local:    {local_digests or '(none)'}\n"
        "The local image does not match the preset's pin. Either "
        "pull the matching image (`docker pull <image>@<digest>`) or "
        "rerun with --strict-image=off if you've intentionally "
        "swapped the image."
    )
    return 1


def _run_apply_phase() -> int:
    """Apply patches via the orchestrator. Returns 0 on success, non-zero
    on failure (caller should abort the launch in that case)."""
    _io.step(1, 2, "Applying SNDR patches at boot")
    try:
        from vllm.sndr_core.apply import run as apply_run
        stats = apply_run(verbose=False, apply=True)
    except Exception as e:
        _io.error(f"apply phase crashed: {type(e).__name__}: {e}")
        return 1
    _io.success(
        f"applied={stats.applied_count}, "
        f"skipped={stats.skipped_count}, failed={stats.failed_count}"
    )
    if stats.failed_count > 0:
        _io.error("patch application FAILED — refusing to launch vllm")
        return 1
    return 0


def _run_docker_pull(cfg) -> int:
    """`--pull`: docker-pull the preset's image. Returns shell rc."""
    import shutil
    docker = getattr(cfg, "docker", None)
    if docker is None:
        _io.info("--pull: bare-metal config, skipping docker pull")
        return 0
    image = docker.effective_image_ref()
    if not image:
        _io.warn("--pull: preset has no docker.image — nothing to pull")
        return 0
    if shutil.which("docker") is None:
        _io.error("--pull: docker binary not on PATH")
        return 2
    _io.step(0, 0, f"docker pull {image}")
    r = subprocess.run(["docker", "pull", image])
    if r.returncode != 0:
        _io.error(f"docker pull failed (rc={r.returncode})")
    return r.returncode


def _run_check_deps(cfg, key: str) -> int:
    """`--check-deps`: run host dep inspection and abort on missing required deps.

    Uses the canonical `vllm.sndr_core.deps` planner — the same code path
    that drives `sndr deps plan` — as the primary signal. Returns
    non-zero whenever `plan.blockers()` is non-empty (Docker / NVIDIA
    runtime / model directory / Python deps / vLLM pin).

    The legacy caveat matcher (env-flag combinations etc.) is kept as a
    secondary check; either signal can fail the preflight.

    Degrades to a WARN when the deps module isn't importable (e.g. CI
    container without `psutil`, sandboxed dev box) rather than blocking
    launch — the caller still sees the diagnostic.
    """
    _io.step(0, 0, "Checking host dependencies")

    # ─── Canonical planner check (same path as `sndr deps plan`) ──────
    plan = None
    try:
        from vllm.sndr_core.deps import inspect_host, plan_changes
        inv = inspect_host()
        plan = plan_changes(cfg, inv)
    except Exception as e:
        _io.warn(f"--check-deps: deps planner unavailable ({e}); skipping planner path")
        plan = None

    plan_blockers: list = []
    if plan is not None:
        plan_blockers = list(plan.blockers())
        for item in plan_blockers:
            _io.error(f"[{item.scope}] {item.target}")
            _io.error(f"    {item.reason}")
            if item.suggested_command:
                _io.error(f"    hint: {item.suggested_command}")

    # ─── Legacy caveat matcher (secondary; env-flag combos etc.) ──────
    try:
        from vllm.sndr_core.deps.checkers import inspect_host as _inspect_host
        from vllm.sndr_core.caveats import match_caveats
        facts = _inspect_host().to_dict()
        facts.setdefault("genesis_env", dict(getattr(cfg, "genesis_env", {}) or {}))
        triggered = match_caveats(facts)
    except Exception as e:
        _io.warn(f"--check-deps: caveats matcher unavailable ({e}); skipping legacy path")
        triggered = []

    caveat_errs = [c for c in triggered if c.severity == "error"]
    for c in caveat_errs:
        _io.error(f"[{c.id}] {c.title}")
        _io.error(f"    {c.message}")

    total_problems = len(plan_blockers) + len(caveat_errs)
    if total_problems:
        _io.error(
            f"--check-deps: {len(plan_blockers)} planner blocker(s) + "
            f"{len(caveat_errs)} caveat error(s); fix host before launching"
        )
        return 2
    return 0


def run_launch(opts: argparse.Namespace) -> int:
    """Render `cfg.to_launch_script()` + apply patches + exec the script."""
    cfg, key = _resolve_config(opts.config_key, opts.non_interactive)
    _maybe_override_port(cfg, opts.port)

    _io.banner(f"SNDR Launch: {key}",
               f"port={getattr(getattr(cfg, 'docker', None), 'port', None) or 'config'}")

    # Optional preludes — each runs only when its flag is passed and
    # short-circuits the launch on failure.
    if getattr(opts, "check_deps", False):
        rc = _run_check_deps(cfg, key)
        if rc != 0:
            return rc
    if getattr(opts, "pull", False):
        rc = _run_docker_pull(cfg)
        if rc != 0:
            return rc

    # F-016: resolve symbolic mounts via host.yaml when the renderer needs them.
    # live launch uses strict_mounts=True so missing host.yaml entries
    # fail loudly instead of producing an unbootable script with literal
    # `${var}` placeholders. Dry-run keeps strict=False so previews show
    # the placeholder pattern.
    # `--preflight-only` also drops strict_mounts so the render produces
    # a partial script and then this layer surfaces the missing
    # host-paths as a structured diagnostic — the whole point of
    # preflight is "tell me what's not set up", never a Python traceback.
    host_paths = _load_host_paths()
    is_preflight = getattr(opts, "preflight_only", False)
    strict_mounts = not (opts.dry_run or is_preflight)
    try:
        script = cfg.to_launch_script(
            host_paths=host_paths,
            strict_mounts=strict_mounts,
        )
    except Exception as e:
        if is_preflight:
            _emit_preflight_render_diagnostic(cfg, host_paths, e)
            return 1
        _io.fatal(f"render failed: {type(e).__name__}: {e}", 1)

    if opts.dry_run:
        # L-01: emit the schema-rendered script verbatim.
        sys.stdout.write(script)
        sys.stdout.flush()
        # explicit unresolved-mount diagnostics
        # in dry-run. Operators have repeatedly missed that
        # `${models_dir}` style placeholders in the rendered script
        # mean their host.yaml is incomplete. We surface them on stderr
        # with a clear "fix host.yaml" pointer so the preview output
        # stays clean for piping but the warning is visible.
        _emit_unresolved_mount_diagnostics(script, host_paths)
        return 0

    # T1.8 (audit closure §7.2): check declared `constraints` block.
    # min_gpu_memory / min_gpu_count / forbidden_flags violations abort
    # the launch with a precise message, instead of failing inside vllm
    # boot with a confusing CUDA error / OOM.
    constraints = getattr(cfg, "constraints", None)
    if constraints is not None:
        violations = constraints.check(
            hw=getattr(cfg, "hardware", None),
            vllm_extra_args=list(getattr(cfg, "vllm_extra_args", []) or []),
        )
        if violations:
            _io.error("preset constraints violated:")
            for v in violations:
                _io.error(f"  - {v}")
            _io.info(
                "Either fix the violation, or remove the constraint from "
                "the preset YAML if it no longer applies."
            )
            return 2

    # T1.6 (audit closure §7.4): verify image digest BEFORE we hand off
    # to bash. This catches "operator pulled `:latest` but the preset
    # was tested against a specific digest" — a class of regression
    # that previously surfaced as mysterious behavioral diffs only
    # visible by tailing container logs.
    digest_mode = getattr(opts, "strict_image", "auto")
    rc = _verify_image_digest(cfg, digest_mode)
    if rc != 0:
        return rc

    # `--preflight-only`: every gate above has passed (resolve, render,
    # constraints, digest); now return before the slow apply + exec.
    if getattr(opts, "preflight_only", False):
        _io.info("--preflight-only: all checks passed — exiting without exec")
        return 0

    # host apply phase is meaningful for
    # bare-metal launches (host == target vllm). For Docker launches
    # the patches must apply INSIDE the container — host site-packages
    # may not even have vllm/torch installed. The container bootstrap
    # (rendered into the launch script) handles its own apply.
    is_docker = bool(getattr(cfg, "docker", None))
    if opts.skip_apply:
        _io.info("(--skip-apply: assuming patches already applied)")
    elif is_docker:
        _io.info(
            "(docker config — host apply phase skipped; container "
            "bootstrap will apply patches inside the container)"
        )
    else:
        rc = _run_apply_phase()
        if rc != 0:
            return rc

    _io.step(2, 2, "Invoking vllm serve via rendered script")

    # Persist the script so the running process has a clean handle and so
    # operators can re-run / inspect after launch.
    fd, path = tempfile.mkstemp(prefix=f"sndr-launch-{key}-", suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)
        _io.info(f"rendered script: {path}")
        try:
            os.execvp("bash", ["bash", path])
        except FileNotFoundError:
            _io.fatal("`bash` not on PATH", 2)
        except Exception as e:
            _io.fatal(f"exec failed: {type(e).__name__}: {e}", 1)
    except BaseException:
        # exec replaces the process — only reached on failure paths
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return 0  # unreachable on success (execvp replaces process)


__all__ = ["add_argparser", "run_launch"]
