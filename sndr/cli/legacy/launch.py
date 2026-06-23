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
import copy
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
        "--skip-autodetect", action="store_true",
        help="Skip the A1-A9 autodetect preflight gate (GPU count, vLLM "
             "pin, model/drafter path existence, HF cache mount, "
             "max-model-len sanity, served-name default, port conflict, "
             "repo resolution). Use only when you know the host state is "
             "fine and want to bypass the checks.",
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
    p.add_argument(
        "--policy",
        choices=("compat", "safe", "minimal"),
        default=None,
        help=(
            "filter cfg.genesis_env through the "
            "patch_plan resolver before rendering the launch script. "
            "compat keeps every truthy toggle; safe drops role='no_op'; "
            "minimal additionally drops role in {suspected_regression, "
            "unknown}. Non-toggle GENESIS_* parameter keys always pass "
            "through. Omit the flag to keep the legacy unfiltered "
            "launch matrix."
        ),
    )
    p.add_argument(
        "--extra-env", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Inject an extra env var into the rendered docker-run (or "
            "bare-metal export block) for this launch only. Repeatable. "
            "KEY starting with GENESIS_ or SNDR_ lands in cfg.genesis_env; "
            "everything else lands in cfg.system_env. Last-wins on conflict "
            "with the preset's own env. Useful for one-shot operator probes "
            "(e.g. GENESIS_ENABLE_PN248_ACCEPTANCE_TRACE=1) without editing "
            "the preset/profile YAML."
        ),
    )
    p.set_defaults(func=run_launch)


def _list_available_configs() -> list[str]:
    """Return sorted list of available model_config keys (V1 + V2 aliases).

    Phase 10.5 (2026-06-01): V1 monolithic preset tier fully retired,
    so the V1 registry is empty by design. V2 alias files under
    `builtin/presets/<alias>.yaml` are the operator-facing canonical
    keys post-sunset — include them here so `_resolve_config` does not
    fatal-on-empty when only V2 aliases exist.
    """
    keys: set[str] = set()
    try:
        from sndr.model_configs.registry import list_keys
        keys.update(list_keys())
    except Exception:
        pass
    try:
        from sndr.model_configs.registry_v2 import list_presets
        keys.update(list_presets())
    except Exception:
        pass
    return sorted(keys)


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
        from sndr.model_configs.registry_v2 import (
            load_alias as _v2_load_alias,
        )
        from sndr.model_configs.schema import SchemaError as _SchemaError
        try:
            cfg = _v2_load_alias(key)
            return cfg, key
        except _SchemaError:
            pass  # Not a V2 alias; try V1 registry below.
    except ImportError:
        pass

    try:
        from sndr.model_configs.registry import get as get_config
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

    All output goes to stderr (`err=True`): this helper is only ever
    called on the `--dry-run` path, where stdout must carry ONLY the
    rendered script so `sndr launch <preset> --dry-run -y | bash`
    (or `bash -n`) stays runnable. Previously these lines printed to
    stdout, contaminating the script and breaking the pipe — the
    docstring claimed stderr but the code did not deliver it.
    """
    import re
    placeholders = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", script)))
    if not placeholders:
        return
    known = set((host_paths or {}).keys())
    missing = [p for p in placeholders if p not in known]
    if not missing:
        return
    _io.warn("\nUNRESOLVED MOUNTS — host.yaml is missing entries for:", err=True)
    for var in missing:
        _io.info(f"  ${{{var}}}", err=True)
    _io.info(
        "\nThe rendered script above is a PREVIEW. Live `sndr launch` "
        "(without --dry-run) would refuse these placeholders.\n"
        "Fix:  edit ~/.sndr/host.yaml (or ~/.genesis/host.yaml legacy) "
        "and add the missing keys.\n"
        "Or:   set them in the environment before re-running.",
        err=True,
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
    from sndr.model_configs.host import _ENV_OVERRIDES

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
        from sndr.model_configs.host import load_host_config
        hc = load_host_config()
        return dict(hc.paths) if hc and hc.paths else None
    except Exception:
        return None


def _maybe_override_port(cfg: Any, port: int | None) -> Any:
    """Apply a `--port` override and return the config to render.

    B8 (2026-06-22): the override used to mutate the passed config IN
    PLACE. `cfg` can come from a process-cached registry, so mutating it
    leaked the override into every later caller in the same process
    (e.g. a second `sndr launch` of a different preset would inherit a
    stale port). We now deep-copy the config before mutating so the
    shared/cached object stays pristine, and return the copy.

    When `port` is None there is nothing to override → return `cfg`
    unchanged (no copy, keeps the common path cheap).
    """
    if port is None:
        return cfg
    cfg = copy.deepcopy(cfg)
    if hasattr(cfg, "docker") and cfg.docker:
        cfg.docker.port = port
    if hasattr(cfg, "port"):
        cfg.port = port
    return cfg


def _run_autodetect_gate(cfg: Any, host_paths: dict[str, str] | None) -> int:
    """A1-A9 autodetect preflight (see cli/legacy/preflight.py).

    Runs BEFORE render/exec so cryptic engine-init failures (missing
    model/drafter checkpoint, GPU-count mismatch, port conflict, pin
    mismatch, over-long max_model_len) become clear operator messages.

    Returns 0 when there are no errors (warnings are surfaced but allow
    the launch to proceed), non-zero when at least one error fired and
    the caller must abort.

    Note: the A7 served-model-name default mutates `cfg` in place. The
    caller passes an already-isolated config (deep-copied on the --port
    path; for the no-port path we copy here) so the shared/cached
    registry object is never touched.
    """
    from . import preflight as _preflight

    result = _preflight.run_autodetect_preflight(cfg, host_paths)
    for w in result.warnings:
        _io.warn(f"preflight [{w.code}] {w.message}")
    for e in result.errors:
        _io.error(f"preflight [{e.code}] {e.message}")
    if not result.ok:
        _io.info(
            "Autodetect preflight found blocking issues (above). Fix them, "
            "or pass --skip-autodetect to bypass."
        )
        return 2
    return 0


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
        from sndr.apply import run as apply_run
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

    Uses the canonical `sndr.deps` planner — the same code path
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
        from sndr.deps import inspect_host, plan_changes
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
        from sndr.deps.checkers import inspect_host as _inspect_host
        from sndr.caveats import match_caveats
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


def _parse_extra_env(items: list[str]) -> dict[str, str]:
    """Parse `--extra-env KEY=VALUE` items into a dict.

    Split is on the FIRST `=` only — values may contain further `=`
    characters (e.g. JSON payloads). Empty values are allowed (some
    env vars are presence-checked). Empty key or missing `=` are
    rejected with a helpful message.
    """
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            _io.fatal(
                f"--extra-env: expected KEY=VALUE, got {raw!r} "
                f"(no '=' separator)",
                2,
            )
        k, _, v = raw.partition("=")
        k = k.strip()
        if not k:
            _io.fatal(
                f"--extra-env: empty key in {raw!r}", 2,
            )
        out[k] = v
    return out


def _apply_extra_env(
    cfg: Any, extra_env: dict[str, str], *, err: bool = False,
) -> None:
    """Merge `--extra-env` overrides into the right env block on cfg.

    GENESIS_* / SNDR_* keys go into cfg.genesis_env (so they get the
    same rendering + visibility as preset-declared toggles); other
    keys go into cfg.system_env. Last-wins on conflict with the
    preset's own env. `err=True` (passed on the `--dry-run` path)
    routes the override log to stderr so the rendered script on stdout
    stays clean for piping.
    """
    if not extra_env:
        return
    if not hasattr(cfg, "genesis_env") or cfg.genesis_env is None:
        cfg.genesis_env = {}
    if not hasattr(cfg, "system_env") or cfg.system_env is None:
        cfg.system_env = {}
    for k, v in extra_env.items():
        target_block = "genesis_env" if (
            k.startswith("GENESIS_") or k.startswith("SNDR_")
        ) else "system_env"
        target = getattr(cfg, target_block)
        prev = target.get(k)
        target[k] = v
        if prev is not None and prev != v:
            _io.warn(
                f"  --extra-env override: {target_block}[{k}] "
                f"{prev!r} → {v!r}",
                err=err,
            )
        else:
            _io.info(f"  --extra-env: {target_block}[{k}]={v!r}", err=err)


def _maybe_apply_patch_policy(
    cfg: Any, opts: argparse.Namespace, *, err: bool = False,
) -> None:
    """Apply --policy filter to cfg.genesis_env in place.

    No-op when ``opts.policy`` is None (legacy unfiltered path).

    The resolver itself is read-only — it doesn't mutate cfg — so we
    overwrite cfg.genesis_env explicitly. The replacement is the
    union of policy-included toggles AND parameter passthrough, so
    dependent patches keep their configuration keys (PN95 stays
    armed via GENESIS_PN95_CONFIG_KEY, etc.).

    `err=True` (passed on the `--dry-run` path) routes the policy
    summary + warnings to stderr so stdout carries only the rendered
    script.
    """
    policy = getattr(opts, "policy", None)
    if policy is None:
        return
    from sndr.model_configs.patch_plan import resolve_patch_plan
    plan = resolve_patch_plan(cfg, policy=policy)
    _io.info(
        f"  patch plan policy={policy}: "
        f"{len(plan.included)} included / {len(plan.excluded)} excluded / "
        f"{len(plan.passthrough)} passthrough",
        err=err,
    )
    if plan.warnings:
        _io.warn(f"  ⚠ {len(plan.warnings)} patch_plan warning(s):", err=err)
        for w in plan.warnings:
            _io.warn(f"    {w}", err=err)
    cfg.genesis_env = plan.env


def _warn_about_non_full_enabled_patches(cfg: Any, *, err: bool = False) -> None:
    """Scan cfg.genesis_env and warn when an enabled toggle maps to a
    patch with implementation_status in {partial, placeholder,
    marker_only}.

    Why this matters: a partial patch (e.g. PN95) may install but
    skip critical wiring; a placeholder (PN64) has no real
    implementation; a marker_only entry with default_on=True (legacy
    P1/P17/etc.) appears enabled but has no apply_module so nothing
    fires. Surfacing the status before launch avoids silent
    "feature is on" mistakes.

    `err=True` routes every advisory line to stderr. Passed on the
    `--dry-run` path so these warnings do not contaminate the rendered
    script on stdout (which callers pipe to `bash`).
    """
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except Exception:
        return
    flag_to_meta: dict[str, tuple[str, dict]] = {}
    for pid, meta in PATCH_REGISTRY.items():
        if not isinstance(meta, dict):
            continue
        flag = meta.get("env_flag")
        if isinstance(flag, str) and flag:
            flag_to_meta[flag] = (pid, meta)

    partial: list[tuple[str, str]] = []
    placeholder: list[tuple[str, str]] = []
    marker: list[tuple[str, str]] = []
    for flag, value in (cfg.genesis_env or {}).items():
        if str(value).strip().lower() not in ("1", "true", "yes", "on"):
            continue
        hit = flag_to_meta.get(flag)
        if hit is None:
            continue
        pid, meta = hit
        impl = meta.get("implementation_status", "full")
        if impl == "partial":
            partial.append((pid, flag))
        elif impl == "placeholder":
            placeholder.append((pid, flag))
        elif impl == "marker_only":
            marker.append((pid, flag))

    if partial:
        _io.warn(
            f"  ⚠ {len(partial)} enabled patch(es) have "
            f"implementation_status='partial' — wiring incomplete, "
            f"runtime behaviour may differ from the documented effect:",
            err=err,
        )
        for pid, flag in partial:
            _io.warn(f"    {pid:<10} {flag}", err=err)
    if placeholder:
        _io.warn(
            f"  ⚠ {len(placeholder)} enabled patch(es) have "
            f"implementation_status='placeholder' — no real "
            f"implementation, runtime is a no-op:",
            err=err,
        )
        for pid, flag in placeholder:
            _io.warn(f"    {pid:<10} {flag}", err=err)
    if marker:
        # marker_only with default_on=True is a legacy registry pattern
        # — informational only. Lowered to info severity so the warning
        # block stays focused on actual breakage classes.
        _io.info(
            f"  · {len(marker)} enabled marker_only patch(es) "
            f"(advisory/historical, no apply module):",
            err=err,
        )
        for pid, flag in marker[:5]:
            _io.info(f"    {pid:<10} {flag}", err=err)
        if len(marker) > 5:
            _io.info(f"    … and {len(marker) - 5} more", err=err)


def run_launch(opts: argparse.Namespace) -> int:
    """Render `cfg.to_launch_script()` + apply patches + exec the script."""
    cfg, key = _resolve_config(opts.config_key, opts.non_interactive)
    # B8: returns a deep-copied cfg when --port is set; mutating in place
    # would leak the override into the process-cached registry object.
    cfg = _maybe_override_port(cfg, opts.port)

    # On `--dry-run`, ALL advisory output must go to stderr so stdout
    # carries ONLY the rendered script — `sndr launch <preset> --dry-run
    # -y | bash` (or `bash -n`) must stay runnable. The live launch path
    # keeps its existing stdout behaviour (err=False).
    advisory_err = bool(opts.dry_run)

    _io.banner(f"SNDR Launch: {key}",
               f"port={getattr(getattr(cfg, 'docker', None), 'port', None) or 'config'}",
               err=advisory_err)

    # Apply optional patch_plan policy filter (--policy compat|safe|minimal).
    # No-op when --policy isn't passed; otherwise replaces cfg.genesis_env
    # with the policy-filtered + parameter-passthrough union.
    _maybe_apply_patch_policy(cfg, opts, err=advisory_err)

    # `--extra-env KEY=VALUE` overrides land AFTER policy filtering so an
    # operator can force-enable a key that policy would have dropped.
    extra_env_items = getattr(opts, "extra_env", None) or []
    if extra_env_items:
        _apply_extra_env(cfg, _parse_extra_env(extra_env_items), err=advisory_err)

    # Surface partial / placeholder / marker_only patches that ended up
    # in the launch env after policy filtering. These are the silent-
    # success failure classes — patch enabled, dispatcher applies, but
    # the runtime behaviour differs from what the patch advertises.
    _warn_about_non_full_enabled_patches(cfg, err=advisory_err)

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

    # A1-A9 autodetect preflight gate. Runs for live launch AND
    # --preflight-only, BEFORE the render so A7 (served-model-name
    # default) reaches the rendered script. Skipped for --dry-run (a pure
    # preview must succeed on any machine without touching nvidia-smi /
    # sockets) and when the operator passes --skip-autodetect. The gate
    # mutates cfg (A7) and reads cfg, so isolate it from the cached
    # registry object first (no-op extra cost when --port already copied).
    if not opts.dry_run and not getattr(opts, "skip_autodetect", False):
        cfg = copy.deepcopy(cfg)
        rc = _run_autodetect_gate(cfg, host_paths)
        if rc != 0:
            return rc

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
