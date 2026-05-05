# SPDX-License-Identifier: Apache-2.0
"""Genesis model-config CLI — comprehensive launch+verify orchestrator.

Subcommands:
    list                          enumerate all available configs
    show <key>                    print full YAML
    render <key>                  emit launch script to stdout
    save <key> <path>             write launch script to disk
    audit <key>                   run audit_rules (16 rules: P98/P67/PN59/...)
    validate <key>                schema + audit + cross-ref PATCH_REGISTRY
    preflight <key>               pre-launch env checks (mounts/GPU/pin)
    diagnose <key>                runtime diagnose (running container)
    verify <key>                  bench + diff vs reference (CI gate)
    where <key>                   show source tier
    new <key> --template <other>  clone existing
    new <key> --from-running <c>  capture from running docker
    launch <key> [--dry-run]      execute the rendered script
    bench-and-update <key>        boot + bench + write metrics back
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vllm._genesis.model_configs import (
    ModelConfig, load_all, get, list_keys, dump_yaml,
)
from vllm._genesis.model_configs.registry import source_of
from vllm._genesis.model_configs.audit_rules import audit
from vllm._genesis.model_configs.preflight import (
    preflight_all, has_blockers as preflight_blockers,
)
from vllm._genesis.model_configs.diagnose import (
    diagnose_all, has_blockers as diagnose_blockers,
)
from vllm._genesis.model_configs.verify import (
    verify, has_blockers as verify_blockers,
)


def _cfg_or_die(key: str):
    """Return cfg or print error + raise SystemExit(1) for CLI use."""
    cfg = get(key)
    if cfg is None:
        print(f"ERROR: config '{key}' not found", file=sys.stderr)
        print(f"Available: {', '.join(list_keys())}", file=sys.stderr)
        raise SystemExit(1)
    return cfg


def cmd_list(args) -> int:
    configs = load_all()
    if not configs:
        print("(no configs found in vllm/_genesis/model_configs/builtin/)")
        return 0
    print(f"Genesis model configs ({len(configs)}):\n")
    print(f"  {'KEY':<38}  {'TIER':<10}  {'TPS':>7}  {'TOOL':<7}  {'CV%':>6}  TITLE")
    print(f"  {'-'*38}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}  -----")
    for k in sorted(configs):
        c = configs[k]
        rm = c.reference_metrics
        tier = source_of(k) or "?"
        tps = f"{rm.long_gen_sustained_tps:.1f}" if rm else "—"
        tool = rm.tool_call_score if rm else "—"
        cv = f"{rm.stability_cv_pct:.2f}" if rm else "—"
        print(f"  {k:<38}  {tier:<10}  {tps:>7}  {tool:<7}  {cv:>6}  {c.title}")
    print(
        "\n  Use:  genesis model-config validate <key>     # schema + audit"
        "\n        genesis model-config preflight <key>    # env check"
        "\n        genesis model-config launch <key>       # boot"
        "\n        genesis model-config diagnose <key>     # runtime check"
        "\n        genesis model-config verify <key>       # bench vs reference"
    )
    return 0


def cmd_show(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"# Source tier: {source_of(args.key)}")
    print(dump_yaml(cfg))
    return 0


def cmd_render(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(cfg.to_launch_script())
    return 0


def cmd_save(args) -> int:
    cfg = _cfg_or_die(args.key)
    out = Path(args.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cfg.to_launch_script())
    out.chmod(0o755)
    print(f"Wrote launch script to {out}")
    return 0


def _print_section(title: str, items: list, name_fn, msg_fn, sev_fn,
                   passed_fn) -> tuple[int, int, int]:
    """Helper — print a section, return (errors, warnings, infos)."""
    e = w = i = 0
    if not items:
        print(f"  ✓ no {title.lower()} issues")
        return 0, 0, 0
    for item in items:
        sev = sev_fn(item)
        passed = passed_fn(item)
        if sev == "error" and not passed:
            mark, e_ = "✗ ERROR  ", 1
            e += 1
        elif sev == "warning" and not passed:
            mark, e_ = "⚠ WARN   ", 1
            w += 1
        else:
            mark, e_ = "✓ ok     ", 0
            i += 1
        print(f"  {mark}{name_fn(item):<35}  {msg_fn(item)}")
    return e, w, i


def cmd_audit(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== audit {args.key} ===\n")
    issues = audit(cfg)
    e = w = 0
    if not issues:
        print("  ✓ all 16 rules pass")
        return 0
    for rid, sev, title, msg in issues:
        if sev == "error":
            mark = "✗ ERROR  "; e += 1
        elif sev == "warning":
            mark = "⚠ WARN   "; w += 1
        else:
            mark = "ℹ INFO   "
        print(f"  {mark}[{rid}] {title}")
        for line in msg.splitlines():
            print(f"             {line}")
    print()
    print(f"  Summary: {e} errors, {w} warnings, "
          f"{len([i for i in issues if i[1] == 'info'])} info")
    return 1 if e > 0 else 0


def cmd_validate(args) -> int:
    """Combined: schema check + audit_rules. Exit 1 on any error severity."""
    cfg = _cfg_or_die(args.key)
    print(f"=== validate {args.key} ===\n")

    # Schema validation already happened on load. Re-run for explicit confirm.
    print("[1/2] schema check")
    try:
        cfg.validate()
        print("  ✓ schema OK")
    except Exception as ex:
        print(f"  ✗ schema FAIL: {ex}")
        return 1

    print("\n[2/2] audit_rules (cross-patch consistency, env checks)")
    rc = cmd_audit(args)
    return rc


def cmd_preflight(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== preflight {args.key} ===\n")
    checks = preflight_all(cfg)
    if not checks:
        print("  (no preflight checks applicable)")
        return 0
    e = w = 0
    for c in checks:
        if c.severity == "error" and not c.passed:
            print(f"  ✗ ERROR  {c.name:<35}  {c.message}"); e += 1
        elif c.severity == "warning" and not c.passed:
            print(f"  ⚠ WARN   {c.name:<35}  {c.message}"); w += 1
        else:
            print(f"  ✓ ok     {c.name:<35}  {c.message}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_diagnose(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== diagnose {args.key} (runtime) ===\n")
    findings = diagnose_all(cfg, port=args.port)
    e = w = 0
    for f in findings:
        if f.severity == "error" and not f.passed:
            print(f"  ✗ ERROR  {f.name:<35}  {f.message}"); e += 1
        elif f.severity == "warning" and not f.passed:
            print(f"  ⚠ WARN   {f.name:<35}  {f.message}"); w += 1
        else:
            print(f"  ✓ ok     {f.name:<35}  {f.message}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_verify(args) -> int:
    cfg = _cfg_or_die(args.key)
    print(f"=== verify {args.key} (bench vs reference) ===\n")
    if cfg.reference_metrics is None:
        print("  ✗ ERROR  no reference_metrics — run `bench-and-update` first")
        return 1

    rm = cfg.reference_metrics
    print(f"Reference: {rm.long_gen_sustained_tps:.1f} TPS / "
          f"{rm.tool_call_score} tool / "
          f"CV {rm.stability_cv_pct:.2f}% / "
          f"VRAM {rm.vram_total_mib} MiB")
    print(f"Bench'd:   {rm.measured_at} on {rm.vllm_pin}")
    print()
    results = verify(cfg, port=args.port)
    e = w = 0
    for r in results:
        if r.severity == "error" and not r.passed:
            mark = "✗ ERROR  "; e += 1
        elif r.severity == "warning" and not r.passed:
            mark = "⚠ WARN   "; w += 1
        else:
            mark = "✓ ok     "
        print(f"  {mark}{r.metric:<20}  expected={r.expected:<15} "
              f"actual={r.actual:<15} {r.delta}")
    print(f"\n  Summary: {e} blockers, {w} warnings")
    return 1 if e > 0 else 0


def cmd_where(args) -> int:
    src = source_of(args.key)
    if src is None:
        print(f"ERROR: config '{args.key}' not found", file=sys.stderr)
        return 1
    cfg = get(args.key)
    print(f"{args.key}:")
    print(f"  tier:           {src}")
    print(f"  title:          {cfg.title}")
    print(f"  schema_version: {cfg.schema_version}")
    print(f"  maintainer:     {cfg.maintainer}")
    if cfg.last_validated:
        print(f"  last_validated: {cfg.last_validated}")
    if cfg.genesis_pin:
        print(f"  genesis_pin:    {cfg.genesis_pin}")
    if cfg.vllm_pin_required:
        print(f"  vllm_pin:       {cfg.vllm_pin_required}")
    return 0


def cmd_launch(args) -> int:
    cfg = _cfg_or_die(args.key)
    script = cfg.to_launch_script()
    if args.dry_run:
        print("# DRY RUN — would execute:")
        print(script)
        return 0
    if not args.skip_preflight:
        print(f"=== preflight {args.key} ===")
        checks = preflight_all(cfg)
        for c in checks:
            if c.severity == "error" and not c.passed:
                print(f"  ✗ ERROR: {c.name} — {c.message}")
        if preflight_blockers(checks):
            print("\nERROR: preflight has blockers. Use --skip-preflight to "
                  "override (not recommended).", file=sys.stderr)
            return 1
        print("  ✓ preflight clean\n")
    import subprocess
    print(f"=== launching {args.key} ===")
    proc = subprocess.run(["bash", "-c", script], check=False)
    return proc.returncode


def cmd_new(args) -> int:
    if args.template:
        src = get(args.template)
        if src is None:
            print(f"ERROR: template '{args.template}' not found",
                  file=sys.stderr)
            return 1
        from copy import deepcopy
        new_cfg = deepcopy(src)
        new_cfg.key = args.key
        new_cfg.title = f"{src.title} (copy: {args.key})"
        new_cfg.maintainer = "<your-username>"
        new_cfg.last_validated = None
        new_cfg.reference_metrics = None
        new_cfg.verified_on = []
        new_cfg.lifecycle = "experimental"
        from vllm._genesis.model_configs.registry import _user_dir
        out_dir = _user_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.key}.yaml"
        if out_path.exists() and not args.force:
            print(f"ERROR: {out_path} exists. Use --force to overwrite.",
                  file=sys.stderr)
            return 1
        out_path.write_text(dump_yaml(new_cfg))
        print(f"✓ Created {out_path}")
        print(f"  Edit it, then `genesis model-config launch {args.key}`.")
        print(f"  After bench, `genesis model-config bench-and-update "
              f"{args.key}` to capture metrics.")
        return 0
    elif args.from_running:
        print("ERROR: --from-running requires Layer 4 docker-inspect-based "
              "captor — not implemented yet. Use --template for now.",
              file=sys.stderr)
        return 1
    else:
        print("ERROR: --template OR --from-running required", file=sys.stderr)
        return 1


def cmd_bench_and_update(args) -> int:
    """SCAFFOLD — run bench against running config, write metrics into YAML."""
    cfg = _cfg_or_die(args.key)
    print(f"=== bench-and-update {args.key} ===")
    p = args.port or (cfg.docker.port if cfg.docker else 8000)
    results = verify(cfg, port=p)
    print("Bench results would be written back into YAML. SCAFFOLD only "
          "for now — copy values manually.")
    for r in results:
        print(f"  {r.metric}: actual={r.actual}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genesis model-config",
        description="Manage vetted model launch configurations",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="enumerate all configs").set_defaults(
        func=cmd_list)

    p_show = sub.add_parser("show", help="print full YAML")
    p_show.add_argument("key")
    p_show.set_defaults(func=cmd_show)

    p_render = sub.add_parser("render", help="emit launch script")
    p_render.add_argument("key")
    p_render.set_defaults(func=cmd_render)

    p_save = sub.add_parser("save", help="write launch script to file")
    p_save.add_argument("key")
    p_save.add_argument("path")
    p_save.set_defaults(func=cmd_save)

    p_audit = sub.add_parser("audit",
                             help="run 16 audit_rules (cross-patch checks)")
    p_audit.add_argument("key")
    p_audit.set_defaults(func=cmd_audit)

    p_validate = sub.add_parser("validate",
                                help="schema + audit (recommended pre-launch)")
    p_validate.add_argument("key")
    p_validate.set_defaults(func=cmd_validate)

    p_pre = sub.add_parser("preflight",
                           help="pre-launch environment checks")
    p_pre.add_argument("key")
    p_pre.set_defaults(func=cmd_preflight)

    p_diag = sub.add_parser("diagnose",
                            help="runtime diagnose — query running container")
    p_diag.add_argument("key")
    p_diag.add_argument("--port", type=int, default=None)
    p_diag.set_defaults(func=cmd_diagnose)

    p_ver = sub.add_parser("verify",
                           help="bench vs reference_metrics (CI gate)")
    p_ver.add_argument("key")
    p_ver.add_argument("--port", type=int, default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_where = sub.add_parser("where", help="show source tier")
    p_where.add_argument("key")
    p_where.set_defaults(func=cmd_where)

    p_new = sub.add_parser("new", help="create a user config")
    p_new.add_argument("key")
    p_new.add_argument("--template")
    p_new.add_argument("--from-running")
    p_new.add_argument("--force", action="store_true")
    p_new.set_defaults(func=cmd_new)

    p_lau = sub.add_parser("launch", help="execute the rendered script")
    p_lau.add_argument("key")
    p_lau.add_argument("--dry-run", action="store_true")
    p_lau.add_argument("--skip-preflight", action="store_true")
    p_lau.set_defaults(func=cmd_launch)

    p_bau = sub.add_parser("bench-and-update",
                           help="bench + write metrics back into YAML")
    p_bau.add_argument("key")
    p_bau.add_argument("--port", type=int, default=None)
    p_bau.set_defaults(func=cmd_bench_and_update)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
