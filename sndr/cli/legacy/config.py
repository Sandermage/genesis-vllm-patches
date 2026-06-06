# SPDX-License-Identifier: Apache-2.0
"""C8 (UNIFIED_CONFIG plan 2026-05-09) — `sndr config` registry browser.

Native subcommands focused on what's MISSING from the bridged
`sndr model-config` (list/show/audit/validate already exist there):

  sndr config diff <a> <b> [--field PATH]
      Field-by-field diff between two preset configs. Useful for
      operators comparing PROD vs an alternative (e.g. 35B-PROD vs
      27B-PROD) before deciding which to deploy.

  sndr config explain <key>
      Plain-English walkthrough of what the preset declares —
      hardware requirements, model + KV dtype, spec-decode setup,
      enabled patches, schema blocks (Y3 artifacts / Y11 upstream /
      Y10 service / etc).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from . import _io


__all__ = [
    "add_argparser",
    "run_diff",
    "run_explain",
    "run_new",
    "run_list",
    "run_checksum",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "config",
        help="Native preset config browser — diff + explain + new (UNIFIED_CONFIG C8/C7).",
        description=(
            "Native config inspection + scaffold generator. For list/show/audit/"
            "validate use bridged `sndr model-config`. This native command "
            "adds diff (field-by-field comparison), explain (operator-"
            "friendly preset walkthrough), and `new --from-detect` "
            "(detect host + scaffold a starter YAML)."
        ),
    )
    sub = p.add_subparsers(dest="config_cmd", required=True)

    p_diff = sub.add_parser("diff",
                              help="Field-by-field diff between two presets.")
    p_diff.add_argument("a", help="left preset key")
    p_diff.add_argument("b", help="right preset key")
    p_diff.add_argument("--field", default=None,
                         help="Limit diff to a top-level field (e.g. genesis_env).")
    p_diff.add_argument("--json", action="store_true",
                         help="Emit machine-readable JSON.")
    p_diff.set_defaults(func=run_diff)

    p_explain = sub.add_parser("explain",
                                 help="Plain-English walkthrough of a preset.")
    p_explain.add_argument("config_key", help="preset key (e.g. a5000-2x-35b-prod)")
    p_explain.add_argument("--json", action="store_true",
                            help="Emit JSON instead of prose.")
    p_explain.set_defaults(func=run_explain)

    # C7 (UNIFIED_CONFIG plan 2026-05-09): --from-detect wizard
    p_new = sub.add_parser(
        "new",
        help="Generate a starter YAML from detected host (C7).",
        description=(
            "Probes host (deps.inspect_host) + nearest builtin preset, "
            "writes a starter YAML to the model-config registry user dir "
            "(default: ~/.sndr/model_configs/<key>.yaml). "
            "--from-detect: auto-fill hardware + system_env from probe. "
            "--from-template <key>: copy a builtin as starter."
        ),
    )
    p_new.add_argument("key", help="new config key (kebab-case)")
    p_new.add_argument("--from-detect", action="store_true",
                         help="Probe host + autofill hardware section.")
    p_new.add_argument("--from-template", default=None,
                         help="Copy from existing preset key as starter.")
    p_new.add_argument("--out", default=None,
                         help="Output path. Default: user model_config dir/<key>.yaml")
    p_new.add_argument("--force", action="store_true",
                         help="Overwrite if output exists.")
    p_new.set_defaults(func=run_new)

    # Deterministic content-checksum for community submissions.
    # Hashes the canonical YAML form (sorted keys, normalized whitespace)
    # so submitters can freeze a known-good config and reviewers can
    # verify the file hasn't drifted in transit.
    p_checksum = sub.add_parser(
        "checksum",
        help="Compute a deterministic SHA256 of the preset YAML.",
        description=(
            "Hashes the canonical YAML form (sorted keys, normalized "
            "whitespace) so community submissions stay verifiable. The "
            "checksum belongs in `community_credit` or a sidecar "
            ".checksum file shipped with the PR."
        ),
    )
    p_checksum.add_argument("config_key", help="preset key")
    p_checksum.add_argument("--verify", default=None,
                              help="Compare against the given hex digest.")
    p_checksum.set_defaults(func=run_checksum)

    # Convenience alias to `sndr model-config list` so list/show/audit/
    # validate sit next to diff/explain/new. Bridged `model-config`
    # keeps the full implementation.
    p_list = sub.add_parser(
        "list",
        help="List available model configs (alias of `sndr model-config list`).",
        description=(
            "Convenience alias to `sndr model-config list`. Shows all "
            "presets registered via the model-config registry (builtin "
            "+ user overlay). For show/audit/validate, use the bridged "
            "`sndr model-config` subcommands directly."
        ),
    )
    p_list.add_argument("--json", action="store_true",
                         help="Emit JSON instead of the table.")
    p_list.set_defaults(func=run_list)


def _resolve(key: str):
    from vllm.sndr_core.model_configs.registry import get
    cfg = get(key)
    if cfg is None:
        _io.warn(f"unknown preset key {key!r}")
        try:
            from vllm.sndr_core.model_configs.registry import list_keys
            _io.info(f"available: {', '.join(sorted(list_keys()))}")
        except Exception:
            pass
    return cfg


def _to_dict(cfg) -> dict:
    """Best-effort recursive dataclass→dict for diff comparison."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return {}


def _diff_dicts(a: dict, b: dict, path: str = "") -> list[dict]:
    """Recursive dict-diff. Returns list of {path, side, value} entries."""
    diffs: list[dict] = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        sub_path = f"{path}.{k}" if path else k
        va = a.get(k, "<missing>")
        vb = b.get(k, "<missing>")
        if va == vb:
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_dicts(va, vb, sub_path))
        elif isinstance(va, list) and isinstance(vb, list):
            if va != vb:
                diffs.append({"path": sub_path, "left": va, "right": vb})
        else:
            diffs.append({"path": sub_path, "left": va, "right": vb})
    return diffs


# ─── diff

def run_diff(args: argparse.Namespace) -> int:
    cfg_a = _resolve(args.a)
    cfg_b = _resolve(args.b)
    if cfg_a is None or cfg_b is None:
        return 2

    da = _to_dict(cfg_a)
    db = _to_dict(cfg_b)
    if args.field:
        da = {args.field: da.get(args.field)}
        db = {args.field: db.get(args.field)}
    diffs = _diff_dicts(da, db)

    if args.json:
        print(json.dumps({
            "left": args.a, "right": args.b,
            "field_filter": args.field,
            "n_diffs": len(diffs),
            "diffs": diffs,
        }, indent=2, sort_keys=True, default=str))
        return 0

    print(f"sndr config diff {args.a!r} vs {args.b!r}")
    if args.field:
        print(f"  (filtered to field: {args.field})")
    print("─" * 60)
    if not diffs:
        print("  (no differences)")
        return 0
    for d in diffs:
        print(f"  {d['path']}:")
        print(f"    left  ({args.a}):  {d['left']!r}")
        print(f"    right ({args.b}):  {d['right']!r}")
    print()
    print(f"  Total diffs: {len(diffs)}")
    return 0


# ─── explain

def _format_explain(cfg) -> list[str]:
    """Build the human-readable preset walkthrough lines."""
    lines = [
        f"sndr config explain '{cfg.key}'",
        "─" * 60,
        f"  Title:        {cfg.title}",
        f"  Description:  {cfg.description.strip()[:200]}",
        f"  Maintainer:   {cfg.maintainer}",
        f"  Lifecycle:    {cfg.lifecycle}",
        f"  Validated:    {cfg.last_validated or '_unset_'}",
        "",
        "  Hardware:",
        f"    GPU keys:   {cfg.hardware.gpu_match_keys}",
        f"    n_gpus:     {cfg.hardware.n_gpus}",
        f"    min VRAM:   {cfg.hardware.min_vram_per_gpu_mib} MiB/GPU",
        "",
        "  Model + serve:",
        f"    Path:           {cfg.model_path}",
        f"    Served as:      {cfg.served_model_name}",
        f"    KV dtype:       {cfg.kv_cache_dtype or 'default'}",
        f"    max_model_len:  {cfg.max_model_len}",
        f"    max_num_seqs:   {cfg.max_num_seqs}",
        f"    GPU mem util:   {cfg.gpu_memory_utilization}",
    ]
    if cfg.spec_decode:
        lines.extend([
            "",
            "  Spec-decode:",
            f"    method:                  {cfg.spec_decode.method}",
            f"    num_speculative_tokens:  {cfg.spec_decode.num_speculative_tokens}",
        ])

    enabled = sorted(
        k.replace("GENESIS_ENABLE_", "")
        for k, v in cfg.genesis_env.items()
        if k.startswith("GENESIS_ENABLE_") and v == "1"
    )
    lines.extend([
        "",
        f"  Enabled patches ({len(enabled)}):",
    ])
    # Group in lines of 4
    for i in range(0, len(enabled), 4):
        lines.append(f"    {' '.join(p.ljust(28) for p in enabled[i:i+4])}")

    # Schema blocks present
    blocks = []
    for attr, label in (
        ("upstream", "Y11 UpstreamPinPolicy"),
        ("overrides", "Y12 OverridesPolicy"),
        ("offload", "OffloadConfig (Path A)"),
        ("artifacts", "Y3 Artifacts"),
        ("service", "Y10 ServiceConfig"),
        ("package_versions", "Y1 PackageVersions"),
        ("package_sources", "Y2 PackageSources"),
        ("gpu_tuning", "Y8 GpuTuningConfig"),
        ("observability", "Y14 ObservabilityConfig"),
    ):
        v = getattr(cfg, attr, None)
        if v is not None:
            blocks.append(label)
    if cfg.cache_config and getattr(cfg.cache_config, "tiers", []):
        blocks.append(f"PathC tiers ({len(cfg.cache_config.tiers)})")
    if blocks:
        lines.extend([
            "",
            f"  Schema blocks declared ({len(blocks)}):",
            *(f"    • {b}" for b in blocks),
        ])

    if cfg.reference_metrics:
        rm = cfg.reference_metrics
        lines.extend([
            "",
            "  Reference metrics:",
            f"    measured_at: {rm.measured_at}",
            f"    wall_TPS:    {rm.long_gen_sustained_tps}",
            f"    tool_call:   {rm.tool_call_score}",
        ])
    if cfg.notes:
        lines.append("")
        lines.append("  Notes:")
        for n in cfg.notes:
            lines.append(f"    {n}")
    return lines


def run_explain(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config_key)
    if cfg is None:
        return 2

    if args.json:
        d = _to_dict(cfg)
        print(json.dumps(d, indent=2, sort_keys=True, default=str))
        return 0
    print("\n".join(_format_explain(cfg)))
    return 0


# ─── new (C7 wizard)

def _detect_hardware_block() -> dict:
    """Auto-fill the `hardware` section from `deps.inspect_host()`."""
    try:
        from vllm.sndr_core.deps import inspect_host
        inv = inspect_host()
        nv = inv.nvidia
        if not nv.installed or nv.n_gpus == 0:
            return {
                "gpu_match_keys": ["unknown"],
                "n_gpus": 0,
                "min_vram_per_gpu_mib": 0,
            }
        # Pick first GPU name as match key (lowercased)
        first_name = (nv.gpu_names[0] if nv.gpu_names else "unknown").lower()
        # Strip "nvidia " prefix
        first_name = first_name.replace("nvidia ", "").strip()
        # min VRAM = floor of detected VRAM (round down to GiB)
        min_vram = (
            int(nv.gpu_total_vram_mib[0]) if nv.gpu_total_vram_mib else 0
        )
        return {
            "gpu_match_keys": [first_name],
            "n_gpus": nv.n_gpus,
            "min_vram_per_gpu_mib": min_vram,
        }
    except Exception:
        return {
            "gpu_match_keys": ["unknown"],
            "n_gpus": 0,
            "min_vram_per_gpu_mib": 0,
        }


_NEW_TEMPLATE = """# SPDX-License-Identifier: Apache-2.0
# Generated by `sndr config new` ({mode}, 2026-05-09)
# Operator: edit fields below to match your model + workload, then validate:
#   sndr model-config validate {key}

key: {key}
title: {key} (operator-edited title)
description: >-
  Generated starter — replace this with a 1-2 sentence description.
schema_version: 1
maintainer: {user}
last_validated: '2026-05-09'
lifecycle: experimental

hardware:
{hw_yaml}

model_path: /models/REPLACE-WITH-YOUR-MODEL-DIR
served_model_name: REPLACE-WITH-A-SHORT-NAME
quantization: null
kv_cache_dtype: fp16

max_model_len: 32768
gpu_memory_utilization: 0.90
max_num_seqs: 2
max_num_batched_tokens: 4096
enable_chunked_prefill: true
dtype: float16

enable_auto_tool_choice: true
tool_call_parser: qwen3_coder
reasoning_parser: qwen3

genesis_env: {{}}
system_env:
  PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
  VLLM_NO_USAGE_STATS: '1'

api_key: genesis-local
host: 0.0.0.0

docker:
  image: vllm/vllm-openai:nightly
  container_name: vllm-{key}
  port: 8000
  shm_size: 8g
  memory_limit: 64g
  gpus: all
  mounts:
    - ${{models_dir}}:/models:ro
"""


def run_new(args: argparse.Namespace) -> int:
    """C7: scaffold a new YAML starter."""
    import getpass
    from pathlib import Path

    key = args.key
    if not key.replace("-", "").replace("_", "").isalnum():
        _io.error(f"key must be alphanumeric/hyphen/underscore (got {key!r})")
        return 2

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        from vllm.sndr_core.locations.project_paths import model_configs_user_dir
        out_path = model_configs_user_dir() / f"{key}.yaml"
    if out_path.exists() and not args.force:
        _io.error(f"output exists: {out_path} — pass --force to overwrite")
        return 2

    if args.from_template:
        # Copy from existing preset
        cfg = _resolve(args.from_template)
        if cfg is None:
            return 2
        from vllm.sndr_core.model_configs.schema import dump_yaml
        body = dump_yaml(cfg)
        # Replace key + title to mark as scratch
        body = body.replace(
            f"key: {args.from_template}",
            f"key: {key}", 1,
        )
        mode = f"copied from template '{args.from_template}'"
    else:
        # Detect mode (default if --from-detect or no template)
        if args.from_detect:
            hw = _detect_hardware_block()
            mode = "detected from host"
        else:
            hw = {
                "gpu_match_keys": ["replace-with-gpu-name"],
                "n_gpus": 1,
                "min_vram_per_gpu_mib": 24000,
            }
            mode = "blank scaffold"
        hw_yaml = "\n".join(
            f"  {k}: {v}" if not isinstance(v, list) else
            f"  {k}: {v}"
            for k, v in hw.items()
        )
        body = _NEW_TEMPLATE.format(
            key=key, mode=mode, user=getpass.getuser(),
            hw_yaml=hw_yaml,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)

    _io.success(f"wrote new config: {out_path}")
    print(f"  mode: {mode}")
    print("  edit, then validate with:")
    print(f"    sndr model-config validate {key}")
    return 0


def run_checksum(args: argparse.Namespace) -> int:
    """Compute / verify a deterministic SHA256 of the preset YAML.

    The hash is taken over a canonical form: comments stripped, YAML
    re-emitted with `sort_keys=True`, trailing whitespace normalized.
    This way two functionally-identical files (comment edits, key
    reordering) produce the same digest.
    """
    import hashlib
    from vllm.sndr_core.model_configs.registry import path_for

    key = args.config_key
    p = path_for(key)
    if p is None:
        _io.error(f"preset {key!r} not found in registry")
        return 2
    try:
        import yaml
    except ImportError:
        _io.error("PyYAML required for checksum (pip install pyyaml)")
        return 2
    raw = p.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    canonical = yaml.safe_dump(
        parsed, sort_keys=True, default_flow_style=False,
        allow_unicode=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if args.verify:
        if digest == args.verify.lower():
            _io.success(f"checksum OK: {key} = {digest}")
            return 0
        _io.error(
            f"checksum MISMATCH for {key}:\n"
            f"  expected: {args.verify}\n"
            f"  actual:   {digest}"
        )
        return 1
    print(f"{digest}  {p}")
    return 0


def run_list(args: argparse.Namespace) -> int:
    """Convenience alias forwarding to `sndr model-config list`.

    Operators expect `list/show/audit/validate` next to `diff/explain/new`.
    The bridged model-config CLI holds the full implementation; this
    function constructs an argparse Namespace matching its `cmd_list`
    signature and dispatches.
    """
    from vllm.sndr_core.compat.model_config_cli import cmd_list as _cmd_list

    bridged_ns = argparse.Namespace(
        json=getattr(args, "json", False),
        include_tested=False,
    )
    rc = _cmd_list(bridged_ns)
    if rc == 0 and not getattr(args, "json", False):
        print()
        print("  (alias of `sndr model-config list` — show/audit/validate "
              "available there)")
    return rc
