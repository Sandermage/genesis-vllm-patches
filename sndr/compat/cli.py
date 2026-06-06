# SPDX-License-Identifier: Apache-2.0
"""Genesis unified CLI dispatcher.

One entry-point that routes to all 14 sub-CLIs:

    python3 -m vllm.sndr_core.compat.cli doctor
    python3 -m vllm.sndr_core.compat.cli explain PN14
    python3 -m vllm.sndr_core.compat.cli categories --category spec_decode
    python3 -m vllm.sndr_core.compat.cli recipe save my-prod \\
        --from-container vllm-server-mtp-test
    python3 -m vllm.sndr_core.compat.cli plugins list
    python3 -m vllm.sndr_core.compat.cli telemetry status
    python3 -m vllm.sndr_core.compat.cli update-channel check
    ...

Each subcommand is a thin pass-through to the corresponding
`compat/<module>.py::main()`. The per-module CLIs continue to work
unchanged for backwards compatibility — operators with existing
scripts that call the long form keep working.

Subcommand names use **dashes** externally (closer to typical CLI
convention) and are mapped to underscore module names internally.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib
import logging
import sys

log = logging.getLogger("genesis.compat.cli")


# ─── Subcommand → module + function mapping ─────────────────────────────


# Order here is the order shown in --help. Group by purpose.
_SUBCOMMAND_MAP: dict[str, str] = {
    # Diagnostic
    "doctor":          "vllm.sndr_core.compat.doctor",
    "explain":         "vllm.sndr_core.compat.explain",
    "init":            "vllm.sndr_core.compat.init_wizard",
    # Models
    "list-models":     "vllm.sndr_core.compat.models.list_cli",
    "pull":            "vllm.sndr_core.compat.models.pull",
    # Registry validation
    "lifecycle-audit": "vllm.sndr_core.compat.lifecycle_audit_cli",
    "validate-schema": "vllm.sndr_core.compat.schema_validator",
    # Navigation + migration
    "categories":      "vllm.sndr_core.compat.categories",
    "migrate":         "vllm.sndr_core.compat.migrate",
    # Deploy / share
    "recipe":          "vllm.sndr_core.compat.recipes",
    "preset":          "vllm.sndr_core.compat.presets",
    # Community + telemetry + updates
    "plugins":         "vllm.sndr_core.compat.plugins",
    "telemetry":       "vllm.sndr_core.compat.telemetry",
    "update-channel":  "vllm.sndr_core.compat.update_channel",
    # Operator sanity check
    "self-test":       "vllm.sndr_core.compat.self_test",
    "verify":          "vllm.sndr_core.compat.verify",
    "preflight":       "vllm.sndr_core.compat.preflight_checks",
    # Benchmarking
    "bench":           "vllm.sndr_core.compat.bench",
    # Vetted model configs (single-source-of-truth for launch+verify)
    "model-config":    "vllm.sndr_core.compat.model_config_cli",
}


# Public alias for tests + introspection
KNOWN_SUBCOMMANDS = frozenset(_SUBCOMMAND_MAP.keys())


# Short descriptions for the dispatcher's --help banner
_DESCRIPTIONS: dict[str, str] = {
    "doctor":           "system diagnostic — hw + sw + model + patches",
    "explain":          "per-patch deep-dive (id, applies_to, decision)",
    "init":             "interactive first-run wizard",
    "list-models":      "browse curated model registry",
    "pull":             "HF download + tailored launch script",
    "lifecycle-audit":  "PATCH_REGISTRY lifecycle states (CI exit 1 on unknown)",
    "validate-schema":  "schema-validate PATCH_REGISTRY shape",
    "categories":       "browse patches by category",
    "migrate":          "pin-bump runbook against an upstream-vllm clone",
    "recipe":           "save / load / share launch configurations",
    "preset":           "curated launch bundles per (gpu × workload)",
    "plugins":          "community plugin entry-points (opt-in)",
    "telemetry":        "opt-in anonymized stats reporting",
    "update-channel":   "apt-style stable/beta/dev update channel",
    "self-test":        "structural sanity check (post-pull / pin bump)",
    "verify":           "post-install smoke test (--quick / --boot / --full)",
    "preflight":        "preflight checks: PN60 quant validator + club#34/#43 rules",
    "bench":            "Genesis benchmark suite (decode TPOT, wall TPS, stress)",
    "model-config":     "vetted model launch configs — list/show/render/launch/verify",
}


# ─── Dispatch ───────────────────────────────────────────────────────────


def _run_subcommand(name: str, argv: list[str]) -> int:
    """Resolve and call the target sub-CLI's main(argv)."""
    mod_path = _SUBCOMMAND_MAP.get(name)
    if mod_path is None:
        return 2

    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        log.error("[genesis] failed to import %s for subcommand %r: %s",
                  mod_path, name, e)
        return 3

    fn = getattr(mod, "main", None)
    if not callable(fn):
        log.error("[genesis] %s has no main() callable", mod_path)
        return 3

    try:
        rc = fn(argv)
    except SystemExit as e:
        # argparse-driven sub-CLIs typically exit with SystemExit;
        # propagate the code cleanly
        code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        return code

    if rc is None:
        return 0
    if isinstance(rc, int):
        return rc
    return 0


def _print_help() -> None:
    print("genesis — unified CLI for the Genesis vLLM compat layer")
    print()
    print("Usage:")
    print("  python3 -m vllm.sndr_core.compat.cli <subcommand> [args...]")
    print()
    print("Available subcommands:")
    width = max(len(s) for s in _SUBCOMMAND_MAP)
    for sub in _SUBCOMMAND_MAP:
        desc = _DESCRIPTIONS.get(sub, "")
        print(f"  {sub:<{width}}  {desc}")
    print()
    print("For per-subcommand help:")
    print("  python3 -m vllm.sndr_core.compat.cli <subcommand> --help")
    print()
    print("The legacy per-module form continues to work, e.g.:")
    print("  python3 -m vllm.sndr_core.compat.doctor")
    print("  python3 -m vllm.sndr_core.compat.recipe show my-prod")
    print("  python3 -m vllm.sndr_core.compat.update_channel check")


# ─── main() ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        # Empty args: return 0 (help is fine), else explicit help also 0
        return 0

    sub = argv[0]
    rest = argv[1:]

    if sub not in _SUBCOMMAND_MAP:
        # Friendly error
        print(f"genesis: unknown subcommand {sub!r}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"Available subcommands: {', '.join(sorted(_SUBCOMMAND_MAP))}",
              file=sys.stderr)
        print(file=sys.stderr)
        print("Run `python3 -m vllm.sndr_core.compat.cli --help` for full help.",
              file=sys.stderr)
        return 2

    return _run_subcommand(sub, rest)


if __name__ == "__main__":
    sys.exit(main())
