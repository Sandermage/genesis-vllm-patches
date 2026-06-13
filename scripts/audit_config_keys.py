#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Canonical env-key registry audit (Consolidated Roadmap §10.3 item 4 / §6.7).

Walks every committed V1 monolithic + V2 layered YAML and runs
`sndr config-keys-validate` on each. The validator computes the canonical
union of every Genesis/SNDR env key the codebase knows about
(PATCH_REGISTRY entries + V2 model.patches blocks + V1 genesis_env
blocks) and checks that every key in the YAML is in that union.

A typo or undocumented patch produces an "unknown key" hit, which
this audit promotes to a release gate.

Loader-key pass (PN379 mirror, vllm#45196, 2026-06-11): the same
fail-fast rules the PN379 runtime patch enforces at LoadConfig /
DefaultModelLoader construction are applied STATICALLY to loader-
related keys found in the YAMLs (``safetensors_load_strategy``,
``enable_multithread_load``, ``num_threads``,
``model_loader_extra_config``) — a misconfigured multithread-load
experiment (``enable_multithread_load: true, num_threads: 8`` is the
intended server-stage shape) fails here at audit time instead of at
boot on the rig. Regex line-scan by design (no pyyaml dependency,
consistent with the repo's other YAML audits).

Scope:

  • `vllm/sndr_core/model_configs/builtin/*.yaml`           (V1 monolithic)
  • `vllm/sndr_core/model_configs/builtin/model/*.yaml`     (V2 ModelDef)
  • `vllm/sndr_core/model_configs/builtin/hardware/*.yaml`  (V2 HardwareDef)
  • `vllm/sndr_core/model_configs/builtin/profile/*.yaml`   (V2 ProfileDef)

Preset alias triplets (`builtin/presets/*.yaml`) don't carry env keys
of their own — they're pure pointers. They are skipped.

Exit codes:
  0 — every YAML's keys are in the canonical registry AND every
      loader key satisfies the PN379 rules.
  1 — at least one YAML has an unknown env key or a loader-key
      violation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import the validator directly so we don't pay subprocess startup per
# YAML (would be 40+ python launches otherwise).
sys.path.insert(0, str(REPO_ROOT))
from sndr.cli.legacy import config_keys as _ck  # type: ignore  # noqa: E402  — import after sys.path mutation


SCAN_DIRS = (
    "sndr/model_configs/builtin",
    "sndr/model_configs/builtin/model",
    "sndr/model_configs/builtin/hardware",
    "sndr/model_configs/builtin/profile",
)

SKIP_DIRS = ("presets",)


def _gather_yamls() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for s in SCAN_DIRS:
        p = REPO_ROOT / s
        if not p.is_dir():
            continue
        for f in p.glob("*.yaml"):
            if any(skip in f.parts for skip in SKIP_DIRS):
                continue
            if f in seen:
                continue
            seen.add(f)
            out.append(f)
    return sorted(out)


# ─── PN379 loader-key rules (vllm#45196 mirror) ────────────────────────
#
# Mirrors sndr/engines/vllm/patches/loader/pn379_load_config_fail_fast.py:
# the runtime patch rejects these shapes at engine construction; this
# static pass rejects them at audit time, pre-deploy.

_LOADER_VALID_STRATEGIES = {"lazy", "eager", "prefetch", "torchao"}
_LOADER_NULLS = {"", "null", "~", "none"}
_LOADER_KEY_RE = re.compile(
    r"^\s*(safetensors_load_strategy|enable_multithread_load"
    r"|num_threads|model_loader_extra_config)\s*:\s*(.*)$"
)


def _strip_yaml_scalar(raw: str) -> tuple[str, bool]:
    """Return (scalar, was_quoted) with trailing comments stripped."""
    val = raw.strip()
    # Strip a trailing comment (regex line-scan; loader values are
    # simple scalars so a bare ` #` reliably opens a comment here).
    val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
        return val[1:-1], True
    return val, False


def audit_loader_keys(text: str) -> list[str]:
    """Validate loader-related keys in one YAML text (PN379 rules).

    Returns a list of human-readable violation strings (empty = clean).
    """
    findings: list[str] = []
    strategy: str | None = None
    multithread_true = False
    for lineno, line in enumerate(text.split("\n"), start=1):
        if line.lstrip().startswith("#"):
            continue
        m = _LOADER_KEY_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        val, quoted = _strip_yaml_scalar(raw)
        if key == "safetensors_load_strategy":
            if val.lower() in _LOADER_NULLS:
                strategy = None
            elif val in _LOADER_VALID_STRATEGIES:
                strategy = val
            else:
                findings.append(
                    f"line {lineno}: safetensors_load_strategy={val!r} is not "
                    f"one of {sorted(_LOADER_VALID_STRATEGIES)} (PN379: a typo "
                    f"silently fell back to lazy pre-patch)"
                )
        elif key == "enable_multithread_load":
            if quoted or val not in ("true", "false"):
                findings.append(
                    f"line {lineno}: enable_multithread_load={val!r} must be a "
                    f"bare YAML bool (true|false) — PN379 rejects non-bool at "
                    f"engine construction"
                )
            elif val == "true":
                multithread_true = True
        elif key == "num_threads":
            if quoted or not val.isdigit() or int(val) <= 0:
                findings.append(
                    f"line {lineno}: num_threads={val!r} must be a bare "
                    f"positive integer — PN379 rejects it at engine "
                    f"construction (pre-patch this died deep inside "
                    f"ThreadPoolExecutor)"
                )
        elif key == "model_loader_extra_config":
            # Block mapping (empty value) or flow mapping ({...}) are
            # fine; any other inline scalar is a non-dict.
            if val not in _LOADER_NULLS and not val.startswith("{"):
                findings.append(
                    f"line {lineno}: model_loader_extra_config={val!r} must be "
                    f"a mapping — PN379 rejects non-dict at engine construction"
                )
    if multithread_true and strategy is not None and strategy != "lazy":
        findings.append(
            f"enable_multithread_load: true does not support "
            f"safetensors_load_strategy={strategy!r} — the multi-thread "
            f"loader only implements the default lazy strategy (PN379)"
        )
    return findings


def audit() -> dict:
    canon = _ck.load_canonical_registry()
    canonical_keys = set(canon.keys()) if hasattr(canon, "keys") else set(canon)
    results: list[dict] = []
    total_unknown = 0
    total_loader_violations = 0
    for fp in _gather_yamls():
        try:
            keys = _ck._extract_keys_from_yaml(fp)
        except RuntimeError:
            keys = []
        unknown = sorted({
            k for k in keys
            if (k.startswith("GENESIS_") or k.startswith("SNDR_"))
            and k not in canonical_keys
        })
        rel = fp.relative_to(REPO_ROOT).as_posix()
        try:
            loader_violations = audit_loader_keys(
                fp.read_text(encoding="utf-8")
            )
        except OSError:
            loader_violations = []
        results.append({
            "yaml": rel,
            "unknown_keys": unknown,
            "count": len(unknown),
            "loader_violations": loader_violations,
        })
        total_unknown += len(unknown)
        total_loader_violations += len(loader_violations)
    return {
        "canonical_count": len(canonical_keys),
        "yaml_count": len(results),
        "total_unknown": total_unknown,
        "total_loader_violations": total_loader_violations,
        "per_yaml": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "audit-config-keys: "
            f"{report['yaml_count']} YAMLs scanned against "
            f"{report['canonical_count']} canonical keys"
        )
        print("─" * 70)
        bad = [r for r in report["per_yaml"] if r["count"] > 0]
        if bad:
            for r in bad[:10]:
                print(f"  ✗ {r['yaml']}: {r['count']} unknown key(s)")
                for k in r["unknown_keys"][:5]:
                    print(f"      · {k}")
                if r["count"] > 5:
                    print(f"      ... ({r['count'] - 5} more)")
            if len(bad) > 10:
                print(f"  ... ({len(bad) - 10} more YAMLs with unknown keys)")
            print()
            print(f"  FAIL — {report['total_unknown']} unknown key(s)")
        else:
            print("  ✓ every YAML's Genesis/SNDR keys are canonical")
        loader_bad = [
            r for r in report["per_yaml"] if r["loader_violations"]
        ]
        if loader_bad:
            for r in loader_bad:
                print(f"  ✗ {r['yaml']}: loader-key violations (PN379 rules)")
                for v in r["loader_violations"]:
                    print(f"      · {v}")
            print()
            print(
                f"  FAIL — {report['total_loader_violations']} "
                f"loader-key violation(s)"
            )
        else:
            print("  ✓ loader keys satisfy the PN379 fail-fast rules")
        if not bad and not loader_bad:
            print()
            print("  OK — env-key drift gate clean")
    return (
        0
        if report["total_unknown"] == 0
        and report["total_loader_violations"] == 0
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
