#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit for stale `vllm_version_range` upper bounds in PATCH_REGISTRY.

CLAUDE.md Class 5 known-bug surface: "Anchor drift after vllm pin bump."
This audit catches a softer variant — patches whose `applies_to.vllm_version_range`
upper bound EXCLUDES the current operational pin, causing:

  - `applies_to` constraint check fails on every boot
  - When env_flag is set (opt-in), the patch still applies but logs a
    WARNING about the mismatch — spurious noise in production boot logs
  - When env_flag is unset (default_on path), the patch silently skips
    via the strict-opt-in guard (which fires first), so the version
    range never gets checked — but the range field is still wrong

Behaviour
---------

The audit examines every PATCH_REGISTRY entry's
`applies_to.vllm_version_range` and reports cases where the upper
bound looks stale (would exclude the current operational pin).

Operational pin: read from `vllm.__version__` when available; else from
the `--pin` flag; else assumes `0.21.1rc1+` (current as of v11.3.0).

Severity classification:

  CRITICAL — patch is `default_on=True` + version range excludes current
             pin. Patch silently skips for every operator without
             opt-in override. (As of v11.3.0 audit: 0 entries.)

  WARN     — patch is opt-in (default_on=False) but enabled by some
             prod-* preset. Operator gets spurious WARN noise on boot.

  INFO     — patch is opt-in and not enabled by any preset. Only shows
             up if operator explicitly enables.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ audit (CLAUDE.md Class 5 surface).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# Default current-pin assumption when vllm not importable.
# Update on each pin bump (or pull dynamically when vllm is installed).
# 2026-06-13 (wave-2): bumped from the stale 0.21.1rc1.dev354 to the
# canonical pin so torch-less audit runs (CI / collection envs) evaluate
# ranges against the version actually deployed — the 0.21.1 default
# false-flagged every correct `>=0.22.0` range as stale.
DEFAULT_PIN = "0.23.1rc1.dev101+g4c6266331"  # pin-bump candidate 2026-06-17 (was dev259)


# v11.3.0 BUG #14 baseline allowlist — known patches with stale
# upper bounds whose verification on the current pin is queued for a
# bulk-update PR. Each entry documents the patch's status. `--strict`
# mode exits 1 only on CRITICAL entries NOT in this allowlist — new
# additions force review without blocking known-debt entries.
#
# Cleanup workflow when bumping a range:
#   1. Verify patch applies cleanly on the new pin (anchor + smoke).
#   2. Update `applies_to.vllm_version_range` in registry.py.
#   3. Remove the patch ID from this set + add a one-line "verified
#      on pin <X> via bench <Y>" note in commit message.
_BASELINE_CRITICAL_STALE: frozenset[str] = frozenset({
    # Default-on (always-skip without env override) — currently 0
    # entries; legacy `legacy`/`retired` lifecycles are filtered out
    # by `_audit` so do not appear here.
    #
    # v11.3.0 BUG #14 follow-through (commit pending): empirically
    # verified on rig 0.21.1rc1 via direct apply() probe — 17 of the
    # original 19 patches now bumped to `<0.22.0` (P70, P72, P82,
    # P103, P107, PN12, PN14, PN16, PN71, PN91, PN92, PN96, PN106,
    # PN125, PN201, SNDR_WORKSPACE_001, PN90). Each verified by:
    #   - apply() returns "applied" on writable mount, OR
    #   - apply() returns "skipped" with "read_only_mount" reason
    #     (PN14, PN96 — bind-mount issue, not patch breakage), OR
    #   - apply() returns "skipped" with self-detected drift marker
    #     (PN90 — upstream merged equivalent, intentional self-skip)
    #
    # v11.3.0 P0.1+P0.2 anchor rework: both originally-baseline patches
    # (P67, PN73) had their anchors refreshed to match upstream
    # 0.21.x refactors. Baseline now empty — any new CRITICAL entry
    # forces review.
    # P67 — fixed: anchor updated for `mm_prefix_range_tensor` +
    #       multi-line `if (` form introduced by upstream multimodal-
    #       prefix-range refactor.
    # PN73 — fixed: anchor updated for `function = item.get("function")`
    #        extracted-variable refactor in _postprocess_messages.
    #
    # 2026-06-13 (wave-2): DEFAULT_PIN bumped 0.21.1 → 0.22.1 (canonical).
    # That correctly surfaces two PRE-EXISTING 0.21-era entries whose
    # `<0.22.0` upper bound now excludes the deployed pin and which are
    # enabled in builtin YAMLs — pre-existing debt unrelated to the
    # wave-2 registry integration, queued for per-patch re-verification
    # on 0.22.1 before the range is bumped (audit workflow step 1):
    "PN90",   # probabilistic-draft MTP; ('>=0.20.2rc1.dev9', '<0.22.0').
              # Self-skips via drift marker on merged-equivalent upstream
              # (intentional), so the YAML "enable" is already a no-op by
              # design — verify on 0.22.1, then bump or retire.
    "PN125",  # warmup-orchestrator FULL_AND_PIECEWISE; ('>=0.20.0',
              # '<0.22.0'), Qwen3.5/Next arch-gated. Needs a 0.22.1 boot
              # probe before the upper bound moves to <0.23.0.
    #
    # 2026-06-14: DELIBERATE cross-pin gates (NOT debt-to-bump). The four
    # entries below are capped '<0.22.1rc1.dev491' ON PURPOSE by the
    # deep-audit #1 version-gate work. Each stays ENABLED in the shared
    # qwen3.6 / a5000-2x builtin YAMLs so a rollback to the dev259 image
    # (which still has the old code) keeps protection; the runtime
    # version-gate (GENESIS_ENFORCE_VERSION_RANGE=1) makes the per-pin
    # decision — APPLY on dev259, SKIP on dev491. They surface as CRITICAL
    # only under `--pin dev491` (the deployed pin), which is correct and
    # benign by design. Do NOT bump the bound to re-include dev491 — that
    # would re-introduce the exact corruption each one avoids. These became
    # visible only once the `--pin dev491` audit crash (_ver_key TypeError
    # on rc/dev bounds) was fixed in this commit; before that the dev491
    # audit could not run, so the version-gate work never allowlisted them.
    "P64",    # qwen3_coder MTP streaming wrap. dev491 #45171 remapped the
              # engine-native parser; the dev259-era wrap CORRUPTS it (leaks
              # tool-call XML to content) — must skip on dev491, apply dev259.
    "P61c",   # qwen3_coder deferred-commit streaming wrap — same #45171
              # parser remap; skip dev491, apply dev259 rollback.
    "PN56",   # qwen3_coder XML-fallback streaming wrap — same #45171 parser
              # remap; skip dev491, apply dev259 rollback.
    "PN347",  # MarlinFP8 N==K corruption fix. dev491 REFACTORED the buggy
              # `if w_q.shape != (...)` transpose guard out of
              # kernels/linear/scaled_mm/marlin.py (transpose moved to caller
              # via the explicit `size_k_first` contract) so the bug cannot
              # occur and the anchor is correctly absent; the dev259 image
              # still has the guard at marlin.py:87. vllm#44113 CLOSED-unmerged
              # (upstream solved it structurally). Skip dev491, apply dev259.
    # ── 0.23.1 reverify 2026-06-17 (Workflow + adversarial verify) ──────
    # Intentionally capped <0.23.0 on the 0.23.1 pin — each verified live.
    # (a) Upstream supersedes on 0.23.x (the fix shipped / the bug is gone):
    "PN30",   # DS conv spec-decode — upstream fused-postprocess kernel rewrote
              # get_conv_copy_spec (iron-rule #11a). Skip 0.23.1, apply <0.23.0.
    "PN133",  # MTP scheduler empty-output — pre-fix anchor gone on 0.23.1.
    "PN362",  # VLLM_TRITON_FORCE_FIRST_CONFIG — merged into 0.23.x.
    "PN370",  # async accepted-counts race — superseded by our 0.23.1-native PN398.
    "PN373",  # parallel_tool_calls null — vllm#44955 merged into 0.23.x.
    "PN378",  # recovered-token vocab mask — complete vllm#45060 ships in 0.23.1.
    "PN383",  # offload MTP/EAGLE gate — vllm#44784 merged 2026-06-16 (0.23.x).
    "PN51",   # qwen3 enable_thinking routing — merged into 0.23.x.
    "PN125",  # FULL_AND_PIECEWISE redundant (v1 default engages it on hybrid).
    "P29_HEAL",            # heals the qwen3coder parser DELETED by #45588.
    "G4_14",  # gemma4 pad-strip wraps Gemma4ToolParser, DELETED by #45588. New
              # Gemma4EngineToolParser is a skip_special_tokens=False rewrite —
              # #39392 raw-token pad-leak mode no longer exists. Skip 0.23.1,
              # apply <0.23.0 (iron-rule #11a). #39392 still OPEN: redesign vs
              # the new class with a failing repro test before lifting the cap.
    "SNDR_MTP_DYNAMIC_K_001",  # #26504 DynamicProposer — bench NOT_SIGNIFICANT.
    # (b) Bug still live on 0.23.1 but the anchor target was refactored away —
    # capped pending a redesign (cannot byte-exact re-anchor; see registry):
    "PN374",  # qwen3xml quoted-keys — bug fixed upstream; anchors gone.
})


def _resolve_current_pin(override: str | None = None) -> str:
    if override:
        return override
    try:
        import vllm
        ver = getattr(vllm, "__version__", None)
        if ver:
            return ver
    except Exception:
        pass
    return DEFAULT_PIN


def _parse_pep440(spec: str) -> tuple[str | None, str | None]:
    """Best-effort parse of a single PEP 440 specifier like `<0.21.0`
    or `>=0.20.2rc1.dev9`. Returns (operator, version) or (None, None)
    on parse fail."""
    spec = spec.strip()
    m = re.match(r"^(>=|<=|>|<|==|!=|~=)\s*(\S+)$", spec)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _excludes_pin(constraint: str, pin: str) -> bool:
    """Does this single PEP 440 specifier EXCLUDE the current pin?

    Returns True iff the constraint is well-formed AND it deterministically
    rejects pin. Conservative: returns False on parse fail.

    Uses ``packaging.version`` for PEP 440-correct ordering — the same parser
    the runtime version-gate relies on (``sndr/compat/version_check.py``). The
    prior hand-rolled ``_ver_key`` tokeniser built mixed int/str tuples that
    raised ``TypeError: '<' not supported between 'str' and 'int'`` whenever a
    range bound carried rc/dev components (e.g. ``<0.22.1rc1.dev491``), which
    crashed the audit under ``--pin dev491``. Version objects order
    final > rc > dev natively, so a ``.devN`` upper bound now compares cleanly.
    """
    op, ver = _parse_pep440(constraint)
    if op is None:
        return False
    try:
        from packaging.version import Version
        # Drop the vllm local `+gSHA` segment; keep pre/dev components.
        pin_v = Version(pin.split("+", 1)[0])
        ver_v = Version(ver.split("+", 1)[0])
    except Exception:
        return False
    if op == "<":
        return pin_v >= ver_v
    if op == "<=":
        return pin_v > ver_v
    if op == ">":
        return pin_v <= ver_v
    if op == ">=":
        return pin_v < ver_v
    if op == "==":
        return pin_v != ver_v
    return False


def _check_range_excludes_pin(rng, pin: str) -> bool:
    """Given a vllm_version_range (tuple, list, or string), does it
    exclude the current pin?"""
    if isinstance(rng, str):
        # Comma-separated specifier string
        parts = [p.strip() for p in rng.split(",")]
        return any(_excludes_pin(p, pin) for p in parts if p)
    if isinstance(rng, (tuple, list)):
        return any(_excludes_pin(p, pin) for p in rng if isinstance(p, str))
    return False


def _import_registry():
    sys.path.insert(0, str(REPO_ROOT))
    from sndr.dispatcher.registry import PATCH_REGISTRY
    return PATCH_REGISTRY


def _build_builtin_yaml_env_map() -> dict[str, list[str]]:
    """v11.3.0 BUG #14 helper: walk every model_configs/builtin/**.yaml
    and return {env_flag → [yaml_filename, ...]} for truthy
    `GENESIS_ENABLE_<X>: '1'` lines.

    Used by the severity classifier to escalate opt-in patches that
    are enabled by some builtin YAML — those are operationally critical
    when the version range excludes the current pin (operators set the
    flag expecting the patch; the patch silently no-ops).
    """
    out: dict[str, list[str]] = {}
    yaml_dir = REPO_ROOT / "sndr" / "model_configs" / "builtin"
    if not yaml_dir.is_dir():
        return out
    flag_re = re.compile(
        r"^\s*(GENESIS_ENABLE_[A-Z0-9_]+)\s*:\s*['\"]?([^'\"\s#]+)"
    )
    for yp in yaml_dir.rglob("*.yaml"):
        try:
            text = yp.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = yp.name
        for line in text.splitlines():
            m = flag_re.match(line)
            if not m:
                continue
            value = m.group(2).strip()
            if value not in ("1", "true", "True"):
                continue
            flag = m.group(1)
            out.setdefault(flag, []).append(rel)
    return out


def _audit(pin: str) -> dict:
    registry = _import_registry()
    builtin_yaml_envs = _build_builtin_yaml_env_map()
    rows: list[dict] = []
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        lifecycle = meta.get("lifecycle", "")
        if lifecycle in ("retired", "deprecated", "legacy"):
            continue
        applies_to = meta.get("applies_to") or {}
        if not isinstance(applies_to, dict):
            continue
        rng = applies_to.get("vllm_version_range")
        if not rng:
            continue
        if not _check_range_excludes_pin(rng, pin):
            continue
        # v11.3.0 BUG #14 severity escalation: pre-compute enabled-in-prod
        # status. Patches that are opt-in (default_on=False) BUT enabled
        # by some builtin model/profile YAML are operationally CRITICAL
        # even though the legacy severity tagged them WARN — operators
        # set the env flag expecting the patch to apply, but on the
        # current pin it silently no-ops via version-range mismatch.
        default_on = bool(meta.get("default_on"))
        env_flag = meta.get("env_flag")
        enabled_in_yamls = bool(env_flag) and (env_flag in builtin_yaml_envs)
        if default_on:
            severity = "CRITICAL"
            severity_reason = "default_on=True; silently skips on every boot"
        elif enabled_in_yamls:
            severity = "CRITICAL"
            severity_reason = (
                f"enabled in {len(builtin_yaml_envs.get(env_flag, []))} "
                f"builtin YAML(s); silent no-op on this pin"
            )
        else:
            severity = "WARN"
            severity_reason = "opt-in only; not enabled by any builtin YAML"
        rows.append({
            "patch_id": pid,
            "severity": severity,
            "severity_reason": severity_reason,
            "vllm_version_range": rng,
            "lifecycle": lifecycle,
            "default_on": default_on,
            "env_flag": env_flag,
            "family": meta.get("family"),
            "enabled_in_yamls": enabled_in_yamls,
            "yaml_consumers": builtin_yaml_envs.get(env_flag, []),
        })
    # Sort by severity (CRITICAL first) then patch_id
    rows.sort(key=lambda r: (0 if r["severity"] == "CRITICAL" else 1,
                              r["patch_id"]))
    return {
        "pin": pin,
        "total_stale_ranges": len(rows),
        "critical_count": sum(1 for r in rows if r["severity"] == "CRITICAL"),
        "warn_count": sum(1 for r in rows if r["severity"] == "WARN"),
        "rows": rows,
    }


def _print_human(result: dict) -> None:
    print("=" * 70)
    print(f"Stale vllm_version_range audit — pin = {result['pin']}")
    print("=" * 70)
    print()
    print(f"Total stale ranges:    {result['total_stale_ranges']}")
    print(f"  CRITICAL: {result['critical_count']} "
          f"(default_on=True silent-skip OR enabled-in-builtin-YAML silent-no-op)")
    print(f"  WARN:     {result['warn_count']} "
          f"(opt-in only, not enabled by any YAML)")
    print()
    if result["critical_count"] > 0:
        print(
            "⚠⚠⚠ CRITICAL entries — patch silently skips when version "
            "range excludes current pin. v11.3.0 BUG #14 escalation: "
            "opt-in patches that are ENABLED IN BUILTIN YAMLs are also "
            "critical (operator-visible silent no-op)."
        )
        print()
    if result["rows"]:
        print(f"{'Severity':<10} {'Patch':<25} {'Range':<35} {'env_flag':<40}")
        print("-" * 110)
        for r in result["rows"]:
            range_str = str(r["vllm_version_range"])
            if len(range_str) > 34:
                range_str = range_str[:31] + "..."
            print(
                f"{r['severity']:<10} {r['patch_id']:<25} {range_str:<35} "
                f"{r.get('env_flag') or '':<40}"
            )
        print()
    if result["total_stale_ranges"] == 0:
        print(
            "✓ No stale version ranges. All active patches' ranges "
            "include the current pin."
        )
    else:
        print(
            f"Recommendation: bulk-update the {result['total_stale_ranges']} "
            "stale ranges to reflect the current support window. Common "
            "fix: bump upper bound from `<0.21.0` to `<0.22.0` if the "
            "patch is verified working on 0.21.x."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pin", help="override current pin (defaults to vllm.__version__)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 if any CRITICAL entries found",
    )
    args = parser.parse_args()

    pin = _resolve_current_pin(args.pin)
    result = _audit(pin)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)

    if args.strict:
        # v11.3.0 BUG #14: --strict fails on CRITICAL entries NOT in
        # baseline allowlist. Baseline entries surface as INFO in the
        # human report but don't block CI.
        new_critical = [
            r for r in result["rows"]
            if r["severity"] == "CRITICAL"
            and r["patch_id"] not in _BASELINE_CRITICAL_STALE
        ]
        if new_critical:
            print(
                f"\n⚠ --strict failed: {len(new_critical)} CRITICAL "
                f"stale-range entries not in v11.3.0 baseline allowlist:"
            )
            for r in new_critical:
                print(
                    f"  - {r['patch_id']}: {r['vllm_version_range']} "
                    f"({r['severity_reason']})"
                )
            print(
                "\nEither bump the upper bound in registry.py (recommended) "
                "OR add to _BASELINE_CRITICAL_STALE with a justification."
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
