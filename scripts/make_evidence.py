#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 supplement deliverable — `make evidence` aggregate gate.

Pending since Entry 1 of `ROADMAP_EVIDENCE_LEDGER`. Runs every release
audit gate in one command + emits a structured summary that can be
pasted directly into a new ledger entry. Operator runs this before
release; CI runs it on every PR.

Design:

  • Each gate is a `Gate` dataclass with name, Makefile target, severity
    (gating | informational), and an optional "release-only" flag.
  • The aggregate is idempotent — running it twice produces no side
    effects beyond updating the per-run output file.
  • The script captures stdout + stderr + exit code per gate so
    failures are diagnosable without re-running individual targets.

Modes:

  python3 scripts/make_evidence.py               # human report; exit 1 if any gating gate fails
  python3 scripts/make_evidence.py --json        # machine-readable summary
  python3 scripts/make_evidence.py --release     # include release-only gates (SBOM, dirty-state-release)
  python3 scripts/make_evidence.py --emit-md FILE  # write a ready-to-paste ledger entry

Exit codes:
  0 — every gating gate passed
  1 — at least one gating gate failed
  2 — internal error (e.g. make / subprocess invocation broke)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Gate:
    """Definition of one audit gate."""
    name: str
    make_target: str
    description: str
    severity: str          # "gating" | "informational"
    release_only: bool = False


# Single source of truth — every gate the project surfaces via `make audit-*`.
# Add to this list when a new gate ships.
GATES: tuple[Gate, ...] = (
    # ── Always-on gating audits ────────────────────────────────────────
    Gate("audit", "audit",
         "legacy aggregate (legacy-imports + public-paths + upstream + doc-sync)",
         "gating"),
    Gate("audit-configs", "audit-configs",
         "every V2 preset alias composes cleanly",
         "gating"),
    Gate("audit-community", "audit-community",
         "community SDK release-tier validator (R-1..R-7)",
         "gating"),
    Gate("audit-no-new-v1", "audit-no-new-v1",
         "Phase 9 freeze — top-level builtin/*.yaml matches frozen baseline",
         "gating"),
    Gate("audit-patches-prove-all", "audit-patches-prove-all",
         "§6.8 every PATCH_REGISTRY entry passes static checks",
         "gating"),
    Gate("audit-all-referents", "audit-all-referents",
         "§8 F822 — every `__all__` name resolves",
         "gating"),
    Gate("audit-readme-counters", "audit-readme-counters",
         "§8 README counters match live registry",
         "gating"),
    Gate("audit-model-baselines", "audit-model-baselines",
         "every V2 model's reference_metrics_ref points at an existing JSON",
         "gating"),
    Gate("audit-launch-coverage", "audit-launch-coverage",
         "§4.2 every V2 hardware YAML covers canonical mount + env schema",
         "gating"),
    Gate("audit-v2-env-keys", "audit-v2-env-keys",
         "§4.2 every Genesis/SNDR key across V2 model+profile+resolved-alias is canonical",
         "gating"),
    Gate("audit-bench-methodology", "audit-bench-methodology",
         "§6.8/§5 every bench_delta.methodology_sha matches current methodology contract",
         "gating"),
    Gate("audit-no-hardcoded-paths", "audit-no-hardcoded-paths",
         "§6.10 active config uses ${var} placeholders, no /home/USER paths",
         "gating"),
    Gate("audit-v2-required-fields", "audit-v2-required-fields",
         "§4.2 each V2 model/hardware/profile/preset has the required top-level fields",
         "gating"),
    Gate("audit-v2-freshness", "audit-v2-freshness",
         "§4.2 V2 model last_validated not older than 180 days",
         "informational"),
    Gate("audit-v2-id-consistency", "audit-v2-id-consistency",
         "§4.2 each V2 model/hardware/profile YAML id equals its filename stem",
         "gating"),
    Gate("audit-v2-license-coverage", "audit-v2-license-coverage",
         "§4.2/§6.10 each V2 model has SPDX license + non-empty maintainer",
         "gating"),
    Gate("audit-v2-cross-reference", "audit-v2-cross-reference",
         "§4.2 every profile.parent_model + preset triplet ref resolves",
         "gating"),
    Gate("audit-v2-vllm-pin-consistency", "audit-v2-vllm-pin-consistency",
         "§4.2 model.versions.vllm_pin_required matches baseline JSON's vllm_version",
         "gating"),
    Gate("audit-v2-patch-lifecycle", "audit-v2-patch-lifecycle",
         "§4.2 enabled-retired patches in V2 models must be on operator allowlist",
         "gating"),
    Gate("audit-v2-hardware-sanity", "audit-v2-hardware-sanity",
         "§4.2 V2 hardware numeric fields within sane bounds + cross-field VRAM budget",
         "gating"),
    Gate("audit-v2-patch-dependencies", "audit-v2-patch-dependencies",
         "§4.2 every enabled V2 patch's requires_patches + conflicts_with satisfied",
         "gating"),
    Gate("audit-v2-default-on-mismatch", "audit-v2-default-on-mismatch",
         "§4.2 surfaces explicit operator overrides of default_on=True patches",
         "informational"),
    Gate("audit-v2-capability-coverage", "audit-v2-capability-coverage",
         "§4.2 V2 model.capabilities strings in frozen allowed set",
         "gating"),
    Gate("audit-v2-versions-pin-format", "audit-v2-versions-pin-format",
         "§4.2 V2 model.versions pin fields match canonical format regex",
         "gating"),
    Gate("audit-v2-quantization-coverage", "audit-v2-quantization-coverage",
         "§4.2 V2 model.quantization + dtype in frozen allowed set",
         "gating"),
    Gate("audit-v2-context-length-sanity", "audit-v2-context-length-sanity",
         "§4.2 V2 hardware sizing max_model_len + batch sane + consistent",
         "gating"),
    Gate("audit-v2-runtime-image-pin", "audit-v2-runtime-image-pin",
         "§4.2 V2 hardware.runtime.docker.image_digest is a canonical sha256 pin",
         "gating"),
    Gate("audit-v2-network-port-consistency", "audit-v2-network-port-consistency",
         "§4.2 V2 hardware.runtime.docker network/ports/shm valid",
         "gating"),
    Gate("audit-runtime-hook-ratchet", "audit-runtime-hook-ratchet",
         "§4.2 P2.3 stable patches declare stable_kind; runtime-hook requires ≥2 production pins",
         "gating"),
    Gate("audit-no-stub", "audit-no-stub",
         "§10.3 #2 / §10.5 no-stub gate: bare NotImplementedError / TODO(...) / sentinel pass in vllm/sndr_core",
         "gating"),
    Gate("audit-schema-sync", "audit-schema-sync",
         "P0-3 (audit 2026-05-14): package + root patch-entry schemas byte-identical",
         "gating"),
    Gate("audit-patch-attribution", "audit-patch-attribution",
         "Phase A (2026-05-16): ModelDef.patches_attribution keys + role-presence consistency",
         "gating"),
    Gate("audit-patch-plan-resolves", "audit-patch-plan-resolves",
         "Phase D (2026-05-16): every V2 preset resolves cleanly under compat/safe/minimal",
         "gating"),
    Gate("audit-engine-boundary", "audit-engine-boundary",
         "§10.3 #5 engine boundary: only optional-discovery vllm.sndr_engine imports in sndr_core",
         "gating"),
    Gate("audit-config-keys", "audit-config-keys",
         "§10.3 #4 / §6.7 every committed YAML's Genesis/SNDR keys in canonical registry",
         "gating"),
    Gate("audit-evidence-freshness", "audit-evidence-freshness",
         "§10.3 #3 evidence ledger freshness (skipped on CI when ledger absent)",
         "informational"),
    Gate("audit-license-anchor", "audit-license-anchor",
         "P1-6 license trust anchor: warn when dev-only anchor still active",
         "informational"),
    Gate("audit-license-anchor-release", "audit-license-anchor-release",
         "P1-6 release-strict: refuse release when dev-only anchor still active",
         "gating", release_only=True),
    Gate("audit-docs-stale", "audit-docs-stale",
         "§supplement-3 forbid stale tokens (retired verbs, renamed modules) in public docs",
         "gating"),
    Gate("audit-public-docs", "audit-public-docs",
         "§6.10 public/private docs boundary (no _internal links, private IPs, operator paths, retired verbs)",
         "gating"),
    # ── Informational gates (warnings only) ────────────────────────────
    Gate("audit-security", "audit-security",
         "Phase 4.6 security scan (warning-only — pre-existing operator paths)",
         "informational"),
    Gate("audit-patches-prove", "audit-patches-prove",
         "§6.8 dead-patch detector (informational coverage report)",
         "informational"),
    Gate("audit-proof-status", "audit-proof-status",
         "§6.8 read-side: per-patch bucket summary (informational)",
         "informational"),
    Gate("audit-release-check", "audit-release-check",
         "§6.8/§10.3 release-gate: every patch has ≥static proof artefact (require-static)",
         "gating"),
    # ── Release-only gates (require --release flag) ────────────────────
    Gate("audit-dirty-state-release", "audit-dirty-state-release",
         "dirty-state policy release tier — worktree must be clean",
         "gating", release_only=True),
    Gate("audit-artifacts-release", "audit-artifacts-release",
         "release artefact policy — SBOM + constraints present",
         "gating", release_only=True),
    Gate("audit-security-release", "audit-security-release",
         "security scan release-strict mode",
         "gating", release_only=True),
)


@dataclass
class GateResult:
    gate: Gate
    exit_code: int
    duration_s: float
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def blocks_release(self) -> bool:
        """A gating-severity failure blocks; informational failures don't."""
        return self.gate.severity == "gating" and not self.passed


def _run_gate(gate: Gate, *, timeout_s: int = 180) -> GateResult:
    """Invoke `make <target>` and capture exit + tail of output."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["make", "--no-print-directory", gate.make_target],
            cwd=REPO_ROOT,
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code = 124    # Conventional timeout code.
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = f"TIMEOUT after {timeout_s}s"
    except FileNotFoundError:
        exit_code = 2
        stdout = ""
        stderr = "`make` not found in PATH"
    dt = time.monotonic() - t0
    return GateResult(
        gate=gate,
        exit_code=exit_code,
        duration_s=round(dt, 2),
        stdout_tail="\n".join((stdout or "").splitlines()[-8:]),
        stderr_tail="\n".join((stderr or "").splitlines()[-4:]),
    )


def _gates_for_mode(*, include_release: bool) -> tuple[Gate, ...]:
    if include_release:
        return GATES
    return tuple(g for g in GATES if not g.release_only)


def run_evidence(*, include_release: bool = False,
                 timeout_s: int = 180) -> list[GateResult]:
    """Run every applicable gate, return per-gate results."""
    return [_run_gate(g, timeout_s=timeout_s)
            for g in _gates_for_mode(include_release=include_release)]


# ─── Output renderers ────────────────────────────────────────────────


def render_text(results: list[GateResult]) -> str:
    lines = []
    lines.append(f"make evidence — {len(results)} gate(s)")
    lines.append("─" * 70)
    blocking_failures = 0
    informational_failures = 0
    for r in results:
        sym = "✓" if r.passed else "✗"
        sev_pad = r.gate.severity.upper().ljust(13)
        line = f"  {sym} [{sev_pad}] {r.gate.name:36s} {r.duration_s}s"
        lines.append(line)
        if not r.passed:
            if r.blocks_release:
                blocking_failures += 1
            else:
                informational_failures += 1
            for tail_line in r.stdout_tail.splitlines()[-3:]:
                lines.append(f"      | {tail_line}")
    lines.append("")
    if blocking_failures == 0:
        lines.append(f"  ✓ {sum(1 for r in results if r.passed)}/{len(results)} "
                     f"gate(s) green; {informational_failures} informational warning(s)")
    else:
        lines.append(
            f"  ✗ RELEASE BLOCKED — {blocking_failures} gating gate(s) failed"
        )
    return "\n".join(lines)


def render_json(results: list[GateResult]) -> str:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": _detect_host(),
        "commit_sha": _git_short_sha(),
        "include_release_gates": any(r.gate.release_only for r in results),
        "total_gates": len(results),
        "passed": sum(1 for r in results if r.passed),
        "blocking_failures": sum(1 for r in results if r.blocks_release),
        "informational_failures": sum(
            1 for r in results
            if not r.passed and not r.blocks_release
        ),
        "release_blocked": any(r.blocks_release for r in results),
        "gates": [
            {
                "name": r.gate.name,
                "make_target": r.gate.make_target,
                "severity": r.gate.severity,
                "release_only": r.gate.release_only,
                "exit_code": r.exit_code,
                "duration_s": r.duration_s,
                "passed": r.passed,
                "blocks_release": r.blocks_release,
                "stdout_tail": r.stdout_tail,
                "stderr_tail": r.stderr_tail,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_markdown_ledger_entry(results: list[GateResult]) -> str:
    """A ready-to-paste markdown block matching the existing ledger
    entry style."""
    blocking = [r for r in results if r.blocks_release]
    informational = [r for r in results if not r.passed and not r.blocks_release]
    passed = [r for r in results if r.passed]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = _git_short_sha()
    host = _detect_host()

    lines = [
        "### Entry X — `make evidence` aggregate run",
        "",
        f"- timestamp: {ts}",
        f"- host: {host}",
        f"- commit: {sha}",
        f"- gates: {len(results)} ({len(passed)} green, "
        f"{len(blocking)} blocking failures, {len(informational)} informational)",
        "",
    ]
    if not blocking:
        lines.append("**Release status:** OK — every gating gate passed.")
    else:
        lines.append(
            f"**Release status:** BLOCKED — {len(blocking)} gating gate(s) failed."
        )
    lines.append("")
    lines.append("```text")
    for r in results:
        sym = "✓" if r.passed else "✗"
        lines.append(
            f"{sym} [{r.gate.severity:13s}] {r.gate.name:36s} "
            f"({r.duration_s}s)"
        )
    lines.append("```")
    if blocking:
        lines.append("")
        lines.append("**Blocking gate output (tails):**")
        lines.append("")
        for r in blocking:
            lines.append("```text")
            lines.append(f"# {r.gate.name}")
            lines.append(r.stdout_tail or "(no stdout)")
            if r.stderr_tail:
                lines.append("---stderr---")
                lines.append(r.stderr_tail)
            lines.append("```")
    return "\n".join(lines) + "\n"


# ─── Provenance helpers (mirror prove.py) ─────────────────────────────


def _git_short_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _detect_host() -> str:
    import os
    import socket
    forced = os.environ.get("SNDR_HOST_LABEL")
    if forced:
        return forced
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


# ─── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", action="store_true",
                    help="Include release-only gates (dirty-state, SBOM, etc.).")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON summary.")
    ap.add_argument("--emit-md", metavar="FILE", default=None,
                    help="Write a paste-ready markdown ledger entry to FILE.")
    ap.add_argument("--timeout", type=int, default=180,
                    help="Per-gate timeout in seconds (default: 180).")
    ap.add_argument("--only", metavar="NAME", default=None,
                    help="Run only the named gate (for debugging).")
    args = ap.parse_args()

    if args.only:
        target_gate = next((g for g in GATES if g.name == args.only), None)
        if target_gate is None:
            sys.stderr.write(
                f"make_evidence: --only={args.only!r} not in known gates: "
                f"{[g.name for g in GATES]}\n"
            )
            return 2
        results = [_run_gate(target_gate, timeout_s=args.timeout)]
    else:
        results = run_evidence(
            include_release=args.release,
            timeout_s=args.timeout,
        )

    if args.emit_md:
        Path(args.emit_md).write_text(
            render_markdown_ledger_entry(results),
            encoding="utf-8",
        )

    if args.json:
        print(render_json(results))
    else:
        print(render_text(results))

    blocking = sum(1 for r in results if r.blocks_release)
    return 0 if blocking == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
