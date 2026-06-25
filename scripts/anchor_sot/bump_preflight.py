#!/usr/bin/env python3
"""Bump-preflight gate — surface the silent retire-breakage class on a pin bump.

Given the OLD-pin and NEW-pin anchor manifests (each a ``pins/<pin>/`` dir or an
``anchors.json`` with a sibling ``drift.rej.json``), print an ACTIONABLE
checklist and exit NON-ZERO if any HIGH-severity perf dependent is broken on the
new pin. Reports:

  (a) patches NEWLY retired / version-gated-out on the new pin (vs old),
  (b) their BROKEN dependents — the retire-impact detector edges, ranked,
  (c) PERF-tier patches that moved ok -> skip/drift between the two pins
      (the perf-landmine list: a perf optimization that silently went dead),
  (d) a reminder that a canonical A/B (``genesis_bench_suite.py``) MUST be run
      for any perf-tier delta (iron-rule #9).

This is the gate the dev148->dev301 bump lacked: PN353A retired, PN399's anchor
went missing, PN399 SKIPPED as a benign skip, its decode-scratch perf
optimization no-op'd, and a real -5.5% TPS regression hit the 35B with
``genuine_drift=0`` (clean). With this gate the bump fails loudly on the broken
HIGH-severity PERF dependent until a canonical A/B clears it.

Usage:
    bump_preflight.py <old_pin_dir|old_anchors.json> <new_pin_dir|new_anchors.json>

Exit codes:
    0  no HIGH-severity perf dependent broken on the new pin (clean / advisory)
    1  >=1 HIGH-severity perf dependent broken — run a canonical A/B before bump
    2  usage / unreadable input
"""
import json
import os
import sys

# Make the repo root importable so the detector + spec layer resolve when this
# script is invoked directly (mirrors compare_manifest._repo_root()).
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))


def _resolve_pair(path):
    """Return (anchors.json, drift.rej.json) paths from a dir or anchors file."""
    if os.path.isdir(path):
        return (os.path.join(path, "anchors.json"),
                os.path.join(path, "drift.rej.json"))
    base = os.path.dirname(path)
    return path, os.path.join(base, "drift.rej.json")


def _load(path):
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def _ok_patch_ids(manifest):
    """patch ids with >=1 applied (ok) anchor in a manifest."""
    out = set()
    for fe in (manifest.get("files") or {}).values():
        for pid in ((fe or {}).get("patches") or {}):
            pe = fe["patches"][pid]
            if (pe or {}).get("anchors"):
                out.add(pid)
    return out


def _rej_ids_by_status(rej, statuses):
    """patch ids in the reject set whose status is in ``statuses``."""
    out = set()
    for e in (rej.get("rejected") or []):
        if e.get("status") in statuses:
            out.add(str(e.get("key", "")).split("::", 1)[0])
    return out


def _perf_patch_ids():
    """Set of patch ids that are performance-bearing (live registry)."""
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        from sndr.dispatcher.spec import iter_patch_specs
        from sndr.engines.vllm.retire_impact import is_perf_signal
    except Exception:  # noqa: BLE001 — registry unavailable: empty set
        return set()
    out = set()
    for s in iter_patch_specs():
        credit = (PATCH_REGISTRY.get(s.patch_id) or {}).get("credit", "")
        if is_perf_signal(s.category, s.title, credit):
            out.add(s.patch_id)
    return out


def preflight(old_dir, new_dir):
    old_anchors_p, old_rej_p = _resolve_pair(old_dir)
    new_anchors_p, new_rej_p = _resolve_pair(new_dir)
    old_anchors, new_anchors = _load(old_anchors_p), _load(new_anchors_p)
    old_rej, new_rej = _load(old_rej_p) or {}, _load(new_rej_p)
    if old_anchors is None or new_anchors is None or new_rej is None:
        miss = [p for p, v in (
            (old_anchors_p, old_anchors), (new_anchors_p, new_anchors),
            (new_rej_p, new_rej)) if v is None]
        print("FATAL: missing/unreadable input: %s" % miss, file=sys.stderr)
        return 2

    old_pin = (old_anchors.get("pins") or {}).get("vllm", "?")
    new_pin = (new_anchors.get("pins") or {}).get("vllm", "?")
    print("=== bump preflight: %s -> %s ===" % (old_pin, new_pin))

    _RETIRE_STATUSES = ("retired", "version_gated")
    old_retired = _rej_ids_by_status(old_rej, _RETIRE_STATUSES)
    new_retired = _rej_ids_by_status(new_rej, _RETIRE_STATUSES)
    newly_retired = sorted(new_retired - old_retired)

    # (a) newly retired / gated-out on the new pin
    print("\n(a) newly retired / version-gated-out on %s (%d):"
          % (new_pin, len(newly_retired)))
    for pid in newly_retired:
        print("    - %s" % pid)
    if not newly_retired:
        print("    (none)")

    # (b) broken dependents — read from the new pin's dependency_breakage
    # section (already computed by build_manifest), falling back to a live
    # detector run if the section is absent (older manifest).
    breakage = new_rej.get("dependency_breakage")
    if breakage is None:
        try:
            from sndr.engines.vllm.retire_impact import detect_on_live_registry
            breakage = detect_on_live_registry(
                gated_out=_rej_ids_by_status(new_rej, ("version_gated",))
            ).to_dict()
        except Exception:  # noqa: BLE001
            breakage = {"high_count": 0, "medium_count": 0, "edges": []}
    edges = breakage.get("edges") or []
    high_edges = [e for e in edges if e.get("severity") == "HIGH"]
    # APPLY-STATE-AWARE: a HIGH edge flagged ``mitigated`` is already handled (the
    # dependent has a working fallback anchor independent of the retired id — the
    # PN399 native-form C2 sibling). It is still LISTED, but the gate fails ONLY on
    # genuinely-UNMITIGATED HIGH edges (the real PN399-incident class: a dependent
    # whose only path references the retired id).
    high_unmitigated = [e for e in high_edges if not e.get("mitigated")]
    high_mitigated = [e for e in high_edges if e.get("mitigated")]
    print("\n(b) retire-broken dependents (HIGH=%d [mitigated=%d unmitigated=%d] "
          "MEDIUM=%d):" % (breakage.get("high_count", 0), len(high_mitigated),
                           len(high_unmitigated), breakage.get("medium_count", 0)))
    for e in edges:
        if e.get("severity") == "HIGH":
            mark = "HIGH-MITIGATED" if e.get("mitigated") else "HIGH"
        else:
            mark = "med "
        print("    [%s] %s" % (mark, e.get("detail")))
    if not edges:
        print("    (none)")

    # (c) perf-tier patches that moved ok -> skip/drift between the two pins
    perf_ids = _perf_patch_ids()
    old_ok = _ok_patch_ids(old_anchors)
    new_ok = _ok_patch_ids(new_anchors)
    new_skipped = _rej_ids_by_status(
        new_rej, ("anchor_drift", "retired", "version_gated",
                  "ambiguous", "optional_absent", "upstream_merged"))
    perf_landmines = sorted(
        pid for pid in (old_ok - new_ok)
        if pid in perf_ids and pid in new_skipped
    )
    print("\n(c) PERF-tier patches that went ok -> skip/drift on %s (%d) "
          "— perf landmines:" % (new_pin, len(perf_landmines)))
    for pid in perf_landmines:
        print("    * %s (was applied on %s, now skipped/drifted)"
              % (pid, old_pin))
    if not perf_landmines:
        print("    (none)")

    # (d) iron-rule #9 reminder + verdict
    perf_delta = bool(high_unmitigated) or bool(perf_landmines)
    print("\n(d) iron-rule #9 — bench methodology:")
    if perf_delta:
        print("    A canonical A/B (genesis_bench_suite.py --quick) is REQUIRED "
              "for the perf-tier delta above before promoting this pin. "
              "Custom/temp=0 single-prompt benches give a systematic offset — "
              "apples-to-apples only.")
    else:
        print("    No perf-tier delta detected — no A/B gate triggered "
              "(still bench-validate the pin per the bump playbook).")

    if high_unmitigated:
        print("\nRESULT: FAIL — %d UNMITIGATED HIGH-severity PERF dependent(s) "
              "broken on %s (no working fallback anchor). Re-anchor the dependent "
              "against the new pin OR run a canonical A/B proving no regression "
              "before promoting." % (len(high_unmitigated), new_pin))
        return 1
    if high_mitigated:
        print("\nRESULT: PASS — %d HIGH-severity edge(s) are MITIGATED (the "
              "dependent has a working fallback anchor independent of the retired "
              "id — already handled); no unmitigated HIGH on %s."
              % (len(high_mitigated), new_pin))
        return 0
    print("\nRESULT: PASS — no HIGH-severity perf dependent broken on %s."
          % new_pin)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: bump_preflight.py <old_pin_dir|anchors.json> "
              "<new_pin_dir|anchors.json>", file=sys.stderr)
        return 2
    return preflight(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main())
