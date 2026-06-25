#!/usr/bin/env python3
"""TASK 4 — human summary of a pin's drift.rej.json (commit-coverage hygiene).

build_manifest.py writes pins/<pin>/drift.rej.json next to anchors.json recording
every anchor that did NOT make it into the manifest, by status. This prints a
concise breakdown (counts by status: anchor_drift / retired / upstream_merged /
ambiguous / version_gated / optional_absent / target_missing) plus the per-patch
merge tri-state roll-up, so the operator can see at a glance which patches were
dropped and why on a given pin — instead of the dropped set being invisible.

``retired`` is printed separately from ``anchor_drift``: a retired patch's anchor
is gone by design (its code was superseded / absorbed upstream), so it is NOT a
re-anchor candidate. The operator sees "N retired patches' anchors are gone, as
expected" rather than a false re-anchor backlog.

Usage:
    summarize_rej.py <pin/drift.rej.json>
    summarize_rej.py <pin_dir>            # resolves <pin_dir>/drift.rej.json
    summarize_rej.py                      # summarize every committed pin's rej

Exit codes: 0 ok (report only) · 2 no rej file found.
"""
import json
import os
import sys

_STATUS_ORDER = (
    "anchor_drift",
    "retired",
    "upstream_merged",
    "ambiguous",
    "version_gated",
    "optional_absent",
    "target_missing",
)


def _repo_pins_dir():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "sndr", "engines", "vllm", "pins"))


def _resolve(path):
    """Return a drift.rej.json file path from a file/dir argument."""
    if os.path.isdir(path):
        return os.path.join(path, "drift.rej.json")
    return path


def _summarize_one(rej_path):
    if not os.path.isfile(rej_path):
        print("no drift.rej.json at %s" % rej_path, file=sys.stderr)
        return False
    data = json.load(open(rej_path))
    pin = data.get("pin", "?")
    cov = data.get("coverage") or {}
    counts = data.get("counts") or {}
    rejected = data.get("rejected") or []
    merge = data.get("merge_status") or {}

    print("=== %s ===" % pin)
    if cov:
        print("coverage: discovered=%s  ok=%s  rejected=%s" % (
            cov.get("discovered", "?"), cov.get("ok", "?"), cov.get("rejected", "?")))

    print("rejected by status:")
    seen = set()
    for status in _STATUS_ORDER:
        if status in counts:
            print("  %-16s %d" % (status, counts[status]))
            seen.add(status)
    for status, n in sorted(counts.items()):
        if status not in seen:
            print("  %-16s %d" % (status, n))

    # the actionable subset — what a human must re-anchor on a bump.
    # EXCLUDES retired (status `retired`): a retired patch's anchor is gone by
    # design, so it is never a re-anchor candidate.
    genuine = [e for e in rejected if e.get("status") == "anchor_drift"]
    if genuine:
        print("genuine anchor_drift (re-anchor these %d):" % len(genuine))
        for e in genuine[:20]:
            print("  * %s (%s)" % (e.get("key"), e.get("target_rel")))
        if len(genuine) > 20:
            print("  ... and %d more" % (len(genuine) - 20))

    # retired patches whose anchors are gone, as EXPECTED — informational, not
    # actionable. Printed separately so the operator does not mistake them for
    # re-anchor work.
    retired = [e for e in rejected if e.get("status") == "retired"]
    if retired:
        print("retired (anchors gone, as expected — do NOT re-anchor %d):"
              % len(retired))
        for e in retired[:20]:
            print("  - %s (%s)" % (e.get("key"), e.get("target_rel")))
        if len(retired) > 20:
            print("  ... and %d more" % (len(retired) - 20))

    # retire-impact / dependency-breakage — the bug class that slipped through
    # on dev148->dev301: a retired patch silently breaks a DIFFERENT patch that
    # depends on it, no-op'ing a perf optimization without surfacing as drift.
    # A perf-tier dependent that SKIPS due to a retired dependency must NOT be
    # lumped with benign skips — it is flagged here as a perf/correctness risk.
    breakage = data.get("dependency_breakage") or {}
    edges = breakage.get("edges") or []
    if edges:
        hi = [e for e in edges if e.get("severity") == "HIGH"]
        md = [e for e in edges if e.get("severity") != "HIGH"]
        print("⚠ retire-broken dependents (%d) — perf/correctness risk:"
              % len(edges))
        if hi:
            print("  HIGH (perf — its optimization NO-OPs; run a canonical A/B "
                  "before promoting):")
            for e in hi:
                print("    * %s -> %s  via=%s" % (
                    e.get("retired"), e.get("dependent"),
                    ",".join(e.get("via") or [])))
        if md:
            print("  MEDIUM (behaviour change):")
            for e in md:
                print("    - %s -> %s  via=%s" % (
                    e.get("retired"), e.get("dependent"),
                    ",".join(e.get("via") or [])))

    # per-patch upstream-merge tri-state roll-up
    if merge:
        by_status = {}
        for pid, m in merge.items():
            by_status.setdefault(m.get("merge_status", "?"), []).append(pid)
        print("merge_status roll-up:")
        for status in ("fully_merged", "partially_merged", "not_merged"):
            pids = sorted(by_status.get(status, []))
            if pids:
                head = pids[:12]
                more = "" if len(pids) <= 12 else " ... (+%d)" % (len(pids) - 12)
                print("  %-18s %d  %s%s" % (status, len(pids), head, more))
    return True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        ok = _summarize_one(_resolve(argv[0]))
        return 0 if ok else 2

    # no arg: walk every committed pin
    pins_dir = _repo_pins_dir()
    if not os.path.isdir(pins_dir):
        print("pins dir not found: %s" % pins_dir, file=sys.stderr)
        return 2
    found = False
    for name in sorted(os.listdir(pins_dir)):
        rej = os.path.join(pins_dir, name, "drift.rej.json")
        if os.path.isfile(rej):
            _summarize_one(rej)
            found = True
    if not found:
        print("no drift.rej.json committed for any pin — run make rebuild-pin "
              "on the rig to populate it.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
