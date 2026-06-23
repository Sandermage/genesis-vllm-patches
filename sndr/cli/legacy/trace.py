# SPDX-License-Identifier: Apache-2.0
"""§6.H6-H9 (UNIFIED_DEVELOPMENT_PLAN) — `sndr trace list/collect/
summarize` + `sndr support-bundle`.

Verbs for the plan's §6.H trace surface:

  * H6 ``sndr trace list`` — list known traces, optionally pull live
    size/mtime from a running container.
  * H7 ``sndr trace collect --container <name>`` — copy live traces
    out of a container into a host directory; default output is a
    timestamped subfolder under cwd.
  * H8 ``sndr trace summarize <log-file>`` — quick stat analysis on
    a collected trace; auto-detects trace kind from basename and
    dispatches to a per-kind summarizer (generic line-count / size
    / first-last lines plus kind-specific helpers — currently
    `boot` extracts apply/skip/fail counts + failing patch ids).
  * H9 ``sndr support-bundle --container <name>`` — composes H7
    + H8 + host/container facts into a single ``.tar.gz`` ready
    for off-rig analysis. ``sndr support-bundle`` is registered
    under the TOP-level CLI (alongside ``sndr trace``) to match the
    plan §6.H9 spec; the implementation lives here so it can reuse
    the trace primitives.

H6 surfaces the operator-facing question "what diagnostic traces exist
and which are currently being written?" via the catalog at
``sndr.observability.trace_catalog``. ``--container <name>``
runs ``docker exec ls -l`` against the container's ``/tmp/`` and joins
the result with the catalog so the output shows which traces are
actually present and how big they are.

Output modes:

  Default (human view) — categories with a one-line summary per trace
  + per-line annotation when ``--container`` is used.

  ``--json`` — machine-readable; the same structure suitable for the
  H9 support-bundle aggregator.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from typing import Any, Optional

from . import _io


__all__ = [
    "add_argparser", "add_support_bundle_argparser",
    "run_list", "run_collect", "run_summarize", "run_support_bundle",
    "detect_trace_kind", "summarize_boot_log", "summarize_generic",
    "summarize_pn248_acceptance",
    "collect_host_facts", "collect_container_facts",
]


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "trace",
        help=(
            "Genesis diagnostic-trace catalog (§6.H6) — list known "
            "trace files, optionally introspect a live container."
        ),
        description=(
            "Inspect Genesis's known diagnostic traces. By default "
            "prints the static catalog (every trace emitted by some "
            "patch). With --container, runs `docker exec ls -l` "
            "against the named container's /tmp/ and annotates each "
            "catalog entry with its live size + mtime."
        ),
    )
    sub = p.add_subparsers(dest="trace_cmd", required=True)

    p_list = sub.add_parser(
        "list",
        help="List every known trace + (with --container) live state.",
    )
    p_list.add_argument(
        "--container", default=None,
        help=(
            "Name of a running vLLM container; introspect its /tmp/ "
            "via `docker exec` and annotate each entry with the live "
            "file size + mtime. Requires docker on $PATH."
        ),
    )
    p_list.add_argument(
        "--category", default=None,
        help=(
            "Filter to a single category. Valid: " +
            "/".join(_categories())
        ),
    )
    p_list.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON instead of the human view.",
    )
    p_list.add_argument(
        "--all", action="store_true",
        help=(
            "With --container: list catalog entries even when the "
            "live file is absent. Default: hide absent files so the "
            "operator only sees what's actually being written."
        ),
    )
    p_list.set_defaults(func=run_list)

    # ── collect (§6.H7) ─────────────────────────────────────────────
    p_collect = sub.add_parser(
        "collect",
        help=(
            "Copy live trace files out of a container into a host "
            "directory."
        ),
        description=(
            "For every TRACE_CATALOG entry that has a live file in "
            "the named container, `docker cp` it to the host. Default "
            "output dir is `./genesis_traces_<container>_<timestamp>/`."
        ),
    )
    p_collect.add_argument(
        "--container", required=True,
        help="Name of a running vLLM container.",
    )
    p_collect.add_argument(
        "--trace", default=None, dest="trace_id",
        help=(
            "Limit collection to a single trace id (e.g. "
            "`pn248_acceptance`). Without this flag, every present "
            "trace is collected."
        ),
    )
    p_collect.add_argument(
        "--output-dir", default=None,
        help=(
            "Host directory to copy traces into. Created if absent. "
            "Default: `./genesis_traces_<container>_<UTC-timestamp>/`."
        ),
    )
    p_collect.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON report instead of human view.",
    )
    p_collect.set_defaults(func=run_collect)

    # ── summarize (§6.H8) ───────────────────────────────────────────
    p_sum = sub.add_parser(
        "summarize",
        help=(
            "Quick stat analysis on a collected trace file (file "
            "size, line count, kind-specific summary)."
        ),
        description=(
            "Reads the named trace file from disk and emits a summary. "
            "Trace kind is auto-detected from the basename (`genesis_"
            "<kind>_*` convention); a per-kind summarizer extracts "
            "the most operator-actionable signals (e.g. boot log → "
            "apply/skip/fail counts + failing patch ids). Always also "
            "emits generic stats (size, line count, first / last line)."
        ),
    )
    p_sum.add_argument(
        "log_file",
        help="Path to a trace file on disk (e.g. one collected by H7).",
    )
    p_sum.add_argument(
        "--max-line-preview", type=int, default=160, dest="max_preview",
        help=(
            "Truncate first / last line previews to this many chars "
            "(default: 160). Set 0 to disable truncation."
        ),
    )
    p_sum.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON instead of human view.",
    )
    p_sum.set_defaults(func=run_summarize)


def add_support_bundle_argparser(subparsers: Any) -> None:
    """Register ``sndr support-bundle`` under the TOP-level parser
    (NOT under ``sndr trace ...``). The plan §6.H9 spec uses the
    flat form; the implementation lives here so it can reuse the
    trace primitives without a cross-module import cycle."""
    p = subparsers.add_parser(
        "support-bundle",
        help=(
            "§6.H9 — bundle all live traces + host/container facts "
            "into a single .tar.gz for off-rig analysis."
        ),
        description=(
            "Composes `sndr trace collect` + `sndr trace summarize` "
            "+ host & container metadata into one tarball ready to "
            "ship off-rig. Output filename defaults to "
            "`genesis_support_<container>_<UTC-timestamp>.tar.gz` "
            "in cwd."
        ),
    )
    p.add_argument(
        "--container", required=True,
        help="Name of a running vLLM container.",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output tarball path. Default: "
            "`./genesis_support_<container>_<UTC-stamp>.tar.gz`."
        ),
    )
    p.add_argument(
        "--no-traces", action="store_true",
        help=(
            "Skip trace collection (host/container facts only). "
            "Useful for a fast summary when traces aren't relevant."
        ),
    )
    p.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON report instead of human view.",
    )
    p.set_defaults(func=run_support_bundle)


# ─── Helpers ──────────────────────────────────────────────────────────


def _categories() -> tuple[str, ...]:
    from sndr.observability.trace_catalog import TRACE_CATEGORIES
    return TRACE_CATEGORIES


def _container_ls_tmp(container: str) -> dict[str, tuple[int, str]]:
    """Run ``docker exec <container> ls -la /tmp/`` and parse the
    matching genesis_* lines into ``{basename: (size_bytes, mtime_str)}``.

    Returns an empty dict on any failure — never raises. Detection
    failure is surfaced via a one-line warning in the CLI, not via
    an exception (the verb must still print the static catalog).
    """
    if not shutil.which("docker"):
        return {}
    try:
        r = subprocess.run(
            ["docker", "exec", container, "ls", "-la", "--time-style=full-iso", "/tmp"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if r.returncode != 0:
        return {}
    out: dict[str, tuple[int, str]] = {}
    # Typical row:
    #   -rw-r--r-- 1 root root  12345  2026-05-30 13:00:00.000000000 +0000 genesis_pn248_acceptance_trace.log
    # ls -la output also includes . / .. / non-matching files; we only
    # care about genesis_* files in /tmp/ root.
    row_re = re.compile(
        r"^\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[^\s]*\s+\S+)\s+"
        r"(genesis_\S+)\s*$"
    )
    for line in r.stdout.splitlines():
        m = row_re.match(line.strip())
        if m is None:
            continue
        size, mtime, name = m.group(1), m.group(2), m.group(3)
        out[name] = (int(size), mtime)
    return out


def _spec_to_dict(spec, live: Optional[tuple[int, str]]) -> dict:
    d = asdict(spec)
    if live is None:
        d["live"] = None
    else:
        size, mtime = live
        d["live"] = {"size_bytes": size, "mtime": mtime}
    return d


def _fmt_size(n: int) -> str:
    """Human-friendly size — 1024-based, single decimal for >1KB."""
    if n < 1024:
        return f"{n}B"
    for unit in ("KB", "MB", "GB"):
        n_f = n / 1024.0
        if n_f < 1024 or unit == "GB":
            return f"{n_f:.1f}{unit}"
        n //= 1024
    return f"{n}B"  # unreachable


def _docker_cp(container: str, src: str, dst: str) -> tuple[bool, str]:
    """Run ``docker cp <container>:<src> <dst>``.

    Returns ``(ok, message)``. Failures (no docker, container stopped,
    file missing, timeout) become ``(False, <reason>)`` — never raise.
    """
    if not shutil.which("docker"):
        return False, "docker binary not on $PATH"
    try:
        r = subprocess.run(
            ["docker", "cp", f"{container}:{src}", dst],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker cp raised: {exc}"
    if r.returncode != 0:
        # docker cp prints the error on stderr.
        return False, (r.stderr or r.stdout).strip().splitlines()[-1]
    return True, "ok"


def _default_output_dir(container: str) -> str:
    """`./genesis_traces_<container>_<UTC-stamp>/`.

    UTC stamp avoids cross-timezone confusion when bundles travel
    between operators / rigs.
    """
    import datetime
    stamp = datetime.datetime.now(
        datetime.timezone.utc,
    ).strftime("%Y%m%dT%H%M%SZ")
    # Sanitize container name — replace path separators with `-` so the
    # directory name is always safe.
    safe = container.replace("/", "-").replace("\\", "-")
    return f"./genesis_traces_{safe}_{stamp}"


# ─── Runner ───────────────────────────────────────────────────────────


def run_list(args: argparse.Namespace) -> int:
    from sndr.observability.trace_catalog import (
        TRACE_CATALOG, TRACE_CATEGORIES, iter_by_category,
    )

    if args.category and args.category not in TRACE_CATEGORIES:
        _io.error(
            f"unknown category {args.category!r}. "
            f"Valid: {', '.join(TRACE_CATEGORIES)}"
        )
        return 2

    live_state: dict[str, tuple[int, str]] = {}
    container_warning: Optional[str] = None
    if args.container:
        live_state = _container_ls_tmp(args.container)
        if not live_state and not shutil.which("docker"):
            container_warning = (
                "docker binary not on $PATH — --container ignored, "
                "showing static catalog only."
            )
        elif not live_state:
            container_warning = (
                f"docker exec {args.container} ls /tmp returned no "
                "genesis_* files. Either the container is stopped, the "
                "name is wrong, or no trace flag is enabled."
            )

    grouped = iter_by_category()

    # Filter by --category if requested.
    if args.category:
        grouped = {args.category: grouped[args.category]}

    # When --container is set and --all is NOT, drop catalog entries
    # whose live file is absent (operator wants to see "what's being
    # written right now", not "what could be written").
    if args.container and not args.all:
        new_grouped: dict[str, tuple] = {}
        for cat, specs in grouped.items():
            kept = tuple(
                s for s in specs
                if s.container_path.split("/")[-1] in live_state
            )
            if kept:
                new_grouped[cat] = kept
        grouped = new_grouped

    # ── JSON output ──────────────────────────────────────────────────
    if args.json:
        out_specs = []
        for cat in TRACE_CATEGORIES:
            for s in grouped.get(cat, ()):
                live = live_state.get(s.container_path.split("/")[-1])
                out_specs.append(_spec_to_dict(s, live))
        out_payload = {
            "container": args.container,
            "container_warning": container_warning,
            "category_filter": args.category,
            "all": bool(args.all),
            "total": len(out_specs),
            "traces": out_specs,
        }
        print(json.dumps(out_payload, indent=2, sort_keys=False))
        return 0

    # ── Human view ──────────────────────────────────────────────────
    if container_warning:
        _io.warn(container_warning)

    total_listed = sum(len(v) for v in grouped.values())
    if total_listed == 0:
        if args.container and not args.all:
            print(
                f"No live genesis_* traces in {args.container}:/tmp/. "
                "Pass --all to see the full catalog."
            )
        else:
            print("(no traces match filter)")
        return 0

    print("sndr trace — known diagnostic traces"
          + (f" (container={args.container})" if args.container else ""))
    print("─" * 70)
    for cat in TRACE_CATEGORIES:
        specs = grouped.get(cat, ())
        if not specs:
            continue
        print(f"\n  {cat.upper()}")
        for s in specs:
            basename = s.container_path.split("/")[-1]
            live = live_state.get(basename)
            live_suffix = ""
            if args.container:
                if live is None:
                    live_suffix = "  (not present)"
                else:
                    size, mtime = live
                    live_suffix = f"  [{_fmt_size(size)}, {mtime}]"
            env_suffix = (
                f"  env={s.enable_env}" if s.enable_env else "  (always)"
            )
            print(f"    · {s.id:24s}{env_suffix}{live_suffix}")
            print(f"      patch={s.patch_id}")
            print(f"      path={s.container_path}")
            # Re-wrap description at 72 chars for readability.
            desc = s.description.strip()
            indent = "      "
            wrapped = []
            line = ""
            for word in desc.split():
                if len(line) + len(word) + 1 > 72 and line:
                    wrapped.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                wrapped.append(line)
            for w in wrapped:
                print(f"{indent}{w}")

    print()
    print(f"  Total: {total_listed} trace(s).")
    return 0


# ─── run_collect (§6.H7) ──────────────────────────────────────────────


def run_collect(args: argparse.Namespace) -> int:
    """Copy every present trace from a container into a host directory."""
    import os
    from sndr.observability.trace_catalog import (
        TRACE_CATALOG, find_by_id,
    )

    # Resolve which traces to attempt.
    if args.trace_id:
        spec = find_by_id(args.trace_id)
        if spec is None:
            _io.error(
                f"unknown trace id {args.trace_id!r}. "
                f"Run `sndr trace list` to see valid ids."
            )
            return 2
        wanted = (spec,)
    else:
        wanted = TRACE_CATALOG

    # Snapshot live state once so we know what's actually present.
    live_state = _container_ls_tmp(args.container)
    if not live_state and not shutil.which("docker"):
        _io.error(
            "docker binary not on $PATH — `sndr trace collect` needs "
            "docker to run `docker exec` + `docker cp`."
        )
        return 3
    if not live_state:
        # Either container stopped / wrong name / nothing being written.
        if args.json:
            print(json.dumps({
                "container": args.container,
                "output_dir": None,
                "collected": [],
                "skipped": [{"id": s.id, "reason": "not present"}
                            for s in wanted],
                "errors": [],
            }, indent=2))
        else:
            _io.warn(
                f"docker exec {args.container} ls /tmp returned no "
                "genesis_* files — nothing to collect."
            )
        return 0

    # Determine output dir.
    output_dir = args.output_dir or _default_output_dir(args.container)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        _io.error(f"could not create output dir {output_dir!r}: {exc}")
        return 4

    collected: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []

    for spec in wanted:
        basename = spec.container_path.split("/")[-1]
        live = live_state.get(basename)
        if live is None:
            skipped.append({
                "id": spec.id, "container_path": spec.container_path,
                "reason": "not present in container",
            })
            continue
        dst = os.path.join(output_dir, basename)
        ok, msg = _docker_cp(
            container=args.container,
            src=spec.container_path,
            dst=dst,
        )
        if not ok:
            errors.append({
                "id": spec.id, "container_path": spec.container_path,
                "dst": dst, "error": msg,
            })
            continue
        size_bytes, mtime = live
        collected.append({
            "id": spec.id,
            "patch_id": spec.patch_id,
            "container_path": spec.container_path,
            "dst": dst,
            "size_bytes": size_bytes,
            "mtime": mtime,
        })

    # ── JSON report ─────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "container": args.container,
            "output_dir": output_dir,
            "collected": collected,
            "skipped": skipped,
            "errors": errors,
        }, indent=2))
        return 0 if not errors else 1

    # ── Human view ─────────────────────────────────────────────────
    print(f"sndr trace collect — {args.container} → {output_dir}")
    print("─" * 70)
    if collected:
        print(f"\n  Collected ({len(collected)}):")
        for c in collected:
            print(f"    ✓ {c['id']:24s}  {_fmt_size(c['size_bytes'])}")
            print(f"        from: {c['container_path']}")
            print(f"        to:   {c['dst']}")
    if skipped:
        print(f"\n  Skipped — not present ({len(skipped)}):")
        for s in skipped:
            print(f"    · {s['id']:24s}  ({s['reason']})")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e['id']:24s}  {e['error']}")

    print()
    if errors:
        print(f"  Done with errors: {len(collected)} ok, "
              f"{len(errors)} failed.")
        return 1
    print(f"  Done: {len(collected)} trace(s) collected.")
    return 0


# ─── Summarize (§6.H8) ───────────────────────────────────────────────


# Maps basename pattern → trace kind. Order matters: more-specific
# substrings first so e.g. `genesis_pn258_oracle_trace.log` resolves
# to `oracle` rather than the generic `unknown`.
_KIND_BY_SUBSTRING: tuple[tuple[str, str], ...] = (
    ("genesis_boot", "boot"),
    ("genesis_pn248_acceptance", "acceptance"),
    ("genesis_pn258_oracle", "oracle"),
    ("genesis_pn260_kernel", "kernel"),
    ("genesis_pn254_fire", "kernel"),
    ("genesis_pn255_kv_write", "kv_write"),
    ("genesis_pn256_route", "routing"),
    ("genesis_pn261_tq_impl_init", "routing"),
    ("genesis_pn241_mtp", "mtp"),
    ("genesis_tq_forward", "tq_forward"),
)


def detect_trace_kind(path: str) -> str:
    """Look up the trace kind for a filename.

    Returns one of TRACE_CATEGORIES, or ``"unknown"`` if no substring
    matches. The lookup is on basename only — full host paths work too.
    """
    import os
    base = os.path.basename(path).lower()
    for substr, kind in _KIND_BY_SUBSTRING:
        if substr in base:
            return kind
    return "unknown"


def _read_lines_streaming(path: str, max_lines_first: int = 1,
                          max_lines_last: int = 1) -> tuple[int, list[str], list[str]]:
    """Single-pass file reader for size-conscious files.

    Returns ``(total_line_count, first_lines, last_lines)``. Uses a
    sliding window for last_lines so the function works on multi-GB
    files without loading them into memory. The ``first_lines`` /
    ``last_lines`` lists are stripped of trailing newlines.
    """
    from collections import deque
    first: list[str] = []
    last: deque[str] = deque(maxlen=max(1, max_lines_last))
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            if len(first) < max_lines_first:
                first.append(line.rstrip("\n"))
            last.append(line.rstrip("\n"))
    return total, first, list(last)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…"


def summarize_generic(path: str, max_preview: int = 160) -> dict:
    """Generic kind-agnostic summary: size, total lines, first / last
    line preview."""
    import os
    size = os.path.getsize(path)
    total, first, last = _read_lines_streaming(
        path, max_lines_first=1, max_lines_last=1,
    )
    first_line = first[0] if first else ""
    last_line = last[-1] if last else ""
    return {
        "size_bytes": size,
        "total_lines": total,
        "first_line": _truncate(first_line, max_preview),
        "last_line": _truncate(last_line, max_preview),
    }


# Pattern for the boot-log entries written by `python3 -m
# sndr.apply`. Each line ends with `<PATCH_ID>: <STATUS>`
# (status ∈ {applied, skipped, failed}); flexible whitespace and
# optional bracketed prefix `[Genesis]` tolerated.
_BOOT_ENTRY_RE = re.compile(
    r"(?:\[Genesis\][^\w]*)?"
    r"\b(?P<patch_id>[A-Z][A-Za-z0-9_]*)\b"
    r"\s*[:=]\s*"
    r"(?P<status>applied|skipped|failed|ok|error)"
    r"\b",
    re.IGNORECASE,
)


def summarize_boot_log(path: str, max_preview: int = 160) -> dict:
    """Per-kind summary for `genesis_boot.log`: counts of apply /
    skip / fail outcomes + list of failing patches.

    The boot log is the most operator-actionable trace because a single
    failed apply often explains why a patch didn't fire in PROD. We
    surface the failing patch ids verbatim so the operator can grep
    them straight away.
    """
    base = summarize_generic(path, max_preview=max_preview)
    counts = {"applied": 0, "skipped": 0, "failed": 0,
              "other": 0}
    failed_patches: list[str] = []
    skipped_patches: list[str] = []
    applied_patches: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _BOOT_ENTRY_RE.search(line)
            if m is None:
                continue
            status = m.group("status").lower()
            pid = m.group("patch_id")
            # Normalise the few synonyms that some patches emit.
            if status in ("ok", "applied"):
                counts["applied"] += 1
                applied_patches.append(pid)
            elif status == "skipped":
                counts["skipped"] += 1
                skipped_patches.append(pid)
            elif status in ("failed", "error"):
                counts["failed"] += 1
                failed_patches.append(pid)
            else:
                counts["other"] += 1
    base["kind"] = "boot"
    base["counts"] = counts
    base["failed_patches"] = failed_patches
    base["skipped_patches"] = skipped_patches[:50]  # cap for readability
    base["applied_patches"] = applied_patches[:50]
    return base


# PN248 trace format anchors. See
# sndr/engines/vllm/patches/spec_decode/probes/pn248_acceptance_trace.py
# for the producer side; format is stable across patch versions.
#
# ENTER:   `[PN248 call=<N>] ENTER ... num_draft_tokens=[<a>, <b>, ...] ...`
# EXIT:    `[PN248 call=<N>] EXIT  ... accepted_per_req=[<a>, <b>, ...]`
# err:     `[PN248 call=<N>] ENTER err=<exc>: <msg>`  (or EXIT err=...)
_PN248_NUM_DRAFTS_RE = re.compile(r"num_draft_tokens=\[([^\]]*)\]")
_PN248_ACCEPTED_RE = re.compile(r"accepted_per_req=\[([^\]]*)\]")
_PN248_CALL_ID_RE = re.compile(r"\[PN248 call=(\d+)\]")
_PN248_ERR_RE = re.compile(r"\[PN248 call=\d+\]\s+(ENTER|EXIT)\s+err=")


def _parse_int_list(raw: str) -> list[int]:
    """Parse a `[1, 2, 3]` inner string into a list of ints.

    Tolerates trailing commas + arbitrary whitespace. Returns an empty
    list on any parse failure — the summarizer must never crash on a
    partially-formed log line."""
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out


def summarize_pn248_acceptance(
    path: str, max_preview: int = 160,
) -> dict:
    """Per-kind summary for `genesis_pn248_acceptance_trace.log`.

    Single streaming pass extracts:
      * total_calls      — distinct `[PN248 call=<N>]` ids observed
      * total_drafts_proposed   — Σ num_draft_tokens across all ENTER lines
      * total_drafts_accepted   — Σ accepted_per_req across all EXIT lines
      * acceptance_rate         — accepted / proposed (None if proposed=0)
      * acceptance_histogram    — {accept_count: line_count} from per-req
                                  accepted_per_req values
      * mean_accepted_per_request — float; None if no EXIT lines parsed
      * capture_errors          — count of `ENTER err=` / `EXIT err=` lines
                                  (PN248 records its own probe failures)

    A high `capture_errors` count signals the wrap itself is misfiring
    — the operator should investigate the producer before trusting the
    rate figures.
    """
    base = summarize_generic(path, max_preview=max_preview)
    total_proposed = 0
    total_accepted = 0
    call_ids: set[int] = set()
    histogram: dict[int, int] = {}
    capture_errors = 0
    req_lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Track distinct call ids (gives total_calls without
            # double-counting ENTER+EXIT for the same step).
            m = _PN248_CALL_ID_RE.search(line)
            if m is not None:
                call_ids.add(int(m.group(1)))

            if _PN248_ERR_RE.search(line):
                capture_errors += 1
                continue

            m_drafts = _PN248_NUM_DRAFTS_RE.search(line)
            if m_drafts is not None:
                total_proposed += sum(_parse_int_list(m_drafts.group(1)))
                continue

            m_acc = _PN248_ACCEPTED_RE.search(line)
            if m_acc is not None:
                per_req = _parse_int_list(m_acc.group(1))
                total_accepted += sum(per_req)
                req_lines += len(per_req)
                for v in per_req:
                    histogram[v] = histogram.get(v, 0) + 1

    base["kind"] = "acceptance"
    base["total_calls"] = len(call_ids)
    base["total_drafts_proposed"] = total_proposed
    base["total_drafts_accepted"] = total_accepted
    base["acceptance_rate"] = (
        total_accepted / total_proposed if total_proposed > 0 else None
    )
    base["mean_accepted_per_request"] = (
        total_accepted / req_lines if req_lines > 0 else None
    )
    # Sort histogram by accept count for stable output.
    base["acceptance_histogram"] = dict(sorted(histogram.items()))
    base["capture_errors"] = capture_errors
    return base


def run_summarize(args: argparse.Namespace) -> int:
    import os
    path = args.log_file
    if not os.path.isfile(path):
        _io.error(f"trace file {path!r} does not exist (or not a file).")
        return 2

    kind = detect_trace_kind(path)
    max_preview = args.max_preview if args.max_preview is not None else 160

    try:
        if kind == "boot":
            summary = summarize_boot_log(path, max_preview=max_preview)
        elif kind == "acceptance":
            summary = summarize_pn248_acceptance(
                path, max_preview=max_preview,
            )
        else:
            summary = summarize_generic(path, max_preview=max_preview)
            summary["kind"] = kind
    except OSError as exc:
        _io.error(f"could not read {path!r}: {exc}")
        return 3

    if args.json:
        out = {"path": path, "summary": summary}
        print(json.dumps(out, indent=2))
        return 0

    print(f"sndr trace summarize — {path}")
    print("─" * 70)
    print(f"  kind:        {summary['kind']}")
    print(f"  size:        {_fmt_size(summary['size_bytes'])} "
          f"({summary['size_bytes']:,} bytes)")
    print(f"  total lines: {summary['total_lines']:,}")
    print(f"  first line:  {summary['first_line']!r}")
    print(f"  last line:   {summary['last_line']!r}")

    if summary["kind"] == "boot":
        c = summary["counts"]
        print()
        print("  Boot apply summary:")
        print(f"    applied  = {c['applied']}")
        print(f"    skipped  = {c['skipped']}")
        print(f"    failed   = {c['failed']}")
        if c["other"]:
            print(f"    other    = {c['other']}")
        if summary["failed_patches"]:
            print()
            print(f"  Failed patches ({len(summary['failed_patches'])}):")
            for pid in summary["failed_patches"]:
                print(f"    ✗ {pid}")

    elif summary["kind"] == "acceptance":
        print()
        print("  PN248 acceptance summary:")
        print(f"    total decode calls:   {summary['total_calls']:,}")
        print(f"    drafts proposed:      "
              f"{summary['total_drafts_proposed']:,}")
        print(f"    drafts accepted:      "
              f"{summary['total_drafts_accepted']:,}")
        rate = summary["acceptance_rate"]
        if rate is None:
            print("    acceptance rate:      (no drafts proposed)")
        else:
            print(f"    acceptance rate:      {rate:.3%}")
        mean = summary["mean_accepted_per_request"]
        if mean is None:
            print("    mean accepted/req:    (no EXIT lines parsed)")
        else:
            print(f"    mean accepted/req:    {mean:.2f}")
        if summary["capture_errors"]:
            print(f"    ⚠ capture errors:     {summary['capture_errors']} "
                  "(probe itself misfired — investigate producer)")
        if summary["acceptance_histogram"]:
            print()
            print("  Acceptance histogram (accepted-per-request → "
                  "line count):")
            for accept_count, n in summary["acceptance_histogram"].items():
                bar = "█" * min(40, n)
                print(f"    {accept_count:>3d}: {n:>7,d}  {bar}")

    return 0


# ─── Support bundle (§6.H9) ──────────────────────────────────────────


def _run_subprocess_capture(
    argv: list[str], timeout: int = 10,
) -> tuple[bool, str]:
    """Helper: capture stdout of a host command. Returns (ok, output);
    failure yields (False, "<reason>"). Never raises."""
    if not shutil.which(argv[0]):
        return False, f"{argv[0]} not on $PATH"
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{argv[0]} raised: {exc}"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip()
    return True, r.stdout


def collect_host_facts() -> dict[str, dict]:
    """Best-effort gather of host-side diagnostic facts.

    Each fact is a dict ``{"ok": bool, "output": str}``. Missing
    binaries / failures surface as ``{"ok": False, "output": "<reason>"}``
    rather than raising — the bundle must always be produceable.
    """
    out: dict[str, dict] = {}
    facts = {
        "uname": ["uname", "-a"],
        "free_disk": ["df", "-h"],
        "free_memory": ["free", "-h"],
        "nvidia_smi": ["nvidia-smi"],
        "docker_version": ["docker", "--version"],
    }
    for key, argv in facts.items():
        ok, output = _run_subprocess_capture(argv, timeout=10)
        out[key] = {"ok": ok, "output": output}
    return out


def collect_container_facts(container: str) -> dict[str, dict]:
    """Best-effort gather of container-side metadata.

    Pulls:
      * Image name + digest (via ``docker inspect --format``).
      * Env block (filtered to GENESIS_* / VLLM_* / NCCL_* / CUDA_*).
      * Process list inside the container (``ps -ef``).
      * Recent container logs tail (last 200 lines).
    """
    out: dict[str, dict] = {}

    # Image + digest.
    ok, img = _run_subprocess_capture(
        ["docker", "inspect", container,
         "--format", "{{.Config.Image}}\n{{.Image}}"],
        timeout=10,
    )
    out["image"] = {"ok": ok, "output": img}

    # Env block — filter to genesis-relevant prefixes.
    ok, env_raw = _run_subprocess_capture(
        ["docker", "inspect", container,
         "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
        timeout=10,
    )
    if ok:
        wanted_prefixes = ("GENESIS_", "SNDR_", "VLLM_", "NCCL_",
                            "CUDA_", "TORCH_", "PYTORCH_", "TRITON_")
        filtered = "\n".join(
            line for line in env_raw.splitlines()
            if any(line.startswith(p) for p in wanted_prefixes)
        )
        out["env_filtered"] = {"ok": True, "output": filtered}
    else:
        out["env_filtered"] = {"ok": False, "output": env_raw}

    # Process list inside container.
    ok, ps = _run_subprocess_capture(
        ["docker", "exec", container, "ps", "-ef"], timeout=10,
    )
    out["processes"] = {"ok": ok, "output": ps}

    # Tail of container logs (last 200 lines).
    ok, log_tail = _run_subprocess_capture(
        ["docker", "logs", "--tail", "200", container], timeout=20,
    )
    out["logs_tail"] = {"ok": ok, "output": log_tail}

    return out


def _default_bundle_path(container: str) -> str:
    import datetime
    stamp = datetime.datetime.now(
        datetime.timezone.utc,
    ).strftime("%Y%m%dT%H%M%SZ")
    safe = container.replace("/", "-").replace("\\", "-")
    return f"./genesis_support_{safe}_{stamp}.tar.gz"


def run_support_bundle(args: argparse.Namespace) -> int:
    """Build the §6.H9 support bundle."""
    import os
    import tarfile
    import tempfile
    from sndr.observability.trace_catalog import TRACE_CATALOG

    if not args.no_traces and not shutil.which("docker"):
        _io.error(
            "docker binary not on $PATH — support-bundle needs docker "
            "for trace collection. Re-run with --no-traces to skip "
            "traces and bundle host facts only."
        )
        return 3

    output_path = args.output or _default_bundle_path(args.container)

    # Build the bundle inside a tempdir, then tar.gz to output_path.
    collected_count = 0
    failed_collects: list[dict] = []
    summary_count = 0
    with tempfile.TemporaryDirectory() as workdir:
        # 1. Trace collection (H7 reuse).
        traces_dir = os.path.join(workdir, "traces")
        summaries_dir = os.path.join(workdir, "summaries")
        if not args.no_traces:
            os.makedirs(traces_dir, exist_ok=True)
            os.makedirs(summaries_dir, exist_ok=True)
            live_state = _container_ls_tmp(args.container)
            for spec in TRACE_CATALOG:
                basename = spec.container_path.split("/")[-1]
                if basename not in live_state:
                    continue
                dst = os.path.join(traces_dir, basename)
                ok, msg = _docker_cp(
                    container=args.container,
                    src=spec.container_path, dst=dst,
                )
                if not ok:
                    failed_collects.append({
                        "id": spec.id, "error": msg,
                    })
                    continue
                collected_count += 1
                # 2. Auto-summarize each collected trace (H8 reuse).
                try:
                    kind = detect_trace_kind(basename)
                    if kind == "boot":
                        summary = summarize_boot_log(dst)
                    elif kind == "acceptance":
                        summary = summarize_pn248_acceptance(dst)
                    else:
                        summary = summarize_generic(dst)
                        summary["kind"] = kind
                    summary_path = os.path.join(
                        summaries_dir, f"{spec.id}.json",
                    )
                    with open(summary_path, "w",
                              encoding="utf-8") as fh:
                        json.dump(
                            {"id": spec.id, "patch_id": spec.patch_id,
                             "summary": summary},
                            fh, indent=2,
                        )
                    summary_count += 1
                except Exception as exc:
                    failed_collects.append({
                        "id": spec.id,
                        "error": f"summarize raised: {exc}",
                    })

        # 3. Host facts.
        host_dir = os.path.join(workdir, "host")
        os.makedirs(host_dir, exist_ok=True)
        host_facts = collect_host_facts()
        for key, fact in host_facts.items():
            with open(os.path.join(host_dir, f"{key}.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(f"# ok={fact['ok']}\n")
                fh.write(fact["output"])

        # 4. Container facts.
        container_dir = os.path.join(workdir, "container")
        os.makedirs(container_dir, exist_ok=True)
        container_facts: dict[str, dict] = {}
        if shutil.which("docker"):
            container_facts = collect_container_facts(args.container)
            for key, fact in container_facts.items():
                with open(os.path.join(container_dir, f"{key}.txt"),
                          "w", encoding="utf-8") as fh:
                    fh.write(f"# ok={fact['ok']}\n")
                    fh.write(fact["output"])

        # 5. Manifest.
        import datetime
        manifest = {
            "container": args.container,
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trace_collection": {
                "skipped": bool(args.no_traces),
                "collected_count": collected_count,
                "summarized_count": summary_count,
                "failed": failed_collects,
            },
            "host_facts": list(host_facts.keys()),
            "container_facts": list(container_facts.keys()),
        }
        with open(os.path.join(workdir, "manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        # 6. Tar.gz the workdir → output_path.
        try:
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(workdir, arcname="genesis_support_bundle")
        except OSError as exc:
            _io.error(f"could not write {output_path!r}: {exc}")
            return 4

    # ── Report ─────────────────────────────────────────────────────
    final_size = os.path.getsize(output_path)
    if args.json:
        print(json.dumps({
            "container": args.container,
            "output": output_path,
            "size_bytes": final_size,
            "trace_collection": {
                "skipped": bool(args.no_traces),
                "collected": collected_count,
                "summarized": summary_count,
                "failed": failed_collects,
            },
            "host_facts_collected": list(host_facts.keys()),
            "container_facts_collected": list(container_facts.keys()),
        }, indent=2))
        return 0 if not failed_collects else 1

    print(f"sndr support-bundle → {output_path}")
    print("─" * 70)
    print(f"  container:       {args.container}")
    print(f"  bundle size:     {_fmt_size(final_size)} "
          f"({final_size:,} bytes)")
    if args.no_traces:
        print("  traces:          skipped (--no-traces)")
    else:
        print(f"  traces collected: {collected_count}")
        print(f"  traces summarized: {summary_count}")
    print(f"  host facts:      {len(host_facts)} captured")
    print(f"  container facts: {len(container_facts)} captured")
    if failed_collects:
        print()
        print(f"  Failures ({len(failed_collects)}):")
        for f in failed_collects:
            print(f"    ✗ {f['id']:24s}  {f['error']}")
    return 0 if not failed_collects else 1
