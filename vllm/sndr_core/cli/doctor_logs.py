# SPDX-License-Identifier: Apache-2.0
"""Host log forensics helper for `sndr doctor-system --logs`.

Audit P2-1 closure (2026-05-12): the operator must have a fast way
to check whether the last N hours show signs of hardware / runtime
instability — without having to manually dig through dmesg/journalctl/docker.

What we collect (read-only, no actions):

  • dmesg | OOM-kills (`Out of memory: Killed process …`)
  • dmesg | NVRM Xid errors (`NVRM: Xid (PCI:…): <code>` —
    14/31/43/45/79 are especially alarming — fatal MMU/ECC/timeout).
  • docker ps --filter status=restarting (containers in a restart loop).
  • journalctl -u genesis-vllm.service (if the systemd unit is installed) —
    the last 20 lines containing the words error|fatal|oom|cuda.
  • Recent dmesg warnings/errors filtered by the last N hours.

Design:

  • All sources are optional — absence of dmesg/journalctl/docker returns
    an empty list with a reason, does not fail.
  • Does not try to replace full observability (Prometheus/Grafana) — this
    is a fast triage tool for an operator session.
  • JSON-output is compatible with the `--json` mode of doctor-system: returns
    a dict that is placed into `facts["log_forensics"]`.

Test contract — `tests/unit/cli/test_doctor_logs.py`:

  • Mock subprocess.run → correct shape of returned dicts.
  • OOM regex matches typical dmesg lines.
  • Xid severity categorization (fatal vs warning vs info).
  • Graceful when binaries missing.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field


__all__ = [
    "LogForensicsResult",
    "OomEvent",
    "XidEvent",
    "RestartingContainer",
    "collect_log_forensics",
    "summarize_for_text",
    "FATAL_XIDS",
]


# Xid codes that should put the operator on serious alert.
# Source: NVIDIA "Understanding Xid Errors" tech brief + field experience.
#   13  — Graphics Engine Exception (often recoverable, but repeats → bad GPU)
#   14  — Display Engine Error
#   31  — GPU memory page fault (MMU; almost always a runtime bug or ECC)
#   43  — Reset Channel (CUDA program killed/hung)
#   45  — Preemptive cleanup, due to previous errors (consequence of 31/43)
#   62  — Internal micro-controller breakpoint/warning
#   63  — ECC page retirement (memory degradation)
#   64  — ECC page retirement recording failure
#   74  — NVLink error
#   79  — GPU fell off the bus (HARD failure — PCIe link lost)
#   119 — GSP RPC timeout (driver/firmware compat issue, usually non-fatal)
FATAL_XIDS: frozenset[int] = frozenset({31, 43, 45, 63, 64, 74, 79})


@dataclass(frozen=True)
class OomEvent:
    """OOM-kill event from dmesg."""
    timestamp_seconds_ago: int | None  # None if dmesg did not give uptime
    killed_process: str  # name of the process that the kernel killed
    raw_line: str  # for troubleshooting

    def to_dict(self) -> dict:
        return {
            "timestamp_seconds_ago": self.timestamp_seconds_ago,
            "killed_process": self.killed_process,
            "raw_line": self.raw_line,
        }


@dataclass(frozen=True)
class XidEvent:
    """NVRM Xid error from dmesg."""
    timestamp_seconds_ago: int | None
    xid_code: int
    pci_addr: str  # e.g. "PCI:0000:01:00"
    severity: str  # "fatal" | "warning" | "info"
    raw_line: str

    def to_dict(self) -> dict:
        return {
            "timestamp_seconds_ago": self.timestamp_seconds_ago,
            "xid_code": self.xid_code,
            "pci_addr": self.pci_addr,
            "severity": self.severity,
            "raw_line": self.raw_line,
        }


@dataclass(frozen=True)
class RestartingContainer:
    """Container in a restart loop."""
    name: str
    image: str
    status: str  # "Restarting (X) Y ago"
    started_at: str  # iso-8601 timestamp if available

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image": self.image,
            "status": self.status,
            "started_at": self.started_at,
        }


@dataclass
class LogForensicsResult:
    """Composite result of host log forensics."""
    window_hours: int
    oom_events: list[OomEvent] = field(default_factory=list)
    xid_events: list[XidEvent] = field(default_factory=list)
    restarting_containers: list[RestartingContainer] = field(default_factory=list)
    service_journal_tail: list[str] = field(default_factory=list)
    sources_unavailable: list[str] = field(default_factory=list)

    @property
    def has_fatal_signals(self) -> bool:
        """True if there is a fatal Xid or OOM within the window."""
        return (
            any(x.severity == "fatal" for x in self.xid_events)
            or bool(self.oom_events)
            or bool(self.restarting_containers)
        )

    def to_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "oom_events": [e.to_dict() for e in self.oom_events],
            "xid_events": [e.to_dict() for e in self.xid_events],
            "restarting_containers": [
                c.to_dict() for c in self.restarting_containers
            ],
            "service_journal_tail": list(self.service_journal_tail),
            "sources_unavailable": list(self.sources_unavailable),
            "has_fatal_signals": self.has_fatal_signals,
        }


# ──── dmesg parsing ─────────────────────────────────────────────────────

# Line format:
#   [12345.678] Out of memory: Killed process 12345 (vllm) total-vm:...
# Modern dmesg --ctime gives a human-readable time; we parse both forms.
_OOM_RE = re.compile(
    r"Out of memory:\s+Killed process\s+\d+\s+\(([^)]+)\)",
    re.IGNORECASE,
)

# Xid line format (NVRM driver):
#   [12345.678] NVRM: Xid (PCI:0000:01:00): 31, pid=12345, name=python ...
# or with --ctime:
#   Mon May 12 03:14:15 host kernel: NVRM: Xid (PCI:0000:01:00): 31, ...
_XID_RE = re.compile(
    r"NVRM:\s+Xid\s+\((PCI:[0-9a-fA-F:.]+)\):\s+(\d+)",
)

# Uptime-relative timestamps `[12345.678]` — seconds since boot.
_UPTIME_TS_RE = re.compile(r"^\[\s*(\d+)\.\d+\]")


def _read_dmesg(extra_args: list[str] | None = None) -> tuple[str, str | None]:
    """Runs dmesg, returns (stdout, reason_if_unavailable).

    Etap 3.1 (audit 2026-05-12): prefer plain dmesg over `--ctime`.
    Our timestamp parser handles only the uptime-relative `[12345.678]`
    prefix produced by the default; ctime lines result in `ts=None` for
    every event, which bypasses `--logs-hours` filtering. We fall back
    to `--ctime` only if plain dmesg fails (rare).

    Returns ("", reason) if dmesg is missing / no permissions / fails.
    """
    if not shutil.which("dmesg"):
        return "", "dmesg binary not found"
    base = ["dmesg"]
    if extra_args:
        base.extend(extra_args)
    for cmd in (base, base + ["--ctime"]):
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
                check=False,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout, None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    return "", "dmesg unavailable (permission denied? try sudo)"


def _parse_oom_events(dmesg_output: str) -> list[OomEvent]:
    events: list[OomEvent] = []
    for line in dmesg_output.splitlines():
        m = _OOM_RE.search(line)
        if not m:
            continue
        proc = m.group(1)
        ts = None
        u = _UPTIME_TS_RE.match(line)
        if u:
            # uptime seconds = how long ago from boot. Convert to
            # "seconds ago" via current uptime if available.
            try:
                with open("/proc/uptime", "r") as f:
                    now_uptime = float(f.read().split()[0])
                ts = int(now_uptime - float(u.group(1)))
                if ts < 0:
                    ts = None
            except Exception:
                ts = None
        events.append(OomEvent(
            timestamp_seconds_ago=ts,
            killed_process=proc,
            raw_line=line.strip(),
        ))
    return events


def _classify_xid(code: int) -> str:
    if code in FATAL_XIDS:
        return "fatal"
    # 13/14/62/119 — repeats may indicate a problem, but single occurrences are usually ok.
    if code in {13, 14, 62, 119}:
        return "warning"
    return "info"


def _parse_xid_events(dmesg_output: str) -> list[XidEvent]:
    events: list[XidEvent] = []
    for line in dmesg_output.splitlines():
        m = _XID_RE.search(line)
        if not m:
            continue
        pci, code_str = m.group(1), m.group(2)
        try:
            code = int(code_str)
        except ValueError:
            continue
        ts = None
        u = _UPTIME_TS_RE.match(line)
        if u:
            try:
                with open("/proc/uptime", "r") as f:
                    now_uptime = float(f.read().split()[0])
                ts = int(now_uptime - float(u.group(1)))
                if ts < 0:
                    ts = None
            except Exception:
                ts = None
        events.append(XidEvent(
            timestamp_seconds_ago=ts,
            xid_code=code,
            pci_addr=pci,
            severity=_classify_xid(code),
            raw_line=line.strip(),
        ))
    return events


def _filter_within_window(
    events: list, window_seconds: int, *, strict: bool = False,
) -> list:
    """Keeps events whose ts_ago fits the window.

    Etap 3.3 (audit 2026-05-12): added `strict` mode. By default events
    with unknown timestamp (parser couldn't decode) are kept — visible
    to the operator. In strict mode they're dropped so a long-uptime
    host doesn't surface very old events as "recent".
    """
    out = []
    for e in events:
        ts = e.timestamp_seconds_ago
        if ts is None:
            if not strict:
                out.append(e)
            continue
        if ts <= window_seconds:
            out.append(e)
    return out


# ──── docker / containers ────────────────────────────────────────────────

def _collect_restarting_containers() -> tuple[list[RestartingContainer], str | None]:
    """`docker ps --filter status=restarting --format=...`.

    Returns ([], reason) if docker is not installed / daemon unavailable.
    """
    if not shutil.which("docker"):
        return [], "docker binary not found"
    try:
        # Format: name|image|status|startedat
        r = subprocess.run(
            ["docker", "ps", "--filter", "status=restarting",
             "--format", "{{.Names}}|{{.Image}}|{{.Status}}|{{.RunningFor}}"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return [], f"docker ps failed: {e!r}"
    if r.returncode != 0:
        msg = r.stderr.strip() or "non-zero exit"
        return [], f"docker ps returned {r.returncode}: {msg[:80]}"
    out: list[RestartingContainer] = []
    for line in r.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        name, image, status, started = parts
        out.append(RestartingContainer(
            name=name.strip(),
            image=image.strip(),
            status=status.strip(),
            started_at=started.strip(),
        ))
    return out, None


# ──── journalctl (genesis-vllm.service) ─────────────────────────────────

def _collect_service_journal(unit: str, hours: int) -> tuple[list[str], str | None]:
    if not shutil.which("journalctl"):
        return [], "journalctl not found"
    try:
        r = subprocess.run(
            ["journalctl", "-u", unit, "--since", f"{hours} hours ago",
             "--no-pager", "-n", "200"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return [], f"journalctl failed: {e!r}"
    if r.returncode != 0:
        return [], f"journalctl returned {r.returncode}"
    # Filter only lines with suspicious keywords.
    keep_re = re.compile(
        r"\b(error|fatal|panic|oom|cuda|nvrm|xid|killed|exit)\b",
        re.IGNORECASE,
    )
    tail = [
        line.strip() for line in r.stdout.splitlines()
        if keep_re.search(line)
    ]
    # Don't return too many — take the last 20 interesting lines.
    return tail[-20:], None


# ──── Top-level collector ────────────────────────────────────────────────

# Etap 3.2 (audit 2026-05-12): default container-name prefixes that we
# consider "Genesis-related". Anything outside this set is ignored when
# deciding fatal signals — otherwise unrelated restart loops (e.g.
# `nvidia-gpu-exporter`) keep `has_fatal_signals=True` permanently.
_DEFAULT_CONTAINER_PREFIXES: tuple[str, ...] = ("vllm", "genesis", "sndr")


def _filter_containers(
    containers: list[RestartingContainer],
    *, prefixes: tuple[str, ...] | None, all_containers: bool,
) -> list[RestartingContainer]:
    """Keep only containers whose name starts with one of `prefixes`.

    `all_containers=True` short-circuits the filter (operator opted in
    to see every restart loop on the host).
    """
    if all_containers:
        return list(containers)
    effective = prefixes if prefixes is not None else _DEFAULT_CONTAINER_PREFIXES
    return [
        c for c in containers
        if any(c.name.startswith(p) for p in effective)
    ]


def collect_log_forensics(
    window_hours: int = 24,
    service_unit: str = "genesis-vllm.service",
    dmesg_reader=_read_dmesg,
    container_collector=_collect_restarting_containers,
    journal_collector=_collect_service_journal,
    *,
    strict_window: bool = False,
    container_prefixes: tuple[str, ...] | None = None,
    all_containers: bool = False,
) -> LogForensicsResult:
    """Collects all log forensics sources.

    Args:
        window_hours: time window for event filtering.
        service_unit: systemd unit name for journalctl tailing.
        dmesg_reader / container_collector / journal_collector: injected
            for testability.
        strict_window: Etap 3.3 — drop events with unknown timestamps
            (otherwise included by default).
        container_prefixes: Etap 3.2 — keep only containers whose name
            starts with one of these prefixes. `None` → use
            `_DEFAULT_CONTAINER_PREFIXES`.
        all_containers: Etap 3.2 — bypass prefix filter entirely.
    """
    result = LogForensicsResult(window_hours=window_hours)
    window_seconds = window_hours * 3600

    # dmesg → OOM + Xid
    dmesg_out, dmesg_reason = dmesg_reader(None)
    if dmesg_reason:
        result.sources_unavailable.append(f"dmesg: {dmesg_reason}")
    else:
        oom_all = _parse_oom_events(dmesg_out)
        xid_all = _parse_xid_events(dmesg_out)
        result.oom_events = _filter_within_window(
            oom_all, window_seconds, strict=strict_window,
        )
        result.xid_events = _filter_within_window(
            xid_all, window_seconds, strict=strict_window,
        )

    # docker → restarting containers (filtered by name prefix)
    containers, c_reason = container_collector()
    if c_reason:
        result.sources_unavailable.append(f"docker: {c_reason}")
    else:
        result.restarting_containers = _filter_containers(
            containers,
            prefixes=container_prefixes,
            all_containers=all_containers,
        )

    # journalctl → service logs tail
    journal, j_reason = journal_collector(service_unit, window_hours)
    if j_reason:
        result.sources_unavailable.append(f"journalctl: {j_reason}")
    else:
        result.service_journal_tail = journal

    return result


# ──── Text rendering ────────────────────────────────────────────────────

def summarize_for_text(r: LogForensicsResult) -> list[str]:
    """Human-readable lines for `sndr doctor-system --logs` (no --json)."""
    lines: list[str] = []
    lines.append(f"  Log forensics (last {r.window_hours}h):")

    if r.oom_events:
        lines.append(f"    ✗ OOM-kill events: {len(r.oom_events)}")
        for e in r.oom_events[:3]:
            ago = (
                f" ({e.timestamp_seconds_ago // 60} min ago)"
                if e.timestamp_seconds_ago is not None else ""
            )
            lines.append(f"        — process={e.killed_process}{ago}")
    else:
        lines.append("    ✓ no OOM-kills")

    fatal_xids = [x for x in r.xid_events if x.severity == "fatal"]
    warn_xids = [x for x in r.xid_events if x.severity == "warning"]
    if fatal_xids:
        lines.append(f"    ✗ NVRM Xid (FATAL): {len(fatal_xids)}")
        for x in fatal_xids[:3]:
            ago = (
                f" ({x.timestamp_seconds_ago // 60} min ago)"
                if x.timestamp_seconds_ago is not None else ""
            )
            lines.append(
                f"        — Xid {x.xid_code} on {x.pci_addr}{ago}"
            )
    elif warn_xids:
        lines.append(f"    ⚠ NVRM Xid (warning): {len(warn_xids)}")
    else:
        lines.append("    ✓ no NVRM Xid errors")

    if r.restarting_containers:
        lines.append(
            f"    ✗ Restarting containers: {len(r.restarting_containers)}"
        )
        for c in r.restarting_containers[:5]:
            lines.append(f"        — {c.name} ({c.image}): {c.status}")
    else:
        lines.append("    ✓ no restarting containers")

    if r.service_journal_tail:
        lines.append(
            f"    ⚠ journalctl: {len(r.service_journal_tail)} "
            "suspicious lines (last 20 shown):"
        )
        for line in r.service_journal_tail[-5:]:
            # Trim long lines for readability
            clip = line if len(line) <= 120 else line[:117] + "…"
            lines.append(f"        {clip}")

    if r.sources_unavailable:
        lines.append("    sources unavailable:")
        for s in r.sources_unavailable:
            lines.append(f"        · {s}")

    return lines
