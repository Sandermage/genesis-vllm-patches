# SPDX-License-Identifier: Apache-2.0
"""Audit P2-1 closure (2026-05-12): unit tests for host log forensics
(`sndr doctor-system --logs`).

Cover:

  • dmesg OOM-kill parsing (classic format + variations).
  • NVRM Xid parsing and severity classification (fatal vs warning vs info).
  • Window filter (events older than N hours are dropped).
  • Docker restarting containers shape.
  • Graceful degradation when binaries are unavailable.
  • Top-level composition (collect_log_forensics).
  • Text summarization shape.
"""
from __future__ import annotations

import pytest

from sndr.cli.legacy.doctor_logs import (
    FATAL_XIDS,
    LogForensicsResult,
    OomEvent,
    RestartingContainer,
    XidEvent,
    _classify_xid,
    _filter_within_window,
    _parse_oom_events,
    _parse_xid_events,
    collect_log_forensics,
    summarize_for_text,
)


# ─── OOM parsing ────────────────────────────────────────────────────────


class TestOomParsing:
    def test_classic_uptime_format(self):
        text = (
            "[12345.678] Out of memory: Killed process 9999 (vllm) "
            "total-vm:65535MB, anon-rss:4096MB"
        )
        events = _parse_oom_events(text)
        assert len(events) == 1
        assert events[0].killed_process == "vllm"
        assert "vllm" in events[0].raw_line

    def test_multiple_events(self):
        text = "\n".join([
            "[100.0] Out of memory: Killed process 1 (proc_a)",
            "[200.0] unrelated kernel message",
            "[300.0] Out of memory: Killed process 2 (proc_b)",
        ])
        events = _parse_oom_events(text)
        assert {e.killed_process for e in events} == {"proc_a", "proc_b"}

    def test_no_oom_in_clean_log(self):
        assert _parse_oom_events(
            "[1.0] usb 1-1: new high-speed USB device"
        ) == []

    def test_case_insensitive(self):
        text = "[1.0] out of memory: Killed process 1 (lower)"
        assert len(_parse_oom_events(text)) == 1


# ─── Xid parsing ───────────────────────────────────────────────────────


class TestXidParsing:
    def test_basic_xid(self):
        text = "[12345.678] NVRM: Xid (PCI:0000:01:00): 31, pid=1234"
        events = _parse_xid_events(text)
        assert len(events) == 1
        e = events[0]
        assert e.xid_code == 31
        assert e.pci_addr == "PCI:0000:01:00"
        assert e.severity == "fatal"

    def test_classification(self):
        assert _classify_xid(31) == "fatal"
        assert _classify_xid(43) == "fatal"
        assert _classify_xid(79) == "fatal"
        assert _classify_xid(13) == "warning"
        assert _classify_xid(14) == "warning"
        assert _classify_xid(62) == "warning"
        assert _classify_xid(119) == "warning"
        # Unknown / info-level
        assert _classify_xid(99) == "info"
        assert _classify_xid(1) == "info"

    def test_all_fatal_codes_classified(self):
        """Sanity: every code in FATAL_XIDS must return 'fatal'."""
        for code in FATAL_XIDS:
            assert _classify_xid(code) == "fatal"

    def test_no_xid_in_clean_log(self):
        assert _parse_xid_events("[1.0] random message") == []

    def test_malformed_xid_skipped(self):
        text = "[1.0] NVRM: Xid (PCI:0000:01:00): not-a-number, pid=1"
        assert _parse_xid_events(text) == []


# ─── Window filter ──────────────────────────────────────────────────────


class TestWindowFilter:
    def test_includes_unknown_timestamps(self):
        e1 = OomEvent(timestamp_seconds_ago=None, killed_process="x", raw_line="")
        out = _filter_within_window([e1], window_seconds=3600)
        assert out == [e1]

    def test_drops_events_older_than_window(self):
        recent = OomEvent(timestamp_seconds_ago=600, killed_process="a", raw_line="")
        old = OomEvent(timestamp_seconds_ago=10_000, killed_process="b", raw_line="")
        out = _filter_within_window([recent, old], window_seconds=3600)
        assert recent in out
        assert old not in out


# ─── LogForensicsResult ────────────────────────────────────────────────


class TestLogForensicsResult:
    def test_empty_result_no_fatal_signals(self):
        r = LogForensicsResult(window_hours=24)
        assert r.has_fatal_signals is False

    def test_oom_triggers_fatal(self):
        r = LogForensicsResult(
            window_hours=24,
            oom_events=[OomEvent(timestamp_seconds_ago=60,
                                  killed_process="vllm", raw_line="x")],
        )
        assert r.has_fatal_signals is True

    def test_fatal_xid_triggers(self):
        r = LogForensicsResult(
            window_hours=24,
            xid_events=[XidEvent(
                timestamp_seconds_ago=60, xid_code=31,
                pci_addr="PCI:0:1:0", severity="fatal", raw_line="x",
            )],
        )
        assert r.has_fatal_signals is True

    def test_warning_xid_does_not_trigger_fatal(self):
        r = LogForensicsResult(
            window_hours=24,
            xid_events=[XidEvent(
                timestamp_seconds_ago=60, xid_code=13,
                pci_addr="PCI:0:1:0", severity="warning", raw_line="x",
            )],
        )
        assert r.has_fatal_signals is False

    def test_restarting_container_triggers(self):
        r = LogForensicsResult(
            window_hours=24,
            restarting_containers=[RestartingContainer(
                name="vllm-test", image="img", status="Restarting (1)",
                started_at="5 min ago",
            )],
        )
        assert r.has_fatal_signals is True

    def test_to_dict_shape(self):
        r = LogForensicsResult(window_hours=24)
        d = r.to_dict()
        for key in (
            "window_hours", "oom_events", "xid_events",
            "restarting_containers", "service_journal_tail",
            "sources_unavailable", "has_fatal_signals",
        ):
            assert key in d


# ─── collect_log_forensics (top-level composition) ─────────────────────


def _fake_dmesg_clean(_=None):
    return "[1.0] usb 1-1: nothing interesting\n", None


def _fake_dmesg_with_oom_and_xid(_=None):
    # Without an [uptime] prefix — otherwise the window filter would
    # compare ts_ago against the real system's /proc/uptime and events
    # could fall outside the window. Parsing with [uptime] is covered
    # separately in TestOomParsing / TestXidParsing.
    return (
        "Mon May 12 03:14:15 host kernel: Out of memory: Killed process 999 (vllm)\n"
        "Mon May 12 03:14:16 host kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=999\n"
        "Mon May 12 03:14:17 host kernel: NVRM: Xid (PCI:0000:01:00): 13, pid=42\n",
        None,
    )


def _fake_dmesg_unavailable(_=None):
    return "", "dmesg unavailable (test)"


def _fake_containers_empty():
    return [], None


def _fake_containers_restart():
    return [RestartingContainer(
        name="vllm-pn95", image="vllm:nightly",
        status="Restarting (3) 12 seconds ago",
        started_at="5 min ago",
    )], None


def _fake_journal_empty(unit, hours):
    return [], None


def _fake_journal_with_errors(unit, hours):
    return [
        "kernel: CUDA error: out of memory",
        "vllm-genesis[1]: FATAL: model load failed",
    ], None


class TestCollectLogForensics:
    def test_clean_environment(self):
        r = collect_log_forensics(
            window_hours=24,
            dmesg_reader=_fake_dmesg_clean,
            container_collector=_fake_containers_empty,
            journal_collector=_fake_journal_empty,
        )
        assert r.oom_events == []
        assert r.xid_events == []
        assert r.restarting_containers == []
        assert r.has_fatal_signals is False
        assert r.sources_unavailable == []

    def test_oom_and_xid_detected(self):
        r = collect_log_forensics(
            window_hours=24,
            dmesg_reader=_fake_dmesg_with_oom_and_xid,
            container_collector=_fake_containers_empty,
            journal_collector=_fake_journal_empty,
        )
        assert len(r.oom_events) == 1
        assert r.oom_events[0].killed_process == "vllm"
        assert len(r.xid_events) == 2
        # One fatal (31), one warning (13)
        assert sum(1 for x in r.xid_events if x.severity == "fatal") == 1
        assert sum(1 for x in r.xid_events if x.severity == "warning") == 1
        assert r.has_fatal_signals is True

    def test_restart_loop_detected(self):
        r = collect_log_forensics(
            window_hours=24,
            dmesg_reader=_fake_dmesg_clean,
            container_collector=_fake_containers_restart,
            journal_collector=_fake_journal_empty,
        )
        assert len(r.restarting_containers) == 1
        assert r.restarting_containers[0].name == "vllm-pn95"
        assert r.has_fatal_signals is True

    def test_unavailable_dmesg_recorded(self):
        r = collect_log_forensics(
            window_hours=24,
            dmesg_reader=_fake_dmesg_unavailable,
            container_collector=_fake_containers_empty,
            journal_collector=_fake_journal_empty,
        )
        assert r.oom_events == []
        assert any("dmesg" in s for s in r.sources_unavailable)

    def test_journal_lines_kept(self):
        r = collect_log_forensics(
            window_hours=24,
            dmesg_reader=_fake_dmesg_clean,
            container_collector=_fake_containers_empty,
            journal_collector=_fake_journal_with_errors,
        )
        assert len(r.service_journal_tail) == 2


# ─── Text summarization ────────────────────────────────────────────────


class TestSummarizeText:
    def test_clean_result_renders(self):
        r = LogForensicsResult(window_hours=24)
        lines = summarize_for_text(r)
        # Should be three checkmarks (no OOM, no Xid, no restarts)
        joined = "\n".join(lines)
        assert "no OOM-kills" in joined
        assert "no NVRM Xid errors" in joined
        assert "no restarting containers" in joined

    def test_fatal_result_marks_x(self):
        r = LogForensicsResult(
            window_hours=24,
            oom_events=[OomEvent(timestamp_seconds_ago=60,
                                  killed_process="vllm", raw_line="x")],
            xid_events=[XidEvent(
                timestamp_seconds_ago=300, xid_code=31,
                pci_addr="PCI:0:1:0", severity="fatal", raw_line="x",
            )],
            restarting_containers=[RestartingContainer(
                name="vllm-test", image="i", status="Restarting (5)",
                started_at="1 min ago",
            )],
        )
        lines = summarize_for_text(r)
        joined = "\n".join(lines)
        assert "OOM-kill events: 1" in joined
        assert "FATAL" in joined
        assert "Restarting containers: 1" in joined

    def test_sources_unavailable_rendered(self):
        r = LogForensicsResult(window_hours=24,
                                sources_unavailable=["dmesg: not available"])
        lines = summarize_for_text(r)
        assert any("dmesg" in line for line in lines)


# ─── Etap 3.2 — container allowlist filter ──────────────────────────────


class TestContainerPrefixFilter:
    """Etap 3.2 (audit 2026-05-12): by default only Genesis-related
    containers (name prefixes: vllm/genesis/sndr) feed `has_fatal_signals`.
    Unrelated restart loops (e.g. `nvidia-gpu-exporter`) should not
    permanently red-flag the host."""

    def _restarting(self, name: str) -> RestartingContainer:
        return RestartingContainer(
            name=name, image="img", status="Restarting (1)",
            started_at="1 min ago",
        )

    def _container_fixture(self, containers):
        def collector():
            return list(containers), None
        return collector

    def test_default_filter_excludes_unrelated(self):
        containers = [
            self._restarting("nvidia-gpu-exporter"),
            self._restarting("vllm-pn95-2xa5000"),
        ]
        r = collect_log_forensics(
            dmesg_reader=lambda _=None: ("", "unavailable"),
            container_collector=self._container_fixture(containers),
            journal_collector=lambda u, h: ([], None),
        )
        names = [c.name for c in r.restarting_containers]
        assert names == ["vllm-pn95-2xa5000"]
        assert r.has_fatal_signals is True  # vllm-pn95 is fatal

    def test_default_filter_filters_out_unrelated_only(self):
        containers = [
            self._restarting("nvidia-gpu-exporter"),
            self._restarting("docker-sandbox-1"),
        ]
        r = collect_log_forensics(
            dmesg_reader=lambda _=None: ("", "unavailable"),
            container_collector=self._container_fixture(containers),
            journal_collector=lambda u, h: ([], None),
        )
        assert r.restarting_containers == []
        assert r.has_fatal_signals is False

    def test_custom_prefix_accepted(self):
        containers = [self._restarting("myapp-1")]
        r = collect_log_forensics(
            dmesg_reader=lambda _=None: ("", "unavailable"),
            container_collector=self._container_fixture(containers),
            journal_collector=lambda u, h: ([], None),
            container_prefixes=("myapp",),
        )
        assert len(r.restarting_containers) == 1

    def test_all_containers_bypasses_filter(self):
        containers = [
            self._restarting("nvidia-gpu-exporter"),
            self._restarting("docker-sandbox-1"),
        ]
        r = collect_log_forensics(
            dmesg_reader=lambda _=None: ("", "unavailable"),
            container_collector=self._container_fixture(containers),
            journal_collector=lambda u, h: ([], None),
            all_containers=True,
        )
        assert len(r.restarting_containers) == 2
        assert r.has_fatal_signals is True


# ─── Etap 3.3 — strict window mode ──────────────────────────────────────


class TestStrictWindow:
    """Etap 3.3 (audit 2026-05-12): with `strict_window=True`, events
    with unknown timestamps are dropped (long-uptime host won't surface
    very old events as 'possibly recent')."""

    def _dmesg_with_unknown_ts(self, _=None):
        # ctime-style line — our parser yields ts=None for these
        return (
            "Mon May 12 03:14:15 host kernel: Out of memory: "
            "Killed process 1 (foo)\n",
            None,
        )

    def test_default_includes_unknown_ts(self):
        r = collect_log_forensics(
            dmesg_reader=self._dmesg_with_unknown_ts,
            container_collector=lambda: ([], None),
            journal_collector=lambda u, h: ([], None),
        )
        assert len(r.oom_events) == 1

    def test_strict_drops_unknown_ts(self):
        r = collect_log_forensics(
            dmesg_reader=self._dmesg_with_unknown_ts,
            container_collector=lambda: ([], None),
            journal_collector=lambda u, h: ([], None),
            strict_window=True,
        )
        assert r.oom_events == []

    def test_strict_keeps_events_within_window(self):
        from sndr.cli.legacy.doctor_logs import (
            OomEvent, _filter_within_window,
        )
        recent = OomEvent(timestamp_seconds_ago=600, killed_process="x", raw_line="")
        old = OomEvent(timestamp_seconds_ago=999_999, killed_process="y", raw_line="")
        unknown = OomEvent(timestamp_seconds_ago=None, killed_process="z", raw_line="")
        # Non-strict: all three present (recent + unknown)
        ns = _filter_within_window([recent, old, unknown], window_seconds=3600)
        assert {e.killed_process for e in ns} == {"x", "z"}
        # Strict: only recent (unknown dropped)
        s = _filter_within_window(
            [recent, old, unknown], window_seconds=3600, strict=True,
        )
        assert {e.killed_process for e in s} == {"x"}
