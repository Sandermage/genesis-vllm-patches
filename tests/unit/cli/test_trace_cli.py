# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr trace list` — §6.H6 of the unified development plan.

Covers:
  * Static catalog output (no --container, both human and JSON).
  * --container with `docker` missing (fallback warning).
  * --container with mocked `docker exec` output (live size + mtime).
  * --category filter.
  * --all flag semantics (with --container).
  * Argparse wiring registered under the top-level `sndr` parser.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(*args: str, env_extra: dict | None = None,
         text_in: str | None = None) -> subprocess.CompletedProcess:
    """Run `sndr trace ...` via the package's CLI entry."""
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "vllm.sndr_core.cli", "trace", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env=env, check=False, input=text_in,
    )


# ─── Argparse registration ───────────────────────────────────────────


def test_trace_subcommand_is_registered() -> None:
    """`sndr trace list --help` must surface our help text — proves the
    add_argparser() was wired into the root CLI."""
    r = _run("list", "--help")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # The subcommand's own --help renders its argparse options; the
    # description text only appears in the parent (sndr trace --help).
    # The args themselves are sufficient surface to lock.
    assert "--container" in out
    assert "--category" in out
    assert "--json" in out
    assert "--all" in out


def test_trace_top_help_lists_subcommands() -> None:
    r = _run("--help")
    assert r.returncode == 0
    assert "list" in r.stdout


# ─── Default catalog output ──────────────────────────────────────────


def test_list_human_view_lists_every_category_with_specs() -> None:
    r = _run("list")
    assert r.returncode == 0, r.stderr
    # Human view banner.
    assert "sndr trace — known diagnostic traces" in r.stdout
    # Every category that has specs prints its header.
    from vllm.sndr_core.observability.trace_catalog import iter_by_category
    grouped = iter_by_category()
    for cat, specs in grouped.items():
        if specs:
            assert cat.upper() in r.stdout, (
                f"category {cat!r} has specs but its header is missing "
                "from human view"
            )


def test_list_human_view_prints_total_line() -> None:
    r = _run("list")
    assert r.returncode == 0
    assert "Total:" in r.stdout


def test_list_json_payload_matches_catalog() -> None:
    r = _run("list", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    from vllm.sndr_core.observability.trace_catalog import TRACE_CATALOG
    assert payload["total"] == len(TRACE_CATALOG)
    assert payload["container"] is None
    assert payload["category_filter"] is None
    assert isinstance(payload["traces"], list)
    # Spot-check: every catalog entry surfaces by id.
    payload_ids = {t["id"] for t in payload["traces"]}
    catalog_ids = {s.id for s in TRACE_CATALOG}
    assert payload_ids == catalog_ids


def test_list_json_includes_live_field_as_null_without_container() -> None:
    r = _run("list", "--json")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    for t in payload["traces"]:
        assert t["live"] is None


# ─── --category filter ───────────────────────────────────────────────


def test_list_category_filter_keeps_only_matching() -> None:
    r = _run("list", "--category", "boot", "--json")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["category_filter"] == "boot"
    assert all(t["category"] == "boot" for t in payload["traces"])
    assert payload["total"] >= 1  # boot always-on entry must be present


def test_list_category_invalid_returns_nonzero_with_message() -> None:
    r = _run("list", "--category", "not_a_category")
    assert r.returncode != 0
    # Either stderr or stdout — depends on the _io.error contract.
    combined = r.stdout + r.stderr
    assert "unknown category" in combined.lower()


# ─── --container with no docker available ────────────────────────────


def test_list_container_without_docker_emits_warning(monkeypatch) -> None:
    """The CLI must fall back to the static catalog with a clear warn
    when docker isn't on PATH — operators may run `sndr trace list`
    on a machine without docker."""
    # Force PATH to a single directory that has no docker.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # No docker binary in tmpdir.
        env_extra = {"PATH": tmpdir}
        r = _run("list", "--container", "fake-container",
                 env_extra=env_extra)
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert ("docker binary not on $PATH" in combined or
            "docker exec" in combined.lower()), combined


def test_list_container_without_docker_json_carries_warning() -> None:
    """In JSON mode the warning must surface in the
    `container_warning` field so machine consumers can branch on it."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        env_extra = {"PATH": tmpdir}
        r = _run("list", "--container", "fake-container", "--all", "--json",
                 env_extra=env_extra)
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["container"] == "fake-container"
    assert isinstance(payload["container_warning"], str)
    assert "docker" in payload["container_warning"].lower()


# ─── Live container introspection via mocked docker ──────────────────


def test_container_ls_parses_typical_row(monkeypatch) -> None:
    """Unit-test the parser directly so the contract is locked even
    when no live container is reachable from the test env."""
    import vllm.sndr_core.cli.trace as tmod

    typical_ls_output = (
        "total 12\n"
        "drwxrwxrwt 1 root root  4096 2026-05-30 13:00:00.000000000 +0000 .\n"
        "drwxr-xr-x 1 root root  4096 2026-05-30 12:00:00.000000000 +0000 ..\n"
        "-rw-r--r-- 1 root root  1234 2026-05-30 14:00:00.000000000 +0000 "
        "genesis_pn248_acceptance_trace.log\n"
        "-rw-r--r-- 1 root root 99999 2026-05-30 14:30:00.000000000 +0000 "
        "genesis_boot.log\n"
        "-rw-r--r-- 1 root root    42 2026-05-30 14:31:00.000000000 +0000 "
        "unrelated.log\n"
    )

    class FakeRun:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod.subprocess, "run",
        lambda *a, **kw: FakeRun(typical_ls_output),
    )

    out = tmod._container_ls_tmp("any-container-name")
    assert "genesis_pn248_acceptance_trace.log" in out
    assert "genesis_boot.log" in out
    assert "unrelated.log" not in out, (
        "non-genesis files must be filtered out at parse time"
    )
    size, mtime = out["genesis_pn248_acceptance_trace.log"]
    assert size == 1234
    assert mtime.startswith("2026-05-30")


def test_container_ls_returns_empty_on_subprocess_error(monkeypatch) -> None:
    """Subprocess failures (timeout, exec error, container stopped)
    must surface as empty dict — never raise."""
    import vllm.sndr_core.cli.trace as tmod

    class FakeRun:
        returncode = 1
        stdout = ""
        stderr = "Error: container not running\n"

    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod.subprocess, "run", lambda *a, **kw: FakeRun()
    )
    assert tmod._container_ls_tmp("any") == {}


def test_container_ls_handles_timeout(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod

    def raise_timeout(*a, **kw):
        raise tmod.subprocess.TimeoutExpired(cmd=a[0], timeout=10)

    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(tmod.subprocess, "run", raise_timeout)
    assert tmod._container_ls_tmp("any") == {}


# ─── --all semantics ────────────────────────────────────────────────


def test_default_container_view_hides_absent_files(monkeypatch) -> None:
    """Without --all, the human view only lists traces actually present.
    Verified at the run_list level by stubbing _container_ls_tmp to
    return a single hit."""
    import vllm.sndr_core.cli.trace as tmod

    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {"genesis_boot.log": (12345, "2026-05-30 14:00:00 +0000")},
    )

    import argparse
    args = argparse.Namespace(
        trace_cmd="list", container="any", category=None, json=False,
        all=False, func=tmod.run_list,
    )
    # Capture stdout for the assertion.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_list(args)
    assert rc == 0
    out = buf.getvalue()
    # Only `boot` shows up — all other traces are absent.
    assert "boot" in out
    assert "pn248_acceptance" not in out


def test_all_flag_shows_full_catalog_with_present_annotations(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod

    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {"genesis_boot.log": (12345, "2026-05-30 14:00:00 +0000")},
    )

    import argparse
    args = argparse.Namespace(
        trace_cmd="list", container="any", category=None, json=False,
        all=True, func=tmod.run_list,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_list(args)
    assert rc == 0
    out = buf.getvalue()
    # boot is present.
    assert "12.1KB" in out or "12345" in out
    # pn248 is in catalog → header line still appears even though
    # its live file is absent.
    assert "pn248_acceptance" in out
    assert "(not present)" in out


# ─── fmt_size helper ────────────────────────────────────────────────


def test_fmt_size_renders_bytes_kb_mb() -> None:
    import vllm.sndr_core.cli.trace as tmod
    assert tmod._fmt_size(0) == "0B"
    assert tmod._fmt_size(1023) == "1023B"
    assert tmod._fmt_size(1024) == "1.0KB"
    assert tmod._fmt_size(1536) == "1.5KB"
    assert tmod._fmt_size(1024 * 1024) == "1.0MB"
    assert tmod._fmt_size(5 * 1024 * 1024) == "5.0MB"
    assert tmod._fmt_size(1024 * 1024 * 1024) == "1.0GB"


# ─── collect (§6.H7) ─────────────────────────────────────────────────


def test_collect_subcommand_is_registered() -> None:
    r = _run("collect", "--help")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--container" in out
    assert "--trace" in out
    assert "--output-dir" in out
    assert "--json" in out


def test_collect_requires_container_flag() -> None:
    r = _run("collect")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "--container" in combined.lower() or "required" in combined.lower()


def test_default_output_dir_uses_container_and_utc_stamp(monkeypatch) -> None:
    """The default output dir must encode the container name + a UTC
    timestamp so multiple collections don't overwrite each other and
    cross-timezone bundles stay unambiguous."""
    import vllm.sndr_core.cli.trace as tmod
    out = tmod._default_output_dir("vllm-35b-prod")
    assert out.startswith("./genesis_traces_vllm-35b-prod_")
    # 8-digit date + T + 6-digit time + Z (UTC).
    assert "T" in out and out.endswith("Z")


def test_default_output_dir_sanitizes_path_separators(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod
    out = tmod._default_output_dir("foo/bar")
    assert "/bar" not in out.replace("./", "")
    assert "foo-bar" in out


def test_collect_unknown_trace_id_returns_2() -> None:
    r = _run("collect", "--container", "any", "--trace", "nope_id")
    assert r.returncode == 2
    combined = r.stdout + r.stderr
    assert "unknown trace id" in combined.lower()


def test_collect_no_docker_returns_3() -> None:
    """Without docker on PATH, collect must surface a clear error."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        env_extra = {"PATH": tmpdir}
        r = _run(
            "collect", "--container", "any",
            env_extra=env_extra,
        )
    assert r.returncode == 3
    combined = r.stdout + r.stderr
    assert "docker" in combined.lower()


def test_collect_no_live_traces_returns_zero_and_empty_json(monkeypatch) -> None:
    """When the container exists but no genesis_* file is being
    written, collect returns 0 with an empty `collected` list — the
    operator just learns there's nothing to grab."""
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(tmod, "_container_ls_tmp", lambda c: {})

    import argparse
    args = argparse.Namespace(
        trace_cmd="collect", container="empty", trace_id=None,
        output_dir=None, json=True, func=tmod.run_collect,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_collect(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["collected"] == []
    assert payload["output_dir"] is None
    assert payload["container"] == "empty"


def test_collect_copies_each_present_trace(monkeypatch, tmp_path) -> None:
    """End-to-end happy path: container reports two genesis_* files;
    collect calls docker cp once per file and reports both as
    collected. docker cp is stubbed — we only verify the call shape
    and the reported payload, not real I/O."""
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {
            "genesis_boot.log": (10240, "2026-05-30 14:00:00 +0000"),
            "genesis_pn248_acceptance_trace.log":
                (2048, "2026-05-30 14:01:00 +0000"),
        },
    )

    cp_calls: list[tuple[str, str, str]] = []

    def fake_cp(container, src, dst):
        cp_calls.append((container, src, dst))
        return True, "ok"

    monkeypatch.setattr(tmod, "_docker_cp", fake_cp)

    import argparse
    args = argparse.Namespace(
        trace_cmd="collect", container="vllm-rig",
        trace_id=None,
        output_dir=str(tmp_path), json=True,
        func=tmod.run_collect,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_collect(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert len(payload["collected"]) == 2
    assert all(
        c["dst"].startswith(str(tmp_path)) for c in payload["collected"]
    )
    assert len(cp_calls) == 2
    # All cp calls hit our container.
    assert all(c[0] == "vllm-rig" for c in cp_calls)


def test_collect_trace_id_filter_limits_to_one(monkeypatch, tmp_path) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {
            "genesis_boot.log": (10240, "2026-05-30 14:00:00 +0000"),
            "genesis_pn248_acceptance_trace.log":
                (2048, "2026-05-30 14:01:00 +0000"),
        },
    )

    cp_calls: list[tuple[str, str, str]] = []

    def fake_cp(container, src, dst):
        cp_calls.append((container, src, dst))
        return True, "ok"

    monkeypatch.setattr(tmod, "_docker_cp", fake_cp)

    import argparse
    args = argparse.Namespace(
        trace_cmd="collect", container="vllm-rig",
        trace_id="boot",
        output_dir=str(tmp_path), json=True,
        func=tmod.run_collect,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_collect(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert len(payload["collected"]) == 1
    assert payload["collected"][0]["id"] == "boot"
    # docker cp was called exactly once, only for the boot log.
    assert len(cp_calls) == 1
    assert cp_calls[0][1] == "/tmp/genesis_boot.log"


def test_collect_docker_cp_error_surfaces_in_errors_list(
    monkeypatch, tmp_path,
) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        tmod, "_container_ls_tmp",
        lambda c: {"genesis_boot.log": (1024, "2026-05-30 14:00:00 +0000")},
    )
    def fake_cp_fail(container, src, dst):
        return False, "Error: No such container"

    monkeypatch.setattr(tmod, "_docker_cp", fake_cp_fail)

    import argparse
    args = argparse.Namespace(
        trace_cmd="collect", container="missing",
        trace_id=None,
        output_dir=str(tmp_path), json=True,
        func=tmod.run_collect,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tmod.run_collect(args)
    # Exit 1 because at least one error.
    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert payload["collected"] == []
    assert len(payload["errors"]) == 1
    assert "No such container" in payload["errors"][0]["error"]


# ─── _docker_cp helper ─────────────────────────────────────────────


def test_docker_cp_no_binary_returns_false(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: None)
    ok, msg = tmod._docker_cp("c", "/tmp/x", "/tmp/y")
    assert ok is False
    assert "PATH" in msg or "docker" in msg


def test_docker_cp_nonzero_returncode_returns_false(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    class FakeRun:
        returncode = 1
        stdout = ""
        stderr = "Error response from daemon: No such container: nope\n"

    monkeypatch.setattr(tmod.subprocess, "run", lambda *a, **kw: FakeRun())
    ok, msg = tmod._docker_cp("nope", "/tmp/x", "/tmp/y")
    assert ok is False
    assert "No such container" in msg


def test_docker_cp_success_returns_true(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    class FakeRun:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(tmod.subprocess, "run", lambda *a, **kw: FakeRun())
    ok, msg = tmod._docker_cp("c", "/tmp/x", "/tmp/y")
    assert ok is True
    assert msg == "ok"


def test_docker_cp_timeout_returns_false(monkeypatch) -> None:
    import vllm.sndr_core.cli.trace as tmod
    monkeypatch.setattr(tmod.shutil, "which", lambda name: "/usr/bin/docker")

    def raise_timeout(*a, **kw):
        raise tmod.subprocess.TimeoutExpired(cmd=a[0], timeout=30)

    monkeypatch.setattr(tmod.subprocess, "run", raise_timeout)
    ok, msg = tmod._docker_cp("c", "/tmp/x", "/tmp/y")
    assert ok is False
    assert "raised" in msg or "TimeoutExpired" in msg


# ─── summarize (§6.H8) ───────────────────────────────────────────────


def test_summarize_subcommand_is_registered() -> None:
    r = _run("summarize", "--help")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "log_file" in out.lower() or "log_file" in out
    assert "--json" in out
    assert "--max-line-preview" in out


def test_detect_trace_kind_for_known_basenames() -> None:
    import vllm.sndr_core.cli.trace as tmod
    assert tmod.detect_trace_kind("/tmp/genesis_boot.log") == "boot"
    assert tmod.detect_trace_kind("genesis_boot.log") == "boot"
    assert tmod.detect_trace_kind(
        "/x/y/genesis_pn248_acceptance_trace.log"
    ) == "acceptance"
    assert tmod.detect_trace_kind(
        "genesis_pn258_oracle_trace.log"
    ) == "oracle"
    assert tmod.detect_trace_kind(
        "/a/b/genesis_pn260_kernel_trace.log"
    ) == "kernel"
    assert tmod.detect_trace_kind(
        "genesis_pn254_fire.log"
    ) == "kernel"
    assert tmod.detect_trace_kind(
        "genesis_pn255_kv_write.log"
    ) == "kv_write"
    assert tmod.detect_trace_kind(
        "genesis_pn256_route.log"
    ) == "routing"
    assert tmod.detect_trace_kind(
        "genesis_pn241_mtp_trace.log"
    ) == "mtp"
    assert tmod.detect_trace_kind(
        "genesis_tq_forward.log"
    ) == "tq_forward"


def test_detect_trace_kind_unknown_returns_unknown() -> None:
    import vllm.sndr_core.cli.trace as tmod
    assert tmod.detect_trace_kind("random.log") == "unknown"
    assert tmod.detect_trace_kind("") == "unknown"
    assert tmod.detect_trace_kind("/var/log/messages") == "unknown"


def test_summarize_missing_file_returns_2(tmp_path) -> None:
    nonexistent = tmp_path / "no-such-file.log"
    r = _run("summarize", str(nonexistent))
    assert r.returncode == 2
    combined = r.stdout + r.stderr
    assert "does not exist" in combined.lower()


def test_summarize_generic_returns_size_lines_first_last(tmp_path) -> None:
    """Generic summary on any file returns the canonical 4-field
    payload + the detected kind."""
    import vllm.sndr_core.cli.trace as tmod
    p = tmp_path / "random.log"
    p.write_text("first line\nmiddle line\nlast line\n")
    s = tmod.summarize_generic(str(p))
    assert s["total_lines"] == 3
    assert s["first_line"] == "first line"
    assert s["last_line"] == "last line"
    # size matches bytes written.
    assert s["size_bytes"] == len("first line\nmiddle line\nlast line\n")


def test_summarize_generic_truncates_long_lines(tmp_path) -> None:
    import vllm.sndr_core.cli.trace as tmod
    long_line = "x" * 500
    p = tmp_path / "x.log"
    p.write_text(long_line + "\n")
    s = tmod.summarize_generic(str(p), max_preview=80)
    assert s["first_line"].endswith("…")
    assert len(s["first_line"]) <= 81


def test_summarize_generic_max_preview_zero_disables_truncation(tmp_path) -> None:
    import vllm.sndr_core.cli.trace as tmod
    long_line = "x" * 500
    p = tmp_path / "x.log"
    p.write_text(long_line + "\n")
    s = tmod.summarize_generic(str(p), max_preview=0)
    assert s["first_line"] == long_line


def test_summarize_boot_log_counts_applied_skipped_failed(tmp_path) -> None:
    """The boot summarizer must parse the canonical `[Genesis] <PID>:
    <status>` format and produce correct counters + failure list."""
    import vllm.sndr_core.cli.trace as tmod
    boot_text = (
        "[Genesis] PN125: applied (FULL_AND_PIECEWISE flip)\n"
        "[Genesis] PN286: applied\n"
        "[Genesis] PN16: skipped (env flag off)\n"
        "[Genesis] PN999: failed (anchor drift)\n"
        "[Genesis] PN1000: applied\n"
        "random line with no patch info\n"
        "[Genesis] PN1001: error (boot-time crash)\n"
    )
    p = tmp_path / "genesis_boot.log"
    p.write_text(boot_text)
    s = tmod.summarize_boot_log(str(p))
    assert s["kind"] == "boot"
    assert s["counts"]["applied"] == 3
    assert s["counts"]["skipped"] == 1
    # `failed` + `error` both fold into failed.
    assert s["counts"]["failed"] == 2
    assert s["failed_patches"] == ["PN999", "PN1001"]
    assert s["skipped_patches"] == ["PN16"]
    # applied list is capped at 50 but we only have 3 here.
    assert s["applied_patches"] == ["PN125", "PN286", "PN1000"]


def test_summarize_boot_log_handles_zero_failed(tmp_path) -> None:
    import vllm.sndr_core.cli.trace as tmod
    p = tmp_path / "genesis_boot.log"
    p.write_text("[Genesis] PN125: applied\n[Genesis] PN16: skipped\n")
    s = tmod.summarize_boot_log(str(p))
    assert s["counts"]["failed"] == 0
    assert s["failed_patches"] == []


def test_summarize_dispatches_to_boot_handler_via_filename(tmp_path) -> None:
    """End-to-end: a file named genesis_boot.log should automatically
    surface boot-specific fields in the JSON payload."""
    p = tmp_path / "genesis_boot.log"
    p.write_text("[Genesis] PN125: applied\n[Genesis] PN999: failed\n")
    r = _run("summarize", str(p), "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    summary = payload["summary"]
    assert summary["kind"] == "boot"
    assert summary["counts"]["applied"] == 1
    assert summary["counts"]["failed"] == 1
    assert "PN999" in summary["failed_patches"]


def test_summarize_unknown_kind_falls_back_to_generic(tmp_path) -> None:
    """A non-boot file gets generic stats + kind tag = 'unknown' (or
    the detected kind if known but no specialized handler)."""
    p = tmp_path / "random.log"
    p.write_text("alpha\nbeta\ngamma\n")
    r = _run("summarize", str(p), "--json")
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    summary = payload["summary"]
    assert summary["kind"] == "unknown"
    assert summary["total_lines"] == 3
    assert "counts" not in summary  # no boot-specific fields


def test_summarize_human_view_renders_boot_failure_list(tmp_path) -> None:
    p = tmp_path / "genesis_boot.log"
    p.write_text(
        "[Genesis] PN125: applied\n"
        "[Genesis] PN999: failed\n"
        "[Genesis] PN1000: failed\n"
    )
    r = _run("summarize", str(p))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # Human view surfaces the failure list under a clear heading.
    assert "Boot apply summary" in out
    assert "PN999" in out
    assert "PN1000" in out
