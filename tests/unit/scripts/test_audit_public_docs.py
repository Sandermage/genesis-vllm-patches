# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_public_docs.py` — §6.10 public/private docs boundary.

Covers the six checks D-1..D-6 plus the live committed corpus, which
must pass cleanly now that the gate has been promoted from informational
to gating in `scripts/make_evidence.py`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_public_docs.py"


def _import():
    name = "_audit_public_docs_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """tmp_path with REPO_ROOT rebound so `_grep` can `relative_to` it."""
    mod = _import()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return tmp_path


def _scratch_doc(root: Path, body: str) -> list[Path]:
    p = root / "doc.md"
    p.write_text(body, encoding="utf-8")
    return [p]


# ─── D-1: no _internal links ──────────────────────────────────────────


class TestD1NoInternalLinks:
    def test_internal_link_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"see docs/_internal/foo.md for details\n")
        assert mod.check_d1_no_internal_links(files)

    def test_clean_doc(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"see docs/PATCHES.md for details\n")
        assert mod.check_d1_no_internal_links(files) == []


# ─── D-2: no private IPs ──────────────────────────────────────────────


class TestD2NoPrivateIPs:
    @pytest.mark.parametrize("ip", [
        "10.0.0.1", "10.20.30.40",
        "172.16.5.5", "172.31.255.1",
        "192.168.1.50", "192.168.255.255",
    ])
    def test_rfc1918_caught(self, fake_repo, ip):
        mod = _import()
        files = _scratch_doc(fake_repo,f"HOST=http://{ip}:8000\n")
        assert mod.check_d2_no_private_ips(files)

    @pytest.mark.parametrize("ip", [
        "8.8.8.8",         # public DNS
        "172.15.0.1",      # outside 172.16-31 range
        "172.32.0.1",      # outside 172.16-31 range
    ])
    def test_public_ip_clean(self, fake_repo, ip):
        mod = _import()
        files = _scratch_doc(fake_repo,f"see {ip}\n")
        assert mod.check_d2_no_private_ips(files) == []


# ─── D-3: no operator paths ───────────────────────────────────────────


class TestD3NoOperatorPaths:
    def test_home_sander_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"cp file /home/sander/data\n")
        assert mod.check_d3_no_operator_paths(files)

    def test_users_sander_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"/Users/sander/Documents\n")
        assert mod.check_d3_no_operator_paths(files)

    def test_placeholder_path_clean(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"cp file ${HOME}/data\n")
        assert mod.check_d3_no_operator_paths(files) == []


# ─── D-4: no server container names ───────────────────────────────────


class TestD4NoServerContainers:
    def test_vllm_server_mtp_test_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"docker logs vllm-server-mtp-test\n")
        assert mod.check_d4_no_server_container_names(files)

    def test_vllm_pn95_2xa5000_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"docker logs vllm-pn95-2xa5000-bench\n")
        assert mod.check_d4_no_server_container_names(files)

    def test_generic_name_clean(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"docker logs vllm-server\n")
        assert mod.check_d4_no_server_container_names(files) == []


# ─── D-5: no retired CLI verbs ────────────────────────────────────────


class TestD5NoRetiredVerbs:
    @pytest.mark.parametrize("verb", [
        "genesis doctor", "genesis verify", "genesis migrate",
    ])
    def test_retired_verb_caught(self, fake_repo, verb):
        mod = _import()
        files = _scratch_doc(fake_repo,f"run `{verb}` first\n")
        assert mod.check_d5_no_retired_verbs(files)

    def test_launch_script_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"./scripts/launch.sh my-key\n")
        assert mod.check_d5_no_retired_verbs(files)

    def test_sndr_verb_clean(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"run `sndr doctor` first\n")
        assert mod.check_d5_no_retired_verbs(files) == []


# ─── D-6: actionable TODO / placeholder / NotImplementedError markers ─


class TestD6Markers:
    """The refined D-6 only flags actionable markers, not the plain
    English noun "placeholder" used to describe a patch (e.g. PN64).
    """

    def test_todo_with_paren_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"TODO(sandermage): finish this\n")
        assert mod.check_d6_no_unresolved_todos(files)

    def test_fixme_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"FIXME: this is broken\n")
        assert mod.check_d6_no_unresolved_todos(files)

    def test_xxx_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"XXX investigate this\n")
        assert mod.check_d6_no_unresolved_todos(files)

    def test_placeholder_slot_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"fill in <PLACEHOLDER> here\n")
        assert mod.check_d6_no_unresolved_todos(files)

    def test_notimplementederror_bare_caught(self, fake_repo):
        mod = _import()
        files = _scratch_doc(fake_repo,"raises NotImplementedError on call\n")
        assert mod.check_d6_no_unresolved_todos(files)

    def test_notimplementederror_in_backticks_clean(self, fake_repo):
        mod = _import()
        # backticked = identifier reference, not unresolved marker
        files = _scratch_doc(
            fake_repo,
            "replaces `NotImplementedError` raise in upstream code\n",
        )
        assert mod.check_d6_no_unresolved_todos(files) == []

    def test_english_placeholder_clean(self, fake_repo):
        mod = _import()
        # PN64 is described as a "placeholder" in plain English prose;
        # this is legitimate noun usage, NOT an unresolved TODO.
        files = _scratch_doc(
            fake_repo,
            "PN64 — Marlin MoE sm_120 placeholder (env-gated)\n",
        )
        assert mod.check_d6_no_unresolved_todos(files) == []

    def test_allow_marker_skips_line(self, fake_repo):
        mod = _import()
        files = _scratch_doc(
            fake_repo,
            "TODO(sandermage): finish <!-- audit-public-docs: allow -->\n",
        )
        assert mod.check_d6_no_unresolved_todos(files) == []


# ─── D-7: stale-version-as-current (GATE-EXTEND) ──────────────────────


class TestD7StaleVersion:
    """GATE-EXTEND adds D-7 to catch current-state claims that anchor a
    non-current Genesis version (v7.5x or v11.0.x).

    Historical phrasings (`Removed in v11.0.0`, `renamed in v11.0.0`,
    `pre-v11 scripts`) must NOT be flagged."""

    @pytest.mark.parametrize("phrasing", [
        "## Quick start (canonical, v11.0.0+)",
        "### Repository layout (v11.0.0)",
        "vLLM v7.52 stack tested on A5000",
        "install.sh --pin v11.0",
        "## Quick start (canonical, v7.52+)",
    ])
    def test_stale_as_current_phrasings_caught(self, fake_repo, phrasing):
        mod = _import()
        files = _scratch_doc(fake_repo, phrasing + "\n")
        assert mod.check_d7_no_stale_version_as_current(files), (
            f"phrasing should be caught: {phrasing!r}"
        )

    @pytest.mark.parametrize("phrasing", [
        # Historical references — describe past events, not current state.
        "Removed in v11.0.0:",
        "namespace has been removed entirely in v11.0.0",
        "ImportError: pre-v11 scripts",
        "Update the script: rewrite imports `vllm._genesis.*` → ...",
        # Current version anchor — must NOT trigger.
        "## Quick start (canonical, v12.0.0)",
        "### Repository layout (v12.0.0)",
    ])
    def test_historical_or_current_clean(self, fake_repo, phrasing):
        mod = _import()
        files = _scratch_doc(fake_repo, phrasing + "\n")
        assert mod.check_d7_no_stale_version_as_current(files) == [], (
            f"phrasing should be clean: {phrasing!r}"
        )

    def test_transition_allowlist_suppresses_known_sites(
        self, fake_repo, monkeypatch
    ):
        """An entry in `_D7_TRANSITION_ALLOWLIST` for (rel, line)
        suppresses the otherwise-matching pattern at that line.
        Removed by CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL."""
        mod = _import()
        body = "skip\n## Quick start (canonical, v11.0.0+)\nfollow\n"
        # synthetic doc.md is `fake_repo / "doc.md"`.
        files = _scratch_doc(fake_repo, body)
        # Without allowlist — caught.
        assert mod.check_d7_no_stale_version_as_current(files)
        # Inject allowlist entry for ("doc.md", 2) — same as the line
        # the phrasing lives on after the leading "skip\n".
        monkeypatch.setattr(
            mod, "_D7_TRANSITION_ALLOWLIST",
            frozenset({("doc.md", 2)}),
        )
        assert mod.check_d7_no_stale_version_as_current(files) == []


# ─── D-8: stale-pin-as-current (GATE-EXTEND) ──────────────────────────


class TestD8StalePin:
    """GATE-EXTEND adds D-8 to catch current-state phrasings anchored to
    a pre-current vLLM pin (dev16, dev93, dev209, dev212).

    Historical pin references (BENCHMARKS Wave 7 snapshot, CHANGELOG,
    CREDITS attribution, "Previous v7.59 baseline") must NOT be
    flagged."""

    @pytest.mark.parametrize("phrasing", [
        "currently `0.20.1rc1.dev16+g7a1eb8ac2`",
        "pip install --pre vllm==0.20.1rc1.dev16+g7a1eb8ac2",
        "# vllm 0.20.1rc1.dev16+g7a1eb8ac2",
        "Not in nightly image as of dev93+g51f22dcfd.",
        "Primary — full v7.52 stack tested ... vLLM dev212+g7a1eb8ac2",
    ])
    def test_stale_pin_caught(self, fake_repo, phrasing):
        mod = _import()
        files = _scratch_doc(fake_repo, phrasing + "\n")
        assert mod.check_d8_no_stale_pin_as_current(files), (
            f"phrasing should be caught: {phrasing!r}"
        )

    @pytest.mark.parametrize("phrasing", [
        # Current pin — must NOT trigger.
        "vLLM `0.20.2rc1.dev371+gbf610c2f5`",
        "currently `0.20.2rc1.dev371+gbf610c2f5`",
        "pip install --pre vllm==0.20.2rc1.dev371+gbf610c2f5",
        # Historical narrative — should NOT trigger (file exempt OR
        # phrasing-level safe).
        "> Previous v7.59 baseline (2026-04-28): vLLM dev212+g8cd174fa3 era —",
        "Wave 7 / v7.72 (dev9) is pre-v11-rename",
    ])
    def test_current_pin_or_historical_clean(self, fake_repo, phrasing):
        mod = _import()
        files = _scratch_doc(fake_repo, phrasing + "\n")
        hits = mod.check_d8_no_stale_pin_as_current(files)
        # NOTE: the "Previous v7.59 baseline" line is exempted via the
        # permanent exempt list only when its rel_path:line matches
        # the production site. In the fake-repo unit test the file
        # is "doc.md:1", not "docs/CONFIGURATION.md:37" — so the
        # pattern WILL match here. The live-corpus test in
        # TestLiveCorpus validates the production exempt path.
        if "Previous v7.59 baseline" in phrasing:
            # Allow the unit-fake-repo to fire here; the real exempt is
            # validated in TestLiveCorpus.
            return
        assert hits == [], (
            f"phrasing should be clean: {phrasing!r}; got hits: {hits}"
        )

    def test_transition_allowlist_suppresses(self, fake_repo, monkeypatch):
        mod = _import()
        files = _scratch_doc(
            fake_repo,
            "skip\nbody: currently `0.20.1rc1.dev16+gabc`\n",
        )
        assert mod.check_d8_no_stale_pin_as_current(files)
        monkeypatch.setattr(
            mod, "_D8_TRANSITION_ALLOWLIST",
            frozenset({("doc.md", 2)}),
        )
        assert mod.check_d8_no_stale_pin_as_current(files) == []

    def test_permanent_exempt_suppresses(self, fake_repo, monkeypatch):
        mod = _import()
        files = _scratch_doc(
            fake_repo,
            "> Previous v7.59 baseline: vLLM dev212+gabc12345 era —\n",
        )
        assert mod.check_d8_no_stale_pin_as_current(files)
        monkeypatch.setattr(
            mod, "_D8_PERMANENT_EXEMPT",
            frozenset({("doc.md", 1)}),
        )
        assert mod.check_d8_no_stale_pin_as_current(files) == []

    def test_file_level_exempt_changelog(self, fake_repo):
        """File-level exempt covers CHANGELOG.md — every pin mention
        there is intentional engineering history."""
        mod = _import()
        target = fake_repo / "CHANGELOG.md"
        target.write_text(
            "## Wave 8 — Pinned to `0.20.1rc1.dev16+g7a1eb8ac2`\n"
        )
        files = [target]
        assert mod.check_d8_no_stale_pin_as_current(files) == []


# ─── Live committed corpus must be clean ──────────────────────────────


class TestLiveCorpus:
    """The actual public docs corpus in this repo must pass every check —
    this is what the gating-tier `make audit-public-docs` verifies on CI.

    D-7 and D-8 are kept clean via:
      - transition allowlists (sites scheduled for
        CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL fix)
      - permanent line-level exempts (genuine historical references)
      - file-level exempts (CHANGELOG.md, docs/CREDITS.md)
    """

    def test_all_checks_clean_on_repo(self):
        mod = _import()
        files = mod._gather_public_doc_files()
        for check_name, check_fn in [
            ("D-1", mod.check_d1_no_internal_links),
            ("D-2", mod.check_d2_no_private_ips),
            ("D-3", mod.check_d3_no_operator_paths),
            ("D-4", mod.check_d4_no_server_container_names),
            ("D-5", mod.check_d5_no_retired_verbs),
            ("D-6", mod.check_d6_no_unresolved_todos),
            ("D-7", mod.check_d7_no_stale_version_as_current),
            ("D-8", mod.check_d8_no_stale_pin_as_current),
        ]:
            hits = check_fn(files)
            assert hits == [], (
                f"{check_name} produced unexpected hits on live corpus:\n"
                + "\n".join(hits[:10])
            )
