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
        "192.168.1.10", "192.168.255.255",
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


# ─── Live committed corpus must be clean ──────────────────────────────


class TestLiveCorpus:
    """The actual public docs corpus in this repo must pass every check —
    this is what the gating-tier `make audit-public-docs` verifies on CI.
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
        ]:
            hits = check_fn(files)
            assert hits == [], (
                f"{check_name} produced unexpected hits on live corpus:\n"
                + "\n".join(hits[:10])
            )
