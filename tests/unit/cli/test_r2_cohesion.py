# SPDX-License-Identifier: Apache-2.0
"""v12 UX R2 — CLI cohesion: one coherent ``sndr`` surface, full back-compat.

R1 collapsed the no-args experience into the wizard. R2 closes the remaining
split-brain: the beginner verbs (``verify`` / ``pull`` / ``list-models`` /
``model-config``, plus ``doctor`` already promoted in v12) lived ONLY on the
legacy ``genesis`` tree and the ``sndr.compat.cli`` bridge. A novice reading
the docs typed ``sndr verify`` and hit ``invalid choice``.

This module pins the cohesion contract:

  1. Promotion parity — each promoted beginner verb on the canonical ``sndr``
     surface dispatches to the SAME compat implementation the legacy/compat
     entry points use (mock the impl, assert it was called with the tail).
  2. ``model pull`` spaced alias resolves to the same ``pull`` impl.
  3. Dotted + spaced both resolve — ``engines list`` (spaced, beginner-natural)
     resolves identically to ``engines.list`` (dotted, canonical); same for
     ``engines info`` / ``pins list``.
  4. ``genesis <verb>`` still works AND emits exactly one soft-deprecation note
     on stderr; the deprecation note must NOT leak into stdout.
  5. R1 behaviors (no-args wizard gate, ``run``/``chat``) stay intact — covered
     by their own modules; here we only assert the new verbs did not displace
     the existing registry entries.

No implementation logic is rewritten: the legacy/compat code path remains the
single source of truth, so the canonical and ``genesis`` entry points cannot
drift.
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

# The promoted beginner verbs and the compat subcommand they each forward to.
# (Identity here — the canonical verb name equals the compat subcommand name.)
_BEGINNER_VERBS = ("verify", "doctor", "pull", "list-models", "model-config")


# ── 1. Promotion parity: canonical verb → same compat impl ──────────────────


class TestBeginnerVerbsPromoted:
    @pytest.mark.parametrize("verb", _BEGINNER_VERBS)
    def test_verb_registered_on_canonical_surface(self, verb):
        # The verb must appear in the canonical COMMAND_REGISTRY so it shows in
        # ``sndr --help`` and resolves (no more ``invalid choice``).
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()  # the real registration path
        assert verb in COMMAND_REGISTRY, (
            f"`sndr {verb}` must resolve on the canonical surface"
        )

    @pytest.mark.parametrize("verb", _BEGINNER_VERBS)
    def test_verb_dispatches_to_same_compat_impl(self, verb, monkeypatch):
        # The canonical pass-through must route through ``sndr.compat.cli.main``
        # with the verb as the first token — the SAME entry the legacy
        # ``genesis`` tree uses — so the two surfaces cannot drift.
        import sndr.compat.cli as compat_cli
        from sndr.cli.main import main

        seen: dict[str, list[str]] = {}

        def fake_main(argv):
            seen["argv"] = list(argv)
            return 0

        monkeypatch.setattr(compat_cli, "main", fake_main)
        rc = main([verb, "--flag", "x"])
        assert rc == 0
        assert seen.get("argv") == [verb, "--flag", "x"], (
            f"`sndr {verb}` must forward verbatim to compat.cli.main([{verb!r}, ...])"
        )

    def test_model_pull_spaced_alias_hits_pull_impl(self, monkeypatch):
        # ``sndr model pull <args>`` is the beginner-natural spaced form of the
        # ``pull`` verb; it must resolve to the SAME pull implementation.
        import sndr.compat.models.pull as pull_mod
        from sndr.cli.main import main

        seen: dict[str, list[str]] = {}

        def fake_main(argv):
            seen["argv"] = list(argv)
            return 0

        monkeypatch.setattr(pull_mod, "main", fake_main)
        rc = main(["model", "pull", "Qwen/Qwen3-32B"])
        assert rc == 0
        assert seen.get("argv") == ["Qwen/Qwen3-32B"], (
            "`sndr model pull X` must forward to the pull impl with the tail"
        )


# ── 2. Dotted + spaced both resolve ─────────────────────────────────────────


class TestDottedAndSpacedResolve:
    @pytest.mark.parametrize(
        "spaced,dotted",
        [
            (["engines", "list"], "engines.list"),
            (["engines", "info"], "engines.info"),
            (["pins", "list"], "pins.list"),
        ],
    )
    def test_spaced_resolves_to_same_command_as_dotted(self, spaced, dotted, monkeypatch):
        # The spaced form (beginner-natural, Docker/git style) must resolve to
        # the SAME command object as the canonical dotted name.
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser, main

        build_parser()
        assert dotted in COMMAND_REGISTRY

        calls: dict[str, int] = {}

        def fake_execute(self, args):  # noqa: ANN001
            calls["hit"] = calls.get("hit", 0) + 1
            return 0

        target_cls = type(COMMAND_REGISTRY[dotted])
        monkeypatch.setattr(target_cls, "execute", fake_execute)

        # The spaced verb (with a positional, e.g. engines info vllm) must reach
        # the same command's execute().
        argv = list(spaced)
        if dotted == "engines.info":
            argv.append("vllm")
        rc = main(argv)
        assert rc == 0
        assert calls.get("hit"), f"`sndr {' '.join(spaced)}` must reach {dotted} execute()"

    @pytest.mark.parametrize(
        "bare,dotted",
        [
            (["engines"], "engines.list"),
            (["pins"], "pins.list"),
        ],
    )
    def test_bare_group_verb_defaults_to_list(self, bare, dotted, monkeypatch):
        # A beginner who types just ``sndr engines`` / ``sndr pins`` (no
        # subcommand) should get the obvious "show me" — the group's ``list`` —
        # not a raw argparse ``invalid choice`` wall. Mirrors R1's bare-verb
        # philosophy (bare ``sndr`` → wizard).
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser, main

        build_parser()
        calls: dict[str, int] = {}

        def fake_execute(self, args):  # noqa: ANN001
            calls["hit"] = calls.get("hit", 0) + 1
            return 0

        monkeypatch.setattr(type(COMMAND_REGISTRY[dotted]), "execute", fake_execute)
        rc = main(list(bare))
        assert rc == 0
        assert calls.get("hit"), f"bare `sndr {bare[0]}` must default to {dotted}"

    def test_bare_group_normalization_is_surgical(self):
        # The bare-group default rewrites ONLY a bare group verb (or group +
        # leading flags) — never a real spaced subcommand or a positional.
        from sndr.cli.main import _normalize_spaced_verbs

        # bare → list
        assert _normalize_spaced_verbs(["engines"]) == ["engines.list"]
        assert _normalize_spaced_verbs(["pins"]) == ["pins.list"]
        # group + leading flag → list (flag rides through)
        assert _normalize_spaced_verbs(["pins", "--output", "json"]) == [
            "pins.list", "--output", "json",
        ]
        # real spaced subcommands untouched (resolve via the existing aliases)
        assert _normalize_spaced_verbs(["engines", "list"]) == ["engines.list"]
        assert _normalize_spaced_verbs(["engines", "info", "vllm"]) == [
            "engines.info", "vllm",
        ]
        # a non-group verb is never touched
        assert _normalize_spaced_verbs(["doctor"]) == ["doctor"]

    def test_spaced_engines_list_help_does_not_error(self):
        # ``sndr engines list`` must not raise ``invalid choice``.
        # ``engines list`` reaches the modular engines schemas (pydantic
        # models); the light CI test leg installs no pydantic, so the
        # subprocess would exit on ModuleNotFoundError before argparse ever
        # resolves the verb. Skip cleanly there — matching the
        # ``importorskip("pydantic")`` convention used across the modular
        # domain tests — rather than failing on an unrelated env gap.
        pytest.importorskip("pydantic")

        import subprocess
        import sys

        rc = subprocess.run(
            [sys.executable, "-m", "sndr.cli.main", "engines", "list"],
            capture_output=True, text=True,
        )
        assert "invalid choice" not in rc.stderr
        assert rc.returncode == 0


# ── 3. genesis soft-deprecation ─────────────────────────────────────────────


class TestGenesisSoftDeprecation:
    def test_genesis_verb_emits_single_deprecation_note_on_stderr(self, monkeypatch):
        # ``genesis <verb>`` (the genesis console entry) must still fully work
        # AND print exactly one deprecation note on stderr (never stdout).
        from sndr.cli.legacy import genesis_main

        # Stub the bridge target so the test is offline + deterministic.
        import sndr.compat.cli as compat_cli
        monkeypatch.setattr(compat_cli, "main", lambda argv: 0)

        err = io.StringIO()
        out = io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = genesis_main(["verify", "--quick"])
        assert rc == 0
        note = err.getvalue()
        assert "deprecated" in note.lower()
        assert "genesis verify" in note and "sndr verify" in note
        assert "removed in v13" in note
        assert note.lower().count("deprecated") == 1, "exactly one deprecation note"
        # Must not pollute stdout.
        assert "deprecated" not in out.getvalue().lower()

    def test_genesis_help_and_bare_stay_quiet(self, monkeypatch):
        # Bare introspection (``genesis --help`` / bare ``genesis``) must NOT
        # print the nudge — scripted help-scrapers stay clean.
        from sndr.cli.legacy import genesis_main

        for probe in (["--help"], ["-h"], []):
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                try:
                    genesis_main(list(probe))
                except SystemExit:
                    pass
            assert "deprecated" not in err.getvalue().lower(), (
                f"`genesis {' '.join(probe)}` must not warn"
            )

    def test_genesis_version_flag_quiet_at_note_layer(self):
        # The note layer itself exempts ``--version`` (whether or not the
        # downstream version dispatch happens to work).
        import io as _io
        from contextlib import redirect_stderr as _rse
        from sndr.cli.legacy import _emit_genesis_deprecation_note

        for probe in (["--version"], ["--help"], ["-h"], ["help"], []):
            err = _io.StringIO()
            with _rse(err):
                _emit_genesis_deprecation_note(list(probe))
            assert err.getvalue() == "", f"{probe!r} must stay quiet"

    def test_genesis_still_dispatches_after_note(self, monkeypatch):
        from sndr.cli.legacy import genesis_main
        import sndr.compat.cli as compat_cli

        seen: dict[str, list[str]] = {}
        monkeypatch.setattr(
            compat_cli, "main",
            lambda argv: (seen.__setitem__("argv", list(argv)) or 0),
        )
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            rc = genesis_main(["doctor", "--full"])
        assert rc == 0
        assert seen.get("argv") == ["doctor", "--full"], (
            "deprecation note must not change what genesis dispatches"
        )

    def test_sndr_canonical_does_not_emit_deprecation_note(self, monkeypatch):
        # The note is a *genesis*-only nudge; the canonical sndr surface must
        # stay quiet (no double-warning when sndr internally reuses the bridge).
        import sndr.compat.cli as compat_cli
        from sndr.cli.main import main

        monkeypatch.setattr(compat_cli, "main", lambda argv: 0)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = main(["verify", "--quick"])
        assert rc == 0
        assert "deprecated" not in err.getvalue().lower()


# ── 4. R1 behaviors unbroken ────────────────────────────────────────────────


class TestR1Unbroken:
    def test_run_and_chat_still_registered(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()
        assert "run" in COMMAND_REGISTRY
        assert "chat" in COMMAND_REGISTRY

    def test_existing_promoted_v12_commands_intact(self):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()
        for name in ("report", "doctor", "preset", "bench", "tune", "config"):
            assert name in COMMAND_REGISTRY
