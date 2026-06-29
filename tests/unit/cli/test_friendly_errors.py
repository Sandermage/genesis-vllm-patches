# SPDX-License-Identifier: Apache-2.0
"""v12 UX R5 — friendly CLI errors for an unknown command / typo'd verb.

R1-R4 made the happy path a one-liner; R5 makes the *unhappy* path equally
kind. Before R5 a novice who typed ``sndr lauch`` hit a raw argparse wall::

    sndr: error: argument COMMAND: invalid choice: 'lauch' (choose from
    'bench', 'chat', 'config', ...)

— a dump of every verb with no pointer to the next action. R5 intercepts the
unknown command before argparse and prints a short, rustup-style message that
matches the ``install.sh`` tone:

    sndr: unknown command 'lauch' — did you mean 'launch'?
          Run 'sndr' for the guided menu or 'sndr --help' for all commands.

Contract under test:
  1. an unknown verb close to a real one names the nearest match AND the
     next step, and exits non-zero;
  2. a typo with no close match still prints the next-step pointer (no bogus
     suggestion) and exits non-zero;
  3. a VALID command is never intercepted (no friendly-error text, no extra
     exit) — R1-R4 behaviour is untouched;
  4. ``--help`` / ``--version`` / a bare ``sndr`` are never treated as an
     unknown command;
  5. the spaced aliases (``engines list``) and promoted verbs (``doctor``)
     still resolve — the interceptor must not shadow them.
"""
from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

import sndr.cli.main as cli_main  # noqa: E402
from sndr.cli.main import main  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_QUICKSTART = _REPO_ROOT / "docs" / "QUICKSTART.md"


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run ``main(argv)`` capturing (rc, stdout, stderr). argparse raises
    ``SystemExit`` for a real ``invalid choice``; the friendly path must
    return a code instead, so a SystemExit here is itself a contract failure
    for the unknown-command cases."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


# ── 1. close typo → nearest match + next step + non-zero ────────────────────


class TestCloseTypoSuggestion:
    def test_lauch_suggests_launch(self):
        rc, _out, err = _run(["lauch"])
        assert rc != 0, "an unknown command must exit non-zero"
        low = err.lower()
        assert "lauch" in err, "the offending token must be echoed back"
        assert "did you mean" in low and "launch" in err, (
            f"must suggest the nearest verb 'launch'; got: {err!r}"
        )

    def test_suggestion_points_to_next_step(self):
        _rc, _out, err = _run(["lauch"])
        # The next-step hint: the guided menu and/or --help. A true beginner
        # must always know what to type next.
        assert "sndr --help" in err, "must point at `sndr --help`"
        assert "sndr'" in err or "`sndr`" in err or "sndr for" in err.lower(), (
            f"must point at the bare `sndr` guided menu; got: {err!r}"
        )

    @pytest.mark.parametrize(
        "typo,expected",
        [
            ("doctr", "doctor"),
            ("opne", "open"),
            ("verfy", "verify"),
            ("rnu", "run"),
        ],
    )
    def test_other_close_typos_suggest_right_verb(self, typo, expected):
        rc, _out, err = _run([typo])
        assert rc != 0
        assert expected in err, f"`sndr {typo}` should suggest {expected!r}: {err!r}"


# ── 1b. muscle-memory synonym → the canonical verb (curated, beats fuzzy) ───


class TestSynonymHints:
    @pytest.mark.parametrize(
        "synonym,expected",
        [
            ("serve", "run"),
            ("start", "run"),
            ("stop", "down"),
            ("status", "engines"),
            ("ps", "engines"),
            ("models", "list-models"),
            ("ls", "list-models"),
        ],
    )
    def test_muscle_memory_synonym_points_to_canonical(self, synonym, expected):
        # A Docker/Ollama/systemd reflex (serve / ps / stop / models …) should
        # be pointed at the real sndr verb, not left as a bare "unknown command"
        # or a misleading fuzzy match.
        rc, _out, err = _run([synonym])
        assert rc != 0, "an unknown command must still exit non-zero"
        assert synonym in err, "the offending token must be echoed back"
        assert expected in err, f"`sndr {synonym}` should point to {expected!r}: {err!r}"

    def test_curated_synonym_beats_misleading_fuzzy_match(self):
        # Before the synonym table, difflib matched 'ps' → 'pins' and
        # 'models' → 'model' — both wrong. The curated hint must win.
        _rc, _out, err_ps = _run(["ps"])
        assert "engines" in err_ps and "pins" not in err_ps, (
            f"'ps' must point to 'engines', not 'pins': {err_ps!r}"
        )
        _rc, _out, err_models = _run(["models"])
        assert "list-models" in err_models, (
            f"'models' must point to 'list-models': {err_models!r}"
        )


# ── 2. no close match → next-step pointer, no bogus suggestion ──────────────


class TestGlobalFlagBeforeVerb:
    def test_global_output_flag_before_unknown_verb_stays_friendly(self):
        # `sndr --output json bogusverb` — a global flag before the verb must NOT
        # bypass the R5 friendly handler back into the raw argparse wall.
        rc, _out, err = _run(["--output", "json", "bogusverb"])
        assert rc != 0
        assert "bogusverb" in err
        assert "invalid choice" not in err.lower(), (
            f"global flag must not bypass the friendly handler: {err!r}"
        )
        assert "unknown command" in err.lower()


class TestNoCloseMatch:
    def test_nonsense_token_still_helps(self):
        rc, _out, err = _run(["zzqqxx"])
        assert rc != 0, "an unknown command must exit non-zero"
        assert "zzqqxx" in err
        # No fabricated suggestion when nothing is close.
        assert "did you mean" not in err.lower(), (
            f"must NOT fabricate a suggestion for a far-off token; got: {err!r}"
        )
        # But the operator must still get the next step.
        assert "sndr --help" in err and ("sndr'" in err or "`sndr`" in err), (
            f"a no-match token must still point at the menu + help; got: {err!r}"
        )


# ── 3. valid commands are never intercepted ─────────────────────────────────


class TestValidCommandsNotIntercepted:
    def test_health_runs_normally(self, monkeypatch):
        # health is a real command — it must execute, not hit the friendly
        # error path. (Force non-interactive so no wizard side effects.)
        monkeypatch.setattr(cli_main, "_interactive_no_args", lambda: False)
        rc, out, err = _run(["health"])
        assert rc == 0
        assert "unknown command" not in err.lower()
        assert "did you mean" not in err.lower()

    def test_spaced_alias_not_intercepted(self, monkeypatch):
        # `engines list` (spaced alias of engines.list) must resolve, not be
        # flagged as unknown.
        calls = {}

        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()

        def fake_execute(self, args):  # noqa: ANN001
            calls["hit"] = True
            return 0

        monkeypatch.setattr(type(COMMAND_REGISTRY["engines.list"]), "execute", fake_execute)
        rc, _out, err = _run(["engines", "list"])
        assert rc == 0
        assert calls.get("hit"), "spaced alias must reach the real command"
        assert "unknown command" not in err.lower()

    def test_promoted_verb_not_intercepted(self, monkeypatch):
        # A promoted pass-through verb (doctor) must forward, not be flagged.
        import sndr.compat.cli as compat_cli

        seen = {}
        monkeypatch.setattr(
            compat_cli, "main",
            lambda argv: (seen.__setitem__("argv", list(argv)) or 0),
        )
        rc, _out, err = _run(["doctor", "--full"])
        assert rc == 0
        assert seen.get("argv") == ["doctor", "--full"]
        assert "unknown command" not in err.lower()


# ── 4. introspection flags / bare sndr never flagged ────────────────────────


class TestIntrospectionNotFlagged:
    def test_help_flag_not_flagged(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit) as exc:
                main(["--help"])
        assert exc.value.code == 0
        assert "unknown command" not in out.getvalue().lower()

    def test_version_flag_not_flagged(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit) as exc:
                main(["--version"])
        assert exc.value.code == 0
        assert "unknown command" not in out.getvalue().lower()

    def test_bare_sndr_non_tty_prints_help_not_error(self, monkeypatch):
        monkeypatch.setattr(cli_main, "_interactive_no_args", lambda: False)
        rc, out, err = _run([])
        assert rc == 0
        assert "unknown command" not in (out + err).lower()


# ── 5. legacy `--version` no longer crashes (UX R5 one-line fix) ─────────────


class TestLegacyVersionDoesNotCrash:
    """The legacy `genesis --version` path did `from sndr import
    SNDR_CORE_VERSION` — a symbol the top-level package does NOT export — so it
    raised ImportError. R5 sources it from `sndr.version` (the single source of
    truth, where it IS defined)."""

    def test_legacy_version_prints_and_exits_zero(self):
        from sndr.cli.legacy import cli_main as legacy_main

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = legacy_main(["--version"])
        assert rc == 0
        from sndr.version import SNDR_CORE_VERSION

        assert SNDR_CORE_VERSION in out.getvalue()
        assert "SNDR Core" in out.getvalue()


# ── 6. QUICKSTART beginner path is internally consistent ────────────────────


def _beginner_section() -> str:
    """The literal '5-minute path' section of QUICKSTART (everything before the
    '## More' advanced section). A true beginner must be able to copy-paste only
    from here and reach a chat prompt."""
    text = _QUICKSTART.read_text(encoding="utf-8")
    head, _sep, _tail = text.partition("\n## More")
    return head


def _shell_commands(markdown: str) -> list[str]:
    """Every line inside a ```bash fenced block of the given markdown."""
    cmds: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", markdown, flags=re.S):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()  # drop trailing comments
            if line:
                cmds.append(line)
    return cmds


class TestQuickstartConsistency:
    def test_quickstart_exists_and_has_more_section(self):
        text = _QUICKSTART.read_text(encoding="utf-8")
        assert "## The 5-minute path" in text
        assert "## More" in text, "advanced commands belong under a 'More' section"

    def test_beginner_path_uses_only_sndr_not_legacy_surfaces(self):
        section = _beginner_section()
        # No retired / non-canonical entry points in the path a novice copies.
        for forbidden in (
            "genesis doctor", "genesis verify", "genesis run", "genesis launch",
            "python -m sndr.compat.cli", "python3 -m sndr.compat.cli",
            "./scripts/launch.sh",
        ):
            assert forbidden not in section, (
                f"beginner path must not reference {forbidden!r} — use `sndr ...`"
            )

    def test_beginner_path_leads_with_install_then_run(self):
        cmds = _shell_commands(_beginner_section())
        assert cmds, "the 5-minute path must contain runnable commands"
        joined = "\n".join(cmds)
        assert "install.sh" in cmds[0], "first command must be the install one-liner"
        # The single simplest next step is `sndr run` (terminal) with the GUI
        # path (`sndr up` / `sndr open`) offered alongside.
        assert "sndr run" in joined, "must show `sndr run` as the simplest path"
        assert "sndr up" in joined and "sndr open" in joined, (
            "must show the GUI path (`sndr up` + `sndr open`)"
        )

    def test_every_sndr_verb_in_quickstart_resolves(self):
        # Every `sndr <verb>` shown in a RUNNABLE command block of QUICKSTART
        # must resolve on the real surface — no doc-only ghost commands. We scan
        # the fenced ```bash blocks (where copy-paste commands live), not prose
        # (so phrasing like "sndr is up" is not mistaken for a `sndr is` verb).
        from sndr.cli.main import _known_verbs

        known = set(_known_verbs())
        text = _QUICKSTART.read_text(encoding="utf-8")
        cmds = _shell_commands(text)  # all bash-fence command lines
        verbs: list[str] = []
        for line in cmds:
            m = re.match(r"sndr\s+([a-z][a-z0-9._-]*)", line)
            if m:
                verbs.append(m.group(1))
        # Spaced-alias second tokens (`engines list`) resolve via the alias map;
        # the leading token is what must be a known verb.
        unknown = sorted({v for v in verbs if v not in known})
        assert not unknown, (
            f"QUICKSTART command blocks reference sndr verbs that don't "
            f"resolve: {unknown}"
        )
        assert verbs, "expected at least some `sndr ...` commands in QUICKSTART"
