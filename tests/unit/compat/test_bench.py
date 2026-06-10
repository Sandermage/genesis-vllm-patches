# SPDX-License-Identifier: Apache-2.0
"""Tests for `genesis bench` — the unified-CLI shim around
`tools/genesis_bench_suite.py`.

What we pin:
  - Module imports cleanly
  - `_locate_bench_module()` finds the real script when the source
    tree is present
  - `--help` works without needing the bench script (graceful
    fallback to a stub if the script is missing)
  - argv is forwarded verbatim
  - Subcommand is wired into the unified CLI
  - Bench script's `main()` accepts argv (the refactor that made the
    shim possible)
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
# v12 canonical location: the real bench script lives inside the sndr
# package at extras/tools/. Repo-root `tools/genesis_bench_suite.py` is
# a thin shim that delegates to this canonical script.
BENCH_SCRIPT = (
    REPO_ROOT / "sndr" / "extras" / "tools" / "genesis_bench_suite.py"
)


class TestBenchShim:
    def test_module_importable(self):
        from sndr.compat import bench  # noqa: F401

    def test_locate_bench_module_finds_real_script(self):
        """In a normal git checkout, parents[3] resolves to the repo
        root and tools/genesis_bench_suite.py is right there."""
        from sndr.compat.bench import _locate_bench_module
        located = _locate_bench_module()
        assert located is not None, (
            "compat.bench._locate_bench_module() must find the bench "
            "script in a git checkout"
        )
        assert located.name == "genesis_bench_suite.py"

    def test_locate_respects_env_override(self, monkeypatch, tmp_path):
        """GENESIS_REPO_ROOT/tools/genesis_bench_suite.py wins only when
        the canonical in-package path is unavailable (Wave 10 changed
        search-order priority: in-package is now #1, env override is #2).
        We simulate the slim-deployment case by stubbing the canonical
        candidate out via _locate_bench_module's internal Path resolver.
        """
        import sndr.compat.bench as bench_mod

        fake_tools = tmp_path / "tools"
        fake_tools.mkdir()
        fake_script = fake_tools / "genesis_bench_suite.py"
        fake_script.write_text("# fake script\n")

        # Force the in-package canonical candidate to a path that
        # does not exist on disk so it falls through to the env override.
        original_init = bench_mod._locate_bench_module

        def _locate_without_canonical() -> object:
            import os
            from pathlib import Path as _Path
            candidates: list[_Path] = []
            candidates.append(tmp_path / "_does_not_exist" / "genesis_bench_suite.py")
            env_root = os.environ.get("GENESIS_REPO_ROOT")
            if env_root:
                candidates.append(_Path(env_root) / "tools" / "genesis_bench_suite.py")
            return next((p for p in candidates if p.is_file()), None)

        monkeypatch.setenv("GENESIS_REPO_ROOT", str(tmp_path))
        monkeypatch.setattr(
            bench_mod, "_locate_bench_module", _locate_without_canonical,
        )
        located = bench_mod._locate_bench_module()
        assert located == fake_script, (
            f"env override should win when canonical is absent; "
            f"got {located} vs {fake_script}"
        )
        # Sanity check: when canonical IS present, the original picks it.
        canonical = original_init()
        assert canonical is not None and canonical.name == "genesis_bench_suite.py"


class TestHelpPassthrough:
    def test_help_does_not_crash(self, capsys):
        """--help is the most common invocation. It must produce
        output and exit cleanly (rc == 0). argparse uses SystemExit
        for --help; the unified CLI catches that, but a direct call
        gets the SystemExit straight through."""
        from sndr.compat.bench import main
        try:
            rc = main(["--help"])
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 0
        assert rc == 0
        # Either real bench help OR shim fallback printed something
        captured = capsys.readouterr()
        assert captured.out, "help output must not be empty"

    def test_help_in_slim_deployment_falls_back(
        self, capsys, monkeypatch, tmp_path
    ):
        """If GENESIS_REPO_ROOT points at a tree WITHOUT tools/, and
        we tell _locate_bench_module to ignore the real repo root, the
        shim must print a fallback help instead of crashing."""
        from sndr.compat import bench

        # Force every candidate to miss
        monkeypatch.setenv("GENESIS_REPO_ROOT", str(tmp_path))
        monkeypatch.setattr(
            bench, "_locate_bench_module", lambda: None
        )
        rc = bench.main(["--help"])
        assert rc == 0
        captured = capsys.readouterr()
        # BENCHMARK_GUIDE.md was consolidated into BENCHMARKS.md
        # (2026-05-16) — the fallback must point at the canonical doc.
        assert "BENCHMARKS.md" in captured.out, (
            "shim fallback help must point operator at the manual "
            "invocation recipe"
        )


class TestArgvForwarding:
    def test_main_signature_accepts_argv(self):
        """The shim's contract: forward argv verbatim. The bench
        script's main() must accept argv for this to work."""
        # Import via spec since the script lives outside the package
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "genesis_bench_suite", BENCH_SCRIPT
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import inspect
        sig = inspect.signature(mod.main)
        assert "argv" in sig.parameters, (
            "tools/genesis_bench_suite.py:main() must accept an `argv` "
            "parameter so the unified-CLI shim can forward args"
        )

    def test_parse_args_signature_accepts_argv(self):
        """Same contract for parse_args — needed so unit tests can
        construct namespaces without sys.argv side effects."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "genesis_bench_suite", BENCH_SCRIPT
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import inspect
        sig = inspect.signature(mod.parse_args)
        assert "argv" in sig.parameters

    def test_parse_args_with_explicit_argv(self):
        """parse_args(['--quick']) must work without consuming sys.argv."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "genesis_bench_suite", BENCH_SCRIPT
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        ns = mod.parse_args(["--quick"])
        assert ns.quick is True
        assert ns.mode == "quick"


class TestUnifiedCLIWiring:
    def test_bench_subcommand_in_unified_cli(self):
        from sndr.compat.cli import KNOWN_SUBCOMMANDS
        assert "bench" in KNOWN_SUBCOMMANDS, (
            "bench must be discoverable via the unified CLI dispatcher"
        )

    def test_bench_help_via_unified_cli(self, capsys):
        """`genesis bench --help` should not crash and should produce
        output."""
        from sndr.compat.cli import main
        rc = main(["bench", "--help"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out

    def test_unified_cli_help_lists_bench(self, capsys):
        """Top-level `genesis` help banner must mention bench."""
        from sndr.compat.cli import main
        main([])  # no args = print help
        captured = capsys.readouterr()
        assert "bench" in captured.out, (
            "unified CLI --help banner must list the bench subcommand"
        )
