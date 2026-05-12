#!/usr/bin/env bash
# CI-gate: forbids legacy imports in active code.
#
# Etap 5.4/5.5 (audit 2026-05-12): delegates to the Python AST-based
# scanner `scripts/check_no_legacy_imports.py`. The old shell regex
# missed `from vllm.sndr_core import patches` and only scanned .py/.sh/.md;
# the Python scanner covers every import shape plus YAML/TOML/JSON.
#
# Kept as a `.sh` entry point because pre-commit and Makefile already
# invoke it via `bash`; the heavy lifting now lives in Python.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
exec python3 scripts/check_no_legacy_imports.py "$@"
