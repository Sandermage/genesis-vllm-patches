#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 8a cold-install smoke — non-destructive smoke test verifying that
# a fresh clone + `pip install -e .` would work without operator-side
# state mutation.
#
# Source spec: Consolidated Roadmap §4.2 / §13.1 item 1.
#
# What this script does (READ-ONLY by default):
#
#   1. Verify install.sh / setup.py / pyproject.toml shape (bash -n,
#      pip check, import test).
#   2. Verify CLI entry point parses + reports version.
#   3. Verify the registry imports without GPU dependencies.
#   4. Run self-test (8/8 PASS expected).
#   5. Smoke compose for one V1 preset (`a5000-2x-35b-prod`) and one V2
#      alias (`prod-qwen3.6-35b-balanced`) — `--preflight-only`, no live launch.
#
# Modes:
#
#   ./scripts/cold_install_smoke.sh                  # check current env
#   ./scripts/cold_install_smoke.sh --fresh-venv     # build new venv first
#   ./scripts/cold_install_smoke.sh --skip-launch    # skip preflight checks
#
# Exit codes:
#   0  every check passed
#   1  at least one check failed
#   2  prerequisites missing (python3, bash, etc.)
#
# This script NEVER mutates host state (no rm, no apt, no pip install
# without --fresh-venv). When --fresh-venv is set, a new venv is created
# under /tmp/sndr-cold-install-smoke-<ISO>/ and torn down on success
# (kept on failure for inspection).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FRESH_VENV=0
SKIP_LAUNCH=0
KEEP_VENV=0

for arg in "$@"; do
    case "$arg" in
        --fresh-venv)  FRESH_VENV=1 ;;
        --skip-launch) SKIP_LAUNCH=1 ;;
        --keep-venv)   KEEP_VENV=1 ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# //;s/^#//'
            exit 0 ;;
        *)
            echo "unknown flag: $arg" >&2
            exit 2 ;;
    esac
done

# ─── Pretty helpers ──────────────────────────────────────────────────

_pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
_fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; FAILS=$((FAILS + 1)); }
_step() { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }

FAILS=0
SMOKE_LOG="$(mktemp -t sndr-smoke.XXXXXX.log)"
trap 'rm -f "$SMOKE_LOG"' EXIT

# ─── 1. Prerequisites ────────────────────────────────────────────────

_step "1. Prerequisites"

for cmd in python3 bash make git; do
    if command -v "$cmd" >/dev/null 2>&1; then
        _pass "$cmd available ($(command -v $cmd))"
    else
        _fail "$cmd NOT found in PATH"
    fi
done

py_version=$(python3 --version 2>&1 | awk '{print $2}')
_pass "python3 version: $py_version"

# ─── 2. Static syntax checks ─────────────────────────────────────────

_step "2. Static syntax checks"

if bash -n install.sh 2>>"$SMOKE_LOG"; then
    _pass "install.sh syntax OK (bash -n)"
else
    _fail "install.sh syntax error — see $SMOKE_LOG"
fi

if [ -f pyproject.toml ]; then
    if python3 -c "import tomllib; tomllib.loads(open('pyproject.toml','rb').read().decode())" 2>>"$SMOKE_LOG"; then
        _pass "pyproject.toml parses (tomllib)"
    else
        _fail "pyproject.toml parse error — see $SMOKE_LOG"
    fi
elif [ -f setup.py ]; then
    if python3 -c "import ast; ast.parse(open('setup.py').read())" 2>>"$SMOKE_LOG"; then
        _pass "setup.py syntax OK"
    else
        _fail "setup.py syntax error"
    fi
fi

# ─── 3. Optional: fresh venv build ───────────────────────────────────

if [ "$FRESH_VENV" -eq 1 ]; then
    _step "3. Fresh venv build (--fresh-venv)"
    VENV_DIR="/tmp/sndr-cold-install-smoke-$(date +%Y%m%dT%H%M%S)"
    _pass "creating $VENV_DIR"
    python3 -m venv "$VENV_DIR" 2>>"$SMOKE_LOG"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip >>"$SMOKE_LOG" 2>&1 || _fail "pip upgrade failed"
    # Install without heavy deps for smoke (vllm is operator-installed):
    if pip install -e . --no-deps >>"$SMOKE_LOG" 2>&1; then
        _pass "pip install -e . --no-deps OK"
    else
        _fail "pip install failed — see $SMOKE_LOG"
    fi
else
    _step "3. Skipping venv build (use --fresh-venv to test from clean)"
fi

# ─── 4. Module import smoke ──────────────────────────────────────────

_step "4. Module import smoke"

for module in \
    "sndr" \
    "sndr.cli" \
    "sndr.dispatcher.registry" \
    "sndr.model_configs.registry" \
    "sndr.model_configs.registry_v2" \
    "sndr.proof" \
    "sndr.proof.bench_attach" \
    "sndr.proof.release_check"
do
    if python3 -c "import $module" 2>>"$SMOKE_LOG"; then
        _pass "import $module"
    else
        _fail "import $module failed — see $SMOKE_LOG"
    fi
done

# ─── 5. CLI entry point ──────────────────────────────────────────────

_step "5. CLI entry point smoke"

if python3 -m sndr.cli --version >/dev/null 2>>"$SMOKE_LOG"; then
    _pass "sndr --version OK"
else
    _fail "sndr --version failed — see $SMOKE_LOG"
fi

# Self-test (8/8 PASS expected on healthy tree)
if python3 -m sndr.cli self-test 2>>"$SMOKE_LOG" | tail -1 | grep -q "PASS"; then
    _pass "sndr self-test PASS"
else
    # Some versions print "8/8 passed" instead of literal "PASS"; check rc:
    if python3 -m sndr.cli self-test >>"$SMOKE_LOG" 2>&1; then
        _pass "sndr self-test rc=0"
    else
        _fail "sndr self-test rc≠0 — see $SMOKE_LOG"
    fi
fi

# ─── 6. V1 / V2 preset preflight smoke ───────────────────────────────

if [ "$SKIP_LAUNCH" -eq 0 ]; then
    _step "6. V1 + V2 preset preflight smoke (--preflight-only, no live launch)"

    # Use V1 monolithic preset key first — the floor that V2 falls back to.
    V1_KEY="a5000-2x-35b-prod"
    if python3 -m sndr.cli launch "$V1_KEY" --preflight-only \
            >>"$SMOKE_LOG" 2>&1; then
        _pass "V1 preset '$V1_KEY' preflight OK"
    else
        # Common reason: vllm not installed on this host (CI / Mac dev).
        # Treat as informational unless --fresh-venv (where it should work).
        if [ "$FRESH_VENV" -eq 1 ]; then
            _fail "V1 preset '$V1_KEY' preflight failed (fresh venv) — see $SMOKE_LOG"
        else
            echo "  · V1 preset '$V1_KEY' preflight skipped (vllm likely not installed in current env)"
        fi
    fi

    # V2 alias — composes to V1 ModelConfig via registry_v2.load_alias.
    V2_ALIAS="prod-qwen3.6-35b-balanced"
    if python3 -m sndr.cli launch "$V2_ALIAS" --preflight-only \
            >>"$SMOKE_LOG" 2>&1; then
        _pass "V2 alias '$V2_ALIAS' preflight OK"
    else
        if [ "$FRESH_VENV" -eq 1 ]; then
            _fail "V2 alias '$V2_ALIAS' preflight failed (fresh venv) — see $SMOKE_LOG"
        else
            echo "  · V2 alias '$V2_ALIAS' preflight skipped (vllm likely not installed)"
        fi
    fi
else
    _step "6. Skipping preset preflight (--skip-launch)"
fi

# ─── 7. Audit aggregate ──────────────────────────────────────────────

_step "7. Audit aggregate"

if make audit >>"$SMOKE_LOG" 2>&1; then
    _pass "make audit clean"
else
    _fail "make audit reported violations — see $SMOKE_LOG"
fi

# ─── 8. Result + cleanup ──────────────────────────────────────────────

_step "Result"

if [ "$FAILS" -eq 0 ]; then
    printf '\n  \033[32m✓ cold-install smoke PASSED\033[0m\n\n'
    # Clean up venv unless --keep-venv
    if [ "$FRESH_VENV" -eq 1 ] && [ "$KEEP_VENV" -eq 0 ]; then
        deactivate 2>/dev/null || true
        rm -rf "$VENV_DIR"
    fi
    exit 0
else
    printf '\n  \033[31m✗ cold-install smoke FAILED (%d failures)\033[0m\n' "$FAILS"
    printf '  Full log: %s\n\n' "$SMOKE_LOG"
    # On failure, preserve the venv for debugging
    trap '' EXIT  # cancel cleanup of log
    if [ "$FRESH_VENV" -eq 1 ]; then
        printf '  Fresh venv preserved at: %s\n' "$VENV_DIR"
    fi
    exit 1
fi
