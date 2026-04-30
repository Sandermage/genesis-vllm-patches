# Genesis self-test

> **Operator question this answers:** "Is Genesis itself working on my box?"
>
> Different from `doctor`, which answers "is my SYSTEM healthy?".
> A `doctor` failure can be hardware/config; a `self-test` failure is a
> Genesis bug or a botched install.

## When to run

- Right after a fresh `git pull` on a Genesis checkout
- Right after a vLLM pin bump
- When a contributor sends a draft patch and you want a fast structural sanity check before reading the diff
- As an extra gate inside CI (already wired in `.github/workflows/test.yml`)

## Quick reference

```bash
# Default: verbose, color-free, copy-paste friendly
python3 -m vllm._genesis.compat.cli self-test

# Show only fail/warn/skip rows (good for `git pull` muscle memory)
python3 -m vllm._genesis.compat.cli self-test --quiet

# Machine-readable
python3 -m vllm._genesis.compat.cli self-test --json
```

Legacy form `python3 -m vllm._genesis.compat.self_test` still works.

## What it checks

Eight structural checks run in order. All run regardless of failures —
self-test is designed to never crash and to surface every problem in
one pass.

| # | Check | What it verifies |
|---|---|---|
| 1 | **version constant** | `vllm._genesis.__version__` is a non-empty string |
| 2 | **compat imports** | All 18 `vllm._genesis.compat.*` modules import cleanly |
| 3 | **wiring imports** | All `vllm/_genesis/wiring/patch_*.py` modules import; SKIPPED if `vllm` not installed in this env |
| 4 | **schema validator** | `PATCH_REGISTRY` validates against `schemas/patch_entry.schema.json` (no errors) |
| 5 | **lifecycle audit** | Every entry has a known lifecycle state; no unknown states |
| 6 | **categories build** | The categories index builds without errors and every patch is placed in at least one category |
| 7 | **predicates evaluator** | Every `applies_to` clause in the registry can be evaluated against an empty environment without raising |
| 8 | **schema file** | `schemas/patch_entry.schema.json` is parseable and has the required keys; SKIPPED in slim deployments where the source tree is not mounted |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All `fail`-class checks passed; safe to proceed |
| 1 | At least one check returned `fail`; **operator action required** |

`warn` and `skip` rows do not change the exit code. They're informational.

## Status legend

| Symbol | Status | Meaning |
|---|---|---|
| ✓ | `pass` | Check ran and verified the invariant |
| ✗ | `fail` | Check ran and the invariant is violated — **fix required** |
| ⚠ | `warn` | Check ran but found a non-fatal issue (e.g. schema file missing optional keys) |
| • | `skip` | Check could not run in this environment (e.g. `vllm` not installed, source tree not mounted) — not an error |

## Container deployments

When you run self-test inside a container that mounts only the
`vllm/_genesis/` package (without the source tree), the schema file
check returns `skip` rather than `fail`. The `schemas/` directory is a
repo-only artifact that does not ship with the installed package.

If you have the source tree available somewhere outside the package
location, point self-test at it via env var:

```bash
GENESIS_REPO_ROOT=/path/to/genesis-vllm-patches \
    python3 -m vllm._genesis.compat.cli self-test
```

The check now returns `pass` if the schema file is found at
`$GENESIS_REPO_ROOT/schemas/patch_entry.schema.json`.

Verified inside the live `vllm-server-mtp-test` PROD container:
**7 pass, 0 fail, 1 skip, exit 0**.

## JSON output for CI

```bash
python3 -m vllm._genesis.compat.cli self-test --json
```

returns:

```json
{
  "checks": [
    {"name": "version constant", "status": "pass", "message": "version: v7.63.x"},
    {"name": "compat imports", "status": "pass", "message": "18 compat modules import cleanly"},
    ...
  ],
  "summary": {
    "passed": 7,
    "failed": 0,
    "warned": 0,
    "skipped": 1,
    "total": 8
  }
}
```

This is the format the GitHub Actions workflow consumes (and what any
external monitoring should consume — do not parse the human-readable
table; it's not a contract).

## Adding a new check

Self-test checks live in `vllm/_genesis/compat/self_test.py`. To add
one:

1. Write a function `_check_<name>() -> tuple[str, str]` that returns
   `(status, message)` where `status ∈ {"pass", "fail", "warn", "skip"}`.
2. Append it to the `_CHECKS` list at the bottom of the module.
3. Add a unit test in `vllm/_genesis/tests/compat/test_self_test.py`
   that pins the new check's name (so a rename surfaces immediately).

The contract: a check **must never raise**. The `_check()` wrapper
converts any uncaught exception into `("fail", "<ExceptionType>: <msg>")`,
but checks should aim to return a clean status themselves so the
message is operator-friendly.

## Related operator commands

| Command | Use case |
|---|---|
| `genesis self-test` | "Is Genesis itself working?" — structural sanity check |
| `genesis doctor` | "Is my SYSTEM healthy?" — hardware/software/model walk |
| `genesis lifecycle-audit` | Lifecycle states only, machine-readable for CI |
| `genesis validate-schema` | Schema validation only, exit 1 on violation |
| `genesis explain <patch_id>` | Per-patch deep-dive |

`self-test` is the broadest of these — it runs the same
structural-sanity checks the lifecycle audit and schema validator
provide, plus version, imports, categories build, and predicates
evaluator. If you're going to run only one thing after a `git pull`,
run `self-test`.
