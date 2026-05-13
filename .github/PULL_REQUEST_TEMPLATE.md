## What this PR does

<!-- One-sentence summary. -->

## Type of change

- [ ] Bug fix (non-breaking, fixes a reported issue)
- [ ] New patch (adds `vllm/sndr_core/integrations/<family>/<file>.py` + registry entry)
- [ ] Patch retire (per iron-rule-#11 deep-diff — see provenance section below)
- [ ] Documentation
- [ ] Community model config (lifecycle: community-test on submission)
- [ ] Refactor / cleanup (no behavior change)
- [ ] Backport from upstream vLLM PR (link upstream PR below)
- [ ] Pin bump (KNOWN_GOOD_VLLM_PINS update + EXPECTED_PINS sync)

## Backport reference (if applicable)

- Upstream PR: vllm-project/vllm#XXXX
- Upstream author handle: @upstream-handle
- Their state at merge: OPEN / MERGED / CLOSED
- Why backport now: <reason>

## Iron rule #11 — retire provenance (if retiring a patch)

Per Sander's 2026-05-11 strategic mandate: never blindly retire on PR-title match. Required:

- [ ] Deep-diff'd our patch vs upstream merged code line-by-line
- [ ] Categorized as ONE of:
  - **(a) Byte-identical** → set `lifecycle="retired"` + `superseded_by` + `vllm_version_range` upper bound
  - **(b) Our patch does MORE** → keep patch active; update to drop the dups; add iron-rule-#11 note in registry
  - **(c) Different approach** → keep patch active; verify anchors still apply OR mark with `enables_upstream_feature: True` / `_INTENTIONAL_INVERSE_WAIVER`
- [ ] [`tests/unit/dispatcher/test_iron_rule_11_enforcement.py`](../tests/unit/dispatcher/test_iron_rule_11_enforcement.py) passes locally

## How tested

- [ ] `python3 -m pytest tests/unit/` — pass count: X passed / 0 failed / Y skipped / Z xfailed
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`) — see [.pre-commit-config.yaml](../.pre-commit-config.yaml) for the 7 Genesis gates
- [ ] New unit test in `tests/unit/integrations/<family>/test_p<NN>_<name>.py` (for new patches)
- [ ] Family contract auto-covers new patch (no manual change if family already has contract — just add to `<family>_PATCHES` list)
- [ ] Manual boot test on rig: <hardware>
- [ ] `python3 -m vllm.sndr_core.compat.cli self-test` succeeds
- [ ] (If config) `python3 -m vllm.sndr_core.compat.cli model-config verify <key>` succeeds

## Pin-gate (new patches only — recommended)

New patches should declare `vllm_version_range` in `applies_to`:

- [ ] PROD-active broad: `(">=0.20.0", "<0.21.0")` (default for new patches)
- [ ] Anchor-tight: `(">=0.20.2rc1.devN", "<0.21.0")` (anchor appeared at pin N)
- [ ] Retire upper bound: `"<0.20.2rc1.devN"` (upstream merged at pin N — paired with `lifecycle="retired"`)

See [docs/CONTRIBUTING.md "Pin-bump playbook"](../docs/CONTRIBUTING.md#pin-bump-playbook) for full pin-gate semantics.

## Compose / patch interaction

- [ ] No conflict with existing patches (run `python3 -m vllm.sndr_core.compat.cli lifecycle-audit`)
- [ ] PATCH_REGISTRY entry has correct `applies_to.<model_class|is_*>` matrix
- [ ] PATCH_REGISTRY `family` field matches the actual `integrations/<family>/` directory
- [ ] If text-patch: anchor is verbatim upstream, unique, marker constant declared + injected into replacement
- [ ] If runtime hook (no TextPatcher): family contract `test_genesis_marker_exists` will skip cleanly (documented)

## Docs sync

- [ ] Updated `CHANGELOG.md` with brief description (one paragraph + bullet list of changes)
- [ ] `docs/PATCHES_AUTO.md` regenerates clean: `python3 scripts/generate_patches_md.py --check`
- [ ] `docs/CONFIGS_AUTO.md` regenerates clean: `python3 scripts/generate_configs_md.py --check`
- [ ] If env flag introduced: documented in patch source code + `docs/CONFIGURATION.md` if user-facing
- [ ] If new family: family contract added via `make_family_contract_class("<family>", PATCHES)` factory

## Genesis convention checks

- [ ] No automated-tool co-author trailers in commits
- [ ] No paths to `~/Genesis_internal_docs/` or other operator-specific dirs hardcoded in source
- [ ] No hardcoded paths in active scripts — use `vllm.sndr_core.locations.project_paths` helpers + bash env vars
- [ ] Renames respect back-compat aliases (e.g., `from vllm.sndr_core.patches` still resolves via `__getattr__`)

## CI gates (will run automatically)

The PR will fail-fast on any of these 4 gates:

1. **Pin-gate adoption** (`tests/unit/dispatcher/test_pin_gate.py`) — 13 tests
2. **Iron rule #11 enforcement** (`tests/unit/dispatcher/test_iron_rule_11_enforcement.py`) — 4 tests
3. **Family contracts** (`tests/unit/integrations/`) — 17 contracts, ~700 tests
4. **Upstream-status audit** (`scripts/audit_upstream_status.py`) — informational at PR time; strict weekly cron

Run locally via `pre-commit run --all-files` to catch failures before push.

## Open questions / known caveats

<!-- Any decisions you'd like reviewer input on, or known limitations. -->
