# Dispatcher baseline snapshot fixtures — M.1.1.T1.A

These JSON snapshots lock the **observable surface** of
`vllm.sndr_core.dispatcher` + `vllm.sndr_core.apply._state` against
unintentional drift during the M.1.1.T1.B / .T2 / .T3 refactor queue.

## What's protected

| Fixture | Surface | Why it matters |
|---|---|---|
| `apply_registry.json` | `apply._state.PATCH_REGISTRY` — ordered list of `(name, wrapped_name)` pairs | Tier 3 will collapse 145 retired stubs into a declarative `_RETIRED_PATCHES` dict; the registered-callable list must stay byte-identical so boot order + observability metric labels don't shift |
| `spec_set.json` | `iter_patch_specs()` — per-spec stable subset of fields | Tier 1.B will split `should_apply` into helpers and Tier 2 will split `validate_registry`; spec metadata (family / lifecycle / tier / default_on / has_upstream_pr / apply_module) must remain unchanged |
| `apply_module_coverage.json` | `validate_apply_module_coverage()` — `(total, mapped, unmapped, intentionally_unmapped)` | Per-patch `apply_module` derivation must stay stable; any drift signals an accidental relocation |
| `decision_no_env.json` | `should_apply(<patch_id>)` swept across all 228 entries with **every canonical env-flag prefix unset** | Tier 1.B `should_apply` decomposition must preserve every `(applied, reason)` byte-identically; reason strings are operator-visible in boot logs |

## How to regenerate

When you intentionally add/edit a patch (new registry entry, new
apply_module, new env_flag), the snapshot tests will fail. Update
them in the same commit:

```bash
SNDR_SNAPSHOT_REGEN=1 python3 -m pytest tests/unit/dispatcher/test_baseline_snapshots.py -q
```

This writes the new fixture content. Review the JSON diff against
git, confirm it's the intended drift (e.g. a new patch_id appears),
then commit fixture + the registry change together.

**Do NOT** regenerate as part of a refactor commit (T1.B / T2 / T3).
A refactor that needs to update these fixtures is by definition
**not** a behaviour-preserving refactor — that's exactly what this
safety net catches.

## Fixture format notes

- JSON, indented 2, sorted keys at the dict level, **insertion order
  preserved at the list level** (PATCH_REGISTRY iteration order =
  file order = boot order; spec_set order = same).
- `apply_module` paths are dotted Python module paths or `null` for
  intentionally-unmapped entries.
- `decision_no_env.json` reason strings are byte-identical to live
  `should_apply()` output with the env-clear test setup; if you
  change reason wording in `decision.py`, you're changing operator
  contract — regen + document the change.
