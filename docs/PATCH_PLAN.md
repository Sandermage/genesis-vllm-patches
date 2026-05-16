# Patch Plan — `--policy compat|safe|minimal`

Genesis ships **169 patches** across the registry. A typical preset
enables ~50–80 of them. Two questions follow naturally:

1. **Which patches actually need to be on for this deployment?**
2. **Which ones could I safely drop without giving up the speedups
   I care about?**

`patch_plan` is the answer. It's a read-only resolver that reads
each preset's `genesis_env` and the structured `patches_attribution`
metadata, applies a policy filter, and produces:

- the **included** env-flag set (what the runtime should actually export),
- the **excluded** set (what the policy dropped and why),
- the **passthrough** set (`GENESIS_*` parameter keys that are not
  patch toggles and therefore always survive),
- a **warnings** list (conflicts between included patches,
  `candidate_when` predicate mismatches against your rig).

The resolver does **not** change runtime apply decisions — the
dispatcher's `should_apply()` remains the canonical runtime gate.
`patch_plan` operates *upstream* of the runtime: it decides what
gets exported in the first place.

## Policy modes

| Policy | What it drops | When to use |
|---|---|---|
| `compat` *(default)* | Nothing. Every truthy toggle survives. Excluded set only contains operator-disabled (`value=0`) toggles. | Legacy / pre-attribution operators. Byte-identical to old behaviour. |
| `safe` | Toggles whose attribution declares `role: no_op`. Conservative — only kills patches **explicitly documented** as inactive on this preset. | Recommended starting point once attribution coverage is meaningful. |
| `minimal` | Above + `role: suspected_regression` + `role: unknown` (no attribution at all). Aggressive — drops everything that isn't explicitly documented as needed. | Advanced operators who curate attribution and want a lean stack. |

**Parameter keys always pass through.** `GENESIS_PN95_CONFIG_KEY`,
`GENESIS_BUFFER_MODE`, `GENESIS_PROFILE_RUN_CAP_M`, and similar
*configuration* values are not patch toggles — they're how the
patches *behave* once they fire. Filtering them would silently
break the patches that depend on them. The resolver detects
toggles by the `GENESIS_ENABLE_` / `GENESIS_DISABLE_` prefix; every
other `GENESIS_*` key passes through every policy unchanged.

## CLI surface

### Inspect a plan

```bash
sndr patches plan --preset prod-35b
sndr patches plan --preset prod-35b --policy compat --explain
sndr patches plan --preset prod-35b --policy minimal --json
```

`--explain` adds the `role`, `note`, and `bench_evidence` fields per
decision so reviewers see *why* each patch was kept or dropped.

### Diff between policies (A/B before flipping)

```bash
sndr compose plan-diff prod-35b --from compat --to minimal
sndr compose plan-diff prod-35b --from compat --to safe --json
```

Shows newly-excluded toggles, newly-included toggles, and the
passthrough delta (almost always empty). Use this before changing
the policy on a real launch to see exactly which env flags would
move.

### Render a compose file with the filtered env

```bash
sndr compose render prod-35b --policy minimal
sndr compose render prod-35b --policy safe -o /etc/sndr/compose.yml
```

The rendered YAML carries a header block stating policy + counts +
regeneration commands so anyone reading the file weeks later sees
the provenance.

### Launch with the filter applied

```bash
sndr launch prod-35b --policy minimal --dry-run
sndr launch prod-35b --policy safe
```

The rendered launch script exports only the included toggles +
every passthrough parameter. A one-line banner at boot states the
counts: `patch plan policy=minimal: 10 included / 49 excluded / 14 passthrough`.

### Diagnose against the same policy used at launch

```bash
sndr model-config diagnose prod-35b --policy minimal
```

If you launched with `--policy minimal`, run diagnose with the
same flag. Otherwise diagnose compares against the raw `genesis_env`
and flags every policy-excluded toggle as a missing-env error —
that's a false positive.

### Capture the plan in a report bundle

```bash
sndr report bundle --preset prod-35b --scope quality
```

The bundle now includes `patch_plan.json` with the resolver output
under all three policies side-by-side. Reviewers triaging the
bundle three weeks later can see exactly what would have launched
under any policy without rerunning the resolver against possibly
drifted code.

## Authoring attribution

`patches_attribution` lives **inside** the model YAML, keyed by
registry patch ID (e.g. `PN204`, not the env-flag name). Schema:

```yaml
patches_attribution:
  PN204:
    role: optional_perf
    bench_evidence: "dev371 35B conc=8: 689 TPS / TTFT 237 ms"
    note: |
      Enabled in 35b-multiconc profile via patches_delta.enable.
      Hopper SM 9.0+ gives +5-10% TPOT per upstream PR.
    candidate_when:
      max_num_seqs_gte: 4
```

### Role taxonomy

| Role | Semantics | Required fields |
|---|---|---|
| `load_bearing` | Removal causes measurable regression / boot break. | `note` |
| `defensive` | Cheap safety guard; default-include. | — |
| `optional_perf` | Conditional perf knob; needs evidence to justify. | `bench_evidence` |
| `suspected_regression` | Bench showed regression; documented but kept off. | `note` |
| `no_op` | Included for cross-config consistency but inactive on this preset. | — |
| `unknown` | Not yet classified (defaults when no attribution exists). | — |

### `candidate_when` predicates

`candidate_when` is an optional dict of operator-authored predicates.
When the predicate doesn't match the current `cfg`, the resolver
emits an **advisory warning** (it does not exclude the patch).

Supported predicate keys:

```yaml
candidate_when:
  max_num_seqs_gte: 4       # cfg.max_num_seqs >= 4
  max_num_seqs_lte: 8       # cfg.max_num_seqs <= 8
  max_model_len_gte: 100000
  n_gpus_eq: 2
  tool_call_parser: ["qwen3_coder", "qwen3_xml"]   # list membership
  reasoning_parser: ["qwen3"]
  kv_cache_dtype: ["fp8_e5m2", "turboquant_3bit_nc"]
  quantization: ["auto_round"]
```

Unknown predicate keys produce a warning and don't fail closed —
forward-compat for predicates the resolver hasn't been updated to
recognise.

### Profile-level overrides

A `ProfileDef.patches_delta.attribution` map can override or extend
the model's attribution per patch ID:

```yaml
# In profile/long-ctx-27b.yaml
patches_delta:
  enable:
    GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ: '1'
  attribution:
    PN204:
      role: load_bearing
      note: |
        Long-ctx profile keeps PN204 mandatory because OOM risk
        compounds with context length on the multi-conc path.
```

Per-key full replacement (not field merge): the profile entry is
the entry that lands in the composed `ModelConfig.patches_attribution`.

## Audit gates

Two gates in `make evidence` enforce the contract:

```bash
make audit-patch-attribution       # AT-1 (key matches registry) + AT-2 (role-presence consistency)
make audit-patch-plan-resolves     # every V2 preset resolves cleanly under all 3 policies
```

The coverage ratchet is optional:

```bash
python3 scripts/audit_patch_attribution.py --min-coverage 30
```

Operators who want to enforce a coverage floor over time can add
this in CI. Default: coverage is a ratchet *target*, not a hard gate.

## Migration

The policy filter is **opt-in everywhere**. If no `--policy` flag
is passed:

- `sndr patches plan` runs the dispatcher simulator (legacy);
- `sndr compose render` produces byte-identical output to pre-2026-05-16;
- `sndr launch` exports the full `genesis_env` matrix;
- `sndr diagnose` compares against the raw matrix.

The resolver still runs in the background under `sndr patches plan`
**without `--policy`** to surface advisory warnings (conflicts +
`candidate_when` mismatches). Those warnings are always visible —
they don't change behaviour, just visibility.

## Reference

- Resolver code: [`vllm/sndr_core/model_configs/patch_plan.py`](../vllm/sndr_core/model_configs/patch_plan.py)
- Schema: `PatchAttribution` in [`schema.py`](../vllm/sndr_core/model_configs/schema.py),
  `PatchesDelta.attribution` in [`schema_v2.py`](../vllm/sndr_core/model_configs/schema_v2.py).
- Audit gates: `scripts/audit_patch_attribution.py`,
  `scripts/audit_patch_plan_resolves.py`.
