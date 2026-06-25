# `sndr` CLI Reference

Complete command + parameter reference for the `sndr` CLI exposed by
`vllm-sndr-core`. Every subcommand is grouped by operator workflow:
install, run, inspect, configure, report.

> **Source of truth**: `sndr --help` and the
> per-subcommand `--help` always reflect the installed surface. This
> document tracks the same content with extra context, examples, and
> a stability badge per subcommand.

## Cheatsheet — first day on a rig → weekly maintenance

Top commands, ordered by operator workflow. Long-form per-subcommand
reference follows from §1 below.

```bash
# Install + first boot
curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash
sndr launch prod-qwen3.6-35b-balanced                            # V2 alias; V1 keys also accepted
sndr launch prod-qwen3.6-35b-balanced --dry-run                  # render only, no exec
sndr launch prod-qwen3.6-35b-balanced --preflight-only           # gate; never exec vLLM

# Health + smoke
sndr doctor                                     # full system diagnostic
sndr doctor --json                              # machine-readable
sndr doctor-system                              # extended host probe
sndr verify --quick                             # 10-prompt smoke (~60 s)
sndr self-test                                  # structural sanity (no GPU needed)
sndr verify prod-qwen3.6-35b-balanced                            # bench vs reference_metrics

# Browse + diff presets
sndr config list                                # V1 + V2 inventory
sndr config show prod-qwen3.6-35b-balanced
sndr config diff prod-qwen3.6-35b-balanced prod-qwen3.6-35b-multiconc
sndr config explain prod-qwen3.6-35b-balanced
sndr profile show qwen3.6-35b-balanced                  # V2 profile patches_delta

# Patches
sndr patches list --default-on                  # opt-out catalogue
sndr patches plan --preset prod-qwen3.6-35b-balanced             # dispatcher simulation
sndr patches plan --preset prod-qwen3.6-35b-balanced --policy compat --explain
sndr patches explain PN67
sndr patches doctor                             # registry validator
sndr patches release-check --mode require-static

# Capture a running container into a YAML
sndr model-config new my-rig --from-running vllm-test-container

# Service lifecycle (docker_compose / systemd / podman_quadlet / k8s / proxmox)
sndr service install prod-qwen3.6-35b-balanced
sndr service start prod-qwen3.6-35b-balanced
sndr service status prod-qwen3.6-35b-balanced
sndr service logs prod-qwen3.6-35b-balanced --lines 200
sndr service stop prod-qwen3.6-35b-balanced
sndr service uninstall prod-qwen3.6-35b-balanced

# Memory + caveats
sndr memory --preset prod-qwen3.6-35b-balanced                   # VRAM waterfall
sndr memory --live                              # query running container
sndr caveats list

# Reporting
sndr report bundle --preset prod-qwen3.6-35b-balanced            # tarball for issues
sndr report cudagraph-coverage                  # hit-rate snapshot

# Uninstall
bash ~/.sndr/install.sh --uninstall
```

For env-var knobs see [`CONFIGURATION.md`](CONFIGURATION.md); for the
`--from-running` captor + lxc_proxmox renderer see
[`QUICKSTART.md`](QUICKSTART.md).

## Conventions

| Badge | Meaning |
|---|---|
| **stable** | Production-ready, semver-protected. |
| **beta** | Functional, no breaking changes expected, but the JSON schema may grow fields. |
| **experimental** | Surface may change; useful for advanced operators. |
| **deferred** | Declared but not implemented yet; commands return a clean error. |

`<preset>` is a **V2 alias**, e.g. `prod-qwen3.6-35b-balanced`,
`prod-qwen3.6-27b-tq-k8v4`, `long-ctx-qwen3.6-27b` (under
`builtin/presets/`). The V1 monolithic preset tier (flat
`builtin/<key>.yaml`) was fully retired 2026-06-01 (Phase 10 sunset);
the V1+V2 resolver still accepts a V1 key as an opaque arg for
back-compat dispatch (`_resolve_preset_v1_or_v2`) but every shipped
V1 file is gone — operators get a clean "preset not found" error.

The resolver feeds `sndr launch`, `sndr compose`,
`sndr patches plan`, `sndr memory`, `sndr model-config diagnose`.

---

## 1. Install + first run

### `sndr install` — **stable**

Bootstrap a fresh machine end-to-end: preflight, hardware detect,
workload picker, vLLM pin allowlist, repo clone/update, plugin
install, host paths, launch script generation, smoke test.

```bash
sndr install                              # Interactive wizard
sndr install --config prod-qwen3.6-35b-balanced --yes      # Non-interactive, pick preset upfront
sndr install --config prod-qwen3.6-35b-balanced --prepare  # Plan only; write nothing
sndr install --config prod-qwen3.6-35b-balanced --print-script   # Emit launch script + exit
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--config <preset>` | (interactive) | Skip the workload picker; use this preset directly. |
| `--yes`, `-y` | off | Non-interactive; fail rather than prompt. |
| `--prepare` | off | Render artifacts but skip the live launch step. |
| `--print-script` | off | Emit the launch script to stdout and exit. |
| `--update` | off | Pull new commits / refresh plugin install. |
| `--target-pin <vllm>` | (allowlist default) | Pin vLLM nightly; verified against `KNOWN_GOOD_VLLM_PINS`. |

### `sndr launch` — **stable**

Render `cfg.to_launch_script()` + apply patches + exec the resulting
shell script under your shell (or container runtime).

```bash
sndr launch                                       # Interactive preset pick
sndr launch prod-qwen3.6-35b-balanced                              # Live launch
sndr launch prod-qwen3.6-35b-balanced --dry-run                    # Render + diagnose only
sndr launch prod-qwen3.6-35b-balanced --preflight-only             # Preflight gate; no exec
sndr launch prod-qwen3.6-35b-balanced --pull                       # `docker pull` before launch
sndr launch prod-qwen3.6-35b-balanced --check-deps                 # Validate deps before exec
sndr launch prod-qwen3.6-35b-balanced --policy minimal             # Filter env through patch_plan
sndr launch prod-qwen3.6-35b-balanced --policy safe --dry-run      # Preview filtered script
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `config_key` (positional) | (interactive) | Preset key or V2 alias. |
| `--dry-run` | off | Render the launch script to stdout; do not exec. |
| `--port <int>` | preset value | Override the preset's HTTP port. |
| `--skip-apply` | off | Bypass dispatcher apply phase (use only when patches are pre-applied). |
| `--non-interactive`, `-y` | off | No prompts; requires explicit `config_key`. |
| `--strict-image {on,off,auto}` | `auto` | Refuse launch when local image digest mismatches `docker.image_digest`. `auto` enforces only when the preset declares a digest. |
| `--preflight-only` | off | Run preflight gates and exit; never exec vLLM. |
| `--pull` | off | `docker pull` the preset's image before exec. |
| `--check-deps` | off | Run `sndr deps inspect` against the preset; abort on missing dep. |
| `--policy {compat,safe,minimal}` | unset | Filter `cfg.genesis_env` through the `patch_plan` resolver. See [PATCHES.md § patch-plan policy](PATCHES.md). |

Pre-launch warnings surface for enabled patches with
`implementation_status` in `{partial, placeholder, marker_only}` so
operators don't run "advertised" features that have no real
implementation.

---

## 2. Inspect

### `sndr doctor` — **stable**

Single-command "is my Genesis healthy" check. Calls patches doctor,
schema validator, apply.shadow, and host preflights in series.

```bash
sndr doctor                            # human report
sndr doctor --json                     # machine-readable
```

### `sndr doctor-system` — **stable**

Extended host/runtime diagnostic. Probes nvidia-smi, Docker, host
config, plugin install, etc. Useful as the first stop when something
breaks on a fresh box.

```bash
sndr doctor-system
sndr doctor-system --json
```

### `sndr verify` — **stable**

Bench-vs-reference verification (CI gate). Runs the preset for a
short bench, compares against `reference_metrics`, exits non-zero on
out-of-tolerance.

```bash
sndr verify prod-qwen3.6-35b-balanced                  # default tolerance bands
sndr verify prod-qwen3.6-35b-balanced --json
sndr verify prod-qwen3.6-35b-balanced --strict         # tighter tolerance, fail-fast
```

### `sndr memory` — **stable**

VRAM budget estimator + live memory diagnostics.

```bash
sndr memory                                # plan estimator for active preset
sndr memory --preset prod-qwen3.6-35b-balanced              # explicit preset
sndr memory --live                         # query running container
sndr memory --json
```

### `sndr caveats` — **stable**

Runtime caveats registry — known host-condition issues that affect
specific patches or presets.

```bash
sndr caveats list
sndr caveats inspect <preset>
```

### `sndr self-test` — **stable**

Structural sanity check after a fresh `git pull` or vLLM pin bump.
Answers the question "is Genesis itself working on this box?" —
different from `doctor` ("is my SYSTEM healthy?"). A `doctor` failure
can be hardware / config; a `self-test` failure is a Genesis bug or a
botched install.

```bash
sndr self-test                                          # human, all checks
sndr self-test --quiet                                  # only fail/warn/skip rows
sndr self-test --json                                   # machine-readable
```

**Eight checks**, run in order, all run regardless of failures
(self-test never crashes — it surfaces every problem in one pass):

| # | Check | What it verifies |
| --- | --- | --- |
| 1 | version constant | `sndr.version.__version__` is a non-empty string. |
| 2 | compat imports | All `sndr.compat.*` modules import cleanly. |
| 3 | integrations imports | All `sndr/engines/vllm/patches/**/*.py` modules import; SKIP if `vllm` not installed. |
| 4 | schema validator | `PATCH_REGISTRY` validates against `schemas/patch_entry.schema.json`. |
| 5 | lifecycle audit | Every entry has a known lifecycle state. |
| 6 | categories build | Categories index builds without errors and every patch is placed in at least one category. |
| 7 | predicates evaluator | Every `applies_to` clause can be evaluated against an empty environment without raising. |
| 8 | schema file | `schemas/patch_entry.schema.json` is parseable; SKIP in slim deployments where the source tree is not mounted. |

**Exit codes:** `0` = all `fail`-class checks passed; `1` = at least
one `fail`. `warn` and `skip` do not change the exit code.

**Status symbols:** ✓ `pass`, ✗ `fail`, ⚠ `warn`, • `skip`.

**Slim deployments.** If only the `sndr/` package is mounted
(no source tree), the schema file check returns `skip` rather than
`fail`. Point at an external source tree via env var:

```bash
GENESIS_REPO_ROOT=/path/to/genesis-vllm-patches sndr self-test
```

**Adding a new check.** Self-test lives in
`sndr/compat/self_test.py`. Add a function
`_check_<name>() -> tuple[str, str]` returning
`(status, message)`, append to the `_CHECKS` list, and add a unit
test pinning the new check name. Contract: a check must never raise.

Companion utilities:

- `sndr patches lifecycle-audit` — lifecycle states only,
  machine-readable for CI.
- `sndr patches validate-schema` — schema validation only,
  exit 1 on violation.

---

## 3. Patches

### `sndr patches list` — **stable**

Browse the patch registry with filters.

```bash
sndr patches list
sndr patches list --tier community --lifecycle stable
sndr patches list --family attention.turboquant --json
sndr patches list --default-on
sndr patches list --opt-in
sndr patches list --has-upstream
sndr patches list --no-upstream
```

The JSON output carries an extra `production_default` field with
honest values: `applied` / `marker` / `opt-in` / `blocked`.

### `sndr patches explain <patch_id>` — **stable**

Detailed view of a single registry entry: family, env_flag,
applies_to, dependencies, conflicts, lifecycle, upstream PR, evidence.

```bash
sndr patches explain PN204
sndr patches explain P67 --json
```

### `sndr patches doctor` — **stable**

Validator over `PATCH_REGISTRY` + apply layer wiring.

```bash
sndr patches doctor
sndr patches doctor --json
```

### `sndr patches plan` — **stable**

Simulate dispatcher decisions for a preset, optionally filtered by
the `patch_plan` resolver policy.

```bash
sndr patches plan --preset prod-qwen3.6-35b-balanced                        # legacy simulator
sndr patches plan --preset prod-qwen3.6-35b-balanced --policy compat        # resolver view
sndr patches plan --preset prod-qwen3.6-35b-balanced --policy safe --explain
sndr patches plan --preset prod-qwen3.6-35b-balanced --policy minimal --json
sndr patches plan --preset prod-qwen3.6-35b-balanced --profile production   # block partial/placeholder
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--preset <key>` | (required) | V1 key or V2 alias. |
| `--json` | off | Machine-readable output. |
| `--profile {any,production}` | `any` | `production` blocks the plan when any included patch has `implementation_status ∈ {partial, placeholder}` or `lifecycle ∈ {research, retired}`. |
| `--policy {compat,safe,minimal}` | unset | Add resolver view to output. |
| `--explain` | off | Include `role`, `note`, `bench_evidence` per decision. Only meaningful with `--policy`. |

When `--policy` is **not** passed, the legacy simulator output still
runs, **and** the resolver runs silently for warnings only (advisory
section surfaces `conflicts_with` + `candidate_when` mismatches).

### `sndr patches prove` — **stable**

Run static proof checks over every registry entry.

```bash
sndr patches prove --all
sndr patches prove --all --no-write
sndr patches prove --filter PN95
```

### `sndr patches release-check` — **stable**

Gate the registry against a release policy.

```bash
sndr patches release-check                              # default (report mode)
sndr patches release-check --mode require-static       # CI gate
sndr patches release-check --mode require-bench-attached
sndr patches release-check --mode require-baseline     # strict
sndr patches release-check --show-passing
```

### `sndr patches bench-attach` — **stable**

Attach bench output to a patch's proof artifact.

```bash
sndr patches bench-attach PN119 --bench out/bench.json --pin <vllm_pin>
```

### `sndr patches proof-status` — **stable**

Bucket every entry by proof tier (`static_only`,
`bench_with_baseline`, `dead`).

```bash
sndr patches proof-status
sndr patches proof-status --json
```

### `sndr patches diff-upstream` — **stable**

Surface patches whose upstream PR has merged and may be retire-ready.

```bash
sndr patches diff-upstream
sndr patches diff-upstream --json
```

### `sndr patches bundles` — **stable**

Predefined patch bundles for common workloads.

```bash
sndr patches bundles list
sndr patches bundles explain long-context-stack
```

### `sndr patches pn95` — **stable**

Tier-aware KV cache (PN95) inspect/dump/disk-tier-show subcommand
tree.

```bash
sndr patches pn95 status
sndr patches pn95 dump --json
sndr patches pn95 disk-tier-show
```

---

## 4. Compose / launch surfaces

### `sndr compose render <preset>` — **stable**

Render preset → `docker-compose.yml` for stdin or file.

```bash
sndr compose render prod-qwen3.6-35b-balanced                       # legacy unfiltered
sndr compose render prod-qwen3.6-35b-balanced -o /etc/sndr/compose.yml
sndr compose render prod-qwen3.6-35b-balanced --policy compat
sndr compose render prod-qwen3.6-35b-balanced --policy safe
sndr compose render prod-qwen3.6-35b-balanced --policy minimal
```

The rendered header carries a "Patch policy:" block with included /
excluded / passthrough counts plus `regenerate:` and `inspect:`
commands so anyone reading the file weeks later sees provenance.

### `sndr compose up <preset>` — **stable**

`docker compose up -d` against the preset's compose file (renders
inline if needed).

```bash
sndr compose up prod-qwen3.6-35b-balanced
sndr compose up prod-qwen3.6-35b-balanced --detach
```

### `sndr compose down <preset>` — **stable**

`docker compose down`.

```bash
sndr compose down prod-qwen3.6-35b-balanced
```

### `sndr compose logs <preset>` — **stable**

```bash
sndr compose logs prod-qwen3.6-35b-balanced
sndr compose logs prod-qwen3.6-35b-balanced -n 500 --follow
```

### `sndr compose plan-diff <preset>` — **stable**

A/B between two policies. Read-only — no YAML rendered.

```bash
sndr compose plan-diff prod-qwen3.6-35b-balanced --from compat --to minimal
sndr compose plan-diff prod-qwen3.6-35b-balanced --from compat --to safe --json
```

Surfaces:

- `newly_excluded` — toggles that move from included to excluded.
- `newly_included` — opposite direction (rare).
- `unchanged_included`, `unchanged_excluded` — set membership stable.
- `passthrough_diff` — almost always empty; passthrough is policy-independent.

### `sndr quadlet <preset>` — **beta**

Render systemd/podman quadlet unit for the preset.

```bash
sndr quadlet render prod-qwen3.6-35b-balanced
```

### `sndr k8s render <preset>` — **beta**

Render Kubernetes manifests for the preset.

```bash
sndr k8s render prod-qwen3.6-35b-balanced
```

---

## 5. Service lifecycle

### `sndr service` — **beta**

Service lifecycle manager across backends (systemd / docker_compose /
podman_quadlet / bare_metal). Reads `ServiceConfig` from the preset.

```bash
sndr service install <preset>           # backend-specific provisioning
sndr service start <preset>             # backend-specific start
sndr service stop <preset>              # backend-specific stop
sndr service status <preset>            # backend-specific status
sndr service logs <preset>              # backend-specific logs
sndr service uninstall <preset>         # cleanup
```

Backend behaviour:

| Backend | install | start | stop | status | logs |
|---|---|---|---|---|---|
| `systemd` | render unit + enable | systemctl start | systemctl stop | systemctl status | journalctl |
| `docker_compose` | render `~/.sndr/compose/<key>.yml` | `docker compose -f … up -d` (falls back to `docker start <container>`) | `docker compose down` (falls back to `docker stop`) | `docker ps --filter` | `docker logs` |
| `podman_quadlet` | (points operator at `sndr quadlet`) | systemctl start | systemctl stop | systemctl status | journalctl |
| `kubernetes` | (delegates to `sndr k8s render`) | not yet wired | not yet wired | not yet wired | not yet wired |
| `bare_metal` | informational | run `sndr launch` directly | n/a | n/a | n/a |

---

## 6. Proxmox

### `sndr proxmox` — **beta**

Render Proxmox provisioning commands. Modes: `lxc`, `vm`, `host`.

```bash
sndr proxmox doctor <preset>
sndr proxmox render <preset>            # operator-readable command script
sndr proxmox apply <preset>             # (deferred)
sndr proxmox status <preset>
```

Unknown modes produce a fail-fast error (`exit 2`) instead of warning
and proceeding.

---

## 7. Model config

### `sndr model-config show <key>` — **stable**

Print one resolved `ModelConfig` (V1).

```bash
sndr model-config show a5000-2x-35b-prod
sndr model-config show a5000-2x-35b-prod --json
```

### `sndr model-config list` — **stable**

List V1 preset keys + brief metadata.

```bash
sndr model-config list
sndr model-config list --json
```

### `sndr model-config explain <key>` — **stable**

Human-readable explanation of a preset's env, mounts, runtime
command, validation status.

```bash
sndr model-config explain prod-qwen3.6-35b-balanced
```

### `sndr model-config new <key>` — **stable**

Create a user preset from a template.

```bash
sndr model-config new my-preset --template a5000-2x-35b-prod
sndr model-config new my-preset --template prod-qwen3.6-35b-balanced --force
```

### `sndr model-config promote <key>` — **stable**

Promote a community config through the `community-test → -dev → -prod`
ladder. Schema gates enforce cross-rig validation + reference_metrics.

```bash
sndr model-config promote my-preset --tier community-dev
sndr model-config promote my-preset --tier community-prod
```

### `sndr model-config audit <key>` — **stable**

Static audit (audit_rules) against one preset.

```bash
sndr model-config audit prod-qwen3.6-35b-balanced
```

### `sndr model-config preflight <key>` — **stable**

Host preflight (mounts, GPU, container name) without running vLLM.

```bash
sndr model-config preflight prod-qwen3.6-35b-balanced
```

### `sndr model-config diagnose <key>` — **stable**

Runtime diagnose against a **running** container.

```bash
sndr model-config diagnose prod-qwen3.6-35b-balanced
sndr model-config diagnose prod-qwen3.6-35b-balanced --port 8002
sndr model-config diagnose prod-qwen3.6-35b-balanced --policy minimal
```

`--policy` lets diagnose compare against the policy-filtered env
rather than `cfg.genesis_env` raw. Use when you launched with the
same `--policy`, otherwise diagnose flags policy-excluded toggles as
missing.

### `sndr model-config verify <key>` — **stable**

Bench vs reference_metrics (same as top-level `sndr verify`).

```bash
sndr model-config verify prod-qwen3.6-35b-balanced
```

### `sndr model-config diff <a> <b>` — **stable**

Diff two presets' merged env + sizing + mounts.

```bash
sndr model-config diff prod-qwen3.6-35b-balanced prod-qwen3.6-35b-multiconc
```

---

## 8. V2 model registry + preset catalog

### `sndr model list-v2` — **stable**

List V2 ModelDef registry entries.

```bash
sndr model list-v2
sndr model list-v2 --json
```

### `sndr model pull <key>` / `sndr model list` — **stable**

Model download + registry inventory (HuggingFace + local).

```bash
sndr model pull <id>
sndr model list
```

### `sndr hardware list` — **stable**

List V2 HardwareDef entries.

```bash
sndr hardware list
```

### `sndr profile list` — **stable**

List V2 ProfileDef entries.

```bash
sndr profile list
sndr profile list --model qwen3.6-35b-a3b-fp8
```

### `sndr profile show <id>` — **stable**

Show one profile's `patches_delta` + sizing override + attribution.

```bash
sndr profile show qwen3.6-35b-balanced
```

### `sndr profile diff <a> <b>` — **stable**

Diff two profiles.

```bash
sndr profile diff qwen3.6-35b-balanced qwen3.6-35b-multiconc
```

### `sndr preset list` — **stable**

List V2 presets with their `PresetCard` metadata. Operator-product
surface — distinct from `sndr config list` (V1 + V2 inventory of
config keys) and `sndr profile list` (patches delta layer). See
[`PRESETS.md`](PRESETS.md) for the card schema and when to use which
command.

```bash
sndr preset list                                       # all presets, table view
sndr preset list --json                                # machine-readable
sndr preset list --status production_candidate          # filter by card.status
sndr preset list --family qwen3_6_35b_a3b_fp8          # filter by card.routing_family
sndr preset list --workload free_chat                  # workload_allow intersection
sndr preset list --hardware a5000-2x-24gbvram-16cpu-128gbram
sndr preset list --mode throughput                     # filter by card.mode
```

Unannotated presets (no `card:` block in YAML) are shown but tagged
`(unannotated)`. They are skipped by `sndr preset recommend`.

### `sndr preset show <preset_id>` — **stable**

Card-formatted view of one preset: identity, workload contract,
operating envelope, evidence, tradeoffs, "do not use". For raw YAML
dump use `sndr config explain <preset_id>`.

```bash
sndr preset show prod-qwen3.6-35b-balanced
sndr preset show prod-qwen3.6-35b-balanced --json
sndr preset show prod-qwen3.6-35b-balanced --field card.evidence_refs.0.path
```

`--field <dot.path>` walks nested attributes / list indices / dict
keys (e.g. `card.evidence_refs.0.path`, `card.concurrency.canonical`).
Errors include the failed segment for self-correction.

### `sndr preset explain <preset_id>` — **stable**

Operator walkthrough: card narrative + composed runtime dry-run +
single-row diff vs `card.fallback_preset`. Used to validate that the
preset's YAML triplet actually composes to the runtime claimed in the
card.

```bash
sndr preset explain prod-qwen3.6-35b-balanced
sndr preset explain prod-gemma4-26b-multiconc --json
```

The "Composed runtime (dry-run)" section reports `composed_key`,
`kv_cache_dtype`, `max_model_len`, `max_num_seqs`,
`gpu_memory_utilization`, `spec_decode_method`, `spec_decode_K`, and
`enabled_patches_count` — the field-set most operators care about.

### `sndr preset recommend` — **stable**

Inverse lookup: operator declares a workload (with optional hardware
and concurrency constraints), CLI ranks matching presets. Ranking order:

1. `card.status` priority (production > production_candidate > internal_validated > others)
2. `card.default_for_family` (true sorts before false)
3. `card.primary_metric.value` descending
4. preset id ascending (deterministic tie-break)

```bash
sndr preset recommend --workload free_chat \
                      --hardware a5000-2x-24gbvram-16cpu-128gbram \
                      --concurrency 8
sndr preset recommend --workload structured_json.short --top 3
sndr preset recommend --workload custom:my-task --json
```

Workload values are from a frozen taxonomy: `free_chat`,
`structured_json.short`, `structured_json.long`, `tool_call.short`,
`tool_call.long`, `summarization`, `code_gen`, `long_context_qa`.
Custom workloads are accepted via the `custom:<slug>` escape (slug
matches `[a-z0-9._-]+`).

Safety rule: a preset is **excluded** from results when the queried
workload is in its `card.workload_deny`, even if `workload_allow` is
broad or empty. Concrete example — `--workload free_chat --concurrency 8`
will not return `prod-gemma4-26b-mtp-k4` (K=4 structured) because
its `workload_deny` lists `free_chat`.

---

## 9. Reporting

### `sndr report bundle` — **stable**

Collect a redacted tar.gz of diagnostic artifacts.

```bash
sndr report bundle                                  # ~/.sndr/reports/<ts>.tar.gz
sndr report bundle --output /tmp/report.tar.gz
sndr report bundle --preset prod-qwen3.6-35b-balanced
sndr report bundle --container vllm-server
sndr report bundle --no-redact                      # internal use only
sndr report bundle --scope all                      # default
sndr report bundle --scope deps
sndr report bundle --scope launch
sndr report bundle --scope quality
sndr report bundle --scope patches
```

Scope filters the artifacts collected:

| Scope | Artifacts |
|---|---|
| `all` | doctor.json + patches.json + patch_plan.json + launch_dryrun.sh + vllm_boot.log + host_yaml.txt + nvidia_smi.txt + pip_freeze.txt + git_log.txt + image_inspect.json |
| `deps` | doctor + host_yaml + nvidia_smi + pip_freeze + git_log |
| `launch` | doctor + launch_dryrun + vllm_boot + host_yaml + image_inspect + patch_plan |
| `quality` | doctor + patches + patch_plan + launch_dryrun + vllm_boot + git_log |
| `patches` | doctor + patches + patch_plan + git_log |

### `sndr report cudagraph-coverage` — **stable**

CUDA-graph dispatch hit-rate snapshot for the running process.

```bash
sndr report cudagraph-coverage
sndr report cudagraph-coverage --json
```

Requires `GENESIS_CUDAGRAPH_DISPATCH_TRACE=1` at boot.

---

## 10. Dependencies + upstream tracking

### `sndr deps inspect <preset>` — **stable**

Host dependency inventory + per-preset plan.

```bash
sndr deps inspect prod-qwen3.6-35b-balanced
sndr deps inspect prod-qwen3.6-35b-balanced --json
```

### `sndr upstream list` / `sndr upstream check` — **stable**

vLLM pin allowlist + per-preset upstream policy.

```bash
sndr upstream list
sndr upstream check prod-qwen3.6-35b-balanced
```

---

## 11. Community SDK

### `sndr community list` — **stable**

List community-submitted patches and their lifecycle stage.

```bash
sndr community list
sndr community list --json
```

### `sndr community validate <path>` — **stable**

Validate a community patch manifest.

```bash
sndr community validate ./plugins/foo/PN999/
```

### `sndr community new-patch <id>` — **stable**

Scaffold a new community patch.

```bash
sndr community new-patch PN999 --author you --description "..."
```

---

## 12. Native preset config (UNIFIED_CONFIG C8)

### `sndr config` — **stable**

Native preset browser; alternative to `sndr model-config`.

```bash
sndr config list
sndr config show prod-qwen3.6-35b-balanced
sndr config diff prod-qwen3.6-35b-balanced prod-qwen3.6-35b-multiconc
sndr config explain prod-qwen3.6-35b-balanced
```

`sndr config` is the V1 + V2 preset inventory surface — distinct from
`sndr config-catalog` (§13), which is the **derived catalog** view of
the same V2 corpus with row-level redaction and a fixed-DSL query
language.

---

## 13. Derived config catalog (CONFIG-UX.5)

### `sndr config-catalog` — **stable**

Read-only API over the V2 model-config corpus + registry. The catalog
is a **derived view**, not a new source of truth: every row is rebuilt
on demand from `model_configs/builtin/*.yaml` and
`dispatcher/registry.py`. **No catalog JSON is committed to git.**
Operators regenerate locally; CI verifies determinism and redaction.

Five row types — `preset`, `profile`, `model`, `hardware`, `baseline` —
each with its own schema (see
[`sndr/model_configs/catalog_schema.py`](../sndr/model_configs/catalog_schema.py)).

**Redaction.** Maintainer-private tree paths and `visibility: private`
evidence refs are stripped at output time on every leaf (`--redact-private`
is the default). The same generator drives both the public catalog and
any maintainer-private debug view — redaction is the only difference.

#### `sndr config-catalog build` — **stable**

Rebuild the catalog and emit JSON.

```bash
sndr config-catalog build --stdout                # print to stdout
sndr config-catalog build --out catalog.json      # write to a path
sndr config-catalog build --check                 # verify deterministic
                                                   # regeneration (CI mode)
```

`--check` exits non-zero if a second build produces different bytes.

#### `sndr config-catalog verify` — **stable**

Thin wrapper over
[`scripts/audit_generated_config_catalog.py`](../scripts/audit_generated_config_catalog.py).
Verifies the derived catalog regenerates deterministically AND that no
private paths leak.

```bash
sndr config-catalog verify --strict --json
```

Default mode is informational (warnings exit 0); `--strict` exits 1 on
any finding.

#### `sndr config-catalog show <row_id>` — **stable**

Print a single row from the derived catalog. Collision-aware row IDs:
when the same `id` exists across row types (e.g. a preset and a
profile both named `prod-qwen3.6-35b-balanced`), use the prefixed form:

```bash
sndr config-catalog show preset/prod-qwen3.6-35b-balanced
sndr config-catalog show prod-qwen3.6-35b-balanced               # bare form OK if unambiguous
sndr config-catalog show profile/qwen3.6-35b-dflash --section override_policy
```

`--section <SECTION>` narrows the output to a top-level catalog field.

#### `sndr config-catalog query` — **stable**

Filter the derived catalog. **AND-only, five fixed flags**, no
expressions / joins / sort / JSONPath.

```bash
# All profiles with override_class = bench
sndr config-catalog query --row-type profile \
                          --field override_class --equals bench

# Substring match
sndr config-catalog query --row-type preset \
                          --field card.status --contains production

# Time-window filter (only `--field override_expires_at`)
sndr config-catalog query --row-type profile \
                          --field override_expires_at \
                          --expires-before 2026-09-01
```

Flags:

| Flag | Meaning |
|---|---|
| `--row-type {preset,profile,model,hardware,baseline,any}` | Restrict by row type. `any` includes all five. |
| `--field FIELD` | Top-level row field (`override_class`, `card.status`, etc.). Unknown fields error out with the available-fields list. |
| `--equals VALUE` | Exact match (mutually exclusive with `--contains` / `--expires-before`). |
| `--contains VALUE` | Substring match. |
| `--expires-before YYYY-MM-DD` | Only valid on date fields. Invalid date format exits 2. |
| `--json` | Machine-readable output (default is human-formatted). |

#### `sndr config-catalog` — `--from` and `--strict-fresh`

`build` / `verify` / `show` / `query` accept `--from PATH` to read from
a previously-emitted catalog JSON file instead of regenerating:

```bash
sndr config-catalog show prod-qwen3.6-35b-balanced --from /tmp/catalog.json --strict-fresh
```

With `--strict-fresh`, the CLI compares the file's mtime against the
source YAMLs and **exits non-zero on staleness** — it does NOT silently
regenerate. This is the gating behaviour for CI / release pipelines
that pin a specific catalog snapshot.

#### Umbrella alias

```bash
make config-catalog        # → sndr config-catalog build
```

For the deterministic regeneration + redaction audit gate:

```bash
make audit-generated-config-catalog
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Operation failed (non-blocker) |
| `2` | Validation / preflight failed; user action required |
| `>2` | Specific subcommand error code (see `--help`) |

## Environment variables affecting the CLI

| Variable | Purpose |
|---|---|
| `GENESIS_DISABLE_<NAME>=1` | Force-disable a patch even when its env_flag is enabled. |
| `SNDR_DISABLE_<NAME>=1` | Same as above; either prefix works. |
| `VLLM_USE_FLASHINFER_SAMPLER=1` | Routes top-k/top-p through FlashInfer; PN132 then becomes a no-op fallback guard. |
| `VLLM_LOGGING_LEVEL=WARNING` | Cuts uvicorn access-log noise; recommended for prod. |

For Genesis-specific env keys see `docs/CONFIGURATION.md` and the
inline comments inside each model YAML.

---

## See also

- [PATCHES.md § patch-plan policy](PATCHES.md) — `--policy compat|safe|minimal` deep dive
- [CONFIGURATION.md](CONFIGURATION.md) — runtime env knobs + preset selection
- [PATCHES.md](PATCHES.md) — patch taxonomy + lifecycle
- [INSTALL.md](INSTALL.md) — first-time install walkthrough
- [BENCHMARKS.md § methodology](BENCHMARKS.md) — bench methodology + reproduction
