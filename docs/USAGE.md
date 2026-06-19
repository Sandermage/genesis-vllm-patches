# Using Genesis vLLM Patches — Operator Manual

A single-page narrative for going from `git clone` to a production
preset on your own rig. Covers the **installer**, the **`sndr`
launcher**, the **config system**, and the **patch authoring loop** —
in that order, because that is the order you will hit them.

The deep references for each layer live in dedicated docs (linked at
the end of every section). This page is the connective tissue: enough
to get unstuck without context-switching, but not a replacement for
the per-topic deep-dives.

> Stack as of 2026-06-01:
> Genesis `v12.0.0` (317 PATCH_REGISTRY entries) ·
> vLLM `0.23.1rc1.dev148+gb4c80ec0f` (previous / rollback: `dev101`) ·
> Reference rig: 2× RTX A5000 24 GB · driver ≥ 580.126 · CUDA 13.

## 1. What you are running

Genesis is a **runtime patch package** for vLLM, not a fork. At every
process start the plugin attaches itself via vLLM's
`vllm.general_plugins` entry point and applies 317 small surgical
changes — text edits at known anchors, class-rebind wrappers, FastAPI
middleware — that together turn an out-of-the-box vLLM into a
production-grade Qwen3.6 inference server on consumer NVIDIA
hardware. Patches retire automatically when upstream merges the
underlying fix; the dispatcher keeps a registry, a per-pin
applicability window, and an audit trail.

Four practical entry points exist:

| Layer | Command | When to use |
| --- | --- | --- |
| **Installer** | `install.sh` | First setup on a fresh host. |
| **Launcher** | `sndr launch <preset>` | Boot any preset (V1 monolithic or V2 alias). |
| **Configs** | `sndr model-config` + `sndr config` | Inspect, edit, scaffold, validate presets. |
| **Patches** | `sndr patches`, `vllm/sndr_core/integrations/` | Browse the catalogue, author new ones. |

Everything else (`sndr doctor`, `sndr verify`, `sndr memory`,
`sndr deps`, `sndr upstream`, …) is operator instrumentation around
these four. The full subcommand list is `sndr --help`; per-subcommand
detail is [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

---

## 2. Installer — `install.sh`

The installer is a single Bash script (~700 lines, dependency-free
beyond `git` + `curl` + Python 3.10+) that takes a fresh host to a
working Genesis install in **3-5 minutes**.

### What it does

1. **Pre-flight** — verifies OS, Python ≥ 3.10, disk budget (~80 GiB
   free), Git, curl. Refuses to continue if any are missing rather
   than failing midway through.
2. **GPU detection** — runs `nvidia-smi --query-gpu` to capture model
   / count / VRAM / SM compute capability. Used to auto-match a
   preset later. Skipped gracefully on hosts without GPU (you can
   still install for offline preset browsing).
3. **vLLM detection** — finds an existing vLLM install via `import
   vllm` probe; records the pin string. The installer does not
   install vLLM itself — that is your runtime/container concern.
4. **Proxmox VE caveat probe** — surfaces the
   uvloop-on-PVE-kernel-6.17 crash (noonghunna/club-3090#49) so
   operators auto-route to bare-metal install when applicable.
5. **Workload picker** — at most one interactive question
   (`balanced` / `long_context` / `high_throughput` / `tool_agent`).
   Skipped if `--workload` or `-y` is given.
6. **Pin resolution** — `stable` (latest tag), `dev` (dev branch
   tip), or any `--pin <commit/tag>`.
7. **Clone** — into `$SNDR_HOME` (default `~/.sndr`; legacy
   `$GENESIS_HOME` honoured). Idempotent: re-running pulls + resets.
8. **Plugin install** — `pip install -e tools/genesis_vllm_plugin`.
   Registers `vllm.general_plugins.genesis_v7` so vLLM auto-loads
   Genesis in main process + engine + every worker rank. Installs the
   `sndr` and `genesis` console scripts (the latter is the legacy
   compat-CLI surface, kept for back-compat).
9. **PYTHONPATH wire** — adds the clone directory to PYTHONPATH so
   `import vllm.sndr_core` resolves at runtime.
10. **Launch script generation** — picks a preset for your detected
    `(GPU × workload)` combo and writes a runnable launch script
    under `$SNDR_HOME/launch/`.
11. **Smoke verify** — runs `sndr verify --quick`, which loads a tiny
    model and fires 10 inferences. Non-fatal: failure prints a
    diagnostic hint, install still completes.

### Invocation forms

```bash
# Default — interactive workload prompt, default pin
curl -sSL https://raw.githubusercontent.com/Sandermage/genesis-vllm-patches/main/install.sh | bash

# Non-interactive — pin + workload + skip prompt
curl -sSL .../install.sh | bash -s -- --pin dev --workload tool_agent -y

# Manual clone path (no installer)
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
pip install -e tools/genesis_vllm_plugin
```

### Flags

| Flag | Effect |
| --- | --- |
| `--pin <ref>` | `stable` / `dev` / any commit / tag / branch (default: stable). |
| `--workload <name>` | `balanced` / `long_context` / `high_throughput` / `tool_agent`. |
| `--home <path>` | Override `$SNDR_HOME`. |
| `--python <bin>` | Override `python3`. |
| `--no-verify` | Skip post-install smoke test. |
| `--no-plugin` | Skip the `pip install -e` of the plugin (PYTHONPATH-only mode). |
| `--bare-metal` | Skip docker hints; print bare-metal `vllm serve` recipe. Auto-enabled on Proxmox VE 8.x. |
| `--system` | Use system pip (default: `--user`). |
| `--uninstall` | Remove Genesis + plugin entry point. |
| `-y` | Non-interactive (use defaults). |

Env-var equivalents: `GENESIS_REPO`, `GENESIS_HOME`, `GENESIS_PIN`,
`GENESIS_WORKLOAD`, `GENESIS_NON_INTERACTIVE`, `GENESIS_NO_VERIFY`,
`GENESIS_NO_PLUGIN_INSTALL`, `PYTHON_BIN`.

### Recovery

If anything broke mid-install:

```bash
bash install.sh --uninstall                  # clean slate
bash install.sh --pin dev --workload balanced -y --no-verify
```

Both paths are idempotent; re-running is always safe.

Deep reference: [`INSTALL.md`](INSTALL.md).

---

## 3. Launcher — `sndr launch`

The launcher is a thin orchestration layer over `vllm serve`. It
resolves a preset → renders the launch script → runs preflight →
boots vLLM with Genesis env exports → streams the structured Genesis
boot summary.

### Day-to-day commands

```bash
# Boot a preset
sndr launch prod-qwen3.6-35b-balanced                  # V2 alias (3-pointer model+hw+profile)
sndr launch a5000-2x-35b-prod         # V1 monolithic key (legacy form)

# Inspect the rendered command without booting
sndr launch prod-qwen3.6-35b-balanced --dry-run

# Preflight only — env, mounts, GPU, quant args coherent — no boot
sndr launch prod-qwen3.6-35b-balanced --preflight-only

# Override config port
sndr launch prod-qwen3.6-35b-balanced --port 8101

# Strict image digest mode — refuse to boot if local image ≠ preset's
sndr launch prod-qwen3.6-35b-balanced --strict-image on
```

### What happens on boot

1. **Resolve** the alias / key. V2 aliases resolve to a composed
   `ModelConfig` via `model + hardware + profile + runtime`
   pointers. V1 keys are loaded directly from
   `vllm/sndr_core/model_configs/builtin/*.yaml`.
2. **Preflight** — mount paths exist, GPU count matches preset,
   declared vLLM pin matches `$VLLM_BUILD_COMMIT` (or the
   `--strict-image` setting governs the response), quantization
   args don't conflict, VRAM budget fits.
3. **Render** — emit a `docker run …` (or `podman` / `bare_metal` /
   `kubernetes` / `lxc_proxmox`) command with every env var, mount,
   port, and CLI flag set. Use `--dry-run` to inspect it.
4. **Apply** — call the orchestrator which walks the patch registry
   and emits a structured boot summary identifying which patches
   applied, skipped (env off / model-mismatch), or failed.
5. **Boot** — `docker run … vllm serve …` (or platform-equivalent).

First boot takes 2–5 minutes (Triton kernel JIT, CUDA graph capture).
Warm restarts of the same image+config are 30–90 seconds.

### Other deployment runtimes

`sndr model-config render <preset> --runtime <name>` emits the right
artefact:

| `--runtime` | What you get |
| --- | --- |
| `docker` *(default)* | `docker run …` shell script. |
| `bare_metal` | `python3 -m venv` + `vllm serve …` shell script. |
| `podman` | docker render with podman binary + `--device nvidia.com/gpu=all`. |
| `kubernetes` | Single-stream Deployment + Service + ConfigMap manifest. |
| `lxc_proxmox` | Runnable Proxmox VE LXC bootstrap script. |

The k8s and Proxmox lifecycles also expose `sndr service install /
start / stop / status / logs / uninstall` symmetry for end-to-end
lifecycle management.

### Capture an existing running container

If you already hand-tuned a container by trial-and-error:

```bash
sndr model-config new my-rig --from-running vllm-test-container
```

The captor reads `docker inspect` (entrypoint + cmd + env + mounts)
and reverse-engineers a `ModelConfig` YAML you can validate, edit,
and launch.

Deep reference: [`CLI_REFERENCE.md` → `sndr launch` /
`sndr model-config`](CLI_REFERENCE.md).

---

## 4. Configs — how presets work

A **preset** = the full set of inputs vLLM needs to boot one specific
production workload: model checkpoint path, tensor-parallel count,
context length, quantization scheme, speculative-decode draft model,
every `GENESIS_ENABLE_*` env, every Docker mount, every NCCL tunable,
the bench reference metrics. One YAML file = one reproducible boot.

### Schema history — V1 retired, V2 canonical

| Schema | Where | Shape | Status |
| --- | --- | --- | --- |
| **V1 (monolithic)** | `vllm/sndr_core/model_configs/builtin/<key>.yaml` (flat) | One big YAML containing everything (model + hardware + env + genesis_env + docker mounts + reference_metrics). | **Retired 2026-06-01** (Phase 10 sunset cascade, commit `607385f1`) — every shipped V1 YAML deleted. |
| **V2 (layered)** | `vllm/sndr_core/model_configs/builtin/{model,hardware,profile,presets}/` | Three pointer files (`model_id`, `hardware_id`, `profile_id`) wired together by a 3-line alias YAML under `presets/`. Composed on load. | **Canonical** — only active schema. |

V2 is the operator interface — adding a new rig means writing one
hardware YAML and reusing every model file. The V1+V2 resolver
(`_resolve_preset_v1_or_v2`) still accepts a V1 key as an opaque arg
for back-compat dispatch, but every shipped V1 file is gone, so a
bare V1 key resolves to a clean "preset not found" error. New configs
MUST be written in V2 form.

### Browsing

```bash
sndr model-config list                # all configs the registry sees
sndr config diff prod-qwen3.6-35b-balanced prod-qwen3.6-27b-tq-k8v4 # field-by-field diff
sndr config explain prod-qwen3.6-35b-balanced          # plain-English walkthrough
sndr model-config show prod-qwen3.6-35b-balanced       # the resolved full YAML
sndr patches plan prod-qwen3.6-35b-balanced            # preview which patches will apply
```

The 12 builtin configs and their reference metrics are
auto-inventoried in [`CONFIGS_AUTO.md`](CONFIGS_AUTO.md). The
narrative recipe for adding your own (V2 layered) is in
[`MODELS.md`](MODELS.md).

### What a preset gives you

Beyond the obvious model + hardware fields, every preset carries:

- **`genesis_env`** — the dict of `GENESIS_ENABLE_*` flags that gate
  individual patches. Editing one flag here toggles one patch in
  production.
- **`system_env`** — non-Genesis env (NCCL, CUDA_VISIBLE_DEVICES,
  PYTORCH_CUDA_ALLOC_CONF, …).
- **`docker.image_digest`** — pin against image drift; the launcher
  refuses to boot when local ≠ preset under `--strict-image=on`.
- **`reference_metrics`** — TPS / TPOT / CV / tool-call score
  captured on a validated rig. `sndr model-config verify <preset>`
  re-runs the bench and compares against these values.
- **`mounts`** — explicit mount list so models / triton-cache /
  evidence-dir / huggingface-cache aren't accidentally bind-mounted
  from the wrong host directory.
- **`network.port`** — single source of truth; the launcher reads
  this rather than guessing.

### Editing

Three valid editing flows:

1. **`sndr model-config new <key> --from-running <container>`** —
   capture an existing tuned container.
2. **Copy + diff** — `cp` a close-enough builtin preset to
   `vllm/sndr_core/model_configs/community/<key>.yaml`, edit fields,
   then `sndr config diff <yours> <closest-builtin>` to verify your
   delta is small + intentional.
3. **`sndr config new --from-detect`** — scaffold a starter YAML
   from auto-detected hardware shape; fill in the rest.

After editing:

```bash
sndr model-config validate <key>     # schema + cross-field audit
sndr model-config preflight <key>    # env + mount + GPU + pin
sndr launch <key> --dry-run          # render without booting
```

Deep reference: [`MODELS.md` § "Adding a model recipe"](MODELS.md),
[`CONFIGURATION.md`](CONFIGURATION.md) (per-env-var tuning catalogue).

### V1 ↔ V2 parity

After the 2026-05-16 audit closure, the V1 monolithic configs and the
V2 layered presets stay structurally in sync (10 missing-patch +
2 stale-value differences were closed in commit `a669325f`). Drift
between the two schemas is now a CI gate
(`audit_v2_cross_reference`). When you author a new config, picking
either schema is fine; the cross-reference gate enforces consistency
if both forms exist for the same logical preset.

---

## 5. Patches — registry, applying, authoring

The patch registry is the heart of Genesis. 317 entries live in
`sndr/dispatcher/registry.py` as a single Python dict.
Each entry declares: id, title, family, env_flag, default_on,
lifecycle, applies_to (hardware/model gates), conflicts_with,
requires_patches, upstream PR reference, anchor manifest entry.

The wiring code lives in `vllm/sndr_core/integrations/<family>/`. At
process start the orchestrator walks the registry, evaluates each
`applies_to` predicate against the live runtime, and calls the
patch's `apply()` for entries whose env-flag is on. Three possible
returns: `("applied", reason)`, `("skipped", reason)`,
`("failed", reason)`. The boot summary surfaces all three.

### Browsing the catalogue

```bash
sndr patches list                                # full registry, paged
sndr patches list --default-on                   # only on-by-default
sndr patches list --lifecycle stable             # only stable-promoted
sndr patches list --family attention.turboquant  # one family

sndr patches explain P67                         # per-patch deep-dive
sndr patches plan prod-qwen3.6-35b-balanced                       # which patches the preset enables
sndr patches doctor                              # registry contract sanity
```

The full per-patch table is auto-generated:
[`PATCHES_AUTO.md`](PATCHES_AUTO.md). The narrative reference (per
family, with compatibility matrices) is [`PATCHES.md`](PATCHES.md).
Design appendices for the load-bearing patches (PN95 tier-aware KV
cache, GDN kernel fusion, MTP streaming): [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md).

### Lifecycle states

| State | Meaning |
| --- | --- |
| `experimental` | Default for new patches. Validated on the author's rig only. |
| `stable` | Promoted via 5-condition ratchet (patch_id, TextPatcher, anchor manifest, build_anchor_manifest passing, lifecycle audit). |
| `retired` | Upstream merged the underlying fix, or the patch was superseded. Dispatcher skips it. |
| `research` | Code-complete but not production-validated. Audit R-01 closure (2026-05-16) hard-rules these out of `production_default=eligible` regardless of impl_status. |

`sndr patches release-check --scope production-subset --mode require-bench`
filters evaluation to the 109 patches that ship in any `prod-*`
preset (the union of preset-enabled + default_on). The 76+
experimental opt-in patches outside that subset stay under the
cheaper `require-static` gate.

### Authoring a new patch

Five steps. Each has a recipe; the full version is in
[`CONTRIBUTING.md` § "How to add a new patch"](CONTRIBUTING.md).

1. **Pick a family directory** under
   `vllm/sndr_core/integrations/<family>/` (mirror the registry
   `family` field).
2. **Create `pNN_descriptive_name.py`** with a docstring stating the
   problem / solution / safety model + an `apply()` function that
   returns one of `(applied|skipped|failed, reason)`. Never raises.
3. **Register in `dispatcher/registry.py`** with full metadata: tier,
   family, env_flag (must match exactly an `env.py` Flags class
   constant — silent-ignore bugs cost real perf when this drifts),
   default_on (default False), category, credit, upstream_pr,
   implementation_status, lifecycle, applies_to (hardware / model /
   pin gates), conflicts_with, requires_patches.
4. **Add a unit test** under
   `tests/unit/integrations/<family>/test_pNN_<name>.py`. Minimum:
   anchor exists, replacement well-formed, marker present in
   replacement, idempotent on second apply.
5. **Run the suite**:

   ```bash
   python3 -m pytest -q tests/unit/integrations/<family>/
   python3 scripts/check_doc_sync.py --strict
   python3 scripts/generate_patches_md.py        # regenerate auto-doc
   python3 -m vllm.sndr_core.cli patches doctor  # registry sanity
   make evidence                                  # 43+ gates
   ```

Promoting to `lifecycle="stable"` requires a 5-step ratchet — see
[`CONTRIBUTING.md` § "Promoting a patch to lifecycle=stable"](CONTRIBUTING.md).

### Bench proof workflow (R-01 closure path)

For hardened-release readiness the registry has a per-patch proof
system. Each patch gets a JSON artefact under
`evidence/patch_proof/<id>__<vllm-pin>.json` carrying static checks +
optional bench measurements. Buckets:

| Bucket | What it means |
| --- | --- |
| `bench_with_baseline` | Static green + bench measurement + delta-percent against a prior baseline. Hardened-release target. |
| `bench_attached` | Static green + raw bench numbers (no delta). |
| `static_only` | Static green, no bench evidence (default after `sndr patches prove --all`). |
| `static_failed` / `dead` | Artefact static checks failed / no artefact at all. |

To promote a preset's patches from `static_only` → `bench_with_baseline`:

```bash
# 1. Generate static proofs (per-host, per-pin)
sndr patches prove --all

# 2. Bench the live endpoint
python3 sndr/extras/tools/genesis_bench_suite.py \
    --host localhost --port 8000 --api-key genesis-local \
    --model qwen3.6-27b --quick \
    --out tools/bench_results/<preset>_<date>.json

# 3. Attach the bench measurements to every patch the preset enables
python3 scripts/attach_bench_proof.py \
    --bench tools/bench_results/<preset>_<date>.json \
    --preset prod-qwen3.6-27b-dflash-multiconc \
    --baseline tests/integration/baselines/27b_dflash_multiconc.json

# 4. Gate the release on the subset
sndr patches release-check \
    --scope production-subset \
    --mode require-bench
```

Tools involved:

- `tools/genesis_bench_suite.py` — the canonical bench harness.
- `scripts/attach_bench_proof.py` — writes bench measurements into
  the proof JSON; with `--baseline` promotes the bucket from
  `bench_attached` → `bench_with_baseline`.
- `sndr patches release-check --scope production-subset` — gates
  evaluation to the 109 patches in any `prod-*` preset, skipping the
  experimental opt-in tail.

Per-rig per-preset deployment cycle: bench once, attach, commit the
new baseline under `tests/integration/baselines/`. The next bench
against the same preset will yield non-zero delta-percent values
that the regression budget can act on
([CLI_REFERENCE.md → `sndr patches release-check`](CLI_REFERENCE.md)).

---

## 6. Day-1 acceptance — does it actually work?

Six checks, ~15 minutes total. Each has a clear pass signal.

```bash
sndr doctor                   # hardware + software + plugin + patches
sndr verify --quick           # 10-inference smoke
sndr model-config list        # browse what is available
sndr launch <preset> --preflight-only      # config-vs-host coherence
sndr launch <preset>          # actually boot
sndr model-config verify <preset>          # bench vs reference_metrics
```

Detailed pass/fail diagnostics for each step:
[`QUICKSTART.md` § 5](QUICKSTART.md).

---

## 7. Recovery

If anything went sideways:

1. **`sndr doctor`** — most issues are environment drift; re-run.
   Pinpoints version mismatch, missing plugin registration,
   NCCL/CUDA config problems.
2. **`docker logs <container>`** — last 200 lines for the actual
   boot error. Look for `[Genesis] FAILED:` lines (anchor drift),
   `OOM` (memory budget), `[FAIL]` for the per-patch apply table.
3. **`docs/TROUBLESHOOTING.md`** — named cliffs (1–8), OOM recipes,
   the operational cookbook (10 named scenarios), and the rollback
   playbook (R-001 … R-008) all live there.
4. **Open an issue** with the output of `sndr doctor --json`
   attached.

For the rollback playbook specifically (revert a regressing
deployment without losing model weights or evidence):
[`TROUBLESHOOTING.md` § "Rollback playbook"](TROUBLESHOOTING.md).

---

## 8. What to read next

| Topic | Doc |
| --- | --- |
| Installer flags + troubleshooting | [`INSTALL.md`](INSTALL.md) |
| Per-subcommand `sndr` reference | [`CLI_REFERENCE.md`](CLI_REFERENCE.md) |
| Every `GENESIS_*` env var | [`CONFIGURATION.md`](CONFIGURATION.md) |
| Model catalogue + adding your own | [`MODELS.md`](MODELS.md) |
| Hardware envelope (3090 / 4090 / 5090 / A5000 / H100 / …) | [`HARDWARE.md`](HARDWARE.md) |
| Patch catalogue + compatibility matrix | [`PATCHES.md`](PATCHES.md) |
| Patch design appendices (PN95 / GDN / MTP) | [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md) |
| Bench methodology + canonical numbers | [`BENCHMARKS.md`](BENCHMARKS.md) |
| OOM / cliff / rollback recipes | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| Authoring a patch end-to-end | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Release-policy modes (require-static / -bench / -baseline) | [`RELEASE_POLICY.md`](RELEASE_POLICY.md) |
| Glossary (TQ / MTP / GDN / FA2 / A3B / Marlin / …) | [`GLOSSARY.md`](GLOSSARY.md) |
