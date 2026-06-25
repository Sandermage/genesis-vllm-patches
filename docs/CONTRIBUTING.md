# Contributing to Genesis vLLM Patches

Thanks for considering a contribution. Genesis is a runtime-patch package for vLLM, focused on running Qwen3.6 (and other long-context, hybrid, and spec-decode-heavy models) on consumer Ampere GPUs (RTX 3090, RTX 4090, RTX A5000) without forking vLLM itself.

This guide covers how to file useful issues, add a new patch, add a new launch recipe, and what review looks like. The maintainer is Sander (Александр Барзов, Odessa, Ukraine). The project is licensed under Apache-2.0 — by submitting a contribution you agree it is licensed under the same terms.

---

## Welcome and scope

### What we accept

- **Bug fixes for existing patches.** Anchor drift on a new vLLM pin, off-by-one in a Triton kernel, missing guard, etc.
- **New patches with empirical evidence.** A bug or a measurable speed-up, with a reproducer and `n >= 3` benchmark runs.
- **Doc improvements.** Typos, clarifications, broken links, missing cross-references.
- **New model recipes.** Launch scripts for models we don't ship today (Llama, Mistral, Gemma, DeepSeek, Qwen variants), provided you tested boot + a tool-call sanity check.
- **New launcher recipes.** Container compose files, systemd units, k8s manifests — as long as they're tested.
- **Cross-engine learnings.** If you found a relevant fix in SGLang, TensorRT-LLM, or llama.cpp, please open an issue with a link. Even if you can't port it yourself, it's valuable.

### What we don't accept (yet)

- **Forks of vLLM itself.** Genesis is deliberately a *runtime patch package* — we monkey-patch vLLM at boot. PRs that vendor or fork vLLM source are out of scope.
- **Kernels requiring AMD ROCm, CPU-only, or XPU port.** Genesis is Ampere-focused (sm_86, sm_89, sm_90 best-effort). Contributions that *guard* existing kernels behind GPU detection are welcome; contributions that port them away from CUDA are not.
- **Speculative architectural rewrites without empirical backing.** "This *should* be faster" is not enough. Show numbers.

If you're not sure whether your idea fits, open a Discussion first. Cheap to ask.

---

## How to add a new patch

Step-by-step. Read [PATCHES.md](PATCHES.md) (which absorbs the former `COMPATIBILITY.md` content) first to understand the conventions.

### 1. Pick the right directory

`sndr/engines/vllm/patches/` (path updated 2026-06-01 v12 refactor; was `vllm/sndr_core/integrations/` in v11, `wiring/` pre-v11) is split by `family` — same vocabulary as registry's `family` field. Current 20 top-level family directories on disk; the registry adds three further sub-family tags (`attention.gdn`, `attention.flash`, `attention.turboquant`) that all live under `integrations/attention/`, bringing the canonical PATCH_REGISTRY family count to 23:

| Directory | What goes here |
| --- | --- |
| `attention/` | FA2/FA3 backends, GDN, Mamba, TurboQuant (subfolders for each variant) |
| `compile_safety/` | torch.compile guards, cudagraph capture safety, custom_op registration |
| `kernels/` | Triton/CUDA kernel hooks (the kernels themselves live in `sndr/engines/vllm/kernels/`) |
| `kv_cache/` | KV cache dtype, TurboQuant KV, prefix caching, hash backends |
| `loader/` | Weight loading, quantization checkpoint loaders, model arch routing |
| `lora/` | LoRA adapter integration |
| `memory/` | Allocator scoping, fragmentation mitigation, cache release timing |
| `middleware/` | Logging, metrics, telemetry, lazy-reasoner request hooks, observability |
| `moe/` | Fused MoE, router softmax, expert intermediate cache |
| `multimodal/` | Vision encoders, multimodal scratch sizing |
| `observability/` | Per-patch metrics, audit instrumentation |
| `offload/` | CPU offload, PrefetchOffloader pinned-allocator pool (PN102) |
| `quantization/` | FP8 block-scaled, AutoRound row-parallel, FP8 lm_head |
| `reasoning/` | Reasoning parsers, `<think>` handling, multi-turn boundary fixes |
| `scheduler/` | Async scheduling, batching, profile_run caps |
| `serving/` | OpenAI-compatible chat-completions, stream generators, MTP truncation detector |
| `spec_decode/` | MTP, ngram, DFlash, rejection sampling, draft acceptance, prepare_next_token |
| `streaming/` | Streaming-response orchestration helpers (chunked tool-call assembly, partial-tag guards) |
| `tool_parsing/` | Qwen3Coder XML fallback, tool-call argument parsing |
| `worker/` | gpu_model_runner integrations, profile_run, thinking-budget, prompt_logprobs |

Pick the closest match (mirror the registry `family` field). If genuinely unclear, place under `worker/` or `serving/` and the reviewer will move it. Cross-cutting changes split into multiple files in different families.

### 2. Create `patch_NN_descriptive_name.py`

`NN` is the next free integer in the project (check [../docs/PATCHES.md](../docs/PATCHES.md) — don't reuse). Name should be terse and grep-friendly: `patch_67_tq_multi_query_kernel.py`, not `patch_67_fix.py`.

Scaffold:

```python
"""Genesis Patch NN — short title

Problem: What breaks or runs slow today, in concrete terms.

Solution: What this patch does, at the level a reviewer can verify.
Mention which upstream file(s) get text-patched.

Who benefits: Workload + hardware combinations where this patch helps.

Safety model:
- default_on: True/False and why
- env flag: GENESIS_PNN
- conflicts_with / requires_patches
- failure mode if anchor drifts (silent skip vs hard fail)

Attribution: Genesis-original / port of <upstream PR> / cross-engine learning.
"""
from sndr.kernel import TextPatcher, TextPatch  # path updated 2026-06-01 v12 refactor (was vllm.sndr_core.core in v11, sndr_core.wiring.text_patch pre-v11)

GENESIS_PNN_MARKER = "Genesis PNN v7.NN_descriptive_name"


def apply():
    """Apply patch NN. Returns (status, reason)."""
    # Anchor must be VERBATIM upstream code (copy-paste, no whitespace edits)
    sub_patches = [
        TextPatch(
            name="main_anchor",
            anchor='''<verbatim upstream lines>''',
            replacement=f'''<replacement that includes {GENESIS_PNN_MARKER}>''',
            required=True,
        ),
    ]
    patcher = TextPatcher(
        patch_name="PNN <title>",
        target_file="<full path>",
        marker=GENESIS_PNN_MARKER,
        sub_patches=sub_patches,
    )
    return patcher.apply()
```

`apply()` must return one of:
- `("applied", "<reason>")` — all required sub-patches landed
- `("skipped", "<reason>")` — env flag off, or already applied (idempotent), or applies_to filter rejected the model
- `("failed", "<reason>")` — anchor missed and `required=True`, or unexpected exception

Never raise out of `apply()` — wrap with `try/except` and return `("failed", str(e))`. Boot must continue even if a single patch breaks.

### 3. Register in `sndr/dispatcher/registry.py`

> Path updated 2026-05-11 audit (was `dispatcher.py` pre-v11 refactor — now split across `dispatcher/{registry,spec,decision,audit}.py`).

Add an entry to `PATCH_REGISTRY` dict with full metadata:

```python
"PNN": {
    "title": "TurboQuant multi-query Triton kernel",
    "tier": "community",                          # community (Apache 2.0) or engine (commercial — sndr_engine namespace, currently empty)
    "family": "attention.turboquant",             # one of 23 families — see docs/PATCHES_AUTO.md "By family" section
    "env_flag": "GENESIS_ENABLE_PNN",              # MUST match exactly env.py Flags class constant — verify before commit!
    "default_on": False,                          # False unless validated on 2+ models
    "category": "kernels",
    "credit": "Genesis-original (Sander)",        # or "port of vllm#NNNNN by @author"
    "upstream_pr": None,                           # or 41268 for backports
    "implementation_status": "full",              # full|partial|scaffold|placeholder — anything < full = not PROD-eligible
    "lifecycle": "experimental",                  # experimental → tested → stable (via ratchet; see STABLE_PROMOTION_CHECKLIST)
    "applies_to": {
        "is_turboquant": [True],
        # Pin-gate (recommended; see "Pin-bump playbook" below). Declares
        # the vllm version range the patch is validated against. Out-of-range
        # pins get a clean dispatcher skip with reason "VERSION:..." instead
        # of crashing on a moved anchor.
        #   PROD-critical broad range:  (">=0.20.0", "<0.21.0")
        #   Anchor-specific tight range: (">=0.20.2rc1.dev9", "<0.21.0")
        #   Retire (upstream merged):    "<0.20.2rc1.devN" (gates OFF post-merge)
        "vllm_version_range": (">=0.20.0", "<0.21.0"),
    },
    "conflicts_with": ["P65"],                    # patches that mutate the same code path
    "requires_patches": ["P4"],                   # patches that must be applied first
},
```

**CRITICAL: env_flag exact match** — `env.py` Flags class enumerates canonical short or long form for each. Wrong-name `-e GENESIS_ENABLE_X=1` is silently ignored (audit 2026-05-11: P94/P103 short form bug cost ~6% TPS until fixed). Verify:
```bash
grep "PNN" sndr/env.py  # must find canonical form
```

`default_on=True` reserved для bug fixes validated на 2+ workloads. New patches start False; promoted в later PR after bench.

`lifecycle="stable"` blocked by ratchet (see [CONTRIBUTING.md § Promoting a patch to lifecycle=stable](CONTRIBUTING.md#promoting-a-patch-to-lifecyclestable)): **14** currently. Don't claim stable без green ratchet.

After registering, regenerate auto-docs + verify:
```bash
python3 scripts/generate_patches_md.py        # regenerate docs/PATCHES_AUTO.md
python3 scripts/check_doc_sync.py --strict    # verify README/PATCHES/etc counts in sync
```

### 4. Add a unit test

`tests/unit/integrations/<family>/test_pNN_<name>.py` (path updated 2026-05-11 — pre-v11 was `vllm/sndr_core/tests/`). Minimum coverage:

- Anchor exists in current vLLM pin (read the file, assert substring present).
- Replacement is well-formed (parseable Python if it's a Python text-patch).
- Marker is in the replacement.
- After-apply state is idempotent (running `apply()` twice is a no-op on the second call).

For kernel patches, add a CPU-only smoke test with tiny shapes if at all possible. If the patch *cannot* be tested without a GPU, mark the test with `@pytest.mark.gpu` and document that in the PR.

### 5. Run the test suite

```bash
python3 -m pytest tests/unit/ -v
```

(Path updated 2026-05-11: pre-v11 was `vllm/sndr_core/tests/`.) Must pass. CI will gate the PR on this.

For new patches, also verify the registry meta-tests pass:

```bash
python3 -m pytest tests/unit/dispatcher/test_pin_gate.py tests/unit/dispatcher/test_iron_rule_11_enforcement.py -v
```

These enforce pin-gate adoption discipline + iron-rule-#11 retire provenance.

### 6. Bench empirically

On the GPU you have access to, run:

```bash
python tools/genesis_bench_suite.py \
    --base-url http://localhost:8000/v1 \
    --model <served-name> \
    --runs 5 \
    --output bench_pNN.json
```

Report `wall_TPS` mean, std, and CV. If CV > 8%, do more runs or investigate noise (other tenants on the box, thermal throttling, etc.). Keep the JSON — paste the summary in the PR description.

### 7. Open the PR

Required PR contents (see [PR template below](#commit-and-pr-style)):
- Problem statement (1-2 sentences).
- Solution summary (what files, what change).
- Evidence (`n >= 3` bench runs, before/after numbers, CV).
- Risk (boot failure modes, regressions on other model families).
- Tested-on (model + quant + GPU + vLLM pin).
- If applicable: link to the upstream PR/issue you're porting or to the cross-engine source.

---

## How to add a new launch script

Genesis ships launchers under `scripts/`. Adding a new one for your model is a great first contribution.

### 1. Choose a name

Convention: `start_<MODEL>_<KV>_<MODE>.sh` for OpenAI-API server launches, `bare_metal_<MODEL>_<KV>_<MODE>.sh` for offline/throughput runs.

Examples in-tree: `start_27b_int4_TQ_k8v4.sh`, `start_35b_fp8_PROD.sh`, `start_27b_int4_fp8_e5m2_long_256K.sh`.

### 2. Copy from the closest existing template

Don't write from scratch. The existing scripts encode hard-won env-var settings (CUDA visible devices, NCCL timeouts, allocator tuning) that you almost certainly want.

### 3. Update three things

- `--model` and `--served-model-name`
- Genesis env flags (`GENESIS_ENABLE_PNN=1` for whatever subset you tested)
- vLLM serve flags relevant to your model (max-model-len, gpu-memory-utilization, spec-config, KV dtype)

### 4. Test boot and a tool-call

```bash
bash scripts/start_<your>.sh > boot.log 2>&1 &
# wait for "Application startup complete"
curl http://localhost:8000/v1/models
# tool-call sanity (sample in QUICKSTART.md)
```

Boot log must show `[GENESIS]` summary with all expected patches `APPLY` (no `FAILED`).

### 5. Bench

`n=5` runs with `tools/genesis_bench_suite.py`. Include the numbers in the PR.

### 6. Open the PR

Same template as patch PRs but the focus is reproducibility: someone with the same GPU should be able to copy your script and get within ~5% of your numbers.

---

## Audit & maintenance tools

Genesis ships with several automated audit scripts that close recurring drift classes. Operators (and CI) run them periodically; the iron rule #11 enforcement is gated automatically.

| Tool | Purpose | Run |
| --- | --- | --- |
| [`scripts/audit_upstream_status.py`](../scripts/audit_upstream_status.py) | Cross-references PATCH_REGISTRY `upstream_pr` fields against GitHub merge state. Surfaces actionable iron-rule-#11 retire candidates. 9-category classification (NEWLY-MERGED, STALE-RETIRED, ISSUE-CLOSED, INTENTIONAL-INVERSE, ENABLES-UPSTREAM, RETIRED-INTERNAL, ERROR, WATCH, SUPERSEDED-OK). | `python3 scripts/audit_upstream_status.py` (weekly via [upstream_audit_status.yml](../.github/workflows/upstream_audit_status.yml)) |
| [`scripts/emit_paths_env.py`](../scripts/emit_paths_env.py) | Renders canonical paths from `project_paths.py` as a sourcable bash env file. Operator workflow: `python3 scripts/emit_paths_env.py > ~/.genesis_paths.env`; start-scripts source it. | `python3 scripts/emit_paths_env.py` (modes: `--print`, `--prefix SNDR`) |
| [`scripts/check_doc_sync.py`](../scripts/check_doc_sync.py) | Parses registry.py count vs README/PATCHES/INSTALL/MODELS/BENCHMARKS doc claims. CI gate. | `python3 scripts/check_doc_sync.py --strict` |
| [`scripts/generate_patches_md.py`](../scripts/generate_patches_md.py) | Auto-gen `docs/PATCHES_AUTO.md` from registry. | `--check` mode in CI; bare run regenerates |
| [`scripts/generate_configs_md.py`](../scripts/generate_configs_md.py) | Auto-gen `docs/CONFIGS_AUTO.md` from builtin YAMLs. | same pattern |
| [`tools/audit_yaml_vs_runtime.sh`](../tools/audit_yaml_vs_runtime.sh) | YAML `genesis_env` vs `docker inspect` env drift. | `bash tools/audit_yaml_vs_runtime.sh <yaml> <container> [<ssh_host>]` |
| [`scripts/audit_override_policy.py`](../scripts/audit_override_policy.py) | CONFIG-UX.4: every profile with `sizing_override` carries a matching `override_policy`. Four Class-4 forbidden predicates fire unconditionally (GMU > 1.0, TP > n_gpus, kv-dtype downgrade, spec-decode method change). | `make audit-override-policy`; `--strict` exits 1 on any finding |
| [`scripts/audit_v1_migration.py`](../scripts/audit_v1_migration.py) | CONFIG-UX.4.1: classifies legacy V1 keys into transparent / needs\_operator\_choice / deprecated / tombstone buckets via the frozen `_v1_migration_table.json`. | `make audit-v1-migration`; severity matrix follows `SNDR_V1_ROLLOUT_STAGE` |
| [`scripts/audit_config_catalog.py`](../scripts/audit_config_catalog.py) | CONFIG-UX.audit: preset card shape + `production_candidate` evidence-visibility hints. | `make audit-config-catalog` |
| [`scripts/audit_generated_config_catalog.py`](../scripts/audit_generated_config_catalog.py) | CONFIG-UX.5.1: derived catalog regenerates deterministically AND no private paths leak. | `make audit-generated-config-catalog` (default informational; `--strict` for CI) |
| [`scripts/audit_public_docs.py`](../scripts/audit_public_docs.py) | D-1..D-8 public-docs boundary: no maintainer-private-tree links, no RFC-1918 IPs, no operator paths, no server container names, no retired CLI verbs, no unresolved TODOs/placeholders, no stale-version-as-current anchors, no stale-pin-as-current anchors. | `make audit-public-docs` |

### Release-strict Stage 2 (CONFIG-UX.4.2)

`make evidence-release` runs the three CONFIG-UX audits with
`SNDR_V1_ROLLOUT_STAGE=2` exported, which flips the severity matrix
so non-tombstone buckets escalate from WARN to ERROR:

```bash
make evidence-release        # local mirror of tag-push CI
```

The release-strict job lives in
[`.github/workflows/release.yml`](../.github/workflows/release.yml)
(`rollout_strict_audit`). It fires only on tag push; PR CI uses
Stage 1 defaults. The gate runs:

- `audit_override_policy.py --strict` — fails on any
  `missing_override_policy` OR Class-4 forbidden predicate.
- `audit_v1_migration.py --strict` — fails on `tombstone` bucket
  hits and on any `deprecated` key still referenced by an active
  preset.
- `audit_generated_config_catalog.py --strict` — fails on
  non-deterministic regeneration or any private-path leak.

Operator escape hatch: a one-release transition window remains
available via `SNDR_V1_ROLLOUT_STAGE=0` (explicit revert to
informational). Class-4 errors remain ERROR regardless — they encode
physics, not rollout policy. See
[`CONFIGURATION.md`](CONFIGURATION.md) → "Override policy" and
"Rollout staging" for the operator-facing surface.

### Iron rule #11 (retire provenance)

Every patch where upstream merged an equivalent must declare its supersession:

```python
"PN52": {
    ...
    "lifecycle": "retired",  # ← required
    "superseded_by": "vllm#41411 (merged 2026-05-04, byte-equivalent ...)",  # ← required
    "applies_to": {
        "vllm_version_range": "<0.20.2rc1.dev209",  # ← required (pin-gate upper bound)
    },
}
```

If a patch is retired for reasons OTHER than upstream supersession (hypothesis disproven, internal-only retire), add the patch ID to one of the waiver constants in [test_iron_rule_11_enforcement.py](../tests/unit/dispatcher/test_iron_rule_11_enforcement.py):

- `_RETIRED_NO_SUPERSEDE_WAIVER` — hypothesis disproven / internal-only retire
- `_INTERNAL_SUPERSESSION_WAIVER` — superseded by another Genesis patch
- `_INTENTIONAL_INVERSE_WAIVER` — deliberate revert of merged upstream (perf regression on our HW)
- `enables_upstream_feature: True` registry field — convenience activator/wrapper, not a backport

Iron-rule-#11 meta-test enforces this on every PR; missing provenance fails the build.

---

## Pin-bump playbook

Genesis is a runtime-patch package — every text-patch anchors verbatim upstream lines. When vLLM advances its pin, anchors can move and patches degrade silently. The pin-gate (`applies_to.vllm_version_range` + `KNOWN_GOOD_VLLM_PINS` allowlist + per-patch drift detector) is the safety net. Adopting it correctly is the difference between a clean bump and a 6-hour debug session.

### Pin-gate components

| Layer | File | Purpose |
| --- | --- | --- |
| **Allowlist (boot)** | [`sndr/engines/vllm/detection/guards.py`](../sndr/engines/vllm/detection/guards.py) `KNOWN_GOOD_VLLM_PINS` | Boot-time `assert_vllm_pin_allowed` — orchestrator logs WARN if running pin not in tuple. Strict mode (`GENESIS_STRICT_PIN=1`) `sys.exit(2)`. |
| **Per-patch range** | `PATCH_REGISTRY[<id>]["applies_to"]["vllm_version_range"]` (PEP 440 spec tuple) | `decision._check_applies_to` calls `compat.version_check.check_version_constraints` — patches out-of-range return `(False, "VERSION: ...")` and dispatcher skips cleanly. |
| **Anchor drift** | `TextPatcher.apply` returns `("skipped", "anchor not found")` | Last line of defense — if anchor moved on a "validated" pin, drift detector skips with explicit reason instead of corrupting source. |

The three are **layered** — pin-gate prevents the attempt when version is known-incompatible; anchor drift catches the case where the version satisfies the range but upstream still moved the line.

### When bumping the vllm pin

1. **Discover target SHA + dev-counter.** Latest main HEAD: `gh api repos/vllm-project/vllm/commits/main --jq '.sha[0:9]'`. The dev-counter (`devN`) lands when you `pip install` the wheel.
2. **Read upstream changelog since current pin.** `gh api repos/vllm-project/vllm/compare/<current_sha>...main --jq '.commits | length'` for commit count; `--jq '.commits[] | .commit.message | split("\n")[0]'` for one-line subjects. Watch for breaking refactors near our anchor lines.
3. **Server-side install** in throwaway container (NOT PROD):

   ```bash
   ssh <your-user>@<your-host>
   docker run --rm -it --gpus all <base> pip install \
       "vllm @ git+https://github.com/vllm-project/vllm.git@<sha>"
   docker exec <container> python -c 'import vllm; print(vllm.__version__)'
   # → capture exact 0.20.2rc1.devN+gSHA string
   ```

4. **Boot smoke + drift detection.** Start container with Genesis dispatcher in verbose mode (`GENESIS_VERBOSE=1`); read boot log for:

   - `[Genesis pin-gate] running vllm pin = X` — confirms detection
   - `[Genesis pin-gate] OK/WARN/...` — allowlist status
   - `[Genesis dispatcher] <PID>: VERSION: ...` — patches skipped via pin-gate
   - `[Genesis dispatcher] <PID>: anchor not found` — drift skips

5. **Canonical bench.** `tools/genesis_bench_suite.py --quick --ctx 8k` on PROD-equivalent model. 27B target ≥130 TPS, 35B ≥220 TPS (Wave 8 baselines).
6. **Promote**: add validated pin to `KNOWN_GOOD_VLLM_PINS` tuple in `detection/guards.py` with full SHA + a comment naming the bench config + date + observed drift skips. Update `EXPECTED_PINS` in [`tests/unit/dispatcher/test_pin_gate.py`](../tests/unit/dispatcher/test_pin_gate.py) to match. Write CHANGELOG entry.
7. **Per-patch range updates.** For each `VERSION:` skip observed in step 4: either widen the range (patch genuinely works on new pin → relax upper bound), or document the supersession (upstream merged → tighten upper bound, set `lifecycle="retired"`).

### When the gate fires unexpectedly

A `VERSION: ...` skip on a patch you expect to apply means one of:

- The declared range is wrong (too tight, or wrong PEP 440 syntax — note dev-counter ordering rules).
- The version detector returned `None` (`compat/version_check.py`) — gate is "conservative pass" in that case, so this can't actually cause an unexpected skip; if you see one, the detector returned a parseable value that doesn't satisfy the spec.
- An operator overrode the env outside the YAML (check the start-script).

Diagnose with:

```bash
docker exec <container> python -c "
from sndr.compat.version_check import detect_versions, check_version_constraints
profile = detect_versions()
print('detected:', profile.vllm)
# replicate the patch's declared range
print(check_version_constraints({'vllm_version_range': ('>=0.20.0', '<0.21.0')}, profile=profile))
"
```

### Adding a new pin-gated patch

Step-by-step in "How to add a new patch" §3 (`applies_to.vllm_version_range`). Three common shapes:

- **PROD-active broad** (`(">=0.20.0", "<0.21.0")`) — patch is general-purpose, validated on multiple pins, no specific anchor-dependence. Default for new patches.
- **Anchor-tight** (`(">=0.20.2rc1.dev9", "<0.21.0")`) — patch text-patches a specific upstream line that didn't exist before pin X. Lower bound = pin where anchor appeared.
- **Retire upper bound** (`"<0.20.2rc1.devN"`) — upstream merged the equivalent. Upper bound = pin where upstream integration landed. Combined with `lifecycle="retired"` + a CHANGELOG note explaining the supersession.

After editing, run pin-gate tests:

```bash
python3 -m pytest tests/unit/dispatcher/test_pin_gate.py -v
```

---

## Promoting a patch to `lifecycle="stable"`

Promoting a patch from `experimental` (or unspecified) to
`lifecycle="stable"` is a contract: the patch is production-ready
and will not drift silently. The patcher framework enforces this
via the ratchet test in
[`tests/unit/infra/test_stable_manifest_policy.py`](../tests/unit/infra/test_stable_manifest_policy.py)
— marking a patch stable without performing the steps below makes
the ratchet test fail and rolls back the commit.

### Why the ratchet exists

Without it, `lifecycle="stable"` would be a marketing label.
With it, the promotion forces three concrete guarantees:

- **Drift detection precision** — manifest md5s pinpoint the exact
  anchor that changed, instead of a vague "anchor missing
  somewhere".
- **Boot speed-up** — manifest fast-path replaces O(N×M) anchor
  scan with O(1) byte-offset splice.
- **Reproducibility** — the pristine fixture committed at
  promotion time documents the exact upstream state the patch was
  validated against.

### When to promote

A patch is ready for `lifecycle="stable"` when ALL of these hold:

1. **A/B validated on the production rig** (not just dev / Mac).
2. **Reference metrics committed** in at least one builtin
   `model_config` (see `reference_metrics:` in `*-prod.yaml`).
3. **No open known regressions** carrying the patch ID.
4. **Cross-pin tested** — verified on at least two different vLLM
   pins (catches anchor brittleness early).
5. **Documented** in [`PATCHES.md`](PATCHES.md) or
   [`../CHANGELOG.md`](../CHANGELOG.md) with an operating envelope
   (what it does, what it does not, known conflicts).

If any of those is missing, keep the patch at `experimental` or
its current state.

### The five steps

1. **Add `patch_id` to every `TextPatcher`** in the wiring module.
   Naming convention: `<PatchID>.Sub-<N>` where `<N>` is 1-based
   across all TextPatchers in the patch.
   [`pn79_inplace_ssm_state.py`](../sndr/engines/vllm/patches/attention/gdn/pn79_inplace_ssm_state.py)
   is the reference example (`PN79.Sub-1` … `PN79.Sub-4` across
   four files).
2. **Add a `register_for_manifest(pristine_root)` function** that
   mirrors PN79's pattern. It registers a build-mode patcher
   pointed at the pristine fixture (not live vLLM) so
   `scripts/build_anchor_manifest.py` can populate the manifest
   without a running vLLM install.
3. **Commit a pristine fixture** to
   `tests/legacy/pristine_fixtures/<basename>.py` — a verbatim
   copy of the upstream vLLM file at the pinned commit. The
   fixture is frozen at the pinned commit; bumping
   `KNOWN_GOOD_VLLM_PINS` makes the manifest detect the md5
   mismatch and fall back to the legacy anchor scan until the
   builder is re-run.
4. **Extend `_KNOWN_REL_PATHS`** in
   `scripts/build_anchor_manifest.py` with the
   `fixture_basename → vllm_rel_path` mapping. Without this, the
   builder doesn't know where in the vLLM tree the fixture maps
   to and skips it with a warning.
5. **Build + commit the manifest**:

   ```bash
   python scripts/build_anchor_manifest.py
   git add sndr/manifests/anchor_manifest.json
   pytest tests/unit/infra/test_stable_manifest_policy.py -v
   ```

   Then bump `lifecycle` in the registry from old value to
   `"stable"` in the same commit.

### Runtime-hook patches (stable_kind extension)

The five-step ratchet assumes STABLE = TextPatcher. Production-
validated runtime-hook patches (no `TextPatcher`, no anchor) are
covered by a parallel sub-track `stable_kind = "runtime-hook"`
that requires:

- `apply_module` declared in the registry, AND
- structured `production_validated_pins` evidence (list of
  `(genesis_pin, vllm_pin)` tuples documenting the pins on which
  the patch was empirically validated).

This sub-track skips manifest checks — there is no anchor to
drift-detect — but enforces the operational-evidence contract.
PN33 and PN35 are the reference examples.

### What NOT to do

- **Do not** mark a patch stable just to clear the experimental
  label. The lifecycle field is a contract; lying about it leaks
  tech debt into production.
- **Do not** skip the pristine fixture by reading from live vLLM
  at build time. The fixture is the contract — what the patch was
  validated against. Live vLLM drifts.
- **Do not** add `patch_id` to a patcher without the manifest
  entry. The runtime fast-path will abstain via gate G3 and the
  patcher silently uses the legacy path — fine functionally, but
  you have added complexity without benefit.
- **Do not** demote a stable patch back to experimental without
  also removing the `patch_id` and manifest entry. Stale manifest
  entries are noise and confuse drift diagnosis.

## Code style

### Text patches

- **Anchors must be VERBATIM upstream.** Copy-paste from the live source file. Don't reformat, don't normalize whitespace, don't refactor while patching. If upstream uses tab indents, your anchor uses tab indents.
- **Markers must include version.** Format: `Genesis PNN v7.NN_descriptive_name`. The version is the Genesis release where this patch shipped or was last revised.
- **`required=True` for critical sub-patches.** If the patch makes no sense without this sub-patch landing, mark it `required=True` so a missed anchor surfaces as `failed` instead of silent skip.
- **`required=False` only for truly optional sub-patches.** E.g., adding a debug log alongside the real fix.
- **Defensive imports inside functions.** `apply()` should import from `vllm.*` lazily. Module-level imports break boot if the user is on a vLLM pin that renamed the module.

### Triton kernels

- Power-of-2 dims wherever possible. If you must support non-power-of-2 (e.g., GQA=24/4=6 heads-per-KV), use `next_power_of_2` + a `lane_valid` mask. Document the cliff in [TROUBLESHOOTING.md (cliffs section)](TROUBLESHOOTING.md).
- Sanitize Inf/NaN at dequant boundaries. We've been bitten by silent NaN propagation through softmax — see the v7.22 P67 sanitized variant in [PATCHES.md](PATCHES.md).
- BLOCK_SIZE / num_warps / num_stages should be configurable via env override for sweep tuning.

### General Python

- We don't enforce a formatter on contributors, but we do run `ruff` on the maintainer side. PRs may be reformatted before merge.
- Type hints encouraged on public surfaces (the modules under `sndr/dispatcher/`, `sndr/kernel/text_patch.py` (TextPatcher API), and `sndr/apply/orchestrator.py`).
- Logging via `logger = logging.getLogger("genesis.<area>.<patch_id>")` (e.g. `genesis.wiring.pn519_swa_tile_base`; the apply pipeline logs under `genesis.apply_all`). The pre-v11 `vllm._genesis` namespace was removed in v11.0.0 and the `vllm/sndr_core/` overlay was renamed to top-level `sndr/` in the v12 refactor — any new code (and any pre-v11 code being touched) must import from `sndr.*`. Print only in the boot-summary path.

---

## Testing requirements

### Per-PR minimum

- **Unit test for every wiring patch.** `test_pNN_*.py` validates anchor exists, replacement is sane, marker present, idempotent.
- **Family contract** — added automatically if your new patch belongs to one of the 23 covered families. The factory pattern means no extra work: if you put the file under `integrations/<family>/<file>.py` and register in PATCH_REGISTRY with the correct `family` field, the existing family contract covers your patch via the next pytest run.
- **Boot smoke test.** Add your patch to a launch script, run it, paste the boot log section showing `APPLY` in the PR.
- **Empirical bench.** `n >= 3` runs (5 preferred) with `tools/genesis_bench_suite.py`. Report mean, std, CV.

### Adding a new family contract

When introducing a brand-new family (not just a new patch in an existing family), add a 40-line family contract using the factory:

```python
# tests/unit/integrations/<new_family>/test_<new_family>_family_contract.py
from tests.unit.integrations._family_contract_helpers import (
    make_family_contract_class, make_family_registry_class,
)

PATCHES = [
    ("sndr.engines.vllm.patches.<new_family>.<file>", "<PATCH_ID>"),
    # ...
]

class TestNewFamilyPatchContract(
    make_family_contract_class("<new_family>", PATCHES)
):
    pass

class TestNewFamilyFamilyRegistry(
    make_family_registry_class("<new_family>", PATCHES)
):
    pass
```

For nested families (e.g. `attention.gdn`), pass `filesystem_dir="attention/gdn"` to `make_family_registry_class()`.

The factory enforces 6 invariants per patch (module importable / Genesis marker / apply() callable / env_flag documented / no top-level torch / family field matches) + 2 family-level checks (registry has all entries + filesystem matches). Refining invariants in [_family_contract_helpers.py](../tests/unit/integrations/_family_contract_helpers.py) propagates to all 23 family contracts at once.

### CI

GitHub Actions runs `python3 -m pytest tests/unit/` on every PR (path updated 2026-05-11). CPU-only — no GPU CI yet. GPU validation is the maintainer's responsibility on the staging rig. 4 explicit gates fail-fast on regression:

- **Pin-gate adoption** (`tests/unit/dispatcher/test_pin_gate.py`) — KNOWN_GOOD_VLLM_PINS allowlist drift, version range semantics
- **Iron rule #11 enforcement** (`tests/unit/dispatcher/test_iron_rule_11_enforcement.py`) — every `lifecycle="retired"` patch carries `superseded_by` + `vllm_version_range` (or explicit waiver)
- **Family contracts** (`tests/unit/integrations/`) — 23 family contracts, ~2300 tests, covering 20 of 20 integration family dirs (full coverage as of 2026-06-01 after the streaming family contract landed)
- **Upstream-status audit** (`scripts/audit_upstream_status.py --skip-network`) — informational at PR-time; strict weekly gate via [upstream_audit_status.yml](../.github/workflows/upstream_audit_status.yml)

Doc-sync gate also runs (`check_doc_sync.py --strict` + `generate_patches_md.py --check` + `generate_configs_md.py --check`).

### Integration tests

`scripts/run_validation_suite.sh` runs the full integration suite (requires GPU). Not part of CI but contributors with a GPU are welcome to run it locally and paste the summary.

---

## Commit and PR style

### Conventional commits

```
feat(patch): P88 SGLang fused_gdn_gating port (+2.1% TPS on 27B)
fix(patch): P67 anchor drift on vllm pin fe9c3d6c5
docs(cliffs): add Cliff 7 (DFlash 24GB OOM at >80K ctx)
perf(kernel): P67 LOG2E fuse +2.1% on TQ k8v4
test: add unit test for P94 prefix-cache hash backend
chore: bump pin reference in PATCHES.md
```

Allowed types: `feat`, `fix`, `docs`, `perf`, `test`, `chore`, `refactor`, `revert`.

### One patch = one commit

Squash before merge if review produced fixup commits. The final history should read as one logical change per patch.

### PR description template

```markdown
## Problem
<1-2 sentences. What breaks or what's slow.>

## Solution
<What this PR does. Which files. Which subsystem.>

## Evidence
- Bench: `n=5`, before mean=X.X TPS (CV Y.Y%), after mean=X.X TPS (CV Y.Y%), Welch p=Z.ZZ
- Reproducer: <command or test file>
- Boot log excerpt showing APPLY: <paste>

## Risk
- Boot failure if anchor drifts: <yes/no, mitigation>
- Regression possibility on <other model/quant>: <assessed how>

## Tested on
- Model: <HF name + revision>
- Quant: <none / AutoRound int4 / FP8 / ...>
- KV dtype: <auto / fp8_e5m2 / turboquant_k8v4>
- GPU: <2× A5000 / 1× 4090 / ...>
- vLLM pin: <commit sha>

## Upstream reference (if applicable)
- vLLM PR: <link>
- SGLang/TRT-LLM/llama.cpp issue: <link>
```

### Review

The maintainer reviews everything personally. Turnaround is typically 24-48 hours, longer on weekends or during a deploy push. Be patient; nudge politely after a week if no response.

---

## Security

**Do not commit:**
- Anything from `~/.claude/`, the internal-docs tree, `snapshots/`, or any path that's `.gitignore`d. <!-- audit-public-docs: allow -->
- Anything under the project's gitignored planning/research directories — these are intentionally excluded from public PRs.
- Hugging Face tokens, OpenAI keys, GitHub PATs, AWS credentials, anything in a `.env`.
- Personal data — names, emails, IPs of internal infrastructure.
- Internal sprint plans, roadmap drafts, third-party correspondence.

If you discover a security issue (e.g., a patch that allows code injection through model config), **do not open a public issue.** Use the maintainer contact in `SPONSORS.md` with details. We'll acknowledge within 72 hours and coordinate disclosure.

---

## Translation

All public docs are in **English**. This includes README, PATCHES, MODELS, CHANGELOG, CONFIGURATION, and the `docs/` tree.

**Russian translations are welcome** but live as separate files: `docs/<file>.ru.md`. Don't replace the English version. If you submit a Russian translation, the English version is the source of truth — translations track it.

The maintainer writes natively in Russian. Machine-translation help is fine for PR comments and discussions; please flag it briefly (`(translated)`) so reviewers can adjust expectations on phrasing.

---

## Communication

| Channel | Use for |
|---|---|
| GitHub Issues | Bug reports, feature requests, model recipe requests |
| GitHub Discussions | General questions, design proposals, "is this a good idea" |
| PR | Code, doc, and config changes |
| Maintainer contact (in SPONSORS.md) | Security disclosures only |

Please don't email for support questions — use Discussions so the answer helps the next person.

---

## Authoring a community patch

Two paths exist depending on how you want to ship your patch.

### Path 1 — pip-installable plugin (versioned independently)

Ship the patch as its own Python package. Genesis auto-discovers it
at boot via the `vllm_genesis_patches` entry-point group. Best when
you want to release the patch on your own cadence, separate from the
Genesis release train.

A working scaffold is in
[`tools/examples/genesis-plugin-hello-world/`](../tools/examples/genesis-plugin-hello-world/),
tested in CI (`tests/unit/compat/test_plugin_example.py`).

Minimal layout:

```text
my-genesis-plugin/
├── pyproject.toml
└── my_genesis_plugin/
    ├── __init__.py
    └── patch.py
```

`pyproject.toml`:

```toml
[project]
name = "my-genesis-plugin"
version = "0.1.0"
dependencies = []                      # Genesis is the only requirement

[project.entry-points."vllm_genesis_patches"]
my_patch = "my_genesis_plugin.patch:get_patch_metadata"
```

`patch.py`:

```python
"""Example community plugin for Genesis."""
from __future__ import annotations


def get_patch_metadata() -> dict:
    return {
        "patch_id": "MY_HANDLE_001",
        "title": "Short, descriptive title",
        "applies_to": {
            "model_archs": ["qwen3_5_moe"],
            "kv_dtypes":  ["turboquant_k8v4"],
        },
        "default_on": False,
        "lifecycle": "experimental",
        "credit": "your-name",
    }
```

Operator workflow:

```bash
pip install my-genesis-plugin
python3 -m sndr.cli plugins list
python3 -m sndr.cli plugins show MY_HANDLE_001
python3 -m sndr.cli plugins validate
```

Author etiquette:

- Open an issue at
  <https://github.com/Sandermage/genesis-vllm-patches/issues>
  describing your plugin's purpose before publishing.
- Use a unique `patch_id` namespace (`MY_HANDLE_X`) to avoid future
  collisions with core patches.
- Document the hardware you tested on.
- If your patch supersedes a core patch, mention it in
  `community_credit`.

### Path 2 — in-repo SDK (bundled with Genesis releases)

Ship the patch inside the repo at
`plugins/community/<author>/<patch-id>/` with a `manifest.yaml`. The
SDK validates manifests, discovers them via filesystem +
entry-points, and scaffolds new patches from a template. Best when
you want the patch bundled into the Genesis release.

```bash
# Scaffold
sndr community new-patch \
    --id PN999 \
    --author your_handle \
    --family spec_decode \
    --title "PN999 — short description"

# Creates:
#   plugins/community/your_handle/PN999/
#     manifest.yaml
#     patch.py            (apply() stub — replace with real logic)
#     tests/test_pn999.py

# Implement the apply() hook, then:
sndr community list
sndr community validate

# Promote draft → review when ready
$EDITOR plugins/community/your_handle/PN999/manifest.yaml
```

Validator rules:

- **R-1** unique `patch_id` (no collision with core or other
  community).
- **R-2** family must be one of the canonical families.
- **R-3** `applies_to` predicate parses.
- **R-4** lifecycle is one of `draft`, `review`, `experimental`,
  `stable`, `retired`.
- **R-5** `apply()` must exist and be importable torch-less.
- **R-6** unit tests pass (`pytest plugins/community/<author>/<id>/tests/`).
- **R-7** stable lifecycle requires `default_on=True` only after
  cross-rig validation.

Reference code:

- `sndr/model_configs/schema_v2.py` — `PatchManifest` +
  `PatchCompatibility` + `PatchTargetFile` + `PatchAnchor`.
- `sndr/community/manifest.py` — load + path enumeration.
- `sndr/community/discovery.py` — filesystem +
  entry-points.
- `sndr/community/validator.py` — release-tier rules.
- `sndr/community/scaffold.py` — scaffold generator.
- `sndr/cli/legacy/community.py` — CLI surface.
- `plugins/community/_template/` — reference example layout.

## Project map — where things live

Pointers for first-time contributors. The repo is large; this is a
quick "I want to change X, where is X" reference.

### Entry points

| Command | File | Purpose |
| --- | --- | --- |
| `sndr launch <preset>` | `sndr/cli/legacy/launch.py` | Render + exec a preset. |
| `sndr doctor` | `sndr/compat/cli.py` (legacy bridge) + `sndr/cli/legacy/doctor_system.py` (extended) | Health check (basic + UNIFIED_CONFIG C1 extended). |
| `sndr model-config` | `sndr/compat/model_config_cli.py` | Preset CRUD + bench. |
| `sndr patches` | `sndr/cli/legacy/patches.py` | Registry browse / plan / release-check. |
| `sndr service` | `sndr/cli/legacy/service.py` | systemd / compose / quadlet / k8s / proxmox lifecycle. |
| `sndr community` | `sndr/cli/legacy/community.py` | In-repo SDK. |
| `sndr report` | `sndr/cli/legacy/report.py` | Diagnostic tar.gz bundle. |
| `install.sh` | `install.sh` (root) | One-command bootstrap. |
| Bench harness | `sndr/extras/tools/genesis_bench_suite.py` | Canonical bench. Shim under `tools/`. |

### `sndr/` subpackages

| Subpackage | What's inside |
| --- | --- |
| `apply/` | `orchestrator`, `verify`, `shadow`, `_per_patch_dispatch` — the legacy register that drives apply at boot. |
| `audit/` | Audit gates consumed by `make evidence` (release_check, patch_proof, schema validator). |
| `caveats/` | Host-condition caveats registry consumed by `sndr caveats`. |
| `cli/` | One module per `sndr` subcommand. |
| `community/` | In-repo SDK (manifest loader, discovery, validator, scaffold). |
| `compat/` | Operator-facing wrappers (bench, doctor, install, model_config_cli, plugins, ...). |
| `core/` | `TextPatcher`, anchor manifest, drift markers. |
| `dispatcher/` | `registry.py` (PATCH_REGISTRY) + `spec.py` (PatchSpec) + `registry_metadata.py` (derived production_default). |
| `integrations/` | 20 top-level family subdirectories of wiring modules (one per patch); 23 PATCH_REGISTRY family-name values total when sub-families under `attention/` (flash, gdn, turboquant) are counted separately. |
| `kernels/` | Triton / CUDA kernel source (`p67_multi_query_kernel.py`, …). |
| `locations/` | `project_paths.py` + `vllm_targets.py` — portable filesystem layout. |
| `middleware/` | Per-request hooks (lazy reasoner, observability). |
| `model_configs/` | V1 schema + V2 layered registry + composer + audit_rules + preflight + verify + patch_plan resolver. |
| `tools/` | Bench harness (`genesis_bench_suite.py`) — the canonical bench. |

### `scripts/` (release-tier audit gates)

| Script | Purpose |
| --- | --- |
| `check_doc_sync.py` | Patch-count consistency across docs. |
| `check_md_links.py` | Active-doc broken-link finder. |
| `audit_public_docs.py` | D-1..D-6 boundary rules. |
| `audit_no_stub.py` | No-stub gate. |
| `check_dirty_state.py` | dev / audit / release tier dirty-state. |
| `audit_artifacts.py` | Release-artifact policy. |
| `security_scan.py` | Public-release security scan. |
| `make_evidence.py` | Aggregate runner of every gate above. |
| `generate_patches_md.py` | Auto-regen `docs/PATCHES_AUTO.md`. |
| `generate_configs_md.py` | Auto-regen `docs/CONFIGS_AUTO.md`. |
| `sync_readme_counters.py` | Sync patch counts in README sections. |

### `tests/`

| Directory | What's inside |
| --- | --- |
| `tests/unit/` | Per-module unit tests, hermetic (no GPU, no network). |
| `tests/unit/integrations/<family>/` | Per-family contract tests (factory pattern). |
| `tests/integration/` | Tests that exercise a running container; gated on GPU availability. |
| `tests/integration/baselines/` | Per-preset bench baselines (regression bounds). |
| `tests/legacy/` | Pre-v11 tests kept for back-compat; do not add new tests here. |
| `tests/conftest.py` | Shared fixtures and the skip-list for legacy naming drift. |

### `Makefile` (operator-facing)

`make help` enumerates ~21 targets. Most common:

- `make evidence` — full release-tier audit aggregate.
- `make gates` — fast-fail subset (~2 s).
- `make audit-release-check` — current public release gate.
- `make audit-yaml` — drift between local YAML and live container.
- `make precommit-install` — wire pre-commit hooks.
- `make clean` — pycache / pytest-cache cleanup.

For the full audit list see [`RELEASE_POLICY.md`](RELEASE_POLICY.md).

## Cross-references

- [`PATCHES.md`](PATCHES.md) — full patch catalog with metadata
  (incl. compatibility matrix + patch plan resolver).
- [`MODELS.md`](MODELS.md) — model selection, V2 layered configs,
  community config submission pipeline.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — cliffs, OOM recipes,
  cookbook, rollback playbook.
- [`BENCHMARKS.md`](BENCHMARKS.md) — canonical numbers + reproduction.
- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) — every `sndr` subcommand.
- [`PATCH_DESIGNS.md`](PATCH_DESIGNS.md) — PN95 / GDN fusion /
  reasoning contract / v11 rename appendices.
- [`CREDITS.md`](CREDITS.md) — attributions, including upstream PRs
  we ported.

Thanks for contributing.
