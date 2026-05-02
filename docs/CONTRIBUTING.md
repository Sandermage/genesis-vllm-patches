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

Step-by-step. Read [../docs/PATCHES.md](../docs/PATCHES.md) and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) first to understand the conventions.

### 1. Pick the right directory

`vllm/_genesis/wiring/` is split by concern:

| Directory | What goes here |
|---|---|
| `spec_decode/` | Anything touching MTP, ngram, DFlash, rejection sampling, draft acceptance |
| `structured_output/` | Reasoning parsers, tool-call extraction, grammar masks |
| `kv_cache/` | KV cache dtype, TurboQuant, prefix caching, hash backends |
| `kernels/` | Triton/CUDA kernel hooks (the kernels themselves live in `vllm/_genesis/kernels/`) |
| `compile_safety/` | torch.compile guards, cudagraph capture safety, custom_op registration |
| `perf_hotfix/` | Pure perf wins not tied to a specific subsystem |
| `hybrid/` | GDN / Mamba / hybrid-attention specific (Qwen3.5, Qwen3-Next) |
| `middleware/` | Logging, metrics, telemetry, instrumentation |
| `legacy/` | Patches superseded but kept for backporting to older pins |

Pick the closest match. If genuinely unclear, default to `perf_hotfix/` and the reviewer will move it.

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
from vllm._genesis.wiring.text_patch import TextPatcher, TextPatch

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

### 3. Register in `dispatcher.py`

Add an entry to `PATCH_REGISTRY` with full metadata:

```python
"PNN": {
    "title": "TurboQuant multi-query Triton kernel",
    "env_flag": "GENESIS_ENABLE_PNN",
    "default_on": False,                        # False unless we are sure
    "category": "kernels",
    "credit": "Genesis-original (Sander)",      # or "port of vllm#NNNNN"
    "upstream_pr": None,                         # or 41268
    "applies_to": {
        "is_turboquant": [True],
    },
    "conflicts_with": ["P65"],                   # patches that mutate the same code path
    "requires_patches": ["P4"],                  # patches that must be applied first
},
```

`default_on=True` is reserved for patches that fix a bug and have been validated on at least two distinct workloads. New patches start `default_on=False` and get promoted in a later PR.

### 4. Add a unit test

`vllm/_genesis/tests/test_pNN_<name>.py`. Minimum coverage:

- Anchor exists in current vLLM pin (read the file, assert substring present).
- Replacement is well-formed (parseable Python if it's a Python text-patch).
- Marker is in the replacement.
- After-apply state is idempotent (running `apply()` twice is a no-op on the second call).

For kernel patches, add a CPU-only smoke test with tiny shapes if at all possible. If the patch *cannot* be tested without a GPU, mark the test with `@pytest.mark.gpu` and document that in the PR.

### 5. Run the test suite

```bash
pytest vllm/_genesis/tests/ -v
```

Must pass. CI will gate the PR on this.

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

## Code style

### Text patches

- **Anchors must be VERBATIM upstream.** Copy-paste from the live source file. Don't reformat, don't normalize whitespace, don't refactor while patching. If upstream uses tab indents, your anchor uses tab indents.
- **Markers must include version.** Format: `Genesis PNN v7.NN_descriptive_name`. The version is the Genesis release where this patch shipped or was last revised.
- **`required=True` for critical sub-patches.** If the patch makes no sense without this sub-patch landing, mark it `required=True` so a missed anchor surfaces as `failed` instead of silent skip.
- **`required=False` only for truly optional sub-patches.** E.g., adding a debug log alongside the real fix.
- **Defensive imports inside functions.** `apply()` should import from `vllm.*` lazily. Module-level imports break boot if the user is on a vLLM pin that renamed the module.

### Triton kernels

- Power-of-2 dims wherever possible. If you must support non-power-of-2 (e.g., GQA=24/4=6 heads-per-KV), use `next_power_of_2` + a `lane_valid` mask. Document the cliff in [docs/CLIFFS.md](docs/CLIFFS.md).
- Sanitize Inf/NaN at dequant boundaries. We've been bitten by silent NaN propagation through softmax — see the v7.22 P67 sanitized variant in [../docs/PATCHES.md](../docs/PATCHES.md).
- BLOCK_SIZE / num_warps / num_stages should be configurable via env override for sweep tuning.

### General Python

- We don't enforce a formatter on contributors, but we do run `ruff` on the maintainer side. PRs may be reformatted before merge.
- Type hints encouraged on public surfaces (anything in `dispatcher.py`, `text_patch.py`, `apply_all.py`).
- Logging via `logger = logging.getLogger("vllm._genesis")`. Print only in the boot-summary path.

---

## Testing requirements

### Per-PR minimum

- **Unit test for every wiring patch.** `test_pNN_*.py` validates anchor exists, replacement is sane, marker present, idempotent.
- **Boot smoke test.** Add your patch to a launch script, run it, paste the boot log section showing `APPLY` in the PR.
- **Empirical bench.** `n >= 3` runs (5 preferred) with `tools/genesis_bench_suite.py`. Report mean, std, CV.

### CI

GitHub Actions runs `pytest vllm/_genesis/tests/` on every PR. CPU-only — no GPU CI yet. GPU validation is the maintainer's responsibility on the staging rig.

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
chore: bump pin reference in COMPATIBILITY.md
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
- Anything from `~/.claude/`, `docs/_internal/`, `snapshots/`, or any path that's `.gitignore`d.
- Hugging Face tokens, OpenAI keys, GitHub PATs, AWS credentials, anything in a `.env`.
- Personal data — names, emails, IPs of internal infrastructure.
- Internal sprint plans, roadmap drafts, third-party correspondence.

If you discover a security issue (e.g., a patch that allows code injection through model config), **do not open a public issue.** Use the maintainer contact in `SPONSORS.md` with details. We'll acknowledge within 72 hours and coordinate disclosure.

---

## Translation

All public docs are in **English**. This includes README, PATCHES, MODELS, CHANGELOG, CONFIGURATION, and the `docs/` tree.

**Russian translations are welcome** but live as separate files: `docs/<file>.ru.md`. Don't replace the English version. If you submit a Russian translation, the English version is the source of truth — translations track it.

The maintainer writes natively in Russian. AI translation help is fine for PR comments and discussions; please flag it briefly (`(translated with AI assistance)`) so reviewers can adjust expectations on phrasing.

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

## Cross-references

- [../docs/PATCHES.md](../docs/PATCHES.md) — full patch catalog with metadata
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) — supported vLLM pins, models, GPUs
- [docs/CONFIGS.md](docs/CONFIGS.md) — adding your own model recipe
- [docs/CLIFFS.md](docs/CLIFFS.md) — known performance and correctness cliffs
- [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) — how to bench reproducibly
- [docs/SELF_TEST.md](docs/SELF_TEST.md) — running the validation suite
- [../docs/CREDITS.md](../docs/CREDITS.md) — attributions, including upstream PRs we ported

Thanks for contributing.
