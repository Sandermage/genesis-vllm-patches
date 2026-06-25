# Pin Upgrade — operator policy + launcher template

**Audience**: operators upgrading the vLLM (or other engine) pin.

> **This is the short policy summary + launcher reference.** The canonical,
> end-to-end pin-bump procedure is [`../PIN_BUMP_PLAYBOOK.md`](../PIN_BUMP_PLAYBOOK.md);
> the anchor source-of-truth + bump-gate tooling is documented in
> [`../ANCHOR_SOT.md`](../ANCHOR_SOT.md). Read those for the actual bump.
> This page exists so an operator can recall the **policy** and the
> **launcher signature** without paging through the full playbook.
>
> Current pin: `0.23.1rc1.dev424+g3f5a1e173` · rollback:
> `0.23.1rc1.dev301+g04c2a8dea` (`dev301`). `dev148` is dropped.

## What is a pin?

A **pin** is a specific upstream commit + version of the engine package
(e.g. `0.23.1rc1.dev424+g3f5a1e173` = vLLM commit `3f5a1e173`). Each
supported pin is committed under `sndr/engines/<engine>/pins/<pin>/` as a
per-pin **anchor source-of-truth** manifest (`anchors.json`) plus a triage
report (`drift.rej.json`). See [`../ANCHOR_SOT.md`](../ANCHOR_SOT.md).

When upstream changes a file an anchor can move or vanish; the patch then
stops applying. We call this **drift**. Drift is surfaced by
`make audit-pin` (live engine vs committed manifest) and `make summarize-rej`
(per-pin reject report), and shown in the GUI.

## Pin policy (read before bumping)

This project holds **at most two pins** — the current canonical pin plus an
optional **previous** pin retained for rollback during validation. The
rules:

1. **No proactive pulls.** Never `docker pull` a newer nightly just because
   one exists upstream. A bump happens **only** on an explicit operator
   instruction that names the target pin.
2. **The candidate image must already be on the rig** before evaluation. If
   it is absent, the extractor prints the exact `docker pull` command for
   the operator to run deliberately — it does not pull for you.
3. **Validate before promote.** A new pin is promoted to canonical only
   after anchor-drift is resolved, the bump-gate (`make bump-preflight`)
   is clean, and boot-smoke + tokenizer-fingerprint + canonical bench pass
   on a throwaway container (never on PROD).
4. **Keep the previous pin** as rollback during the validation window.
5. **Drop the oldest.** Once the new pin is fully validated, delete the
   2-back pin so the server again holds at most current + previous.

## Universal launcher template

Every model deployment uses the same launcher signature:

    SNDR_ROOT=/opt/sndr-platform
    docker run -d \
      --name sndr-<engine>-<config> \
      --gpus all --shm-size=8g --memory=64g \
      -p $PORT:$PORT \
      -v $SNDR_ROOT/sndr:/usr/local/lib/python3.12/dist-packages/sndr:ro \
      -v $SNDR_ROOT/configs/<config>.yaml:/sndr_config.yaml:ro \
      -v /nfs/models:/models:ro \
      --security-opt label=disable \
      -e SNDR_ENGINE=<engine> \
      -e SNDR_ENGINE_PIN=<pin> \
      -e SNDR_CONFIG=/sndr_config.yaml \
      vllm/vllm-openai:nightly-<pin> \
      <model_path> \
      --tensor-parallel-size=2 \
      --port=$PORT

The mount target is top-level `sndr/` (the v12 package layout). The legacy
`BATCH9`-style launchers continue to work in v12.x via shims. Prefer
booting through a preset — `sndr launch <preset>` — which renders this
command for you with the correct pin, mounts, and env.

## Upgrade procedure (summary)

The full procedure is [`../PIN_BUMP_PLAYBOOK.md`](../PIN_BUMP_PLAYBOOK.md).
In brief, on an authorized bump whose candidate image is already on the rig:

1. **Extract the candidate tree** (read-only, server-side):
   `tools/extract_candidate_tree.sh --image vllm/vllm-openai:nightly-<sha> …`
   (no `docker pull`; the container is never started).
2. **Regenerate the per-pin SOT**: `make rebuild-pin SSH_HOST=<user@host>`
   → writes `sndr/engines/vllm/pins/<new-pin>/`.
3. **Read the triage** + resolve drift: `make summarize-rej PIN=<new-pin>`;
   re-derive every `genuine_anchor_drift` anchor; iron-rule-#11 deep-diff
   every `upstream_merged` row (never retire on a PR title).
4. **Run the bump gate**: `make bump-preflight OLD=<old-pin> NEW=<new-pin>`
   — must exit 0, or clear every flagged perf dependent with a canonical
   A/B (`tools/genesis_bench_suite.py`).
5. **Inspect engine state**: `sndr engines.info vllm` (active version,
   install root, patch counts, supported pins), `sndr pins.list`.
6. **Boot smoke** on a throwaway container + **tokenizer-fingerprint gate**
   (`make tokenizer-fingerprint`) + **canonical bench** vs `reference_metrics`.
7. **Promote**: add the pin to `KNOWN_GOOD_VLLM_PINS`
   (`sndr/engines/vllm/detection/guards.py`), pair-update `EXPECTED_PINS`
   (`tests/unit/dispatcher/test_pin_gate.py`; `make test-pin-gate`), bump
   `vllm_pin_required` in the model YAMLs + README badge + CHANGELOG, and
   bump validated patches' `applies_to` upper bounds (boot-log proof only).
8. **Rotate tags** per the pin policy (re-tag `:nightly`, keep previous,
   drop oldest).

## Rollback

If the new pin fails any check, re-point the running tag to the retained
previous pin and restart the container — no code changes required:

    docker tag vllm/vllm-openai:nightly-<previous-sha> vllm/vllm-openai:nightly

The previous pin is now active again.

## Pin compatibility

| sndr version | current canonical pin | rollback |
| --- | --- | --- |
| 12.0.x | `0.23.1rc1.dev424+g3f5a1e173` | `0.23.1rc1.dev301+g04c2a8dea` |

The authoritative allowlist is `KNOWN_GOOD_VLLM_PINS` in
`sndr/engines/vllm/detection/guards.py`; the per-pin manifests under
`sndr/engines/vllm/pins/` are the source of truth for what each pin's
patch matrix looks like.

## See also

- [`../PIN_BUMP_PLAYBOOK.md`](../PIN_BUMP_PLAYBOOK.md) — **canonical**
  end-to-end pin-bump procedure.
- [`../ANCHOR_SOT.md`](../ANCHOR_SOT.md) — anchor source-of-truth +
  `rebuild-pin` / `audit-pin` / `bump-preflight` / `summarize-rej` manual.
- [`COMMERCIAL_TIER.md`](COMMERCIAL_TIER.md) — how engine-tier patches
  integrate.
