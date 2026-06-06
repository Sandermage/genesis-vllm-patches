# Pin Upgrade Playbook

**Audience**: Operators upgrading the vLLM (or other engine) pin.

## What is a pin?

A **pin** is a specific upstream commit + version of the engine package. We
track each supported pin under `sndr/engines/<engine>/pins/<pin>/` with a
YAML manifest that records the md5 of every file our patches touch.

When upstream changes a file, the live md5 differs from the manifest. We
call this **drift**. Drift detection runs daily via CI and surfaces
warnings in the GUI.

## Universal launcher template

Every model deployment uses the same launcher signature::

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

The legacy `BATCH9`-style launchers continue to work in v12.x via shims.

## Upgrade procedure (manual, v12)

1. **Pull new image**

       docker pull vllm/vllm-openai:nightly

2. **Tag previous current as `nightly-previous`**

       docker tag vllm/vllm-openai:nightly vllm/vllm-openai:nightly-previous

3. **Re-tag new image with explicit SHA**

       SHA=$(docker run --rm --gpus all --entrypoint /usr/bin/python3 \
         vllm/vllm-openai:nightly -c "import vllm; print(vllm.__version__)")
       docker tag vllm/vllm-openai:nightly vllm/vllm-openai:nightly-<short-sha>

4. **Generate manifest for the new pin**

       python3 tools/manifest_gen.py --engine vllm --pin <pin-id>

5. **Run drift check**

       python3 tools/drift_check.py --engine vllm --pin <pin-id>

   Output classifies each file as `ok`, `benign`, `drift`, or `blocked`.

6. **Inspect engine state**

       sndr engines.list
       sndr engines.info vllm
       sndr pins.list

   `engines.info vllm` reports active version, install root, patch
   counts, and supported-pin list. `pins.list` enumerates known pins.

7. **Boot smoke**

       bash /tmp/start_27b_universal.sh

   Verify boot completes and apply matrix is unchanged.

8. **Bench validation**

       python3 tools/genesis_bench_suite.py --quick --model qwen3.6-35b-fp8

   Compare TPS against baseline. Acceptable: ±2%.

9. **Promote pin**

   Add the pin to `KNOWN_GOOD_VLLM_PINS` in
   `sndr/engines/vllm/detection/guards.py` and update
   `docs/concepts/PINS.md`. Commit the change.

## Rollback

If the new pin fails any check:

    docker tag vllm/vllm-openai:nightly-previous vllm/vllm-openai:nightly

Restart the container. The previous pin is now active again. No code
changes required.

## Pin compatibility matrix

| sndr version | vllm pins supported |
|---|---|
| 12.0.x | 0.21.1, 0.22.0, 0.22.1 |
| 12.1.x | TBD (next release) |
| 13.0.x | TBD (legacy pins dropped) |

## See also

- `docs/concepts/PINS.md` — what a pin is and why we track them
- `docs/concepts/DRIFT.md` — how drift detection works
- `docs/guides/COMMERCIAL_TIER.md` — how engine-tier patches integrate
- `tools/manifest_gen.py` — manifest generator source
- `tools/drift_check.py` — drift checker source
