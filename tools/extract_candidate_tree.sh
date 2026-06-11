#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# extract_candidate_tree.sh — candidate pin acquisition for pin_preflight.
#
# Extracts the PRISTINE vllm package tree from a candidate vllm image
# already present on the server, plus a PROVENANCE.json block (image
# ref, content digest, internal vllm version, extraction timestamp) so
# the preflight report can never mislabel which pin it judged (the
# pin-provenance-mislabeling failure class).
#
# Server-side mechanics: docker create → docker cp → docker rm.
#   - NO `docker pull`: per pin policy the operator must explicitly
#     request any image download. If the image is absent we exit 2
#     with a clear message instead of pulling.
#   - NO GPU use: `docker create` never starts the container; the
#     version probe runs `--entrypoint python3` without `--gpus`.
#   - NO PROD interaction: a throwaway container id, removed at the end.
#
# Usage:
#   tools/extract_candidate_tree.sh --image <ref> \
#       [--ssh-host <user@rig-host>] \
#       [--staging /tmp/candidate_pin] \
#       [--timestamp <ISO-8601>] \
#       [--rsync-to <local-dir>] [--py-only]
#
#   --image      Candidate image ref (e.g. vllm/vllm-openai:nightly-<sha>).
#   --ssh-host   SSH target holding the image (default: $GENESIS_RIG_SSH_HOST;
#                required via flag or env — no baked-in host).
#   --staging    Server-side staging dir (default /tmp/candidate_pin).
#                Must live under /tmp/ — it is wiped before extraction.
#   --timestamp  extracted_at value for PROVENANCE.json (default: now UTC).
#   --rsync-to   Optionally rsync the staged tree to this LOCAL dir for
#                Mac-side preflight (tree + PROVENANCE.json).
#   --py-only    Restrict the rsync to *.py files (preflight only reads
#                Python sources; skips ~GBs of shared objects).
#
# Exit codes: 0 ok | 1 extraction failure | 2 invocation error / image absent.
#
# Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

set -euo pipefail

SSH_HOST="${GENESIS_RIG_SSH_HOST:-}"
STAGING="/tmp/candidate_pin"
IMAGE=""
TIMESTAMP=""
RSYNC_TO=""
PY_ONLY=0
VLLM_PKG_PATH="/usr/local/lib/python3.12/dist-packages/vllm"

usage() {
    sed -n '5,40p' "$0" | sed 's/^# \{0,1\}//'
    echo "usage: $0 --image <ref> [--ssh-host <host>] [--staging <dir>]"
    echo "          [--timestamp <iso>] [--rsync-to <local-dir>] [--py-only]"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --image)     IMAGE="${2:?--image needs a value}"; shift 2 ;;
        --ssh-host)  SSH_HOST="${2:?--ssh-host needs a value}"; shift 2 ;;
        --staging)   STAGING="${2:?--staging needs a value}"; shift 2 ;;
        --timestamp) TIMESTAMP="${2:?--timestamp needs a value}"; shift 2 ;;
        --rsync-to)  RSYNC_TO="${2:?--rsync-to needs a value}"; shift 2 ;;
        --py-only)   PY_ONLY=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "extract_candidate_tree: unknown argument: $1" >&2
           usage >&2; exit 2 ;;
    esac
done

if [ -z "$SSH_HOST" ]; then
    echo "extract_candidate_tree: --ssh-host (or GENESIS_RIG_SSH_HOST env) is required" >&2
    usage >&2
    exit 2
fi

if [ -z "$IMAGE" ]; then
    echo "extract_candidate_tree: --image is required" >&2
    usage >&2
    exit 2
fi

case "$STAGING" in
    /tmp/*) : ;;
    *) echo "extract_candidate_tree: --staging must live under /tmp/ " \
            "(it is wiped before extraction); got: $STAGING" >&2
       exit 2 ;;
esac

if [ -z "$TIMESTAMP" ]; then
    TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

echo "[extract] candidate image: $IMAGE on $SSH_HOST" >&2

# ── 1. Image MUST already be on the server (pin policy: no pull) ─────────
if ! ssh "$SSH_HOST" "docker image inspect --format ok $IMAGE" >/dev/null 2>&1; then
    echo "extract_candidate_tree: image $IMAGE is NOT present on $SSH_HOST." >&2
    echo "Pin policy forbids automatic 'docker pull' — if this candidate is" >&2
    echo "wanted, the operator must pull it explicitly first, e.g.:" >&2
    echo "    ssh $SSH_HOST \"docker pull $IMAGE\"" >&2
    exit 2
fi

# ── 2. Provenance: digest + internal version ─────────────────────────────
DIGEST="$(ssh "$SSH_HOST" "docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' $IMAGE")"
echo "[extract] digest: $DIGEST" >&2

echo "[extract] probing internal vllm version (CPU-only, no --gpus)..." >&2
INTERNAL_VERSION="$(ssh "$SSH_HOST" "docker run --rm --entrypoint python3 $IMAGE -c 'import vllm; print(vllm.__version__)'" | tr -d '[:space:]')"
if [ -z "$INTERNAL_VERSION" ]; then
    echo "extract_candidate_tree: could not read vllm.__version__ from $IMAGE" >&2
    exit 1
fi
echo "[extract] internal version: $INTERNAL_VERSION" >&2

# ── 3. Stage the pristine tree: create → cp → rm ─────────────────────────
ssh "$SSH_HOST" "rm -rf $STAGING && mkdir -p $STAGING"

CID="$(ssh "$SSH_HOST" "docker create $IMAGE")"
echo "[extract] throwaway container: $CID" >&2
cleanup() { ssh "$SSH_HOST" "docker rm -f $CID >/dev/null 2>&1 || true"; }
trap cleanup EXIT

ssh "$SSH_HOST" "docker cp $CID:$VLLM_PKG_PATH $STAGING/vllm"
ssh "$SSH_HOST" "docker rm $CID >/dev/null"
trap - EXIT

if ! ssh "$SSH_HOST" "test -d $STAGING/vllm/v1"; then
    echo "extract_candidate_tree: $STAGING/vllm/v1 missing after docker cp —" >&2
    echo "wrong package path inside image? expected $VLLM_PKG_PATH" >&2
    exit 1
fi

# ── 4. PROVENANCE.json ───────────────────────────────────────────────────
ssh "$SSH_HOST" "cat > $STAGING/PROVENANCE.json" <<EOF
{
  "image_ref": "$IMAGE",
  "digest": "$DIGEST",
  "internal_version": "$INTERNAL_VERSION",
  "extracted_at": "$TIMESTAMP",
  "extracted_by": "tools/extract_candidate_tree.sh",
  "package_path": "$VLLM_PKG_PATH",
  "host": "$SSH_HOST"
}
EOF
echo "[extract] PROVENANCE.json written to $STAGING/PROVENANCE.json" >&2

# ── 5. Optional rsync to the local machine ───────────────────────────────
if [ -n "$RSYNC_TO" ]; then
    mkdir -p "$RSYNC_TO"
    if [ "$PY_ONLY" = "1" ]; then
        echo "[extract] rsync (*.py + PROVENANCE.json only) -> $RSYNC_TO" >&2
        rsync -a --prune-empty-dirs \
            --include='*/' --include='*.py' --include='PROVENANCE.json' \
            --exclude='*' \
            "$SSH_HOST:$STAGING/" "$RSYNC_TO/"
    else
        echo "[extract] rsync (full tree) -> $RSYNC_TO" >&2
        rsync -a "$SSH_HOST:$STAGING/" "$RSYNC_TO/"
    fi
    echo "[extract] local candidate root: $RSYNC_TO/vllm" >&2
fi

echo "[extract] DONE — candidate root on server: $STAGING/vllm" >&2
echo "$STAGING/vllm"
