#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the IN-CONTAINER command from a `sndr launch --dry-run` render.

The dry-run render is a HOST `docker run` launcher: it boots the container
itself and passes the real work as the `-c 'payload'` argument to an in-image
`/bin/bash` entrypoint. Tooling that wants to run that payload under a DIFFERENT
image/mounts (the fleet boot-smoke gate does) must extract the payload, not run
the host script — running the whole render inside a container tries
docker-in-docker and dies with `docker: command not found`.

Reads the render on stdin, prints one TAB-separated line: `<inner_port>\t<payload>`.
Exits non-zero if no `docker run ... -c <payload>` is found.
"""
from __future__ import annotations

import shlex
import sys


def main() -> int:
    text = sys.stdin.read()
    lines = text.splitlines()
    try:
        start = next(
            i for i, ln in enumerate(lines) if ln.strip().startswith("docker run")
        )
    except StopIteration:
        print("no 'docker run' in render", file=sys.stderr)
        return 2
    # Join the backslash-continued docker-run invocation into one command line.
    joined = " ".join(ln.rstrip().rstrip("\\") for ln in lines[start:])
    try:
        toks = shlex.split(joined)
    except ValueError as e:
        print(f"shlex parse failed: {e}", file=sys.stderr)
        return 2
    payload = None
    inner_port = "8000"
    for i, t in enumerate(toks):
        if t == "-c" and i + 1 < len(toks):
            payload = toks[i + 1]
        elif t in ("-p", "--publish") and i + 1 < len(toks):
            inner_port = toks[i + 1].split(":")[-1]
    if not payload or "vllm serve" not in payload:
        print("no in-container '-c' payload with a vllm serve found", file=sys.stderr)
        return 2
    # Single logical line already (shlex collapsed it); guard against stray NLs.
    payload = payload.replace("\n", " ")
    sys.stdout.write(f"{inner_port}\t{payload}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
