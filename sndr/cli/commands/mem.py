# SPDX-License-Identifier: Apache-2.0
"""CLI commands: ``sndr mem remember / recall / search / stats``.

Headless access to the persistent neural-graph memory — the SAME
``/api/v1/memory/*`` routes the GUI drives, over HTTP against the running
product-API daemon (``sndr up`` / the genesis-memory container). This is the
scriptable surface for cron jobs and automation; operators mostly use the GUI.

Distinct from ``sndr memory`` (the legacy VRAM/KV fit estimator) — that is a
static GPU calculator, this talks to the live memory store. Connection:

  --url    daemon base URL   (else $SNDR_MEMORY_URL / $SNDR_GUI_URL, default
           http://127.0.0.1:8765)
  --owner  owner id          (else $SNDR_MEMORY_OWNER, default 1)
  --token  API key           (else $GENESIS_MEMORY_API_KEY; sent as Bearer)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

_DEFAULT_URL = "http://127.0.0.1:8765"


def _make_client(args: argparse.Namespace):
    """Build a MemoryHTTPClient from args + env (module-level so tests can fake it)."""
    from sndr.memory.client import MemoryHTTPClient

    url = getattr(args, "url", None) or os.environ.get("SNDR_MEMORY_URL") \
        or os.environ.get("SNDR_GUI_URL") or _DEFAULT_URL
    token = getattr(args, "token", None) or os.environ.get("GENESIS_MEMORY_API_KEY") or None
    owner = int(getattr(args, "owner", None) or os.environ.get("SNDR_MEMORY_OWNER") or 1)
    return MemoryHTTPClient(url, owner_id=owner, token=token)


def _owner(args: argparse.Namespace) -> int:
    return int(getattr(args, "owner", None) or os.environ.get("SNDR_MEMORY_OWNER") or 1)


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=None,
                        help="daemon base URL (else $SNDR_MEMORY_URL/$SNDR_GUI_URL, "
                             f"default {_DEFAULT_URL})")
    parser.add_argument("--owner", type=int, default=None,
                        help="owner id (else $SNDR_MEMORY_OWNER, default 1)")
    parser.add_argument("--token", default=None,
                        help="API key (else $GENESIS_MEMORY_API_KEY; sent as Bearer)")


def _run(args: argparse.Namespace, fn) -> int:
    """Call ``fn(client)`` with friendly, actionable errors instead of a traceback."""
    try:
        return fn(_make_client(args))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print("memory: unauthorized — set --token or $GENESIS_MEMORY_API_KEY to the "
                  "daemon's GENESIS_MEMORY_API_KEY.", file=sys.stderr)
        else:
            print(f"memory: daemon returned HTTP {exc.code} ({exc.reason}).", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"memory: daemon not reachable ({exc}). Is it up? Try `sndr up` or set "
              "--url to the daemon.", file=sys.stderr)
        return 1


def _emit_hits(hits, output: str) -> None:
    rows = [{"id": h.node.id, "score": round(h.score, 4), "content": h.node.content}
            for h in hits]
    if output == "json":
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("(no matches)")
        return
    for r in rows:
        print(f"  [{r['id']:>5}] {r['score']:.3f}  {r['content']}")


class MemRememberCommand:
    name = "mem.remember"
    help = "Store a memory (persistent graph) via the running daemon."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("text", help="the text to remember")
        parser.add_argument("--kind", default="note", help="node kind (default: note)")
        parser.add_argument("--importance", type=float, default=0.0,
                            help="seed importance in [0,1] (default: 0.0)")
        _add_connection_args(parser)

    def execute(self, args: argparse.Namespace) -> int:
        def _do(client) -> int:
            node_id = client.remember(
                owner_id=_owner(args), text=args.text,
                kind=args.kind, importance=args.importance,
            )
            if getattr(args, "output", "text") == "json":
                print(json.dumps({"id": node_id}))
            else:
                print(f"remembered #{node_id}")
            return 0
        return _run(args, _do)


class MemRecallCommand:
    name = "mem.recall"
    help = "Brain recall (graph expand + reinforce) via the running daemon."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query", help="the recall query")
        parser.add_argument("--limit", type=int, default=10, help="max hits (default: 10)")
        parser.add_argument("--depth", type=int, default=2,
                            help="graph expansion depth (default: 2)")
        parser.add_argument("--no-reinforce", action="store_true", dest="no_reinforce",
                            help="do not strengthen recalled nodes (read-only recall)")
        _add_connection_args(parser)

    def execute(self, args: argparse.Namespace) -> int:
        def _do(client) -> int:
            hits = client.recall(
                owner_id=_owner(args), query=args.query, limit=args.limit,
                expand_depth=args.depth, reinforce=not args.no_reinforce,
            )
            _emit_hits(hits, getattr(args, "output", "text"))
            return 0
        return _run(args, _do)


class MemSearchCommand:
    name = "mem.search"
    help = "Search memory (vector/hybrid, no side effects) via the running daemon."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("query", help="the search query")
        parser.add_argument("--limit", type=int, default=10, help="max hits (default: 10)")
        _add_connection_args(parser)

    def execute(self, args: argparse.Namespace) -> int:
        def _do(client) -> int:
            hits = client.search(owner_id=_owner(args), query=args.query, limit=args.limit)
            _emit_hits(hits, getattr(args, "output", "text"))
            return 0
        return _run(args, _do)


class MemStatsCommand:
    name = "mem.stats"
    help = "Show this owner's memory counts (nodes/edges) via the running daemon."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        _add_connection_args(parser)

    def execute(self, args: argparse.Namespace) -> int:
        def _do(client) -> int:
            stats = client.stats(owner_id=_owner(args))
            if getattr(args, "output", "text") == "json":
                print(json.dumps(stats))
            else:
                print(f"memory: {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges "
                      f"(owner {_owner(args)})")
            return 0
        return _run(args, _do)


__all__ = [
    "MemRecallCommand",
    "MemRememberCommand",
    "MemSearchCommand",
    "MemStatsCommand",
]
