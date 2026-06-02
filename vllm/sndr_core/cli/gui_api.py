# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for the read-only SNDR GUI Product API daemon."""
from __future__ import annotations

from typing import Any


__all__ = ["add_argparser", "run_gui_api"]


def _open_browser_soon(url: str, delay: float = 1.5) -> None:
    """Open the UI in a browser after a short delay (best-effort, non-blocking)."""
    import threading
    import webbrowser

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Timer(delay, _open).start()


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "gui-api",
        help="Run the read-only SNDR Product API for GUI/web clients.",
        description=(
            "Starts a local read-only FastAPI daemon exposing typed SNDR "
            "Product API snapshots for GUI, web, and remote desktop clients. "
            "This API does not write V2 YAML, patch registries, or runtime "
            "artifacts."
        ),
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Default: 127.0.0.1.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port. Default: 8765.",
    )
    p.add_argument(
        "--log-level",
        default="info",
        help="uvicorn log level. Default: info.",
    )
    p.add_argument(
        "--enable-apply",
        action="store_true",
        help=(
            "Enable real service-action execution (status/logs always; "
            "start/stop/restart require an explicit confirm). OFF by default — "
            "without this the daemon stays read-only/dry-run."
        ),
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the served UI in a browser shortly after the daemon starts.",
    )
    p.set_defaults(func=run_gui_api)


def run_gui_api(args) -> int:
    try:
        from vllm.sndr_core.product_api.http_app import run_server
    except RuntimeError as exc:
        print(f"sndr gui-api: {exc}")
        return 3
    url = f"http://{args.host}:{args.port}"
    print(f"sndr gui-api: serving UI + API on {url}")
    if getattr(args, "open", False):
        _open_browser_soon(url)
    try:
        run_server(
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            enable_apply=bool(getattr(args, "enable_apply", False)),
        )
    except RuntimeError as exc:
        print(f"sndr gui-api: {exc}")
        return 3
    except KeyboardInterrupt:
        return 130
    return 0
