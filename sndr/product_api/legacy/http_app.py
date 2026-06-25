# SPDX-License-Identifier: Apache-2.0
"""Read-only FastAPI app for SNDR GUI and remote desktop clients.

This module is safe to import without FastAPI installed. Heavy web runtime
dependencies are imported inside ``create_app()`` / ``run_server()`` only,
so the core CLI remains lightweight unless an operator starts the GUI API.
"""
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from sndr.version import SNDR_CORE_VERSION

# Module-level so the websocket route's ``websocket: WebSocket`` annotation
# resolves via get_type_hints (which only sees module globals, not the local
# import inside create_app — ``from __future__ import annotations`` stringifies
# the hint). Optional dep: None when fastapi is absent (create_app raises first).
try:  # pragma: no cover - import shape depends on the environment
    from fastapi import WebSocket
except Exception:  # pragma: no cover
    WebSocket = None  # type: ignore[assignment,misc]

from .capabilities import collect_capabilities
from .config_editor import (
    apply_v2_config_plan,
    apply_v2_layer,
    collect_v2_config_catalog,
    get_v2_layer,
    list_user_presets,
    plan_v2_config_edit,
    preview_v2_config,
)
from .doctor import collect_doctor_report
from .environment import collect_environment_report
from .host_profiles import (
    delete_host_profile,
    host_profile_payload,
    list_host_profiles,
    upsert_host_profile,
)
from .jobs import (
    apply_service_action,
    create_dry_run_job,
    get_job,
    list_events,
    list_jobs,
    record_event,
)
from .launch_plan import build_launch_plan
from .memory import estimate_fit
from .model_cache import collect_model_cache_report
from .reports import REPORT_TYPES, generate_report_bundle
from .service_plan import build_service_plan
from .patches.bundles import explain_bundle, list_bundles
from .patches.diff_upstream import diff_upstream
from .overview import collect_catalog_summary, collect_product_overview
from .presets import (
    PresetComposeError,
    PresetNotFoundError,
    UnknownWorkloadError,
    explain_preset,
    get_preset,
    list_presets,
    recommend_presets,
)
from .patches.doctor import run_doctor as run_patch_doctor
from .patches.explain import explain_patch, suggest_candidates
from .patches.listing import list_patches


DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


# Light per-container version probe (run inside a container via `docker exec`).
# Reads versions through importlib.metadata so it does NOT `import vllm` (which
# loads torch/CUDA and made a cold synchronous probe block past the request
# window). Emits one flat JSON object on stdout. See _do_sndr_state.
_SNDR_VERSION_PROBE = b"""
import json
o = {"vllm": None, "sndr": None, "configs": None, "patches": None}
# vLLM is pip-installed in the engine image, so its dist metadata is the running
# version (and reading it avoids a heavy `import vllm`).
try:
    from importlib.metadata import version
    o["vllm"] = version("vllm")
except Exception:
    o["vllm"] = None
# SNDR Core is MOUNTED into the container at runtime, so the source __version__
# is what's actually running - prefer it over the image's (stale) pip metadata.
try:
    from sndr.version import SNDR_CORE_VERSION
    o["sndr"] = SNDR_CORE_VERSION
except Exception:
    try:
        from sndr.version import SNDR_CORE_VERSION as _v
        o["sndr"] = _v
    except Exception:
        try:
            from importlib.metadata import version as _ver
            o["sndr"] = _ver("vllm-sndr-core")
        except Exception:
            o["sndr"] = None
try:
    import os, glob, sndr
    d = os.path.join(os.path.dirname(sndr.__file__), "model_configs", "builtin")
    o["configs"] = len(glob.glob(d + "/**/*.yaml", recursive=True)) or None
except Exception:
    pass
try:
    # Count registry entries by reading the source (import-free): importing
    # sndr.dispatcher.registry triggers ~9s of heavy transitive imports inside
    # the container, and we only need the COUNT for a version chip. Top-level
    # PATCH_REGISTRY entries are exactly 4-space-indented quoted keys (nested
    # keys are 8+ spaces, so they never start with '    "').
    import os as _os, sndr as _sndr
    _rp = _os.path.join(_os.path.dirname(_sndr.__file__), "dispatcher", "registry.py")
    _n = 0
    with open(_rp, encoding="utf-8") as _fh:
        for _ln in _fh:
            if _ln.startswith('    "') and '": ' in _ln[:48]:
                _n += 1
    o["patches"] = _n or None
except Exception:
    pass
print(json.dumps(o))
"""


def _own_container_id() -> str:
    """The container id THIS daemon runs in, read from /proc (empty if not in a
    container). Unlike ``socket.gethostname()`` this works under ``--network
    host``, where the hostname is the HOST's, not the container id. Used to
    refuse recreating our own container (which would kill the request)."""
    import re
    for path in ("/proc/self/cgroup", "/proc/self/mountinfo"):
        try:
            with open(path, encoding="utf-8") as fh:
                m = re.search(r"[0-9a-f]{64}", fh.read())
                if m:
                    return m.group(0)
        except Exception:
            continue
    return ""


def _is_management_daemon(name: str) -> bool:
    """Name looks like the SNDR management daemon (never safe to recreate from
    the panel — it serves the very request doing the recreate)."""
    nm = (name or "").lower()
    return "sndr-daemon" in nm or "sndr_daemon" in nm or nm.endswith("-daemon")


def _require_fastapi():
    try:
        from fastapi import Body, FastAPI, Header, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SNDR GUI API requires FastAPI runtime dependencies. "
            "Install with: pip install 'vllm-sndr-core[gui-api]' "
            "or pip install fastapi 'uvicorn[standard]'."
        ) from exc
    return Body, FastAPI, Header, HTTPException, Query, CORSMiddleware


def _dataclass_payload(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _dataclass_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_payload(item) for item in value]
    return value


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def terminal_gate(apply_on: bool, ssh_available: bool, host_ids: set[str], host_id: str) -> Optional[dict[str, str]]:
    """Security gate for the PTY terminal websocket — returns an error payload to
    send-and-close, or ``None`` when the connection may proceed.

    A full remote shell is the most dangerous capability in the GUI, so it is
    hard-gated behind ``SNDR_ENABLE_APPLY`` (``apply_on``). Extracted as a pure
    function so the policy is unit-tested independently of the websocket
    transport (the starlette TestClient cannot drive websockets cleanly)."""
    if not apply_on:
        return {"type": "error", "data": "Terminal disabled — start the daemon with SNDR_ENABLE_APPLY=1 to allow remote shell."}
    if not ssh_available:
        return {"type": "error", "data": "paramiko not installed — pip install 'vllm-sndr-core[gui-remote]'"}
    if host_id not in host_ids:
        return {"type": "error", "data": f"unknown host: {host_id}"}
    return None


def create_app(
    *,
    allowed_origins: Optional[tuple[str, ...]] = DEFAULT_ALLOWED_ORIGINS,
    enable_apply: Optional[bool] = None,
    bind_host: str = "127.0.0.1",
):
    """Create the SNDR Product API FastAPI app.

    ``enable_apply`` opts into real service-action execution (default OFF, also
    controllable via ``SNDR_ENABLE_APPLY``). When off, apply endpoints stay
    dry-run. ``bind_host`` informs the auth subsystem whether the daemon is
    exposed beyond loopback (affecting the ``auto`` auth default).
    """
    Body, FastAPI, Header, HTTPException, Query, CORSMiddleware = _require_fastapi()
    from .runtime_exec import (
        ApplyDisabledError,
        ConfirmationRequiredError,
        apply_enabled,
        execute_service_action,
    )

    apply_on = apply_enabled() if enable_apply is None else bool(enable_apply)

    app = FastAPI(
        title="SNDR Product API",
        version=SNDR_CORE_VERSION,
        description=(
            "Read-only API for SNDR GUI/TUI/web clients. The API exposes "
            "typed Product API snapshots and never writes V2 YAML, patch "
            "registries, or runtime artifacts."
        ),
    )
    app.state.read_only = True

    # User-aware authentication (local accounts + optional PAM/OAuth, 2FA),
    # adapting to the deployment context. Backward compatible: a configured
    # SNDR_GUI_TOKEN still works as a service/API bearer token.
    auth_config = _install_auth(app, bind_host=bind_host, apply_on=apply_on)

    # gzip the static bundle (index JS ~650 KB → ~190 KB) and every JSON
    # response. Added before CORS so CORS/security headers wrap it (the body is
    # compressed inside; the headers they add are not). minimum_size skips tiny
    # payloads where the gzip framing would cost more than it saves.
    from starlette.middleware.gzip import GZipMiddleware  # noqa: PLC0415
    app.add_middleware(GZipMiddleware, minimum_size=512)

    if allowed_origins is not None:
        # CORS is installed LAST so it is the OUTERMOST middleware — it must wrap
        # the auth middleware so even a 401 (auth-required) response carries the
        # Access-Control-Allow-Origin header. Otherwise a cross-origin GUI (the
        # fleet/cluster case: the GUI on one host, node daemons on others) sees
        # the 401 as a CORS error and never gets to show the login.
        #
        # The localhost regex covers a same-host GUI (Vite dev ports, etc.). For
        # a fleet the GUI runs on a different origin, so open it with:
        #   SNDR_ALLOW_ALL_ORIGINS=1  -> allow any origin (LAN homelab; auth is
        #                                still required on non-loopback binds)
        #   SNDR_ALLOWED_ORIGINS=a,b  -> allow these explicit origins
        _extra = [o.strip() for o in (os.environ.get("SNDR_ALLOWED_ORIGINS") or "").split(",") if o.strip()]
        _allow_all = (os.environ.get("SNDR_ALLOW_ALL_ORIGINS") or "").strip().lower() in ("1", "true", "yes", "on")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins) + _extra,
            allow_origin_regex=r".*" if _allow_all else r"http://(localhost|127\.0\.0\.1)(:\d+)?",
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # ── Security response headers (defense-in-depth) ──────────────────────
    # Applied to every response. The bundled SPA loads only same-origin
    # script/css (no inline <script>), so a strict script-src 'self' holds;
    # style-src keeps 'unsafe-inline' for React inline styles + ANSI log colours;
    # connect-src stays open (the app's job is talking to API daemons + the
    # terminal WebSocket). The strict CSP is skipped for the Swagger UI, which
    # loads its assets from a CDN and uses inline scripts. Set SNDR_DISABLE_CSP=1
    # to drop only the CSP header (other headers stay) if a deployment needs it.
    _csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "connect-src 'self' http: https: ws: wss:; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; "
        "object-src 'none'"
    )
    _csp_off = (os.environ.get("SNDR_DISABLE_CSP") or "").strip().lower() in ("1", "true", "yes", "on")
    _csp_skip = ("/docs", "/redoc", "/openapi.json")
    _force_hsts = (os.environ.get("SNDR_FORCE_HSTS") or "").strip().lower() in ("1", "true", "yes", "on")

    def _is_https(request) -> bool:  # type: ignore[no-untyped-def]
        # Direct TLS, or a reverse proxy that terminated TLS and set the
        # de-facto-standard forwarded-proto header.
        if request.url.scheme == "https":
            return True
        fwd = request.headers.get("x-forwarded-proto", "")
        return "https" in fwd.split(",")[0].strip().lower()

    @app.middleware("http")
    async def _security_headers(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"
        )
        # Cross-window isolation (defence-in-depth alongside frame-ancestors).
        # `-allow-popups` keeps any redirect/OAuth popup flow working.
        response.headers.setdefault(
            "Cross-Origin-Opener-Policy", "same-origin-allow-popups"
        )
        # HSTS only over TLS — pinning it on a plain-http homelab origin would
        # make the daemon unreachable. A reverse proxy fronting TLS gets it via
        # X-Forwarded-Proto; SNDR_FORCE_HSTS=1 forces it unconditionally.
        if _force_hsts or _is_https(request):
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        if not _csp_off and not any(request.url.path.startswith(p) for p in _csp_skip):
            response.headers.setdefault("Content-Security-Policy", _csp)
        return response

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        # read_only must reflect the ACTUAL apply gate (apply_on) — reporting a
        # constant True is misleading: a daemon started with SNDR_ENABLE_APPLY=1
        # would still announce itself read-only, masking that mutating endpoints
        # (node install, container ops, terminal) are live. apply_enabled is the
        # positive form so monitors don't have to invert it.
        return {
            "status": "ok",
            "service": "sndr-product-api",
            "version": SNDR_CORE_VERSION,
            "read_only": not apply_on,
            "apply_enabled": apply_on,
            "auth_required": auth_config.enabled,
        }

    @app.get("/api/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return _dataclass_payload(collect_capabilities())

    @app.get("/api/v1/catalog/summary")
    async def catalog_summary() -> dict[str, Any]:
        return _dataclass_payload(collect_catalog_summary())

    @app.get("/api/v1/overview")
    async def overview() -> dict[str, Any]:
        return _dataclass_payload(collect_product_overview())

    @app.get("/api/v1/configs/v2/catalog")
    async def configs_v2_catalog() -> dict[str, Any]:
        return _dataclass_payload(collect_v2_config_catalog())

    @app.get("/api/v1/configs/v2/preview")
    async def configs_v2_preview(
        model_id: str = Query(...),
        hardware_id: str = Query(...),
        profile_id: Optional[str] = None,
        runtime: Optional[str] = None,
    ) -> dict[str, Any]:
        return _dataclass_payload(
            preview_v2_config(
                model_id=model_id,
                hardware_id=hardware_id,
                profile_id=profile_id,
                runtime=runtime,
            )
        )

    @app.post("/api/v1/configs/v2/plan")
    async def configs_v2_plan(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            return _dataclass_payload(
                plan_v2_config_edit(
                    preset_id=payload.get("preset_id"),
                    model_id=str(payload["model_id"]),
                    hardware_id=str(payload["hardware_id"]),
                    profile_id=payload.get("profile_id"),
                    runtime=payload.get("runtime"),
                )
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {exc.args[0]}",
            ) from exc

    @app.post("/api/v1/configs/v2/apply")
    async def configs_v2_apply(
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        """Write a validated draft into the operator-local config dir.

        This is the one mutating endpoint, and it is deliberately narrow: it
        never writes the repo builtin catalog and never touches a remote host.
        """
        try:
            result = apply_v2_config_plan(
                preset_id=payload.get("preset_id"),
                model_id=str(payload["model_id"]),
                hardware_id=str(payload["hardware_id"]),
                profile_id=payload.get("profile_id"),
                runtime=payload.get("runtime"),
                expected_plan_id=payload.get("expected_plan_id"),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {exc.args[0]}",
            ) from exc
        body = _dataclass_payload(result)
        if result.status == "conflict":
            raise HTTPException(status_code=409, detail=body)
        if result.status == "blocked":
            raise HTTPException(status_code=422, detail=body)
        from .config_editor import invalidate_v2_config_catalog  # config changed → drop the cache
        invalidate_v2_config_catalog()
        return body

    @app.get("/api/v1/configs/v2/user-presets")
    async def configs_v2_user_presets() -> dict[str, Any]:
        presets = list_user_presets()
        return {
            "count": len(presets),
            "presets": [_dataclass_payload(item) for item in presets],
        }

    @app.post("/api/v1/configs/v2/layer/apply")
    async def configs_v2_layer_apply(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Write an edited layer YAML to the operator-local config dir."""
        result = apply_v2_layer(
            kind=str(payload.get("kind", "")),
            layer_id=str(payload.get("layer_id", "")),
            yaml_text=str(payload.get("yaml_text", "")),
        )
        body = _dataclass_payload(result)
        if result.status == "conflict":
            raise HTTPException(status_code=409, detail=body)
        if result.status == "blocked":
            raise HTTPException(status_code=422, detail=body)
        from .config_editor import invalidate_v2_config_catalog  # config changed → drop the cache
        invalidate_v2_config_catalog()
        return body

    @app.get("/api/v1/configs/v2/layer/{kind}/{layer_id}")
    async def configs_v2_layer(kind: str, layer_id: str) -> dict[str, Any]:
        if kind.lower() not in {"model", "hardware", "profile", "preset"}:
            raise HTTPException(status_code=400, detail=f"Unknown layer kind: {kind}")
        try:
            return get_v2_layer(kind, layer_id)
        except Exception as exc:  # missing/invalid layer id
            raise HTTPException(
                status_code=404,
                detail=f"Layer not found: {kind}/{layer_id} ({exc})",
            ) from exc

    @app.get("/api/v1/models/cache")
    async def models_cache() -> dict[str, Any]:
        return _dataclass_payload(collect_model_cache_report())

    @app.get("/api/v1/models/hub/search")
    def models_hub_search(query: str = "", limit: int = 20) -> dict[str, Any]:
        """Search the Hugging Face Hub for downloadable models."""
        from . import hub

        try:
            return {"results": hub.search_models(query, limit=limit)}
        except Exception as exc:  # network / Hub unavailable
            raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {exc}")

    @app.post("/api/v1/models/download")
    async def models_download(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Download model weights — a catalog model (``model_id``) or any Hugging
        Face repo (``repo_id``). With --enable-apply it runs the pull as a live
        background job (status/log/progress); otherwise a dry-run job. Both ids
        are strictly validated (no arbitrary shell input)."""
        import re

        repo_id = str(payload.get("repo_id") or "").strip()
        model_id = str(payload.get("model_id") or "").strip()

        if repo_id:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
                raise HTTPException(status_code=400, detail="Invalid Hugging Face repo id (expected org/name).")
            label, cli = repo_id, f"huggingface-cli download {repo_id}"
            summary = {"repo_id": repo_id, "source": "huggingface"}
        elif model_id:
            if not re.fullmatch(r"[A-Za-z0-9._\-/]{1,128}", model_id):
                raise HTTPException(status_code=400, detail="Invalid model id.")
            known = {entry.model_id for entry in collect_model_cache_report().models}
            if known and model_id not in known:
                raise HTTPException(status_code=404, detail=f"Unknown model id: {model_id}")
            label, cli = model_id, f"sndr model pull {model_id}"
            summary = {"model_id": model_id, "source": "catalog"}
        else:
            raise HTTPException(status_code=400, detail="Provide model_id (catalog) or repo_id (Hugging Face).")

        if apply_on:
            from .background_exec import run_background_command

            exec_cmd = cli if repo_id else f"python3 -m sndr.cli model pull {model_id}"
            job = run_background_command(kind="model.download", title=f"download {label}", summary=summary, command=exec_cmd)
            return _dataclass_payload(job)
        job = create_dry_run_job(
            kind="model.download",
            title=f"download {label}",
            summary=summary,
            steps=[("Pull weights", cli)],
            cli_mirror=[cli],
            note="Dry-run — start the daemon with --enable-apply to download here, or run the command on the host.",
        )
        return _dataclass_payload(job)

    @app.get("/api/v1/memory/fit")
    async def memory_fit(
        model_id: str = Query(...),
        hardware_id: str = Query(...),
    ) -> dict[str, Any]:
        try:
            return _dataclass_payload(
                estimate_fit(model_id=model_id, hardware_id=hardware_id)
            )
        except Exception as exc:  # unknown model/hardware id
            raise HTTPException(
                status_code=404,
                detail=f"Cannot build fit report for {model_id} × {hardware_id} ({exc})",
            ) from exc

    @app.get("/api/v1/presets")
    async def presets_list(
        family: Optional[str] = None,
        workload: Optional[str] = None,
        hardware: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        return _dataclass_payload(
            list_presets(
                family=family,
                workload=workload,
                hardware=hardware,
                mode=mode,
                status=status,
            )
        )

    @app.get("/api/v1/presets/recommend")
    async def presets_recommend(
        workload: str = Query(...),
        hardware: Optional[str] = None,
        concurrency: Optional[int] = None,
        top: int = 5,
    ) -> dict[str, Any]:
        try:
            return _dataclass_payload(
                recommend_presets(
                    workload=workload,
                    hardware=hardware,
                    concurrency=concurrency,
                    top=top,
                )
            )
        except UnknownWorkloadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/presets/{preset_id}")
    async def presets_get(preset_id: str) -> dict[str, Any]:
        try:
            return _dataclass_payload(get_preset(preset_id))
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/presets/{preset_id}/explain")
    async def presets_explain(preset_id: str) -> dict[str, Any]:
        try:
            return _dataclass_payload(explain_preset(preset_id))
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PresetComposeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/v1/launch/plan")
    async def launch_plan(
        preset_id: str = Query(...),
        runtime_target: str = "docker_compose",
        patch_policy: str = "safe",
        host: str = "127.0.0.1",
        mode: str = "remote",
    ) -> dict[str, Any]:
        try:
            return _dataclass_payload(
                build_launch_plan(
                    preset_id=preset_id,
                    runtime_target=runtime_target,
                    patch_policy=patch_policy,
                    host=host,
                    mode=mode,
                )
            )
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PresetComposeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/launch/apply")
    async def launch_apply(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Launch a preset (= start its service).

        Dry-run by default. When apply is enabled it executes the start action;
        because a launch is mutating it requires ``confirm: true``.
        """
        try:
            preset_id = str(payload["preset_id"])
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        runtime_target = payload.get("runtime_target", "docker_compose")
        host = payload.get("host", "127.0.0.1")
        try:
            if apply_on:
                job = execute_service_action(
                    preset_id=preset_id,
                    action="start",
                    runtime_target=runtime_target,
                    host=host,
                    transport=str(payload.get("transport", "local")),
                    ssh_target=str(payload.get("ssh_target", "")),
                    confirm=bool(payload.get("confirm", False)),
                    enabled=apply_on,
                )
            else:
                job = apply_service_action(
                    preset_id=preset_id,
                    action="start",
                    runtime_target=runtime_target,
                    host=host,
                )
        except ConfirmationRequiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApplyDisabledError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _dataclass_payload(job)

    @app.get("/api/v1/patches")
    async def patches_list(
        tier: Optional[str] = None,
        lifecycle: Optional[str] = None,
        family: Optional[str] = None,
        default_on: Optional[bool] = None,
        has_upstream: Optional[bool] = None,
    ) -> dict[str, Any]:
        all_rows = [_dataclass_payload(row) for row in list_patches()]
        rows = [
            _dataclass_payload(row)
            for row in list_patches(
                tier=tier,
                lifecycle=lifecycle,
                family=family,
                default_on=default_on,
                has_upstream=has_upstream,
            )
        ]
        return {
            "filters": {
                "tier": tier,
                "lifecycle": lifecycle,
                "family": family,
                "default_on": default_on,
                "has_upstream": has_upstream,
            },
            "matched": len(rows),
            "total": len(all_rows),
            "patches": rows,
            "summary": {
                "tier_counts": _count_by(all_rows, "tier"),
                "lifecycle_counts": _count_by(all_rows, "lifecycle"),
                "production_default_counts": _count_by(
                    all_rows, "production_default"
                ),
                "implementation_status_counts": _count_by(
                    all_rows, "implementation_status"
                ),
            },
        }

    @app.get("/api/v1/patches/doctor")
    async def patches_doctor() -> dict[str, Any]:
        return _dataclass_payload(run_patch_doctor())

    @app.get("/api/v1/patches/manifest")
    async def patches_manifest(drift: bool = True) -> dict[str, Any]:
        """Per-pin anchor source-of-truth status: each pin's manifest (vLLM/genesis
        version, file/patch/anchor counts, schema validity), which is ACTIVE for the
        running vLLM, and live drift of the active manifest vs the installed source.
        Read-only. ``drift=false`` skips the (cheap) source verification."""
        from .patches.anchor_status import manifest_status

        return manifest_status(drift=drift)

    @app.get("/api/v1/patches/retire-impact")
    async def patches_retire_impact() -> dict[str, Any]:
        """Anchor-SoT retire-impact: which active dependents a retired patch would
        break. HIGH = perf-bearing dependent whose anchor targets the retired
        patch's emitted bytes (the dev301-class silent regression); MEDIUM =
        registry edge only. The signal the pin-bump preflight gate exists for.
        Read-only; fail-safe (empty report off-engine)."""
        from .patches.retire_impact_status import retire_impact_status

        return retire_impact_status()

    @app.get("/api/v1/license")
    async def license_status() -> dict[str, Any]:
        """License + sndr_engine tier status — installed?, entitled?, subject/expiry,
        and how many engine-tier patches it unlocks. Read-only."""
        from . import licensing

        return licensing.status()

    @app.get("/api/v1/flags/matrix")
    async def flags_matrix(container: Optional[str] = None) -> dict[str, Any]:
        """The full GENESIS_ENABLE_* catalogue with defaults; when ``container``
        names a local engine, overlays its live ON/OFF flags + drift verdicts."""
        from . import flag_matrix

        live: Optional[set[str]] = None
        if container:
            try:
                inspect = _container_op(lambda: _local_control().inspect(container))
                live = flag_matrix.live_flags_from_inspect(inspect)
            except Exception:  # noqa: BLE001 — container may be absent; static matrix still useful
                live = None
        return flag_matrix.build_matrix(live)

    @app.get("/api/v1/patches/overrides")
    async def patches_overrides_get() -> dict[str, Any]:
        from .patch_overrides import load

        return {"overrides": load()}

    @app.post("/api/v1/patches/overrides")
    async def patches_overrides_set(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Operator-local force on/off (or clear) of a patch. Persisted under
        $SNDR_HOME and emitted into the launch env."""
        from .patch_overrides import set_override

        try:
            overrides = set_override(
                str(payload.get("patch_id", "")),
                str(payload.get("state", "")),
                str(payload.get("env_flag", "")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "overrides": overrides}

    @app.get("/api/v1/deploy/targets")
    async def deploy_targets() -> dict[str, Any]:
        """Supported deployment targets + the live host inventory."""
        from . import deployment

        return {"targets": deployment.list_targets(), "host": deployment.host_inventory()}

    @app.post("/api/v1/deploy/plan")
    async def deploy_plan(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Render the deployment artifact + dependency plan + fine launch
        parameters for a preset on a chosen target. Read-only — no host
        mutation. host_paths overrides the symbolic mount defaults."""
        from . import deployment

        preset_id = str(payload.get("preset_id", "")).strip()
        target = str(payload.get("target", "")).strip()
        raw_paths = payload.get("host_paths") or {}
        host_paths = {str(k): str(v) for k, v in raw_paths.items()} if isinstance(raw_paths, dict) else None
        image_override = str(payload.get("image_override") or "").strip() or None
        with_daemon = bool(payload.get("with_daemon", False))
        try:
            return deployment.build_deployment(
                preset_id, target, host_paths=host_paths,
                image_override=image_override, with_daemon=with_daemon,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")

    # --- Remote installer (Setup wizard): plan + dry-run, host-driven ---------
    @app.get("/api/v1/install/targets")
    async def install_targets() -> dict[str, Any]:
        """Deployment targets (incl. Proxmox LXC/VM) + the saved host registry."""
        from . import deployment

        return {
            "targets": deployment.list_targets(),
            "hosts": [_augment_host(h, host_profile_payload(h)) for h in list_host_profiles()],
            "apply_enabled": apply_on,
        }

    @app.post("/api/v1/install/plan")
    async def install_plan(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Read-only, dry-run install plan: render the artifact for a preset/
        target and lay it out as the steps a remote install would run, flagging
        the infrastructure-mutating ones. Executes nothing."""
        from . import installer

        host_id = str(payload.get("host_id") or "").strip()
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        raw_paths = payload.get("host_paths") or {}
        host_paths = {str(k): str(v) for k, v in raw_paths.items()} if isinstance(raw_paths, dict) else None
        try:
            return installer.build_install_plan(
                host={"label": profile.label, "host": profile.host},
                preset_id=str(payload.get("preset_id", "")).strip(),
                target=str(payload.get("target", "")).strip(),
                host_paths=host_paths,
                image_override=str(payload.get("image_override") or "").strip() or None,
                with_daemon=bool(payload.get("with_daemon")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {payload.get('preset_id')}")

    @app.post("/api/v1/fleet/deploy-plan")
    async def fleet_deploy_plan(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Read-only multi-host deploy plan: render the install plan for one
        preset/target across N selected hosts at once and roll the results up
        (per-host ready/error + mutating-step counts). Executes NOTHING — the
        actual apply stays per-host via /install/apply (apply+confirm gated)."""
        from . import installer

        preset_id = str(payload.get("preset_id", "")).strip()
        target = str(payload.get("target", "")).strip()
        host_ids = payload.get("host_ids") or []
        if not isinstance(host_ids, list) or not host_ids:
            raise HTTPException(status_code=400, detail="host_ids must be a non-empty list")
        with_daemon = bool(payload.get("with_daemon"))
        image_override = str(payload.get("image_override") or "").strip() or None

        profiles = {p.id: p for p in list_host_profiles()}
        results: list[dict[str, Any]] = []
        for raw in host_ids:
            hid = str(raw)
            profile = profiles.get(hid)
            if profile is None:
                results.append({"host_id": hid, "ok": False, "error": "unknown host", "plan": None})
                continue
            try:
                plan = installer.build_install_plan(
                    host={"label": profile.label, "host": profile.host},
                    preset_id=preset_id, target=target, host_paths=None,
                    image_override=image_override, with_daemon=with_daemon,
                )
                steps = plan.get("steps", []) if isinstance(plan, dict) else []
                mutating = sum(1 for s in steps if isinstance(s, dict) and s.get("mutating"))
                results.append({
                    "host_id": hid, "label": profile.label, "host": profile.host,
                    "ok": True, "error": None, "mutating_steps": mutating, "plan": plan,
                })
            except ValueError as exc:
                results.append({"host_id": hid, "label": profile.label, "ok": False, "error": str(exc), "plan": None})
            except KeyError:
                results.append({"host_id": hid, "label": profile.label, "ok": False, "error": f"unknown preset: {preset_id}", "plan": None})

        ready = [r for r in results if r["ok"]]
        return {
            "preset_id": preset_id,
            "target": target,
            "results": results,
            "rollup": {
                "hosts": len(results),
                "ready": len(ready),
                "errors": len(results) - len(ready),
                "mutating_steps_total": sum(r.get("mutating_steps", 0) for r in ready),
                "apply_enabled": apply_on,
            },
        }

    def _ssh_target_for(host_id: str):
        """Resolve a stored host profile + build its SSH target dict (creds stay
        server-side, keyed by host_id). Raises 404 for an unknown host."""
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        target = {
            "host": profile.host, "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent", "key_path": profile.ssh_key_path,
            "secret_id": f"ssh:{host_id}",
        }
        return profile, target

    @app.post("/api/v1/install/apply")
    async def install_apply(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Execute an install plan on a host over SSH — MUTATING, double-gated.

        Requires the daemon to run with SNDR_ENABLE_APPLY (apply_on) AND an
        explicit ``confirm: true`` in the body. SFTPs the artifact and runs the
        plan's commands on the host, returning per-step results."""
        from . import installer, ssh_client

        if not apply_on:
            raise HTTPException(status_code=403, detail="apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1")
        if not bool(payload.get("confirm")):
            raise HTTPException(status_code=400, detail="explicit confirm:true is required to run on a host")

        host_id = str(payload.get("host_id") or "").strip()
        profile, ssh_target = _ssh_target_for(host_id)
        try:
            return installer.apply_install_plan(
                host={"label": profile.label, "host": profile.host},
                preset_id=str(payload.get("preset_id", "")).strip(),
                target=str(payload.get("target", "")).strip(),
                ssh_target=ssh_target,
                run_apply=ssh_client.run_apply,
                apply_enabled=True,
                confirm=True,
                image_override=str(payload.get("image_override") or "").strip() or None,
                with_daemon=bool(payload.get("with_daemon")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {payload.get('preset_id')}")

    @app.post("/api/v1/install/node")
    async def install_node(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """One-button node setup over SSH — MUTATING, double-gated. Ships the
        daemon code + runs the management daemon on the engine host so the GUI can
        switch to it. Requires apply_on AND confirm:true."""
        from . import node_setup, ssh_client

        if not apply_on:
            raise HTTPException(status_code=403, detail="apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1")
        if not bool(payload.get("confirm")):
            raise HTTPException(status_code=400, detail="explicit confirm:true is required to run on a host")

        host_id = str(payload.get("host_id") or "").strip()
        profile, ssh_target = _ssh_target_for(host_id)
        try:
            result = node_setup.setup_node(
                ssh_target=ssh_target, run_apply=ssh_client.run_apply,
                apply_enabled=True, confirm=True,
                admin_password=str(payload.get("admin_password") or ""),
                port=int(payload.get("port") or profile.port or 8765),
                engine_port=int(payload.get("engine_port") or profile.engine_port or 8102),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not result.get("applied") and "password" in (result.get("error") or ""):
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    # ─── Container management (scoped to vLLM/engine containers) ──────────
    #
    # One REST shape over two transports (Approach A): the LOCAL family talks the
    # docker socket mounted into this daemon (the node case); the HOST family runs
    # docker over SSH to a registered host (the central-GUI case). Read ops are
    # ungated; lifecycle needs apply+confirm; exec additionally needs
    # SNDR_ENABLE_EXEC. The whitelist lives in container_ops, enforced by both
    # backends, so neither channel can escape the engine-container scope.

    def _container_op(fn):
        from . import container_ops as _co
        try:
            return fn()
        except _co.NotManagedError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except _co.ContainerOpError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    def _local_control():
        from . import container_ops as _co
        from sndr.deps import checkers
        if not checkers._docker_socket_present():
            raise HTTPException(
                status_code=503,
                detail="docker socket not mounted — reinstall the node to enable container management")
        return _co.SocketContainerControl()

    def _host_control(host_id: str):
        from . import container_ops as _co
        _profile, target = _ssh_target_for(host_id)
        return _co.SshContainerControl(target=target)

    # `docker system df` scans every image layer/volume → it is genuinely slow
    # (seconds) AND, over the per-host SSH pool, holds the lock while live stats
    # wait behind it — which made the Containers panel feel sluggish to open.
    # Disk usage barely changes, so cache it per source with a short TTL.
    import time as _time
    _df_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _DF_TTL = 120.0

    def _system_df_cached(control, key: str) -> dict[str, Any]:
        hit = _df_cache.get(key)
        if hit and (_time.time() - hit[0]) < _DF_TTL:
            return hit[1]
        data = _container_op(lambda: control.system_df())
        _df_cache[key] = (_time.time(), data)
        return data

    # Per-container version probe runs a `docker exec` (sub-second now, but with
    # exec overhead) — cache successful results so switching tabs/containers is
    # instant. Versions change only on an image swap, so a short TTL is plenty.
    _sndr_state_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _SNDR_STATE_TTL = 90.0

    def _sndr_state_cached(control, name: str, key: str) -> dict[str, Any]:
        hit = _sndr_state_cache.get(key)
        if hit and (_time.time() - hit[0]) < _SNDR_STATE_TTL:
            return hit[1]
        data = _do_sndr_state(control, name)
        if data.get("ok"):  # never cache a failed probe
            _sndr_state_cache[key] = (_time.time(), data)
        return data

    def _audit(action: str, name: str, source: str, detail: dict[str, Any]) -> None:
        # Reuse the persisted, bounded event feed as the container audit log — the
        # GUI's Events panel + the /events SSE already render it. Every mutating
        # container op leaves a who/what/when trace here.
        try:
            record_event(f"container.{action}", f"{action} {name} ({source})",
                         {"container": name, "source": source, **detail})
        except Exception:
            pass

    def _do_action(control, name: str, payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
        from . import container_ops as _co
        action = str(payload.get("action", "")).strip().lower()
        if action not in ("start", "stop", "restart"):
            raise HTTPException(status_code=400, detail=f"unsupported action: {action!r}")
        gate = _co.gate_lifecycle(apply_on=apply_on, confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        _container_op(lambda: getattr(control, action)(name))
        _audit(action, name, source, {})
        return {"ok": True, "action": action, "container": name}

    def _do_exec(control, name: str, payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
        from . import container_ops as _co
        argv = payload.get("argv")
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv) or not argv:
            raise HTTPException(status_code=400, detail="argv must be a non-empty list of strings")
        gate = _co.gate_exec(apply_on=apply_on, exec_on=_co.exec_enabled(), confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        result = _container_op(lambda: control.exec(name, argv))
        _audit("exec", name, source, {"argv": argv, "exit_code": result.exit_code})
        return {"ok": True, "container": name, **result.to_dict()}

    def _do_pull(control, name: str, payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
        from . import container_ops as _co
        gate = _co.gate_lifecycle(apply_on=apply_on, confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        result = _container_op(lambda: control.pull(name))
        # Optionally restart so the new image takes effect (best-effort, gated already).
        if payload.get("restart"):
            _container_op(lambda: control.restart(name))
            result["restarted"] = True
        _audit("pull", name, source, {"image": result.get("image"), "restarted": bool(payload.get("restart"))})
        return {"ok": True, "container": name, **result}

    def _do_recreate(control, name: str, source: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Recreate a container so a new (or rolled-back) image actually takes
        effect — unlike a plain restart. Guarded: never recreates the management
        daemon from itself (it would kill the request), and never engines (those
        follow the pin policy / manual commands). Records the prior image so a
        rollback can recreate from it."""
        from . import container_ops as _co, update_prefs
        # Self-protection — NEVER recreate the daemon's own container (it serves
        # this request). Two independent checks, because a single one is brittle:
        #   1) name looks like a management daemon (covers a remote daemon too),
        #   2) the target id == our own container id read from /proc (robust
        #      under --network host, where gethostname() is the HOST's name).
        if _is_management_daemon(name):
            raise HTTPException(status_code=400,
                                detail="the SNDR management daemon can't be recreated from the panel — recreate it from the host / reinstall the node")
        inspect = _container_op(lambda: control.inspect(name))
        cid = str(inspect.get("Id") or "")
        own = _own_container_id()
        if own and cid and (cid == own or cid.startswith(own) or own.startswith(cid)):
            raise HTTPException(status_code=400,
                                detail="cannot recreate the management daemon from itself — recreate it from the host")
        image = str((inspect.get("Config") or {}).get("Image") or "")
        if _is_engine_container(name, image):
            raise HTTPException(status_code=400,
                                detail="vLLM engines update via the pin policy (manual commands), not in-panel recreate")
        gate = _co.gate_lifecycle(apply_on=apply_on, confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        if payload.get("rollback"):
            prev = update_prefs.get_previous(source, name)
            if not prev:
                raise HTTPException(status_code=400, detail="no previous image recorded for rollback")
            result = _container_op(lambda: control.recreate(name, image=prev))
            _audit("rollback", name, source, {"image": result.get("image")})
            return {"ok": True, "container": name, "rolled_back": True, **result}
        # apply latest: remember the running image as 'previous', pull, recreate
        prev_running = str(inspect.get("Image") or "")
        _container_op(lambda: control.pull(name))
        result = _container_op(lambda: control.recreate(name))
        update_prefs.set_previous(source, name, prev_running)
        _audit("recreate", name, source, {"image": result.get("image")})
        return {"ok": True, "container": name, **result}

    def _require_apply(detail: str) -> None:
        if not apply_on:
            raise HTTPException(status_code=403, detail=detail)

    def _do_fs(control, name: str, path: str) -> dict[str, Any]:
        # Read-only file browsing: FIXED ls/head commands on a validated absolute
        # path (not operator-arbitrary), so it sits in the read tier — gated by
        # apply (operator-enabled daemon), NOT the stricter SNDR_ENABLE_EXEC which
        # guards arbitrary in-container command execution.
        _require_apply("file browsing needs apply — start the daemon with SNDR_ENABLE_APPLY=1")
        return _container_op(lambda: control.list_dir(name, path))

    def _do_file(control, name: str, path: str, max_bytes: int = 65536) -> dict[str, Any]:
        _require_apply("file browsing needs apply — start the daemon with SNDR_ENABLE_APPLY=1")
        return _container_op(lambda: control.read_file(name, path, max_bytes=max_bytes))

    def _do_settings(control, name: str, payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
        from . import container_ops as _co
        gate = _co.gate_lifecycle(apply_on=apply_on, confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        cpus = payload.get("cpus")
        memory = payload.get("memory")
        rp = payload.get("restart_policy")
        try:
            cpus_v = float(cpus) if cpus not in (None, "") else None
            mem_v = int(memory) if memory not in (None, "") else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="cpus/memory must be numbers")
        result = _container_op(lambda: control.update_settings(
            name, cpus=cpus_v, memory=mem_v, restart_policy=(str(rp) if rp else None)))
        _audit("settings", name, source, {"cpus": cpus_v, "memory": mem_v, "restart_policy": rp})
        return {"ok": True, "container": name, **result}

    def _do_network(control, name: str, payload: dict[str, Any], source: str = "local") -> dict[str, Any]:
        from . import container_ops as _co
        gate = _co.gate_lifecycle(apply_on=apply_on, confirm=bool(payload.get("confirm")))
        if not gate.allowed:
            raise HTTPException(status_code=gate.status, detail=gate.reason)
        network = str(payload.get("network", ""))
        action = str(payload.get("action", "connect")).lower()
        if action not in ("connect", "disconnect"):
            raise HTTPException(status_code=400, detail=f"unsupported network action: {action!r}")
        fn = control.connect_network if action == "connect" else control.disconnect_network
        result = _container_op(lambda: fn(name, network))
        _audit(f"network.{action}", name, source, {"network": network})
        return {"ok": True, "container": name, **result}

    def _stream_logs_response(control, name: str, tail: int):
        """Wrap a backend's blocking log generator as an ND-JSON stream. Each line
        is {"line": "..."}; {"hb": true} heartbeats keep the connection writable so
        a client disconnect is noticed and the SSH channel / socket is released."""
        import asyncio
        import json as _json

        from starlette.responses import StreamingResponse

        sync_it = _container_op(lambda: iter(control.stream_logs(name, tail=tail)))

        def _next_or_none(it):
            try:
                return next(it)
            except StopIteration:
                return None

        async def gen():
            try:
                while True:
                    chunk = await asyncio.to_thread(_next_or_none, sync_it)
                    if chunk is None:
                        break
                    yield (_json.dumps({"line": chunk}) if chunk else '{"hb":true}') + "\n"
            finally:
                close = getattr(sync_it, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    def _is_engine_container(name: str, image: str) -> bool:
        """True for a vLLM *inference engine* container. The management daemon
        runs FROM the vLLM image but is NOT an engine — exclude it so it isn't
        pin-gated or marked critical (semi/auto updates of the sidecar are fine)."""
        nm = (name or "").lower()
        if "sndr-daemon" in nm or "sndr_daemon" in nm or nm.endswith("-daemon"):
            return False
        return "vllm" in ((image or "").lower() + " " + nm)

    def _do_update_plan(control, name: str, source: str = "local") -> dict[str, Any]:
        """Read-only: how to update this container. vLLM engines follow the pin
        policy (deliberate, copyable commands + rollback); other managed
        containers can be guard-updated (pull image + restart). Carries the
        operator's chosen update mode (manual/semi/auto) and whether the
        container is critical (engines — automatic is then forbidden)."""
        from . import updater, update_prefs
        inspect = _container_op(lambda: control.inspect(name))
        image = str((inspect.get("Config") or {}).get("Image") or "")
        is_engine = _is_engine_container(name, image)
        # Update detection (local): the container runs a specific image sha
        # (.Image); compare it to the current local image id of its tag. If they
        # differ, a newer image was pulled but this container wasn't recreated.
        running_sha = str(inspect.get("Image") or "")
        latest_sha = ""
        try:
            latest_sha = control.image_id(image)
        except Exception:
            latest_sha = ""
        update_available = bool(latest_sha) and bool(running_sha) and latest_sha != running_sha
        pins = updater.supported_pins()
        canonical = pins[0] if pins else None
        commands: list[str] = []
        if is_engine and canonical:
            commands = [
                f"docker pull vllm/vllm-openai:{canonical}",
                "docker tag vllm/vllm-openai:nightly vllm/vllm-openai:nightly-previous   # keep for rollback",
                f"docker tag vllm/vllm-openai:{canonical} vllm/vllm-openai:nightly",
                "# then re-run the engine via its start script to pick up the new image",
            ]
        return {
            "container": name, "image": image, "is_engine": is_engine,
            "supported_pins": pins, "canonical_pin": canonical,
            "guarded_update": not is_engine,
            "policy": "vLLM pin moves deliberately — ≤1 active pin plus a 'previous' tag for rollback.",
            "commands": commands,
            # Update-mode selection (manual/semi/auto). Engines are critical:
            # automatic is blocked (pin policy), so the GUI greys it out.
            "mode": update_prefs.get_mode(source, name),
            "is_critical": is_engine,
            "modes": list(update_prefs.VALID_MODES),
            # Update detection — a newer local image exists for this tag.
            "update_available": update_available,
            "running_image_id": running_sha[:19],
            "latest_image_id": latest_sha[:19],
            # Rollback availability — a prior image was recorded at last update.
            "has_previous": bool(update_prefs.get_previous(source, name)),
        }

    def _do_set_update_mode(control, name: str, source: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist the operator's chosen update mode for a container."""
        from . import update_prefs
        inspect = _container_op(lambda: control.inspect(name))
        image = str((inspect.get("Config") or {}).get("Image") or "")
        is_critical = _is_engine_container(name, image)
        return update_prefs.set_mode(source, name, str(payload.get("mode", "")), is_critical=is_critical)

    def _do_sndr_state(control, name: str) -> dict[str, Any]:
        """Read-only: the project versions running INSIDE this container — SNDR
        Core version, vLLM build, builtin-config count and patch-registry size —
        by running the introspection probe in it. Works for both the local socket
        control and the SSH control (both implement ``exec``). The same data the
        Hosts fleet view shows, but per-container in the Containers tab."""
        import base64
        import json as _json
        result: dict[str, Any] = {"ok": False, "container": name, "vllm_version": None,
                                  "sndr_version": None, "configs": None, "patches": None, "error": None}
        # Ship the probe as base64 over stdin (not `python3 -c <multiline>`): the
        # SSH control joins argv into a shell command, so a multi-line arg would
        # break — a single base64 token is shell-safe for both controls.
        # Use a LIGHT probe: read versions via importlib.metadata (no heavy
        # `import vllm`, which loads torch/CUDA and made a cold call block past
        # the request window). Suppress stderr — the socket control merges it into
        # stdout, and patch-registry warnings would pollute the JSON.
        b64 = base64.b64encode(_SNDR_VERSION_PROBE).decode("ascii")
        try:
            res = control.exec(name, ["sh", "-c", f"echo {b64} | base64 -d | python3 - 2>/dev/null"], timeout=25.0)
        except Exception as exc:  # container gone / docker error / SSH down
            result["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return result
        if res.exit_code != 0 or not (res.stdout or "").strip():
            result["error"] = ((res.stderr or res.stdout or "introspection failed").strip())[:300]
            return result
        # The probe emits one flat JSON object, but it may be surrounded by log
        # lines (e.g. "[PatchSpec] … stale") or docker stream frame bytes. Scan
        # for the last brace-delimited block that parses and looks like the probe.
        import re
        data = None
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", res.stdout))):
            try:
                d = _json.loads(m.group(0))
            except Exception:
                continue
            if isinstance(d, dict) and ("vllm" in d or "sndr" in d):
                data = d
                break
        if data is None:
            snippet = (res.stdout or "").strip().replace("\n", " ")[:160]
            result["error"] = f"unparseable introspection output: {snippet}"
            return result
        result.update(ok=True, vllm_version=data.get("vllm"), sndr_version=data.get("sndr"),
                      configs=data.get("configs"), patches=data.get("patches"))
        return result

    # Local family (this daemon's host, via the docker socket) ------------
    @app.get("/api/v1/containers")
    def containers_list() -> dict[str, Any]:
        control = _local_control()
        items = _container_op(control.list_managed)
        return {"containers": [c.to_dict() for c in items], "source": "socket"}

    # Declared BEFORE /containers/{name} so the literal path isn't shadowed by the
    # {name} param. One call returns stats for every managed container.
    @app.get("/api/v1/containers/stats")
    def containers_stats_all() -> dict[str, Any]:
        return {"stats": _container_op(lambda: _local_control().list_stats())}

    @app.get("/api/v1/containers/{name}")
    def container_inspect(name: str) -> dict[str, Any]:
        return _container_op(lambda: _local_control().inspect(name))

    @app.get("/api/v1/containers/{name}/logs")
    def container_logs(name: str, tail: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
        return {"container": name, "logs": _container_op(lambda: _local_control().logs(name, tail=tail))}

    @app.get("/api/v1/containers/{name}/stats")
    def container_stats(name: str) -> dict[str, Any]:
        return {"container": name, "stats": _container_op(lambda: _local_control().stats(name))}

    @app.post("/api/v1/containers/{name}/action")
    def container_action(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_action(_local_control(), name, payload)

    @app.post("/api/v1/containers/{name}/exec")
    def container_exec(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_exec(_local_control(), name, payload)

    @app.get("/api/v1/containers/{name}/top")
    def container_top(name: str) -> dict[str, Any]:
        return {"container": name, **_container_op(lambda: _local_control().top(name))}

    @app.get("/api/v1/containers/{name}/changes")
    def container_changes(name: str) -> dict[str, Any]:
        return {"container": name, "changes": _container_op(lambda: _local_control().changes(name))}

    @app.get("/api/v1/containers/{name}/fs")
    def container_fs(name: str, path: str = Query(default="/")) -> dict[str, Any]:
        return _do_fs(_local_control(), name, path)

    @app.get("/api/v1/containers/{name}/file")
    def container_file(name: str, path: str = Query(...),
                             max_bytes: int = Query(default=65536, ge=1, le=5_000_000)) -> dict[str, Any]:
        return _do_file(_local_control(), name, path, max_bytes)

    @app.get("/api/v1/containers/{name}/update-plan")
    def container_update_plan(name: str) -> dict[str, Any]:
        return _do_update_plan(_local_control(), name, "local")

    @app.post("/api/v1/containers/{name}/update-mode")
    def container_set_update_mode(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_set_update_mode(_local_control(), name, "local", payload)

    @app.get("/api/v1/containers/{name}/sndr-state")
    def container_sndr_state(name: str) -> dict[str, Any]:
        return _sndr_state_cached(_local_control(), name, f"local/{name}")

    @app.post("/api/v1/containers/{name}/pull")
    def container_pull(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_pull(_local_control(), name, payload, source="local")

    @app.post("/api/v1/containers/{name}/recreate")
    def container_recreate(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_recreate(_local_control(), name, "local", payload)

    @app.get("/api/v1/containers/{name}/scan")
    def container_scan(name: str) -> dict[str, Any]:
        return _container_op(lambda: _local_control().scan_image(name))

    def _do_source(control, name: str) -> dict[str, Any]:
        from . import container_link
        inspect = _container_op(lambda: control.inspect(name))
        return container_link.source_report(name, inspect)

    @app.get("/api/v1/containers/{name}/source")
    def container_source(name: str) -> dict[str, Any]:
        return _do_source(_local_control(), name)

    @app.get("/api/v1/containers/{name}/engine")
    def container_engine(name: str) -> dict[str, Any]:
        return _container_op(lambda: _local_control().engine_health(name))

    @app.get("/api/v1/containers/{name}/logs/stream")
    async def container_logs_stream(name: str, tail: int = Query(default=200, ge=1, le=5000)):
        return _stream_logs_response(_local_control(), name, tail)

    @app.get("/api/v1/system/df")
    def system_df() -> dict[str, Any]:
        return _system_df_cached(_local_control(), "local")

    @app.get("/api/v1/system/networks")
    def system_networks() -> dict[str, Any]:
        return {"networks": _container_op(lambda: _local_control().list_networks())}

    @app.post("/api/v1/containers/{name}/settings")
    def container_settings(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_settings(_local_control(), name, payload, source="local")

    @app.post("/api/v1/containers/{name}/network")
    def container_network(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_network(_local_control(), name, payload, source="local")

    # Host family (a registered host, via SSH) ----------------------------
    @app.get("/api/v1/hosts/{host_id}/containers")
    async def host_containers_list(host_id: str) -> dict[str, Any]:
        control = _host_control(host_id)
        items = _container_op(control.list_managed)
        return {"containers": [c.to_dict() for c in items], "source": "ssh"}

    @app.get("/api/v1/hosts/{host_id}/containers/stats")
    async def host_containers_stats_all(host_id: str) -> dict[str, Any]:
        return {"stats": _container_op(lambda: _host_control(host_id).list_stats())}

    @app.get("/api/v1/hosts/{host_id}/containers/{name}")
    async def host_container_inspect(host_id: str, name: str) -> dict[str, Any]:
        return _container_op(lambda: _host_control(host_id).inspect(name))

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/logs")
    async def host_container_logs(host_id: str, name: str, tail: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
        return {"container": name, "logs": _container_op(lambda: _host_control(host_id).logs(name, tail=tail))}

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/stats")
    async def host_container_stats(host_id: str, name: str) -> dict[str, Any]:
        return {"container": name, "stats": _container_op(lambda: _host_control(host_id).stats(name))}

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/action")
    async def host_container_action(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_action(_host_control(host_id), name, payload, source=host_id)

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/exec")
    async def host_container_exec(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_exec(_host_control(host_id), name, payload, source=host_id)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/top")
    async def host_container_top(host_id: str, name: str) -> dict[str, Any]:
        return {"container": name, **_container_op(lambda: _host_control(host_id).top(name))}

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/changes")
    async def host_container_changes(host_id: str, name: str) -> dict[str, Any]:
        return {"container": name, "changes": _container_op(lambda: _host_control(host_id).changes(name))}

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/fs")
    async def host_container_fs(host_id: str, name: str, path: str = Query(default="/")) -> dict[str, Any]:
        return _do_fs(_host_control(host_id), name, path)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/file")
    async def host_container_file(host_id: str, name: str, path: str = Query(...),
                                  max_bytes: int = Query(default=65536, ge=1, le=5_000_000)) -> dict[str, Any]:
        return _do_file(_host_control(host_id), name, path, max_bytes)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/update-plan")
    async def host_container_update_plan(host_id: str, name: str) -> dict[str, Any]:
        return _do_update_plan(_host_control(host_id), name, host_id)

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/update-mode")
    async def host_container_set_update_mode(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_set_update_mode(_host_control(host_id), name, host_id, payload)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/sndr-state")
    async def host_container_sndr_state(host_id: str, name: str) -> dict[str, Any]:
        return _sndr_state_cached(_host_control(host_id), name, f"host:{host_id}/{name}")

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/pull")
    async def host_container_pull(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_pull(_host_control(host_id), name, payload, source=host_id)

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/recreate")
    async def host_container_recreate(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_recreate(_host_control(host_id), name, host_id, payload)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/scan")
    async def host_container_scan(host_id: str, name: str) -> dict[str, Any]:
        return _container_op(lambda: _host_control(host_id).scan_image(name))

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/source")
    async def host_container_source(host_id: str, name: str) -> dict[str, Any]:
        return _do_source(_host_control(host_id), name)

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/engine")
    async def host_container_engine(host_id: str, name: str) -> dict[str, Any]:
        return _container_op(lambda: _host_control(host_id).engine_health(name))

    @app.get("/api/v1/hosts/{host_id}/containers/{name}/logs/stream")
    async def host_container_logs_stream(host_id: str, name: str, tail: int = Query(default=200, ge=1, le=5000)):
        return _stream_logs_response(_host_control(host_id), name, tail)

    @app.get("/api/v1/hosts/{host_id}/system/df")
    async def host_system_df(host_id: str) -> dict[str, Any]:
        return _system_df_cached(_host_control(host_id), f"host:{host_id}")

    @app.get("/api/v1/hosts/{host_id}/system/networks")
    async def host_system_networks(host_id: str) -> dict[str, Any]:
        return {"networks": _container_op(lambda: _host_control(host_id).list_networks())}

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/settings")
    async def host_container_settings(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_settings(_host_control(host_id), name, payload, source=host_id)

    @app.post("/api/v1/hosts/{host_id}/containers/{name}/network")
    async def host_container_network(host_id: str, name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return _do_network(_host_control(host_id), name, payload, source=host_id)

    # ─── Alerts (engine health → Telegram) ───────────────────────────────
    @app.get("/api/v1/alerts/config")
    async def alerts_config_get() -> dict[str, Any]:
        from . import notify
        return notify.get_config()

    @app.post("/api/v1/alerts/config")
    async def alerts_config_set(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        # Storing a bot token + changing daemon behavior → gate behind apply.
        if not apply_on:
            raise HTTPException(status_code=403, detail="apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1")
        from . import notify
        return notify.set_config(
            enabled=payload.get("enabled"),
            chat_id=payload.get("chat_id"),
            bot_token=payload.get("bot_token"),
        )

    @app.post("/api/v1/alerts/test")
    async def alerts_test(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        if not apply_on:
            raise HTTPException(status_code=403, detail="apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1")
        from . import notify
        return notify.send("✅ SNDR test alert — notifications are working.")

    @app.get("/api/v1/operations")
    async def operations_list() -> dict[str, Any]:
        """Curated project maintenance/diagnostic operations (server allowlist)."""
        from . import operations

        return {"operations": operations.list_operations(), "apply_enabled": apply_on}

    @app.post("/api/v1/operations/run")
    async def operations_run(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Run a curated operation by id. The command is server-defined (no
        injection). Executes live with --enable-apply, else returns a dry-run."""
        from . import operations

        op_id = str(payload.get("operation", "")).strip()
        try:
            job = operations.run_operation(op_id, apply_on=apply_on)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown operation: {op_id}")
        return _dataclass_payload(job)

    @app.get("/api/v1/caveats")
    async def caveats_list() -> dict[str, Any]:
        """Runtime caveats registry — known host-condition issues — each
        evaluated against the live host so the GUI can flag which ones fire.
        Read-only; host probe is best-effort (caveats still listed if it fails)."""
        try:
            from sndr.caveats import KNOWN_CAVEATS
        except Exception as exc:  # noqa: BLE001
            return {"caveats": [], "total": 0, "triggered_count": 0,
                    "host_facts_available": False, "facts_error": str(exc)}

        facts: dict[str, Any] | None = None
        facts_error: str | None = None
        try:
            from sndr.deps.checkers import inspect_host
            from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS
            facts = inspect_host().to_dict()
            pin = facts.get("vllm", {}).get("version")
            facts["vllm_pin_in_allowlist"] = (pin in KNOWN_GOOD_VLLM_PINS) if pin else None
        except Exception as exc:  # noqa: BLE001 - host probe best-effort
            facts_error = str(exc)

        items: list[dict[str, Any]] = []
        for c in KNOWN_CAVEATS:
            triggered = c.matches(facts) if facts is not None else None
            items.append({
                "id": c.id, "severity": c.severity, "title": c.title,
                "message": c.message, "docs_url": c.docs_url,
                "triggered": triggered,
            })
        sev_order = {"error": 0, "warning": 1, "info": 2}
        items.sort(key=lambda x: (
            0 if x["triggered"] else 1,
            sev_order.get(x["severity"], 9),
            x["id"],
        ))
        return {
            "caveats": items,
            "total": len(items),
            "triggered_count": sum(1 for x in items if x["triggered"]),
            "host_facts_available": facts is not None,
            "facts_error": facts_error,
        }

    @app.get("/api/v1/config-keys")
    async def config_keys_list() -> dict[str, Any]:
        """Canonical config / env-key glossary (GENESIS_ENABLE_* flags + V1/V2
        config keys + policy keys) with provenance — operator reference. Read-only."""
        from sndr.cli.legacy.config_keys import load_canonical_registry

        canon = load_canonical_registry()
        by_source: dict[str, int] = {}
        for meta in canon.values():
            src = str(meta.get("source", "unknown"))
            by_source[src] = by_source.get(src, 0) + 1
        return {"keys": canon, "total": len(canon), "by_source": by_source}

    @app.get("/api/v1/traces")
    async def trace_catalog_list() -> dict[str, Any]:
        """Diagnostic trace catalog — every per-patch debug trace, the
        container path where it lands, the emitting patch, and the env var
        that enables it. Read-only operator reference (mirrors `sndr trace
        list`). Top-level path so it never collides with /patches/{id}."""
        from dataclasses import asdict

        from sndr.observability.trace_catalog import (
            TRACE_CATALOG,
            TRACE_CATEGORIES,
        )

        traces = [asdict(t) for t in TRACE_CATALOG]
        by_category: dict[str, int] = {}
        for t in traces:
            by_category[t["category"]] = by_category.get(t["category"], 0) + 1
        return {
            "traces": traces,
            "categories": list(TRACE_CATEGORIES),
            "by_category": by_category,
            "total": len(traces),
        }

    @app.get("/api/v1/doctor")
    async def doctor() -> dict[str, Any]:
        return _dataclass_payload(collect_doctor_report())

    @app.get("/api/v1/environment")
    async def environment() -> dict[str, Any]:
        return _dataclass_payload(collect_environment_report())

    @app.get("/api/v1/services/plan")
    async def services_plan(
        preset_id: str = Query(...),
        action: str = "status",
        runtime_target: str = "docker_compose",
        host: str = "127.0.0.1",
    ) -> dict[str, Any]:
        try:
            return _dataclass_payload(
                build_service_plan(
                    preset_id=preset_id,
                    action=action,
                    runtime_target=runtime_target,
                    host=host,
                )
            )
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/services/apply")
    async def services_apply(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Apply a lifecycle action.

        Dry-run by default. When apply is enabled, read-only actions
        (status/logs) execute, and mutating actions require ``confirm: true``.
        """
        try:
            preset_id = str(payload["preset_id"])
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc
        action = payload.get("action", "status")
        runtime_target = payload.get("runtime_target", "docker_compose")
        host = payload.get("host", "127.0.0.1")
        try:
            if apply_on:
                job = execute_service_action(
                    preset_id=preset_id,
                    action=action,
                    runtime_target=runtime_target,
                    host=host,
                    transport=str(payload.get("transport", "local")),
                    ssh_target=str(payload.get("ssh_target", "")),
                    confirm=bool(payload.get("confirm", False)),
                    enabled=apply_on,
                )
            else:
                job = apply_service_action(
                    preset_id=preset_id,
                    action=action,
                    runtime_target=runtime_target,
                    host=host,
                )
        except ConfirmationRequiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApplyDisabledError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except PresetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _dataclass_payload(job)

    def _resolve_host_api_key(profile) -> Optional[str]:
        """Return a host's engine key from the encrypted secrets store.

        Legacy rows that still carry a plaintext ``api_key`` on the profile are
        migrated on first read: the key is lifted into the secrets store and the
        on-disk profile is rewritten without it (``host_profile_payload`` /
        ``upsert`` already drop the field). If the secrets backend is
        unavailable the migration is skipped and the legacy value is still used,
        so a key-protected engine never silently loses auth."""
        from . import secrets_store

        sid = f"apikey:{profile.id}"
        try:
            stored = secrets_store.get_secret(sid)
        except Exception:
            stored = None
        if stored:
            return stored
        legacy = (getattr(profile, "api_key", "") or "").strip()
        if legacy:
            try:
                secrets_store.set_secret(sid, legacy)
                upsert_host_profile(host_profile_payload(profile))  # strip plaintext from disk
            except Exception:
                pass  # secrets unavailable — leave legacy in place, still usable
            return legacy
        return None

    def _engine_key_for(host_id: Optional[str], explicit: Optional[str]) -> Optional[str]:
        """Resolve the engine key for a request: an explicit header wins, else
        the stored secret for ``host_id``. The raw key never leaves the daemon."""
        if explicit:
            return explicit
        if host_id:
            prof = next((p for p in list_host_profiles() if p.id == host_id), None)
            if prof is not None:
                return _resolve_host_api_key(prof)
        return None

    def _augment_host(profile, payload: dict[str, Any]) -> dict[str, Any]:
        """Add boolean presence flags for the stored SSH password and engine API
        key — never the values themselves."""
        from . import secrets_store

        try:
            payload["has_ssh_password"] = secrets_store.has_secret(f"ssh:{payload.get('id')}")
        except Exception:
            payload["has_ssh_password"] = False
        try:
            payload["has_api_key"] = bool(_resolve_host_api_key(profile))
        except Exception:
            payload["has_api_key"] = bool((getattr(profile, "api_key", "") or "").strip())
        return payload

    @app.get("/api/v1/hosts")
    async def hosts_list() -> dict[str, Any]:
        return {"hosts": [_augment_host(host, host_profile_payload(host)) for host in list_host_profiles()]}

    @app.post("/api/v1/hosts")
    async def hosts_upsert(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        from . import secrets_store

        try:
            profile = upsert_host_profile(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # An inline engine key in the form body is routed into the encrypted
        # secrets store (never persisted on the profile). Empty string clears it.
        if "api_key" in payload:
            raw = str(payload.get("api_key") or "").strip()
            try:
                if raw:
                    secrets_store.set_secret(f"apikey:{profile.id}", raw)
                else:
                    secrets_store.delete_secret(f"apikey:{profile.id}")
            except Exception:
                pass
        return _augment_host(profile, host_profile_payload(profile))

    @app.delete("/api/v1/hosts/{host_id}")
    async def hosts_delete(host_id: str) -> dict[str, Any]:
        from . import secrets_store

        for sid in (f"ssh:{host_id}", f"apikey:{host_id}"):  # tidy up stored secrets
            try:
                secrets_store.delete_secret(sid)
            except Exception:
                pass
        return {"deleted": delete_host_profile(host_id)}

    @app.post("/api/v1/hosts/ssh-check")
    def hosts_ssh_check(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Test SSH auth + SFTP to a host (read-only — runs no operator command).

        An inline password is used for this check and, when a ``host_id`` is
        given, persisted encrypted in the secrets store so later connections
        reuse it. ``forget_password`` clears it. The password is never echoed
        back."""
        from . import secrets_store, ssh_client

        host_id = str(payload.get("host_id") or "").strip()
        secret_id = f"ssh:{host_id}" if host_id else None
        auth = str(payload.get("auth_method") or "agent").lower()
        password = payload.get("password")

        if payload.get("forget_password") and secret_id:
            secrets_store.delete_secret(secret_id)
            return {"available": ssh_client.available(), "ssh_ok": False, "sftp_ok": False,
                    "latency_ms": None, "banner": None, "uname": None, "error": None, "forgot": True}
        if auth == "password" and password and secret_id:
            try:
                secrets_store.set_secret(secret_id, str(password))
            except Exception:
                pass
        target = {
            "host": payload.get("host"),
            "port": payload.get("ssh_port") or payload.get("port") or 22,
            "user": payload.get("user"),
            "auth_method": auth,
            "key_path": payload.get("key_path"),
            "password": password,        # inline wins for this check
            "secret_id": secret_id,      # fallback to the stored one
        }
        return ssh_client.check_connectivity(target)

    @app.post("/api/v1/hosts/fetch-api-key")
    def hosts_fetch_api_key(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Auto-discover a host's engine API key over SSH and store it on the
        profile (read-only discovery: reads container env / launch scripts)."""
        from . import ssh_client

        host_id = str(payload.get("host_id") or "").strip()
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        target = {
            "host": profile.host,
            "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent",
            "key_path": profile.ssh_key_path,
            "secret_id": f"ssh:{host_id}",
        }
        containers = tuple(c for c in [str(payload.get("container") or "").strip()] if c)
        result = ssh_client.discover_api_key(target, containers=containers)
        if result.get("found") and result.get("key"):
            from . import secrets_store

            key = str(result["key"])
            try:
                secrets_store.set_secret(f"apikey:{host_id}", key)  # encrypted at rest
            except Exception:
                pass
            # Mask the value in the response — it is stored encrypted; the GUI
            # re-fetches the host list (has_api_key=true) to pick it up.
            result["key_masked"] = (key[:3] + "…" + key[-2:]) if len(key) > 6 else "set"
            result.pop("key", None)
        return result

    @app.post("/api/v1/hosts/discover")
    def hosts_discover(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Auto-discover what a host runs (vLLM containers + ports + GPUs) over
        SSH, probe each engine for version/models, and set the profile's engine
        port to the discovered one — so the operator never hunts for the port."""
        from . import ssh_client

        host_id = str(payload.get("host_id") or "").strip()
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        target = {
            "host": profile.host,
            "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent",
            "key_path": profile.ssh_key_path,
            "secret_id": f"ssh:{host_id}",
        }
        disco = ssh_client.discover_host(target)
        engine_key = _resolve_host_api_key(profile)  # from the encrypted secrets store
        # Enrich each engine with a live probe (version + served models). Probe
        # every published port (API-first) and keep the first that answers
        # /health — so a custom or metrics-first port mapping can't make a
        # running engine look unreachable.
        for eng in disco.get("engines", []):
            candidates = eng.get("host_ports") or ([eng["host_port"]] if eng.get("host_port") else [])
            best = None
            for port in candidates:
                pr = engine_client.probe_host(profile.host, port, api_key=engine_key or None)
                if pr.get("reachable"):
                    best = pr
                    eng["host_port"] = port  # pin to the port that actually responded
                    break
                best = best or pr
            if best is None:
                continue
            eng["reachable"] = best.get("reachable", False)
            eng["version"] = best.get("version")
            eng["models"] = best.get("models", [])
        # Auto-set the profile's engine port to the first reachable engine if the
        # current one isn't among what we found.
        found_ports = [e["host_port"] for e in disco.get("engines", []) if e.get("host_port")]
        reachable_ports = [e["host_port"] for e in disco.get("engines", []) if e.get("reachable")]
        chosen = None
        # Persist the discovered hardware summary on the profile (single source
        # the Planner reads), and auto-set the engine port.
        gpus = disco.get("gpus", [])
        patch: dict[str, Any] = {}
        if found_ports and profile.engine_port not in found_ports:
            chosen = (reachable_ports or found_ports)[0]
            patch["engine_port"] = chosen
        if gpus:
            try:
                patch["gpu_vram_mib"] = int(gpus[0].get("memory_total_mib") or 0)
            except (TypeError, ValueError):
                pass
            patch["gpu_name"] = gpus[0].get("name", "")
            patch["gpu_arch"] = gpus[0].get("arch", "")
            patch["gpus"] = len(gpus)
            if disco.get("interconnect"):
                patch["interconnect"] = disco["interconnect"].get("worst_link", "")
        if patch:
            upsert_host_profile({**host_profile_payload(profile), **patch})
        disco["engine_port_set"] = chosen
        return disco

    @app.get("/api/v1/fleet/overview")
    def fleet_overview() -> dict[str, Any]:
        """One read-only summary across all registered engine hosts — fans out
        over SSH concurrently (discover + live engine probe) so the operator
        sees the whole fleet at a glance (status, model, vLLM version, GPUs,
        live patch count) and drills into a host's card for detail."""
        from . import fleet, ssh_client

        rows = fleet.collect_fleet_overview(
            list(list_host_profiles()),
            discover=ssh_client.discover_host,
            probe=engine_client.probe_host,
            resolve_key=_resolve_host_api_key,
        )
        return {"hosts": rows}

    @app.post("/api/v1/hosts/model-config")
    def hosts_model_config(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Read the real model architecture (config.json + exact weight size) off
        a host's running engine, so the fit calculator uses true dims."""
        from . import ssh_client

        host_id = str(payload.get("host_id") or "").strip()
        container = str(payload.get("container") or "").strip()
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        if not container:
            disco = ssh_client.discover_host({
                "host": profile.host, "port": profile.ssh_port or 22,
                "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
                "auth_method": profile.ssh_auth or "agent", "key_path": profile.ssh_key_path, "secret_id": f"ssh:{host_id}",
            })
            engines = disco.get("engines", [])
            if not engines:
                return {"ok": False, "error": disco.get("error") or "no vLLM container found to read a model from"}
            container = engines[0]["container"]
        target = {
            "host": profile.host, "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent", "key_path": profile.ssh_key_path, "secret_id": f"ssh:{host_id}",
        }
        out = ssh_client.read_model_config(target, container=container)
        out["container"] = container
        return out

    @app.post("/api/v1/hosts/sndr-state")
    def hosts_sndr_state(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Read a host's own sndr_core management identity from inside its running
        container over SSH (read-only): patcher version, vLLM build, builtin config
        count and patch-registry size. The 'light Path B' — a host's management
        state without standing up a daemon on it."""
        from . import ssh_client

        host_id = str(payload.get("host_id") or "").strip()
        container = str(payload.get("container") or "").strip() or None
        profile = next((p for p in list_host_profiles() if p.id == host_id), None)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"unknown host: {host_id}")
        target = {
            "host": profile.host, "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent", "key_path": profile.ssh_key_path, "secret_id": f"ssh:{host_id}",
        }
        return ssh_client.read_sndr_state(target, container=container)

    @app.websocket("/api/v1/hosts/{host_id}/terminal")
    async def host_terminal(websocket: WebSocket, host_id: str):
        """Interactive PTY shell to a host over SSH (xterm.js front end).

        Full remote shell — gated hard behind ``SNDR_ENABLE_APPLY``. Protocol:
        client→server JSON ``{"type":"input","data":...}`` / ``{"type":"resize",
        "cols","rows"}``; server→client raw output as binary frames plus
        ``{"type":"ready|error",...}`` JSON.
        """
        import asyncio
        import json as _json

        from starlette.websockets import WebSocketDisconnect

        from . import ssh_client

        await websocket.accept()
        profiles = list_host_profiles()
        gate = terminal_gate(apply_on, ssh_client.available(), {p.id for p in profiles}, host_id)
        if gate is not None:
            await websocket.send_json(gate)
            await websocket.close()
            return
        profile = next(p for p in profiles if p.id == host_id)

        target = {
            "host": profile.host,
            "port": profile.ssh_port or 22,
            "user": profile.ssh_user or (profile.ssh_target.split("@", 1)[0] if "@" in profile.ssh_target else None),
            "auth_method": profile.ssh_auth or "agent",
            "key_path": profile.ssh_key_path,
            "secret_id": f"ssh:{host_id}",
        }
        try:
            client, chan = await asyncio.to_thread(ssh_client.open_shell, target)
        except Exception as exc:  # auth / network
            await websocket.send_json({"type": "error", "data": f"SSH failed: {exc}"})
            await websocket.close()
            return

        await websocket.send_json({"type": "ready", "data": f"{profile.label} — {profile.host}"})
        stop = asyncio.Event()

        async def pump_out():
            try:
                while not stop.is_set():
                    data = ssh_client.read_nonblocking(chan)
                    if data is None:
                        await asyncio.sleep(0.02)
                        continue
                    if data == b"":  # remote shell closed
                        break
                    await websocket.send_bytes(data)
            except Exception:
                pass
            finally:
                stop.set()

        out_task = asyncio.create_task(pump_out())
        try:
            while not stop.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text is not None:
                    try:
                        obj = _json.loads(text)
                    except Exception:
                        continue
                    if obj.get("type") == "input":
                        chan.send(str(obj.get("data", "")))
                    elif obj.get("type") == "resize":
                        try:
                            chan.resize_pty(width=int(obj.get("cols") or 120), height=int(obj.get("rows") or 32))
                        except Exception:
                            pass
                elif msg.get("bytes") is not None:
                    chan.send(msg["bytes"])
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            stop.set()
            out_task.cancel()
            for closer in (chan.close, client.close):
                try:
                    closer()
                except Exception:
                    pass

    @app.get("/api/v1/hosts/probe")
    def hosts_probe(host: str = Query(...), port: int = 8000,
                    host_id: Optional[str] = Query(default=None),
                    x_engine_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        """Live reachability + engine version probe for a host:port (read-only).

        The engine key is resolved server-side from the host's encrypted secret
        when ``host_id`` is given (an explicit ``X-Engine-Api-Key`` header still
        wins) so the raw key never round-trips through the browser."""
        import time as _t

        from . import reliability
        result = engine_client.probe_host(host, port, api_key=_engine_key_for(host_id, x_engine_api_key))
        reliability.TRACKER.record(host_id or host, bool(result.get("reachable")), now=_t.time())
        return result

    @app.get("/api/v1/hosts/reliability")
    def hosts_reliability() -> dict[str, Any]:
        """Per-host reachability uptime %, sample history and breaker state."""
        import time as _t

        from . import reliability
        return reliability.TRACKER.snapshot_all(now=_t.time())

    @app.get("/api/v1/host/inventory")
    async def host_inventory_route() -> dict[str, Any]:
        """Full inventory of the daemon host — OS / Python / Docker / NVIDIA / vLLM."""
        from . import deployment

        return deployment.host_inventory()

    @app.get("/api/v1/host/gpu")
    async def host_gpu_route() -> dict[str, Any]:
        """Rich live GPU + hardware telemetry for the daemon host (nvidia-smi)."""
        from . import gpu_telemetry

        return _dataclass_payload(gpu_telemetry.collect_local())

    @app.get("/api/v1/hosts/{host_id}/gpu")
    async def host_gpu_remote_route(host_id: str) -> dict[str, Any]:
        """Rich live GPU + hardware telemetry for a registered host (over SSH)."""
        from . import gpu_telemetry

        _profile, target = _ssh_target_for(host_id)
        return _dataclass_payload(gpu_telemetry.collect_remote(target))

    # ── Kubernetes mode (read-only, P1) — degrades gracefully when no cluster ──
    @app.get("/api/v1/k8s/status")
    def k8s_status_route() -> dict[str, Any]:
        """Cluster reachability + version + node/GPU/namespace counts (or a
        structured {available:false, error} when k8s isn't configured)."""
        from . import k8s_client
        return k8s_client.cluster_status()

    @app.get("/api/v1/k8s/nodes")
    def k8s_nodes_route() -> dict[str, Any]:
        """Nodes with GPU capacity/allocatable/requested/free, conditions,
        taints and GPU labels — the GPU-fleet operator's primary view."""
        from . import k8s_client
        return k8s_client.list_nodes()

    @app.get("/api/v1/k8s/pods")
    def k8s_pods_route(namespace: Optional[str] = None) -> dict[str, Any]:
        """Pods (all namespaces or one) — phase, ready, restarts, GPU request,
        node, pending reason. GPU + non-running pods sort first."""
        from . import k8s_client
        return k8s_client.list_pods(namespace=namespace)

    @app.get("/api/v1/k8s/events")
    def k8s_events_route(warnings_only: bool = False) -> dict[str, Any]:
        """Cluster events — surfaces Warning events like FailedScheduling
        'Insufficient nvidia.com/gpu' (why a vLLM pod is stuck pending)."""
        from . import k8s_client
        return k8s_client.list_events(warnings_only=warnings_only)

    @app.get("/api/v1/k8s/kubevirt")
    def k8s_kubevirt_route() -> dict[str, Any]:
        """KubeVirt VMs (VirtualMachineInstances). {installed:false} when the
        KubeVirt CRD isn't present — VMs-as-pods for the Virtualization view."""
        from . import k8s_client
        return k8s_client.list_kubevirt_vms()

    @app.get("/api/v1/proxmox/status")
    def proxmox_status_route() -> dict[str, Any]:
        """Proxmox VE reachability + node/VM/LXC counts (or {available:false,
        error} when Proxmox isn't configured) — mirrors the k8s status shape."""
        from . import proxmox_client
        return proxmox_client.cluster_status()

    @app.get("/api/v1/proxmox/nodes")
    def proxmox_nodes_route() -> dict[str, Any]:
        """Proxmox host nodes with CPU / memory / disk utilization."""
        from . import proxmox_client
        return proxmox_client.list_nodes()

    @app.get("/api/v1/proxmox/guests")
    def proxmox_guests_route() -> dict[str, Any]:
        """Proxmox VMs (qemu) + containers (lxc) with resources, uptime, and the
        SNDR preset they host (via the `sndr-preset-<id>` tag)."""
        from . import proxmox_client
        return proxmox_client.list_guests()

    @app.get("/api/v1/proxmox/guests/{node}/{kind}/{vmid}")
    def proxmox_guest_detail_route(node: str, kind: str, vmid: int) -> dict[str, Any]:
        """Rich detail for one guest: CPU topology, memory, OS, BIOS, boot order,
        passthrough devices (resolved names), disks, networks, guest-agent IPs."""
        from . import proxmox_client
        return proxmox_client.guest_detail(node, kind, vmid)

    @app.get("/api/v1/proxmox/nodes/{node}")
    def proxmox_node_detail_route(node: str) -> dict[str, Any]:
        """Rich detail for a Proxmox node: CPU model/topology, kernel, PVE
        version, load average, swap, root fs and display GPUs present."""
        from . import proxmox_client
        return proxmox_client.node_detail(node)

    @app.get("/api/v1/alerts")
    async def alerts_route() -> dict[str, Any]:
        """Evaluate hardware-threshold rules over the daemon host's live telemetry
        and merge into the shared alert store. The GUI polls this for its bell."""
        import time as _t

        from . import alerts as alerts_mod
        from . import gpu_telemetry

        tele = _dataclass_payload(gpu_telemetry.collect_local())
        host = (tele.get("system") or {}).get("hostname") or "local"
        alerts_mod.STORE.update(alerts_mod.evaluate_hardware(host, tele), now=_t.time())
        return alerts_mod.STORE.snapshot()

    # --- Spec-decode workload routing (same brain as the gateway) -----------
    @app.get("/api/v1/routing/artifacts")
    def routing_artifacts() -> dict[str, Any]:
        """Bench-validated spec-decode profiles + their per-workload economics."""
        from . import routing

        return routing.list_artifacts()

    @app.get("/api/v1/routing/active")
    def routing_active() -> dict[str, Any]:
        """The profile the operator considers live (env override or sole artifact)."""
        from . import routing

        return routing.active_profile()

    @app.post("/api/v1/routing/active")
    def routing_set_active(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Pin the daemon's active routing profile (or clear with a null profile).
        Scopes this daemon's reporting + classify default; persist via the
        SNDR_ACTIVE_PROFILE env for the data-plane gateway."""
        from . import routing

        _audit("routing.set_active", str(payload.get("profile") or ""), "local", {"profile": payload.get("profile")})
        return routing.set_active(payload.get("profile"))

    @app.post("/api/v1/routing/classify")
    def routing_classify(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Classify request signals → profile + accept/fallback + expected TPS delta."""
        from . import routing

        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else payload
        return routing.classify(signals=signals or {}, profile=payload.get("profile"))

    @app.get("/api/v1/jobs")
    async def jobs_list() -> dict[str, Any]:
        return {"jobs": [_dataclass_payload(job) for job in list_jobs()]}

    @app.get("/api/v1/jobs/{job_id}")
    async def jobs_get(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return _dataclass_payload(job)

    @app.get("/api/v1/events/recent")
    async def events_recent(since_seq: int = 0, limit: int = 100) -> dict[str, Any]:
        events = list_events(since_seq=since_seq, limit=limit)
        last = events[-1]["seq"] if events else since_seq
        return {"events": events, "last_seq": last}

    @app.get("/api/v1/events")
    async def events_stream():
        import asyncio
        import json as _json

        from starlette.responses import StreamingResponse

        async def gen():
            # Snapshot first so a freshly-connected client renders immediately.
            snapshot = list_events(since_seq=0, limit=100)
            cursor = snapshot[-1]["seq"] if snapshot else 0
            yield (
                "event: snapshot\n"
                f"data: {_json.dumps({'events': snapshot, 'last_seq': cursor})}\n\n"
            )
            # Then incremental events + heartbeat. Bounded sleep keeps the
            # connection cheap; the client may reconnect at will.
            while True:
                await asyncio.sleep(2)
                fresh = list_events(since_seq=cursor, limit=100)
                for event in fresh:
                    cursor = event["seq"]
                    yield f"event: event\ndata: {_json.dumps(event)}\n\n"
                yield f"event: heartbeat\ndata: {_json.dumps({'cursor': cursor})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/v1/reports/bundle")
    async def reports_bundle(payload: dict = Body(...)) -> dict[str, Any]:
        report_type = str(payload.get("report_type") or "catalog")
        preset_id = str(payload.get("preset_id") or "")
        redact = bool(payload.get("redact", True))
        try:
            result = generate_report_bundle(
                report_type=report_type, preset_id=preset_id, redact=redact
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # keep CORS headers on failure, no crash
            raise HTTPException(
                status_code=500, detail=f"Report generation failed: {exc}"
            ) from exc
        record_event(
            "report",
            f"report bundle generated: {result.bundle_id} ({report_type})",
            {"bundle_id": result.bundle_id, "report_type": report_type, "redacted": redact},
        )
        return _dataclass_payload(result)

    @app.get("/api/v1/reports/types")
    async def reports_types() -> dict[str, Any]:
        return {"types": list(REPORT_TYPES)}

    @app.post("/api/v1/bench/run")
    async def bench_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Queue a benchmark run as a dry-run job (records the exact commands)."""
        preset_id = str(payload.get("preset_id") or "")
        if not preset_id:
            raise HTTPException(status_code=400, detail="Missing field: preset_id")
        profile = str(payload.get("profile") or "quick")
        ctx = str(payload.get("ctx") or "8k")
        job = create_dry_run_job(
            kind="bench.run",
            title=f"bench {profile} {preset_id}",
            summary={"preset_id": preset_id, "profile": profile, "ctx": ctx},
            steps=[
                ("Warmup", f"sndr bench warmup --preset {preset_id}"),
                ("Run", f"sndr bench run --preset {preset_id} --{profile} --ctx {ctx}"),
                ("Attach", "sndr evidence attach-bench --release-check"),
            ],
            cli_mirror=[f"sndr bench run --preset {preset_id} --{profile} --ctx {ctx}"],
            note="Benchmark queued as a dry-run job — copy commands to run on the rig.",
        )
        return _dataclass_payload(job)

    @app.post("/api/v1/evidence/attach")
    async def evidence_attach(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Queue an evidence-collect/attach action as a dry-run job."""
        preset_id = str(payload.get("preset_id") or "")
        if not preset_id:
            raise HTTPException(status_code=400, detail="Missing field: preset_id")
        job = create_dry_run_job(
            kind="evidence.attach",
            title=f"evidence attach {preset_id}",
            summary={"preset_id": preset_id},
            steps=[
                ("Collect", f"sndr evidence collect --preset {preset_id}"),
                ("Attach", "sndr proof attach --release-check"),
            ],
            cli_mirror=[f"sndr evidence collect --preset {preset_id}"],
            note="Evidence action queued as a dry-run job — copy commands to run on the rig.",
        )
        return _dataclass_payload(job)

    @app.get("/api/v1/patches/bundles")
    async def patches_bundles() -> dict[str, Any]:
        return {"bundles": [_dataclass_payload(b) for b in list_bundles()]}

    @app.get("/api/v1/patches/bundles/{name}")
    async def patches_bundle_explain(name: str) -> dict[str, Any]:
        bundle = explain_bundle(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail=f"Unknown bundle: {name}")
        return _dataclass_payload(bundle)

    @app.get("/api/v1/patches/diff-upstream")
    async def patches_diff_upstream() -> dict[str, Any]:
        return _dataclass_payload(diff_upstream())

    @app.get("/api/v1/proof/status")
    async def proof_status_endpoint() -> dict[str, Any]:
        """Best-effort proof artifact bucket summary.

        Optional diagnostic: if the proof subsystem is unavailable we return a
        graceful payload instead of a 500 so the GUI can show an honest state.
        """
        try:
            from .patches.proof_status import proof_status

            return {"available": True, **_dataclass_payload(proof_status())}
        except Exception as exc:  # pragma: no cover - environment dependent
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "total": 0,
                "counts": {},
                "patches": [],
            }

    @app.get("/api/v1/patches/{patch_id}/explain")
    async def patches_explain(patch_id: str) -> dict[str, Any]:
        detail = explain_patch(patch_id)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Unknown patch id: {patch_id}",
                    "candidates": suggest_candidates(patch_id),
                },
            )
        return _dataclass_payload(detail)

    # --- Live engine bridge -------------------------------------------------
    # Unlike the rest of the API (static project state), these reach the running
    # vLLM server. Defined as sync handlers so FastAPI offloads the blocking
    # urllib calls to a threadpool instead of stalling the event loop.
    from . import engine_client

    # The engine API key (for key-protected engines, e.g. 35B PROD on :8102) is
    # forwarded via the X-Engine-Api-Key header so it never lands in URLs/logs.
    # Falls back to operator env (SNDR_ENGINE_API_KEY / VLLM_API_KEY).
    @app.get("/api/v1/engine/status")
    def engine_status_route(host: Optional[str] = None, port: Optional[int] = None,
                            host_id: Optional[str] = None,
                            x_engine_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        return engine_client.engine_status(host, port=port, api_key=_engine_key_for(host_id, x_engine_api_key))

    @app.get("/api/v1/engine/model")
    def engine_model_route(host: Optional[str] = None, port: Optional[int] = None,
                           host_id: Optional[str] = None,
                           x_engine_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        """Detect the running engine's served model(s) and bridge each to the V2
        catalog (capabilities, requirements, pin, presets that run it).

        With no explicit target, auto-discover: the configured/local engine first,
        then any registered host's declared engine endpoint (host + engine_port +
        its stored key) — so the GUI finds a key-protected engine on a remote host
        instead of failing on localhost:8000."""
        from . import engine_model
        if host is None and port is None and host_id is None:
            return engine_model.discover_engine(
                profiles=list_host_profiles(),
                key_for=lambda p: _engine_key_for(p.id, None),
            )
        return engine_model.engine_model_detail(host, port=port, api_key=_engine_key_for(host_id, x_engine_api_key))

    @app.get("/api/v1/engine/metrics")
    def engine_metrics_route(host: Optional[str] = None, port: Optional[int] = None) -> dict[str, Any]:
        return engine_client.engine_metrics(host, port=port)

    @app.post("/api/v1/engine/chat")
    def engine_chat_route(payload: dict[str, Any] = Body(default={}),
                          x_engine_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        host = payload.get("host")
        port = payload.get("port")
        api_key = _engine_key_for(payload.get("host_id"), x_engine_api_key)
        try:
            return engine_client.engine_chat(payload, host=host, port=port, api_key=api_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except engine_client.EngineError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:  # connection refused / timeout -> engine down
            raise HTTPException(status_code=503, detail=f"Engine unreachable: {exc}")

    @app.post("/api/v1/engine/chat/stream")
    def engine_chat_stream_route(payload: dict[str, Any] = Body(default={}),
                                 x_engine_api_key: Optional[str] = Header(default=None)):
        """Stream a chat completion (ND-JSON) from the running engine."""
        import json as _json

        from starlette.responses import StreamingResponse

        host = payload.get("host")
        port = payload.get("port")
        api_key = _engine_key_for(payload.get("host_id"), x_engine_api_key)

        def generate():
            from . import market_tools

            msgs = payload.get("messages") or []
            last_user = next((m.get("content", "") for m in reversed(msgs)
                              if isinstance(m, dict) and m.get("role") == "user"), "")
            # Always ground the model in the current date — it has no clock, so it
            # otherwise refuses "future"-dated questions or reasons from a stale
            # training cutoff. Cheap and offline; market/web grounding is added below.
            grounding: list[str] = [market_tools.date_grounding()]
            # Live crypto data is injected ALWAYS (not gated on web search) — it only
            # fires when the question names a known ticker, so a price/analysis query
            # is grounded in real CoinGecko figures even with web search off. Without
            # it the model refuses ("no live data") or invents prices from stale
            # training. Independent of the web-search toggle below.
            try:
                mg = market_tools.market_grounding(last_user)
                if mg:
                    grounding.append(mg)
            except Exception:  # noqa: BLE001 - best-effort
                pass
            # Web-search grounding. The explicit GUI toggle forces it; otherwise it
            # AUTO-enables when the question shows it needs live web data (temporal
            # cues, news, an explicit search ask, a recent year — see
            # market_tools.needs_web_search). Searches the live web (aggregator
            # SearXNG, no external API; direct-SearXNG fallback) and adds the results
            # as context so the answer is grounded and cites sources. Best-effort —
            # chat proceeds ungrounded on failure. Opt out of auto via
            # SNDR_WEB_SEARCH_AUTO=0.
            _auto_web = os.environ.get("SNDR_WEB_SEARCH_AUTO", "1").strip().lower() not in ("0", "false", "no", "off")
            do_web = bool(payload.get("web_search")) or (_auto_web and market_tools.needs_web_search(last_user))
            if do_web:
                try:
                    from . import external_clients
                    from urllib.parse import urlparse

                    res = external_clients.web_search(last_user, limit=int(payload.get("web_k") or 6))
                    hits = res.get("results") or []
                    if hits:
                        grounding.append(
                            "Live web search results — use these as your source for current facts and cite the URLs; "
                            "if a figure isn't here, say so rather than guessing:\n" + "\n".join(
                                f"[{i + 1}] ({h.get('url')}) {h.get('title')}: {h.get('snippet')}"
                                for i, h in enumerate(hits)))
                    yield _json.dumps({"sources": [
                        {"id": h.get("url"), "kind": "web", "title": h.get("title") or h.get("url"),
                         "ref": (urlparse(h.get("url") or "").hostname or h.get("url") or ""),
                         "snippet": h.get("snippet") or ""}
                        for h in hits]}) + "\n"
                except Exception as exc:  # noqa: BLE001 - search is best-effort
                    yield _json.dumps({"search_error": str(exc)}) + "\n"
            # Prepend the assembled grounding as ONE leading system message (multiple
            # system messages are rejected by some chat templates; downstream
            # _coalesce_system also merges, but keep it single here too).
            if grounding and msgs:
                payload["messages"] = [{"role": "system", "content": "\n\n".join(grounding)}, *msgs]
            try:
                for chunk in engine_client.stream_chat(payload, host=host, port=port, api_key=api_key):
                    yield chunk + "\n"
            except ValueError as exc:
                yield _json.dumps({"error": str(exc)}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    # --- Ops copilot (read-only tool-calling assistant) ---------------------
    @app.get("/api/v1/copilot/tools")
    def copilot_tools() -> dict[str, Any]:
        """The read-only/dry-run tools the copilot can call (for the UI)."""
        from . import copilot

        return {"tools": copilot.tool_catalog()}

    @app.post("/api/v1/copilot/chat")
    def copilot_chat(payload: dict[str, Any] = Body(default={}),
                     x_engine_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        """Run the tool-calling ops-copilot loop against the engine.

        Read-only by design: the copilot only calls read/dry-run tools and
        returns a reply + tool-call trace + proposed (human-applied) actions. It
        never mutates state, so it works regardless of SNDR_ENABLE_APPLY."""
        from . import copilot

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        host, port, model = payload.get("host"), payload.get("port"), payload.get("model")
        api_key = _engine_key_for(payload.get("host_id"), x_engine_api_key)
        try:
            temperature = float(payload.get("temperature") or 0.2)
        except (TypeError, ValueError):
            temperature = 0.2
        max_steps = max(1, min(8, int(payload.get("max_steps") or 5))) if str(payload.get("max_steps") or "5").isdigit() else 5

        def chat_fn(msgs: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
            return engine_client.chat_raw(msgs, tools=tools, model=model, host=host, port=port,
                                          api_key=api_key, temperature=temperature)

        try:
            return copilot.run_copilot(messages, chat_fn=chat_fn, max_steps=max_steps)
        except engine_client.EngineError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # connection refused / timeout -> engine down
            raise HTTPException(status_code=503, detail=f"Engine unreachable: {exc}")

    # --- Prompt library (operator-managed system-prompt templates) ----------
    @app.get("/api/v1/prompts")
    def list_prompts_route() -> dict[str, Any]:
        from . import prompts_store

        return {"prompts": prompts_store.list_prompts()}

    @app.post("/api/v1/prompts")
    def create_prompt_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from . import prompts_store

        try:
            return prompts_store.create_prompt(str(payload.get("name") or ""), str(payload.get("content") or ""),
                                               title=str(payload.get("title") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/v1/prompts/{pid}")
    def update_prompt_route(pid: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from . import prompts_store

        try:
            return prompts_store.update_prompt(pid, name=payload.get("name"),
                                               content=payload.get("content"), title=payload.get("title"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/v1/prompts/{pid}")
    def delete_prompt_route(pid: str) -> dict[str, Any]:
        from . import prompts_store

        return {"deleted": prompts_store.delete_prompt(pid)}

    # --- Managed declarative tools (the GUI tool manager) -------------------
    @app.get("/api/v1/tools/managed")
    def list_managed_tools_route() -> dict[str, Any]:
        from . import tools_store

        return {"tools": tools_store.list_tools()}

    @app.post("/api/v1/tools/managed")
    def create_managed_tool_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from . import tools_store

        try:
            return tools_store.create_tool(
                str(payload.get("name") or ""), str(payload.get("url") or ""),
                description=str(payload.get("description") or ""), title=str(payload.get("title") or ""),
                method=str(payload.get("method") or "GET"), params=payload.get("params") or [],
                enabled=bool(payload.get("enabled", True)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/v1/tools/managed/{tid}")
    def update_managed_tool_route(tid: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from . import tools_store

        try:
            return tools_store.update_tool(tid, **{k: payload[k] for k in
                ("title", "description", "url", "method", "params", "enabled") if k in payload})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/v1/tools/managed/{tid}")
    def delete_managed_tool_route(tid: str) -> dict[str, Any]:
        from . import tools_store

        return {"deleted": tools_store.delete_tool(tid)}

    # --- KV / VRAM fit calculator -------------------------------------------
    @app.get("/api/v1/calc/models")
    def calc_models() -> dict[str, Any]:
        """Curated, GUI-editable model arch registry for the fit calculator."""
        from dataclasses import asdict

        from . import kv_math

        return {
            "models": {k: asdict(v) for k, v in kv_math.known_models().items()},
            "kv_dtypes": kv_math.KV_DTYPE_BYTES,
        }

    @app.post("/api/v1/calc/kv")
    def calc_kv(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Per-GPU VRAM breakdown, fit verdict, max context — plus a per-dtype
        comparison and a context→VRAM curve for the GUI chart. Optional
        ``measured_total_mib`` calibrates the overhead against a real point."""
        from dataclasses import asdict

        from . import kv_math

        # All numeric fields are operator-supplied — coerce them under one guard
        # so malformed input ("context":"abc") is a clean 400, not a 500.
        try:
            arch = kv_math.arch_from_dict(payload)
            context = int(payload.get("context") or 32768)
            concurrency = int(payload.get("concurrency") or 1)
            tp = int(payload.get("tp") or payload.get("gpu_count") or 1)
            gpu_count = int(payload.get("gpu_count") or tp)
            gpu_vram_mib = int(payload.get("gpu_vram_mib") or 24564)
            util = float(payload.get("util") or 0.90)
            measured = payload.get("measured_total_mib")
            measured_val = float(measured) if measured else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid numeric field: {exc}") from exc
        if context <= 0 or concurrency <= 0 or tp <= 0 or gpu_vram_mib <= 0:
            raise HTTPException(status_code=400, detail="context, concurrency, tp and gpu_vram_mib must be positive")

        kv_name = str(payload.get("kv_dtype") or "fp8")
        kv_bytes = float(kv_math.KV_DTYPE_BYTES.get(kv_name, 1.0))

        overhead = payload.get("overhead_mib")
        if measured_val is not None:
            overhead = kv_math.calibrate_overhead(
                arch, measured_total_mib=measured_val,
                context=context, concurrency=concurrency, tp=tp, kv_bytes=kv_bytes)
        try:
            overhead = float(overhead if overhead is not None else 1500.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid overhead_mib: {exc}") from exc

        common = dict(concurrency=concurrency, tp=tp, gpu_count=gpu_count,
                      gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead)
        result = kv_math.estimate(arch, context=context, kv_bytes=kv_bytes, **common)

        by_dtype = {
            name: kv_math.estimate(arch, context=context, kv_bytes=b, **common)["max_context"]
            for name, b in kv_math.KV_DTYPE_BYTES.items()
        }
        # How max context scales with tensor-parallel width (more GPUs shard both
        # weights and KV) — answers "should I add a GPU?" for capacity planning.
        by_tp = {
            str(t): kv_math.estimate(
                arch, context=context, kv_bytes=kv_bytes,
                concurrency=concurrency, tp=t, gpu_count=t,
                gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead,
            )["max_context"]
            for t in (1, 2, 4, 8)
        }
        # Context → VRAM curve with a weights/KV/overhead breakdown (stacked area).
        ceiling = max(8192, int(result["max_context"] * 1.3) or 65536)
        step = max(1, ceiling // 28)
        curve = []
        for c in range(step, ceiling + 1, step):
            e = kv_math.estimate(arch, context=c, kv_bytes=kv_bytes, **common)
            curve.append({"context": c, "weights_mib": e["weights_per_gpu_mib"], "kv_mib": e["kv_per_gpu_mib"],
                          "overhead_mib": e["overhead_mib"], "total_mib": e["total_per_gpu_mib"], "fits": e["fits"]})

        # Operating envelope: concurrency × context fit grid.
        env_contexts = sorted({int(ceiling * f) for f in (0.08, 0.16, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)} | {context})
        env_conc = [1, 2, 4, 8, 16, 32]
        envelope = {
            "contexts": env_contexts, "concurrencies": env_conc,
            "grid": kv_math.fit_envelope(arch, contexts=env_contexts, concurrencies=env_conc,
                                         kv_bytes=kv_bytes, tp=tp, gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead),
        }
        # Recommendation: highest-fidelity KV that fits the target operating point.
        recommendation = kv_math.recommend(arch, target_context=context, target_concurrency=concurrency, tp=tp,
                                           gpu_vram_mib=gpu_vram_mib, util=util, overhead_mib=overhead)
        # Arch-aware advisory (when a GPU name/cap is supplied).
        arch_advice = None
        if payload.get("gpu_name") or payload.get("compute_cap"):
            from . import gpu_arch

            arch_advice = gpu_arch.classify(name=payload.get("gpu_name"), compute_cap=payload.get("compute_cap"))

        return {"arch": asdict(arch), "kv_dtype": kv_name, "overhead_mib": round(overhead),
                "rig": {"tp": tp, "gpu_count": gpu_count, "gpu_vram_mib": gpu_vram_mib, "util": util},
                "result": result, "by_dtype": by_dtype, "by_tp": by_tp, "curve": curve,
                "envelope": envelope, "recommendation": recommendation, "arch_advice": arch_advice}

    # --- Quality / bench baselines + regression diff ------------------------
    @app.get("/api/v1/baselines")
    def baselines_list() -> dict[str, Any]:
        from . import baselines

        return {"baselines": baselines.list_baselines()}

    @app.get("/api/v1/baselines/trend")
    def baselines_trend(metric: Optional[str] = None, scenario: Optional[str] = None) -> dict[str, Any]:
        """Time-ordered series of one metric across saved baselines (regression trend)."""
        from . import baselines

        return baselines.trend(metric, scenario=scenario)

    @app.post("/api/v1/baselines")
    def baselines_save(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        from . import baselines

        result = payload.get("result")
        if not isinstance(result, dict):
            raise HTTPException(status_code=400, detail="result object required")
        return baselines.save_baseline(result, label=payload.get("label"))

    @app.delete("/api/v1/baselines/{baseline_id}")
    def baselines_delete(baseline_id: str) -> dict[str, Any]:
        from . import baselines

        return {"deleted": baselines.delete_baseline(baseline_id)}

    @app.post("/api/v1/baselines/diff")
    def baselines_diff(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Diff a current result against a saved baseline (id) or an inline one."""
        from . import baselines

        current = payload.get("current")
        if not isinstance(current, dict):
            raise HTTPException(status_code=400, detail="current result required")
        base = payload.get("baseline")
        if base is None and payload.get("baseline_id"):
            rec = baselines.get_baseline(str(payload["baseline_id"]))
            if rec is None:
                raise HTTPException(status_code=404, detail="unknown baseline")
            base = rec["result"]
        if not isinstance(base, dict):
            raise HTTPException(status_code=400, detail="baseline or baseline_id required")
        return baselines.diff_results(current, base, threshold_pct=float(payload.get("threshold_pct") or 5.0))

    # --- Pin-gated self-updater (GUI + sndr_core patcher) -------------------
    @app.get("/api/v1/update/status")
    def update_status_route() -> dict[str, Any]:
        """Read-only: patcher version, patcher-supported vLLM pins, git/GUI state."""
        from . import updater

        return updater.collect_status()

    @app.get("/api/v1/update/check")
    def update_check_route() -> dict[str, Any]:
        """Read-only remote check (git ls-remote) — is a newer commit available?"""
        from . import updater

        return updater.check_remote()

    @app.post("/api/v1/update/plan")
    def update_plan_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Read-only, pin-gated update plan. Runs nothing."""
        from . import updater

        return updater.build_plan(payload.get("target_pin") or None)

    @app.post("/api/v1/update/apply")
    def update_apply_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Apply the local update steps — gated by apply + confirm + pin gate +
        clean tree. The server vLLM-pin step stays manual (pin policy)."""
        from . import updater

        return updater.apply_plan(
            confirm=bool(payload.get("confirm")),
            apply_enabled=apply_on,
            target_pin=payload.get("target_pin") or None,
        )

    @app.get("/api/v1/chat/retrieve")
    def chat_retrieve_route(query: str = "", k: int = 5) -> dict[str, Any]:
        """Read-only project-knowledge retrieval (RAG) for grounded chat.

        Ranks the patch registry, presets and V2 config catalog against the
        query with a stdlib BM25-lite scorer. Never mutates project state.
        """
        from . import chat_rag

        k = max(1, min(int(k), 12))
        return chat_rag.retrieve(query, k=k).as_dict()

    @app.post("/api/v1/chat/retrieve")
    def chat_retrieve_post_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Retrieve across selected sources: project corpus + local notes vaults.

        Body: ``{query, k, project: bool, vaults: [path, ...]}``. Vault dirs
        (Obsidian / markdown / txt) are read-only and bounded.
        """
        from . import chat_rag

        k = max(1, min(int(payload.get("k") or 5), 12))
        include_project = payload.get("project", True) is not False
        vaults = payload.get("vaults") or []
        if not isinstance(vaults, list):
            vaults = []
        vaults = [str(v) for v in vaults if str(v).strip()][:8]
        result = chat_rag.retrieve(
            str(payload.get("query") or ""), k=k,
            include_project=include_project, vaults=tuple(vaults),
        )
        return result.as_dict()

    @app.post("/api/v1/chat/rag/preview")
    def chat_rag_preview_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Validate a notes-vault directory and report how much it indexes."""
        from . import chat_rag

        return chat_rag.preview_vault(str(payload.get("path") or ""))

    @app.post("/api/v1/engine/bench")
    def engine_bench_route(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Run a real micro-benchmark against the running engine (TTFT/TPOT/TPS)."""
        from . import engine_bench

        try:
            return engine_bench.run_bench(payload, host=payload.get("host"))
        except engine_client.EngineError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:  # engine down / unreachable
            raise HTTPException(status_code=503, detail=f"Engine unreachable: {exc}")

    # Serve the built web UI from the daemon itself (integration / packaging).
    # API routes are registered above, so they take precedence over the mount.
    # Absent a build (e.g. unit tests, dev with a separate Vite server), the
    # daemon stays API-only — unchanged behavior.
    static_dir = _resolve_gui_static_dir()
    if static_dir is not None:
        from starlette.staticfiles import StaticFiles

        class _UiStatic(StaticFiles):
            """Serve the SPA with correct caching: ``index.html`` (and any HTML)
            must revalidate so a fresh deploy is picked up immediately, while the
            content-hashed assets are immutable and cached for a year."""

            async def get_response(self, path: str, scope):  # type: ignore[override]
                resp = await super().get_response(path, scope)
                if path.endswith(".html") or path in ("", "."):
                    resp.headers["Cache-Control"] = "no-cache"
                elif "assets/" in path:
                    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                return resp

        app.mount("/", _UiStatic(directory=str(static_dir), html=True), name="ui")

    # Warm the slow read paths off the request thread so the first GUI load is
    # instant: the host-inventory probe (nvidia-smi/docker shell-outs) and the
    # V2 config catalog (reads + parses every config YAML).
    import threading as _threading

    def _warm() -> None:
        from . import config_editor, deployment

        deployment.warm_host_inventory()
        try:
            config_editor.collect_v2_config_catalog()
        except Exception:  # noqa: BLE001 - best-effort warm-up
            pass

    _threading.Thread(target=_warm, daemon=True).start()

    # Background health-watch: alerts when a managed (engine) container goes down.
    # Idempotent (starts once/process); each tick is a no-op unless alerts are
    # enabled AND the local docker socket is reachable (the node case).
    try:
        from . import container_watch
        from sndr.deps import checkers as _checkers

        def _watch_control():
            from . import container_ops as _co
            if not _checkers._docker_socket_present():
                raise RuntimeError("no docker socket")
            return _co.SocketContainerControl()

        container_watch.start_watch(_watch_control)
    except Exception:  # noqa: BLE001 - never block app creation on the watcher
        pass

    return app


def _resolve_gui_static_dir():
    """Locate the built web UI directory, or None if not present.

    Resolution order: ``SNDR_GUI_STATIC`` env → packaged ``web_static`` beside
    this module → repo ``gui/web/dist`` (dev build). Requires an ``index.html``.
    """
    from pathlib import Path

    candidates = []
    env_dir = os.environ.get("SNDR_GUI_STATIC", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    here = Path(__file__).resolve()
    candidates.append(here.parent / "web_static")
    # repo layout: sndr/product_api/legacy/http_app.py -> parents[3] = repo root
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "gui" / "web" / "dist")
    for candidate in candidates:
        try:
            if (candidate / "index.html").is_file():
                return candidate
        except OSError:
            continue
    return None


_SESSION_COOKIE = "sndr_session"
_OAUTH_STATE_COOKIE = "sndr_oauth_state"


def _install_auth(app, *, bind_host: str, apply_on: bool):
    """Install the auth subsystem: build the service, the request guard, and the
    ``/api/v1/auth/*`` endpoints. Returns the resolved :class:`AuthConfig`."""
    from fastapi import Body, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, RedirectResponse

    # ``from __future__ import annotations`` stringizes endpoint annotations;
    # FastAPI resolves them against module globals, so the locally-imported
    # Request/Response types must be visible there for dependency injection.
    globals()["Request"] = Request
    globals()["Response"] = Response

    from .auth import AuthError, AuthService, load_config
    from .auth import oauth as oauth_mod
    from .auth.store import UserStore

    store = UserStore()
    config = load_config(bind_host=bind_host, has_users=store.count() > 0)
    service = AuthService(store, config)

    if config.manage_accounts:
        generated = service.bootstrap(admin_password=os.environ.get("SNDR_ADMIN_PASSWORD") or None)
        if generated is not None:
            admin = config.system_user
            print(  # noqa: T201 - intentional one-time operator notice
                "\n[SNDR auth] No accounts found — created initial admin.\n"
                f"[SNDR auth]   username: {admin}\n"
                f"[SNDR auth]   password: {generated}\n"
                "[SNDR auth] Store this now; it is shown only once. "
                "Change it after first login.\n",
                flush=True,
            )

    open_api_paths = {
        "/api/v1/health",
        "/api/v1/auth/status",
        "/api/v1/auth/login",
        "/api/v1/auth/login/2fa",
        "/api/v1/auth/logout",
    }

    def _extract_token(request: Request) -> str:
        cookie = request.cookies.get(_SESSION_COOKIE, "")
        if cookie:
            return cookie
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return request.headers.get("x-sndr-token", "")

    def _set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            _SESSION_COOKIE,
            token,
            max_age=config.session_ttl,
            httponly=True,
            samesite="lax",
            secure=config.public_base_url.startswith("https"),
            path="/",
        )

    def _current_user(request: Request):
        return getattr(request.state, "user", None)

    def _require_user(request: Request):
        user = _current_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return user

    def _require_admin(request: Request):
        user = _require_user(request)
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator role required.")
        return user

    _MUTATING = {"POST", "PUT", "DELETE", "PATCH"}

    def _csrf_ok(request: Request) -> bool:
        """CSRF defence for cookie-authenticated browser requests.

        Bearer-token clients are immune (the attacker cannot read the token).
        For session-cookie requests we require a same-origin signal — browsers
        always send ``Sec-Fetch-Site`` / ``Origin`` on cross-site POSTs."""
        if not request.cookies.get(_SESSION_COOKIE):
            return True  # no session cookie -> bearer or anonymous, not a CSRF vector
        if request.headers.get("authorization", "").lower().startswith("bearer "):
            return True
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site:
            return fetch_site in {"same-origin", "same-site", "none"}
        origin = request.headers.get("origin")
        if not origin:
            return True  # non-browser client (no Origin) — cookies aren't auto-attached cross-site
        from urllib.parse import urlparse

        return urlparse(origin).netloc == request.headers.get("host", "")

    @app.middleware("http")
    async def _auth_guard(request: Request, call_next):
        request.state.user = None
        token = _extract_token(request)
        if token:
            request.state.user = service.verify_session(token)
            # Fall back to a managed API token (Bearer PAT) for programmatic use.
            if request.state.user is None and token.startswith("sndr_pat_"):
                request.state.user = service.verify_api_token(token)
        path = request.url.path
        # CSRF: cookie-authenticated mutations must be same-origin. OAuth
        # callbacks (cross-site form_post from Apple) are exempted by path.
        if (
            request.method in _MUTATING
            and path.startswith("/api/v1/")
            and not path.startswith("/api/v1/auth/oauth/")
            and not _csrf_ok(request)
        ):
            return JSONResponse({"detail": "Cross-origin request blocked (CSRF)."}, status_code=403)
        if config.enabled and path.startswith("/api/v1/") and request.method != "OPTIONS":
            if path in open_api_paths or path.startswith("/api/v1/auth/oauth/"):
                return await call_next(request)
            if config.legacy_token and token == config.legacy_token:
                return await call_next(request)
            if request.state.user is None:
                return JSONResponse({"detail": "Authentication required."}, status_code=401)
        return await call_next(request)

    @app.get("/api/v1/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        user = _current_user(request)
        return {
            "auth_required": config.enabled,
            "apply_enabled": apply_on,
            "backends": config.backends,
            "oauth_providers": list(config.oauth.keys()),
            "context": {
                "in_container": config.in_container,
                "system_user": config.system_user,
                "pam_enabled": config.pam_enabled,
            },
            "user": user.public_dict() if user else None,
        }

    def _audit(event: str, username: str, detail: dict | None = None) -> None:
        record_event("auth", f"{event}: {username}", {"event": event, "user": username, **(detail or {})})

    @app.post("/api/v1/auth/login")
    async def auth_login(response: Response, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        username = str(payload.get("username", ""))
        result = service.authenticate(username, str(payload.get("password", "")))
        if not result.ok:
            if result.locked:
                _audit("login.locked", username)
                raise HTTPException(status_code=429, detail=result.error or "Too many attempts.")
            _audit("login.failed", username)
            raise HTTPException(status_code=401, detail=result.error or "Login failed.")
        if result.needs_2fa:
            return {"ok": True, "needs_2fa": True, "username": result.user.username}
        _set_session_cookie(response, result.token)
        _audit("login.success", result.user.username)
        return {"ok": True, "needs_2fa": False, "token": result.token, "user": result.user.public_dict()}

    @app.post("/api/v1/auth/login/2fa")
    async def auth_login_2fa(response: Response, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        username = str(payload.get("username", ""))
        result = service.complete_2fa(username, str(payload.get("code", "")))
        if not result.ok:
            if result.locked:
                _audit("login.2fa.locked", username)
                raise HTTPException(status_code=429, detail=result.error or "Too many attempts.")
            _audit("login.2fa.failed", username)
            raise HTTPException(status_code=401, detail=result.error or "Invalid code.")
        _set_session_cookie(response, result.token)
        _audit("login.success", result.user.username, {"twofa": True})
        return {"ok": True, "token": result.token, "user": result.user.public_dict()}

    @app.post("/api/v1/auth/logout")
    async def auth_logout(request: Request, response: Response) -> dict[str, Any]:
        user = _current_user(request)
        if user:
            _audit("logout", user.username)
        response.delete_cookie(_SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.post("/api/v1/auth/sessions/revoke")
    async def auth_revoke_sessions(request: Request, response: Response) -> dict[str, Any]:
        user = _require_user(request)
        service.revoke_sessions(user)
        _audit("sessions.revoked", user.username)
        response.delete_cookie(_SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/v1/auth/me")
    async def auth_me(request: Request) -> dict[str, Any]:
        return _require_user(request).public_dict()

    @app.get("/api/v1/auth/tokens")
    async def auth_tokens_list(request: Request) -> dict[str, Any]:
        _require_user(request)
        return {"tokens": [_dataclass_payload(token) for token in service.list_api_tokens()]}

    @app.post("/api/v1/auth/tokens")
    async def auth_tokens_create(request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        user = _require_user(request)
        label = str(payload.get("label", "")).strip()[:64]
        plaintext, token = service.issue_api_token(label, created_by=user.username)
        _audit("token.issued", user.username, {"id": token.id, "label": token.label})
        # The plaintext is returned exactly once — never persisted in clear.
        return {"token": plaintext, "record": _dataclass_payload(token)}

    @app.delete("/api/v1/auth/tokens/{token_id}")
    async def auth_tokens_revoke(request: Request, token_id: str) -> dict[str, Any]:
        user = _require_user(request)
        revoked = service.revoke_api_token(token_id)
        if revoked:
            _audit("token.revoked", user.username, {"id": token_id})
        return {"revoked": revoked}

    @app.post("/api/v1/auth/password")
    async def auth_password(request: Request, response: Response, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        user = _require_user(request)
        try:
            service.set_password(user, current=str(payload.get("current", "")), new=str(payload.get("new", "")))
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _audit("password.changed", user.username)
        # The password change rotated the token epoch — re-issue this session.
        _set_session_cookie(response, service.issue_session(user))
        return {"ok": True}

    @app.get("/api/v1/auth/users")
    async def auth_users(request: Request) -> dict[str, Any]:
        _require_admin(request)
        return {"users": [u.public_dict() for u in service.store.list_users()]}

    @app.post("/api/v1/auth/users")
    async def auth_create_user(request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        _require_admin(request)
        try:
            user = service.create_user(
                username=str(payload.get("username", "")),
                password=str(payload.get("password", "")),
                role=str(payload.get("role", "operator")),
            )
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _audit("user.created", _current_user(request).username, {"target": user.username, "role": user.role})
        return user.public_dict()

    @app.delete("/api/v1/auth/users/{username}")
    async def auth_delete_user(request: Request, username: str) -> dict[str, Any]:
        acting = _require_admin(request)
        try:
            service.delete_user(username, acting=acting)
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _audit("user.deleted", acting.username, {"target": username})
        return {"ok": True}

    @app.post("/api/v1/auth/2fa/enroll")
    async def auth_2fa_enroll(request: Request) -> dict[str, Any]:
        user = _require_user(request)
        secret, uri = service.enroll_2fa(user)
        return {"secret": secret, "otpauth_uri": uri}

    @app.post("/api/v1/auth/2fa/activate")
    async def auth_2fa_activate(request: Request, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        user = _require_user(request)
        try:
            recovery_codes = service.activate_2fa(user, str(payload.get("code", "")))
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _audit("2fa.enabled", user.username)
        return {"ok": True, "totp_enabled": True, "recovery_codes": recovery_codes}

    @app.post("/api/v1/auth/2fa/recovery")
    async def auth_2fa_recovery(request: Request) -> dict[str, Any]:
        user = _require_user(request)
        try:
            codes = service.regenerate_recovery_codes(user)
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _audit("2fa.recovery.regenerated", user.username)
        return {"ok": True, "recovery_codes": codes}

    @app.post("/api/v1/auth/2fa/disable")
    async def auth_2fa_disable(request: Request) -> dict[str, Any]:
        user = _require_user(request)
        service.disable_2fa(user)
        _audit("2fa.disabled", user.username)
        return {"ok": True, "totp_enabled": False}

    @app.get("/api/v1/auth/oauth/{provider}/login")
    async def auth_oauth_login(provider: str):
        prov = config.oauth.get(provider)
        if prov is None:
            raise HTTPException(status_code=404, detail=f"OAuth provider '{provider}' is not configured.")
        import secrets as _secrets

        nonce = _secrets.token_urlsafe(16)
        state = service.sign_state(f"{provider}:{nonce}")
        url = oauth_mod.authorize_url(prov, config.public_base_url, state=state, nonce=nonce)
        redirect = RedirectResponse(url, status_code=307)
        redirect.set_cookie(
            _OAUTH_STATE_COOKIE, f"{state}|{nonce}", max_age=600, httponly=True, samesite="lax", path="/"
        )
        return redirect

    async def _oauth_callback(provider: str, request: Request):
        prov = config.oauth.get(provider)
        if prov is None:
            raise HTTPException(status_code=404, detail=f"OAuth provider '{provider}' is not configured.")
        if request.method == "POST":
            body = (await request.body()).decode("utf-8")
            import urllib.parse as _url

            form = {k: v[0] for k, v in _url.parse_qs(body).items()}
        else:
            form = dict(request.query_params)
        code, state = form.get("code", ""), form.get("state", "")
        cookie = request.cookies.get(_OAUTH_STATE_COOKIE, "")
        expected_state, _, nonce = cookie.partition("|")
        if not code or not state or state != expected_state or service.unsign_state(state) is None:
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
        try:
            token_response = oauth_mod.exchange_code(prov, config.public_base_url, code)
            identity = oauth_mod.identity_from_token_response(token_response, expected_nonce=nonce)
        except Exception as exc:  # pragma: no cover - network dependent
            raise HTTPException(status_code=502, detail=f"OAuth exchange failed: {exc}")
        if identity is None:
            raise HTTPException(status_code=400, detail="OAuth provider returned no usable identity.")
        user = service.upsert_oauth_user(provider=provider, subject=identity["sub"], email=identity.get("email"))
        token = service.issue_session(user)
        redirect = RedirectResponse("/", status_code=303)
        _set_session_cookie(redirect, token)
        redirect.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
        return redirect

    @app.get("/api/v1/auth/oauth/{provider}/callback")
    async def auth_oauth_callback_get(provider: str, request: Request):
        return await _oauth_callback(provider, request)

    @app.post("/api/v1/auth/oauth/{provider}/callback")
    async def auth_oauth_callback_post(provider: str, request: Request):
        return await _oauth_callback(provider, request)

    return config


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_level: str = "info",
    enable_apply: bool = False,
) -> None:
    """Run the GUI Product API server via uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SNDR GUI API requires uvicorn. Install with: "
            "pip install 'vllm-sndr-core[gui-api]' or pip install uvicorn."
        ) from exc
    # enable_apply OR the env flag enables real execution; default stays OFF.
    apply_on = enable_apply or None
    uvicorn.run(
        create_app(enable_apply=apply_on, bind_host=host), host=host, port=port, log_level=log_level
    )


__all__ = ["DEFAULT_ALLOWED_ORIGINS", "create_app", "run_server"]
