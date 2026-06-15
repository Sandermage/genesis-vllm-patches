# SPDX-License-Identifier: Apache-2.0
"""sndr.product_api — FastAPI backend for the Control Center GUI.

Architecture (Phase 5 onward):

  routes/      One file per resource (engines, patches, pins, drift, ...)
  domain/      Business logic invoked by routes
  schemas/     Pydantic models — single source of truth for API contracts
  streaming/   SSE event stream, WebSocket terminal/log streams
  auth/        Authentication subsystem (PAM, OAuth, TOTP, sessions)
  middleware   CORS, request-id, rate limit, auth gate

The HTTP server is created via :func:`create_app`, which composes all routers
and middleware.

During v12.x migration, this package coexists with the legacy
``sndr.product_api.legacy.http_app`` monolith. New endpoints (engine-aware,
pin-aware, drift-aware) live here; legacy endpoints remain in the monolith
until Phase 11 cleanup.
"""
# This package is intentionally lazy — importing sndr.product_api does NOT
# eagerly import FastAPI or routes. Callers explicitly import what they need.
