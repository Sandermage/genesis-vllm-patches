# SPDX-License-Identifier: Apache-2.0
"""HTTP routes — thin layer between FastAPI and domain services.

Each route file declares an :class:`APIRouter` named ``router`` that the
main app composer mounts under ``/api/v1/``.

Architecture rule: routes contain no business logic. They:
  1. Parse and validate the request (FastAPI + Pydantic do most of this).
  2. Call a domain service.
  3. Render the response with :class:`Envelope`.

This keeps the HTTP layer testable in isolation and lets the domain layer
be reused by the CLI, the background worker, and future API versions.
"""
