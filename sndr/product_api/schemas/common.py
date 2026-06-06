# SPDX-License-Identifier: Apache-2.0
"""Common Pydantic schemas shared across resources."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    """Metadata returned with every response."""

    api_version: str = Field(default="v1")
    request_id: str
    engine: str | None = None
    pin: str | None = None
    timestamp: datetime


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    number: int = Field(ge=1, description="Current page number (1-indexed)")
    size: int = Field(ge=1, le=200, description="Page size")
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class Envelope(BaseModel, Generic[T]):
    """Standard response envelope.

    Every successful response has this shape::

        {
          "data": { ... },
          "meta": { "api_version": "v1", ... }
        }
    """

    data: T
    meta: ResponseMeta


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP errors."""

    type: str = Field(description="Stable URL identifying the problem type")
    title: str = Field(description="Short, human-readable summary")
    status: int = Field(description="HTTP status code")
    detail: str | None = Field(default=None, description="Human-readable explanation")
    instance: str | None = Field(default=None, description="URI of the failing operation")
    extensions: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Envelope", "PaginationMeta", "ProblemDetail", "ResponseMeta"]
