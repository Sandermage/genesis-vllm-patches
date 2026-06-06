# SPDX-License-Identifier: Apache-2.0
"""Business logic invoked by HTTP routes.

Routes are thin: parse request → call domain function → render response.
All business logic lives here so it is unit-testable without an HTTP layer.
"""
