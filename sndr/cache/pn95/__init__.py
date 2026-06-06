# SPDX-License-Identifier: Apache-2.0
"""PN95 runtime split package — M.4.1 onwards.

The legacy ``vllm.sndr_core.cache._pn95_runtime`` monolith was 3390 LOC
mixing 11 distinct concerns. M.4.1 extracts the env-gates + metrics
concerns into ``pn95.gates`` and ``pn95.metrics``; the legacy module
keeps byte-identical re-exports so existing tests and text-patch
anchors are unaffected.

See ``sndr_private/planning/audits/M4_PN95_SPLIT_R_2026-05-27_RU.md``
for the full decomposition plan.
"""
