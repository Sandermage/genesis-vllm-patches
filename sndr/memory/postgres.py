# SPDX-License-Identifier: Apache-2.0
"""`PostgresStore` — the production backend (Postgres + pgvector).

Implements the same `MemoryStore` contract as the in-memory reference; the
shared `recall` / `_retention` live in the base class, so this backend only
provides data-access primitives in SQL. Verified by running the identical
contract suite against a live Postgres+pgvector (the `integration` marker /
MEMORY_TEST_DSN).

Vectors are passed as pgvector string literals cast to `::vector`, so the only
runtime dependency is `psycopg` (no numpy, no pgvector-python). Timestamps are
stored as epoch-seconds `double precision` driven by the injected `clock`, so
decay is deterministic and numerically identical to the reference backend.

Connections: a single autocommit connection guarded by a lock — correct and
simple for the store contract. Pooling / async write-back is layered at the
product-API, not here. Identifiers are composed with `psycopg.sql` (no
injection); the schema name additionally must be a plain identifier.
"""
from __future__ import annotations

import contextlib
import re
import threading
import time
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg import sql
from psycopg.types.json import Json

from sndr.memory.model import (
    CO_ACCESS_REL,
    HEBBIAN_ETA,
    HEBBIAN_LAMBDA,
    MemoryNode,
    SearchHit,
)
from sndr.memory.store import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_SYMMETRIC_RELS = ["co_access", "similar_to"]
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _vec_literal(values: Sequence[float], dim: int) -> str:
    """Render a vector as a pgvector literal, zero-padded/truncated to `dim`."""
    v = list(values)[:dim]
    v.extend([0.0] * (dim - len(v)))
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _parse_vec(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in text.strip("[]").split(",") if x != ""]


class PostgresStore(MemoryStore):
    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 1024,
        schema: str = "public",
        clock: Any = time.time,
        ensure: bool = True,
    ) -> None:
        if not _IDENT_RE.match(schema):
            raise ValueError(f"invalid schema identifier: {schema!r}")
        self._dim = dim
        self._schema = schema
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._node = sql.Identifier(schema, "mem_node")
        self._edge = sql.Identifier(schema, "mem_edge")
        # Filtered-ANN tuning: relaxed_order iterative scan keeps owner-scoped
        # HNSW from starving the candidate set (else recall collapses); a higher
        # ef_search trades a little latency for recall. GUCs exist in pgvector
        # >= 0.8; guard for older builds.
        with self._lock, self._conn.cursor() as cur:
            for stmt in (
                "SET hnsw.iterative_scan = 'relaxed_order'",
                "SET hnsw.ef_search = 100",
            ):
                with contextlib.suppress(psycopg.Error):
                    cur.execute(stmt)
        if ensure:
            self.ensure_schema()

    # ── schema ───────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(self._schema)
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {node} ("
                    " id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
                    " owner_id BIGINT NOT NULL,"
                    " kind TEXT NOT NULL,"
                    " content TEXT NOT NULL,"
                    " embedding vector({dim}),"
                    " importance REAL NOT NULL DEFAULT 0,"
                    " strength REAL NOT NULL DEFAULT 1,"
                    " access_count INT NOT NULL DEFAULT 0,"
                    " community_id INT,"
                    " properties JSONB NOT NULL DEFAULT '{{}}',"
                    " created_at DOUBLE PRECISION NOT NULL DEFAULT 0,"
                    " accessed_at DOUBLE PRECISION NOT NULL DEFAULT 0)"
                ).format(node=self._node, dim=sql.Literal(self._dim))
            )
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {edge} ("
                    " src_id BIGINT NOT NULL REFERENCES {node}(id) ON DELETE CASCADE,"
                    " dst_id BIGINT NOT NULL REFERENCES {node}(id) ON DELETE CASCADE,"
                    " rel TEXT NOT NULL,"
                    " weight DOUBLE PRECISION NOT NULL DEFAULT 0,"  # parity w/ Python double
                    " properties JSONB NOT NULL DEFAULT '{{}}',"
                    " valid_at DOUBLE PRECISION NOT NULL DEFAULT 0,"
                    " invalid_at DOUBLE PRECISION,"
                    " PRIMARY KEY (src_id, dst_id, rel))"
                ).format(edge=self._edge, node=self._node)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {ix} ON {node} (owner_id, kind)"
                ).format(ix=sql.Identifier(f"{self._schema}_node_owner"), node=self._node)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {ix} ON {edge} (src_id)"
                ).format(ix=sql.Identifier(f"{self._schema}_edge_src"), edge=self._edge)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {ix} ON {edge} (dst_id)"
                ).format(ix=sql.Identifier(f"{self._schema}_edge_dst"), edge=self._edge)
            )
            # ANN index (the design's main index). HNSW needs dim <= 2000.
            if self._dim <= 2000:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {ix} ON {node}"
                        " USING hnsw (embedding vector_cosine_ops)"
                        " WITH (m = 16, ef_construction = 200)"
                    ).format(ix=sql.Identifier(f"{self._schema}_node_hnsw"), node=self._node)
                )
            # Lexical index for hybrid (keyword/BM25-style) search.
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {ix} ON {node}"
                    " USING gin (to_tsvector('simple', content))"
                ).format(ix=sql.Identifier(f"{self._schema}_node_tsv"), node=self._node)
            )

    def close(self) -> None:
        self._conn.close()

    def _row_to_node(self, r: tuple) -> MemoryNode:
        return MemoryNode(
            id=r[0], owner_id=r[1], kind=r[2], content=r[3],
            embedding=_parse_vec(r[4]), importance=r[5], strength=r[6],
            access_count=r[7], community_id=r[8], properties=r[9] or {},
            created_at=r[10], accessed_at=r[11],
        )

    _COLS = ("id, owner_id, kind, content, embedding::text, importance, strength, "
             "access_count, community_id, properties, created_at, accessed_at")

    # ── nodes ────────────────────────────────────────────────────────────
    def add_node(
        self,
        *,
        owner_id: int,
        kind: str,
        content: str,
        embedding: Sequence[float],
        importance: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> int:
        now = self._clock()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {node} (owner_id, kind, content, embedding,"
                    " importance, properties, created_at, accessed_at)"
                    " VALUES (%s,%s,%s,%s::vector,%s,%s,%s,%s) RETURNING id"
                ).format(node=self._node),
                (owner_id, kind, content, _vec_literal(embedding, self._dim),
                 importance, Json(properties or {}), now, now),
            )
            return cur.fetchone()[0]

    def get_node(self, node_id: int) -> MemoryNode | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT " + self._COLS + " FROM {node} WHERE id=%s").format(
                    node=self._node
                ),
                (node_id,),
            )
            row = cur.fetchone()
        return self._row_to_node(row) if row else None

    def iter_nodes(self, owner_id: int) -> Iterator[MemoryNode]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT " + self._COLS + " FROM {node} WHERE owner_id=%s ORDER BY id"
                ).format(node=self._node),
                (owner_id,),
            )
            rows = cur.fetchall()
        return iter([self._row_to_node(r) for r in rows])

    # ── edges ────────────────────────────────────────────────────────────
    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        rel: str,
        *,
        weight: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        now = self._clock()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {edge} (src_id,dst_id,rel,weight,properties,valid_at)"
                    " VALUES (%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT (src_id,dst_id,rel) DO UPDATE"
                    " SET weight=EXCLUDED.weight, properties=EXCLUDED.properties,"
                    " valid_at=EXCLUDED.valid_at, invalid_at=NULL"
                ).format(edge=self._edge),
                (src_id, dst_id, rel, weight, Json(properties or {}), now),
            )

    def edge_weight(self, src_id: int, dst_id: int, rel: str) -> float:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT weight FROM {edge} WHERE src_id=%s AND dst_id=%s AND rel=%s"
                ).format(edge=self._edge),
                (src_id, dst_id, rel),
            )
            row = cur.fetchone()
        return float(row[0]) if row else 0.0

    # ── recall primitives ────────────────────────────────────────────────
    def search(
        self, *, owner_id: int, query: Sequence[float], limit: int = 15
    ) -> list[SearchHit]:
        qv = _vec_literal(query, self._dim)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT " + self._COLS + ", 1 - (embedding <=> %s::vector) AS score"
                    " FROM {node} WHERE owner_id=%s"
                    " ORDER BY embedding <=> %s::vector LIMIT %s"
                ).format(node=self._node),
                (qv, owner_id, qv, limit),
            )
            rows = cur.fetchall()
        return [SearchHit(node=self._row_to_node(r), score=float(r[12])) for r in rows]

    def keyword_search(
        self, *, owner_id: int, query: str, limit: int = 15
    ) -> list[SearchHit]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT " + self._COLS + ", ts_rank_cd(to_tsvector('simple', content),"
                    " plainto_tsquery('simple', %s)) AS rank FROM {node}"
                    " WHERE owner_id=%s"
                    " AND to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)"
                    " ORDER BY rank DESC LIMIT %s"
                ).format(node=self._node),
                (query, owner_id, query, limit),
            )
            rows = cur.fetchall()
        return [SearchHit(node=self._row_to_node(r), score=float(r[12])) for r in rows]

    def find_by_content(self, *, owner_id: int, content: str) -> int | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT id FROM {node} WHERE owner_id=%s AND content=%s LIMIT 1"
                ).format(node=self._node),
                (owner_id, content),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def neighbors(
        self, node_id: int, *, min_weight: float = 0.0
    ) -> list[tuple[int, str, float]]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT dst_id, rel, weight FROM {edge}"
                    " WHERE src_id=%s AND invalid_at IS NULL AND weight>=%s"
                    " UNION ALL "
                    "SELECT src_id, rel, weight FROM {edge}"
                    " WHERE dst_id=%s AND rel = ANY(%s) AND invalid_at IS NULL"
                    " AND weight>=%s"
                ).format(edge=self._edge),
                (node_id, min_weight, node_id, _SYMMETRIC_RELS, min_weight),
            )
            return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

    def _touch(self, node_ids: Sequence[int], now: float) -> None:
        if not node_ids:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "UPDATE {node} SET access_count=access_count+1, accessed_at=%s"
                    " WHERE id = ANY(%s)"
                ).format(node=self._node),
                (now, list(node_ids)),
            )

    # ── brain mechanics ──────────────────────────────────────────────────
    def reinforce_co_access(self, node_ids: Sequence[int]) -> None:
        uniq = sorted(set(node_ids))
        srcs, dsts = [], []
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                srcs.append(uniq[i])
                dsts.append(uniq[j])
        if not srcs:
            return
        now = self._clock()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {edge} AS e (src_id,dst_id,rel,weight,valid_at)"
                    " SELECT u.a, u.b, %(rel)s, %(eta)s, %(now)s"
                    " FROM unnest(%(srcs)s::bigint[], %(dsts)s::bigint[]) AS u(a,b)"
                    " ON CONFLICT (src_id,dst_id,rel) DO UPDATE"
                    " SET weight = LEAST(1.0, e.weight * %(lam)s + %(eta)s),"
                    " invalid_at = NULL"
                ).format(edge=self._edge),
                {"rel": CO_ACCESS_REL, "eta": HEBBIAN_ETA, "lam": HEBBIAN_LAMBDA,
                 "now": now, "srcs": srcs, "dsts": dsts},
            )

    # ── maintenance (leak-bounding) ──────────────────────────────────────
    def prune(self, *, owner_id: int, max_nodes: int) -> int:
        from sndr.memory.model import EBBINGHAUS_S

        now = self._clock()
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT count(*) FROM {node} WHERE owner_id=%s").format(
                    node=self._node
                ),
                (owner_id,),
            )
            total = cur.fetchone()[0]
            k = total - max_nodes
            if k <= 0:
                return 0
            # Salience identical to the reference: importance + retention +
            # 0.1*access_count; tie-break by id DESC (oldest id evicted first).
            cur.execute(
                sql.SQL(
                    "DELETE FROM {node} WHERE id IN ("
                    " SELECT id FROM {node} WHERE owner_id=%(owner)s"
                    " ORDER BY (importance"
                    " + exp(- GREATEST(0, %(now)s - accessed_at)"
                    "        / (%(s)s * (1 + GREATEST(0, importance))))"
                    " + 0.1 * access_count) ASC, id DESC"
                    " LIMIT %(k)s)"
                ).format(node=self._node),
                {"owner": owner_id, "now": now, "s": EBBINGHAUS_S, "k": k},
            )
            return cur.rowcount

    def set_communities(self, mapping: dict[int, int]) -> None:
        if not mapping:
            return
        ids = list(mapping.keys())
        vals = [mapping[i] for i in ids]
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "UPDATE {node} AS n SET community_id = v.c"
                    " FROM unnest(%(ids)s::bigint[], %(vals)s::int[]) AS v(id, c)"
                    " WHERE n.id = v.id"
                ).format(node=self._node),
                {"ids": ids, "vals": vals},
            )

    def set_importance(self, mapping: dict[int, float]) -> None:
        if not mapping:
            return
        ids = list(mapping.keys())
        vals = [float(mapping[i]) for i in ids]
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "UPDATE {node} AS n SET importance = v.imp"
                    " FROM unnest(%(ids)s::bigint[], %(vals)s::double precision[])"
                    " AS v(id, imp) WHERE n.id = v.id"
                ).format(node=self._node),
                {"ids": ids, "vals": vals},
            )

    def count_nodes(self, owner_id: int | None = None) -> int:
        with self._lock, self._conn.cursor() as cur:
            if owner_id is None:
                cur.execute(
                    sql.SQL("SELECT count(*) FROM {node}").format(node=self._node)
                )
            else:
                cur.execute(
                    sql.SQL("SELECT count(*) FROM {node} WHERE owner_id=%s").format(
                        node=self._node
                    ),
                    (owner_id,),
                )
            return cur.fetchone()[0]

    def count_edges(self) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {edge}").format(edge=self._edge))
            return cur.fetchone()[0]
