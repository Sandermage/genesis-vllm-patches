# SPDX-License-Identifier: Apache-2.0
"""Read-only project-knowledge retrieval (RAG) for the GUI chat.

The chat can ground its answers in *this project's* own state — the patch
registry, the preset catalog and the V2 config catalog — so the operator can
ask "what does PN95 do?", "which preset for long context on 2x A5000?" or
"which patches touch the KV cache?" and get an answer grounded in real
metadata instead of the model's stale priors.

Design constraints:

* **Read-only.** Every source is pulled through the same import-safe
  product-API helpers the Patches / Presets / Configs views already use.
  Nothing here mutates the registry or any config file.
* **No heavy deps.** Ranking is a pure-stdlib BM25-lite scorer — no
  embeddings, no network, works on the air-gapped server.
* **Torch-less collection.** Every source import is lazy and wrapped so a
  single failing source never breaks retrieval.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Common words that add noise to short technical queries.
_STOP = frozenset(
    "the a an of for to in on and or is are be with by at as it this that "
    "what which who how do does can i we our your my".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOP]


@dataclass(frozen=True)
class Doc:
    """One retrievable unit of project knowledge."""

    id: str
    kind: str  # "patch" | "preset" | "model" | "hardware" | "profile" | "config"
    title: str
    text: str
    ref: str  # human-facing source label, e.g. "patch:PN95"
    score: float = 0.0

    def snippet(self, limit: int = 320) -> str:
        body = self.text.strip()
        return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "ref": self.ref,
            "snippet": self.snippet(),
            "score": round(float(self.score), 4),
        }


@dataclass(frozen=True)
class RetrieveResult:
    """Ranked retrieval result for a query."""

    query: str
    matched: int
    docs: tuple[Doc, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "docs": [d.as_dict() for d in self.docs],
        }


# ─── Corpus assembly ────────────────────────────────────────────────────────


def _patch_docs() -> list[Doc]:
    from .patches.listing import list_patches

    docs: list[Doc] = []
    for row in list_patches():
        upstream = f" upstream PR #{row.upstream_pr}." if row.upstream_pr else ""
        text = (
            f"Patch {row.patch_id} [{row.tier} / {row.family}] {row.title}. "
            f"lifecycle={row.lifecycle}, status={row.production_default}, "
            f"default_on={row.default_on}, env_flag={row.env_flag or 'none'}, "
            f"apply_module={row.apply_module or 'none'}.{upstream}"
        )
        docs.append(
            Doc(
                id=row.patch_id,
                kind="patch",
                title=row.title or row.patch_id,
                text=text,
                ref=f"patch:{row.patch_id}",
            )
        )
    return docs


def _preset_docs() -> list[Doc]:
    from .presets import list_presets

    docs: list[Doc] = []
    for rec in list_presets().presets:
        card = rec.card or {}
        notes = " ".join(
            str(card.get(k, "")) for k in ("title", "summary", "notes", "intent")
        ).strip()
        bits = ", ".join(
            f"{k}={v}"
            for k, v in (
                ("model", rec.model),
                ("hardware", rec.hardware),
                ("profile", rec.profile),
                ("runtime", rec.runtime),
            )
            if v
        )
        text = f"Preset {rec.id}: {bits}. {notes}".strip()
        docs.append(
            Doc(
                id=rec.id,
                kind="preset",
                title=rec.id,
                text=text,
                ref=f"preset:{rec.id}",
            )
        )
    return docs


def _config_docs() -> list[Doc]:
    from .config_editor import collect_v2_config_catalog

    catalog = collect_v2_config_catalog()
    docs: list[Doc] = []
    for group in (catalog.models, catalog.hardware, catalog.profiles, catalog.presets):
        for item in group:
            extra = ", ".join(
                f"{k}={v}"
                for k, v in (
                    ("model", item.model),
                    ("hardware", item.hardware),
                    ("profile", item.profile),
                    ("runtime", item.runtime),
                )
                if v
            )
            text = f"{item.kind} {item.id}: {item.title}. {item.summary} {extra}".strip()
            docs.append(
                Doc(
                    id=item.id,
                    kind=item.kind or "config",
                    title=item.title or item.id,
                    text=text,
                    ref=f"{item.kind or 'config'}:{item.id}",
                )
            )
    return docs


_SOURCES = (_patch_docs, _preset_docs, _config_docs)


def build_corpus() -> list[Doc]:
    """Assemble the full read-only knowledge corpus.

    Each source is best-effort: a source that raises (missing optional dep,
    registry import guard) is skipped rather than failing the whole corpus.
    """
    corpus: list[Doc] = []
    for source in _SOURCES:
        try:
            corpus.extend(source())
        except Exception:
            # Best-effort: a broken source must not break retrieval.
            continue
    return corpus


# ─── External knowledge: Obsidian / notes vaults ────────────────────────────
#
# A "vault" is any local directory of plain-text notes (an Obsidian vault, a
# markdown wiki, a folder of .txt memos). The operator points the chat at it and
# its notes become retrievable knowledge alongside the project corpus. Reads are
# bounded (text files only, size/count caps) and the directory is never written.

_TEXT_EXTS = frozenset({".md", ".markdown", ".mdx", ".txt", ".text", ".org", ".rst"})
_SKIP_DIRS = frozenset({".git", ".obsidian", ".trash", ".smart-env", "node_modules", ".venv", "__pycache__"})
_VAULT_MAX_FILES = 4000
_VAULT_MAX_FILE_BYTES = 1_000_000
_VAULT_MAX_TOTAL_BYTES = 64_000_000

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def resolve_vault(path: str) -> str:
    """Validate and canonicalise a vault directory path.

    Raises ``ValueError`` if the path is empty, missing, or not a directory.
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError("empty vault path")
    resolved = os.path.realpath(os.path.expanduser(raw))
    if not os.path.exists(resolved):
        raise ValueError(f"path does not exist: {raw}")
    if not os.path.isdir(resolved):
        raise ValueError(f"not a directory: {raw}")
    return resolved


def _iter_vault_files(root: str):
    """Yield readable text files under ``root``, skipping hidden/system dirs."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() not in _TEXT_EXTS:
                continue
            yield os.path.join(dirpath, name)
            count += 1
            if count >= _VAULT_MAX_FILES:
                return


def _chunk_note(text: str) -> list[tuple[str, str]]:
    """Split a note into (heading, body) chunks by markdown headings.

    A note with no headings becomes a single untitled chunk. Frontmatter is
    stripped first so YAML metadata never pollutes the retrievable body.
    """
    body = _FRONTMATTER_RE.sub("", text, count=1)
    chunks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush():
        joined = "\n".join(buf).strip()
        if heading or joined:
            chunks.append((heading, joined))

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return [c for c in chunks if c[0] or c[1]]


def vault_docs(path: str) -> list[Doc]:
    """Index a notes vault directory into retrievable :class:`Doc` chunks."""
    root = resolve_vault(path)
    docs: list[Doc] = []
    total_bytes = 0
    for fpath in _iter_vault_files(root):
        try:
            size = os.path.getsize(fpath)
            if size > _VAULT_MAX_FILE_BYTES:
                continue
            total_bytes += size
            if total_bytes > _VAULT_MAX_TOTAL_BYTES:
                break
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(fpath, root)
        stem = os.path.splitext(os.path.basename(fpath))[0]
        chunks = _chunk_note(text)
        for i, (heading, content) in enumerate(chunks):
            title = heading or stem
            ref = f"note:{rel}" + (f"#{heading}" if heading else "")
            body = f"{title}. {content}".strip()
            docs.append(
                Doc(id=f"{rel}#{i}", kind="note", title=title[:120], text=body, ref=ref)
            )
    return docs


def preview_vault(path: str) -> dict[str, Any]:
    """Validate a vault path and report how much it indexes (for GUI feedback)."""
    try:
        root = resolve_vault(path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "files": 0, "chunks": 0, "sample": []}
    try:
        docs = vault_docs(root)
    except Exception as exc:  # noqa: BLE001 - report any read failure to the GUI
        return {"ok": False, "error": str(exc), "files": 0, "chunks": 0, "sample": []}
    files = len({d.id.split("#", 1)[0] for d in docs})
    sample = [d.title for d in docs[:8]]
    return {"ok": True, "error": None, "path": root, "files": files, "chunks": len(docs), "sample": sample}


# ─── Ranking (BM25-lite, pure stdlib) ───────────────────────────────────────


@dataclass
class _Index:
    docs: list[Doc] = field(default_factory=list)
    doc_tokens: list[list[str]] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    total_len: int = 0


def _index_from_docs(docs: list[Doc]) -> _Index:
    idx = _Index(docs=list(docs))
    for doc in idx.docs:
        toks = _tokenize(f"{doc.id} {doc.title} {doc.text}")
        idx.doc_tokens.append(toks)
        idx.total_len += len(toks)
        for term in set(toks):
            idx.df[term] = idx.df.get(term, 0) + 1
    return idx


def _merge_indices(parts: list[_Index]) -> _Index:
    merged = _Index()
    for part in parts:
        merged.docs.extend(part.docs)
        merged.doc_tokens.extend(part.doc_tokens)
        merged.total_len += part.total_len
        for term, c in part.df.items():
            merged.df[term] = merged.df.get(term, 0) + c
    return merged


# The project corpus is static for a process; cache it. Vault indices are cached
# per (path, change-signature) so edits to your notes are picked up.
_PROJECT_INDEX: _Index | None = None
_VAULT_INDICES: dict[str, tuple[Any, _Index]] = {}


def _project_index() -> _Index:
    global _PROJECT_INDEX
    if _PROJECT_INDEX is None:
        _PROJECT_INDEX = _index_from_docs(build_corpus())
    return _PROJECT_INDEX


def _vault_signature(root: str) -> tuple:
    """Cheap change signature: file count, total size, newest mtime."""
    count = 0
    total = 0
    newest = 0.0
    for fpath in _iter_vault_files(root):
        try:
            st = os.stat(fpath)
        except OSError:
            continue
        count += 1
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return (count, total, newest)


def _vault_index(path: str) -> _Index:
    root = resolve_vault(path)
    sig = _vault_signature(root)
    cached = _VAULT_INDICES.get(root)
    if cached is not None and cached[0] == sig:
        return cached[1]
    idx = _index_from_docs(vault_docs(root))
    _VAULT_INDICES[root] = (sig, idx)
    return idx


def reset_cache() -> None:
    """Drop cached indices (used by tests / after a registry or vault reload)."""
    global _PROJECT_INDEX
    _PROJECT_INDEX = None
    _VAULT_INDICES.clear()


def retrieve(
    query: str,
    k: int = 5,
    *,
    include_project: bool = True,
    vaults: tuple[str, ...] | list[str] = (),
) -> RetrieveResult:
    """Rank knowledge docs against ``query`` across the selected sources.

    Sources: the built-in project corpus (``include_project``) and any number of
    local notes ``vaults`` (Obsidian / markdown / txt directories). A vault that
    fails to resolve is skipped rather than failing the whole query.
    """
    q_terms = _tokenize(query)
    if not q_terms:
        return RetrieveResult(query=query.strip(), matched=0, docs=())

    parts: list[_Index] = []
    if include_project:
        parts.append(_project_index())
    for vault in vaults or ():
        try:
            parts.append(_vault_index(vault))
        except Exception:
            # Best-effort: a missing/unreadable vault doesn't break retrieval.
            continue

    idx = _merge_indices(parts) if len(parts) != 1 else parts[0]
    n = len(idx.docs)
    if n == 0:
        return RetrieveResult(query=query.strip(), matched=0, docs=())

    avg_len = (idx.total_len / n) if n else 0.0
    k1, b = 1.5, 0.75
    scored: list[Doc] = []
    q_set = set(q_terms)
    for doc, toks in zip(idx.docs, idx.doc_tokens):
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in q_set:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            continue
        dl = len(toks)
        score = 0.0
        for term, freq in tf.items():
            df = idx.df.get(term, 1)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * dl / (avg_len or 1))
            score += idf * (freq * (k1 + 1)) / denom
        # Small boost when the query names a doc id directly (e.g. "PN95").
        if doc.id.lower() in q_set:
            score += 5.0
        scored.append(
            Doc(id=doc.id, kind=doc.kind, title=doc.title, text=doc.text, ref=doc.ref, score=score)
        )

    scored.sort(key=lambda d: d.score, reverse=True)
    top = tuple(scored[: max(0, k)])
    return RetrieveResult(query=query.strip(), matched=len(scored), docs=top)


def build_context(docs: tuple[Doc, ...] | list[Doc]) -> str:
    """Render retrieved docs as a grounding context block for the model."""
    if not docs:
        return ""
    lines = [
        "Knowledge retrieved for this question (project registry/presets/configs "
        "and your connected notes). Ground your answer in these facts and cite "
        "the source labels where relevant:",
        "",
    ]
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] ({doc.ref}) {doc.snippet(420)}")
    return "\n".join(lines)
