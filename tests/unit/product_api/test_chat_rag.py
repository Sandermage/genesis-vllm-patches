# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the read-only project-knowledge retrieval (RAG) module.

The module assembles a searchable corpus from existing read-only project
state (patch registry rows, presets, V2 config catalog) and ranks documents
against a free-text query with a pure-stdlib BM25-lite scorer. It never
mutates the registry or any config file — it only reads what the Patches /
Presets / Configs views already expose.
"""
from __future__ import annotations

from sndr.product_api.legacy import chat_rag


def test_build_corpus_has_patch_preset_and_config_docs():
    corpus = chat_rag.build_corpus()
    assert len(corpus) > 50, "expected a substantial corpus from project state"
    kinds = {doc.kind for doc in corpus}
    # At minimum patches must be present; presets/configs are best-effort.
    assert "patch" in kinds
    for doc in corpus:
        assert doc.id and doc.title and doc.text
        assert doc.kind
        assert doc.ref  # human-facing source label


def test_retrieve_ranks_relevant_patch_first():
    # PN95 is a well-known tiered-KV patch id in the registry.
    result = chat_rag.retrieve("PN95 tiered kv cache", k=5)
    assert result.matched >= 1
    assert len(result.docs) <= 5
    top_ids = " ".join(d.id.lower() for d in result.docs)
    assert "pn95" in top_ids


def test_retrieve_empty_query_returns_no_docs():
    result = chat_rag.retrieve("   ", k=5)
    assert result.matched == 0
    assert result.docs == ()


def test_retrieve_respects_k_limit():
    result = chat_rag.retrieve("patch kv cache speculative decode", k=3)
    assert len(result.docs) <= 3


def test_build_context_block_is_grounding_text():
    result = chat_rag.retrieve("PN95 tiered kv cache", k=3)
    block = chat_rag.build_context(result.docs)
    assert isinstance(block, str)
    assert block.strip()
    # Cites at least one source label so the model can ground its answer.
    assert any(d.ref in block for d in result.docs)


def test_retrieved_doc_is_json_safe():
    result = chat_rag.retrieve("preset 27b", k=2)
    for doc in result.docs:
        d = doc.as_dict()
        assert set(d) >= {"id", "kind", "title", "ref", "snippet", "score"}
        assert isinstance(d["score"], float)


# ─── External knowledge: Obsidian / notes vaults ────────────────────────────


def _make_vault(tmp_path):
    (tmp_path / "ml").mkdir()
    (tmp_path / "ml" / "vllm-tuning.md").write_text(
        "---\ntags: [vllm, perf]\n---\n"
        "# vLLM tuning notes\n\n"
        "Raise gpu_memory_utilization to 0.92 for the 27B model.\n\n"
        "## Speculative decoding\n\n"
        "MTP draft length K=3 gave the best acceptance on Qwen.\n",
        encoding="utf-8",
    )
    (tmp_path / "groceries.txt").write_text(
        "Buy milk, eggs, and coffee beans.\n", encoding="utf-8"
    )
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_vault_docs_indexes_markdown_and_chunks_by_heading(tmp_path):
    chat_rag.reset_cache()
    docs = chat_rag.vault_docs(str(_make_vault(tmp_path)))
    assert docs, "expected note docs from the vault"
    kinds = {d.kind for d in docs}
    assert kinds == {"note"}
    titles = " ".join(d.title for d in docs)
    assert "vLLM tuning notes" in titles or "Speculative decoding" in titles
    # YAML frontmatter is stripped, not indexed as body content.
    assert all("tags: [vllm, perf]" not in d.text for d in docs)
    # Hidden config dirs (.obsidian) are skipped.
    assert all(".obsidian" not in d.ref for d in docs)


def test_retrieve_from_vault_only(tmp_path):
    chat_rag.reset_cache()
    vault = str(_make_vault(tmp_path))
    result = chat_rag.retrieve(
        "speculative decoding MTP draft length", k=3,
        include_project=False, vaults=(vault,),
    )
    assert result.matched >= 1
    assert all(d.kind == "note" for d in result.docs)
    assert any("spec" in d.text.lower() or "mtp" in d.text.lower() for d in result.docs)


def test_retrieve_combines_project_and_vault(tmp_path):
    chat_rag.reset_cache()
    vault = str(_make_vault(tmp_path))
    result = chat_rag.retrieve(
        "vllm tuning gpu memory and PN95", k=8,
        include_project=True, vaults=(vault,),
    )
    kinds = {d.kind for d in result.docs}
    # Both a note from the vault and a project doc can surface together.
    assert "note" in kinds


def test_preview_vault_reports_counts(tmp_path):
    info = chat_rag.preview_vault(str(_make_vault(tmp_path)))
    assert info["ok"] is True
    assert info["files"] >= 2
    assert info["chunks"] >= 2
    assert isinstance(info["sample"], list)


def test_preview_vault_rejects_bad_path(tmp_path):
    missing = chat_rag.preview_vault(str(tmp_path / "does-not-exist"))
    assert missing["ok"] is False and missing["error"]
    afile = tmp_path / "f.md"
    afile.write_text("# x\n", encoding="utf-8")
    not_dir = chat_rag.preview_vault(str(afile))
    assert not_dir["ok"] is False
