# SPDX-License-Identifier: Apache-2.0
"""Invalidation-correctness contract for the V2 layer parse cache.

`registry_v2.load_model` / `load_hardware` / `load_profile` memoise the
parsed-and-validated dataclass keyed by the resolved file's (mtime_ns, size).
The GUI overview/catalog/observability paths and
``list_profiles(parent_model=...)`` re-parse the same YAMLs repeatedly, so the
cache turns a ~10 ms `yaml.safe_load` + `typing.get_type_hints` materialisation
into a dict lookup.

A stale cache is worse than no cache: these tests pin the contract that an
operator edit (the GUI write-path bumps the file mtime) is picked up live and
that a cache hit returns the *same* object only while the file is unchanged.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sndr.model_configs import registry_v2


@pytest.fixture(autouse=True)
def _clean_layer_cache():
    """Each test starts from an empty layer cache and leaves it empty."""
    registry_v2.reset_layer_def_cache()
    yield
    registry_v2.reset_layer_def_cache()


def _redirect_builtin_model_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point `load_model` at a writable temp `model/` dir so the test can
    mutate a YAML on disk without touching committed fixtures."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    real_builtin_dir = registry_v2._builtin_dir

    def fake_builtin_dir(layer: str) -> Path:
        if layer == "model":
            return model_dir
        return real_builtin_dir(layer)

    monkeypatch.setattr(registry_v2, "_builtin_dir", fake_builtin_dir)
    return model_dir


def _seed_model_yaml(model_dir: Path, model_id: str, source_id: str) -> Path:
    """Copy a real builtin model YAML into the temp dir, rewriting its `id`
    so it loads as `model_id` and validates against the schema."""
    src = registry_v2._PKG_ROOT / "builtin" / "model" / f"{source_id}.yaml"
    text = src.read_text(encoding="utf-8")
    # The `id:` field must match the filename stem for downstream consistency.
    text = text.replace(f"id: {source_id}", f"id: {model_id}", 1)
    dst = model_dir / f"{model_id}.yaml"
    dst.write_text(text, encoding="utf-8")
    return dst


def _a_real_model_id() -> str:
    # Read the *real* package dir directly so this works even after the test
    # has redirected `_builtin_dir("model")` at an empty temp directory.
    real_dir = registry_v2._PKG_ROOT / "builtin" / "model"
    ids = sorted(
        p.stem for p in real_dir.glob("*.yaml")
        if p.is_file() and not p.stem.startswith("_")
    )
    assert ids, "no builtin models found"
    return ids[0]


def test_repeated_load_returns_cached_object(monkeypatch, tmp_path):
    """Two loads of an unchanged file return the identical object (cache hit)."""
    model_dir = _redirect_builtin_model_dir(monkeypatch, tmp_path)
    source = _a_real_model_id()
    _seed_model_yaml(model_dir, "cache-fixture", source)

    first = registry_v2.load_model("cache-fixture")
    second = registry_v2.load_model("cache-fixture")
    assert first is second, "unchanged file must serve the same cached object"


def test_edit_invalidates_cache(monkeypatch, tmp_path):
    """Rewriting the YAML (new mtime/size) must surface the new content."""
    model_dir = _redirect_builtin_model_dir(monkeypatch, tmp_path)
    source = _a_real_model_id()
    path = _seed_model_yaml(model_dir, "cache-fixture", source)

    first = registry_v2.load_model("cache-fixture")
    original_title = first.title

    # Operator edits the title (the GUI write-path rewrites the file on disk).
    new_text = path.read_text(encoding="utf-8").replace(
        f"title: {original_title}", "title: Edited By Operator", 1
    )
    assert "Edited By Operator" in new_text, "fixture must have a `title:` field"
    # Force a distinct mtime even on coarse-resolution clocks.
    path.write_text(new_text, encoding="utf-8")
    future = time.time() + 5
    os.utime(path, (future, future))

    second = registry_v2.load_model("cache-fixture")
    assert second is not first, "edited file must bypass the stale cache entry"
    assert second.title == "Edited By Operator", "new content must be reflected"


def test_size_change_invalidates_even_on_same_mtime(monkeypatch, tmp_path):
    """A content change that lands on the same mtime stamp still invalidates,
    because the file signature also tracks size."""
    model_dir = _redirect_builtin_model_dir(monkeypatch, tmp_path)
    source = _a_real_model_id()
    path = _seed_model_yaml(model_dir, "cache-fixture", source)

    first = registry_v2.load_model("cache-fixture")
    frozen = path.stat().st_mtime

    new_text = path.read_text(encoding="utf-8").replace(
        f"title: {first.title}",
        "title: Same Mtime Different Size XXXXXXXXXX",
        1,
    )
    path.write_text(new_text, encoding="utf-8")
    # Pin the mtime back to the original value: only size differs now.
    os.utime(path, (frozen, frozen))

    second = registry_v2.load_model("cache-fixture")
    assert second.title == "Same Mtime Different Size XXXXXXXXXX", (
        "size-only change must still invalidate the cache"
    )


def test_reset_hook_clears_cache(monkeypatch, tmp_path):
    """`reset_layer_def_cache()` forces a fresh parse (test isolation hook)."""
    model_dir = _redirect_builtin_model_dir(monkeypatch, tmp_path)
    source = _a_real_model_id()
    _seed_model_yaml(model_dir, "cache-fixture", source)

    first = registry_v2.load_model("cache-fixture")
    registry_v2.reset_layer_def_cache()
    second = registry_v2.load_model("cache-fixture")
    assert first is not second, "reset must drop the cached object"
    assert second.title == first.title, "content is unchanged after a pure reset"


def test_distinct_ids_do_not_collide(monkeypatch, tmp_path):
    """Two different ids must not share a cache slot."""
    model_dir = _redirect_builtin_model_dir(monkeypatch, tmp_path)
    source = _a_real_model_id()
    _seed_model_yaml(model_dir, "cache-a", source)
    _seed_model_yaml(model_dir, "cache-b", source)

    a = registry_v2.load_model("cache-a")
    b = registry_v2.load_model("cache-b")
    assert a.id == "cache-a"
    assert b.id == "cache-b"
    assert a is not b
