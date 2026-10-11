"""E2E contracts for the verified, materialized MIL case cache."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from histology_data import mil_case_cache
from histology_data.bags import load_case_features
from histology_data.readiness import build_pretrain_bundle


def _hash_tree(root: Path) -> dict[str, str]:
    from histology_data.io import file_hash

    return {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()}


@pytest.fixture(scope="module")
def governed_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from test_pretrain import make_feature_fixture, make_reviewed_governance

    root = tmp_path_factory.mktemp("mil-case-cache")
    source = make_feature_fixture(root / "source", patients=6)
    reviewed = make_reviewed_governance(source, root / "governance")
    result = build_pretrain_bundle(
        Path(reviewed["governance"]["path"]),
        source["feature_root"],
        root / "bundle-output",
        expected_sources=1,
        expected_vectors=source["png_count"],
        seed=42,
    )
    assert result["training_ready"] is True
    return {"bundle": Path(result["bundle_path"]), "feature_root": source["feature_root"]}


def test_case_cache_matches_public_loader_and_keeps_sources_unchanged(
    governed_bundle: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = governed_bundle["bundle"]
    feature_root = governed_bundle["feature_root"]
    before_bundle = _hash_tree(bundle)
    before_features = _hash_tree(feature_root)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    case_ids = manifest["cohort_case_ids"]
    source_cases = {
        case_id: load_case_features(bundle, case_id, feature_root, max_case_bytes=512 * 1024 * 1024)
        for case_id in case_ids
    }

    cache_path = mil_case_cache.prepare_case_cache(
        bundle, feature_root, tmp_path / "case-cache", deadline=time.monotonic() + 120,
    )
    dataset = mil_case_cache.CacheDataset(bundle, feature_root, cache_path, case_ids)
    assert dataset.case_ids == case_ids
    assert dataset.identity["bundle_id"] == manifest["bundle_id"]
    assert dataset.identity["split_id"] == manifest["split_id"]
    assert dataset.identity["cached_source_manifest_sha"]
    cache_manifest = json.loads((cache_path / "cache_manifest.json").read_text(encoding="utf-8"))
    assert cache_manifest["bundle_id"] == manifest["bundle_id"]
    assert cache_manifest["feature_id"] == manifest["feature_id"]
    assert cache_manifest["split_id"] == manifest["split_id"]

    for case_id in case_ids:
        cached = dataset.load_case(case_id)
        original = source_cases[case_id]
        assert cached.case_id == original["case_id"]
        assert cached.patient_id == original["patient_id"]
        assert cached.target == original["case_label"]
        for lens in ("4", "10", "40"):
            np.testing.assert_array_equal(cached.features[lens], original["features"][int(lens)])
        assert dataset.loaded_tile_order(case_id) == {
            lens: original["tile_order"][int(lens)] for lens in ("4", "10", "40")
        }
    assert _hash_tree(bundle) == before_bundle
    assert _hash_tree(feature_root) == before_features


def test_partial_cache_resume_skips_committed_case_loader(
    governed_bundle: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = governed_bundle["bundle"]
    feature_root = governed_bundle["feature_root"]
    case_ids = json.loads((bundle / "bundle.json").read_text())["cohort_case_ids"]
    real_loader = mil_case_cache.load_case_features
    first_calls: list[str] = []

    def interrupt_after_first(bundle_arg: Path, case_id: str, feature_arg: Path, **kwargs: Any) -> dict[str, Any]:
        first_calls.append(case_id)
        if len(first_calls) == 2:
            raise RuntimeError("injected cache interruption")
        return real_loader(bundle_arg, case_id, feature_arg, **kwargs)

    monkeypatch.setattr(mil_case_cache, "load_case_features", interrupt_after_first)
    cache_root = tmp_path / "resumable-cache"
    with pytest.raises(RuntimeError, match="injected cache interruption"):
        mil_case_cache.prepare_case_cache(bundle, feature_root, cache_root, time.monotonic() + 120)
    assert first_calls == case_ids[:2]
    version = next(path for path in cache_root.iterdir() if path.is_dir())
    first_folder = version / "cases" / hashlib.sha256(case_ids[0].encode("utf-8")).hexdigest()
    assert (first_folder / "commit.json").is_file()

    resume_calls: list[str] = []

    def count_remaining(bundle_arg: Path, case_id: str, feature_arg: Path, **kwargs: Any) -> dict[str, Any]:
        resume_calls.append(case_id)
        return real_loader(bundle_arg, case_id, feature_arg, **kwargs)

    monkeypatch.setattr(mil_case_cache, "load_case_features", count_remaining)
    completed = mil_case_cache.prepare_case_cache(bundle, feature_root, cache_root, time.monotonic() + 120)
    assert completed == version
    assert case_ids[0] not in resume_calls
    assert resume_calls == case_ids[1:]
    assert len(list((completed / "cases").iterdir())) == len(case_ids)


def test_cache_open_binds_original_manifest_and_enforces_case_memory_cap(
    governed_bundle: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = governed_bundle["bundle"]
    feature_root = governed_bundle["feature_root"]
    cache_path = mil_case_cache.prepare_case_cache(
        bundle, feature_root, tmp_path / "memory-cache", time.monotonic() + 120,
    )
    case_ids = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))["cohort_case_ids"]
    bounded = mil_case_cache.CacheDataset(bundle, feature_root, cache_path, case_ids, max_case_bytes=1)
    with pytest.raises(MemoryError, match="above max_case_bytes"):
        bounded.load_case(case_ids[0])

    copied_bundle = tmp_path / "changed-bundle"
    shutil.copytree(bundle, copied_bundle)
    with (copied_bundle / "bundle.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprint|source bundle|manifest"):
        mil_case_cache.CacheDataset(copied_bundle, feature_root, cache_path, case_ids)
