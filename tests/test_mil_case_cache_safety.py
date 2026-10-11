"""Adversarial integrity tests for the verified full-case cache snapshot."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from histology_data.io import fingerprint
from histology_data.mil_case_cache import CacheDataset, prepare_case_cache
from histology_data.mil_data import BundleDataset

ROOT = Path(__file__).resolve().parents[1]
LENSES = ("4", "10", "40")


@pytest.fixture(scope="module")
def governed_case_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the existing 12-case synthetic feature→governance→bundle fixture."""
    root = tmp_path_factory.mktemp("mil-case-cache-governed")
    helpers = runpy.run_path(str(ROOT / "tests" / "test_pretrain.py"))
    feature = helpers["make_feature_fixture"](root / "source", patients=6)
    review = helpers["make_reviewed_governance"](feature, root / "governance")
    build_bundle = helpers["build_pretrain_bundle"]
    common = build_bundle(
        Path(review["governance"]["path"]),
        Path(feature["feature_root"]),
        root / "bundles",
        cohort_mode="common",
        expected_sources=1,
        expected_vectors=feature["png_count"],
    )
    all_cases = build_bundle(
        Path(review["governance"]["path"]),
        Path(feature["feature_root"]),
        root / "bundles",
        cohort_mode="all",
        expected_sources=1,
        expected_vectors=feature["png_count"],
    )
    common_bundle = Path(common["bundle_path"])
    bundle_manifest = json.loads((common_bundle / "bundle.json").read_text(encoding="utf-8"))
    case_ids = bundle_manifest["cohort_case_ids"]
    assert len(case_ids) == 12
    assert len(set(case_ids)) == len(case_ids)
    return {
        "root": root,
        "feature_root": Path(feature["feature_root"]),
        "common_bundle": common_bundle,
        "all_bundle": Path(all_cases["bundle_path"]),
        "case_ids": case_ids,
    }


@pytest.fixture(scope="module")
def prepared_case_cache(governed_case_sources: dict[str, Any]) -> dict[str, Any]:
    """Prepare the full source cohort once; all test reads stay on this snapshot."""
    sources = governed_case_sources
    cache_path = prepare_case_cache(
        sources["common_bundle"],
        sources["feature_root"],
        sources["root"] / "case-cache",
        deadline=time.monotonic() + 300,
    )
    return {**sources, "cache_path": cache_path}


def _case_dir(cache_path: Path, case_id: str) -> Path:
    folder = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return cache_path / "cases" / folder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_dataset(sources: dict[str, Any], cache_path: Path, *, feature_root: Path | None = None) -> CacheDataset:
    return CacheDataset(
        sources["common_bundle"],
        feature_root or sources["feature_root"],
        cache_path,
        sources["case_ids"],
    )


def test_cache_materializes_all_governed_cases_and_preserves_arrays_order_and_labels(
    prepared_case_cache: dict[str, Any],
) -> None:
    sources = prepared_case_cache
    bundle = sources["common_bundle"]
    feature_root = sources["feature_root"]
    cache_path = sources["cache_path"]
    source = BundleDataset(bundle, feature_root, sources["case_ids"])
    cached = _cache_dataset(sources, cache_path)
    root_manifest = json.loads((cache_path / "cache_manifest.json").read_text(encoding="utf-8"))

    assert cached.case_ids == sources["case_ids"]
    assert root_manifest["case_ids"] == sorted(sources["case_ids"])
    assert len(list((cache_path / "cases").iterdir())) == len(sources["case_ids"]) == 12
    assert cached.identity["cache_id"] == root_manifest["cache_id"]
    assert cached.identity["bundle_id"] == root_manifest["bundle_id"]
    assert cached.identity["feature_id"] == root_manifest["feature_id"]
    assert cached.identity["split_id"] == root_manifest["split_id"]
    assert cached.identity["source_manifest_sha256"] == _sha256(bundle / "bundle.json")
    assert cached.identity["cached_source_manifest_sha"] == root_manifest["manifest_sha256"]

    for case_id in sources["case_ids"]:
        source_case = source.load_case(case_id)
        cached_case = cached.load_case(case_id)
        assert cached_case.case_id == source_case.case_id == case_id
        assert cached_case.patient_id == source_case.patient_id
        assert cached_case.target == source_case.target
        assert cached_case.presence_mask == source_case.presence_mask
        assert cached.loaded_tile_order(case_id) == source.loaded_tile_order(case_id)

        commit_path = _case_dir(cache_path, case_id) / "commit.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        assert commit["case_id"] == case_id
        assert commit["patient_id"] == source_case.patient_id
        assert commit["target"] == source_case.target
        assert commit["presence_mask"] == list(source_case.presence_mask)
        for lens in LENSES:
            np.testing.assert_array_equal(cached_case.features[lens], source_case.features[lens])
            assert cached_case.features[lens].shape == source_case.features[lens].shape
            assert cached_case.features[lens].dtype == source_case.features[lens].dtype == np.float32
            array_path = _case_dir(cache_path, case_id) / f"lens_{lens}.npy"
            tile_path = _case_dir(cache_path, case_id) / f"tile_order_{lens}.json"
            assert commit["features"][lens]["sha256"] == _sha256(array_path)
            assert commit["tile_order"][lens]["sha256"] == _sha256(tile_path)
            assert json.loads(tile_path.read_text(encoding="utf-8")) == source.loaded_tile_order(case_id)[lens]
            assert commit["features"][lens]["shape"] == list(source_case.features[lens].shape)
            assert commit["tile_order"][lens]["count"] == source_case.features[lens].shape[0]


def test_cache_identity_is_bound_to_exact_source_bundle_and_artifact_inventory(
    prepared_case_cache: dict[str, Any],
) -> None:
    sources = prepared_case_cache
    common_id = json.loads((sources["common_bundle"] / "bundle.json").read_text(encoding="utf-8"))["bundle_id"]
    all_id = json.loads((sources["all_bundle"] / "bundle.json").read_text(encoding="utf-8"))["bundle_id"]
    assert common_id != all_id

    with pytest.raises(ValueError, match="another source bundle or artifact inventory"):
        CacheDataset(
            sources["all_bundle"],
            sources["feature_root"],
            sources["cache_path"],
            sources["case_ids"],
        )


@pytest.mark.parametrize(
    "damage",
    ["array_corruption", "tile_file_missing", "commit_tampering", "case_target_rehashed"],
)
def test_cached_case_payload_corruption_or_loss_is_rejected(
    prepared_case_cache: dict[str, Any],
    tmp_path: Path,
    damage: str,
) -> None:
    sources = prepared_case_cache
    damaged_cache = tmp_path / damage
    shutil.copytree(sources["cache_path"], damaged_cache)
    case_id = sources["case_ids"][0]
    case_dir = _case_dir(damaged_cache, case_id)

    if damage == "array_corruption":
        path = case_dir / "lens_4.npy"
        path.write_bytes(path.read_bytes() + b"corruption")
    elif damage == "tile_file_missing":
        (case_dir / "tile_order_10.json").unlink()
    else:
        commit_path = case_dir / "commit.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["target"] = 1 - commit["target"]
        if damage == "case_target_rehashed":
            commit_body = {key: value for key, value in commit.items() if key != "commit_id"}
            commit["commit_id"] = fingerprint(commit_body)
        commit_path.write_text(json.dumps(commit), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum|commit fingerprint|schema|payload|identity|label|file set"):
        cached = _cache_dataset(sources, damaged_cache)
        cached.load_case(case_id)


def test_cache_root_manifest_corruption_is_rejected_before_case_access(
    prepared_case_cache: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = prepared_case_cache
    damaged_cache = tmp_path / "bad-root-manifest"
    shutil.copytree(sources["cache_path"], damaged_cache)
    manifest_path = damaged_cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_id"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        _cache_dataset(sources, damaged_cache)


def test_rehashed_forged_root_manifest_does_not_authorize_changed_case_label(
    prepared_case_cache: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = prepared_case_cache
    forged_cache = tmp_path / "forged-source-authorization"
    shutil.copytree(sources["cache_path"], forged_cache)
    manifest_path = forged_cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"cache_id", "manifest_sha256", "verified_at_utc"}
    }
    case_id = sources["case_ids"][0]
    body["case_records"][case_id]["target"] = 1 - body["case_records"][case_id]["target"]
    forged_id = fingerprint(body)
    manifest_path.write_text(
        json.dumps({
            **body,
            "cache_id": forged_id,
            "manifest_sha256": fingerprint(body),
            "verified_at_utc": manifest["verified_at_utc"],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another source bundle|artifact inventory"):
        _cache_dataset(sources, forged_cache)


def test_rehashed_case_header_and_shortened_arrays_cannot_change_source_instance_count(
    prepared_case_cache: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = prepared_case_cache
    damaged_cache = tmp_path / "forged-source-count"
    shutil.copytree(sources["cache_path"], damaged_cache)
    case_id = sources["case_ids"][0]
    source = BundleDataset(sources["common_bundle"], sources["feature_root"], [case_id])
    source_case = source.load_case(case_id)
    case_dir = _case_dir(damaged_cache, case_id)
    commit_path = case_dir / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))

    lens = max(LENSES, key=lambda key: source_case.features[key].shape[0])
    source_count = source_case.features[lens].shape[0]
    assert source_count > 0
    array_path = case_dir / f"lens_{lens}.npy"
    original = np.load(array_path, allow_pickle=False)
    shortened = np.array(original[:-1], dtype=np.float32, copy=True)
    with array_path.open("wb") as stream:
        np.save(stream, shortened, allow_pickle=False)
    tile_path = case_dir / f"tile_order_{lens}.json"
    tile_ids = json.loads(tile_path.read_text(encoding="utf-8"))
    tile_path.write_text(json.dumps(tile_ids[:-1]), encoding="utf-8")

    lens_index = LENSES.index(lens)
    commit["features"][lens]["shape"] = list(shortened.shape)
    commit["features"][lens]["sha256"] = _sha256(array_path)
    commit["tile_order"][lens]["count"] = len(tile_ids) - 1
    commit["tile_order"][lens]["sha256"] = _sha256(tile_path)
    commit["presence_mask"][lens_index] = shortened.shape[0] > 0
    commit_body = {key: value for key, value in commit.items() if key != "commit_id"}
    commit["commit_id"] = fingerprint(commit_body)
    commit_path.write_text(json.dumps(commit), encoding="utf-8")

    with pytest.raises(ValueError, match="count|presence|source|metadata|commit"):
        cached = _cache_dataset(sources, damaged_cache)
        cached.load_case(case_id)

    assert source_count - 1 == shortened.shape[0]


def test_source_mutation_before_cache_preparation_fails_without_publishing_cache(
    governed_case_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = governed_case_sources
    changed_feature_root = tmp_path / "mutated-feature-release"
    shutil.copytree(sources["feature_root"], changed_feature_root)
    source_array = next(changed_feature_root.rglob("features.npy"))
    source_array.write_bytes(source_array.read_bytes() + b"source changed before prepare")
    output_root = tmp_path / "rejected-cache"

    with pytest.raises((ValueError, OSError)):
        prepare_case_cache(
            sources["common_bundle"],
            changed_feature_root,
            output_root,
            deadline=time.monotonic() + 300,
        )

    assert not list(output_root.rglob("cache_manifest.json"))
    assert not list(output_root.rglob("commit.json"))


def test_prepared_cache_is_an_immutable_snapshot_after_source_feature_changes(
    prepared_case_cache: dict[str, Any],
    tmp_path: Path,
) -> None:
    sources = prepared_case_cache
    changed_feature_root = tmp_path / "source-after-snapshot"
    shutil.copytree(sources["feature_root"], changed_feature_root)
    case_id = sources["case_ids"][0]
    cached = _cache_dataset(sources, sources["cache_path"], feature_root=changed_feature_root)
    before = cached.load_case(case_id)
    expected = {lens: np.array(before.features[lens], copy=True) for lens in LENSES}
    source_array = next(changed_feature_root.rglob("features.npy"))
    source_array.write_bytes(source_array.read_bytes() + b"source changed after snapshot")

    after = cached.load_case(case_id)

    assert after.case_id == before.case_id
    assert after.patient_id == before.patient_id
    assert after.target == before.target
    assert after.presence_mask == before.presence_mask
    for lens in LENSES:
        np.testing.assert_array_equal(after.features[lens], expected[lens])
