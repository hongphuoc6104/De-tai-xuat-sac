"""Cached versus uncached nested-study E2E on a governed synthetic bundle."""
from __future__ import annotations

import csv
import hashlib
import json
import runpy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from histology_data import bags, mil_case_cache, mil_data
from histology_data.io import file_hash
from histology_data.mil_case_cache import CacheDataset, prepare_case_cache, verify_source_readonly
from histology_data.mil_checkpoint import CheckpointStore
from histology_data.mil_study import StudyConfig, run_study
from histology_data.readiness import verify_pretrain_bundle


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _hash_differences(before: dict[str, str], after: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    return {
        path: (before.get(path), after.get(path))
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    }


def _fit_states(output: Path) -> dict[str, dict[str, Any]]:
    states = {}
    for config_path in sorted((output / "fits" / "E0").glob("outer_*/**/resolved_config.json")):
        fit_dir = config_path.parent
        identity = json.loads(config_path.read_text(encoding="utf-8"))
        state = CheckpointStore(fit_dir / "checkpoints", identity).load_latest()
        assert state is not None and state["finished"] is True
        states[fit_dir.relative_to(output / "fits" / "E0").as_posix()] = state
    return states


def _oof_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


@pytest.fixture(scope="module")
def cached_study_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mil-cached-study")
    helper_path = Path(__file__).resolve().parents[1] / "tests" / "test_pretrain.py"
    helpers = runpy.run_path(str(helper_path))
    source = helpers["make_feature_fixture"](root / "source", patients=6)
    reviewed = helpers["make_reviewed_governance"](source, root / "governance")
    built = helpers["build_pretrain_bundle"](
        Path(reviewed["governance"]["path"]),
        source["feature_root"],
        root / "bundles",
        cohort_mode="common",
        expected_sources=1,
        expected_vectors=source["png_count"],
    )
    bundle = Path(built["bundle_path"])
    feature_root = source["feature_root"]
    verified = verify_pretrain_bundle(bundle, feature_root, deadline=time.monotonic() + 300)
    assert verified["training_ready"] is True
    source_hashes = {"bundle": _tree_hashes(bundle), "features": _tree_hashes(feature_root)}
    cache_path = prepare_case_cache(
        bundle,
        feature_root,
        root / "case-caches",
        time.monotonic() + 300,
    )
    assert _tree_hashes(bundle) == source_hashes["bundle"]
    assert _tree_hashes(feature_root) == source_hashes["features"]
    return {
        "root": root,
        "bundle": bundle,
        "feature_root": feature_root,
        "cache_path": cache_path,
        "case_count": len(source["case_labels"]),
        "patient_count": len(set(source["case_patients"].values())),
        "source_hashes": source_hashes,
    }


def _config() -> StudyConfig:
    return StudyConfig(
        variants=("E0",),
        max_epochs=1,
        patience=1,
        seed=42,
        device="cpu",
        memory_limit_bytes=2 * 1024**3,
        projection_dim=128,
        attention_dim=64,
        dropout=0.25,
    )


def _run(bundle: Path, feature_root: Path, output: Path, *, case_cache: Path | None) -> dict[str, Any]:
    return run_study(
        bundle,
        feature_root,
        output,
        _config(),
        deadline=time.monotonic() + 600,
        allow_synthetic_cpu=True,
        case_cache=case_cache,
    )


def _attention_directory(output: Path, fold: int, case_id: str) -> Path:
    case_hash = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return output / "fits" / "E0" / f"outer_{fold}" / "refit" / "attention" / case_hash


def test_cached_study_matches_uncached_weights_oof_and_verified_callback_order(
    cached_study_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = cached_study_inputs["bundle"]
    feature_root = cached_study_inputs["feature_root"]
    cache_path = cached_study_inputs["cache_path"]
    uncached_output = tmp_path / "uncached-study"
    cached_output = tmp_path / "cached-study"
    source_hashes_before = {
        "bundle": cached_study_inputs["source_hashes"]["bundle"],
        "features": cached_study_inputs["source_hashes"]["features"],
    }

    uncached_result = _run(bundle, feature_root, uncached_output, case_cache=None)
    source_hashes_after_uncached = {
        "bundle": _tree_hashes(bundle),
        "features": _tree_hashes(feature_root),
    }
    assert uncached_result["phase"] == "complete"
    assert uncached_result["oof_rows"] == cached_study_inputs["case_count"] == 12

    # Verify source metadata in the read-only mirror before forbidding the public
    # per-case loader for the cached study run.
    assert verify_source_readonly(bundle, feature_root, time.monotonic() + 300)["training_ready"]
    source_loader_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forbid_source_case_load(*args: Any, **kwargs: Any) -> None:
        source_loader_calls.append((args, kwargs))
        raise AssertionError("cached study attempted to reload source feature cases")

    # Cache preparation is complete. No fit, validation, OOF or attention callback
    # may read the public feature source again.
    monkeypatch.setattr(mil_case_cache, "load_case_features", forbid_source_case_load)
    monkeypatch.setattr(mil_data, "load_case_features", forbid_source_case_load)
    monkeypatch.setattr(bags, "load_case_features", forbid_source_case_load)
    monkeypatch.setattr(mil_data.BundleDataset, "load_case", forbid_source_case_load)

    cached_result = _run(bundle, feature_root, cached_output, case_cache=cache_path)
    source_hashes_after_cached = {
        "bundle": _tree_hashes(bundle),
        "features": _tree_hashes(feature_root),
    }
    assert cached_result["phase"] == "complete"
    assert cached_result["oof_rows"] == cached_study_inputs["case_count"] == 12
    assert source_loader_calls == []

    uncached_identity = json.loads((uncached_output / "resolved_config.json").read_text(encoding="utf-8"))
    cached_identity = json.loads((cached_output / "resolved_config.json").read_text(encoding="utf-8"))
    assert uncached_identity["case_cache_identity"] is None
    assert cached_identity["case_cache_identity"]["cache_id"]
    assert uncached_identity["bundle_manifest_sha256"] == cached_identity["bundle_manifest_sha256"]
    assert uncached_identity["split_id"] == cached_identity["split_id"]

    baseline_states = _fit_states(uncached_output)
    cached_states = _fit_states(cached_output)
    assert set(baseline_states) == set(cached_states)
    assert len(cached_states) == 9
    for fit_id, baseline_state in baseline_states.items():
        cached_state = cached_states[fit_id]
        assert baseline_state["epoch"] == cached_state["epoch"] == 1
        assert baseline_state["global_step"] == cached_state["global_step"]
        assert baseline_state["best_epoch"] == cached_state["best_epoch"]
        assert baseline_state["model"].keys() == cached_state["model"].keys()
        for name, baseline_tensor in baseline_state["model"].items():
            torch.testing.assert_close(cached_state["model"][name], baseline_tensor, rtol=1e-6, atol=1e-8)

    baseline_oof = _oof_rows(uncached_output / "evaluation" / "oof_predictions.csv")
    cached_oof = _oof_rows(cached_output / "evaluation" / "oof_predictions.csv")
    assert set(baseline_oof) == set(cached_oof) == set(
        json.loads((bundle / "splits.json").read_text(encoding="utf-8"))["case_ids"]
    )
    for case_id, baseline_row in baseline_oof.items():
        cached_row = cached_oof[case_id]
        for field in ("patient_id", "target", "variant", "outer_fold", "threshold", "predicted_label"):
            assert cached_row[field] == baseline_row[field]
        for field in ("probability", "logit"):
            assert float(cached_row[field]) == pytest.approx(float(baseline_row[field]), rel=1e-6, abs=1e-8)

    splits = json.loads((bundle / "splits.json").read_text(encoding="utf-8"))
    for outer in splits["outer"]:
        cache_dataset = CacheDataset(bundle, feature_root, cache_path, outer["test_cases"])
        for case_id in outer["test_cases"]:
            cache_case = cache_dataset.load_case(case_id)
            tile_order = cache_dataset.loaded_tile_order(case_id)
            cached_case_dir = _attention_directory(cached_output, outer["fold"], case_id)
            uncached_case_dir = _attention_directory(uncached_output, outer["fold"], case_id)
            for lens in ("4", "10", "40"):
                refs = json.loads((cached_case_dir / f"references_{lens}.json").read_text(encoding="utf-8"))
                assert [row["tile_id"] for row in refs] == tile_order[lens]
                baseline_refs = json.loads(
                    (uncached_case_dir / f"references_{lens}.json").read_text(encoding="utf-8"),
                )
                assert refs == baseline_refs
                cached_weights = np.load(cached_case_dir / f"attention_{lens}.npy", allow_pickle=False)
                baseline_weights = np.load(uncached_case_dir / f"attention_{lens}.npy", allow_pickle=False)
                np.testing.assert_allclose(cached_weights, baseline_weights, rtol=1e-6, atol=1e-8)
            assert cache_case.case_id == case_id

    assert (
        source_hashes_after_uncached == source_hashes_before
        and source_hashes_after_cached == source_hashes_before
    ), (
        "Study mutated immutable inputs: "
        f"uncached bundle={_hash_differences(source_hashes_before['bundle'], source_hashes_after_uncached['bundle'])!r}, "
        f"uncached features={_hash_differences(source_hashes_before['features'], source_hashes_after_uncached['features'])!r}, "
        f"cached bundle={_hash_differences(source_hashes_before['bundle'], source_hashes_after_cached['bundle'])!r}, "
        f"cached features={_hash_differences(source_hashes_before['features'], source_hashes_after_cached['features'])!r}"
    )
