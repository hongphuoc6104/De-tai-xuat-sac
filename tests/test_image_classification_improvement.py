"""Tiny source→two whole-frame releases→fixed-patient comparison E2E."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from histology_data import (
    image_classification as core,
    image_classification_improvement as improvement,
    image_classification_studies as studies,
)
from histology_data.io import file_hash
from tests.test_image_classification import TinyPixelEncoder
from tests.test_image_classification_studies import source as original_source


@pytest.fixture
def releases(tmp_path: Path) -> tuple[studies.StudyData, Path, Path]:
    original, baseline = original_source.__wrapped__(tmp_path)
    incoming = core.extract_original_image_features(
        Path(original.release["inventory_path"]), tmp_path / "variant", tmp_path / "weights",
        encoder=TinyPixelEncoder(), config=core.FeatureExtractionConfig(
            max_long_edge=64, device="cpu", shard_size=7, budget_minutes=5, reserve_minutes=0))
    return original, baseline, incoming


def test_source_producer_to_new_resolution_fixed_folds_and_resume(releases: tuple[studies.StudyData, Path, Path]) -> None:
    old, baseline, new = releases
    before = file_hash(baseline / "study.commit.json")
    with pytest.raises(ValueError, match="Baseline does not belong"):
        studies.load_study_data(new, baseline)
    data = improvement.load_comparison_data(new, old.run_root, baseline)
    assert data.assignment_sha256 == old.assignment_sha256
    assert data.release["release_id"] != old.release["release_id"]
    assert [row["image_id"] for row in data.rows] == [row["image_id"] for row in old.rows]
    result = studies.run_study(data, studies.StudyConfig(
        lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0))
    rerun = studies.run_study(data, studies.StudyConfig(
        lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0))
    assert result["state"] == rerun["state"] == "complete"
    assert result["study_id"] == rerun["study_id"]
    assert result["patient_overlap_count"] == 0
    assert file_hash(baseline / "study.commit.json") == before


def test_comparison_lineage_rejects_patient_label_and_source_changes(releases: tuple[studies.StudyData, Path, Path]) -> None:
    old, baseline, new = releases
    data = improvement.load_comparison_data(new, old.run_root, baseline)
    for key, changed in (("patient_id", "other"), ("image_target", 1 - data.rows[0]["image_target"]),
                         ("original_image_sha256", "0" * 64), ("raw_glade", "99"),
                         ("objective_lens", 99), ("source_member", "other.tif")):
        altered = [dict(row) for row in data.rows]
        altered[0][key] = changed
        with pytest.raises(ValueError, match="lineage"):
            improvement.validate_source_lineage(old.rows, altered)


def test_comparison_lineage_requires_unique_exhaustive_images(releases: tuple[studies.StudyData, Path, Path]) -> None:
    old, _, _ = releases
    with pytest.raises(ValueError, match="coverage"):
        improvement.validate_source_lineage(old.rows, old.rows[:-1])
    with pytest.raises(ValueError, match="coverage"):
        improvement.validate_source_lineage(old.rows, old.rows[:-1] + [old.rows[0]])


def test_comparison_writes_and_verifies_binding_receipt(releases: tuple[studies.StudyData, Path, Path], tmp_path: Path) -> None:
    old, baseline, new = releases
    receipt = improvement.comparison_receipt(new, old.run_root, baseline)
    assert receipt["source_feature_release_id"] == old.release["release_id"]
    assert receipt["image_count"] == len(old.rows)
    assert receipt["same_source_lineage"] is True
    assert json.loads(json.dumps(receipt))["assignment_sha256"] == old.assignment_sha256


def test_benchmark_producer_wholeframes_checks_source_and_commits(releases: tuple[studies.StudyData, Path, Path], tmp_path: Path) -> None:
    old, _, _ = releases
    inventory = core._load_inventory(Path(old.release["inventory_path"]))
    paths = {int(k): Path(v) for k, v in inventory["source_paths"].items()}
    chosen = [row["image_id"] for row in old.rows[:4]]
    encoder = TinyPixelEncoder()
    result = improvement.benchmark_whole_frames(
        Path(old.release["inventory_path"]), old.rows, paths, chosen,
        {"control": (encoder, 48), "larger": (encoder, 64)}, tmp_path / "bench")
    assert result["training_ready"] is False and result["selected_images"] == 4
    assert result["variants"]["control"]["feature_shape"] == [4, 4]
    assert result["variants"]["larger"]["max_long_edge"] == 64
    assert {image.size for image in encoder.images} == {(48, 48), (64, 64)}
    assert (Path(result["benchmark_root"]) / "benchmark.commit.json").is_file()
