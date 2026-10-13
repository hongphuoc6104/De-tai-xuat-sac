"""Real producer-to-study E2E with tiny original images, never clinical data."""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression

from histology_data import (
    image_classification as core,
    image_classification_lab as lab,
    image_classification_studies as studies,
)
from histology_data.io import file_hash, fingerprint
from tests.image_classification_fixtures import OriginalImageFixture, make_original_image_fixture, rewrite_metadata
from tests.test_image_classification import TinyPixelEncoder, _make_features


@pytest.fixture
def source(tmp_path: Path) -> tuple[studies.StudyData, Path]:
    """Both image classes per patient/lens ensure valid per-lens inner support."""
    fixture = make_original_image_fixture(tmp_path / "photos", patients=6)
    rows = list(fixture.rows)
    paths = dict(fixture.image_paths)
    for row in fixture.rows:
        incoming = paths[row["Ten_File"]]
        outgoing = incoming.with_name("ALT_" + incoming.name)
        with Image.open(incoming) as opened:
            Image.fromarray(255 - np.asarray(opened.convert("RGB"))).save(outgoing)
        changed = dict(row)
        changed["Id"] = str(len(rows) + 1)
        changed["Ten_File"] = outgoing.name
        changed["Glade"] = "3" if float(row["Glade"]) == 0 else "0"
        rows.append(changed)
        paths[outgoing.name] = outgoing
        lens = int(row["Do_Phong_Dai"][:-1])
        with zipfile.ZipFile(fixture.archives[lens], "a") as archive:
            archive.writestr("source/" + outgoing.name, outgoing.read_bytes())
    rewrite_metadata(fixture.metadata, rows)
    fixture = OriginalImageFixture(fixture.metadata, fixture.archives, paths, tuple(rows))
    run_root, _ = _make_features(fixture, tmp_path / "baseline")
    baseline = core.train_original_image_classifier(run_root, config=core.TrainingConfig(
        inner_folds=2, c_values=(0.1, 1.0), max_iter=1000, budget_minutes=5, reserve_minutes=0))
    baseline_root = Path(baseline["study_root"])
    return studies.load_study_data(run_root, baseline_root), baseline_root


def test_weight_mass_and_numerical_scale_invariance() -> None:
    x = np.array([[-2, 1], [-1, 0], [-0.5, 0.2], [0.5, -0.2], [1, 0], [2, -1]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    patients = np.array(["a", "a", "b", "c", "c", "d"], dtype=object)
    for scheme in studies.WEIGHT_SCHEMES:
        assert studies.sample_weights(y, patients, scheme).sum() == pytest.approx(len(y))
    original = core._patient_sample_weights(y, patients)
    b = len(y) / original.sum()
    first = LogisticRegression(C=10, tol=1e-10, max_iter=1000).fit(x, y, sample_weight=original)
    second = LogisticRegression(C=10 / b, tol=1e-10, max_iter=1000).fit(x, y, sample_weight=original * b)
    assert first.predict_proba(x) == pytest.approx(second.predict_proba(x), abs=1e-8)


def test_study_resumes_and_preserves_baseline_and_cache(source: tuple[studies.StudyData, Path]) -> None:
    data, baseline = source
    original = file_hash(baseline / "study.commit.json")
    config = studies.StudyConfig(pca_components=2, lambda_values=(0.5, 0.05), budget_minutes=5, reserve_minutes=0)
    first = studies.run_study(data, config)
    second = studies.run_study(data, config)
    assert first["study_id"] == second["study_id"] and second["state"] == "complete"
    assert file_hash(baseline / "study.commit.json") == original
    assert first["assignment_sha256"] == data.assignment_sha256
    assert first["image_count"] == len(data.rows)
    for fold in first["folds"]:
        train = set(fold["train_image_ids"])
        test = set(fold["test_image_ids"])
        assert not train & test
        for head in fold["heads"].values():
            assert set(head["training"]["pca_fit_image_ids"]) <= train
            for split in head["tuning"]["inner_splits"]:
                assert not set(split["pca_fit_image_ids"]) & set(split["validation_image_ids"])
                assert split["weight_mean"] == pytest.approx(1.0)


def test_budget_stop_is_partial_and_resumable(source: tuple[studies.StudyData, Path]) -> None:
    data, _ = source
    result = studies.run_study(data, studies.StudyConfig(budget_minutes=1e-12, reserve_minutes=0))
    assert result["state"] == "partial"
    assert not (Path(result["study_root"]) / "run.lock").exists()
    assert not (Path(result["study_root"]) / "study.commit.json").exists()
    resumed = studies.run_study(data, studies.StudyConfig(budget_minutes=5, reserve_minutes=0))
    assert resumed["state"] == "complete" and resumed["study_id"] == result["study_id"]


def test_per_lens_routing_and_config_identity(source: tuple[studies.StudyData, Path]) -> None:
    data, _ = source
    config = studies.StudyConfig(head_mode="per_lens", lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0)
    result = studies.run_study(data, config)
    root = Path(result["study_root"])
    assert set(result["deployment"]) == {"4", "10", "40"}
    assert set(result["metrics"]["pooled"]["thresholds_by_fold_and_head"]["1"]) == {"4", "10", "40"}
    for i, row in enumerate(data.rows):
        prediction = studies.predict_vector(root, data.features[i], row["objective_lens"])
        assert prediction["head"] == str(row["objective_lens"])
        assert 0 <= prediction["cancer_probability"] <= 1
    with pytest.raises(ValueError, match="known objective"):
        studies.predict_vector(root, data.features[0])
    assert studies._descriptor(data, config) != studies._descriptor(data, studies.StudyConfig())
    assert studies._descriptor(data, config) != studies._descriptor(data, studies.StudyConfig(weight_scheme="none"))
    assert len(studies.read_predictions(root)) == len(data.rows)


def test_tamper_is_rejected(source: tuple[studies.StudyData, Path]) -> None:
    data, _ = source
    result = studies.run_study(data, studies.StudyConfig(lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0))
    root = Path(result["study_root"])
    (root / "oof_predictions.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        studies.load_study(root)


def test_verified_fold_rejects_wrong_head_even_with_valid_file_commit(source: tuple[studies.StudyData, Path]) -> None:
    data, _ = source
    result = studies.run_study(data, studies.StudyConfig(head_mode="per_lens", lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0))
    folder = Path(result["study_root"]) / "folds/fold-01"
    predictions = json.loads((folder / "predictions.json").read_text())
    predictions[0]["head"] = "40" if predictions[0]["head"] != "40" else "4"
    (folder / "predictions.json").write_text(json.dumps(predictions))
    commit = json.loads((folder / "commit.json").read_text())
    commit["files"]["predictions.json"] = file_hash(folder / "predictions.json")
    body = {k: v for k, v in commit.items() if k != "commit_id"}
    commit["commit_id"] = fingerprint(body)
    (folder / "commit.json").write_text(json.dumps(commit))
    with pytest.raises(ValueError, match="head|tuning"):
        studies._verified_fold(data, folder, result["study_id"], 1, *data.outer[0])


def test_invalid_pca_support_fails_before_creating_study(source: tuple[studies.StudyData, Path]) -> None:
    data, _ = source
    with pytest.raises(ValueError, match="PCA exceeds"):
        studies.run_study(data, studies.StudyConfig(pca_components=1000))
    assert not (data.run_root / "head_studies").exists()


def test_baseline_fold_corruption_is_rejected(source: tuple[studies.StudyData, Path]) -> None:
    data, baseline = source
    path = baseline / "folds/fold-01/fold.json"
    value = json.loads(path.read_text())
    value["test_image_ids"] = value["train_image_ids"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        studies.load_study_data(data.run_root, baseline)


def test_normalized_patient_study_reproduces_legacy_objective(source: tuple[studies.StudyData, Path]) -> None:
    data, baseline = source
    result = studies.run_study(data, studies.StudyConfig(lambda_values=(5.0, 0.5), max_iter=1000,
                                                       budget_minutes=5, reserve_minutes=0))
    old = lab._parse_rows(baseline / "oof_predictions.csv")
    new = lab._parse_rows(Path(result["study_root"]) / "oof_predictions.csv")
    old_by_id = {r["image_id"]: r for r in old}
    for row in new:
        assert row["predicted_probability"] == pytest.approx(old_by_id[row["image_id"]]["predicted_probability"], abs=1e-6)
    comparison = lab.paired_deltas(old, new, repeats=20)
    assert comparison["AUROC_delta"] == pytest.approx(0, abs=1e-6)


def test_complete_lab_uses_cache_and_publishes_source_review(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    data, baseline = source
    inventory = core._load_inventory(Path(data.release["inventory_path"]))
    archives = {int(k): Path(v) for k, v in inventory["source_paths"].items()}
    result = lab.run_lab(data.run_root, baseline, tmp_path / "lab", variants=("patient_lr", "per_lens_pca32"),
                         pca_override=2, budget_minutes=5, reserve_minutes=0, archives=archives)
    assert result["state"] == "complete" and result["image_count"] == len(data.rows)
    root = Path(result["lab_root"])
    assert (root / "lab.commit.json").is_file()
    commit = json.loads((root / "lab.commit.json").read_text())
    assert {"baseline_diagnostics.json", "review/review_receipt.json", "review/4x-review.png"} <= set(commit["files"])
    assert set(commit["study_commits"]) == {"patient_lr", "per_lens_pca32"}
    receipt = json.loads((root / "review/review_receipt.json").read_text())
    assert 0 < receipt["image_count"] <= 24
    assert all((root / f"review/{lens}x-review.png").is_file() for lens in (4, 10, 40))
    for row in json.loads((root / "review/private/references.json").read_text()):
        assert file_hash(root / "review" / row["review_file"]) == row["source_sha256"]


def test_paired_comparison_rejects_changed_outer_assignment() -> None:
    rows = [{"image_id": "x", "patient_id": "a", "image_target": 0, "objective_lens": 4,
             "outer_fold": 1, "predicted_probability": 0.1}]
    changed = [{**rows[0], "outer_fold": 2}]
    with pytest.raises(ValueError, match="assignments"):
        lab.paired_deltas(rows, changed, repeats=1)


def test_packaged_cli_lab_e2e(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    from scripts.build_image_classification_runtime import build_runtime

    data, baseline = source
    repo = Path(__file__).resolve().parents[1]
    archive = tmp_path / "runtime.zip"
    build_runtime(repo, archive)
    unpacked = tmp_path / "runtime"
    with zipfile.ZipFile(archive) as package:
        package.extractall(unpacked)
    process = subprocess.run([sys.executable, str(unpacked / "scripts/run_image_classification.py"),
        "head-lab", "--run-root", str(data.run_root), "--baseline-root", str(baseline),
        "--output-root", str(tmp_path / "cli-lab"), "--variants", "patient_lr,class_lr",
        "--budget-minutes", "5", "--reserve-minutes", "0"],
        capture_output=True, text=True, timeout=90, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result["state"] == "complete" and result["image_count"] == len(data.rows)
    assert (Path(result["lab_root"]) / "lab.commit.json").is_file()


def test_one_whole_image_matches_cached_vector_with_lens_head(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    data, _ = source
    result = studies.run_study(data, studies.StudyConfig(head_mode="per_lens", pca_components=2,
        lambda_values=(0.5,), budget_minutes=5, reserve_minutes=0))
    root = Path(result["study_root"])
    image = tmp_path / "photos/originals" / data.rows[0]["file_name"]
    raw = studies.predict_image(root, image, tmp_path / "weights", objective_lens=data.rows[0]["objective_lens"],
                                encoder=TinyPixelEncoder())
    cached = studies.predict_vector(root, data.features[0], data.rows[0]["objective_lens"])
    assert raw["cancer_probability"] == pytest.approx(cached["cancer_probability"], abs=1e-7)
    assert raw["head"] == cached["head"] and raw["image_target"] == cached["image_target"]


def test_partial_archive_map_fails_before_training(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    data, baseline = source
    inventory = core._load_inventory(Path(data.release["inventory_path"]))
    archives = {4: Path(inventory["source_paths"]["4"])}
    with pytest.raises(ValueError, match="three|all"):
        lab.run_lab(data.run_root, baseline, tmp_path / "lab", variants=("patient_lr",),
                    budget_minutes=5, reserve_minutes=0, archives=archives)
    assert not (data.run_root / "head_studies").exists()


def test_source_review_refuses_existing_private_symlink(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    data, baseline = source
    inventory = core._load_inventory(Path(data.release["inventory_path"]))
    archives = {int(k): Path(v) for k, v in inventory["source_paths"].items()}
    output = tmp_path / "review"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "private").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|existing|receipt"):
        lab.review_originals(data, lab._parse_rows(baseline / "oof_predictions.csv"), output, archives)
    assert not list(outside.iterdir())


def test_lab_budget_stop_and_same_manifest_resume(source: tuple[studies.StudyData, Path], tmp_path: Path) -> None:
    data, baseline = source
    output = tmp_path / "lab"
    partial = lab.run_lab(data.run_root, baseline, output, variants=("patient_lr",), budget_minutes=1e-12, reserve_minutes=0)
    assert partial["state"] == "partial"
    assert not (Path(partial["lab_root"]) / "run.lock").exists()
    resumed = lab.run_lab(data.run_root, baseline, output, variants=("patient_lr",), budget_minutes=5, reserve_minutes=0)
    assert resumed["state"] == "complete" and resumed["lab_id"] == Path(partial["lab_root"]).name
    lab.verify_lab(Path(resumed["lab_root"]))
    (Path(resumed["lab_root"]) / "baseline_diagnostics.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        lab.verify_lab(Path(resumed["lab_root"]))
