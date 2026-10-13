"""Tiny synthetic regression checks for the post-training analyzer."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_image_study import ExpectedCounts, VerificationError, analyze_study  # noqa: E402

LENSES = (4, 10, 40)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _patient_rows() -> list[dict[str, object]]:
    rows = []
    for patient_index in range(6):
        patient_id = f"patient_private_sentinel_{patient_index:02d}"
        outer_fold = patient_index // 2 + 1
        for lens_index, lens in enumerate(LENSES):
            grade_slot = (patient_index + lens_index) % 3
            grade = 0 if grade_slot == 0 else (3, 4, 5)[(patient_index + 2 * lens_index) % 3]
            target = int(grade != 0)
            if target == 0:
                probability = 0.2 if patient_index % 2 == 0 else 0.6
            else:
                probability = {4: 0.7, 10: 0.55, 40: 0.9}[lens]
            threshold = 0.61
            rows.append({
                "image_id": f"image_private_sentinel_{patient_index:02d}_{lens}x",
                "file_name": f"original_private_image_{patient_index:02d}_{lens}x.tif",
                "patient_id": patient_id,
                "objective_lens": lens,
                "raw_glade": str(grade),
                "image_target": target,
                "outer_fold": outer_fold,
                "predicted_probability": probability,
                "prediction_at_0_5": int(probability >= 0.5),
                "selected_threshold": threshold,
                "prediction_at_selected_threshold": int(probability >= threshold),
            })
    return rows


def _saved_metric(rows: list[dict[str, object]], pred_key: str) -> dict[str, object]:
    labels = np.asarray([int(row["image_target"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([float(row["predicted_probability"]) for row in rows], dtype=np.float64)
    predictions = np.asarray([int(row[pred_key]) for row in rows], dtype=np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {
        "image_count": len(rows),
        "positive_images": tp + fn,
        "negative_images": tn + fp,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": (tn + tp) / len(rows),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _saved_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"pooled": {}, "by_objective_lens": {}}
    pooled = result["pooled"]
    pooled["fixed_threshold_0_5"] = _saved_metric(rows, "prediction_at_0_5")
    pooled["validation_selected_threshold"] = _saved_metric(rows, "prediction_at_selected_threshold")
    pooled["selected_thresholds_by_fold"] = {str(fold): 0.61 for fold in (1, 2, 3)}
    for lens in LENSES:
        selected = [row for row in rows if row["objective_lens"] == lens]
        result["by_objective_lens"][str(lens)] = {
            "fixed_threshold_0_5": _saved_metric(selected, "prediction_at_0_5"),
            "validation_selected_threshold": _saved_metric(selected, "prediction_at_selected_threshold"),
        }
    return result


def _study_artifact(rows: list[dict[str, object]], fold_audits: list[dict[str, object]],
                    config: dict[str, object]) -> dict[str, object]:
    counts = Counter(int(row["image_target"]) for row in rows)
    return {
        "schema_version": 1,
        "protocol": "original-image-frozen-resnet50-v1",
        "study_id": _json_sha(config),
        "release_id": "synthetic-release-private-sentinel",
        "inventory_id": "synthetic-inventory-private-sentinel",
        "training_config": config,
        "image_count": len(rows),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "class_counts": {str(label): counts[label] for label in (0, 1)},
        "outer_fold_count": 3,
        "folds": fold_audits,
        "metrics": _saved_metrics(rows),
        "deployment_model": {"scaler_fit_image_ids": [str(row["image_id"]) for row in rows]},
        "patient_split_leakage_check": {"passed": True, "outer_folds": 3, "patient_overlap_count": 0},
        "prediction_unit": "original_image",
        "training_target_field": "image_target",
        "case_target_used": False,
        "patch_labels_used": False,
    }


def _commit_inputs(private_dir: Path) -> None:
    """Produce the exact file-bound envelope used by the real trainer."""
    study = json.loads((private_dir / "study.json").read_text())
    body = {"study_id": study["study_id"], "release_id": study["release_id"], "complete": True,
            "files": {name: hashlib.sha256((private_dir / name).read_bytes()).hexdigest()
                      for name in ("study.json", "oof_predictions.csv")}}
    body["commit_id"] = _json_sha(body)
    (private_dir / "study.commit.json").write_text(json.dumps(body))
    (private_dir / "study.commit.json").chmod(0o600)


def _make_private_artifacts(private_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    private_dir.mkdir(parents=True, exist_ok=True)
    rows = _patient_rows()
    row_ids_by_patient: dict[str, list[str]] = {}
    for row in rows:
        row_ids_by_patient.setdefault(str(row["patient_id"]), []).append(str(row["image_id"]))
    patients = sorted(row_ids_by_patient)
    config: dict[str, object] = {
        "protocol": "original-image-frozen-resnet50-v1",
        "release_id": "synthetic-release-private-sentinel",
        "outer_folds": 3,
        "inner_folds": 2,
        "seed": 17,
        "c_values": [0.1, 1.0],
        "balanced_patient_weights": True,
        "max_iter": 200,
    }
    fold_audits = []
    scaler_mean = [0.25, 0.5, 0.75]
    scaler_scale = [1.0, 1.0, 1.0]
    scaler_mean_hash = hashlib.sha256(np.asarray(scaler_mean, dtype=np.float64).tobytes()).hexdigest()
    scaler_scale_hash = hashlib.sha256(np.asarray(scaler_scale, dtype=np.float64).tobytes()).hexdigest()
    for fold_number in (1, 2, 3):
        test_patients = patients[(fold_number - 1) * 2:fold_number * 2]
        train_patients = [patient for patient in patients if patient not in test_patients]
        train_ids = [str(row["image_id"]) for row in rows if row["patient_id"] in train_patients]
        test_ids = [str(row["image_id"]) for row in rows if row["patient_id"] in test_patients]
        inner_splits = []
        inner_audit_by_c = {"0.1": [], "1.0": []}
        for split_number in (1, 2):
            validation_patients = train_patients[(split_number - 1) * 2:split_number * 2]
            inner_train_patients = [patient for patient in train_patients if patient not in validation_patients]
            inner_train_ids = [str(row["image_id"]) for row in rows if row["patient_id"] in inner_train_patients]
            inner_validation_ids = [str(row["image_id"]) for row in rows if row["patient_id"] in validation_patients]
            inner_splits.append({
                "split": split_number,
                "train_image_ids": inner_train_ids,
                "validation_image_ids": inner_validation_ids,
                "scaler_fit_image_ids": inner_train_ids,
                "train_patient_hashes": sorted(_sha(patient) for patient in inner_train_patients),
                "validation_patient_hashes": sorted(_sha(patient) for patient in validation_patients),
                "patient_overlap_count": 0,
            })
            for c_key in inner_audit_by_c:
                inner_audit_by_c[c_key].append({
                    "split": split_number,
                    "validation_auc": 0.8,
                    "scaler_fit_row_count": len(inner_train_ids),
                    "scaler_mean_sha256": scaler_mean_hash,
                    "scaler_scale_sha256": scaler_scale_hash,
                })
        train_patient_hashes = sorted(_sha(patient) for patient in train_patients)
        test_patient_hashes = sorted(_sha(patient) for patient in test_patients)
        fold_audits.append({
            "fold_id": f"fold-{fold_number:02d}",
            "outer_fold": fold_number,
            "train_image_count": len(train_ids),
            "test_image_count": len(test_ids),
            "train_image_ids": train_ids,
            "test_image_ids": test_ids,
            "scaler_fit_image_ids": train_ids,
            "train_patient_hashes": train_patient_hashes,
            "test_patient_hashes": test_patient_hashes,
            "patient_overlap_count": 0,
            "selected_C": 1.0,
            "selected_threshold": 0.61,
            "selected_C_validation_roc_auc": 0.8,
            "threshold_selection_validation_balanced_accuracy": 0.75,
            "c_validation_scores": [
                {"C": 0.1, "validation_roc_auc": 0.7, "validation_selected_threshold": 0.61,
                 "validation_balanced_accuracy_at_selected_threshold": 0.75},
                {"C": 1.0, "validation_roc_auc": 0.8, "validation_selected_threshold": 0.61,
                 "validation_balanced_accuracy_at_selected_threshold": 0.75},
            ],
            "inner_validation_splits": inner_splits,
            "inner_validation_scaler_audit": inner_audit_by_c,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "scaler_mean_sha256": scaler_mean_hash,
            "scaler_scale_sha256": scaler_scale_hash,
            "feature_dim": len(scaler_mean),
        })

    study = _study_artifact(rows, fold_audits, config)
    (private_dir / "study.json").write_text(json.dumps(study, ensure_ascii=False), encoding="utf-8")
    (private_dir / "training_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    fields = [
        "image_id", "file_name", "patient_id", "objective_lens", "raw_glade", "image_target", "outer_fold",
        "predicted_probability", "prediction_at_0_5", "selected_threshold", "prediction_at_selected_threshold",
    ]
    with (private_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _commit_inputs(private_dir)
    private_dir.chmod(0o700)
    for name in ("study.json", "training_config.json", "oof_predictions.csv"):
        (private_dir / name).chmod(0o600)
    return rows, study


def test_analyzer_verifies_synthetic_study_and_writes_aggregate_only_reports(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    rows, _study = _make_private_artifacts(private_dir)
    report_dir = tmp_path / "report"
    expected = ExpectedCounts(
        image_count=18,
        patient_count=6,
        negative_images=6,
        positive_images=12,
        lens_counts=((4, 6), (10, 6), (40, 6)),
    )

    result = analyze_study(private_dir, report_dir, expected=expected, bootstrap_replicates=60, seed=91)

    assert result["counts"]["image_count"] == 18
    assert result["counts"]["patient_count"] == 6
    assert result["checks"]["outer_folds"] == 3
    assert result["checks"]["patient_overlap_count"] == 0
    assert result["checks"]["inner_splits_checked"] == 6
    assert result["metrics"]["pooled"]["fixed_threshold_0_5"]["confusion_matrix"] == {
        "tn": 3, "fp": 3, "fn": 0, "tp": 12,
    }
    assert result["metrics"]["pooled"]["validation_selected_threshold"]["confusion_matrix"] == {
        "tn": 6, "fp": 0, "fn": 4, "tp": 8,
    }
    labels = np.asarray([int(row["image_target"]) for row in rows])
    probabilities = np.asarray([float(row["predicted_probability"]) for row in rows])
    assert result["metrics"]["pooled"]["fixed_threshold_0_5"]["roc_auc"] == pytest.approx(
        roc_auc_score(labels, probabilities), abs=1e-12,
    )
    assert result["metrics"]["pooled"]["fixed_threshold_0_5"]["auprc"] == pytest.approx(
        average_precision_score(labels, probabilities), abs=1e-12,
    )
    bootstrap = result["metrics"]["patient_group_bootstrap_ci"]
    assert bootstrap["replicates_requested"] == 60
    assert bootstrap["replicates_valid"] == 60
    assert "do not refit" in bootstrap["method"]

    for name in ("verification.json", "report_vi.md", "roc_oof.png", "confusion_selected_threshold.png", "precision_recall_oof.png"):
        artifact = report_dir / name
        assert artifact.is_file()
        assert artifact.stat().st_size < 1_000_000
    report_bytes = b"".join(path.read_bytes() for path in report_dir.iterdir() if path.is_file())
    for row in rows:
        for private_id in (str(row["patient_id"]), str(row["image_id"]), str(row["file_name"])):
            assert private_id.encode("utf-8") not in report_bytes
    assert "không xác lập hiệu quả lâm sàng" in (report_dir / "report_vi.md").read_text(encoding="utf-8")


def test_analyzer_rejects_count_mismatch_without_creating_report(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    _make_private_artifacts(private_dir)
    report_dir = tmp_path / "must-not-exist"
    expected = ExpectedCounts(
        image_count=19,
        patient_count=6,
        negative_images=6,
        positive_images=12,
        lens_counts=((4, 6), (10, 6), (40, 6)),
    )

    with pytest.raises(VerificationError, match="Expected 19 OOF images"):
        analyze_study(private_dir, report_dir, expected=expected, bootstrap_replicates=0)
    assert not report_dir.exists()


def test_analyzer_rejects_patient_overlap_in_a_fold_audit(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    _make_private_artifacts(private_dir)
    study_path = private_dir / "study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    test_image_id = study["folds"][0]["test_image_ids"][0]
    study["folds"][0]["train_image_ids"].append(test_image_id)
    study_path.write_text(json.dumps(study, ensure_ascii=False), encoding="utf-8")
    study_path.chmod(0o600)
    _commit_inputs(private_dir)
    report_dir = tmp_path / "must-not-exist"

    with pytest.raises(VerificationError, match="disjoint full-cohort partition"):
        analyze_study(
            private_dir,
            report_dir,
            expected=ExpectedCounts(18, 6, 6, 12, ((4, 6), (10, 6), (40, 6))),
            bootstrap_replicates=0,
        )
    assert not report_dir.exists()


def test_report_accepts_committed_producer_v1_without_optional_scale_hash(tmp_path: Path) -> None:
    private = tmp_path / "private"
    _make_private_artifacts(private)
    study = json.loads((private / "study.json").read_text())
    for fold in study["folds"]:
        del fold["scaler_scale_sha256"]
    (private / "study.json").write_text(json.dumps(study))
    _commit_inputs(private)
    result = analyze_study(private, tmp_path / "report", expected=ExpectedCounts(18, 6, 6, 12, ((4, 6), (10, 6), (40, 6))), bootstrap_replicates=0)
    assert result["checks"]["study_commit_verified"]
    assert result["checks"]["serialized_scaler_scales_covered_by_study_hash"]


def test_report_rejects_scale_tampering_without_recommitting_study(tmp_path: Path) -> None:
    private = tmp_path / "private"
    _make_private_artifacts(private)
    study = json.loads((private / "study.json").read_text())
    study["folds"][0]["scaler_scale"][0] = 2.0
    (private / "study.json").write_text(json.dumps(study))
    with pytest.raises(VerificationError, match="Committed input checksum"):
        analyze_study(private, tmp_path / "report", expected=ExpectedCounts(18, 6, 6, 12, ((4, 6), (10, 6), (40, 6))), bootstrap_replicates=0)
    assert not (tmp_path / "report").exists()


def test_report_rejects_nonpositive_scales_even_with_valid_commit(tmp_path: Path) -> None:
    private = tmp_path / "private"
    _make_private_artifacts(private)
    study = json.loads((private / "study.json").read_text())
    del study["folds"][0]["scaler_scale_sha256"]
    study["folds"][0]["scaler_scale"][0] = 0.0
    (private / "study.json").write_text(json.dumps(study))
    _commit_inputs(private)
    with pytest.raises(VerificationError, match="finite and positive"):
        analyze_study(private, tmp_path / "report", expected=ExpectedCounts(18, 6, 6, 12, ((4, 6), (10, 6), (40, 6))), bootstrap_replicates=0)
    assert not (tmp_path / "report").exists()


def test_analyzer_cli_accepts_an_actual_v1_producer_study(tmp_path: Path) -> None:
    """E2E uses the real inventory/extractor/trainer rather than invented audits."""
    from histology_data.image_classification import TrainingConfig, train_original_image_classifier
    from tests.image_classification_fixtures import make_original_image_fixture
    from tests.test_image_classification import _make_features

    fixture = make_original_image_fixture(tmp_path / "source", patients=6)
    feature_root, _ = _make_features(fixture, tmp_path / "pipeline")
    trained = train_original_image_classifier(feature_root, config=TrainingConfig(
        inner_folds=2, c_values=(0.1, 1.0), max_iter=1000, budget_minutes=5, reserve_minutes=0))
    assert trained["state"] == "complete"
    study_root = Path(trained["study_root"])
    study = json.loads((study_root / "study.json").read_text())
    assert all("scaler_scale_sha256" not in fold for fold in study["folds"])
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    for name in ("study.json", "training_config.json", "oof_predictions.csv", "study.commit.json"):
        path = private / name
        shutil.copyfile(study_root / name, path)
        path.chmod(0o600)
    output = tmp_path / "report"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_image_study.py"
    process = subprocess.run([sys.executable, str(script), "--input-dir", str(private),
        "--report-dir", str(output), "--expected-images", "18", "--expected-patients", "6",
        "--expected-negative", "6", "--expected-positive", "12", "--expected-lens-4", "6",
        "--expected-lens-10", "6", "--expected-lens-40", "6", "--bootstrap-replicates", "0"],
        capture_output=True, text=True, timeout=60, check=False)
    assert process.returncode == 0, process.stderr
    verification = json.loads((output / "verification.json").read_text())
    assert verification["checks"]["study_commit_verified"]
    assert verification["checks"]["patient_overlap_count"] == 0
    assert verification["counts"]["image_count"] == 18
