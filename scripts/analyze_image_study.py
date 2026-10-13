#!/usr/bin/env python3
"""Verify and report a completed original-image, patient-grouped OOF study.

The analyzer reads only the committed owner-private study artifacts and writes
aggregate tables/figures. It never writes image-, filename-, or patient-level
records to the report directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

TARGET_BY_GRADE = {0: 0, 3: 1, 4: 1, 5: 1}
DEFAULT_EXECUTION_ROOT = Path(__file__).resolve().parent.parent


class VerificationError(ValueError):
    """Input artifact inconsistency or failed study verification."""


@dataclass(frozen=True)
class ExpectedCounts:
    """Strict cohort totals; tests may supply a smaller synthetic profile."""

    image_count: int = 2802
    patient_count: int = 18
    negative_images: int = 2298
    positive_images: int = 504
    lens_counts: tuple[tuple[int, int], ...] = ((4, 364), (10, 830), (40, 1608))

    @property
    def class_counts(self) -> dict[int, int]:
        return {0: self.negative_images, 1: self.positive_images}

    @property
    def lens_count_map(self) -> dict[int, int]:
        return dict(self.lens_counts)


DEFAULT_EXPECTED_COUNTS = ExpectedCounts()


@dataclass(frozen=True)
class StudyArtifacts:
    """Read-only paths for the private training outputs."""

    input_dir: Path
    study_path: Path
    predictions_path: Path
    config_path: Path
    commit_path: Path

    @classmethod
    def from_private_dir(cls, input_dir: Path) -> "StudyArtifacts":
        source = Path(input_dir).expanduser()
        if source.is_symlink() or not source.is_dir():
            raise VerificationError("Private input directory is missing or is not a regular directory.")
        root = source.resolve()
        return cls(
            input_dir=root,
            study_path=root / "study.json",
            predictions_path=root / "oof_predictions.csv",
            config_path=root / "training_config.json",
            commit_path=root / "study.commit.json",
        )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _fail_samples(label: str, values: Iterable[str], reason: str) -> VerificationError:
    samples = sorted({_short_hash(str(value)) for value in values})[:5]
    return VerificationError(f"{reason} (count={len(samples)} hashed examples for {label}: {samples}).")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Unable to read required JSON artifact: {path.name}.") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object in {path.name}.")
    return value


def _check_private_permissions(artifacts: StudyArtifacts) -> None:
    """Require input IDs to remain in an owner-only private directory/files."""
    if os.name != "posix":
        raise VerificationError("Owner-only input permission checks require a POSIX filesystem.")
    directory_mode = artifacts.input_dir.stat().st_mode & 0o777
    if directory_mode & 0o077:
        raise VerificationError("Private input directory must be owner-only (mode 0700 or stricter).")
    for path in (artifacts.study_path, artifacts.predictions_path, artifacts.config_path, artifacts.commit_path):
        if path.is_symlink() or not path.is_file():
            raise VerificationError(f"Required private artifact is missing or unsafe: {path.name}.")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise VerificationError(f"Private artifact must be owner-only (mode 0600 or stricter): {path.name}.")


def _verify_input_commit(artifacts: StudyArtifacts, study: dict[str, Any]) -> None:
    """Bind serialized audit vectors and OOF rows to the producer's study commit.

    Version 1 producers store scaler scales in the committed study without a
    separate scale hash. The whole-study digest covers these values; supplied
    per-array hashes remain additional checks, never invented fields.
    """
    commit = _read_json(artifacts.commit_path)
    body = {key: value for key, value in commit.items() if key != "commit_id"}
    if (commit.get("commit_id") != _canonical_sha256(body)
            or commit.get("complete") is not True
            or commit.get("study_id") != study.get("study_id")
            or commit.get("release_id") != study.get("release_id")):
        raise VerificationError("Study commit identity is invalid or incomplete.")
    files = commit.get("files")
    if not isinstance(files, dict):
        raise VerificationError("Study commit lacks its file hash manifest.")
    for path in (artifacts.study_path, artifacts.predictions_path):
        if files.get(path.name) != _file_sha256(path):
            raise VerificationError(f"Committed input checksum failed: {path.name}.")


def _parse_int(value: Any, field: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise VerificationError(f"Invalid integer value in field {field}.")
    return int(text)


def _parse_grade(value: Any) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        raise VerificationError("OOF contains a non-numeric raw Glade value.")
    grade = int(text.split(".", maxsplit=1)[0])
    if grade not in TARGET_BY_GRADE:
        raise VerificationError("OOF contains a Glade value outside the confirmed {0,3,4,5} set.")
    return grade


def _parse_lens(value: Any) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"(4|10|40)(?:[xX×])?", text)
    if not match:
        raise VerificationError("OOF contains an unsupported objective lens value.")
    return int(match.group(1))


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    required = {
        "image_id", "file_name", "patient_id", "objective_lens", "raw_glade", "image_target",
        "outer_fold", "predicted_probability", "prediction_at_0_5", "selected_threshold",
        "prediction_at_selected_threshold",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise VerificationError("OOF CSV is missing required image-level prediction columns.")
            source_rows = list(reader)
    except OSError as exc:
        raise VerificationError("Unable to read the OOF prediction CSV.") from exc
    if not source_rows:
        raise VerificationError("OOF prediction CSV is empty.")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_names: list[str] = []
    for index, source in enumerate(source_rows, start=2):
        image_id = str(source["image_id"]).strip()
        file_name = str(source["file_name"]).strip()
        patient_id = str(source["patient_id"]).strip()
        if not image_id or not file_name or not patient_id:
            raise VerificationError(f"OOF row {index} has an empty image or patient identity.")
        if image_id in seen_ids:
            duplicate_ids.append(image_id)
        if file_name in seen_names:
            duplicate_names.append(file_name)
        seen_ids.add(image_id)
        seen_names.add(file_name)
        grade = _parse_grade(source["raw_glade"])
        target = _parse_int(source["image_target"], "image_target")
        expected_target = TARGET_BY_GRADE[grade]
        if target != expected_target:
            raise VerificationError("OOF image_target does not match its raw Glade grade mapping.")
        lens = _parse_lens(source["objective_lens"])
        probability = float(source["predicted_probability"])
        threshold = float(source["selected_threshold"])
        if (not math.isfinite(probability) or not 0 <= probability <= 1
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise VerificationError("OOF contains an invalid probability or selected threshold.")
        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "patient_id": patient_id,
            "objective_lens": lens,
            "raw_glade": grade,
            "image_target": target,
            "outer_fold": _parse_int(source["outer_fold"], "outer_fold"),
            "predicted_probability": probability,
            "prediction_at_0_5": _parse_int(source["prediction_at_0_5"], "prediction_at_0_5"),
            "selected_threshold": threshold,
            "prediction_at_selected_threshold": _parse_int(
                source["prediction_at_selected_threshold"], "prediction_at_selected_threshold",
            ),
        })
    if duplicate_ids:
        raise _fail_samples("image_id", duplicate_ids, "Duplicate OOF image IDs")
    if duplicate_names:
        raise _fail_samples("file_name", duplicate_names, "Duplicate OOF filenames")
    return rows


def _hash_patient_ids(patient_ids: Iterable[str]) -> list[str]:
    return sorted(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in set(patient_ids))


def _verify_hash_list(actual: Any, expected: list[str], label: str) -> None:
    if not isinstance(actual, list) or sorted(actual) != expected:
        raise VerificationError(f"{label} does not match the hashed patient IDs reproduced from OOF rows.")


def _array_sha256(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise VerificationError("Scaler audit vector is malformed or non-finite.")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _verify_study_and_splits(study: dict[str, Any], config: dict[str, Any],
                             rows: list[dict[str, Any]]) -> dict[str, Any]:
    if config != study.get("training_config"):
        raise VerificationError("training_config.json differs from the config bound into study.json.")
    if study.get("study_id") != _canonical_sha256(config):
        raise VerificationError("study_id is not the fingerprint of training_config.json.")
    if study.get("prediction_unit") != "original_image" or study.get("training_target_field") != "image_target":
        raise VerificationError("Study does not declare original-image predictions and targets.")
    if study.get("case_target_used") is not False or study.get("patch_labels_used") is not False:
        raise VerificationError("Study records case or patch labels in the original-image target path.")

    row_by_id = {row["image_id"]: row for row in rows}
    all_ids = set(row_by_id)
    patients = {row["patient_id"] for row in rows}
    folds = study.get("folds")
    requested_folds = _parse_int(config.get("outer_folds", 0), "outer_folds")
    if not isinstance(folds, list) or not 2 <= len(folds) <= requested_folds:
        raise VerificationError("Study must contain between two folds and its configured outer-fold count.")
    if study.get("outer_fold_count") != len(folds):
        raise VerificationError("Study outer_fold_count disagrees with its fold audits.")

    seen_test_ids: set[str] = set()
    fold_by_patient: dict[str, int] = {}
    fold_by_id: dict[str, int] = {}
    fold_audits: list[dict[str, Any]] = []
    fold_numbers: set[int] = set()
    inner_fold_count = _parse_int(config.get("inner_folds", 0), "inner_folds")
    for audit in folds:
        fold_number = _parse_int(audit.get("outer_fold"), "outer_fold")
        if fold_number in fold_numbers:
            raise VerificationError("Study contains duplicate outer fold numbers.")
        fold_numbers.add(fold_number)
        train_ids = audit.get("train_image_ids")
        test_ids = audit.get("test_image_ids")
        scaler_ids = audit.get("scaler_fit_image_ids")
        if not all(isinstance(value, list) for value in (train_ids, test_ids, scaler_ids)):
            raise VerificationError(f"Fold {fold_number} is missing image-ID audit lists.")
        train_set, test_set = set(train_ids), set(test_ids)
        if len(train_set) != len(train_ids) or len(test_set) != len(test_ids):
            raise VerificationError(f"Fold {fold_number} repeats an image ID in its partition audit.")
        if train_set & test_set or train_set | test_set != all_ids:
            raise VerificationError(f"Fold {fold_number} does not form a disjoint full-cohort partition.")
        if audit.get("train_image_count") != len(train_set) or audit.get("test_image_count") != len(test_set):
            raise VerificationError(f"Fold {fold_number} image counts disagree with its partition IDs.")
        if scaler_ids != train_ids:
            raise VerificationError(f"Fold {fold_number} scaler-fit IDs differ from its outer training rows.")
        if seen_test_ids & test_set:
            raise VerificationError("An OOF image appears in more than one outer test fold.")
        seen_test_ids.update(test_set)
        if fold_by_id and test_set & set(fold_by_id):
            raise VerificationError("Outer test image IDs overlap across folds.")

        train_patients = {row_by_id[image_id]["patient_id"] for image_id in train_set if image_id in row_by_id}
        test_patients = {row_by_id[image_id]["patient_id"] for image_id in test_set if image_id in row_by_id}
        if len(train_patients) + len(test_patients) == 0 or train_patients & test_patients:
            raise VerificationError(f"Fold {fold_number} leaks patients between train and test.")
        if audit.get("patient_overlap_count") != 0:
            raise VerificationError(f"Fold {fold_number} audit reports patient overlap.")
        _verify_hash_list(audit.get("train_patient_hashes"), _hash_patient_ids(train_patients),
                          f"Fold {fold_number} train_patient_hashes")
        _verify_hash_list(audit.get("test_patient_hashes"), _hash_patient_ids(test_patients),
                          f"Fold {fold_number} test_patient_hashes")
        for patient in test_patients:
            previous = fold_by_patient.get(patient)
            if previous is not None and previous != fold_number:
                raise VerificationError("Images from one patient appear in multiple outer test folds.")
            fold_by_patient[patient] = fold_number
        for image_id in test_set:
            if image_id not in row_by_id:
                raise VerificationError("Fold audit references an image absent from OOF rows.")
            row = row_by_id[image_id]
            if row["outer_fold"] != fold_number:
                raise VerificationError(f"Fold {fold_number} labels/predictions do not match OOF row assignments.")
            expected_05 = int(row["predicted_probability"] >= 0.5)
            expected_selected = int(row["predicted_probability"] >= row["selected_threshold"])
            if row["prediction_at_0_5"] != expected_05:
                raise VerificationError("Stored 0.5-threshold prediction disagrees with the probability.")
            if row["prediction_at_selected_threshold"] != expected_selected:
                raise VerificationError("Stored selected-threshold prediction disagrees with the probability.")
            if not math.isclose(float(audit.get("selected_threshold")), row["selected_threshold"], abs_tol=1e-12):
                raise VerificationError(f"Fold {fold_number} threshold does not match its OOF rows.")
            fold_by_id[image_id] = fold_number

        train_labels = {row_by_id[image_id]["image_target"] for image_id in train_set}
        test_labels = {row_by_id[image_id]["image_target"] for image_id in test_set}
        if train_labels != {0, 1} or test_labels != {0, 1}:
            raise VerificationError(f"Fold {fold_number} lacks both image-level classes in train or test.")

        if audit.get("scaler_mean_sha256") != _array_sha256(audit.get("scaler_mean", [])):
            raise VerificationError(f"Fold {fold_number} scaler mean hash does not match its serialized audit vector.")
        scale = np.asarray(audit.get("scaler_scale", []), dtype=np.float64)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise VerificationError(f"Fold {fold_number} scaler scales must be finite and positive.")
        if "scaler_scale_sha256" in audit and audit["scaler_scale_sha256"] != _array_sha256(scale):
            raise VerificationError(f"Fold {fold_number} scaler scale hash does not match its serialized audit vector.")
        feature_dim = _parse_int(audit.get("feature_dim"), "feature_dim")
        if len(audit.get("scaler_mean", [])) != feature_dim or len(audit.get("scaler_scale", [])) != feature_dim:
            raise VerificationError(f"Fold {fold_number} scaler audit dimension differs from feature_dim.")

        _verify_inner_splits(audit, train_set, row_by_id, fold_number, inner_fold_count)
        _verify_validation_selection(audit, fold_number)
        fold_audits.append({
            "fold": fold_number,
            "train_images": len(train_set),
            "test_images": len(test_set),
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "train_class_counts": {
                str(label): sum(row_by_id[image_id]["image_target"] == label for image_id in train_set)
                for label in (0, 1)
            },
            "test_class_counts": {
                str(label): sum(row_by_id[image_id]["image_target"] == label for image_id in test_set)
                for label in (0, 1)
            },
            "patient_overlap_count": 0,
            "scaler_fit_rows": len(scaler_ids),
        })

    if fold_numbers != set(range(1, len(folds) + 1)):
        raise VerificationError("Outer fold numbers are not a contiguous 1-based sequence.")
    if seen_test_ids != all_ids or len(fold_by_id) != len(all_ids):
        raise VerificationError("OOF test folds do not cover every image exactly once.")
    if set(fold_by_patient) != patients:
        raise VerificationError("Outer folds do not assign every patient to exactly one test fold.")
    if study.get("patient_count") != len(patients) or study.get("image_count") != len(rows):
        raise VerificationError("Study patient/image totals disagree with the OOF rows.")
    leakage = study.get("patient_split_leakage_check", {})
    if (leakage.get("passed") is not True or leakage.get("patient_overlap_count") != 0
            or leakage.get("outer_folds") != len(folds)):
        raise VerificationError("Study-level patient leakage check is absent or failed.")
    deployment = study.get("deployment_model", {})
    if deployment.get("scaler_fit_image_ids") != [row["image_id"] for row in rows]:
        raise VerificationError("Deployment scaler-fit rows do not cover the full training cohort in OOF order.")
    return {
        "outer_folds": len(folds),
        "patient_overlap_count": 0,
        "test_coverage_count": len(seen_test_ids),
        "folds": sorted(fold_audits, key=lambda item: item["fold"]),
        "scaler_fit_audits_passed": True,
        "inner_splits_checked": sum(len(audit.get("inner_validation_splits", [])) for audit in folds),
        "validation_selection_checks_passed": True,
    }


def _verify_inner_splits(audit: dict[str, Any], outer_train: set[str],
                         row_by_id: Mapping[str, dict[str, Any]], fold_number: int,
                         expected_count: int) -> None:
    splits = audit.get("inner_validation_splits")
    if not isinstance(splits, list) or not 2 <= len(splits) <= expected_count:
        raise VerificationError(f"Fold {fold_number} lacks inner grouped validation audits.")
    for split in splits:
        train_ids = split.get("train_image_ids")
        validation_ids = split.get("validation_image_ids")
        scaler_ids = split.get("scaler_fit_image_ids")
        if not all(isinstance(value, list) for value in (train_ids, validation_ids, scaler_ids)):
            raise VerificationError(f"Fold {fold_number} inner split is missing image-ID lists.")
        train_set, validation_set = set(train_ids), set(validation_ids)
        if (len(train_set) != len(train_ids) or len(validation_set) != len(validation_ids)
                or train_set & validation_set or train_set | validation_set != outer_train):
            raise VerificationError(f"Fold {fold_number} inner split is not a disjoint partition of outer training rows.")
        if scaler_ids != train_ids:
            raise VerificationError(f"Fold {fold_number} inner scaler-fit IDs differ from inner training rows.")
        train_patients = {row_by_id[image_id]["patient_id"] for image_id in train_set}
        validation_patients = {row_by_id[image_id]["patient_id"] for image_id in validation_set}
        if train_patients & validation_patients or split.get("patient_overlap_count") != 0:
            raise VerificationError(f"Fold {fold_number} inner split leaks patient groups.")
        _verify_hash_list(split.get("train_patient_hashes"), _hash_patient_ids(train_patients),
                          f"Fold {fold_number} inner train_patient_hashes")
        _verify_hash_list(split.get("validation_patient_hashes"), _hash_patient_ids(validation_patients),
                          f"Fold {fold_number} inner validation_patient_hashes")
        if {row_by_id[image_id]["image_target"] for image_id in train_set} != {0, 1}:
            raise VerificationError(f"Fold {fold_number} inner training split lacks a class.")
        if {row_by_id[image_id]["image_target"] for image_id in validation_set} != {0, 1}:
            raise VerificationError(f"Fold {fold_number} inner validation split lacks a class.")
    scaler_audit = audit.get("inner_validation_scaler_audit", {})
    if not isinstance(scaler_audit, dict):
        raise VerificationError(f"Fold {fold_number} inner scaler summary is malformed.")
    expected_rows = {split["split"]: len(split["scaler_fit_image_ids"]) for split in splits}
    for c_key, c_audits in scaler_audit.items():
        if not isinstance(c_audits, list) or len(c_audits) != len(splits):
            raise VerificationError(f"Fold {fold_number} inner scaler audit count is invalid for C={c_key}.")
        for item in c_audits:
            split_number = _parse_int(item.get("split"), "inner_split")
            if item.get("scaler_fit_row_count") != expected_rows.get(split_number):
                raise VerificationError(f"Fold {fold_number} inner scaler row count does not match its train IDs.")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("scaler_mean_sha256", ""))):
                raise VerificationError(f"Fold {fold_number} inner scaler mean hash is missing or malformed.")


def _verify_validation_selection(audit: dict[str, Any], fold_number: int) -> None:
    scores = audit.get("c_validation_scores")
    if not isinstance(scores, list) or not scores:
        raise VerificationError(f"Fold {fold_number} lacks inner-validation C scores.")
    candidates: list[tuple[float, float]] = []
    by_c: dict[float, dict[str, Any]] = {}
    for item in scores:
        c_value = float(item["C"])
        auc_value = item.get("validation_roc_auc")
        auc = float(auc_value) if auc_value is not None else float("nan")
        if not math.isfinite(c_value) or c_value <= 0 or (auc_value is not None and not math.isfinite(auc)):
            raise VerificationError(f"Fold {fold_number} has invalid C-selection scores.")
        if c_value in by_c:
            raise VerificationError(f"Fold {fold_number} repeats a candidate C value.")
        by_c[c_value] = item
        candidates.append((c_value, auc))
    selected_c = float(audit.get("selected_C"))
    if selected_c not in by_c:
        raise VerificationError(f"Fold {fold_number} selected C is absent from validation results.")
    best_auc = max((auc if math.isfinite(auc) else -1.0) for _, auc in candidates)
    expected_c = min(c for c, auc in candidates if (auc if math.isfinite(auc) else -1.0) == best_auc)
    if selected_c != expected_c:
        raise VerificationError(f"Fold {fold_number} selected C does not match grouped validation AUROC.")
    selected_entry = by_c[selected_c]
    if not math.isclose(float(audit.get("selected_threshold")),
                        float(selected_entry["validation_selected_threshold"]), abs_tol=1e-12):
        raise VerificationError(f"Fold {fold_number} threshold is not the selected-C validation threshold.")
    if not math.isclose(float(audit.get("selected_C_validation_roc_auc")),
                        float(selected_entry["validation_roc_auc"]), abs_tol=1e-12):
        raise VerificationError(f"Fold {fold_number} selected-C validation AUROC audit is inconsistent.")
    if not math.isclose(float(audit.get("threshold_selection_validation_balanced_accuracy")),
                        float(selected_entry["validation_balanced_accuracy_at_selected_threshold"]),
                        abs_tol=1e-12):
        raise VerificationError(f"Fold {fold_number} validation threshold score is inconsistent.")


def _metric_record(labels: np.ndarray, probabilities: np.ndarray,
                   predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    negatives, positives = tn + fp, tp + fn
    sensitivity = float(tp / positives) if positives else None
    specificity = float(tn / negatives) if negatives else None
    balanced = ((sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)
    roc_auc = float(roc_auc_score(labels, probabilities)) if positives and negatives else None
    auprc = float(average_precision_score(labels, probabilities)) if positives else None
    return {
        "image_count": int(len(labels)),
        "positive_images": positives,
        "negative_images": negatives,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float((tn + tp) / len(labels)) if len(labels) else None,
        "balanced_accuracy": balanced,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _close_value(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise VerificationError(f"Saved metric {label} differs from its recomputed value.")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise VerificationError(f"Saved metric {label} differs from its recomputed value.")


def _verify_saved_metrics(study: dict[str, Any], computed: dict[str, Any]) -> None:
    saved = study.get("metrics")
    if not isinstance(saved, dict) or not isinstance(saved.get("pooled"), dict):
        raise VerificationError("Study is missing its saved pooled OOF metrics.")
    def compare_record(saved_record: Any, expected_record: dict[str, Any], label: str) -> None:
        if not isinstance(saved_record, dict):
            raise VerificationError(f"Study is missing saved metric record {label}.")
        for key in ("image_count", "positive_images", "negative_images", "confusion_matrix",
                    "accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity"):
            if key == "confusion_matrix":
                if saved_record.get(key) != expected_record[key]:
                    raise VerificationError(f"Saved confusion matrix {label} differs from recomputation.")
            elif key in {"image_count", "positive_images", "negative_images"}:
                if saved_record.get(key) != expected_record[key]:
                    raise VerificationError(f"Saved count {label}.{key} differs from recomputation.")
            else:
                _close_value(saved_record.get(key), expected_record[key], f"{label}.{key}")

    saved_pooled = saved["pooled"]
    for subset in ("fixed_threshold_0_5", "validation_selected_threshold"):
        compare_record(saved_pooled.get(subset), computed["pooled"][subset], f"pooled.{subset}")
    saved_thresholds = saved_pooled.get("selected_thresholds_by_fold", {})
    for fold, expected_threshold in computed["pooled"]["selected_thresholds_by_fold"].items():
        _close_value(saved_thresholds.get(fold), expected_threshold, f"pooled.selected_thresholds_by_fold.{fold}")

    saved_by_lens = saved.get("by_objective_lens")
    if not isinstance(saved_by_lens, dict):
        raise VerificationError("Study is missing per-objective-lens OOF metrics.")
    for lens, lens_metrics in computed["by_objective_lens"].items():
        saved_lens = saved_by_lens.get(lens)
        if not isinstance(saved_lens, dict):
            raise VerificationError(f"Study is missing saved metrics for lens {lens}.")
        for subset in ("fixed_threshold_0_5", "validation_selected_threshold"):
            compare_record(saved_lens.get(subset), lens_metrics[subset], f"lens-{lens}.{subset}")


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["image_target"] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["predicted_probability"] for row in rows], dtype=np.float64)
    fixed = np.asarray([row["prediction_at_0_5"] for row in rows], dtype=np.int64)
    selected = np.asarray([row["prediction_at_selected_threshold"] for row in rows], dtype=np.int64)
    pooled = {
        "fixed_threshold_0_5": _metric_record(labels, probabilities, fixed),
        "validation_selected_threshold": _metric_record(labels, probabilities, selected),
        "selected_thresholds_by_fold": {
            str(fold): float(next(row["selected_threshold"] for row in rows if row["outer_fold"] == fold))
            for fold in sorted({row["outer_fold"] for row in rows})
        },
    }
    by_lens: dict[str, Any] = {}
    for lens in sorted({row["objective_lens"] for row in rows}):
        selected_rows = [row for row in rows if row["objective_lens"] == lens]
        sub_y = np.asarray([row["image_target"] for row in selected_rows], dtype=np.int64)
        sub_p = np.asarray([row["predicted_probability"] for row in selected_rows], dtype=np.float64)
        by_lens[str(lens)] = {
            "fixed_threshold_0_5": _metric_record(
                sub_y, sub_p, np.asarray([row["prediction_at_0_5"] for row in selected_rows], dtype=np.int64),
            ),
            "validation_selected_threshold": _metric_record(
                sub_y, sub_p, np.asarray([row["prediction_at_selected_threshold"] for row in selected_rows], dtype=np.int64),
            ),
        }
    return {"pooled": pooled, "by_objective_lens": by_lens}


def _bootstrap_patient_ci(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any] | None:
    if replicates == 0:
        return None
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_patient[row["patient_id"]].append(index)
    patient_ids = sorted(by_patient)
    if len(patient_ids) < 2:
        raise VerificationError("Patient-group bootstrap requires at least two independent patients.")
    rng = np.random.default_rng(seed)
    records: dict[str, list[float]] = {"roc_auc": [], "sensitivity": [], "specificity": []}
    skipped = 0
    for _ in range(replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        indices = np.concatenate([np.asarray(by_patient[patient], dtype=np.int64) for patient in sampled])
        labels = np.asarray([rows[index]["image_target"] for index in indices], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            skipped += 1
            continue
        probabilities = np.asarray([rows[index]["predicted_probability"] for index in indices], dtype=np.float64)
        predictions = np.asarray([rows[index]["prediction_at_selected_threshold"] for index in indices], dtype=np.int64)
        metrics = _metric_record(labels, probabilities, predictions)
        for key in records:
            value = metrics[key]
            if value is not None and math.isfinite(float(value)):
                records[key].append(float(value))
    intervals = {}
    for key, values in records.items():
        if not values:
            intervals[key] = None
        else:
            low, high = np.quantile(np.asarray(values), (0.025, 0.975))
            intervals[key] = {"low_95": float(low), "high_95": float(high)}
    return {
        "unit": "patient_group",
        "replicates_requested": replicates,
        "replicates_valid": len(records["roc_auc"]),
        "replicates_skipped_single_class": skipped,
        "method": "Resample patients with replacement; reuse saved OOF probabilities and fold thresholds; do not refit models.",
        "intervals": intervals,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _plot_roc(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=True)
    groups: list[tuple[str, list[dict[str, Any]]]] = [("Gộp", rows)]
    groups.extend((f"{lens}×", [row for row in rows if row["objective_lens"] == lens])
                  for lens in (4, 10, 40))
    colors = ("#203864", "#4472C4", "#70AD47", "#ED7D31")
    for (name, selected), color in zip(groups, colors, strict=True):
        labels = np.asarray([row["image_target"] for row in selected], dtype=np.int64)
        probabilities = np.asarray([row["predicted_probability"] for row in selected], dtype=np.float64)
        if set(labels.tolist()) != {0, 1}:
            continue
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc_value = float(roc_auc_score(labels, probabilities))
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name} (n={len(selected):,}; AUROC={auc_value:.3f})")
    ax.plot((0, 1), (0, 1), linestyle="--", color="#808080", linewidth=1.0, label="Ngẫu nhiên")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Tỷ lệ dương tính giả",
           ylabel="Độ nhạy", title="ROC từ dự đoán ngoài fold theo ảnh gốc")
    ax.grid(True, color="#D9E1F2", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", metadata={"Title": "ROC OOF ảnh gốc"})
    plt.close(fig)


def _plot_precision_recall(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Display ranking quality alongside the pooled positive-class prevalence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=True)
    groups = [("Gộp", rows)]
    groups.extend((f"{lens}×", [row for row in rows if row["objective_lens"] == lens]) for lens in (4, 10, 40))
    for (name, selected), color in zip(groups, ("#203864", "#4472C4", "#70AD47", "#ED7D31"), strict=True):
        labels = np.asarray([row["image_target"] for row in selected], dtype=np.int64)
        probability = np.asarray([row["predicted_probability"] for row in selected], dtype=np.float64)
        if set(labels.tolist()) != {0, 1}:
            continue
        precision, recall, _ = precision_recall_curve(labels, probability)
        ap = float(average_precision_score(labels, probability))
        ax.step(recall, precision, where="post", color=color, linewidth=2, label=f"{name} (AP={ap:.3f})")
    prevalence = sum(row["image_target"] for row in rows) / len(rows)
    ax.axhline(prevalence, color="#808080", linestyle="--", label=f"Tỷ lệ ảnh dương gộp: {prevalence:.1%}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Độ nhạy", ylabel="Precision",
           title="Precision–recall từ dự đoán ngoài fold theo ảnh gốc")
    ax.grid(True, color="#D9E1F2", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray([row["image_target"] for row in rows], dtype=np.int64)
    fixed = np.asarray([row["prediction_at_0_5"] for row in rows], dtype=np.int64)
    selected = np.asarray([row["prediction_at_selected_threshold"] for row in rows], dtype=np.int64)
    matrices = []
    for predictions in (fixed, selected):
        matrices.append(np.asarray([
            [np.sum((labels == 0) & (predictions == 0)), np.sum((labels == 0) & (predictions == 1))],
            [np.sum((labels == 1) & (predictions == 0)), np.sum((labels == 1) & (predictions == 1))],
        ], dtype=np.int64))
    titles = ("Ngưỡng cố định 0,5", "Ngưỡng chọn trên validation theo fold")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, titles, strict=True):
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_title(title)
        axis.set_xticks((0, 1), labels=("Không ung thư", "Ung thư"))
        axis.set_yticks((0, 1), labels=("Không ung thư", "Ung thư"))
        axis.set_xlabel("Dự đoán")
        axis.set_ylabel("Nhãn bác sĩ")
        midpoint = max(1, int(matrix.max())) / 2
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:,}",
                          ha="center", va="center",
                          color="white" if matrix[row_index, column_index] > midpoint else "#172033",
                          fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.84, label="Số ảnh")
    fig.suptitle("Ma trận nhầm lẫn OOF theo ảnh gốc", fontsize=13)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", metadata={"Title": "Confusion OOF ảnh gốc"})
    plt.close(fig)


def _markdown_report(verification: dict[str, Any]) -> str:
    counts = verification["counts"]
    metrics = verification["metrics"]
    pooled = metrics["pooled"]
    lines = [
        "# Báo cáo kiểm chứng phân loại ảnh gốc",
        "",
        "Nhãn là cấp ảnh gốc theo `Glade`: 0 = không ung thư; 3/4/5 = ung thư. Mã người bệnh chỉ dùng để chia train/validation/test theo nhóm.",
        "",
        f"Đã kiểm chứng {counts['image_count']:,} dự đoán ngoài fold từ {counts['patient_count']} người bệnh độc lập.",
        f"Số ảnh âm: {counts['labels']['0']:,}; số ảnh dương: {counts['labels']['1']:,}.",
        "",
        "| Vật kính | Số ảnh | AUROC | AUPRC | Độ nhạy (0,5) | Độ đặc hiệu (0,5) | Độ nhạy (ngưỡng validation) | Độ đặc hiệu (ngưỡng validation) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_lens = sorted(metrics["by_objective_lens"], key=int)
    for lens in ordered_lens:
        entry = metrics["by_objective_lens"][lens]
        fixed = entry["fixed_threshold_0_5"]
        selected = entry["validation_selected_threshold"]
        lines.append(
            f"| {lens}× | {fixed['image_count']:,} | {_fmt(fixed['roc_auc'])} | {_fmt(fixed['auprc'])} "
            f"| {_fmt(fixed['sensitivity'])} | {_fmt(fixed['specificity'])} "
            f"| {_fmt(selected['sensitivity'])} | {_fmt(selected['specificity'])} |"
        )
    fixed = pooled["fixed_threshold_0_5"]
    selected = pooled["validation_selected_threshold"]
    lines.extend([
        f"| **Gộp** | **{fixed['image_count']:,}** | **{_fmt(fixed['roc_auc'])}** | **{_fmt(fixed['auprc'])}** "
        f"| **{_fmt(fixed['sensitivity'])}** | **{_fmt(fixed['specificity'])}** "
        f"| **{_fmt(selected['sensitivity'])}** | **{_fmt(selected['specificity'])}** |",
        "",
        "Ma trận nhầm lẫn gộp được tính với ngưỡng cố định 0,5 và ngưỡng validation riêng của từng outer fold. ROC, ma trận và các chỉ số được tính lại từ CSV OOF; kết quả phải khớp các chỉ số đã lưu trong `study.json`.",
        "AUPRC được tính theo average precision với nội suy bậc thang.",
        "",
        "![ROC OOF theo ảnh và vật kính](roc_oof.png)",
        "",
        "![Precision–recall OOF](precision_recall_oof.png)",
        "",
        "![Ma trận nhầm lẫn OOF](confusion_selected_threshold.png)",
        "",
    ])
    bootstrap = metrics.get("patient_group_bootstrap_ci")
    if bootstrap is not None:
        lines.extend([
            "## Khoảng tin cậy bootstrap theo người bệnh",
            "",
            f"Đã dùng {bootstrap['replicates_valid']:,}/{bootstrap['replicates_requested']:,} mẫu hợp lệ; đơn vị lấy mẫu là người bệnh.",
            "Các mẫu bootstrap tái sử dụng xác suất OOF và ngưỡng validation đã lưu; mô hình không được fit lại.",
            "",
            "| Chỉ số | Cận dưới 95% | Cận trên 95% |",
            "|---|---:|---:|",
        ])
        for key, label in (("roc_auc", "AUROC"), ("sensitivity", "Độ nhạy"), ("specificity", "Độ đặc hiệu")):
            interval = bootstrap["intervals"].get(key)
            if interval is None:
                lines.append(f"| {label} | — | — |")
            else:
                lines.append(f"| {label} | {_fmt(interval['low_95'])} | {_fmt(interval['high_95'])} |")
        lines.append("")
    lines.extend([
        "## Giới hạn diễn giải",
        "",
        "Đây là ước lượng nội bộ trên dự đoán ngoài fold theo **ảnh gốc**. Số người bệnh độc lập là giới hạn suy luận; kết quả không xác lập hiệu quả lâm sàng. Không so sánh trực tiếp AUROC này với AUROC của mô hình cấp ca E4 vì đơn vị dự đoán và thiết kế đánh giá khác nhau.",
        "",
        f"Kiểm tra nhóm bệnh nhân: {verification['checks']['outer_folds']} fold; không có bệnh nhân giao giữa train và test; {verification['checks']['inner_splits_checked']} inner split đã được kiểm tra.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def analyze_study(
    input_dir: Path,
    report_dir: Path,
    *,
    expected: ExpectedCounts = DEFAULT_EXPECTED_COUNTS,
    bootstrap_replicates: int = 1000,
    seed: int = 20261012,
) -> dict[str, Any]:
    """Verify private artifacts and produce aggregate JSON, Markdown and PNGs."""
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative.")
    artifacts = StudyArtifacts.from_private_dir(input_dir)
    _check_private_permissions(artifacts)
    study = _read_json(artifacts.study_path)
    config = _read_json(artifacts.config_path)
    _verify_input_commit(artifacts, study)
    rows = _read_predictions(artifacts.predictions_path)

    image_count = len(rows)
    patient_ids = {row["patient_id"] for row in rows}
    label_counts = Counter(row["image_target"] for row in rows)
    grade_counts = Counter(row["raw_glade"] for row in rows)
    lens_counts = Counter(row["objective_lens"] for row in rows)
    if image_count != expected.image_count:
        raise VerificationError(f"Expected {expected.image_count} OOF images; found {image_count}.")
    if len(patient_ids) != expected.patient_count:
        raise VerificationError(f"Expected {expected.patient_count} distinct patients; found {len(patient_ids)}.")
    if dict(label_counts) != expected.class_counts:
        raise VerificationError(f"Image-level class counts mismatch: expected {expected.class_counts}, found {dict(label_counts)}.")
    if dict(lens_counts) != expected.lens_count_map:
        raise VerificationError(f"Objective-lens counts mismatch: expected {expected.lens_count_map}, found {dict(lens_counts)}.")
    if set(grade_counts) != set(TARGET_BY_GRADE):
        raise VerificationError("OOF must contain exactly the confirmed Glade grades 0, 3, 4 and 5.")
    if study.get("image_count") != image_count or study.get("patient_count") != len(patient_ids):
        raise VerificationError("study.json cohort totals disagree with the OOF prediction rows.")
    raw_class_counts = study.get("class_counts")
    if not isinstance(raw_class_counts, dict):
        raise VerificationError("study.json class_counts field is missing or malformed.")
    saved_class_counts = {int(key): int(value) for key, value in raw_class_counts.items()}
    if saved_class_counts != dict(label_counts):
        raise VerificationError("study.json class counts disagree with the OOF prediction rows.")

    split_checks = _verify_study_and_splits(study, config, rows)
    metrics = _compute_metrics(rows)
    _verify_saved_metrics(study, metrics)
    bootstrap = _bootstrap_patient_ci(rows, bootstrap_replicates, seed)
    metrics["patient_group_bootstrap_ci"] = bootstrap
    checks = {
        **split_checks,
        "glade_target_mapping_passed": True,
        "unique_image_rows_passed": True,
        "count_profile_passed": True,
        "saved_metrics_reproduced": True,
        "owner_only_private_inputs": True,
        "study_commit_verified": True,
        "serialized_scaler_scales_covered_by_study_hash": True,
    }
    verification = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_unit": "original_image",
        "label_source": "raw_glade",
        "target_mapping": {str(key): value for key, value in TARGET_BY_GRADE.items()},
        "metric_definitions": {
            "auroc": "sklearn.metrics.roc_auc_score",
            "auprc": "sklearn.metrics.average_precision_score (stepwise interpolation)",
            "thresholds": "0.5 fixed and saved per-outer-fold validation-selected threshold",
        },
        "input_sha256": {
            artifacts.study_path.name: _file_sha256(artifacts.study_path),
            artifacts.predictions_path.name: _file_sha256(artifacts.predictions_path),
            artifacts.config_path.name: _file_sha256(artifacts.config_path),
            artifacts.commit_path.name: _file_sha256(artifacts.commit_path),
        },
        "counts": {
            "image_count": image_count,
            "patient_count": len(patient_ids),
            "labels": {str(key): int(label_counts.get(key, 0)) for key in (0, 1)},
            "grades": {str(key): int(grade_counts.get(key, 0)) for key in sorted(TARGET_BY_GRADE)},
            "objective_lens": {str(key): int(lens_counts.get(key, 0)) for key in (4, 10, 40)},
        },
        "checks": checks,
        "metrics": metrics,
        "bootstrap_interpretation": (
            "Conditional on saved OOF models and predictions; patient groups resampled with replacement; models not refit."
            if bootstrap is not None else None
        ),
        "clinical_validity_established": False,
        "case_level_e4_comparison_performed": False,
    }

    report_path = Path(report_dir).expanduser()
    if report_path.is_symlink():
        raise VerificationError("Report output path cannot be a symlink.")
    report_root = report_path.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    _plot_roc(rows, report_root / "roc_oof.png")
    _plot_confusion(rows, report_root / "confusion_selected_threshold.png")
    _plot_precision_recall(rows, report_root / "precision_recall_oof.png")
    markdown = _markdown_report(verification)
    _atomic_write(report_root / "verification.json",
                  json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
    _atomic_write(report_root / "report_vi.md", markdown.encode("utf-8"))
    return verification


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_EXECUTION_ROOT / "private")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_EXECUTION_ROOT / "report")
    parser.add_argument("--expected-images", type=int, default=2802)
    parser.add_argument("--expected-patients", type=int, default=18)
    parser.add_argument("--expected-negative", type=int, default=2298)
    parser.add_argument("--expected-positive", type=int, default=504)
    parser.add_argument("--expected-lens-4", type=int, default=364)
    parser.add_argument("--expected-lens-10", type=int, default=830)
    parser.add_argument("--expected-lens-40", type=int, default=1608)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    expected = ExpectedCounts(
        image_count=args.expected_images,
        patient_count=args.expected_patients,
        negative_images=args.expected_negative,
        positive_images=args.expected_positive,
        lens_counts=((4, args.expected_lens_4), (10, args.expected_lens_10), (40, args.expected_lens_40)),
    )
    try:
        result = analyze_study(args.input_dir, args.report_dir, expected=expected,
                               bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    except (VerificationError, ValueError, OSError, ImportError) as exc:
        print(f"Study analysis stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "verified",
        "report_dir": str(Path(args.report_dir).resolve()),
        "image_count": result["counts"]["image_count"],
        "patient_count": result["counts"]["patient_count"],
        "outer_folds": result["checks"]["outer_folds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
