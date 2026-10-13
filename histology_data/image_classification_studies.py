"""Patient-locked head/weight/PCA studies on verified original-image features.

This module never changes source images, image targets, baseline assignments,
or feature shards. Its versioned protocol is independent of legacy LR studies.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import image_classification as core
from .io import atomic_json, file_hash, fingerprint

PROTOCOL = "original-image-fixed-patient-ablation-v1"
WEIGHT_SCHEMES = ("patient_class", "class", "none")


@dataclass(frozen=True)
class StudyConfig:
    """All statistical choices enter study identity; deadlines do not."""

    head_mode: str = "global"
    weight_scheme: str = "patient_class"
    pca_components: int | None = None
    lambda_values: tuple[float, ...] = (50.0, 5.0, 0.5, 0.05)
    inner_folds: int = 2
    seed: int = 17
    max_iter: int = 2000
    budget_minutes: float = 60.0
    reserve_minutes: float = 10.0
    recover_lock: bool = False

    def validate(self) -> None:
        if self.head_mode not in {"global", "per_lens"} or self.weight_scheme not in WEIGHT_SCHEMES:
            raise ValueError("Unknown head mode or weighting objective.")
        if self.pca_components is not None and (isinstance(self.pca_components, bool)
                                               or not isinstance(self.pca_components, int)
                                               or self.pca_components < 1):
            raise ValueError("PCA dimension must be a positive integer or None.")
        if self.inner_folds != 2:
            raise ValueError("This comparison protocol fixes inner_folds=2.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if (not self.lambda_values or len(set(self.lambda_values)) != len(self.lambda_values)
                or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in self.lambda_values)):
            raise ValueError("Regularization lambdas must be distinct finite positive values.")
        if (not math.isfinite(self.budget_minutes) or not math.isfinite(self.reserve_minutes)
                or not 0 <= self.reserve_minutes < self.budget_minutes):
            raise ValueError("Invalid study budget/reserve.")


@dataclass
class StudyData:
    """Read once on Colab; multiple studies share this immutable feature source."""

    run_root: Path
    features: np.ndarray
    rows: list[dict[str, Any]]
    release: dict[str, Any]
    baseline: dict[str, Any]
    outer: list[tuple[np.ndarray, np.ndarray]]
    assignment_sha256: str


def load_study_data(run_root: Path, baseline_root: Path) -> StudyData:
    """Verify source and replay exact baseline outer image/patient assignments."""
    features, rows, release = core._load_feature_matrix(run_root)
    baseline, _ = core._verify_study(baseline_root)
    core._audit_original_image_content(run_root, rows)
    if (baseline.get("release_id") != release["release_id"]
            or baseline.get("prediction_unit") != "original_image"
            or baseline.get("case_target_used") is not False
            or baseline.get("patch_labels_used") is not False):
        raise ValueError("Baseline does not belong to this original-image feature release.")
    positions = {row["image_id"]: i for i, row in enumerate(rows)}
    universe = set(positions)
    assignments: dict[str, int] = {}
    patient_fold: dict[str, int] = {}
    outer = []
    folds = sorted(baseline["folds"], key=lambda f: f["outer_fold"])
    if [f["outer_fold"] for f in folds] != [1, 2, 3]:
        raise ValueError("Exactly three baseline outer folds are required.")
    for fold in folds:
        train_ids, test_ids = fold["train_image_ids"], fold["test_image_ids"]
        train, test = set(train_ids), set(test_ids)
        if (len(train) != len(train_ids) or len(test) != len(test_ids)
                or train & test or train | test != universe or set(assignments) & test):
            raise ValueError("Baseline outer assignments are not exhaustive and disjoint.")
        ti = np.array([positions[x] for x in train_ids], dtype=np.int64)
        vi = np.array([positions[x] for x in test_ids], dtype=np.int64)
        train_patients = {rows[i]["patient_id"] for i in ti}
        if train_patients & {rows[i]["patient_id"] for i in vi}:
            raise ValueError("Baseline outer assignments leak patients.")
        for i in vi:
            row = rows[i]
            number = fold["outer_fold"]
            previous = patient_fold.setdefault(row["patient_id"], number)
            if previous != number:
                raise ValueError("One patient occurs in multiple baseline outer test folds.")
            assignments[row["image_id"]] = number
        outer.append((ti, vi))
    if set(assignments) != universe:
        raise ValueError("Baseline outer tests omit feature images.")
    return StudyData(Path(run_root).resolve(), features, rows, release, baseline, outer,
                     fingerprint(dict(sorted(assignments.items()))))


def sample_weights(labels: np.ndarray, patients: np.ndarray, scheme: str) -> np.ndarray:
    """Each scheme has mass N; relative objectives remain explicitly different."""
    if set(labels.tolist()) != {0, 1} or len(labels) != len(patients):
        raise ValueError("Weighting requires both classes and aligned patient rows.")
    if scheme == "patient_class":
        weights = core._patient_sample_weights(labels, patients)
    elif scheme == "class":
        counts = Counter(labels.tolist())
        weights = np.array([1.0 / counts[int(label)] for label in labels], dtype=np.float64)
    elif scheme == "none":
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        raise ValueError("Unknown weighting scheme.")
    weights = weights * (len(labels) / weights.sum())
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Non-finite or nonpositive sample weights.")
    return weights


def _arrays(data: StudyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([r["image_target"] for r in data.rows], dtype=np.int64),
            np.array([r["patient_id"] for r in data.rows], dtype=object),
            np.array([r["objective_lens"] for r in data.rows], dtype=np.int64))


def _heads(config: StudyConfig) -> tuple[str, ...]:
    return ("all",) if config.head_mode == "global" else ("4", "10", "40")


def _scope(indices: np.ndarray, lenses: np.ndarray, head: str) -> np.ndarray:
    return indices if head == "all" else indices[lenses[indices] == int(head)]


def _preprocess(x: np.ndarray, indices: np.ndarray, config: StudyConfig,
                seed: int) -> tuple[list[tuple[str, Any]], np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if config.pca_components is not None and config.pca_components > min(len(indices), x.shape[1]):
        raise ValueError("PCA dimension is infeasible for this training subset; refusing silent clipping.")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(x[indices])
    steps: list[tuple[str, Any]] = [("scaler", scaler)]
    if config.pca_components is not None:
        pca = PCA(n_components=config.pca_components, svd_solver="randomized", random_state=seed)
        transformed = pca.fit_transform(transformed)
        steps.append(("pca", pca))
    return steps, transformed


def _fit_classifier(z: np.ndarray, y: np.ndarray, weights: np.ndarray,
                    penalty: float, config: StudyConfig, seed: int) -> Any:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(C=1.0 / (penalty * float(weights.sum())),
                                   max_iter=config.max_iter, solver="lbfgs", random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(z, y, sample_weight=weights)
    return estimator


def _fit_head(data: StudyData, indices: np.ndarray, penalty: float, config: StudyConfig,
              seed: int) -> tuple[Any, dict[str, Any]]:
    from sklearn.pipeline import Pipeline

    labels, patients, _ = _arrays(data)
    steps, transformed = _preprocess(data.features, indices, config, seed)
    weights = sample_weights(labels[indices], patients[indices], config.weight_scheme)
    estimator = _fit_classifier(transformed, labels[indices], weights, penalty, config, seed)
    pipeline = Pipeline([*steps, ("classifier", estimator)])
    ids = [data.rows[i]["image_id"] for i in indices]
    audit = {"fit_image_ids": ids, "scaler_fit_image_ids": ids,
             "pca_fit_image_ids": ids if config.pca_components is not None else [],
             "pca_components": config.pca_components, "feature_dim": data.features.shape[1],
             "weight_scheme": config.weight_scheme, "weight_sum": float(weights.sum()),
             "weight_mean": float(weights.mean()),
             "weight_sum_by_class": {str(k): float(weights[labels[indices] == k].sum()) for k in (0, 1)},
             "lambda": penalty, "effective_C": estimator.C,
             "solver_iterations": estimator.n_iter_.tolist()}
    return pipeline, audit


def _tune_head(data: StudyData, indices: np.ndarray, config: StudyConfig, seed: int,
               deadline: float) -> dict[str, Any]:
    from sklearn.pipeline import Pipeline

    labels, patients, _ = _arrays(data)
    inner = core._grouped_folds(labels[indices], patients[indices], requested_folds=2,
                               seed=seed, context="fixed-outer head inner validation")
    scores = {value: np.full(len(indices), np.nan) for value in config.lambda_values}
    audits = []
    for split, (local_train, local_valid) in enumerate(inner, start=1):
        core._check_deadline(deadline)
        train, valid = indices[local_train], indices[local_valid]
        steps, z = _preprocess(data.features, train, config, seed + split)
        preprocessing = Pipeline(steps)
        z_valid = preprocessing.transform(data.features[valid])
        weights = sample_weights(labels[train], patients[train], config.weight_scheme)
        ids = [data.rows[i]["image_id"] for i in train]
        audits.append({"split": split, "train_image_ids": ids,
                       "validation_image_ids": [data.rows[i]["image_id"] for i in valid],
                       "scaler_fit_image_ids": ids,
                       "pca_fit_image_ids": ids if config.pca_components is not None else [],
                       "weight_sum": float(weights.sum()), "weight_mean": float(weights.mean()),
                       "patient_overlap_count": len(set(patients[train]) & set(patients[valid]))})
        for value in config.lambda_values:
            core._check_deadline(deadline)
            estimator = _fit_classifier(z, labels[train], weights, value, config, seed + split)
            scores[value][local_valid] = estimator.predict_proba(z_valid)[:, 1]
    candidates = []
    for value, prediction in scores.items():
        if not np.isfinite(prediction).all():
            raise ValueError("Inner OOF coverage is incomplete.")
        metrics = core._metrics(labels[indices], prediction, prediction >= 0.5)
        threshold, bacc = core._choose_threshold(labels[indices], prediction)
        candidates.append({"lambda": value, "validation_roc_auc": metrics["roc_auc"],
                           "threshold": threshold, "validation_balanced_accuracy": bacc})
    # Larger lambda is stronger regularization, equivalent to smaller legacy C.
    chosen = sorted(candidates, key=lambda r: (-r["validation_roc_auc"], -r["lambda"]))[0]
    return {**chosen, "candidates": candidates, "inner_splits": audits}


def preflight(data: StudyData, config: StudyConfig) -> list[dict[str, Any]]:
    """Check each lens/head outer and inner support before publishing a study."""
    config.validate()
    labels, patients, lenses = _arrays(data)
    support = []
    for fold, (train, test) in enumerate(data.outer, start=1):
        for head in _heads(config):
            ti, vi = _scope(train, lenses, head), _scope(test, lenses, head)
            if set(labels[ti].tolist()) != {0, 1} or set(labels[vi].tolist()) != {0, 1}:
                raise ValueError(f"Attention required: fold {fold}, head {head} lacks two-class support.")
            inner = core._grouped_folds(labels[ti], patients[ti], requested_folds=2,
                                       seed=config.seed + fold * 100 + (0 if head == "all" else int(head)),
                                       context="head support preflight")
            smallest = min(len(t) for t, _ in inner)
            if config.pca_components is not None and config.pca_components > min(smallest, data.features.shape[1]):
                raise ValueError(f"Attention required: PCA exceeds inner training support in fold {fold}, head {head}.")
            support.append({"fold": fold, "head": head, "train_images": len(ti), "test_images": len(vi),
                            "train_patients": len(set(patients[ti])), "test_patients": len(set(patients[vi])),
                            "train_class_counts": {str(k): int(np.sum(labels[ti] == k)) for k in (0, 1)},
                            "test_class_counts": {str(k): int(np.sum(labels[vi] == k)) for k in (0, 1)},
                            "smallest_inner_train": smallest})
    return support


def _descriptor(data: StudyData, config: StudyConfig) -> dict[str, Any]:
    import scipy
    import sklearn

    statistical = asdict(config)
    for key in ("budget_minutes", "reserve_minutes", "recover_lock"):
        statistical.pop(key)
    statistical = json.loads(json.dumps(statistical))
    return {"protocol": PROTOCOL, "release_id": data.release["release_id"],
            "baseline_study_id": data.baseline["study_id"],
            "assignment_sha256": data.assignment_sha256, "config": statistical,
            "weight_scale": "mass equals subset image count",
            "penalty": "effective_C=1/(lambda*sum_sample_weights)",
            "selection": "pooled inner OOF AUROC; threshold inner OOF balanced accuracy",
            "scaler": "train-only unweighted StandardScaler",
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "implementation_sha256": file_hash(Path(__file__)),
            "core_sha256": file_hash(Path(core.__file__))}


def _verified_fold(data: StudyData, root: Path, study_id: str, fold: int,
                   train: np.ndarray, test: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit, predictions = core._verify_fold_artifact(root, study_id)
    descriptor = json.loads((root.parents[1] / "config.json").read_text())
    if fingerprint(descriptor) != study_id:
        raise ValueError("Fold config identity differs from its study.")
    _validate_routing(audit, predictions, descriptor["config"]["head_mode"])
    if (audit.get("assignment_sha256") != data.assignment_sha256
            or audit.get("study_id") != study_id or audit.get("outer_fold") != fold
            or audit.get("train_image_ids") != [data.rows[i]["image_id"] for i in train]
            or audit.get("test_image_ids") != [data.rows[i]["image_id"] for i in test]):
        raise ValueError("Fold commit does not match fixed baseline assignments/config.")
    expected = {data.rows[i]["image_id"]: data.rows[i] for i in test}
    if len(predictions) != len(expected) or {r["image_id"] for r in predictions} != set(expected):
        raise ValueError("Fold predictions do not cover fixed outer test images exactly once.")
    for prediction in predictions:
        source = expected[prediction["image_id"]]
        p, threshold = prediction["predicted_probability"], prediction["selected_threshold"]
        if (not math.isfinite(p) or not 0 <= p <= 1 or not 0 <= threshold <= 1
                or prediction["image_target"] != source["image_target"]
                or prediction["patient_id"] != source["patient_id"]
                or prediction["objective_lens"] != source["objective_lens"]
                or prediction["outer_fold"] != fold
                or prediction["prediction_at_0_5"] != int(p >= 0.5)
                or prediction["prediction_at_selected_threshold"] != int(p >= threshold)):
            raise ValueError("Saved OOF lineage/probability/decision validation failed.")
    return audit, predictions


def _report_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    pooled, by_lens = core._compute_reports(rows)
    pooled.pop("selected_thresholds_by_fold", None)
    thresholds: dict[str, dict[str, float]] = {}
    for row in rows:
        thresholds.setdefault(str(row["outer_fold"]), {})[row["head"]] = row["selected_threshold"]
    pooled["thresholds_by_fold_and_head"] = thresholds
    for subset, metrics in [(rows, pooled), *[([r for r in rows if str(r["objective_lens"]) == lens], m)
                                             for lens, m in by_lens.items()]]:
        y = np.array([r["image_target"] for r in subset])
        p = np.array([r["predicted_probability"] for r in subset])
        for key in ("fixed_threshold_0_5", "validation_selected_threshold"):
            metrics[key]["auprc"] = float(average_precision_score(y, p))
    return {"pooled": pooled, "by_objective_lens": by_lens}


def run_study(data: StudyData, config: StudyConfig = StudyConfig(), *,
              absolute_deadline: float | None = None) -> dict[str, Any]:
    """Fit, atomically commit and resume a fixed-patient nested head study."""
    import sklearn

    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("Study protocol requires scikit-learn==1.8.0.")
    support = preflight(data, config)
    descriptor = _descriptor(data, config)
    study_id = fingerprint(descriptor)
    root = data.run_root / "head_studies" / study_id
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != descriptor:
        raise ValueError("Existing study config does not match identity.")
    atomic_json(config_path, descriptor)
    deadline = time.monotonic() + 60 * (config.budget_minutes - config.reserve_minutes)
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    labels, _, lenses = _arrays(data)
    with core._RunLock(root, recover=config.recover_lock):
        if (root / "study.commit.json").is_file():
            study, _ = load_study(root)
            return {"state": "complete", "study_root": str(root), **study}
        try:
            atomic_json(root / "status.json", {"state": "running", "study_id": study_id})
            all_predictions = []
            audits = []
            for fold, (train, test) in enumerate(data.outer, start=1):
                core._check_deadline(deadline)
                folder = root / "folds" / f"fold-{fold:02d}"
                if not (folder / "commit.json").is_file():
                    if folder.exists():
                        raise ValueError("Uncommitted fold folder exists; refusing overwrite.")
                    models = {}
                    predictions = []
                    head_audits = {}
                    for head in _heads(config):
                        ti, vi = _scope(train, lenses, head), _scope(test, lenses, head)
                        seed = config.seed + fold * 100 + (0 if head == "all" else int(head))
                        tuning = _tune_head(data, ti, config, seed, deadline)
                        estimator, fitted = _fit_head(data, ti, tuning["lambda"], config, seed)
                        models[head] = estimator
                        head_audits[head] = {"training": fitted, "tuning": tuning}
                        probability = estimator.predict_proba(data.features[vi])[:, 1]
                        for i, score in zip(vi, probability, strict=True):
                            row = data.rows[i]
                            predictions.append({**row, "outer_fold": fold, "head": head,
                                "predicted_probability": float(score), "selected_lambda": tuning["lambda"],
                                "selected_threshold": tuning["threshold"],
                                "prediction_at_0_5": int(score >= 0.5),
                                "prediction_at_selected_threshold": int(score >= tuning["threshold"])})
                    audit = {"fold_id": f"fold-{fold:02d}", "outer_fold": fold,
                             "study_id": study_id, "assignment_sha256": data.assignment_sha256,
                             "train_image_ids": [data.rows[i]["image_id"] for i in train],
                             "test_image_ids": [data.rows[i]["image_id"] for i in test],
                             "test_image_count": len(test), "heads": head_audits,
                             "patient_overlap_count": 0}
                    core._commit_fold_artifact(root, study_id, fold, audit, predictions, models)
                audit, predictions = _verified_fold(data, folder, study_id, fold, train, test)
                audits.append(audit)
                all_predictions.extend(predictions)
                atomic_json(root / "status.json", {"state": "running", "study_id": study_id,
                                                      "completed_outer_folds": fold, "expected_outer_folds": 3})
            by_id = {r["image_id"]: r for r in all_predictions}
            if len(by_id) != len(all_predictions) or set(by_id) != {r["image_id"] for r in data.rows}:
                raise ValueError("OOF coverage is not exactly one prediction per original image.")
            predictions = [by_id[r["image_id"]] for r in data.rows]
            models, deployment = {}, {}
            full = np.arange(len(labels), dtype=np.int64)
            for head in _heads(config):
                core._check_deadline(deadline)
                ti = _scope(full, lenses, head)
                seed = config.seed + 10000 + (0 if head == "all" else int(head))
                tuning = _tune_head(data, ti, config, seed, deadline)
                models[head], fitted = _fit_head(data, ti, tuning["lambda"], config, seed)
                deployment[head] = {"threshold": tuning["threshold"], "lambda": tuning["lambda"],
                                    "training": fitted, "tuning": tuning}
            core._atomic_joblib_dump(models, root / "deployment_models.joblib")
            fields = ("image_id", "patient_id", "file_name", "objective_lens", "raw_glade", "image_target",
                      "outer_fold", "head", "predicted_probability", "selected_lambda", "selected_threshold",
                      "prediction_at_0_5", "prediction_at_selected_threshold")
            core._write_csv(root / "oof_predictions.csv", [{k: r[k] for k in fields} for r in predictions], fields)
            study = {"protocol": PROTOCOL, "study_id": study_id, "descriptor": descriptor,
                     "release_id": data.release["release_id"], "image_count": len(data.rows),
                     "patient_count": len({r["patient_id"] for r in data.rows}), "prediction_unit": "original_image",
                     "feature_dimension": data.features.shape[1],
                     "assignment_sha256": data.assignment_sha256, "metrics": _report_metrics(predictions),
                     "support": support, "folds": audits, "deployment": deployment,
                     "patient_overlap_count": 0, "case_target_used": False, "patch_labels_used": False}
            atomic_json(root / "study.json", study)
            names = ("config.json", "study.json", "oof_predictions.csv", "deployment_models.joblib")
            commit = {"study_id": study_id, "release_id": data.release["release_id"], "complete": True,
                      "files": {name: file_hash(root / name) for name in names},
                      "fold_commits": {f"fold-{i:02d}": json.loads((root / "folds" / f"fold-{i:02d}" / "commit.json").read_text())["commit_id"]
                                       for i in (1, 2, 3)}}
            commit["commit_id"] = fingerprint(commit)
            atomic_json(root / "study.commit.json", commit)
            load_study(root)
            atomic_json(root / "status.json", {"state": "complete", "study_id": study_id})
            return {"state": "complete", "study_root": str(root), **study}
        except core.BudgetExhausted:
            atomic_json(root / "status.json", {"state": "partial", "study_id": study_id, "reason": "budget"})
            return {"state": "partial", "study_id": study_id, "study_root": str(root)}
        except BaseException:
            atomic_json(root / "status.json", {"state": "attention_required", "study_id": study_id})
            raise


def _validate_routing(audit: dict[str, Any], predictions: list[dict[str, Any]], mode: str) -> None:
    """Bind each OOF decision to its own head's committed validation tuning."""
    expected_heads = {"all"} if mode == "global" else {"4", "10", "40"}
    if mode not in {"global", "per_lens"} or set(audit.get("heads", {})) != expected_heads:
        raise ValueError("Committed fold head routing is incomplete.")
    for prediction in predictions:
        head = "all" if mode == "global" else str(prediction["objective_lens"])
        if prediction.get("head") != head or head not in expected_heads:
            raise ValueError("OOF row uses the wrong lens head.")
        tuning = audit["heads"][head]["tuning"]
        threshold = prediction["selected_threshold"]
        penalty = prediction["selected_lambda"]
        if (not math.isfinite(threshold) or not 0 <= threshold <= 1
                or not math.isfinite(penalty) or penalty <= 0
                or not math.isclose(threshold, tuning["threshold"], abs_tol=1e-12)
                or not math.isclose(penalty, tuning["lambda"], abs_tol=1e-12)):
            raise ValueError("OOF decision does not match its head tuning.")


def load_study(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify complete file/fold identities before loading owned model artifacts."""
    import joblib

    root = Path(root)
    commit = json.loads((root / "study.commit.json").read_text())
    body = {k: v for k, v in commit.items() if k != "commit_id"}
    if commit.get("commit_id") != fingerprint(body) or commit.get("complete") is not True:
        raise ValueError("Invalid head-study commit.")
    required = {"config.json", "study.json", "oof_predictions.csv", "deployment_models.joblib"}
    if set(commit.get("files", {})) != required:
        raise ValueError("Study file manifest is incomplete.")
    for name, sha in commit["files"].items():
        if file_hash(root / name) != sha:
            raise ValueError(f"Head-study checksum failed: {name}")
    descriptor = json.loads((root / "config.json").read_text())
    study = json.loads((root / "study.json").read_text())
    if (fingerprint(descriptor) != commit["study_id"] or study.get("study_id") != commit["study_id"]
            or study.get("descriptor") != descriptor or study.get("protocol") != PROTOCOL
            or study.get("release_id") != commit["release_id"]):
        raise ValueError("Head-study config/descriptor identity mismatch.")
    if set(commit.get("fold_commits", {})) != {f"fold-{i:02d}" for i in (1, 2, 3)}:
        raise ValueError("Study does not contain three committed folds.")
    for name, identity in commit["fold_commits"].items():
        path = root / "folds" / name
        audit, predictions = core._verify_fold_artifact(path, study["study_id"])
        _validate_routing(audit, predictions, descriptor["config"]["head_mode"])
        if json.loads((path / "commit.json").read_text())["commit_id"] != identity:
            raise ValueError("Study/fold commit identity differs.")
    models = joblib.load(root / "deployment_models.joblib")
    expected = {"all"} if descriptor["config"]["head_mode"] == "global" else {"4", "10", "40"}
    if not isinstance(models, dict) or set(models) != expected or set(study["deployment"]) != expected:
        raise ValueError("Deployment head routing is incomplete.")
    for head, model in models.items():
        decision = study["deployment"][head]
        threshold, penalty = decision["threshold"], decision["lambda"]
        if (not math.isfinite(threshold) or not 0 <= threshold <= 1
                or not math.isfinite(penalty) or penalty <= 0
                or not math.isclose(threshold, decision["tuning"]["threshold"], abs_tol=1e-12)
                or not math.isclose(penalty, decision["tuning"]["lambda"], abs_tol=1e-12)):
            raise ValueError("Deployment threshold/lambda differs from head tuning.")
        if (not hasattr(model, "named_steps") or "scaler" not in model.named_steps
                or not np.array_equal(model.named_steps["classifier"].classes_, [0, 1])):
            raise ValueError(f"Invalid deployment pipeline for head {head}.")
        requested = descriptor["config"]["pca_components"]
        names = [name for name, _ in model.steps]
        expected_names = ["scaler", "classifier"] if requested is None else ["scaler", "pca", "classifier"]
        if names != expected_names or len(model.named_steps["scaler"].mean_) != study["feature_dimension"]:
            raise ValueError("Deployment preprocessing dimensions differ from config.")
        if requested is not None and model.named_steps["pca"].n_components_ != requested:
            raise ValueError("Deployment PCA dimension differs from config.")
    return study, models


def predict_vector(root: Path, vector: np.ndarray, objective_lens: int | None = None) -> dict[str, Any]:
    """One original-image vector → explicit global/lens-head binary decision."""
    study, models = load_study(root)
    if study["descriptor"]["config"]["head_mode"] == "per_lens":
        if isinstance(objective_lens, bool) or objective_lens not in {4, 10, 40}:
            raise ValueError("Per-lens prediction requires the known objective lens (4, 10, or 40).")
        head = str(objective_lens)
    else:
        head = "all"
    x = np.asarray(vector, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    model = models[head]
    if x.shape != (1, len(model.named_steps["scaler"].mean_)) or not np.isfinite(x).all():
        raise ValueError("Prediction requires one finite vector with the trained feature dimension.")
    probability = float(model.predict_proba(x)[0, 1])
    threshold = float(study["deployment"][head]["threshold"])
    return {"study_id": study["study_id"], "head": head, "cancer_probability": probability,
            "threshold": threshold, "image_target": int(probability >= threshold),
            "prediction_unit": "original_image"}


def predict_image(root: Path, image: Any, weights_dir: Path, *, objective_lens: int | None = None,
                  device: str = "auto", encoder: Any = None) -> dict[str, Any]:
    """Whole-frame image inference uses the exact frozen feature descriptor."""
    from PIL import Image

    study, _ = load_study(root)
    release = json.loads((Path(root).parents[1] / "release.json").read_text())
    if release["release_id"] != study["release_id"]:
        raise ValueError("Study does not match its original-image feature release.")
    if encoder is None:
        encoder = core.OriginalImageEncoder(weights_dir, device=device,
                                           precision=release["descriptor"]["feature_encoder"]["precision"], download=False)
    if encoder.descriptor != release["descriptor"]["feature_encoder"]:
        raise ValueError("Inference encoder differs from the source feature release.")
    if isinstance(image, Image.Image):
        original = image.copy()
    else:
        with Image.open(image) as opened:
            original = opened.copy()
    try:
        processed = core.resize_pad_rgb(original, release["descriptor"]["preprocessing"]["max_long_edge"])
    finally:
        original.close()
    try:
        vector = encoder.encode([processed])
    finally:
        processed.close()
    return predict_vector(root, vector, objective_lens)


def read_predictions(root: Path) -> list[dict[str, Any]]:
    """Read checksum-verified OOF records for reporting/paired comparisons."""
    study, _ = load_study(root)
    with (Path(root) / "oof_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != study["image_count"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Saved OOF coverage differs from study.")
    return rows
