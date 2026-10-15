"""Immutable-source RetCCL head ablations with shared splits and per-fit commits.

The v1 producers/readers stay unchanged. This CPU-only protocol imports their
verified source artifacts, then publishes into a separate v2 study namespace.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import image_classification as core, image_classification_lab as lab, image_classification_studies as legacy
from .io import atomic_json, file_hash, fingerprint

PROTOCOL = "original-image-retccl-head-ablation-v2"
WORKER_MODULE = "histology_data.image_classification_r0_r3"
ARMS = ("R1", "R2", "R2_SAGA_CONTROL", "R3")


@dataclass(frozen=True)
class Config:
    """Scientific settings are identity-bound; wall budgets are operational."""

    lambdas: tuple[float, ...] = (50.0, 5.0, 0.5, 0.05)
    ratios: tuple[float, ...] = (0.3, 0.7)
    iterations: tuple[int, ...] = (2000, 10000, 50000)
    seed: int = 17
    lbfgs_tol: float = 1e-4
    saga_tol: float = 1e-6
    score_tolerance: float = 1e-3
    objective_tolerance: float = 1e-4
    bootstrap_repeats: int = 2000
    bootstrap_seed: int = 17016
    fit_timeout_seconds: float = 900.0
    budget_minutes: float = 180.0
    reserve_minutes: float = 20.0

    def validate(self) -> None:
        """Reject invalid grids before opening an output study."""
        for values in (self.lambdas, self.ratios, self.iterations):
            if not values or len(set(values)) != len(values):
                raise ValueError("Grid values must be nonempty and unique.")
        if any(not math.isfinite(v) or v <= 0 for v in self.lambdas):
            raise ValueError("Lambda must be positive and finite.")
        if any(not 0 < v < 1 for v in self.ratios):
            raise ValueError("ElasticNet ratios must exclude the separate L2 control.")
        if any(type(v) is not int or v < 1 for v in self.iterations) or tuple(sorted(self.iterations)) != self.iterations:
            raise ValueError("Iteration escalation must be increasing positive integers.")
        if (type(self.seed) is not int or self.seed < 0 or type(self.bootstrap_repeats) is not int or self.bootstrap_repeats < 1
                or not 0 <= self.reserve_minutes < self.budget_minutes):
            raise ValueError("Invalid seed, bootstrap or budget.")
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.lbfgs_tol, self.saga_tol, self.score_tolerance,
                self.objective_tolerance, self.fit_timeout_seconds, self.budget_minutes)):
            raise ValueError("Tolerances and deadlines must be finite positive values.")

    def scientific(self) -> dict[str, Any]:
        """Exclude only session-specific budgeting from the immutable recipe."""
        result = asdict(self)
        for key in ("budget_minutes", "reserve_minutes"):
            result.pop(key)
        return json.loads(json.dumps(result))


@dataclass
class Source:
    """Verified frozen features and actual archived inner/outer assignments."""

    features: np.ndarray
    rows: list[dict[str, Any]]
    baseline: dict[str, Any]
    oof: list[dict[str, Any]]
    receipt: dict[str, Any]
    release_root: Path | None = None
    baseline_root: Path | None = None

    def positions(self, ids: list[str]) -> np.ndarray:
        """Resolve exact image order; duplicate/unknown IDs fail closed."""
        lookup = {r["image_id"]: i for i, r in enumerate(self.rows)}
        if len(set(ids)) != len(ids) or not set(ids) <= set(lookup):
            raise ValueError("Invalid shared assignment image identities.")
        return np.asarray([lookup[k] for k in ids], dtype=np.int64)


def load_source(release_root: Path, baseline_root: Path) -> Source:
    """Use the real v1 readers; never rewrite a completed release or study."""
    import sklearn

    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("Head ablation requires scikit-learn==1.8.0.")
    features, rows, release = core._load_feature_matrix(release_root)
    # The v1 audit producer writes a timestamp into the source directory.
    # A completed source must instead be inspected without producer side effects.
    if any(count > 1 for count in Counter(r["original_image_sha256"] for r in rows).values()):
        raise ValueError("Duplicate original image content in frozen source.")
    baseline, _ = legacy.load_study(baseline_root)
    if (release["release_id"] != baseline["release_id"]
            or baseline["descriptor"]["config"]["head_mode"] != "global"
            or baseline.get("case_target_used") is not False
            or baseline.get("patch_labels_used") is not False):
        raise ValueError("R0 must be a verified global image-level study on this release.")
    oof = lab._parse_rows(baseline_root / "oof_predictions.csv")
    receipt = {"release_id": release["release_id"], "study_id": baseline["study_id"],
               "assignment_sha256": baseline["assignment_sha256"],
               "baseline_commit_sha256": file_hash(baseline_root / "study.commit.json"),
               "oof_sha256": file_hash(baseline_root / "oof_predictions.csv"),
               "image_order_sha256": fingerprint([r["image_id"] for r in rows]),
               "feature_shape": list(features.shape),
               "versions": baseline["descriptor"]["versions"]}
    if (baseline["descriptor"]["config"]["weight_scheme"] != "patient_class"
            or baseline["descriptor"]["config"]["pca_components"] is not None):
        raise ValueError("R0 must use patient-class weights and no PCA.")
    source = Source(features, rows, baseline, oof, receipt, Path(release_root), Path(baseline_root))
    preflight(source)
    metrics = report_metrics(oof)
    expected = baseline["metrics"]["pooled"]["fixed_threshold_0_5"]
    for key, old in (("AUROC", "roc_auc"), ("AP", "auprc")):
        if not math.isclose(metrics["all"][key], expected[old], abs_tol=1e-12, rel_tol=0):
            raise ValueError("R0 metrics differ from the immutable original study.")
    return source


def _arrays(source: Source) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray([r["image_target"] for r in source.rows]),
            np.asarray([r["patient_id"] for r in source.rows]),
            np.asarray([r["objective_lens"] for r in source.rows]))


def scopes(source: Source) -> list[tuple[str, np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    """Extract actual B1 records including independent full-data tuning."""
    result = []
    for f in source.baseline["folds"]:
        result.append((f"fold-{f['outer_fold']:02d}", source.positions(f["train_image_ids"]),
                       source.positions(f["test_image_ids"]), f["heads"]["all"]["tuning"]["inner_splits"]))
    result.append(("deployment", np.arange(len(source.rows)), np.arange(len(source.rows)),
                   source.baseline["deployment"]["all"]["tuning"]["inner_splits"]))
    return result


def _restrict(source: Source, indices: np.ndarray, head: str) -> np.ndarray:
    return indices if head == "all" else indices[_arrays(source)[2][indices] == int(head)]


def preflight(source: Source) -> list[dict[str, Any]]:
    """Check source/OOF alignment, shared coverage and all per-lens supports."""
    y, patients, lenses = _arrays(source)
    ids = [r["image_id"] for r in source.rows]
    if (len(set(ids)) != len(ids) or set(lenses) != {4, 10, 40}
            or source.features.ndim != 2 or source.features.shape[0] != len(ids)
            or not np.isfinite(source.features).all()
            or np.any(np.linalg.norm(source.features, axis=1) == 0)):
        raise ValueError("Invalid finite feature coverage, lenses or zero vectors.")
    by_id = {r["image_id"]: r for r in source.oof}
    if len(by_id) != len(source.oof) or set(by_id) != set(ids):
        raise ValueError("R0 OOF does not cover each original exactly once.")
    tests: list[int] = []
    result = []
    contexts = scopes(source)
    if [s[0] for s in contexts] != ["fold-01", "fold-02", "fold-03", "deployment"]:
        raise ValueError("Exactly three ordered outer folds are required.")
    for name, train, test, inner in contexts:
        if name != "deployment":
            if (set(train) & set(test) or set(train) | set(test) != set(range(len(ids)))
                    or set(patients[train]) & set(patients[test])):
                raise ValueError("Outer assignment leaks patients or omits images.")
            tests.extend(test.tolist())
            for i in test:
                row = by_id[ids[i]]
                if (any(row[k] != source.rows[i][k] for k in ("image_target", "patient_id", "objective_lens"))
                        or row["outer_fold"] != int(name[-2:])
                        or not 0 <= row["predicted_probability"] <= 1
                        or row["prediction_at_0_5"] != int(row["predicted_probability"] >= .5)
                        or row["prediction_at_selected_threshold"] != int(row["predicted_probability"] >= row["selected_threshold"])):
                    raise ValueError("R0 OOF lineage/decision differs from source.")
        if len(inner) != 2:
            raise ValueError("Exactly two archived inner splits are required.")
        validation: list[int] = []
        for split in inner:
            ti, vi = source.positions(split["train_image_ids"]), source.positions(split["validation_image_ids"])
            if (set(ti) & set(vi) or set(ti) | set(vi) != set(train)
                    or set(patients[ti]) & set(patients[vi])):
                raise ValueError("Shared inner assignment leaks patients or changes scope.")
            validation.extend(vi.tolist())
            for head in ("4", "10", "40"):
                for partition, indices in (("train", ti), ("validation", vi), ("test", test)):
                    selected = _restrict(source, indices, head)
                    if set(y[selected]) != {0, 1}:
                        raise ValueError("Shared lens scope lacks both classes; refusing fallback.")
                    result.append({"scope": name, "split": split["split"], "head": head,
                                   "partition": partition, "images": len(selected),
                                   "class_patient_support": {str(k): len(set(patients[selected][y[selected] == k])) for k in (0, 1)}})
        if sorted(validation) != sorted(train.tolist()):
            raise ValueError("Inner validation must cover scope exactly once.")
    if sorted(tests) != list(range(len(ids))):
        raise ValueError("Outer test coverage must be exactly once.")
    return result


def _publish(root: Path, binding: dict[str, Any], files: list[str]) -> dict[str, Any]:
    """Publish commit last; readers reject mutations or incomplete payloads."""
    commit = {"binding": binding, "files": {n: file_hash(root / n) for n in files}}
    commit["commit_id"] = fingerprint(commit)
    atomic_json(root / "commit.json", commit)
    return commit


def _verify(root: Path, binding: dict[str, Any]) -> bool:
    path = root / "commit.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if (value.get("binding") != binding
            or fingerprint({k: v for k, v in value.items() if k != "commit_id"}) != value.get("commit_id")
            or not value.get("files")):
        raise ValueError("Artifact commit binding is invalid.")
    for name, sha in value["files"].items():
        p = root / name
        if Path(name).name != name or p.is_symlink() or file_hash(p) != sha:
            raise ValueError("Artifact checksum mismatch.")
    return True


def _bounded(command: list[str], timeout: float, log: Path) -> int:
    """Cancel only this job's worker and always reap it, including interruptions."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _worker(folder: Path) -> int:
    """Fit one bounded CPU model; no labels from evaluation enter fitting."""
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    request = json.loads((folder / "request.json").read_text())
    with np.load(folder / "input.npz", allow_pickle=False) as data:
        x, y, weights, evaluate = (data[k] for k in ("x", "y", "weights", "evaluate"))
    steps = []
    if request["arm"] != "R1":
        steps.append(("normalizer", Normalizer(norm="l2")))
    steps.append(("scaler", StandardScaler()))
    preprocessing = Pipeline(steps)
    z = preprocessing.fit_transform(x)
    ze = preprocessing.transform(evaluate)
    saga = request["arm"] in {"R3", "R2_SAGA_CONTROL"}
    attempts = []
    start = time.monotonic()
    for limit in request["iterations"]:
        model = LogisticRegression(C=1 / (request["lambda"] * float(weights.sum())),
                                   solver="saga" if saga else "lbfgs", l1_ratio=request["ratio"],
                                   tol=request["tol"], random_state=request["seed"], max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(z, y, sample_weight=weights)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        attempts.append({"max_iter": limit, "iterations": model.n_iter_.tolist(),
                         "warnings": [str(w.message) for w in caught], "converged": converged})
        if converged:
            break
    receipt = {"attempts": attempts, "seconds": time.monotonic() - start,
               "converged": converged, "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
               "weight_sum": float(weights.sum()), "C": model.C, "scaler_fit_rows": len(x)}
    import resource
    receipt["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1)
    atomic_json(folder / "fit_receipt.json", receipt)
    if not converged or not np.isfinite(model.coef_).all() or not np.array_equal(model.classes_, [0, 1]):
        return 2
    pipeline = Pipeline([*preprocessing.steps, ("classifier", model)])
    prediction, train_prediction = model.predict_proba(ze)[:, 1], model.predict_proba(z)[:, 1]
    if not np.isfinite(prediction).all():
        return 2
    logits = model.decision_function(z)
    logloss = np.average(np.logaddexp(0, logits) - y * logits, weights=weights)
    ratio = request["ratio"]
    penalty = request["lambda"] * ((1 - ratio) * np.square(model.coef_).sum() / 2 + ratio * np.abs(model.coef_).sum())
    receipt.update(objective=float(logloss + penalty))
    atomic_json(folder / "fit_receipt.json", receipt)
    joblib.dump(pipeline, folder / "model.joblib")
    np.savez(folder / "predictions.npz", evaluate=prediction, train=train_prediction)
    return 0


class Runner:
    """Serial fit dispatcher with immutable, verifiable resume units."""

    def __init__(self, source: Source, root: Path, config: Config, max_new_fits: int | None = None):
        self.source, self.root, self.config = source, root, config
        self.deadline = time.monotonic() + (config.budget_minutes - config.reserve_minutes) * 60
        self.max_new_fits, self.new_fits = max_new_fits, 0

    def fit(self, arm: str, scope: str, head: str, split: str, train: np.ndarray,
            evaluate: np.ndarray, penalty: float, ratio: float = 0) -> tuple[np.ndarray, dict[str, Any], Path]:
        """Reuse only matching committed predictions and owned model bytes."""
        request = {"arm": arm, "lambda": penalty, "ratio": ratio,
                   "iterations": list(self.config.iterations), "seed": self.config.seed,
                   "tol": self.config.saga_tol if arm in {"R3", "R2_SAGA_CONTROL"} else self.config.lbfgs_tol}
        request["lambda"], request["ratio"] = float(penalty), float(ratio)
        binding = {"request": request, "scope": scope, "head": head, "split": split,
                   "train_ids": [self.source.rows[i]["image_id"] for i in train],
                   "evaluation_ids": [self.source.rows[i]["image_id"] for i in evaluate],
                   "source": self.source.receipt, "protocol": PROTOCOL}
        target = self.root / "arms" / arm / scope / "heads" / head / split / fingerprint(request)
        if not _verify(target, binding):
            if (time.monotonic() >= self.deadline or
                    self.max_new_fits is not None and self.new_fits >= self.max_new_fits):
                raise TimeoutError("Budget/new-fit limit; committed units remain reusable.")
            target.mkdir(parents=True, exist_ok=True)
            y, patients, _ = _arrays(self.source)
            weights = legacy.sample_weights(y[train], patients[train], "patient_class")
            wall_start = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="r0-r3-fit-") as scratch:
                folder = Path(scratch)
                atomic_json(folder / "request.json", request)
                np.savez(folder / "input.npz", x=self.source.features[train], y=y[train], weights=weights,
                         evaluate=self.source.features[evaluate])
                timeout = min(self.config.fit_timeout_seconds, self.deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("No time left before reserve.")
                self.event("fit_started", arm=arm, scope=scope, head=head, split=split)
                try:
                    code = _bounded([sys.executable, "-m", WORKER_MODULE, "--worker", str(folder)], timeout, folder / "worker.log")
                except subprocess.TimeoutExpired as error:
                    shutil.copyfile(folder / "worker.log", target / "last_worker.log")
                    timeout_path = target / "timeouts.json"
                    count = json.loads(timeout_path.read_text())["count"] + 1 if timeout_path.exists() else 1
                    atomic_json(timeout_path, {"count": count, "binding_sha256": fingerprint(binding), "timeout_seconds": timeout})
                    if count >= 2:
                        raise RuntimeError("Repeated fit timeout; attention required, no further automatic retry.") from error
                    raise TimeoutError("Fit deadline exceeded; worker stopped.") from error
                if code != 0:
                    shutil.copyfile(folder / "worker.log", target / "last_worker.log")
                    if (folder / "fit_receipt.json").exists():
                        shutil.copyfile(folder / "fit_receipt.json", target / "failed_fit_receipt.json")
                    raise RuntimeError("Fit failed or did not converge; inspect saved worker receipt.")
                fit_receipt = json.loads((folder / "fit_receipt.json").read_text())
                fit_receipt.update(wall_seconds=time.monotonic() - wall_start,
                                   training_images=len(train), evaluation_images=len(evaluate),
                                   feature_dimension=self.source.features.shape[1])
                atomic_json(folder / "fit_receipt.json", fit_receipt)
                for name in ("model.joblib", "predictions.npz", "fit_receipt.json"):
                    temporary = target / (name + ".partial")
                    shutil.copyfile(folder / name, temporary)
                    os.replace(temporary, target / name)
                _publish(target, binding, ["model.joblib", "predictions.npz", "fit_receipt.json"])
                self.new_fits += 1
                self.event("fit_committed", arm=arm, scope=scope, head=head, split=split)
        with np.load(target / "predictions.npz", allow_pickle=False) as values:
            prediction = values["evaluate"].copy()
        if prediction.shape != (len(evaluate),) or not np.isfinite(prediction).all():
            raise ValueError("Committed prediction dimensions differ.")
        return prediction, json.loads((target / "fit_receipt.json").read_text()), target

    def event(self, event: str, **fields: Any) -> None:
        """Append durable operator events without printing patient identities."""
        value = {"event": event, "utc_epoch": time.time(), **fields}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)


def _selection(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(c["AUROC"] for c in candidates)
    return sorted([c for c in candidates if abs(c["AUROC"] - best) <= 1e-12],
                  key=lambda c: (-c["lambda"], c["ratio"]))[0]


def _run_head(runner: Runner, arm: str, name: str, train: np.ndarray, test: np.ndarray,
              inner: list[dict[str, Any]], head: str, dependency: dict[str, Any] | None = None) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    source, config = runner.source, runner.config
    train, test = _restrict(source, train, head), _restrict(source, test, head)
    y = _arrays(source)[0]
    ratios = config.ratios if arm == "R3" else (0.0,)
    candidates = []
    control_checks = []
    for penalty in config.lambdas:
        for ratio in ratios:
            scores = np.full(len(source.rows), np.nan)
            for split in inner:
                ti = _restrict(source, source.positions(split["train_image_ids"]), head)
                vi = _restrict(source, source.positions(split["validation_image_ids"]), head)
                prediction, receipt, path = runner.fit(arm, name, head, f"inner-{split['split']:02d}", ti, vi, penalty, ratio)
                scores[vi] = prediction
                if arm == "R2_SAGA_CONTROL":
                    reference, old, old_path = runner.fit("R2", name, head, f"inner-{split['split']:02d}", ti, vi, penalty)
                    with np.load(path / "predictions.npz") as a, np.load(old_path / "predictions.npz") as b:
                        delta = max(float(np.max(np.abs(a[k] - b[k]))) for k in ("train", "evaluate"))
                    objective_delta = abs(receipt["objective"] - old["objective"]) / max(1, abs(old["objective"]))
                    check = {"lambda": penalty, "split": split["split"], "max_score_delta": delta,
                             "relative_objective_delta": objective_delta,
                             "passed": delta <= config.score_tolerance and objective_delta <= config.objective_tolerance}
                    control_checks.append(check)
                    if not check["passed"] or reference.shape != prediction.shape:
                        atomic_json(runner.root / "solver_control_failure.json", check)
                        raise RuntimeError("SAGA L2 control differs from R2; R3 interpretation blocked.")
            if not np.isfinite(scores[train]).all():
                raise ValueError("Incomplete inner OOF coverage.")
            threshold, bacc = core._choose_threshold(y[train], scores[train])
            candidates.append({"lambda": penalty, "ratio": ratio, "AUROC": float(roc_auc_score(y[train], scores[train])),
                               "threshold": threshold, "balanced_accuracy": bacc})
    chosen = dependency["selected"] if dependency is not None else _selection(candidates)
    prediction, receipt, model_root = runner.fit(arm, name, head, "refit", train, test, chosen["lambda"], chosen["ratio"])
    if arm == "R2_SAGA_CONTROL":
        old_prediction = np.asarray(dependency["probabilities"])
        check = {"scope": name, "refit_score_delta": float(np.max(np.abs(old_prediction - prediction))),
                 "relative_objective_delta": abs(receipt["objective"] - dependency["fit_receipt"]["objective"]) / max(1, abs(dependency["fit_receipt"]["objective"]))}
        check["passed"] = check["refit_score_delta"] <= config.score_tolerance and check["relative_objective_delta"] <= config.objective_tolerance
        control_checks.append(check)
        if not check["passed"]:
            atomic_json(runner.root / "solver_control_failure.json", check)
            raise RuntimeError("SAGA L2 refit control failed.")
    result = {"selected": chosen, "candidates": candidates, "test_image_ids": [source.rows[i]["image_id"] for i in test],
              "probabilities": prediction.tolist(), "fit_receipt": receipt,
              "model_root": str(model_root.relative_to(runner.root)), "control_checks": control_checks}
    destination = runner.root / "arms" / arm / name / "heads" / head
    atomic_json(destination / "selection.json", result)
    _publish(destination, {"study_id": runner.root.name, "arm": arm, "scope": name, "head": head}, ["selection.json"])
    return result


def _prediction_rows(source: Source, arm: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {r["image_id"]: r for r in source.rows}
    output = []
    for name in ("fold-01", "fold-02", "fold-03"):
        for head, result in results[name].items():
            selected = result["selected"]
            for image_id, probability in zip(result["test_image_ids"], result["probabilities"], strict=True):
                output.append({**lookup[image_id], "outer_fold": int(name[-2:]), "head": head,
                               "predicted_probability": probability, "selected_threshold": selected["threshold"],
                               "selected_lambda": selected["lambda"], "selected_l1_ratio": selected["ratio"],
                               "prediction_at_0_5": int(probability >= .5),
                               "prediction_at_selected_threshold": int(probability >= selected["threshold"]), "arm": arm})
    by_id = {r["image_id"]: r for r in output}
    if len(by_id) != len(source.rows) or len(output) != len(source.rows):
        raise ValueError("Final OOF does not cover each image exactly once.")
    return [by_id[r["image_id"]] for r in source.rows]


def report_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Image endpoints and explicitly patient-balanced descriptive endpoints."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    output = {}
    subsets = {"all": rows, **{str(lens): [r for r in rows if r["objective_lens"] == lens] for lens in (4, 10, 40)}}
    for name, subset in subsets.items():
        if not subset:
            output[name] = {"images": 0, "AUROC": None, "AP": None}
            continue
        y = np.asarray([r["image_target"] for r in subset])
        p = np.asarray([r["predicted_probability"] for r in subset])
        patients = np.asarray([r.get("bootstrap_patient", r["patient_id"]) for r in subset])
        valid = set(y) == {0, 1}
        result: dict[str, Any] = {"images": len(y), "patients": len(set(patients)),
                                  "AUROC": float(roc_auc_score(y, p)) if valid else None,
                                  "AP": float(average_precision_score(y, p)) if valid else None}
        if valid:
            weights = legacy.sample_weights(y, patients, "patient_class")
            result["patient_class_weighted_image_AUROC"] = float(roc_auc_score(y, p, sample_weight=weights))
        for key, field in (("fixed_0_5", "prediction_at_0_5"), ("inner_threshold", "prediction_at_selected_threshold")):
            decision = np.asarray([r[field] for r in subset])
            tp, fn = int(np.sum((y == 1) & (decision == 1))), int(np.sum((y == 1) & (decision == 0)))
            tn, fp = int(np.sum((y == 0) & (decision == 0))), int(np.sum((y == 0) & (decision == 1)))
            sensitivity, specificity = (tp / (tp + fn) if tp + fn else None), (tn / (tn + fp) if tn + fp else None)
            macro = {}
            for target, title in ((1, "sensitivity"), (0, "specificity")):
                rates = [float(np.mean(decision[(patients == pat) & (y == target)] == target))
                         for pat in set(patients[y == target])]
                macro[title] = float(np.mean(rates)) if rates else None
            result[key] = {"TP": tp, "FN": fn, "TN": tn, "FP": fp,
                           "accuracy": (tp + tn) / len(y), "sensitivity": sensitivity, "specificity": specificity,
                           "balanced_accuracy": (sensitivity + specificity) / 2 if valid else None, "patient_macro": macro}
        output[name] = result
    lens_auc = [output[str(lens)]["AUROC"] for lens in (4, 10, 40)]
    output["macro_lens_AUROC"] = float(np.mean(lens_auc)) if all(x is not None for x in lens_auc) else None
    return output


def paired_intervals(before: list[dict[str, Any]], after: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Paired cluster percentile intervals; resampling preserves multiplicity."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    before = sorted(before, key=lambda r: r["image_id"])
    after = sorted(after, key=lambda r: r["image_id"])
    if [(r["image_id"], r["patient_id"], r["image_target"], r["objective_lens"]) for r in before] != [
            (r["image_id"], r["patient_id"], r["image_target"], r["objective_lens"]) for r in after]:
        raise ValueError("Paired rows are not identically aligned.")
    patients = sorted({r["patient_id"] for r in before})
    by_patient = {p: [i for i, r in enumerate(before) if r["patient_id"] == p] for p in patients}
    y = np.asarray([r["image_target"] for r in before])
    lens = np.asarray([r["objective_lens"] for r in before])
    a, b = (np.asarray([r["predicted_probability"] for r in rows]) for rows in (before, after))
    values: dict[str, dict[str, list[float]]] = {s: {"AUROC": [], "AP": []} for s in ("all", "4", "10", "40")}
    rng = np.random.default_rng(config.bootstrap_seed)
    for _ in range(config.bootstrap_repeats):
        indices = np.concatenate([by_patient[p] for p in rng.choice(patients, len(patients), replace=True)])
        for scope in values:
            idx = indices if scope == "all" else indices[lens[indices] == int(scope)]
            if set(y[idx]) != {0, 1}:
                continue
            for metric, function in (("AUROC", roc_auc_score), ("AP", average_precision_score)):
                values[scope][metric].append(float(function(y[idx], b[idx]) - function(y[idx], a[idx])))
    return {scope: {metric: {"valid": len(v), "requested": config.bootstrap_repeats,
                            "invalid_reason": "class absent" if len(v) < config.bootstrap_repeats else None,
                            "CI95": np.percentile(v, [2.5, 97.5]).tolist() if len(v) >= 1000 else None}
                    for metric, v in metrics.items()} for scope, metrics in values.items()}


def _write_report(folder: Path, report: dict[str, Any]) -> list[str]:
    """Publish a compact human report and a scientific comparison heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ("R0", "R1", "R2", "R3")
    lines = ["# Kết quả R0–R3 trên frozen whole-image features", "",
             "Đánh giá OOF theo người bệnh; kết quả thăm dò nội bộ, chưa phải test độc lập.", "",
             "| Nhánh | AUROC4× | AUROC10× | AUROC40× | AUROC gộp | AP | Nhạy@0,5 | Đặc hiệu@0,5 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    matrix = []
    for arm in arms:
        metrics = report["metrics"][arm]
        values = [metrics[str(lens)]["AUROC"] for lens in (4, 10, 40)] + [metrics["all"]["AUROC"]]
        matrix.append(values)
        fixed = metrics["all"]["fixed_0_5"]
        lines.append("| " + arm + " | " + " | ".join(f"{v:.3f}" for v in values) +
                     f" | {metrics['all']['AP']:.3f} | {fixed['sensitivity']:.1%} | {fixed['specificity']:.1%} |")
    lines += ["", f"Phương án ưu tiên theo gate đã khóa: **{report['selected_for_followup']}**.", "",
              "Ngưỡng inner validation được lưu riêng từng fold/head; không chỉnh bằng outer labels.",
              "CI trong comparison.json lấy mẫu người bệnh từ OOF đã lưu, không refit; không bao quát toàn bộ bất định lựa chọn mô hình.",
              "R0 và features giữ nguyên. R2_SAGA_CONTROL chỉ kiểm tương đương số học, không là mô hình chính để chọn.",
              "Score logistic chưa được xác minh là xác suất nguy cơ lâm sàng đã hiệu chuẩn.", "",
              "Xem comparison.json, solver_control.json, fit_summary và CSV riêng tư để truy vết."]
    (folder / "BAO_CAO.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    shown = ax.imshow(matrix, vmin=.5, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(4), ["4x", "10x", "40x", "Pooled"])
    ax.set_yticks(range(4), arms)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > .8 else "black")
    ax.set_title("AUROC — patient-disjoint internal OOF")
    fig.colorbar(shown, ax=ax, label="AUROC")
    fig.savefig(folder / "auroc_comparison.png", dpi=160)
    plt.close(fig)
    return ["BAO_CAO.md", "auroc_comparison.png"]


def run_suite(source: Source, output: Path, config: Config = Config(), *,
              max_new_fits: int | None = None, refit: bool = False) -> dict[str, Any]:
    """Run committed CPU ablations; a partial run is never reported complete."""
    config.validate()
    if max_new_fits is not None and (type(max_new_fits) is not int or max_new_fits < 0):
        raise ValueError("New-fit limit must be a nonnegative integer.")
    if any(p is not None and Path(output).resolve().is_relative_to(p.resolve())
           for p in (source.release_root, source.baseline_root)):
        raise ValueError("New study output must not be inside the immutable source.")
    support = preflight(source)
    import importlib.metadata

    descriptor = {"protocol": PROTOCOL, "source": source.receipt, "settings": config.scientific(),
                  "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn")},
                  "baseline_inner_sha256": fingerprint([s[3] for s in scopes(source)]),
                  "implementation_sha256": file_hash(Path(__file__)),
                  "legacy_core_sha256": file_hash(Path(core.__file__)),
                  "legacy_studies_sha256": file_hash(Path(legacy.__file__))}
    root = Path(output) / fingerprint(descriptor)
    root.mkdir(parents=True, exist_ok=True)
    binding = {"descriptor": descriptor}
    if (root / "config.json").exists() and json.loads((root / "config.json").read_text()) != json.loads(json.dumps(descriptor)):
        raise ValueError("Existing study descriptor mismatch.")
    atomic_json(root / "config.json", descriptor)
    runner = Runner(source, root, config, max_new_fits)
    status = {"state": "running", "study_root": str(root), "study_id": root.name,
              "protocol": PROTOCOL, "deployment_requested": refit}
    with core._RunLock(root, recover=False):
        if _verify(root / "report", binding) and not refit:
            _verify_unit_manifest(root)
            return {**status, "state": "complete", "report": json.loads((root / "report/comparison.json").read_text())}
        atomic_json(root / "R0_reference.json", source.receipt)
        atomic_json(root / "support.json", support)
        all_results: dict[str, Any] = {}
        try:
            # Technical pilots are canonical candidate units and are reused later.
            name, train, _, inner = scopes(source)[0]
            split = inner[0]
            pilots = {}
            for arm, ratio in (("R2", 0.0), ("R3", max(config.ratios))):
                _, pilot, _ = runner.fit(arm, name, "all", f"inner-{split['split']:02d}",
                                        source.positions(split["train_image_ids"]), source.positions(split["validation_image_ids"]),
                                        min(config.lambdas), ratio)
                pilots[arm] = pilot
            k, ratios = len(config.lambdas), len(config.ratios)
            lbfgs_fits, saga_fits = 12 * (2 * k + 1), 3 * (2 * k + 1) + 3 * (2 * k * ratios + 1)
            if refit:
                lbfgs_fits += 4 * (2 * k + 1)
                saga_fits += (2 * k + 1) + (2 * k * ratios + 1)
            estimated = 1.5 * (lbfgs_fits * pilots["R2"]["wall_seconds"] + saga_fits * pilots["R3"]["wall_seconds"])
            atomic_json(root / "resource_benchmark.json", {"pilot_committed": True, "pilots": pilots,
                        "unique_fits_expected": lbfgs_fits + saga_fits,
                        "conservative_estimated_fit_seconds": estimated,
                        "reserve_seconds": config.reserve_minutes * 60,
                        "note": "Estimate extrapolates scope/solver pilots; hard per-unit deadlines still apply."})
            for arm in ARMS:
                results = {}
                for name, train, test, inner in scopes(source):
                    if name == "deployment" and not refit:
                        continue
                    results[name] = {}
                    for head in (("4", "10", "40") if arm == "R1" else ("all",)):
                        dependency = all_results["R2"][name][head] if arm == "R2_SAGA_CONTROL" else None
                        results[name][head] = _run_head(runner, arm, name, train, test, inner, head, dependency)
                    atomic_json(root / "status.json", {**status, "arm": arm, "scope": name, "new_fits": runner.new_fits})
                all_results[arm] = results
            rows = {"R0": source.oof, **{arm: _prediction_rows(source, arm, result) for arm, result in all_results.items()}}
            report = {"interpretation": f"Exploratory saved OOF on {len({r['patient_id'] for r in source.rows})} patients; not independent testing.",
                      "source": source.receipt, "metrics": {a: report_metrics(r) for a, r in rows.items()},
                      "by_fold": {a: {str(f): report_metrics([r for r in rr if r["outer_fold"] == f]) for f in (1, 2, 3)} for a, rr in rows.items()},
                      "paired": {}, "gates": {}}
            for before, after in (("R0", "R1"), ("R0", "R2"), ("R0", "R3"), ("R2", "R3")):
                report["paired"][f"{after}-{before}"] = paired_intervals(rows[before], rows[after], config)
            reference = report["metrics"]["R0"]
            for arm in ("R1", "R2", "R3"):
                m = report["metrics"][arm]
                report["gates"][arm] = {"priority_passed": m["all"]["AUROC"] - reference["all"]["AUROC"] >= .02
                                        and m["all"]["AP"] >= reference["all"]["AP"]
                                        and all(m[str(lens)]["AUROC"] - reference[str(lens)]["AUROC"] >= -.02 for lens in (4, 10, 40))}
            eligible = [a for a in ("R2", "R1", "R3") if report["gates"][a]["priority_passed"]]
            if eligible:
                best = max(report["metrics"][a]["all"]["AUROC"] for a in eligible)
                eligible = [a for a in eligible if abs(report["metrics"][a]["all"]["AUROC"] - best) <= 1e-12]
                best_ap = max(report["metrics"][a]["all"]["AP"] for a in eligible)
                eligible = [a for a in eligible if abs(report["metrics"][a]["all"]["AP"] - best_ap) <= 1e-12]
            report["selected_for_followup"] = eligible[0] if eligible else "R0"
            report["fit_summary"] = {arm: {scope: {head: {"selected": result["selected"],
                                     "fit_receipt": result["fit_receipt"]} for head, result in heads.items()}
                                    for scope, heads in scopes_results.items()} for arm, scopes_results in all_results.items()}
            report_root = root / "report"
            report_root.mkdir(exist_ok=True)
            files = []
            for arm, records in rows.items():
                fields = ["image_id", "patient_id", "objective_lens", "raw_glade", "image_target", "outer_fold",
                          "predicted_probability", "selected_threshold", "prediction_at_0_5", "prediction_at_selected_threshold"]
                name = f"{arm}_oof.csv"
                with (report_root / name).open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(records)
                files.append(name)
            atomic_json(report_root / "comparison.json", report)
            atomic_json(report_root / "solver_control.json", {scope: result["all"]["control_checks"]
                        for scope, result in all_results["R2_SAGA_CONTROL"].items()})
            atomic_json(report_root / "gate_decision.json", {"priority_only": True,
                        "gates": report["gates"], "selected": report["selected_for_followup"]})
            files.extend(_write_report(report_root, report))
            atomic_json(root / "results.json", all_results)
            atomic_json(report_root / "results.json", all_results)
            unit_manifest = {str(p.relative_to(root)): file_hash(p) for p in (root / "arms").rglob("commit.json")}
            atomic_json(report_root / "unit_manifest.json", unit_manifest)
            _publish(report_root, binding, [*files, "comparison.json", "results.json", "unit_manifest.json",
                                          "solver_control.json", "gate_decision.json"])
            status.update(state="complete", new_fits=runner.new_fits, report=report)
        except TimeoutError as error:
            status.update(state="partial", reason=str(error), new_fits=runner.new_fits)
        except Exception as error:
            atomic_json(root / "status.json", {**status, "state": "attention_required", "reason": str(error)})
            raise
        atomic_json(root / "status.json", {k: v for k, v in status.items() if k != "report"})
    return status


def _verify_unit_manifest(root: Path) -> None:
    """A complete report is valid only while every referenced fit remains valid."""
    manifest = json.loads((root / "report/unit_manifest.json").read_text())
    if not manifest:
        raise ValueError("Missing fit manifest.")
    for name, sha in manifest.items():
        path = root / name
        if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                or file_hash(path) != sha):
            raise ValueError("Fit manifest checksum mismatch.")
        value = json.loads(path.read_text())
        _verify(path.parent, value["binding"])


def predict_vector(root: Path, arm: str, vector: np.ndarray, objective_lens: int | None = None) -> dict[str, Any]:
    """Verify complete deployment graph before loading this owned pipeline."""
    import joblib

    descriptor = json.loads((root / "config.json").read_text())
    if descriptor["protocol"] != PROTOCOL or fingerprint(descriptor) != root.name:
        raise ValueError("Deployment protocol identity mismatch.")
    if not _verify(root / "report", {"descriptor": descriptor}):
        raise ValueError("Deployment requires a complete report.")
    _verify_unit_manifest(root)
    if arm not in {"R1", "R2", "R3"}:
        raise ValueError("Select a declared main arm.")
    if arm == "R1" and (type(objective_lens) is not int or objective_lens not in {4, 10, 40}):
        raise ValueError("R1 requires known objective lens4/10/40.")
    head = str(objective_lens) if arm == "R1" else "all"
    results = json.loads((root / "report/results.json").read_text())
    if "deployment" not in results[arm]:
        raise ValueError("Deployment refit has not been completed.")
    selected = results[arm]["deployment"][head]
    path = root / selected["model_root"] / "model.joblib"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Deployment model escapes study.")
    model = joblib.load(path)
    expected = ["scaler", "classifier"] if arm == "R1" else ["normalizer", "scaler", "classifier"]
    x = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if (list(model.named_steps) != expected or not np.array_equal(model.classes_, [0, 1])
            or x.shape[1] != descriptor["source"]["feature_shape"][1] or not np.isfinite(x).all()):
        raise ValueError("Invalid deployment preprocessing or vector.")
    probability = float(model.predict_proba(x)[0, 1])
    threshold = selected["selected"]["threshold"]
    return {"score": probability, "threshold": threshold, "image_target": int(probability >= threshold), "arm": arm, "head": head}


def main() -> int:
    """CLI for the bounded worker and mounted-Drive cached-head run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-fits", type=int)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        return _worker(args.worker)
    if any(v is None for v in (args.release_root, args.baseline_root, args.output)):
        parser.error("Provide release-root, baseline-root and output.")
    result = run_suite(load_source(args.release_root, args.baseline_root), args.output,
                       max_new_fits=args.max_new_fits, refit=args.refit)
    print(json.dumps({k: v for k, v in result.items() if k != "report"}), flush=True)
    return 0 if result["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
