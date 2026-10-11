"""Nested B1 novelty-guided MIL with matched full-bag beta-zero controls."""
from __future__ import annotations

import json
import math
import pickle
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dual_common import ArtifactStore, check_deadline
from .dual_member import prepare_member_states
from .dual_mil import make_novelty_model_factory, novelty_model_identity
from .dual_novelty import equal_weight_member_ensemble
from .io import atomic_json, fingerprint
from .mil_checkpoint import CheckpointStore
from .mil_model import ModelConfig
from .mil_study import metrics as case_metrics
from .mil_training import TrainConfig, evaluate, run_fit, source_fingerprint

_BETAS = (0.0, 0.5)
_BETA_SELECTION_DELTA = 0.01
_PATIENCE = 5


def _beta_key(beta: float) -> str:
    if beta == 0.0:
        return "beta_0"
    if beta == 0.5:
        return "beta_0p5"
    raise ValueError("B1 beta must be 0.0 or 0.5.")


def select_beta(inner_bce_by_beta: Mapping[float, Sequence[float]], *, minimum_improvement: float = 0.01) -> dict[str, Any]:
    """Choose beta=.5 only when mean inner ensemble BCE improves by 0.01 or more."""
    if set(inner_bce_by_beta) != set(_BETAS):
        raise ValueError("Inner BCE must be supplied for both beta=0.0 and beta=0.5.")
    if (isinstance(minimum_improvement, bool) or not isinstance(minimum_improvement, (int, float))
            or not math.isfinite(minimum_improvement) or minimum_improvement < 0):
        raise ValueError("minimum_improvement must be finite and nonnegative.")
    means: dict[float, float] = {}
    for beta in _BETAS:
        values = tuple(inner_bce_by_beta[beta])
        if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                             for value in values):
            raise ValueError("Every beta needs finite nonnegative inner validation BCE values.")
        means[beta] = float(np.mean(values, dtype=np.float64))
    improvement = means[0.0] - means[0.5]
    selected = 0.5 if improvement >= minimum_improvement else 0.0
    return {
        "selected_beta": selected,
        "mean_inner_ensemble_bce": {"beta_0": means[0.0], "beta_0p5": means[0.5]},
        "improvement_from_beta_0": improvement,
        "minimum_improvement": float(minimum_improvement),
    }


def outer_epoch_budget(best_epochs_by_inner_scope: Sequence[Sequence[int]]) -> int:
    """Median member best epochs per inner scope, then median scopes and ceil."""
    if not best_epochs_by_inner_scope:
        raise ValueError("At least one completed inner scope is required to derive refit epochs.")
    medians: list[float] = []
    for scope_epochs in best_epochs_by_inner_scope:
        values = tuple(scope_epochs)
        if not values or any(type(epoch) is not int or epoch < 1 for epoch in values):
            raise ValueError("Each inner scope needs positive committed member best epochs.")
        medians.append(float(np.median(values)))
    return max(1, math.ceil(float(np.median(medians))))


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    identity = getattr(dataset, "identity", None)
    if not isinstance(identity, Mapping):
        raise TypeError("dataset_all must expose a verified identity mapping.")
    keys = ("feature_id", "bundle_id", "split_id", "governance_id", "cache_id", "source_manifest_sha256", "mode")
    return {key: identity[key] for key in keys if key in identity}


def _validate_dataset_view(dataset: Any, case_ids: Sequence[str], *, name: str) -> None:
    ids = tuple(dataset.case_ids)
    requested = tuple(case_ids)
    if len(ids) != len(requested) or set(ids) != set(requested):
        raise ValueError(f"{name} dataset factory returned different case membership.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} dataset factory returned duplicate case IDs.")


def _validate_nested_splits(dataset_all: Any, splits: Mapping[str, Any], *, smoke: bool) -> tuple[dict[str, Any], ...]:
    if not isinstance(splits, Mapping) or not isinstance(splits.get("outer"), list) or not splits["outer"]:
        raise ValueError("splits must contain a nonempty outer-fold list.")
    if not smoke and (len(splits["outer"]) != 3 or splits.get("outer_folds") != 3
                      or splits.get("inner_folds") != 2 or splits.get("seed") != 42):
        raise ValueError("Production B1 requires the locked 3x2 patient folds with seed 42.")
    if smoke and (len(splits["outer"]) != 1 or len(splits["outer"][0].get("inner_folds", [])) != 1):
        raise ValueError("Smoke mode is limited to one outer and one inner scope.")
    all_ids = set(dataset_all.case_ids)
    records = dataset_all.records
    expected = set(splits.get("case_ids", ())) or all_ids
    if expected != all_ids:
        raise ValueError("Nested split case coverage differs from the public dataset.")
    test_membership: list[str] = []
    seen_outer: set[int] = set()
    for outer in splits["outer"]:
        fold = outer.get("fold")
        if type(fold) is not int or fold < 0 or fold in seen_outer:
            raise ValueError("Outer folds must have unique nonnegative integer indices.")
        seen_outer.add(fold)
        train_ids = tuple(outer.get("train_cases", ()))
        test_ids = tuple(outer.get("test_cases", ()))
        if not train_ids or not test_ids or len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
            raise ValueError("Outer train/test membership must be nonempty and unique.")
        outer_membership = set(train_ids) | set(test_ids)
        if set(train_ids) & set(test_ids) or outer_membership != all_ids:
            raise ValueError("Outer fold contains a case outside the verified dataset.")
        train_patients = {records[case_id]["patient_id"] for case_id in train_ids}
        test_patients = {records[case_id]["patient_id"] for case_id in test_ids}
        if train_patients & test_patients:
            raise ValueError("Outer split leaks a patient between training and test.")
        test_membership.extend(test_ids)
        inner_folds = outer.get("inner_folds")
        if not isinstance(inner_folds, list) or not inner_folds:
            raise ValueError("Every outer fold requires inner train/validation scopes.")
        if not smoke and len(inner_folds) != 2:
            raise ValueError("Every production outer fold must contain the two locked inner scopes.")
        inner_indices: set[int] = set()
        for inner in inner_folds:
            inner_index = inner.get("fold")
            inner_train = tuple(inner.get("train_cases", ()))
            validation = tuple(inner.get("validation_cases", ()))
            if type(inner_index) is not int or inner_index < 0 or inner_index in inner_indices:
                raise ValueError("Inner folds must have unique nonnegative integer indices.")
            inner_indices.add(inner_index)
            if (not inner_train or not validation or len(set(inner_train)) != len(inner_train)
                    or len(set(validation)) != len(validation)):
                raise ValueError("Inner train/validation membership must be nonempty and unique.")
            if (set(inner_train) & set(validation)
                    or (set(inner_train) | set(validation)) != set(train_ids)):
                raise ValueError("Inner train/validation cases overlap or escape their outer train scope.")
            if set(validation) & set(test_ids):
                raise ValueError("Outer test cases may not enter inner validation.")
            inner_patients = {records[case_id]["patient_id"] for case_id in inner_train}
            validation_patients = {records[case_id]["patient_id"] for case_id in validation}
            if inner_patients & validation_patients:
                raise ValueError("Inner split leaks a patient between training and validation.")
    if not smoke and (len(test_membership) != len(set(test_membership)) or set(test_membership) != expected):
        raise ValueError("Locked outer test folds must cover every case exactly once.")
    return tuple(splits["outer"])


def _load_transform(shared_store: ArtifactStore, scope: str) -> dict[str, Any]:
    name = f"scopes/{scope}/transform.pkl"
    payload = shared_store.read(name)
    if payload is None:
        raise ValueError(f"Required committed shared transform is missing: {name}.")
    value = pickle.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Committed shared transform payload must be a mapping.")
    if value.get("dimensions") != 32 or not isinstance(value.get("lenses"), Mapping):
        raise ValueError("Committed shared transform does not follow the PCA32 protocol.")
    return value


def _model_seed(scope: str, member_id: str) -> int:
    return int(fingerprint({"protocol": "B1-paired-initialization-v1", "scope": scope,
                            "member_id": member_id})[:8], 16) % (2**31 - 1)


def _train_config(*, device: str, mode: str, epochs: int, seed: int, smoke: bool) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(variant="E4", projection_dim=128, attention_dim=64,
                          dropout=0.25, feature_dim=2048),
        mode=mode,
        max_epochs=epochs,
        patience=_PATIENCE,
        learning_rate=1e-4,
        weight_decay=1e-3,
        seed=seed,
        device=device,
        max_steps=20 if smoke else None,
        memory_limit_bytes=2 * 1024**3 if smoke else 8 * 1024**3,
        num_threads=1 if smoke else 2,
        deterministic=True,
    )


def _fit_identity(
    *,
    study_id: str,
    split_id: str,
    scope: str,
    outer_fold: int,
    beta: float,
    member: Mapping[str, Any],
    stage: str,
    inner_fold: int | None,
) -> dict[str, Any]:
    return {
        "protocol": "B1-negative-anchor-full-bag-v1",
        "study_id": study_id,
        "split_id": split_id,
        "scope": scope,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "stage": stage,
        "beta": beta,
        "member_id": member["member_id"],
        "member_identity_id": member["member_identity_id"],
        "anchor_token": member["anchor_token"],
    }


def _fit_dir(output: Path, *, stage: str, outer: int, inner: int | None, beta: float,
             member_id: str) -> Path:
    path = output / "fits" / f"outer_{outer}"
    if inner is not None:
        path = path / f"inner_{inner}"
    return path / stage / _beta_key(beta) / "members" / member_id


def _load_model(
    fit_dir: Path,
    config: TrainConfig,
    factory: Callable[[TrainConfig], torch.nn.Module],
    model_identity: dict[str, Any],
    *,
    best: bool,
) -> torch.nn.Module:
    config_path = fit_dir / "resolved_config.json"
    if not config_path.is_file():
        raise ValueError("Completed fit has no resolved identity file.")
    identity = json.loads(config_path.read_text(encoding="utf-8"))
    if identity.get("model_identity") != model_identity:
        raise ValueError("Fit checkpoint model/bank identity differs from its current member state.")
    store = CheckpointStore(fit_dir / "checkpoints", identity)
    latest = store.load_latest(map_location="cpu")
    if latest is None or not latest.get("finished"):
        raise ValueError("Prediction requires a fully committed completed member fit.")
    state = store.load_best(map_location="cpu") if best else latest
    if state is None:
        raise ValueError("Fit has no committed best checkpoint for prediction.")
    model = factory(config).to(config.device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _prediction_payload(
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, tuple[int, ...]]:
    case_ids = tuple(str(row["case_id"]) for row in predictions)
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(expected_case_ids):
        raise ValueError("Member predictions have missing, extra, or duplicate query cases.")
    by_case = {str(row["case_id"]): row for row in predictions}
    ordered = tuple(str(case_id) for case_id in expected_case_ids)
    probabilities = np.asarray([by_case[case]["probability"] for case in ordered], dtype=np.float32)
    targets = tuple(int(by_case[case]["target"]) for case in ordered)
    if (not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or any(target not in (0, 1) for target in targets)):
        raise ValueError("Member prediction probabilities or targets are invalid.")
    return ordered, probabilities, targets


def _ensemble_rows(
    member_predictions: Sequence[Sequence[Mapping[str, Any]]],
    member_ids: Sequence[str],
    query_case_ids: Sequence[str],
    *,
    outer_fold: int,
    beta: float,
) -> tuple[list[dict[str, Any]], float]:
    if len(member_predictions) != len(member_ids) or not member_predictions:
        raise ValueError("Member predictions and IDs must be nonempty and aligned.")
    query_ids = tuple(str(case_id) for case_id in query_case_ids)
    aligned = [_prediction_payload(rows, expected_case_ids=query_ids) for rows in member_predictions]
    targets = aligned[0][2]
    if any(row[2] != targets for row in aligned[1:]):
        raise ValueError("Member target metadata differs for the same query case order.")
    probabilities = equal_weight_member_ensemble(
        [row[1] for row in aligned],
        member_query_case_ids=[row[0] for row in aligned],
    )
    rows = [
        {"case_id": case_id, "target": target, "probability": float(probability),
         "outer_fold": outer_fold, "beta": beta}
        for case_id, target, probability in zip(query_ids, targets, probabilities, strict=True)
    ]
    clipped = np.clip(probabilities.astype(np.float64), np.finfo(np.float64).eps,
                      1.0 - np.finfo(np.float64).eps)
    labels = np.asarray(targets, dtype=np.float64)
    bce = float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))))
    if not math.isfinite(bce):
        raise FloatingPointError("Ensemble BCE is non-finite.")
    return rows, bce


def _metric_rows(rows: Sequence[Mapping[str, Any]], dataset_all: Any) -> dict[str, Any]:
    expanded = [
        {**row, "patient_id": dataset_all.records[row["case_id"]]["patient_id"]}
        for row in rows
    ]
    return case_metrics(expanded, bootstrap_samples=0)


def _metric_report(rows: Sequence[Mapping[str, Any]], dataset_all: Any,
                   outer_folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "pooled_oof": _metric_rows(rows, dataset_all),
        "outer": {
            str(outer["fold"]): _metric_rows(
                [row for row in rows if row["outer_fold"] == outer["fold"]], dataset_all,
            )
            for outer in outer_folds
        },
    }


def _status(output: Path, *, phase: str, study_id: str, completed_fits: int, expected_fits: int,
            **extra: Any) -> dict[str, Any]:
    value = {"phase": phase, "study_id": study_id, "completed_fits": completed_fits,
             "expected_fits": expected_fits, **extra}
    atomic_json(output / "status.json", value)
    return value


def _commit_member_predictions(
    store: ArtifactStore,
    name: str,
    *,
    study_id: str,
    scope: str,
    beta: float,
    member: Mapping[str, Any],
    fit_identity_id: str,
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    store.json(name, {
        "schema": 1,
        "study_id": study_id,
        "scope": scope,
        "beta": beta,
        "member_id": member["member_id"],
        "member_identity_id": member["member_identity_id"],
        "fit_identity_id": fit_identity_id,
        "predictions": [
            {"case_id": row["case_id"], "target": int(row["target"]),
             "probability": float(row["probability"])}
            for row in predictions
        ],
    })


def _fit_validation_member(
    *,
    dataset_factory: Callable[[Sequence[str]], Any],
    validation_ids: Sequence[str],
    member: Mapping[str, Any],
    beta: float,
    config: TrainConfig,
    fit_identity: dict[str, Any],
    fit_dir: Path,
    model_factory: Callable[[TrainConfig], torch.nn.Module],
    model_identity: dict[str, Any],
    deadline: float,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    train_dataset = dataset_factory(member["head_train_ids"])
    validation_dataset = dataset_factory(validation_ids)
    _validate_dataset_view(train_dataset, member["head_train_ids"], name="member head training")
    _validate_dataset_view(validation_dataset, validation_ids, name="inner validation")
    check_deadline(deadline)
    fit = run_fit(
        train_dataset,
        validation_dataset,
        fit_dir,
        config,
        fit_identity=fit_identity,
        deadline=min(deadline, time.monotonic() + 600.0) if config.mode == "smoke" else deadline,
        model_factory=model_factory,
        model_identity=model_identity,
    )
    if fit.get("phase") != "complete":
        return fit, None
    best_epoch = fit.get("best_epoch")
    if type(best_epoch) is not int or best_epoch < 1:
        raise ValueError("Completed inner member fit has no committed best epoch.")
    model = _load_model(fit_dir, config, model_factory, model_identity, best=True)
    predictions = evaluate(model, validation_dataset, config.device, config=config, deadline=deadline)["predictions"]
    return fit, predictions


def run_neural_study(
    dataset_all: Any,
    dataset_factory: Callable[[Sequence[str]], Any],
    splits: Mapping[str, Any],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
    b_store: ArtifactStore,
    output: Path,
    deadline: float,
    device: str = "cuda",
    max_epochs: int = 30,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run nested β=0/.5 member ensembles, fresh refits, and paired OOF predictions.

    Production uses the locked 3x2 patient folds and one sequential device
    owner. ``smoke=True`` is limited to a single synthetic outer/inner fixture,
    two epochs, and CPU-sized step/memory caps; it runs the inner candidate
    loop and resume path but deliberately produces no outer OOF predictions.
    """
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError("A future finite monotonic deadline is required.")
    if not isinstance(shared_store, ArtifactStore) or not isinstance(b_store, ArtifactStore):
        raise TypeError("shared_store and b_store must be verified ArtifactStore instances.")
    if not isinstance(smoke, bool):
        raise TypeError("smoke must be boolean.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda; no fallback is permitted.")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or not 1 <= max_epochs <= 30:
        raise ValueError("max_epochs must be between 1 and 30.")
    if smoke:
        if _dataset_identity(dataset_all).get("mode") != "smoke":
            raise ValueError("Smoke B1 is permitted only on explicit smoke fixtures.")
        if max_epochs > 2:
            raise ValueError("Smoke B1 is capped at two epochs.")
    else:
        if _dataset_identity(dataset_all).get("mode") == "smoke":
            raise ValueError("Smoke fixtures cannot be promoted into the production B1 study.")
        if device != "cuda":
            raise ValueError("Production B1 runs require the explicitly requested CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback for production B1.")
    outer_folds = _validate_nested_splits(dataset_all, splits, smoke=smoke)
    split_id = str(splits.get("split_id", fingerprint(splits)))
    source_identity = source_map.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("Verified source-photo map must include its source dataset identity.")
    source_identity = {
        key: source_identity[key]
        for key in ("feature_id", "bundle_id", "split_id", "governance_id", "cache_id", "source_manifest_sha256", "mode")
        if key in source_identity
    }
    if source_identity != _dataset_identity(dataset_all):
        raise ValueError("Source-photo map and public dataset identities differ.")
    identity = {
        "schema": 1,
        "protocol": "B1-negative-anchor-full-bag-nested-v1",
        "split_id": split_id,
        "split_fingerprint": fingerprint(splits),
        "dataset_identity": _dataset_identity(dataset_all),
        "source_identity": source_identity,
        "shared_store_identity_id": shared_store.identity_id,
        "b_store_identity_id": b_store.identity_id,
        "betas": list(_BETAS),
        "minimum_beta_improvement_bce": _BETA_SELECTION_DELTA,
        "max_epochs": max_epochs,
        "patience": _PATIENCE,
        "device": device,
        "smoke": smoke,
        "source_fingerprint": source_fingerprint(),
    }
    study_id = fingerprint(identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    if resolved_path.exists() and json.loads(resolved_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Neural study output belongs to a different split/source/model/resource protocol.")
    atomic_json(resolved_path, identity)
    prefix = f"B1/neural/{study_id}"
    result_name = f"{prefix}/{'smoke_result' if smoke else 'final_result'}.json"
    committed_result = b_store.read(result_name)
    if committed_result is not None:
        result = json.loads(committed_result)
        if result.get("study_id") != study_id:
            raise ValueError("Committed neural study result belongs to a different identity.")
        atomic_json(output / "result.json", result)
        if not smoke:
            atomic_json(output / "evaluation" / "metrics.json", result["metrics"])
            atomic_json(output / "evaluation" / "oof_predictions.json", result["oof_predictions"])
        _status(output, phase=result["phase"], study_id=study_id,
                completed_fits=result.get("completed_fits", result.get("member_fit_count", 0)),
                expected_fits=result.get("expected_fits", 0),
                outer_refits_performed=result.get("outer_refits_performed", False),
                oof_rows=result.get("oof_rows", 0),
                selected_beta_by_outer=result.get("selected_beta_by_outer", {}))
        return result

    estimated_inner = sum(
        len([case for case in inner["train_cases"] if dataset_all.records[case]["target"] == 0])
        for outer in outer_folds for inner in outer["inner_folds"]
    ) * len(_BETAS)
    estimated_outer = sum(
        len([case for case in outer["train_cases"] if dataset_all.records[case]["target"] == 0]) * 3
        for outer in outer_folds
    )
    expected_fits = estimated_inner + (0 if smoke else estimated_outer)
    completed_fits = 0
    inner_reports: dict[int, dict[int, dict[float, dict[str, Any]]]] = {}
    selections: dict[int, dict[str, Any]] = {}
    _status(output, phase="running", study_id=study_id, completed_fits=completed_fits,
            expected_fits=expected_fits, stage="preflight")
    try:
        for outer in outer_folds:
            check_deadline(deadline)
            outer_index = outer["fold"]
            inner_reports[outer_index] = {}
            for inner in outer["inner_folds"]:
                check_deadline(deadline)
                inner_index = inner["fold"]
                scope = f"outer_{outer_index}/inner_{inner_index}"
                transforms = _load_transform(shared_store, scope)
                members: list[dict[str, Any]] = []
                negative_anchors = [
                    case_id for case_id in inner["train_cases"]
                    if dataset_all.records[case_id]["target"] == 0
                ]
                if not negative_anchors:
                    raise ValueError(f"Inner scope {scope} has no negative bank anchors.")
                for anchor in negative_anchors:
                    check_deadline(deadline)
                    try:
                        member = prepare_member_states(
                            dataset_all,
                            dataset_factory,
                            transforms,
                            source_map,
                            inner["train_cases"],
                            anchor,
                            b_store,
                            scope,
                            deadline,
                        )
                    except TimeoutError:
                        return _status(output, phase="paused_budget", study_id=study_id,
                                       completed_fits=completed_fits, expected_fits=expected_fits,
                                       next_scope=scope, next_stage="member_state", next_anchor_token=fingerprint(anchor))
                    members.append(member)
                inner_reports[outer_index][inner_index] = {}
                for beta in _BETAS:
                    member_predictions: list[list[dict[str, Any]]] = []
                    member_ids: list[str] = []
                    best_epochs: list[int] = []
                    fit_identities: list[str] = []
                    for member in members:
                        check_deadline(deadline)
                        config = _train_config(
                            device=device,
                            mode="smoke" if smoke else "inner",
                            epochs=max_epochs,
                            seed=_model_seed(scope, member["member_id"]),
                            smoke=smoke,
                        )
                        model_factory = make_novelty_model_factory(
                            member["lens_states"], beta=beta, member_id=member["member_id"],
                        )
                        model_identity = novelty_model_identity(
                            config.model, member["lens_states"], beta=beta, member_id=member["member_id"],
                        )
                        fit_identity = _fit_identity(
                            study_id=study_id,
                            split_id=split_id,
                            scope=scope,
                            outer_fold=outer_index,
                            beta=beta,
                            member=member,
                            stage="inner",
                            inner_fold=inner_index,
                        )
                        fit_dir = _fit_dir(
                            output, stage="inner", outer=outer_index, inner=inner_index,
                            beta=beta, member_id=member["member_id"],
                        )
                        _status(output, phase="running", study_id=study_id,
                                completed_fits=completed_fits, expected_fits=expected_fits,
                                stage="inner_fit", scope=scope, beta=beta,
                                member_id=member["member_id"])
                        try:
                            fit, predictions = _fit_validation_member(
                                dataset_factory=dataset_factory,
                                validation_ids=inner["validation_cases"],
                                member=member,
                                beta=beta,
                                config=config,
                                fit_identity=fit_identity,
                                fit_dir=fit_dir,
                                model_factory=model_factory,
                                model_identity=model_identity,
                                deadline=deadline,
                            )
                        except TimeoutError:
                            return _status(output, phase="paused_budget", study_id=study_id,
                                           completed_fits=completed_fits, expected_fits=expected_fits,
                                           next_fit=fit_identity)
                        if fit.get("phase") != "complete" or predictions is None:
                            return _status(output, phase=fit.get("phase", "attention_required"),
                                           study_id=study_id, completed_fits=completed_fits,
                                           expected_fits=expected_fits, next_fit=fit_identity,
                                           checkpoint_epoch=fit.get("epoch"))
                        completed_fits += 1
                        member_predictions.append(predictions)
                        member_ids.append(member["member_id"])
                        best_epochs.append(int(fit["best_epoch"]))
                        fit_identities.append(fit["identity_id"])
                        member_artifact_name = (
                            f"{prefix}/inner/{scope}/{_beta_key(beta)}/members/"
                            f"{member['member_id']}/predictions.json"
                        )
                        _commit_member_predictions(
                            b_store,
                            member_artifact_name,
                            study_id=study_id,
                            scope=scope,
                            beta=beta,
                            member=member,
                            fit_identity_id=fit["identity_id"],
                            predictions=predictions,
                        )
                    query_ids = tuple(inner["validation_cases"])
                    ensemble, bce = _ensemble_rows(
                        member_predictions,
                        member_ids,
                        query_ids,
                        outer_fold=outer_index,
                        beta=beta,
                    )
                    candidate = {
                        "beta": beta,
                        "mean_ensemble_bce": bce,
                        "member_ids": member_ids,
                        "member_fit_identity_ids": fit_identities,
                        "member_best_epochs": best_epochs,
                        "ensemble_predictions": ensemble,
                    }
                    inner_reports[outer_index][inner_index][beta] = candidate
                    ensemble_name = f"{prefix}/inner/{scope}/{_beta_key(beta)}/ensemble_predictions.json"
                    b_store.json(ensemble_name, {
                        "study_id": study_id,
                        "scope": scope,
                        "beta": beta,
                        "member_ids": member_ids,
                        "predictions": ensemble,
                        "ensemble_bce": bce,
                    })
                    _status(
                        output,
                        phase="running",
                        study_id=study_id,
                        completed_fits=completed_fits,
                        expected_fits=expected_fits,
                        stage="inner_complete",
                        scope=scope,
                        beta=beta,
                        validation_ensemble_bce=bce,
                    )
            beta_values = {
                beta: [inner_reports[outer_index][inner["fold"]][beta]["mean_ensemble_bce"]
                       for inner in outer["inner_folds"]]
                for beta in _BETAS
            }
            selection = select_beta(beta_values, minimum_improvement=_BETA_SELECTION_DELTA)
            selections[outer_index] = selection
            atomic_json(output / f"outer_{outer_index}_selection.json", selection)
            b_store.json(f"{prefix}/selection/outer_{outer_index}.json", {
                "study_id": study_id,
                "outer_fold": outer_index,
                **selection,
                "inner_scope_bce": {
                    str(inner_index): {
                        _beta_key(beta): inner_reports[outer_index][inner_index][beta]["mean_ensemble_bce"]
                        for beta in _BETAS
                    }
                    for inner_index in inner_reports[outer_index]
                },
            })
            if smoke:
                smoke_result = {
                    "phase": "smoke_complete",
                    "study_id": study_id,
                    "software_only": True,
                    "outer_refits_performed": False,
                    "oof_predictions": [],
                    "selected_beta_by_outer": {str(key): value["selected_beta"] for key, value in selections.items()},
                    "selection": selection,
                    "inner_scopes": {
                        str(inner_index): {
                            _beta_key(beta): {
                                "ensemble_bce": inner_reports[outer_index][inner_index][beta]["mean_ensemble_bce"],
                                "member_best_epochs": inner_reports[outer_index][inner_index][beta]["member_best_epochs"],
                            }
                            for beta in _BETAS
                        }
                        for inner_index in inner_reports[outer_index]
                    },
                }
                smoke_result.update({
                    "completed_fits": completed_fits,
                    "expected_fits": expected_fits,
                    "oof_rows": 0,
                    "selected_beta": selection["selected_beta"],
                })
                b_store.json(result_name, smoke_result)
                atomic_json(output / "result.json", smoke_result)
                _status(output, phase="smoke_complete", study_id=study_id,
                        completed_fits=completed_fits, expected_fits=expected_fits,
                        outer_refits_performed=False, oof_rows=0,
                        selected_beta=selection["selected_beta"])
                return smoke_result

        beta0_oof: list[dict[str, Any]] = []
        selected_oof: list[dict[str, Any]] = []
        selected_beta05_oof: list[dict[str, Any]] = []
        for outer in outer_folds:
            check_deadline(deadline)
            outer_index = outer["fold"]
            selected_beta = selections[outer_index]["selected_beta"]
            outer_scope = f"outer_{outer_index}/refit"
            transforms = _load_transform(shared_store, outer_scope)
            members: list[dict[str, Any]] = []
            outer_anchors = [
                case_id for case_id in outer["train_cases"]
                if dataset_all.records[case_id]["target"] == 0
            ]
            for anchor in outer_anchors:
                check_deadline(deadline)
                try:
                    members.append(prepare_member_states(
                        dataset_all,
                        dataset_factory,
                        transforms,
                        source_map,
                        outer["train_cases"],
                        anchor,
                        b_store,
                        outer_scope,
                        deadline,
                    ))
                except TimeoutError:
                    return _status(output, phase="paused_budget", study_id=study_id,
                                   completed_fits=completed_fits, expected_fits=expected_fits,
                                   next_scope=outer_scope, next_stage="member_state", next_anchor_token=fingerprint(anchor))
            if not members:
                raise ValueError(f"Outer refit scope {outer_scope} has no negative bank members.")
            refit_predictions: dict[float, list[list[dict[str, Any]]]] = {0.0: []}
            refit_member_ids: dict[float, list[str]] = {0.0: []}
            candidate_betas = [0.0] + ([0.5] if selected_beta == 0.5 else [])
            for beta in candidate_betas:
                inner_epoch_lists = [
                    inner_reports[outer_index][inner["fold"]][beta]["member_best_epochs"]
                    for inner in outer["inner_folds"]
                ]
                refit_epochs = outer_epoch_budget(inner_epoch_lists)
                refit_predictions[beta] = []
                refit_member_ids[beta] = []
                for member in members:
                    check_deadline(deadline)
                    config = _train_config(
                        device=device,
                        mode="refit",
                        epochs=refit_epochs,
                        seed=_model_seed(outer_scope, member["member_id"]),
                        smoke=False,
                    )
                    model_factory = make_novelty_model_factory(
                        member["lens_states"], beta=beta, member_id=member["member_id"],
                    )
                    model_identity = novelty_model_identity(
                        config.model, member["lens_states"], beta=beta, member_id=member["member_id"],
                    )
                    fit_identity = _fit_identity(
                        study_id=study_id,
                        split_id=split_id,
                        scope=outer_scope,
                        outer_fold=outer_index,
                        beta=beta,
                        member=member,
                        stage="outer_refit",
                        inner_fold=None,
                    )
                    fit_dir = _fit_dir(
                        output, stage="outer_refit", outer=outer_index, inner=None,
                        beta=beta, member_id=member["member_id"],
                    )
                    train_dataset = dataset_factory(member["head_train_ids"])
                    _validate_dataset_view(train_dataset, member["head_train_ids"], name="outer member head training")
                    _status(output, phase="running", study_id=study_id,
                            completed_fits=completed_fits, expected_fits=expected_fits,
                            stage="outer_refit", scope=outer_scope, beta=beta,
                            member_id=member["member_id"], epochs=refit_epochs)
                    try:
                        fit = run_fit(
                            train_dataset,
                            None,
                            fit_dir,
                            config,
                            fit_identity=fit_identity,
                            deadline=deadline,
                            model_factory=model_factory,
                            model_identity=model_identity,
                        )
                    except TimeoutError:
                        return _status(output, phase="paused_budget", study_id=study_id,
                                       completed_fits=completed_fits, expected_fits=expected_fits,
                                       next_fit=fit_identity)
                    if fit.get("phase") != "complete":
                        return _status(output, phase=fit.get("phase", "attention_required"),
                                       study_id=study_id, completed_fits=completed_fits,
                                       expected_fits=expected_fits, next_fit=fit_identity,
                                       checkpoint_epoch=fit.get("epoch"))
                    completed_fits += 1
                    model = _load_model(fit_dir, config, model_factory, model_identity, best=False)
                    test_dataset = dataset_factory(outer["test_cases"])
                    _validate_dataset_view(test_dataset, outer["test_cases"], name="outer test")
                    try:
                        predictions = evaluate(
                            model, test_dataset, device, config=config, deadline=deadline,
                        )["predictions"]
                    except TimeoutError:
                        return _status(output, phase="paused_budget", study_id=study_id,
                                       completed_fits=completed_fits, expected_fits=expected_fits,
                                       next_prediction=fit_identity)
                    refit_predictions[beta].append(predictions)
                    refit_member_ids[beta].append(member["member_id"])
                    member_artifact_name = (
                        f"{prefix}/outer/{outer_scope}/{_beta_key(beta)}/members/"
                        f"{member['member_id']}/predictions.json"
                    )
                    _commit_member_predictions(
                        b_store,
                        member_artifact_name,
                        study_id=study_id,
                        scope=outer_scope,
                        beta=beta,
                        member=member,
                        fit_identity_id=fit["identity_id"],
                        predictions=predictions,
                    )
                rows, ensemble_bce = _ensemble_rows(
                    refit_predictions[beta],
                    refit_member_ids[beta],
                    outer["test_cases"],
                    outer_fold=outer_index,
                    beta=beta,
                )
                b_store.json(f"{prefix}/outer/{outer_scope}/{_beta_key(beta)}/ensemble_predictions.json", {
                    "study_id": study_id,
                    "outer_fold": outer_index,
                    "beta": beta,
                    "member_ids": refit_member_ids[beta],
                    "predictions": rows,
                    "ensemble_bce": ensemble_bce,
                })
                if beta == 0.0:
                    beta0_oof.extend(rows)
                if beta == selected_beta:
                    selected_oof.extend(rows)
                if beta == 0.5:
                    selected_beta05_oof.extend(rows)

        expected_cases = tuple(splits.get("case_ids", dataset_all.case_ids))
        for name, rows in (("B1_beta0_control", beta0_oof), ("B1_selected", selected_oof)):
            ids = [row["case_id"] for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(expected_cases):
                raise ValueError(f"{name} outer predictions do not cover each patient exactly once.")
        metrics = {
            "B1_beta0_control": _metric_report(beta0_oof, dataset_all, outer_folds),
            "B1_selected": _metric_report(selected_oof, dataset_all, outer_folds),
        }
        if len(selected_beta05_oof) == len(expected_cases):
            metrics["B1_beta05_selected_folds"] = _metric_report(
                selected_beta05_oof, dataset_all, outer_folds,
            )
        result = {
            "phase": "complete",
            "study_id": study_id,
            "protocol": identity["protocol"],
            "software_only": False,
            "outer_refits_performed": True,
            "selected_beta_by_outer": {str(key): value["selected_beta"] for key, value in selections.items()},
            "selection": {str(key): value for key, value in selections.items()},
            "refit_epochs": {
                str(outer["fold"]): {
                    "beta_0": outer_epoch_budget([
                        inner_reports[outer["fold"]][inner["fold"]][0.0]["member_best_epochs"]
                        for inner in outer["inner_folds"]
                    ]),
                    **({"beta_0p5": outer_epoch_budget([
                        inner_reports[outer["fold"]][inner["fold"]][0.5]["member_best_epochs"]
                        for inner in outer["inner_folds"]
                    ])} if selections[outer["fold"]]["selected_beta"] == 0.5 else {}),
                }
                for outer in outer_folds
            },
            "oof_predictions": {
                "B1_beta0_control": beta0_oof,
                "B1_selected": selected_oof,
                "B1_beta05_selected_folds": selected_beta05_oof,
            },
            "metrics": metrics,
            "oof_rows": len(expected_cases),
            "case_ids": list(expected_cases),
            "member_fit_count": completed_fits,
        }
        result["completed_fits"] = completed_fits
        result["expected_fits"] = expected_fits
        b_store.json(result_name, result)
        atomic_json(output / "result.json", result)
        atomic_json(output / "evaluation" / "metrics.json", metrics)
        atomic_json(output / "evaluation" / "oof_predictions.json", result["oof_predictions"])
        _status(output, phase="complete", study_id=study_id,
                completed_fits=completed_fits, expected_fits=expected_fits,
                outer_refits_performed=True, oof_rows=len(expected_cases),
                selected_beta_by_outer=result["selected_beta_by_outer"])
        return result
    except TimeoutError:
        return _status(output, phase="paused_budget", study_id=study_id,
                       completed_fits=completed_fits, expected_fits=expected_fits,
                       stage="neural_study")


__all__ = ["outer_epoch_budget", "run_neural_study", "select_beta"]
