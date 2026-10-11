"""Resumable nested case-level studies for atlas A1 and reference-bank B0.

This module is the study runner around the pure clustering/novelty primitives.
It loads only committed train-scope transforms, verifies that split and source
photo lineage agree with the dataset, and commits each reusable model,
descriptor, assignment and prediction through branch-owned ``ArtifactStore``
jobs. Case metadata is used for scope checks and scoring only; numeric heads
receive descriptors alone.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .dual_atlas import (
    CaseHistogram,
    ClusterCaseLabelCounts,
    HDBSCANConfig,
    WardTree,
    assign_to_centers,
    cluster_case_label_crosstab,
    fit_case_classifier,
    fit_kmeans_dictionary,
    fit_ward_tree,
    hdbscan_resampling_stability,
    kmeans_seed_stability,
    photo_balanced_histogram,
    run_hdbscan_grid,
    summarize_partition,
    ward_multicut,
)
from .dual_common import (
    LENSES,
    ArtifactStore,
    check_deadline,
    gzip_csv_bytes,
    photos_for_case,
    project_case,
)
from .dual_novelty import (
    LinearNoveltyHead,
    NegativePrototypeBank,
    PhotoBalancedDescriptors,
    descriptor_vector_by_lens,
    equal_weight_member_ensemble,
    fit_linear_novelty_head,
    photo_balanced_descriptors,
    predict_linear_novelty_head,
    query_novelty,
)
from .splits import validate_split_structure

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]
DatasetFactory = Callable[[list[str]], Any]
Branch = Literal["A", "B"]

_K_VALUES = (4, 8, 16)
_C_VALUES = (0.1, 1.0)
_ATLAS_STABILITY_CONFIGS: tuple[tuple[int, int, str], ...] = (
    (8, 4, "eom"),
    (16, 4, "eom"),
    (16, 4, "leaf"),
    (32, 4, "leaf"),
)
_ASSIGNMENT_FIELDS = [
    "scope", "case_id", "patient_id", "target", "lens", "tile_id", "photo_id",
    "cluster_id", "center_distance", "center_margin",
]
_SAMPLE_FIELDS = [
    "scope", "case_id", "patient_id", "target", "lens", "tile_id", "photo_id", "cluster_id",
]


def _opaque_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _transform_layer_digest(
    layer: Mapping[str, Any],
    scope: str,
    train_case_ids: Sequence[str],
    lens: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(pickle.dumps((scope, tuple(train_case_ids), lens), protocol=5))
    for name, value in (
        ("projected", layer.get("sample_projected")),
        ("weights", layer.get("sample_weights")),
        ("pca_mean", getattr(layer.get("pca"), "mean_", None)),
        ("pca_components", getattr(layer.get("pca"), "components_", None)),
    ):
        if value is None:
            raise ValueError(f"Train transform is missing required {name} for {scope}/{lens}.")
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    refs = layer.get("sample_refs")
    if not isinstance(refs, list):
        raise ValueError(f"Train transform sample refs are missing for {scope}/{lens}.")
    ref_keys = (
        "case_id", "patient_id", "source_case_label", "lens", "tile_id", "photo_id", "part_id", "row_index",
    )
    digest.update(pickle.dumps(tuple(tuple(ref.get(key) for key in ref_keys) for ref in refs), protocol=5))
    return digest.hexdigest()


def _array_digest(values: object) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sequence_token(values: Sequence[str]) -> str:
    return hashlib.sha256(pickle.dumps(tuple(values), protocol=5)).hexdigest()


def _case_ids(values: object, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of case IDs.")
    result = list(values)
    if not result or any(not isinstance(value, str) or not value or value != value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty, trimmed case IDs.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate case IDs.")
    return result


def _records(dataset: Any) -> dict[str, dict[str, Any]]:
    case_ids = _case_ids(dataset.case_ids, "dataset.case_ids")
    records = dataset.records
    if not isinstance(records, Mapping) or set(records) != set(case_ids):
        raise ValueError("Dataset case records must cover exactly dataset.case_ids.")
    normalized: dict[str, dict[str, Any]] = {}
    patient_to_case: dict[str, str] = {}
    for case_id in case_ids:
        row = records[case_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"Dataset record for {case_id!r} is invalid.")
        patient = row.get("patient_id")
        target = row.get("target")
        if not isinstance(patient, str) or not patient or patient != patient.strip():
            raise ValueError(f"Dataset patient_id is invalid for {case_id!r}.")
        if type(target) is not int or target not in (0, 1):
            raise ValueError(f"Dataset target is not a reviewed binary case label for {case_id!r}.")
        if patient in patient_to_case:
            raise ValueError(
                f"Linear case study requires one case per patient; {case_id!r} and "
                f"{patient_to_case[patient]!r} share a patient_id."
            )
        patient_to_case[patient] = case_id
        normalized[case_id] = {"patient_id": patient, "target": target}
    return normalized


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _make_dataset(dataset_factory: DatasetFactory, case_ids: Sequence[str]) -> Any:
    requested = _case_ids(list(case_ids), "dataset_factory case_ids")
    dataset = dataset_factory(requested)
    actual = _case_ids(dataset.case_ids, "dataset_factory result.case_ids")
    if len(actual) != len(requested) or set(actual) != set(requested):
        raise ValueError("dataset_factory must return exactly the requested case membership; silent sampling is forbidden.")
    return dataset


def _validate_splits(splits: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], dataset: Any) -> list[dict[str, Any]]:
    if not isinstance(splits, Mapping) or not isinstance(splits.get("outer"), list):
        raise ValueError("splits must be the original nested split manifest with an outer list.")
    split_cases = [
        {"case_id": case_id, "patient_id": row["patient_id"], "case_label": row["target"]}
        for case_id, row in records.items()
    ]
    validate_split_structure(dict(splits), split_cases)
    outer_rows = sorted(splits["outer"], key=lambda row: row["fold"])
    if [row.get("fold") for row in outer_rows] != list(range(len(outer_rows))):
        raise ValueError("Outer fold IDs must be unique, zero-based, and contiguous.")
    is_smoke = getattr(dataset, "mode", None) == "smoke"
    if not is_smoke and (len(outer_rows) != 3 or splits.get("outer_folds") != 3
                         or splits.get("inner_folds") != 2 or splits.get("seed") != 42):
        raise ValueError("Production linear studies require the locked 3×2 patient folds with seed 42.")
    if not outer_rows:
        raise ValueError("Nested splits contain no outer folds.")
    all_cases = set(records)
    test_seen: list[str] = []
    for outer_index, outer in enumerate(outer_rows):
        train_ids = _case_ids(outer.get("train_cases"), f"outer[{outer_index}].train_cases")
        test_ids = _case_ids(outer.get("test_cases"), f"outer[{outer_index}].test_cases")
        if set(train_ids) | set(test_ids) != all_cases or set(train_ids) & set(test_ids):
            raise ValueError(f"Outer fold {outer_index} does not partition the complete dataset.")
        if set(outer.get("train_patients", [])) != {records[case]["patient_id"] for case in train_ids}:
            raise ValueError(f"Outer fold {outer_index} train patient metadata differs from dataset records.")
        if set(outer.get("test_patients", [])) != {records[case]["patient_id"] for case in test_ids}:
            raise ValueError(f"Outer fold {outer_index} test patient metadata differs from dataset records.")
        if set(outer["train_patients"]) & set(outer["test_patients"]):
            raise ValueError(f"Outer fold {outer_index} leaks a patient across train/test.")
        test_seen.extend(test_ids)
        raw_inner_rows = outer.get("inner_folds")
        if not isinstance(raw_inner_rows, list) or not raw_inner_rows:
            raise ValueError(f"Outer fold {outer_index} requires at least one inner fold.")
        inner_rows = sorted(raw_inner_rows, key=lambda row: row.get("fold", -1))
        if [row.get("fold") for row in inner_rows] != list(range(len(inner_rows))):
            raise ValueError(f"Inner fold IDs of outer {outer_index} must be unique, zero-based, and contiguous.")
        validation_seen: list[str] = []
        for inner_index, inner in enumerate(inner_rows):
            inner_train = _case_ids(inner.get("train_cases"), f"inner[{inner_index}].train_cases")
            inner_validation = _case_ids(inner.get("validation_cases"), f"inner[{inner_index}].validation_cases")
            if (set(inner_train) | set(inner_validation) != set(train_ids)
                    or set(inner_train) & set(inner_validation)):
                raise ValueError(f"Inner fold {outer_index}/{inner_index} does not partition outer-train.")
            if set(inner.get("train_patients", [])) != {records[case]["patient_id"] for case in inner_train}:
                raise ValueError(f"Inner fold {outer_index}/{inner_index} train patient metadata mismatch.")
            if set(inner.get("validation_patients", [])) != {records[case]["patient_id"] for case in inner_validation}:
                raise ValueError(f"Inner fold {outer_index}/{inner_index} validation patient metadata mismatch.")
            if set(inner["train_patients"]) & set(inner["validation_patients"]):
                raise ValueError(f"Inner fold {outer_index}/{inner_index} leaks a patient across train/validation.")
            for name, ids in (("train", inner_train), ("validation", inner_validation)):
                if {records[case]["target"] for case in ids} != {0, 1}:
                    raise ValueError(f"Inner fold {outer_index}/{inner_index} {name} needs both classes.")
            validation_seen.extend(inner_validation)
        if sorted(validation_seen) != sorted(train_ids):
            raise ValueError(f"Inner validation folds of outer {outer_index} must partition outer-train once.")
    if sorted(test_seen) != sorted(records):
        raise ValueError("Outer test folds must contain each dataset case exactly once.")
    return outer_rows


def _scope_patient_ids(case_ids: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    patients = [records[case]["patient_id"] for case in case_ids]
    if len(patients) != len(set(patients)):
        raise ValueError("A train/query scope contains duplicate patient IDs.")
    return patients


def _validate_source_map(dataset: Any, records: Mapping[str, Mapping[str, Any]], source_map: Mapping[str, Any]) -> None:
    if not isinstance(source_map, Mapping) or not isinstance(source_map.get("rows"), Mapping):
        raise ValueError("source_map must contain verified rows and case_tile_order.")
    order = source_map.get("case_tile_order")
    if not isinstance(order, Mapping) or set(order) != set(records):
        raise ValueError("source_map case order must cover the complete dataset.")
    source_identity = source_map.get("source_identity")
    if not isinstance(source_identity, Mapping) or source_identity != dataset.identity:
        raise ValueError("Source photo map identity differs from dataset identity.")
    global_tiles: set[str] = set()
    for case_id in records:
        if not isinstance(order[case_id], Mapping) or set(order[case_id]) != set(LENSES):
            raise ValueError(f"Source photo map has invalid lens keys for {case_id!r}.")
        seen_tiles: set[str] = set()
        for lens in LENSES:
            tile_ids = order[case_id][lens]
            if not isinstance(tile_ids, (list, tuple)):
                raise ValueError("Source map tile order must be an ordered sequence.")
            photos = photos_for_case(dict(source_map), case_id, lens)
            if len(photos) != len(tile_ids):
                raise ValueError("Source photo IDs do not align with tile order.")
            for tile_id, photo_id in zip(tile_ids, photos, strict=True):
                if (not isinstance(tile_id, str) or not tile_id or tile_id in seen_tiles
                        or tile_id in global_tiles):
                    raise ValueError(f"Tile order contains an invalid/duplicate tile for {case_id!r}.")
                seen_tiles.add(tile_id)
                global_tiles.add(tile_id)
                ref = source_map["rows"].get(tile_id)
                if (not isinstance(ref, Mapping) or ref.get("case_id") != case_id
                        or str(ref.get("lens")) != lens or ref.get("photo_id") != photo_id):
                    raise ValueError(f"Tile/source-photo lineage mismatch for {tile_id!r}.")


def _load_transforms(
    scope: str,
    train_case_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
) -> dict[str, Any]:
    name = f"scopes/{scope}/transform.pkl"
    payload = shared_store.read(name)
    if payload is None:
        raise FileNotFoundError(f"Required committed train-only transform is missing: {name}")
    transforms = pickle.loads(payload)
    if not isinstance(transforms, dict) or set(transforms.get("lenses", {})) != set(LENSES):
        raise ValueError(f"Committed transform for {scope} has invalid lens structure.")
    members = transforms.get("members")
    if not isinstance(members, Mapping) or set(members) != set(train_case_ids):
        raise ValueError(f"Committed transform for {scope} belongs to different training cases.")
    if transforms.get("dimensions") != 32:
        raise ValueError("Dual-track study requires the locked PCA32 train-only transform.")
    allowed_patients = {records[case]["patient_id"] for case in train_case_ids}
    for case_id in train_case_ids:
        member = members[case_id]
        if (not isinstance(member, Mapping) or member.get("patient_id") != records[case_id]["patient_id"]
                or int(member.get("target", -1)) != records[case_id]["target"]):
            raise ValueError(f"Train transform member metadata differs for {case_id!r}.")
    for lens in LENSES:
        state = transforms["lenses"][lens]
        matrix = np.asarray(state.get("sample_projected"))
        refs = state.get("sample_refs")
        weights = np.asarray(state.get("sample_weights"), dtype=np.float64)
        if (matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != 32 or not len(matrix)
                or not np.isfinite(matrix).all() or not isinstance(refs, list) or len(refs) != len(matrix)
                or weights.shape != (len(matrix),) or not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError(f"Committed transform sample contract is invalid for {scope}/{lens}.")
        for ref in refs:
            case_id = ref.get("case_id") if isinstance(ref, Mapping) else None
            if case_id not in train_case_ids or ref.get("patient_id") not in allowed_patients:
                raise ValueError(f"Transform sample for {scope}/{lens} contains an out-of-scope case.")
            if (ref.get("patient_id") != records[case_id]["patient_id"]
                    or ref.get("source_case_label") != records[case_id]["target"]
                    or ref.get("lens") != lens):
                raise ValueError(f"Transform sample metadata disagrees with the case/lens for {case_id!r}.")
            tile_id = ref.get("tile_id")
            source_ref = source_map["rows"].get(tile_id)
            if (not isinstance(source_ref, Mapping) or source_ref.get("case_id") != case_id
                    or str(source_ref.get("lens")) != lens or source_ref.get("photo_id") != ref.get("photo_id")):
                raise ValueError(f"Transform sample source photo differs from source_map for {tile_id!r}.")
    return transforms


def _dictionary(
    transforms: Mapping[str, Any],
    train_case_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    store: ArtifactStore,
    scope: str,
    lens: str,
    n_clusters: int,
    deadline: float,
) -> MiniBatchKMeans:
    name = f"scopes/{scope}/kmeans/k{n_clusters}/lens_{lens}/dictionary.pkl"
    layer = transforms["lenses"][lens]
    transform_id = _transform_layer_digest(layer, scope, train_case_ids, lens)
    train_membership = tuple(train_case_ids)

    def fit() -> dict[str, Any]:
        check_deadline(deadline)
        refs = layer["sample_refs"]
        patients = [ref["patient_id"] for ref in refs]
        allowed_patients = set(_scope_patient_ids(train_case_ids, records))
        for patient in patients:
            if patient not in allowed_patients:
                raise ValueError("MiniBatchKMeans sample includes an out-of-scope patient.")
        model = fit_kmeans_dictionary(
            layer["sample_projected"],
            patients,
            _scope_patient_ids(train_case_ids, records),
            n_clusters=n_clusters,
            sample_weight=layer["sample_weights"],
            random_state=42,
            n_init=10,
            batch_size=1024,
            max_iter=100,
        )
        check_deadline(deadline)
        return {
            "model": model,
            "scope": scope,
            "lens": lens,
            "n_clusters": n_clusters,
            "train_case_ids": train_membership,
            "transform_id": transform_id,
        }

    committed = store.object(name, fit)
    if (not isinstance(committed, dict) or committed.get("scope") != scope
            or committed.get("lens") != lens or committed.get("n_clusters") != n_clusters
            or committed.get("train_case_ids") != train_membership
            or committed.get("transform_id") != transform_id):
        raise ValueError(f"Committed KMeans dictionary belongs to another transform/scope: {name}")
    model = committed.get("model")
    if not isinstance(model, MiniBatchKMeans) or model.cluster_centers_.shape != (n_clusters, 32):
        raise ValueError(f"Committed KMeans dictionary has the wrong shape: {name}")
    return model


def _probability_matrix(model: Pipeline, features: FloatMatrix) -> FloatMatrix:
    values = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    if values.shape != (len(features),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise FloatingPointError("Case-level logistic model produced invalid probabilities.")
    return values


def _targets(case_ids: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> NDArray[np.int32]:
    return np.asarray([records[case]["target"] for case in case_ids], dtype=np.int32)


def _patients(case_ids: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return _scope_patient_ids(case_ids, records)


def _loss(case_ids: Sequence[str], probabilities: object, records: Mapping[str, Mapping[str, Any]]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(case_ids),) or not np.isfinite(values).all():
        raise ValueError("Log-loss probabilities must be one finite value per case.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Log-loss probabilities must be in [0, 1].")
    result = float(log_loss(_targets(case_ids, records), values, labels=[0, 1]))
    if not math.isfinite(result):
        raise FloatingPointError("Case log loss is non-finite.")
    return result


def _scope_token(value: str) -> str:
    return value.replace("/", "_")


def _assignment_path(scope: str, k: int, lens: str, case_id: str) -> str:
    return f"assignments/{_scope_token(scope)}/k{k}/lens_{lens}/{_opaque_token(case_id)}.csv.gz"


def _project_scope_cases(
    case_ids: Sequence[str],
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    dataset_factory: DatasetFactory,
    deadline: float,
) -> dict[str, dict[str, FloatMatrix]]:
    ids = _case_ids(list(case_ids), "projected scope case_ids")
    dataset = _make_dataset(dataset_factory, ids)
    projected_cases: dict[str, dict[str, FloatMatrix]] = {}
    for case_id in ids:
        check_deadline(deadline)
        with threadpool_limits(limits=1):
            projected = project_case(dataset, case_id, dict(transforms))
        if hasattr(dataset, "loaded_tile_order"):
            if dataset.loaded_tile_order(case_id) != source_map["case_tile_order"][case_id]:
                raise ValueError(f"Loaded public feature order differs from source map for {case_id!r}.")
        projected_cases[case_id] = projected
    return projected_cases


def _case_descriptors(
    scope: str,
    train_case_ids: Sequence[str],
    case_ids: Sequence[str],
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    projected_cases: Mapping[str, Mapping[str, FloatMatrix]],
    store: ArtifactStore,
    n_clusters: int,
    deadline: float,
) -> dict[str, Any]:
    cases = _case_ids(case_ids, "descriptor case_ids")
    if not set(train_case_ids).issubset(set(records)) or not set(cases).issubset(set(records)):
        raise ValueError("Descriptor cases are outside the dataset records.")
    if set(projected_cases) != set(cases):
        raise ValueError("Projected case cache must cover exactly the descriptor query cases.")
    feature_rows: list[FloatVector] = []
    tile_counts: list[int] = []
    photo_counts: list[int] = []
    lens_tile_counts: list[list[int]] = []
    lens_photo_counts: list[list[int]] = []
    patient_ids: list[str] = []
    for case_id in cases:
        check_deadline(deadline)
        projected = projected_cases[case_id]
        expected_order = source_map["case_tile_order"][case_id]
        per_lens: list[FloatVector] = []
        case_tile_total = 0
        case_photo_total: set[str] = set()
        case_lens_tile_counts: list[int] = []
        case_lens_photo_counts: list[int] = []
        for lens in LENSES:
            matrix = np.asarray(projected[lens])
            tile_ids = list(expected_order[lens])
            photo_ids = photos_for_case(dict(source_map), case_id, lens)
            if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape != (len(tile_ids), 32):
                raise ValueError(f"Projected full bag and ordered tile refs disagree for {case_id!r}/{lens}.")
            if not len(matrix):
                raise ValueError(f"Primary three-lens descriptor requires nonempty {lens}x bags.")
            if len(photo_ids) != len(tile_ids):
                raise ValueError(f"Photo lineage does not align with rows for {case_id!r}/{lens}.")
            patient_id = records[case_id]["patient_id"]
            centers = _dictionary(
                transforms, train_case_ids, records, store, scope, lens, n_clusters, deadline,
            ).cluster_centers_
            with threadpool_limits(limits=1):
                labels, distances, margins = assign_to_centers(matrix, centers, chunk_size=512)
            hist: CaseHistogram = photo_balanced_histogram(
                labels,
                photo_ids,
                [patient_id] * len(matrix),
                n_clusters=n_clusters,
            )
            if hist.patient_ids.tolist() != [patient_id]:
                raise ValueError("Per-case histogram did not reduce to its expected patient ID.")
            per_lens.append(hist.descriptors[0])
            case_tile_total += len(matrix)
            case_photo_total.update(photo_ids)
            case_lens_tile_counts.append(len(matrix))
            case_lens_photo_counts.append(len(set(photo_ids)))
            target = records[case_id]["target"]
            assignment_rows: list[dict[str, Any]] = []
            for row_index, (tile_id, photo_id, cluster_id, distance, margin) in enumerate(
                zip(tile_ids, photo_ids, labels, distances, margins, strict=True)
            ):
                check_deadline(deadline)
                source_ref = source_map["rows"][tile_id]
                if (source_ref["case_id"] != case_id or str(source_ref["lens"]) != lens
                        or source_ref["photo_id"] != photo_id):
                    raise ValueError(f"Full-bag assignment lineage mismatch at {case_id}/{lens}/{row_index}.")
                assignment_rows.append({
                    "scope": scope, "case_id": case_id, "patient_id": patient_id, "target": target,
                    "lens": lens, "tile_id": tile_id, "photo_id": photo_id,
                    "cluster_id": int(cluster_id), "center_distance": float(distance),
                    "center_margin": float(margin),
                })
            store.publish(
                _assignment_path(scope, n_clusters, lens, case_id),
                gzip_csv_bytes(assignment_rows, _ASSIGNMENT_FIELDS),
            )
        feature_rows.append(np.concatenate(per_lens).astype(np.float32, copy=False))
        patient_ids.append(records[case_id]["patient_id"])
        tile_counts.append(case_tile_total)
        photo_counts.append(len(case_photo_total))
        lens_tile_counts.append(case_lens_tile_counts)
        lens_photo_counts.append(case_lens_photo_counts)
    matrix = np.vstack(feature_rows).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Photo-balanced case descriptors contain non-finite values.")
    transform_hashes = {
        lens: _transform_layer_digest(transforms["lenses"][lens], scope, train_case_ids, lens)
        for lens in LENSES
    }
    return {
        "scope": scope,
        "train_case_ids": tuple(train_case_ids),
        "n_clusters": n_clusters,
        "transform_hashes": transform_hashes,
        "case_ids": tuple(cases),
        "patient_ids": tuple(patient_ids),
        "features": matrix,
        "tile_counts": np.asarray(tile_counts, dtype=np.int64),
        "photo_counts": np.asarray(photo_counts, dtype=np.int64),
        "lens_tile_counts": np.asarray(lens_tile_counts, dtype=np.int64),
        "lens_photo_counts": np.asarray(lens_photo_counts, dtype=np.int64),
    }


def _cached_case_descriptors(
    scope: str,
    train_case_ids: Sequence[str],
    case_ids: Sequence[str],
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    projected_cases: Mapping[str, Mapping[str, FloatMatrix]],
    store: ArtifactStore,
    n_clusters: int,
    deadline: float,
) -> dict[str, Any]:
    # Recompute row assignments on resume so a missing per-case provenance file
    # can be completed before the descriptor aggregate is accepted.
    name = f"descriptors/{_scope_token(scope)}/k{n_clusters}/case_matrix.pkl"
    computed = _case_descriptors(
        scope, train_case_ids, case_ids, transforms, source_map, records, projected_cases,
        store, n_clusters, deadline,
    )
    payload = pickle.dumps(computed, protocol=5)
    prior = store.read(name)
    if prior is None:
        store.publish(name, payload)
        return computed
    cached = pickle.loads(prior)
    if (not isinstance(cached, dict) or cached.get("scope") != computed["scope"]
            or cached.get("train_case_ids") != computed["train_case_ids"]
            or cached.get("n_clusters") != computed["n_clusters"]
            or cached.get("transform_hashes") != computed["transform_hashes"]
            or cached.get("case_ids") != computed["case_ids"]
            or cached.get("patient_ids") != computed["patient_ids"]
            or not np.array_equal(cached.get("features"), computed["features"])
            or not np.array_equal(cached.get("tile_counts"), computed["tile_counts"])
            or not np.array_equal(cached.get("photo_counts"), computed["photo_counts"])
            or not np.array_equal(cached.get("lens_tile_counts"), computed["lens_tile_counts"])
            or not np.array_equal(cached.get("lens_photo_counts"), computed["lens_photo_counts"])):
        raise ValueError(f"Committed descriptor matrix differs from recomputed inputs: {name}")
    return cached


def _indices_for_cases(descriptors: Mapping[str, Any], case_ids: Sequence[str]) -> tuple[FloatMatrix, list[str]]:
    lookup = {case_id: index for index, case_id in enumerate(descriptors["case_ids"])}
    if not set(case_ids).issubset(lookup):
        raise ValueError("Requested case descriptors are missing from the scope aggregate.")
    indices = [lookup[case_id] for case_id in case_ids]
    return np.asarray(descriptors["features"][indices], dtype=np.float32), list(case_ids)


def _counts_features(
    descriptors: Mapping[str, Any],
    case_ids: Sequence[str],
) -> FloatMatrix:
    lookup = {case_id: index for index, case_id in enumerate(descriptors["case_ids"])}
    indices = [lookup[case] for case in case_ids]
    tile_counts = np.asarray(descriptors["lens_tile_counts"][indices], dtype=np.float32)
    photo_counts = np.asarray(descriptors["lens_photo_counts"][indices], dtype=np.float32)
    values = np.stack((tile_counts, photo_counts), axis=2).reshape(len(indices), -1)
    features = np.log1p(values)
    if not np.isfinite(features).all():
        raise FloatingPointError("Count-only baseline features are non-finite.")
    return features.astype(np.float32, copy=False)


def _fit_a_head(
    store: ArtifactStore,
    scope: str,
    candidate_id: str,
    features: FloatMatrix,
    case_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    c_value: float,
    deadline: float,
) -> Pipeline:
    name = f"heads/{_scope_token(scope)}/{candidate_id}.pkl"
    train_membership = tuple(case_ids)
    feature_digest = _array_digest(features)

    def fit() -> dict[str, Any]:
        check_deadline(deadline)
        model = fit_case_classifier(
            features,
            _targets(case_ids, records),
            _patients(case_ids, records),
            _patients(case_ids, records),
            regularization_c=c_value,
            random_state=42,
        )
        check_deadline(deadline)
        return {
            "model": model,
            "scope": scope,
            "candidate_id": candidate_id,
            "train_case_ids": train_membership,
            "regularization_c": float(c_value),
            "feature_digest": feature_digest,
        }

    committed = store.object(name, fit)
    if (not isinstance(committed, dict) or committed.get("scope") != scope
            or committed.get("candidate_id") != candidate_id
            or committed.get("train_case_ids") != train_membership
            or committed.get("regularization_c") != float(c_value)
            or committed.get("feature_digest") != feature_digest):
        raise ValueError(f"Committed A1/count head belongs to another fold, input, or candidate: {name}")
    model = committed.get("model")
    if not isinstance(model, Pipeline) or model.n_features_in_ != features.shape[1]:
        raise ValueError(f"Committed linear head does not match its descriptor input: {name}")
    return model


def _case_predictions(
    scope: str,
    model: Pipeline,
    features: FloatMatrix,
    case_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    *,
    selected_k: int | None,
    selected_c: float,
) -> list[dict[str, Any]]:
    probabilities = _probability_matrix(model, features)
    return [
        {
            "case_id": case_id,
            "patient_id": records[case_id]["patient_id"],
            "target": records[case_id]["target"],
            "probability": float(probability),
            "scope": scope,
            "outer_fold": int(scope.split("/")[0].split("_")[1]),
            "selected_k": selected_k,
            "selected_c": float(selected_c),
        }
        for case_id, probability in zip(case_ids, probabilities, strict=True)
    ]


def _verify_oof(rows: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(records) or set(case_ids) != set(records) or len(case_ids) != len(set(case_ids)):
        raise ValueError("OOF manifest must contain exactly one prediction for every unique cohort case.")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["case_id"])
    for row in ordered:
        case_id = row["case_id"]
        if (row.get("patient_id") != records[case_id]["patient_id"]
                or row.get("target") != records[case_id]["target"]
                or not math.isfinite(float(row["probability"]))
                or not 0.0 <= float(row["probability"]) <= 1.0):
            raise ValueError(f"OOF metadata/probability is invalid for {case_id!r}.")
    return ordered


def _branch_identity(store: ArtifactStore, branch: str, dataset: Any, splits: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch": branch,
        "branch_store_identity_id": store.identity_id,
        "source_identity": dataset.identity,
        "split_id": splits.get("split_id"),
    }


def _write_final(store: ArtifactStore, result: dict[str, Any]) -> dict[str, Any]:
    store.json("results/linear_study.json", result)
    return result


def run_linear_study(
    branch: Branch,
    dataset_all: Any,
    dataset_factory: DatasetFactory,
    splits: dict[str, Any],
    source_map: dict[str, Any],
    shared_store: ArtifactStore,
    branch_store: ArtifactStore,
    deadline: float,
) -> dict[str, Any]:
    """Run A0+A1 or B0 on the original nested splits and committed transforms.

    The shared store is read-only from this runner: all nine transform jobs
    must already be committed at ``scopes/<scope>/transform.pkl``. Branch-owned
    models, assignment tables, descriptors and predictions use the supplied
    branch store, whose ArtifactStore receipts provide resume integrity.
    """
    if branch not in {"A", "B"}:
        raise ValueError("branch must be 'A' or 'B'.")
    if not math.isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic-clock timestamp.")
    check_deadline(deadline)
    records = _records(dataset_all)
    if set(source_map.get("case_tile_order", {})) != set(records):
        raise ValueError("source_map must cover every case in dataset_all.")
    _validate_source_map(dataset_all, records, source_map)
    outer_rows = _validate_splits(splits, records, dataset_all)
    _preflight_all_transforms(outer_rows, splits, records, source_map, shared_store, deadline)
    context = _branch_identity(branch_store, branch, dataset_all, splits)
    if branch == "A":
        result = _run_branch_a(
            dataset_all, dataset_factory, splits, outer_rows, records, source_map,
            shared_store, branch_store, deadline,
        )
    else:
        result = _run_branch_b(
            dataset_all, dataset_factory, splits, outer_rows, records, source_map,
            shared_store, branch_store, deadline,
        )
    result["study_identity"] = context
    return _write_final(branch_store, result)


def _run_branch_a(
    dataset_all: Any,
    dataset_factory: DatasetFactory,
    splits: Mapping[str, Any],
    outer_rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
    branch_store: ArtifactStore,
    deadline: float,
) -> dict[str, Any]:
    atlas = _run_atlas_a0(
        dataset_all, dataset_factory, splits, outer_rows[0], records, source_map,
        shared_store, branch_store, deadline,
    )
    candidate_results: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    count_oof_rows: list[dict[str, Any]] = []
    outer_reports: list[dict[str, Any]] = []
    logical_head_fits = 0
    count_head_fits = 0
    for outer in outer_rows:
        check_deadline(deadline)
        outer_index = int(outer["fold"])
        inner_candidates: dict[tuple[int, float], list[tuple[float, int]]] = {
            (k, c): [] for k in _K_VALUES for c in _C_VALUES
        }
        for inner in outer["inner_folds"]:
            scope = f"outer_{outer_index}/inner_{int(inner['fold'])}"
            train_ids = list(inner["train_cases"])
            validation_ids = list(inner["validation_cases"])
            all_ids = _ordered_unique([*train_ids, *validation_ids])
            transforms = _load_transforms(scope, train_ids, records, source_map, shared_store)
            projected_cases = _project_scope_cases(all_ids, transforms, source_map, dataset_factory, deadline)
            count_desc: dict[str, Any] | None = None
            for k in _K_VALUES:
                descriptors = _cached_case_descriptors(
                    scope, train_ids, all_ids, transforms, source_map, records,
                    projected_cases, branch_store, k, deadline,
                )
                train_features, _ = _indices_for_cases(descriptors, train_ids)
                validation_features, _ = _indices_for_cases(descriptors, validation_ids)
                for c_value in _C_VALUES:
                    candidate_id = f"k{k}_c{str(c_value).replace('.', '_')}"
                    model = _fit_a_head(
                        branch_store, scope, candidate_id, train_features, train_ids,
                        records, c_value, deadline,
                    )
                    probabilities = _probability_matrix(model, validation_features)
                    value = _loss(validation_ids, probabilities, records)
                    inner_candidates[(k, c_value)].append((value, len(validation_ids)))
                    candidate_results.append({
                        "scope": scope, "k": k, "c": c_value, "validation_log_loss": value,
                    })
                    logical_head_fits += 1
                count_desc = descriptors
            if count_desc is None:
                raise ArithmeticError("Count control requires a descriptor scope.")
            count_train = _counts_features(count_desc, train_ids)
            count_validation = _counts_features(count_desc, validation_ids)
            count_model = _fit_a_head(
                branch_store, scope, "counts_only_fixed_c0_1", count_train, train_ids,
                records, 0.1, deadline,
            )
            count_probs = _probability_matrix(count_model, count_validation)
            count_loss = _loss(validation_ids, count_probs, records)
            candidate_results.append({"scope": scope, "candidate": "counts_only", "validation_log_loss": count_loss})
            count_head_fits += 1

        inner_means = {
            key: float(sum(loss * count for loss, count in losses) / sum(count for _, count in losses))
            for key, losses in inner_candidates.items()
        }
        if any(len(losses) != len(outer["inner_folds"]) for losses in inner_candidates.values()):
            raise ArithmeticError("A1 candidate is missing an inner-fold score.")
        best_loss = min(inner_means.values())
        tied = [key for key, value in inner_means.items() if value <= best_loss + 0.001]
        selected_k, selected_c = min(tied, key=lambda item: (item[0], item[1]))
        refit_scope = f"outer_{outer_index}/refit"
        refit_train = list(outer["train_cases"])
        test_ids = list(outer["test_cases"])
        refit_ids = _ordered_unique([*refit_train, *test_ids])
        transforms = _load_transforms(refit_scope, refit_train, records, source_map, shared_store)
        projected_cases = _project_scope_cases(refit_ids, transforms, source_map, dataset_factory, deadline)
        descriptors = _cached_case_descriptors(
            refit_scope, refit_train, refit_ids, transforms, source_map, records,
            projected_cases, branch_store, selected_k, deadline,
        )
        train_features, _ = _indices_for_cases(descriptors, refit_train)
        test_features, _ = _indices_for_cases(descriptors, test_ids)
        final_model = _fit_a_head(
            branch_store,
            refit_scope,
            f"selected_k{selected_k}_c{str(selected_c).replace('.', '_')}",
            train_features,
            refit_train,
            records,
            selected_c,
            deadline,
        )
        oof_rows.extend(_case_predictions(
            refit_scope, final_model, test_features, test_ids, records,
            selected_k=selected_k, selected_c=selected_c,
        ))
        count_features = _counts_features(descriptors, refit_train)
        test_count_features = _counts_features(descriptors, test_ids)
        count_model = _fit_a_head(
            branch_store, refit_scope, "counts_only_fixed_c0_1", count_features,
            refit_train, records, 0.1, deadline,
        )
        count_oof_rows.extend(_case_predictions(
            refit_scope, count_model, test_count_features, test_ids, records,
            selected_k=None, selected_c=0.1,
        ))
        count_head_fits += 1
        logical_head_fits += 1
        outer_reports.append({
            "outer_fold": outer_index,
            "inner_mean_log_loss": {
                f"k{k}_c{c}": inner_means[(k, c)] for k in _K_VALUES for c in _C_VALUES
            },
            "selected_k": selected_k,
            "selected_c": selected_c,
            "selection_log_loss": inner_means[(selected_k, selected_c)],
            "tie_tolerance": 0.001,
            "tie_break": "smaller_k_then_smaller_c",
        })
    verified_oof = _verify_oof(oof_rows, records)
    verified_counts = _verify_oof(count_oof_rows, records)
    expected_head_fits = (
        len(_K_VALUES) * len(_C_VALUES) * sum(len(row["inner_folds"]) for row in outer_rows)
        + len(outer_rows)
    )
    if logical_head_fits != expected_head_fits:
        raise ArithmeticError(
            f"A1 logical head fit count should be {expected_head_fits}, got {logical_head_fits}."
        )
    result = {
        "schema_version": 1,
        "branch": "A",
        "phase": "complete",
        "method": "A0_train_only_atlas_A1_photo_balanced_kmeans_logistic",
        "a1": {
            "candidates": [{"k": k, "c": c} for k in _K_VALUES for c in _C_VALUES],
            "outer_selection": outer_reports,
            "inner_candidate_scores": candidate_results,
            "logical_head_fits": logical_head_fits,
            "expected_head_fits": expected_head_fits,
            "oof_count": len(verified_oof),
            "oof_predictions": verified_oof,
        },
        "counts_only_control": {
            "feature_contract": "log1p(tile_count, source_photo_count) for each lens",
            "c": 0.1,
            "logical_head_fits": count_head_fits,
            "oof_count": len(verified_counts),
            "oof_predictions": verified_counts,
        },
        "a0": atlas,
    }
    branch_store.json("results/a1_oof.json", verified_oof)
    branch_store.json("results/counts_only_oof.json", verified_counts)
    return result


def _preflight_all_transforms(
    outer_rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
    deadline: float,
) -> None:
    """Validate every scope transform before a branch publishes any fit."""
    scopes: list[tuple[str, list[str]]] = []
    for outer in outer_rows:
        outer_index = int(outer["fold"])
        for inner in outer["inner_folds"]:
            scopes.append((
                f"outer_{outer_index}/inner_{int(inner['fold'])}",
                list(inner["train_cases"]),
            ))
        scopes.append((f"outer_{outer_index}/refit", list(outer["train_cases"])))
    for scope, train_ids in scopes:
        check_deadline(deadline)
        _load_transforms(scope, train_ids, records, source_map, shared_store)


def _sample_rows(
    scope: str,
    lens: str,
    refs: Sequence[Mapping[str, Any]],
    labels: object,
    source_map: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cluster_ids = np.asarray(labels)
    if cluster_ids.ndim != 1 or len(cluster_ids) != len(refs):
        raise ValueError("Atlas labels and sample refs must preserve the exact same row order.")
    rows: list[dict[str, Any]] = []
    for ref, cluster_id in zip(refs, cluster_ids, strict=True):
        case_id, tile_id = ref.get("case_id"), ref.get("tile_id")
        if case_id not in records or case_id not in source_map["case_tile_order"]:
            raise ValueError("Atlas sample reference contains an unknown case.")
        source_ref = source_map["rows"].get(tile_id)
        if (not isinstance(source_ref, Mapping) or source_ref.get("case_id") != case_id
                or str(source_ref.get("lens")) != lens or source_ref.get("photo_id") != ref.get("photo_id")):
            raise ValueError(f"Atlas sample reference lineage mismatch for {tile_id!r}.")
        if ref.get("source_case_label") != records[case_id]["target"]:
            raise ValueError(f"Atlas sample source case label differs from verified dataset for {case_id!r}.")
        rows.append({
            "scope": scope,
            "case_id": case_id,
            "patient_id": ref.get("patient_id"),
            "target": int(ref["source_case_label"]),
            "lens": lens,
            "tile_id": tile_id,
            "photo_id": ref["photo_id"],
            "cluster_id": int(cluster_id),
        })
    return rows


def _crosstab_rows(values: Sequence[ClusterCaseLabelCounts]) -> list[dict[str, Any]]:
    return [asdict(row) for row in values]


def _run_atlas_a0(
    dataset_all: Any,
    dataset_factory: DatasetFactory,
    splits: Mapping[str, Any],
    outer: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
    branch_store: ArtifactStore,
    deadline: float,
) -> dict[str, Any]:
    del dataset_all, dataset_factory  # A0 only consumes the verified train reservoir.
    scope = f"outer_{int(outer['fold'])}/inner_0"
    development_inner = next(
        (row for row in outer["inner_folds"] if int(row["fold"]) == 0),
        None,
    )
    if development_inner is None:
        raise ValueError("A0 requires the locked outer_0/inner_0 development train scope.")
    train_ids = list(development_inner["train_cases"])
    transforms = _load_transforms(scope, train_ids, records, source_map, shared_store)
    summary: dict[str, Any] = {"scope": scope, "lenses": {}}
    for lens in LENSES:
        check_deadline(deadline)
        layer = transforms["lenses"][lens]
        vectors = np.asarray(layer["sample_projected"])
        refs = layer["sample_refs"]
        patients = [ref["patient_id"] for ref in refs]
        photos = [ref["photo_id"] for ref in refs]
        patient_scope = _scope_patient_ids(train_ids, records)
        density_reports: list[dict[str, Any]] = []
        density_lookup: set[tuple[int, int, str]] = set()
        for min_cluster_size in (8, 16, 32, 64):
            for min_samples in (4, 8):
                for selection_method in ("eom", "leaf"):
                    check_deadline(deadline)
                    config = HDBSCANConfig(min_cluster_size, min_samples, selection_method)
                    key = (min_cluster_size, min_samples, selection_method)
                    density_lookup.add(key)
                    path_base = (
                        f"atlas/{_scope_token(scope)}/hdbscan/lens_{lens}/"
                        f"mcs{min_cluster_size}_ms{min_samples}_{selection_method}"
                    )
                    metrics_name = f"{path_base}/metrics.json"
                    assignment_name = f"{path_base}/assignments.csv.gz"
                    cached_metrics = branch_store.read(metrics_name)
                    cached_assignment = branch_store.read(assignment_name)
                    if cached_metrics is not None and cached_assignment is not None:
                        report = json.loads(cached_metrics)
                        if report.get("config") != asdict(config) or report.get("assignment_rows") != len(refs):
                            raise ValueError(f"Committed HDBSCAN report has the wrong configuration: {metrics_name}")
                        density_reports.append(report)
                        continue
                    run = run_hdbscan_grid(
                        vectors,
                        patients,
                        photos,
                        patient_scope,
                        min_cluster_sizes=(min_cluster_size,),
                        min_samples_values=(min_samples,),
                        selection_methods=(selection_method,),
                    )[0]
                    check_deadline(deadline)
                    if run.config != config:
                        raise ArithmeticError("HDBSCAN helper returned a different configuration.")
                    sample_assignment = _sample_rows(scope, lens, refs, run.labels, source_map, records)
                    branch_store.publish(
                        assignment_name,
                        gzip_csv_bytes(sample_assignment, _SAMPLE_FIELDS),
                    )
                    label_rows = cluster_case_label_crosstab(
                        run.labels, patients, [ref["source_case_label"] for ref in refs],
                    )
                    report = {
                        "config": asdict(config),
                        "partition": asdict(run.summary),
                        "source_case_label_crosstab": _crosstab_rows(label_rows),
                        "assignment_rows": len(sample_assignment),
                    }
                    branch_store.json(metrics_name, report)
                    density_reports.append(report)

        ward_tree = branch_store.object(
            f"atlas/{_scope_token(scope)}/ward/lens_{lens}/tree.pkl",
            lambda: fit_ward_tree(
                vectors, patients, patient_scope, photo_ids=photos, maximum_rows=5000,
            ),
        )
        if not isinstance(ward_tree, WardTree):
            raise ValueError("Committed Ward payload has an invalid type.")
        cuts = [value for value in (2, 4, 8, 16) if value < len(vectors)]
        if len(cuts) < 2:
            raise ValueError("Ward atlas requires at least three reservoir rows for multiple cuts.")
        ward_reports: list[dict[str, Any]] = []
        for count, labels, partition in ward_multicut(ward_tree, vectors, cuts, photos, patients):
            sample_assignment = _sample_rows(scope, lens, refs, labels, source_map, records)
            base = f"atlas/{_scope_token(scope)}/ward/lens_{lens}/k{count}"
            branch_store.publish(f"{base}/assignments.csv.gz", gzip_csv_bytes(sample_assignment, _SAMPLE_FIELDS))
            crosstab = cluster_case_label_crosstab(labels, patients, [ref["source_case_label"] for ref in refs])
            report = {
                "cluster_count": count,
                "partition": asdict(partition),
                "source_case_label_crosstab": _crosstab_rows(crosstab),
                "assignment_rows": len(sample_assignment),
            }
            branch_store.json(f"{base}/metrics.json", report)
            ward_reports.append(report)

        kmeans_reports: list[dict[str, Any]] = []
        kmeans_stability: list[dict[str, Any]] = []
        for k in _K_VALUES:
            model = _dictionary(
                transforms, train_ids, records, branch_store, scope, lens, k, deadline,
            )
            labels, distances, margins = assign_to_centers(vectors, model.cluster_centers_, chunk_size=512)
            sample_rows = _sample_rows(scope, lens, refs, labels, source_map, records)
            for row, distance, margin in zip(sample_rows, distances, margins, strict=True):
                row["center_distance"] = float(distance)
                row["center_margin"] = float(margin)
            base = f"atlas/{_scope_token(scope)}/kmeans/lens_{lens}/k{k}"
            branch_store.publish(
                f"{base}/assignments.csv.gz",
                gzip_csv_bytes(sample_rows, [*_SAMPLE_FIELDS, "center_distance", "center_margin"]),
            )
            partition = summarize_partition(vectors, labels, photos, patients)
            crosstab = cluster_case_label_crosstab(labels, patients, [ref["source_case_label"] for ref in refs])
            report = {
                "cluster_count": k,
                "partition": asdict(partition),
                "source_case_label_crosstab": _crosstab_rows(crosstab),
                "assignment_rows": len(sample_rows),
            }
            branch_store.json(f"{base}/metrics.json", report)
            kmeans_reports.append(report)
            assignments, comparisons = kmeans_seed_stability(
                vectors,
                patients,
                patient_scope,
                vectors,
                patients,
                n_clusters=k,
                seeds=(42, 43, 44),
                sample_weight=layer["sample_weights"],
                n_init=10,
                batch_size=1024,
                max_iter=100,
            )
            stability_rows = [asdict(comparison) for comparison in comparisons]
            for seed_name, seed_labels in assignments.items():
                seed_assignment = _sample_rows(scope, lens, refs, seed_labels, source_map, records)
                stability_path = f"{base}/seed_assignments/{seed_name}.csv.gz"
                branch_store.publish(stability_path, gzip_csv_bytes(seed_assignment, _SAMPLE_FIELDS))
            branch_store.json(f"{base}/seed_stability.json", {
                "seeds": [42, 43, 44],
                "comparisons": stability_rows,
                "fixed_probe_rows": len(vectors),
            })
            kmeans_stability.append({"k": k, "comparisons": stability_rows, "probe_rows": len(vectors)})

        probe_patients = sorted(set(patients))
        if len(probe_patients) < 2:
            raise ValueError("HDBSCAN patient-resampling stability requires at least two train patients.")
        anchor_patients = set(probe_patients[:min(2, len(probe_patients) - 1)])
        probe_indices = np.flatnonzero(np.isin(patients, list(anchor_patients)))
        density_stability: list[dict[str, Any]] = []
        for mcs, min_samples, method in _ATLAS_STABILITY_CONFIGS:
            if (mcs, min_samples, method) not in density_lookup:
                raise ArithmeticError("Predeclared HDBSCAN stability config was absent from the grid.")
            assignments, comparisons = hdbscan_resampling_stability(
                vectors,
                patients,
                photos,
                patient_scope,
                probe_indices=probe_indices,
                min_cluster_size=mcs,
                min_samples=min_samples,
                selection_method=method,
                seeds=(42, 43, 44),
                patient_fraction=0.8,
            )
            comparison_rows = [asdict(comparison) for comparison in comparisons]
            name = f"atlas/{_scope_token(scope)}/hdbscan/lens_{lens}/stability_mcs{mcs}_ms{min_samples}_{method}.json"
            branch_store.json(name, {
                "config": {"min_cluster_size": mcs, "min_samples": min_samples, "selection_method": method},
                "probe_patient_ids": sorted(anchor_patients),
                "probe_row_count": len(probe_indices),
                "probe_refs": [
                    {"case_id": refs[int(index)]["case_id"],
                     "patient_id": refs[int(index)]["patient_id"],
                     "tile_id": refs[int(index)]["tile_id"],
                     "photo_id": refs[int(index)]["photo_id"]}
                    for index in probe_indices
                ],
                "resample_seeds": [42, 43, 44],
                "assignments": {name: values.tolist() for name, values in assignments.items()},
                "comparisons": comparison_rows,
                "noise_caveat": "Inspect full ARI with jointly-assigned ARI and coverage.",
            })
            density_stability.append({"config": {"min_cluster_size": mcs, "min_samples": min_samples,
                                                  "selection_method": method},
                                      "comparisons": comparison_rows})
        lens_report = {
            "sample_rows": len(vectors),
            "sample_patient_count": len(set(patients)),
            "sample_photo_count": len(set(photos)),
            "hdbscan_grid": density_reports,
            "ward_multicut": ward_reports,
            "kmeans_partitions": kmeans_reports,
            "kmeans_seed_stability": kmeans_stability,
            "hdbscan_patient_resampling_stability": density_stability,
        }
        branch_store.json(f"atlas/{_scope_token(scope)}/lens_{lens}/summary.json", lens_report)
        summary["lenses"][lens] = lens_report
    branch_store.json(f"atlas/{_scope_token(scope)}/summary.json", summary)
    return summary


def _descriptor_for_member_case(
    dataset: Any,
    case_id: str,
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    lens_states: Mapping[str, Any],
) -> FloatVector:
    with threadpool_limits(limits=1):
        projected = project_case(dataset, case_id, dict(transforms))
    if hasattr(dataset, "loaded_tile_order"):
        if dataset.loaded_tile_order(case_id) != source_map["case_tile_order"][case_id]:
            raise ValueError(f"B0 public feature order differs from source map for {case_id!r}.")
    lens_descriptors: dict[str, PhotoBalancedDescriptors] = {}
    for lens in LENSES:
        values = projected[lens]
        photo_ids = photos_for_case(dict(source_map), case_id, lens)
        if not len(values) or len(values) != len(photo_ids):
            raise ValueError(f"B0 requires aligned, nonempty projected features and photo IDs for {case_id}/{lens}.")
        bank = lens_states[lens].bank
        if not isinstance(bank, NegativePrototypeBank) or bank.lens != lens:
            raise ValueError(f"B0 member has an invalid {lens}x bank.")
        with threadpool_limits(limits=1):
            scores = query_novelty(values, bank, chunk_size=512)
        lens_descriptors[lens] = photo_balanced_descriptors(scores.distances, photo_ids)
    return descriptor_vector_by_lens(lens_descriptors, lens_order=LENSES)


def _member_case_descriptors(
    scope: str,
    member: Mapping[str, Any],
    case_ids: Sequence[str],
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    dataset_factory: DatasetFactory,
    store: ArtifactStore,
    deadline: float,
) -> dict[str, Any]:
    ids = _case_ids(list(case_ids), "member query case_ids")
    anchor = member.get("anchor_case_id")
    if anchor is not None and anchor in ids:
        raise ValueError("Member anchor may not be self-scored.")
    if anchor is None:
        # The shared member helper stores the anchor in its opaque identity; the
        # caller passes a private `_anchor_case_id` only in the local wrapper.
        anchor = member.get("_anchor_case_id")
    if not isinstance(anchor, str) or anchor in ids:
        raise ValueError("Member query IDs must exclude its negative anchor.")
    bank_ids = {lens: member["lens_states"][lens].bank.bank_id for lens in LENSES}
    name = (
        f"scopes/{_scope_token(scope)}/members/{member['anchor_token']}/"
        f"descriptors/{_sequence_token(ids)}.pkl"
    )
    prior = store.read(name)
    if prior is not None:
        payload = pickle.loads(prior)
        if (not isinstance(payload, dict) or payload.get("scope") != scope
                or payload.get("member_identity_id") != member["member_identity_id"]
                or payload.get("bank_ids") != bank_ids or payload.get("case_ids") != tuple(ids)):
            raise ValueError("Committed B0 descriptor artifact belongs to another query case order.")
        matrix = np.asarray(payload.get("features"))
        if (matrix.dtype != np.float32 or matrix.shape != (len(ids), 12) or not np.isfinite(matrix).all()):
            raise ValueError("Committed B0 descriptor matrix has invalid shape or values.")
        return payload
    dataset = _make_dataset(dataset_factory, ids)
    rows: list[FloatVector] = []
    for case_id in ids:
        check_deadline(deadline)
        rows.append(_descriptor_for_member_case(
            dataset, case_id, transforms, source_map, member["lens_states"],
        ))
    features = np.vstack(rows).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise FloatingPointError("B0 case descriptors are non-finite.")
    payload = {
        "scope": scope,
        "member_identity_id": member["member_identity_id"],
        "bank_ids": bank_ids,
        "case_ids": tuple(ids),
        "features": features,
    }
    store.publish(name, pickle.dumps(payload, protocol=5))
    return payload


def _fit_b_head(
    store: ArtifactStore,
    scope: str,
    member: Mapping[str, Any],
    train_features: FloatMatrix,
    head_train_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    c_value: float,
    deadline: float,
) -> LinearNoveltyHead:
    lens_states = member["lens_states"]
    bank = lens_states[LENSES[0]].bank
    bank_ids = {lens: lens_states[lens].bank.bank_id for lens in LENSES}
    member_identity = {
        "member_identity_id": member["member_identity_id"],
        "member_id": member["member_id"],
        "all_lens_bank_ids": bank_ids,
    }
    anchor = member["_anchor_case_id"]
    train_membership = tuple(head_train_ids)
    feature_digest = _array_digest(train_features)
    name = (
        f"heads/{_scope_token(scope)}/members/{member['anchor_token']}/"
        f"c{str(c_value).replace('.', '_')}.pkl"
    )

    def fit() -> dict[str, Any]:
        check_deadline(deadline)
        with threadpool_limits(limits=1):
            head = fit_linear_novelty_head(
                train_features,
                [records[case]["target"] for case in head_train_ids],
                head_train_ids,
                bank=bank,
                anchor_case_id=anchor,
                c=c_value,
                random_state=42,
                production=getattr(member.get("_dataset"), "mode", None) != "smoke",
                model_identity=member_identity,
            )
        check_deadline(deadline)
        return {
            "head": head,
            "scope": scope,
            "anchor_token": member["anchor_token"],
            "member_identity_id": member["member_identity_id"],
            "all_lens_bank_ids": bank_ids,
            "train_case_ids": train_membership,
            "regularization_c": float(c_value),
            "feature_digest": feature_digest,
        }

    committed = store.object(name, fit)
    if (not isinstance(committed, dict) or committed.get("scope") != scope
            or committed.get("anchor_token") != member["anchor_token"]
            or committed.get("member_identity_id") != member["member_identity_id"]
            or committed.get("all_lens_bank_ids") != bank_ids
            or committed.get("train_case_ids") != train_membership
            or committed.get("regularization_c") != float(c_value)
            or committed.get("feature_digest") != feature_digest):
        raise ValueError("Committed B0 head belongs to another fold, member, bank set, or training input.")
    head = committed.get("head")
    if not isinstance(head, LinearNoveltyHead) or head.bank_id != bank.bank_id:
        raise ValueError("Committed B0 head does not belong to the member's 4x bank.")
    if head.case_ids != train_membership or head.c != float(c_value):
        raise ValueError("Committed B0 head has different train membership or C.")
    return head


def _prepare_member(
    dataset_all: Any,
    dataset_factory: DatasetFactory,
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    train_ids: Sequence[str],
    anchor: str,
    store: ArtifactStore,
    scope: str,
    deadline: float,
) -> dict[str, Any]:
    try:
        from .dual_member import prepare_member_states
    except ImportError as exc:  # pragma: no cover - integration ordering only
        raise RuntimeError("B0 requires the branch-B dual_member.py helper before execution.") from exc
    with threadpool_limits(limits=1):
        member = prepare_member_states(
            dataset_all,
            dataset_factory,
            dict(transforms),
            dict(source_map),
            list(train_ids),
            anchor,
            store,
            scope,
            deadline,
        )
    if not isinstance(member, Mapping) or set(member.get("lens_states", {})) != set(LENSES):
        raise ValueError("B0 member helper returned invalid per-lens state.")
    expected_train = tuple(case for case in train_ids if case != anchor)
    if member.get("head_train_ids") != expected_train:
        raise ValueError("B0 member helper returned incorrect anchor-excluded head membership/order.")
    result = dict(member)
    result["_anchor_case_id"] = anchor
    result["_dataset"] = dataset_all
    for lens in LENSES:
        state = result["lens_states"][lens]
        if state.bank.anchor_case_id != anchor or state.bank.lens != lens:
            raise ValueError(f"B0 bank anchor/lens mismatch for {scope}/{anchor}/{lens}.")
    return result


def _run_branch_b(
    dataset_all: Any,
    dataset_factory: DatasetFactory,
    splits: Mapping[str, Any],
    outer_rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    source_map: Mapping[str, Any],
    shared_store: ArtifactStore,
    branch_store: ArtifactStore,
    deadline: float,
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    count_oof_rows: list[dict[str, Any]] = []
    outer_reports: list[dict[str, Any]] = []
    logical_head_fits = 0
    count_head_fits = 0
    minimum_anchor_cases = 2 if getattr(dataset_all, "mode", None) == "smoke" else 3
    for outer in outer_rows:
        check_deadline(deadline)
        outer_index = int(outer["fold"])
        inner_losses: dict[float, list[tuple[float, int]]] = {c: [] for c in _C_VALUES}
        for inner in outer["inner_folds"]:
            scope = f"outer_{outer_index}/inner_{int(inner['fold'])}"
            train_ids = list(inner["train_cases"])
            validation_ids = list(inner["validation_cases"])
            transforms = _load_transforms(scope, train_ids, records, source_map, shared_store)
            anchors = [case for case in train_ids if records[case]["target"] == 0]
            if len(anchors) < minimum_anchor_cases:
                raise ValueError(f"B0 scope {scope} has too few negative anchors for the locked class gate.")
            descriptors_by_anchor: dict[str, dict[str, Any]] = {}
            members: dict[str, dict[str, Any]] = {}
            for anchor in anchors:
                member = _prepare_member(
                    dataset_all, dataset_factory, transforms, source_map, train_ids,
                    anchor, branch_store, scope, deadline,
                )
                members[anchor] = member
                head_train_ids = list(member["head_train_ids"])
                query_ids = _ordered_unique([*head_train_ids, *validation_ids])
                descriptors_by_anchor[anchor] = _member_case_descriptors(
                    scope, member, query_ids, transforms, source_map,
                    dataset_factory, branch_store, deadline,
                )
            member_predictions: dict[float, list[FloatVector]] = {c: [] for c in _C_VALUES}
            for anchor in anchors:
                member = members[anchor]
                descriptors = descriptors_by_anchor[anchor]
                train_features, _ = _indices_for_cases(descriptors, member["head_train_ids"])
                validation_features, _ = _indices_for_cases(descriptors, validation_ids)
                for c_value in _C_VALUES:
                    head = _fit_b_head(
                        branch_store, scope, member, train_features, member["head_train_ids"],
                        records, c_value, deadline,
                    )
                    predictions = predict_linear_novelty_head(
                        head, validation_features, validation_ids,
                        bank=member["lens_states"][LENSES[0]].bank,
                    )
                    member_predictions[c_value].append(predictions)
                    logical_head_fits += 1
            for c_value in _C_VALUES:
                ensemble = equal_weight_member_ensemble(
                    member_predictions[c_value],
                    member_query_case_ids=[tuple(validation_ids)] * len(anchors),
                )
                loss = _loss(validation_ids, ensemble, records)
                inner_losses[c_value].append((loss, len(validation_ids)))
                candidate_results.append({
                    "scope": scope,
                    "c": c_value,
                    "anchor_count": len(anchors),
                    "validation_log_loss": loss,
                })
            count_descriptors = _count_descriptors(dataset_factory, source_map, records,
                                                   _ordered_unique([*train_ids, *validation_ids]))
            count_train = _counts_features(count_descriptors, train_ids)
            count_validation = _counts_features(count_descriptors, validation_ids)
            count_model = _fit_a_head(
                branch_store, scope, "counts_only_fixed_c0_1", count_train,
                train_ids, records, 0.1, deadline,
            )
            count_probs = _probability_matrix(count_model, count_validation)
            candidate_results.append({"scope": scope, "candidate": "counts_only",
                                      "validation_log_loss": _loss(validation_ids, count_probs, records)})
            count_head_fits += 1

        if any(len(losses) != len(outer["inner_folds"]) for losses in inner_losses.values()):
            raise ArithmeticError("B0 C candidate is missing an inner ensemble score.")
        inner_means = {
            c: float(sum(loss * count for loss, count in values) / sum(count for _, count in values))
            for c, values in inner_losses.items()
        }
        selected_c = min(_C_VALUES, key=lambda c: (inner_means[c], c))
        refit_scope = f"outer_{outer_index}/refit"
        train_ids = list(outer["train_cases"])
        test_ids = list(outer["test_cases"])
        transforms = _load_transforms(refit_scope, train_ids, records, source_map, shared_store)
        anchors = [case for case in train_ids if records[case]["target"] == 0]
        if len(anchors) < minimum_anchor_cases:
            raise ValueError(f"B0 outer scope {refit_scope} has too few negative anchors.")
        member_predictions: list[FloatVector] = []
        for anchor in anchors:
            member = _prepare_member(
                dataset_all, dataset_factory, transforms, source_map, train_ids,
                anchor, branch_store, refit_scope, deadline,
            )
            head_train_ids = list(member["head_train_ids"])
            query_ids = _ordered_unique([*head_train_ids, *test_ids])
            descriptors = _member_case_descriptors(
                refit_scope, member, query_ids, transforms, source_map,
                dataset_factory, branch_store, deadline,
            )
            train_features, _ = _indices_for_cases(descriptors, head_train_ids)
            test_features, _ = _indices_for_cases(descriptors, test_ids)
            head = _fit_b_head(
                branch_store, refit_scope, member, train_features, head_train_ids,
                records, selected_c, deadline,
            )
            member_predictions.append(predict_linear_novelty_head(
                head, test_features, test_ids, bank=member["lens_states"][LENSES[0]].bank,
            ))
            logical_head_fits += 1
        ensemble = equal_weight_member_ensemble(
            member_predictions,
            member_query_case_ids=[tuple(test_ids)] * len(anchors),
        )
        oof_rows.extend([
            {"case_id": case_id, "patient_id": records[case_id]["patient_id"],
             "target": records[case_id]["target"], "probability": float(probability),
             "scope": refit_scope, "outer_fold": outer_index,
             "selected_c": float(selected_c), "member_count": len(anchors)}
            for case_id, probability in zip(test_ids, ensemble, strict=True)
        ])
        count_descriptors = _count_descriptors(dataset_factory, source_map, records,
                                               _ordered_unique([*train_ids, *test_ids]))
        count_model = _fit_a_head(
            branch_store, refit_scope, "counts_only_fixed_c0_1",
            _counts_features(count_descriptors, train_ids), train_ids, records, 0.1, deadline,
        )
        count_oof_rows.extend(_case_predictions(
            refit_scope, count_model, _counts_features(count_descriptors, test_ids),
            test_ids, records, selected_k=None, selected_c=0.1,
        ))
        count_head_fits += 1
        outer_reports.append({
            "outer_fold": outer_index,
            "inner_mean_ensemble_log_loss": {str(c): inner_means[c] for c in _C_VALUES},
            "selected_c": selected_c,
            "anchor_count": len(anchors),
            "tie_break": "stronger_regularization_lower_c",
        })
    verified_oof = _verify_oof(oof_rows, records)
    verified_counts = _verify_oof(count_oof_rows, records)
    expected_head_fits = (
        2 * sum(sum(records[case]["target"] == 0 for case in inner["train_cases"])
                for outer in outer_rows for inner in outer["inner_folds"])
        + sum(sum(records[case]["target"] == 0 for case in outer["train_cases"]) for outer in outer_rows)
    )
    if logical_head_fits != expected_head_fits:
        raise ArithmeticError(
            f"B0 logical head fit count should be {expected_head_fits}, got {logical_head_fits}."
        )
    result = {
        "schema_version": 1,
        "branch": "B",
        "phase": "complete",
        "method": "B0_anchor_excluded_negative_bank_linear_ensemble",
        "b0": {
            "candidate_c": list(_C_VALUES),
            "outer_selection": outer_reports,
            "inner_candidate_scores": candidate_results,
            "logical_head_fits": logical_head_fits,
            "expected_head_fits": expected_head_fits,
            "oof_count": len(verified_oof),
            "oof_predictions": verified_oof,
        },
        "counts_only_control": {
            "feature_contract": "log1p(tile_count, source_photo_count) for each lens",
            "c": 0.1,
            "logical_head_fits": count_head_fits,
            "oof_count": len(verified_counts),
            "oof_predictions": verified_counts,
        },
    }
    branch_store.json("results/b0_oof.json", verified_oof)
    branch_store.json("results/counts_only_oof.json", verified_counts)
    return result


def _count_descriptors(
    dataset_factory: DatasetFactory,
    source_map: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    case_ids: Sequence[str],
) -> dict[str, Any]:
    ids = _case_ids(list(case_ids), "count descriptor case_ids")
    dataset = _make_dataset(dataset_factory, ids)
    tile_counts: list[int] = []
    photo_counts: list[int] = []
    lens_tile_counts: list[list[int]] = []
    lens_photo_counts: list[list[int]] = []
    for case_id in ids:
        data = dataset.load_case(case_id)
        expected_order = source_map["case_tile_order"][case_id]
        if hasattr(dataset, "loaded_tile_order") and dataset.loaded_tile_order(case_id) != expected_order:
            raise ValueError(f"Count control source order differs for {case_id!r}.")
        total_tiles = 0
        total_photos: set[str] = set()
        case_lens_tile_counts: list[int] = []
        case_lens_photo_counts: list[int] = []
        for lens in LENSES:
            total_tiles += len(data.features[lens])
            lens_photos = photos_for_case(dict(source_map), case_id, lens)
            total_photos.update(lens_photos)
            case_lens_tile_counts.append(len(data.features[lens]))
            case_lens_photo_counts.append(len(set(lens_photos)))
        if not total_tiles or not total_photos:
            raise ValueError(f"Count-only control requires nonempty source bags for {case_id!r}.")
        tile_counts.append(total_tiles)
        photo_counts.append(len(total_photos))
        lens_tile_counts.append(case_lens_tile_counts)
        lens_photo_counts.append(case_lens_photo_counts)
    return {
        "case_ids": tuple(ids),
        "patient_ids": tuple(records[case]["patient_id"] for case in ids),
        "features": np.zeros((len(ids), 1), dtype=np.float32),
        "tile_counts": np.asarray(tile_counts, dtype=np.int64),
        "photo_counts": np.asarray(photo_counts, dtype=np.int64),
        "lens_tile_counts": np.asarray(lens_tile_counts, dtype=np.int64),
        "lens_photo_counts": np.asarray(lens_photo_counts, dtype=np.int64),
    }


__all__ = ["run_linear_study"]
