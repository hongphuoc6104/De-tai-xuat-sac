"""Train-scoped clustering and case descriptors for the dual-atlas study.

The functions in this module consume already transformed ``float32`` feature
matrices. They do not read image data, infer patient identity, or interpret a
cluster as a biological label. Patient and photo identifiers are explicit
scope/grouping metadata and are never appended to model features.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
StringArray = NDArray[np.str_]

DEFAULT_LENSES: tuple[str, str, str] = ("4", "10", "40")
_DEFAULT_CHUNK_SIZE = 512


@dataclass(frozen=True)
class CaseHistogram:
    """Photo-balanced case histograms and their fixed-bin-drop descriptors."""

    patient_ids: StringArray
    photo_counts: IntArray
    full_histograms: FloatArray
    descriptors: FloatArray


@dataclass(frozen=True)
class ClusterGeometry:
    """Size, geometry, and source concentration for one assigned cluster."""

    cluster_id: int
    row_count: int
    row_fraction: float
    patient_count: int
    photo_count: int
    max_patient_row_share: float
    max_photo_row_share: float
    mean_radius: float
    q90_radius: float


@dataclass(frozen=True)
class ClusterCaseLabelCounts:
    """Descriptive source-case-label counts for one partition label."""

    cluster_id: int
    row_counts_by_target: tuple[int, int]
    patient_counts_by_target: tuple[int, int]


@dataclass(frozen=True)
class PartitionSummary:
    """Whole-partition coverage plus metrics for each non-noise cluster."""

    row_count: int
    cluster_count: int
    assigned_count: int
    coverage: float
    noise_fraction: float
    clusters: tuple[ClusterGeometry, ...]


@dataclass(frozen=True)
class HDBSCANConfig:
    """One density-clustering configuration."""

    min_cluster_size: int
    min_samples: int
    selection_method: str


@dataclass(frozen=True)
class HDBSCANRun:
    """Labels and source/geometry summary for one HDBSCAN configuration."""

    config: HDBSCANConfig
    labels: IntArray
    summary: PartitionSummary


@dataclass(frozen=True)
class WardTree:
    """A fitted Ward linkage matrix tied to its original row count."""

    linkage_matrix: NDArray[np.float64]
    row_count: int
    feature_sha256: str
    patient_sha256: str
    photo_sha256: str


@dataclass(frozen=True)
class StabilityComparison:
    """Pairwise stability with separate handling for density noise."""

    left_name: str
    right_name: str
    full_probe_ari: float
    jointly_assigned_ari: float | None
    jointly_assigned_count: int
    jointly_assigned_fraction: float
    left_coverage: float
    right_coverage: float


def _features(values: object, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; implicit feature conversion is forbidden.")
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional N×D array with D ≥ 1.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    for start in range(0, array.shape[0], _DEFAULT_CHUNK_SIZE):
        if not np.isfinite(array[start:start + _DEFAULT_CHUNK_SIZE]).all():
            raise ValueError(f"{name} contains non-finite feature values.")
    return np.ascontiguousarray(array)


def _ids(values: object, name: str, *, expected_length: int | None = None) -> StringArray:
    array = np.asarray(values)
    if array.ndim != 1 or (expected_length is not None and len(array) != expected_length):
        suffix = f" with length {expected_length}" if expected_length is not None else ""
        raise ValueError(f"{name} must be a one-dimensional string array{suffix}.")
    if array.size == 0:
        return np.asarray([], dtype=np.str_)
    if array.dtype.kind not in {"U", "S", "O"}:
        raise TypeError(f"{name} must contain strings.")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{name} contains a non-UTF-8 identifier.") from exc
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} contains an empty or untrimmed identifier.")
        normalized.append(value)
    return np.asarray(normalized, dtype=np.str_)


def _positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer ≥ {minimum}.")
    return int(value)


def _readonly(array: NDArray) -> NDArray:
    array.setflags(write=False)
    return array


def _array_sha256(array: NDArray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def guard_train_scope(
    sample_patient_ids: object,
    train_patient_ids: object,
    *,
    forbidden_patient_ids: object = (),
    context: str = "fit",
) -> None:
    """Reject any row outside the declared training patients before fitting.

    ``forbidden_patient_ids`` lets a caller pass the held-out fold explicitly,
    so an invalid or accidentally widened train manifest fails before model
    construction. IDs are only checked here; they are not model inputs.
    """
    samples = _ids(sample_patient_ids, "sample_patient_ids")
    train = _ids(train_patient_ids, "train_patient_ids")
    forbidden = _ids(forbidden_patient_ids, "forbidden_patient_ids")
    train_set = set(train.tolist())
    forbidden_set = set(forbidden.tolist())
    overlap = train_set & forbidden_set
    if overlap:
        raise ValueError(f"{context} scope includes forbidden held-out patients: {sorted(overlap)[:3]}.")
    outside = set(samples.tolist()) - train_set
    if outside:
        raise ValueError(f"{context} rows include patients outside the training scope: {sorted(outside)[:3]}.")
    if not len(samples):
        raise ValueError(f"{context} scope cannot be empty.")


def _training_arrays(
    values: object,
    patient_ids: object,
    train_patient_ids: object,
    *,
    forbidden_patient_ids: object = (),
    context: str,
) -> tuple[FloatArray, StringArray]:
    matrix = _features(values, "features")
    patients = _ids(patient_ids, "patient_ids", expected_length=len(matrix))
    guard_train_scope(
        patients,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context=context,
    )
    return matrix, patients


def _weights(values: object | None, row_count: int) -> NDArray[np.float64]:
    if values is None:
        return np.ones(row_count, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != row_count:
        raise ValueError("sample_weight must be a one-dimensional array matching the feature rows.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample_weight values must be finite and strictly positive.")
    if not np.isfinite(weights.sum(dtype=np.float64)):
        raise ValueError("sample_weight total must be finite.")
    return weights


def fit_kmeans_dictionary(
    features: object,
    patient_ids: object,
    train_patient_ids: object,
    *,
    n_clusters: int,
    sample_weight: object | None = None,
    random_state: int = 42,
    n_init: int = 10,
    batch_size: int = 1024,
    max_iter: int = 100,
    forbidden_patient_ids: object = (),
) -> MiniBatchKMeans:
    """Fit a weighted, seeded MiniBatchKMeans dictionary on train patients.

    Reservoir construction and any L2/PCA transform happen upstream. Distinct
    rows are checked exactly so a requested K cannot silently exceed the
    available prototypes. Native numerical libraries are limited to one
    thread during fitting.
    """
    matrix, _ = _training_arrays(
        features,
        patient_ids,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="MiniBatchKMeans",
    )
    clusters = _positive_int(n_clusters, "n_clusters", minimum=2)
    if clusters > len(matrix):
        raise ValueError("n_clusters cannot exceed the number of training rows.")
    distinct = np.unique(matrix, axis=0).shape[0]
    if clusters > distinct:
        raise ValueError(f"n_clusters={clusters} exceeds the {distinct} distinct training vectors.")
    seed = _positive_int(random_state, "random_state", minimum=0)
    starts = _positive_int(n_init, "n_init")
    batch = _positive_int(batch_size, "batch_size", minimum=2)
    iterations = _positive_int(max_iter, "max_iter")
    weights = _weights(sample_weight, len(matrix))
    estimator = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=starts,
        batch_size=batch,
        max_iter=iterations,
        reassignment_ratio=0.01,
    )
    with threadpool_limits(limits=1):
        estimator.fit(matrix, sample_weight=weights)
    return estimator


def assign_to_centers(
    features: object,
    centers: object,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Assign rows by nearest Euclidean center with bounded chunk buffers.

    Returns ``(cluster_id, nearest_center_distance, second_minus_nearest)``.
    The distance/margin arrays contain one value per input row; the temporary
    distance matrix is at most ``chunk_size × number_of_centers``.
    """
    matrix = _features(features, "features")
    center_matrix = _features(centers, "centers")
    if matrix.shape[1] != center_matrix.shape[1] or len(center_matrix) < 2:
        raise ValueError("centers must have matching feature width and at least two rows.")
    chunk = _positive_int(chunk_size, "chunk_size")
    center_norm = np.einsum("ij,ij->i", center_matrix, center_matrix)
    labels = np.empty(len(matrix), dtype=np.int32)
    distances = np.empty(len(matrix), dtype=np.float32)
    margins = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(matrix), chunk):
        stop = min(start + chunk, len(matrix))
        block = matrix[start:stop]
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + center_norm[None, :]
            - np.float32(2.0) * (block @ center_matrix.T)
        )
        np.maximum(squared, np.float32(0.0), out=squared)
        nearest_two = np.argpartition(squared, kth=1, axis=1)[:, :2]
        nearest_values = np.take_along_axis(squared, nearest_two, axis=1)
        order = np.argsort(nearest_values, axis=1)
        nearest_two = np.take_along_axis(nearest_two, order, axis=1)
        nearest_values = np.take_along_axis(nearest_values, order, axis=1)
        closest = np.sqrt(nearest_values[:, 0]).astype(np.float32, copy=False)
        second = np.sqrt(nearest_values[:, 1]).astype(np.float32, copy=False)
        labels[start:stop] = nearest_two[:, 0]
        distances[start:stop] = closest
        margins[start:stop] = second - closest
    return _readonly(labels), _readonly(distances), _readonly(margins)


def photo_balanced_histogram(
    cluster_ids: object,
    photo_ids: object,
    patient_ids: object,
    *,
    n_clusters: int,
) -> CaseHistogram:
    """Average normalized per-photo histograms within each patient.

    Every source photo receives equal weight regardless of its tile count. The
    last, fixed ``K-1`` bin is omitted from ``descriptors`` because each full
    histogram sums to one.
    """
    labels = np.asarray(cluster_ids)
    if labels.ndim != 1 or labels.dtype.kind not in {"i", "u"} or len(labels) == 0:
        raise ValueError("cluster_ids must be a nonempty one-dimensional integer array.")
    clusters = _positive_int(n_clusters, "n_clusters", minimum=2)
    if np.any(labels < 0) or np.any(labels >= clusters):
        raise ValueError("cluster_ids must be in [0, n_clusters); noise is not a KMeans histogram bin.")
    photos = _ids(photo_ids, "photo_ids", expected_length=len(labels))
    patients = _ids(patient_ids, "patient_ids", expected_length=len(labels))
    photo_patient: dict[str, str] = {}
    per_patient_photo_counts: dict[str, list[NDArray[np.float64]]] = {}
    for photo in np.unique(photos):
        indices = np.flatnonzero(photos == photo)
        owners = np.unique(patients[indices])
        if len(owners) != 1:
            raise ValueError(f"source photo {photo!r} maps to multiple patients.")
        patient = str(owners[0])
        if photo in photo_patient and photo_patient[photo] != patient:
            raise ValueError(f"source photo {photo!r} has inconsistent patient ownership.")
        photo_patient[str(photo)] = patient
        counts = np.bincount(labels[indices].astype(np.int64), minlength=clusters).astype(np.float64)
        per_patient_photo_counts.setdefault(patient, []).append(counts / float(len(indices)))
    patient_order = np.asarray(sorted(per_patient_photo_counts), dtype=np.str_)
    histograms = np.vstack([
        np.mean(per_patient_photo_counts[str(patient)], axis=0) for patient in patient_order
    ]).astype(np.float32)
    if not np.allclose(histograms.sum(axis=1), 1.0, rtol=0, atol=2e-6):
        raise ArithmeticError("photo-balanced histograms failed the unit-sum invariant.")
    photo_counts = np.asarray(
        [len(per_patient_photo_counts[str(patient)]) for patient in patient_order],
        dtype=np.int32,
    )
    return CaseHistogram(
        patient_ids=_readonly(patient_order),
        photo_counts=_readonly(photo_counts),
        full_histograms=_readonly(histograms),
        descriptors=_readonly(histograms[:, :-1].copy()),
    )


def fit_case_classifier(
    descriptors: object,
    targets: object,
    patient_ids: object,
    train_patient_ids: object,
    *,
    regularization_c: float,
    random_state: int = 42,
    forbidden_patient_ids: object = (),
    max_iter: int = 2000,
) -> Pipeline:
    """Fit train-only StandardScaler plus L2 case-level logistic regression.

    One row must represent one unique patient/case. The classifier is
    unweighted and consumes only the numeric descriptor matrix and reviewed
    binary case targets; IDs are checked for scope and discarded before fit.
    """
    matrix = _features(descriptors, "descriptors")
    patients = _ids(patient_ids, "patient_ids", expected_length=len(matrix))
    if len(set(patients.tolist())) != len(patients):
        raise ValueError("patient_ids must be unique: each classifier row represents one case.")
    guard_train_scope(
        patients,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="case classifier",
    )
    labels_raw = np.asarray(targets)
    if labels_raw.ndim != 1 or len(labels_raw) != len(matrix):
        raise ValueError("targets must be one-dimensional and match the descriptor rows.")
    if labels_raw.dtype.kind not in {"i", "u"} or not np.isin(labels_raw, (0, 1)).all():
        raise ValueError("targets must contain reviewed integer case labels 0 or 1.")
    labels = labels_raw.astype(np.int32, copy=False)
    if len(np.unique(labels)) != 2:
        raise ValueError("case classifier training requires both case-level classes.")
    if isinstance(regularization_c, bool) or not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("regularization_c must be finite and strictly positive.")
    seed = _positive_int(random_state, "random_state", minimum=0)
    iterations = _positive_int(max_iter, "max_iter")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(regularization_c),
            fit_intercept=True,
            class_weight=None,
            solver="lbfgs",
            l1_ratio=0.0,
            max_iter=iterations,
            random_state=seed,
        )),
    ])
    with threadpool_limits(limits=1):
        model.fit(matrix, labels)
    return model


def summarize_partition(
    features: object,
    cluster_ids: object,
    photo_ids: object,
    patient_ids: object,
    *,
    geometry_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> PartitionSummary:
    """Measure coverage, noise, cluster radii, and patient/photo concentration."""
    matrix = _features(features, "features")
    labels = np.asarray(cluster_ids)
    if labels.ndim != 1 or len(labels) != len(matrix) or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("cluster_ids must be an integer vector matching the feature rows.")
    if np.any(labels < -1):
        raise ValueError("cluster_ids may use -1 for noise; other negative IDs are invalid.")
    photos = _ids(photo_ids, "photo_ids", expected_length=len(matrix))
    patients = _ids(patient_ids, "patient_ids", expected_length=len(matrix))
    chunk = _positive_int(geometry_chunk_size, "geometry_chunk_size")
    cluster_ids_sorted = np.unique(labels[labels >= 0])
    assigned_count = int(np.count_nonzero(labels >= 0))
    geometry: list[ClusterGeometry] = []
    for cluster_id in cluster_ids_sorted.tolist():
        indices = np.flatnonzero(labels == cluster_id)
        cluster_features = matrix[indices]
        centroid = cluster_features.mean(axis=0, dtype=np.float64).astype(np.float32)
        radii = np.empty(len(indices), dtype=np.float32)
        for start in range(0, len(indices), chunk):
            stop = min(start + chunk, len(indices))
            delta = cluster_features[start:stop] - centroid
            radii[start:stop] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        _, patient_counts = np.unique(patients[indices], return_counts=True)
        _, photo_counts = np.unique(photos[indices], return_counts=True)
        geometry.append(ClusterGeometry(
            cluster_id=int(cluster_id),
            row_count=len(indices),
            row_fraction=len(indices) / len(matrix),
            patient_count=len(patient_counts),
            photo_count=len(photo_counts),
            max_patient_row_share=float(patient_counts.max() / len(indices)),
            max_photo_row_share=float(photo_counts.max() / len(indices)),
            mean_radius=float(radii.mean(dtype=np.float64)),
            q90_radius=float(np.quantile(radii, 0.9)),
        ))
    coverage = assigned_count / len(matrix)
    return PartitionSummary(
        row_count=len(matrix),
        cluster_count=len(geometry),
        assigned_count=assigned_count,
        coverage=float(coverage),
        noise_fraction=float(1.0 - coverage),
        clusters=tuple(geometry),
    )


def cluster_case_label_crosstab(
    cluster_ids: object,
    patient_ids: object,
    case_targets: object,
) -> tuple[ClusterCaseLabelCounts, ...]:
    """Count sampled rows and patients by source case target per cluster.

    This is a descriptive atlas diagnostic only. Targets are never used to fit
    or assign clusters. Noise (label ``-1``) is retained as its own row so the
    crosstab does not silently omit unassigned samples.
    """
    labels = np.asarray(cluster_ids)
    if labels.ndim != 1 or len(labels) == 0 or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("cluster_ids must be a nonempty one-dimensional integer vector.")
    if np.any(labels < -1):
        raise ValueError("cluster_ids may use -1 for noise; other negative IDs are invalid.")
    patients = _ids(patient_ids, "patient_ids", expected_length=len(labels))
    targets = np.asarray(case_targets)
    if targets.ndim != 1 or len(targets) != len(labels):
        raise ValueError("case_targets must match cluster_ids and patient_ids.")
    if targets.dtype.kind not in {"i", "u"} or not np.isin(targets, (0, 1)).all():
        raise ValueError("case_targets must contain integer case labels 0 or 1.")
    for patient in np.unique(patients):
        if len(np.unique(targets[patients == patient])) != 1:
            raise ValueError(f"patient {patient!r} has conflicting source case labels.")
    rows: list[ClusterCaseLabelCounts] = []
    for cluster_id in np.unique(labels).tolist():
        cluster_mask = labels == cluster_id
        row_counts = tuple(int(np.count_nonzero(cluster_mask & (targets == target))) for target in (0, 1))
        patient_counts = tuple(
            int(len(np.unique(patients[cluster_mask & (targets == target)]))) for target in (0, 1)
        )
        rows.append(ClusterCaseLabelCounts(int(cluster_id), row_counts, patient_counts))
    return tuple(rows)


def run_hdbscan_grid(
    features: object,
    patient_ids: object,
    photo_ids: object,
    train_patient_ids: object,
    *,
    min_cluster_sizes: Sequence[int] = (8, 16, 32, 64),
    min_samples_values: Sequence[int] = (4, 8),
    selection_methods: Sequence[str] = ("eom", "leaf"),
    forbidden_patient_ids: object = (),
) -> tuple[HDBSCANRun, ...]:
    """Run an explicitly requested HDBSCAN grid on train-scoped rows only."""
    matrix, patients = _training_arrays(
        features,
        patient_ids,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="HDBSCAN atlas",
    )
    photos = _ids(photo_ids, "photo_ids", expected_length=len(matrix))
    sizes = tuple(_positive_int(v, "min_cluster_size", minimum=2) for v in min_cluster_sizes)
    samples = tuple(_positive_int(v, "min_samples") for v in min_samples_values)
    methods = tuple(selection_methods)
    if not sizes or not samples or not methods:
        raise ValueError("HDBSCAN grid dimensions cannot be empty.")
    if any(size > len(matrix) for size in sizes):
        raise ValueError("min_cluster_size cannot exceed the number of training rows.")
    if any(method not in {"eom", "leaf"} for method in methods):
        raise ValueError("selection_methods must contain only 'eom' and/or 'leaf'.")
    runs: list[HDBSCANRun] = []
    with threadpool_limits(limits=1):
        for size, min_samples, method in product(sizes, samples, methods):
            estimator = HDBSCAN(
                min_cluster_size=size,
                min_samples=min_samples,
                cluster_selection_method=method,
                n_jobs=1,
                copy=False,
            )
            labels = estimator.fit_predict(matrix).astype(np.int32, copy=False)
            summary = summarize_partition(matrix, labels, photos, patients)
            runs.append(HDBSCANRun(
                config=HDBSCANConfig(size, min_samples, method),
                labels=_readonly(labels.copy()),
                summary=summary,
            ))
    return tuple(runs)


def fit_ward_tree(
    features: object,
    patient_ids: object,
    train_patient_ids: object,
    *,
    photo_ids: object,
    maximum_rows: int = 5000,
    forbidden_patient_ids: object = (),
) -> WardTree:
    """Build a bounded Ward linkage tree over the train-only reservoir."""
    matrix, patients = _training_arrays(
        features,
        patient_ids,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="Ward tree",
    )
    photos = _ids(photo_ids, "photo_ids", expected_length=len(matrix))
    cap = _positive_int(maximum_rows, "maximum_rows", minimum=2)
    if len(matrix) < 2:
        raise ValueError("Ward linkage requires at least two rows.")
    if len(matrix) > cap:
        raise ValueError(f"Ward input has {len(matrix)} rows, above the configured cap of {cap}.")
    with threadpool_limits(limits=1):
        tree = linkage(matrix, method="ward", optimal_ordering=False)
    return WardTree(
        linkage_matrix=_readonly(np.asarray(tree, dtype=np.float64)),
        row_count=len(matrix),
        feature_sha256=_array_sha256(matrix),
        patient_sha256=_array_sha256(patients),
        photo_sha256=_array_sha256(photos),
    )


def ward_multicut(
    tree: WardTree,
    features: object,
    cluster_counts: Sequence[int],
    photo_ids: object,
    patient_ids: object,
) -> tuple[tuple[int, IntArray, PartitionSummary], ...]:
    """Cut one Ward tree at several cluster counts and summarize each cut."""
    matrix = _features(features, "features")
    patients = _ids(patient_ids, "patient_ids", expected_length=len(matrix))
    photos = _ids(photo_ids, "photo_ids", expected_length=len(matrix))
    if not isinstance(tree, WardTree) or tree.row_count != len(matrix):
        raise ValueError("WardTree row_count must match the original feature matrix.")
    if _array_sha256(matrix) != tree.feature_sha256:
        raise ValueError("WardTree feature values or row order do not match the original fit.")
    if _array_sha256(patients) != tree.patient_sha256:
        raise ValueError("WardTree patient metadata or row order do not match the original fit.")
    if _array_sha256(photos) != tree.photo_sha256:
        raise ValueError("WardTree photo metadata or row order do not match the original fit.")
    cuts = tuple(_positive_int(k, "cluster_count") for k in cluster_counts)
    if not cuts:
        raise ValueError("cluster_counts cannot be empty.")
    if any(k >= len(matrix) for k in cuts):
        raise ValueError("each Ward cut must be smaller than the number of input rows.")
    result = cut_tree(tree.linkage_matrix, n_clusters=sorted(set(cuts)))
    by_count: dict[int, IntArray] = {}
    for column, count in enumerate(sorted(set(cuts))):
        by_count[count] = result[:, column].astype(np.int32, copy=False)
    return tuple(
        (count, _readonly(by_count[count].copy()), summarize_partition(matrix, by_count[count], photos, patients))
        for count in cuts
    )


def compare_clusterings(assignments: Mapping[str, object]) -> tuple[StabilityComparison, ...]:
    """Compare assignments on the same probe, exposing density-noise coverage.

    ``full_probe_ari`` treats -1 as a label. ``jointly_assigned_ari`` excludes
    every row that either fit called noise, so readers can see whether a high
    full-set score came from widespread shared noise. Joint coverage is always
    reported alongside the restricted ARI.
    """
    if not isinstance(assignments, Mapping) or len(assignments) < 2:
        raise ValueError("assignments must contain at least two named label vectors.")
    names = sorted(assignments)
    labels: dict[str, IntArray] = {}
    expected_length: int | None = None
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("assignment names must be nonempty strings.")
        array = np.asarray(assignments[name])
        if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
            raise ValueError(f"assignments[{name!r}] must be a one-dimensional integer vector.")
        if np.any(array < -1):
            raise ValueError("cluster assignments may use -1 for noise; other negative IDs are invalid.")
        if expected_length is None:
            expected_length = len(array)
        elif len(array) != expected_length:
            raise ValueError("all assignments must refer to the same fixed probe rows.")
        if len(array) < 1:
            raise ValueError("assignment vectors cannot be empty.")
        labels[name] = array.astype(np.int32, copy=False)
    comparisons: list[StabilityComparison] = []
    for left_name, right_name in combinations(names, 2):
        left = labels[left_name]
        right = labels[right_name]
        paired = (left >= 0) & (right >= 0)
        paired_count = int(np.count_nonzero(paired))
        assigned_ari = (
            float(adjusted_rand_score(left[paired], right[paired])) if paired_count >= 2 else None
        )
        comparisons.append(StabilityComparison(
            left_name=left_name,
            right_name=right_name,
            full_probe_ari=float(adjusted_rand_score(left, right)),
            jointly_assigned_ari=assigned_ari,
            jointly_assigned_count=paired_count,
            jointly_assigned_fraction=paired_count / len(left),
            left_coverage=float(np.count_nonzero(left >= 0) / len(left)),
            right_coverage=float(np.count_nonzero(right >= 0) / len(right)),
        ))
    return tuple(comparisons)


def kmeans_seed_stability(
    features: object,
    patient_ids: object,
    train_patient_ids: object,
    probe_features: object,
    probe_patient_ids: object,
    *,
    n_clusters: int,
    seeds: Sequence[int] = (42, 43, 44),
    sample_weight: object | None = None,
    n_init: int = 10,
    batch_size: int = 1024,
    max_iter: int = 100,
    forbidden_patient_ids: object = (),
) -> tuple[dict[str, IntArray], tuple[StabilityComparison, ...]]:
    """Fit seeded KMeans dictionaries and compare predictions on one probe."""
    matrix, _ = _training_arrays(
        features,
        patient_ids,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="KMeans stability",
    )
    probe = _features(probe_features, "probe_features")
    probe_patients = _ids(probe_patient_ids, "probe_patient_ids", expected_length=len(probe))
    guard_train_scope(probe_patients, train_patient_ids, context="KMeans fixed probe")
    if probe.shape[1] != matrix.shape[1]:
        raise ValueError("probe_features width must match training feature width.")
    seed_values = tuple(_positive_int(seed, "seed", minimum=0) for seed in seeds)
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain at least two unique values.")
    assignments: dict[str, IntArray] = {}
    for seed in seed_values:
        estimator = fit_kmeans_dictionary(
            matrix,
            patient_ids,
            train_patient_ids,
            n_clusters=n_clusters,
            sample_weight=sample_weight,
            random_state=seed,
            n_init=n_init,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        assigned, _, _ = assign_to_centers(probe, estimator.cluster_centers_)
        assignments[f"seed-{seed}"] = _readonly(assigned)
    return assignments, compare_clusterings(assignments)


def _patient_subsample_indices(
    patient_ids: StringArray,
    probe_indices: NDArray[np.int64],
    *,
    fraction: float,
    seed: int,
) -> NDArray[np.int64]:
    probe_patients = set(patient_ids[probe_indices].tolist())
    all_patients = set(patient_ids.tolist())
    resample_pool = sorted(all_patients - probe_patients)
    if not resample_pool:
        raise ValueError("patient resampling needs at least one non-probe patient group.")
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("patient_fraction must be strictly between zero and one.")
    take = max(1, min(len(resample_pool), int(np.ceil(len(resample_pool) * fraction))))
    rng = np.random.default_rng(seed)
    selected_patients = probe_patients | set(rng.choice(resample_pool, size=take, replace=False).tolist())
    return np.flatnonzero(np.isin(patient_ids, list(selected_patients))).astype(np.int64)


def hdbscan_resampling_stability(
    features: object,
    patient_ids: object,
    photo_ids: object,
    train_patient_ids: object,
    *,
    probe_indices: object,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str,
    seeds: Sequence[int] = (42, 43, 44),
    patient_fraction: float = 0.8,
    forbidden_patient_ids: object = (),
) -> tuple[dict[str, IntArray], tuple[StabilityComparison, ...]]:
    """Refit HDBSCAN on patient subsamples and compare a fixed train-only probe.

    All rows from patients represented in the fixed probe are retained in each
    fit; other train patients are sampled without replacement. This ensures
    identical probe rows while perturbing the surrounding patient mixture.
    It is an unsupervised stability analysis with shared anchors, not an
    out-of-sample HDBSCAN prediction API.
    """
    matrix, patients = _training_arrays(
        features,
        patient_ids,
        train_patient_ids,
        forbidden_patient_ids=forbidden_patient_ids,
        context="HDBSCAN resampling stability",
    )
    photos = _ids(photo_ids, "photo_ids", expected_length=len(matrix))
    probe = np.asarray(probe_indices)
    if probe.ndim != 1 or probe.dtype.kind not in {"i", "u"} or len(probe) < 2:
        raise ValueError("probe_indices must be a one-dimensional integer vector with at least two rows.")
    probe_indices_array = probe.astype(np.int64, copy=False)
    if len(np.unique(probe_indices_array)) != len(probe_indices_array):
        raise ValueError("probe_indices must be unique.")
    if np.any(probe_indices_array < 0) or np.any(probe_indices_array >= len(matrix)):
        raise ValueError("probe_indices contains a row outside the training matrix.")
    size = _positive_int(min_cluster_size, "min_cluster_size", minimum=2)
    samples = _positive_int(min_samples, "min_samples")
    if size > len(matrix):
        raise ValueError("min_cluster_size cannot exceed the number of training rows.")
    if selection_method not in {"eom", "leaf"}:
        raise ValueError("selection_method must be 'eom' or 'leaf'.")
    seed_values = tuple(_positive_int(seed, "seed", minimum=0) for seed in seeds)
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain at least two unique values.")
    assignments: dict[str, IntArray] = {}
    for seed in seed_values:
        selected = _patient_subsample_indices(
            patients,
            probe_indices_array,
            fraction=patient_fraction,
            seed=seed,
        )
        selected_position = {int(row): pos for pos, row in enumerate(selected.tolist())}
        if any(int(row) not in selected_position for row in probe_indices_array):
            raise ArithmeticError("fixed probe rows were lost while constructing patient resample.")
        with threadpool_limits(limits=1):
            estimator = HDBSCAN(
                min_cluster_size=size,
                min_samples=samples,
                cluster_selection_method=selection_method,
                n_jobs=1,
                copy=False,
            )
            fitted = estimator.fit_predict(matrix[selected]).astype(np.int32, copy=False)
        positions = np.asarray([selected_position[int(row)] for row in probe_indices_array], dtype=np.int64)
        assignments[f"patient-resample-{seed}"] = _readonly(fitted[positions].copy())
    # Validate provenance arrays here as part of the atlas contract even though
    # stability itself consumes only fixed-probe assignments.
    if len(photos) != len(patients):  # defensive; _ids already enforces this
        raise ArithmeticError("photo and patient metadata lost row alignment.")
    return assignments, compare_clusterings(assignments)
