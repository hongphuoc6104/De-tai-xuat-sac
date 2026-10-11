"""Reference-bank novelty scores and case-level descriptor controls.

All functions in this module operate on already transformed, float32 patch
features. Image/source-photo identifiers and case metadata are explicit inputs
used for provenance and balanced aggregation; they are never model features.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Float32Matrix = NDArray[np.float32]
Float32Vector = NDArray[np.float32]

_DESCRIPTOR_NAMES = ("median", "q90", "q95", "tail5_mean")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _float32_matrix(value: Any, name: str) -> Float32Matrix:
    matrix = np.asarray(value)
    if matrix.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; received {matrix.dtype}.")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [rows, features] matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(matrix)


def _float32_vector(value: Any, name: str) -> Float32Vector:
    vector = np.asarray(value)
    if vector.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; received {vector.dtype}.")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector)


def _metadata_rows(metadata: Sequence[Mapping[str, Any]], length: int) -> tuple[Mapping[str, Any], ...]:
    if isinstance(metadata, (str, bytes)) or len(metadata) != length:
        raise ValueError("metadata must contain one mapping per feature row.")
    rows = tuple(metadata)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("Every metadata row must be a mapping.")
    return rows


def _hash_payload(payload: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NegativePrototypeBank:
    """One anchor-owned, immutable set of negative prototypes for a single lens."""

    lens: str
    anchor_case_id: str
    prototypes: Float32Matrix
    prototype_row_ids: tuple[str, ...]
    prototype_photo_ids: tuple[str, ...]
    bank_id: str = ""

    def __post_init__(self) -> None:
        lens = _identifier(self.lens, "lens")
        anchor = _identifier(self.anchor_case_id, "anchor_case_id")
        prototypes = _float32_matrix(self.prototypes, "prototypes").copy()
        if prototypes.shape[0] != 32:
            raise ValueError("The active negative-reference protocol requires exactly 32 prototypes per lens.")
        row_ids = tuple(_identifier(value, "prototype_row_id") for value in self.prototype_row_ids)
        photo_ids = tuple(_identifier(value, "prototype_photo_id") for value in self.prototype_photo_ids)
        if len(row_ids) != prototypes.shape[0] or len(photo_ids) != prototypes.shape[0]:
            raise ValueError("Prototype row and source-photo provenance must match the prototype count.")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Each prototype must resolve to a distinct source feature row.")
        if np.unique(prototypes, axis=0).shape[0] != prototypes.shape[0]:
            raise ValueError("Prototype bank contains duplicate vectors.")
        payload = {"version": 1, "lens": lens, "anchor_case_id": anchor,
                   "prototype_row_ids": row_ids, "prototype_photo_ids": photo_ids}
        expected_id = _hash_payload(payload, (prototypes,))
        if self.bank_id and self.bank_id != expected_id:
            raise ValueError("bank_id does not match the prototype payload and provenance.")
        prototypes.setflags(write=False)
        object.__setattr__(self, "lens", lens)
        object.__setattr__(self, "anchor_case_id", anchor)
        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "prototype_row_ids", row_ids)
        object.__setattr__(self, "prototype_photo_ids", photo_ids)
        object.__setattr__(self, "bank_id", expected_id)


def fit_negative_prototype_bank(
    vectors: Any,
    metadata: Sequence[Mapping[str, Any]],
    *,
    lens: str,
    anchor_case_id: str,
    n_prototypes: int = 32,
    random_state: int = 42,
) -> NegativePrototypeBank:
    """Fit a fixed-size, photo-balanced KMeans bank from one negative anchor.

    ``vectors`` must already be transformed with the scope's train-only feature
    transform. Each metadata mapping must provide ``case_id``, integer
    ``target``, ``photo_id`` and unique ``row_id`` provenance. This routine
    refuses mixed-case, positive, undersized, or duplicate-vector banks.
    """
    matrix = _float32_matrix(vectors, "vectors")
    anchor = _identifier(anchor_case_id, "anchor_case_id")
    lens_name = _identifier(lens, "lens")
    if isinstance(n_prototypes, bool) or not isinstance(n_prototypes, int) or n_prototypes != 32:
        raise ValueError("The active negative-reference protocol requires exactly 32 prototypes per lens.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer.")
    rows = _metadata_rows(metadata, matrix.shape[0])
    case_ids: list[str] = []
    photo_ids: list[str] = []
    row_ids: list[str] = []
    for row in rows:
        case_id = _identifier(row.get("case_id"), "metadata.case_id")
        target = row.get("target")
        if type(target) is not int or target != 0:
            raise ValueError("Negative prototype rows must have reviewed integer target 0.")
        case_ids.append(case_id)
        photo_ids.append(_identifier(row.get("photo_id"), "metadata.photo_id"))
        row_ids.append(_identifier(row.get("row_id"), "metadata.row_id"))
    if set(case_ids) != {anchor}:
        raise ValueError("A negative bank may contain rows from its anchor case only.")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Anchor feature row IDs must be unique.")
    if matrix.shape[0] < n_prototypes or np.unique(matrix, axis=0).shape[0] < n_prototypes:
        raise ValueError(
            f"Bank requires at least {n_prototypes} rows and distinct vectors; no padding is allowed."
        )

    photo_counts = Counter(photo_ids)
    sample_weight = np.asarray([1.0 / photo_counts[value] for value in photo_ids], dtype=np.float64)
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state, n_init=10, algorithm="lloyd")
    labels = kmeans.fit_predict(matrix, sample_weight=sample_weight)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    if centers.shape != (n_prototypes, matrix.shape[1]):
        raise ValueError("KMeans returned an unexpected prototype shape.")
    if not np.isfinite(centers).all() or np.unique(centers, axis=0).shape[0] != n_prototypes:
        raise ValueError("KMeans produced fewer distinct prototypes than requested.")

    reference_indices: list[int] = []
    for prototype_index in range(n_prototypes):
        members = np.flatnonzero(labels == prototype_index)
        if members.size == 0:
            raise ValueError("KMeans produced an empty prototype; refusing to pad the bank.")
        delta = matrix[members] - centers[prototype_index]
        local = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        reference_indices.append(int(members[local]))
    if len(set(reference_indices)) != n_prototypes:
        raise ValueError("Prototype provenance did not resolve to distinct source rows.")
    return NegativePrototypeBank(
        lens=lens_name,
        anchor_case_id=anchor,
        prototypes=centers,
        prototype_row_ids=tuple(row_ids[index] for index in reference_indices),
        prototype_photo_ids=tuple(photo_ids[index] for index in reference_indices),
    )


@dataclass(frozen=True)
class NoveltyScores:
    """Bounded CPU query output with distance and prototype-index per patch."""

    distances: Float32Vector
    nearest_prototype_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        distances = _float32_vector(self.distances, "distances").copy()
        indices = np.asarray(self.nearest_prototype_indices)
        if indices.dtype != np.int64 or indices.ndim != 1 or indices.shape != distances.shape:
            raise ValueError("nearest_prototype_indices must be an int64 vector matching distances.")
        if np.any(indices < 0):
            raise ValueError("nearest prototype indices must be nonnegative.")
        distances.setflags(write=False)
        indices = np.ascontiguousarray(indices.copy())
        indices.setflags(write=False)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "nearest_prototype_indices", indices)


def query_novelty(
    query_vectors: Any,
    bank: NegativePrototypeBank,
    *,
    chunk_size: int = 512,
) -> NoveltyScores:
    """Return exact nearest-prototype Euclidean distances using bounded chunks."""
    queries = _float32_matrix(query_vectors, "query_vectors")
    if not isinstance(bank, NegativePrototypeBank):
        raise TypeError("bank must be a NegativePrototypeBank.")
    if queries.shape[1] != bank.prototypes.shape[1]:
        raise ValueError("Query and reference banks have different feature dimensions.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1 or chunk_size > 512:
        raise ValueError("chunk_size must be between 1 and 512 to preserve the bounded distance buffer.")
    bank_matrix = bank.prototypes
    bank_squared = np.einsum("ij,ij->i", bank_matrix, bank_matrix)[None, :]
    distances = np.empty(queries.shape[0], dtype=np.float32)
    nearest = np.empty(queries.shape[0], dtype=np.int64)
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        chunk = queries[start:stop]
        squared = (
            np.einsum("ij,ij->i", chunk, chunk)[:, None]
            + bank_squared
            - 2.0 * chunk @ bank_matrix.T
        )
        np.maximum(squared, 0.0, out=squared)
        local_nearest = np.argmin(squared, axis=1)
        nearest[start:stop] = local_nearest
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local_nearest])
    if not np.isfinite(distances).all():
        raise FloatingPointError("Novelty query produced non-finite distances.")
    return NoveltyScores(distances, nearest)


@dataclass(frozen=True)
class PhotoBalancedDescriptors:
    """Four source-photo-balanced summaries of one case's novelty distances."""

    median: float
    q90: float
    q95: float
    tail5_mean: float

    def as_array(self) -> Float32Vector:
        values = np.asarray([self.median, self.q90, self.q95, self.tail5_mean], dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Novelty descriptors must be finite.")
        return values


def photo_balanced_descriptors(scores: Any, photo_ids: Sequence[str]) -> PhotoBalancedDescriptors:
    """Compute per-photo summaries then average photos with equal weight.

    Patch counts therefore cannot make one source photograph dominate the
    case descriptor. The tail mean is the largest ceil(5% of patches) within
    each photograph; no maximum score is used.
    """
    values = _float32_vector(scores, "scores")
    if isinstance(photo_ids, (str, bytes)) or len(photo_ids) != values.size:
        raise ValueError("photo_ids must contain one source-photo ID per score.")
    normalized_ids = [_identifier(value, "photo_id") for value in photo_ids]
    unique = sorted(set(normalized_ids))
    rows: list[tuple[float, float, float, float]] = []
    for photo_id in unique:
        photo_values = values[np.asarray([value == photo_id for value in normalized_ids], dtype=bool)]
        tail_count = max(1, int(math.ceil(photo_values.size * 0.05)))
        tail = np.partition(photo_values, photo_values.size - tail_count)[-tail_count:]
        rows.append((
            float(np.quantile(photo_values, 0.5)),
            float(np.quantile(photo_values, 0.9)),
            float(np.quantile(photo_values, 0.95)),
            float(np.mean(tail, dtype=np.float64)),
        ))
    result = np.asarray(rows, dtype=np.float64).mean(axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Photo-balanced descriptor aggregation is non-finite.")
    return PhotoBalancedDescriptors(*(float(value) for value in result))


def descriptor_vector_by_lens(
    descriptors: Mapping[str, PhotoBalancedDescriptors],
    *,
    lens_order: Sequence[str] = ("4", "10", "40"),
) -> Float32Vector:
    """Flatten lens descriptors in a stable, explicit lens-major order."""
    if not isinstance(descriptors, Mapping):
        raise TypeError("descriptors must be a lens-keyed mapping.")
    order = tuple(_identifier(value, "lens") for value in lens_order)
    if len(set(order)) != len(order) or set(descriptors) != set(order):
        raise ValueError("Descriptor lenses must exactly match the unique requested lens_order.")
    flattened: list[float] = []
    for lens in order:
        value = descriptors[lens]
        if not isinstance(value, PhotoBalancedDescriptors):
            raise TypeError(f"descriptors[{lens!r}] must be PhotoBalancedDescriptors.")
        flattened.extend((value.median, value.q90, value.q95, value.tail5_mean))
    result = np.asarray(flattened, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Flattened descriptor vector is non-finite.")
    return result


@dataclass(frozen=True)
class RobustNoveltyScale:
    """Train-only balanced median/IQR scale with a finite zero-IQR policy."""

    median: float
    iqr: float
    epsilon: float = 1e-6
    degenerate: bool = False

    def __post_init__(self) -> None:
        values = (self.median, self.iqr, self.epsilon)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
               for value in values):
            raise ValueError("Novelty scale values must be finite numbers.")
        if self.iqr < 0 or self.epsilon <= 0:
            raise ValueError("Novelty IQR must be nonnegative and epsilon must be positive.")
        if type(self.degenerate) is not bool:
            raise TypeError("degenerate must be boolean.")
        if self.degenerate != (self.iqr <= self.epsilon):
            raise ValueError("degenerate must match the fixed epsilon zero-IQR policy.")

    @property
    def denominator(self) -> float:
        return 1.0 if self.degenerate else self.iqr


def _weighted_quantile(values: Float32Vector, weights: NDArray[np.float64], quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered[min(index, ordered.size - 1)])


def fit_robust_novelty_scale(
    values: Any,
    photo_ids: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    *,
    anchor_case_id: str,
    allowed_train_case_ids: Sequence[str],
    epsilon: float = 1e-6,
) -> RobustNoveltyScale:
    """Fit equal-case, equal-photo, equal-row median/IQR from T-minus-anchor rows.

    Each metadata record must identify ``case_id`` and ``partition``. Case IDs
    must also belong to the caller's explicit allowed training membership; the
    metadata label alone is not trusted to distinguish training from held-out
    rows. Weights allocate equal total mass to each allowed case represented,
    then split that mass equally across its source photos and rows.
    """
    scores = _float32_vector(values, "values")
    anchor = _identifier(anchor_case_id, "anchor_case_id")
    if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite.")
    if isinstance(photo_ids, (str, bytes)) or len(photo_ids) != scores.size:
        raise ValueError("photo_ids must contain one source-photo ID per novelty score.")
    photos = tuple(_identifier(value, "photo_id") for value in photo_ids)
    rows = _metadata_rows(metadata, scores.size)
    if isinstance(allowed_train_case_ids, (str, bytes)):
        raise TypeError("allowed_train_case_ids must be a sequence of case IDs, not a string.")
    allowed_cases = tuple(_identifier(value, "allowed_train_case_id") for value in allowed_train_case_ids)
    if not allowed_cases or len(set(allowed_cases)) != len(allowed_cases):
        raise ValueError("allowed_train_case_ids must be a nonempty sequence of unique case IDs.")
    if anchor in allowed_cases:
        raise ValueError("allowed_train_case_ids must exclude the negative anchor.")
    allowed_set = set(allowed_cases)
    case_ids: list[str] = []
    for row in rows:
        case_id = _identifier(row.get("case_id"), "metadata.case_id")
        if case_id == anchor:
            raise ValueError("Anchor rows are forbidden when fitting the T-minus-anchor novelty scale.")
        if case_id not in allowed_set:
            raise ValueError("Novelty scale row is outside the explicit T-minus-anchor training membership.")
        if row.get("partition") != "train":
            raise ValueError("Novelty scale may be fit from training rows only.")
        case_ids.append(case_id)

    case_photo_counts = Counter(zip(case_ids, photos, strict=True))
    photo_counts_per_case = Counter(case_id for case_id, _photo_id in case_photo_counts)
    n_cases = len(photo_counts_per_case)
    weights = np.asarray([
        1.0 / n_cases / photo_counts_per_case[case_id] / case_photo_counts[(case_id, photo)]
        for case_id, photo in zip(case_ids, photos, strict=True)
    ], dtype=np.float64)
    median = _weighted_quantile(scores, weights, 0.5)
    q25 = _weighted_quantile(scores, weights, 0.25)
    q75 = _weighted_quantile(scores, weights, 0.75)
    iqr = max(0.0, q75 - q25)
    return RobustNoveltyScale(median, iqr, float(epsilon), iqr <= epsilon)


def standardize_novelty(values: Any, scale: RobustNoveltyScale, *, clip: float = 3.0) -> Float32Vector:
    """Apply a frozen robust scale; degenerate train distributions map to zero."""
    scores = _float32_vector(values, "values")
    if not isinstance(scale, RobustNoveltyScale):
        raise TypeError("scale must be a RobustNoveltyScale.")
    if not isinstance(clip, (int, float)) or isinstance(clip, bool) or not math.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite.")
    if scale.degenerate:
        return np.zeros(scores.shape, dtype=np.float32)
    standardized = np.clip((scores.astype(np.float64) - scale.median) / scale.iqr, -clip, clip)
    result = np.asarray(standardized, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Standardized novelty contains non-finite values.")
    return result


@dataclass(frozen=True)
class LinearNoveltyHead:
    """Train-only standardized logistic head bound to one member's bank."""

    estimator: Pipeline
    anchor_case_id: str
    bank_id: str
    case_ids: tuple[str, ...]
    head_id: str
    c: float


def validate_anchor_head_membership(
    case_ids: Sequence[str],
    targets: Sequence[int],
    *,
    anchor_case_id: str,
    production: bool = True,
) -> tuple[str, ...]:
    """Guard member exclusion and the production two-per-class eligibility gate."""
    anchor = _identifier(anchor_case_id, "anchor_case_id")
    if len(case_ids) != len(targets) or len(case_ids) == 0:
        raise ValueError("Head case IDs and targets must have equal, nonzero length.")
    normalized_ids = tuple(_identifier(value, "case_id") for value in case_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Head case IDs must be unique.")
    if anchor in normalized_ids:
        raise ValueError("Negative anchor must be excluded from supervised head training.")
    labels = tuple(targets)
    if any(type(value) is not int or value not in (0, 1) for value in labels):
        raise ValueError("Head targets must be reviewed binary integers.")
    counts = Counter(labels)
    minimum = 2 if production else 1
    if counts[0] < minimum or counts[1] < minimum:
        raise ValueError(f"Head requires at least {minimum} cases from each class.")
    return normalized_ids


def fit_linear_novelty_head(
    features: Any,
    targets: Sequence[int],
    case_ids: Sequence[str],
    *,
    bank: NegativePrototypeBank,
    anchor_case_id: str,
    c: float = 1.0,
    random_state: int = 42,
    production: bool = True,
    model_identity: Mapping[str, Any] | None = None,
) -> LinearNoveltyHead:
    """Fit StandardScaler+logistic regression on case descriptors only."""
    matrix = _float32_matrix(features, "features")
    if not isinstance(bank, NegativePrototypeBank):
        raise TypeError("bank must be a NegativePrototypeBank.")
    anchor = _identifier(anchor_case_id, "anchor_case_id")
    if bank.anchor_case_id != anchor:
        raise ValueError("Member head anchor does not match its negative bank anchor.")
    normalized_ids = validate_anchor_head_membership(
        case_ids, targets, anchor_case_id=anchor, production=production,
    )
    if matrix.shape[0] != len(normalized_ids) or len(targets) != len(normalized_ids):
        raise ValueError("One descriptor row and target are required for every unique training case.")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not math.isfinite(c) or c <= 0:
        raise ValueError("Logistic C must be positive and finite.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer.")
    identity = dict(model_identity or {})
    estimator = Pipeline([
        ("standardizer", StandardScaler()),
        ("logistic", LogisticRegression(C=float(c), solver="liblinear", max_iter=1000,
                                        random_state=random_state)),
    ])
    estimator.fit(matrix, np.asarray(targets, dtype=np.int64))
    head_id = _hash_payload(
        {"version": 1, "bank_id": bank.bank_id, "anchor_case_id": anchor,
         "case_ids": normalized_ids, "targets": tuple(targets), "c": float(c),
         "random_state": random_state, "model_identity": identity,
         "feature_dim": int(matrix.shape[1])},
        (matrix,),
    )
    return LinearNoveltyHead(estimator, anchor, bank.bank_id, normalized_ids, head_id, float(c))


def predict_linear_novelty_head(
    head: LinearNoveltyHead,
    features: Any,
    case_ids: Sequence[str],
    *,
    bank: NegativePrototypeBank,
) -> Float32Vector:
    """Predict one probability per query case after member-bank ownership checks."""
    if not isinstance(head, LinearNoveltyHead):
        raise TypeError("head must be a LinearNoveltyHead.")
    if not isinstance(bank, NegativePrototypeBank):
        raise TypeError("bank must be a NegativePrototypeBank.")
    if bank.bank_id != head.bank_id or bank.anchor_case_id != head.anchor_case_id:
        raise ValueError("Query bank does not belong to this trained head member.")
    query_ids = tuple(_identifier(value, "query_case_id") for value in case_ids)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Query case IDs must be unique.")
    if head.anchor_case_id in query_ids:
        raise ValueError("The negative anchor may not be self-scored by its supervised member.")
    matrix = _float32_matrix(features, "features")
    if matrix.shape[0] != len(query_ids) or matrix.shape[1] != head.estimator.n_features_in_:
        raise ValueError("Query case descriptors do not match the head's rows or feature dimension.")
    probabilities = np.asarray(head.estimator.predict_proba(matrix)[:, 1], dtype=np.float32)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise FloatingPointError("Logistic head produced invalid probabilities.")
    return probabilities


def equal_weight_member_ensemble(
    probabilities: Sequence[Any],
    *,
    member_query_case_ids: Sequence[Sequence[str]],
) -> Float32Vector:
    """Average member probabilities after verifying identical query case order."""
    if isinstance(probabilities, (str, bytes)) or len(probabilities) == 0:
        raise ValueError("At least one member probability vector is required.")
    if isinstance(member_query_case_ids, (str, bytes)) or len(member_query_case_ids) != len(probabilities):
        raise ValueError("One explicit query case-ID sequence is required for every member.")
    arrays = [_float32_vector(value, f"probabilities[{index}]") for index, value in enumerate(probabilities)]
    id_sequences = []
    for index, case_ids in enumerate(member_query_case_ids):
        if isinstance(case_ids, (str, bytes)):
            raise TypeError(f"member_query_case_ids[{index}] must be a sequence of case IDs, not a string.")
        id_sequences.append(tuple(
            _identifier(case_id, f"member_query_case_ids[{index}]") for case_id in case_ids
        ))
    if any(len(set(case_ids)) != len(case_ids) for case_ids in id_sequences):
        raise ValueError("Each member query case-ID sequence must be unique.")
    if any(case_ids != id_sequences[0] for case_ids in id_sequences[1:]):
        raise ValueError("All members must predict the same ordered query case IDs.")
    if any(array.shape != (len(id_sequences[0]),) for array in arrays):
        raise ValueError("Member probabilities must match their explicit query case IDs.")
    if any(np.any((array < 0.0) | (array > 1.0)) for array in arrays):
        raise ValueError("Member values must be probabilities in [0, 1].")
    result = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Ensemble probabilities are non-finite.")
    return result


__all__ = [
    "LinearNoveltyHead",
    "NegativePrototypeBank",
    "NoveltyScores",
    "PhotoBalancedDescriptors",
    "RobustNoveltyScale",
    "descriptor_vector_by_lens",
    "equal_weight_member_ensemble",
    "fit_linear_novelty_head",
    "fit_negative_prototype_bank",
    "fit_robust_novelty_scale",
    "photo_balanced_descriptors",
    "predict_linear_novelty_head",
    "query_novelty",
    "standardize_novelty",
    "validate_anchor_head_membership",
]
