"""Scope-bound transforms, source-photo lineage and immutable dual-track artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .features import verify_feature_part
from .io import atomic_json, file_hash, fingerprint
from .mil_checkpoint import FitLock

LENSES = ("4", "10", "40")


def check_deadline(deadline: float) -> None:
    """Cooperatively stop without promoting partially written work."""
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("Dual-track work budget exhausted.")


def _relative(name: str) -> str:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or "\\" in name or value.as_posix() != name:
        raise ValueError("Artifact path must be canonical and relative.")
    return name


class ArtifactStore:
    """Publish checksum-bound immutable jobs; only committed payloads may resume."""

    def __init__(
        self,
        root: Path,
        identity: dict[str, Any],
        *,
        recover_stale: bool = False,
        recovery_evidence: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ValueError("Artifact root must not be a symlink.")
        self.root.mkdir(parents=True, exist_ok=True)
        if not isinstance(identity, dict) or not identity:
            raise ValueError("Artifact identity must be a nonempty object.")
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.identity_id = fingerprint(self.identity)
        self.recover_stale = recover_stale
        self.recovery_evidence = dict(recovery_evidence or {})
        path = self.root / "identity.json"
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Artifact identity must be a regular file.")
        if path.exists():
            if json.loads(path.read_text()) != self.identity:
                raise ValueError("Artifact root belongs to another protocol/source identity.")
        else:
            with self.lease(self.root):
                if path.exists():
                    if path.is_symlink() or json.loads(path.read_text()) != self.identity:
                        raise ValueError("Artifact identity changed before initialization lease.")
                else:
                    atomic_json(path, self.identity)

    def lease(self, root: Path) -> FitLock:
        """Opt-in dead-owner recovery; never reclaim a live/foreign unproven lease."""
        relative = Path(root).absolute().relative_to(self.root).as_posix()
        evidence = self.recovery_evidence.get(relative)
        return FitLock(root, recover_stale=self.recover_stale, recovery_evidence=evidence)

    def _path(self, name: str) -> Path:
        identity_path = self.root / "identity.json"
        if (
            identity_path.is_symlink()
            or not identity_path.is_file()
            or json.loads(identity_path.read_text()) != self.identity
        ):
            raise ValueError("Artifact identity changed after initialization.")
        path = self.root / _relative(name)
        cursor = path.parent
        while cursor != self.root:
            if cursor.is_symlink():
                raise ValueError("Artifact directory is symlinked.")
            cursor = cursor.parent
        if path.is_symlink():
            raise ValueError("Artifact payload must not be a symlink.")
        return path

    def read(self, name: str) -> bytes | None:
        """Verify identity/receipt/hash before returning a committed payload."""
        path = self._path(name)
        receipt_path = self._path(name + ".commit.json")
        if not receipt_path.exists():
            return None
        receipt = json.loads(receipt_path.read_text())
        body = {key: value for key, value in receipt.items() if key != "commit_id"}
        if (
            receipt.get("commit_id") != fingerprint(body)
            or receipt.get("identity_id") != self.identity_id
            or receipt.get("name") != name
            or not path.is_file()
        ):
            raise ValueError("Invalid artifact receipt/identity.")
        if path.stat().st_size != receipt.get("bytes") or file_hash(path) != receipt.get("sha256"):
            raise ValueError("Committed artifact checksum mismatch.")
        return path.read_bytes()

    def publish(self, name: str, payload: bytes) -> dict[str, Any]:
        """Commit a new artifact under a per-job writer lock; preserve last-good."""
        if not isinstance(payload, bytes):
            raise TypeError("Artifact payload must be bytes.")
        path = self._path(name)
        lock_root = self.root / "locks" / fingerprint(name)
        lock_root.mkdir(parents=True, exist_ok=True)
        with self.lease(lock_root):
            existing = self.read(name)
            if existing is not None:
                if existing != payload:
                    raise ValueError("Refusing to overwrite a committed artifact.")
                return json.loads(self._path(name + ".commit.json").read_text())
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            digest = hashlib.sha256(payload).hexdigest()
            try:
                with temp.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                if file_hash(temp) != digest:
                    raise ValueError("Temporary artifact checksum failed.")
                os.replace(temp, path)
                if file_hash(path) != digest:
                    raise ValueError("Published artifact read-back failed.")
                body = {"name": name, "identity_id": self.identity_id, "sha256": digest, "bytes": len(payload)}
                receipt = {**body, "commit_id": fingerprint(body)}
                atomic_json(self._path(name + ".commit.json"), receipt)
                self.read(name)
                return receipt
            finally:
                temp.unlink(missing_ok=True)

    def object(self, name: str, producer: Any) -> Any:
        """Load verified trusted internal models, or compute/commit one job.

        Pickle is restricted to this code-owned checksum/identity-verified store;
        it is never used to deserialize uploaded source data or unknown models.
        """
        content = self.read(name)
        if content is not None:
            return pickle.loads(content)
        compute_root = self.root / "computations" / fingerprint(name)
        compute_root.mkdir(parents=True, exist_ok=True)
        with self.lease(compute_root):
            content = self.read(name)
            if content is not None:
                return pickle.loads(content)
            value = producer()
            self.publish(name, pickle.dumps(value, protocol=5))
            return value

    def json(self, name: str, value: Any) -> dict[str, Any]:
        """Publish a finite, canonical JSON artifact."""
        return self.publish(name, json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize finite float32 features; reject zero-norm rows explicitly."""
    x = np.asarray(values)
    if x.dtype != np.float32 or x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty float32 matrix.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Invalid/zero feature norm; no silent normalization replacement.")
    return (x / norms).astype(np.float32)


def photo_sample_indices(photo_ids: list[str], cap: int, seed: int) -> np.ndarray:
    """Sample without replacement, distributing quota across source photos."""
    if not photo_ids or cap < 1 or any(not isinstance(x, str) or not x for x in photo_ids):
        raise ValueError("Source photos and sample cap must be valid.")
    rng = np.random.default_rng(seed)
    photo_order = rng.permutation(sorted(set(photo_ids))).tolist()
    groups = {photo: rng.permutation(np.flatnonzero(np.array(photo_ids) == photo)).tolist() for photo in photo_order}
    selected = []
    while len(selected) < min(cap, len(photo_ids)):
        progressed = False
        for photo in groups:
            if groups[photo] and len(selected) < min(cap, len(photo_ids)):
                selected.append(groups[photo].pop())
                progressed = True
        if not progressed:
            break
    return np.array(sorted(selected), dtype=np.int64)


def _build_source_photo_map(
    dataset: Any, bundle: Path, feature_root: Path, store: ArtifactStore, deadline: float
) -> dict[str, Any]:
    """Verify part indexes and join only authorized governed refs to source photos."""
    existing = store.read("source_photo_map.pkl")
    if existing is not None:
        return pickle.loads(existing)
    feature_id = dataset.identity["feature_id"]
    authorized: dict[tuple[str, int], dict[str, Any]] = {}
    order_by_case: dict[str, dict[str, list[str]]] = {case: {lens: [] for lens in LENSES} for case in dataset.case_ids}
    with (Path(bundle) / "instance_refs.jsonl").open() as stream:
        for line in stream:
            check_deadline(deadline)
            ref = json.loads(line)
            case = ref["case_id"]
            if case not in order_by_case:
                continue
            lens = str(ref["objective_lens"])
            key = (ref["part_id"], ref["row_index"])
            if (
                lens not in LENSES
                or ref["feature_id"] != feature_id
                or key in authorized
                or any(k in ref for k in ("patch_target", "label", "target"))
            ):
                raise ValueError("Governed refs have invalid duplicate/scope/lens/label metadata.")
            authorized[key] = ref
            order_by_case[case][lens].append(ref["tile_id"])
    mapping = {}
    version_root = Path(feature_root) / feature_id / "parts"
    for part_id in sorted({key[0] for key in authorized}):
        if Path(part_id).name != part_id or part_id in {".", ".."}:
            raise ValueError("Unsafe feature part id.")
        part = version_root / part_id
        if part.is_symlink():
            raise ValueError("Feature part path must not be symlinked.")
        verify_feature_part(part, feature_id=feature_id, deadline=deadline)
        with (part / "tile_index.jsonl").open() as stream:
            for row_index, line in enumerate(stream):
                check_deadline(deadline)
                key = (part_id, row_index)
                if key not in authorized:
                    continue
                ref, row = authorized[key], json.loads(line)
                lens = str(ref["objective_lens"])
                if (
                    row["tile_id"] != ref["tile_id"]
                    or row["row_index"] != row_index
                    or str(row.get("objective_lens")) != lens
                    or row.get("candidate_case_id") != ref["case_id"]
                    or not isinstance(row.get("image_id"), str)
                    or not row["image_id"]
                ):
                    raise ValueError("Tile/source-photo lineage differs from governed case/lens/order.")
                tile = ref["tile_id"]
                if tile in mapping:
                    raise ValueError("Source-photo map contains duplicate governed tile id.")
                mapping[tile] = {
                    "case_id": ref["case_id"],
                    "lens": lens,
                    "part_id": part_id,
                    "row_index": row_index,
                    "photo_id": fingerprint({"case": ref["case_id"], "lens": lens, "image": row["image_id"]}),
                    "tile_id": tile,
                    "source_png_sha256": row.get("source_png_sha256"),
                    "source_member": row.get("source_member"),
                    "source_name": row.get("source_name"),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "original_crop_geometry": "unverified",
                }
    if len(mapping) != len(authorized):
        raise ValueError("One or more governed tile refs have no verified source photo.")
    for case in dataset.case_ids:
        check_deadline(deadline)
        data = dataset.load_case(case)
        actual = dataset.loaded_tile_order(case)
        if actual != order_by_case[case] or any(len(actual[lens]) != len(data.features[lens]) for lens in LENSES):
            raise ValueError("Source photo rows and public cached feature order differ.")
    result = {"source_identity": dataset.identity, "rows": mapping, "case_tile_order": order_by_case}
    store.publish("source_photo_map.pkl", pickle.dumps(result, protocol=5))
    store.json(
        "source_photo_map_summary.json",
        {
            "tiles": len(mapping),
            "cases": len(dataset.case_ids),
            "photos": len({r["photo_id"] for r in mapping.values()}),
            "labels_are_patch_targets": False,
        },
    )
    return result


def build_source_photo_map(
    dataset: Any, bundle: Path, feature_root: Path, store: ArtifactStore, deadline: float
) -> dict[str, Any]:
    """Deduplicate the complete verified lineage job before reading full indexes."""
    return store.object(
        "source_photo_map.pkl",
        lambda: _build_source_photo_map(
            dataset,
            bundle,
            feature_root,
            store,
            deadline,
        ),
    )


def photos_for_case(source_map: dict[str, Any], case_id: str, lens: str) -> list[str]:
    """Resolve source photos in the exact public-loader tile order."""
    return [source_map["rows"][tile]["photo_id"] for tile in source_map["case_tile_order"][case_id][lens]]


@dataclass(frozen=True)
class WeightedPCA:
    """Frozen weighted PCA with an ordinary PCA-compatible transform interface."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray
    n_components_: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values)
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError("Weighted PCA query shape/values mismatch.")
        return ((x - self.mean_) @ self.components_.T).astype(np.float32)


def fit_weighted_pca(values: np.ndarray, weights: np.ndarray, dimensions: int, seed: int) -> WeightedPCA:
    """Fit weighted covariance, so cases/photos with more rows cannot dominate."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 2
        or w.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or np.any(w <= 0)
        or dimensions > min(x.shape)
        or dimensions < 1
    ):
        raise ValueError("Weighted PCA inputs/rank/weights are invalid.")
    w = w / w.sum()
    mean = np.sum(x * w[:, None], axis=0)
    centered = (x - mean) * np.sqrt(w[:, None])
    total_variance = float(np.square(centered).sum())
    if total_variance <= np.finfo(np.float64).tiny:
        raise ValueError("Training reservoir has no feature variance.")
    _, singular, components = randomized_svd(centered, n_components=dimensions, random_state=seed)
    return WeightedPCA(
        mean.astype(np.float32),
        components.astype(np.float32),
        (singular**2 / total_variance).astype(np.float32),
        dimensions,
    )


def fit_scope_transforms(
    dataset: Any,
    source_map: dict[str, Any],
    train_ids: list[str],
    *,
    authorized_train_ids: list[str],
    cap: int = 256,
    dimensions: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit independent per-lens PCA on authorized patient/photo-balanced training refs."""
    if (
        set(train_ids) != set(authorized_train_ids)
        or len(authorized_train_ids) != len(set(authorized_train_ids))
        or set(train_ids) != set(dataset.case_ids)
        or len(train_ids) != len(set(train_ids))
    ):
        raise ValueError("Transform dataset must contain exactly the authorized train cases.")
    source_identity = source_map.get("source_identity")
    if not isinstance(source_identity, dict) or not isinstance(dataset.identity, dict):
        raise ValueError("Source-photo map identity is missing.")
    ignored = {"case_ids", "fingerprint", "case_records_fingerprint"}
    source_domain = {k: v for k, v in source_identity.items() if k not in ignored}
    dataset_domain = {k: v for k, v in dataset.identity.items() if k not in ignored}
    if source_domain != dataset_domain:
        raise ValueError("Source-photo map belongs to another dataset identity.")
    members = {case: dataset.records[case] for case in train_ids}
    if len({r["patient_id"] for r in members.values()}) != len(members):
        raise ValueError("Training scope must contain distinct patients.")
    samples: dict[str, list[np.ndarray]] = {lens: [] for lens in LENSES}
    sample_refs: dict[str, list[dict[str, Any]]] = {lens: [] for lens in LENSES}
    weights: dict[str, list[float]] = {lens: [] for lens in LENSES}
    for case in train_ids:
        data = dataset.load_case(case)
        for lens in LENSES:
            ids = photos_for_case(source_map, case, lens)
            tiles = source_map["case_tile_order"][case][lens]
            if (
                len(tiles) != len(data.features[lens])
                or len(tiles) != len(set(tiles))
                or dataset.loaded_tile_order(case)[lens] != tiles
            ):
                raise ValueError("Source-photo map does not match public-loader row order.")
            if not len(data.features[lens]):
                if ids:
                    raise ValueError("Missing lens has nonempty source-photo refs.")
                continue
            chosen = photo_sample_indices(ids, cap, seed)
            samples[lens].append(normalize_rows(data.features[lens][chosen]))
            chosen_photos = [ids[int(i)] for i in chosen]
            unique = set(chosen_photos)
            for index, photo in zip(chosen, chosen_photos):
                weights[lens].append(1 / len(train_ids) / len(unique) / chosen_photos.count(photo))
                tile = source_map["case_tile_order"][case][lens][int(index)]
                sample_refs[lens].append(
                    {**source_map["rows"][tile], "patient_id": data.patient_id, "source_case_label": data.target}
                )
    transforms = {}
    for lens in LENSES:
        if not samples[lens]:
            raise ValueError("Training scope has no observed data for a required lens.")
        x = np.concatenate(samples[lens])
        if dimensions > min(x.shape):
            raise ValueError("PCA dimension exceeds train-only sample rank bound.")
        pca = fit_weighted_pca(x, np.array(weights[lens]), dimensions, seed)
        transforms[lens] = {
            "pca": pca,
            "sample_projected": pca.transform(x).astype(np.float32),
            "sample_refs": sample_refs[lens],
            "sample_weights": np.array(weights[lens]),
            "normalized_sample": x,
        }
    return {"members": members, "seed": seed, "dimensions": dimensions, "cap": cap, "lenses": transforms}


def project_case(dataset: Any, case_id: str, transforms: dict[str, Any], *, chunk: int = 512) -> dict[str, np.ndarray]:
    """Apply a locked transform to all vectors, never fit using query cases."""
    if chunk < 1:
        raise ValueError("Chunk size must be positive.")
    data = dataset.load_case(case_id)
    result = {}
    for lens in LENSES:
        array = data.features[lens]
        pca = transforms["lenses"][lens]["pca"]
        blocks = [
            pca.transform(normalize_rows(array[start : start + chunk])).astype(np.float32)
            for start in range(0, len(array), chunk)
        ]
        result[lens] = np.concatenate(blocks) if blocks else np.empty((0, pca.n_components_), dtype=np.float32)
    return result


def gzip_csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    """Produce deterministic compressed rows for small summaries/ref sets."""
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(text.getvalue().encode(), mtime=0)
