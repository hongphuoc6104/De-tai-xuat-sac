"""Bounded, provenance-aware datasets for multiscale MIL experiments.

The synthetic and exported fixtures in this module are smoke-test inputs only.
They carry an explicit marker that prevents them from being mistaken for the
full governed feature release used by production training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .bags import BUNDLE_CASE_FIELDS, LENSES, load_case_features
from .feature_encoder import FEATURE_DIM
from .io import file_hash, fingerprint
from .readiness import verify_pretrain_bundle
from .splits import validate_split_structure

FIXTURE_SCHEMA_VERSION = 1
FIXTURE_MODE = "smoke"
MAX_TOY_CASES = 16
MAX_TOY_ROWS_PER_LENS = 64
_LENS_KEYS = tuple(str(lens) for lens in LENSES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty, trimmed string of at most 256 characters.")
    if value in {".", ".."} or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} contains an unsafe path/control character.")
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _target(value: Any, name: str = "target") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in {0, 1}:
        raise ValueError(f"{name} must be a reviewed integer 0 or 1.")
    return int(value)


def _readonly_feature(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise ValueError(f"{name} must be a float32 N×{FEATURE_DIM} array.")
    for start in range(0, array.shape[0], 256):
        if not np.isfinite(array[start:start + 256]).all():
            raise ValueError(f"{name} contains non-finite feature values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CaseData:
    """One reviewed case with case-level target and three lens feature bags."""

    case_id: str
    patient_id: str
    target: int
    features: dict[str, np.ndarray]
    presence_mask: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.patient_id, "patient_id")
        object.__setattr__(self, "target", _target(self.target))
        if not isinstance(self.features, dict) or set(self.features) != set(_LENS_KEYS):
            raise ValueError(f"features must contain exactly the lens keys {_LENS_KEYS}.")
        readonly = {lens: _readonly_feature(self.features[lens], f"features[{lens!r}]")
                    for lens in _LENS_KEYS}
        if (not isinstance(self.presence_mask, tuple) or len(self.presence_mask) != len(_LENS_KEYS)
                or any(type(value) is not bool for value in self.presence_mask)):
            raise ValueError("presence_mask must be a three-element tuple of booleans in 4x, 10x, 40x order.")
        expected = tuple(readonly[lens].shape[0] > 0 for lens in _LENS_KEYS)
        if self.presence_mask != expected:
            raise ValueError("presence_mask does not agree with the feature bag row counts.")
        object.__setattr__(self, "features", readonly)


def _canonical_array_path(case_id: str, lens: str) -> str:
    case_key = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return f"arrays/{case_key}/lens_{lens}.npy"


def _write_npy(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return file_hash(path)


def _publish_fixture(root: Path, records: list[dict[str, Any]], provenance: dict[str, Any]) -> Path:
    root = Path(root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to replace existing fixture path: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        case_entries = []
        seen_cases: set[str] = set()
        seen_patients: set[str] = set()
        for record in records:
            case_id = _identifier(record["case_id"], "case_id")
            patient_id = _identifier(record["patient_id"], "patient_id")
            target = _target(record["target"])
            if case_id in seen_cases:
                raise ValueError(f"Duplicate fixture case_id: {case_id}")
            if patient_id in seen_patients:
                raise ValueError(f"Duplicate fixture patient_id: {patient_id}")
            seen_cases.add(case_id)
            seen_patients.add(patient_id)
            arrays = record.get("features")
            if not isinstance(arrays, dict) or set(arrays) != set(_LENS_KEYS):
                raise ValueError(f"Fixture case {case_id} must have all three lens arrays.")
            array_entries: dict[str, Any] = {}
            for lens in _LENS_KEYS:
                array = _readonly_feature(arrays[lens], f"{case_id}/{lens}")
                if array.shape[0] < 1:
                    raise ValueError(f"Fixture case {case_id}/{lens} must contain at least one feature row.")
                relative = _canonical_array_path(case_id, lens)
                digest = _write_npy(staging / relative, array)
                array_entries[lens] = {
                    "path": relative,
                    "sha256": digest,
                    "shape": [int(array.shape[0]), FEATURE_DIM],
                    "dtype": "float32",
                }
            case_entries.append({
                "case_id": case_id,
                "patient_id": patient_id,
                "target": target,
                "arrays": array_entries,
            })
        if not case_entries:
            raise ValueError("A fixture must contain at least one case.")
        if len({entry["target"] for entry in case_entries}) != 2:
            raise ValueError("Smoke fixtures must contain both case-level classes.")
        body = {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "mode": FIXTURE_MODE,
            "training_ready": False,
            "production_eligible": False,
            "provenance": provenance,
            "case_count": len(case_entries),
            "case_ids": [entry["case_id"] for entry in case_entries],
            "cases": case_entries,
        }
        manifest = {**body, "fixture_id": fingerprint(body)}
        manifest_path = staging / "fixture.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def create_toy_fixture(
    root: Path,
    seed: int = 42,
    case_count: int = 4,
    rows_per_lens: int = 16,
) -> Path:
    """Create an isolated, synthetic smoke fixture with a real 2048-D signal."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    _positive_int(case_count, "case_count", maximum=MAX_TOY_CASES)
    _positive_int(rows_per_lens, "rows_per_lens", maximum=MAX_TOY_ROWS_PER_LENS)
    if case_count < 2:
        raise ValueError("case_count must be at least two to include both classes.")
    rng = np.random.default_rng(seed)
    records = []
    for index in range(case_count):
        target = index % 2
        features: dict[str, np.ndarray] = {}
        for lens_index, lens in enumerate(_LENS_KEYS):
            # A broad low-amplitude noise floor plus a class-dependent signal in
            # the first 32 embedding dimensions makes the target learnable.
            values = rng.normal(0.0, 0.05, size=(rows_per_lens, FEATURE_DIM)).astype(np.float32)
            values[:, :32] += np.float32(1.5 if target else -1.5)
            values[:, 32 + lens_index] += np.float32((lens_index + 1) * 0.1)
            features[lens] = values
        records.append({
            "case_id": f"toy-case-{index:04d}",
            "patient_id": f"toy-patient-{index:04d}",
            "target": target,
            "features": features,
        })
    return _publish_fixture(Path(root), records, {"kind": "synthetic", "seed": seed})


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError(f"{name} must be an existing regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"Invalid non-finite JSON value: {constant}")


def _fixture_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "smoke_fixture",
        "fixture_id": manifest["fixture_id"],
        "mode": manifest["mode"],
        "case_ids": list(manifest["case_ids"]),
        "provenance": manifest["provenance"],
    }


class FixtureDataset:
    """Load arrays from a checksum-verified synthetic or exported smoke fixture."""

    def __init__(self, root: Path) -> None:
        fixture_root = Path(root).expanduser().absolute()
        if fixture_root.is_symlink():
            raise ValueError("Fixture root must be an existing, non-symlink directory.")
        self.root = fixture_root.resolve()
        if not self.root.is_dir():
            raise ValueError("Fixture root must be an existing, non-symlink directory.")
        manifest = _read_json_object(self.root / "fixture.json", "fixture manifest")
        fixture_id = manifest.get("fixture_id")
        unsigned = {key: value for key, value in manifest.items() if key != "fixture_id"}
        if not isinstance(fixture_id, str) or not _SHA256.fullmatch(fixture_id) or fixture_id != fingerprint(unsigned):
            raise ValueError("Fixture manifest fingerprint mismatch.")
        if set(manifest) != {
            "schema_version", "mode", "training_ready", "production_eligible", "provenance",
            "case_count", "case_ids", "cases", "fixture_id",
        }:
            raise ValueError("Fixture manifest schema mismatch.")
        if (manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION or manifest.get("mode") != FIXTURE_MODE
                or manifest.get("training_ready") is not False or manifest.get("production_eligible") is not False):
            raise ValueError("Fixture marker is invalid; smoke fixtures cannot be production inputs.")
        provenance = manifest.get("provenance")
        cases = manifest.get("cases")
        case_ids = manifest.get("case_ids")
        if not isinstance(provenance, dict) or not isinstance(cases, list) or not cases:
            raise ValueError("Fixture provenance or case rows are invalid.")
        if (not isinstance(case_ids, list) or len(case_ids) != len(cases)
                or manifest.get("case_count") != len(cases)):
            raise ValueError("Fixture case count/list differs from its case rows.")
        records: dict[str, dict[str, Any]] = {}
        seen_patients: set[str] = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"case_id", "patient_id", "target", "arrays"}:
                raise ValueError("Fixture case row schema mismatch.")
            case_id = _identifier(case["case_id"], "case_id")
            patient_id = _identifier(case["patient_id"], "patient_id")
            target = _target(case["target"])
            if case_id in records or patient_id in seen_patients:
                raise ValueError("Fixture case IDs and patient IDs must be unique.")
            seen_patients.add(patient_id)
            arrays = case["arrays"]
            if not isinstance(arrays, dict) or set(arrays) != set(_LENS_KEYS):
                raise ValueError(f"Fixture arrays for {case_id} must list exactly the three lenses.")
            for lens, entry in arrays.items():
                if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "shape", "dtype"}:
                    raise ValueError(f"Fixture array metadata schema mismatch for {case_id}/{lens}.")
                relative = _canonical_array_path(case_id, lens)
                if entry.get("path") != relative or not _SHA256.fullmatch(str(entry.get("sha256", ""))):
                    raise ValueError(f"Unsafe path or invalid checksum for fixture array {case_id}/{lens}.")
                shape = entry.get("shape")
                if (entry.get("dtype") != "float32" or not isinstance(shape, list) or len(shape) != 2
                        or shape[1] != FEATURE_DIM or isinstance(shape[0], bool)
                        or not isinstance(shape[0], int) or shape[0] < 1):
                    raise ValueError(f"Invalid feature schema for fixture array {case_id}/{lens}.")
                array_path = self.root / relative
                if (array_path.is_symlink() or array_path.parent.is_symlink()
                        or array_path.parent.parent.is_symlink() or not array_path.is_file()):
                    raise ValueError(f"Fixture array is missing or is a symlink: {relative}")
                if array_path.resolve() != self.root / relative:
                    raise ValueError(f"Fixture array escapes its declared directory: {relative}")
            records[case_id] = {
                "patient_id": patient_id,
                "target": target,
                "arrays": MappingProxyType({lens: MappingProxyType(dict(arrays[lens])) for lens in _LENS_KEYS}),
            }
        if case_ids != list(records):
            raise ValueError("Fixture case_ids order/content differs from the case manifest.")
        if {record["target"] for record in records.values()} != {0, 1}:
            raise ValueError("Smoke fixture must include both case-level classes.")
        if provenance.get("kind") == "governed_bundle_subset":
            _validate_export_provenance(provenance, records)
        elif (provenance.get("kind") != "synthetic" or set(provenance) != {"kind", "seed"}
              or isinstance(provenance.get("seed"), bool)
              or not isinstance(provenance.get("seed"), int) or provenance["seed"] < 0):
            raise ValueError("Fixture provenance kind must be synthetic or governed_bundle_subset.")
        self._manifest = MappingProxyType(manifest)
        self._records = MappingProxyType(records)
        self._case_ids = tuple(records)
        self._identity_payload = _fixture_identity_payload(manifest)
        self._identity = {**self._identity_payload, "fingerprint": fingerprint(self._identity_payload)}

    @property
    def case_ids(self) -> list[str]:
        return list(self._case_ids)

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return {case_id: {"patient_id": row["patient_id"], "target": row["target"]}
                for case_id, row in self._records.items()}

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._identity, ensure_ascii=False))

    @property
    def identity_hash(self) -> str:
        return str(self._identity["fingerprint"])

    @property
    def mode(self) -> str:
        return FIXTURE_MODE

    def load_case(self, case_id: str) -> CaseData:
        case_id = _identifier(case_id, "case_id")
        if case_id not in self._records:
            raise KeyError(f"Case is not in this fixture: {case_id}")
        row = self._records[case_id]
        arrays: dict[str, np.ndarray] = {}
        presence = []
        for lens in _LENS_KEYS:
            entry = row["arrays"][lens]
            relative = _canonical_array_path(case_id, lens)
            path = self.root / relative
            if path.is_symlink() or not path.is_file() or file_hash(path) != entry["sha256"]:
                raise ValueError(f"Fixture array checksum mismatch: {relative}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != entry["shape"] or array.dtype != np.float32:
                raise ValueError(f"Fixture array schema changed: {relative}")
            arrays[lens] = array
            presence.append(array.shape[0] > 0)
        return CaseData(case_id, row["patient_id"], row["target"], arrays, tuple(presence))


def _parse_bundle_metadata(bundle: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    bundle = Path(bundle).expanduser().absolute()
    if bundle.is_symlink():
        raise ValueError("Bundle path must be an existing, non-symlink directory.")
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ValueError("Bundle path must be an existing, non-symlink directory.")
    manifest = _read_json_object(bundle / "bundle.json", "bundle manifest")
    bundle_id = manifest.get("bundle_id")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if (not isinstance(bundle_id, str) or not _SHA256.fullmatch(bundle_id)
            or bundle_id != fingerprint(unsigned)):
        raise ValueError("Bundle manifest fingerprint mismatch.")
    if manifest.get("schema_version") != 1 or manifest.get("training_ready") is not True:
        raise ValueError("Bundle schema is unsupported or bundle is not training-ready.")
    feature_id = manifest.get("feature_id")
    split_id = manifest.get("split_id")
    if not isinstance(feature_id, str) or not _SHA256.fullmatch(feature_id):
        raise ValueError("Bundle feature_id is invalid.")
    if not isinstance(split_id, str) or not _SHA256.fullmatch(split_id):
        raise ValueError("Bundle split_id is invalid.")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Bundle artifact checksums are missing.")
    for name, digest in artifact_hashes.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ValueError("Bundle artifact checksum table contains an unsafe path or digest.")
    for name in ("cases.csv", "splits.json"):
        digest = artifact_hashes.get(name)
        path = bundle / name
        if (not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                or path.is_symlink() or not path.is_file() or file_hash(path) != digest):
            raise ValueError(f"Bundle metadata checksum mismatch: {name}")
    case_path = bundle / "cases.csv"
    with case_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BUNDLE_CASE_FIELDS:
            raise ValueError("Bundle case metadata schema mismatch.")
        rows = list(reader)
    if not rows or len(rows) != manifest.get("cohort_case_count"):
        raise ValueError("Bundle case metadata count mismatch.")
    records: dict[str, dict[str, Any]] = {}
    split_cases = []
    for row in rows:
        case_id = _identifier(row.get("case_id"), "case_id")
        patient_id = _identifier(row.get("patient_id"), "patient_id")
        try:
            if row.get("case_label") not in {"0", "1"}:
                raise ValueError
            target = int(row["case_label"])
            presence = json.loads(row["lens_presence_json"])
            counts = json.loads(row["instance_counts_by_lens_json"])
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid case metadata for {case_id}.") from exc
        if case_id in records:
            raise ValueError(f"Duplicate case row in bundle: {case_id}")
        if (not isinstance(presence, dict) or set(presence) != set(_LENS_KEYS)
                or any(type(presence[lens]) is not bool for lens in _LENS_KEYS)
                or not isinstance(counts, dict) or set(counts) != set(_LENS_KEYS)):
            raise ValueError(f"Invalid lens metadata for {case_id}.")
        clean_counts = {}
        for lens in _LENS_KEYS:
            count = counts[lens]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"Invalid lens instance count for {case_id}/{lens}.")
            if bool(count) != presence[lens]:
                raise ValueError(f"Lens presence/count mismatch for {case_id}/{lens}.")
            clean_counts[lens] = count
        records[case_id] = {
            "patient_id": patient_id,
            "target": target,
            "presence_mask": tuple(presence[lens] for lens in _LENS_KEYS),
            "instance_counts": MappingProxyType(clean_counts),
            "feature_id": feature_id,
        }
        split_cases.append({"case_id": case_id, "patient_id": patient_id, "case_label": target})
    cohort_ids = manifest.get("cohort_case_ids")
    if not isinstance(cohort_ids, list) or cohort_ids != list(records):
        raise ValueError("Bundle manifest case IDs differ from cases.csv order/content.")
    split_path = bundle / "splits.json"
    splits = _read_json_object(split_path, "nested split manifest")
    split_body = {key: value for key, value in splits.items() if key != "split_id"}
    if splits.get("split_id") != split_id or fingerprint(split_body) != split_id:
        raise ValueError("Nested split fingerprint differs from bundle manifest.")
    validate_split_structure(splits, split_cases)
    assignment = splits.get("assignments", {})
    for case_id, record in records.items():
        if assignment.get(case_id, {}).get("patient_id") != record["patient_id"]:
            raise ValueError(f"Split identity differs from bundle case metadata: {case_id}")
    return manifest, records, splits


class BundleDataset:
    """Lazy one-case adapter over an immutable governed feature bundle.

    Constructor work is limited to case and split metadata. Each ``load_case``
    delegates to the verified public ``load_case_features`` API, which checks
    bundle artifacts, feature part checksums, row references and finite values.
    """

    def __init__(
        self,
        bundle: Path,
        feature_root: Path,
        partition_ids: list[str],
        max_case_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        source_bundle = Path(bundle).expanduser().absolute()
        if source_bundle.is_symlink():
            raise ValueError("Bundle path must be an existing, non-symlink directory.")
        self.bundle = source_bundle.resolve()
        self.feature_root = Path(feature_root).expanduser().resolve()
        self.max_case_bytes = _positive_int(max_case_bytes, "max_case_bytes")
        if not isinstance(partition_ids, list) or not partition_ids:
            raise ValueError("partition_ids must be a nonempty list of case IDs.")
        selected = [_identifier(case_id, "partition case_id") for case_id in partition_ids]
        if len(selected) != len(set(selected)):
            raise ValueError("partition_ids must not contain duplicates.")
        manifest, records, _ = _parse_bundle_metadata(self.bundle)
        absent = set(selected) - set(records)
        if absent:
            raise ValueError(f"Partition includes cases outside the governed bundle: {sorted(absent)}")
        self._manifest_id = str(manifest["bundle_id"])
        self._feature_id = str(manifest["feature_id"])
        self._split_id = str(manifest["split_id"])
        self._case_ids = tuple(selected)
        self._records = MappingProxyType({case_id: records[case_id] for case_id in selected})
        self._loaded_tile_order: dict[str, dict[str, tuple[str, ...]]] = {}
        identity_payload = {
            "kind": "governed_bundle_partition",
            "bundle_id": self._manifest_id,
            "feature_id": self._feature_id,
            "split_id": self._split_id,
            "case_ids": sorted(selected),
            "case_records_fingerprint": fingerprint({
                case_id: {"patient_id": records[case_id]["patient_id"], "target": records[case_id]["target"]}
                for case_id in sorted(selected)
            }),
        }
        self._identity = {**identity_payload, "fingerprint": fingerprint(identity_payload)}

    @property
    def case_ids(self) -> list[str]:
        return list(self._case_ids)

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return {case_id: {"patient_id": row["patient_id"], "target": row["target"]}
                for case_id, row in self._records.items()}

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._identity, ensure_ascii=False))

    @property
    def identity_hash(self) -> str:
        return str(self._identity["fingerprint"])

    def loaded_tile_order(self, case_id: str) -> dict[str, list[str]]:
        """Return tile IDs in the order used by the last verified case load."""
        case_id = _identifier(case_id, "case_id")
        if case_id not in self._loaded_tile_order:
            raise KeyError(f"Case has not been loaded in this dataset partition: {case_id}")
        return {lens: list(tile_ids) for lens, tile_ids in self._loaded_tile_order[case_id].items()}

    def load_case(self, case_id: str) -> CaseData:
        case_id = _identifier(case_id, "case_id")
        if case_id not in self._records:
            raise KeyError(f"Case is not in this dataset partition: {case_id}")
        current_manifest = _read_json_object(self.bundle / "bundle.json", "bundle manifest")
        if current_manifest.get("bundle_id") != self._manifest_id:
            raise ValueError("Bundle identity changed after dataset metadata was loaded.")
        loaded = load_case_features(
            self.bundle, case_id, self.feature_root, max_case_bytes=self.max_case_bytes,
        )
        expected = self._records[case_id]
        if (loaded.get("case_id") != case_id or loaded.get("patient_id") != expected["patient_id"]
                or loaded.get("case_label") != expected["target"]
                or loaded.get("feature_id") != self._feature_id):
            raise ValueError(f"Loaded feature metadata differs from the governed case row: {case_id}")
        raw_features = loaded.get("features")
        raw_presence = loaded.get("presence_mask")
        if not isinstance(raw_features, dict) or not isinstance(raw_presence, dict):
            raise ValueError(f"Loaded feature payload schema mismatch: {case_id}")
        features = {lens: raw_features[int(lens)] for lens in _LENS_KEYS}
        presence = tuple(raw_presence[int(lens)] for lens in _LENS_KEYS)
        if presence != expected["presence_mask"]:
            raise ValueError(f"Loaded lens presence differs from bundle case metadata: {case_id}")
        for lens in _LENS_KEYS:
            if features[lens].shape[0] != expected["instance_counts"][lens]:
                raise ValueError(f"Loaded feature count differs from bundle metadata: {case_id}/{lens}")
        tile_order = loaded.get("tile_order")
        if not isinstance(tile_order, dict):
            raise ValueError(f"Loaded tile order metadata schema mismatch: {case_id}")
        normalized_tile_order: dict[str, tuple[str, ...]] = {}
        for lens in _LENS_KEYS:
            tile_ids = tile_order.get(int(lens))
            if (not isinstance(tile_ids, list) or len(tile_ids) != features[lens].shape[0]
                    or any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids)
                    or len(tile_ids) != len(set(tile_ids))):
                raise ValueError(f"Loaded tile order differs from its feature bag: {case_id}/{lens}")
            normalized_tile_order[lens] = tuple(tile_ids)
        self._loaded_tile_order[case_id] = normalized_tile_order
        return CaseData(case_id, expected["patient_id"], expected["target"], features, presence)


def _validate_export_provenance(provenance: dict[str, Any], records: Mapping[str, dict[str, Any]]) -> None:
    required = {
        "kind", "source_bundle_id", "source_feature_id", "source_split_id",
        "outer_fold", "inner_fold", "selection_seed", "selected_cases",
    }
    if set(provenance) != required or provenance.get("kind") != "governed_bundle_subset":
        raise ValueError("Governed fixture provenance schema mismatch.")
    for key in ("source_bundle_id", "source_feature_id", "source_split_id"):
        if not isinstance(provenance.get(key), str) or not _SHA256.fullmatch(provenance[key]):
            raise ValueError(f"Invalid source digest in fixture provenance: {key}")
    for key in ("outer_fold", "inner_fold", "selection_seed"):
        if isinstance(provenance.get(key), bool) or not isinstance(provenance.get(key), int) or provenance[key] < 0:
            raise ValueError(f"Invalid export selection value: {key}")
    selected = provenance.get("selected_cases")
    if not isinstance(selected, list) or len(selected) != len(records):
        raise ValueError("Fixture provenance does not list every exported case.")
    seen: set[str] = set()
    patients: set[str] = set()
    for entry in selected:
        if not isinstance(entry, dict) or set(entry) != {"case_id", "patient_id", "target", "tiles_by_lens"}:
            raise ValueError("Invalid source case row in fixture provenance.")
        case_id = _identifier(entry["case_id"], "source case_id")
        patient_id = _identifier(entry["patient_id"], "source patient_id")
        target = _target(entry["target"], "source target")
        if case_id not in records or case_id in seen or patient_id in patients:
            raise ValueError("Export provenance case/patient IDs are duplicated or unknown.")
        if records[case_id]["patient_id"] != patient_id or records[case_id]["target"] != target:
            raise ValueError("Export provenance identity/target differs from fixture case metadata.")
        seen.add(case_id)
        patients.add(patient_id)
        tiles_by_lens = entry["tiles_by_lens"]
        if not isinstance(tiles_by_lens, dict) or set(tiles_by_lens) != set(_LENS_KEYS):
            raise ValueError("Export provenance must contain tile references for all three lenses.")
        all_tile_ids: set[str] = set()
        for lens in _LENS_KEYS:
            refs = tiles_by_lens[lens]
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"Export provenance is missing selected {lens}x tile references.")
            array_entry = records[case_id]["arrays"][lens]
            if array_entry["shape"][0] != len(refs):
                raise ValueError(f"Exported {lens}x row count differs from its source tile references.")
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != {"tile_id", "part_id", "row_index"}:
                    raise ValueError("Source tile reference schema mismatch.")
                tile_id = _identifier(ref["tile_id"], "tile_id")
                _identifier(ref["part_id"], "part_id")
                row_index = ref["row_index"]
                if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
                    raise ValueError("Source row_index must be a nonnegative integer.")
                if tile_id in all_tile_ids:
                    raise ValueError("Duplicate source tile reference in exported case.")
                all_tile_ids.add(tile_id)


def _read_source_tile_refs(bundle: Path, case_id: str, lens: str) -> list[dict[str, Any]]:
    bag_path = bundle / "bags.jsonl"
    refs_path = bundle / "instance_refs.jsonl"
    selected_bag = None
    with bag_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("case_id") == case_id and str(row.get("lens")) == lens:
                if selected_bag is not None:
                    raise ValueError(f"Duplicate case/lens bag row: {case_id}/{lens}")
                selected_bag = row
    if selected_bag is None:
        raise ValueError(f"Missing case/lens bag row: {case_id}/{lens}")
    start = selected_bag.get("ref_start_byte")
    end = selected_bag.get("ref_end_byte")
    count = selected_bag.get("instance_count")
    if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int)
            or isinstance(count, bool) or not isinstance(count, int) or start < 0 or end < start or count < 1):
        raise ValueError(f"Invalid source bag bounds: {case_id}/{lens}")
    refs = []
    with refs_path.open("rb") as stream:
        stream.seek(start)
        while stream.tell() < end:
            line = stream.readline()
            if not line or stream.tell() > end:
                raise ValueError("Source bag byte range ended inside a tile reference.")
            try:
                ref = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid source tile reference JSON.") from exc
            refs.append(ref)
        final_position = stream.tell()
    if final_position != end or len(refs) != count:
        raise ValueError("Source bag byte range/count mismatch.")
    return refs


def _choose_case_ids(
    records: Mapping[str, dict[str, Any]],
    candidates: list[str],
    bundle_id: str,
    outer_fold: int,
    inner_fold: int,
    max_cases: int,
    selection_seed: int,
) -> list[str]:
    ranked = sorted(candidates, key=lambda case_id: hashlib.sha256(
        f"{selection_seed}:{bundle_id}:{outer_fold}:{inner_fold}:{case_id}".encode("utf-8")).hexdigest())
    by_target = {
        target: [case_id for case_id in ranked if records[case_id]["target"] == target]
        for target in (0, 1)
    }
    first_zero = next((case_id for case_id in by_target[0]), None)
    if first_zero is None:
        raise ValueError("Selected inner training partition has no eligible class-0 case.")
    first_one = next((case_id for case_id in by_target[1]
                      if records[case_id]["patient_id"] != records[first_zero]["patient_id"]), None)
    if first_one is None:
        raise ValueError("Selected inner training partition has no distinct-patient class-1 case.")
    selected = [first_zero, first_one]
    used_patients = {records[case_id]["patient_id"] for case_id in selected}
    for case_id in ranked:
        if len(selected) >= max_cases:
            break
        if case_id not in selected and records[case_id]["patient_id"] not in used_patients:
            selected.append(case_id)
            used_patients.add(records[case_id]["patient_id"])
    return selected


def export_training_fixture(
    bundle: Path,
    feature_root: Path,
    output: Path,
    outer_fold: int = 0,
    inner_fold: int = 0,
    max_cases: int = 4,
    max_rows: int = 64,
) -> Path:
    """Export a small smoke subset from one nested inner-training partition.

    The governed source bundle is fully verified first. Only selected case
    features are loaded, through ``load_case_features``, and at most
    ``max_rows`` rows per lens are written.
    """
    for name, value in (("outer_fold", outer_fold), ("inner_fold", inner_fold)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    _positive_int(max_cases, "max_cases", maximum=4)
    if max_cases < 2:
        raise ValueError("max_cases must be between two and four.")
    _positive_int(max_rows, "max_rows", maximum=64)
    bundle = Path(bundle).expanduser().absolute()
    if bundle.is_symlink():
        raise ValueError("Bundle path must be an existing, non-symlink directory.")
    bundle = bundle.resolve()
    feature_root = Path(feature_root).expanduser().resolve()
    verify_pretrain_bundle(bundle, feature_root)
    manifest, records, splits = _parse_bundle_metadata(bundle)
    outer_rows = [row for row in splits["outer"] if row.get("fold") == outer_fold]
    if len(outer_rows) != 1:
        raise ValueError(f"Unknown or duplicate outer fold: {outer_fold}")
    outer = outer_rows[0]
    inner_rows = [row for row in outer["inner_folds"] if row.get("fold") == inner_fold]
    if len(inner_rows) != 1:
        raise ValueError(f"Unknown or duplicate inner fold: {inner_fold}")
    inner = inner_rows[0]
    outer_train = set(outer["train_cases"])
    outer_test = set(outer["test_cases"])
    inner_train = set(inner["train_cases"])
    inner_validation = set(inner["validation_cases"])
    if (inner_train & inner_validation or not inner_train <= outer_train
            or inner_train & outer_test or inner_train & inner_validation):
        raise ValueError("Nested training partition violates outer/inner split boundaries.")
    # Selection candidates must come only from the verified inner train side and
    # contain usable arrays at every lens. The common cohort is expected to meet
    # this condition; the explicit check keeps all-cohort bundles safe as well.
    candidates = [case_id for case_id in inner["train_cases"]
                  if case_id in records and all(records[case_id]["presence_mask"])
                  and all(records[case_id]["instance_counts"][lens] > 0 for lens in _LENS_KEYS)]
    if len(candidates) < 2:
        raise ValueError("Selected inner training partition has fewer than two complete cases.")
    selection_seed = 42
    selected_ids = _choose_case_ids(
        records, candidates, manifest["bundle_id"], outer_fold, inner_fold, max_cases, selection_seed,
    )
    if any(case_id not in inner_train or case_id in outer_test or case_id in inner_validation for case_id in selected_ids):
        raise ValueError("Smoke fixture selection crossed an outer/inner held-out partition.")
    dataset = BundleDataset(bundle, feature_root, selected_ids)
    array_records = []
    source_cases = []
    for case_id in selected_ids:
        case = dataset.load_case(case_id)
        sliced: dict[str, np.ndarray] = {}
        tile_refs: dict[str, list[dict[str, Any]]] = {}
        for lens in _LENS_KEYS:
            source_refs = _read_source_tile_refs(bundle, case_id, lens)
            tile_ids = [ref.get("tile_id") for ref in source_refs]
            if tile_ids != dataset.loaded_tile_order(case_id)[lens]:
                raise ValueError(f"Loaded tile order differs from verified source references: {case_id}/{lens}")
            if any(ref.get("case_id") != case_id or str(ref.get("objective_lens")) != lens
                   for ref in source_refs):
                raise ValueError(f"Source tile reference crossed a case/lens boundary: {case_id}/{lens}")
            available = len(source_refs)
            rng_seed = int.from_bytes(hashlib.sha256(
                f"{selection_seed}:{manifest['bundle_id']}:{case_id}:{lens}".encode("utf-8")).digest()[:8], "big")
            if available > max_rows:
                indices = np.sort(np.random.default_rng(rng_seed).choice(available, max_rows, replace=False))
            else:
                indices = np.arange(available)
            sliced[lens] = np.asarray(case.features[lens][indices], dtype=np.float32)
            tile_refs[lens] = [{
                "tile_id": _identifier(source_refs[int(index)].get("tile_id"), "tile_id"),
                "part_id": _identifier(source_refs[int(index)].get("part_id"), "part_id"),
                "row_index": int(source_refs[int(index)].get("row_index")),
            } for index in indices]
        array_records.append({
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "target": case.target,
            "features": sliced,
        })
        source_cases.append({
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "target": case.target,
            "tiles_by_lens": tile_refs,
        })
    provenance = {
        "kind": "governed_bundle_subset",
        "source_bundle_id": manifest["bundle_id"],
        "source_feature_id": manifest["feature_id"],
        "source_split_id": manifest["split_id"],
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "selection_seed": selection_seed,
        "selected_cases": source_cases,
    }
    return _publish_fixture(Path(output), array_records, provenance)
