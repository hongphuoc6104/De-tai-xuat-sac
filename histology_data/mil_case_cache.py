"""Materialize checksum-bound per-case MIL arrays after full source verification."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .bags import LENSES, load_case_features
from .feature_encoder import FEATURE_DIM
from .io import atomic_json, file_hash, fingerprint
from .mil_data import BundleDataset, CaseData
from .readiness import verify_pretrain_bundle

SCHEMA_VERSION = 1
MAX_CASE_BYTES = 512 * 1024 * 1024
_LENS_KEYS = tuple(str(lens) for lens in LENSES)
_SHA256_LENGTH = 64
_SAFE_ID = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")


def _check_deadline(deadline: float) -> None:
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic timestamp.")
    if time.monotonic() >= deadline:
        raise TimeoutError("MIL case-cache deadline reached.")


def _sha256(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if deadline is not None:
                _check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _safe_relative(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path for {name}.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"Unsafe relative path for {name}: {value!r}")
    return value


def _sha_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"Invalid SHA-256 identity: {name}")
    return value


def _identifier(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip() or value in {".", ".."}
            or not _SAFE_ID.fullmatch(value)):
        raise ValueError(f"Invalid {name}.")
    return value


def _regular_file(root: Path, relative: str, name: str) -> Path:
    relative = _safe_relative(relative, name)
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlink source directory is not allowed: {cursor}")
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing, symlinked or escaped source file: {name}")
    return path


def _hardlink_feature_mirror(feature_root: Path, deadline: float) -> tuple[Path, Path]:
    source = Path(feature_root).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("feature_root must be a directory.")
    temporary = Path(tempfile.mkdtemp(prefix=".mil-case-cache-verify-", dir=source.parent))
    mirror = temporary / "features"
    mirror.mkdir()
    for current, directory_names, filenames in os.walk(source, followlinks=False):
        _check_deadline(deadline)
        current_path = Path(current)
        relative_dir = current_path.relative_to(source)
        for name in directory_names:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"Feature release has a symlink directory: {child}")
        target_dir = mirror / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _check_deadline(deadline)
            src = current_path / filename
            if src.is_symlink() or not src.is_file():
                raise ValueError(f"Feature release has a non-regular source file: {src}")
            target = target_dir / filename
            try:
                os.link(src, target)
            except OSError as exc:
                raise OSError("Verified MIL cache preparation requires local hard-link support.") from exc
    return temporary, mirror


def verify_source_readonly(bundle: Path, feature_root: Path, deadline: float) -> dict[str, Any]:
    """Run the complete source verifier in a mirror, leaving every source byte untouched."""
    temporary, mirror = _hardlink_feature_mirror(feature_root, deadline)
    try:
        return verify_pretrain_bundle(bundle, mirror, deadline=deadline)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _validate_bundle_source_files(bundle: Path, deadline: float) -> dict[str, Any]:
    bundle = Path(bundle).resolve(strict=True)
    manifest = _json_object(_regular_file(bundle, "bundle.json", "bundle manifest"), "bundle manifest")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Bundle artifact hashes are missing.")
    names = {"bundle.json", "training_readiness.json", *artifacts}
    for name in names:
        _check_deadline(deadline)
        _regular_file(bundle, name, "bundle input")
    return manifest


def _source_identity(bundle: Path, feature_root: Path, dataset: BundleDataset,
                     deadline: float | None = None) -> dict[str, Any]:
    bundle_manifest_path = Path(bundle) / "bundle.json"
    bundle_manifest = _json_object(bundle_manifest_path, "bundle manifest")
    bundle_id = _sha_id(bundle_manifest.get("bundle_id"), "bundle_id")
    feature_id = _sha_id(bundle_manifest.get("feature_id"), "feature_id")
    split_id = _sha_id(bundle_manifest.get("split_id"), "split_id")
    if dataset.identity.get("bundle_id") != bundle_id or dataset.identity.get("feature_id") != feature_id:
        raise ValueError("BundleDataset identity differs from source bundle metadata.")
    artifact_hashes = bundle_manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Bundle artifact hashes are missing.")
    for name, digest in artifact_hashes.items():
        _safe_relative(name, "bundle artifact")
        _sha_id(digest, f"bundle/{name}")
    root = Path(feature_root).resolve(strict=True)
    status_path = _regular_file(root, "status.json", "feature status")
    config_path = _regular_file(root, f"{feature_id}/feature_config.json", "feature config")
    release_path = _regular_file(root, f"{feature_id}/release.json", "feature release")
    status = _json_object(status_path, "feature status")
    config = _json_object(config_path, "feature config")
    release = _json_object(release_path, "feature release")
    if (status.get("feature_id") != feature_id or fingerprint(config) != feature_id
            or release.get("feature_id") != feature_id):
        raise ValueError("Feature status/config/release has a different feature_id.")
    if (_sha256(config_path, deadline) != bundle_manifest.get("feature_config_sha256")
            or _sha256(release_path, deadline) != bundle_manifest.get("feature_release_sha256")):
        raise ValueError("Feature config/release hashes differ from the governed bundle.")
    audit_relative = _safe_relative(release.get("audit_file"), "feature audit")
    audit_path = _regular_file(root, f"{feature_id}/{audit_relative}", "feature audit")
    audit_hash = release.get("audit_sha256")
    if audit_hash != bundle_manifest.get("feature_audit_sha256") or _sha256(audit_path, deadline) != audit_hash:
        raise ValueError("Feature audit hash differs from the governed bundle.")
    source_hashes = {
        "bundle_manifest_sha256": _sha256(bundle_manifest_path, deadline),
        "bundle_artifact_sha256": dict(sorted(artifact_hashes.items())),
        "feature_status_sha256": _sha256(status_path, deadline),
        "feature_config_sha256": str(bundle_manifest["feature_config_sha256"]),
        "feature_release_sha256": str(bundle_manifest["feature_release_sha256"]),
        "feature_audit_sha256": str(bundle_manifest["feature_audit_sha256"]),
    }
    records = {
        case_id: {"patient_id": dataset.records[case_id]["patient_id"], "target": dataset.records[case_id]["target"]}
        for case_id in sorted(dataset.case_ids)
    }
    instance_counts: dict[str, dict[str, int]] = {}
    with (Path(bundle) / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"case_id", "instance_counts_by_lens_json"}.issubset(reader.fieldnames or ()):
            raise ValueError("Bundle case-count metadata schema mismatch.")
        for row in reader:
            case_id = row.get("case_id", "")
            if case_id not in records:
                continue
            try:
                counts = json.loads(row["instance_counts_by_lens_json"], parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid source instance counts for {case_id}.") from exc
            if not isinstance(counts, dict) or set(counts) != set(_LENS_KEYS):
                raise ValueError(f"Invalid source lens counts for {case_id}.")
            clean_counts = {}
            for lens in _LENS_KEYS:
                count = counts[lens]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"Invalid source instance count for {case_id}/{lens}.")
                clean_counts[lens] = count
            if case_id in instance_counts:
                raise ValueError(f"Duplicate source case count row: {case_id}")
            instance_counts[case_id] = clean_counts
    if set(instance_counts) != set(records):
        raise ValueError("Source case instance counts differ from bundle membership.")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": "verified_materialized_case_cache",
        "bundle_id": bundle_id,
        "feature_id": feature_id,
        "split_id": split_id,
        "source_hashes": source_hashes,
        "case_ids": sorted(dataset.case_ids),
        "case_records": records,
        "case_instance_counts": instance_counts,
    }
    return {**body, "cache_id": fingerprint(body)}


def _cache_manifest_body(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in {"cache_id", "manifest_sha256", "verified_at_utc"}}


def _validate_cache_manifest(cache_path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = _json_object(cache_path / "cache_manifest.json", "case-cache manifest")
    body = _cache_manifest_body(manifest)
    cache_id = manifest.get("cache_id")
    if set(manifest) != set(body) | {"cache_id", "manifest_sha256", "verified_at_utc"}:
        raise ValueError("Case-cache manifest schema mismatch.")
    if (not isinstance(cache_id, str) or cache_id != fingerprint(body)
            or manifest.get("manifest_sha256") != fingerprint(body)):
        raise ValueError("Case-cache manifest fingerprint mismatch.")
    expected_body = None if expected is None else {key: value for key, value in expected.items() if key != "cache_id"}
    if expected is not None and (body != expected_body or cache_id != expected["cache_id"]):
        raise ValueError("Existing case cache belongs to another source bundle or artifact inventory.")
    return manifest


def _case_folder(case_id: str) -> str:
    _identifier(case_id, "case_id")
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def _case_commit(
    case_dir: Path,
    case_id: str,
    patient_id: str,
    target: int,
    cache_id: str,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    if case_dir.is_symlink() or not case_dir.is_dir():
        raise ValueError(f"Case-cache directory is missing or symlinked: {case_id}")
    commit = _json_object(case_dir / "commit.json", "case-cache commit")
    body = {key: value for key, value in commit.items() if key != "commit_id"}
    if set(commit) != {
        "schema_version", "cache_id", "case_id", "patient_id", "target", "presence_mask",
        "features", "tile_order", "commit_id",
    } or commit.get("commit_id") != fingerprint(body):
        raise ValueError(f"Case-cache commit fingerprint/schema mismatch: {case_id}")
    if (commit.get("schema_version") != SCHEMA_VERSION or commit.get("cache_id") != cache_id
            or commit.get("case_id") != case_id or commit.get("patient_id") != patient_id
            or type(commit.get("target")) is not int or commit.get("target") != target):
        raise ValueError(f"Case-cache identity/label differs from governed metadata: {case_id}")
    features = commit.get("features")
    tile_order = commit.get("tile_order")
    presence = commit.get("presence_mask")
    if (not isinstance(features, dict) or set(features) != set(_LENS_KEYS)
            or not isinstance(tile_order, dict) or set(tile_order) != set(_LENS_KEYS)
            or not isinstance(presence, list) or len(presence) != len(_LENS_KEYS)
            or any(type(value) is not bool for value in presence)
            or any(not isinstance(features[lens], dict) for lens in _LENS_KEYS)
            or any(not isinstance(tile_order[lens], dict) for lens in _LENS_KEYS)):
        raise ValueError(f"Case-cache lens count schema mismatch: {case_id}")
    for index, lens in enumerate(_LENS_KEYS):
        entry, tile_entry = features[lens], tile_order[lens]
        shape = entry.get("shape") if isinstance(entry, dict) else None
        count = tile_entry.get("count") if isinstance(tile_entry, dict) else None
        if (not isinstance(shape, list) or shape != [expected_counts[lens], FEATURE_DIM]
                or entry.get("path") != f"lens_{lens}.npy" or entry.get("dtype") != "float32"
                or tile_entry.get("path") != f"tile_order_{lens}.json"
                or isinstance(count, bool) or count != expected_counts[lens]
                or presence[index] != (expected_counts[lens] > 0)):
            raise ValueError(f"Cached feature count differs from verified source: {case_id}/{lens}")
        _sha_id(entry.get("sha256"), f"{case_id}/{lens} feature")
        _sha_id(tile_entry.get("sha256"), f"{case_id}/{lens} tile order")
    return commit


def _prepare_case_tree(cache_path: Path, case_ids: list[str], *, complete: bool) -> Path:
    cases_root = cache_path / "cases"
    if cases_root.is_symlink():
        raise ValueError("Case-cache cases directory must not be a symlink.")
    cases_root.mkdir(exist_ok=True)
    expected = {_case_folder(case_id): case_id for case_id in case_ids}
    actual = set()
    for child in cases_root.iterdir():
        if child.is_symlink():
            raise ValueError(f"Symlink in case-cache directory: {child.name}")
        if child.name.startswith("."):
            if complete or not any(child.name.startswith(f".{folder}.") for folder in expected):
                raise ValueError(f"Unexpected temporary cache entry: {child.name}")
            shutil.rmtree(child)
            continue
        if child.name not in expected or not child.is_dir():
            raise ValueError(f"Unexpected case-cache entry: {child.name}")
        actual.add(child.name)
    if complete and actual != set(expected):
        raise ValueError("Case cache is incomplete; every source case needs a committed payload.")
    return cases_root


def _validate_cache_root_entries(cache_path: Path) -> None:
    for child in cache_path.iterdir():
        if child.is_symlink():
            raise ValueError(f"Symlink in cache version root: {child.name}")
        if child.name == "cache_manifest.json":
            if not child.is_file():
                raise ValueError("cache_manifest.json must be a regular file.")
        elif child.name == "cases":
            if not child.is_dir():
                raise ValueError("Case cache payload root must be a directory.")
        else:
            raise ValueError(f"Unexpected file in case-cache version root: {child.name}")


def _case_files(case_dir: Path, case_id: str, patient_id: str, target: int,
                cache_id: str, max_case_bytes: int,
                expected_counts: dict[str, int],
                deadline: float | None = None) -> tuple[CaseData, dict[str, list[str]]]:
    if case_dir.is_symlink() or not case_dir.is_dir():
        raise ValueError(f"Case-cache directory is missing or symlinked: {case_id}")
    expected_names = {"commit.json"}
    expected_names.update(f"lens_{lens}.npy" for lens in _LENS_KEYS)
    expected_names.update(f"tile_order_{lens}.json" for lens in _LENS_KEYS)
    actual_names = set()
    for path in case_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected non-file in case-cache directory: {path}")
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise ValueError(f"Case-cache file set mismatch: {case_id}")
    commit = _case_commit(case_dir, case_id, patient_id, target, cache_id, expected_counts)
    feature_entries = commit.get("features")
    tile_entries = commit.get("tile_order")
    presence = commit.get("presence_mask")
    if (not isinstance(feature_entries, dict) or set(feature_entries) != set(_LENS_KEYS)
            or not isinstance(tile_entries, dict) or set(tile_entries) != set(_LENS_KEYS)
            or not isinstance(presence, list) or len(presence) != len(_LENS_KEYS)
            or any(type(value) is not bool for value in presence)):
        raise ValueError(f"Case-cache payload schema mismatch: {case_id}")
    total_bytes = 0
    rows_by_lens: dict[str, int] = {}
    for lens in _LENS_KEYS:
        if deadline is not None:
            _check_deadline(deadline)
        entry = feature_entries[lens]
        tile_entry = tile_entries[lens]
        if (not isinstance(entry, dict) or entry.get("path") != f"lens_{lens}.npy"
                or entry.get("dtype") != "float32" or not isinstance(entry.get("shape"), list)
                or len(entry["shape"]) != 2 or entry["shape"][1] != FEATURE_DIM
                or isinstance(entry["shape"][0], bool) or not isinstance(entry["shape"][0], int)
                or entry["shape"][0] < 0):
            raise ValueError(f"Invalid cached feature schema: {case_id}/{lens}")
        if (not isinstance(tile_entry, dict) or tile_entry.get("path") != f"tile_order_{lens}.json"
                or isinstance(tile_entry.get("count"), bool) or not isinstance(tile_entry.get("count"), int)
                or tile_entry["count"] != entry["shape"][0]):
            raise ValueError(f"Invalid cached tile-order schema: {case_id}/{lens}")
        rows = entry["shape"][0]
        rows_by_lens[lens] = rows
        total_bytes += rows * FEATURE_DIM * np.dtype(np.float32).itemsize
    if total_bytes > max_case_bytes:
        raise MemoryError(f"Case {case_id} requires {total_bytes} bytes, above max_case_bytes={max_case_bytes}.")

    features: dict[str, np.ndarray] = {}
    orders: dict[str, list[str]] = {}
    for index, lens in enumerate(_LENS_KEYS):
        if deadline is not None:
            _check_deadline(deadline)
        feature_entry = feature_entries[lens]
        tile_entry = tile_entries[lens]
        array_path = case_dir / feature_entry["path"]
        tile_path = case_dir / tile_entry["path"]
        for path, expected_hash in ((array_path, feature_entry.get("sha256")),
                                    (tile_path, tile_entry.get("sha256"))):
            _sha_id(expected_hash, f"{case_id}/{lens} file")
            if path.is_symlink() or not path.is_file() or _sha256(path, deadline) != expected_hash:
                raise ValueError(f"Case-cache checksum mismatch: {case_id}/{lens}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or list(array.shape) != feature_entry["shape"]:
            raise ValueError(f"Cached feature array schema changed: {case_id}/{lens}")
        try:
            tile_ids = json.loads(tile_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot read cached tile order: {case_id}/{lens}") from exc
        if (not isinstance(tile_ids, list) or len(tile_ids) != rows_by_lens[lens]
                or any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_ids)
                or len(tile_ids) != len(set(tile_ids))):
            raise ValueError(f"Cached tile order is invalid: {case_id}/{lens}")
        for offset in range(0, len(array), 256):
            if deadline is not None:
                _check_deadline(deadline)
            if not np.isfinite(array[offset:offset + 256]).all():
                raise ValueError(f"Cached feature array contains non-finite values: {case_id}/{lens}")
        if presence[index] != (rows_by_lens[lens] > 0):
            raise ValueError(f"Cached lens-presence mask differs from arrays: {case_id}/{lens}")
        features[lens] = array
        orders[lens] = tile_ids
    return CaseData(case_id, patient_id, target, features, tuple(presence)), orders


def _write_case(cache_path: Path, case: CaseData, tile_order: dict[str, list[str]], cache_id: str,
                expected_counts: dict[str, int], deadline: float) -> None:
    cases_root = cache_path / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    folder = _case_folder(case.case_id)
    final_dir = cases_root / folder
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"Uncommitted cache path exists for case {case.case_id}.")
    temporary = Path(tempfile.mkdtemp(prefix=f".{folder}.", dir=cases_root))
    try:
        feature_entries: dict[str, Any] = {}
        tile_entries: dict[str, Any] = {}
        for lens in _LENS_KEYS:
            _check_deadline(deadline)
            array = case.features[lens]
            if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != FEATURE_DIM:
                raise ValueError(f"Public source loader returned an invalid array: {case.case_id}/{lens}")
            for offset in range(0, len(array), 256):
                _check_deadline(deadline)
                if not np.isfinite(array[offset:offset + 256]).all():
                    raise ValueError(f"Public source loader returned non-finite features: {case.case_id}/{lens}")
            array_path = temporary / f"lens_{lens}.npy"
            with array_path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            tile_path = temporary / f"tile_order_{lens}.json"
            atomic_json(tile_path, tile_order[lens])
            feature_entries[lens] = {
                "path": array_path.name,
                "sha256": _sha256(array_path, deadline),
                "shape": [int(array.shape[0]), FEATURE_DIM],
                "dtype": "float32",
            }
            tile_entries[lens] = {
                "path": tile_path.name,
                "sha256": _sha256(tile_path, deadline),
                "count": len(tile_order[lens]),
            }
        body = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": cache_id,
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "target": case.target,
            "presence_mask": list(case.presence_mask),
            "features": feature_entries,
            "tile_order": tile_entries,
        }
        atomic_json(temporary / "commit.json", {**body, "commit_id": fingerprint(body)})
        _case_files(temporary, case.case_id, case.patient_id, case.target, cache_id, MAX_CASE_BYTES,
                    expected_counts, deadline)
        _check_deadline(deadline)
        os.rename(temporary, final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_case_cache(bundle: Path, feature_root: Path, cache_root: Path, deadline: float) -> Path:
    """Verify the governed release, then atomically materialize each full case once."""
    _check_deadline(deadline)
    bundle = Path(bundle).expanduser().resolve(strict=True)
    feature_root = Path(feature_root).expanduser().resolve(strict=True)
    cache_root = Path(cache_root).expanduser().absolute()
    if cache_root.is_symlink():
        raise ValueError("cache_root must not be a symlink.")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root.resolve(strict=True)
    if (cache_root == bundle or cache_root.is_relative_to(bundle) or bundle.is_relative_to(cache_root)
            or cache_root == feature_root or cache_root.is_relative_to(feature_root)
            or feature_root.is_relative_to(cache_root)):
        raise ValueError("cache_root must be separate from the immutable bundle and feature source.")

    _validate_bundle_source_files(bundle, deadline)
    verify_source_readonly(bundle, feature_root, deadline)
    initial_manifest = _json_object(bundle / "bundle.json", "bundle manifest")
    all_case_ids = initial_manifest.get("cohort_case_ids")
    if not isinstance(all_case_ids, list) or not all_case_ids or len(all_case_ids) != len(set(all_case_ids)):
        raise ValueError("Bundle case IDs must be nonempty and unique.")
    source_dataset = BundleDataset(bundle, feature_root, all_case_ids, max_case_bytes=MAX_CASE_BYTES)
    expected = _source_identity(bundle, feature_root, source_dataset, deadline)
    cache_path = cache_root / expected["cache_id"]
    cache_path.mkdir(exist_ok=True)
    if cache_path.is_symlink() or not cache_path.is_dir():
        raise ValueError("Case-cache version path is not a regular directory.")
    manifest_path = cache_path / "cache_manifest.json"
    if manifest_path.exists():
        _validate_cache_manifest(cache_path, expected)
    else:
        if any(cache_path.iterdir()):
            raise ValueError("Unmanifested files exist in the requested case-cache version.")
        manifest_body = {key: value for key, value in expected.items() if key != "cache_id"}
        atomic_json(manifest_path, {
            **manifest_body,
            "cache_id": expected["cache_id"],
            "manifest_sha256": fingerprint(manifest_body),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        _validate_cache_manifest(cache_path, expected)

    _validate_cache_root_entries(cache_path)
    _prepare_case_tree(cache_path, source_dataset.case_ids, complete=False)

    for case_id in source_dataset.case_ids:
        _check_deadline(deadline)
        metadata = source_dataset.records[case_id]
        expected_counts = expected["case_instance_counts"][case_id]
        case_dir = cache_path / "cases" / _case_folder(case_id)
        if case_dir.exists() or case_dir.is_symlink():
            _case_files(case_dir, case_id, metadata["patient_id"], metadata["target"],
                        expected["cache_id"], MAX_CASE_BYTES, expected_counts, deadline)
            continue
        loaded = load_case_features(bundle, case_id, feature_root, max_case_bytes=MAX_CASE_BYTES, deadline=deadline)
        if (loaded.get("case_id") != case_id or loaded.get("patient_id") != metadata["patient_id"]
                or loaded.get("case_label") != metadata["target"]
                or loaded.get("feature_id") != expected["feature_id"]):
            raise ValueError(f"Public loader metadata differs from governed case row: {case_id}")
        features = {lens: loaded["features"][int(lens)] for lens in _LENS_KEYS}
        presence = tuple(loaded["presence_mask"][int(lens)] for lens in _LENS_KEYS)
        if any(features[lens].shape[0] != expected_counts[lens] for lens in _LENS_KEYS):
            raise ValueError(f"Public loader feature counts differ from governed case metadata: {case_id}")
        case = CaseData(case_id, metadata["patient_id"], metadata["target"], features, presence)
        tile_order = {lens: loaded["tile_order"][int(lens)] for lens in _LENS_KEYS}
        for lens in _LENS_KEYS:
            if (not isinstance(tile_order[lens], list)
                    or any(not isinstance(tile_id, str) or not tile_id for tile_id in tile_order[lens])
                    or len(tile_order[lens]) != case.features[lens].shape[0]
                    or len(tile_order[lens]) != len(set(tile_order[lens]))):
                raise ValueError(f"Public loader tile order differs from features: {case_id}/{lens}")
        _write_case(cache_path, case, tile_order, expected["cache_id"], expected_counts, deadline)
    _check_deadline(deadline)
    _prepare_case_tree(cache_path, source_dataset.case_ids, complete=True)
    return cache_path


class CacheDataset:
    """One-case adapter over an immutable cache bound to the original bundle."""

    def __init__(
        self,
        bundle: Path,
        feature_root: Path,
        cache_path: Path,
        partition_ids: list[str],
        max_case_bytes: int = MAX_CASE_BYTES,
    ) -> None:
        if isinstance(max_case_bytes, bool) or not isinstance(max_case_bytes, int) or max_case_bytes < 1:
            raise ValueError("max_case_bytes must be a positive integer.")
        self.bundle = Path(bundle).expanduser().resolve(strict=True)
        self.feature_root = Path(feature_root).expanduser().resolve(strict=True)
        cache_input = Path(cache_path).expanduser().absolute()
        if cache_input.is_symlink():
            raise ValueError("cache_path must not be a symlink.")
        self.cache_path = cache_input.resolve(strict=True)
        if not self.cache_path.is_dir():
            raise ValueError("cache_path must be an existing, non-symlink directory.")
        manifest = _validate_cache_manifest(self.cache_path)
        _validate_cache_root_entries(self.cache_path)
        expected_ids = manifest.get("case_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            raise ValueError("Case-cache manifest has no case IDs.")
        self._source_dataset = BundleDataset(self.bundle, self.feature_root, expected_ids,
                                             max_case_bytes=max_case_bytes)
        identity = _source_identity(self.bundle, self.feature_root, self._source_dataset)
        _validate_cache_manifest(self.cache_path, identity)
        _prepare_case_tree(self.cache_path, expected_ids, complete=True)
        source_hashes = manifest.get("source_hashes", {})
        if file_hash(self.bundle / "bundle.json") != source_hashes.get("bundle_manifest_sha256"):
            raise ValueError("Original bundle manifest changed after case-cache preparation.")
        for case_id in expected_ids:
            if self._source_dataset.records[case_id] != manifest["case_records"][case_id]:
                raise ValueError(f"Case-cache labels/identity differ from source bundle: {case_id}")
            metadata = self._source_dataset.records[case_id]
            _case_commit(self.cache_path / "cases" / _case_folder(case_id), case_id,
                         metadata["patient_id"], metadata["target"], manifest["cache_id"],
                         manifest["case_instance_counts"][case_id])
        if (not isinstance(partition_ids, list) or not partition_ids
                or any(not isinstance(case_id, str) for case_id in partition_ids)
                or len(partition_ids) != len(set(partition_ids))):
            raise ValueError("partition_ids must be a nonempty list of unique case IDs.")
        if set(partition_ids) - set(expected_ids):
            raise ValueError("partition_ids include cases outside the materialized cache.")
        self.max_case_bytes = max_case_bytes
        self._manifest_id = str(manifest["manifest_sha256"])
        self._cache_id = str(manifest["cache_id"])
        self._bundle_manifest_sha256 = str(source_hashes["bundle_manifest_sha256"])
        self._case_ids = tuple(partition_ids)
        self._records = MappingProxyType({case_id: self._source_dataset.records[case_id] for case_id in partition_ids})
        self._source_instance_counts = MappingProxyType({
            case_id: MappingProxyType(dict(manifest["case_instance_counts"][case_id]))
            for case_id in partition_ids
        })
        identity_body = {
            "kind": "verified_materialized_case_cache",
            "cache_id": self._cache_id,
            "bundle_id": manifest["bundle_id"],
            "feature_id": manifest["feature_id"],
            "split_id": manifest["split_id"],
            "source_manifest_sha256": self._bundle_manifest_sha256,
            "cached_source_manifest_sha": self._manifest_id,
            "case_ids": sorted(partition_ids),
        }
        self._identity = {**identity_body, "fingerprint": fingerprint(identity_body)}
        self._loaded_tile_order: dict[str, dict[str, tuple[str, ...]]] = {}

    @property
    def case_ids(self) -> list[str]:
        return list(self._case_ids)

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return {case_id: dict(row) for case_id, row in self._records.items()}

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._identity, ensure_ascii=False))

    @property
    def identity_hash(self) -> str:
        return str(self._identity["fingerprint"])

    def loaded_tile_order(self, case_id: str) -> dict[str, list[str]]:
        if case_id not in self._loaded_tile_order:
            raise KeyError(f"Case has not been loaded from this cache: {case_id}")
        return {lens: list(ids) for lens, ids in self._loaded_tile_order[case_id].items()}

    def load_case(self, case_id: str) -> CaseData:
        if case_id not in self._records:
            raise KeyError(f"Case is outside this cache partition: {case_id}")
        current = _validate_cache_manifest(self.cache_path)
        if (current.get("manifest_sha256") != self._manifest_id or current.get("cache_id") != self._cache_id
                or file_hash(self.bundle / "bundle.json") != self._bundle_manifest_sha256):
            raise ValueError("Cache/source manifest changed after dataset initialization.")
        record = self._records[case_id]
        case_dir = self.cache_path / "cases" / _case_folder(case_id)
        case, tile_order = _case_files(
            case_dir, case_id, record["patient_id"], record["target"], self._cache_id, self.max_case_bytes,
            self._source_instance_counts[case_id],
        )
        self._loaded_tile_order[case_id] = {lens: tuple(tile_order[lens]) for lens in _LENS_KEYS}
        return case
