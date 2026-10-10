"""Streamed case-by-lens feature references and one-case loading adapter."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .feature_encoder import FEATURE_DIM
from .features import BudgetExhausted, verify_feature_part
from .governance import CASE_FIELDS
from .io import file_hash, fingerprint

LENSES = (4, 10, 40)
BAG_FIELDS = (
    "bag_id", "case_id", "lens", "feature_id", "present", "instance_count",
    "ref_start_byte", "ref_end_byte", "ref_start_line", "ref_end_line",
)
BUNDLE_CASE_FIELDS = CASE_FIELDS + (
    "cohort_mode", "lens_presence_json", "instance_counts_by_lens_json",
)
_SAFE_PART_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _check_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic timestamp or None.")
    if time.monotonic() >= deadline:
        raise BudgetExhausted("Case/lens feature reference deadline reached.")


def _bounded_file_hash(path: Path, deadline: float | None) -> str:
    if deadline is None:
        return file_hash(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            _check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _jsonl_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _csv_payload(fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output.getvalue().encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _stable_bag_id(case_id: str, lens: int) -> str:
    return "case-lens-v1-" + fingerprint({"case_id": case_id, "lens": lens})[:24]


def write_bundle_tables(
    bundle_dir: Path,
    cases: list[dict[str, Any]],
    instance_rows: Iterable[dict[str, Any]],
    *,
    feature_id: str,
    cohort_mode: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Write case/lens manifests and references without copying feature vectors."""
    _check_deadline(deadline)
    if cohort_mode not in {"common", "all"}:
        raise ValueError("cohort_mode must be 'common' or 'all'.")
    if not _SHA256.fullmatch(feature_id):
        raise ValueError("feature_id must be a SHA-256 fingerprint.")
    case_by_id = {str(case["case_id"]): case for case in cases}
    if len(case_by_id) != len(cases) or not cases:
        raise ValueError("Bundle cases must be nonempty and unique.")
    output = Path(bundle_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = {case_id: {lens: 0 for lens in LENSES} for case_id in case_by_id}
    group_paths: dict[tuple[str, int], Path] = {}
    group_handles: dict[tuple[str, int], Any] = {}
    with tempfile.TemporaryDirectory(prefix="pretrain-bags-") as temporary:
        scratch = Path(temporary)
        try:
            for reference_number, ref in enumerate(instance_rows, start=1):
                if reference_number % 256 == 0:
                    _check_deadline(deadline)
                case_id = str(ref.get("case_id", ""))
                if case_id not in case_by_id:
                    raise ValueError(f"Feature reference has no selected governed case: {case_id}")
                lens = ref.get("objective_lens")
                if isinstance(lens, str) and lens.isdigit():
                    lens = int(lens)
                if lens not in LENSES:
                    raise ValueError(f"Invalid lens in feature reference for {case_id}: {lens!r}")
                if ref.get("feature_id") != feature_id:
                    raise ValueError("Feature reference version differs from the bundle feature_id.")
                if any(key in ref for key in ("target", "patch_target", "case_label", "label")):
                    raise ValueError("Instance references cannot contain labels or targets.")
                if int(ref.get("row_index", -1)) < 0 or not _SAFE_PART_ID.fullmatch(str(ref.get("part_id", ""))):
                    raise ValueError("Feature reference has an invalid part_id or row_index.")
                group = (case_id, int(lens))
                if group not in group_paths:
                    group_paths[group] = scratch / (fingerprint(group) + ".jsonl")
                    group_handles[group] = group_paths[group].open("wb")
                payload = _jsonl_bytes(ref)
                group_handles[group].write(payload)
                counts[case_id][int(lens)] += 1
        finally:
            for handle in group_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

        refs_path = output / "instance_refs.jsonl"
        bags_path = output / "bags.jsonl"
        bag_rows: list[dict[str, Any]] = []
        ref_offset = 0
        line_number = 0
        with refs_path.open("wb") as refs_stream:
            for case_id in sorted(case_by_id):
                for lens in LENSES:
                    start_byte = ref_offset
                    start_line = line_number
                    source_path = group_paths.get((case_id, lens))
                    if source_path is not None:
                        with source_path.open("rb") as source_stream:
                            while block := source_stream.read(1024 * 1024):
                                _check_deadline(deadline)
                                refs_stream.write(block)
                                ref_offset += len(block)
                                line_number += block.count(b"\n")
                    count = counts[case_id][lens]
                    if line_number - start_line != count:
                        raise ValueError("Bag reference count differs from the streamed instance rows.")
                    bag_rows.append({
                        "bag_id": _stable_bag_id(case_id, lens),
                        "case_id": case_id,
                        "lens": lens,
                        "feature_id": feature_id,
                        "present": count > 0,
                        "instance_count": count,
                        "ref_start_byte": start_byte,
                        "ref_end_byte": ref_offset,
                        "ref_start_line": start_line,
                        "ref_end_line": line_number,
                    })
            refs_stream.flush()
            os.fsync(refs_stream.fileno())
        _write_atomic(bags_path, b"".join(_jsonl_bytes(row) for row in bag_rows))

    case_rows = []
    for case_number, case_id in enumerate(sorted(case_by_id), start=1):
        if case_number % 256 == 0:
            _check_deadline(deadline)
        case = case_by_id[case_id]
        presence = {str(lens): counts[case_id][lens] > 0 for lens in LENSES}
        case_rows.append({
            **{key: case[key] for key in CASE_FIELDS},
            "cohort_mode": cohort_mode,
            "lens_presence_json": json.dumps(presence, sort_keys=True, separators=(",", ":")),
            "instance_counts_by_lens_json": json.dumps(
                {str(lens): counts[case_id][lens] for lens in LENSES}, sort_keys=True, separators=(",", ":"),
            ),
        })
    cases_path = output / "cases.csv"
    _write_atomic(cases_path, _csv_payload(BUNDLE_CASE_FIELDS, case_rows))
    return {
        "cases_path": cases_path,
        "bags_path": bags_path,
        "instance_refs_path": refs_path,
        "case_count": len(case_rows),
        "bag_count": len(bag_rows),
        "instance_count": line_number,
        "instance_counts_by_case": counts,
        "artifact_sha256": {
            "cases.csv": _bounded_file_hash(cases_path, deadline),
            "bags.jsonl": _bounded_file_hash(bags_path, deadline),
            "instance_refs.jsonl": _bounded_file_hash(refs_path, deadline),
        },
    }


def _validate_bundle(
    bundle_dir: Path,
    feature_root: Path,
    *,
    deadline: float | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _check_deadline(deadline)
    bundle_dir = Path(bundle_dir)
    feature_root = Path(feature_root)
    manifest_path = bundle_dir / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_id = manifest.pop("bundle_id", None)
    if bundle_id != fingerprint(manifest):
        raise ValueError("Bundle fingerprint mismatch.")
    if not manifest.get("training_ready"):
        raise ValueError("Bundle is blocked and cannot load features.")
    for name, expected_sha in manifest["artifact_sha256"].items():
        _check_deadline(deadline)
        path = bundle_dir / name
        if not path.is_file() or _bounded_file_hash(path, deadline) != expected_sha:
            raise ValueError(f"Bundle artifact checksum mismatch: {name}")
    readiness = json.loads((bundle_dir / "training_readiness.json").read_text(encoding="utf-8"))
    if (not readiness.get("training_ready") or readiness.get("bundle_id") != bundle_id
            or readiness.get("feature_id") != manifest.get("feature_id")):
        raise ValueError("Bundle readiness report is not bound to this bundle/feature version.")
    feature_id = manifest["feature_id"]
    if not _SHA256.fullmatch(feature_id):
        raise ValueError("Invalid bundle feature_id.")
    feature_dir = feature_root / feature_id
    config_path = feature_dir / "feature_config.json"
    release_path = feature_dir / "release.json"
    if not config_path.is_file() or not release_path.is_file():
        raise ValueError("Bundle feature configuration/release is missing.")
    if _bounded_file_hash(config_path, deadline) != manifest.get("feature_config_sha256"):
        raise ValueError("Feature configuration checksum differs from bundle manifest.")
    if _bounded_file_hash(release_path, deadline) != manifest.get("feature_release_sha256"):
        raise ValueError("Feature release checksum differs from bundle manifest.")
    feature_config = json.loads(config_path.read_text(encoding="utf-8"))
    if fingerprint(feature_config) != feature_id:
        raise ValueError("Feature configuration fingerprint changed since bundle creation.")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release.get("feature_id") != feature_id or release.get("source_complete") is not True
            or release.get("feature_complete") is not True or release.get("audit_complete") is not True
            or release.get("observed_sources") != manifest.get("expected_sources")
            or release.get("observed_pngs") != manifest.get("expected_feature_vectors")
            or release.get("committed_vectors") != manifest.get("expected_feature_vectors")):
        raise ValueError("Feature release is no longer full and audited.")
    with (bundle_dir / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BUNDLE_CASE_FIELDS:
            raise ValueError("Bundle cases.csv schema mismatch.")
        cases = {str(row["case_id"]): row for row in reader}
    if not cases or len(cases) != int(manifest["cohort_case_count"]):
        raise ValueError("Bundle case count is invalid.")
    return manifest, cases


def _read_case_bags(
    bundle_dir: Path, case_id: str, *, deadline: float | None = None,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with (Path(bundle_dir) / "bags.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number % 256 == 0:
                _check_deadline(deadline)
            row = json.loads(line)
            if row.get("case_id") != case_id:
                continue
            lens = int(row["lens"])
            if lens in selected:
                raise ValueError(f"Duplicate case/lens bag row: {case_id}/{lens}")
            selected[lens] = row
    if set(selected) != set(LENSES):
        raise ValueError(f"Case {case_id} must have exactly one bag row for each of {LENSES}.")
    return selected


def _read_reference_range(
    path: Path, bag: dict[str, Any], *, deadline: float | None = None,
) -> list[dict[str, Any]]:
    start, end = int(bag["ref_start_byte"]), int(bag["ref_end_byte"])
    expected_count = int(bag["instance_count"])
    if start < 0 or end < start:
        raise ValueError("Invalid bag reference byte range.")
    refs: list[dict[str, Any]] = []
    with Path(path).open("rb") as stream:
        stream.seek(start)
        while stream.tell() < end:
            if len(refs) % 256 == 0:
                _check_deadline(deadline)
            line = stream.readline()
            if not line or stream.tell() > end:
                raise ValueError("Bag reference range ended inside an instance row.")
            refs.append(json.loads(line))
        if stream.tell() != end or len(refs) != expected_count:
            raise ValueError("Bag reference range/count mismatch.")
    return refs


def load_case_features(
    bundle: Path,
    case_id: str,
    feature_root: Path,
    *,
    max_case_bytes: int = 512 * 1024 * 1024,
    chunk_rows: int = 1024,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Load one governed case's three feature arrays by verified part/row references.

    The bundle is read as ordinary NumPy data. One case is materialized at a time;
    source feature matrices stay memory-mapped and are never copied as a cohort.
    """
    _check_deadline(deadline)
    if isinstance(max_case_bytes, bool) or not isinstance(max_case_bytes, int) or max_case_bytes < 1:
        raise ValueError("max_case_bytes must be a positive integer.")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer.")
    bundle_dir = Path(bundle)
    manifest, case_rows = _validate_bundle(bundle_dir, Path(feature_root), deadline=deadline)
    if case_id not in case_rows:
        raise KeyError(f"Case is not in this cohort bundle: {case_id}")
    case = case_rows[case_id]
    bag_rows = _read_case_bags(bundle_dir, case_id, deadline=deadline)
    refs_by_lens = {lens: _read_reference_range(bundle_dir / "instance_refs.jsonl", bag_rows[lens], deadline=deadline)
                    for lens in LENSES}
    total_vectors = sum(len(refs) for refs in refs_by_lens.values())
    required_bytes = total_vectors * FEATURE_DIM * np.dtype(np.float32).itemsize
    if required_bytes > max_case_bytes:
        raise MemoryError(
            f"Case {case_id} requires {required_bytes} bytes, above max_case_bytes={max_case_bytes}; "
            "increase the explicit per-case limit or defer loading."
        )
    for lens, refs in refs_by_lens.items():
        if bool(bag_rows[lens]["present"]) != bool(refs):
            raise ValueError(f"Lens presence flag disagrees with references for {case_id}/{lens}.")
        for ref in refs:
            if ref.get("case_id") != case_id or int(ref.get("objective_lens", -1)) != lens:
                raise ValueError("Instance reference crossed case/lens bag boundaries.")
            if ref.get("feature_id") != manifest["feature_id"]:
                raise ValueError("Instance reference uses a different feature_id.")

    feature_id = manifest["feature_id"]
    feature_dir = Path(feature_root) / feature_id
    matrices: dict[str, np.ndarray] = {}
    index_rows: dict[tuple[str, int], dict[str, Any]] = {}
    needed_by_part: dict[str, set[int]] = defaultdict(set)
    for refs in refs_by_lens.values():
        for ref in refs:
            part_id = str(ref["part_id"])
            if not _SAFE_PART_ID.fullmatch(part_id):
                raise ValueError("Unsafe feature part_id in bundle reference.")
            row_index = int(ref["row_index"])
            if row_index < 0:
                raise ValueError("Negative feature row_index in bundle reference.")
            needed_by_part[part_id].add(row_index)

    for part_id, needed_rows in needed_by_part.items():
        part_dir = feature_dir / "parts" / part_id
        verify_feature_part(part_dir, feature_id=feature_id, deadline=deadline)
        matrix_path = part_dir / "features.npy"
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise ValueError(f"Feature matrix schema changed in {part_id}.")
        if any(row >= matrix.shape[0] for row in needed_rows):
            raise ValueError(f"Feature row reference is out of range in {part_id}.")
        matrices[part_id] = matrix
        with (part_dir / "tile_index.jsonl").open(encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                if row_index % 256 == 0:
                    _check_deadline(deadline)
                if row_index in needed_rows:
                    index_rows[(part_id, row_index)] = json.loads(line)
                if len(index_rows) and all((part_id, row) in index_rows for row in needed_rows):
                    break
        if any((part_id, row) not in index_rows for row in needed_rows):
            raise ValueError(f"Feature index is missing referenced rows in {part_id}.")

    features: dict[int, np.ndarray] = {}
    tile_order: dict[int, list[str]] = {}
    for lens, refs in refs_by_lens.items():
        array = np.empty((len(refs), FEATURE_DIM), dtype=np.float32)
        tile_ids = []
        for offset, ref in enumerate(refs):
            if offset % chunk_rows == 0:
                _check_deadline(deadline)
            part_id = str(ref["part_id"])
            row_index = int(ref["row_index"])
            indexed = index_rows[(part_id, row_index)]
            for ref_key, index_key in (
                ("tile_id", "tile_id"), ("image_id", "image_id"), ("x", "x"), ("y", "y"),
                ("source_png_sha256", "source_png_sha256"), ("rgb_pixel_sha256", "rgb_pixel_sha256"),
            ):
                if ref.get(ref_key) != indexed.get(index_key):
                    raise ValueError(f"Feature reference differs from indexed row: {part_id}/{row_index}/{ref_key}")
            vector = np.asarray(matrices[part_id][row_index], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite feature vector at {part_id}/{row_index}.")
            array[offset] = vector
            tile_ids.append(str(ref["tile_id"]))
        for offset in range(0, len(array), chunk_rows):
            _check_deadline(deadline)
            if not np.isfinite(array[offset:offset + chunk_rows]).all():
                raise ValueError(f"Non-finite loaded feature chunk for {case_id}/{lens}.")
        features[lens] = array
        tile_order[lens] = tile_ids
    _check_deadline(deadline)
    return {
        "case_id": case_id,
        "patient_id": case["patient_id"],
        "case_label": int(case["case_label"]),
        "feature_id": feature_id,
        "features": features,
        "presence_mask": {lens: len(refs_by_lens[lens]) > 0 for lens in LENSES},
        "tile_order": tile_order,
        "bag_ids": {lens: bag_rows[lens]["bag_id"] for lens in LENSES},
    }
