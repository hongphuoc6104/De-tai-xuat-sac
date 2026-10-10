"""Feature release, duplicate review, cohort and bundle readiness gates."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .bags import LENSES, write_bundle_tables
from .catalog import safe_member
from .feature_encoder import (
    FEATURE_DIM,
    OFFICIAL_WEIGHTS_SHA256,
    WEIGHTS_ENUM,
)
from .features import BudgetExhausted, verify_feature_release
from .governance import CASE_FIELDS, load_governance
from .io import file_hash, fingerprint
from .splits import (
    SplitError,
    create_nested_patient_splits,
    validate_split_structure,
)

DUPLICATE_REVIEW_FIELDS = (
    "kind", "sha256", "keep_tile_ids_json", "exclude_tile_ids_json", "canonical_tile_id",
    "reviewer", "reviewed_at", "evidence",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _check_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic timestamp or None.")
    if time.monotonic() >= deadline:
        raise BudgetExhausted("Pretraining readiness deadline reached; no new ready bundle was published.")


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


class ReadinessBlocked(ValueError):
    """One or more explicit pre-training gates remain unsatisfied."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _official_v1_contract(descriptor: dict[str, Any]) -> list[str]:
    encoder = descriptor.get("encoder", {})
    prep = encoder.get("preprocessing", {})
    blockers = []
    if descriptor.get("feature_dim") != FEATURE_DIM or descriptor.get("output_dtype") != "float32":
        blockers.append("feature_schema_must_be_2048_float32")
    if encoder.get("name") != "resnet50" or encoder.get("weights") != WEIGHTS_ENUM:
        blockers.append("encoder_must_be_resnet50_imagenet1k_v1")
    if encoder.get("weights_sha256") != OFFICIAL_WEIGHTS_SHA256:
        blockers.append("encoder_weights_sha256_mismatch")
    expected_preprocessing = {
        "input_color": "RGB",
        "transform": f"{WEIGHTS_ENUM}.transforms()",
        "resize_size": [256],
        "crop_size": [224],
        "crop": "center",
        "interpolation": "bilinear",
        "antialias": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    if prep != expected_preprocessing:
        blockers.append("preprocessing_contract_mismatch")
    if descriptor.get("max_patches_per_part") is not None:
        blockers.append("smoke_or_capped_features_are_not_training_input")
    return blockers


def inspect_feature_release(
    feature_root: Path,
    *,
    expected_sources: int = 25,
    expected_vectors: int = 148991,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Readiness-check a full frozen feature release and run its global audit.

    The existing feature module owns checksum, vector and index validation. Its
    `training_ready=False` means governance is still required; it is not itself a
    blocker once this module builds and verifies a governed bundle.
    """
    _check_deadline(deadline)
    root = Path(feature_root)
    blockers: list[str] = []
    if isinstance(expected_sources, bool) or not isinstance(expected_sources, int) or expected_sources < 1:
        raise ValueError("expected_sources must be a positive integer.")
    if isinstance(expected_vectors, bool) or not isinstance(expected_vectors, int) or expected_vectors < 1:
        raise ValueError("expected_vectors must be a positive integer.")
    status_path = root / "status.json"
    if not status_path.is_file():
        return {"training_ready": False, "blockers": ["feature_status_missing"], "feature_id": None}
    status = _read_json(status_path)
    feature_id = status.get("feature_id")
    if not isinstance(feature_id, str) or not _SHA256.fullmatch(feature_id):
        return {"training_ready": False, "blockers": ["feature_id_missing_or_invalid"], "feature_id": feature_id}
    release_root = root / feature_id
    descriptor_path = release_root / "feature_config.json"
    if not descriptor_path.is_file():
        return {"training_ready": False, "blockers": ["feature_config_missing"], "feature_id": feature_id}
    descriptor = _read_json(descriptor_path)
    if fingerprint(descriptor) != feature_id:
        blockers.append("feature_config_fingerprint_mismatch")
    blockers.extend(_official_v1_contract(descriptor))
    release_path = release_root / "release.json"
    if not release_path.is_file():
        return {"training_ready": False, "blockers": sorted(set(blockers + ["feature_release_missing"])),
                "feature_id": feature_id, "feature_config_sha256": file_hash(descriptor_path)}
    release = _read_json(release_path)
    if release.get("feature_id") != feature_id:
        blockers.append("release_feature_id_mismatch")
    if status.get("status") != "complete" or not status.get("feature_complete"):
        blockers.append("feature_run_not_complete")
    for field in ("source_complete", "feature_complete", "audit_complete"):
        if release.get(field) is not True:
            blockers.append(f"release_{field}_false")
    if release.get("source_complete") is not True or release.get("observed_sources") != expected_sources:
        blockers.append("feature_source_count_mismatch")
    if release.get("expected_sources") != expected_sources:
        blockers.append("feature_expected_source_count_mismatch")
    if release.get("observed_pngs") != expected_vectors or release.get("expected_pngs") != expected_vectors:
        blockers.append("feature_source_vector_count_mismatch")
    if release.get("committed_vectors") != expected_vectors:
        blockers.append("feature_committed_vector_count_mismatch")
    if descriptor.get("source_kind") == "zip" and len(release.get("parts", [])) != expected_sources:
        blockers.append("feature_part_count_mismatch")
    if descriptor.get("source_kind") not in {"zip", "directory"}:
        blockers.append("feature_source_kind_invalid")
    if blockers:
        return {
            "training_ready": False,
            "blockers": sorted(set(blockers)),
            "feature_id": feature_id,
            "feature_config_sha256": file_hash(descriptor_path),
            "release_sha256": file_hash(release_path),
            "release": release,
            "descriptor": descriptor,
        }
    try:
        audit = verify_feature_release(root, feature_id, deadline=deadline)
    except BudgetExhausted:
        raise
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        blockers.append(f"feature_global_audit_failed:{type(exc).__name__}:{exc}")
        return {
            "training_ready": False,
            "blockers": blockers,
            "feature_id": feature_id,
            "feature_config_sha256": file_hash(descriptor_path),
            "release_sha256": file_hash(release_path),
            "release": release,
            "descriptor": descriptor,
        }
    _check_deadline(deadline)
    if not audit.get("feature_complete") or not audit.get("source_complete"):
        blockers.append("feature_global_audit_incomplete")
    stored_audit_member = release.get("audit_file")
    stored_audit_sha256 = release.get("audit_sha256")
    if (not isinstance(stored_audit_member, str) or not stored_audit_member
            or not _SHA256.fullmatch(str(stored_audit_sha256 or ""))):
        blockers.append("release_audit_provenance_missing")
    else:
        try:
            safe_member(stored_audit_member)
        except ValueError:
            blockers.append("release_audit_path_invalid")
        else:
            stored_audit_path = release_root / stored_audit_member
            if (not stored_audit_path.is_file() or file_hash(stored_audit_path) != stored_audit_sha256):
                blockers.append("release_audit_provenance_checksum_mismatch")
    _check_deadline(deadline)
    return {
        "training_ready": not blockers,
        "blockers": sorted(set(blockers)),
        "feature_id": feature_id,
        "feature_config_sha256": file_hash(descriptor_path),
        "release_sha256": file_hash(release_path),
        "release": release,
        "descriptor": descriptor,
        "audit": audit,
        "audit_sha256": stored_audit_sha256,
        "release_root": release_root,
    }


def _load_duplicate_groups(
    feature_report: dict[str, Any], *, deadline: float | None = None,
) -> list[dict[str, Any]]:
    _check_deadline(deadline)
    audit = feature_report["audit"]
    release_root = Path(feature_report["release_root"])
    member = str(audit.get("duplicate_group_file", ""))
    if not member:
        raise ValueError("Feature audit does not reference duplicate groups.")
    relative = Path(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe duplicate group path in feature audit.")
    path = release_root / relative
    if not path.is_file() or _bounded_file_hash(path, deadline) != audit.get("duplicate_group_file_sha256"):
        raise ValueError("Duplicate group audit checksum mismatch.")
    groups = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number % 256 == 0:
                _check_deadline(deadline)
            group = json.loads(line)
            if group.get("kind") not in {"source_png_sha256", "rgb_pixel_sha256"}:
                raise ValueError(f"Unknown duplicate hash kind on line {line_number}.")
            if not _SHA256.fullmatch(str(group.get("sha256", ""))):
                raise ValueError(f"Invalid duplicate hash on line {line_number}.")
            members = group.get("members")
            if not isinstance(members, list) or len(members) < 2:
                raise ValueError(f"Malformed duplicate group on line {line_number}.")
            tile_ids = [str(row.get("tile_id", "")) for row in members]
            if not all(tile_ids) or len(tile_ids) != len(set(tile_ids)):
                raise ValueError(f"Duplicate tile IDs inside duplicate group on line {line_number}.")
            groups.append({**group, "group_key": (str(group["kind"]), str(group["sha256"]))})
    _check_deadline(deadline)
    return groups


def _load_duplicate_dispositions(
    path: Path | None,
    duplicate_groups: list[dict[str, Any]],
    patient_for_case: dict[str, str],
    *,
    deadline: float | None = None,
) -> tuple[set[str], list[list[str]], dict[str, Any]]:
    """Require a provenance-bearing decision for every exact duplicate group."""
    if not duplicate_groups and path is None:
        return set(), [], {"reviewed_group_count": 0, "disposition_sha256": None}
    if path is None:
        raise ReadinessBlocked("Exact duplicate groups require duplicate_dispositions.csv review.")
    _check_deadline(deadline)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DUPLICATE_REVIEW_FIELDS:
            raise ReadinessBlocked(f"duplicate_dispositions.csv columns must exactly match: {DUPLICATE_REVIEW_FIELDS}")
        rows = []
        for row_number, row in enumerate(reader, start=1):
            if row_number % 256 == 0:
                _check_deadline(deadline)
            rows.append(row)
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("kind", "")).strip(), str(row.get("sha256", "")).strip().lower())
        if key in by_key:
            raise ReadinessBlocked(f"Duplicate disposition row: {key[0]}/{key[1]}")
        by_key[key] = row
    group_keys = {group["group_key"] for group in duplicate_groups}
    if set(by_key) != group_keys:
        missing = sorted(group_keys - set(by_key))
        extra = sorted(set(by_key) - group_keys)
        details = []
        if missing:
            details.append(f"unreviewed_groups={len(missing)}")
        if extra:
            details.append(f"unknown_groups={len(extra)}")
        raise ReadinessBlocked("Duplicate dispositions do not exactly cover the audited groups: " + ",".join(details))

    excluded: set[str] = set()
    for group_number, group in enumerate(duplicate_groups, start=1):
        if group_number % 256 == 0:
            _check_deadline(deadline)
        row = by_key[group["group_key"]]
        members = group["members"]
        member_ids = {str(member["tile_id"]) for member in members}
        try:
            keep = json.loads(row["keep_tile_ids_json"])
            drop = json.loads(row["exclude_tile_ids_json"])
        except json.JSONDecodeError as exc:
            raise ReadinessBlocked(f"Invalid duplicate tile list for {group['group_key']}.") from exc
        if not isinstance(keep, list) or not isinstance(drop, list):
            raise ReadinessBlocked(f"Duplicate tile lists must be JSON arrays for {group['group_key']}.")
        keep_ids = [str(value) for value in keep]
        drop_ids = [str(value) for value in drop]
        if (len(keep_ids) != len(set(keep_ids)) or len(drop_ids) != len(set(drop_ids))
                or set(keep_ids) & set(drop_ids) or set(keep_ids) | set(drop_ids) != member_ids or not keep_ids):
            raise ReadinessBlocked(f"Duplicate disposition must partition every member exactly once: {group['group_key']}.")
        canonical = str(row.get("canonical_tile_id", "")).strip()
        if canonical not in keep_ids:
            raise ReadinessBlocked(f"Canonical tile must be in the keep set: {group['group_key']}.")
        for field in ("reviewer", "reviewed_at", "evidence"):
            if not str(row.get(field, "")).strip():
                raise ReadinessBlocked(f"Duplicate disposition requires {field}: {group['group_key']}.")
        excluded.update(drop_ids)
        for member in members:
            tile_id = str(member["tile_id"])
            if tile_id not in keep_ids:
                continue
            case_id = str(member.get("candidate_case_id", ""))
            if case_id not in patient_for_case:
                raise ReadinessBlocked(f"Duplicate member has no reviewed case mapping: {case_id}.")
    links = []
    for group_number, group in enumerate(duplicate_groups, start=1):
        if group_number % 256 == 0:
            _check_deadline(deadline)
        row = by_key[group["group_key"]]
        keep_ids = {str(value) for value in json.loads(row["keep_tile_ids_json"])}
        kept_patients = {
            patient_for_case[str(member["candidate_case_id"])]
            for member in group["members"]
            if str(member["tile_id"]) in keep_ids and str(member["tile_id"]) not in excluded
            and str(member.get("candidate_case_id", "")) in patient_for_case
        }
        if len(kept_patients) > 1:
            links.append(sorted(kept_patients))
    return excluded, links, {
        "reviewed_group_count": len(duplicate_groups),
        "disposition_sha256": _bounded_file_hash(Path(path), deadline),
        "cross_patient_link_count": len(links),
    }


def _iter_feature_rows(
    feature_report: dict[str, Any], *, deadline: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream committed index rows in part/row order; vectors remain memory-mapped."""
    _check_deadline(deadline)
    feature_id = feature_report["feature_id"]
    release_root = Path(feature_report["release_root"])
    release = feature_report["release"]
    for entry in release["parts"]:
        _check_deadline(deadline)
        part_id = str(entry["part_id"])
        if not _SAFE_NAME.fullmatch(part_id):
            raise ValueError(f"Unsafe part_id in release: {part_id}")
        part_root = release_root / "parts" / part_id
        matrix = np.load(part_root / "features.npy", mmap_mode="r", allow_pickle=False)
        if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise ValueError(f"Feature matrix is not N×2048 float32 in {part_id}.")
        count = 0
        with (part_root / "tile_index.jsonl").open(encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                if row_index % 256 == 0:
                    _check_deadline(deadline)
                row = json.loads(line)
                if row.get("row_index") != row_index:
                    raise ValueError(f"Feature row order mismatch in {part_id}/{row_index}.")
                lens = row.get("objective_lens")
                if isinstance(lens, str) and lens.isdigit():
                    lens = int(lens)
                if lens not in LENSES:
                    raise ValueError(f"Invalid objective lens in {part_id}/{row_index}.")
                if not row.get("candidate_case_id") or not row.get("image_id") or not row.get("tile_id"):
                    raise ValueError(f"Feature index lacks case/image/tile identity in {part_id}/{row_index}.")
                for digest_key in ("source_png_sha256", "rgb_pixel_sha256"):
                    if not _SHA256.fullmatch(str(row.get(digest_key, ""))):
                        raise ValueError(f"Invalid {digest_key} in {part_id}/{row_index}.")
                if any(key in row for key in ("target", "patch_target", "label", "case_label")):
                    raise ValueError("Feature index contains a forbidden patch/case target.")
                yield {**row, "objective_lens": lens, "part_id": part_id,
                       "row_index": row_index, "feature_id": feature_id}
                count += 1
        if count != matrix.shape[0] or count != entry.get("tile_count"):
            raise ValueError(f"Feature matrix/index row count mismatch in {part_id}.")
    _check_deadline(deadline)


def _case_rows(governance_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in governance_rows:
        result.append({**row, "case_label": int(row["case_label"])})
    return result


def _csv_write(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = output.getvalue().encode("utf-8")
    with Path(path).open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_readiness_report(
    output_root: Path,
    body: dict[str, Any],
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    _check_deadline(deadline)
    report_body = {**body, "training_ready": False}
    report_id = fingerprint(report_body)
    report_dir = Path(output_root) / "readiness" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {**report_body, "readiness_id": report_id}
    path = report_dir / "training_readiness.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Readiness report path is immutable: {path}")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    _check_deadline(deadline)
    return {**report, "path": str(path)}


def _read_duplicate_disposition_copy(
    path: Path | None, output: Path, *, deadline: float | None = None,
) -> str:
    _check_deadline(deadline)
    if path is None:
        payload = (",".join(DUPLICATE_REVIEW_FIELDS) + "\n").encode("utf-8")
        with Path(output).open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_stream, Path(output).open("wb") as output_stream:
        while True:
            _check_deadline(deadline)
            block = input_stream.read(1024 * 1024)
            if not block:
                break
            output_stream.write(block)
            digest.update(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _check_deadline(deadline)
    return digest.hexdigest()


def _validate_duplicate_split_isolation(
    splits: dict[str, Any],
    selected_case_ids: set[str],
    duplicate_groups: list[dict[str, Any]],
    excluded_tile_ids: set[str],
    *,
    deadline: float | None = None,
) -> None:
    """Ensure each retained exact-content group stays in one side of every split."""
    for group_number, group in enumerate(duplicate_groups, start=1):
        if group_number % 256 == 0:
            _check_deadline(deadline)
        member_cases = {
            str(member.get("candidate_case_id", ""))
            for member in group["members"]
            if str(member.get("tile_id", "")) not in excluded_tile_ids
            and str(member.get("candidate_case_id", "")) in selected_case_ids
        }
        if len(member_cases) < 2:
            continue
        for outer in splits["outer"]:
            train, test = set(outer["train_cases"]), set(outer["test_cases"])
            if member_cases & train and member_cases & test:
                raise SplitError(f"Duplicate {group['group_key']} crosses outer fold {outer['fold']}.")
            for inner in outer["inner_folds"]:
                inner_train = set(inner["train_cases"])
                validation = set(inner["validation_cases"])
                if member_cases & inner_train and member_cases & validation:
                    raise SplitError(
                        f"Duplicate {group['group_key']} crosses outer {outer['fold']} inner {inner['fold']}."
                    )


def _verify_bundle_references(
    bundle_dir: Path,
    feature_report: dict[str, Any],
    case_rows: list[dict[str, Any]],
    expected_instance_count: int,
    *,
    deadline: float | None = None,
) -> int:
    """Verify bag ranges and every part/row reference using bounded SQLite storage."""
    _check_deadline(deadline)
    bundle_dir = Path(bundle_dir)
    case_by_id = {str(row["case_id"]): row for row in case_rows}
    expected_bags = {(case_id, lens) for case_id in case_by_id for lens in LENSES}
    observed_bags: set[tuple[str, int]] = set()
    refs_path = bundle_dir / "instance_refs.jsonl"
    refs_size = refs_path.stat().st_size
    expected_offset = 0
    reference_count = 0
    with tempfile.TemporaryDirectory(prefix="pretrain-reference-audit-") as temporary:
        database_path = Path(temporary) / "refs.sqlite"
        with sqlite3.connect(database_path) as db:
            db.execute(
                "CREATE TABLE refs (part_id TEXT, row_index INTEGER, tile_id TEXT UNIQUE, case_id TEXT, "
                "patient_id TEXT, image_id TEXT, lens INTEGER, x INTEGER, y INTEGER, source_member TEXT, "
                "source_png_sha256 TEXT, rgb_pixel_sha256 TEXT, verified INTEGER DEFAULT 0, "
                "PRIMARY KEY(part_id,row_index))"
            )
            with (bundle_dir / "bags.jsonl").open(encoding="utf-8") as bags_stream:
                for line_number, line in enumerate(bags_stream, start=1):
                    if line_number % 256 == 0:
                        _check_deadline(deadline)
                    bag = json.loads(line)
                    case_id = str(bag.get("case_id", ""))
                    lens = bag.get("lens")
                    if isinstance(lens, bool) or lens not in LENSES or case_id not in case_by_id:
                        raise ValueError(f"Invalid case/lens bag row {line_number}.")
                    key = (case_id, int(lens))
                    if key in observed_bags:
                        raise ValueError(f"Duplicate bag row: {case_id}/{lens}")
                    observed_bags.add(key)
                    start, end = int(bag["ref_start_byte"]), int(bag["ref_end_byte"])
                    start_line, end_line = int(bag["ref_start_line"]), int(bag["ref_end_line"])
                    count = int(bag["instance_count"])
                    if start != expected_offset or end < start or end > refs_size or end_line - start_line != count:
                        raise ValueError(f"Bag reference range is noncontiguous or invalid: {case_id}/{lens}")
                    if bool(bag.get("present")) != (count > 0):
                        raise ValueError(f"Bag presence flag disagrees with count: {case_id}/{lens}")
                    with refs_path.open("rb") as refs_stream:
                        refs_stream.seek(start)
                        seen = 0
                        while refs_stream.tell() < end:
                            if seen % 256 == 0:
                                _check_deadline(deadline)
                            raw = refs_stream.readline()
                            if not raw or refs_stream.tell() > end:
                                raise ValueError("Bag byte range ended inside an instance reference.")
                            ref = json.loads(raw)
                            if ref.get("case_id") != case_id or ref.get("objective_lens") != lens:
                                raise ValueError(f"Instance reference is in the wrong bag: {case_id}/{lens}")
                            if any(name in ref for name in ("target", "patch_target", "label", "case_label", "raw_glade")):
                                raise ValueError("Instance references cannot carry targets or raw grade fields.")
                            if ref.get("patient_id") != case_by_id[case_id]["patient_id"]:
                                raise ValueError(f"Instance patient mapping differs from cases.csv: {case_id}")
                            if ref.get("feature_id") != feature_report["feature_id"]:
                                raise ValueError("Instance reference feature_id differs from source release.")
                            digests = (str(ref.get("source_png_sha256", "")), str(ref.get("rgb_pixel_sha256", "")))
                            if any(not _SHA256.fullmatch(value) for value in digests):
                                raise ValueError("Instance reference has invalid source/RGB SHA-256.")
                            part_id, row_index = str(ref.get("part_id", "")), ref.get("row_index")
                            if (not _SAFE_NAME.fullmatch(part_id) or isinstance(row_index, bool)
                                    or not isinstance(row_index, int) or row_index < 0):
                                raise ValueError("Instance reference has unsafe part_id or row_index.")
                            db.execute(
                                "INSERT INTO refs(part_id,row_index,tile_id,case_id,patient_id,image_id,lens,x,y,"
                                "source_member,source_png_sha256,rgb_pixel_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                (part_id, row_index, str(ref["tile_id"]), case_id, str(ref["patient_id"]),
                                 str(ref["image_id"]), int(lens), int(ref["x"]), int(ref["y"]),
                                 str(ref["source_member"]), digests[0], digests[1]),
                            )
                            seen += 1
                            reference_count += 1
                        if refs_stream.tell() != end or seen != count:
                            raise ValueError(f"Bag reference count mismatch: {case_id}/{lens}")
                    expected_offset = end
            if observed_bags != expected_bags:
                raise ValueError("Bundle does not contain exactly one bag for every selected case/lens.")
            if expected_offset != refs_size or reference_count != expected_instance_count:
                raise ValueError("Bundle reference bytes/count differ from bundle metadata.")
            db.commit()
            release_root = Path(feature_report["release_root"])
            for entry in feature_report["release"]["parts"]:
                part_id = str(entry["part_id"])
                part_root = release_root / "parts" / part_id
                with (part_root / "tile_index.jsonl").open(encoding="utf-8") as index_stream:
                    for row_index, line in enumerate(index_stream):
                        if row_index % 256 == 0:
                            _check_deadline(deadline)
                        expected = db.execute(
                            "SELECT tile_id,case_id,image_id,lens,x,y,source_member,source_png_sha256,rgb_pixel_sha256 "
                            "FROM refs WHERE part_id=? AND row_index=?",
                            (part_id, row_index),
                        ).fetchone()
                        if expected is None:
                            continue
                        row = json.loads(line)
                        actual = (
                            str(row.get("tile_id")), str(row.get("candidate_case_id")), str(row.get("image_id")),
                            int(row.get("objective_lens")), int(row.get("x")), int(row.get("y")),
                            str(row.get("source_member")), str(row.get("source_png_sha256")),
                            str(row.get("rgb_pixel_sha256")),
                        )
                        if actual != expected:
                            raise ValueError(f"Bundle reference differs from source index: {part_id}/{row_index}")
                        db.execute("UPDATE refs SET verified=1 WHERE part_id=? AND row_index=?", (part_id, row_index))
                db.commit()
            verified_count = db.execute("SELECT COUNT(*) FROM refs WHERE verified=1").fetchone()[0]
            if verified_count != reference_count:
                raise ValueError("Not every bundle feature reference matched the audited source index.")
    _check_deadline(deadline)
    return reference_count


def _write_directory_immutable(
    source: Path, destination: Path, *, deadline: float | None = None,
) -> None:
    _check_deadline(deadline)
    if destination.exists():
        source_manifest = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
        destination_manifest = json.loads((destination / "bundle.json").read_text(encoding="utf-8"))
        if source_manifest != destination_manifest:
            raise ValueError(f"Immutable bundle path already contains a different bundle: {destination}")
        for name, digest in source_manifest["artifact_sha256"].items():
            if _bounded_file_hash(destination / name, deadline) != digest:
                raise ValueError(f"Existing immutable bundle artifact differs: {name}")
        if (_bounded_file_hash(destination / "training_readiness.json", deadline)
                != _bounded_file_hash(source / "training_readiness.json", deadline)):
            raise ValueError("Existing immutable bundle readiness report differs.")
        return
    _check_deadline(deadline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def build_pretrain_bundle(
    governance_path: Path,
    feature_root: Path,
    output_root: Path,
    *,
    cohort_mode: str = "common",
    duplicate_dispositions: Path | None = None,
    expected_sources: int = 25,
    expected_vectors: int = 148991,
    seed: int = 42,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Build a versioned case×lens bundle or publish explicit readiness blockers."""
    _check_deadline(deadline)
    if cohort_mode not in {"common", "all"}:
        raise ValueError("cohort_mode must be 'common' or 'all'.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    governance, raw_governance_rows = load_governance(Path(governance_path))
    governance_rows = _case_rows(raw_governance_rows)
    feature = inspect_feature_release(
        Path(feature_root), expected_sources=expected_sources, expected_vectors=expected_vectors,
        deadline=deadline,
    )
    if not feature["training_ready"]:
        report = _write_readiness_report(Path(output_root), {
            "schema_version": 1,
            "governance_id": governance["governance_id"],
            "feature_id": feature.get("feature_id"),
            "cohort_mode": cohort_mode,
            "blockers": sorted(set(feature["blockers"])),
            "expected_sources": expected_sources,
            "expected_vectors": expected_vectors,
        }, deadline=deadline)
        return {"status": "blocked", "training_ready": False,
                "blockers": report["blockers"], "readiness_path": report["path"],
                "governance_id": governance["governance_id"]}

    duplicate_groups = _load_duplicate_groups(feature, deadline=deadline)
    governance_by_case = {str(row["case_id"]): row for row in governance_rows}
    patient_for_case = {case_id: str(row["patient_id"]) for case_id, row in governance_by_case.items()}
    try:
        excluded_tile_ids, duplicate_patient_links, duplicate_meta = _load_duplicate_dispositions(
            Path(duplicate_dispositions) if duplicate_dispositions else None,
            duplicate_groups,
            patient_for_case,
            deadline=deadline,
        )
    except ReadinessBlocked as exc:
        report = _write_readiness_report(Path(output_root), {
            "schema_version": 1,
            "governance_id": governance["governance_id"],
            "feature_id": feature["feature_id"],
            "cohort_mode": cohort_mode,
            "blockers": [str(exc)],
            "duplicate_group_count": len(duplicate_groups),
        }, deadline=deadline)
        return {"status": "blocked", "training_ready": False, "blockers": report["blockers"],
                "readiness_path": report["path"], "governance_id": governance["governance_id"],
                "feature_id": feature["feature_id"]}

    raw_counts: Counter[tuple[str, int]] = Counter()
    effective_counts: Counter[tuple[str, int]] = Counter()
    feature_case_ids: set[str] = set()
    image_owner: dict[str, str] = {}
    feature_errors: set[str] = set()
    feature_total = 0
    for ref in _iter_feature_rows(feature, deadline=deadline):
        feature_total += 1
        case_id = str(ref["candidate_case_id"])
        if case_id not in governance_by_case:
            feature_errors.add(f"feature_case_missing_review:{case_id}")
            continue
        feature_case_ids.add(case_id)
        lens = int(ref["objective_lens"])
        raw_counts[(case_id, lens)] += 1
        previous_case = image_owner.setdefault(str(ref["image_id"]), case_id)
        if previous_case != case_id:
            feature_errors.add(f"image_id_maps_to_multiple_cases:{ref['image_id']}")
        if str(ref["tile_id"]) not in excluded_tile_ids:
            effective_counts[(case_id, lens)] += 1
    if feature_total != expected_vectors:
        feature_errors.add("feature_index_row_count_mismatch")
    if feature_errors:
        report = _write_readiness_report(Path(output_root), {
            "schema_version": 1,
            "governance_id": governance["governance_id"],
            "feature_id": feature["feature_id"],
            "cohort_mode": cohort_mode,
            "blockers": sorted(feature_errors),
            "feature_vector_count": feature_total,
        }, deadline=deadline)
        return {"status": "blocked", "training_ready": False, "blockers": report["blockers"],
                "readiness_path": report["path"], "governance_id": governance["governance_id"],
                "feature_id": feature["feature_id"]}

    excluded_cases: list[dict[str, Any]] = []
    selected_case_ids = set()
    for case_id in sorted(feature_case_ids):
        counts = {lens: effective_counts[(case_id, lens)] for lens in LENSES}
        if cohort_mode == "common" and any(counts[lens] == 0 for lens in LENSES):
            missing = [lens for lens in LENSES if counts[lens] == 0]
            excluded_cases.append({"case_id": case_id, "reason": "missing_required_lens",
                                   "detail": json.dumps(missing), "raw_instance_count": sum(raw_counts[(case_id, lens)] for lens in LENSES),
                                   "effective_instance_count": sum(counts.values()),
                                   "included_in_other_bundle": "all"})
            continue
        if sum(counts.values()) == 0:
            excluded_cases.append({"case_id": case_id, "reason": "all_instances_excluded_after_duplicate_review",
                                   "detail": "", "raw_instance_count": sum(raw_counts[(case_id, lens)] for lens in LENSES),
                                   "effective_instance_count": 0, "included_in_other_bundle": "no"})
            continue
        selected_case_ids.add(case_id)
    for case_id in sorted(set(governance_by_case) - feature_case_ids):
        excluded_cases.append({"case_id": case_id, "reason": "no_patch_coverage_in_approved_release",
                               "detail": "master metadata case is outside observed patch cohort",
                               "raw_instance_count": 0, "effective_instance_count": 0,
                               "included_in_other_bundle": "no"})
    if not selected_case_ids:
        blockers = ["cohort_has_no_cases_after_review_and_qc"]
        report = _write_readiness_report(Path(output_root), {
            "schema_version": 1, "governance_id": governance["governance_id"],
            "feature_id": feature["feature_id"], "cohort_mode": cohort_mode, "blockers": blockers,
        }, deadline=deadline)
        return {"status": "blocked", "training_ready": False, "blockers": blockers,
                "readiness_path": report["path"]}

    selected_cases = [governance_by_case[case_id] for case_id in sorted(selected_case_ids)]
    excluded_tile_count = len(excluded_tile_ids)
    selected_vector_count = sum(effective_counts[(case_id, lens)] for case_id in selected_case_ids for lens in LENSES)
    duplicate_links = [link for link in duplicate_patient_links
                       if sum(patient in {str(case["patient_id"]) for case in selected_cases} for patient in link) > 1]
    try:
        splits = create_nested_patient_splits(
            selected_cases,
            duplicate_patient_links=duplicate_links,
            seed=seed,
        )
        validate_split_structure(splits, selected_cases)
        _validate_duplicate_split_isolation(
            splits, selected_case_ids, duplicate_groups, excluded_tile_ids, deadline=deadline,
        )
    except SplitError as exc:
        report = _write_readiness_report(Path(output_root), {
            "schema_version": 1,
            "governance_id": governance["governance_id"],
            "feature_id": feature["feature_id"],
            "cohort_mode": cohort_mode,
            "case_count": len(selected_cases),
            "patient_count": len({case["patient_id"] for case in selected_cases}),
            "independent_split_group_count": splits.get("independent_split_group_count") if "splits" in locals() else None,
            "blockers": [f"nested_patient_split_infeasible:{exc}"],
        }, deadline=deadline)
        return {"status": "blocked", "training_ready": False,
                "blockers": report["blockers"], "readiness_path": report["path"],
                "governance_id": governance["governance_id"], "feature_id": feature["feature_id"]}

    excluded_path_fieldnames = ("case_id", "reason", "detail", "raw_instance_count",
                                "effective_instance_count", "included_in_other_bundle")
    all_instance_rows = _iter_feature_rows(feature, deadline=deadline)
    selected_rows = (
        {
            "feature_id": ref["feature_id"],
            "part_id": ref["part_id"],
            "row_index": ref["row_index"],
            "tile_id": ref["tile_id"],
            "case_id": ref["candidate_case_id"],
            "patient_id": patient_for_case[str(ref["candidate_case_id"])],
            "image_id": ref["image_id"],
            "objective_lens": ref["objective_lens"],
            "x": ref["x"],
            "y": ref["y"],
            "source_member": ref["source_member"],
            "source_png_sha256": ref["source_png_sha256"],
            "rgb_pixel_sha256": ref["rgb_pixel_sha256"],
            "stored_width": ref.get("stored_width"),
            "stored_height": ref.get("stored_height"),
        }
        for ref in all_instance_rows
        if str(ref["candidate_case_id"]) in selected_case_ids and str(ref["tile_id"]) not in excluded_tile_ids
    )
    output_root = Path(output_root)
    _check_deadline(deadline)
    bundles_root = output_root / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-build-", dir=bundles_root))
    try:
        bag_info = write_bundle_tables(
            temporary,
            selected_cases,
            selected_rows,
            feature_id=feature["feature_id"],
            cohort_mode=cohort_mode,
            deadline=deadline,
        )
        if bag_info["instance_count"] != selected_vector_count:
            raise ValueError("Written bundle reference count differs from the selected feature rows.")
        verified_reference_count = _verify_bundle_references(
            temporary, feature, selected_cases, bag_info["instance_count"], deadline=deadline,
        )
        if verified_reference_count != selected_vector_count:
            raise ValueError("A selected feature vector did not resolve to its audited source row.")
        split_payload = json.dumps(splits, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        (temporary / "splits.json").write_bytes(split_payload)
        _csv_write(temporary / "cohort_exclusions.csv", excluded_path_fieldnames, excluded_cases)
        duplicate_copy_sha = _read_duplicate_disposition_copy(
            Path(duplicate_dispositions) if duplicate_dispositions else None,
            temporary / "duplicate_dispositions.csv",
            deadline=deadline,
        )
        snapshot_dir = temporary / "governance_snapshot"
        source_governance = Path(governance_path)
        if source_governance.is_file():
            source_governance = source_governance.parent
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in ("governance.json", "cases.csv", "case_review.csv"):
            shutil.copyfile(source_governance / name, snapshot_dir / name)

        artifact_names = (
            "bags.jsonl", "instance_refs.jsonl", "cases.csv", "splits.json", "cohort_exclusions.csv",
            "duplicate_dispositions.csv", "governance_snapshot/governance.json",
            "governance_snapshot/cases.csv", "governance_snapshot/case_review.csv",
        )
        artifact_hashes = {name: _bounded_file_hash(temporary / name, deadline) for name in artifact_names}
        body = {
            "schema_version": 1,
            "cohort_mode": cohort_mode,
            "feature_id": feature["feature_id"],
            "feature_config_sha256": feature["feature_config_sha256"],
            "feature_release_sha256": feature["release_sha256"],
            "feature_audit_sha256": feature["audit_sha256"],
            "expected_sources": expected_sources,
            "expected_feature_vectors": expected_vectors,
            "observed_feature_sources": feature["release"]["observed_sources"],
            "observed_feature_vectors": feature_total,
            "governance_id": governance["governance_id"],
            "governance_case_count": governance["case_count"],
            "cohort_case_count": len(selected_cases),
            "cohort_case_ids": [case["case_id"] for case in selected_cases],
            "cohort_patient_count": len({str(case["patient_id"]) for case in selected_cases}),
            "independent_split_group_count": splits["independent_split_group_count"],
            "cohort_instance_count": bag_info["instance_count"],
            "excluded_case_count": len(excluded_cases),
            "excluded_feature_vector_count": feature_total - bag_info["instance_count"],
            "excluded_duplicate_vector_count": excluded_tile_count,
            "common_cohort_excluded_case_count": sum(row["reason"] == "missing_required_lens" for row in excluded_cases),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_review": duplicate_meta,
            "split_id": splits["split_id"],
            "seed": seed,
            "outer_folds": 3,
            "inner_folds": 2,
            "feature_encoder_training_ready_flag_ignored": feature["release"].get("training_ready") is False,
            "training_ready": True,
            "artifact_sha256": artifact_hashes,
            "duplicate_dispositions_sha256": duplicate_copy_sha,
        }
        bundle_id = fingerprint(body)
        bundle_manifest = {**body, "bundle_id": bundle_id}
        manifest_payload = json.dumps(bundle_manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        _check_deadline(deadline)
        (temporary / "bundle.json").write_bytes(manifest_payload)
        ready = {
            "schema_version": 1,
            "training_ready": True,
            "blockers": [],
            "bundle_id": bundle_id,
            "feature_id": feature["feature_id"],
            "governance_id": governance["governance_id"],
            "cohort_mode": cohort_mode,
            "case_count": len(selected_cases),
            "patient_count": len({str(case["patient_id"]) for case in selected_cases}),
            "feature_vector_count": feature_total,
            "selected_instance_count": bag_info["instance_count"],
            "excluded_case_count": len(excluded_cases),
            "excluded_feature_vector_count": feature_total - bag_info["instance_count"],
            "split_id": splits["split_id"],
            "verified_checks": [
                "full_feature_source_and_audit",
                "reviewed_case_identity_and_binary_labels",
                "feature_reference_order_and_row_alignment",
                "finite_float32_2048_vectors",
                "patient_grouped_nested_splits",
                "zero_cross_split_patient_image_png_rgb_overlap",
                "duplicate_content_reviewed",
            ],
        }
        ready_payload = json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        _check_deadline(deadline)
        (temporary / "training_readiness.json").write_bytes(ready_payload)
        destination = bundles_root / bundle_id
        _write_directory_immutable(temporary, destination, deadline=deadline)
        if temporary.exists():
            shutil.rmtree(temporary)
        return {"status": "ready", "training_ready": True, "bundle_id": bundle_id,
                "bundle_path": str(destination), "governance_id": governance["governance_id"],
                "feature_id": feature["feature_id"], "cohort_mode": cohort_mode,
                "case_count": len(selected_cases), "patient_count": len({str(case["patient_id"]) for case in selected_cases}),
                "instance_count": bag_info["instance_count"],
                "excluded_case_count": len(excluded_cases),
                "excluded_feature_vector_count": feature_total - bag_info["instance_count"]}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_pretrain_bundle(
    bundle_path: Path,
    feature_root: Path,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Verify immutable bundle contents and its full audited feature source."""
    _check_deadline(deadline)
    bundle_dir = Path(bundle_path)
    manifest = _read_json(bundle_dir / "bundle.json")
    bundle_id = manifest.get("bundle_id")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if bundle_id != fingerprint(unsigned):
        raise ValueError("Bundle manifest fingerprint mismatch.")
    if not manifest.get("training_ready"):
        raise ReadinessBlocked("Bundle is blocked and is not a training input.")
    for name, expected_sha in manifest["artifact_sha256"].items():
        _check_deadline(deadline)
        path = bundle_dir / name
        if not path.is_file() or _bounded_file_hash(path, deadline) != expected_sha:
            raise ValueError(f"Bundle artifact checksum mismatch: {name}")
    governance, governance_rows = load_governance(bundle_dir / "governance_snapshot")
    if governance["governance_id"] != manifest.get("governance_id"):
        raise ValueError("Bundle governance snapshot differs from its manifest.")
    feature = inspect_feature_release(
        feature_root,
        expected_sources=int(manifest["expected_sources"]),
        expected_vectors=int(manifest["expected_feature_vectors"]),
        deadline=deadline,
    )
    if not feature["training_ready"]:
        raise ReadinessBlocked("Feature source no longer passes readiness: " + "; ".join(feature["blockers"]))
    if feature["feature_id"] != manifest.get("feature_id"):
        raise ValueError("Feature source changed after bundle creation.")
    if feature["release_sha256"] != manifest.get("feature_release_sha256"):
        raise ValueError("Feature release fingerprint changed after bundle creation.")
    if feature["feature_config_sha256"] != manifest.get("feature_config_sha256"):
        raise ValueError("Feature configuration fingerprint changed after bundle creation.")
    if feature["audit_sha256"] != manifest.get("feature_audit_sha256"):
        raise ValueError("Feature global audit fingerprint changed after bundle creation.")
    with (bundle_dir / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CASE_FIELDS + ("cohort_mode", "lens_presence_json", "instance_counts_by_lens_json"):
            raise ValueError("Bundle case schema mismatch.")
        cases = list(reader)
    cases_int = [{**case, "case_label": int(case["case_label"])} for case in cases]
    if len(cases_int) != manifest.get("cohort_case_count"):
        raise ValueError("Bundle case count differs from its manifest.")
    splits = _read_json(bundle_dir / "splits.json")
    if splits.get("split_id") != manifest.get("split_id"):
        raise ValueError("Bundle split_id differs from its manifest.")
    validate_split_structure(splits, cases_int)
    if _bounded_file_hash(bundle_dir / "duplicate_dispositions.csv", deadline) != manifest.get("duplicate_dispositions_sha256"):
        raise ValueError("Duplicate disposition provenance changed.")
    duplicate_groups = _load_duplicate_groups(feature, deadline=deadline)
    patient_for_case = {str(row["case_id"]): str(row["patient_id"]) for row in governance_rows}
    excluded_tile_ids, _, _ = _load_duplicate_dispositions(
        bundle_dir / "duplicate_dispositions.csv", duplicate_groups, patient_for_case, deadline=deadline,
    )
    _validate_duplicate_split_isolation(
        splits, {str(case["case_id"]) for case in cases_int}, duplicate_groups, excluded_tile_ids,
        deadline=deadline,
    )
    verified_reference_count = _verify_bundle_references(
        bundle_dir, feature, cases_int, int(manifest["cohort_instance_count"]), deadline=deadline,
    )
    if verified_reference_count != int(manifest["cohort_instance_count"]):
        raise ValueError("Verified bundle references differ from the manifest count.")
    readiness = _read_json(bundle_dir / "training_readiness.json")
    if (not readiness.get("training_ready") or readiness.get("bundle_id") != bundle_id
            or readiness.get("feature_id") != manifest.get("feature_id")):
        raise ValueError("Training readiness report is not bound to this bundle.")
    _check_deadline(deadline)
    return {"status": "ready", "training_ready": True, "bundle_id": bundle_id,
            "feature_id": feature["feature_id"], "governance_id": governance["governance_id"],
            "case_count": len(cases_int), "patient_count": len({str(case["patient_id"]) for case in cases_int}),
            "instance_count": manifest["cohort_instance_count"]}
