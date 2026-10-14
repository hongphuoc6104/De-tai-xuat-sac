"""Read-only reuse and attempt-scoped paths for whole-image experiments.

These helpers let a resumed execution consume a pinned, complete N0 audit and
completed feature releases without calling their writers.  New attempt logs
get a unique directory while persistent vector/benchmark roots remain stable.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import json
import math
import re
import urllib.error
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import image_classification as core, image_classification_audit as audit
from .io import file_hash, fingerprint

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RETCCL_SCHEMA_ERROR_PREFIXES = (
    "RetCCL checkpoint must contain a direct state dictionary.",
    "RetCCL checkpoint keys differ from the complete encoder-only schema ",
    "RetCCL checkpoint value ",
    "RetCCL checkpoint tensor ",
)
_RETCCL_REMOTE_NOT_READY = (
    "Official RetCCL checkpoint is not ready for download according to its HEAD metadata."
)
_SOURCE_FEATURE_FIELDS = (
    "file_name",
    "patient_id",
    "objective_lens",
    "raw_glade",
    "image_target",
    "source_archive_sha256",
    "source_archive_name",
    "source_member",
    "member_crc32",
    "member_byte_size",
)
_AUDIT_INVENTORY_FIELDS = (
    "image_id",
    "file_name",
    "image_stem",
    "patient_id",
    "objective_lens",
    "raw_glade",
    "image_target",
    "source_archive_sha256",
    "source_archive_name",
    "source_member",
    "member_crc32",
    "member_byte_size",
)


@dataclass(frozen=True)
class VerifiedAudit:
    """Read-only view of a pinned, complete original-image audit."""

    audit_root: Path
    status: dict[str, Any]
    summary: dict[str, Any]
    binding: dict[str, Any]
    representative_image_ids: tuple[str, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class CompletedFeatureRelease:
    """Fully verified matrix and source rows for one exact release identity."""

    run_root: Path
    features: Any
    rows: tuple[dict[str, Any], ...]
    release: dict[str, Any]
    status: dict[str, Any]


@dataclass(frozen=True)
class ExecutionAttemptPaths:
    """Per-attempt evidence location plus stable, shared cache roots."""

    attempt_root: Path
    n0_audit_root: Path
    benchmark_root: Path
    feature_roots: Mapping[str, Path]


class RetCCLFailureKind(str, Enum):
    """Whether an optional RetCCL branch may be marked pending."""

    REMOTE_UNAVAILABLE = "remote_unavailable"
    CHECKPOINT_SCHEMA_INCOMPATIBLE = "checkpoint_schema_incompatible"
    FATAL = "fatal"


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase SHA-256 digest.")
    return value


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular, non-symlinked file: {path}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _assert_audit_chunks_are_committed(run_root: Path, image_count: int) -> None:
    """Preflight the loader's destructive incomplete-chunk branch read-only."""
    chunk_root = run_root / "chunks"
    if chunk_root.is_symlink() or not chunk_root.is_dir():
        raise ValueError("Completed audit has no regular chunks directory.")
    expected_count = math.ceil(image_count / audit.CHUNK_SIZE)
    entries = sorted(chunk_root.iterdir())
    if len(entries) != expected_count:
        raise ValueError("Completed audit chunk count differs from full image coverage.")
    for index, folder in enumerate(entries):
        expected_name = f"chunk-{index:06d}"
        if folder.name != expected_name or folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Invalid or noncanonical audit chunk path: {folder}")
        if {path.name for path in folder.iterdir()} != {"commit.json", "records.jsonl"}:
            raise ValueError(f"Audit chunk has incomplete or unexpected entries: {folder}")
        for filename in ("commit.json", "records.jsonl"):
            _require_regular_file(folder / filename, f"audit chunk {filename}")


def _validate_inventory_identity(inventory_path: Path) -> dict[str, Any]:
    _require_regular_file(inventory_path, "image inventory")
    _require_regular_file(inventory_path.parent / "inventory.commit.json", "inventory receipt")
    inventory = core._load_inventory(inventory_path)
    ids = [row.get("image_id") for row in inventory["images"]]
    if not ids or any(not isinstance(image_id, str) for image_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Image inventory must have unique, nonempty image IDs.")
    return inventory


def _validate_source_rows(
    records: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    feature_bindings: Mapping[str, Mapping[str, Any]] | None,
    feature_rows: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    inventory_rows = {str(row["image_id"]): row for row in inventory["images"]}
    record_ids = [str(row.get("image_id", "")) for row in records]
    expected_ids = [str(row["image_id"]) for row in inventory["images"]]
    if record_ids != expected_ids:
        raise ValueError("Committed audit records do not cover inventory IDs in exact order.")

    for record in records:
        image_id = str(record["image_id"])
        source = inventory_rows[image_id]
        for field in _AUDIT_INVENTORY_FIELDS:
            if record.get(field) != source.get(field):
                raise ValueError(f"Committed audit source field {field!r} differs for {image_id}.")
        for field in ("original_width", "original_height", "rgb_width", "rgb_height"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Committed audit dimension {field!r} is invalid for {image_id}.")
        original_sha = record.get("original_image_sha256")
        if not isinstance(original_sha, str) or not _SHA256_RE.fullmatch(original_sha):
            raise ValueError(f"Committed audit image SHA-256 is invalid for {image_id}.")
        if feature_bindings is None:
            if record.get("verified_feature_row_original_image_sha256") != "":
                raise ValueError("Audit rows claim a feature binding that was not supplied.")
            continue

        bound_sha = feature_bindings[image_id]["original_image_sha256"]
        if (original_sha != bound_sha
                or record.get("verified_feature_row_original_image_sha256") != bound_sha):
            raise ValueError(f"Committed audit pixels differ from the verified feature row for {image_id}.")
        feature_row = (feature_rows or {}).get(image_id, {})
        for field in (*_SOURCE_FEATURE_FIELDS, "original_width", "original_height", "source_format"):
            if field in feature_row and record.get(field) != feature_row[field]:
                raise ValueError(f"Committed audit/feature field {field!r} differs for {image_id}.")


def _validate_records_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _require_regular_file(path, "audit records CSV")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(audit.RECORD_FIELDS):
                raise ValueError("Audit records CSV has an unexpected schema.")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Unreadable audit records CSV: {path}") from exc
    if len(rows) != len(records):
        raise ValueError("Audit records CSV coverage differs from committed chunks.")
    for csv_row, record in zip(rows, records, strict=True):
        if any(csv_row.get(field) != str(record.get(field, "")) for field in audit.RECORD_FIELDS):
            raise ValueError(f"Audit records CSV differs from committed chunk row {record['image_id']}.")


def _validate_selection_csv(path: Path, selected: Sequence[Mapping[str, Any]]) -> None:
    _require_regular_file(path, "representative selection CSV")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(audit.SELECTION_FIELDS):
                raise ValueError("Representative selection CSV has an unexpected schema.")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Unreadable representative selection CSV: {path}") from exc
    if len(rows) != len(selected):
        raise ValueError("Representative selection coverage differs from the pinned audit.")
    for order, (csv_row, source) in enumerate(zip(rows, selected, strict=True), start=1):
        expected = {"selection_order": str(order), **{key: str(source[key]) for key in audit.SELECTION_FIELDS
                                                      if key != "selection_order"}}
        if csv_row != expected:
            raise ValueError("Representative selection CSV differs from deterministic selection.")


def load_verified_audit(
    inventory_path: Path,
    audit_root: Path,
    *,
    expected_audit_id: str,
    expected_commit_id: str,
    expected_commit_sha256: str,
    verified_feature_rows: Sequence[Mapping[str, Any]] | None = None,
    selection_limit: int = 48,
) -> VerifiedAudit:
    """Load a completed audit without invoking its writer or changing files.

    The caller must pin the audit and final commit IDs/hashes from its trusted
    checkpoint. Feature rows are required when the audit binding records them.
    Incomplete chunks are rejected before the existing chunk loader is called,
    because that writer-oriented loader removes uncommitted chunk folders.
    """
    _require_sha256(expected_audit_id, "expected_audit_id")
    _require_sha256(expected_commit_id, "expected_commit_id")
    _require_sha256(expected_commit_sha256, "expected_commit_sha256")
    if isinstance(selection_limit, bool) or not isinstance(selection_limit, int) or selection_limit < 1:
        raise ValueError("selection_limit must be a positive integer.")

    inventory_file = Path(inventory_path).expanduser()
    inventory = _validate_inventory_identity(inventory_file)
    inventory_sha256 = file_hash(inventory_file)
    root = Path(audit_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Audit root must be a regular directory: {root}")
    if root.name != expected_audit_id:
        raise ValueError("Audit root path does not match the pinned audit ID.")
    if (root / "audit.lock").exists() or (root / "audit.lock").is_symlink():
        raise ValueError("Audit root has a writer lock; refusing a concurrent read.")

    manifest = _read_json_object(root / "audit.json", "audit manifest")
    binding = _read_json_object(root / "binding.json", "audit binding")
    status = _read_json_object(root / "status.json", "audit status")
    summary = _read_json_object(root / "summary.json", "audit summary")
    expected_ids = [str(row["image_id"]) for row in inventory["images"]]
    source_identity = audit._source_identity(inventory)
    descriptor = manifest.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError("Audit manifest has no descriptor object.")
    if (manifest.get("audit_id") != expected_audit_id
            or manifest.get("inventory_id") != inventory["inventory_id"]
            or manifest.get("image_count") != len(expected_ids)
            or manifest.get("image_ids") != expected_ids
            or fingerprint(descriptor) != expected_audit_id
            or descriptor.get("schema_version") != audit.AUDIT_SCHEMA_VERSION
            or descriptor.get("protocol") != audit.AUDIT_PROTOCOL
            or descriptor.get("inventory_id") != inventory["inventory_id"]
            or descriptor.get("source_identity") != source_identity
            or descriptor.get("chunk_size") != audit.CHUNK_SIZE):
        raise ValueError("Audit manifest differs from the pinned inventory or audit identity.")

    feature_bindings, feature_rows_sha256 = audit._verified_feature_bindings(
        verified_feature_rows, inventory,
    )
    if descriptor.get("verified_feature_rows_sha256") != feature_rows_sha256:
        raise ValueError("Audit feature-row identity differs from the supplied verified rows.")
    feature_rows_by_id = None
    if verified_feature_rows is not None:
        feature_rows_by_id = {str(row["image_id"]): row for row in verified_feature_rows}
    expected_binding = audit._make_binding(
        expected_audit_id, inventory, descriptor, source_identity, feature_rows_sha256,
    )
    if binding != expected_binding:
        raise ValueError("Audit binding does not match the pinned inventory and feature rows.")

    _assert_audit_chunks_are_committed(root, len(expected_ids))
    chunks, chunk_commit_ids = audit._load_committed_chunks(
        root,
        audit_id=expected_audit_id,
        inventory_id=inventory["inventory_id"],
        source_identity=source_identity,
        expected_ids=expected_ids,
    )
    records = [row for index in sorted(chunks) for row in chunks[index]]
    if len(records) != len(expected_ids) or len(chunk_commit_ids) != math.ceil(len(expected_ids) / audit.CHUNK_SIZE):
        raise ValueError("Committed audit chunks do not cover every inventory image.")
    _validate_source_rows(records, inventory, feature_bindings, feature_rows_by_id)
    _validate_records_csv(root / "records.csv", records)

    selected = audit.select_representative_images(records, limit=selection_limit)
    _validate_selection_csv(root / "representative_selection.csv", selected)
    final_commit_path = root / "audit.commit.json"
    _require_regular_file(final_commit_path, "final audit commit")
    actual_commit_sha256 = file_hash(final_commit_path)
    if actual_commit_sha256 != expected_commit_sha256:
        raise ValueError("Final audit commit SHA-256 differs from the pinned checkpoint.")
    final_commit = _read_json_object(final_commit_path, "final audit commit")
    if final_commit.get("commit_id") != expected_commit_id:
        raise ValueError("Final audit commit ID differs from the pinned checkpoint.")
    verified_root = audit._verify_final_commit(
        root,
        audit_id=expected_audit_id,
        source_identity=source_identity,
        binding_id=expected_binding["binding_id"],
        chunk_commit_ids=chunk_commit_ids,
        image_count=len(records),
    )
    if verified_root != root:
        raise ValueError("Final audit commit is missing or incomplete.")

    if (status.get("audit_id") != expected_audit_id
            or status.get("inventory_id") != inventory["inventory_id"]
            or status.get("expected_image_count") != len(expected_ids)
            or status.get("committed_image_count") != len(expected_ids)
            or status.get("remaining_image_count") != 0
            or status.get("audit_complete") is not True
            or status.get("duplicate_analysis_complete") is not True
            or status.get("state") != summary.get("state")
            or status.get("training_ready") != summary.get("training_ready")
            or status.get("ready_for_fit") != summary.get("ready_for_fit")
            or summary.get("audit_complete") is not True
            or summary.get("image_count") != len(expected_ids)
            or summary.get("representative_selection_count") != len(selected)):
        raise ValueError("Audit status/summary do not describe the complete pinned cohort.")
    if (root / "audit.lock").exists() or (root / "audit.lock").is_symlink():
        raise ValueError("Audit acquired a writer lock while it was being verified.")
    if file_hash(final_commit_path) != expected_commit_sha256:
        raise ValueError("Final audit commit changed while it was being verified.")
    if file_hash(inventory_file) != inventory_sha256:
        raise ValueError("Image inventory changed while the audit was being verified.")

    receipt = {
        "audit_id": expected_audit_id,
        "audit_commit_id": expected_commit_id,
        "audit_commit_sha256": actual_commit_sha256,
        "inventory_id": inventory["inventory_id"],
        "binding_id": expected_binding["binding_id"],
        "image_count": len(records),
        "chunk_count": len(chunks),
        "chunk_commit_ids": {str(index): chunk_commit_ids[index] for index in sorted(chunk_commit_ids)},
        "representative_selection_count": len(selected),
        "audit_complete": True,
        "ready_for_fit": bool(status.get("ready_for_fit")),
    }
    return VerifiedAudit(
        audit_root=root,
        status=status,
        summary=summary,
        binding=expected_binding,
        representative_image_ids=tuple(str(row["image_id"]) for row in selected),
        receipt=receipt,
    )


def _validate_feature_rows(
    rows: Sequence[Mapping[str, Any]], inventory: Mapping[str, Any], expected_ids: Sequence[str],
) -> None:
    image_rows = {str(row["image_id"]): row for row in inventory["images"]}
    actual_ids = [str(row.get("image_id", "")) for row in rows]
    if actual_ids != list(expected_ids):
        raise ValueError("Completed feature rows do not cover inventory IDs in exact order.")
    for row in rows:
        image_id = str(row["image_id"])
        expected = image_rows[image_id]
        for field in _SOURCE_FEATURE_FIELDS:
            if row.get(field) != expected.get(field):
                raise ValueError(f"Completed feature source field {field!r} differs for {image_id}.")
        for field in ("original_width", "original_height", "padded_width", "padded_height"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Completed feature dimension {field!r} is invalid for {image_id}.")
        if not isinstance(row.get("source_format"), str) or not row["source_format"]:
            raise ValueError(f"Completed feature source format is invalid for {image_id}.")


def _preflight_feature_shards(run_root: Path, release_id: str) -> None:
    features_root = run_root / "features"
    if features_root.is_symlink() or not features_root.is_dir():
        raise ValueError("Completed feature release has no regular feature-shard directory.")
    entries = sorted(features_root.iterdir())
    if not entries:
        raise ValueError("Completed feature release has no committed shards.")
    for index, shard in enumerate(entries):
        if shard.name != f"part-{index:06d}" or shard.is_symlink() or not shard.is_dir():
            raise ValueError(f"Invalid or noncanonical feature shard path: {shard}")
        commit = _read_json_object(shard / "commit.json", "feature shard commit")
        files = commit.get("files")
        if not isinstance(files, dict) or set(files) != {"features.npy", "rows.jsonl", "qc.json"}:
            raise ValueError(f"Feature shard manifest has an unexpected schema: {shard}")
        if {path.name for path in shard.iterdir()} != {"commit.json", *files}:
            raise ValueError(f"Feature shard has incomplete or unexpected entries: {shard}")
        for filename in ("commit.json", *files):
            _require_regular_file(shard / filename, "feature shard file")
        if commit.get("release_id") != release_id or commit.get("complete") is not True:
            raise ValueError(f"Feature shard does not belong to completed release {release_id}.")


def select_completed_feature_release(
    inventory_path: Path,
    output_root: Path,
    encoder: core.ImageFeatureEncoder,
    config: core.FeatureExtractionConfig,
) -> CompletedFeatureRelease | None:
    """Select one exact complete release without rewriting its manifest/status.

    A missing identity path returns ``None`` so a caller can create it.  An
    existing but incomplete, tampered, symlinked, or mismatched path raises;
    callers must not silently fall back to another release or re-extract it.
    """
    inventory_file = Path(inventory_path).expanduser()
    inventory = _validate_inventory_identity(inventory_file)
    inventory_sha256 = file_hash(inventory_file)
    config.validate()
    descriptor = core._release_descriptor(inventory, encoder, config)
    release_id = core.fingerprint(descriptor)
    output = Path(output_root).expanduser()
    if output.is_symlink():
        raise ValueError("Feature output root must not be a symlink.")
    runs_root = output / "runs"
    if runs_root.is_symlink():
        raise ValueError("Feature runs directory must not be a symlink.")
    if not runs_root.exists():
        return None
    if not runs_root.is_dir():
        raise ValueError("Feature runs path is not a directory.")
    run_root = runs_root / release_id
    if run_root.is_symlink():
        raise ValueError("Matching feature release path must not be a symlink.")
    if not run_root.exists():
        return None
    if not run_root.is_dir():
        raise ValueError("Matching feature release path is not a directory.")
    if (run_root / "run.lock").exists() or (run_root / "run.lock").is_symlink():
        raise ValueError("Matching feature release still has a writer lock.")

    release = _read_json_object(run_root / "release.json", "feature release manifest")
    status = _read_json_object(run_root / "status.json", "feature release status")
    expected_ids = [str(row["image_id"]) for row in inventory["images"]]
    if (release.get("release_id") != release_id
            or release.get("inventory_id") != inventory["inventory_id"]
            or release.get("descriptor") != descriptor
            or release.get("image_count") != len(expected_ids)
            or release.get("image_ids") != expected_ids
            or release.get("smoke") is not False
            or release.get("extraction_complete") is not True
            or release.get("training_ready") is not True):
        raise ValueError("Matching feature release manifest is incomplete or has a different identity.")
    if (status.get("schema_version") != 1
            or status.get("protocol") != core.CLASSIFICATION_PROTOCOL
            or status.get("state") != "complete"
            or status.get("release_id") != release_id
            or status.get("inventory_id") != inventory["inventory_id"]
            or status.get("expected_image_count") != len(expected_ids)
            or status.get("committed_image_count") != len(expected_ids)
            or status.get("remaining_image_count") != 0
            or status.get("training_ready") is not True
            or status.get("expected_image_ids_sha256") != hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
            or status.get("committed_image_ids_sha256") != hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()):
        raise ValueError("Matching feature release status does not prove complete coverage.")

    _preflight_feature_shards(run_root, release_id)
    features, rows, loaded_release = core._load_feature_matrix(run_root)
    if loaded_release != release:
        raise ValueError("Loaded feature release differs from its pinned manifest.")
    _validate_feature_rows(rows, inventory, expected_ids)
    if (run_root / "run.lock").exists() or (run_root / "run.lock").is_symlink():
        raise ValueError("Feature release acquired a writer lock while it was being verified.")
    if file_hash(inventory_file) != inventory_sha256:
        raise ValueError("Image inventory changed while the feature release was being verified.")
    return CompletedFeatureRelease(
        run_root=run_root,
        features=features,
        rows=tuple(dict(row) for row in rows),
        release=loaded_release,
        status=status,
    )


def create_execution_attempt(
    execution_root: Path,
    cache_root: Path,
    verified_audit: VerifiedAudit,
    *,
    feature_variants: Sequence[str] = ("A1", "B1"),
    attempt_id: str | None = None,
) -> ExecutionAttemptPaths:
    """Create a unique evidence directory while returning stable cache paths.

    The verified N0 audit is only referenced in the returned layout.  Its files
    and the shared feature/benchmark cache roots are never copied or modified.
    """
    if not isinstance(verified_audit, VerifiedAudit) or not verified_audit.receipt.get("audit_complete"):
        raise ValueError("A complete VerifiedAudit is required for a new execution attempt.")
    pinned_audit_root = verified_audit.audit_root
    pinned_commit = pinned_audit_root / "audit.commit.json"
    if (pinned_audit_root.is_symlink() or not pinned_audit_root.is_dir()
            or pinned_commit.is_symlink() or not pinned_commit.is_file()
            or file_hash(pinned_commit) != verified_audit.receipt.get("audit_commit_sha256")):
        raise ValueError("Previously verified N0 audit changed before attempt creation.")
    commit_value = _read_json_object(pinned_commit, "final audit commit")
    if commit_value.get("commit_id") != verified_audit.receipt.get("audit_commit_id"):
        raise ValueError("Previously verified N0 audit commit ID changed before attempt creation.")
    variants = tuple(feature_variants)
    if (not variants or len(set(variants)) != len(variants)
            or any(not isinstance(name, str) or not _ATTEMPT_ID_RE.fullmatch(name) for name in variants)):
        raise ValueError("feature_variants must be unique path-safe names.")

    root = Path(execution_root).expanduser()
    cache = Path(cache_root).expanduser()
    if root.is_symlink() or cache.is_symlink():
        raise ValueError("Execution and cache roots must not be symlinks.")
    if cache.exists() and not cache.is_dir():
        raise ValueError("Persistent cache root must be a directory when present.")
    if attempt_id is None:
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:10]
    if not isinstance(attempt_id, str) or not _ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise ValueError("attempt_id must be a short path-safe identifier.")

    feature_roots = {name: cache / "features" / name for name in variants}
    cache_paths = (*feature_roots.values(), cache / "benchmarks")
    for path in cache_paths:
        cursor = path
        while cursor != cache and cursor != cursor.parent:
            if cursor.is_symlink():
                raise ValueError(f"Persistent cache path must not be a symlink: {cursor}")
            cursor = cursor.parent

    root.mkdir(parents=True, exist_ok=True)
    attempt_root = root / f"attempt-{attempt_id}"
    if attempt_root.exists() or attempt_root.is_symlink():
        raise FileExistsError(f"Execution attempt already exists: {attempt_root}")
    attempt_root.mkdir(mode=0o700)
    return ExecutionAttemptPaths(
        attempt_root=attempt_root,
        n0_audit_root=verified_audit.audit_root,
        benchmark_root=cache / "benchmarks",
        feature_roots=feature_roots,
    )


def classify_retccl_failure(error: Exception) -> RetCCLFailureKind:
    """Classify only remote transport and explicit checkpoint-schema failures as optional.

    Network timeouts raised by ``urlopen``/``response.read`` must be wrapped
    as ``URLError`` at that transport boundary.  Bare ``TimeoutError`` or
    ``ConnectionError`` may come from local Drive/file operations and remain
    fatal, as do local checkpoint SHA mismatches, assertions, CUDA OOM, and
    implementation errors.
    """
    if isinstance(error, (urllib.error.URLError, http.client.HTTPException)):
        return RetCCLFailureKind.REMOTE_UNAVAILABLE
    if isinstance(error, RuntimeError) and str(error) == _RETCCL_REMOTE_NOT_READY:
        return RetCCLFailureKind.REMOTE_UNAVAILABLE
    if (isinstance(error, ValueError)
            and str(error).startswith(_RETCCL_SCHEMA_ERROR_PREFIXES)):
        return RetCCLFailureKind.CHECKPOINT_SCHEMA_INCOMPATIBLE
    return RetCCLFailureKind.FATAL
