"""Bounded feature extraction with verified parts and restartable Drive outputs.

This module performs inference with a frozen encoder. It never trains a model,
assigns patch cancer targets, changes source pixels, or manages Colab runtimes.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import socket
import sqlite3
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io import atomic_json, file_hash, fingerprint, verified_copy

FEATURE_IMPLEMENTATION = "precut-features-v1"
MAX_IMAGE_PIXELS = 25_000_000
PART_FILES = ("features.npy", "tile_index.jsonl", "qc.json")


class Encoder(Protocol):
    """Interface for inference; test encoders are supplied only through Python."""

    feature_dim: int

    @property
    def descriptor(self) -> dict[str, Any]: ...

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...

    def release_memory(self) -> None: ...


@dataclass(frozen=True)
class FeatureRunConfig:
    """Paths are runtime bindings, rather than persistent feature identities."""

    metadata: Path
    source_root: Path
    source_kind: str
    output_root: Path
    work_root: Path
    weights_dir: Path
    batch_size: int = 32
    device: str = "cuda"
    precision: str = "fp32"
    expected_archives: int | None = 25
    expected_pngs: int | None = 148991
    directory_part_size: int = 1024
    max_new_parts: int | None = None
    max_patches_per_part: int | None = None
    budget_minutes: float = 240.0
    reserve_minutes: float = 30.0
    retries: int = 3
    aliases: dict[str, Any] | None = None
    recover_lock: bool = False

    def validate(self) -> None:
        """Reject ambiguous or unsafe run settings before loading any weights."""
        if self.source_kind not in {"zip", "directory"}:
            raise ValueError("source_kind must be zip or directory.")
        for name in ("batch_size", "directory_part_size", "retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("expected_archives", "expected_pngs", "max_new_parts", "max_patches_per_part"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be null or a positive integer.")
        for name in ("budget_minutes", "reserve_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0 <= self.reserve_minutes < self.budget_minutes:
            raise ValueError("budget_minutes must exceed the nonnegative reserve_minutes.")
        if not isinstance(self.recover_lock, bool):
            raise ValueError("recover_lock must be a boolean.")
        roots = [Path(getattr(self, name)).expanduser().resolve()
                 for name in ("source_root", "output_root", "work_root", "weights_dir")]
        source, output, work, weights = roots
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and output must be separate directories without nesting.")
        if work == source or work == output or source in work.parents or output in work.parents:
            raise ValueError("work_root must be separate scratch storage.")
        if work in source.parents or work in output.parents or work == weights:
            raise ValueError("Scratch cannot contain durable sources, output or weights.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FeatureRunConfig:
        """Parse a strict JSON config; reject unknown fields and string booleans."""
        if not isinstance(values, dict):
            raise ValueError("Feature config must be a JSON object.")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown feature config fields: {sorted(unknown)}")
        result = dict(values)
        for name in ("metadata", "source_root", "output_root", "work_root", "weights_dir"):
            if not isinstance(result.get(name), (str, Path)) or not str(result[name]).strip():
                raise ValueError(f"Missing path: {name}")
            result[name] = Path(result[name]).expanduser()
        try:
            config = cls(**result)
        except TypeError as exc:
            raise ValueError(f"Invalid feature configuration: {exc}") from exc
        config.validate()
        return config


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RunWriter:
    """Single writer guard, released on normal exits including raised errors."""

    def __init__(self, root: Path, recover: bool) -> None:
        self.root = root
        self.lock = root / "extractor.lock"
        self.run_id = "extract-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.recover = recover
        self.events: Path | None = None

    def __enter__(self) -> _RunWriter:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                previous = json.loads(self.lock.read_text(encoding="utf-8"))
                if not isinstance(previous, dict):
                    raise ValueError("Invalid output lock schema.")
            except (ValueError, OSError) as exc:
                if not self.recover:
                    raise ValueError("Output is locked by an unreadable marker. Stop the old runtime and recover explicitly.") from exc
                previous = {}
            alive = True
            if previous.get("hostname") == socket.gethostname():
                try:
                    os.kill(int(previous["pid"]), 0)
                except ProcessLookupError:
                    alive = False
                else:
                    raise ValueError("Output is locked by an active process; recovery cannot replace a live writer.")
            if alive and not self.recover:
                raise ValueError("Output is locked. Stop the previous runtime, then set recover_lock=true to resume.")
            self.lock.unlink()
        descriptor = {"run_id": self.run_id, "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": _utc()}
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(descriptor, stream)
            stream.flush()
            os.fsync(stream.fileno())
        directory = self.root / "runs" / self.run_id
        directory.mkdir(parents=True)
        self.events = directory / "events.jsonl"
        return self

    def emit(self, event: str, **values: Any) -> None:
        row = {"event": event, "timestamp": _utc(), "run_id": self.run_id, **values}
        text = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if self.events is not None:
            with self.events.open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text, flush=True)

    def __exit__(self, *_: Any) -> None:
        if self.lock.is_file():
            current = json.loads(self.lock.read_text(encoding="utf-8"))
            if current.get("run_id") == self.run_id:
                self.lock.unlink()


class BudgetExhausted(RuntimeError):
    """An incomplete part is deliberately left uncommitted for the next run."""


def _deadline_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise BudgetExhausted("Configured compute budget reached; resume this uncommitted part next time.")


def _bounded_hash(path: Path, deadline: float | None = None) -> str:
    """Check deadlines between bounded reads, including resume verification."""
    if deadline is None:
        return file_hash(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            _deadline_check(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _choose_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    by_lens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_lens[row.get("objective_lens", 0)].append(row)
    result = []
    offset = 0
    while len(result) < limit:
        found = False
        for lens in sorted(by_lens):
            if offset < len(by_lens[lens]):
                result.append(by_lens[lens][offset])
                found = True
                if len(result) == limit:
                    break
        if not found:
            break
        offset += 1
    return result


def _decode(raw: bytes, row: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"PNG size/format gate failed: {row['source_member']}")
            image.verify()
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            mode = source.mode
            rgb = source.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"PNG decode failed: {row['source_member']}: {exc}") from exc
    sha = hashlib.sha256(raw).hexdigest()
    tile_id = fingerprint({"version": "precut-feature-instance-v1", "member": row["source_member"],
                           "image_id": row["image_id"], "x": row["x"], "y": row["y"], "png_sha256": sha})
    index = {key: row.get(key) for key in (
        "source_member", "read_member", "image_id", "metadata_filename", "candidate_case_id", "objective_lens", "x", "y",
        "coordinate_variant", "raw_glade", "slide_group_id", "candidate_label_from_conclusion",
        "source_mtime_ns", "alias_reconstructed",
    )}
    index.update(tile_id=tile_id, source_png_sha256=sha,
                 rgb_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), stored_width=rgb.width,
                 stored_height=rgb.height, source_mode=mode, patient_id=None, bag_id=None,
                 label_status="candidate_unreviewed_case_label", metadata_match=row.get("metadata_match", False),
                 crop_geometry_status="filename_coordinates_only_original_crop_size_unverified",
                 tissue_fraction_estimate=float(np.mean(np.max(pixels, axis=2) < 245)),
                 tissue_fraction_method="rgb_max_lt_245_v1_observation_only")
    return rgb, index


def _encode(images: list[Image.Image], encoder: Encoder, batch_size: int,
            writer: _RunWriter, deadline: float) -> tuple[np.ndarray, int]:
    from .feature_encoder import is_oom

    chunks = []
    offset = 0
    effective = batch_size
    while offset < len(images):
        _deadline_check(deadline)
        selected = images[offset:offset + effective]
        try:
            values = np.asarray(encoder.encode(selected))
        except RuntimeError as exc:
            if not is_oom(exc) or effective == 1:
                raise
            effective = max(1, effective // 2)
            values = None
        if values is None:
            # Leave the exception handler first so its traceback/tensors are released.
            encoder.release_memory()
            writer.emit("batch_reduced_after_oom", batch_size=effective)
            continue
        if values.shape != (len(selected), encoder.feature_dim) or values.dtype != np.float32:
            raise ValueError("Encoder returned wrong feature shape/dtype; expected float32 N x feature_dim.")
        if not np.isfinite(values).all():
            raise ValueError("Encoder returned NaN or infinite feature values.")
        chunks.append(values)
        offset += len(selected)
    return np.concatenate(chunks, axis=0), effective


def verify_feature_part(part_root: Path, *, feature_id: str | None = None,
                        deadline: float | None = None) -> dict[str, Any]:
    """Verify hashes, row order, uniqueness, shape and finite vector values."""
    root = Path(part_root)
    commit = json.loads((root / "commit.json").read_text(encoding="utf-8"))
    body = dict(commit)
    identity = body.pop("commit_id", None)
    if identity != fingerprint(body) or not commit.get("complete"):
        raise ValueError(f"Invalid feature commit: {root}")
    if feature_id is not None and commit.get("feature_id") != feature_id:
        raise ValueError("Feature part belongs to a different encoder/preprocess version.")
    for name in PART_FILES:
        if _bounded_hash(root / name, deadline) != commit["files"][name]:
            raise ValueError(f"Feature part checksum failed: {root / name}")
    matrix = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (commit["tile_count"], commit["feature_dim"]) or matrix.dtype != np.float32:
        raise ValueError("Committed feature matrix shape/dtype is invalid.")
    for offset in range(0, len(matrix), 1024):
        if deadline is not None:
            _deadline_check(deadline)
        if not np.isfinite(matrix[offset:offset + 1024]).all():
            raise ValueError("Committed feature matrix contains NaN/Inf.")
    count = 0
    seen = set()
    order = hashlib.sha256()
    with (root / "tile_index.jsonl").open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            if deadline is not None and count % 256 == 0:
                _deadline_check(deadline)
            row = json.loads(line)
            if row["row_index"] != count - 1 or row["tile_id"] in seen:
                raise ValueError("Index row order or tile ID uniqueness failed.")
            if row.get("patient_id") is not None or row.get("bag_id") is not None or "patch_target" in row:
                raise ValueError("Unreviewed feature index violates weak-label/identity policy.")
            seen.add(row["tile_id"])
            order.update((row["tile_id"] + "\n").encode())
    if count != commit["tile_count"] or order.hexdigest() != commit["row_order_sha256"]:
        raise ValueError("Feature matrix and index counts/order do not agree.")
    return commit


def _verify_source_on_reuse(part: dict[str, Any], directory: Path, commit: dict[str, Any], deadline: float) -> None:
    """Verify immutable input bytes without staging or running the encoder again."""
    source = Path(part["source_path"])
    if part["source_kind"] == "zip":
        if _bounded_hash(source, deadline) != commit["source_archive_sha256"]:
            raise ValueError("ZIP bytes changed after feature extraction; keep source releases immutable.")
        return
    records = {row["source_member"]: row for row in part["records"]}
    checked: dict[str, str] = {}
    with (directory / "tile_index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            old = json.loads(line)
            current = records[old["source_member"]]
            member = current["read_member"]
            if member not in checked:
                checked[member] = _bounded_hash(source / member, deadline)
            if checked[member] != old["source_png_sha256"]:
                raise ValueError(f"Directory PNG changed after feature extraction: {old['source_member']}")


def _process_part(config: FeatureRunConfig, part: dict[str, Any], encoder: Encoder,
                  feature_id: str, metadata_sha: str, writer: _RunWriter,
                  deadline: float) -> dict[str, Any]:
    from .feature_sources import read_feature_payload, stage_feature_part

    destination = Path(config.output_root) / feature_id / "parts" / part["part_id"]
    if (destination / "commit.json").is_file():
        commit = verify_feature_part(destination, feature_id=feature_id, deadline=deadline)
        if commit["source_signature"] != part["source_signature"] or commit["metadata_sha256"] != metadata_sha:
            raise ValueError("Committed part source/metadata differs; use a reviewed new output version.")
        _verify_source_on_reuse(part, destination, commit, deadline)
        return {"reused": True, "commit": commit}
    if part.get("blocked") or any(not row.get("metadata_match") for row in part["records"]):
        raise ValueError(f"Part metadata/coordinate gate blocked: {part.get('blockers', part['part_id'])}")
    rows = _choose_records(part["records"], config.max_patches_per_part)
    if not rows:
        raise ValueError("Feature part has no eligible patch records.")
    # Smoke selects records BEFORE any source staging or PNG reads.
    selected_part = {**part, "records": rows}
    Path(config.work_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="feature-output-", dir=config.work_root) as temporary:
        scratch = Path(temporary)
        matrix = np.lib.format.open_memmap(scratch / "features.npy", mode="w+", dtype=np.float32,
                                          shape=(len(rows), encoder.feature_dim))
        order = hashlib.sha256()
        coverage: dict[str, int] = defaultdict(int)
        stored_shapes: dict[str, int] = defaultdict(int)
        batch_size = config.batch_size
        done = 0
        source_mtime = Path(part["source_path"]).stat().st_mtime_ns if part["source_kind"] == "zip" else None
        started = time.monotonic()
        try:
            with stage_feature_part(selected_part, config.work_root, retries=config.retries,
                                    deadline=deadline) as staged:
                archive_sha = staged.get("source_archive_sha256")
                with (scratch / "tile_index.jsonl").open("w", encoding="utf-8") as index_file:
                    while done < len(rows):
                        _deadline_check(deadline)
                        batch_rows = rows[done:done + batch_size]
                        images: list[Image.Image] = []
                        indices = []
                        try:
                            for row in batch_rows:
                                rgb, index = _decode(read_feature_payload(staged, row), row)
                                images.append(rgb)
                                indices.append(index)
                            vectors, batch_size = _encode(images, encoder, batch_size, writer, deadline)
                            matrix[done:done + len(indices)] = vectors
                            for offset, index in enumerate(indices):
                                index.update(row_index=done + offset, source_kind=part["source_kind"],
                                             source_name=part["source_name"])
                                index_file.write(json.dumps(index, ensure_ascii=False, allow_nan=False) + "\n")
                                order.update((index["tile_id"] + "\n").encode())
                                coverage[str(index["objective_lens"])] += 1
                                stored_shapes[f"{index['stored_width']}x{index['stored_height']}"] += 1
                        finally:
                            for image in images:
                                image.close()
                        done += len(indices)
                        if done == len(rows) or done % (config.batch_size * 10) == 0:
                            writer.emit("part_progress", part_id=part["part_id"], vectors=done,
                                        total=len(rows), effective_batch_size=batch_size)
                    index_file.flush()
                    os.fsync(index_file.fileno())
        finally:
            matrix.flush()
            del matrix
        atomic_json(scratch / "qc.json", {"coverage": dict(coverage), "stored_shapes": dict(stored_shapes),
                    "tile_count": done, "pixel_operations_on_source": [], "stain_normalization": "none",
                    "crop_geometry_status": "original_crop_size_unverified",
                    "metadata_sha256": metadata_sha, "training_ready": False})
        hashes = {name: file_hash(scratch / name) for name in PART_FILES}
        commit = {"schema_version": 1, "implementation_version": FEATURE_IMPLEMENTATION,
                  "feature_id": feature_id, "part_id": part["part_id"], "source_name": part["source_name"],
                  "source_kind": part["source_kind"], "source_signature": part["source_signature"],
                  "source_archive_sha256": archive_sha, "source_mtime_ns": source_mtime,
                  "metadata_sha256": metadata_sha, "tile_count": len(rows), "source_part_patch_count": len(part["records"]),
                  "feature_dim": encoder.feature_dim, "dtype": "float32", "files": hashes,
                  "row_order_sha256": order.hexdigest(), "coverage": dict(coverage),
                  "smoke": config.max_patches_per_part is not None,
                  "complete": True, "training_ready": False, "effective_batch_size": batch_size,
                  "elapsed_seconds": round(time.monotonic() - started, 3), "completed_at": _utc()}
        commit["commit_id"] = fingerprint(commit)
        atomic_json(scratch / "commit.json", commit)
        verify_feature_part(scratch, feature_id=feature_id)
        destination.mkdir(parents=True, exist_ok=True)
        for name in PART_FILES:
            verified_copy(scratch / name, destination / name, hashes[name])
        # A commit marker is published only after all payloads are durably verified.
        verified_copy(scratch / "commit.json", destination / "commit.json", file_hash(scratch / "commit.json"))
        verify_feature_part(destination, feature_id=feature_id)
        return {"reused": False, "commit": commit}


def verify_feature_release(output_root: Path, feature_id: str | None = None, *,
                            deadline: float | None = None, release_path: Path | None = None) -> dict[str, Any]:
    """Verify all referenced parts and scan duplicate PNG/RGB hashes globally."""
    root = Path(output_root)
    if feature_id is None:
        feature_id = json.loads((root / "status.json").read_text())["feature_id"]
    if not isinstance(feature_id, str) or len(feature_id) != 64 or any(c not in "0123456789abcdef" for c in feature_id):
        raise ValueError("Invalid feature_id.")
    release_root = root / feature_id
    descriptor = json.loads((release_root / "feature_config.json").read_text(encoding="utf-8"))
    if fingerprint(descriptor) != feature_id:
        raise ValueError("Encoder/preprocess descriptor does not match feature_id.")
    release = json.loads((release_path or release_root / "release.json").read_text(encoding="utf-8"))
    plan_id = release["source_plan_id"]
    if not isinstance(plan_id, str) or len(plan_id) != 64 or any(c not in "0123456789abcdef" for c in plan_id):
        raise ValueError("Invalid source plan identity.")
    source_plan = json.loads((release_root / "source_plans" / (plan_id + ".json")).read_text(encoding="utf-8"))
    if fingerprint(source_plan) != plan_id:
        raise ValueError("Source plan snapshot fingerprint failed.")
    planned = {part["part_id"]: part for part in source_plan["parts"]}
    referenced = {part["part_id"] for part in release["parts"]}
    complete_claim = bool(release.get("feature_complete") or release.get("extraction_complete"))
    if complete_claim and (referenced != set(planned) or not source_plan["source_complete"]
                           or descriptor["max_patches_per_part"] is not None):
        raise ValueError("A complete feature release must match the full planned uncapped source.")
    total = 0
    with tempfile.TemporaryDirectory(prefix="feature-audit-") as temporary:
        with sqlite3.connect(Path(temporary) / "hashes.sqlite") as db:
            db.execute("CREATE TABLE hashes(kind TEXT,sha TEXT,tile_id TEXT,case_id TEXT,image_id TEXT,"
                       "part_id TEXT,member TEXT,row_index INTEGER,PRIMARY KEY(kind,tile_id))")
            for entry in release["parts"]:
                part = release_root / "parts" / entry["part_id"]
                commit = verify_feature_part(part, feature_id=feature_id, deadline=deadline)
                if commit["commit_id"] != entry["commit_id"]:
                    raise ValueError("Release references a different part commit.")
                if commit["part_id"] not in planned or commit["source_signature"] != planned[commit["part_id"]]["source_signature"]:
                    raise ValueError("Feature part is not present in the source plan.")
                if complete_claim and commit["tile_count"] != planned[commit["part_id"]]["png_members"]:
                    raise ValueError("Complete release has missing planned vectors.")
                total += commit["tile_count"]
                with (part / "tile_index.jsonl").open(encoding="utf-8") as stream:
                    for row_number, line in enumerate(stream):
                        if deadline is not None and row_number % 256 == 0:
                            _deadline_check(deadline)
                        row = json.loads(line)
                        for kind in ("source_png_sha256", "rgb_pixel_sha256"):
                            db.execute("INSERT INTO hashes VALUES(?,?,?,?,?,?,?,?)",
                                       (kind, row[kind], row["tile_id"], row["candidate_case_id"], row["image_id"],
                                        commit["part_id"], row["source_member"], row["row_index"]))
                db.commit()
            duplicates = {}
            db.execute("CREATE INDEX hash_groups ON hashes(kind,sha)")
            for kind in ("source_png_sha256", "rgb_pixel_sha256"):
                duplicates[kind] = db.execute(
                    "SELECT COUNT(*) FROM (SELECT sha FROM hashes WHERE kind=? GROUP BY sha HAVING COUNT(*)>1)",
                    (kind,),
                ).fetchone()[0]
            cross_case = db.execute(
                "SELECT COUNT(*) FROM (SELECT kind,sha FROM hashes GROUP BY kind,sha HAVING COUNT(DISTINCT case_id)>1)"
            ).fetchone()[0]
            duplicate_path = Path(temporary) / "duplicate_groups.jsonl"
            with duplicate_path.open("w", encoding="utf-8") as stream:
                grouped = db.execute("SELECT kind,sha,COUNT(*) FROM hashes GROUP BY kind,sha HAVING COUNT(*)>1")
                for kind, sha, count in grouped:
                    if deadline is not None:
                        _deadline_check(deadline)
                    # Stream instances; a large blank-image group must not grow RAM.
                    stream.write(json.dumps({"kind": kind, "sha256": sha, "count": count})[:-1] + ', "members": [')
                    first = True
                    for tile, case, image, part_id, member, row_index in db.execute(
                        "SELECT tile_id,case_id,image_id,part_id,member,row_index FROM hashes WHERE kind=? AND sha=? ORDER BY tile_id",
                        (kind, sha),
                    ):
                        if not first:
                            stream.write(",")
                        first = False
                        stream.write(json.dumps({"tile_id": tile, "candidate_case_id": case, "image_id": image,
                                                 "part_id": part_id, "source_member": member, "row_index": row_index}))
                    stream.write("]}\n")
                stream.flush()
                os.fsync(stream.fileno())
            duplicate_sha = _bounded_hash(duplicate_path, deadline)
            duplicate_member = f"duplicate_groups/{duplicate_sha}.jsonl"
            verified_copy(duplicate_path, release_root / duplicate_member, duplicate_sha)
            if deadline is not None:
                _deadline_check(deadline)
    if total != release["committed_vectors"]:
        raise ValueError("Release vector count does not match its parts.")
    audit = {"feature_id": feature_id, "verified_parts": len(release["parts"]), "verified_vectors": total,
             "source_plan_id": plan_id,
             "source_complete": release["source_complete"], "feature_complete": complete_claim,
             "exact_duplicate_groups": duplicates, "cross_candidate_case_duplicate_groups": cross_case,
             "duplicate_group_file": duplicate_member, "duplicate_group_file_sha256": duplicate_sha,
             "training_ready": False, "training_blockers": ["patient_identity_unverified", "case_labels_not_reviewed",
                "patient_splits_not_created", "duplicate_review_required", "MIL_not_trained"], "verified_at": _utc()}
    audit_id = fingerprint(audit)
    audit_path = release_root / "audits" / (audit_id + ".json")
    atomic_json(audit_path, audit)
    atomic_json(release_root / "dataset_audit.json", audit)
    return {**audit, "audit_file": f"audits/{audit_id}.json", "audit_sha256": file_hash(audit_path)}


def run_feature_extraction(config: FeatureRunConfig, *, encoder: Encoder | None = None) -> dict[str, Any]:
    """Process available independent parts; reruns verify commits without recoding."""
    from .feature_sources import build_source_plan

    config.validate()
    started = time.monotonic()
    deadline = started + (config.budget_minutes - config.reserve_minutes) * 60
    hard_deadline = started + config.budget_minutes * 60
    root = Path(config.output_root)
    with _RunWriter(root, config.recover_lock) as writer:
        status: dict[str, Any] = {"status": "running", "run_id": writer.run_id, "feature_id": None,
                                  "training_ready": False, "started_at": _utc()}
        resolved = {field.name: getattr(config, field.name) for field in fields(config)}
        for key, value in resolved.items():
            if isinstance(value, Path):
                resolved[key] = str(value)
        atomic_json(root / "runs" / writer.run_id / "resolved_config.json", resolved)
        atomic_json(root / "status.json", status)
        try:
            plan = build_source_plan(config.metadata, config.source_root, config.source_kind,
                                     directory_part_size=config.directory_part_size,
                                     expected_archives=config.expected_archives, expected_pngs=config.expected_pngs,
                                     aliases=config.aliases)
            writer.emit("source_inventory", observed_sources=plan["observed_sources"],
                        observed_pngs=plan["observed_pngs"], source_complete=plan["source_complete"])
            status.update(observed_pngs=plan["observed_pngs"], expected_pngs=config.expected_pngs,
                          expected_sources=plan["expected_sources"], available_parts=len(plan["parts"]),
                          source_complete=plan["source_complete"], source_errors=plan["source_errors"])
            if not plan["parts"]:
                status.update(status="error" if plan["source_errors"] else "awaiting_sources",
                              completed_parts=0, committed_vectors=0, finished_at=_utc())
                atomic_json(root / "status.json", status)
                writer.emit("run_finished", **status)
                return status
            if encoder is None:
                from .feature_encoder import ResNet50Encoder

                encoder = ResNet50Encoder(config.weights_dir, device=config.device, precision=config.precision)
            descriptor = {"implementation_version": FEATURE_IMPLEMENTATION, "encoder": encoder.descriptor,
                          "implementation_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
                              "features.py", "feature_sources.py", "feature_encoder.py",
                          )},
                          "feature_dim": encoder.feature_dim, "output_dtype": "float32",
                          "max_patches_per_part": config.max_patches_per_part,
                          "source_kind": config.source_kind,
                          "directory_part_size": config.directory_part_size if config.source_kind == "directory" else None,
                          "stain_normalization": "none", "source_patch_policy": "approved pixels retained"}
            feature_id = fingerprint(descriptor)
            status["feature_id"] = feature_id
            version_root = root / feature_id
            version_root.mkdir(parents=True, exist_ok=True)
            atomic_json(version_root / "feature_config.json", descriptor)
            compact_plan = {key: plan[key] for key in (
                "schema_version", "metadata_sha256", "source_kind", "observed_sources", "expected_sources",
                "observed_pngs", "source_complete", "coverage", "source_errors",
            )}
            compact_plan["expected_pngs"] = config.expected_pngs
            compact_plan["parts"] = [{key: part[key] for key in (
                "part_id", "source_kind", "source_name", "source_size", "source_signature",
            )} | {"png_members": len(part["records"])} for part in plan["parts"]]
            plan_id = fingerprint(compact_plan)
            atomic_json(version_root / "source_plans" / (plan_id + ".json"), compact_plan)
            atomic_json(version_root / "source_plan.json", compact_plan)
            failures = [row for row in plan["source_errors"] if row.get("status") != "incomplete_upload"]
            available: dict[str, dict[str, Any]] = {}
            failed_ids: set[str] = set()
            new = 0
            reused = 0
            durations = []
            stopping = None
            if config.max_patches_per_part is not None:
                selection_path = version_root / "smoke_selection.json"
                if selection_path.is_file():
                    selected_ids = json.loads(selection_path.read_text())["part_ids"]
                else:
                    selected_ids = [row["part_id"] for row in plan["parts"][:config.max_new_parts or 1]]
                    atomic_json(selection_path, {"part_ids": selected_ids})
                active_parts = [part for part in plan["parts"] if part["part_id"] in selected_ids]
            else:
                active_parts = plan["parts"]
            # Audit every existing active commit before any new work. A budget/part
            # limit must not drop earlier successful parts from the release index.
            for part in active_parts:
                if (version_root / "parts" / part["part_id"] / "commit.json").is_file():
                    try:
                        previous = _process_part(config, part, encoder, feature_id, plan["metadata_sha256"], writer, deadline)
                        available[part["part_id"]] = previous["commit"]
                        reused += 1
                    except BudgetExhausted:
                        stopping = "budget_exhausted"
                        break
                    except (OSError, ValueError, RuntimeError) as exc:
                        failures.append({"part_id": part["part_id"], "source_name": part["source_name"],
                                         "error_type": type(exc).__name__, "message": str(exc)})
                        failed_ids.add(part["part_id"])
                        writer.emit("existing_part_failed", part_id=part["part_id"], message=str(exc))
            for part in ([] if stopping else active_parts):
                if part["part_id"] in failed_ids or part["part_id"] in available:
                    continue
                estimate = max(durations, default=300.0)
                if time.monotonic() + estimate >= deadline:
                    stopping = "budget_exhausted"
                    writer.emit("budget_stop_before_part", part_id=part["part_id"], estimated_seconds=estimate)
                    break
                if config.max_new_parts is not None and new >= config.max_new_parts:
                    stopping = "part_limit_reached"
                    break
                tick = time.monotonic()
                writer.emit("part_started", part_id=part["part_id"], source_name=part["source_name"])
                try:
                    result = _process_part(config, part, encoder, feature_id, plan["metadata_sha256"], writer, deadline)
                except (BudgetExhausted, TimeoutError):
                    stopping = "budget_exhausted"
                    break
                except (OSError, ValueError, RuntimeError) as exc:
                    failure = {"part_id": part["part_id"], "source_name": part["source_name"],
                               "error_type": type(exc).__name__, "message": str(exc)}
                    failures.append(failure)
                    writer.emit("part_failed", **failure, traceback=traceback.format_exc(limit=8))
                    continue
                available[part["part_id"]] = result["commit"]
                if not result["reused"]:
                    new += 1
                    durations.append(time.monotonic() - tick)
                status.update(completed_parts=len(available), committed_vectors=sum(c["tile_count"] for c in available.values()),
                              new_parts=new, reused_parts=reused, last_part=part["part_id"], updated_at=_utc())
                atomic_json(root / "status.json", status)
                writer.emit("part_complete", part_id=part["part_id"], reused=result["reused"],
                            vectors=result["commit"]["tile_count"])
            commits = [available[part["part_id"]] for part in active_parts if part["part_id"] in available]
            full = (plan["source_complete"] and len(commits) == len(plan["parts"]) and not failures
                    and config.max_patches_per_part is None)
            release = {"schema_version": 1, "feature_id": feature_id, "metadata_sha256": plan["metadata_sha256"],
                       "source_plan_id": plan_id,
                       "parts": commits, "committed_vectors": sum(c["tile_count"] for c in commits),
                       "expected_sources": plan["expected_sources"], "expected_pngs": config.expected_pngs,
                       "observed_sources": plan["observed_sources"], "observed_pngs": plan["observed_pngs"],
                       "source_complete": plan["source_complete"], "feature_complete": False,
                       "extraction_complete": full, "audit_complete": False,
                       "source_errors": failures, "training_ready": False, "updated_at": _utc()}
            release_path = version_root / "release.json"
            existing_complete = release_path.is_file() and json.loads(release_path.read_text()).get("feature_complete")
            snapshot_path = root / "runs" / writer.run_id / "release_snapshot.json"
            atomic_json(snapshot_path, release)
            if full:
                status.update(status="auditing", feature_complete=False)
                atomic_json(root / "status.json", status)
                writer.emit("release_audit_started")
                try:
                    audit = verify_feature_release(root, feature_id, deadline=hard_deadline, release_path=snapshot_path)
                except BudgetExhausted:
                    full = False
                    stopping = "budget_exhausted"
                else:
                    release.update(feature_complete=True, audit_complete=True,
                                   audit_file=audit["audit_file"], audit_sha256=audit["audit_sha256"])
                    status["audit_path"] = str(version_root / "dataset_audit.json")
                    status["exact_duplicate_groups"] = audit["exact_duplicate_groups"]
            if existing_complete and not full:
                release_path = version_root / "partial_release.json"
            atomic_json(snapshot_path, release)
            atomic_json(release_path, release)
            final = "error" if failures else stopping or (
                "smoke_complete" if config.max_patches_per_part is not None and commits else "complete" if full else "awaiting_sources")
            status.update(status=final, completed_parts=len(commits), committed_vectors=release["committed_vectors"],
                          new_parts=new, reused_parts=reused, feature_complete=full, failures=failures,
                          elapsed_seconds=round(time.monotonic() - started, 3), finished_at=_utc(),
                          report_path=str(release_path))
            atomic_json(root / "status.json", status)
            writer.emit("run_finished", **status)
            return status
        except KeyboardInterrupt:
            status.update(status="interrupted", finished_at=_utc())
            atomic_json(root / "status.json", status)
            writer.emit("run_interrupted")
            raise
        except Exception as exc:
            status.update(status="error", error_type=type(exc).__name__, message=str(exc), finished_at=_utc())
            atomic_json(root / "status.json", status)
            writer.emit("run_failed", error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(limit=8))
            raise
