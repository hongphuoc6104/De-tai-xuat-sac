"""Inventory approved PNG patches from ZIPs and retain row-level provenance.

This module does not recut, filter, recolor, resize, or assign patch labels.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .catalog import safe_member
from .io import atomic_json, file_hash, fingerprint

SCHEMA_VERSION = 1
IMPORT_VERSION = "precut-index-v1"
COORDINATE_NAME = re.compile(r"^(\d+)_(\d+)(?:\((\d+)\))?\.png$", re.IGNORECASE)
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_LENSES = {4, 10, 40}
REQUIRED_METADATA = {"Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade", "Ket_Luan", "Ten_Slide"}


def _metadata_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Map image stems to metadata context without promoting weak labels."""
    source = Path(path)
    frame = (pd.read_csv(source, dtype=str, keep_default_na=False)
             if source.suffix.lower() == ".csv"
             else pd.read_excel(source, dtype=str, keep_default_na=False))
    missing = REQUIRED_METADATA - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing required fields: {sorted(missing)}")
    names = frame["Ten_File"].astype(str).str.strip()
    if names.duplicated().any():
        raise ValueError("Metadata has duplicate Ten_File values.")
    rows: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        filename = str(record["Ten_File"]).strip()
        if not filename or PurePosixPath(filename).name != filename:
            raise ValueError(f"Metadata Ten_File must be a basename: {filename!r}")
        match = re.fullmatch(r"(4|10|40)[xX×]?", str(record["Do_Phong_Dai"]).strip())
        if not match:
            raise ValueError(f"Unrecognized objective lens for {filename!r}")
        year = re.sub(r"[\s-]+", "", str(record["Ma_Nam"])).upper()
        number = str(record["Ma_So"]).strip()
        if not year or not number:
            raise ValueError(f"Missing case identifier for {filename!r}")
        conclusion = str(record["Ket_Luan"]).strip().upper()
        if "CARCIN" in conclusion:
            candidate_label: int | None = 1
        elif "TĂNG SẢN" in conclusion or "HYPERPLASIA" in conclusion:
            candidate_label = 0
        else:
            candidate_label = None
        rows[Path(filename).stem] = {
            "metadata_filename": filename,
            "candidate_case_id": f"{year}_{number}",
            "objective_lens": int(match.group(1)),
            "raw_glade": str(record["Glade"]),
            "candidate_label_from_conclusion": candidate_label,
            "label_status": "candidate_unreviewed_case_label",
            "patient_id": None,
            "slide_group_id": str(record["Ten_Slide"]),
        }
    return rows


def _archive_entries(path: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read central-directory records only; do not decompress patch payloads."""
    path = Path(path).resolve()
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError(f"Source is not a readable ZIP archive: {path}")
    stat = path.stat()
    records: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    ignored_non_png = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            safe_member(info.filename.rstrip("/"))
            if info.is_dir():
                continue
            member = safe_member(info.filename)
            if member in seen_members:
                raise ValueError(f"Duplicate ZIP member: {path.name}:{member}")
            seen_members.add(member)
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and (mode & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink is not allowed: {path.name}:{member}")
            if info.flag_bits & 1:
                raise ValueError(f"Encrypted ZIP member is not supported: {path.name}:{member}")
            if PurePosixPath(member).suffix.lower() != ".png":
                ignored_non_png += 1
                continue
            parts = PurePosixPath(member).parts
            if len(parts) < 2:
                raise ValueError(f"Patch path must include an image folder: {member!r}")
            image_id = parts[-2]
            match = COORDINATE_NAME.fullmatch(parts[-1])
            if not match:
                raise ValueError(f"Patch filename must encode x_y.png coordinates: {member!r}")
            x, y = int(match.group(1)), int(match.group(2))
            context = metadata.get(image_id)
            records.append({
                "source_archive": path.name, "source_member": member, "image_id": image_id,
                "x": x, "y": y, "coordinate_variant": int(match.group(3) or 0),
                "coordinate_collision": False,
                "byte_size": info.file_size, "zip_crc32": f"{info.CRC:08x}",
                "metadata_match": context is not None, **(context or {}),
            })
    records.sort(key=lambda row: (
        row.get("objective_lens", 999), row.get("candidate_case_id", ""),
        row["image_id"], row["y"], row["x"], row["source_member"],
    ))
    central_digest = fingerprint(sorted(
        (row["source_member"], row["byte_size"], row["zip_crc32"]) for row in records
    ))
    return {
        "archive_name": path.name, "archive_path": str(path), "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns, "central_directory_sha256": central_digest,
        "png_members": len(records), "ignored_non_png_members": ignored_non_png, "records": records,
    }


def build_inventory(metadata_path: Path, sources: list[Path],
                    lenses: list[int] | None = None) -> dict[str, Any]:
    """Create a lightweight, deterministic ZIP/member inventory."""
    if not sources:
        raise ValueError("At least one source ZIP is required.")
    selected_lenses = sorted(set(lenses or [4, 10, 40]))
    if not selected_lenses or any(isinstance(lens, bool) or lens not in ALLOWED_LENSES
                                  for lens in selected_lenses):
        raise ValueError("Select one or more supported lenses: 4, 10, 40.")
    metadata = _metadata_rows(Path(metadata_path))
    archives = [_archive_entries(Path(source), metadata) for source in sources]
    names = [archive["archive_name"] for archive in archives]
    if len(names) != len(set(names)):
        raise ValueError("Source ZIP basenames must be unique.")
    seen_paths: set[str] = set()
    coordinate_rows: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    unmatched: list[dict[str, str]] = []
    coverage = {str(lens): 0 for lens in selected_lenses}
    labels_by_case: dict[str, set[int | None]] = defaultdict(set)
    for archive in archives:
        for row in archive["records"]:
            if row["source_member"] in seen_paths:
                raise ValueError(f"Patch member path occurs in multiple ZIPs: {row['source_member']}")
            seen_paths.add(row["source_member"])
            coordinate = (row["image_id"], row["x"], row["y"])
            coordinate_rows[coordinate].append({
                "archive": archive["archive_name"], "member": row["source_member"],
            })
            if not row["metadata_match"]:
                unmatched.append({"archive": archive["archive_name"], "member": row["source_member"]})
                continue
            lens = row["objective_lens"]
            if lens in selected_lenses:
                coverage[str(lens)] += 1
                labels_by_case[row["candidate_case_id"]].add(row["candidate_label_from_conclusion"])
    conflicts = sorted(case for case, labels in labels_by_case.items() if len(labels) > 1)
    collisions = [
        {"image_id": key[0], "x": key[1], "y": key[2], "members": members}
        for key, members in sorted(coordinate_rows.items()) if len(members) > 1
    ]
    colliding_members = {entry["member"] for collision in collisions for entry in collision["members"]}
    for archive in archives:
        for row in archive["records"]:
            row["coordinate_collision"] = row["source_member"] in colliding_members
    body = {
        "schema_version": SCHEMA_VERSION, "import_version": IMPORT_VERSION,
        "mode": "full_inventory_only", "metadata_name": Path(metadata_path).name,
        "metadata_sha256": file_hash(Path(metadata_path)), "lenses": selected_lenses,
        "archives": archives, "coverage": coverage,
        "unmatched_png_count": len(unmatched), "unmatched_pngs": unmatched,
        "coordinate_collision_count": len(collisions), "coordinate_collisions": collisions,
        "conflicting_candidate_case_label_count": len(conflicts),
        "conflicting_candidate_case_ids": conflicts, "training_ready": False,
        "training_blockers": ["patient_identity_unverified", "case_labels_not_reviewed", "features_not_created"]
        + (["coordinate_collisions_require_review"] if collisions else []),
    }
    body["inventory_id"] = fingerprint(body)
    return body


def write_inventory(path: Path, inventory: dict[str, Any]) -> Path:
    """Atomically persist inventory JSON."""
    destination = Path(path)
    atomic_json(destination, inventory)
    return destination


def load_inventory(path: Path) -> dict[str, Any]:
    """Load inventory and reject edited or unsupported manifests."""
    with Path(path).open(encoding="utf-8") as stream:
        inventory = json.load(stream)
    inventory_id = inventory.pop("inventory_id", None)
    if inventory_id != fingerprint(inventory):
        raise ValueError("Inventory fingerprint is invalid or its contents changed.")
    inventory["inventory_id"] = inventory_id
    if inventory.get("schema_version") != SCHEMA_VERSION or not isinstance(inventory.get("archives"), list):
        raise ValueError("Unsupported precut inventory schema.")
    return inventory


def _select_smoke(inventory: dict[str, Any], per_lens: int) -> dict[str, list[dict[str, Any]]]:
    """Pick a stable label/case round-robin sample for each available lens."""
    if isinstance(per_lens, bool) or not isinstance(per_lens, int) or per_lens < 1:
        raise ValueError("per_lens must be a positive integer.")
    grouped: dict[int, dict[tuple[int | None, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for archive in inventory["archives"]:
        for row in archive["records"]:
            if row.get("metadata_match") and row["objective_lens"] in inventory["lenses"]:
                grouped[row["objective_lens"]][
                    (row["candidate_label_from_conclusion"], row["candidate_case_id"])
                ].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for lens in inventory["lenses"]:
        groups = grouped.get(lens, {})
        labels = sorted({key[0] for key in groups}, key=lambda value: (value is None, str(value)))
        per_label: dict[int | None, list[dict[str, Any]]] = {}
        for label in labels:
            cases = sorted(key for key in groups if key[0] == label)
            depth = max((len(groups[key]) for key in cases), default=0)
            per_label[label] = [
                groups[key][offset]
                for offset in range(depth) for key in cases if offset < len(groups[key])
            ]
        interleaved = [
            per_label[label][offset]
            for offset in range(max((len(rows) for rows in per_label.values()), default=0))
            for label in labels if offset < len(per_label[label])
        ]
        chosen = interleaved[:per_lens]
        if not chosen:
            raise ValueError(f"No metadata-matched patches for {lens}X in inventory.")
        selected[str(lens)] = chosen
    return selected


def _read_patch(archive_source: Path | zipfile.ZipFile, row: dict[str, Any]) -> dict[str, Any]:
    """Read, CRC-check, and fully decode one bounded PNG from an open archive."""
    owns_archive = not isinstance(archive_source, zipfile.ZipFile)
    archive = zipfile.ZipFile(archive_source) if owns_archive else archive_source
    try:
        try:
            info = archive.getinfo(row["source_member"])
        except KeyError as exc:
            raise ValueError(f"Patch missing from its ZIP: {row['source_member']}") from exc
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"Patch exceeds byte safety limit: {row['source_member']}")
        if info.file_size != row["byte_size"] or f"{info.CRC:08x}" != row["zip_crc32"]:
            raise ValueError(f"ZIP central record changed since inventory: {row['source_member']}")
        with archive.open(info) as member:
            raw = member.read(MAX_MEMBER_BYTES + 1)
    finally:
        if owns_archive:
            archive.close()
    if len(raw) != info.file_size:
        raise ValueError(f"Patch byte length does not match ZIP metadata: {row['source_member']}")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"PNG format or image-size gate failed: {row['source_member']}")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            source_mode, width, height = image.mode, *image.size
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"PNG decode failed for {row['source_member']}: {exc}") from exc
    return {
        "source_png_sha256": hashlib.sha256(raw).hexdigest(),
        "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "source_mode": source_mode, "stored_width": width, "stored_height": height,
        "stored_rgb_min": [int(value) for value in rgb.min(axis=(0, 1))],
        "stored_rgb_max": [int(value) for value in rgb.max(axis=(0, 1))],
        "tissue_fraction_estimate": float(np.mean(np.max(rgb, axis=2) < 245)),
        "tissue_fraction_method": "rgb_max_lt_245_v1_observation_only",
        "crop_geometry_status": "filename_coordinates_only_original_crop_size_unverified",
    }


def _sample_row(inventory: dict[str, Any], row: dict[str, Any], lens: str) -> dict[str, Any]:
    archive_path = Path(next(archive["archive_path"] for archive in inventory["archives"]
                             if archive["archive_name"] == row["source_archive"]))
    qc = _read_patch(archive_path, row)
    tile_id = fingerprint({
        "version": IMPORT_VERSION, "source_archive": row["source_archive"],
        "image_id": row["image_id"], "member": row["source_member"],
        "png_sha256": qc["source_png_sha256"],
        "x": row["x"], "y": row["y"],
        "coordinate_variant": row.get("coordinate_variant", 0),
        "coordinate_collision": row.get("coordinate_collision", False),
    })
    return {
        "tile_id": tile_id, "source_archive": row["source_archive"],
        "source_member": row["source_member"], "image_id": row["image_id"],
        "candidate_case_id": row["candidate_case_id"], "patient_id": None,
        "objective_lens": int(lens), "x": row["x"], "y": row["y"],
        "raw_glade": row["raw_glade"],
        "candidate_label_from_conclusion": row["candidate_label_from_conclusion"],
        "label_status": "candidate_unreviewed_case_label", **qc,
    }


def inspect_inventory(inventory: dict[str, Any], per_lens: int = 2) -> dict[str, Any]:
    """Decode a few patch payloads per lens for a bounded local smoke audit."""
    selected = _select_smoke(inventory, per_lens)
    checked = [_sample_row(inventory, row, lens)
               for lens, rows in selected.items() for row in rows]
    collisions: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for archive in inventory["archives"]:
        for row in archive["records"]:
            if row.get("coordinate_collision"):
                collisions[(row["image_id"], row["x"], row["y"])].append(row)
    collision_checks = [
        [_sample_row(inventory, row, str(row["objective_lens"])) for row in rows[:2]]
        for _, rows in sorted(collisions.items())[:10]
    ]
    hashes: dict[str, list[str]] = defaultdict(list)
    for row in checked:
        hashes[row["source_png_sha256"]].append(row["tile_id"])
    return {
        "schema_version": SCHEMA_VERSION, "inventory_id": inventory["inventory_id"],
        "mode": "local_smoke", "per_lens_limit": per_lens,
        "selected_patch_count": len(checked),
        "selected_by_lens": {lens: len(rows) for lens, rows in selected.items()},
        "checked": checked, "coordinate_collision_checks": collision_checks,
        "coordinate_collision_groups_in_inventory": len(collisions),
        "duplicate_png_sha256_groups_in_sample": [ids for ids in hashes.values() if len(ids) > 1],
        "training_ready": False,
        "training_blockers": ["smoke_sample_only", "patient_identity_unverified",
                              "case_labels_not_reviewed", "features_not_created",
                              "global_content_duplicate_audit_pending"],
    }


def _part_id(archive: dict[str, Any], inventory_id: str) -> str:
    suffix = fingerprint((inventory_id, archive["central_directory_sha256"]))[:12]
    return f"{Path(archive['archive_name']).stem}-{suffix}"


def _central_directory_digest(archive_path: Path) -> str:
    """Fingerprint the PNG member names, uncompressed sizes, and CRCs in one ZIP."""
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".png":
                continue
            member = safe_member(info.filename)
            records.append((member, info.file_size, f"{info.CRC:08x}"))
    return fingerprint(sorted(records))


def _stage_archive(archive: dict[str, Any], inventory_id: str, work_root: Path) -> Path:
    """Copy one ZIP to local scratch so patch extraction does not seek a Drive mount."""
    source = Path(archive["archive_path"])
    before = source.stat()
    if before.st_size != archive["archive_size"] or before.st_mtime_ns != archive["archive_mtime_ns"]:
        raise ValueError(f"Source ZIP changed since inventory was built: {archive['archive_name']}")
    stage_dir = Path(work_root) / inventory_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / archive["archive_name"]
    if staged.is_file():
        if (staged.stat().st_size == archive["archive_size"]
                and _central_directory_digest(staged) == archive["central_directory_sha256"]):
            return staged
        staged.unlink()
    for stale in stage_dir.glob(f".{archive['archive_name']}.*.part"):
        stale.unlink()
    free_bytes = shutil.disk_usage(stage_dir).free
    if free_bytes < archive["archive_size"]:
        raise OSError(f"Scratch space is insufficient for {archive['archive_name']}.")
    temp = stage_dir / f".{archive['archive_name']}.{uuid.uuid4().hex}.part"
    try:
        shutil.copyfile(source, temp)
        after = source.stat()
        if (after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
                or temp.stat().st_size != archive["archive_size"]):
            raise ValueError(f"Source ZIP changed while staging: {archive['archive_name']}")
        if _central_directory_digest(temp) != archive["central_directory_sha256"]:
            raise ValueError(f"Staged ZIP central directory differs from inventory: {archive['archive_name']}")
        os.replace(temp, staged)
    finally:
        temp.unlink(missing_ok=True)
    return staged


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temp.open("wb") as stream:
            for row in rows:
                line = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
                stream.write(line)
                digest.update(line)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return count, digest.hexdigest()


def verify_part(part_root: Path, commit: dict[str, Any] | None = None) -> bool:
    """Verify row-index and QC files against a part commit."""
    root = Path(part_root)
    if commit is None:
        with (root / "commit.json").open(encoding="utf-8") as stream:
            commit = json.load(stream)
    commit_body = dict(commit)
    commit_id = commit_body.pop("commit_id", None)
    if not commit.get("complete") or commit_id != fingerprint(commit_body):
        return False
    index, qc = root / "tile_index.jsonl", root / "qc.json"
    if file_hash(index) != commit.get("tile_index_sha256") or file_hash(qc) != commit.get("qc_sha256"):
        return False
    with index.open(encoding="utf-8") as stream:
        count = sum(1 for line in stream if line.strip())
    return count == commit.get("tile_count")


def import_archive(inventory: dict[str, Any], archive: dict[str, Any],
                   output_root: Path, source_for_reading: Path | None = None) -> dict[str, Any]:
    """Decode/hash one ZIP and atomically publish its index/QC/commit."""
    archive_path = Path(archive["archive_path"])
    stat = archive_path.stat()
    if stat.st_size != archive["archive_size"] or stat.st_mtime_ns != archive["archive_mtime_ns"]:
        raise ValueError(f"ZIP changed since inventory was built: {archive['archive_name']}")
    part_id = _part_id(archive, inventory["inventory_id"])
    final = Path(output_root) / inventory["inventory_id"] / part_id
    commit_path = final / "commit.json"
    if commit_path.is_file():
        with commit_path.open(encoding="utf-8") as stream:
            commit = json.load(stream)
        if (commit.get("inventory_id") == inventory["inventory_id"]
                and commit.get("archive_central_directory_sha256") == archive["central_directory_sha256"]
                and verify_part(final, commit)):
            return {"part_id": part_id, "reused": True, "commit": commit}
        raise ValueError(f"Existing part is corrupt or incompatible: {final}")
    read_path = Path(source_for_reading) if source_for_reading is not None else archive_path
    if (read_path.stat().st_size != archive["archive_size"]
            or _central_directory_digest(read_path) != archive["central_directory_sha256"]):
        raise ValueError(f"ZIP to import does not match inventory: {archive['archive_name']}")
    staging = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        index_rows: list[dict[str, Any]] = []
        with sqlite3.connect(staging / "dedup.sqlite") as connection:
            connection.execute("CREATE TABLE hashes (sha256 TEXT PRIMARY KEY, occurrences INTEGER NOT NULL)")
            with zipfile.ZipFile(read_path) as source_zip:
                for row in archive["records"]:
                    qc = _read_patch(source_zip, row)
                    tile_id = fingerprint({
                        "version": IMPORT_VERSION, "source_archive": row["source_archive"],
                        "image_id": row["image_id"], "member": row["source_member"],
                        "png_sha256": qc["source_png_sha256"],
                        "x": row["x"], "y": row["y"],
                    })
                    connection.execute(
                        "INSERT INTO hashes VALUES (?, 1) ON CONFLICT(sha256) "
                        "DO UPDATE SET occurrences = occurrences + 1",
                        (qc["source_png_sha256"],),
                    )
                    matched = row.get("metadata_match", False)
                    index_rows.append({
                        "tile_id": tile_id, "source_archive": row["source_archive"],
                        "source_member": row["source_member"], "image_id": row["image_id"],
                        "candidate_case_id": row.get("candidate_case_id"), "patient_id": None,
                        "objective_lens": row.get("objective_lens"), "x": row["x"], "y": row["y"],
                        "metadata_match": matched,
                        "coordinate_variant": row["coordinate_variant"],
                        "coordinate_collision": row["coordinate_collision"],
                        "raw_glade": row.get("raw_glade"), "slide_group_id": row.get("slide_group_id"),
                        "candidate_label_from_conclusion": row.get("candidate_label_from_conclusion"),
                        "label_status": "candidate_unreviewed_case_label" if matched else "unmatched_metadata",
                        "bag_id": None, **qc,
                    })
            duplicate_count = connection.execute(
                "SELECT COUNT(*) FROM hashes WHERE occurrences > 1"
            ).fetchone()[0]
        count, index_sha = _write_jsonl(staging / "tile_index.jsonl", iter(index_rows))
        atomic_json(staging / "qc.json", {
            "schema_version": SCHEMA_VERSION, "inventory_id": inventory["inventory_id"],
            "archive_name": archive["archive_name"],
            "archive_central_directory_sha256": archive["central_directory_sha256"],
            "png_member_count": archive["png_members"],
            "metadata_matched_count": sum(row.get("metadata_match", False) for row in index_rows),
            "metadata_unmatched_count": sum(not row.get("metadata_match", False) for row in index_rows),
            "exact_png_content_duplicate_groups_in_part": duplicate_count,
            "crop_geometry_status": "original_crop_size_unverified", "pixel_operations": [],
            "patch_label_policy": "candidate case metadata only; no patch target assigned",
            "training_ready": False,
        })
        (staging / "dedup.sqlite").unlink()
        commit = {
            "schema_version": SCHEMA_VERSION, "import_version": IMPORT_VERSION,
            "inventory_id": inventory["inventory_id"], "part_id": part_id,
            "archive_name": archive["archive_name"], "archive_size": archive["archive_size"],
            "archive_mtime_ns": archive["archive_mtime_ns"],
            "archive_central_directory_sha256": archive["central_directory_sha256"],
            "archive_sha256": file_hash(read_path),
            "tile_count": count, "tile_index_sha256": index_sha,
            "qc_sha256": file_hash(staging / "qc.json"),
            "complete": True, "training_ready": False,
        }
        commit["commit_id"] = fingerprint(commit)
        atomic_json(staging / "commit.json", commit)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return {"part_id": part_id, "reused": False, "commit": commit}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def import_selected(inventory: dict[str, Any], output_root: Path, archive_names: list[str] | None = None,
                    max_new_parts: int | None = None, work_root: Path | None = None) -> dict[str, Any]:
    """Process ZIP parts independently; reruns verify commits and advance."""
    if max_new_parts is not None and (
        isinstance(max_new_parts, bool) or not isinstance(max_new_parts, int) or max_new_parts < 1
    ):
        raise ValueError("max_new_parts must be a positive integer.")
    archives = inventory["archives"]
    if archive_names:
        unknown = set(archive_names) - {archive["archive_name"] for archive in archives}
        if unknown:
            raise ValueError(f"Unknown ZIP archives: {sorted(unknown)}")
        archives = [archive for archive in archives if archive["archive_name"] in set(archive_names)]
    parts: list[dict[str, Any]] = []
    new_count = 0
    for archive in archives:
        original_path = Path(archive["archive_path"])
        stat = original_path.stat()
        if stat.st_size != archive["archive_size"] or stat.st_mtime_ns != archive["archive_mtime_ns"]:
            raise ValueError(f"Source ZIP changed since inventory was built: {archive['archive_name']}")
        part = Path(output_root) / inventory["inventory_id"] / _part_id(archive, inventory["inventory_id"])
        if (part / "commit.json").is_file():
            result = import_archive(inventory, archive, output_root)
        elif work_root is None:
            result = import_archive(inventory, archive, output_root)
        else:
            staged = _stage_archive(archive, inventory["inventory_id"], work_root)
            try:
                result = import_archive(inventory, archive, output_root, staged)
            finally:
                staged.unlink(missing_ok=True)
        commit = result["commit"]
        parts.append({"part_id": result["part_id"], "reused": result["reused"],
                      "tile_count": commit["tile_count"]})
        if not result["reused"]:
            new_count += 1
        if max_new_parts is not None and new_count >= max_new_parts:
            break
    return {
        "inventory_id": inventory["inventory_id"], "parts": parts, "new_parts": new_count,
        "reused_parts": sum(part["reused"] for part in parts), "training_ready": False,
    }


def audit_import(inventory: dict[str, Any], parts_root: Path, output_path: Path,
                 scratch_root: Path | None = None) -> dict[str, Any]:
    """Verify every ZIP part and scan exact pixel duplicates across the full release."""
    root = Path(parts_root) / inventory["inventory_id"]
    database_root = Path(scratch_root) if scratch_root is not None else root
    database_root.mkdir(parents=True, exist_ok=True)
    temp_db = database_root / f".precut-audit-{uuid.uuid4().hex}.sqlite"
    output: dict[str, Any]
    try:
        with sqlite3.connect(temp_db) as connection:
            connection.execute(
                "CREATE TABLE fingerprints (kind TEXT, sha256 TEXT, tile_id TEXT, "
                "archive TEXT, member TEXT, case_id TEXT, lens INTEGER)"
            )
            connection.execute("CREATE INDEX fingerprint_lookup ON fingerprints(kind, sha256)")
            verified_parts = 0
            total_rows = 0
            lens_counts: dict[str, int] = defaultdict(int)
            for archive in inventory["archives"]:
                part_id = _part_id(archive, inventory["inventory_id"])
                part_root = root / part_id
                commit_path = part_root / "commit.json"
                if not commit_path.is_file():
                    raise ValueError(f"ZIP part is not imported yet: {archive['archive_name']}")
                with commit_path.open(encoding="utf-8") as stream:
                    commit = json.load(stream)
                if (commit.get("inventory_id") != inventory["inventory_id"]
                        or commit.get("archive_name") != archive["archive_name"]
                        or not verify_part(part_root, commit)):
                    raise ValueError(f"ZIP part commit failed verification: {archive['archive_name']}")
                expected_rows = archive["png_members"]
                if expected_rows != commit.get("tile_count"):
                    raise ValueError(f"Imported row count differs from inventory: {archive['archive_name']}")
                with (part_root / "tile_index.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        expected_label_status = (
                            "candidate_unreviewed_case_label" if row.get("metadata_match") else "unmatched_metadata"
                        )
                        if (row.get("source_archive") != archive["archive_name"]
                                or row.get("patient_id") is not None
                                or row.get("label_status") != expected_label_status):
                            raise ValueError(f"Patch provenance/label gate failed in {archive['archive_name']}")
                        for kind in ("source_png_sha256", "rgb_pixel_sha256"):
                            digest = row.get(kind)
                            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                                raise ValueError(f"Missing valid {kind} in {archive['archive_name']}")
                            connection.execute(
                                "INSERT INTO fingerprints VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (kind, digest, row["tile_id"], row["source_archive"], row["source_member"],
                                 row["candidate_case_id"], row["objective_lens"]),
                            )
                        if row.get("objective_lens") in ALLOWED_LENSES:
                            lens_counts[str(row["objective_lens"])] += 1
                        else:
                            lens_counts["unmatched"] += 1
                        total_rows += 1
                verified_parts += 1
            connection.commit()
            duplicate_groups: list[dict[str, Any]] = []
            duplicate_summary: dict[str, int] = {}
            for kind in ("source_png_sha256", "rgb_pixel_sha256"):
                count = connection.execute(
                    "SELECT COUNT(*) FROM (SELECT sha256 FROM fingerprints WHERE kind=? "
                    "GROUP BY sha256 HAVING COUNT(*)>1)", (kind,),
                ).fetchone()[0]
                duplicate_summary[kind] = count
                samples = connection.execute(
                    "SELECT sha256, COUNT(*), GROUP_CONCAT(tile_id, ',') FROM fingerprints "
                    "WHERE kind=? GROUP BY sha256 HAVING COUNT(*)>1 ORDER BY sha256 LIMIT 50", (kind,),
                ).fetchall()
                duplicate_groups.extend({
                    "kind": kind, "sha256": row[0], "instance_count": row[1],
                    "tile_ids": row[2].split(",") if row[2] else [],
                } for row in samples)
        blockers = [
            "patient_identity_unverified", "case_labels_not_reviewed",
            "feature_vectors_not_created",
        ]
        if any(duplicate_summary.values()):
            blockers.append("global_duplicate_content_requires_review")
        if inventory.get("coordinate_collision_count", 0):
            blockers.append("duplicate_coordinates_require_review")
        if inventory.get("unmatched_png_count", 0):
            blockers.append("patches_without_metadata_match")
        if inventory.get("conflicting_candidate_case_label_count", 0):
            blockers.append("candidate_case_label_conflicts")
        output = {
            "schema_version": SCHEMA_VERSION, "inventory_id": inventory["inventory_id"],
            "verified_zip_parts": verified_parts, "expected_zip_parts": len(inventory["archives"]),
            "verified_patch_rows": total_rows, "patch_rows_by_lens": dict(sorted(lens_counts.items())),
            "exact_duplicate_groups": duplicate_summary,
            "duplicate_group_examples": duplicate_groups,
            "coordinate_collision_count": inventory.get("coordinate_collision_count", 0),
            "unmatched_png_count": inventory.get("unmatched_png_count", 0),
            "training_ready": False, "training_blockers": blockers,
        }
        atomic_json(Path(output_path), output)
        return output
    finally:
        temp_db.unlink(missing_ok=True)
