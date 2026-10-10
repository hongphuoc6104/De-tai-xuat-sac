"""Plan and stage bounded raw PNG sources for resumable feature extraction.

Inventory reads metadata and filesystem/ZIP structure only. Payload bytes are
read later, one bounded PNG at a time, from the staged part.
"""
from __future__ import annotations

import binascii
import errno
import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
import zlib
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .catalog import safe_member
from .io import file_hash, fingerprint
from .precut import COORDINATE_NAME, _metadata_rows

MAX_PNG_BYTES = 64 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
FREE_DISK_RESERVE_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_TRANSIENT_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
    errno.EIO,
    errno.ETIMEDOUT,
    getattr(errno, "ESTALE", errno.EIO),
}


def _checked_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None.")
    return value


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Feature part staging deadline exceeded.")


def _source_error(source_errors: list[dict[str, str]], source_name: str,
                  message: str, member: str | None = None) -> None:
    entry = {"source_name": source_name, "error": message}
    if member is not None:
        entry["source_member"] = member
    source_errors.append(entry)


def _archive_structure(archive: zipfile.ZipFile,
                       deadline: float | None = None) -> tuple[str, list[zipfile.ZipInfo]]:
    """Validate central-directory names and return a metadata-only signature."""
    seen: set[str] = set()
    entries: list[tuple[str, int, int, int, int, int]] = []
    files: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        _check_deadline(deadline)
        member_name = info.filename.rstrip("/") if info.is_dir() else info.filename
        member = safe_member(member_name)
        if info.is_dir():
            continue
        if member in seen:
            raise ValueError(f"Duplicate ZIP member: {member}")
        seen.add(member)
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise ValueError(f"ZIP symlink is not allowed: {member}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP member is not supported: {member}")
        entries.append((member, info.file_size, info.CRC, info.compress_type,
                        info.flag_bits, mode & 0o170000))
        files.append(info)
    signature = fingerprint(sorted(entries))
    return signature, files


def _image_record(member: str, *, source_kind: str, read_member: str | None,
                  byte_size: int, zip_crc32: str | None, source_mtime_ns: int,
                  metadata: dict[str, dict[str, Any]], alias_reconstructed: bool = False,
                  alias_expected_sha256: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Build a provenance row, retaining malformed PNGs with a blocking error."""
    errors: list[str] = []
    parts = PurePosixPath(member).parts
    image_id = parts[-2] if len(parts) >= 2 else None
    if image_id is None:
        errors.append("PNG path must include an image folder")
    match = COORDINATE_NAME.fullmatch(parts[-1]) if parts else None
    if match is None:
        errors.append("PNG filename must encode unambiguous x_y coordinates")
        x = y = coordinate_variant = None
    else:
        x, y = int(match.group(1)), int(match.group(2))
        coordinate_variant = int(match.group(3) or 0)
    context = metadata.get(image_id, {}) if image_id is not None else {}
    row: dict[str, Any] = {
        "source_member": member,
        "read_member": read_member,
        "image_id": image_id,
        "x": x,
        "y": y,
        "coordinate_variant": coordinate_variant,
        "byte_size": byte_size,
        "zip_crc32": zip_crc32,
        "source_mtime_ns": source_mtime_ns,
        "metadata_match": bool(context),
        "alias_reconstructed": alias_reconstructed,
        "coordinate_collision": False,
        **context,
    }
    if alias_expected_sha256 is not None:
        row["alias_expected_sha256"] = alias_expected_sha256.lower()
    if byte_size < 0 or byte_size > MAX_PNG_BYTES:
        errors.append(f"PNG byte size exceeds {MAX_PNG_BYTES} byte read limit")
    if source_kind not in {"zip", "directory"}:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    return row, errors


def _directory_aliases(root: Path, aliases: dict[str, Any] | None,
                       discovered: set[str]) -> list[dict[str, Any]]:
    """Validate explicit virtual-member mappings without reading image bytes."""
    if aliases is None:
        return []
    if not isinstance(aliases, dict):
        raise ValueError("aliases must map virtual member paths to alias descriptors.")
    checked: list[dict[str, Any]] = []
    used_virtual: set[str] = set()
    for virtual_name, descriptor in aliases.items():
        virtual = safe_member(str(virtual_name))
        virtual_relative = virtual.removeprefix("Tiles/") if virtual.startswith("Tiles/") else virtual
        virtual_relative = safe_member(virtual_relative)
        if virtual_relative in discovered or virtual_relative in used_virtual:
            raise ValueError(f"Alias virtual path already exists: {virtual_relative}")
        if not virtual_relative.lower().endswith(".png"):
            raise ValueError(f"Alias virtual path must name a PNG: {virtual_relative}")
        if not isinstance(descriptor, dict):
            raise ValueError(f"Alias descriptor must be an object: {virtual_relative}")
        actual_value = descriptor.get("source_member")
        digest_value = descriptor.get("expected_sha256")
        if not isinstance(actual_value, str) or not isinstance(digest_value, str) or not _SHA256.fullmatch(digest_value):
            raise ValueError(f"Alias requires source_member and a SHA-256 digest: {virtual_relative}")
        actual = safe_member(actual_value)
        if not actual.lower().endswith(".png"):
            raise ValueError(f"Alias source must name a PNG: {actual}")
        candidate = root / Path(*PurePosixPath(actual).parts)
        current = root
        for component in PurePosixPath(actual).parts:
            current = current / component
            if current.is_symlink():
                raise ValueError(f"Alias source contains a symlink: {actual}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Alias source is missing or escapes the Tiles root: {actual}") from exc
        if candidate.is_symlink() or not resolved.is_file():
            raise ValueError(f"Alias source must be a regular file: {actual}")
        source_stat = resolved.stat()
        if source_stat.st_size > MAX_PNG_BYTES:
            raise ValueError(f"Alias source exceeds {MAX_PNG_BYTES} byte read limit: {actual}")
        checked.append({
            "virtual_member": virtual_relative,
            "source_member": actual,
            "expected_sha256": digest_value.lower(),
            "byte_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        })
        used_virtual.add(virtual_relative)
    return checked


def _part_identity(source_kind: str, source_name: str, source_signature: str) -> str:
    digest = fingerprint({"source_kind": source_kind, "source_name": source_name,
                          "source_signature": source_signature})
    return f"{source_kind}-{digest[:20]}"


def _part_block(part: dict[str, Any], reason: str) -> None:
    if reason not in part["blockers"]:
        part["blockers"].append(reason)
    part["blocked"] = True


def _base_part(source_kind: str, source_name: str, source_path: Path,
               source_size: int, source_signature: str, records: list[dict[str, Any]],
               source_mtime_ns: int | None = None) -> dict[str, Any]:
    return {
        "part_id": _part_identity(source_kind, source_name, source_signature),
        "source_kind": source_kind,
        "source_name": source_name,
        "source_path": str(source_path),
        "source_size": source_size,
        "source_signature": source_signature,
        "source_mtime_ns": source_mtime_ns,
        "records": records,
        "blocked": False,
        "blockers": [],
    }


def _scan_zip(path: Path, metadata: dict[str, dict[str, Any]], source_errors: list[dict[str, str]]) -> tuple[dict[str, Any] | None, int]:
    source_name = path.name
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("ZIP source must be a regular non-symlink file")
        archive_stat = path.stat()
        with zipfile.ZipFile(path) as archive:
            source_signature, entries = _archive_structure(archive)
            records: list[dict[str, Any]] = []
            png_count = 0
            row_errors: list[tuple[str, str]] = []
            for info in entries:
                member = safe_member(info.filename)
                if PurePosixPath(member).suffix.lower() != ".png":
                    continue
                png_count += 1
                row, errors = _image_record(
                    member,
                    source_kind="zip",
                    read_member=None,
                    byte_size=info.file_size,
                    zip_crc32=f"{info.CRC:08x}",
                    source_mtime_ns=archive_stat.st_mtime_ns,
                    metadata=metadata,
                )
                records.append(row)
                row_errors.extend((member, error) for error in errors)
            records.sort(key=lambda row: (
                row["image_id"] or "", row["y"] if row["y"] is not None else -1,
                row["x"] if row["x"] is not None else -1,
                row["coordinate_variant"] if row["coordinate_variant"] is not None else -1,
                row["source_member"],
            ))
        part = _base_part("zip", source_name, path.resolve(), archive_stat.st_size,
                          source_signature, records, archive_stat.st_mtime_ns)
        if png_count == 0:
            _part_block(part, "no_png_members")
            _source_error(source_errors, source_name, "ZIP contains no PNG members")
        for member, message in row_errors:
            _part_block(part, "invalid_png_record")
            _source_error(source_errors, source_name, message, member)
        return part, png_count
    except (OSError, EOFError, zipfile.BadZipFile, ValueError, RuntimeError) as exc:
        entry = {"source_name": source_name, "error": str(exc)}
        if isinstance(exc, (EOFError, zipfile.BadZipFile)):
            entry["status"] = "incomplete_upload"
        else:
            entry["status"] = "invalid_source"
        source_errors.append(entry)
        return None, 0


def _scan_directory(root: Path, metadata: dict[str, dict[str, Any]],
                    aliases: dict[str, Any] | None,
                    directory_part_size: int,
                    source_errors: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    source_name = "Tiles"
    if root.is_symlink() or not root.is_dir():
        _source_error(source_errors, source_name, "Source root must be a regular directory")
        return [], 0
    resolved_root = root.resolve()
    discovered: list[dict[str, Any]] = []
    discovered_relatives: set[str] = set()
    png_count = 0
    traversal_errors: list[str] = []
    for current, dirs, files in os.walk(resolved_root, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for dirname in dirs:
            dir_path = current_path / dirname
            if dir_path.is_symlink():
                traversal_errors.append(f"Symlink directory is not allowed: {dir_path.relative_to(resolved_root)}")
            else:
                safe_dirs.append(dirname)
        dirs[:] = safe_dirs
        for filename in files:
            path = current_path / filename
            if path.is_symlink():
                traversal_errors.append(f"Symlink file is not allowed: {path.relative_to(resolved_root)}")
                if path.suffix.lower() == ".png":
                    png_count += 1
                continue
            if path.suffix.lower() != ".png":
                continue
            png_count += 1
            relative = path.relative_to(resolved_root).as_posix()
            try:
                member = safe_member(relative)
                file_stat = path.stat()
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError("PNG source must be a regular file")
                discovered_relatives.add(member)
                discovered.append({
                    "member": member,
                    "byte_size": file_stat.st_size,
                    "source_mtime_ns": file_stat.st_mtime_ns,
                })
            except (OSError, ValueError) as exc:
                traversal_errors.append(f"{relative}: {exc}")
    for message in traversal_errors:
        _source_error(source_errors, source_name, message)

    try:
        alias_records = _directory_aliases(resolved_root, aliases, discovered_relatives)
    except ValueError as exc:
        _source_error(source_errors, source_name, str(exc))
        alias_records = []

    items: list[dict[str, Any]] = []
    for item in discovered:
        item_member = item["member"]
        row, errors = _image_record(
            f"Tiles/{item_member}", source_kind="directory", read_member=item_member,
            byte_size=item["byte_size"], zip_crc32=None,
            source_mtime_ns=item["source_mtime_ns"], metadata=metadata,
        )
        items.append({"row": row, "errors": errors, "identity": {
            "source_member": row["source_member"], "read_member": item_member,
            "byte_size": item["byte_size"], "alias_reconstructed": False,
            "alias_expected_sha256": None,
        }})
    for alias in alias_records:
        logical_member = f"Tiles/{alias['virtual_member']}"
        row, errors = _image_record(
            logical_member, source_kind="directory", read_member=alias["source_member"],
            byte_size=alias["byte_size"], zip_crc32=None,
            source_mtime_ns=alias["source_mtime_ns"], metadata=metadata,
            alias_reconstructed=True, alias_expected_sha256=alias["expected_sha256"],
        )
        items.append({"row": row, "errors": errors, "identity": {
            "source_member": logical_member, "read_member": alias["source_member"],
            "byte_size": alias["byte_size"], "alias_reconstructed": True,
            "alias_expected_sha256": alias["expected_sha256"],
        }})
    items.sort(key=lambda item: (
        item["row"]["image_id"] or "",
        item["row"]["y"] if item["row"]["y"] is not None else -1,
        item["row"]["x"] if item["row"]["x"] is not None else -1,
        item["row"]["coordinate_variant"] if item["row"]["coordinate_variant"] is not None else -1,
        item["row"]["source_member"],
    ))
    if not items and png_count == 0:
        _source_error(source_errors, source_name, "Directory contains no PNG files")

    parts: list[dict[str, Any]] = []
    for offset in range(0, len(items), directory_part_size):
        chunk = items[offset:offset + directory_part_size]
        rows = [item["row"] for item in chunk]
        signature = fingerprint([item["identity"] for item in chunk])
        byte_size = sum(row["byte_size"] for row in rows)
        mtimes = [row["source_mtime_ns"] for row in rows]
        part = _base_part("directory", source_name, resolved_root, byte_size,
                          signature, rows, max(mtimes, default=None))
        for item in chunk:
            for error in item["errors"]:
                _part_block(part, "invalid_png_record")
                _source_error(source_errors, source_name, error, item["row"]["source_member"])
        parts.append(part)
    return parts, png_count + len(alias_records)


def _collect_collisions(parts: list[dict[str, Any]], source_errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    coordinates: dict[tuple[str, int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for part in parts:
        for row in part["records"]:
            if row["image_id"] is not None and row["x"] is not None and row["y"] is not None:
                coordinates[(row["image_id"], row["x"], row["y"])].append((part, row))
    collisions: list[dict[str, Any]] = []
    for key, entries in sorted(coordinates.items()):
        if len(entries) < 2:
            continue
        members = []
        variant_counts: dict[int | None, int] = defaultdict(int)
        for _, row in entries:
            variant_counts[row["coordinate_variant"]] += 1
        ambiguous = any(count > 1 for count in variant_counts.values())
        for part, row in entries:
            row["coordinate_collision"] = True
            if ambiguous:
                _part_block(part, "ambiguous_duplicate_coordinates")
            members.append({
                "part_id": part["part_id"], "source_name": part["source_name"],
                "source_member": row["source_member"],
                "coordinate_variant": row["coordinate_variant"],
            })
        collisions.append({"image_id": key[0], "x": key[1], "y": key[2], "members": members})
        if ambiguous:
            source_errors.append({"source_name": entries[0][0]["source_name"],
                                  "error": "Ambiguous duplicate patch coordinates",
                                  "source_member": entries[0][1]["source_member"],
                                  "status": "invalid_source"})
    zip_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for part in parts:
        if part["source_kind"] != "zip":
            continue
        for row in part["records"]:
            zip_members[row["source_member"]].append(part)
    for member, owners in sorted(zip_members.items()):
        unique_parts = {part["part_id"]: part for part in owners}
        if len(unique_parts) < 2:
            continue
        for part in unique_parts.values():
            _part_block(part, "overlapping_zip_member_path")
        source_errors.append({"source_name": ",".join(sorted(part["source_name"] for part in unique_parts.values())),
                              "error": "PNG member path occurs in multiple ZIPs",
                              "source_member": member})
    for part in parts:
        part["blockers"].sort()
    return collisions


def build_source_plan(metadata: Path, source_root: Path, source_kind: str = "zip", *,
                      directory_part_size: int = 1024, expected_archives: int | None = 25,
                      expected_pngs: int | None = 148991,
                      aliases: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inventory PNG metadata and source structure without reading image bytes."""
    metadata_path = Path(metadata)
    try:
        source_root = Path(source_root).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve source root: {exc}") from exc
    if source_kind not in {"zip", "directory"}:
        raise ValueError("source_kind must be 'zip' or 'directory'.")
    if isinstance(directory_part_size, bool) or not isinstance(directory_part_size, int) or directory_part_size < 1:
        raise ValueError("directory_part_size must be a positive integer.")
    if expected_archives is not None:
        _checked_nonnegative_int(expected_archives, "expected_archives")
    if expected_pngs is not None:
        _checked_nonnegative_int(expected_pngs, "expected_pngs")
    if source_kind == "zip" and aliases:
        raise ValueError("Aliases are supported only for directory sources.")

    metadata_rows = _metadata_rows(metadata_path)
    metadata_sha256 = file_hash(metadata_path)
    source_errors: list[dict[str, str]] = []
    parts: list[dict[str, Any]] = []
    if source_kind == "zip":
        if source_root.is_symlink() or not source_root.is_dir():
            observed_sources = 0
            _source_error(source_errors, source_root.name or "source_root", "ZIP source root must be a regular directory")
            observed_pngs = 0
        else:
            archives = sorted(source_root.glob("Tiles-*.zip"), key=lambda path: path.name)
            observed_sources = len(archives)
            observed_pngs = 0
            for archive_path in archives:
                part, png_count = _scan_zip(archive_path, metadata_rows, source_errors)
                observed_pngs += png_count
                if part is not None:
                    parts.append(part)
        expected_sources = expected_archives
    else:
        directory_parts, observed_pngs = _scan_directory(
            source_root, metadata_rows, aliases, directory_part_size, source_errors,
        )
        parts.extend(directory_parts)
        observed_sources = 1 if source_root.is_dir() and not source_root.is_symlink() else 0
        expected_sources = 1

    coordinate_collisions = _collect_collisions(parts, source_errors)
    coverage = {"4": 0, "10": 0, "40": 0}
    unmatched_png_count = 0
    for part in parts:
        for row in part["records"]:
            if not row["metadata_match"]:
                unmatched_png_count += 1
                _part_block(part, "unmatched_metadata")
                source_errors.append({"source_name": part["source_name"],
                                      "error": "PNG has no matching metadata row",
                                      "source_member": row["source_member"]})
                continue
            if row["objective_lens"] in {4, 10, 40} and row["x"] is not None and row["y"] is not None:
                coverage[str(row["objective_lens"])] += 1
    for part in parts:
        part["blockers"].sort()

    pending_sources = max(0, expected_sources - observed_sources) if expected_sources is not None else 0
    pending_pngs = max(0, expected_pngs - observed_pngs) if expected_pngs is not None else 0
    excess_sources = max(0, observed_sources - expected_sources) if expected_sources is not None else 0
    excess_pngs = max(0, observed_pngs - expected_pngs) if expected_pngs is not None else 0
    if excess_sources:
        source_errors.append({"source_name": str(source_root),
                              "error": f"Observed {observed_sources} sources; configured maximum is {expected_sources}",
                              "status": "invalid_source"})
    if excess_pngs:
        source_errors.append({"source_name": str(source_root),
                              "error": f"Observed {observed_pngs} PNGs; configured maximum is {expected_pngs}",
                              "status": "invalid_source"})
    expected_counts_met = pending_sources == 0 and pending_pngs == 0 and excess_sources == 0 and excess_pngs == 0
    pending_reasons = []
    if pending_sources:
        pending_reasons.append("awaiting_sources")
    if pending_pngs:
        pending_reasons.append("awaiting_pngs")
    source_complete = expected_counts_met and not source_errors
    return {
        "schema_version": 1,
        "metadata_sha256": metadata_sha256,
        "source_kind": source_kind,
        "parts": parts,
        "observed_sources": observed_sources,
        "expected_sources": expected_sources,
        "pending_sources": pending_sources,
        "observed_pngs": observed_pngs,
        "expected_pngs": expected_pngs,
        "pending_pngs": pending_pngs,
        "pending_reasons": pending_reasons,
        "source_complete": source_complete,
        "source_errors": source_errors,
        "coverage": coverage,
        "coordinate_collisions": coordinate_collisions,
        "unmatched_png_count": unmatched_png_count,
    }


def _require_unblocked_part(part: dict[str, Any]) -> None:
    if part.get("blocked"):
        reasons = ", ".join(part.get("blockers", [])) or "unspecified source issue"
        raise ValueError(f"Cannot stage blocked part {part.get('part_id')}: {reasons}")


def _ensure_disk_space(work_root: Path, required_bytes: int) -> None:
    free_bytes = shutil.disk_usage(work_root).free
    if free_bytes < required_bytes + FREE_DISK_RESERVE_BYTES:
        raise OSError(errno.ENOSPC, f"Insufficient scratch space for {required_bytes} staged bytes")


def _copy_stream(source: Path, destination: Path, expected_size: int,
                 work_root: Path, deadline: float | None) -> tuple[int, str]:
    digest = hashlib.sha256()
    copied = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while True:
            _check_deadline(deadline)
            block = input_stream.read(COPY_CHUNK_BYTES)
            if not block:
                break
            if copied + len(block) > expected_size:
                raise ValueError(f"Source grew while staging: {source}")
            _ensure_disk_space(work_root, expected_size - copied)
            output_stream.write(block)
            digest.update(block)
            copied += len(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if copied != expected_size:
        raise ValueError(f"Staged byte size differs from inventory: {source}")
    return copied, digest.hexdigest()


def _directory_source_file(root: Path, read_member: str, row: dict[str, Any]) -> Path:
    member = safe_member(read_member)
    candidate = root / Path(*PurePosixPath(member).parts)
    current = root
    for component in PurePosixPath(member).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Directory source contains a symlink: {member}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Directory source escapes its root: {member}") from exc
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Directory source is not a regular file: {member}")
    if before.st_size != row["byte_size"] or before.st_mtime_ns != row["source_mtime_ns"]:
        raise ValueError(f"Directory source changed since inventory: {member}")
    return resolved


def _copy_directory_record(root: Path, staging_root: Path, row: dict[str, Any],
                           work_root: Path, deadline: float | None,
                           remaining_bytes: int) -> None:
    source = _directory_source_file(root, row["read_member"], row)
    destination_member = row["source_member"].removeprefix("Tiles/")
    destination = staging_root / Path(*PurePosixPath(safe_member(destination_member)).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        opened = os.fstat(input_stream.fileno())
        if opened.st_size != before.st_size or opened.st_mtime_ns != before.st_mtime_ns:
            raise ValueError(f"Directory source changed while opening: {row['read_member']}")
        while True:
            _check_deadline(deadline)
            block = input_stream.read(COPY_CHUNK_BYTES)
            if not block:
                break
            if copied + len(block) > row["byte_size"]:
                raise ValueError(f"Directory source grew while staging: {row['read_member']}")
            _ensure_disk_space(work_root, remaining_bytes - copied)
            output_stream.write(block)
            digest.update(block)
            copied += len(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
        after = os.fstat(input_stream.fileno())
    if copied != row["byte_size"] or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError(f"Directory source changed while staging: {row['read_member']}")
    expected_alias = row.get("alias_expected_sha256")
    if row.get("alias_reconstructed"):
        if not expected_alias or digest.hexdigest() != expected_alias:
            raise ValueError(f"Alias SHA-256 mismatch: {row['source_member']}")


def _prepare_staged_part(part: dict[str, Any], task_dir: Path, work_root: Path,
                         deadline: float | None) -> tuple[dict[str, Any], zipfile.ZipFile | None]:
    source_kind = part.get("source_kind")
    source_path = Path(part["source_path"])
    expected_size = part.get("source_size")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("Part source_size must be a nonnegative integer.")
    _ensure_disk_space(work_root, expected_size)
    staged_part = dict(part)
    staged_part["records"] = [dict(row) for row in part["records"]]
    if source_kind == "zip":
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("ZIP source must remain a regular non-symlink file")
        archive_name = safe_member(Path(part["source_name"]).name)
        staged_path = task_dir / archive_name
        _, archive_sha256 = _copy_stream(source_path, staged_path, expected_size, work_root, deadline)
        handle: zipfile.ZipFile | None = None
        try:
            handle = zipfile.ZipFile(staged_path)
            signature, _ = _archive_structure(handle, deadline)
            if signature != part["source_signature"]:
                raise ValueError("Staged ZIP structural signature differs from plan")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            if handle is not None:
                handle.close()
            raise ValueError(f"Staged ZIP is corrupt: {exc}") from exc
        except BaseException:
            if handle is not None:
                handle.close()
            raise
        assert handle is not None
        staged_part["source_path"] = str(staged_path)
        staged_part["source_archive_sha256"] = archive_sha256
        staged_part["zip_handle"] = handle
        return staged_part, handle
    if source_kind == "directory":
        if source_path.is_symlink() or not source_path.is_dir():
            raise ValueError("Directory source must remain a regular non-symlink directory")
        remaining_bytes = expected_size
        for row in staged_part["records"]:
            _check_deadline(deadline)
            _copy_directory_record(source_path, task_dir, row, work_root, deadline, remaining_bytes)
            remaining_bytes -= row["byte_size"]
        staged_part["source_path"] = str(task_dir)
        staged_part["source_archive_sha256"] = None
        return staged_part, None
    raise ValueError(f"Unsupported part source kind: {source_kind!r}")


def _retryable_oserror(error: OSError) -> bool:
    return error.errno in _TRANSIENT_ERRNOS


def _cleanup_staging(task_dir: Path, zip_handle: zipfile.ZipFile | None) -> None:
    try:
        if zip_handle is not None:
            zip_handle.close()
    finally:
        shutil.rmtree(task_dir)


@contextmanager
def stage_feature_part(part: dict[str, Any], work_root: Path, *, retries: int = 3,
                       deadline: float | None = None) -> Iterator[dict[str, Any]]:
    """Copy one source part to unique SSD scratch, verify it, then clean it up."""
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer.")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))):
        raise ValueError("deadline must be a monotonic timestamp or None.")
    _require_unblocked_part(part)
    scratch_root = Path(work_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Any] | None = None
    zip_handle: zipfile.ZipFile | None = None
    task_dir: Path | None = None
    for attempt in range(retries + 1):
        _check_deadline(deadline)
        task_dir = Path(tempfile.mkdtemp(prefix=f"feature-{part['part_id']}-", dir=scratch_root))
        try:
            prepared, zip_handle = _prepare_staged_part(part, task_dir, scratch_root, deadline)
            break
        except OSError as exc:
            _cleanup_staging(task_dir, zip_handle)
            zip_handle = None
            task_dir = None
            if attempt >= retries or not _retryable_oserror(exc):
                raise
            backoff = 0.1 * (2 ** attempt)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Feature part staging deadline exceeded.") from exc
                time.sleep(min(backoff, remaining))
            else:
                time.sleep(backoff)
        except BaseException:
            _cleanup_staging(task_dir, zip_handle)
            zip_handle = None
            raise
    if prepared is None or task_dir is None:
        raise RuntimeError("Feature part staging did not produce a staged part.")
    try:
        yield prepared
    finally:
        _cleanup_staging(task_dir, zip_handle)


def read_feature_payload(part: dict[str, Any], row: dict[str, Any]) -> bytes:
    """Read and validate one PNG payload from a staged part, capped at 64 MiB."""
    expected_size = row.get("byte_size")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or not 0 <= expected_size <= MAX_PNG_BYTES:
        raise ValueError(f"Invalid PNG byte size for {row.get('source_member')!r}")
    if part.get("source_kind") == "zip":
        member = safe_member(row["source_member"])
        handle = part.get("zip_handle")
        owns_handle = handle is None
        if owns_handle:
            handle = zipfile.ZipFile(part["source_path"])
        try:
            try:
                info = handle.getinfo(member)
            except KeyError as exc:
                raise ValueError(f"ZIP member is missing: {member}") from exc
            if info.file_size != expected_size or f"{info.CRC:08x}" != row.get("zip_crc32"):
                raise ValueError(f"ZIP member metadata differs from plan: {member}")
            with handle.open(info, "r") as stream:
                payload = stream.read(MAX_PNG_BYTES + 1)
            if len(payload) > MAX_PNG_BYTES:
                raise ValueError(f"PNG byte size exceeds {MAX_PNG_BYTES} byte read limit: {member}")
            if len(payload) != expected_size:
                raise ValueError(f"ZIP member size mismatch: {member}")
            actual_crc = f"{binascii.crc32(payload) & 0xffffffff:08x}"
            if actual_crc != row.get("zip_crc32"):
                raise ValueError(f"ZIP member CRC mismatch: {member}")
            return payload
        except (zipfile.BadZipFile, EOFError, RuntimeError, zlib.error) as exc:
            raise ValueError(f"ZIP member CRC or decompression failure: {member}") from exc
        finally:
            if owns_handle:
                handle.close()
    if part.get("source_kind") == "directory":
        root = Path(part["source_path"])
        path = root / Path(*PurePosixPath(safe_member(row["source_member"].removeprefix("Tiles/"))).parts)
        try:
            path.resolve(strict=True).relative_to(root.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Staged directory member escapes or is missing: {row['source_member']}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Staged directory member is not a regular file: {row['source_member']}")
        with path.open("rb") as stream:
            payload = stream.read(MAX_PNG_BYTES + 1)
        if len(payload) > MAX_PNG_BYTES:
            raise ValueError(f"PNG byte size exceeds {MAX_PNG_BYTES} byte read limit: {row['source_member']}")
        if len(payload) != expected_size:
            raise ValueError(f"Staged directory member size mismatch: {row['source_member']}")
        expected_alias = row.get("alias_expected_sha256")
        if row.get("alias_reconstructed") and file_hash_bytes(payload) != expected_alias:
            raise ValueError(f"Alias SHA-256 mismatch: {row['source_member']}")
        return payload
    raise ValueError(f"Unsupported part source kind: {part.get('source_kind')!r}")


def file_hash_bytes(payload: bytes) -> str:
    """Return a SHA-256 digest for an already bounded payload."""
    return hashlib.sha256(payload).hexdigest()
