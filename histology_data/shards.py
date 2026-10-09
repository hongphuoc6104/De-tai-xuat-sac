"""Deterministic, resumable raw-image TAR shards with verified staging."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tarfile
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from .catalog import IMAGE_SUFFIXES, safe_member
from .io import atomic_json, file_hash, fingerprint

_CHUNK_SIZE = 1024 * 1024
_RECORD_SIZE = tarfile.RECORDSIZE
_IMAGE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CRC32 = re.compile(r"zip-crc32:[0-9a-f]{8}\Z")


class _DigestingReader:
    """Bounded reader that hashes exactly the bytes passed to ``tarfile``."""

    def __init__(self, stream: BinaryIO, expected_size: int) -> None:
        self._stream = stream
        self.expected_size = expected_size
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = _CHUNK_SIZE
        block = self._stream.read(min(size, _CHUNK_SIZE))
        self.size += len(block)
        if self.size > self.expected_size:
            raise ValueError("Source member grew while it was being packaged.")
        self.digest.update(block)
        return block


def _json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _catalog_sources(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate catalog identity and return source mappings by source ID."""
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("Unsupported or malformed catalog schema.")
    catalog_id = catalog.get("catalog_id")
    sources = catalog.get("sources")
    images = catalog.get("images")
    if not isinstance(catalog_id, str) or not isinstance(sources, list) or not isinstance(images, list):
        raise ValueError("Catalog must include catalog_id, sources and images.")
    fingerprint_payload = {key: value for key, value in catalog.items() if key not in {"catalog_id", "sources"}}
    if fingerprint(fingerprint_payload) != catalog_id:
        raise ValueError("Catalog fingerprint does not match its contents.")

    source_map: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Malformed catalog source definition.")
        source_id, source_path, kind = source.get("source_id"), source.get("path"), source.get("kind")
        if (not isinstance(source_id, str) or not source_id or not isinstance(source_path, str)
                or not source_path or kind not in {"directory", "zip"} or source_id in source_map):
            raise ValueError("Catalog source definitions require unique IDs, paths and valid kinds.")
        source_map[source_id] = source

    image_ids: set[str] = set()
    source_members: set[tuple[str, str]] = set()
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Malformed catalog image record.")
        image_id = image.get("image_id")
        file_name = image.get("file_name")
        source_id = image.get("source_id")
        source_member = image.get("source_member")
        byte_size = image.get("byte_size")
        signature = image.get("source_signature")
        if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id) or image_id in image_ids:
            raise ValueError("Catalog image IDs must be unique safe basenames.")
        if not isinstance(file_name, str) or PurePosixPath(safe_member(file_name)).name != file_name:
            raise ValueError(f"Catalog file name is not a basename: {file_name!r}")
        suffix = Path(file_name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported catalog image suffix: {suffix!r}")
        if not isinstance(source_id, str) or source_id not in source_map:
            raise ValueError(f"Image references unknown source: {source_id!r}")
        if not isinstance(source_member, str):
            raise ValueError("Catalog image is missing source_member.")
        safe_member(source_member)
        if (not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0):
            raise ValueError("Catalog image byte_size must be a nonnegative integer.")
        kind = source_map[source_id]["kind"]
        if ((kind == "directory" and (not isinstance(signature, str) or not _SHA256.fullmatch(signature)))
                or (kind == "zip" and (not isinstance(signature, str) or not _CRC32.fullmatch(signature)))):
            raise ValueError(f"Invalid source signature for {image_id}.")
        key = (source_id, source_member)
        if key in source_members:
            raise ValueError(f"Duplicate source image member: {source_member}")
        image_ids.add(image_id)
        source_members.add(key)
    if not images:
        raise ValueError("Cannot package an empty image catalog.")
    return source_map


def _output_member(image: dict[str, Any]) -> str:
    suffix = Path(image["file_name"]).suffix.lower()
    return safe_member(f"images/{image['image_id']}{suffix}")


def _tar_info(member: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(member)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _tar_header_size(member: str, size: int) -> int:
    """Return the exact deterministic header/PAX record size for one member."""
    return len(_tar_info(member, size).tobuf(tarfile.PAX_FORMAT, encoding="utf-8", errors="surrogateescape"))


def _member_record_size(member: str, size: int) -> int:
    return (_tar_header_size(member, size)
            + math.ceil(size / tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE)


def _rounded_archive_size(record_bytes: int) -> int:
    return math.ceil((record_bytes + 2 * tarfile.BLOCKSIZE) / _RECORD_SIZE) * _RECORD_SIZE


def _archive_size(images: list[dict[str, Any]]) -> int:
    record_bytes = sum(_member_record_size(image["member"], image["byte_size"]) for image in images)
    return _rounded_archive_size(record_bytes)


def _shard_paths(root: Path, shard_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"shard-[0-9]{6}", shard_id):
        raise ValueError(f"Invalid shard ID: {shard_id!r}")
    return root / f"{shard_id}.tar", root / f"{shard_id}.json"


def _safe_regular_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"Missing {description}: {path}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ValueError(f"{description} must be a regular file: {path}")


@contextmanager
def _open_source(image: dict[str, Any], source: dict[str, Any]) -> Iterator[BinaryIO]:
    """Open one exact catalog member, rejecting links and ambiguous ZIP names."""
    source_path = Path(source["path"]).expanduser().resolve()
    member = image["source_member"]
    if source["kind"] == "directory":
        target = source_path.joinpath(*PurePosixPath(member).parts)
        if target.is_symlink() or not target.resolve().is_relative_to(source_path):
            raise ValueError(f"Source member escapes its directory: {member}")
        if not target.is_file() or target.stat().st_size != image["byte_size"]:
            raise ValueError(f"Source member size or type changed since cataloging: {member}")
        with target.open("rb") as stream:
            yield stream
        return

    if not source_path.is_file() or not zipfile.is_zipfile(source_path):
        raise ValueError(f"Catalog ZIP source is unavailable: {source_path}")
    with zipfile.ZipFile(source_path, "r") as archive:
        matching = [info for info in archive.infolist() if not info.is_dir() and info.filename == member]
        if len(matching) != 1:
            raise ValueError(f"ZIP member is missing or duplicated: {member}")
        info = matching[0]
        if info.flag_bits & 1 or info.file_size != image["byte_size"]:
            raise ValueError(f"ZIP member size or encryption state changed since cataloging: {member}")
        if f"zip-crc32:{info.CRC:08x}" != image["source_signature"]:
            raise ValueError(f"ZIP member CRC changed since cataloging: {member}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError(f"ZIP image member is a symbolic link: {member}")
        try:
            with archive.open(info, "r") as stream:
                yield stream
        except zipfile.BadZipFile as exc:
            raise ValueError(f"ZIP member failed CRC verification: {member}") from exc


def _read_archive_member(stream: BinaryIO, expected_size: int) -> str:
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(_CHUNK_SIZE), b""):
        count += len(block)
        if count > expected_size:
            raise ValueError("TAR member is longer than its declared size.")
        digest.update(block)
    if count != expected_size:
        raise ValueError("TAR member size does not match its manifest.")
    return digest.hexdigest()


def _manifest_records(manifest: dict[str, Any], catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate one manifest's full catalog provenance and return its member map."""
    records = manifest.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Shard manifest has no images: {manifest.get('shard_id', '<unknown>')}")
    catalog_by_id = {image["image_id"]: image for image in catalog["images"]}
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed image record in shard manifest.")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or image_id not in catalog_by_id:
            raise ValueError("Shard manifest references an image outside the catalog.")
        catalog_image = catalog_by_id[image_id]
        member = record.get("member")
        expected_member = _output_member(catalog_image)
        if member != expected_member or member in expected:
            raise ValueError(f"Unexpected or duplicate TAR member in manifest: {member!r}")
        safe_member(member)
        if set(record) != {*catalog_image, "member", "sha256"}:
            raise ValueError(f"Shard image record has unexpected fields: {image_id}")
        if any(record.get(key) != value for key, value in catalog_image.items()):
            raise ValueError(f"Shard image provenance differs from catalog: {image_id}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Invalid image SHA-256 in manifest: {image_id}")
        expected[member] = record
    return expected


def _verify_tar(tar_path: Path, manifest: dict[str, Any], catalog: dict[str, Any], max_bytes: int) -> None:
    _safe_regular_file(tar_path, "TAR archive")
    records = manifest.get("images")
    if tar_path.stat().st_size > max_bytes:
        raise ValueError(f"Shard exceeds release size limit: {tar_path.name}")
    expected = _manifest_records(manifest, catalog)
    if _archive_size(records) != tar_path.stat().st_size:
        raise ValueError(f"TAR archive size is not canonical for its manifest: {tar_path.name}")
    found: set[str] = set()
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            for item in archive:
                name = item.name
                safe_member(name)
                if name not in expected or name in found:
                    raise ValueError(f"Unexpected or duplicate TAR member: {name!r}")
                if not item.isreg():
                    raise ValueError(f"TAR member is not a regular file: {name!r}")
                record = expected[name]
                if item.size != record["byte_size"]:
                    raise ValueError(f"TAR member size differs from manifest: {name}")
                stream = archive.extractfile(item)
                if stream is None:
                    raise ValueError(f"Cannot read TAR member: {name}")
                with stream:
                    digest = _read_archive_member(stream, item.size)
                if digest != record["sha256"]:
                    raise ValueError(f"TAR member checksum mismatch: {name}")
                found.add(name)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Unreadable TAR archive: {tar_path.name}") from exc
    if found != set(expected):
        raise ValueError(f"TAR archive is missing manifest members: {tar_path.name}")


def _load_release_metadata(
    release_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate release/catalog/manifests without reading any TAR archive bytes."""
    root = Path(release_root)
    descriptor_path = root / "release.json"
    _safe_regular_file(descriptor_path, "release descriptor")
    descriptor = _json(descriptor_path)
    catalog = descriptor.get("catalog")
    source_map = _catalog_sources(catalog) if isinstance(catalog, dict) else None
    if source_map is None:
        raise ValueError("Release descriptor has no valid catalog.")
    catalog_id = catalog["catalog_id"]
    if descriptor.get("catalog_id") != catalog_id:
        raise ValueError("Release and catalog identifiers do not match.")
    max_bytes = descriptor.get("max_bytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("Release descriptor has an invalid max_bytes.")
    shards = descriptor.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Release descriptor shards must be a list.")

    all_images = {image["image_id"] for image in catalog["images"]}
    covered: list[str] = []
    manifests: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    seen_shards: set[str] = set()
    for index, entry in enumerate(shards, 1):
        if not isinstance(entry, dict):
            raise ValueError("Malformed shard descriptor entry.")
        shard_id = f"shard-{index:06d}"
        manifest_name = f"{shard_id}.json"
        tar_name = f"{shard_id}.tar"
        if (entry.get("shard_id") != shard_id or entry.get("manifest_name") != manifest_name
                or entry.get("tar_name") != tar_name):
            raise ValueError("Shard descriptors must use sequential generated names.")
        manifest_path = root / manifest_name
        _safe_regular_file(manifest_path, "shard manifest")
        manifest_digest = entry.get("manifest_sha256")
        if not isinstance(manifest_digest, str) or not _SHA256.fullmatch(manifest_digest):
            raise ValueError(f"Invalid manifest checksum: {manifest_name}")
        if file_hash(manifest_path) != manifest_digest:
            raise ValueError(f"Shard manifest checksum mismatch: {manifest_name}")
        manifest = _json(manifest_path)
        if (manifest.get("shard_id") != shard_id or manifest.get("catalog_id") != catalog_id
                or manifest.get("tar_name") != tar_name):
            raise ValueError(f"Shard manifest identity mismatch: {manifest_name}")
        if manifest.get("tar_sha256") != entry.get("tar_sha256"):
            raise ValueError(f"Shard TAR checksum differs from descriptor: {tar_name}")
        tar_digest = entry.get("tar_sha256")
        if not isinstance(tar_digest, str) or not _SHA256.fullmatch(tar_digest):
            raise ValueError(f"Invalid TAR checksum: {tar_name}")
        records = manifest.get("images")
        if not isinstance(records, list) or entry.get("image_count") != len(records):
            raise ValueError(f"Shard image count differs from descriptor: {manifest_name}")
        _manifest_records(manifest, catalog)
        size_bytes = entry.get("size_bytes")
        if (not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 1
                or size_bytes > max_bytes or _archive_size(records) != size_bytes):
            raise ValueError(f"Shard size metadata is invalid: {tar_name}")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("image_id"), str):
                raise ValueError("Malformed image record in shard manifest.")
            image_id = record["image_id"]
            if image_id not in all_images:
                raise ValueError(f"Shard references unknown catalog image: {image_id}")
            if image_id in seen_shards:
                raise ValueError("An image appears in multiple committed shards.")
            seen_shards.add(image_id)
            covered.append(image_id)
        manifests[shard_id] = manifest
        entries[shard_id] = entry

    image_order = [image["image_id"] for image in catalog["images"]]
    if covered != image_order[:len(covered)]:
        raise ValueError("Committed shards must cover a prefix of catalog image order.")
    complete = len(covered) == len(image_order)
    if descriptor.get("complete") is not complete:
        raise ValueError("Release complete flag does not match committed image coverage.")
    return descriptor, manifests, entries, catalog


def _verify_shard_archive(root: Path, entry: dict[str, Any], manifest: dict[str, Any],
                          catalog: dict[str, Any], max_bytes: int) -> None:
    tar_path = _verify_shard_checksum(root, entry, manifest)
    _verify_tar(tar_path, manifest, catalog, max_bytes)


def _verify_shard_checksum(root: Path, entry: dict[str, Any], manifest: dict[str, Any]) -> Path:
    tar_path = root / manifest["tar_name"]
    _safe_regular_file(tar_path, "TAR archive")
    if tar_path.stat().st_size != entry["size_bytes"] or file_hash(tar_path) != entry["tar_sha256"]:
        raise ValueError(f"Shard TAR checksum or size mismatch: {tar_path.name}")
    return tar_path


def _verify_release(release_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(release_root)
    descriptor, manifests, entries, catalog = _load_release_metadata(root)
    for shard_id, manifest in manifests.items():
        _verify_shard_archive(root, entries[shard_id], manifest, catalog, descriptor["max_bytes"])
    return descriptor, manifests


def verify_release(release_root: Path) -> dict[str, Any]:
    """Verify release metadata, shard archives, and every stored image digest."""
    descriptor, _ = _verify_release(Path(release_root))
    return descriptor


def load_shard_manifest(release_root: Path, shard_id: str) -> dict[str, Any]:
    """Validate release/catalog/manifests and return one manifest without reading TARs."""
    _, manifests, _, _ = _load_release_metadata(Path(release_root))
    if shard_id not in manifests:
        raise ValueError(f"Shard is not committed in this release: {shard_id}")
    return manifests[shard_id]


def _plan_shards(images: list[dict[str, Any]], max_bytes: int) -> list[list[dict[str, Any]]]:
    planned: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_record_bytes = 0
    for catalog_image in images:
        record = dict(catalog_image, member=_output_member(catalog_image))
        record_bytes = _member_record_size(record["member"], record["byte_size"])
        if _rounded_archive_size(record_bytes) > max_bytes:
            raise ValueError(f"Single image cannot fit within max_bytes: {catalog_image['image_id']}")
        if current and _rounded_archive_size(current_record_bytes + record_bytes) > max_bytes:
            planned.append(current)
            current = [record]
            current_record_bytes = record_bytes
        else:
            current.append(record)
            current_record_bytes += record_bytes
    if current:
        planned.append(current)
    return planned


def _write_shard(root: Path, shard_id: str, catalog_id: str,
                 records: list[dict[str, Any]], source_map: dict[str, dict[str, Any]],
                 max_bytes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    tar_path, manifest_path = _shard_paths(root, shard_id)
    temp_tar = tar_path.with_name(f".{tar_path.name}.{uuid.uuid4().hex}.part")
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    committed_images: list[dict[str, Any]] = []
    try:
        with tarfile.open(temp_tar, mode="w", format=tarfile.PAX_FORMAT,
                          encoding="utf-8", errors="surrogateescape") as archive:
            for record in records:
                image = {key: value for key, value in record.items() if key != "member"}
                source = source_map[image["source_id"]]
                with _open_source(image, source) as stream:
                    reader = _DigestingReader(stream, image["byte_size"])
                    archive.addfile(_tar_info(record["member"], image["byte_size"]), reader)
                    if reader.read(1):
                        raise ValueError(f"Source member grew while being packaged: {image['source_member']}")
                    if reader.size != image["byte_size"]:
                        raise ValueError(f"Source member size changed while being packaged: {image['source_member']}")
                    source_signature = image["source_signature"]
                    digest = reader.digest.hexdigest()
                    if source["kind"] == "directory" and digest != source_signature:
                        raise ValueError(f"Directory source SHA-256 changed since cataloging: {image['source_member']}")
                    committed_images.append(dict(record, sha256=digest))
        actual_size = temp_tar.stat().st_size
        if actual_size > max_bytes or actual_size != _archive_size(committed_images):
            raise ValueError(f"Produced TAR archive violates its planned size: {shard_id}")
        tar_digest = file_hash(temp_tar)
        manifest = dict(shard_id=shard_id, catalog_id=catalog_id,
                        tar_name=tar_path.name, tar_sha256=tar_digest,
                        images=committed_images)
        with temp_manifest.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_tar, tar_path)
        os.replace(temp_manifest, manifest_path)
        entry = dict(shard_id=shard_id, manifest_name=manifest_path.name,
                     manifest_sha256=file_hash(manifest_path), tar_name=tar_path.name,
                     tar_sha256=tar_digest, image_count=len(committed_images),
                     size_bytes=actual_size)
        return entry, manifest
    finally:
        temp_tar.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)


def pack_catalog(catalog: dict, output: Path, max_bytes: int,
                 max_shards: int | None = None) -> dict:
    """Package catalog images into deterministic, independently verifiable TAR shards.

    ``max_shards`` limits new shards in this invocation. Calling again with the
    same catalog resumes after verified commits and eventually marks the release
    complete once every catalog image is committed.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")
    if max_shards is not None and (not isinstance(max_shards, int) or isinstance(max_shards, bool)
                                   or max_shards < 1):
        raise ValueError("max_shards must be a positive integer when provided.")
    source_map = _catalog_sources(catalog)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    release_path = root / "release.json"
    if release_path.exists():
        descriptor, _ = _verify_release(root)
        if descriptor["catalog_id"] != catalog["catalog_id"]:
            raise ValueError("Cannot resume a release with a different catalog.")
        committed = descriptor["shards"]
        if any(entry["size_bytes"] > max_bytes for entry in committed):
            raise ValueError("New max_bytes is below the size of an immutable committed shard.")
    else:
        committed = []
        descriptor = dict(catalog_id=catalog["catalog_id"], catalog=catalog,
                          complete=False, max_bytes=max_bytes, shards=[])

    catalog_images = catalog["images"]
    committed_count = sum(entry["image_count"] for entry in committed)
    pending = catalog_images[committed_count:]
    planned = _plan_shards(pending, max_bytes)
    new_shards = planned if max_shards is None else planned[:max_shards]
    current_max = len(committed)
    for offset, records in enumerate(new_shards, 1):
        shard_id = f"shard-{current_max + offset:06d}"
        entry, _ = _write_shard(root, shard_id, catalog["catalog_id"], records, source_map, max_bytes)
        committed.append(entry)
        committed_count += entry["image_count"]
        descriptor = dict(catalog_id=catalog["catalog_id"], catalog=catalog,
                          complete=committed_count == len(catalog_images),
                          max_bytes=max_bytes, shards=committed)
        atomic_json(release_path, descriptor)

    if not new_shards:
        descriptor = dict(catalog_id=catalog["catalog_id"], catalog=catalog,
                          complete=committed_count == len(catalog_images),
                          max_bytes=max_bytes, shards=committed)
        atomic_json(release_path, descriptor)
    return _verify_release(root)[0]


def _verify_staged(stage_root: Path, manifest: dict[str, Any]) -> None:
    marker_path = stage_root / ".staged.json"
    _safe_regular_file(marker_path, "staged marker")
    marker = _json(marker_path)
    if (marker.get("shard_id") != manifest["shard_id"]
            or marker.get("catalog_id") != manifest["catalog_id"]
            or marker.get("tar_sha256") != manifest["tar_sha256"]):
        raise ValueError("Staged marker does not match the verified shard.")
    expected = {record["member"]: record for record in manifest["images"]}
    if marker.get("files") != [dict(member=record["member"], sha256=record["sha256"],
                                     byte_size=record["byte_size"])
                               for record in manifest["images"]]:
        raise ValueError("Staged marker file list differs from the shard manifest.")
    actual: set[str] = set()
    expected_dirs = {
        parent.as_posix()
        for member in expected
        for parent in list(PurePosixPath(member).parents)[:-1]
        if parent.as_posix() != "."
    }
    actual_dirs: set[str] = set()
    for path in stage_root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"Staged files must be regular files: {path}")
        if stat.S_ISDIR(mode):
            directory = path.relative_to(stage_root).as_posix()
            if directory not in expected_dirs:
                raise ValueError(f"Unexpected staged directory: {directory}")
            actual_dirs.add(directory)
        elif path != marker_path:
            member = path.relative_to(stage_root).as_posix()
            if member not in expected:
                raise ValueError(f"Unexpected staged file: {member}")
            record = expected[member]
            if path.stat().st_size != record["byte_size"] or file_hash(path) != record["sha256"]:
                raise ValueError(f"Staged file checksum mismatch: {member}")
            actual.add(member)
    if actual != set(expected):
        raise ValueError("Staged directory is missing shard files.")
    if actual_dirs != expected_dirs:
        raise ValueError("Staged directory tree differs from the shard manifest.")


def stage_shard(release_root: Path, shard_id: str,
                work_root: Path) -> tuple[Path, dict]:
    """Verify and safely extract one committed shard into a reusable work directory."""
    root = Path(release_root)
    descriptor, manifests, entries, _ = _load_release_metadata(root)
    if shard_id not in manifests:
        raise ValueError(f"Shard is not committed in this release: {shard_id}")
    manifest = manifests[shard_id]
    tar_path = _verify_shard_checksum(root, entries[shard_id], manifest)
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    staging_parent = work / "staged" / descriptor["catalog_id"]
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage_root = staging_parent / shard_id
    if stage_root.is_symlink():
        raise ValueError(f"Staging destination cannot be a symbolic link: {stage_root}")
    if stage_root.exists():
        _verify_staged(stage_root, manifest)
        return stage_root, manifest

    required_bytes = sum(record["byte_size"] for record in manifest["images"])
    if shutil.disk_usage(work).free < required_bytes:
        raise OSError(f"Insufficient free space to stage {shard_id} ({required_bytes} bytes required).")

    temp_root = staging_parent / f".{shard_id}.{uuid.uuid4().hex}.tmp"
    temp_root.mkdir()
    try:
        expected = {record["member"]: record for record in manifest["images"]}
        found: set[str] = set()
        with tarfile.open(tar_path, mode="r:") as archive:
            for item in archive:
                member = safe_member(item.name)
                if member not in expected or member in found or not item.isreg():
                    raise ValueError(f"Unsafe TAR member during staging: {member}")
                found.add(member)
                record = expected[member]
                if item.size != record["byte_size"]:
                    raise ValueError(f"TAR member size mismatch during staging: {member}")
                destination = temp_root.joinpath(*PurePosixPath(member).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.resolve().is_relative_to(temp_root.resolve()):
                    raise ValueError(f"TAR member escapes staging directory: {member}")
                source_stream = archive.extractfile(item)
                if source_stream is None:
                    raise ValueError(f"Cannot read TAR member during staging: {member}")
                digest = hashlib.sha256()
                count = 0
                with source_stream, destination.open("xb") as target:
                    for block in iter(lambda: source_stream.read(_CHUNK_SIZE), b""):
                        count += len(block)
                        if count > record["byte_size"]:
                            raise ValueError(f"TAR member exceeded its manifest size: {member}")
                        digest.update(block)
                        target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                if count != record["byte_size"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"TAR member checksum mismatch during staging: {member}")

        if found != set(expected):
            raise ValueError("TAR archive is missing manifest members during staging.")
        actual_members = {path.relative_to(temp_root).as_posix() for path in temp_root.rglob("*") if path.is_file()}
        if actual_members != set(expected):
            raise ValueError("Staged TAR contents do not match the shard manifest.")
        marker = dict(schema_version=1, shard_id=shard_id, catalog_id=descriptor["catalog_id"],
                      tar_sha256=manifest["tar_sha256"],
                      files=[dict(member=record["member"], sha256=record["sha256"],
                                  byte_size=record["byte_size"])
                             for record in manifest["images"]])
        atomic_json(temp_root / ".staged.json", marker)
        os.replace(temp_root, stage_root)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
    _verify_staged(stage_root, manifest)
    return stage_root, manifest
