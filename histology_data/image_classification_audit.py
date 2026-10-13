"""Streaming technical QC and duplicate audit for inventoried original images.

The audit is descriptive: it reports technical measurements and duplicate
candidates without excluding, relabeling, or rewriting any source image.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import socket
import time
import uuid
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from . import image_classification as core
from .io import atomic_json, file_hash, fingerprint

AUDIT_SCHEMA_VERSION = 1
AUDIT_PROTOCOL = "original-image-technical-audit-v1"
CHUNK_SIZE = 32
LENSES = (4, 10, 40)
GRADE_ORDER = ("0", "3", "4", "5")
RECORD_FIELDS = (
    "image_id", "file_name", "image_stem", "patient_id", "objective_lens",
    "raw_glade", "image_target", "source_archive_sha256", "source_archive_name",
    "source_member", "member_crc32", "member_byte_size", "original_image_sha256",
    "original_width", "original_height", "source_format", "rgb_width", "rgb_height",
    "rgb_pixel_sha256", "verified_feature_row_original_image_sha256", "phash_hex",
    "qc_mean_brightness", "qc_mean_saturation",
    "qc_tissue_fraction_estimate", "qc_laplacian_variance",
)
CANDIDATE_FIELDS = (
    "candidate_id", "image_id_left", "image_id_right", "objective_lens_left",
    "objective_lens_right", "same_lens", "candidate_type", "patient_relation",
    "target_disagreement", "target_conflict",
    "image_target_left", "image_target_right", "exact_byte_duplicate",
    "exact_pixel_duplicate", "review_required", "blocks_ready_for_fit",
    "phash_hamming_distance", "candidate_reasons",
)
SELECTION_FIELDS = (
    "selection_order", "image_id", "file_name", "patient_id", "objective_lens",
    "raw_glade", "image_target", "source_archive_sha256", "source_member",
)
Notify = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ImageAuditResult:
    """Committed or resumable audit references returned to notebook callers."""

    audit_root: Path
    status: dict[str, Any]
    binding: dict[str, Any]
    representative_image_ids: list[str]


class _BudgetExhausted(RuntimeError):
    """The configured time budget ended before the next chunk committed."""


class _RunLock:
    """Prevent concurrent writers and recover only when the previous owner stopped."""

    def __init__(self, root: Path, *, recover: bool) -> None:
        self.root = root
        self.path = root / "audit.lock"
        self.run_id = uuid.uuid4().hex
        self.recover = recover

    def __enter__(self) -> _RunLock:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink():
                raise ValueError("Refusing a symlinked audit lock.")
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Unreadable audit lock; refusing automatic recovery.") from exc
            if not isinstance(owner, dict) or not isinstance(owner.get("pid"), int):
                raise ValueError("Audit lock has no verifiable owner; refusing recovery.")
            if owner.get("hostname") != socket.gethostname():
                raise ValueError("Cross-host audit-lock recovery is unsupported.")
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                if not self.recover:
                    raise ValueError(
                        "A previous runtime left an audit lock; prove it stopped and set recover_lock=True."
                    )
                self.path.unlink()
            except (PermissionError, ValueError, TypeError) as exc:
                raise ValueError("Audit lock owner cannot be proven stopped.") from exc
            else:
                raise ValueError("An active audit already owns this output directory.")
        body = {"run_id": self.run_id, "pid": os.getpid(),
                "hostname": socket.gethostname()}
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(body, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.path.is_file() and not self.path.is_symlink():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if owner.get("run_id") == self.run_id:
                self.path.unlink()


def _source_identity(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "inventory_id": inventory["inventory_id"],
        "metadata_sha256": inventory["metadata_sha256"],
        "sources": {
            str(lens): {
                "source_name": inventory["sources"][str(lens)]["source_name"],
                "archive_sha256": inventory["sources"][str(lens)]["archive_sha256"],
                "archive_bytes": inventory["sources"][str(lens)]["archive_bytes"],
                "member_count": inventory["sources"][str(lens)]["member_count"],
            }
            for lens in LENSES
        },
    }


def _verified_feature_bindings(
    rows: Sequence[Mapping[str, Any]] | None,
    inventory: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    if rows is None:
        return None, None
    if isinstance(rows, (str, bytes)):
        raise ValueError("verified_feature_rows must be a sequence of feature-row mappings.")
    expected = {row["image_id"]: row for row in inventory["images"]}
    bindings: dict[str, dict[str, Any]] = {}
    for source in rows:
        if not isinstance(source, Mapping):
            raise ValueError("Every verified feature row must be a mapping.")
        image_id = source.get("image_id")
        if not isinstance(image_id, str) or image_id not in expected:
            raise ValueError("Verified feature rows contain an unknown image_id.")
        if image_id in bindings:
            raise ValueError(f"Verified feature rows duplicate image_id {image_id}.")
        inventory_row = expected[image_id]
        required = ("patient_id", "objective_lens", "image_target", "original_image_sha256")
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(f"Verified feature row {image_id} is missing fields: {missing}")
        raw_sha = source["original_image_sha256"]
        if (not isinstance(raw_sha, str) or len(raw_sha) != 64
                or any(char not in "0123456789abcdef" for char in raw_sha)):
            raise ValueError(f"Verified feature row has an invalid original_image_sha256: {image_id}")
        if (source["patient_id"] != inventory_row["patient_id"]
                or int(source["objective_lens"]) != int(inventory_row["objective_lens"])
                or int(source["image_target"]) != int(inventory_row["image_target"])):
            raise ValueError(f"Verified feature row metadata differs from the inventory for {image_id}.")
        if "raw_glade" in source and str(source["raw_glade"]) != str(inventory_row["raw_glade"]):
            raise ValueError(f"Verified feature row grade differs from the inventory for {image_id}.")
        if ("source_archive_sha256" in source
                and source["source_archive_sha256"] != inventory_row["source_archive_sha256"]):
            raise ValueError(f"Verified feature row source archive differs for {image_id}.")
        bindings[image_id] = {
            "image_id": image_id,
            "patient_id": inventory_row["patient_id"],
            "objective_lens": int(inventory_row["objective_lens"]),
            "raw_glade": str(inventory_row["raw_glade"]),
            "image_target": int(inventory_row["image_target"]),
            "source_archive_sha256": inventory_row["source_archive_sha256"],
            "original_image_sha256": raw_sha,
        }
    if set(bindings) != set(expected):
        missing = sorted(set(expected) - set(bindings))
        raise ValueError(f"Verified feature rows do not cover every inventory image: {missing[:5]}")
    normalized = [bindings[image_id] for image_id in sorted(bindings)]
    return bindings, fingerprint(normalized)


def _make_binding(audit_id: str, inventory: dict[str, Any], descriptor: dict[str, Any],
                  source_identity: dict[str, Any], feature_rows_sha256: str | None) -> dict[str, Any]:
    body = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": audit_id,
        "inventory_id": inventory["inventory_id"],
        "metadata_sha256": inventory["metadata_sha256"],
        "source_archive_sha256": {
            lens: source_identity["sources"][lens]["archive_sha256"] for lens in ("4", "10", "40")
        },
        "verified_feature_rows_sha256": feature_rows_sha256,
        "descriptor_sha256": fingerprint(descriptor),
    }
    return {**body, "binding_id": fingerprint(body)}


def _load_binding(run_root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "binding.json"
    if path.is_symlink():
        raise ValueError("Refusing a symlinked image-audit binding.")
    if not path.exists():
        atomic_json(path, expected)
        return expected
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unreadable image-audit binding.") from exc
    body = {key: value for key, value in binding.items() if key != "binding_id"}
    if binding != expected or binding.get("binding_id") != fingerprint(body):
        raise ValueError("Image-audit binding differs from the verified source identity.")
    return binding


def _result(run_root: Path, binding: dict[str, Any],
            selection: Sequence[dict[str, Any]]) -> ImageAuditResult:
    status_path = run_root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return ImageAuditResult(
        audit_root=run_root,
        status=status,
        binding=binding,
        representative_image_ids=[str(row["image_id"]) for row in selection],
    )


def _validate_source_archives(
    inventory: dict[str, Any], archives: Mapping[int, Path],
) -> dict[int, Path]:
    if set(archives) != set(LENSES):
        raise ValueError("archives must provide exactly the 4X, 10X, and 40X ZIP files.")
    resolved: dict[int, Path] = {}
    for lens in LENSES:
        requested = Path(archives[lens]).expanduser()
        if requested.is_symlink() or not requested.is_file() or not zipfile.is_zipfile(requested):
            raise ValueError(f"Expected a regular ZIP archive for {lens}X: {requested}")
        path = requested.resolve()
        expected = inventory["sources"][str(lens)]
        actual_sha256 = file_hash(path)
        if actual_sha256 != expected["archive_sha256"]:
            raise ValueError(
                f"{lens}X source archive SHA-256 differs from the verified inventory "
                f"(expected {expected['archive_sha256']}, found {actual_sha256})."
            )
        if path.stat().st_size != expected["archive_bytes"]:
            raise ValueError(f"{lens}X source archive byte size differs from the inventory.")
        resolved[lens] = path
    return resolved


def _pixel_digest(rgb: Image.Image) -> tuple[str, int, int]:
    width, height = map(int, rgb.size)
    digest = hashlib.sha256()
    digest.update(b"histology-rgb-pixels-v1\0")
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest(), width, height


def _phash(rgb: Image.Image, *, thumbnail_size: int, hash_size: int) -> str:
    try:
        from scipy.fft import dctn
    except ImportError as exc:
        raise RuntimeError("The original-image audit requires scipy for its small pHash DCT.") from exc
    gray = rgb.convert("L").resize(
        (thumbnail_size, thumbnail_size), resample=Image.Resampling.LANCZOS,
    )
    coefficients = dctn(np.asarray(gray, dtype=np.float64), type=2, norm="ortho")
    low = coefficients[:hash_size, :hash_size].reshape(-1)
    values = low[1:]
    median = float(np.median(values))
    bits = low >= median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    gray.close()
    return f"{value:0{math.ceil(hash_size * hash_size / 4)}x}"


def _quality_metrics(rgb: Image.Image) -> dict[str, float]:
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = np.divide(
        maximum - minimum, maximum,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    brightness = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
    tissue = (brightness < 0.92) & ((saturation > 0.05) | (brightness < 0.75))
    if brightness.shape[0] >= 3 and brightness.shape[1] >= 3:
        center = brightness[1:-1, 1:-1]
        laplacian = (
            brightness[:-2, 1:-1] + brightness[2:, 1:-1]
            + brightness[1:-1, :-2] + brightness[1:-1, 2:] - 4.0 * center
        )
        sharpness = float(laplacian.var(dtype=np.float64))
    else:
        sharpness = 0.0
    return {
        "qc_mean_brightness": round(float(brightness.mean(dtype=np.float64)), 8),
        "qc_mean_saturation": round(float(saturation.mean(dtype=np.float64)), 8),
        "qc_tissue_fraction_estimate": round(float(tissue.mean(dtype=np.float64)), 8),
        "qc_laplacian_variance": round(sharpness, 8),
    }


def _audit_image(archive: zipfile.ZipFile, row: dict[str, Any], *,
                 max_image_bytes: int, max_image_pixels: int,
                 thumbnail_size: int, hash_size: int,
                 verified_feature_row: dict[str, Any] | None) -> dict[str, Any]:
    rgb, original_sha256, original_size, source_format = core._load_member_image(
        archive, row, max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
    )
    try:
        if (verified_feature_row is not None
                and original_sha256 != verified_feature_row["original_image_sha256"]):
            raise ValueError(
                f"Original SHA-256 differs from the verified feature row for {row['image_id']}."
            )
        pixels_sha256, rgb_width, rgb_height = _pixel_digest(rgb)
        record = {
            key: row[key] for key in (
                "image_id", "file_name", "image_stem", "patient_id", "objective_lens",
                "raw_glade", "image_target", "source_archive_sha256", "source_archive_name",
                "source_member", "member_crc32", "member_byte_size",
            )
        }
        record.update({
            "original_image_sha256": original_sha256,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "source_format": source_format,
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "rgb_pixel_sha256": pixels_sha256,
            "verified_feature_row_original_image_sha256": (
                verified_feature_row["original_image_sha256"] if verified_feature_row else ""
            ),
            "phash_hex": _phash(rgb, thumbnail_size=thumbnail_size, hash_size=hash_size),
            **_quality_metrics(rgb),
        })
        return record
    finally:
        rgb.close()


def _write_text_atomic(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_text(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable committed audit chunk: {path}") from exc


def _commit_chunk(run_root: Path, chunk_index: int, rows: list[dict[str, Any]], *,
                  audit_id: str, inventory_id: str,
                  source_identity: dict[str, Any]) -> dict[str, Any]:
    folder = run_root / "chunks" / f"chunk-{chunk_index:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError(f"Refusing a symlinked audit chunk: {folder}")
    data_path = folder / "records.jsonl"
    commit_path = folder / "commit.json"
    if commit_path.exists() or commit_path.is_symlink():
        raise ValueError(f"Refusing to overwrite committed audit chunk {chunk_index}.")
    if data_path.is_symlink():
        raise ValueError(f"Refusing a symlinked uncommitted audit chunk: {data_path}")
    data_path.unlink(missing_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    _write_text_atomic(data_path, text)
    body = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": audit_id,
        "inventory_id": inventory_id,
        "chunk_index": chunk_index,
        "image_count": len(rows),
        "image_ids": [row["image_id"] for row in rows],
        "records_sha256": file_hash(data_path),
        "source_identity": source_identity,
        "complete": True,
    }
    commit = {**body, "commit_id": fingerprint(body)}
    atomic_json(commit_path, commit)
    return commit


def _load_committed_chunks(run_root: Path, *, audit_id: str, inventory_id: str,
                           source_identity: dict[str, Any], expected_ids: list[str]
                           ) -> tuple[dict[int, list[dict[str, Any]]], dict[int, str]]:
    root = run_root / "chunks"
    if root.is_symlink():
        raise ValueError("Refusing a symlinked audit chunk directory.")
    if not root.exists():
        return {}, {}
    chunks: dict[int, list[dict[str, Any]]] = {}
    commit_ids: dict[int, str] = {}
    for folder in sorted(root.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Invalid audit chunk path: {folder}")
        if not folder.name.startswith("chunk-"):
            raise ValueError(f"Unexpected path in audit chunk directory: {folder.name}")
        try:
            index = int(folder.name.removeprefix("chunk-"))
        except ValueError as exc:
            raise ValueError(f"Invalid audit chunk folder name: {folder.name}") from exc
        if folder.name != f"chunk-{index:06d}" or index in chunks or index in commit_ids:
            raise ValueError(f"Noncanonical or duplicate audit chunk folder: {folder.name}")
        commit_path, records_path = folder / "commit.json", folder / "records.jsonl"
        if not commit_path.is_file() or commit_path.is_symlink():
            # Data without a commit is untrusted and will be rebuilt if needed.
            if records_path.is_symlink():
                raise ValueError(f"Refusing a symlinked incomplete audit chunk: {records_path}")
            records_path.unlink(missing_ok=True)
            folder.rmdir()
            continue
        if not records_path.is_file() or records_path.is_symlink():
            raise ValueError(f"Committed audit chunk is missing its data: {records_path}")
        try:
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable audit chunk commit: {commit_path}") from exc
        body = {key: value for key, value in commit.items() if key != "commit_id"}
        rows = _read_json_lines(records_path)
        expected_chunk_ids = expected_ids[index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE]
        if (
            commit.get("commit_id") != fingerprint(body)
            or commit.get("audit_id") != audit_id
            or commit.get("inventory_id") != inventory_id
            or commit.get("source_identity") != source_identity
            or commit.get("chunk_index") != index
            or commit.get("complete") is not True
            or commit.get("image_count") != len(rows)
            or commit.get("image_ids") != expected_chunk_ids
            or [row.get("image_id") for row in rows] != expected_chunk_ids
            or commit.get("records_sha256") != file_hash(records_path)
        ):
            raise ValueError(f"Audit chunk {index} failed commit verification.")
        chunks[index] = rows
        commit_ids[index] = commit["commit_id"]
    if set(chunks) != set(range(len(chunks))):
        raise ValueError("Committed audit chunks are not a contiguous prefix.")
    return chunks, commit_ids


class _BKTree:
    """Small Hamming-distance tree used to find conservative pHash candidates."""

    def __init__(self) -> None:
        self.root: tuple[int, list[dict[str, Any]], dict[int, Any]] | None = None

    def add(self, value: int, row: dict[str, Any]) -> None:
        if self.root is None:
            self.root = (value, [row], {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(row)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [row], {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[dict[str, Any], int]]:
        matches: list[tuple[dict[str, Any], int]] = []
        if self.root is None:
            return matches
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                matches.extend((row, distance) for row in node[1])
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return matches


def _duplicate_candidates(records: list[dict[str, Any]], *, threshold: int) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def add(left: dict[str, Any], right: dict[str, Any], *, reason: str,
            distance: int | None = None) -> None:
        first, second = sorted((left, right), key=lambda row: row["image_id"])
        key = first["image_id"], second["image_id"]
        candidate = pairs.setdefault(key, {"left": first, "right": second, "reasons": set(), "distance": None})
        candidate["reasons"].add(reason)
        if distance is not None:
            previous = candidate["distance"]
            candidate["distance"] = distance if previous is None else min(previous, distance)

    exact_bytes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_pixels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        exact_bytes[row["original_image_sha256"]].append(row)
        exact_pixels[row["rgb_pixel_sha256"]].append(row)
        by_lens[int(row["objective_lens"])].append(row)
    for group in exact_bytes.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                add(left, right, reason="exact_bytes")
    for group in exact_pixels.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                add(left, right, reason="exact_pixels")
    for lens_rows in by_lens.values():
        tree = _BKTree()
        for row in lens_rows:
            value = int(row["phash_hex"], 16)
            for previous, distance in tree.query(value, threshold):
                add(previous, row, reason="phash_near", distance=distance)
            tree.add(value, row)

    output: list[dict[str, Any]] = []
    for candidate in pairs.values():
        left, right = candidate["left"], candidate["right"]
        same_lens = int(left["objective_lens"]) == int(right["objective_lens"])
        reasons = candidate["reasons"]
        exact_duplicate = bool({"exact_bytes", "exact_pixels"} & reasons)
        target_disagreement = int(left["image_target"]) != int(right["image_target"])
        target_conflict = exact_duplicate and target_disagreement
        patient_relation = "same_patient" if left["patient_id"] == right["patient_id"] else "cross_patient"
        blocks_ready = patient_relation == "cross_patient" or target_conflict
        body = {
            "image_ids": sorted((left["image_id"], right["image_id"])),
            "reasons": sorted(reasons),
            "phash_hamming_distance": candidate["distance"],
        }
        output.append({
            "candidate_id": fingerprint(body),
            "image_id_left": left["image_id"],
            "image_id_right": right["image_id"],
            "objective_lens_left": left["objective_lens"],
            "objective_lens_right": right["objective_lens"],
            "same_lens": same_lens,
            "candidate_type": "exact_duplicate" if exact_duplicate else "near_similarity",
            "patient_relation": patient_relation,
            "target_disagreement": target_disagreement,
            "target_conflict": target_conflict,
            "image_target_left": left["image_target"],
            "image_target_right": right["image_target"],
            "exact_byte_duplicate": "exact_bytes" in reasons,
            "exact_pixel_duplicate": "exact_pixels" in reasons,
            "review_required": True,
            "blocks_ready_for_fit": blocks_ready,
            "phash_hamming_distance": candidate["distance"],
            "candidate_reasons": ";".join(sorted(reasons)),
        })
    return sorted(output, key=lambda row: (row["image_id_left"], row["image_id_right"]))


def select_representative_images(
    records: Sequence[dict[str, Any]], *, limit: int = 48,
) -> list[dict[str, Any]]:
    """Select a deterministic technical review sample balanced by lens and grade.

    The round-robin selection spreads picks across available lens/grade groups,
    then prefers patients with fewer already selected images. Duplicate and QC
    measurements are never used to exclude or favor an image.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["objective_lens"]), str(row["raw_glade"]))].append(dict(row))
    for rows in groups.values():
        rows.sort(key=lambda row: (row["patient_id"], row["image_id"]))
    keys = [(lens, grade) for lens in LENSES for grade in GRADE_ORDER if groups[(lens, grade)]]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    patient_counts: dict[str, int] = defaultdict(int)
    while len(selected) < limit:
        made_progress = False
        for key in keys:
            if len(selected) >= limit:
                break
            candidates = [row for row in groups[key] if row["image_id"] not in selected_ids]
            if not candidates:
                continue
            row = min(candidates, key=lambda item: (patient_counts[item["patient_id"]], item["image_id"]))
            selected.append(row)
            selected_ids.add(row["image_id"])
            patient_counts[row["patient_id"]] += 1
            made_progress = True
        if not made_progress:
            break
    return selected


def _summary(records: list[dict[str, Any]], candidates: list[dict[str, Any]],
             selection: list[dict[str, Any]], *, threshold: int) -> dict[str, Any]:
    cross_patient = sum(row["patient_relation"] == "cross_patient" for row in candidates)
    target_conflicts = sum(bool(row["target_conflict"]) for row in candidates)
    target_disagreements = sum(bool(row["target_disagreement"]) for row in candidates)
    blocking_candidates = sum(bool(row["blocks_ready_for_fit"]) for row in candidates)
    exact_candidates = sum(row["candidate_type"] == "exact_duplicate" for row in candidates)
    near_candidates = len(candidates) - exact_candidates
    gate_reasons = []
    if cross_patient:
        gate_reasons.append("cross_patient_duplicate_candidates_require_review")
    if target_conflicts:
        gate_reasons.append("duplicate_target_conflicts_require_review")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol": AUDIT_PROTOCOL,
        "state": "attention_required" if gate_reasons else "complete",
        "audit_complete": True,
        "training_ready": not gate_reasons,
        "ready_for_fit": not gate_reasons,
        "candidate_review_required": bool(candidates),
        "blocking_review_required": bool(gate_reasons),
        "qc_is_descriptive_only": True,
        "automatic_exclusion_count": 0,
        "automatic_relabel_count": 0,
        "image_count": len(records),
        "duplicate_candidate_count": len(candidates),
        "review_logged_candidate_count": len(candidates),
        "blocking_candidate_count": blocking_candidates,
        "exact_duplicate_candidate_count": exact_candidates,
        "near_similarity_candidate_count": near_candidates,
        "cross_patient_candidate_count": cross_patient,
        "target_conflict_candidate_count": target_conflicts,
        "target_disagreement_candidate_count": target_disagreements,
        "near_similarity_target_disagreement_count": sum(
            row["candidate_type"] == "near_similarity" and row["target_disagreement"]
            for row in candidates
        ),
        "same_patient_candidate_count": len(candidates) - cross_patient,
        "phash_hamming_threshold": threshold,
        "gate_reasons": gate_reasons,
        "target_conflict_semantics": "label disagreement is a conflict only for exact byte or decoded RGB-pixel duplicates; pHash similarity alone is not proof of a label conflict",
        "candidate_review_semantics": "all candidates are logged; cross-patient candidates and exact-duplicate label conflicts block fit readiness; within-patient pHash-only label disagreements remain descriptive",
        "representative_selection_count": len(selection),
        "representative_selection_requested": 48,
        "representative_selection_purpose": "technical_review_only; does_not_change_training_inclusion",
        "qc_fields": {
            "qc_mean_brightness": "mean Rec.709 luminance in [0,1]",
            "qc_mean_saturation": "mean HSV saturation in [0,1]",
            "qc_tissue_fraction_estimate": "fraction with luminance < 0.92 and saturation > 0.05, or luminance < 0.75",
            "qc_laplacian_variance": "variance of four-neighbor luminance Laplacian; descriptive sharpness proxy",
        },
    }


def _notify(notify: Notify | None, *, state: str, processed: int, expected: int,
            run_root: Path) -> None:
    if notify is not None:
        notify({"state": state, "processed_images": processed,
                "expected_images": expected, "output_path": str(run_root)})


def _verify_final_commit(run_root: Path, *, audit_id: str,
                         source_identity: dict[str, Any],
                         binding_id: str, chunk_commit_ids: dict[int, str],
    image_count: int) -> Path | None:
    path = run_root / "audit.commit.json"
    if path.is_symlink():
        raise ValueError("Refusing a symlinked final image-audit commit.")
    if not path.is_file():
        return None
    try:
        commit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unreadable final image-audit commit.") from exc
    body = {key: value for key, value in commit.items() if key != "commit_id"}
    if (commit.get("commit_id") != fingerprint(body)
            or commit.get("audit_id") != audit_id
            or commit.get("source_identity") != source_identity
            or commit.get("binding_id") != binding_id
            or commit.get("image_count") != image_count
            or commit.get("chunk_commit_ids") != {str(index): value for index, value in sorted(chunk_commit_ids.items())}
            or commit.get("complete") is not True):
        raise ValueError("Final image-audit commit identity is invalid.")
    expected_files = {
        "audit.json", "binding.json", "records.csv", "duplicate_candidates.csv",
        "representative_selection.csv", "summary.json", "status.json",
    }
    files = commit.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("Final image-audit commit has an unexpected output schema.")
    for filename, expected_sha256 in files.items():
        file_path = run_root / filename
        if file_path.is_symlink() or not file_path.is_file() or file_hash(file_path) != expected_sha256:
            raise ValueError(f"Final image-audit output failed checksum verification: {filename}")
    return run_root


def audit_original_images(
    inventory_path: Path,
    archives: Mapping[int, Path],
    output_root: Path,
    *,
    verified_feature_rows: Sequence[Mapping[str, Any]] | None = None,
    budget_minutes: float = 240.0,
    reserve_minutes: float = 15.0,
    recover_lock: bool = False,
    phash_hamming_threshold: int = 2,
    thumbnail_size: int = 32,
    hash_size: int = 8,
    max_image_bytes: int = 512 * 1024 * 1024,
    max_image_pixels: int = 178_000_000,
    notify: Notify | None = None,
) -> ImageAuditResult:
    """Audit every inventoried original and commit resumable technical evidence.

    All three source ZIPs are fully SHA-256 checked before creating or changing
    any audit output. Budget expiry preserves only completed 32-image chunks;
    QC values and duplicate candidates never modify labels or inclusion.
    """
    inventory = core._load_inventory(Path(inventory_path))
    for name, value in (("budget_minutes", budget_minutes), ("reserve_minutes", reserve_minutes)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if budget_minutes <= 0 or reserve_minutes < 0 or reserve_minutes >= budget_minutes:
        raise ValueError("budget_minutes must be positive and exceed reserve_minutes.")
    if not isinstance(recover_lock, bool):
        raise ValueError("recover_lock must be boolean.")
    if isinstance(phash_hamming_threshold, bool) or not isinstance(phash_hamming_threshold, int) or phash_hamming_threshold < 0:
        raise ValueError("phash_hamming_threshold must be a nonnegative integer.")
    if isinstance(thumbnail_size, bool) or not isinstance(thumbnail_size, int) or thumbnail_size < 8:
        raise ValueError("thumbnail_size must be an integer >= 8.")
    if isinstance(hash_size, bool) or not isinstance(hash_size, int) or not 2 <= hash_size <= thumbnail_size:
        raise ValueError("hash_size must be an integer in [2,thumbnail_size].")
    if hash_size * hash_size > 64:
        raise ValueError("hash_size may not exceed 8 because candidate hashes use 64-bit Hamming distance.")
    if phash_hamming_threshold >= hash_size * hash_size:
        raise ValueError("phash_hamming_threshold must be smaller than the pHash bit count.")
    core._positive_int(max_image_bytes, "max_image_bytes")
    core._positive_int(max_image_pixels, "max_image_pixels")
    resolved_archives = _validate_source_archives(inventory, archives)
    feature_bindings, feature_rows_sha256 = _verified_feature_bindings(
        verified_feature_rows, inventory,
    )

    source_identity = _source_identity(inventory)
    descriptor = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol": AUDIT_PROTOCOL,
        "inventory_id": inventory["inventory_id"],
        "source_identity": source_identity,
        "chunk_size": CHUNK_SIZE,
        "phash_hamming_threshold": phash_hamming_threshold,
        "thumbnail_size": thumbnail_size,
        "hash_size": hash_size,
        "max_image_bytes": max_image_bytes,
        "max_image_pixels": max_image_pixels,
        "verified_feature_rows_sha256": feature_rows_sha256,
    }
    audit_id = fingerprint(descriptor)
    binding = _make_binding(
        audit_id, inventory, descriptor, _source_identity(inventory), feature_rows_sha256,
    )
    source_identity = _source_identity(inventory)
    run_root = Path(output_root).expanduser().resolve() / "runs" / audit_id
    if run_root.is_symlink():
        raise ValueError("Refusing a symlinked image-audit run directory.")
    images = list(inventory["images"])
    expected_ids = [row["image_id"] for row in images]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Inventory contains duplicate image IDs.")
    selection = select_representative_images(images, limit=48)
    started = time.monotonic()
    deadline = started + (budget_minutes - reserve_minutes) * 60.0

    with _RunLock(run_root, recover=recover_lock):
        manifest_path = run_root / "audit.json"
        if manifest_path.exists():
            if manifest_path.is_symlink():
                raise ValueError("Refusing a symlinked image-audit manifest.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("audit_id") != audit_id or manifest.get("descriptor") != descriptor:
                raise ValueError("Image-audit output belongs to a different source or configuration.")
        else:
            if any(path.name != "audit.lock" for path in run_root.iterdir()):
                raise ValueError("Image-audit output exists without its identity manifest; refusing overwrite.")
            atomic_json(manifest_path, {
                "audit_id": audit_id,
                "inventory_id": inventory["inventory_id"],
                "descriptor": descriptor,
                "image_count": len(images),
                "image_ids": expected_ids,
                "created_at": core._utc(),
            })
        binding = _load_binding(run_root, binding)
        committed, commit_ids = _load_committed_chunks(
            run_root, audit_id=audit_id, inventory_id=inventory["inventory_id"],
            source_identity=source_identity, expected_ids=expected_ids,
        )
        committed_count = sum(len(rows) for rows in committed.values())
        verified_final = _verify_final_commit(
            run_root, audit_id=audit_id, source_identity=source_identity,
            binding_id=binding["binding_id"],
            chunk_commit_ids=commit_ids, image_count=committed_count,
        )
        if verified_final is not None:
            return _result(verified_final, binding, selection)
        _write_text_atomic(
            run_root / "representative_selection.csv",
            _csv_text([
                {"selection_order": index, **{key: row[key] for key in SELECTION_FIELDS if key != "selection_order"}}
                for index, row in enumerate(selection, start=1)
            ], SELECTION_FIELDS),
        )
        atomic_json(run_root / "status.json", {
            "state": "running",
            "audit_id": audit_id,
            "inventory_id": inventory["inventory_id"],
            "expected_image_count": len(images),
            "committed_image_count": committed_count,
            "remaining_image_count": len(images) - committed_count,
            "audit_complete": False,
            "training_ready": False,
            "ready_for_fit": False,
            "duplicate_analysis_complete": False,
            "duplicate_candidate_count": None,
            "candidate_review_required": None,
            "blocking_review_required": None,
            "target_conflict_semantics": "label disagreement is a conflict only for exact byte or decoded RGB-pixel duplicates; pHash similarity alone is not proof of a label conflict",
            "updated_at": core._utc(),
        })
        _notify(notify, state="running", processed=committed_count,
                expected=len(images), run_root=run_root)

        try:
            with ExitStack() as stack:
                zip_archives = {
                    lens: stack.enter_context(zipfile.ZipFile(resolved_archives[lens], "r"))
                    for lens in LENSES
                }
                chunk_count = math.ceil(len(images) / CHUNK_SIZE)
                for chunk_index in range(chunk_count):
                    if chunk_index in committed:
                        continue
                    selected = images[chunk_index * CHUNK_SIZE:(chunk_index + 1) * CHUNK_SIZE]
                    chunk_records: list[dict[str, Any]] = []
                    try:
                        for row in selected:
                            if time.monotonic() >= deadline:
                                raise _BudgetExhausted
                            record = _audit_image(
                                zip_archives[int(row["objective_lens"])], row,
                                max_image_bytes=max_image_bytes,
                                max_image_pixels=max_image_pixels,
                                thumbnail_size=thumbnail_size,
                                hash_size=hash_size,
                                verified_feature_row=(feature_bindings or {}).get(row["image_id"]),
                            )
                            chunk_records.append(record)
                        if time.monotonic() >= deadline:
                            raise _BudgetExhausted
                    except _BudgetExhausted:
                        raise
                    chunk_commit = _commit_chunk(
                        run_root, chunk_index, chunk_records,
                        audit_id=audit_id, inventory_id=inventory["inventory_id"],
                        source_identity=source_identity,
                    )
                    committed[chunk_index] = chunk_records
                    commit_ids[chunk_index] = chunk_commit["commit_id"]
                    committed_count += len(chunk_records)
                    atomic_json(run_root / "status.json", {
                        "state": "running",
                        "audit_id": audit_id,
                        "inventory_id": inventory["inventory_id"],
                        "expected_image_count": len(images),
                        "committed_image_count": committed_count,
                        "remaining_image_count": len(images) - committed_count,
                        "audit_complete": False,
                        "training_ready": False,
                        "ready_for_fit": False,
                        "duplicate_analysis_complete": False,
                        "duplicate_candidate_count": None,
                        "candidate_review_required": None,
                        "blocking_review_required": None,
                        "target_conflict_semantics": "label disagreement is a conflict only for exact byte or decoded RGB-pixel duplicates; pHash similarity alone is not proof of a label conflict",
                        "last_committed_chunk": chunk_index,
                        "updated_at": core._utc(),
                    })
                    _notify(notify, state="running", processed=committed_count,
                            expected=len(images), run_root=run_root)
        except _BudgetExhausted:
            atomic_json(run_root / "status.json", {
                "state": "partial",
                "audit_id": audit_id,
                "inventory_id": inventory["inventory_id"],
                "expected_image_count": len(images),
                "committed_image_count": committed_count,
                "remaining_image_count": len(images) - committed_count,
                "audit_complete": False,
                "training_ready": False,
                "ready_for_fit": False,
                "duplicate_analysis_complete": False,
                "duplicate_candidate_count": None,
                "candidate_review_required": None,
                "blocking_review_required": None,
                "target_conflict_semantics": "label disagreement is a conflict only for exact byte or decoded RGB-pixel duplicates; pHash similarity alone is not proof of a label conflict",
                "stop_reason": "budget",
                "updated_at": core._utc(),
            })
            _notify(notify, state="partial", processed=committed_count,
                    expected=len(images), run_root=run_root)
            return _result(run_root, binding, selection)
        except Exception as exc:
            atomic_json(run_root / "status.json", {
                "state": "failed",
                "audit_id": audit_id,
                "inventory_id": inventory["inventory_id"],
                "expected_image_count": len(images),
                "committed_image_count": committed_count,
                "remaining_image_count": len(images) - committed_count,
                "audit_complete": False,
                "training_ready": False,
                "ready_for_fit": False,
                "duplicate_analysis_complete": False,
                "duplicate_candidate_count": None,
                "candidate_review_required": None,
                "blocking_review_required": None,
                "target_conflict_semantics": "label disagreement is a conflict only for exact byte or decoded RGB-pixel duplicates; pHash similarity alone is not proof of a label conflict",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": core._utc(),
            })
            raise

        records = [record for index in sorted(committed) for record in committed[index]]
        if [record["image_id"] for record in records] != expected_ids:
            raise ValueError("Committed audit records do not exactly cover the verified inventory.")
        candidates = _duplicate_candidates(records, threshold=phash_hamming_threshold)
        record_selection = select_representative_images(records, limit=48)
        if [row["image_id"] for row in record_selection] != [row["image_id"] for row in selection]:
            raise ValueError("Representative selection changed between inventory and decoded audit rows.")
        summary = _summary(records, candidates, selection, threshold=phash_hamming_threshold)
        _write_text_atomic(run_root / "records.csv", _csv_text(records, RECORD_FIELDS))
        _write_text_atomic(run_root / "duplicate_candidates.csv", _csv_text(candidates, CANDIDATE_FIELDS))
        _write_text_atomic(
            run_root / "representative_selection.csv",
            _csv_text([
                {"selection_order": index, **{key: row[key] for key in SELECTION_FIELDS if key != "selection_order"}}
                for index, row in enumerate(selection, start=1)
            ], SELECTION_FIELDS),
        )
        atomic_json(run_root / "summary.json", summary)
        atomic_json(run_root / "status.json", {
            "state": summary["state"],
            "audit_id": audit_id,
            "inventory_id": inventory["inventory_id"],
            "expected_image_count": len(images),
            "committed_image_count": len(records),
            "remaining_image_count": 0,
            "audit_complete": True,
            "training_ready": summary["training_ready"],
            "ready_for_fit": summary["ready_for_fit"],
            "duplicate_analysis_complete": True,
            "duplicate_candidate_count": summary["duplicate_candidate_count"],
            "cross_patient_candidate_count": summary["cross_patient_candidate_count"],
            "exact_duplicate_candidate_count": summary["exact_duplicate_candidate_count"],
            "near_similarity_candidate_count": summary["near_similarity_candidate_count"],
            "target_conflict_candidate_count": summary["target_conflict_candidate_count"],
            "target_disagreement_candidate_count": summary["target_disagreement_candidate_count"],
            "candidate_review_required": summary["candidate_review_required"],
            "blocking_review_required": summary["blocking_review_required"],
            "target_conflict_semantics": summary["target_conflict_semantics"],
            "candidate_review_semantics": summary["candidate_review_semantics"],
            "gate_reasons": summary["gate_reasons"],
            "updated_at": core._utc(),
        })
        files = {
            name: file_hash(run_root / name)
            for name in (
                "audit.json", "binding.json", "records.csv", "duplicate_candidates.csv",
                "representative_selection.csv", "summary.json", "status.json",
            )
        }
        commit_body = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_id": audit_id,
            "inventory_id": inventory["inventory_id"],
            "source_identity": source_identity,
            "binding_id": binding["binding_id"],
            "image_count": len(records),
            "chunk_commit_ids": {str(index): commit_ids[index] for index in sorted(commit_ids)},
            "files": files,
            "complete": True,
            "state": summary["state"],
        }
        atomic_json(run_root / "audit.commit.json", {
            **commit_body,
            "commit_id": fingerprint(commit_body),
        })
        _notify(notify, state=summary["state"], processed=len(records),
                expected=len(images), run_root=run_root)
        _verify_final_commit(
            run_root, audit_id=audit_id, source_identity=source_identity,
            binding_id=binding["binding_id"],
            chunk_commit_ids=commit_ids, image_count=len(records),
        )
        return _result(run_root, binding, selection)
