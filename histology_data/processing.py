"""Bounded CPU tiling and heuristic H&E QC; tissue thresholds are not clinically calibrated."""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io import atomic_json, file_hash, fingerprint, verified_copy
from .shards import load_shard_manifest, stage_shard

TISSUE_METHOD = "normalized_od_he_contrast_v1"
TISSUE_METHOD_CALIBRATION = "heuristic_not_clinically_calibrated"
DEFAULT_CONFIG: dict[str, int | float | str] = {
    "tile_size": 256,
    "stride": 256,
    "min_tissue": 0.2,
    "max_tiles_per_image": 0,
    "blur_threshold": 0.0,
    "tissue_method": TISSUE_METHOD,
    "tissue_od_mean_threshold": 0.05,
    "tissue_green_contrast_threshold": 0.03,
    "background_neutral_chroma_threshold": 0.2,
    "background_reference_percentile": 99.0,
}
PROCESSING_VERSION = "cpu-tiles-v3"
_MAX_TILE_SIZE = 4096
_BACKGROUND_SAMPLE_LIMIT = 4096
_IMAGE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_LENSES = {4, 10, 40}


def _validated_config(config: dict[str, Any] | None) -> dict[str, int | float | str]:
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None.")
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown processing configuration fields: {sorted(unknown)}")
    values: dict[str, Any] = {**DEFAULT_CONFIG, **supplied}
    for name in ("tile_size", "stride", "max_tiles_per_image"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if not 1 <= values["tile_size"] <= _MAX_TILE_SIZE:
        raise ValueError(f"tile_size must be between 1 and {_MAX_TILE_SIZE}.")
    if not 1 <= values["stride"] <= _MAX_TILE_SIZE:
        raise ValueError(f"stride must be between 1 and {_MAX_TILE_SIZE}.")
    if values["max_tiles_per_image"] < 0:
        raise ValueError("max_tiles_per_image cannot be negative.")
    for name in (
        "min_tissue", "blur_threshold", "tissue_od_mean_threshold",
        "tissue_green_contrast_threshold", "background_neutral_chroma_threshold",
        "background_reference_percentile",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if values["tissue_method"] != TISSUE_METHOD:
        raise ValueError(f"tissue_method must be {TISSUE_METHOD!r}.")
    if not 0.0 <= values["min_tissue"] <= 1.0:
        raise ValueError("min_tissue must be in [0, 1].")
    if values["blur_threshold"] < 0.0:
        raise ValueError("blur_threshold cannot be negative.")
    if not 0.0 <= values["tissue_od_mean_threshold"] <= 5.0:
        raise ValueError("tissue_od_mean_threshold must be in [0, 5].")
    if not 0.0 <= values["tissue_green_contrast_threshold"] <= 5.0:
        raise ValueError("tissue_green_contrast_threshold must be in [0, 5].")
    if not 0.0 <= values["background_neutral_chroma_threshold"] <= 3.0:
        raise ValueError("background_neutral_chroma_threshold must be in [0, 3].")
    if not 50.0 <= values["background_reference_percentile"] <= 100.0:
        raise ValueError("background_reference_percentile must be in [50, 100].")
    return {
        "tile_size": int(values["tile_size"]),
        "stride": int(values["stride"]),
        "min_tissue": float(values["min_tissue"]),
        "max_tiles_per_image": int(values["max_tiles_per_image"]),
        "blur_threshold": float(values["blur_threshold"]),
        "tissue_method": str(values["tissue_method"]),
        "tissue_od_mean_threshold": float(values["tissue_od_mean_threshold"]),
        "tissue_green_contrast_threshold": float(values["tissue_green_contrast_threshold"]),
        "background_neutral_chroma_threshold": float(values["background_neutral_chroma_threshold"]),
        "background_reference_percentile": float(values["background_reference_percentile"]),
    }


def _manifest_images(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Committed shard manifest must contain at least one image.")
    seen_ids: set[str] = set()
    seen_members: set[str] = set()
    checked: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Invalid image entry in shard manifest.")
        image_id = image.get("image_id")
        member = image.get("member")
        sha256 = image.get("sha256")
        if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id) or image_id in {".", ".."}:
            raise ValueError(f"Unsafe or missing image_id in shard manifest: {image_id!r}")
        if not isinstance(member, str):
            raise ValueError(f"Missing source member for {image_id}.")
        path = PurePosixPath(member)
        if (path.is_absolute() or ".." in path.parts or "\\" in member
                or path.as_posix() != member or len(path.parts) != 2 or path.parts[0] != "images"):
            raise ValueError(f"Unsafe source member in shard manifest: {member!r}")
        if image_id in seen_ids or member in seen_members:
            raise ValueError("Duplicate image ID or member in shard manifest.")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Invalid source checksum for {image_id}.")
        lens = image.get("objective_lens")
        if isinstance(lens, bool) or not isinstance(lens, int) or lens not in _LENSES:
            raise ValueError(f"Unsupported objective lens for {image_id}: {lens!r}")
        if not isinstance(image.get("case_id"), str) or not image["case_id"].strip():
            raise ValueError(f"Missing case provenance for {image_id}.")
        if not isinstance(image.get("slide_group_id"), str) or not image["slide_group_id"].strip():
            raise ValueError(f"Missing bag provenance for {image_id}.")
        patient_id = image.get("patient_id")
        if patient_id is not None and (not isinstance(patient_id, str) or not patient_id.strip()):
            raise ValueError(f"Invalid patient provenance for {image_id}.")
        byte_size = image.get("byte_size")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 1:
            raise ValueError(f"Invalid source size for {image_id}.")
        seen_ids.add(image_id)
        seen_members.add(member)
        checked.append(image)
    return sorted(checked, key=lambda item: item["image_id"])


def _image_path(stage_root: Path, member: str) -> Path:
    root = stage_root.resolve()
    path = stage_root.joinpath(*PurePosixPath(member).parts)
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f"Staged image is missing or unsafe: {member}")
    return path


def _background_reference(rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Estimate a bright neutral field background from at most 4096 grid samples."""
    height, width = rgb.shape[:2]
    pixel_count = height * width
    sample_count = min(pixel_count, _BACKGROUND_SAMPLE_LIMIT)
    sample_indices = np.linspace(0, pixel_count - 1, num=sample_count, dtype=np.int64)
    sample = rgb.reshape(-1, 3)[sample_indices].astype(np.float32)
    maximum = sample.max(axis=1)
    minimum = sample.min(axis=1)
    mean = sample.mean(axis=1)
    chroma = (maximum - minimum) / np.maximum(mean, 1.0)
    white_od = np.log(256.0 / (sample + 1.0))
    he_evidence = _he_evidence_mask(white_od, config)
    neutral = sample[
        (chroma <= float(config["background_neutral_chroma_threshold"])) & ~he_evidence
    ]
    if neutral.size == 0:
        return np.full(3, 255.0, dtype=np.float32)
    reference = np.percentile(
        neutral,
        float(config["background_reference_percentile"]),
        axis=0,
    )
    return np.clip(reference, 1.0, 255.0).astype(np.float32)


def _tissue_count(
    rgb: np.ndarray, reference_rgb: np.ndarray, config: dict[str, Any],
    y0: int, y1: int, x0: int = 0, x1: int | None = None,
) -> int:
    """Count H&E-like pixels by background-normalized OD and green stain evidence."""
    width = rgb.shape[1]
    x_end = width if x1 is None else x1
    block = rgb[y0:y1, x0:x_end].astype(np.float32)
    optical_density = (reference_rgb.reshape(1, 1, 3) + 1.0) / (block + 1.0)
    np.log(optical_density, out=optical_density)
    np.maximum(optical_density, 0.0, out=optical_density)
    mask = _he_evidence_mask(optical_density, config)
    return int(np.count_nonzero(mask))


def _he_evidence_mask(optical_density: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Require meaningful normalized OD and green stain absorbance above red/blue."""
    od_mean = optical_density.mean(axis=-1)
    green_contrast = optical_density[..., 1] - np.maximum(
        optical_density[..., 0], optical_density[..., 2],
    )
    return (
        (od_mean >= float(config["tissue_od_mean_threshold"]))
        & (green_contrast >= float(config["tissue_green_contrast_threshold"]))
    )


def _tissue_fraction(rgb: np.ndarray, reference_rgb: np.ndarray,
                     config: dict[str, Any]) -> float:
    height, width = rgb.shape[:2]
    total = 0
    for y0 in range(0, height, 256):
        y1 = min(height, y0 + 256)
        total += _tissue_count(rgb, reference_rgb, config, y0, y1)
    return total / (height * width)


def _gray_rows(rgb: np.ndarray, y0: int, y1: int) -> np.ndarray:
    block = rgb[y0:y1]
    weighted = (block[..., 0].astype(np.uint16) * 77
                + block[..., 1].astype(np.uint16) * 150
                + block[..., 2].astype(np.uint16) * 29 + 128)
    return (weighted >> 8).astype(np.uint8)


def _focus_score(rgb: np.ndarray) -> float:
    """Return variance of the 4-neighbor Laplacian using row-bounded buffers."""
    height, width = rgb.shape[:2]
    count = 0
    total = 0.0
    total_squared = 0.0
    for y0 in range(0, height, 128):
        y1 = min(height, y0 + 128)
        source_y0 = max(0, y0 - 1)
        source_y1 = min(height, y1 + 1)
        gray = _gray_rows(rgb, source_y0, source_y1)
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        laplacian = (
            padded[2:, 1:-1].astype(np.float32)
            + padded[:-2, 1:-1].astype(np.float32)
            + padded[1:-1, 2:].astype(np.float32)
            + padded[1:-1, :-2].astype(np.float32)
            - 4.0 * padded[1:-1, 1:-1].astype(np.float32)
        )
        offset = y0 - source_y0
        valid = laplacian[offset:offset + (y1 - y0)]
        count += valid.size
        total += float(np.sum(valid, dtype=np.float64))
        total_squared += float(np.sum(valid * valid, dtype=np.float64))
    if count == 0:
        return 0.0
    mean = total / count
    return max(0.0, total_squared / count - mean * mean)


def _content_tile_id(
    image_sha256: str, x: int, y: int, width: int, height: int, config_id: str,
) -> str:
    return fingerprint({"image_sha256": image_sha256, "x": x, "y": y,
                        "w": width, "h": height, "config_id": config_id})


def _tile_id(
    image_id: str, image_sha256: str, x: int, y: int, width: int, height: int,
    config_id: str,
) -> str:
    """Identify a source tile instance, including its stable catalog identity."""
    return fingerprint({"image_id": image_id, "image_sha256": image_sha256,
                        "x": x, "y": y, "w": width, "h": height,
                        "config_id": config_id})


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def _jsonl_write(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _iter_tiles(
    image: dict[str, Any], rgb: np.ndarray, reference_rgb: np.ndarray,
    config: dict[str, Any],
    image_sha256: str, config_id: str,
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    height, width = rgb.shape[:2]
    tile_size = int(config["tile_size"])
    stride = int(config["stride"])
    minimum = float(config["min_tissue"])
    cap = int(config["max_tiles_per_image"])
    emitted = 0
    for y in range(0, height, stride):
        valid_h = min(tile_size, height - y)
        for x in range(0, width, stride):
            valid_w = min(tile_size, width - x)
            tissue = _tissue_count(
                rgb, reference_rgb, config, y, y + valid_h, x, x + valid_w,
            ) / (valid_h * valid_w)
            if tissue <= 0.0 or tissue < minimum:
                continue
            if cap and emitted >= cap:
                continue
            if valid_w == tile_size and valid_h == tile_size:
                tile = np.ascontiguousarray(rgb[y:y + tile_size, x:x + tile_size])
            else:
                tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                tile[:valid_h, :valid_w] = rgb[y:y + valid_h, x:x + valid_w]
            content_identifier = _content_tile_id(
                image_sha256, x, y, tile_size, tile_size, config_id,
            )
            identifier = _tile_id(
                image["image_id"], image_sha256, x, y, tile_size, tile_size, config_id,
            )
            row = {
                "tile_id": identifier,
                "content_tile_id": content_identifier,
                "patch_member": f"patches/{identifier}.png",
                "image_id": image["image_id"],
                "case_id": image.get("case_id"),
                "patient_id": image.get("patient_id"),
                "bag_id": image.get("slide_group_id"),
                "slide_group_id": image.get("slide_group_id"),
                "objective_lens": image["objective_lens"],
                "x": x,
                "y": y,
                "w": tile_size,
                "h": tile_size,
                "valid_w": valid_w,
                "valid_h": valid_h,
                "tissue_fraction": tissue,
                "source_sha256": image_sha256,
            }
            emitted += 1
            yield row, tile


def _encode_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _process_images(stage_root: Path, manifest: dict[str, Any], config: dict[str, Any],
                    config_id: str, patches_path: Path, tiles_path: Path,
                    qc_path: Path) -> dict[str, int]:
    stats = {"images_seen": 0, "tiles_written": 0, "images_review": 0,
             "images_duplicate_source": 0,
             "images_no_tissue": 0, "tiles_below_tissue_threshold": 0}
    images = _manifest_images(manifest)
    duplicate_source_ids: dict[str, list[str]] = {}
    source_ids_by_hash: dict[str, list[str]] = {}
    for image in images:
        source_ids_by_hash.setdefault(image["sha256"], []).append(image["image_id"])
    for image_sha256, image_ids in source_ids_by_hash.items():
        if len(image_ids) > 1:
            duplicate_source_ids[image_sha256] = sorted(image_ids)
    with tarfile.open(patches_path, mode="w", format=tarfile.PAX_FORMAT) as patches, \
            tiles_path.open("w", encoding="utf-8", newline="\n") as tiles, \
            qc_path.open("w", encoding="utf-8", newline="\n") as qc:
        for image in images:
            stats["images_seen"] += 1
            path = _image_path(stage_root, image["member"])
            actual_sha = file_hash(path)
            if actual_sha != image["sha256"]:
                raise ValueError(f"Staged source checksum mismatch: {image['image_id']}")
            expected_size = image.get("byte_size")
            if expected_size is not None and path.stat().st_size != expected_size:
                raise ValueError(f"Staged source size mismatch: {image['image_id']}")
            try:
                with Image.open(path) as source:
                    source_mode = source.mode
                    frame_count = getattr(source, "n_frames", 1)
                    if frame_count != 1:
                        raise ValueError(f"Multi-frame image is unsupported: {image['image_id']}")
                    source.load()
                    source_width, source_height = source.size
                    if source_width < 1 or source_height < 1:
                        raise ValueError(f"Invalid image geometry: {image['image_id']}")
                    if "A" in source.getbands() or "transparency" in source.info:
                        rgba = source.convert("RGBA")
                        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                        normalized = Image.alpha_composite(white, rgba).convert("RGB")
                    else:
                        normalized = source.convert("RGB")
                    rgb = np.asarray(normalized, dtype=np.uint8).copy()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValueError(f"Failed to decode source image {image['image_id']}: {exc}") from exc
            if rgb.ndim != 3 or rgb.shape != (source_height, source_width, 3):
                raise ValueError(f"Unexpected RGB geometry for {image['image_id']}")
            background_reference = _background_reference(rgb, config)
            tissue = _tissue_fraction(rgb, background_reference, config)
            focus = _focus_score(rgb)
            reasons: list[str] = []
            if tissue <= 0.0:
                reasons.append("no_tissue")
                stats["images_no_tissue"] += 1
            repeated_image_ids = duplicate_source_ids.get(actual_sha, [])
            if repeated_image_ids:
                reasons.append("duplicate_source_content")
                stats["images_duplicate_source"] += 1
            if float(config["blur_threshold"]) > 0 and focus < float(config["blur_threshold"]):
                reasons.append("focus_below_review_threshold")
            eligible = 0
            saved = 0
            for row, tile in _iter_tiles(
                image, rgb, background_reference, config, actual_sha, config_id,
            ):
                eligible += 1
                member = f"patches/{row['tile_id']}.png"
                encoded = _encode_png(tile)
                patches.addfile(_tar_info(member, len(encoded)), io.BytesIO(encoded))
                _jsonl_write(tiles, row)
                stats["tiles_written"] += 1
                saved += 1
            max_per_image = int(config["max_tiles_per_image"])
            if max_per_image and eligible == max_per_image:
                # Count additional candidates without retaining patches or pixel buffers.
                cap_candidates = _eligible_tile_count(rgb, background_reference, config)
                eligible = cap_candidates
            if eligible == 0:
                reasons.append("no_tiles_met_tissue_threshold")
            if reasons:
                stats["images_review"] += 1
            rejected = max(0, _grid_tile_count(source_width, source_height, config) - eligible)
            stats["tiles_below_tissue_threshold"] += rejected
            _jsonl_write(qc, {
                "image_id": image["image_id"],
                "case_id": image.get("case_id"),
                "patient_id": image.get("patient_id"),
                "bag_id": image.get("slide_group_id"),
                "source_sha256": actual_sha,
                "duplicate_source_content": bool(repeated_image_ids),
                "duplicate_source_image_ids": repeated_image_ids,
                "tissue_method": config["tissue_method"],
                "tissue_method_calibration": TISSUE_METHOD_CALIBRATION,
                "background_reference_rgb": [float(value) for value in background_reference],
                "tissue_od_mean_threshold": config["tissue_od_mean_threshold"],
                "tissue_green_contrast_threshold": config["tissue_green_contrast_threshold"],
                "background_neutral_chroma_threshold": config["background_neutral_chroma_threshold"],
                "background_reference_percentile": config["background_reference_percentile"],
                "source_mode": source_mode,
                "output_mode": "RGB",
                "width": source_width,
                "height": source_height,
                "objective_lens": image["objective_lens"],
                "decode_ok": True,
                "geometry_ok": True,
                "tissue_fraction": tissue,
                "focus_score": focus,
                "tiles_eligible": eligible,
                "tiles_written": saved,
                "tiles_below_tissue_threshold": rejected,
                "status": "review" if reasons else "pass",
                "reasons": reasons,
            })
            del rgb
    return stats


def _grid_tile_count(width: int, height: int, config: dict[str, Any]) -> int:
    return math.ceil(width / int(config["stride"])) * math.ceil(height / int(config["stride"]))


def _eligible_tile_count(rgb: np.ndarray, reference_rgb: np.ndarray,
                         config: dict[str, Any]) -> int:
    height, width = rgb.shape[:2]
    size, stride = int(config["tile_size"]), int(config["stride"])
    threshold = float(config["min_tissue"])
    count = 0
    for y in range(0, height, stride):
        valid_h = min(size, height - y)
        for x in range(0, width, stride):
            valid_w = min(size, width - x)
            fraction = _tissue_count(
                rgb, reference_rgb, config, y, y + valid_h, x, x + valid_w,
            ) / (valid_h * valid_w)
            count += int(fraction > 0 and fraction >= threshold)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path.name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path.name}:{line_number}")
            rows.append(value)
    return rows


def _verify_output(directory: Path, expected_processing_id: str | None = None,
                   expected_config_id: str | None = None,
                   expected_catalog_id: str | None = None) -> dict[str, Any]:
    commit_path = directory / "commit.json"
    if directory.is_symlink() or not commit_path.is_file() or commit_path.is_symlink():
        raise ValueError(f"Output directory has no committed completion marker: {directory}")
    try:
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid processing commit marker.") from exc
    if not isinstance(commit, dict) or commit.get("complete") is not True:
        raise ValueError("Processing commit marker is incomplete.")
    if commit.get("processing_version") != PROCESSING_VERSION:
        raise ValueError("Processing output was produced by an incompatible processing version.")
    for name, expected in (("processing_id", expected_processing_id),
                           ("config_id", expected_config_id),
                           ("catalog_id", expected_catalog_id)):
        if expected is not None and commit.get(name) != expected:
            raise ValueError(f"Processing output {name} does not match requested input.")
    expected_files = {"patches.tar", "tiles.jsonl", "qc.jsonl"}
    checksums = commit.get("files")
    if not isinstance(checksums, dict) or set(checksums) != expected_files:
        raise ValueError("Processing commit has an invalid output file list.")
    if set(item.name for item in directory.iterdir()) != expected_files | {"commit.json"}:
        raise ValueError("Processing output contains missing or unexpected files.")
    for name in sorted(expected_files):
        path = directory / name
        expected_hash = checksums[name]
        if path.is_symlink() or not path.is_file() or not isinstance(expected_hash, str) or file_hash(path) != expected_hash:
            raise ValueError(f"Processing output checksum mismatch: {name}")
    tile_rows = _read_jsonl(directory / "tiles.jsonl")
    qc_rows = _read_jsonl(directory / "qc.jsonl")
    if len(tile_rows) != commit.get("tiles_written") or len(qc_rows) != commit.get("images_seen"):
        raise ValueError("Processing output record counts do not match commit.")
    tile_ids = [row.get("tile_id") for row in tile_rows]
    content_tile_ids = [row.get("content_tile_id") for row in tile_rows]
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in tile_ids):
        raise ValueError("Invalid tile ID in processing metadata.")
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
           for item in content_tile_ids):
        raise ValueError("Invalid content tile ID in processing metadata.")
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("Duplicate tile ID in processing metadata.")
    expected_members = {f"patches/{tile_id}.png" for tile_id in tile_ids}
    if any(row.get("patch_member") != f"patches/{row['tile_id']}.png" for row in tile_rows):
        raise ValueError("Patch member references do not match tile IDs.")
    try:
        with tarfile.open(directory / "patches.tar", mode="r:") as archive:
            members = archive.getmembers()
            names = [item.name for item in members]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError("Patch archive members do not match tile metadata.")
            if any(not item.isfile() for item in members):
                raise ValueError("Patch archive contains a non-file member.")
            if any(PurePosixPath(item.name).is_absolute() or ".." in PurePosixPath(item.name).parts
                   for item in members):
                raise ValueError("Patch archive contains an unsafe member name.")
            for item in members:
                source = archive.extractfile(item)
                if source is None:
                    raise ValueError(f"Cannot read patch archive member: {item.name}")
                with source, Image.open(source) as patch:
                    patch.load()
                    if patch.mode != "RGB" or patch.size != (commit["config"]["tile_size"],
                                                              commit["config"]["tile_size"]):
                        raise ValueError(f"Invalid encoded patch geometry/mode: {item.name}")
    except (tarfile.TarError, OSError, KeyError, TypeError) as exc:
        raise ValueError("Invalid patch archive in processing output.") from exc
    return commit


def process_shard(release_root: Path, shard_id: str, work_root: Path,
                  output_root: Path, config: dict[str, Any] | None = None,
                  *, keep_staged: bool = False) -> dict[str, Any]:
    """Stage one committed shard, emit lossless RGB tiles and publish verified outputs."""
    config_value = _validated_config(config)
    if not isinstance(keep_staged, bool):
        raise TypeError("keep_staged must be a bool.")
    if not isinstance(shard_id, str) or not _IMAGE_ID.fullmatch(shard_id) or shard_id in {".", ".."}:
        raise ValueError(f"Unsafe shard ID: {shard_id!r}")
    release_root = Path(release_root)
    work_root = Path(work_root)
    output_root = Path(output_root)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve the committed metadata first so verified output reuse never reads the raw TAR.
    manifest = load_shard_manifest(release_root, shard_id)
    catalog_id = manifest.get("catalog_id")
    if not isinstance(catalog_id, str) or not re.fullmatch(r"[0-9a-f]{64}", catalog_id):
        raise ValueError("Shard manifest has an invalid catalog ID.")
    config_id = fingerprint({"processing_version": PROCESSING_VERSION, "config": config_value})
    processing_id = fingerprint({"catalog_id": catalog_id, "shard_id": shard_id,
                                 "config_id": config_id})
    final_directory = output_root / shard_id / processing_id
    if final_directory.exists() or final_directory.is_symlink():
        if not final_directory.is_dir():
            raise ValueError(f"Processing destination is not a directory: {final_directory}")
        result = _verify_output(final_directory, processing_id, config_id, catalog_id)
        staged_path = _stage_cache_path(work_root, catalog_id, shard_id)
        if keep_staged:
            if staged_path.exists() or staged_path.is_symlink():
                _validate_owned_stage_path(staged_path, work_root, catalog_id, shard_id)
                if not staged_path.is_dir():
                    raise ValueError(f"Staged cache path is not a directory: {staged_path}")
                retained_stage = str(staged_path)
            else:
                retained_stage = None
        else:
            _cleanup_stage_cache(work_root, catalog_id, shard_id)
            retained_stage = None
        return dict(result, reused=True, output_path=str(final_directory),
                    staged_path=retained_stage)

    # Stage the source archive only for a new processing fingerprint.
    stage_root, staged_manifest = stage_shard(release_root, shard_id, work_root)
    if not isinstance(staged_manifest, dict) or fingerprint(staged_manifest) != fingerprint(manifest):
        raise ValueError("Staged shard manifest differs from validated release metadata.")
    _validate_owned_stage_path(stage_root, work_root, catalog_id, shard_id)
    release_descriptor = _read_release_descriptor(release_root)
    catalog = release_descriptor.get("catalog")
    if not isinstance(catalog, dict) or catalog.get("catalog_id") != catalog_id:
        raise ValueError("Release descriptor catalog does not match staged shard.")
    source_mode = catalog.get("mode")
    if source_mode not in {"smoke", "full"}:
        raise ValueError("Release catalog has an invalid processing mode.")
    source_training_ready = catalog.get("training_ready") is True
    release_complete = release_descriptor.get("complete") is True

    attempt = uuid.uuid4().hex
    scratch = work_root / f".processing-{processing_id}-{attempt}"
    publish_parent = output_root / shard_id
    publish_parent.mkdir(parents=True, exist_ok=True)
    publish = publish_parent / f".{processing_id}-{attempt}.part"
    scratch.mkdir()
    try:
        patches_path = scratch / "patches.tar"
        tiles_path = scratch / "tiles.jsonl"
        qc_path = scratch / "qc.jsonl"
        stats = _process_images(stage_root, manifest, config_value, config_id,
                                patches_path, tiles_path, qc_path)
        capped = int(config_value["max_tiles_per_image"]) > 0
        training_ready = (source_training_ready and source_mode == "full" and release_complete and not capped
                          and stats["tiles_written"] > 0 and stats["images_review"] == 0)
        checksums = {path.name: file_hash(path) for path in (patches_path, tiles_path, qc_path)}
        _verify_staged_files(scratch, checksums, stats, config_value)

        publish.mkdir()
        for name, checksum in checksums.items():
            verified_copy(scratch / name, publish / name, checksum)
        _verify_staged_files(publish, checksums, stats, config_value)
        commit = {
            "schema_version": 1,
            "processing_version": PROCESSING_VERSION,
            "complete": True,
            "training_ready": training_ready,
            "review_required": stats["images_review"] > 0 or stats["tiles_written"] == 0,
            "mode": "smoke" if capped or source_mode == "smoke" or not release_complete else "full",
            "release_complete": release_complete,
            "catalog_id": catalog_id,
            "shard_id": shard_id,
            "processing_id": processing_id,
            "config_id": config_id,
            "config": config_value,
            "tissue_method": config_value["tissue_method"],
            "tissue_method_calibration": TISSUE_METHOD_CALIBRATION,
            "files": checksums,
            **stats,
        }
        atomic_json(publish / "commit.json", commit)
        os.replace(publish, final_directory)
        result = _verify_output(final_directory, processing_id, config_id, catalog_id)
        if keep_staged:
            retained_stage = str(stage_root)
        else:
            _cleanup_stage_cache(work_root, catalog_id, shard_id, stage_root)
            retained_stage = None
        return dict(result, reused=False, output_path=str(final_directory),
                    staged_path=retained_stage)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if publish.exists():
            shutil.rmtree(publish, ignore_errors=True)


def _stage_cache_path(work_root: Path, catalog_id: str, shard_id: str) -> Path:
    return Path(work_root) / "staged" / catalog_id / shard_id


def _validate_owned_stage_path(stage_path: Path, work_root: Path,
                               catalog_id: str, shard_id: str) -> None:
    work_root_resolved = Path(work_root).resolve()
    expected_path = _stage_cache_path(work_root_resolved, catalog_id, shard_id)
    path = Path(stage_path)
    if any(item.is_symlink() for item in (expected_path.parent.parent, expected_path.parent, expected_path)):
        raise ValueError("Staged cache path cannot include symbolic links.")
    resolved = path.resolve()
    if not resolved.is_relative_to(work_root_resolved) or resolved != expected_path.resolve():
        raise ValueError("stage_shard returned a path outside its owned work cache.")


def _cleanup_stage_cache(work_root: Path, catalog_id: str, shard_id: str,
                         stage_path: Path | None = None) -> None:
    expected = _stage_cache_path(work_root, catalog_id, shard_id)
    path = expected if stage_path is None else Path(stage_path)
    _validate_owned_stage_path(path, work_root, catalog_id, shard_id)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Staged cache path is not a directory: {path}")
    shutil.rmtree(path)


def _read_release_descriptor(release_root: Path) -> dict[str, Any]:
    path = Path(release_root) / "release.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or missing release descriptor.") from exc
    if not isinstance(value, dict):
        raise ValueError("Release descriptor must be an object.")
    return value


def _verify_staged_files(directory: Path, checksums: dict[str, str],
                         stats: dict[str, int], config: dict[str, Any]) -> None:
    """Verify temporary or published artifacts before writing the commit marker."""
    for name, expected in checksums.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Output checksum mismatch before commit: {name}")
    tiles = _read_jsonl(directory / "tiles.jsonl")
    qc = _read_jsonl(directory / "qc.jsonl")
    if len(tiles) != stats["tiles_written"] or len(qc) != stats["images_seen"]:
        raise ValueError("Generated JSONL counts do not match processing counts.")
    member_names = set()
    member_count = 0
    try:
        with tarfile.open(directory / "patches.tar", mode="r:") as archive:
            for member in archive:
                if not member.isfile() or member.name in member_names:
                    raise ValueError("Generated patch archive has an invalid member.")
                member_count += 1
                member_names.add(member.name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable generated patch: {member.name}")
                with stream, Image.open(stream) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (int(config["tile_size"]), int(config["tile_size"])):
                        raise ValueError(f"Generated patch has invalid mode or geometry: {member.name}")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("Generated patch archive failed verification.") from exc
    expected_members = {f"patches/{row['tile_id']}.png" for row in tiles}
    if (member_count != len(tiles) or member_names != expected_members
            or any(row.get("patch_member") != f"patches/{row['tile_id']}.png" for row in tiles)):
        raise ValueError("Generated patch members do not match tile metadata.")
