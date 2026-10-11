"""Whole-original-image cancer classification from frozen ResNet-50 features.

This path is deliberately separate from the coordinate-patch catalog and MIL
interfaces. Each source image receives its own Glade-derived binary target;
patient identity is used only to keep related images together during splits.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import socket
import stat
import threading
import time
import uuid
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps

from .io import atomic_json, file_hash, fingerprint

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
INVENTORY_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
CLASSIFICATION_PROTOCOL = "original-image-frozen-resnet50-v1"
FEATURE_DIM = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_RGB = (124, 116, 104)
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
_PIL_PIXEL_LIMIT_LOCK = threading.Lock()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or ":" in name):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return name


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class OriginalImageInventoryConfig:
    """Inputs for a path-independent, checksum-bound metadata/source audit."""

    metadata: Path
    archives: Mapping[int, Path]
    output_root: Path
    exclude_missing: bool = False
    exclude_unknown_grade: bool = False

    def validate(self) -> None:
        for lens in (4, 10, 40):
            if lens not in self.archives:
                raise ValueError(f"Missing source archive for {lens}X.")
            archive = Path(self.archives[lens]).expanduser()
            if not archive.is_file() or archive.is_symlink() or not zipfile.is_zipfile(archive):
                raise ValueError(f"Expected a regular ZIP archive for {lens}X: {archive}")
        if not Path(self.metadata).is_file() or Path(self.metadata).is_symlink():
            raise ValueError(f"Metadata workbook does not exist as a regular file: {self.metadata}")
        if not isinstance(self.exclude_missing, bool) or not isinstance(self.exclude_unknown_grade, bool):
            raise ValueError("Exclusion switches must be explicit booleans.")
        resolved = [Path(self.archives[lens]).expanduser().resolve() for lens in (4, 10, 40)]
        if len(set(resolved)) != 3:
            raise ValueError("Each magnification must use a distinct source ZIP.")


def _read_metadata(path: Path) -> list[dict[str, str]]:
    """Read required source columns as strings without coercing blank grades."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("XLSX inventory requires pandas and openpyxl; install the image-classification requirements.") from exc
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Metadata must be an XLSX/XLS/CSV file.")
    required = {"Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required original-image columns: {missing}")
    return frame.loc[:, sorted(required)].to_dict("records")


def _parse_lens(value: str, file_name: str) -> int:
    match = re.fullmatch(r"\s*(4|10|40)\s*(?:[xX×])?\s*", str(value))
    if not match:
        raise ValueError(f"Unrecognized objective lens for {file_name}: {value!r}")
    return int(match.group(1))


def _parse_glade(value: str) -> tuple[int | None, str | None]:
    """Map only the human-confirmed original-image grades to binary targets."""
    text = str(value).strip()
    if not text:
        return None, "missing_glade"
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return None, "unknown_glade"
    raw = int(float(text))
    if raw == 0:
        return 0, None
    if raw in {3, 4, 5}:
        return 1, None
    return None, "unknown_glade"


def _patient_id(year_value: str, case_value: str, file_name: str) -> str:
    year = re.sub(r"[\s-]+", "", str(year_value)).upper()
    case_number = str(case_value).strip()
    if not year or not case_number:
        raise ValueError(f"Missing case-to-patient identity fields for {file_name}.")
    return f"{year}_{case_number}"


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_images(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return image members indexed by exact basename without reading payloads."""
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        seen_members: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = _safe_member(info.filename)
            if member in seen_members:
                raise ValueError(f"Duplicate ZIP member: {member}")
            seen_members.add(member)
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlink is not allowed: {member}")
            if info.flag_bits & 1:
                raise ValueError(f"Encrypted ZIP member is not supported: {member}")
            suffix = PurePosixPath(member).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            parts = PurePosixPath(member).parts
            if (len(parts) >= 3 and parts[0].casefold() == "tiles"
                    and re.fullmatch(r"-?\d+_-?\d+(?:_\d+)?\.png", parts[-1], re.IGNORECASE)):
                raise ValueError(
                    f"Source-type mismatch: coordinate-patch member {member!r} is not an original image."
                )
            basename = PurePosixPath(member).name
            by_name[basename].append({
                "source_member": member,
                "member_byte_size": int(info.file_size),
                "member_crc32": f"{info.CRC:08x}",
                "member_compressed_size": int(info.compress_size),
            })
    return by_name


def _inventory_body(records: list[dict[str, Any]], *, metadata_sha256: str,
                    source_identities: dict[str, dict[str, Any]], exclusions: list[dict[str, Any]],
                    mode: str) -> dict[str, Any]:
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "protocol": CLASSIFICATION_PROTOCOL,
        "mode": mode,
        "metadata_sha256": metadata_sha256,
        "sources": source_identities,
        "images": records,
        "exclusions": exclusions,
        "target_mapping": {"0": 0, "3": 1, "4": 1, "5": 1},
        "patient_identity_policy": "Ma_Nam+Ma_So; human-confirmed distinct case per patient; split grouping only",
        "label_level": "original_image_glade",
    }


def build_original_image_inventory(config: OriginalImageInventoryConfig) -> Path:
    """Audit metadata and three ZIPs, then atomically publish an immutable manifest.

    ZIP hashes are streamed exactly once here. Extraction later verifies each
    selected member via ZIP CRC and binds its vector receipt to this archive hash.
    """
    config.validate()
    metadata_path = Path(config.metadata).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    metadata_rows = _read_metadata(metadata_path)
    if not metadata_rows:
        raise ValueError("Metadata workbook contains no rows.")
    if not output_root.is_dir() and output_root.exists():
        raise ValueError("Inventory output root must be a directory.")

    source_paths = {lens: Path(config.archives[lens]).expanduser().resolve() for lens in (4, 10, 40)}
    source_records: dict[str, dict[str, Any]] = {}
    source_paths_manifest: dict[str, str] = {}
    member_maps: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for lens in (4, 10, 40):
        source = source_paths[lens]
        digest = _stream_sha256(source)
        stat_result = source.stat()
        member_maps[lens] = _zip_images(source)
        source_records[str(lens)] = {
            "source_name": source.name,
            "archive_sha256": digest,
            "archive_bytes": int(stat_result.st_size),
            "archive_mtime_ns": int(stat_result.st_mtime_ns),
            "member_count": sum(map(len, member_maps[lens].values())),
        }
        source_paths_manifest[str(lens)] = str(source)

    seen_names: set[str] = set()
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    problems: list[str] = []
    for row_number, source_row in enumerate(metadata_rows, start=2):
        file_name = str(source_row["Ten_File"]).strip()
        if not file_name or PurePosixPath(file_name).name != file_name or "/" in file_name or "\\" in file_name:
            problems.append(f"row {row_number}: Ten_File must be an exact basename: {file_name!r}")
            continue
        if file_name in seen_names:
            problems.append(f"row {row_number}: duplicate metadata Ten_File {file_name!r}")
            continue
        seen_names.add(file_name)
        try:
            lens = _parse_lens(source_row["Do_Phong_Dai"], file_name)
            patient_id = _patient_id(source_row["Ma_Nam"], source_row["Ma_So"], file_name)
            target, grade_error = _parse_glade(source_row["Glade"])
        except ValueError as exc:
            problems.append(f"row {row_number}: {exc}")
            continue
        if grade_error:
            if config.exclude_unknown_grade:
                exclusions.append({"metadata_row": row_number, "file_name": file_name, "reason": grade_error,
                                   "raw_glade": str(source_row["Glade"]).strip()})
                continue
            problems.append(f"row {row_number}: {grade_error} for {file_name!r}; pass exclude_unknown_grade=True to record an explicit exclusion")
            continue
        candidates = member_maps[lens].get(file_name, [])
        if len(candidates) > 1:
            problems.append(f"row {row_number}: duplicate exact-basename ZIP members for {file_name!r}: {[item['source_member'] for item in candidates]}")
            continue
        if not candidates:
            if config.exclude_missing:
                exclusions.append({"metadata_row": row_number, "file_name": file_name,
                                   "reason": "metadata_original_missing_from_lens_archive", "objective_lens": lens,
                                   "raw_glade": str(source_row["Glade"]).strip()})
                continue
            problems.append(f"row {row_number}: original image {file_name!r} absent from {lens}X ZIP; pass exclude_missing=True to record an explicit exclusion")
            continue
        member = candidates[0]
        if member["member_byte_size"] <= 0:
            problems.append(f"row {row_number}: zero-byte original image {file_name!r}")
            continue
        stem = PurePosixPath(file_name).stem
        image_id = f"{lens}x::{file_name}"
        records.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_stem": stem,
            "patient_id": patient_id,
            "objective_lens": lens,
            "raw_glade": str(source_row["Glade"]).strip(),
            "image_target": int(target),
            "source_archive_sha256": source_records[str(lens)]["archive_sha256"],
            "source_archive_name": source_records[str(lens)]["source_name"],
            **member,
        })
    if problems:
        preview = "\n".join(f"- {problem}" for problem in problems[:25])
        suffix = f"\n... and {len(problems) - 25} more" if len(problems) > 25 else ""
        raise ValueError(f"Original-image inventory rejected {len(problems)} metadata/source issue(s):\n{preview}{suffix}")
    if not records:
        raise ValueError("No eligible original images remain after the explicit exclusions.")
    records.sort(key=lambda item: (item["objective_lens"], item["image_id"]))
    source_identity = {lens: {key: value for key, value in record.items() if key != "archive_mtime_ns"}
                       for lens, record in source_records.items()}
    mode = "full" if not exclusions else "explicit_exclusions"
    body = _inventory_body(records, metadata_sha256=file_hash(metadata_path),
                           source_identities=source_identity, exclusions=exclusions, mode=mode)
    inventory_id = fingerprint(body)
    manifest = {
        **body,
        "inventory_id": inventory_id,
        "source_paths": source_paths_manifest,
        "source_runtime_stats": {lens: {"archive_bytes": source_records[lens]["archive_bytes"],
                                         "archive_mtime_ns": source_records[lens]["archive_mtime_ns"]}
                                  for lens in source_records},
        "coverage": {str(lens): sum(row["objective_lens"] == int(lens) for row in records)
                     for lens in ("4", "10", "40")},
        "created_at": _utc(),
    }
    output_dir = output_root / "inventory" / inventory_id
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "inventory.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("inventory_id") != inventory_id:
            raise ValueError("An inventory path exists with a different identity.")
        if existing.get("images") != records:
            raise ValueError("Existing inventory content does not match this audit.")
        return target
    atomic_json(target, manifest)
    atomic_json(output_dir / "inventory.commit.json", {
        "inventory_id": inventory_id,
        "manifest_sha256": file_hash(target),
        "complete": True,
        "commit_id": fingerprint({"inventory_id": inventory_id, "manifest_sha256": file_hash(target), "complete": True}),
    })
    return target


def _load_inventory(path: Path) -> dict[str, Any]:
    inventory_path = Path(path)
    manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_id = manifest.get("inventory_id")
    body = {key: manifest[key] for key in (
        "schema_version", "protocol", "mode", "metadata_sha256", "sources", "images", "exclusions",
        "target_mapping", "patient_identity_policy", "label_level",
    )}
    if inventory_id != fingerprint(body) or manifest.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError("Original-image inventory identity is invalid.")
    commit_path = inventory_path.parent / "inventory.commit.json"
    if commit_path.is_file():
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit_body = {key: commit.get(key) for key in ("inventory_id", "manifest_sha256", "complete")}
        if (commit.get("commit_id") != fingerprint(commit_body) or not commit.get("complete")
                or commit.get("inventory_id") != inventory_id or commit.get("manifest_sha256") != file_hash(inventory_path)):
            raise ValueError("Original-image inventory receipt failed verification.")
    else:
        raise ValueError("Original-image inventory is missing its completion receipt.")
    image_ids = [row.get("image_id") for row in manifest.get("images", [])]
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("Original-image inventory has no images or duplicate image IDs.")
    return manifest


def resize_pad_rgb(image: Image.Image, max_long_edge: int = 512) -> Image.Image:
    """Resize an entire RGB image proportionally and center-pad to a square.

    No center crop, random crop, or patch selection is performed. The output
    side length equals ``max_long_edge`` and padding maps to zero after
    ImageNet normalization.
    """
    max_long_edge = _positive_int(max_long_edge, "max_long_edge")
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    if width < 1 or height < 1:
        raise ValueError("Original image has an empty dimension.")
    scale = max_long_edge / max(width, height)
    resized_width = max(1, min(max_long_edge, int(round(width * scale))))
    resized_height = max(1, min(max_long_edge, int(round(height * scale))))
    if (resized_width, resized_height) != source.size:
        resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        source.close()
        source = resized
    canvas = Image.new("RGB", (max_long_edge, max_long_edge), PAD_RGB)
    left = (max_long_edge - resized_width) // 2
    top = (max_long_edge - resized_height) // 2
    canvas.paste(source, (left, top))
    source.close()
    return canvas


class ImageFeatureEncoder(Protocol):
    """Encoder contract; input images have already been resized and padded."""

    feature_dim: int

    @property
    def descriptor(self) -> dict[str, Any]: ...

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...

    def release_memory(self) -> None: ...


class OriginalImageEncoder:
    """Frozen ImageNet ResNet-50 global-average features for full images."""

    feature_dim = FEATURE_DIM

    def __init__(self, weights_dir: Path, *, device: str = "auto", precision: str = "fp32",
                 download: bool = True) -> None:
        if precision not in {"fp32", "amp_fp16"}:
            raise ValueError("precision must be 'fp32' or 'amp_fp16'.")
        try:
            import torch
            import torchvision
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError("OriginalImageEncoder requires torch and torchvision; install requirements-image-classification.txt") from exc
        from .feature_encoder import (
            OFFICIAL_WEIGHTS_SHA256,
            OFFICIAL_WEIGHTS_URL,
            WEIGHTS_ENUM,
            _ensure_weights,
        )

        self._torch = torch
        self._torchvision = torchvision
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        if self._device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, CPU, or CUDA.")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if precision == "amp_fp16" and self._device.type != "cuda":
            raise ValueError("amp_fp16 requires CUDA.")
        self.precision = precision
        self.weights_path, weights_sha256 = _ensure_weights(torch, Path(weights_dir), download=download)
        if weights_sha256 != OFFICIAL_WEIGHTS_SHA256:
            raise ValueError("ResNet-50 checkpoint checksum is not the pinned ImageNet V1 checkpoint.")
        try:
            state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("Safe torch.load(..., weights_only=True) support is required.") from exc
        if not isinstance(state_dict, dict):
            raise ValueError("ResNet-50 checkpoint must contain a state dictionary.")
        model = models.resnet50(weights=None)
        model.load_state_dict(state_dict, strict=True)
        model.fc = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(self._device)
        self._descriptor = {
            "name": "resnet50_global_average_pooling",
            "weights": WEIGHTS_ENUM,
            "weights_url": OFFICIAL_WEIGHTS_URL,
            "weights_sha256": weights_sha256,
            "feature_dim": self.feature_dim,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "precision": precision,
            "model_mode": "eval; parameters frozen; torch.inference_mode",
            "pooling": "resnet50.layer4+avgpool; flatten; 2048 float32 values",
        }

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._descriptor))

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        torch = self._torch
        tensors = []
        for image in images:
            if image.size[0] != image.size[1]:
                raise ValueError("Encoder expects the square resize-pad canvas from resize_pad_rgb().")
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
            tensors.append((tensor - mean) / std)
        batch = torch.stack(tensors, dim=0).to(self._device)
        with torch.inference_mode():
            if self.precision == "amp_fp16":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self._model(batch)
            else:
                features = self._model(batch)
        result = features.to(device="cpu", dtype=torch.float32).numpy()
        if result.shape != (len(images), self.feature_dim) or not np.isfinite(result).all():
            raise RuntimeError("Frozen ResNet-50 produced invalid pooled features.")
        return np.asarray(result, dtype=np.float32)

    def release_memory(self) -> None:
        gc.collect()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Compute and resume controls for the original-image feature release."""

    batch_size: int = 2
    shard_size: int = 32
    max_long_edge: int = 512
    device: str = "auto"
    precision: str = "fp32"
    budget_minutes: float = 240.0
    reserve_minutes: float = 30.0
    smoke_max_images_per_lens: int | None = None
    recover_lock: bool = False

    def validate(self) -> None:
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.shard_size, "shard_size")
        _positive_int(self.max_long_edge, "max_long_edge")
        if self.smoke_max_images_per_lens is not None:
            _positive_int(self.smoke_max_images_per_lens, "smoke_max_images_per_lens")
        if self.precision not in {"fp32", "amp_fp16"}:
            raise ValueError("precision must be fp32 or amp_fp16.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string.")
        for name in ("budget_minutes", "reserve_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.reserve_minutes < self.budget_minutes:
            raise ValueError("budget_minutes must exceed reserve_minutes.")
        if not isinstance(self.recover_lock, bool):
            raise ValueError("recover_lock must be boolean.")


class BudgetExhausted(RuntimeError):
    """The run stopped before committing all images at its configured deadline."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise BudgetExhausted("Configured compute budget reached; committed shards remain resumable.")


class _RunLock:
    """A run lease with explicit stale recovery and no TTL-based reclamation."""

    def __init__(self, root: Path, *, recover: bool) -> None:
        self.root = root
        self.path = root / "run.lock"
        self.recover = recover
        self.run_id = uuid.uuid4().hex

    def __enter__(self) -> _RunLock:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink():
                raise ValueError("Refusing a symlinked run lock.")
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Unreadable active run lock; refusing automatic recovery.") from exc
            if not isinstance(owner, dict) or not isinstance(owner.get("pid"), int) or not owner.get("run_id"):
                raise ValueError("Run lock does not contain verifiable owner evidence; refusing recovery.")
            if owner.get("hostname") != socket.gethostname():
                raise ValueError("Cross-host run-lock recovery is unsupported; refusing to assume that runtime stopped.")
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                pass
            except (PermissionError, ValueError, TypeError) as exc:
                raise ValueError("Run lock owner cannot be proven stopped; refusing recovery.") from exc
            else:
                raise ValueError("An active run still owns this output; refusing a second writer.")
            if not self.recover:
                raise ValueError("A previous runtime left a lock; stop it and set recover_lock=true to resume.")
            self.path.unlink()
        payload = {"run_id": self.run_id, "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": _utc()}
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.path.is_file():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if owner.get("run_id") == self.run_id:
                self.path.unlink()


def _resize_manifest(max_long_edge: int) -> dict[str, Any]:
    return {
        "method": "full-image aspect-preserving resize then centered square padding",
        "max_long_edge": max_long_edge,
        "canvas": [max_long_edge, max_long_edge],
        "interpolation": "Pillow bilinear",
        "padding_rgb": list(PAD_RGB),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "crop": "none",
        "patches": "none",
    }


def _selected_inventory_images(inventory: dict[str, Any], smoke_cap: int | None) -> list[dict[str, Any]]:
    images = list(inventory["images"])
    if smoke_cap is None:
        return images
    selected = []
    by_lens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in images:
        by_lens[int(row["objective_lens"])].append(row)
    for lens in (4, 10, 40):
        selected.extend(by_lens[lens][:smoke_cap])
    return sorted(selected, key=lambda row: (row["objective_lens"], row["image_id"]))


def _resolve_archive_paths(inventory: dict[str, Any], overrides: Mapping[int, Path] | None) -> dict[str, Path | None]:
    saved = inventory.get("source_paths", {})
    result: dict[str, Path | None] = {}
    for lens in (4, 10, 40):
        explicitly_overridden = overrides is not None
        raw = overrides.get(lens) if overrides is not None else saved.get(str(lens))
        if not raw:
            result[str(lens)] = None
            continue
        requested = Path(raw).expanduser()
        if requested.is_symlink():
            raise ValueError(f"Runtime source must not be a symlink: {requested}")
        path = requested.resolve()
        if not path.is_file() or not zipfile.is_zipfile(path):
            if not explicitly_overridden:
                result[str(lens)] = None
                continue
            raise ValueError(f"Runtime source is not a readable ZIP: {path}")
        result[str(lens)] = path
    return result


def _verify_runtime_archive_binding(run_root: Path, lens: int, path: Path,
                                    expected: dict[str, Any]) -> None:
    """Verify staged ZIP bytes once per stable runtime binding before decode."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Runtime source is not a regular file: {path}")
    stat_result = path.stat()
    if stat_result.st_size != expected["archive_bytes"]:
        raise ValueError(f"Runtime ZIP byte size does not match the inventoried {lens}X source.")
    runtime_signature = {
        "runtime_path": str(path.resolve()),
        "archive_bytes": int(stat_result.st_size),
        "archive_mtime_ns": int(stat_result.st_mtime_ns),
        "archive_device": int(stat_result.st_dev),
        "archive_inode": int(stat_result.st_ino),
    }
    binding_path = run_root / "source_bindings" / f"{lens}x.json"
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        body = {key: value for key, value in binding.items() if key != "binding_id"}
        if (binding.get("binding_id") == fingerprint(body)
                and body.get("source_archive_sha256") == expected["archive_sha256"]
                and all(body.get(key) == value for key, value in runtime_signature.items())):
            return
    actual_sha256 = _stream_sha256(path)
    if actual_sha256 != expected["archive_sha256"]:
        raise ValueError(
            f"Runtime ZIP bytes differ from the inventoried source for {lens}X "
            f"(expected {expected['archive_sha256']}, found {actual_sha256})."
        )
    body = {
        "objective_lens": lens,
        "source_archive_sha256": expected["archive_sha256"],
        "runtime_archive_sha256": actual_sha256,
        **runtime_signature,
        "verified": True,
        "verified_at": _utc(),
    }
    atomic_json(binding_path, {**body, "binding_id": fingerprint(body)})


class SourceNotStaged(RuntimeError):
    """The next ZIP is not present in this runtime; committed lens shards remain resumable."""


def _release_descriptor(inventory: dict[str, Any], encoder: ImageFeatureEncoder,
                        config: FeatureExtractionConfig) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "protocol": CLASSIFICATION_PROTOCOL,
        "inventory_id": inventory["inventory_id"],
        "inventory_mode": inventory["mode"],
        "source_archive_sha256": {
            lens: inventory["sources"][lens]["archive_sha256"] for lens in ("4", "10", "40")
        },
        "feature_encoder": encoder.descriptor,
        "preprocessing": _resize_manifest(config.max_long_edge),
        "smoke_max_images_per_lens": config.smoke_max_images_per_lens,
        "smoke": config.smoke_max_images_per_lens is not None,
    }


def _load_member_image(archive: zipfile.ZipFile, row: dict[str, Any], *,
                       max_image_bytes: int, max_image_pixels: int) -> tuple[Image.Image, str, tuple[int, int], str]:
    """Stream/hash one bounded member, then decode its single payload in memory.

    Pillow's incremental parser leaves some TIFF decoders backed by a closed
    parser stream. A bounded BytesIO keeps multipage/TIFF decoding valid while
    still avoiding ZIP extraction or loading any other image payload.
    """
    info = archive.getinfo(row["source_member"])
    if (info.file_size != row["member_byte_size"] or f"{info.CRC:08x}" != row["member_crc32"]
            or info.file_size > max_image_bytes):
        raise ValueError(f"ZIP member identity/size changed or exceeds the read limit: {row['source_member']}")
    digest = hashlib.sha256()
    payload_stream = io.BytesIO()
    with archive.open(info, "r") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            payload_stream.write(block)
    payload_stream.seek(0)
    try:
        with _PIL_PIXEL_LIMIT_LOCK:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                previous_limit = Image.MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = max_image_pixels
                try:
                    decoded = Image.open(payload_stream)
                    size = (int(decoded.width), int(decoded.height))
                    if decoded.width < 1 or decoded.height < 1 or decoded.width * decoded.height > max_image_pixels:
                        decoded.close()
                        raise ValueError(f"Original image exceeds max_image_pixels: {row['source_member']}")
                    image_format = str(decoded.format or PurePosixPath(row["file_name"]).suffix.lstrip(".")).upper()
                    rgb = ImageOps.exif_transpose(decoded).convert("RGB")
                    rgb.load()
                    decoded.close()
                finally:
                    Image.MAX_IMAGE_PIXELS = previous_limit
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        payload_stream.close()
        raise ValueError(f"Unable to decode original image {row['source_member']}: {exc}") from exc
    payload_stream.close()
    return rgb, digest.hexdigest(), size, image_format


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _encode_adaptive(images: list[Image.Image], encoder: ImageFeatureEncoder,
                     configured_batch_size: int) -> tuple[np.ndarray, int]:
    """Retry only uncommitted inference with smaller batches after an OOM."""
    from .feature_encoder import is_oom

    offset = 0
    effective = min(configured_batch_size, len(images))
    chunks: list[np.ndarray] = []
    while offset < len(images):
        selected = images[offset:offset + effective]
        try:
            values = np.asarray(encoder.encode(selected), dtype=np.float32)
        except (RuntimeError, MemoryError) as exc:
            if not is_oom(exc) or effective == 1:
                raise
            effective = max(1, effective // 2)
            encoder.release_memory()
            continue
        if values.shape != (len(selected), int(encoder.feature_dim)) or not np.isfinite(values).all():
            raise ValueError("Encoder returned invalid feature vectors.")
        chunks.append(values)
        offset += len(selected)
    return np.concatenate(chunks, axis=0), effective


def _verify_feature_shard(path: Path, *, release_id: str | None = None) -> dict[str, Any]:
    root = Path(path)
    commit = json.loads((root / "commit.json").read_text(encoding="utf-8"))
    body = dict(commit)
    commit_id = body.pop("commit_id", None)
    if commit_id != fingerprint(body) or not commit.get("complete"):
        raise ValueError(f"Invalid original-image feature shard receipt: {root}")
    if release_id is not None and commit.get("release_id") != release_id:
        raise ValueError("Feature shard belongs to another encoder/preprocess release.")
    for name, expected in commit["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Feature shard checksum failed: {root / name}")
    matrix = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (commit["image_count"], commit["feature_dim"]) or matrix.dtype != np.float32:
        raise ValueError("Feature shard has an invalid matrix shape/dtype.")
    order = hashlib.sha256()
    seen: set[str] = set()
    with (root / "rows.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row.get("row_index") != index or row.get("image_id") in seen:
                raise ValueError("Feature shard row order or image IDs are invalid.")
            seen.add(row["image_id"])
            vector_sha = _sha256_bytes(np.asarray(matrix[index], dtype=np.float32).tobytes(order="C"))
            if vector_sha != row.get("vector_sha256"):
                raise ValueError(f"Feature vector identity failed for {row['image_id']}.")
            order.update((row["image_id"] + "\n").encode("utf-8"))
            if not np.isfinite(matrix[index]).all():
                raise ValueError(f"Non-finite feature vector for {row['image_id']}.")
    if index + 1 != commit["image_count"] or order.hexdigest() != commit["row_order_sha256"]:
        raise ValueError("Feature shard row index does not match its matrix receipt.")
    return commit


def _commit_feature_shard(run_root: Path, release_id: str, inventory_id: str,
                          shard_number: int, rows: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, Any]:
    features_root = run_root / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    shard_name = f"part-{shard_number:06d}"
    destination = features_root / shard_name
    if destination.exists():
        raise ValueError(f"Uncommitted feature directory already exists: {destination}")
    staging = features_root / f".{shard_name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        matrix_path = staging / "features.npy"
        with matrix_path.open("wb") as stream:
            np.save(stream, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with (staging / "rows.jsonl").open("w", encoding="utf-8") as stream:
            for index, source_row in enumerate(rows):
                row = {**source_row, "row_index": index,
                       "vector_sha256": _sha256_bytes(np.asarray(matrix[index], dtype=np.float32).tobytes(order="C"))}
                stream.write(_row_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        qc = {
            "image_count": len(rows),
            "objective_lens_counts": dict(Counter(str(row["objective_lens"]) for row in rows)),
            "class_counts": dict(Counter(str(row["image_target"]) for row in rows)),
            "preprocessing": "full-image aspect-preserving resize-pad; no crop; no patches",
        }
        atomic_json(staging / "qc.json", qc)
        files = {name: file_hash(staging / name) for name in ("features.npy", "rows.jsonl", "qc.json")}
        order = hashlib.sha256("".join(row["image_id"] + "\n" for row in rows).encode("utf-8")).hexdigest()
        commit = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "release_id": release_id,
            "inventory_id": inventory_id,
            "shard_id": shard_name,
            "image_count": len(rows),
            "feature_dim": int(matrix.shape[1]),
            "dtype": "float32",
            "row_order_sha256": order,
            "files": files,
            "complete": True,
            "committed_at": _utc(),
        }
        commit["commit_id"] = fingerprint(commit)
        atomic_json(staging / "commit.json", commit)
        _verify_feature_shard(staging, release_id=release_id)
        os.replace(staging, destination)
        _verify_feature_shard(destination, release_id=release_id)
        return commit
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _existing_shards(run_root: Path, release_id: str) -> dict[str, dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    features_root = run_root / "features"
    if not features_root.exists():
        return shards
    for path in sorted(features_root.glob("part-*")):
        if not path.is_dir():
            raise ValueError(f"Unexpected file in feature shard directory: {path}")
        commit = _verify_feature_shard(path, release_id=release_id)
        with (path / "rows.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        matrix = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
        for row in rows:
            if row["image_id"] in shards:
                raise ValueError(f"Duplicate committed image feature: {row['image_id']}")
            shards[row["image_id"]] = commit
        del matrix
    return shards


def _status_write(run_root: Path, state: str, *, release_id: str, inventory_id: str,
                  expected_ids: Sequence[str], committed_ids: Sequence[str], **extra: Any) -> None:
    payload = {
        "schema_version": 1,
        "protocol": CLASSIFICATION_PROTOCOL,
        "state": state,
        "release_id": release_id,
        "inventory_id": inventory_id,
        "expected_image_count": len(expected_ids),
        "committed_image_count": len(committed_ids),
        "remaining_image_count": max(0, len(expected_ids) - len(committed_ids)),
        "expected_image_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
        "committed_image_ids_sha256": hashlib.sha256("\n".join(committed_ids).encode()).hexdigest(),
        "updated_at": _utc(),
        **extra,
    }
    atomic_json(run_root / "status.json", payload)


def extract_original_image_features(
    inventory_path: Path,
    output_root: Path,
    weights_dir: Path,
    *,
    encoder: ImageFeatureEncoder | None = None,
    config: FeatureExtractionConfig = FeatureExtractionConfig(),
    archive_paths: Mapping[int, Path] | None = None,
    max_image_bytes: int = 512 * 1024 * 1024,
    max_image_pixels: int = 178_000_000,
) -> Path:
    """Extract and commit original-image vectors, preserving completed shards.

    ``archive_paths`` allows a Colab runtime to rebind streamed SSD staging
    paths. Those local paths are excluded from cache identity; checksums, member
    CRCs, and source names remain in every committed receipt.
    """
    config.validate()
    _positive_int(max_image_bytes, "max_image_bytes")
    _positive_int(max_image_pixels, "max_image_pixels")
    inventory = _load_inventory(inventory_path)
    images = _selected_inventory_images(inventory, config.smoke_max_images_per_lens)
    if not images:
        raise ValueError("No source images selected for feature extraction.")
    paths = _resolve_archive_paths(inventory, archive_paths)
    if encoder is None:
        encoder = OriginalImageEncoder(weights_dir, device=config.device, precision=config.precision)
    descriptor = _release_descriptor(inventory, encoder, config)
    release_id = fingerprint(descriptor)
    run_root = Path(output_root).expanduser().resolve() / "runs" / release_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "release.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("release_id") != release_id or existing.get("descriptor") != descriptor:
            raise ValueError("Feature output path belongs to a different release identity.")
    else:
        atomic_json(manifest_path, {
            "release_id": release_id,
            "inventory_id": inventory["inventory_id"],
            "descriptor": descriptor,
            "inventory_path": str(Path(inventory_path).resolve()),
            "image_count": len(images),
            "image_ids": [row["image_id"] for row in images],
            "smoke": descriptor["smoke"],
            "training_ready": False,
            "extraction_complete": False,
            "created_at": _utc(),
        })
    expected_ids = [row["image_id"] for row in images]
    deadline = time.monotonic() + max(0.0, (config.budget_minutes - config.reserve_minutes) * 60)
    started = time.monotonic()
    with _RunLock(run_root, recover=config.recover_lock):
        existing_by_id = _existing_shards(run_root, release_id)
        unexpected = set(existing_by_id) - set(expected_ids)
        if unexpected:
            raise ValueError(f"Committed cache contains images outside this release: {sorted(unexpected)[:5]}")
        committed_ids = [image_id for image_id in expected_ids if image_id in existing_by_id]
        _status_write(run_root, "running", release_id=release_id, inventory_id=inventory["inventory_id"],
                      expected_ids=expected_ids, committed_ids=committed_ids,
                      smoke=descriptor["smoke"], max_long_edge=config.max_long_edge)
        shard_number = len(list((run_root / "features").glob("part-*"))) if (run_root / "features").exists() else 0
        new_shard_rows: list[dict[str, Any]] = []
        new_shard_vectors: list[np.ndarray] = []
        effective_batch_size = config.batch_size
        processed = set(committed_ids)
        try:
            for lens in (4, 10, 40):
                lens_rows = [row for row in images if int(row["objective_lens"]) == lens and row["image_id"] not in processed]
                if not lens_rows:
                    continue
                archive_path = paths[str(lens)]
                if archive_path is None:
                    raise SourceNotStaged(
                        f"The {lens}X ZIP is not staged in this runtime; stage that ZIP and resume."
                    )
                source_identity = inventory["sources"][str(lens)]
                _verify_runtime_archive_binding(run_root, lens, archive_path, source_identity)
                with zipfile.ZipFile(archive_path) as archive:
                    pending_images: list[Image.Image] = []
                    pending_rows: list[dict[str, Any]] = []
                    for row in lens_rows:
                        _check_deadline(deadline)
                        image, image_sha, original_size, image_format = _load_member_image(
                            archive, row, max_image_bytes=max_image_bytes, max_image_pixels=max_image_pixels)
                        padded = resize_pad_rgb(image, config.max_long_edge)
                        image.close()
                        pending_images.append(padded)
                        pending_rows.append({
                            "image_id": row["image_id"],
                            "file_name": row["file_name"],
                            "patient_id": row["patient_id"],
                            "objective_lens": row["objective_lens"],
                            "raw_glade": row["raw_glade"],
                            "image_target": row["image_target"],
                            "source_archive_sha256": row["source_archive_sha256"],
                            "source_archive_name": row["source_archive_name"],
                            "source_member": row["source_member"],
                            "member_crc32": row["member_crc32"],
                            "member_byte_size": row["member_byte_size"],
                            "original_image_sha256": image_sha,
                            "original_width": original_size[0],
                            "original_height": original_size[1],
                            "source_format": image_format,
                            "padded_width": padded.width,
                            "padded_height": padded.height,
                        })
                        if len(pending_images) >= config.batch_size or row is lens_rows[-1]:
                            try:
                                matrix, selected_batch_size = _encode_adaptive(
                                    pending_images, encoder, effective_batch_size)
                                effective_batch_size = min(effective_batch_size, selected_batch_size)
                                if effective_batch_size < config.batch_size:
                                    atomic_json(run_root / "batch_runtime.json", {
                                        "release_id": release_id,
                                        "configured_batch_size": config.batch_size,
                                        "effective_batch_size": effective_batch_size,
                                        "reason": "oom_retry",
                                        "updated_at": _utc(),
                                    })
                            finally:
                                for item in pending_images:
                                    item.close()
                            new_shard_rows.extend(pending_rows)
                            new_shard_vectors.extend(matrix[index].copy() for index in range(len(matrix)))
                            pending_images.clear()
                            pending_rows.clear()
                            if len(new_shard_rows) >= config.shard_size or row is lens_rows[-1]:
                                chunk = config.shard_size
                                while new_shard_rows:
                                    _check_deadline(deadline)
                                    take = min(chunk, len(new_shard_rows))
                                    rows_chunk = new_shard_rows[:take]
                                    vectors_chunk = np.stack(new_shard_vectors[:take]).astype(np.float32, copy=False)
                                    _commit_feature_shard(run_root, release_id, inventory["inventory_id"],
                                                          shard_number, rows_chunk, vectors_chunk)
                                    shard_number += 1
                                    processed.update(item["image_id"] for item in rows_chunk)
                                    committed_ids = [item for item in expected_ids if item in processed]
                                    _status_write(run_root, "running", release_id=release_id,
                                                  inventory_id=inventory["inventory_id"], expected_ids=expected_ids,
                                                  committed_ids=committed_ids, smoke=descriptor["smoke"],
                                                  max_long_edge=config.max_long_edge,
                                                  completed_shards=shard_number,
                                                  elapsed_seconds=round(time.monotonic() - started, 3))
                                    del new_shard_rows[:take]
                                    del new_shard_vectors[:take]
                                    if len(new_shard_rows) < config.shard_size:
                                        break
                    while new_shard_rows:
                        _check_deadline(deadline)
                        take = min(config.shard_size, len(new_shard_rows))
                        rows_chunk = new_shard_rows[:take]
                        vectors_chunk = np.stack(new_shard_vectors[:take]).astype(np.float32, copy=False)
                        _commit_feature_shard(run_root, release_id, inventory["inventory_id"],
                                              shard_number, rows_chunk, vectors_chunk)
                        shard_number += 1
                        processed.update(item["image_id"] for item in rows_chunk)
                        del new_shard_rows[:take]
                        del new_shard_vectors[:take]
            if new_shard_rows:
                while new_shard_rows:
                    _check_deadline(deadline)
                    take = min(config.shard_size, len(new_shard_rows))
                    _commit_feature_shard(run_root, release_id, inventory["inventory_id"], shard_number,
                                          new_shard_rows[:take], np.stack(new_shard_vectors[:take]).astype(np.float32))
                    shard_number += 1
                    processed.update(item["image_id"] for item in new_shard_rows[:take])
                    del new_shard_rows[:take]
                    del new_shard_vectors[:take]
            committed_ids = [image_id for image_id in expected_ids if image_id in processed]
            complete = committed_ids == expected_ids
            state = "complete" if complete else "partial"
            _status_write(run_root, state, release_id=release_id, inventory_id=inventory["inventory_id"],
                          expected_ids=expected_ids, committed_ids=committed_ids,
                          smoke=descriptor["smoke"], max_long_edge=config.max_long_edge,
                          elapsed_seconds=round(time.monotonic() - started, 3),
                          training_ready=complete and not descriptor["smoke"])
            if complete:
                release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                release_manifest["extraction_complete"] = True
                release_manifest["training_ready"] = not descriptor["smoke"]
                release_manifest["completed_at"] = _utc()
                atomic_json(manifest_path, release_manifest)
        except BudgetExhausted as exc:
            committed_ids = [image_id for image_id in expected_ids if image_id in processed]
            _status_write(run_root, "partial", release_id=release_id, inventory_id=inventory["inventory_id"],
                          expected_ids=expected_ids, committed_ids=committed_ids,
                          smoke=descriptor["smoke"], max_long_edge=config.max_long_edge,
                          stop_reason="deadline", detail=str(exc),
                          elapsed_seconds=round(time.monotonic() - started, 3), training_ready=False)
            release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            release_manifest["extraction_complete"] = False
            release_manifest["training_ready"] = False
            atomic_json(manifest_path, release_manifest)
        except SourceNotStaged as exc:
            committed_ids = [image_id for image_id in expected_ids if image_id in processed]
            _status_write(run_root, "partial", release_id=release_id, inventory_id=inventory["inventory_id"],
                          expected_ids=expected_ids, committed_ids=committed_ids,
                          smoke=descriptor["smoke"], max_long_edge=config.max_long_edge,
                          stop_reason="source_not_staged", detail=str(exc),
                          elapsed_seconds=round(time.monotonic() - started, 3), training_ready=False)
            release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            release_manifest["extraction_complete"] = False
            release_manifest["training_ready"] = False
            atomic_json(manifest_path, release_manifest)
        except Exception as exc:
            committed_ids = [image_id for image_id in expected_ids if image_id in processed]
            _status_write(run_root, "failed", release_id=release_id, inventory_id=inventory["inventory_id"],
                          expected_ids=expected_ids, committed_ids=committed_ids,
                          smoke=descriptor["smoke"], max_long_edge=config.max_long_edge,
                          error_type=type(exc).__name__, error=str(exc), training_ready=False)
            raise
    return run_root


@dataclass(frozen=True)
class TrainingConfig:
    """Nested grouped evaluation and logistic-regression baseline settings."""

    outer_folds: int = 3
    inner_folds: int = 3
    seed: int = 17
    c_values: tuple[float, ...] = DEFAULT_C_VALUES
    balanced_patient_weights: bool = True
    max_iter: int = 2000
    budget_minutes: float = 240.0
    reserve_minutes: float = 15.0
    recover_lock: bool = False

    def validate(self) -> None:
        if self.outer_folds != 3:
            raise ValueError("The original-image baseline protocol locks outer_folds=3.")
        _positive_int(self.inner_folds, "inner_folds")
        _positive_int(self.max_iter, "max_iter")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not self.c_values or any(isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0
                                    for value in self.c_values):
            raise ValueError("c_values must contain positive finite numbers.")
        if len(set(self.c_values)) != len(self.c_values):
            raise ValueError("c_values must be unique.")
        if not isinstance(self.balanced_patient_weights, bool) or not isinstance(self.recover_lock, bool):
            raise ValueError("balanced_patient_weights and recover_lock must be booleans.")
        for name in ("budget_minutes", "reserve_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.reserve_minutes < self.budget_minutes:
            raise ValueError("budget_minutes must exceed reserve_minutes.")


@dataclass(frozen=True)
class LoadedImageClassifier:
    """Checksum-verified deployment pipeline and its image-level decision rule."""

    study_root: Path
    estimator: Any
    threshold: float
    selected_c: float
    max_long_edge: int
    release_id: str
    study_id: str
    study: dict[str, Any]


def _load_feature_matrix(run_root: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(run_root)
    status_path = root / "status.json"
    release_path = root / "release.json"
    if not status_path.is_file() or not release_path.is_file():
        raise ValueError("Training requires a completed original-image feature release.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete" or not status.get("training_ready"):
        raise ValueError(f"Feature extraction is incomplete or not training-ready: {status.get('state')}")
    if release.get("smoke") or not release.get("training_ready"):
        raise ValueError("Smoke-capped releases cannot enter the training path.")
    expected_ids = release.get("image_ids")
    if not isinstance(expected_ids, list) or not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Feature release image ID order is missing or invalid.")
    expected_index = {image_id: index for index, image_id in enumerate(expected_ids)}
    feature_dim = int(release["descriptor"]["feature_encoder"]["feature_dim"])
    features = np.empty((len(expected_ids), feature_dim), dtype=np.float32)
    rows_by_index: list[dict[str, Any] | None] = [None] * len(expected_ids)
    features_root = root / "features"
    shard_paths = sorted(features_root.glob("part-*"))
    if not shard_paths:
        raise ValueError("No committed feature shards are available.")
    for shard_path in shard_paths:
        commit = _verify_feature_shard(shard_path, release_id=release["release_id"])
        if commit.get("inventory_id") != release.get("inventory_id"):
            raise ValueError("Feature shard inventory identity does not match release.")
        with (shard_path / "rows.jsonl").open(encoding="utf-8") as stream:
            shard_rows = [json.loads(line) for line in stream]
        matrix = np.load(shard_path / "features.npy", mmap_mode="r", allow_pickle=False)
        for index, row in enumerate(shard_rows):
            image_id = row.get("image_id")
            if image_id not in expected_index:
                raise ValueError(f"Feature shard contains an image outside the release: {image_id}")
            output_index = expected_index[image_id]
            if rows_by_index[output_index] is not None:
                raise ValueError(f"Duplicate committed feature row: {image_id}")
            if row.get("padded_width") != release["descriptor"]["preprocessing"]["max_long_edge"]:
                raise ValueError(f"Feature preprocessing size differs for {image_id}.")
            target = row.get("image_target")
            grade = row.get("raw_glade")
            expected_target, grade_error = _parse_glade(grade)
            if grade_error or target != expected_target or target not in {0, 1}:
                raise ValueError(f"Original image target does not follow the Glade mapping for {image_id}.")
            if not row.get("patient_id") or row.get("objective_lens") not in {4, 10, 40}:
                raise ValueError(f"Image row lacks patient identity or objective lens: {image_id}.")
            source_sha = row.get("original_image_sha256")
            if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
                raise ValueError(f"Original image checksum is invalid for {image_id}.")
            expected_archive_sha = release["descriptor"]["source_archive_sha256"].get(str(row["objective_lens"]))
            if row.get("source_archive_sha256") != expected_archive_sha:
                raise ValueError(f"Source archive identity differs from feature release for {image_id}.")
            rows_by_index[output_index] = row
            features[output_index] = matrix[index]
        del matrix
    if any(row is None for row in rows_by_index):
        missing = [expected_ids[index] for index, row in enumerate(rows_by_index) if row is None]
        extra: list[str] = []
        raise ValueError(f"Feature coverage mismatch (missing={len(missing)}, extra={len(extra)}).")
    rows = [row for row in rows_by_index if row is not None]
    if features.shape[1] != int(release["descriptor"]["feature_encoder"]["feature_dim"]):
        raise ValueError("Feature dimension differs from the release encoder descriptor.")
    return features, rows, release


def _audit_original_image_content(run_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject byte-identical source photos before a patient-grouped split."""
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sha[row["original_image_sha256"]].append(row)
    duplicate_groups = [(digest, members) for digest, members in sorted(by_sha.items()) if len(members) > 1]
    audit = {
        "schema_version": 1,
        "passed": not duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "duplicated_image_count": sum(len(members) for _, members in duplicate_groups),
        "cross_patient_group_count": sum(len({row["patient_id"] for row in members}) > 1
                                          for _, members in duplicate_groups),
        "conflicting_target_group_count": sum(len({int(row["image_target"]) for row in members}) > 1
                                               for _, members in duplicate_groups),
        "examples": [
            {
                "original_image_sha256": digest,
                "image_ids": [row["image_id"] for row in members],
                "patient_hashes": sorted({_patient_hash(row["patient_id"]) for row in members}),
                "image_targets": sorted({int(row["image_target"]) for row in members}),
            }
            for digest, members in duplicate_groups[:10]
        ],
        "checked_at": _utc(),
    }
    atomic_json(Path(run_root) / "content_duplicate_audit.json", audit)
    if duplicate_groups:
        raise ValueError(
            "Duplicate original image content detected: "
            f"{audit['duplicated_image_count']} rows in {audit['duplicate_group_count']} SHA-256 group(s), "
            f"including {audit['cross_patient_group_count']} cross-patient group(s) and "
            f"{audit['conflicting_target_group_count']} conflicting-target group(s). "
            f"Review {Path(run_root) / 'content_duplicate_audit.json'} before training."
        )
    return audit


def _patient_hash(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def _patient_sample_weights(labels: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Give each patient equal weight within each class, even with many images."""
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in (0, 1):
        label_mask = labels == label
        patient_counts: Counter[str] = Counter(patients[label_mask].tolist())
        patient_total = len(patient_counts)
        if patient_total == 0:
            raise ValueError(f"Training fold has no patients for image target {label}.")
        for index in np.flatnonzero(label_mask):
            weights[index] = 1.0 / patient_total / patient_counts[patients[index]]
    return weights


def _grouped_folds(labels: np.ndarray, patients: np.ndarray, *, requested_folds: int,
                   seed: int, context: str) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        import sklearn
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise RuntimeError("Nested grouped classification requires scikit-learn; install requirements-image-classification.txt") from exc
    if str(sklearn.__version__) != "1.8.0":
        raise RuntimeError(f"Image-classification split protocol requires scikit-learn==1.8.0; found {sklearn.__version__}.")
    class_support = {label: len(set(patients[labels == label].tolist())) for label in (0, 1)}
    fold_count = min(requested_folds, class_support[0], class_support[1])
    if fold_count < 2:
        raise ValueError(f"{context} cannot form at least two patient-grouped folds with both classes: {class_support}")
    for attempt in range(128):
        splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed + attempt)
        folds = list(splitter.split(np.zeros(len(labels), dtype=np.uint8), labels, patients))
        valid = True
        for train_indices, validation_indices in folds:
            if set(labels[train_indices].tolist()) != {0, 1} or set(labels[validation_indices].tolist()) != {0, 1}:
                valid = False
                break
            if set(patients[train_indices].tolist()) & set(patients[validation_indices].tolist()):
                raise ValueError(f"{context} patient leakage detected in grouped splitter.")
        if valid:
            return [(np.asarray(train, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
                    for train, valid_idx in folds]
    raise ValueError(f"{context} could not find a grouped split with both classes in each fold after 128 deterministic seeds.")


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if not np.any(mask):
            return None
        recalls.append(float(np.mean(predictions[mask] == label)))
    return float(sum(recalls) / 2)


def _metrics(labels: np.ndarray, probabilities: np.ndarray,
             predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    positives = tp + fn
    negatives = tn + fp
    auc: float | None = None
    if positives and negatives:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probabilities))
    sensitivity = float(tp / positives) if positives else None
    specificity = float(tn / negatives) if negatives else None
    balanced = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    accuracy = float((tp + tn) / len(labels)) if len(labels) else None
    return {
        "image_count": int(len(labels)),
        "positive_images": positives,
        "negative_images": negatives,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probabilities, dtype=np.float64))))
    choices: list[tuple[float, float]] = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(np.int64)
        score = _balanced_accuracy(labels, prediction)
        if score is not None:
            choices.append((score, float(threshold)))
    if not choices:
        raise ValueError("Validation scores did not contain both classes.")
    best_score = max(item[0] for item in choices)
    best_threshold = min((item[1] for item in choices if item[0] == best_score),
                         key=lambda item: (abs(item - 0.5), item))
    return best_threshold, best_score


def _fit_tuning_fold(features: np.ndarray, labels: np.ndarray, train_indices: np.ndarray,
                     validation_indices: np.ndarray, patients: np.ndarray, *, c_value: float,
                     balanced_patient_weights: bool, max_iter: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = features[train_indices]
    y_train = labels[train_indices]
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)
    classifier = LogisticRegression(C=float(c_value), max_iter=max_iter, solver="lbfgs", random_state=seed)
    sample_weights = _patient_sample_weights(y_train, patients[train_indices]) if balanced_patient_weights else None
    classifier.fit(x_train_scaled, y_train, sample_weight=sample_weights)
    probabilities = classifier.predict_proba(scaler.transform(features[validation_indices]))[:, 1]
    audit = {
        "scaler_fit_row_count": int(len(train_indices)),
        "scaler_mean_sha256": _sha256_bytes(np.asarray(scaler.mean_, dtype=np.float64).tobytes()),
        "scaler_scale_sha256": _sha256_bytes(np.asarray(scaler.scale_, dtype=np.float64).tobytes()),
    }
    return np.asarray(probabilities, dtype=np.float64), audit


def _tune_grouped(features: np.ndarray, labels: np.ndarray, patients: np.ndarray,
                  image_ids: Sequence[str], *, c_values: Sequence[float], inner_folds: int,
                  seed: int, balanced_patient_weights: bool, max_iter: int,
                  deadline: float | None = None) -> dict[str, Any]:
    inner = _grouped_folds(labels, patients, requested_folds=inner_folds, seed=seed,
                           context="inner grouped validation")
    probability_by_c = {float(c_value): np.full(len(labels), np.nan, dtype=np.float64) for c_value in c_values}
    split_audit = []
    validation_quality: dict[float, list[dict[str, Any]]] = {float(c_value): [] for c_value in c_values}
    for split_index, (train_indices, validation_indices) in enumerate(inner, start=1):
        if deadline is not None:
            _check_deadline(deadline)
        train_patients = set(patients[train_indices].tolist())
        validation_patients = set(patients[validation_indices].tolist())
        if train_patients & validation_patients:
            raise ValueError("Inner validation split leaked patient IDs.")
        split_audit.append({
            "split": split_index,
            "train_image_ids": [image_ids[index] for index in train_indices],
            "validation_image_ids": [image_ids[index] for index in validation_indices],
            "scaler_fit_image_ids": [image_ids[index] for index in train_indices],
            "train_patient_hashes": sorted(_patient_hash(value) for value in train_patients),
            "validation_patient_hashes": sorted(_patient_hash(value) for value in validation_patients),
            "patient_overlap_count": len(train_patients & validation_patients),
        })
        for c_value in c_values:
            if deadline is not None:
                _check_deadline(deadline)
            scores, scaler_audit = _fit_tuning_fold(
                features, labels, train_indices, validation_indices, patients,
                c_value=float(c_value), balanced_patient_weights=balanced_patient_weights,
                max_iter=max_iter, seed=seed + split_index,
            )
            probability_by_c[float(c_value)][validation_indices] = scores
            validation_quality[float(c_value)].append({
                "split": split_index,
                "validation_auc": _metrics(labels[validation_indices], scores,
                                            (scores >= 0.5).astype(np.int64))["roc_auc"],
                **scaler_audit,
            })
    if any(not np.isfinite(scores).all() for scores in probability_by_c.values()):
        raise ValueError("Inner grouped validation did not produce exactly one score per training image.")
    c_scores = []
    for c_value, probabilities in probability_by_c.items():
        metrics = _metrics(labels, probabilities, (probabilities >= 0.5).astype(np.int64))
        threshold, bacc = _choose_threshold(labels, probabilities)
        c_scores.append({"C": c_value, "validation_roc_auc": metrics["roc_auc"],
                         "validation_balanced_accuracy_at_0_5": _balanced_accuracy(labels, probabilities >= 0.5),
                         "validation_selected_threshold": threshold,
                         "validation_balanced_accuracy_at_selected_threshold": bacc})
    selected = sorted(c_scores, key=lambda item: (-float(item["validation_roc_auc"] or -1.0), item["C"]))[0]
    chosen_c = float(selected["C"])
    chosen_scores = probability_by_c[chosen_c]
    threshold, threshold_bacc = _choose_threshold(labels, chosen_scores)
    return {
        "selected_C": chosen_c,
        "threshold": threshold,
        "threshold_selection_validation_balanced_accuracy": threshold_bacc,
        "selected_C_validation_roc_auc": selected["validation_roc_auc"],
        "c_validation_scores": c_scores,
        "inner_splits": split_audit,
        "validation_quality_by_split": validation_quality,
        "oof_validation_probabilities": chosen_scores,
    }


def _fit_pipeline(features: np.ndarray, labels: np.ndarray, patients: np.ndarray, *,
                  c_value: float, balanced_patient_weights: bool, max_iter: int,
                  seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline([
        ("scaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
        ("classifier", LogisticRegression(C=float(c_value), max_iter=max_iter,
                                           solver="lbfgs", random_state=seed)),
    ])
    weights = _patient_sample_weights(labels, patients) if balanced_patient_weights else None
    estimator.fit(features, labels, classifier__sample_weight=weights)
    return estimator


def _atomic_joblib_dump(model: Any, destination: Path) -> None:
    import joblib

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_fold_artifact(path: Path, study_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(path)
    commit = json.loads((root / "commit.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in commit.items() if key != "commit_id"}
    if (commit.get("commit_id") != fingerprint(body) or not commit.get("complete")
            or commit.get("study_id") != study_id):
        raise ValueError(f"Invalid fold artifact receipt: {root}")
    for name, digest in commit["files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Fold artifact checksum failed: {root / name}")
    audit = json.loads((root / "fold.json").read_text(encoding="utf-8"))
    predictions = json.loads((root / "predictions.json").read_text(encoding="utf-8"))
    if audit.get("fold_id") != commit.get("fold_id") or len(predictions) != audit.get("test_image_count"):
        raise ValueError("Fold audit and prediction rows do not agree.")
    return audit, predictions


def _commit_fold_artifact(training_root: Path, study_id: str, fold_number: int,
                          audit: dict[str, Any], predictions: list[dict[str, Any]], estimator: Any) -> Path:
    from .io import atomic_json

    fold_id = f"fold-{fold_number:02d}"
    root = training_root / "folds" / fold_id
    if root.exists():
        raise ValueError(f"Refusing to overwrite existing fold artifact: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{fold_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        _atomic_joblib_dump(estimator, staging / "model.joblib")
        atomic_json(staging / "fold.json", audit)
        atomic_json(staging / "predictions.json", predictions)
        files = {name: file_hash(staging / name) for name in ("model.joblib", "fold.json", "predictions.json")}
        commit = {"study_id": study_id, "fold_id": fold_id, "complete": True, "files": files,
                  "test_image_count": len(predictions), "committed_at": _utc()}
        commit["commit_id"] = fingerprint(commit)
        atomic_json(staging / "commit.json", commit)
        os.replace(staging, root)
        _verify_fold_artifact(root, study_id)
        return root
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields_order: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields_order), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _prediction_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "image_id", "file_name", "patient_id", "objective_lens", "raw_glade", "image_target",
        "outer_fold", "predicted_probability", "prediction_at_0_5", "selected_threshold",
        "prediction_at_selected_threshold", "selected_C", "source_archive_sha256",
        "source_member", "original_image_sha256",
    )}


def _collect_predictions(study_root: Path, study_id: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result: dict[str, dict[str, Any]] = {}
    fold_audits = []
    for fold_path in sorted((study_root / "folds").glob("fold-*")):
        audit, predictions = _verify_fold_artifact(fold_path, study_id)
        fold_audits.append(audit)
        for row in predictions:
            if row["image_id"] in result:
                raise ValueError(f"OOF image was predicted more than once: {row['image_id']}")
            result[row["image_id"]] = row
    return result, fold_audits


def _compute_reports(predictions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray([int(row["image_target"]) for row in predictions], dtype=np.int64)
    probabilities = np.asarray([float(row["predicted_probability"]) for row in predictions], dtype=np.float64)
    fixed_predictions = np.asarray([int(row["prediction_at_0_5"]) for row in predictions], dtype=np.int64)
    selected_predictions = np.asarray([int(row["prediction_at_selected_threshold"]) for row in predictions], dtype=np.int64)
    pooled = {
        "fixed_threshold_0_5": _metrics(labels, probabilities, fixed_predictions),
        "validation_selected_threshold": _metrics(labels, probabilities, selected_predictions),
        "selected_thresholds_by_fold": {str(fold): float(next(row["selected_threshold"] for row in predictions
                                                                   if row["outer_fold"] == fold))
                                        for fold in sorted({int(row["outer_fold"]) for row in predictions})},
    }
    by_lens: dict[str, Any] = {}
    for lens in (4, 10, 40):
        selected = [row for row in predictions if int(row["objective_lens"]) == lens]
        if not selected:
            continue
        y = np.asarray([int(row["image_target"]) for row in selected], dtype=np.int64)
        p = np.asarray([float(row["predicted_probability"]) for row in selected], dtype=np.float64)
        by_lens[str(lens)] = {
            "fixed_threshold_0_5": _metrics(y, p, np.asarray([int(row["prediction_at_0_5"]) for row in selected])),
            "validation_selected_threshold": _metrics(y, p, np.asarray([int(row["prediction_at_selected_threshold"]) for row in selected])),
        }
    return pooled, by_lens


def train_original_image_classifier(
    run_root: Path,
    *,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    """Run three outer patient-grouped folds with nested C/threshold selection.

    The persisted OOF result contains one prediction for every original image.
    The separate deployment estimator is tuned using grouped OOF validation on
    all images and then refit on all images; report metrics never use it.
    """
    config.validate()
    root = Path(run_root).expanduser().resolve()
    features, rows, release = _load_feature_matrix(root)
    image_ids = [row["image_id"] for row in rows]
    labels = np.asarray([int(row["image_target"]) for row in rows], dtype=np.int64)
    patients = np.asarray([str(row["patient_id"]) for row in rows], dtype=object)
    lenses = np.asarray([int(row["objective_lens"]) for row in rows], dtype=np.int64)
    content_audit = _audit_original_image_content(root, rows)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Image-level dataset must include both Glade-derived classes.")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs cannot enter original-image training.")
    outer = _grouped_folds(labels, patients, requested_folds=config.outer_folds,
                           seed=config.seed, context="outer patient-grouped evaluation")
    canonical_config = {
        "protocol": CLASSIFICATION_PROTOCOL,
        "release_id": release["release_id"],
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "seed": config.seed,
        "c_values": [float(value) for value in config.c_values],
        "balanced_patient_weights": config.balanced_patient_weights,
        "max_iter": config.max_iter,
    }
    study_id = fingerprint(canonical_config)
    study_root = root / "training" / study_id
    study_root.mkdir(parents=True, exist_ok=True)
    config_path = study_root / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != canonical_config:
        raise ValueError("Training directory contains a different config identity.")
    if not config_path.exists():
        atomic_json(config_path, canonical_config)
    deadline = time.monotonic() + max(0.0, (config.budget_minutes - config.reserve_minutes) * 60)
    with _RunLock(study_root, recover=config.recover_lock):
        if (study_root / "study.commit.json").is_file():
            loaded = load_deployment_classifier(study_root)
            return {"study_id": study_id, "study_root": str(study_root),
                    "state": "complete", "metrics": loaded.study["metrics"]}
        atomic_json(study_root / "training_status.json", {"state": "running", "study_id": study_id,
                                                          "release_id": release["release_id"], "updated_at": _utc()})
        try:
            for fold_number, (train_indices, test_indices) in enumerate(outer, start=1):
                _check_deadline(deadline)
                fold_root = study_root / "folds" / f"fold-{fold_number:02d}"
                if fold_root.is_dir() and (fold_root / "commit.json").is_file():
                    _verify_fold_artifact(fold_root, study_id)
                    continue
                train_patient_ids = set(patients[train_indices].tolist())
                test_patient_ids = set(patients[test_indices].tolist())
                if train_patient_ids & test_patient_ids:
                    raise ValueError(f"Outer fold {fold_number} has patient leakage.")
                tuning = _tune_grouped(
                    features[train_indices], labels[train_indices], patients[train_indices],
                    [image_ids[index] for index in train_indices], c_values=config.c_values,
                    inner_folds=config.inner_folds, seed=config.seed + fold_number * 100,
                    balanced_patient_weights=config.balanced_patient_weights, max_iter=config.max_iter,
                    deadline=deadline,
                )
                selected_c = float(tuning["selected_C"])
                estimator = _fit_pipeline(features[train_indices], labels[train_indices], patients[train_indices],
                                          c_value=selected_c,
                                          balanced_patient_weights=config.balanced_patient_weights,
                                          max_iter=config.max_iter, seed=config.seed + fold_number)
                test_probabilities = estimator.predict_proba(features[test_indices])[:, 1]
                threshold = float(tuning["threshold"])
                predicted_05 = (test_probabilities >= 0.5).astype(np.int64)
                predicted_selected = (test_probabilities >= threshold).astype(np.int64)
                fold_predictions = []
                for offset, index in enumerate(test_indices):
                    source = rows[index]
                    fold_predictions.append({
                        "image_id": image_ids[index],
                        "file_name": source["file_name"],
                        "patient_id": source["patient_id"],
                        "objective_lens": int(lenses[index]),
                        "raw_glade": source["raw_glade"],
                        "image_target": int(labels[index]),
                        "outer_fold": fold_number,
                        "predicted_probability": float(test_probabilities[offset]),
                        "prediction_at_0_5": int(predicted_05[offset]),
                        "selected_threshold": threshold,
                        "prediction_at_selected_threshold": int(predicted_selected[offset]),
                        "selected_C": selected_c,
                        "source_archive_sha256": source["source_archive_sha256"],
                        "source_member": source["source_member"],
                        "original_image_sha256": source["original_image_sha256"],
                    })
                scaler = estimator.named_steps["scaler"]
                train_image_ids = [image_ids[index] for index in train_indices]
                test_image_ids = [image_ids[index] for index in test_indices]
                audit = {
                    "fold_id": f"fold-{fold_number:02d}",
                    "outer_fold": fold_number,
                    "train_image_count": len(train_indices),
                    "test_image_count": len(test_indices),
                    "train_image_ids": train_image_ids,
                    "test_image_ids": test_image_ids,
                    "scaler_fit_image_ids": train_image_ids,
                    "train_patient_hashes": sorted(_patient_hash(patient) for patient in train_patient_ids),
                    "test_patient_hashes": sorted(_patient_hash(patient) for patient in test_patient_ids),
                    "patient_overlap_count": 0,
                    "selected_C": selected_c,
                    "selected_threshold": threshold,
                    "selected_C_validation_roc_auc": tuning["selected_C_validation_roc_auc"],
                    "threshold_selection_validation_balanced_accuracy": tuning["threshold_selection_validation_balanced_accuracy"],
                    "c_validation_scores": tuning["c_validation_scores"],
                    "inner_validation_splits": tuning["inner_splits"],
                    "inner_validation_scaler_audit": tuning["validation_quality_by_split"],
                    "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
                    "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
                    "scaler_mean_sha256": _sha256_bytes(np.asarray(scaler.mean_, dtype=np.float64).tobytes()),
                    "feature_dim": int(features.shape[1]),
                }
                _commit_fold_artifact(study_root, study_id, fold_number, audit, fold_predictions, estimator)
                atomic_json(study_root / "training_status.json", {
                    "state": "running", "study_id": study_id, "release_id": release["release_id"],
                    "completed_outer_folds": fold_number, "expected_outer_folds": len(outer),
                    "updated_at": _utc(),
                })
            prediction_by_id, fold_audits = _collect_predictions(study_root, study_id)
            if set(prediction_by_id) != set(image_ids):
                raise ValueError("Outer OOF folds do not predict every original image exactly once.")
            predictions = [prediction_by_id[image_id] for image_id in image_ids]
            pooled_metrics, lens_metrics = _compute_reports(predictions)
            deployment_tuning = _tune_grouped(
                features, labels, patients, image_ids, c_values=config.c_values,
                inner_folds=config.inner_folds, seed=config.seed + 10000,
                balanced_patient_weights=config.balanced_patient_weights, max_iter=config.max_iter,
                deadline=deadline,
            )
            deployment_estimator = _fit_pipeline(
                features, labels, patients, c_value=float(deployment_tuning["selected_C"]),
                balanced_patient_weights=config.balanced_patient_weights,
                max_iter=config.max_iter, seed=config.seed + 20000,
            )
            _atomic_joblib_dump(deployment_estimator, study_root / "deployment_model.joblib")
            deployment_receipt_body = {
                "study_id": study_id,
                "release_id": release["release_id"],
                "model_sha256": file_hash(study_root / "deployment_model.joblib"),
                "selected_C": float(deployment_tuning["selected_C"]),
                "threshold": float(deployment_tuning["threshold"]),
                "max_long_edge": int(release["descriptor"]["preprocessing"]["max_long_edge"]),
                "training_image_ids_sha256": hashlib.sha256("\n".join(image_ids).encode()).hexdigest(),
                "scaler_fit_image_ids": image_ids,
                "scaler_mean_sha256": _sha256_bytes(np.asarray(deployment_estimator.named_steps["scaler"].mean_, dtype=np.float64).tobytes()),
            }
            deployment_receipt_body["receipt_id"] = fingerprint(deployment_receipt_body)
            atomic_json(study_root / "deployment_model.commit.json", deployment_receipt_body)
            study = {
                "schema_version": 1,
                "protocol": CLASSIFICATION_PROTOCOL,
                "study_id": study_id,
                "release_id": release["release_id"],
                "inventory_id": release["inventory_id"],
                "training_config": canonical_config,
                "image_count": len(rows),
                "patient_count": len(set(patients.tolist())),
                "class_counts": {str(label): int(np.sum(labels == label)) for label in (0, 1)},
                "outer_fold_count": len(fold_audits),
                "folds": fold_audits,
                "metrics": {"pooled": pooled_metrics, "by_objective_lens": lens_metrics},
                "deployment_model": {
                    "selected_C": float(deployment_tuning["selected_C"]),
                    "threshold": float(deployment_tuning["threshold"]),
                    "max_long_edge": int(release["descriptor"]["preprocessing"]["max_long_edge"]),
                    "tuning": {key: value for key, value in deployment_tuning.items() if key != "oof_validation_probabilities"},
                    "scaler_fit_image_ids": image_ids,
                    "scaler_mean": np.asarray(deployment_estimator.named_steps["scaler"].mean_, dtype=np.float64).tolist(),
                },
                "patient_split_leakage_check": {"passed": True, "outer_folds": len(fold_audits),
                                                 "patient_overlap_count": 0},
                "content_duplicate_audit": content_audit,
                "prediction_unit": "original_image",
                "training_target_field": "image_target",
                "case_target_used": False,
                "patch_labels_used": False,
                "created_at": _utc(),
            }
            atomic_json(study_root / "study.json", study)
            csv_path = study_root / "oof_predictions.csv"
            _write_csv(csv_path, [_prediction_csv_row(row) for row in predictions], (
                "image_id", "file_name", "patient_id", "objective_lens", "raw_glade", "image_target",
                "outer_fold", "predicted_probability", "prediction_at_0_5", "selected_threshold",
                "prediction_at_selected_threshold", "selected_C", "source_archive_sha256",
                "source_member", "original_image_sha256",
            ))
            files = {
                "study.json": file_hash(study_root / "study.json"),
                "oof_predictions.csv": file_hash(csv_path),
                "deployment_model.joblib": file_hash(study_root / "deployment_model.joblib"),
                "deployment_model.commit.json": file_hash(study_root / "deployment_model.commit.json"),
            }
            fold_commit_ids = {audit["fold_id"]: json.loads((study_root / "folds" / audit["fold_id"] / "commit.json").read_text())["commit_id"]
                               for audit in fold_audits}
            study_commit = {"study_id": study_id, "release_id": release["release_id"], "complete": True,
                            "files": files, "fold_commit_ids": fold_commit_ids, "committed_at": _utc()}
            study_commit["commit_id"] = fingerprint(study_commit)
            atomic_json(study_root / "study.commit.json", study_commit)
            atomic_json(study_root / "training_status.json", {
                "state": "complete", "study_id": study_id, "release_id": release["release_id"],
                "image_count": len(rows), "updated_at": _utc(),
            })
            atomic_json(study_root / "report.json", {
                "study_id": study_id,
                "prediction_unit": "original_image",
                "metrics": study["metrics"],
                "prediction_csv": str(csv_path),
                "patient_split_leakage_check": study["patient_split_leakage_check"],
            })
            return {"study_id": study_id, "study_root": str(study_root),
                    "state": "complete", "metrics": study["metrics"]}
        except BudgetExhausted as exc:
            atomic_json(study_root / "training_status.json", {
                "state": "partial", "study_id": study_id, "release_id": release["release_id"],
                "stop_reason": "deadline", "detail": str(exc), "updated_at": _utc(),
            })
            return {"study_id": study_id, "study_root": str(study_root), "state": "partial", "reason": str(exc)}
        except Exception as exc:
            atomic_json(study_root / "training_status.json", {
                "state": "failed", "study_id": study_id, "release_id": release["release_id"],
                "error_type": type(exc).__name__, "error": str(exc), "updated_at": _utc(),
            })
            raise


def _verify_study(study_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(study_root)
    commit = json.loads((root / "study.commit.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in commit.items() if key != "commit_id"}
    if commit.get("commit_id") != fingerprint(body) or not commit.get("complete"):
        raise ValueError("Study receipt is incomplete or invalid.")
    for name, expected_sha in commit["files"].items():
        if file_hash(root / name) != expected_sha:
            raise ValueError(f"Study artifact checksum failed: {root / name}")
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    if study.get("study_id") != commit.get("study_id") or study.get("release_id") != commit.get("release_id"):
        raise ValueError("Study JSON identity does not match its receipt.")
    for fold_id, expected_commit in commit.get("fold_commit_ids", {}).items():
        _verify_fold_artifact(root / "folds" / fold_id, study["study_id"])
        actual_commit = json.loads((root / "folds" / fold_id / "commit.json").read_text(encoding="utf-8"))["commit_id"]
        if actual_commit != expected_commit:
            raise ValueError(f"Study references another fold artifact: {fold_id}")
    return study, commit


def load_deployment_classifier(study_root: Path) -> LoadedImageClassifier:
    """Load only a complete checksum-verified internal study/model release."""
    root = Path(study_root).expanduser().resolve()
    study, _ = _verify_study(root)
    receipt = json.loads((root / "deployment_model.commit.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (receipt.get("receipt_id") != fingerprint(body)
            or receipt.get("study_id") != study["study_id"]
            or receipt.get("release_id") != study["release_id"]
            or receipt.get("model_sha256") != file_hash(root / "deployment_model.joblib")
            or receipt.get("max_long_edge") != study.get("deployment_model", {}).get("max_long_edge")):
        raise ValueError("Deployment estimator receipt failed verification.")
    import joblib
    estimator = joblib.load(root / "deployment_model.joblib")
    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "named_steps"):
        raise ValueError("Loaded deployment artifact is not a probability-producing scaler/classifier pipeline.")
    deployment = study["deployment_model"]
    return LoadedImageClassifier(
        study_root=root,
        estimator=estimator,
        threshold=float(deployment["threshold"]),
        selected_c=float(deployment["selected_C"]),
        max_long_edge=int(deployment["max_long_edge"]),
        release_id=study["release_id"],
        study_id=study["study_id"],
        study=study,
    )


def predict_original_image(
    study_root: Path,
    image: Image.Image | Path,
    weights_dir: Path,
    *,
    encoder: ImageFeatureEncoder | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Return one original-image cancer probability from a saved deployment model."""
    loaded = load_deployment_classifier(study_root)
    release_path = loaded.study_root.parents[1] / "release.json"
    # Training roots are nested below the versioned feature release.
    release = json.loads(release_path.read_text(encoding="utf-8"))
    precision = release["descriptor"]["feature_encoder"].get("precision", "fp32")
    if encoder is None:
        encoder = OriginalImageEncoder(weights_dir, device=device, precision=precision, download=False)
    expected_encoder = release["descriptor"]["feature_encoder"]
    if encoder.descriptor != expected_encoder:
        raise ValueError("Inference encoder descriptor does not match the frozen-feature release.")
    if isinstance(image, Image.Image):
        source = image.copy()
        image_name = "in-memory-image"
    else:
        image_path = Path(image)
        image_name = image_path.name
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    try:
        original_size = tuple(map(int, source.size))
        processed = resize_pad_rgb(source, loaded.max_long_edge)
    finally:
        source.close()
    try:
        vector = np.asarray(encoder.encode([processed]), dtype=np.float32)
    finally:
        processed.close()
    if vector.shape != (1, len(loaded.estimator.named_steps["scaler"].mean_)) or not np.isfinite(vector).all():
        raise ValueError("Inference encoder returned a feature vector incompatible with the saved estimator.")
    probability = float(loaded.estimator.predict_proba(vector)[0, 1])
    return {
        "image_name": image_name,
        "prediction_unit": "original_image",
        "cancer_probability": probability,
        "threshold": loaded.threshold,
        "image_target": int(probability >= loaded.threshold),
        "predicted_class": "cancer" if probability >= loaded.threshold else "no_cancer",
        "selected_C": loaded.selected_c,
        "max_long_edge": loaded.max_long_edge,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "release_id": loaded.release_id,
        "study_id": loaded.study_id,
    }


def render_image_classification_report(study_root: Path) -> dict[str, Any]:
    """Verify saved OOF outputs and return the ready-to-display report."""
    root = Path(study_root).expanduser().resolve()
    study, _ = _verify_study(root)
    rows_by_id: dict[str, dict[str, str]] = {}
    with (root / "oof_predictions.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["image_id"] in rows_by_id:
                raise ValueError(f"Duplicate OOF report row: {row['image_id']}")
            rows_by_id[row["image_id"]] = row
    if len(rows_by_id) != study["image_count"]:
        raise ValueError("OOF prediction report coverage does not match the study image count.")
    report = {
        "study_id": study["study_id"],
        "release_id": study["release_id"],
        "prediction_unit": "original_image",
        "image_count": study["image_count"],
        "patient_count": study["patient_count"],
        "metrics": study["metrics"],
        "patient_split_leakage_check": study["patient_split_leakage_check"],
        "prediction_csv": str(root / "oof_predictions.csv"),
        "deployment_model": {
            "selected_C": study["deployment_model"]["selected_C"],
            "threshold": study["deployment_model"]["threshold"],
        },
    }
    atomic_json(root / "report.json", report)
    return report
