"""Verified, shard-level resumable UNI extraction from original image ZIPs."""
from __future__ import annotations

import json
import time
import zipfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import image_classification as core, image_classification_r0_r3 as units
from .image_classification_improvement import SOURCE_FIELDS
from .io import atomic_json, file_hash, fingerprint
from .uni_views import IMAGE_DIM, PROTOCOL, UNI_DIM, ViewConfig, ViewEncoder, embed_image, pool


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Require complete image-level source lineage and the approved label map."""
    if not rows or len({r.get("image_id") for r in rows}) != len(rows):
        raise ValueError("Original image rows must be nonempty and unique.")
    for row in rows:
        expected_target, grade_error = core._parse_glade(row.get("raw_glade"))
        if (any(k not in row for k in SOURCE_FIELDS) or not row["patient_id"]
                or type(row["objective_lens"]) is not int or row["objective_lens"] not in {4, 10, 40}
                or grade_error or type(row["image_target"]) is not int
                or row["image_target"] != expected_target):
            raise ValueError("Invalid image label/lens/source metadata.")
    if len({r["original_image_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate original content requires audit before extraction.")


def _write_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Write one recoverable payload; commit markers are always published last."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(rows: list[dict[str, Any]], archives: dict[int, Path], output: Path, encoder: ViewEncoder, *,
            config: ViewConfig = ViewConfig(), shard_size: int = 32, batch_size: int = 4,
            budget_seconds: float = 14_400, max_new_shards: int | None = None,
            recover_lock: bool = False) -> dict[str, Any]:
    """Hash originals, commit image-shards and reuse only verified exact identities.

    Deadlines are checked before decoding each image and before each encoder
    batch. An interrupted incomplete shard is recomputed; committed shards stay.
    """
    config.validate()
    validate_rows(rows)
    if (type(shard_size) is not int or shard_size < 1 or type(batch_size) is not int or batch_size < 1
            or not np.isfinite(budget_seconds) or budget_seconds <= 0
            or max_new_shards is not None and (type(max_new_shards) is not int or max_new_shards < 0)):
        raise ValueError("Invalid extraction shard/batch/budget limits.")
    lenses = {r["objective_lens"] for r in rows}
    if set(archives) != lenses:
        raise ValueError("Source archives must cover exactly the selected objective lenses.")
    deadline = time.monotonic() + budget_seconds
    archive_hashes = {str(lens): file_hash(path) for lens, path in archives.items()}
    if any(archive_hashes[str(r["objective_lens"])] != r["source_archive_sha256"] for r in rows):
        raise ValueError("Original ZIP checksum differs from baseline lineage.")
    descriptor = json.loads(json.dumps({"protocol": PROTOCOL, "encoder": encoder.descriptor(),
        "views": asdict(config), "batch_size": batch_size, "shard_size": shard_size,
        "source_rows": [{k: r[k] for k in SOURCE_FIELDS} for r in rows], "archive_sha256": archive_hashes,
        "implementation_sha256": file_hash(Path(__file__))}))
    root = Path(output) / fingerprint(descriptor)
    if root.is_symlink():
        raise ValueError("Cache root cannot be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"state": "running", "cache_root": str(root), "new_shards": 0}
    shard_manifest = []

    class BoundedEncoder:
        feature_dim = encoder.feature_dim

        def encode(self, images: list) -> np.ndarray:
            if time.monotonic() >= deadline:
                raise TimeoutError("Extraction budget reached before next batch.")
            return encoder.encode(images)

    with core._RunLock(root, recover=recover_lock), ExitStack() as stack:
        atomic_json(root / "config.json", descriptor)
        zipped = {lens: stack.enter_context(zipfile.ZipFile(path)) for lens, path in archives.items()}
        try:
            for offset in range(0, len(rows), shard_size):
                selected = rows[offset:offset + shard_size]
                folder = root / "shards" / f"{offset // shard_size:05d}"
                if folder.is_symlink():
                    raise ValueError("Shard directory cannot be a symlink.")
                binding = {"cache_id": root.name, "offset": offset,
                           "image_ids": [r["image_id"] for r in selected]}
                if not units._verify(folder, binding):
                    if time.monotonic() >= deadline or max_new_shards is not None and state["new_shards"] >= max_new_shards:
                        raise TimeoutError("Committed shards preserved; resume remaining work.")
                    folder.mkdir(parents=True, exist_ok=True)
                    image_features, local_features, metadata, view_records = [], [], [], []
                    offsets = [0]
                    for row in selected:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Extraction budget reached before next image.")
                        photo, sha, _, _ = core._load_member_image(zipped[row["objective_lens"]], row,
                            max_image_bytes=512 * 1024**2, max_image_pixels=config.max_image_pixels)
                        try:
                            if sha != row["original_image_sha256"]:
                                raise ValueError("Original member checksum mismatch.")
                            combined, features, coordinates = embed_image(photo, BoundedEncoder(),
                                                                          config=config, batch_size=batch_size)
                        finally:
                            photo.close()
                        image_features.append(combined)
                        local_features.extend(features)
                        offsets.append(offsets[-1] + len(features))
                        metadata.append({k: row[k] for k in SOURCE_FIELDS})
                        view_records.append(coordinates)
                    _write_arrays(folder / "features.npz", image_features=np.asarray(image_features, dtype=np.float32),
                                  view_features=np.asarray(local_features, dtype=np.float32), offsets=np.asarray(offsets, dtype=np.int64))
                    atomic_json(folder / "rows.json", {"rows": metadata, "views": view_records})
                    units._publish(folder, binding, ["features.npz", "rows.json"])
                    state["new_shards"] += 1
                shard_manifest.append({"path": str(folder.relative_to(root)), "binding": binding,
                                       "commit_sha256": file_hash(folder / "commit.json")})
                atomic_json(root / "status.json", {**state, "verified_images": min(offset + shard_size, len(rows))})
            atomic_json(root / "release.json", {"descriptor": descriptor, "shards": shard_manifest})
            units._publish(root, {"cache_id": root.name}, ["config.json", "release.json"])
            state.update(state="complete", image_count=len(rows))
        except TimeoutError as error:
            state.update(state="partial", reason=str(error))
        except Exception:
            atomic_json(root / "status.json", {**state, "state": "attention_required"})
            raise
        atomic_json(root / "status.json", state)
    return state


def load_cache(root: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Verify all shard commits, image order, offsets and pooling before training."""
    if not units._verify(root, {"cache_id": root.name}):
        raise ValueError("UNI cache is incomplete.")
    release = json.loads((root / "release.json").read_text())
    descriptor = release["descriptor"]
    if fingerprint(descriptor) != root.name or json.loads((root / "config.json").read_text()) != descriptor:
        raise ValueError("UNI cache descriptor mismatch.")
    rows, combined = [], []
    for index, shard in enumerate(release["shards"]):
        expected_path = f"shards/{index:05d}"
        if shard["path"] != expected_path:
            raise ValueError("Unexpected cache shard path/order.")
        folder = root / expected_path
        if folder.is_symlink() or file_hash(folder / "commit.json") != shard["commit_sha256"] or not units._verify(folder, shard["binding"]):
            raise ValueError("UNI shard checksum mismatch.")
        manifest = json.loads((folder / "rows.json").read_text())
        incoming = manifest["rows"]
        if [r["image_id"] for r in incoming] != shard["binding"]["image_ids"] or shard["binding"]["offset"] != len(rows):
            raise ValueError("UNI shard row alignment mismatch.")
        with np.load(folder / "features.npz", allow_pickle=False) as data:
            values, local, offsets = data["image_features"], data["view_features"], data["offsets"]
            if (values.shape != (len(incoming), IMAGE_DIM) or values.dtype != np.float32
                    or local.ndim != 2 or local.shape[1] != UNI_DIM or local.dtype != np.float32
                    or offsets.shape != (len(incoming) + 1,) or offsets.dtype != np.int64
                    or offsets[0] != 0 or offsets[-1] != len(local) or np.any(np.diff(offsets) < 2)
                    or len(manifest["views"]) != len(incoming) or not np.isfinite(values).all()):
                raise ValueError("UNI cache dimensions/dtypes/offsets mismatch.")
            for i, coords in enumerate(manifest["views"]):
                if len(coords) != offsets[i + 1] - offsets[i] or not np.array_equal(pool(local[offsets[i]:offsets[i + 1]]), values[i]):
                    raise ValueError("UNI pooled vector/view lineage mismatch.")
            combined.extend(values.copy())
        rows.extend(incoming)
    validate_rows(rows)
    if rows != descriptor["source_rows"]:
        raise ValueError("UNI source coverage differs from descriptor.")
    return np.asarray(combined, dtype=np.float32), rows, descriptor
