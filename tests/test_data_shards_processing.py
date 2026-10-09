"""End-to-end checks for CPU shard decoding, tiling and verified publication."""
from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from histology_data.shards import pack_catalog
from PIL import Image

from histology_data.catalog import build_catalog
from histology_data.processing import process_shard


def _make_release(root: Path, pixels: np.ndarray | None = None,
                  raw_image: bytes | None = None) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir(parents=True)
    image_path = source / "field.png"
    if raw_image is not None:
        image_path.write_bytes(raw_image)
    else:
        assert pixels is not None
        Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(image_path, format="PNG")
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade",
                         "Ket_Luan", "Ten_Slide"])
        writer.writerow(["field.png", "2020", "001", "10x", "G3", "review pending", "slide-a"])
    catalog = build_catalog(metadata, [source], lenses=[10], labels_reviewed=False)
    release = root / "release"
    descriptor = pack_catalog(catalog, release, max_bytes=1_000_000)
    assert len(descriptor["shards"]) == 1
    return release, descriptor["shards"][0]["shard_id"]


def _tissue_image(width: int, height: int) -> np.ndarray:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[:, :] = (130, 65, 105)
    return pixels


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_processing_preserves_coordinates_padding_and_provenance(tmp_path: Path) -> None:
    pixels = _tissue_image(5, 3)
    release, shard_id = _make_release(tmp_path, pixels)
    result = process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output",
                           {"tile_size": 4, "stride": 4, "min_tissue": 0.1,
                            "blur_threshold": 1_000_000.0})

    assert result["reused"] is False
    assert result["tiles_written"] == 2
    rows = _rows(tmp_path / "output" / shard_id / result["processing_id"] / "tiles.jsonl")
    assert [(row["x"], row["y"], row["w"], row["h"], row["valid_w"], row["valid_h"])
            for row in rows] == [(0, 0, 4, 4, 4, 3), (4, 0, 4, 4, 1, 3)]
    assert all(row["case_id"] == "2020_001" and row["bag_id"] == "slide-a" for row in rows)
    assert all("case_label" not in row and "isup" not in row for row in rows)

    with tarfile.open(tmp_path / "output" / shard_id / result["processing_id"] / "patches.tar",
                      mode="r:") as archive:
        patch_file = archive.extractfile(f"patches/{rows[1]['tile_id']}.png")
        assert patch_file is not None
        with Image.open(io.BytesIO(patch_file.read())) as patch_image:
            patch = np.asarray(patch_image)
    assert patch.shape == (4, 4, 3)
    assert np.all(patch[:3, 0] == pixels[:, 4])
    assert np.all(patch[:3, 1:] == 255)
    assert np.all(patch[3] == 255)

    qc = _rows(tmp_path / "output" / shard_id / result["processing_id"] / "qc.jsonl")[0]
    assert qc["decode_ok"] is True
    assert qc["source_mode"] == "RGB" and qc["output_mode"] == "RGB"
    assert qc["objective_lens"] == 10
    assert qc["focus_score"] >= 0
    assert "focus_below_review_threshold" in qc["reasons"]
    assert result["review_required"] is True


def test_no_tissue_is_saved_as_review_qc_without_patch_targets(tmp_path: Path) -> None:
    white = np.full((3, 5, 3), 255, dtype=np.uint8)
    release, shard_id = _make_release(tmp_path, white)
    result = process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output",
                           {"tile_size": 4, "stride": 4, "min_tissue": 0.0})

    directory = tmp_path / "output" / shard_id / result["processing_id"]
    assert result["complete"] is True
    assert result["tiles_written"] == 0
    assert result["training_ready"] is False
    qc = _rows(directory / "qc.jsonl")[0]
    assert qc["status"] == "review"
    assert qc["reasons"] == ["no_tissue", "no_tiles_met_tissue_threshold"]
    with tarfile.open(directory / "patches.tar", mode="r:") as archive:
        assert archive.getmembers() == []


def test_cap_marks_output_smoke_and_resume_verifies_committed_files(tmp_path: Path) -> None:
    release, shard_id = _make_release(tmp_path, _tissue_image(8, 4))
    config = {"tile_size": 4, "stride": 4, "min_tissue": 0.1,
              "max_tiles_per_image": 1}
    first = process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output", config)
    resumed = process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output", config)

    assert first["tiles_written"] == 1
    assert first["mode"] == "smoke"
    assert first["training_ready"] is False
    assert resumed["reused"] is True
    assert resumed["processing_id"] == first["processing_id"]

    tiles_path = tmp_path / "output" / shard_id / first["processing_id"] / "tiles.jsonl"
    tiles_path.write_text(tiles_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output", config)


def test_tile_ids_are_stable_across_release_and_work_paths(tmp_path: Path) -> None:
    pixels = _tissue_image(5, 3)
    release_a, shard_a = _make_release(tmp_path / "profile-a", pixels)
    release_b, shard_b = _make_release(tmp_path / "profile-b", pixels)
    config = {"tile_size": 4, "stride": 4, "min_tissue": 0.1}
    result_a = process_shard(release_a, shard_a, tmp_path / "ssd-a", tmp_path / "out-a", config)
    result_b = process_shard(release_b, shard_b, tmp_path / "ssd-b", tmp_path / "out-b", config)
    rows_a = _rows(tmp_path / "out-a" / shard_a / result_a["processing_id"] / "tiles.jsonl")
    rows_b = _rows(tmp_path / "out-b" / shard_b / result_b["processing_id"] / "tiles.jsonl")

    assert result_a["catalog_id"] == result_b["catalog_id"]
    assert [row["tile_id"] for row in rows_a] == [row["tile_id"] for row in rows_b]


def test_corrupt_image_fails_without_publishing_success(tmp_path: Path) -> None:
    release, shard_id = _make_release(tmp_path, raw_image=b"not a valid image")
    with pytest.raises(ValueError, match="Failed to decode source image"):
        process_shard(release, shard_id, tmp_path / "ssd", tmp_path / "output")

    assert list((tmp_path / "output").rglob("commit.json")) == []


@pytest.mark.parametrize("config", [
    {"tile_size": 0},
    {"stride": 0},
    {"min_tissue": 1.01},
    {"max_tiles_per_image": -1},
    {"blur_threshold": float("nan")},
    {"unexpected": True},
])
def test_invalid_processing_parameters_are_rejected(tmp_path: Path, config: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        process_shard(tmp_path / "missing-release", "shard-000001", tmp_path / "ssd",
                      tmp_path / "output", config)
