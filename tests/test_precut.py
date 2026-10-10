"""End-to-end coverage for reading reviewed patch ZIPs and preserving provenance."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from histology_data.precut import (
    audit_import,
    build_inventory,
    import_selected,
    inspect_inventory,
    load_inventory,
    verify_part,
    write_inventory,
)
from scripts.build_data_runtime import build_runtime


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    rows = []
    sources = [tmp_path / f"tiles-{lens}.zip" for lens in (4, 10, 40)]
    for lens, source in zip((4, 10, 40), sources, strict=True):
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for case_index in range(2):
                image_id = f"IMG_{lens}x_{case_index}"
                rows.append({
                    "Ten_File": f"{image_id}.tif", "Ma_Nam": "YCT 26", "Ma_So": str(case_index + 1),
                    "Do_Phong_Dai": f"{lens}X", "Glade": str(case_index), "Ten_Slide": "Slide1-2",
                    "Ket_Luan": "TĂNG SẢN LÀNH TÍNH" if case_index == 0 else "CARCINÔM TUYẾN",
                })
                for x, color in ((0, (255, 255, 255)), (256, (90 + lens, 20 + case_index, 130))):
                    buffer = io.BytesIO()
                    Image.new("RGB", (512, 512), color).save(buffer, format="PNG")
                    archive.writestr(f"Tiles/{image_id}/{x}_0.png", buffer.getvalue())
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return metadata, sources


def test_precut_smoke_reads_examples_per_lens_and_keeps_case_labels_weak(tmp_path: Path) -> None:
    """Run metadata join, ZIP CRC read, PNG decode, and coordinate trace end to end."""
    metadata, sources = _fixture(tmp_path)
    inventory = build_inventory(metadata, sources)
    path = write_inventory(tmp_path / "inventory.json", inventory)
    loaded = load_inventory(path)
    result = inspect_inventory(loaded, per_lens=2)

    assert result["selected_by_lens"] == {"4": 2, "10": 2, "40": 2}
    assert {row["stored_width"] for row in result["checked"]} == {512}
    assert {row["stored_height"] for row in result["checked"]} == {512}
    assert {row["objective_lens"] for row in result["checked"]} == {4, 10, 40}
    assert {row["candidate_case_id"] for row in result["checked"]} == {"YCT26_1", "YCT26_2"}
    assert all(row["patient_id"] is None for row in result["checked"])
    assert all(row["label_status"] == "candidate_unreviewed_case_label" for row in result["checked"])
    assert all(row["crop_geometry_status"].endswith("unverified") for row in result["checked"])
    assert result["training_ready"] is False


def test_precut_import_is_per_zip_resumable_and_verifies_published_index(tmp_path: Path) -> None:
    """Commit one ZIP, resume by verifying it, and fail closed after output tampering."""
    metadata, sources = _fixture(tmp_path)
    inventory = build_inventory(metadata, sources)
    inventory_path = write_inventory(tmp_path / "inventory.json", inventory)
    loaded = load_inventory(inventory_path)
    result = import_selected(loaded, tmp_path / "out", archive_names=[sources[0].name])
    part_id = result["parts"][0]["part_id"]
    part = tmp_path / "out" / loaded["inventory_id"] / part_id

    assert result["new_parts"] == 1 and result["reused_parts"] == 0
    assert verify_part(part)
    assert import_selected(loaded, tmp_path / "out", archive_names=[sources[0].name])["reused_parts"] == 1
    rows = [json.loads(line) for line in (part / "tile_index.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {row["objective_lens"] for row in rows} == {4}
    assert all(row["bag_id"] is None for row in rows)
    assert all(row["source_png_sha256"] and row["rgb_pixel_sha256"] for row in rows)
    assert len(rows[0]["source_png_sha256"]) == 64

    (part / "tile_index.jsonl").write_text("tampered\n")
    assert verify_part(part) is False
    with pytest.raises(ValueError, match="corrupt or incompatible"):
        import_selected(loaded, tmp_path / "out", archive_names=[sources[0].name])


def test_import_stages_one_zip_locally_and_cleans_it_after_commit(tmp_path: Path) -> None:
    """ZIP import uses local scratch and removes the copy after its Drive-ready commit."""
    metadata, sources = _fixture(tmp_path)
    inventory = build_inventory(metadata, sources)
    result = import_selected(
        inventory, tmp_path / "parts", archive_names=[sources[0].name],
        work_root=tmp_path / "scratch",
    )
    stage = tmp_path / "scratch" / inventory["inventory_id"] / sources[0].name

    assert result["new_parts"] == 1
    assert not stage.exists()
    assert result["parts"][0]["tile_count"] == 4


def test_full_import_audit_checks_every_part_and_cross_zip_duplicates(tmp_path: Path) -> None:
    """Global audit joins committed parts and reports identical pixels across ZIPs."""
    metadata, sources = _fixture(tmp_path)
    inventory = build_inventory(metadata, sources)
    result = import_selected(inventory, tmp_path / "parts")
    assert result["new_parts"] == 3
    scratch = tmp_path / "scratch-audit"
    audit = audit_import(inventory, tmp_path / "parts", tmp_path / "dataset-audit.json", scratch)

    assert audit["verified_zip_parts"] == 3
    assert audit["verified_patch_rows"] == sum(inventory["coverage"].values())
    assert audit["exact_duplicate_groups"]["source_png_sha256"] >= 1
    assert audit["exact_duplicate_groups"]["rgb_pixel_sha256"] >= 1
    assert audit["training_ready"] is False
    assert list(scratch.iterdir()) == []


def test_import_records_and_hashes_unmatched_patch_without_training_gate(tmp_path: Path) -> None:
    """An unknown PNG is decoded and traceable, then blocks dataset readiness."""
    metadata, sources = _fixture(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), (3, 4, 5)).save(buffer, format="PNG")
    with zipfile.ZipFile(sources[0], "a") as archive:
        archive.writestr("Tiles/UNKNOWN_FIELD/0_0.png", buffer.getvalue())
    inventory = build_inventory(metadata, sources)
    result = import_selected(inventory, tmp_path / "parts")
    audit = audit_import(inventory, tmp_path / "parts", tmp_path / "audit.json")
    first_part = tmp_path / "parts" / inventory["inventory_id"] / result["parts"][0]["part_id"]
    rows = [json.loads(line) for line in (first_part / "tile_index.jsonl").read_text().splitlines()]
    unknown = next(row for row in rows if row["image_id"] == "UNKNOWN_FIELD")

    assert unknown["source_png_sha256"]
    assert unknown["metadata_match"] is False
    assert unknown["candidate_case_id"] is None
    assert unknown["label_status"] == "unmatched_metadata"
    assert audit["verified_patch_rows"] == sum(inventory["coverage"].values()) + 1
    assert "patches_without_metadata_match" in audit["training_blockers"]


def test_inventory_fingerprint_rejects_manifest_edits(tmp_path: Path) -> None:
    """Inventory row edits cannot silently change source/member provenance."""
    metadata, sources = _fixture(tmp_path)
    path = write_inventory(tmp_path / "inventory.json", build_inventory(metadata, sources))
    document = json.loads(path.read_text())
    document["archives"][0]["records"][0]["x"] = 123
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="fingerprint"):
        load_inventory(path)


def test_pixel_tile_identity_does_not_change_with_metadata_only_edit(tmp_path: Path) -> None:
    """Changing a weak-label metadata cell does not rename an unchanged source patch."""
    metadata, sources = _fixture(tmp_path)
    first = inspect_inventory(build_inventory(metadata, sources), per_lens=2)
    table = pd.read_csv(metadata, dtype=str)
    table.loc[0, "Glade"] = "edited-but-retained"
    table.to_csv(metadata, index=False)
    second = inspect_inventory(build_inventory(metadata, sources), per_lens=2)

    assert [row["tile_id"] for row in first["checked"]] == [row["tile_id"] for row in second["checked"]]


def test_inventory_flags_duplicate_image_coordinates_without_dropping_patch(tmp_path: Path) -> None:
    """A same image/crop coordinate stays in the inventory and is flagged for review."""
    metadata, sources = _fixture(tmp_path)
    with zipfile.ZipFile(sources[1], "a") as archive:
        buffer = io.BytesIO()
        Image.new("RGB", (512, 512), (1, 2, 3)).save(buffer, format="PNG")
        archive.writestr("Alternative/IMG_4x_0/0_0.png", buffer.getvalue())
    inventory = build_inventory(metadata, sources)
    assert inventory["coordinate_collision_count"] == 1
    assert len(inventory["coordinate_collisions"][0]["members"]) == 2
    assert "coordinate_collisions_require_review" in inventory["training_blockers"]


def test_cli_smoke_and_colab_runtime_include_precut_importer(tmp_path: Path) -> None:
    """Exercise the CLI as Colab will call it and ensure the runtime ZIP ships the module."""
    metadata, sources = _fixture(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    smoke_path = tmp_path / "smoke.json"
    command = [sys.executable, "-m", "histology_data", "precut-inventory", "--metadata", str(metadata),
               "--output", str(inventory_path)]
    for source in sources:
        command.extend(["--source", str(source)])
    indexed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert indexed.returncode == 0, indexed.stderr

    smoke = subprocess.run(
        [sys.executable, "-m", "histology_data", "precut-smoke", "--inventory", str(inventory_path),
         "--output", str(smoke_path), "--per-lens", "2"],
        capture_output=True, text=True, check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert json.loads(smoke_path.read_text())["selected_patch_count"] == 6

    runtime = build_runtime(tmp_path / "runtime.zip")
    with zipfile.ZipFile(runtime) as archive:
        names = set(archive.namelist())
    assert "histology_data/precut.py" in names
    assert "docs/PRECUT_IMPORT.md" in names
