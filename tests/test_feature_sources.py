"""End-to-end tests for metadata-only source planning and bounded part staging."""
from __future__ import annotations

import errno
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import histology_data.feature_sources as feature_sources
from histology_data.feature_sources import (
    build_source_plan,
    read_feature_payload,
    stage_feature_part,
)


def _metadata(path: Path, images: list[tuple[str, int]]) -> Path:
    rows = []
    for index, (image_id, lens) in enumerate(images):
        rows.append({
            "Ten_File": f"{image_id}.tif",
            "Ma_Nam": "YCT 26",
            "Ma_So": str(index + 1),
            "Do_Phong_Dai": f"{lens}X",
            "Glade": f"raw-grade-{index}",
            "Ket_Luan": "CARCINÔM TUYẾN",
            "Ten_Slide": f"slide-{index}",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def _plan(metadata: Path, source_root: Path, *, expected_archives: int | None = 1,
          expected_pngs: int | None = 1, source_kind: str = "zip", **kwargs: object) -> dict:
    return build_source_plan(metadata, source_root, source_kind,
                             expected_archives=expected_archives,
                             expected_pngs=expected_pngs, **kwargs)


def test_zip_plan_stages_crc_checked_payload_and_keeps_weak_metadata(tmp_path: Path) -> None:
    """Inventory, stage, read, and clean one ZIP while preserving case-level provenance."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    payload_a = b"png payload A"
    payload_b = b"png payload B"
    source_root = tmp_path / "source"
    archive_path = source_root / "Tiles-4.zip"
    _write_zip(archive_path, {
        "Tiles/FIELD_A/0_0.png": payload_a,
        "Tiles/FIELD_A/8_4(1).png": payload_b,
    })

    plan = _plan(metadata, source_root, expected_pngs=2)
    part = plan["parts"][0]
    assert plan["source_complete"] is True
    assert plan["observed_sources"] == plan["expected_sources"] == 1
    assert plan["observed_pngs"] == 2
    assert plan["coverage"] == {"4": 2, "10": 0, "40": 0}
    assert {row["source_member"] for row in part["records"]} == {
        "Tiles/FIELD_A/0_0.png", "Tiles/FIELD_A/8_4(1).png",
    }
    variant = next(row for row in part["records"] if row["coordinate_variant"] == 1)
    assert variant["metadata_match"] is True
    assert variant["objective_lens"] == 4
    assert variant["raw_glade"] == "raw-grade-0"
    assert variant["patient_id"] is None
    assert variant["zip_crc32"] == f"{zipfile.crc32(payload_b):08x}"

    scratch = tmp_path / "ssd"
    with stage_feature_part(part, scratch) as staged:
        assert staged["source_path"] != part["source_path"]
        assert len(staged["source_archive_sha256"]) == 64
        assert isinstance(staged["zip_handle"], zipfile.ZipFile)
        assert read_feature_payload(staged, variant) == payload_b
        assert list(scratch.iterdir())
    assert list(scratch.iterdir()) == []


def test_staged_payload_lookup_does_not_rescan_zip_central_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated payload reads must use the ZIP name index instead of O(N) scans."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    members = {f"Tiles/FIELD_A/{index}_0.png": f"payload-{index}".encode() for index in range(32)}
    _write_zip(source_root / "Tiles-01.zip", members)
    part = _plan(metadata, source_root, expected_pngs=len(members))["parts"][0]

    with stage_feature_part(part, tmp_path / "scratch") as staged:
        def reject_member_list(_archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
            raise AssertionError("payload lookup rescanned the ZIP central directory")

        monkeypatch.setattr(zipfile.ZipFile, "infolist", reject_member_list)
        for row in staged["records"]:
            assert read_feature_payload(staged, row) == members[row["source_member"]]


def test_part_ids_survive_relocation_later_archives_and_metadata_edits(tmp_path: Path) -> None:
    """Per-part identity excludes root paths, later sources, mtime, and weak labels."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4), ("FIELD_B", 10)])
    archive_bytes_root = tmp_path / "first-root"
    original = archive_bytes_root / "Tiles-4.zip"
    _write_zip(original, {"Tiles/FIELD_A/0_0.png": b"field a"})
    first_id = _plan(metadata, archive_bytes_root, expected_pngs=1)["parts"][0]["part_id"]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copyfile(original, relocated / original.name)
    _write_zip(relocated / "Tiles-10.zip", {"Tiles/FIELD_B/1_0.png": b"field b"})
    plan = _plan(metadata, relocated, expected_archives=2, expected_pngs=2)
    assert plan["source_complete"] is True
    original_part = next(part for part in plan["parts"] if part["source_name"] == "Tiles-4.zip")
    assert original_part["part_id"] == first_id

    table = pd.read_csv(metadata, dtype=str)
    table.loc[0, "Glade"] = "edited raw metadata"
    table.to_csv(metadata, index=False)
    edited = _plan(metadata, relocated, expected_archives=2, expected_pngs=2)
    assert edited["metadata_sha256"] != plan["metadata_sha256"]
    edited_original = next(part for part in edited["parts"] if part["source_name"] == "Tiles-4.zip")
    assert edited_original["part_id"] == first_id


def test_zip_source_root_symlink_resolves_to_same_inventory_and_part_ids(tmp_path: Path) -> None:
    """An explicitly selected Drive-style root shortcut resolves before scanning."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    target_root = tmp_path / "actual-drive-folder"
    _write_zip(target_root / "Tiles-01.zip", {"Tiles/FIELD_A/0_0.png": b"payload"})
    shortcut = tmp_path / "shared-tiles-shortcut"
    shortcut.symlink_to(target_root, target_is_directory=True)

    direct = _plan(metadata, target_root, expected_pngs=1)
    via_shortcut = _plan(metadata, shortcut, expected_pngs=1)
    assert via_shortcut["source_complete"] is True
    assert via_shortcut["parts"][0]["part_id"] == direct["parts"][0]["part_id"]
    assert via_shortcut["parts"][0]["source_path"] == direct["parts"][0]["source_path"]


def test_directory_source_root_shortcut_resolves_but_child_symlinks_are_rejected(tmp_path: Path) -> None:
    """Resolve a selected root shortcut while refusing symlinked members below it."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    target_root = tmp_path / "actual-Tiles"
    image = target_root / "FIELD_A" / "0_0.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"inside")
    shortcut = tmp_path / "Tiles-shortcut"
    shortcut.symlink_to(target_root, target_is_directory=True)

    direct = build_source_plan(metadata, target_root, "directory", expected_archives=None, expected_pngs=1)
    via_shortcut = build_source_plan(metadata, shortcut, "directory", expected_archives=None, expected_pngs=1)
    assert via_shortcut["source_complete"] is True
    assert via_shortcut["parts"][0]["part_id"] == direct["parts"][0]["part_id"]

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "1_0.png").write_bytes(b"outside")
    child_link = target_root / "FIELD_A" / "1_0.png"
    child_link.symlink_to(outside / "1_0.png")
    linked_directory = target_root / "FIELD_EXTERNAL"
    linked_directory.symlink_to(outside, target_is_directory=True)
    rejected = build_source_plan(metadata, shortcut, "directory", expected_archives=None,
                                 expected_pngs=None)
    assert rejected["source_complete"] is False
    assert any("Symlink" in error["error"] for error in rejected["source_errors"])
    assert len(rejected["parts"]) == 1
    assert rejected["parts"][0]["blocked"] is False


def test_directory_alias_stages_virtual_member_and_checks_expected_sha(tmp_path: Path) -> None:
    """Explicit aliases retain virtual provenance and validate bytes only at staging."""
    tiles = tmp_path / "Tiles"
    actual = tiles / "FIELD_SOURCE" / "0_0.png"
    actual.parent.mkdir(parents=True)
    payload = b"directory PNG bytes"
    actual.write_bytes(payload)
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_VIRTUAL", 10), ("FIELD_SOURCE", 4)])
    alias = {
        "FIELD_VIRTUAL/5_3.png": {
            "source_member": "FIELD_SOURCE/0_0.png",
            "expected_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    plan = build_source_plan(metadata, tiles, "directory", directory_part_size=8,
                             expected_archives=None, expected_pngs=2, aliases=alias)
    part = plan["parts"][0]
    row = next(item for item in part["records"] if item["alias_reconstructed"])
    assert plan["source_complete"] is True
    assert row["source_member"] == "Tiles/FIELD_VIRTUAL/5_3.png"
    assert row["read_member"] == "FIELD_SOURCE/0_0.png"
    assert row["alias_reconstructed"] is True
    assert row["objective_lens"] == 10

    scratch = tmp_path / "scratch"
    with stage_feature_part(part, scratch) as staged:
        staged_alias = next(item for item in staged["records"] if item["alias_reconstructed"])
        assert read_feature_payload(staged, staged_alias) == payload
    assert list(scratch.iterdir()) == []

    bad_alias = {"FIELD_VIRTUAL/5_3.png": {
        "source_member": "FIELD_SOURCE/0_0.png", "expected_sha256": "0" * 64,
    }}
    bad_part = build_source_plan(metadata, tiles, "directory", expected_archives=None,
                                 expected_pngs=2, aliases=bad_alias)["parts"][0]
    with pytest.raises(ValueError, match="Alias SHA-256 mismatch"):
        with stage_feature_part(bad_part, scratch):
            pytest.fail("invalid alias must not be yielded")
    assert list(scratch.iterdir()) == []


def test_directory_snapshot_detects_mtime_change_without_renaming_part(tmp_path: Path) -> None:
    """Directory records freeze membership and reject stale same-size source bytes."""
    tiles = tmp_path / "Tiles"
    source = tiles / "FIELD_A" / "2_2.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 40)])
    plan = build_source_plan(metadata, tiles, "directory", expected_archives=None, expected_pngs=1)
    part = plan["parts"][0]
    part_id = part["part_id"]
    old_mtime = source.stat().st_mtime_ns
    source.write_bytes(b"modified")
    source.touch()
    assert source.stat().st_size == part["records"][0]["byte_size"]
    assert source.stat().st_mtime_ns != old_mtime
    rebuilt = build_source_plan(metadata, tiles, "directory", expected_archives=None, expected_pngs=1)
    assert rebuilt["parts"][0]["part_id"] == part_id
    with pytest.raises(ValueError, match="changed since inventory"):
        with stage_feature_part(part, tmp_path / "scratch"):
            pytest.fail("stale directory snapshot must not stage")


def test_directory_and_zip_records_share_numeric_coordinate_order(tmp_path: Path) -> None:
    """Equivalent source layouts sort records by numeric image coordinates."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    members = {
        "Tiles/FIELD_A/10_0.png": b"ten",
        "Tiles/FIELD_A/2_0.png": b"two",
        "Tiles/FIELD_A/0_10.png": b"lower-row",
    }
    tiles = tmp_path / "Tiles"
    for member, payload in members.items():
        relative = Path(*Path(member).parts[1:])
        destination = tiles / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    archive_root = tmp_path / "archives"
    _write_zip(archive_root / "Tiles-01.zip", members)

    directory_plan = build_source_plan(metadata, tiles, "directory", expected_archives=None, expected_pngs=3)
    zip_plan = _plan(metadata, archive_root, expected_pngs=3)
    directory_order = [row["source_member"] for row in directory_plan["parts"][0]["records"]]
    zip_order = [row["source_member"] for row in zip_plan["parts"][0]["records"]]
    assert directory_order == zip_order == [
        "Tiles/FIELD_A/2_0.png", "Tiles/FIELD_A/10_0.png", "Tiles/FIELD_A/0_10.png",
    ]


def test_unsafe_zip_path_is_rejected_while_independent_valid_zip_is_kept(tmp_path: Path) -> None:
    """Traversal in one archive is reported without discarding a separate good part."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    _write_zip(source_root / "Tiles-01.zip", {"Tiles/FIELD_A/0_0.png": b"good"})
    _write_zip(source_root / "Tiles-02.zip", {"../FIELD_A/1_0.png": b"unsafe"})

    plan = _plan(metadata, source_root, expected_archives=2, expected_pngs=1)
    assert plan["observed_sources"] == 2
    assert len(plan["parts"]) == 1
    assert plan["parts"][0]["source_name"] == "Tiles-01.zip"
    assert plan["source_complete"] is False
    assert any("Unsafe source member" in error["error"] for error in plan["source_errors"])


def test_coordinate_variants_are_retained_and_same_variant_duplicates_block(tmp_path: Path) -> None:
    """Suffix variants are reviewable; ambiguous duplicate variants block their part."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    _write_zip(source_root / "Tiles-01.zip", {
        "Tiles/FIELD_A/0_0.png": b"one",
        "Tiles/FIELD_A/0_0(1).png": b"two",
        "Alternative/FIELD_A/0_0.png": b"ambiguous duplicate variant",
    })
    plan = _plan(metadata, source_root, expected_pngs=3)
    part = plan["parts"][0]
    assert len(part["records"]) == 3
    assert {row["coordinate_variant"] for row in part["records"]} == {0, 1}
    assert sum(row["coordinate_collision"] for row in part["records"]) == 3
    assert part["blocked"] is True
    assert len(plan["coordinate_collisions"]) == 1
    assert "ambiguous_duplicate_coordinates" in part["blockers"]
    assert plan["source_complete"] is False
    with pytest.raises(ValueError, match="blocked part"):
        with stage_feature_part(part, tmp_path / "scratch"):
            pytest.fail("ambiguous coordinates must block their part")


def test_unmatched_png_stays_in_plan_and_blocks_only_its_part(tmp_path: Path) -> None:
    """An unmatched patch remains auditable and cannot be silently staged."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    _write_zip(source_root / "Tiles-01.zip", {
        "Tiles/FIELD_A/0_0.png": b"matched",
        "Tiles/UNKNOWN_FIELD/1_0.png": b"unmatched",
    })
    plan = _plan(metadata, source_root, expected_pngs=2)
    part = plan["parts"][0]
    assert plan["unmatched_png_count"] == 1
    unmatched = next(row for row in part["records"] if row["image_id"] == "UNKNOWN_FIELD")
    assert unmatched["metadata_match"] is False
    assert unmatched["source_member"] == "Tiles/UNKNOWN_FIELD/1_0.png"
    assert part["blocked"] is True
    assert "unmatched_metadata" in part["blockers"]


def test_duplicate_zip_paths_block_overlapping_parts(tmp_path: Path) -> None:
    """Repeated logical ZIP paths are reported and neither overlapping part is usable."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    member = "Tiles/FIELD_A/0_0.png"
    _write_zip(source_root / "Tiles-01.zip", {member: b"first"})
    _write_zip(source_root / "Tiles-02.zip", {member: b"second"})
    plan = _plan(metadata, source_root, expected_archives=2, expected_pngs=2)
    assert len(plan["parts"]) == 2
    assert all(part["blocked"] for part in plan["parts"])
    assert all("overlapping_zip_member_path" in part["blockers"] for part in plan["parts"])
    assert any("multiple ZIPs" in error["error"] for error in plan["source_errors"])


def test_staging_detects_payload_crc_corruption_and_cleans_scratch(tmp_path: Path) -> None:
    """Post-inventory byte damage is caught when the payload is consumed."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    archive_path = source_root / "Tiles-01.zip"
    _write_zip(archive_path, {"Tiles/FIELD_A/0_0.png": b"payload to corrupt"})
    part = _plan(metadata, source_root)["parts"][0]

    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("Tiles/FIELD_A/0_0.png")
    data_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    with archive_path.open("r+b") as stream:
        stream.seek(data_offset)
        byte = stream.read(1)
        stream.seek(data_offset)
        stream.write(bytes([byte[0] ^ 0x01]))

    scratch = tmp_path / "scratch"
    with stage_feature_part(part, scratch) as staged:
        with pytest.raises(ValueError, match="CRC"):
            read_feature_payload(staged, staged["records"][0])
    assert list(scratch.iterdir()) == []


def test_partial_source_counts_are_pending_and_incomplete_zip_is_classified(tmp_path: Path) -> None:
    """Expected-count shortfalls wait for uploads, while unreadable archives are tagged."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    _write_zip(source_root / "Tiles-01.zip", {"Tiles/FIELD_A/0_0.png": b"ready"})
    partial = _plan(metadata, source_root, expected_archives=2, expected_pngs=2)
    assert partial["source_complete"] is False
    assert partial["pending_sources"] == 1
    assert partial["pending_pngs"] == 1
    assert partial["pending_reasons"] == ["awaiting_sources", "awaiting_pngs"]
    assert partial["source_errors"] == []

    (source_root / "Tiles-02.zip").write_bytes(b"partial upload")
    interrupted = _plan(metadata, source_root, expected_archives=2, expected_pngs=2)
    assert interrupted["source_complete"] is False
    assert any(error.get("status") == "incomplete_upload" for error in interrupted["source_errors"])


def test_transient_retry_deadline_and_consumer_failure_cleanup(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient I/O retries are bounded; timeout and consumer failures clean scratch."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    source_root = tmp_path / "archives"
    _write_zip(source_root / "Tiles-01.zip", {"Tiles/FIELD_A/0_0.png": b"payload"})
    part = _plan(metadata, source_root)["parts"][0]
    scratch = tmp_path / "scratch"
    original_prepare = feature_sources._prepare_staged_part
    attempts = 0

    def transient_once(*args: object, **kwargs: object) -> tuple[dict, zipfile.ZipFile | None]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, "temporary read failure")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(feature_sources, "_prepare_staged_part", transient_once)
    with stage_feature_part(part, scratch, retries=1) as staged:
        assert read_feature_payload(staged, staged["records"][0]) == b"payload"
    assert attempts == 2
    assert list(scratch.iterdir()) == []

    monkeypatch.setattr(feature_sources, "_prepare_staged_part", original_prepare)
    with pytest.raises(TimeoutError, match="deadline"):
        with stage_feature_part(part, scratch, deadline=0.0):
            pytest.fail("expired deadline must not yield")
    assert list(scratch.iterdir()) == []

    with pytest.raises(RuntimeError, match="consumer failure"):
        with stage_feature_part(part, scratch):
            raise RuntimeError("consumer failure")
    assert list(scratch.iterdir()) == []


def test_metadata_lens_is_mandatory(tmp_path: Path) -> None:
    """Metadata with an absent objective lens is rejected before source inventory."""
    metadata = _metadata(tmp_path / "metadata.csv", [("FIELD_A", 4)])
    table = pd.read_csv(metadata, dtype=str)
    table.loc[0, "Do_Phong_Dai"] = ""
    table.to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="objective lens"):
        build_source_plan(metadata, tmp_path / "missing", "zip")
