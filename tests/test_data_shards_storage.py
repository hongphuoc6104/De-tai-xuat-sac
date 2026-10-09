"""End-to-end coverage for raw catalog sharding, verification and staging."""
from __future__ import annotations

import csv
import io
import json
import random
import struct
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from histology_data.catalog import build_catalog
from histology_data.io import file_hash
from histology_data.shards import pack_catalog, stage_shard, verify_release


def _png(index: int, width: int = 32, height: int = 32) -> bytes:
    pixels = random.Random(index).randbytes(width * height * 3)
    from io import BytesIO

    buffer = BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_catalog(tmp_path: Path, count: int = 4, layout: str = "mixed") -> dict:
    directory = tmp_path / "fields"
    directory.mkdir()
    archive_path = tmp_path / "fields.zip"
    rows = []
    zip_members: list[tuple[str, bytes]] = []
    for index in range(count):
        name = f"field_{index}.png"
        payload = _png(index)
        to_zip = layout == "zip" or (layout == "mixed" and index % 2 == 1)
        if to_zip:
            zip_members.append((f"slides/{name}", payload))
        else:
            (directory / name).write_bytes(payload)
        rows.append({
            "Ten_File": name,
            "Ma_Nam": "2024",
            "Ma_So": str(index + 1),
            "Do_Phong_Dai": "4X",
            "Glade": "unreviewed text",
            "Ket_Luan": "CARCINOMA" if index % 2 == 0 else "PROSTATE HYPERPLASIA",
            "Ten_Slide": f"slide-{index}",
        })
    if zip_members:
        import zipfile

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, payload in zip_members:
                archive.writestr(member, payload)
    sources = [directory]
    if layout == "zip":
        sources = [archive_path]
    elif layout == "mixed":
        sources.append(archive_path)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return build_catalog(metadata, sources, lenses=[4])


def _read_manifest(release_root: Path, entry: dict) -> dict:
    return json.loads((release_root / entry["manifest_name"]).read_text(encoding="utf-8"))


def test_directory_and_zip_shards_preserve_case_provenance_and_stage(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=4, layout="mixed")
    release_root = tmp_path / "release"

    descriptor = pack_catalog(catalog, release_root, max_bytes=40_960)

    assert descriptor["complete"] is True
    assert descriptor["catalog"]["catalog_id"] == catalog["catalog_id"]
    assert len(descriptor["shards"]) == 1
    verified = verify_release(release_root)
    manifest = _read_manifest(release_root, verified["shards"][0])
    assert len(manifest["images"]) == 4
    assert all(record["label_level"] == "case" for record in manifest["images"])
    assert all(record["grade_semantics"] == "unconfirmed" for record in manifest["images"])
    assert {record["case_label"] for record in manifest["images"]} == {0, 1}
    assert all("patch_label" not in record for record in manifest["images"])

    stage_root, staged_manifest = stage_shard(release_root, "shard-000001", tmp_path / "work")
    assert staged_manifest["catalog_id"] == catalog["catalog_id"]
    for record in manifest["images"]:
        path = stage_root.joinpath(*record["member"].split("/"))
        assert path.read_bytes() == _png(int(record["image_id"].split("_")[-1]))
    repeated_root, _ = stage_shard(release_root, "shard-000001", tmp_path / "work")
    assert repeated_root == stage_root
    assert (stage_root / ".staged.json").is_file()


def test_limited_packaging_resumes_from_verified_prefix(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=5, layout="directory")
    release_root = tmp_path / "release"

    first = pack_catalog(catalog, release_root, max_bytes=10_240, max_shards=1)
    assert first["complete"] is False
    assert len(first["shards"]) == 1
    first_manifest = _read_manifest(release_root, first["shards"][0])
    first_ids = [record["image_id"] for record in first_manifest["images"]]
    assert first_ids == [image["image_id"] for image in catalog["images"][:len(first_ids)]]
    assert verify_release(release_root)["complete"] is False
    staged_root, _ = stage_shard(release_root, "shard-000001", tmp_path / "partial-work")
    assert staged_root.is_dir()
    assert verify_release(release_root)["complete"] is False

    second = pack_catalog(catalog, release_root, max_bytes=10_240, max_shards=1)
    assert len(second["shards"]) == 2
    assert second["complete"] is False
    completed = pack_catalog(catalog, release_root, max_bytes=10_240)
    assert completed["complete"] is True
    assert sum(item["image_count"] for item in completed["shards"]) == len(catalog["images"])
    assert verify_release(release_root)["complete"] is True


@pytest.mark.parametrize("layout", ["directory", "zip"])
def test_pack_rejects_changed_source_since_cataloging(tmp_path: Path, layout: str) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout=layout)
    image = catalog["images"][0]
    source = next(item for item in catalog["sources"] if item["source_id"] == image["source_id"])
    if layout == "directory":
        path = Path(source["path"]) / image["source_member"]
        path.write_bytes(_png(22))
    else:
        import zipfile

        path = Path(source["path"])
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(image["source_member"], _png(22))

    with pytest.raises(ValueError, match="changed since cataloging"):
        pack_catalog(catalog, tmp_path / "release", max_bytes=40_960)


@pytest.mark.parametrize("attack", ["traversal", "duplicate", "symlink", "hardlink"])
def test_verifier_rejects_malicious_tar_members(tmp_path: Path, attack: str) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout="directory")
    release_root = tmp_path / "release"
    descriptor = pack_catalog(catalog, release_root, max_bytes=40_960)
    entry = descriptor["shards"][0]
    manifest_path = release_root / entry["manifest_name"]
    manifest = _read_manifest(release_root, entry)
    record = manifest["images"][0]
    source_bytes = _png(0)
    tar_path = release_root / entry["tar_name"]

    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        if attack in {"symlink", "hardlink"}:
            link = tarfile.TarInfo(record["member"])
            link.type = tarfile.SYMTYPE if attack == "symlink" else tarfile.LNKTYPE
            link.linkname = "../../outside.png" if attack == "symlink" else record["member"]
            archive.addfile(link)
        else:
            image = tarfile.TarInfo(record["member"])
            image.size = len(source_bytes)
            archive.addfile(image, io.BytesIO(source_bytes))
            if attack == "duplicate":
                duplicate = tarfile.TarInfo(record["member"])
                duplicate.size = len(source_bytes)
                archive.addfile(duplicate, io.BytesIO(source_bytes))
            elif attack == "traversal":
                extra = tarfile.TarInfo("../../outside.png")
                extra.size = 1
                archive.addfile(extra, io.BytesIO(b"x"))

    new_tar_hash = file_hash(tar_path)
    manifest["tar_sha256"] = new_tar_hash
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    entry["tar_sha256"] = new_tar_hash
    entry["manifest_sha256"] = file_hash(manifest_path)
    (release_root / "release.json").write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="(Unsafe|Unexpected|duplicate|regular file|canonical)"):
        verify_release(release_root)


def test_zip_crc_is_checked_by_consuming_the_member_stream(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout="zip")
    source = Path(catalog["sources"][0]["path"])
    image = catalog["images"][0]
    import zipfile

    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(image["source_member"], _png(0), compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo(image["source_member"])
        offset = info.header_offset
    data = bytearray(source.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
    first_payload_byte = offset + 30 + name_length + extra_length
    data[first_payload_byte] ^= 0xFF
    source.write_bytes(data)

    with pytest.raises(ValueError, match="CRC verification"):
        pack_catalog(catalog, tmp_path / "release", max_bytes=40_960)


def test_pack_rejects_member_that_cannot_fit_and_invalid_limits(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout="directory")
    image_path = Path(catalog["sources"][0]["path"]) / catalog["images"][0]["source_member"]
    image_path.write_bytes(_png(3, width=128, height=128))
    # Rebuild the catalog so this test reaches the TAR capacity check.
    metadata = tmp_path / "metadata.csv"
    catalog = build_catalog(metadata, [Path(catalog["sources"][0]["path"])], lenses=[4])
    with pytest.raises(ValueError, match="cannot fit"):
        pack_catalog(catalog, tmp_path / "too-small", max_bytes=10_240)
    with pytest.raises(ValueError, match="max_bytes"):
        pack_catalog(catalog, tmp_path / "bad-size", max_bytes=0)
    with pytest.raises(ValueError, match="max_shards"):
        pack_catalog(catalog, tmp_path / "bad-count", max_bytes=40_960, max_shards=0)


def test_resume_fails_closed_when_a_committed_tar_is_corrupted(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=2, layout="directory")
    release_root = tmp_path / "release"
    descriptor = pack_catalog(catalog, release_root, max_bytes=40_960)
    tar_path = release_root / descriptor["shards"][0]["tar_name"]
    tar_path.write_bytes(tar_path.read_bytes()[:-1] + b"X")

    with pytest.raises(ValueError, match="checksum or size mismatch"):
        pack_catalog(catalog, release_root, max_bytes=40_960)


def test_catalog_content_must_match_its_identifier(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout="directory")
    catalog["images"][0]["case_label"] = 0

    with pytest.raises(ValueError, match="fingerprint"):
        pack_catalog(catalog, tmp_path / "release", max_bytes=40_960)


def test_tar_bytes_are_deterministic_for_the_same_catalog(tmp_path: Path) -> None:
    catalog = _make_catalog(tmp_path, count=3, layout="mixed")

    left = pack_catalog(catalog, tmp_path / "left", max_bytes=40_960)
    right = pack_catalog(catalog, tmp_path / "right", max_bytes=40_960)

    assert [item["tar_sha256"] for item in left["shards"]] == [
        item["tar_sha256"] for item in right["shards"]
    ]


def test_staging_checks_available_disk_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _make_catalog(tmp_path, count=1, layout="directory")
    release_root = tmp_path / "release"
    pack_catalog(catalog, release_root, max_bytes=40_960)
    monkeypatch.setattr("histology_data.shards.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=0))

    with pytest.raises(OSError, match="Insufficient free space"):
        stage_shard(release_root, "shard-000001", tmp_path / "no-space")
