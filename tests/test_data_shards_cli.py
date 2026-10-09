"""End-to-end CLI coverage with tiny local image sources."""
from __future__ import annotations

import csv
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_COLUMNS = ["Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade", "Ket_Luan", "Ten_Slide"]


@pytest.fixture
def synthetic_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create three colored synthetic fields and equivalent directory/ZIP sources."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = [
        ["field4.png", "2020", "1", "4", "G2", "Carcinoma", "slide-4"],
        ["field10.png", "2020", "2", "10", "G1", "Hyperplasia", "slide-10"],
        ["field40.png", "2020", "3", "40", "G3", "Carcinoma", "slide-40"],
    ]
    for index, row in enumerate(rows):
        Image.new("RGB", (64, 64), (150 + index, 60, 120)).save(image_root / row[0])
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(METADATA_COLUMNS)
        writer.writerows(rows)
    source_zip = tmp_path / "images.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image in sorted(image_root.iterdir()):
            archive.write(image, image.name)
    return metadata, image_root, source_zip


def _run_cli(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "histology_data", *map(str, arguments)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("source_kind", ["directory", "zip"])
def test_prepare_and_verify_directory_or_zip_source(
    tmp_path: Path,
    synthetic_sources: tuple[Path, Path, Path],
    source_kind: str,
) -> None:
    metadata, image_root, source_zip = synthetic_sources
    source = image_root if source_kind == "directory" else source_zip
    release = tmp_path / f"release-{source_kind}"
    prepared = _run_cli(
        "prepare",
        "--metadata",
        metadata,
        "--source",
        source,
        "--release",
        release,
        "--lenses",
        "4",
        "10",
        "40",
        "--per-lens",
        "1",
        "--max-mib",
        "1",
    )
    assert prepared.returncode == 0, prepared.stderr
    assert "Prepared catalog" in prepared.stderr

    verified = _run_cli("verify", "--release", release)
    assert verified.returncode == 0, verified.stderr
    assert "Verified release" in verified.stderr


def test_process_resumes_verified_shard_and_advances_by_new_work(
    tmp_path: Path,
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    metadata, image_root, _source_zip = synthetic_sources
    release = tmp_path / "release"
    prepared = _run_cli(
        "prepare",
        "--metadata",
        metadata,
        "--source",
        image_root,
        "--release",
        release,
        "--lenses",
        "4",
        "10",
        "40",
        "--per-lens",
        "1",
        "--max-mib",
        "1",
    )
    assert prepared.returncode == 0, prepared.stderr

    work_root = tmp_path / "ssd-work"
    output_root = tmp_path / "processed"
    process_args = (
        "process",
        "--release",
        release,
        "--work-root",
        work_root,
        "--output",
        output_root,
        "--tile-size",
        "32",
        "--stride",
        "32",
        "--min-tissue",
        "0",
        "--max-tiles-per-image",
        "2",
        "--max-shards",
        "1",
    )
    first = _run_cli(*process_args)
    assert first.returncode == 0, first.stderr
    assert "1 new" in first.stderr

    resumed = _run_cli(*process_args)
    assert resumed.returncode == 0, resumed.stderr
    assert "0 new" in resumed.stderr
    assert "verified existing" in resumed.stderr


def test_cli_reports_bad_metadata_without_traceback(tmp_path: Path, synthetic_sources: tuple[Path, Path, Path]) -> None:
    _metadata, image_root, _source_zip = synthetic_sources
    invalid_metadata = tmp_path / "bad.csv"
    invalid_metadata.write_text("wrong,columns\na,b\n", encoding="utf-8")
    failed = _run_cli(
        "prepare",
        "--metadata",
        invalid_metadata,
        "--source",
        image_root,
        "--release",
        tmp_path / "bad-release",
        "--per-lens",
        "1",
    )
    assert failed.returncode != 0
    assert "missing fields" in failed.stderr.lower()
    assert "Traceback" not in failed.stderr
