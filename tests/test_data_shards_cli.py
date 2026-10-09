"""End-to-end CLI coverage with tiny local image sources."""
from __future__ import annotations

import csv
import json
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
    descriptor = json.loads((release / "release.json").read_text(encoding="utf-8"))
    shard_id = descriptor["shards"][0]["shard_id"]
    stage_cache = work_root / "staged" / descriptor["catalog_id"] / shard_id
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
    assert not stage_cache.exists()

    resumed = _run_cli(*process_args)
    assert resumed.returncode == 0, resumed.stderr
    assert "0 new" in resumed.stderr
    assert "verified existing" in resumed.stderr
    assert not stage_cache.exists()

    keep_args = list(process_args)
    tile_size_index = keep_args.index("--tile-size") + 1
    keep_args[tile_size_index] = "16"
    keep_args.append("--keep-work")
    kept = _run_cli(*keep_args)
    assert kept.returncode == 0, kept.stderr
    assert "1 new" in kept.stderr
    assert "Retained staged source shard" in kept.stderr
    assert stage_cache.is_dir()

    output_directories = [path for path in (output_root / shard_id).iterdir() if path.is_dir()]
    assert len(output_directories) == 2
    kept_output = next(path for path in output_directories if str(path) in kept.stderr)
    # Runtime-only retention must not change the output's processing fingerprint.
    cleanup_args = [item for item in keep_args if item != "--keep-work"]
    cleaned_reuse = _run_cli(*cleanup_args)
    assert cleaned_reuse.returncode == 0, cleaned_reuse.stderr
    assert "0 new" in cleaned_reuse.stderr
    assert "Verified existing processed shard" in cleaned_reuse.stderr
    assert str(kept_output) in cleaned_reuse.stderr
    assert not stage_cache.exists()


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


def test_prepare_rejects_non_boolean_labels_reviewed_config(
    tmp_path: Path,
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    metadata, image_root, _source_zip = synthetic_sources
    config = tmp_path / "invalid-config.json"
    config.write_text('{"prepare":{"per_lens":1,"labels_reviewed":"false"}}', encoding="utf-8")
    failed = _run_cli(
        "prepare",
        "--metadata",
        metadata,
        "--source",
        image_root,
        "--release",
        tmp_path / "invalid-config-release",
        "--config",
        config,
    )
    assert failed.returncode != 0
    assert "prepare.labels_reviewed must be a JSON boolean" in failed.stderr
    assert "Traceback" not in failed.stderr


def test_process_exposes_tissue_thresholds_and_rejects_unknown_keys(
    tmp_path: Path,
    synthetic_sources: tuple[Path, Path, Path],
) -> None:
    """A configured stain threshold changes real emitted tiles through the CLI."""
    metadata, source, _archive = synthetic_sources
    release = tmp_path / "raw"
    prepared = _run_cli("prepare", "--metadata", metadata, "--source", source,
                        "--release", release, "--lenses", "4", "--per-lens", "1", "--max-mib", "1")
    assert prepared.returncode == 0, prepared.stderr
    config = tmp_path / "thresholds.json"
    config.write_text(json.dumps({"process": {"tissue_green_contrast_threshold": 5.0}}))
    output = tmp_path / "processed"
    processed = _run_cli("process", "--release", release, "--work-root", tmp_path / "work",
                         "--output", output, "--config", config)
    assert processed.returncode == 0, processed.stderr
    commit = json.loads(next(output.glob("shard-*/*/commit.json")).read_text())
    assert commit["config"]["tissue_green_contrast_threshold"] == 5.0
    assert commit["tiles_written"] == 0 and commit["review_required"]
    config.write_text(json.dumps({"process": {"tissue_threshold_typo": 0.2}}))
    invalid = _run_cli("process", "--release", release, "--work-root", tmp_path / "work",
                       "--output", output, "--config", config)
    assert invalid.returncode != 0
    assert "Unknown process configuration" in invalid.stderr
