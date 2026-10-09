"""Exercise the exact source-selection function embedded in the Colab notebook."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_MARKER = "# SOURCE_SELECTION_ADAPTER_BEGIN"
END_MARKER = "# SOURCE_SELECTION_ADAPTER_END"
METADATA_COLUMNS = ["Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade", "Ket_Luan", "Ten_Slide"]


@pytest.fixture
def mixed_sources(tmp_path: Path) -> dict[str, Any]:
    source_directory = tmp_path / "4x-directory"
    source_directory.mkdir()
    zip_directory = tmp_path / "source-zips"
    zip_directory.mkdir()
    metadata = tmp_path / "Metadata.csv"
    rows = [
        ["field4.png", "2020", "1", "4", "G2", "Carcinoma", "slide-4"],
        ["field10.png", "2020", "2", "10", "G1", "Hyperplasia", "slide-10"],
        ["field40.png", "2020", "3", "40", "G3", "Carcinoma", "slide-40"],
    ]
    Image.new("RGB", (64, 64), (150, 60, 120)).save(source_directory / rows[0][0])
    for index, lens in enumerate((10, 40), start=1):
        image_path = tmp_path / f"field{lens}.png"
        Image.new("RGB", (64, 64), (150 + index, 60, 120)).save(image_path)
        with zipfile.ZipFile(zip_directory / f"{lens}x.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(image_path, rows[index][0])
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(METADATA_COLUMNS)
        writer.writerows(rows)
    return {
        "metadata": metadata,
        "source_directory": source_directory,
        "zip_directory": zip_directory,
        "rows": rows,
    }


def _notebook_selector() -> Any:
    notebook_path = PROJECT_ROOT / "notebooks" / "Colab_Data_Shards.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    adapter_source = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if START_MARKER in source and END_MARKER in source:
            start = source.index(START_MARKER) + len(START_MARKER)
            end = source.index(END_MARKER, start)
            adapter_source = source[start:end]
            break
    assert adapter_source is not None, "Notebook source adapter markers are missing."
    namespace: dict[str, Any] = {"Path": Path}
    exec(compile(adapter_source, str(notebook_path), "exec"), namespace)
    return namespace["select_image_sources"]


def _run_prepare(metadata: Path, sources: list[Path], release: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "histology_data",
        "prepare",
        "--metadata",
        str(metadata),
        "--release",
        str(release),
        "--lenses",
        "4",
        "10",
        "40",
        "--per-lens",
        "1",
        "--max-mib",
        "1",
    ]
    for source in sources:
        command.extend(["--source", str(source)])
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def test_notebook_source_adapter_accepts_4x_directory_and_10x_40x_archives(
    tmp_path: Path,
    mixed_sources: dict[str, Any],
) -> None:
    select_sources = _notebook_selector()
    sources = select_sources(mixed_sources["source_directory"], mixed_sources["zip_directory"])
    expected_archives = sorted(mixed_sources["zip_directory"].glob("*.zip"))
    assert sources == [mixed_sources["source_directory"], *expected_archives]

    release = tmp_path / "mixed-release"
    prepared = _run_prepare(mixed_sources["metadata"], sources, release)
    assert prepared.returncode == 0, prepared.stderr
    descriptor = json.loads((release / "release.json").read_text(encoding="utf-8"))
    assert descriptor["complete"] is True
    assert descriptor["catalog"]["mode"] == "smoke"
    assert len(descriptor["catalog"]["images"]) == 3


def test_notebook_passes_duplicate_basenames_to_catalog_for_rejection(
    tmp_path: Path,
    mixed_sources: dict[str, Any],
) -> None:
    duplicate = mixed_sources["zip_directory"] / "duplicate-4x.zip"
    duplicate_image = tmp_path / "duplicate.png"
    Image.new("RGB", (64, 64), (153, 60, 120)).save(duplicate_image)
    with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(duplicate_image, "field4.png")

    sources = _notebook_selector()(
        mixed_sources["source_directory"],
        mixed_sources["zip_directory"],
    )
    failed = _run_prepare(mixed_sources["metadata"], sources, tmp_path / "duplicate-release")
    assert failed.returncode != 0
    assert "Image present in multiple sources: field4.png" in failed.stderr
    assert "Traceback" not in failed.stderr
