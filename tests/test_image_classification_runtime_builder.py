"""Contract tests for the code-only runtime archive builder."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.build_image_classification_runtime import build_runtime


def _source_tree(root: Path) -> Path:
    root = Path(root)
    (root / "histology_data").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "histology_data" / "image_classification.py").write_text("# image runtime\n", encoding="utf-8")
    (root / "histology_data" / "image_classification_cli.py").write_text("# CLI\n", encoding="utf-8")
    (root / "histology_data" / "io.py").write_text("# shared runtime code\n", encoding="utf-8")
    (root / "scripts" / "run_image_classification.py").write_text("# entry point\n", encoding="utf-8")
    (root / "requirements-image-classification.txt").write_text(
        "numpy>=1.24\n-r requirements-features.txt\n", encoding="utf-8",
    )
    (root / "requirements-features.txt").write_text(
        "scikit-learn==1.8.0\n-r requirements-runtime.txt\n", encoding="utf-8",
    )
    (root / "requirements-runtime.txt").write_text("Pillow>=10.0\n", encoding="utf-8")
    (root / "Metadata.xlsx").write_bytes(b"synthetic sensitive workbook marker")
    (root / "data_4x.zip").write_bytes(b"synthetic image archive marker")
    (root / "credentials.json").write_text("SYNTHETIC_SECRET_SENTINEL", encoding="utf-8")
    results = root / "Results"
    results.mkdir()
    (results / "clinical.csv").write_text("synthetic,patient,rows\n", encoding="utf-8")
    return root


def test_runtime_builder_is_deterministic_and_excludes_data_and_credentials(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    first_path = tmp_path / "runtime-one.zip"
    second_path = tmp_path / "runtime-two.zip"

    first = build_runtime(source, first_path)
    second = build_runtime(source, second_path)

    assert first["sha256"] == second["sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["size_bytes"] == first_path.stat().st_size
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first["sha256"]
    assert first_path.with_suffix(".zip.sha256").read_text(encoding="ascii") == f"{first['sha256']}  {first_path.name}\n"
    with zipfile.ZipFile(first_path) as archive:
        names = archive.namelist()
        payloads = [archive.read(name) for name in names]
    assert names == sorted(names)
    assert "histology_data/image_classification.py" in names
    assert "histology_data/image_classification_cli.py" in names
    assert "scripts/run_image_classification.py" in names
    assert "requirements-image-classification.txt" in names
    assert "requirements-features.txt" in names
    assert "requirements-runtime.txt" in names
    assert not any(name.lower().endswith((".xlsx", ".xls", ".csv", ".zip", ".json")) for name in names)
    assert all(b"synthetic sensitive workbook marker" not in payload for payload in payloads)
    assert all(b"SYNTHETIC_SECRET_SENTINEL" not in payload for payload in payloads)
    assert all(b"synthetic image archive marker" not in payload for payload in payloads)
    assert "Metadata.xlsx" not in first["source_files"]
    assert "data_4x.zip" not in first["source_files"]
    assert "credentials.json" not in first["source_files"]


def test_runtime_builder_includes_the_actual_transitive_project_requirements_closure(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    output = tmp_path / "project-runtime.zip"
    receipt = build_runtime(source, output)
    expected_requirements = {
        "requirements-image-classification.txt",
        "requirements-features.txt",
        "requirements-data.txt",
    }

    assert expected_requirements.issubset(set(receipt["source_files"]))
    with zipfile.ZipFile(output) as archive:
        packaged_requirements = {name for name in archive.namelist() if name.startswith("requirements")}
        payloads = {name: archive.read(name) for name in packaged_requirements}
    assert packaged_requirements == expected_requirements
    assert all(payloads[name] == (source / name).read_bytes() for name in expected_requirements)
    assert output.exists() and receipt["sha256"]


def test_runtime_builder_fails_when_a_transitive_requirement_include_is_missing(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    (source / "requirements-runtime.txt").unlink()

    with pytest.raises(ValueError, match="Missing or unsafe runtime requirements"):
        build_runtime(source, tmp_path / "runtime.zip")
    assert not (tmp_path / "runtime.zip").exists()


def test_runtime_builder_rejects_requirement_include_traversal(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    (source / "requirements-runtime.txt").write_text(
        "-r ../requirements-outside.txt\n", encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe requirements include"):
        build_runtime(source, tmp_path / "runtime.zip")
    assert not (tmp_path / "runtime.zip").exists()


def test_runtime_builder_rejects_a_symlinked_source_module(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    outside = tmp_path / "outside.py"
    outside.write_text("# outside source\n", encoding="utf-8")
    module = source / "histology_data" / "io.py"
    module.unlink()
    module.symlink_to(outside)

    with pytest.raises(ValueError, match="Unsafe runtime source"):
        build_runtime(source, tmp_path / "runtime.zip")
    assert module.is_symlink()


def test_runtime_builder_rejects_a_symlinked_output_path(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    target = tmp_path / "target.zip"
    target.write_bytes(b"preserve target")
    output = tmp_path / "runtime.zip"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="regular .zip path"):
        build_runtime(source, output)
    assert target.read_bytes() == b"preserve target"
