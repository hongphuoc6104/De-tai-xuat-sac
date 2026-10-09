"""Real filesystem regressions for smoke sampling and clinical metadata gates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from histology_data import catalog


def make_source(root: Path) -> tuple[Path, Path]:
    """Create six real fields with both weak case labels in a known sort order."""
    source = root / "images"
    source.mkdir()
    rows = []
    for i in range(6):
        name = f"img_{i}.tiff"
        Image.new("RGB", (16, 16), "purple").save(source / name)
        rows.append(dict(Ten_File=name, Ma_Nam="YCT 26", Ma_So=str(i + 1), Do_Phong_Dai="4X",
                         Glade="0" if i < 3 else "4", Ten_Slide="Slide1-2",
                         Ket_Luan="TĂNG SẢN LÀNH TÍNH" if i < 3 else "CARCINÔM TUYẾN TIỀN LIỆT"))
    metadata = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return source, metadata


def test_smoke_alternates_available_case_labels(tmp_path: Path) -> None:
    """Reproduce the former two-benign-field subset through the public catalog API."""
    source, metadata = make_source(tmp_path)
    result = catalog.build_catalog(metadata, [source], lenses=[4], per_lens=2)
    assert {image["case_label"] for image in result["images"]} == {0, 1}
    assert result["mode"] == "smoke" and result["training_ready"] is False


def test_smoke_hashes_only_selected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read real hashes while recording which raw image files were opened."""
    source, metadata = make_source(tmp_path)
    calls: list[Path] = []
    original = catalog.file_hash

    def recorded_hash(path: Path) -> str:
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(catalog, "file_hash", recorded_hash)
    result = catalog.build_catalog(metadata, [source], lenses=[4], per_lens=2)
    assert len([path for path in calls if path.suffix == ".tiff"]) == 2
    assert all(len(image["source_signature"]) == 64 for image in result["images"])


def test_training_requires_confirmed_identity_and_label_review(tmp_path: Path) -> None:
    """Preserve raw grades and require explicit patient identity at training boundary."""
    source, metadata = make_source(tmp_path)
    full = catalog.build_catalog(metadata, [source], lenses=[4])
    assert not full["training_ready"]
    assert {image["grade_semantics"] for image in full["images"]} == {"unconfirmed"}
    with pytest.raises(ValueError, match="verified patient mapping"):
        catalog.require_training_ready(full)
    identities = tmp_path / "patients.csv"
    pd.DataFrame({"case_id": [f"YCT26_{i}" for i in range(1, 7)],
                  "patient_id": [f"person-{i}" for i in range(1, 7)]}).to_csv(identities, index=False)
    reviewed = catalog.build_catalog(metadata, [source], lenses=[4], identity_map=identities, labels_reviewed=True)
    catalog.require_training_ready(reviewed)
    assert reviewed["training_ready"]
    smoke = catalog.build_catalog(metadata, [source], lenses=[4], per_lens=2, identity_map=identities, labels_reviewed=True)
    assert not smoke["training_ready"]


def test_missing_images_and_conflicting_case_labels_are_exposed(tmp_path: Path) -> None:
    """Full metadata coverage and consistent weak case labels remain explicit."""
    source, metadata = make_source(tmp_path)
    (source / "img_5.tiff").unlink()
    result = catalog.build_catalog(metadata, [source], lenses=[4])
    assert result["missing_images"] == ["img_5.tiff"]
    table = pd.read_csv(metadata, dtype=str)
    table.loc[3, "Ma_So"] = "1"
    table.to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="Conflicting case-level"):
        catalog.build_catalog(metadata, [source], lenses=[4])
    assert not result["training_ready"]


@pytest.mark.parametrize("member", ["../bad.tiff", "/bad.tiff", "x//bad.tiff", "C:\\bad.tiff"])
def test_unsafe_source_members_are_rejected(member: str) -> None:
    """Archive source paths cannot escape a source or collide via normalization."""
    with pytest.raises(ValueError):
        catalog.safe_member(member)
    assert member != "images/good.tiff"
