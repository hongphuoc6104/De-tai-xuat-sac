"""Small original-photo fixtures for image-level classifier tests."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

LENSES = (4, 10, 40)
METADATA_COLUMNS = (
    "Id", "Ten_File", "Data_Provider", "Slide_Mot_A", "Ma_So", "Ma_Nam",
    "Ket_Luan", "Ten_Slide", "Do_Phong_Dai", "Glade",
)


@dataclass(frozen=True)
class OriginalImageFixture:
    metadata: Path
    archives: dict[int, Path]
    image_paths: dict[str, Path]
    rows: tuple[dict[str, str], ...]


def _photo_bytes(path: Path, *, target: int, patient_index: int, lens_index: int) -> None:
    """Write a non-square original photo with class signal and edge landmarks."""
    if target == 0:
        color = (35 + patient_index * 3, 145 + lens_index * 4, 65 + patient_index)
    else:
        color = (170 + patient_index * 3, 45 + lens_index * 5, 60 + patient_index * 2)
    image = Image.new("RGB", (90, 30), color)
    draw = ImageDraw.Draw(image)
    # Distinctive bands at both ends expose accidental center cropping.
    draw.rectangle((0, 0, 9, 29), fill=(255, 0, 255))
    draw.rectangle((80, 0, 89, 29), fill=(0, 255, 255))
    # A second, interior tissue marker makes resize interpolation observable.
    draw.rectangle((39, 8, 50, 21), fill=(250, 220, 10))
    image.save(path, format="TIFF")


def make_original_image_fixture(root: Path, *, patients: int = 12) -> OriginalImageFixture:
    """Create three original-image ZIPs and the exact workbook schema."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "originals"
    raw_root.mkdir()
    archive_paths = {lens: root / f"originals_{lens}x.zip" for lens in LENSES}
    archives = {lens: zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED)
                for lens, path in archive_paths.items()}
    rows: list[dict[str, str]] = []
    image_paths: dict[str, Path] = {}
    try:
        for patient_index in range(patients):
            patient_number = f"SYN{patient_index:03d}"
            for lens_index, lens in enumerate(LENSES):
                grade_slot = (patient_index + lens_index) % 3
                if grade_slot == 0:
                    grade = 0
                else:
                    grade = (3, 4, 5)[(patient_index + 2 * lens_index) % 3]
                target = int(grade != 0)
                filename = f"ORIGINAL_{patient_index:03d}_{lens}x.tif"
                image_path = raw_root / filename
                _photo_bytes(image_path, target=target, patient_index=patient_index,
                             lens_index=lens_index)
                image_paths[filename] = image_path
                with image_path.open("rb") as stream:
                    payload = stream.read()
                archives[lens].writestr(f"source/lens-{lens}/{filename}", payload)
                rows.append({
                    "Id": str(len(rows) + 1),
                    "Ten_File": filename,
                    "Data_Provider": "synthetic-provider",
                    "Slide_Mot_A": f"synthetic-major-{patient_index:03d}",
                    "Ma_So": patient_number,
                    "Ma_Nam": "SYNTH-2026",
                    # Deliberately contradictory: only the doctor's image grade is target.
                    "Ket_Luan": "CARCINOMA" if target == 0 else "BENIGN HYPERPLASIA",
                    "Ten_Slide": f"synthetic-slide-{patient_index:03d}",
                    "Do_Phong_Dai": f"{lens}X",
                    "Glade": str(grade),
                })
    finally:
        for archive in archives.values():
            archive.close()

    metadata = root / "Metadata.xlsx"
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_excel(metadata, index=False)
    return OriginalImageFixture(metadata=metadata, archives=archive_paths,
                                image_paths=image_paths, rows=tuple(rows))


def rewrite_metadata(path: Path, rows: list[dict[str, str]]) -> Path:
    """Replace workbook rows while retaining the declared source schema."""
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_excel(path, index=False)
    return path
