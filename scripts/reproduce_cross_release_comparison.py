"""Reproduce the explicit cross-release comparison gap with synthetic originals."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from histology_data import (
    image_classification as core,  # noqa: E402
    image_classification_studies as studies,  # noqa: E402
)
from tests.test_image_classification import TinyPixelEncoder  # noqa: E402
from tests.test_image_classification_studies import source  # noqa: E402

with tempfile.TemporaryDirectory(prefix="image-comparison-e2e-") as folder:
    root = Path(folder)
    original, baseline = source.__wrapped__(root)
    new = core.extract_original_image_features(
        Path(original.release["inventory_path"]), root / "variant", root / "weights",
        encoder=TinyPixelEncoder(), config=core.FeatureExtractionConfig(
            max_long_edge=64, device="cpu", budget_minutes=5, reserve_minutes=0))
    try:
        studies.load_study_data(new, baseline)
    except ValueError as exc:
        assert "Baseline does not belong" in str(exc)
        print(json.dumps({"reproduced": True, "context": "same synthetic original sources, geometry48→64",
                          "guard": str(exc), "baseline_preserved": True}))
    else:
        raise AssertionError("The old same-release loader unexpectedly accepted another release.")
