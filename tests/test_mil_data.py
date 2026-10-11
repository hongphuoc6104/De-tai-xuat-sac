"""Contracts for bounded synthetic and governed MIL feature datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histology_data.io import fingerprint
from histology_data.mil_data import (
    BundleDataset,
    CaseData,
    FixtureDataset,
    create_toy_fixture,
    export_training_fixture,
)
from histology_data.readiness import build_pretrain_bundle


def test_toy_fixture_is_deterministic_learnable_and_smoke_only(tmp_path: Path) -> None:
    first_path = create_toy_fixture(tmp_path / "first", seed=23, case_count=4, rows_per_lens=5)
    second_path = create_toy_fixture(tmp_path / "second", seed=23, case_count=4, rows_per_lens=5)
    first = FixtureDataset(first_path)
    second = FixtureDataset(second_path)

    assert first.case_ids == second.case_ids
    assert first.identity["fingerprint"] == second.identity["fingerprint"]
    assert first.identity["kind"] == "smoke_fixture"
    assert first.mode == "smoke"
    assert first.records["toy-case-0000"] == {"patient_id": "toy-patient-0000", "target": 0}
    zero = first.load_case("toy-case-0000")
    one = first.load_case("toy-case-0001")
    assert zero.presence_mask == (True, True, True)
    assert set(zero.features) == {"4", "10", "40"}
    assert all(array.shape == (5, 2048) and array.dtype == np.float32 for array in zero.features.values())
    assert all(not array.flags.writeable for array in zero.features.values())
    assert float(zero.features["4"][:, :32].mean()) < -1.0
    assert float(one.features["4"][:, :32].mean()) > 1.0
    with pytest.raises(ValueError, match="at most"):
        create_toy_fixture(tmp_path / "too-large", case_count=17, rows_per_lens=1)

    manifest = json.loads((first_path / "fixture.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "smoke"
    assert manifest["training_ready"] is False
    assert manifest["production_eligible"] is False
    assert all(entry["sha256"] for case in manifest["cases"] for entry in case["arrays"].values())


def test_fixture_loader_rejects_array_tampering_and_path_traversal(tmp_path: Path) -> None:
    fixture_path = create_toy_fixture(tmp_path / "fixture", case_count=2, rows_per_lens=2)
    fixture = FixtureDataset(fixture_path)
    manifest = json.loads((fixture_path / "fixture.json").read_text(encoding="utf-8"))
    array_path = fixture_path / manifest["cases"][0]["arrays"]["4"]["path"]
    array_path.write_bytes(array_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fixture.load_case(fixture.case_ids[0])

    unsafe_path = tmp_path / "unsafe"
    unsafe_path.mkdir()
    # JSON output sorts lens keys, so exercise the first row encountered during
    # constructor validation before it checks for the other missing arrays.
    manifest["cases"][0]["arrays"]["10"]["path"] = "../outside.npy"
    unsigned = {key: value for key, value in manifest.items() if key != "fixture_id"}
    manifest["fixture_id"] = fingerprint(unsigned)
    (unsafe_path / "fixture.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe path"):
        FixtureDataset(unsafe_path)


def test_case_data_enforces_lens_shape_finiteness_and_presence() -> None:
    empty = np.empty((0, 2048), dtype=np.float32)
    valid = np.ones((1, 2048), dtype=np.float32)
    features = {"4": valid.copy(), "10": valid.copy(), "40": empty}
    case = CaseData("case-1", "patient-1", 1, features, (True, True, False))
    assert case.target == 1
    with pytest.raises(ValueError, match="presence_mask"):
        CaseData("case-1", "patient-1", 1, features, (True, True, True))
    bad = valid.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        CaseData("case-1", "patient-1", 1, {"4": bad, "10": valid.copy(), "40": empty},
                 (True, True, False))
    with pytest.raises(ValueError, match="reviewed integer"):
        CaseData("case-1", "patient-1", True, features, (True, True, False))


@pytest.fixture(scope="module")
def governed_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    # Reuse the repository's real synthetic extraction/governance pipeline; no
    # project images or external data are read by this bounded integration test.
    from test_pretrain import make_feature_fixture, make_reviewed_governance

    root = tmp_path_factory.mktemp("mil-governed")
    source = make_feature_fixture(root / "source", patients=12)
    reviewed = make_reviewed_governance(source, root / "governance")
    result = build_pretrain_bundle(
        Path(reviewed["governance"]["path"]),
        source["feature_root"],
        root / "bundle-output",
        expected_sources=1,
        expected_vectors=source["png_count"],
        seed=42,
    )
    assert result["status"] == "ready"
    return {"bundle": Path(result["bundle_path"]), "feature_root": source["feature_root"]}


def test_bundle_dataset_loads_only_requested_partition_and_exports_inner_train_smoke(
    governed_bundle: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = governed_bundle["bundle"]
    feature_root = governed_bundle["feature_root"]
    splits = json.loads((bundle / "splits.json").read_text(encoding="utf-8"))
    outer = next(row for row in splits["outer"] if row["fold"] == 0)
    inner = next(row for row in outer["inner_folds"] if row["fold"] == 0)
    candidates = inner["train_cases"]
    metadata = {}
    with (bundle / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        import csv

        metadata = {row["case_id"]: row for row in csv.DictReader(stream)}
    selected = []
    seen_classes = set()
    seen_patients = set()
    for case_id in candidates:
        row = metadata[case_id]
        label = int(row["case_label"])
        if label not in seen_classes and row["patient_id"] not in seen_patients:
            selected.append(case_id)
            seen_classes.add(label)
            seen_patients.add(row["patient_id"])
        if len(selected) == 2:
            break
    assert len(selected) == 2
    dataset = BundleDataset(bundle, feature_root, selected)
    assert dataset.case_ids == selected
    assert dataset.records[selected[0]]["patient_id"] == metadata[selected[0]]["patient_id"]
    assert dataset.identity["bundle_id"]
    with pytest.raises(ValueError, match="outside the governed bundle"):
        BundleDataset(bundle, feature_root, ["not-in-this-bundle"])
    bounded = BundleDataset(bundle, feature_root, selected, max_case_bytes=1)
    with pytest.raises(MemoryError, match="above max_case_bytes"):
        bounded.load_case(selected[0])
    loaded = dataset.load_case(selected[0])
    assert loaded.target == int(metadata[selected[0]]["case_label"])
    assert loaded.patient_id == metadata[selected[0]]["patient_id"]
    assert all(array.shape == (1, 2048) for array in loaded.features.values())

    output = export_training_fixture(
        bundle,
        feature_root,
        tmp_path / "exported-smoke",
        outer_fold=0,
        inner_fold=0,
        max_cases=4,
        max_rows=1,
    )
    exported = FixtureDataset(output)
    manifest = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    selected_ids = set(exported.case_ids)
    assert 2 <= len(selected_ids) <= 4
    assert selected_ids <= set(inner["train_cases"])
    assert selected_ids.isdisjoint(set(inner["validation_cases"]) | set(outer["test_cases"]))
    assert provenance["source_bundle_id"] == json.loads((bundle / "bundle.json").read_text())["bundle_id"]
    assert all(len(refs) == 1 for entry in provenance["selected_cases"]
               for refs in entry["tiles_by_lens"].values())
    assert all(not ({"target", "label", "patch_target", "patch_label"} & set(ref))
               for entry in provenance["selected_cases"]
               for refs in entry["tiles_by_lens"].values() for ref in refs)
    second_output = export_training_fixture(
        bundle,
        feature_root,
        tmp_path / "exported-smoke-repeat",
        outer_fold=0,
        inner_fold=0,
        max_cases=4,
        max_rows=1,
    )
    second_manifest = json.loads((second_output / "fixture.json").read_text(encoding="utf-8"))
    assert second_manifest["fixture_id"] == manifest["fixture_id"]
