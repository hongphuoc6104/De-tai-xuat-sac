"""Verified relative-attention artifact tests using governed synthetic bags."""
from __future__ import annotations

import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from histology_data.mil_data import BundleDataset
from histology_data.mil_model import MILModel, ModelConfig
from histology_data.mil_outputs import publish_case_attention, verify_attention_outputs
from histology_data.readiness import verify_pretrain_bundle


@pytest.fixture(scope="module")
def governed_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mil-attention-output")
    helper_path = Path(__file__).resolve().parents[1] / "tests" / "test_pretrain.py"
    helpers = runpy.run_path(str(helper_path))
    fixture = helpers["make_feature_fixture"](root / "source", patients=6)
    reviewed = helpers["make_reviewed_governance"](fixture, root / "governance")
    build_bundle = helpers["build_pretrain_bundle"]
    bundle_result = build_bundle(
        Path(reviewed["governance"]["path"]),
        fixture["feature_root"],
        root / "bundles",
        cohort_mode="common",
        expected_sources=1,
        expected_vectors=fixture["png_count"],
    )
    bundle = Path(bundle_result["bundle_path"])
    verified = verify_pretrain_bundle(bundle, fixture["feature_root"])
    assert verified["training_ready"] is True
    return {"bundle": bundle, "feature_root": fixture["feature_root"]}


def _publish_real_case(bundle_info: dict[str, Any], target: Path) -> tuple[str, dict[str, Any]]:
    bundle = bundle_info["bundle"]
    feature_root = bundle_info["feature_root"]
    splits = json.loads((bundle / "splits.json").read_text(encoding="utf-8"))
    case_id = splits["case_ids"][0]
    dataset = BundleDataset(bundle, feature_root, [case_id])
    case = dataset.load_case(case_id)
    features = {
        lens: torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        for lens, values in case.features.items()
    }
    model = MILModel(ModelConfig(variant="E4")).eval()
    with torch.no_grad():
        output = model(features, presence_mask=case.presence_mask)
    publish_case_attention(bundle, dataset, case, output, target)
    return case_id, output


def _source_references(bundle: Path, case_id: str, lens: str) -> list[dict[str, Any]]:
    bags = [
        json.loads(line)
        for line in (bundle / "bags.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["case_id"] == case_id and str(json.loads(line)["lens"]) == lens
    ]
    assert len(bags) == 1
    bag = bags[0]
    with (bundle / "instance_refs.jsonl").open("rb") as stream:
        stream.seek(bag["ref_start_byte"])
        payload = stream.read(bag["ref_end_byte"] - bag["ref_start_byte"])
    return [json.loads(line) for line in payload.splitlines()]


def test_attention_publication_preserves_verified_order_and_native_artifacts(
    governed_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "attention"
    case_id, model_output = _publish_real_case(governed_bundle, output_root)
    directory = output_root / hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    receipt_path = directory / "commit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["case_id"] == case_id
    assert receipt["kind"] == "relative_attention_with_source_coordinate_maps"
    assert receipt["pixel_ground_truth"] is False
    assert receipt["registration_between_lenses"] is False
    assert set(receipt["files"]) == {
        *(f"attention_{lens}.npy" for lens in ("4", "10", "40")),
        *(f"references_{lens}.json" for lens in ("4", "10", "40")),
        *(f"coordinates_{lens}.svg" for lens in ("4", "10", "40")),
    }

    bundle = governed_bundle["bundle"]
    for lens in ("4", "10", "40"):
        weights = np.load(directory / f"attention_{lens}.npy", allow_pickle=False)
        assert weights.ndim == 1 and weights.size == 1
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert float(weights.sum(dtype=np.float64)) == pytest.approx(1.0)
        np.testing.assert_allclose(weights, model_output["attention"][lens].detach().cpu().numpy())

        saved_refs = json.loads((directory / f"references_{lens}.json").read_text(encoding="utf-8"))
        assert saved_refs == _source_references(bundle, case_id, lens)
        assert all("target" not in ref and "case_label" not in ref for ref in saved_refs)
        svg = (directory / f"coordinates_{lens}.svg").read_text(encoding="utf-8")
        root = ET.fromstring(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert len(root.findall(".//{http://www.w3.org/2000/svg}circle")) == 1
        assert "relative attention, not cancer probability" in svg
        assert receipt["files"][f"attention_{lens}.npy"] == hashlib.sha256(
            (directory / f"attention_{lens}.npy").read_bytes(),
        ).hexdigest()

    verified = verify_attention_outputs(output_root, [case_id])
    assert verified == {case_id: hashlib.sha256(receipt_path.read_bytes()).hexdigest()}


def test_attention_verification_detects_weight_file_tampering(
    governed_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "tampered-attention"
    case_id, _ = _publish_real_case(governed_bundle, output_root)
    directory = output_root / hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    weights_path = directory / "attention_4.npy"
    weights_path.write_bytes(weights_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        verify_attention_outputs(output_root, [case_id])


def _reference_bundle(root: Path, refs: list[dict[str, Any]]) -> Path:
    root.mkdir(parents=True)
    payload = b"".join(
        (json.dumps(ref, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for ref in refs
    )
    (root / "instance_refs.jsonl").write_bytes(payload)
    bag = {
        "case_id": "synthetic-case",
        "lens": "4",
        "ref_start_byte": 0,
        "ref_end_byte": len(payload),
    }
    (root / "bags.jsonl").write_text(json.dumps(bag) + "\n", encoding="utf-8")
    return root


def _synthetic_references(image_id: str = "source-slide") -> list[dict[str, Any]]:
    return [
        {"tile_id": "tile-a", "image_id": image_id, "x": 0, "y": 0},
        {"tile_id": "tile-b", "image_id": image_id, "x": 4, "y": 8},
    ]


def test_publisher_rejects_attention_in_wrong_verified_tile_order(tmp_path: Path) -> None:
    refs = _synthetic_references()
    bundle = _reference_bundle(tmp_path / "bundle", refs)
    dataset = SimpleNamespace(loaded_tile_order=lambda _case_id: {"4": ["tile-b", "tile-a"]})
    case = SimpleNamespace(case_id="synthetic-case", patient_id="synthetic-patient")
    output_root = tmp_path / "wrong-order"

    with pytest.raises(ValueError, match="references differ from loaded source tile order"):
        publish_case_attention(
            bundle,
            dataset,
            case,
            {"attention": {"4": torch.tensor([0.25, 0.75])}},
            output_root,
        )

    directory = output_root / hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
    assert not (directory / "commit.json").exists()


def test_publisher_rejects_negative_attention_even_when_sum_is_one(tmp_path: Path) -> None:
    refs = _synthetic_references()
    bundle = _reference_bundle(tmp_path / "bundle", refs)
    dataset = SimpleNamespace(loaded_tile_order=lambda _case_id: {"4": ["tile-a", "tile-b"]})
    case = SimpleNamespace(case_id="synthetic-case", patient_id="synthetic-patient")
    output_root = tmp_path / "negative-attention"

    with pytest.raises(ValueError, match="nonnegative"):
        publish_case_attention(
            bundle,
            dataset,
            case,
            {"attention": {"4": torch.tensor([-0.25, 1.25])}},
            output_root,
        )

    directory = output_root / hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
    assert not (directory / "commit.json").exists()


def test_coordinate_svg_escapes_synthetic_source_image_id(tmp_path: Path) -> None:
    image_id = "<script>alert('tile')</script>"
    refs = _synthetic_references(image_id)
    bundle = _reference_bundle(tmp_path / "bundle", refs)
    dataset = SimpleNamespace(loaded_tile_order=lambda _case_id: {"4": ["tile-a", "tile-b"]})
    case = SimpleNamespace(case_id="synthetic-case", patient_id="synthetic-patient")
    output_root = tmp_path / "escaped-svg"

    publish_case_attention(
        bundle,
        dataset,
        case,
        {"attention": {"4": torch.tensor([0.25, 0.75])}},
        output_root,
    )
    directory = output_root / hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
    svg = (directory / "coordinates_4.svg").read_text(encoding="utf-8")

    assert "&lt;script&gt;alert(&#x27;tile&#x27;)&lt;/script&gt;" in svg
    assert "<script>" not in svg
    assert verify_attention_outputs(output_root, [case.case_id])[case.case_id]
