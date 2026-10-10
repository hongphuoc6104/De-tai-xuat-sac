"""Synthetic end-to-end checks for reviewed identity, grouped splits and MIL data bundles."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from histology_data.bags import load_case_features
from histology_data.feature_encoder import FEATURE_DIM, OFFICIAL_WEIGHTS_SHA256, WEIGHTS_ENUM
from histology_data.features import BudgetExhausted, FeatureRunConfig, run_feature_extraction
from histology_data.governance import (
    CASE_REVIEW_FIELDS,
    GovernanceError,
    build_governance,
    create_case_review_draft,
    write_case_review_draft,
)
from histology_data.io import file_hash, fingerprint
from histology_data.pretrain_cli import main as pretrain_cli_main
from histology_data.readiness import (
    DUPLICATE_REVIEW_FIELDS,
    _load_duplicate_groups,
    _validate_duplicate_split_isolation,
    build_pretrain_bundle,
    inspect_feature_release,
    verify_pretrain_bundle,
)
from histology_data.splits import (
    SplitError,
    create_nested_patient_splits,
)

V1_TRANSFORM = {
    "input_color": "RGB",
    "transform": f"{WEIGHTS_ENUM}.transforms()",
    "resize_size": [256],
    "crop_size": [224],
    "crop": "center",
    "interpolation": "bilinear",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class SyntheticResNetEncoder:
    """Deterministic 2048-D fixture; production features still use the real encoder."""

    feature_dim = FEATURE_DIM
    descriptor = {
        "name": "resnet50",
        "weights": WEIGHTS_ENUM,
        "weights_sha256": OFFICIAL_WEIGHTS_SHA256,
        "preprocessing": V1_TRANSFORM,
        "torch_version": "synthetic-fixture",
        "torchvision_version": "synthetic-fixture",
        "pillow_version": "synthetic-fixture",
        "precision": "fp32",
        "implementation_version": "1.0.0",
    }

    def __init__(self, *, nonfinite: bool = False) -> None:
        self.nonfinite = nonfinite

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        result = np.empty((len(images), FEATURE_DIM), dtype=np.float32)
        for index, image in enumerate(images):
            pixel = image.convert("RGB").getpixel((0, 0))
            result[index] = np.float32(pixel[0] * 65536 + pixel[1] * 256 + pixel[2])
        if self.nonfinite and len(result):
            result[0, 0] = np.nan
        return result

    def release_memory(self) -> None:
        return None


def make_feature_fixture(
    root: Path,
    *,
    patients: int = 12,
    missing_lenses: set[tuple[str, int]] | None = None,
    duplicate_pair: tuple[tuple[str, int], tuple[str, int]] | None = None,
    smoke: bool = False,
    nonfinite: bool = False,
) -> dict[str, Any]:
    """Create tiny on-disk metadata/ZIP/features; no project images are read."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source_root = root / "archives"
    source_root.mkdir()
    metadata_rows = []
    case_labels: dict[str, int] = {}
    case_patients: dict[str, str] = {}
    png_members: list[tuple[str, bytes]] = []
    skipped = missing_lenses or set()
    duplicate_source: bytes | None = None

    for patient_index in range(patients):
        patient_id = f"patient-{patient_index:02d}"
        for class_index in (0, 1):
            numeric_case = patient_index * 2 + class_index + 1
            case_id = f"YCT26_{numeric_case}"
            case_label = class_index
            case_labels[case_id] = case_label
            case_patients[case_id] = patient_id
            conclusion = "BENIGN HYPERPLASIA WITH INFLAMMATION" if case_label == 0 else "CARCINOMA"
            for lens in (4, 10, 40):
                image_id = f"IMG_{numeric_case:02d}_{lens}X"
                metadata_rows.append({
                    "Ten_File": image_id + ".tif",
                    "Ma_Nam": "YCT26",
                    "Ma_So": str(numeric_case),
                    "Do_Phong_Dai": f"{lens}X",
                    "Glade": "raw-grade-value-" + str(class_index),
                    "Ket_Luan": conclusion,
                    "Ten_Slide": f"candidate-slide-{numeric_case}",
                })
                if (case_id, lens) in skipped:
                    continue
                color = ((numeric_case * 7 + lens) % 250 + 1,
                         (numeric_case * 13 + lens * 3) % 250 + 1,
                         (numeric_case * 19 + lens * 5) % 250 + 1)
                image_buffer = io.BytesIO()
                Image.new("RGB", (32, 32), color).save(image_buffer, format="PNG")
                raw = image_buffer.getvalue()
                if duplicate_pair and (case_id, lens) == duplicate_pair[0]:
                    duplicate_source = raw
                if duplicate_pair and (case_id, lens) == duplicate_pair[1] and duplicate_source is not None:
                    raw = duplicate_source
                member = f"Tiles/{image_id}/0_0.png"
                png_members.append((member, raw))

    metadata_path = root / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    zip_path = source_root / "Tiles-001.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, raw in png_members:
            archive.writestr(member, raw)

    feature_root = root / "feature-output"
    config = FeatureRunConfig(
        metadata=metadata_path,
        source_root=source_root,
        source_kind="zip",
        output_root=feature_root,
        work_root=root / "scratch",
        weights_dir=root / "weights",
        batch_size=8,
        device="cpu",
        precision="fp32",
        expected_archives=1,
        expected_pngs=len(png_members),
        budget_minutes=30,
        reserve_minutes=1,
        max_patches_per_part=1 if smoke else None,
    )
    feature_status = run_feature_extraction(config, encoder=SyntheticResNetEncoder(nonfinite=nonfinite))
    return {
        "root": root,
        "metadata": metadata_path,
        "source_root": source_root,
        "feature_root": feature_root,
        "feature_status": feature_status,
        "feature_config": config,
        "case_labels": case_labels,
        "case_patients": case_patients,
        "png_count": len(png_members),
    }


def make_reviewed_governance(fixture: dict[str, Any], output_root: Path) -> dict[str, Any]:
    draft = create_case_review_draft(
        fixture["metadata"], fixture["source_root"], expected_sources=1,
        expected_patches=fixture["png_count"],
    )
    draft_dir = write_case_review_draft(draft, output_root)
    review_path = draft_dir / "case_review.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CASE_REVIEW_FIELDS
    assert {row["case_id"] for row in rows} == set(fixture["case_labels"])
    for row in rows:
        row["patient_id"] = fixture["case_patients"][row["case_id"]]
        row["identity_verified"] = "true"
        row["identity_evidence"] = "synthetic reviewer fixture: patient assignment supplied per case"
        row["case_label"] = str(fixture["case_labels"][row["case_id"]])
        row["label_verified"] = "true"
        row["label_evidence"] = "synthetic reviewer fixture: explicit binary case-level target"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    governance = build_governance(fixture["metadata"], review_path, output_root)
    return {"governance": governance, "review_path": review_path, "draft": draft}


def _manifest_rehash(bundle_path: Path) -> None:
    manifest_path = bundle_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("bundle_id")
    for name in manifest["artifact_sha256"]:
        manifest["artifact_sha256"][name] = hashlib.sha256((bundle_path / name).read_bytes()).hexdigest()
    bundle_id = fingerprint(manifest)
    manifest["bundle_id"] = bundle_id
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def test_nested_split_e2e_reproduces_and_fixes_mixed_case_patient_groups() -> None:
    """12 patients × 2 opposite-label cases must fill every 3×2 grouped fold."""
    cases = [
        {"case_id": f"case-{patient:02d}-{label}", "patient_id": f"patient-{patient:02d}", "case_label": label}
        for patient in range(12)
        for label in (0, 1)
    ]
    splits = create_nested_patient_splits(cases, seed=42)
    assert [len(fold["test_patients"]) for fold in splits["outer"]] == [4, 4, 4]
    assert [len(fold["test_cases"]) for fold in splits["outer"]] == [8, 8, 8]
    for outer in splits["outer"]:
        assert len(outer["train_patients"]) == 8
        assert all(len(inner["validation_patients"]) == 4 for inner in outer["inner_folds"])
        assert all({0, 1} == {case["case_label"] for case in cases if case["case_id"] in side}
                   for side in (outer["train_cases"], outer["test_cases"]))
    assert create_nested_patient_splits(cases, seed=42) == splits


def test_full_feature_to_common_and_all_bundle_e2e_roundtrip(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "full")
    assert fixture["feature_status"]["status"] == "complete"
    assert fixture["feature_status"]["training_ready"] is False
    output_root = tmp_path / "pretrain"
    review = make_reviewed_governance(fixture, output_root)
    governance = review["governance"]
    assert governance["unique_patient_count"] == 12
    assert governance["case_count"] == 24
    assert governance["case_labels"] == {"0": 12, "1": 12}

    common = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        cohort_mode="common", expected_sources=1, expected_vectors=fixture["png_count"],
    )
    supplementary = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        cohort_mode="all", expected_sources=1, expected_vectors=fixture["png_count"],
    )
    assert common["training_ready"] is True and supplementary["training_ready"] is True
    assert common["bundle_id"] != supplementary["bundle_id"]
    assert common["governance_id"] == supplementary["governance_id"] == governance["governance_id"]
    assert common["feature_id"] == supplementary["feature_id"] == fixture["feature_status"]["feature_id"]
    assert common["case_count"] == supplementary["case_count"] == 24
    assert common["patient_count"] == supplementary["patient_count"] == 12
    assert common["instance_count"] == supplementary["instance_count"] == fixture["png_count"]
    common_manifest = json.loads((Path(common["bundle_path"]) / "bundle.json").read_text())
    assert common_manifest["feature_encoder_training_ready_flag_ignored"] is True

    verified = verify_pretrain_bundle(Path(common["bundle_path"]), fixture["feature_root"])
    assert verified["training_ready"] is True
    assert verified["case_count"] == 24 and verified["instance_count"] == fixture["png_count"]
    loaded = load_case_features(Path(common["bundle_path"]), "YCT26_1", fixture["feature_root"])
    assert loaded["case_id"] == "YCT26_1" and loaded["case_label"] == 0
    assert loaded["presence_mask"] == {4: True, 10: True, 40: True}
    assert all(loaded["features"][lens].shape == (1, FEATURE_DIM) for lens in (4, 10, 40))
    assert all(loaded["features"][lens].dtype == np.float32 for lens in (4, 10, 40))
    assert all(len(loaded["tile_order"][lens]) == 1 for lens in (4, 10, 40))
    for lens in (4, 10, 40):
        red = (7 + lens) % 250 + 1
        green = (13 + lens * 3) % 250 + 1
        blue = (19 + lens * 5) % 250 + 1
        assert loaded["features"][lens][0, 0] == red * 65536 + green * 256 + blue

    refs = Path(common["bundle_path"]) / "instance_refs.jsonl"
    assert all("target" not in json.loads(line) and "case_label" not in json.loads(line)
               for line in refs.read_text(encoding="utf-8").splitlines())
    with (Path(common["bundle_path"]) / "cases.csv").open(encoding="utf-8", newline="") as stream:
        case_rows = list(csv.DictReader(stream))
    row = next(item for item in case_rows if item["case_id"] == "YCT26_1")
    assert "raw-grade-value-0" in row["raw_glade_values_json"]
    assert "0" in row["candidate_label_codes_json"]


def test_pretrain_cli_draft_build_verify_e2e(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "cli", patients=6)
    output_root = tmp_path / "pretrain-cli"
    assert pretrain_cli_main([
        "draft", "--metadata", str(fixture["metadata"]), "--source-root", str(fixture["source_root"]),
        "--output-root", str(output_root), "--expected-sources", "1",
        "--expected-patches", str(fixture["png_count"]),
    ]) == 0
    draft_files = list((output_root / "governance" / "drafts").glob("*/case_review.csv"))
    assert len(draft_files) == 1
    review_path = draft_files[0]
    with review_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    for row in rows:
        row.update(patient_id=fixture["case_patients"][row["case_id"]], identity_verified="true",
                   identity_evidence="CLI E2E synthetic confirmation", case_label=str(fixture["case_labels"][row["case_id"]]),
                   label_verified="true", label_evidence="CLI E2E synthetic case-level label review")
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assert pretrain_cli_main([
        "build", "--metadata", str(fixture["metadata"]), "--case-review", str(review_path),
        "--feature-root", str(fixture["feature_root"]), "--output-root", str(output_root),
        "--cohort", "common", "--expected-sources", "1", "--expected-vectors", str(fixture["png_count"]),
        "--seed", "42",
    ]) == 0
    bundle_files = list((output_root / "bundles").glob("*/bundle.json"))
    assert len(bundle_files) == 1
    bundle_path = bundle_files[0].parent
    assert pretrain_cli_main([
        "verify", "--bundle", str(bundle_path), "--feature-root", str(fixture["feature_root"]),
    ]) == 0


def test_all_cohort_keeps_missing_lens_mask_and_common_records_exclusion(tmp_path: Path) -> None:
    missing_case = "YCT26_1"
    fixture = make_feature_fixture(tmp_path / "missing", missing_lenses={(missing_case, 40)})
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    common = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        cohort_mode="common", expected_sources=1, expected_vectors=fixture["png_count"],
    )
    all_cases = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        cohort_mode="all", expected_sources=1, expected_vectors=fixture["png_count"],
    )
    assert common["training_ready"] and all_cases["training_ready"]
    assert common["case_count"] == 23 and common["excluded_case_count"] == 1
    assert all_cases["case_count"] == 24
    loaded = load_case_features(Path(all_cases["bundle_path"]), missing_case, fixture["feature_root"])
    assert loaded["features"][40].shape == (0, FEATURE_DIM)
    assert loaded["presence_mask"][40] is False
    assert loaded["features"][4].shape == (1, FEATURE_DIM)


def test_partial_or_smoke_release_never_builds_a_bundle(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "smoke", smoke=True)
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    result = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        expected_sources=1, expected_vectors=fixture["png_count"],
    )
    assert result["status"] == "blocked" and result["training_ready"] is False
    assert "feature_run_not_complete" in result["blockers"]
    assert not list((output_root / "bundles").glob("*/bundle.json"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("patient_id", "", "patient_identity_not_reviewed"),
        ("identity_verified", "false", "patient_identity_not_reviewed"),
        ("identity_evidence", "", "patient_identity_evidence_missing"),
        ("case_label", "2", "case_label_not_reviewed_binary"),
        ("label_verified", "false", "case_label_not_reviewed_binary"),
        ("label_evidence", "", "case_label_evidence_missing"),
    ],
)
def test_missing_or_invalid_human_review_rows_block_governance(
    tmp_path: Path, field: str, value: str, message: str,
) -> None:
    fixture = make_feature_fixture(tmp_path / "review", patients=3)
    output_root = tmp_path / "pretrain"
    draft = create_case_review_draft(
        fixture["metadata"], fixture["source_root"], expected_sources=1,
        expected_patches=fixture["png_count"],
    )
    draft_dir = write_case_review_draft(draft, output_root)
    review_path = draft_dir / "case_review.csv"
    rows = list(csv.DictReader(review_path.open(encoding="utf-8", newline="")))
    for row in rows:
        row.update(patient_id=fixture["case_patients"][row["case_id"]], identity_verified="true",
                   identity_evidence="explicit synthetic reviewer evidence", case_label=str(fixture["case_labels"][row["case_id"]]),
                   label_verified="true", label_evidence="explicit synthetic case-label evidence")
    rows[0][field] = value
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(GovernanceError, match=message):
        build_governance(fixture["metadata"], review_path, output_root)


def test_duplicate_groups_require_disposition_and_union_patients_for_splits(tmp_path: Path) -> None:
    first_case, second_case = "YCT26_1", "YCT26_3"
    fixture = make_feature_fixture(
        tmp_path / "duplicates", duplicate_pair=((first_case, 4), (second_case, 4)),
    )
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    blocked = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        expected_sources=1, expected_vectors=fixture["png_count"],
    )
    assert blocked["training_ready"] is False
    assert "duplicate_dispositions.csv" in blocked["blockers"][0]

    feature = inspect_feature_release(fixture["feature_root"], expected_sources=1,
                                      expected_vectors=fixture["png_count"])
    groups = _load_duplicate_groups(feature)
    dispositions_path = tmp_path / "duplicate_dispositions.csv"
    with dispositions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DUPLICATE_REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for group in groups:
            members = sorted(member["tile_id"] for member in group["members"])
            writer.writerow({
                "kind": group["kind"], "sha256": group["sha256"],
                "keep_tile_ids_json": json.dumps(members), "exclude_tile_ids_json": "[]",
                "canonical_tile_id": members[0], "reviewer": "synthetic reviewer",
                "reviewed_at": "2026-10-10T00:00:00Z", "evidence": "synthetic exact-content review",
            })
    ready = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        duplicate_dispositions=dispositions_path, expected_sources=1,
        expected_vectors=fixture["png_count"],
    )
    assert ready["training_ready"] is True
    bundle_splits = json.loads((Path(ready["bundle_path"]) / "splits.json").read_text())
    assignment = bundle_splits["assignments"]
    assert assignment[first_case]["split_group_id"] == assignment[second_case]["split_group_id"]
    assert verify_pretrain_bundle(Path(ready["bundle_path"]), fixture["feature_root"])["training_ready"] is True

    no_union = create_nested_patient_splits(
        [{"case_id": case_id, "patient_id": fixture["case_patients"][case_id],
          "case_label": fixture["case_labels"][case_id]} for case_id in fixture["case_labels"]],
        seed=42,
    )
    outer_fold_for_case = {case_id: row["outer_test_fold"] for case_id, row in no_union["assignments"].items()}
    pair = next((left, right) for left in fixture["case_labels"] for right in fixture["case_labels"]
                if fixture["case_patients"][left] != fixture["case_patients"][right]
                and outer_fold_for_case[left] != outer_fold_for_case[right])
    fake_duplicate = [{"group_key": ("source_png_sha256", "a" * 64), "members": [
        {"tile_id": "tile-left", "candidate_case_id": pair[0]},
        {"tile_id": "tile-right", "candidate_case_id": pair[1]},
    ]}]
    with pytest.raises(SplitError, match="crosses outer"):
        _validate_duplicate_split_isolation(no_union, set(fixture["case_labels"]), fake_duplicate, set())


def test_nonfinite_encoder_and_feature_id_change_block_readiness(tmp_path: Path) -> None:
    bad = make_feature_fixture(tmp_path / "nan", patients=3, nonfinite=True)
    assert bad["feature_status"]["status"] == "error"
    governance = make_reviewed_governance(bad, tmp_path / "pretrain")["governance"]
    blocked = build_pretrain_bundle(
        Path(governance["path"]), bad["feature_root"], tmp_path / "pretrain",
        expected_sources=1, expected_vectors=bad["png_count"],
    )
    assert blocked["training_ready"] is False
    assert any("feature" in blocker for blocker in blocked["blockers"])

    good = make_feature_fixture(tmp_path / "good", patients=6)
    output_root = tmp_path / "good-pretrain"
    good_governance = make_reviewed_governance(good, output_root)["governance"]
    bundle = build_pretrain_bundle(
        Path(good_governance["path"]), good["feature_root"], output_root,
        expected_sources=1, expected_vectors=good["png_count"],
    )
    assert bundle["training_ready"]
    config_path = good["feature_root"] / good["feature_status"]["feature_id"] / "feature_config.json"
    config = json.loads(config_path.read_text())
    config["feature_dim"] = 7
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = inspect_feature_release(good["feature_root"], expected_sources=1,
                                     expected_vectors=good["png_count"])
    assert result["training_ready"] is False
    assert "feature_config_fingerprint_mismatch" in result["blockers"]


def test_rehashed_bundle_with_wrong_feature_row_reference_is_rejected(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "reference", patients=6)
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    bundle = build_pretrain_bundle(
        Path(governance["path"]), fixture["feature_root"], output_root,
        expected_sources=1, expected_vectors=fixture["png_count"],
    )
    bundle_path = Path(bundle["bundle_path"])
    ref_path = bundle_path / "instance_refs.jsonl"
    lines = ref_path.read_text(encoding="utf-8").splitlines()
    ref = json.loads(lines[0])
    ref["tile_id"] = "f" * 64
    lines[0] = json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ref_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _manifest_rehash(bundle_path)
    with pytest.raises(ValueError, match="reference differs from source index"):
        verify_pretrain_bundle(bundle_path, fixture["feature_root"])


def test_feature_cache_nan_after_commit_fails_existing_audit(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "committed-nan", patients=3)
    feature_id = fixture["feature_status"]["feature_id"]
    release_root = fixture["feature_root"] / feature_id
    release = json.loads((release_root / "release.json").read_text())
    part = release_root / "parts" / release["parts"][0]["part_id"]
    matrix = np.load(part / "features.npy", mmap_mode="r+")
    matrix[0, 0] = np.nan
    matrix.flush()
    commit_path = part / "commit.json"
    commit = json.loads(commit_path.read_text())
    commit.pop("commit_id")
    commit["files"]["features.npy"] = file_hash(part / "features.npy")
    commit["commit_id"] = fingerprint(commit)
    commit_path.write_text(json.dumps(commit, ensure_ascii=False, indent=2), encoding="utf-8")
    release["parts"][0]["commit_id"] = commit["commit_id"]
    (release_root / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    report = inspect_feature_release(fixture["feature_root"], expected_sources=1,
                                     expected_vectors=fixture["png_count"])
    assert report["training_ready"] is False
    assert any("NaN/Inf" in blocker for blocker in report["blockers"])


def test_expired_deadline_blocks_full_fixture_before_bundle_publish(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "expired", patients=6)
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    with pytest.raises(BudgetExhausted, match="deadline"):
        build_pretrain_bundle(
            Path(governance["path"]), fixture["feature_root"], output_root,
            expected_sources=1, expected_vectors=fixture["png_count"],
            deadline=time.monotonic() - 1,
        )
    assert not list((output_root / "bundles").glob("*/bundle.json"))


def test_deadline_during_existing_feature_hash_verification_preserves_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histology_data import features as feature_module

    fixture = make_feature_fixture(tmp_path / "mid-audit", patients=6)
    output_root = tmp_path / "pretrain"
    governance = make_reviewed_governance(fixture, output_root)["governance"]
    feature_id = fixture["feature_status"]["feature_id"]
    release_root = fixture["feature_root"] / feature_id
    release = json.loads((release_root / "release.json").read_text())
    part_dir = release_root / "parts" / release["parts"][0]["part_id"]
    files_before = {name: file_hash(part_dir / name) for name in (
        "features.npy", "tile_index.jsonl", "qc.json", "commit.json",
    )}
    original_hash = feature_module._bounded_hash

    def expire_during_feature_hash(path: Path, deadline: float | None = None) -> str:
        if Path(path).name == "features.npy":
            raise BudgetExhausted("simulated cutoff during feature matrix hash")
        return original_hash(path, deadline)

    monkeypatch.setattr(feature_module, "_bounded_hash", expire_during_feature_hash)
    with pytest.raises(BudgetExhausted, match="simulated cutoff"):
        build_pretrain_bundle(
            Path(governance["path"]), fixture["feature_root"], output_root,
            expected_sources=1, expected_vectors=fixture["png_count"],
            deadline=time.monotonic() + 60,
        )
    files_after = {name: file_hash(part_dir / name) for name in files_before}
    assert files_after == files_before
    assert not list((output_root / "bundles").glob("*/bundle.json"))


def test_review_rows_cannot_duplicate_case_with_conflicting_labels(tmp_path: Path) -> None:
    fixture = make_feature_fixture(tmp_path / "conflict", patients=3)
    output_root = tmp_path / "pretrain"
    draft = create_case_review_draft(
        fixture["metadata"], fixture["source_root"], expected_sources=1,
        expected_patches=fixture["png_count"],
    )
    draft_dir = write_case_review_draft(draft, output_root)
    review_path = draft_dir / "case_review.csv"
    rows = list(csv.DictReader(review_path.open(encoding="utf-8", newline="")))
    for row in rows:
        row.update(patient_id=fixture["case_patients"][row["case_id"]], identity_verified="true",
                   identity_evidence="explicit synthetic evidence", case_label=str(fixture["case_labels"][row["case_id"]]),
                   label_verified="true", label_evidence="explicit synthetic case-label evidence")
    conflict = dict(rows[0], case_label="1" if rows[0]["case_label"] == "0" else "0")
    rows.append(conflict)
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(GovernanceError, match="duplicate_case_id"):
        build_governance(fixture["metadata"], review_path, output_root)
