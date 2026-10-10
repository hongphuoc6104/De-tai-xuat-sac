"""Synthetic end-to-end tests for CPU-only pre-MIL orchestration."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import socket
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import scripts.run_pretrain_colab as runner_module
from histology_data.feature_encoder import FEATURE_DIM, OFFICIAL_WEIGHTS_SHA256, WEIGHTS_ENUM
from histology_data.features import FeatureRunConfig, run_feature_extraction
from histology_data.governance import create_case_review_draft, write_case_review_draft
from histology_data.io import file_hash
from histology_data.splits import SplitError
from scripts.run_pretrain_colab import (
    RunnerError,
    RunnerLock,
    run_pretrain,
)

_V1_TRANSFORM = {
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


class SyntheticEncoder:
    """Small deterministic test encoder; the runner must never invoke it."""

    feature_dim = FEATURE_DIM
    descriptor = {
        "name": "resnet50",
        "weights": WEIGHTS_ENUM,
        "weights_sha256": OFFICIAL_WEIGHTS_SHA256,
        "preprocessing": _V1_TRANSFORM,
        "torch_version": "synthetic-fixture",
        "torchvision_version": "synthetic-fixture",
        "pillow_version": "synthetic-fixture",
        "precision": "fp32",
        "implementation_version": "1.0.0",
    }
    calls = 0

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        type(self).calls += 1
        result = np.empty((len(images), FEATURE_DIM), dtype=np.float32)
        for offset, image in enumerate(images):
            rgb = image.convert("RGB").getpixel((0, 0))
            result[offset] = np.float32(rgb[0] * 65536 + rgb[1] * 256 + rgb[2])
        return result

    def release_memory(self) -> None:
        return None


def _fixture(root: Path, *, duplicate: bool = False, make_features: bool = True) -> dict[str, Any]:
    """Make synthetic metadata/ZIP inputs and, optionally, a tiny feature release."""
    drive_root = Path(root)
    drive_root.mkdir(parents=True, exist_ok=True)
    source_root = drive_root / "archives"
    source_root.mkdir()
    metadata_rows: list[dict[str, str]] = []
    png_members: list[tuple[str, bytes]] = []
    case_labels: dict[str, int] = {}
    patient_map: dict[str, str] = {}
    duplicate_source: bytes | None = None
    selected_duplicate_pair = (("YCT26_1", 4), ("YCT26_3", 4)) if duplicate else None
    png_index = 0

    for patient_index in range(12):
        patient_id = f"synthetic-patient-{patient_index:02d}"
        for class_index in (0, 1):
            case_number = patient_index * 2 + class_index + 1
            case_id = f"YCT26_{case_number}"
            label = class_index
            case_labels[case_id] = label
            patient_map[case_id] = patient_id
            conclusion = "CARCINOMA" if label else "BENIGN HYPERPLASIA WITH INFLAMMATION"
            for lens in (4, 10, 40):
                image_id = f"IMG_{case_number:02d}_{lens}X"
                metadata_rows.append({
                    "Ten_File": image_id + ".tif",
                    "Ma_Nam": "YCT26",
                    "Ma_So": str(case_number),
                    "Do_Phong_Dai": f"{lens}X",
                    "Glade": f"raw-grade-{class_index}",
                    "Ket_Luan": conclusion,
                    "Ten_Slide": f"synthetic-slide-{case_number}",
                })
                color = (png_index + 1, (png_index * 7 + 1) % 256, (png_index * 13 + 1) % 256)
                image_buffer = io.BytesIO()
                Image.new("RGB", (16, 16), color).save(image_buffer, format="PNG")
                raw = image_buffer.getvalue()
                if selected_duplicate_pair and (case_id, lens) == selected_duplicate_pair[0]:
                    duplicate_source = raw
                if selected_duplicate_pair and (case_id, lens) == selected_duplicate_pair[1]:
                    assert duplicate_source is not None
                    raw = duplicate_source
                png_members.append((f"Tiles/{image_id}/0_0.png", raw))
                png_index += 1

    metadata_path = drive_root / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    zip_path = source_root / "Tiles-20261009T164818Z-1-001.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, payload in png_members:
            archive.writestr(member, payload)

    fixture: dict[str, Any] = {
        "drive_root": drive_root,
        "metadata": metadata_path,
        "source_root": source_root,
        "case_labels": case_labels,
        "patient_map": patient_map,
        "png_count": len(png_members),
        "feature_root": drive_root / "features" / "v001",
    }
    if make_features:
        feature_config = FeatureRunConfig(
            metadata=metadata_path,
            source_root=source_root,
            source_kind="zip",
            output_root=fixture["feature_root"],
            work_root=drive_root / "scratch",
            weights_dir=drive_root / "weights",
            batch_size=8,
            device="cpu",
            precision="fp32",
            expected_archives=1,
            expected_pngs=len(png_members),
            budget_minutes=30,
            reserve_minutes=1,
        )
        fixture["feature_status"] = run_feature_extraction(feature_config, encoder=SyntheticEncoder())
    return fixture


def _write_confirmation(fixture: dict[str, Any], *, tamper: str | None = None) -> Path:
    drive_root = Path(fixture["drive_root"])
    draft = create_case_review_draft(
        fixture["metadata"], fixture["source_root"], expected_sources=1,
        expected_patches=fixture["png_count"],
    )
    draft_dir = write_case_review_draft(draft, drive_root)
    template_sha = file_hash(draft_dir / "case_review.csv")
    case_ids = sorted(fixture["case_labels"])
    labels = dict(fixture["case_labels"])
    patients = dict(fixture["patient_map"])
    content: dict[str, Any] = {
        "schema_version": 1,
        "confirmed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "synthetic fixture cases only",
        "identity_confirmation_quote": "Synthetic identity confirmation for this test fixture.",
        "identity_semantics": "Two synthetic case records may share a patient pseudonym.",
        "label_confirmation_quote": "Synthetic binary case-label confirmation for this test fixture.",
        "binary_mapping": {"synthetic_negative": 0, "synthetic_positive": 1},
        "mixed_benign_malignant_case_policy": "Synthetic fixture policy: positive case-level label; no patch labels.",
        "raw_glade_policy": "Preserve unchanged; not a binary target.",
        "case_scope_template_sha256": template_sha,
        "cases": case_ids,
        "patient_pseudonym_policy": "Synthetic, non-identifying fixture pseudonyms.",
        "approved_cohort_plan": {"primary": "common synthetic cohort", "supplementary": "all synthetic cases"},
        "metadata_sha256": file_hash(fixture["metadata"]),
        "case_labels": labels,
        "patient_map": patients,
    }
    if tamper == "metadata_sha":
        content["metadata_sha256"] = "0" * 64
    elif tamper == "template_sha":
        content["case_scope_template_sha256"] = "0" * 64
    elif tamper == "unknown_case":
        content["cases"].append("YCT26_UNKNOWN")
        content["case_labels"]["YCT26_UNKNOWN"] = 0
        content["patient_map"]["YCT26_UNKNOWN"] = "synthetic-unknown"
    elif tamper == "missing_patient_map":
        content["patient_map"].pop(case_ids[0])
    elif tamper == "blank_patient":
        content["patient_map"][case_ids[0]] = "  "
    confirmation_path = drive_root / "governance" / "reviews" / "user_confirmation.json"
    confirmation_path.parent.mkdir(parents=True, exist_ok=True)
    confirmation_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return confirmation_path


def _write_config(fixture: dict[str, Any], config_path: Path) -> Path:
    drive_root = Path(fixture["drive_root"])
    config = {
        "schema_version": 1,
        "drive_root": str(drive_root),
        "metadata": "metadata.csv",
        "source_root": "archives",
        "source_kind": "zip",
        "feature_root": "features/v001",
        "case_review": "governance/reviews/case_review.csv",
        "user_confirmation": "governance/reviews/user_confirmation.json",
        "cohorts": ["common", "all"],
        "expected_sources": 1,
        "expected_vectors": fixture["png_count"],
        "outer_folds": 3,
        "inner_folds": 2,
        "seed": 42,
        "allow_training": False,
        "source_pixels": "approved PNGs retained unchanged",
        "encoder_preprocessing": "synthetic test only; production uses official V1 transforms",
        "runtime_policy": "CPU pretrain tooling; no VM allocation, no encoder or training invocation",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


@pytest.mark.parametrize("mode", ["auto", "build"])
def test_runner_rejects_incompatible_splitter_before_pretrain_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    fixture = _fixture(tmp_path / "drive", make_features=False)
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")

    def reject_version() -> str:
        raise SplitError(
            "Nested patient splits require scikit-learn==1.8.0; found 1.6.1. "
            "Install with `python -m pip install --no-deps scikit-learn==1.8.0` and retry."
        )

    def unexpected_work(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Incompatible splitter must fail before governance or feature work.")

    monkeypatch.setattr(runner_module, "require_supported_splitter_version", reject_version)
    monkeypatch.setattr(runner_module, "_create_review_and_governance", unexpected_work)
    monkeypatch.setattr(runner_module, "_require_feature_ready_now", unexpected_work)
    with pytest.raises(RunnerError, match="require scikit-learn==1[.]8[.]0.*Install with.*pip install"):
        run_pretrain(config_path, mode=mode, deadline_utc=_deadline())
    assert not (fixture["drive_root"] / "pretrain_runs").exists()


def test_draft_mode_does_not_require_the_supported_splitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "drive", make_features=False)
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")

    def unexpected_version_check() -> str:
        pytest.fail("Draft generation does not run nested patient splits.")

    monkeypatch.setattr(runner_module, "require_supported_splitter_version", unexpected_version_check)
    result = run_pretrain(config_path, mode="draft")
    assert result["status"] == "draft_ready"
    assert result["source_complete"] is True


def _deadline(seconds: int = 3600) -> str:
    moment = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)
    return moment.isoformat().replace("+00:00", "Z")


def test_auto_builds_and_verifies_separate_common_and_all_bundles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    encoder_calls = SyntheticEncoder.calls

    result = run_pretrain(config_path, mode="auto", deadline_utc=_deadline())

    assert result["status"] == "data_ready"
    assert result["training_started"] is False
    assert result["data_ready"] is True
    assert set(result["cohorts"]) == {"common", "all"}
    assert result["cohorts"]["common"]["bundle_id"] != result["cohorts"]["all"]["bundle_id"]
    assert result["feature_id"] == fixture["feature_status"]["feature_id"]
    assert result["cohorts"]["common"]["training_ready"] is True
    assert result["cohorts"]["all"]["training_ready"] is True
    assert SyntheticEncoder.calls == encoder_calls
    for cohort in ("common", "all"):
        bundle = Path(result["cohorts"][cohort]["bundle_path"])
        manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        assert manifest["feature_id"] == result["feature_id"]
        assert (bundle / "bags.jsonl").is_file()
        assert (bundle / "instance_refs.jsonl").is_file()
        assert (bundle / "splits.json").is_file()
        assert (bundle / "training_readiness.json").is_file()
        assert not list(bundle.glob("*.npy"))

    review_path = fixture["drive_root"] / "governance/reviews/case_review.csv"
    published_rows = {row["case_id"]: row for row in csv.DictReader(review_path.open(encoding="utf-8", newline=""))}
    confirmation_path = fixture["drive_root"] / "governance/reviews/user_confirmation.json"
    confirmation_sha = file_hash(confirmation_path)
    template_draft = create_case_review_draft(
        fixture["metadata"], fixture["source_root"], expected_sources=1,
        expected_patches=fixture["png_count"],
    )
    template_path = fixture["drive_root"] / "governance" / "drafts" / template_draft["draft_id"] / "case_review.csv"
    template_rows = {row["case_id"]: row for row in csv.DictReader(template_path.open(encoding="utf-8", newline=""))}
    assert set(published_rows) == set(template_rows)
    for case_id, row in published_rows.items():
        assert row["patient_id"] == fixture["patient_map"][case_id]
        assert row["case_label"] == str(fixture["case_labels"][case_id])
        assert confirmation_sha in row["identity_evidence"]
        assert confirmation_sha in row["label_evidence"]
        for field in ("raw_glade_values_json", "raw_conclusion_values_json",
                      "candidate_label_codes_json", "metadata_case_sha256"):
            assert row[field] == template_rows[case_id][field]


@pytest.mark.parametrize(
    ("tamper", "expected_blocker"),
    [
        ("metadata_sha", "user_confirmation_metadata_sha256_mismatch"),
        ("template_sha", "user_confirmation_case_scope_template_mismatch"),
        ("unknown_case", "user_confirmation_case_scope_mismatch"),
        ("missing_patient_map", "user_confirmation_case_map_scope_mismatch"),
        ("blank_patient", "user_confirmation_patient_map_incomplete"),
    ],
)
def test_confirmation_tampering_and_missing_identity_fail_closed(
    tmp_path: Path, tamper: str, expected_blocker: str,
) -> None:
    fixture = _fixture(tmp_path / "drive", make_features=False)
    _write_confirmation(fixture, tamper=tamper)
    config_path = _write_config(fixture, tmp_path / "config.json")

    result = run_pretrain(config_path, mode="auto", deadline_utc=_deadline())

    assert result["status"] == "attention_required"
    assert expected_blocker in result["blockers"]
    assert not (fixture["drive_root"] / "governance/reviews/case_review.csv").exists()
    assert not list((fixture["drive_root"] / "governance").glob("*/governance.json"))


def test_auto_waits_for_24_source_writer_then_uses_full_release(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    status_path = fixture["feature_root"] / "status.json"
    complete_status = json.loads(status_path.read_text(encoding="utf-8"))
    partial_status = {
        **complete_status,
        "status": "awaiting_sources",
        "source_complete": False,
        "feature_complete": False,
        "observed_sources": 0,
        "observed_pngs": 0,
        "committed_vectors": 0,
    }
    status_path.write_text(json.dumps(partial_status), encoding="utf-8")
    lock_path = fixture["feature_root"] / "extractor.lock"
    lock_path.write_text('{"run_id":"human-owned-feature-writer"}', encoding="utf-8")
    sleep_calls: list[float] = []
    encoder_calls = SyntheticEncoder.calls

    def resume_after_owner_run(seconds: float) -> None:
        sleep_calls.append(seconds)
        status_path.write_text(json.dumps(complete_status), encoding="utf-8")
        lock_path.unlink()

    result = run_pretrain(
        config_path,
        mode="auto",
        deadline_utc=_deadline(),
        poll_seconds=30,
        sleep_fn=resume_after_owner_run,
    )

    assert sleep_calls == [30.0]
    assert result["status"] == "data_ready"
    assert not lock_path.exists()
    assert SyntheticEncoder.calls == encoder_calls
    run_dir = fixture["drive_root"] / "pretrain_runs" / result["run_id"]
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "waiting_for_feature" for event in events)
    assert any(event["event"] == "feature_release_ready" for event in events)


def test_writer_deadline_persists_attention_and_never_removes_extractor_lock(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    status_path = fixture["feature_root"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(status="awaiting_sources", source_complete=False, feature_complete=False,
                  observed_sources=0, observed_pngs=0, committed_vectors=0)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    extractor_lock = fixture["feature_root"] / "extractor.lock"
    extractor_lock.write_text('{"run_id":"active-owner-writer"}', encoding="utf-8")
    now = [dt.datetime.now(dt.timezone.utc)]
    deadline = now[0] + dt.timedelta(seconds=60)

    def advance(seconds: float) -> None:
        now[0] += dt.timedelta(seconds=seconds)

    result = run_pretrain(
        config_path,
        mode="auto",
        deadline_utc=deadline.isoformat(),
        poll_seconds=30,
        now_fn=lambda: now[0],
        sleep_fn=advance,
    )

    assert result["status"] == "attention_required"
    assert "feature_wait_deadline_reached" in result["blockers"]
    assert extractor_lock.is_file()
    assert not list((fixture["drive_root"] / "bundles").glob("*"))


@pytest.mark.parametrize("feature_status", ["error", "interrupted"])
def test_terminal_feature_error_or_interrupt_requires_attention(tmp_path: Path, feature_status: str) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    status_path = fixture["feature_root"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(status=feature_status, feature_complete=False, source_complete=False)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    encoder_calls = SyntheticEncoder.calls

    result = run_pretrain(config_path, mode="auto", deadline_utc=_deadline())

    assert result["status"] == "attention_required"
    assert f"feature_run_{feature_status}" in result["blockers"]
    assert SyntheticEncoder.calls == encoder_calls
    assert not list((fixture["drive_root"] / "bundles").glob("*"))


def test_build_mode_blocks_partial_feature_without_waiting(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    status_path = fixture["feature_root"] / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(status="awaiting_sources", source_complete=False, feature_complete=False)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    slept = False

    def forbidden_sleep(_: float) -> None:
        nonlocal slept
        slept = True
        raise AssertionError("build mode must not wait")

    result = run_pretrain(config_path, mode="build", deadline_utc=_deadline(), sleep_fn=forbidden_sleep)

    assert result["status"] == "attention_required"
    assert "feature_writer_active_or_incomplete" in result["blockers"]
    assert slept is False
    assert not (fixture["drive_root"] / "governance/reviews/case_review.csv").exists()


def test_expired_profile_cutoff_prevents_first_bundle_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    now = [dt.datetime.now(dt.timezone.utc)]
    deadline = now[0] + dt.timedelta(seconds=60)
    original_prepare = runner_module._create_review_and_governance
    build_calls: list[str] = []

    def prepare_then_expire(*args: Any, **kwargs: Any) -> Any:
        result = original_prepare(*args, **kwargs)
        now[0] = deadline
        return result

    def unexpected_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        build_calls.append(str(kwargs.get("cohort_mode")))
        raise AssertionError("No bundle build may start after the profile cutoff.")

    monkeypatch.setattr(runner_module, "_create_review_and_governance", prepare_then_expire)
    monkeypatch.setattr(runner_module, "build_pretrain_bundle", unexpected_build)
    result = run_pretrain(
        config_path,
        mode="auto",
        deadline_utc=deadline.isoformat(),
        now_fn=lambda: now[0],
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "attention_required"
    assert "profile_deadline_already_reached" in result["blockers"]
    assert build_calls == []


def test_cutoff_between_cohorts_stops_before_starting_the_second_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "drive")
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    now = [dt.datetime.now(dt.timezone.utc)]
    deadline = now[0] + dt.timedelta(seconds=3600)
    original_build = runner_module.build_pretrain_bundle
    build_calls: list[str] = []

    def build_then_expire(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cohort = str(kwargs["cohort_mode"])
        build_calls.append(cohort)
        result = original_build(*args, **kwargs)
        if cohort == "common":
            now[0] = deadline
        return result

    monkeypatch.setattr(runner_module, "build_pretrain_bundle", build_then_expire)
    result = run_pretrain(
        config_path,
        mode="auto",
        deadline_utc=deadline.isoformat(),
        now_fn=lambda: now[0],
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "attention_required"
    assert "profile_deadline_reached" in result["blockers"]
    assert build_calls == ["common"]
    bundles_root = fixture["drive_root"] / "bundles"
    assert len(list(bundles_root.iterdir())) == 1
    common_manifest = json.loads(next(bundles_root.iterdir()).joinpath("bundle.json").read_text(encoding="utf-8"))
    assert common_manifest["cohort_mode"] == "common"


def test_shared_cli_pretrain_lock_blocks_runner_writes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive", make_features=False)
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")
    cli_lock = fixture["drive_root"] / ".pretrain.lock"
    cli_lock.write_text('{"token":"other-cli-writer"}', encoding="utf-8")

    result = run_pretrain(config_path, mode="auto", deadline_utc=_deadline())

    assert result["status"] == "attention_required"
    assert "pretrain_writer_lock_active" in result["blockers"]
    assert cli_lock.is_file()
    assert not (fixture["drive_root"] / "governance/reviews/case_review.csv").exists()


def test_duplicate_groups_write_review_template_and_block_bundles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "drive", duplicate=True)
    _write_confirmation(fixture)
    config_path = _write_config(fixture, tmp_path / "config.json")

    result = run_pretrain(config_path, mode="auto", deadline_utc=_deadline())

    assert result["status"] == "attention_required"
    assert "duplicate-review-required" in result["blockers"]
    template_path = Path(result["details"]["duplicate_dispositions_template"])
    with template_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == (
        "kind", "sha256", "keep_tile_ids_json", "exclude_tile_ids_json", "canonical_tile_id",
        "reviewer", "reviewed_at", "evidence",
    )
    assert len(rows) >= 2
    assert all(row["kind"] and row["sha256"] for row in rows)
    assert all(not row["keep_tile_ids_json"] and not row["exclude_tile_ids_json"] for row in rows)
    assert not list((fixture["drive_root"] / "bundles").glob("*"))


def test_runner_lock_requires_explicit_recovery_and_refuses_live_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "pretrain_runs" / "runner.lock"
    lock_path.parent.mkdir()
    live = {"run_id": "live", "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": "now"}
    lock_path.write_text(json.dumps(live), encoding="utf-8")
    with pytest.raises(RunnerError, match="explicit recovery"):
        RunnerLock(lock_path, "next", recover_stale=False).acquire()
    with pytest.raises(RunnerError, match="live process"):
        RunnerLock(lock_path, "next", recover_stale=True).acquire()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["run_id"] == "live"

    stale = {**live, "run_id": "stale", "pid": os.getpid() + 10_000_000}
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    recovery = RunnerLock(lock_path, "manual-recovery", recover_stale=True)
    recovery.acquire()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["run_id"] == "manual-recovery"
    assert recovery.recovered is not None and recovery.recovered["operator_recovery_requested"] is True
    recovery.release()
    assert not lock_path.exists()
