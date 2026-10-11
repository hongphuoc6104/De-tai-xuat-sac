"""End-to-end QA for the whole-original-image frozen-feature baseline."""
from __future__ import annotations

import hashlib
import json
import struct
import threading
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from histology_data import image_classification as image_clf
from histology_data.io import file_hash, fingerprint
from tests.image_classification_fixtures import (
    LENSES,
    OriginalImageFixture,
    make_original_image_fixture,
    rewrite_metadata,
)


class TinyPixelEncoder:
    """Deterministic injectable encoder; no pretrained weights or network."""

    feature_dim = 4

    def __init__(self, *, entered: threading.Event | None = None,
                 release: threading.Event | None = None, fail_on_call: int | None = None) -> None:
        self.images: list[Image.Image] = []
        self.calls = 0
        self.entered = entered
        self.release = release
        self.fail_on_call = fail_on_call

    @property
    def descriptor(self) -> dict[str, Any]:
        return {"name": "qa-tiny-pixel-statistics", "version": 1, "feature_dim": self.feature_dim}

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(timeout=10):
            raise TimeoutError("test encoder release timed out")
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected encoder interruption")
        values = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            self.images.append(image.copy())
            values.append((
                float(rgb[:, :, 0].mean()),
                float(rgb[:, :, 1].mean()),
                float(rgb[:, :, 2].mean()),
                float(rgb.std()),
            ))
        return np.asarray(values, dtype=np.float32)

    def release_memory(self) -> None:
        return None


def _inventory(fixture: OriginalImageFixture, output: Path) -> Path:
    return image_clf.build_original_image_inventory(
        image_clf.OriginalImageInventoryConfig(
            metadata=fixture.metadata,
            archives=fixture.archives,
            output_root=output,
        )
    )


def _read_feature_rows(run_root: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[Path]]:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    shards = sorted((run_root / "features").glob("part-*"))
    for shard in shards:
        rows.extend(json.loads(line) for line in (shard / "rows.jsonl").read_text(encoding="utf-8").splitlines())
        vectors.extend(np.load(shard / "features.npy", allow_pickle=False))
    return rows, np.stack(vectors), shards


def _make_features(fixture: OriginalImageFixture, root: Path, encoder: TinyPixelEncoder | None = None,
                   *, budget_minutes: float = 5.0, reserve_minutes: float = 0.0,
                   batch_size: int = 4, shard_size: int = 7) -> tuple[Path, TinyPixelEncoder]:
    inventory = _inventory(fixture, root / "inventory")
    chosen_encoder = encoder or TinyPixelEncoder()
    run_root = image_clf.extract_original_image_features(
        inventory,
        root / "features",
        root / "weights",
        encoder=chosen_encoder,
        config=image_clf.FeatureExtractionConfig(
            batch_size=batch_size,
            shard_size=shard_size,
            max_long_edge=48,
            device="cpu",
            budget_minutes=budget_minutes,
            reserve_minutes=reserve_minutes,
        ),
    )
    return run_root, chosen_encoder


def _expected_vector(path: Path, max_long_edge: int = 48) -> np.ndarray:
    with Image.open(path) as original:
        canvas = image_clf.resize_pad_rgb(original, max_long_edge)
    rgb = np.asarray(canvas, dtype=np.float32)
    canvas.close()
    return np.asarray((rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean(), rgb.std()),
                      dtype=np.float32)


def _rewrite_zip_member(path: Path, member: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "r") as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist() if not info.is_dir()}
    members[member] = payload
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as destination:
        for name, content in members.items():
            destination.writestr(name, content)


def _crc32_four_byte_suffix(prefix: bytes, target_crc32: int) -> bytes:
    """Solve the invertible CRC32 map for the final four suffix bytes."""
    base_crc = zlib.crc32(prefix + bytes(4))
    basis: dict[int, tuple[int, int]] = {}
    for input_bit in range(32):
        suffix = (1 << input_bit).to_bytes(4, "little")
        value = zlib.crc32(prefix + suffix) ^ base_crc
        input_mask = 1 << input_bit
        for output_bit in range(31, -1, -1):
            if value & (1 << output_bit):
                if output_bit in basis:
                    basis_value, basis_mask = basis[output_bit]
                    value ^= basis_value
                    input_mask ^= basis_mask
                else:
                    basis[output_bit] = (value, input_mask)
                    break
    remainder = target_crc32 ^ base_crc
    solution = 0
    for output_bit in range(31, -1, -1):
        if remainder & (1 << output_bit):
            basis_value, basis_mask = basis[output_bit]
            remainder ^= basis_value
            solution ^= basis_mask
    assert remainder == 0
    result = solution.to_bytes(4, "little")
    assert zlib.crc32(prefix + result) == target_crc32
    return result


def _replace_stored_member_bytes_in_place(
    archive_path: Path, member: str, expected: bytes, replacement: bytes,
) -> None:
    assert len(expected) == len(replacement)
    with zipfile.ZipFile(archive_path, "r") as archive:
        info = archive.getinfo(member)
        assert info.compress_type == zipfile.ZIP_STORED
    raw = bytearray(archive_path.read_bytes())
    name_size, extra_size = struct.unpack_from("<HH", raw, info.header_offset + 26)
    data_start = info.header_offset + 30 + name_size + extra_size
    data_end = data_start + info.file_size
    assert bytes(raw[data_start:data_end]) == expected
    raw[data_start:data_end] = replacement
    archive_path.write_bytes(raw)


def test_inventory_maps_each_original_image_grade_and_commits_zip_provenance(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    inventory_path = _inventory(fixture, tmp_path / "inventory")
    manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = manifest["images"]

    assert manifest["label_level"] == "original_image_glade"
    assert manifest["target_mapping"] == {"0": 0, "3": 1, "4": 1, "5": 1}
    assert len(rows) == 36
    assert {row["raw_glade"] for row in rows} == {"0", "3", "4", "5"}
    assert all(row["image_target"] == int(row["raw_glade"] != "0") for row in rows)
    # The deliberately contradictory conclusion is not a target and mixed image labels are allowed per patient.
    labels_by_patient: dict[str, set[int]] = {}
    for row in rows:
        labels_by_patient.setdefault(row["patient_id"], set()).add(row["image_target"])
    assert len(labels_by_patient) == 12
    assert all(labels == {0, 1} for labels in labels_by_patient.values())
    assert all("case_label" not in row for row in rows)

    for lens in LENSES:
        source_receipt = manifest["sources"][str(lens)]
        assert source_receipt["archive_sha256"] == file_hash(fixture.archives[lens])
        assert source_receipt["member_count"] == 12
    for row in rows:
        with zipfile.ZipFile(fixture.archives[row["objective_lens"]]) as archive:
            info = archive.getinfo(row["source_member"])
            assert row["member_crc32"] == f"{info.CRC:08x}"
            assert row["member_byte_size"] == info.file_size
            assert row["source_archive_sha256"] == file_hash(fixture.archives[row["objective_lens"]])

    receipt = json.loads((inventory_path.parent / "inventory.commit.json").read_text(encoding="utf-8"))
    receipt_body = {key: receipt[key] for key in ("inventory_id", "manifest_sha256", "complete")}
    assert receipt["manifest_sha256"] == file_hash(inventory_path)
    assert receipt["commit_id"] == fingerprint(receipt_body)
    assert receipt["complete"] is True


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("Do_Phong_Dai", "", "objective lens"),
        ("Ten_File", "unmatched_original.tif", "absent"),
        ("Glade", "2", "unknown_glade"),
    ],
)
def test_inventory_fails_closed_on_missing_lens_match_or_unknown_grade(
    tmp_path: Path, field: str, replacement: str, message: str,
) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    rows = [dict(row) for row in fixture.rows]
    rows[0][field] = replacement
    rewrite_metadata(fixture.metadata, rows)

    with pytest.raises(ValueError, match=message):
        _inventory(fixture, tmp_path / "inventory")
    assert not (tmp_path / "inventory" / "inventory").exists()


def test_inventory_rejects_duplicate_metadata_and_duplicate_zip_basenames(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    duplicated = [dict(row) for row in fixture.rows]
    duplicated[1]["Ten_File"] = duplicated[0]["Ten_File"]
    rewrite_metadata(fixture.metadata, duplicated)
    with pytest.raises(ValueError, match="duplicate metadata Ten_File"):
        _inventory(fixture, tmp_path / "duplicate-metadata")

    fixture = make_original_image_fixture(tmp_path / "second-source")
    lens4 = fixture.archives[4]
    with zipfile.ZipFile(lens4, "a", compression=zipfile.ZIP_STORED) as archive:
        payload = fixture.image_paths[fixture.rows[0]["Ten_File"]].read_bytes()
        archive.writestr(f"duplicate-path/{fixture.rows[0]['Ten_File']}", payload)
    with pytest.raises(ValueError, match="duplicate exact-basename ZIP members"):
        _inventory(fixture, tmp_path / "duplicate-zip")
    assert fixture.metadata.exists()


def test_inventory_rejects_coordinate_patch_archive_members(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    rows = [dict(row) for row in fixture.rows]
    original_name = rows[0]["Ten_File"]
    rows[0]["Ten_File"] = "0_0.png"
    rewrite_metadata(fixture.metadata, rows)

    archive_path = fixture.archives[4]
    with zipfile.ZipFile(archive_path, "r") as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist() if not info.is_dir()}
    payload = members.pop(next(name for name in members if name.endswith(original_name)))
    members[f"Tiles/{Path(original_name).stem}/0_0.png"] = payload
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as destination:
        for name, content in members.items():
            destination.writestr(name, content)

    with pytest.raises(ValueError, match="coordinate|patch|Tiles"):
        _inventory(fixture, tmp_path / "patch-inventory")
    assert not (tmp_path / "patch-inventory" / "inventory").exists()


def test_resize_pad_keeps_far_edge_tissue_without_center_crop() -> None:
    source = Image.new("RGB", (90, 30), (40, 80, 120))
    pixels = np.asarray(source).copy()
    pixels[:, :10] = (255, 0, 255)
    pixels[:, 80:] = (0, 255, 255)
    source = Image.fromarray(pixels, mode="RGB")
    canvas = image_clf.resize_pad_rgb(source, max_long_edge=48)
    output = np.asarray(canvas)

    assert canvas.size == (48, 48)
    assert np.all(output[:16] == np.asarray(image_clf.PAD_RGB, dtype=np.uint8))
    assert np.all(output[32:] == np.asarray(image_clf.PAD_RGB, dtype=np.uint8))
    tissue = output[16:32]
    assert np.max(tissue[:, :5, 0]) == 255 and np.max(tissue[:, :5, 2]) == 255
    assert np.max(tissue[:, -5:, 1]) == 255 and np.max(tissue[:, -5:, 2]) == 255
    assert len(np.unique(tissue.reshape(-1, 3), axis=0)) > 3
    canvas.close()
    source.close()


def test_e2e_tiny_frozen_encoder_vectors_keep_row_identity_and_hashes(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    run_root, _encoder = _make_features(fixture, tmp_path / "run", batch_size=5, shard_size=8)
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    rows, matrix, shards = _read_feature_rows(run_root)

    assert status["state"] == "complete"
    assert status["expected_image_count"] == status["committed_image_count"] == 36
    assert matrix.shape == (36, 4)
    assert len(rows) == len({row["image_id"] for row in rows}) == 36
    for shard in shards:
        commit = json.loads((shard / "commit.json").read_text(encoding="utf-8"))
        commit_body = dict(commit)
        commit_id = commit_body.pop("commit_id")
        assert commit_id == fingerprint(commit_body)
        for name, digest in commit["files"].items():
            assert file_hash(shard / name) == digest
        shard_rows = [json.loads(line) for line in (shard / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        order_sha = hashlib.sha256("".join(row["image_id"] + "\n" for row in shard_rows).encode()).hexdigest()
        assert order_sha == commit["row_order_sha256"]
        shard_vectors = np.load(shard / "features.npy", allow_pickle=False)
        for row, vector in zip(shard_rows, shard_vectors, strict=True):
            assert row["vector_sha256"] == hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()
            assert vector.tolist() == pytest.approx(
                _expected_vector(fixture.image_paths[row["file_name"]]).tolist(), abs=1e-5,
            )
            assert row["original_image_sha256"] == file_hash(fixture.image_paths[row["file_name"]])
            assert row["padded_width"] == row["padded_height"] == 48
            assert row["image_target"] == int(row["raw_glade"] != "0")


def test_feature_extraction_resumes_only_verified_shards_and_rejects_tampering(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    inventory = _inventory(fixture, tmp_path / "inventory")
    output_root = tmp_path / "features"
    config = image_clf.FeatureExtractionConfig(
        batch_size=1, shard_size=2, max_long_edge=48, device="cpu",
        budget_minutes=5.0, reserve_minutes=0.0,
    )
    interrupted_encoder = TinyPixelEncoder(fail_on_call=3)
    with pytest.raises(RuntimeError, match="injected encoder interruption"):
        image_clf.extract_original_image_features(
            inventory, output_root, tmp_path / "weights", encoder=interrupted_encoder, config=config,
        )

    run_root = next((output_root / "runs").iterdir())
    first_shard = run_root / "features" / "part-000000"
    first_commit_hash = file_hash(first_shard / "commit.json")
    first_rows = [json.loads(line) for line in (first_shard / "rows.jsonl").read_text().splitlines()]
    assert len(first_rows) == 2
    assert json.loads((run_root / "status.json").read_text())["state"] == "failed"
    assert not (run_root / "run.lock").exists()

    resumed_encoder = TinyPixelEncoder()
    resumed_root = image_clf.extract_original_image_features(
        inventory, output_root, tmp_path / "weights", encoder=resumed_encoder, config=config,
    )
    status = json.loads((resumed_root / "status.json").read_text(encoding="utf-8"))
    all_rows, _matrix, shards = _read_feature_rows(resumed_root)
    assert resumed_root == run_root
    assert status["state"] == "complete"
    assert status["committed_image_count"] == status["expected_image_count"] == 36
    assert len(all_rows) == len({row["image_id"] for row in all_rows}) == 36
    assert resumed_encoder.calls == 34
    assert file_hash(first_shard / "commit.json") == first_commit_hash
    assert not (resumed_root / "run.lock").exists()

    with (shards[0] / "features.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        original_byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original_byte[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum failed"):
        image_clf.extract_original_image_features(
            inventory, output_root, tmp_path / "weights", encoder=TinyPixelEncoder(), config=config,
        )
    assert not (resumed_root / "run.lock").exists()


def test_second_writer_cannot_recover_a_live_feature_run_lock(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    inventory = _inventory(fixture, tmp_path / "inventory")
    output_root = tmp_path / "features"
    entered = threading.Event()
    release = threading.Event()
    blocked_encoder = TinyPixelEncoder(entered=entered, release=release)
    config = image_clf.FeatureExtractionConfig(
        batch_size=1, shard_size=2, max_long_edge=48, device="cpu",
        budget_minutes=5.0, reserve_minutes=0.0,
    )
    worker_errors: list[BaseException] = []
    worker_result: list[Path] = []

    def run_first_writer() -> None:
        try:
            worker_result.append(image_clf.extract_original_image_features(
                inventory, output_root, tmp_path / "weights", encoder=blocked_encoder, config=config,
            ))
        except BaseException as exc:  # Propagate worker failures after deterministic join.
            worker_errors.append(exc)

    worker = threading.Thread(target=run_first_writer, name="image-feature-lock-test")
    worker.start()
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(ValueError, match="active run still owns"):
            image_clf.extract_original_image_features(
                inventory, output_root, tmp_path / "weights", encoder=TinyPixelEncoder(), config=config,
            )
    finally:
        release.set()
        worker.join(timeout=15)
    assert not worker.is_alive()
    assert worker_errors == []
    assert len(worker_result) == 1
    assert not (worker_result[0] / "run.lock").exists()


def test_deadline_returns_partial_status_and_never_a_complete_release(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    inventory = _inventory(fixture, tmp_path / "inventory")
    run_root = image_clf.extract_original_image_features(
        inventory,
        tmp_path / "features",
        tmp_path / "weights",
        encoder=TinyPixelEncoder(),
        config=image_clf.FeatureExtractionConfig(
            batch_size=1, shard_size=2, max_long_edge=48, device="cpu",
            budget_minutes=1e-9, reserve_minutes=0.0,
        ),
    )
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    release = json.loads((run_root / "release.json").read_text(encoding="utf-8"))

    assert status["state"] == "partial"
    assert status["stop_reason"] == "deadline"
    assert status["committed_image_count"] < status["expected_image_count"]
    assert status["training_ready"] is False
    assert release["training_ready"] is False
    assert not (run_root / "complete.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        image_clf.train_original_image_classifier(
            run_root,
            config=image_clf.TrainingConfig(
                outer_folds=3, inner_folds=2, c_values=(0.1, 1.0), budget_minutes=1.0, reserve_minutes=0.0,
            ),
        )


def test_cross_host_lock_cannot_be_recovered_without_stopping_evidence(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    inventory = _inventory(fixture, tmp_path / "inventory")
    output_root = tmp_path / "features"
    config = image_clf.FeatureExtractionConfig(
        batch_size=1, shard_size=2, max_long_edge=48, device="cpu",
        budget_minutes=1e-9, reserve_minutes=0.0,
    )
    run_root = image_clf.extract_original_image_features(
        inventory, output_root, tmp_path / "weights", encoder=TinyPixelEncoder(), config=config,
    )
    lock = run_root / "run.lock"
    lock.write_text(json.dumps({
        "run_id": "remote-owner",
        "pid": 87654321,
        "hostname": "another-host",
        "started_at": "2026-10-12T00:00:00+00:00",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)cross-host|cannot be verified|stopping evidence|remote"):
        image_clf.extract_original_image_features(
            inventory,
            output_root,
            tmp_path / "weights",
            encoder=TinyPixelEncoder(),
            config=image_clf.FeatureExtractionConfig(
                batch_size=1, shard_size=2, max_long_edge=48, device="cpu",
                budget_minutes=5.0, reserve_minutes=0.0, recover_lock=True,
            ),
        )
    assert lock.exists()


def test_grouped_nested_training_persists_patient_safe_scaler_and_repeatable_prediction(
    tmp_path: Path,
) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    run_root, encoder = _make_features(fixture, tmp_path / "run", batch_size=5, shard_size=9)
    feature_rows, matrix, _shards = _read_feature_rows(run_root)
    vectors_by_id = {row["image_id"]: vector for row, vector in zip(feature_rows, matrix, strict=True)}
    patient_by_id = {row["image_id"]: row["patient_id"] for row in feature_rows}
    lens_by_id = {row["image_id"]: row["objective_lens"] for row in feature_rows}
    config = image_clf.TrainingConfig(
        outer_folds=3, inner_folds=2, seed=19, c_values=(0.1, 1.0),
        max_iter=300, budget_minutes=5.0, reserve_minutes=0.0,
    )

    result = image_clf.train_original_image_classifier(run_root, config=config)
    study_root = Path(result["study_root"])
    study = json.loads((study_root / "study.json").read_text(encoding="utf-8"))
    status = json.loads((study_root / "training_status.json").read_text(encoding="utf-8"))

    assert result["state"] == status["state"] == "complete"
    assert study["outer_fold_count"] == 3
    assert study["image_count"] == 36 and study["patient_count"] == 12
    assert study["prediction_unit"] == "original_image"
    assert study["training_target_field"] == "image_target"
    assert study["case_target_used"] is False and study["patch_labels_used"] is False
    assert study["patient_split_leakage_check"] == {
        "passed": True, "outer_folds": 3, "patient_overlap_count": 0,
    }

    test_fold_by_patient: dict[str, int] = {}
    for fold in study["folds"]:
        train_ids = set(fold["train_image_ids"])
        test_ids = set(fold["test_image_ids"])
        scaler_ids = set(fold["scaler_fit_image_ids"])
        assert train_ids.isdisjoint(test_ids)
        assert scaler_ids == train_ids
        train_patients = {patient_by_id[image_id] for image_id in train_ids}
        test_patients = {patient_by_id[image_id] for image_id in test_ids}
        assert train_patients.isdisjoint(test_patients)
        assert fold["patient_overlap_count"] == 0
        assert set(fold["train_patient_hashes"]).isdisjoint(fold["test_patient_hashes"])
        assert np.allclose(
            np.asarray(fold["scaler_mean"], dtype=np.float64),
            np.mean(np.stack([vectors_by_id[image_id] for image_id in fold["train_image_ids"]]), axis=0),
            rtol=1e-6,
            atol=1e-6,
        )

        for patient in test_patients:
            assert patient not in test_fold_by_patient
            test_fold_by_patient[patient] = fold["outer_fold"]
        for patient in test_patients:
            patient_ids = {image_id for image_id, value in patient_by_id.items() if value == patient}
            assert patient_ids.issubset(test_ids)
            assert {lens_by_id[image_id] for image_id in patient_ids} == {4, 10, 40}

        for inner in fold["inner_validation_splits"]:
            inner_train = set(inner["train_image_ids"])
            inner_validation = set(inner["validation_image_ids"])
            assert inner_train.isdisjoint(inner_validation)
            assert set(inner["scaler_fit_image_ids"]) == inner_train
            assert inner_train | inner_validation <= train_ids
            inner_train_patients = {patient_by_id[image_id] for image_id in inner_train}
            inner_validation_patients = {patient_by_id[image_id] for image_id in inner_validation}
            assert inner_train_patients.isdisjoint(inner_validation_patients)
            assert inner["patient_overlap_count"] == 0

    assert set(test_fold_by_patient) == set(patient_by_id.values())
    study_commit = json.loads((study_root / "study.commit.json").read_text(encoding="utf-8"))
    commit_body = dict(study_commit)
    commit_id = commit_body.pop("commit_id")
    assert commit_id == fingerprint(commit_body)
    loaded = image_clf.load_deployment_classifier(study_root)
    assert loaded.max_long_edge == 48

    image_id, image_row = next((row["image_id"], row) for row in feature_rows if row["objective_lens"] == 40)
    image_path = fixture.image_paths[image_row["file_name"]]
    first_prediction = image_clf.predict_original_image(
        study_root, image_path, tmp_path / "weights", encoder=encoder,
    )
    loaded_again = image_clf.load_deployment_classifier(study_root)
    second_prediction = image_clf.predict_original_image(
        study_root, image_path, tmp_path / "weights", encoder=encoder,
    )
    expected_probability = float(loaded_again.estimator.predict_proba(
        _expected_vector(image_path, max_long_edge=48)[None, :],
    )[0, 1])
    assert first_prediction["image_name"] == image_path.name
    assert first_prediction["prediction_unit"] == "original_image"
    assert first_prediction["max_long_edge"] == loaded_again.max_long_edge == 48
    assert first_prediction["cancer_probability"] == pytest.approx(second_prediction["cancer_probability"], abs=1e-8)
    assert first_prediction["cancer_probability"] == pytest.approx(expected_probability, abs=1e-6)
    assert first_prediction["image_target"] in {0, 1}
    assert image_id in {row["image_id"] for row in feature_rows}


def test_duplicate_original_bytes_with_conflicting_targets_fail_before_training(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    negative = next(row for row in fixture.rows if row["Ma_So"] == "SYN000" and row["Do_Phong_Dai"] == "4X")
    positive = next(row for row in fixture.rows if row["Ma_So"] == "SYN001" and row["Do_Phong_Dai"] == "4X")
    assert negative["Glade"] == "0" and positive["Glade"] in {"3", "4", "5"}
    assert negative["Ma_So"] != positive["Ma_So"]
    with zipfile.ZipFile(fixture.archives[4], "r") as archive:
        negative_member = next(name for name in archive.namelist() if name.endswith(negative["Ten_File"]))
        positive_member = next(name for name in archive.namelist() if name.endswith(positive["Ten_File"]))
        negative_payload = archive.read(negative_member)
    _rewrite_zip_member(fixture.archives[4], positive_member, negative_payload)

    inventory = _inventory(fixture, tmp_path / "inventory")
    run_root, _encoder = _make_features(fixture, tmp_path / "duplicate-run", batch_size=5, shard_size=8)
    feature_rows, _matrix, _shards = _read_feature_rows(run_root)
    by_filename = {row["file_name"]: row for row in feature_rows}
    duplicate_a = by_filename[negative["Ten_File"]]
    duplicate_b = by_filename[positive["Ten_File"]]

    assert duplicate_a["original_image_sha256"] == duplicate_b["original_image_sha256"]
    assert duplicate_a["patient_id"] != duplicate_b["patient_id"]
    assert {duplicate_a["image_target"], duplicate_b["image_target"]} == {0, 1}
    assert json.loads(inventory.read_text(encoding="utf-8"))["inventory_id"] == json.loads(
        (run_root / "release.json").read_text(encoding="utf-8"),
    )["inventory_id"]

    with pytest.raises(ValueError, match="(?i)duplicate.*(content|image)|same.*content|hash.*patient"):
        image_clf.train_original_image_classifier(
            run_root,
            config=image_clf.TrainingConfig(
                outer_folds=3, inner_folds=2, c_values=(0.1, 1.0), budget_minutes=5.0, reserve_minutes=0.0,
            ),
        )
    assert not (run_root / "training").exists()


def test_same_size_crc_preserving_payload_rebind_fails_before_feature_commit(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source")
    target = fixture.rows[0]
    archive_path = fixture.archives[4]
    with zipfile.ZipFile(archive_path, "r") as archive:
        member = next(name for name in archive.namelist() if name.endswith(target["Ten_File"]))
        original_payload = archive.read(member)
    # Keep the payload a valid TIFF while reserving eight trailing bytes for a CRC collision.
    original_payload += b"orig-tail"
    _rewrite_zip_member(archive_path, member, original_payload)
    inventory_path = _inventory(fixture, tmp_path / "inventory")
    manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_row = next(row for row in manifest["images"] if row["file_name"] == target["Ten_File"])
    original_archive_sha = manifest["sources"]["4"]["archive_sha256"]
    original_archive_size = archive_path.stat().st_size
    target_crc = int(inventory_row["member_crc32"], 16)

    prefix = original_payload[:-8]
    changed_prefix = prefix + b"DIFF"
    changed_payload = changed_prefix + _crc32_four_byte_suffix(changed_prefix, target_crc)
    assert changed_payload != original_payload
    assert len(changed_payload) == len(original_payload)
    assert zlib.crc32(original_payload) == zlib.crc32(changed_payload) == target_crc
    _replace_stored_member_bytes_in_place(archive_path, member, original_payload, changed_payload)

    assert archive_path.stat().st_size == original_archive_size
    assert file_hash(archive_path) != original_archive_sha
    with zipfile.ZipFile(archive_path, "r") as archive:
        rebound_info = archive.getinfo(member)
        assert rebound_info.file_size == inventory_row["member_byte_size"]
        assert f"{rebound_info.CRC:08x}" == inventory_row["member_crc32"]
        assert archive.testzip() is None
        assert archive.read(member) == changed_payload

    output_root = tmp_path / "features-after-rebind"
    encoder = TinyPixelEncoder()
    with pytest.raises(ValueError, match="(?i)(archive|source|zip).*(sha|hash|identity|checksum|bytes|differ)"):
        image_clf.extract_original_image_features(
            inventory_path,
            output_root,
            tmp_path / "weights",
            encoder=encoder,
            config=image_clf.FeatureExtractionConfig(
                batch_size=2, shard_size=4, max_long_edge=48, device="cpu",
                budget_minutes=5.0, reserve_minutes=0.0,
            ),
        )
    assert encoder.calls == 0
    assert not list((output_root / "runs").glob("*/features/part-*"))
