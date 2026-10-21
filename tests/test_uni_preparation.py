"""Small synthetic E2E: images→views→commits→patient folds→head→reload.

No clinical image, authenticated download, torch model or GPU is used here.
These tests establish software readiness only, never UNI accuracy/readiness.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histology_data import image_classification_r0_r3 as heads, uni_cache, uni_smoke, uni_study, uni_views
from histology_data.io import file_hash
from scripts.build_image_classification_runtime import build_runtime


class SyntheticEncoder:
    """Deterministic image statistics with a deliberately non-UNI identity."""

    feature_dim = 1024

    def __init__(self) -> None:
        self.calls = 0

    def descriptor(self) -> dict:
        return {"kind": "SYNTHETIC_TEST_ONLY", "feature_dim": self.feature_dim, "version": 1}

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        self.calls += 1
        return np.array([np.resize(np.asarray(photo, dtype=np.float32).mean(axis=(0, 1)) + 1, 1024)
                         for photo in images], dtype=np.float32)


def make_source(root: Path, patients: int = 6) -> tuple[heads.Source, dict[int, Path]]:
    root.mkdir()
    rows, archives = [], {}
    for lens in (4, 10, 40):
        archive_path = root / f"{lens}.zip"
        archives[lens] = archive_path
        with zipfile.ZipFile(archive_path, "w") as archive:
            for patient in range(patients):
                for label in (0, 1):
                    photo = root / f"p{patient}-{lens}-{label}.png"
                    color = (30 + patient, 45 + lens, 180) if label else (180, 45 + lens, 30 + patient)
                    with Image.new("RGB", (530, 260), color) as image:
                        image.save(photo)
                    archive.write(photo, photo.name)
                    info = archive.getinfo(photo.name)
                    rows.append({"image_id": photo.stem, "patient_id": f"p{patient}", "objective_lens": lens,
                        "image_target": label, "raw_glade": "3" if label else "0", "file_name": photo.name,
                        "source_member": photo.name, "member_crc32": f"{info.CRC:08x}",
                        "member_byte_size": info.file_size, "original_image_sha256": file_hash(photo),
                        "original_width": 530, "original_height": 260, "source_format": "PNG"})
        for row in rows:
            if row["objective_lens"] == lens:
                row["source_archive_sha256"] = file_hash(archive_path)

    def inner(ids: list[str]) -> list[dict]:
        pats = sorted({r["patient_id"] for r in rows if r["image_id"] in ids})
        first = set(pats[:len(pats) // 2])
        left = [r["image_id"] for r in rows if r["image_id"] in ids and r["patient_id"] in first]
        right = [item for item in ids if item not in left]
        return [{"split": 1, "train_image_ids": left, "validation_image_ids": right},
                {"split": 2, "train_image_ids": right, "validation_image_ids": left}]

    ids = [r["image_id"] for r in rows]
    baseline = {"folds": [], "deployment": {"all": {"tuning": {"inner_splits": inner(ids)}}}}
    oof = []
    for fold in range(1, 4):
        test = [r["image_id"] for r in rows if int(r["patient_id"][1:]) % 3 == fold - 1]
        train = [item for item in ids if item not in test]
        baseline["folds"].append({"outer_fold": fold, "train_image_ids": train, "test_image_ids": test,
                                  "heads": {"all": {"tuning": {"inner_splits": inner(train)}}}})
        for row in rows:
            if row["image_id"] in test:
                score = .8 if row["image_target"] else .2
                oof.append({**row, "outer_fold": fold, "predicted_probability": score, "selected_threshold": .5,
                            "prediction_at_0_5": row["image_target"], "prediction_at_selected_threshold": row["image_target"]})
    return heads.Source(np.ones((len(rows), 2048), dtype=np.float32), rows, baseline, oof,
                         {"kind": "SYNTHETIC_TEST_ONLY"}), archives


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, heads.Source, dict[int, Path], SyntheticEncoder]:
    root = tmp_path_factory.mktemp("uni-synthetic")
    source, archives = make_source(root / "originals")
    encoder = SyntheticEncoder()
    result = uni_cache.extract(source.rows, archives, root / "cache", encoder, shard_size=6)
    assert result["state"] == "complete"
    return Path(result["cache_root"]), source, archives, encoder


def test_coordinates_keep_original_edges_and_pooling_is_exact() -> None:
    encoder = SyntheticEncoder()
    with Image.new("RGB", (513, 257), (20, 40, 80)) as image:
        combined, matrix, records = uni_views.embed_image(image, encoder, batch_size=1)
    assert combined.shape == (3072,) and matrix.shape == (3, 1024)
    assert records[1] == {"kind": "local", "x": 0, "y": 0, "width": 512, "height": 257}
    assert records[2] == {"kind": "local", "x": 512, "y": 0, "width": 1, "height": 257}
    assert sum(r["width"] * r["height"] for r in records[1:]) == 513 * 257
    assert np.linalg.norm(combined.reshape(3, 1024), axis=1) == pytest.approx(np.ones(3))
    assert np.array_equal(combined, uni_views.pool(matrix))


def test_pool_rejects_nan_and_zero_embeddings() -> None:
    with pytest.raises(ValueError, match="finite"):
        uni_views.pool(np.full((2, 1024), np.nan))
    with pytest.raises(ValueError, match="Zero"):
        uni_views.pool(np.zeros((2, 1024)))


def test_real_zip_cache_resumes_and_retains_view_provenance(tmp_path: Path) -> None:
    source, archives = make_source(tmp_path / "originals")
    encoder = SyntheticEncoder()
    first = uni_cache.extract(source.rows, archives, tmp_path / "cache", encoder, shard_size=12, max_new_shards=1)
    assert first["state"] == "partial" and first["new_shards"] == 1
    root = Path(first["cache_root"])
    preserved = file_hash(root / "shards/00000/commit.json")
    second = uni_cache.extract(source.rows, archives, tmp_path / "cache", encoder, shard_size=12)
    assert second["state"] == "complete" and second["new_shards"] == 2
    assert file_hash(root / "shards/00000/commit.json") == preserved
    calls = encoder.calls
    third = uni_cache.extract(source.rows, archives, tmp_path / "cache", encoder, shard_size=12)
    assert third["new_shards"] == 0 and encoder.calls == calls
    values, rows, _ = uni_cache.load_cache(root)
    assert values.shape == (36, 3072) and rows == source.rows
    assert not (root / "run.lock").exists()
    (root / "shards/00000/features.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        uni_cache.load_cache(root)


def test_original_zip_tampering_and_weak_label_reassignment_rejected(tmp_path: Path) -> None:
    source, archives = make_source(tmp_path / "originals")
    bad = [dict(row) for row in source.rows]
    bad[0]["image_target"] = 1 - bad[0]["image_target"]
    with pytest.raises(ValueError, match="label"):
        uni_cache.extract(bad, archives, tmp_path / "out", SyntheticEncoder())
    with zipfile.ZipFile(archives[4], "a") as archive:
        archive.writestr("changed", b"new")
    with pytest.raises(ValueError, match="ZIP checksum"):
        uni_cache.extract(source.rows, archives, tmp_path / "out", SyntheticEncoder())


def test_new_cache_binds_to_baseline_without_changing_labels(prepared: tuple) -> None:
    cache, baseline, _, _ = prepared
    incoming = uni_study.source_from_cache(cache, baseline)
    assert incoming.features.shape == (36, 3072)
    assert incoming.baseline == baseline.baseline and incoming.oof == baseline.oof
    assert incoming.receipt["uni_cache_id"] == cache.name
    changed = [dict(row) for row in baseline.rows]
    changed[0]["patient_id"] = "another-patient"
    with pytest.raises(ValueError, match="lineage"):
        uni_study.source_from_cache(cache, replace(baseline, rows=changed))


def test_cli_readiness_never_downloads_or_claims_gpu_checked(tmp_path: Path) -> None:
    completed = subprocess.run([sys.executable, "-m", "histology_data.uni_cli", "check-weights",
                               "--weights", str(tmp_path / "missing.bin"), "--weights-sha256", "a" * 64],
                               text=True, capture_output=True)
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {"ready": False, "reason": "local_weights_missing", "download_attempted": False}
    with pytest.raises(FileNotFoundError, match="access-approved"):
        uni_views.UniEncoder(tmp_path / "missing.bin", "a" * 64)
    checkpoint = tmp_path / "fake.bin"
    checkpoint.write_bytes(b"synthetic checkpoint bytes only")
    assert uni_views.weight_readiness(checkpoint, file_hash(checkpoint))["ready"]
    with pytest.raises(ValueError, match="checksum"):
        uni_views.weight_readiness(checkpoint, "a" * 64)


def test_nested_training_resume_and_single_image_restore(prepared: tuple, tmp_path: Path) -> None:
    cache, baseline, archives, encoder = prepared
    source = uni_study.source_from_cache(cache, baseline)
    config = heads.Config(lambdas=(.5,), bootstrap_repeats=10, budget_minutes=10, reserve_minutes=0)
    first = uni_study.run_study(source, tmp_path, config, max_new_fits=1, refit=True)
    assert first["state"] == "partial" and first["new_fits"] == 1
    study = Path(first["study_root"])
    commits = {p: file_hash(p) for p in study.rglob("commit.json")}
    second = uni_study.run_study(source, tmp_path, config, refit=True)
    assert second["state"] == "complete" and second["new_fits"] == 11
    assert second["real_encoder_checked"] is False
    assert all(file_hash(p) == sha for p, sha in commits.items())
    assert second["report"]["external_test_completed"] is False
    assert second["report"]["selected"] == "R0"  # synthetic R0 already has perfect ranking
    import joblib
    outer = second["report"]["results"]["fold-01"]["all"]
    fitted = joblib.load(study / outer["model_root"] / "model.joblib")
    train = source.positions(source.baseline["folds"][0]["train_image_ids"])
    normalized = source.features[train] / np.linalg.norm(source.features[train], axis=1, keepdims=True)
    assert fitted.named_steps["scaler"].mean_ == pytest.approx(normalized.mean(axis=0, dtype=np.float64), abs=1e-8)
    index = 0
    with zipfile.ZipFile(archives[4]) as archive:
        with archive.open(source.rows[index]["source_member"]) as stream, Image.open(stream) as image:
            with image.convert("RGB") as rgb:
                actual = uni_study.predict_image(study, rgb, encoder)
    cached = uni_study.predict_vector(study, source.features[index], encoder_descriptor=encoder.descriptor())
    assert actual == cached and actual["prediction_unit"] == "original_image"
    assert 0 <= actual["score"] <= 1 and actual["clinical_probability_calibrated"] is False
    again = uni_study.run_study(source, tmp_path, config, refit=True)
    assert again["new_fits"] == 0
    with pytest.raises(ValueError, match="identity"):
        uni_study.predict_vector(study, source.features[index], encoder_descriptor={"kind": "WRONG"})
    model = next(study.glob("arms/UNI/deployment/**/model.joblib"))
    model.write_bytes(b"corrupt owned model")
    with pytest.raises(ValueError, match="checksum"):
        uni_study.predict_vector(study, source.features[index], encoder_descriptor=encoder.descriptor())


def test_patient_leakage_is_rejected_before_fitting(prepared: tuple, tmp_path: Path) -> None:
    cache, baseline, _, _ = prepared
    source = uni_study.source_from_cache(cache, baseline)
    copied = json.loads(json.dumps(source.baseline))
    first = copied["folds"][0]["heads"]["all"]["tuning"]["inner_splits"][0]
    first["train_image_ids"].append(first["validation_image_ids"][0])
    with pytest.raises(ValueError, match="leaks patients"):
        uni_study.run_study(replace(source, baseline=copied), tmp_path)
    assert not list(tmp_path.iterdir())


def test_uni_runtime_is_code_only_and_includes_pinned_requirements(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    receipt = build_runtime(source, tmp_path / "runtime.zip", requirements_name="requirements-uni.txt")
    with zipfile.ZipFile(receipt["path"]) as archive:
        names = set(archive.namelist())
        assert {"histology_data/uni_views.py", "histology_data/uni_cache.py", "histology_data/uni_study.py",
                "histology_data/uni_cli.py", "requirements-uni.txt"} <= names
        assert not any(name.endswith((".bin", ".pth", ".tiff", ".csv", ".xlsx")) for name in names)
        assert b"scikit-learn==1.8.0" in archive.read("requirements-image-classification.txt")
        archive.extractall(tmp_path / "runtime")  # locally produced code-only archive
    assert file_hash(Path(receipt["path"])) == receipt["sha256"]
    completed = subprocess.run([sys.executable, "-m", "histology_data.uni_cli", "--help"],
                               cwd=tmp_path / "runtime", capture_output=True, text=True)
    assert completed.returncode == 0 and "check-weights" in completed.stdout and "smoke" in completed.stdout


def test_resource_smoke_records_synthetic_scope_and_reuses_commits(prepared: tuple, tmp_path: Path) -> None:
    _, source, archives, encoder = prepared
    result = uni_smoke.run(source.rows, archives, tmp_path, encoder)
    assert result["technical_smoke_passed"] and result["quality_evaluated"] is False
    assert result["GPU_checked"] is False and result["encoder_kind"] == "SYNTHETIC_TEST_ONLY"
    assert [r["objective_lens"] for r in result["checks"]] == [4, 10, 40]
    assert all(r["peak_vram_bytes"] is None and r["batch_parity_max_absolute_delta"] == 0 for r in result["checks"])
    calls = encoder.calls
    again = uni_smoke.run(source.rows, archives, tmp_path, encoder)
    assert again == result and encoder.calls == calls


def test_decimal_excel_glade_is_preserved_through_zip_extraction(tmp_path: Path) -> None:
    source, archives = make_source(tmp_path / "originals")
    rows = [{**row, "raw_glade": row["raw_glade"] + ".0"} for row in source.rows]
    result = uni_cache.extract(rows, archives, tmp_path / "cache", SyntheticEncoder())
    assert result["state"] == "complete"
    _, restored, _ = uni_cache.load_cache(Path(result["cache_root"]))
    assert [r["raw_glade"] for r in restored] == [r["raw_glade"] for r in rows]


def test_actual_original_image_producer_and_baseline_loader_bridge(tmp_path: Path) -> None:
    from histology_data import image_classification_studies as studies
    from tests.test_image_classification_studies import source as producer

    data, _ = producer.__wrapped__(tmp_path)
    result = studies.run_study(data, studies.StudyConfig(lambda_values=(.5,), budget_minutes=5, reserve_minutes=0))
    original = heads.load_source(data.run_root, Path(result["study_root"]))
    # Resolve actual synthetic archive paths from the existing inventory producer.
    archives = {lens: tmp_path / "photos" / f"originals_{lens}x.zip" for lens in (4, 10, 40)}
    assert all(path.is_file() for path in archives.values())
    cached = uni_cache.extract(original.rows, archives, tmp_path / "new-cache", SyntheticEncoder())
    incoming = uni_study.source_from_cache(Path(cached["cache_root"]), original)
    assert incoming.rows == [{key: row[key] for key in uni_cache.SOURCE_FIELDS} for row in original.rows]
    assert len(heads.preflight(incoming)) == 72
