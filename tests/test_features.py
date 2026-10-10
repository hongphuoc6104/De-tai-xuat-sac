"""E2E proof for unattended extraction, resume, corruption and budget handling."""
from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from histology_data.features import (
    FeatureRunConfig,
    run_feature_extraction,
    verify_feature_part,
    verify_feature_release,
)


class TinyEncoder:
    """Deterministic fixture used to inspect index alignment, never production."""

    feature_dim = 3
    descriptor = {"name": "test-fixture-only", "weights_sha256": "fixture", "feature_dim": 3}

    def __init__(self, oom_limit: int | None = None, nan: bool = False) -> None:
        self.calls = 0
        self.oom_limit = oom_limit
        self.nan = nan
        self.releases = 0

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        self.calls += 1
        if self.oom_limit is not None and len(images) > self.oom_limit:
            raise RuntimeError("CUDA out of memory (simulated fixture)")
        values = np.array([image.getpixel((0, 0)) for image in images], dtype=np.float32)
        if self.nan:
            values[0, 0] = np.nan
        return values

    def release_memory(self) -> None:
        self.releases += 1


def fixture(tmp_path: Path, *, directory: bool = False, parts: int = 2) -> FeatureRunConfig:
    source = tmp_path / "inputs"
    source.mkdir()
    metadata = []
    for part in range(parts):
        member_bytes = []
        for index, lens in enumerate((4, 10, 40)):
            image = f"IMG_{part}_{lens}"
            metadata.append({"Ten_File": image + ".tif", "Ma_Nam": "YCT26", "Ma_So": str(part),
                             "Do_Phong_Dai": f"{lens}X", "Glade": "raw-string",
                             "Ket_Luan": "CARCINOM", "Ten_Slide": "Slide1"})
            for x in (0, 10):
                buffer = io.BytesIO()
                Image.new("RGB", (32, 32), (30 + part, 20 + index, 80 + x)).save(buffer, "PNG", compress_level=0)
                member = f"Tiles/{image}/{x}_0.png"
                member_bytes.append((member, buffer.getvalue()))
        if directory:
            for member, raw in member_bytes:
                path = source / member.removeprefix("Tiles/")
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(raw)
        else:
            with zipfile.ZipFile(source / f"Tiles-{part:03d}.zip", "w") as archive:
                for member, raw in member_bytes:
                    archive.writestr(member, raw)
    meta = tmp_path / "metadata.csv"
    pd.DataFrame(metadata).to_csv(meta, index=False)
    return FeatureRunConfig(meta, source, "directory" if directory else "zip", tmp_path / "out",
                            tmp_path / "scratch", tmp_path / "weights", batch_size=4, device="cpu",
                            expected_archives=None if directory else parts, expected_pngs=parts * 6,
                            directory_part_size=6)


def test_full_run_order_resume_and_corrupted_output_fail_closed(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    encoder = TinyEncoder()
    first = run_feature_extraction(config, encoder=encoder)
    assert first["status"] == "complete"
    assert first["committed_vectors"] == 12 and first["completed_parts"] == 2
    assert first["training_ready"] is False
    release = json.loads(Path(first["report_path"]).read_text())
    part = config.output_root / first["feature_id"] / "parts" / release["parts"][0]["part_id"]
    rows = [json.loads(line) for line in (part / "tile_index.jsonl").read_text().splitlines()]
    matrix = np.load(part / "features.npy")
    assert [row["row_index"] for row in rows] == list(range(6))
    assert all(row["patient_id"] is None and row["bag_id"] is None for row in rows)
    assert all(row["raw_glade"] == "raw-string" and "patch_target" not in row for row in rows)
    assert all(matrix[i, 2] == 80 + row["x"] for i, row in enumerate(rows))
    calls = encoder.calls
    resumed = run_feature_extraction(config, encoder=encoder)
    assert resumed["reused_parts"] == 2 and resumed["new_parts"] == 0
    assert encoder.calls == calls
    assert not list(config.work_root.iterdir()) and not (config.output_root / "extractor.lock").exists()
    (part / "features.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_feature_part(part)
    failed = run_feature_extraction(config, encoder=encoder)
    assert failed["status"] == "error"
    assert any("checksum" in row["message"] for row in failed["failures"])


def test_partial_upload_then_new_zip_preserves_completed_parts(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    delayed = tmp_path / "delayed.zip"
    shutil.move(config.source_root / "Tiles-001.zip", delayed)
    encoder = TinyEncoder()
    first = run_feature_extraction(config, encoder=encoder)
    assert first["status"] == "awaiting_sources" and first["completed_parts"] == 1
    first_release = json.loads(Path(first["report_path"]).read_text())
    original_id = first_release["parts"][0]["part_id"]
    shutil.move(delayed, config.source_root / "Tiles-001.zip")
    second = run_feature_extraction(config, encoder=encoder)
    assert second["status"] == "complete" and second["reused_parts"] == 1 and second["new_parts"] == 1
    assert second["feature_id"] == first["feature_id"]
    second_release = json.loads(Path(second["report_path"]).read_text())
    assert original_id in [row["part_id"] for row in second_release["parts"]]


def test_complete_release_survives_temporarily_missing_source(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    result = run_feature_extraction(config, encoder=TinyEncoder())
    release_path = Path(result["report_path"])
    before = release_path.read_bytes()
    (config.source_root / "Tiles-001.zip").unlink()
    partial = run_feature_extraction(config, encoder=TinyEncoder())
    assert partial["status"] == "awaiting_sources"
    assert release_path.read_bytes() == before
    assert verify_feature_release(config.output_root, result["feature_id"])["verified_parts"] == 2


def test_directory_same_size_and_mtime_change_is_not_reused(tmp_path: Path) -> None:
    config = fixture(tmp_path, directory=True, parts=1)
    first = run_feature_extraction(config, encoder=TinyEncoder())
    assert first["status"] == "complete"
    path = next(config.source_root.rglob("*.png"))
    stat = path.stat()
    Image.new("RGB", (32, 32), (100, 100, 100)).save(path, "PNG", compress_level=0)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = run_feature_extraction(config, encoder=TinyEncoder())
    assert second["status"] == "error"
    assert any("changed" in failure["message"] for failure in second["failures"])


def test_smoke_repeat_stays_bounded_and_does_not_become_full(tmp_path: Path) -> None:
    config = replace(fixture(tmp_path), max_new_parts=1, max_patches_per_part=3)
    first = run_feature_extraction(config, encoder=TinyEncoder())
    second = run_feature_extraction(config, encoder=TinyEncoder())
    assert first["status"] in {"smoke_complete", "part_limit_reached"}
    assert second["completed_parts"] == 1 and second["committed_vectors"] == 3
    assert second["new_parts"] == 0 and not second["feature_complete"]
    assert first["feature_id"] == second["feature_id"]
    rows = json.loads(Path(second["report_path"]).read_text())["parts"]
    assert rows[0]["coverage"] == {"4": 1, "10": 1, "40": 1}


def test_oom_reduces_batch_without_losing_or_reordering_rows(tmp_path: Path) -> None:
    config = fixture(tmp_path, parts=1)
    encoder = TinyEncoder(oom_limit=1)
    result = run_feature_extraction(config, encoder=encoder)
    assert result["status"] == "complete" and result["committed_vectors"] == 6
    assert encoder.releases == 2
    commit = json.loads(Path(result["report_path"]).read_text())["parts"][0]
    assert commit["effective_batch_size"] == 1 and commit["tile_count"] == 6


def test_nan_or_corrupt_png_never_publishes_success_commit(tmp_path: Path) -> None:
    config = fixture(tmp_path, directory=True, parts=1)
    nan = run_feature_extraction(config, encoder=TinyEncoder(nan=True))
    assert nan["status"] == "error"
    assert not list(config.output_root.rglob("commit.json"))
    assert not list(config.work_root.iterdir())
    next(config.source_root.rglob("*.png")).write_bytes(b"not-a-png")
    corrupt = run_feature_extraction(config, encoder=TinyEncoder())
    assert corrupt["status"] == "error"
    assert not list(config.output_root.rglob("commit.json"))


def test_small_budget_stops_before_new_work_and_cli_status_reads_it(tmp_path: Path) -> None:
    from histology_data.feature_cli import main

    config = replace(fixture(tmp_path), budget_minutes=1, reserve_minutes=0)
    encoder = TinyEncoder()
    result = run_feature_extraction(config, encoder=encoder)
    assert result["status"] == "budget_exhausted" and encoder.calls == 0
    assert main(["status", "--output", str(config.output_root)]) == 0
    assert not list(config.work_root.glob("**/features.npy"))


def test_config_and_writer_lock_reject_unsafe_or_concurrent_runs(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        replace(config, batch_size=True).validate()
    with pytest.raises(ValueError, match="separate"):
        replace(config, output_root=config.source_root / "out").validate()
    config.output_root.mkdir()
    (config.output_root / "extractor.lock").write_text(json.dumps({"hostname": "other-runtime", "pid": 123}))
    with pytest.raises(ValueError, match="locked"):
        run_feature_extraction(config, encoder=TinyEncoder())
    recovered = run_feature_extraction(replace(config, recover_lock=True), encoder=TinyEncoder())
    assert recovered["status"] == "complete"
    assert not (config.output_root / "extractor.lock").exists()


def test_part_limit_keeps_previously_completed_later_parts(tmp_path: Path) -> None:
    config = fixture(tmp_path, parts=3)
    parked = tmp_path / "parked"
    parked.mkdir()
    for name in ("Tiles-000.zip", "Tiles-001.zip"):
        shutil.move(config.source_root / name, parked / name)
    first = run_feature_extraction(config, encoder=TinyEncoder())
    assert first["completed_parts"] == 1
    for name in ("Tiles-000.zip", "Tiles-001.zip"):
        shutil.move(parked / name, config.source_root / name)
    second = run_feature_extraction(replace(config, max_new_parts=1), encoder=TinyEncoder())
    assert second["status"] == "part_limit_reached"
    assert second["completed_parts"] == 2 and second["committed_vectors"] == 12
    assert second["reused_parts"] == 1 and second["new_parts"] == 1


def test_interrupt_cleans_scratch_and_leaves_no_success_commit(tmp_path: Path) -> None:
    class InterruptedEncoder(TinyEncoder):
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            raise KeyboardInterrupt()

    config = fixture(tmp_path, parts=1)
    with pytest.raises(KeyboardInterrupt):
        run_feature_extraction(config, encoder=InterruptedEncoder())
    status = json.loads((config.output_root / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert not list(config.work_root.iterdir())
    assert not list(config.output_root.rglob("commit.json"))
    assert not (config.output_root / "extractor.lock").exists()


def test_explicit_recovery_handles_lock_from_interrupted_initial_write(tmp_path: Path) -> None:
    config = fixture(tmp_path, parts=1)
    config.output_root.mkdir()
    (config.output_root / "extractor.lock").write_text('{"interrupted":')
    recovered = run_feature_extraction(replace(config, recover_lock=True), encoder=TinyEncoder())
    assert recovered["status"] == "complete"
    assert not (config.output_root / "extractor.lock").exists()


def test_resume_hash_validation_obeys_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import histology_data.features as module

    config = fixture(tmp_path, parts=1)
    first = run_feature_extraction(config, encoder=TinyEncoder())
    assert first["status"] == "complete"
    before = [path.read_bytes() for path in config.output_root.rglob("commit.json")]
    counter = [0.0]

    def clock() -> float:
        counter[0] += 120
        return counter[0]

    monkeypatch.setattr(module.time, "monotonic", clock)
    encoder = TinyEncoder()
    result = run_feature_extraction(replace(config, budget_minutes=1, reserve_minutes=0), encoder=encoder)
    assert result["status"] == "budget_exhausted" and encoder.calls == 0
    assert [path.read_bytes() for path in config.output_root.rglob("commit.json")] == before


def test_audit_failure_never_publishes_terminal_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import histology_data.features as module

    config = fixture(tmp_path, parts=1)
    original = module.atomic_json
    statuses = []

    def capture(path: Path, value: dict) -> None:
        if path.name == "status.json":
            statuses.append(value["status"])
        original(path, value)

    def audit_failure(*_args, **_kwargs):
        raise ValueError("simulated final audit rejection")

    monkeypatch.setattr(module, "atomic_json", capture)
    monkeypatch.setattr(module, "verify_feature_release", audit_failure)
    with pytest.raises(ValueError, match="audit rejection"):
        run_feature_extraction(config, encoder=TinyEncoder())
    result = json.loads((config.output_root / "status.json").read_text())
    assert result["status"] == "error" and not result["feature_complete"]
    assert "complete" not in statuses


def test_consistent_truncated_release_cannot_claim_complete(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    result = run_feature_extraction(config, encoder=TinyEncoder())
    path = Path(result["report_path"])
    release = json.loads(path.read_text())
    release["parts"] = release["parts"][:1]
    release["committed_vectors"] = release["parts"][0]["tile_count"]
    path.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="planned|complete|plan"):
        verify_feature_release(config.output_root, result["feature_id"])


def test_recovery_cannot_take_lock_from_known_live_process(tmp_path: Path) -> None:
    import socket

    config = fixture(tmp_path, parts=1)
    config.output_root.mkdir()
    lock = config.output_root / "extractor.lock"
    lock.write_text(json.dumps({"hostname": socket.gethostname(), "pid": os.getpid(), "run_id": "live-writer"}))
    with pytest.raises(ValueError, match="active|live|locked"):
        run_feature_extraction(replace(config, recover_lock=True), encoder=TinyEncoder())
    assert json.loads(lock.read_text())["run_id"] == "live-writer"


def test_global_duplicate_audit_keeps_cross_case_instances_and_reverse_lookup(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    first_path = config.source_root / "Tiles-000.zip"
    second_path = config.source_root / "Tiles-001.zip"
    with zipfile.ZipFile(first_path) as source:
        clone = source.read("Tiles/IMG_0_4/0_0.png")
    with zipfile.ZipFile(second_path) as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(second_path, "w") as target:
        for name, raw in members:
            target.writestr(name, clone if name == "Tiles/IMG_1_40/10_0.png" else raw)
    result = run_feature_extraction(config, encoder=TinyEncoder())
    assert result["status"] == "complete" and result["committed_vectors"] == 12
    audit = verify_feature_release(config.output_root, result["feature_id"])
    assert audit["cross_candidate_case_duplicate_groups"] == 2
    file = config.output_root / result["feature_id"] / audit["duplicate_group_file"]
    groups = [json.loads(line) for line in file.read_text().splitlines()]
    assert len(groups) == 2 and all(group["count"] == 2 for group in groups)
    assert all(len({row["candidate_case_id"] for row in group["members"]}) == 2 for group in groups)
    assert all(row["row_index"] >= 0 and row["part_id"] and row["source_member"]
               for group in groups for row in group["members"])
