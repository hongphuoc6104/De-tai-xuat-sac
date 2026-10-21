"""End-to-end tests for the original-image technical audit."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from histology_data import image_classification as image_clf, image_classification_audit as audit
from tests.image_classification_fixtures import OriginalImageFixture, make_original_image_fixture


def _inventory(fixture: OriginalImageFixture, output: Path) -> Path:
    return image_clf.build_original_image_inventory(
        image_clf.OriginalImageInventoryConfig(
            metadata=fixture.metadata,
            archives=fixture.archives,
            output_root=output,
        )
    )


def _replace_member(path: Path, filename: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "r") as source:
        members = {
            info.filename: source.read(info.filename)
            for info in source.infolist() if not info.is_dir()
        }
    matches = [name for name in members if Path(name).name == filename]
    assert len(matches) == 1
    members[matches[0]] = payload
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as destination:
        for name, content in members.items():
            destination.writestr(name, content)


def _payload(fixture: OriginalImageFixture, filename: str) -> bytes:
    return fixture.image_paths[filename].read_bytes()


def _verified_feature_rows(fixture: OriginalImageFixture, inventory: Path) -> list[dict[str, object]]:
    manifest = image_clf._load_inventory(inventory)
    rows = []
    for source in manifest["images"]:
        with zipfile.ZipFile(fixture.archives[int(source["objective_lens"])]) as archive:
            payload = archive.read(source["source_member"])
        rows.append({
            "image_id": source["image_id"],
            "patient_id": source["patient_id"],
            "objective_lens": source["objective_lens"],
            "raw_glade": source["raw_glade"],
            "image_target": source["image_target"],
            "source_archive_sha256": source["source_archive_sha256"],
            "original_image_sha256": hashlib.sha256(payload).hexdigest(),
        })
    return rows


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_e2e_partial_resume_duplicate_gate_and_source_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_original_image_fixture(tmp_path / "source", patients=12)
    left, right = fixture.rows[0], fixture.rows[3]
    assert left["Do_Phong_Dai"] == right["Do_Phong_Dai"] == "4X"
    assert left["Ma_So"] != right["Ma_So"]
    assert left["Glade"] == "0" and right["Glade"] != "0"
    _replace_member(fixture.archives[4], right["Ten_File"], _payload(fixture, left["Ten_File"]))
    inventory = _inventory(fixture, tmp_path / "inventory")
    feature_rows = _verified_feature_rows(fixture, inventory)

    # Finish exactly one 32-image commit, then expire before the next image.
    fake_times = iter([0.0] + [0.0] * 33 + [0.1])
    monkeypatch.setattr(audit.time, "monotonic", lambda: next(fake_times, 0.1))
    partial_result = audit.audit_original_images(
        inventory, fixture.archives, tmp_path / "audit",
        verified_feature_rows=feature_rows,
        budget_minutes=0.001, reserve_minutes=0,
    )
    monkeypatch.undo()
    partial_root = partial_result.audit_root
    assert partial_result.status["state"] == "partial"
    assert partial_result.status["ready_for_fit"] is False
    assert len(partial_result.representative_image_ids) == 36
    assert partial_result.binding["verified_feature_rows_sha256"]
    partial = json.loads((partial_root / "status.json").read_text(encoding="utf-8"))
    assert partial["state"] == "partial"
    assert partial["audit_complete"] is False
    assert partial["training_ready"] is False
    assert partial["committed_image_count"] == 32
    assert len(list((partial_root / "chunks").glob("chunk-*/commit.json"))) == 1

    completed_result = audit.audit_original_images(
        inventory, fixture.archives, tmp_path / "audit",
        verified_feature_rows=feature_rows,
        budget_minutes=10, reserve_minutes=0,
    )
    completed_root = completed_result.audit_root
    assert completed_root == partial_root
    status = json.loads((completed_root / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((completed_root / "summary.json").read_text(encoding="utf-8"))
    assert status["state"] == "attention_required"
    assert status["audit_complete"] is True
    assert status["training_ready"] is False
    assert status["ready_for_fit"] is False
    assert completed_result.status == status
    assert status["committed_image_count"] == 36
    assert summary["automatic_exclusion_count"] == 0
    assert summary["automatic_relabel_count"] == 0
    assert summary["cross_patient_candidate_count"] >= 1
    assert summary["target_conflict_candidate_count"] >= 1

    record_rows = _rows(completed_root / "records.csv")
    assert len(record_rows) == 36
    assert {row["image_target"] for row in record_rows} == {"0", "1"}
    assert all(row["source_archive_sha256"] for row in record_rows)
    assert all(0 <= float(row["qc_mean_brightness"]) <= 1 for row in record_rows)
    duplicate_rows = _rows(completed_root / "duplicate_candidates.csv")
    duplicate = next(
        row for row in duplicate_rows
        if {row["image_id_left"], row["image_id_right"]}
        == {f"4x::{left['Ten_File']}", f"4x::{right['Ten_File']}"}
    )
    assert duplicate["exact_byte_duplicate"] == "True"
    assert duplicate["exact_pixel_duplicate"] == "True"
    assert duplicate["patient_relation"] == "cross_patient"
    assert duplicate["target_conflict"] == "True"
    assert "exact_bytes" in duplicate["candidate_reasons"]

    committed_before_tamper = (completed_root / "audit.commit.json").read_bytes()
    status_before_tamper = (completed_root / "status.json").read_bytes()
    with zipfile.ZipFile(fixture.archives[4], "a") as source:
        source.comment = b"archive changed after inventory"
    with pytest.raises(ValueError, match="SHA-256"):
        audit.audit_original_images(
            inventory, fixture.archives, tmp_path / "audit",
            budget_minutes=10, reserve_minutes=0,
        )
    assert (completed_root / "audit.commit.json").read_bytes() == committed_before_tamper
    assert (completed_root / "status.json").read_bytes() == status_before_tamper


def test_e2e_pixel_duplicate_finds_different_tiff_encoding(tmp_path: Path) -> None:
    fixture = make_original_image_fixture(tmp_path / "source", patients=4)
    left, right = fixture.rows[0], fixture.rows[3]
    source = Image.open(io.BytesIO(_payload(fixture, left["Ten_File"])))
    encoded = io.BytesIO()
    source.save(encoded, format="TIFF", compression="tiff_lzw")
    source.close()
    replacement = encoded.getvalue()
    assert replacement != _payload(fixture, left["Ten_File"])
    _replace_member(fixture.archives[4], right["Ten_File"], replacement)
    inventory = _inventory(fixture, tmp_path / "inventory")

    run_root = audit.audit_original_images(
        inventory, fixture.archives, tmp_path / "audit",
        budget_minutes=10, reserve_minutes=0,
    ).audit_root
    candidates = _rows(run_root / "duplicate_candidates.csv")
    match = next(
        row for row in candidates
        if {row["image_id_left"], row["image_id_right"]}
        == {f"4x::{left['Ten_File']}", f"4x::{right['Ten_File']}"}
    )
    assert match["exact_byte_duplicate"] == "False"
    assert match["exact_pixel_duplicate"] == "True"
    assert match["patient_relation"] == "cross_patient"
    assert json.loads((run_root / "summary.json").read_text(encoding="utf-8"))["state"] == "attention_required"


def test_representative_selection_is_deterministic_and_covers_lens_grade_groups() -> None:
    records = []
    for lens in (4, 10, 40):
        for grade in ("0", "3", "4", "5"):
            for index in range(5):
                records.append({
                    "image_id": f"{lens}x::{grade}::{index}",
                    "patient_id": f"patient-{lens}-{grade}-{index}",
                    "objective_lens": lens,
                    "raw_glade": grade,
                })
    selected = audit.select_representative_images(records, limit=48)
    repeated = audit.select_representative_images(records, limit=48)
    assert [row["image_id"] for row in selected] == [row["image_id"] for row in repeated]
    assert len(selected) == 48
    assert len({row["image_id"] for row in selected}) == 48
    assert len({row["patient_id"] for row in selected}) == 48
    counts = Counter((row["objective_lens"], row["raw_glade"]) for row in selected)
    assert len(counts) == 12
    assert set(counts.values()) == {4}


def test_phash_only_label_disagreement_is_reviewed_but_not_a_target_conflict() -> None:
    first = {
        "image_id": "4x::first",
        "objective_lens": 4,
        "patient_id": "patient-1",
        "image_target": 0,
        "original_image_sha256": "a" * 64,
        "rgb_pixel_sha256": "c" * 64,
        "phash_hex": "0000000000000000",
    }
    second = {
        "image_id": "4x::second",
        "objective_lens": 4,
        "patient_id": "patient-1",
        "image_target": 1,
        "original_image_sha256": "b" * 64,
        "rgb_pixel_sha256": "d" * 64,
        "phash_hex": "0000000000000001",
    }
    candidates = audit._duplicate_candidates([first, second], threshold=1)
    assert len(candidates) == 1
    assert candidates[0]["candidate_type"] == "near_similarity"
    assert candidates[0]["target_disagreement"] is True
    assert candidates[0]["target_conflict"] is False
    assert candidates[0]["review_required"] is True
    assert candidates[0]["blocks_ready_for_fit"] is False
    summary = audit._summary([first, second], candidates, [], threshold=1)
    assert summary["state"] == "complete"
    assert summary["ready_for_fit"] is True
    assert summary["near_similarity_target_disagreement_count"] == 1


def test_stale_audit_lock_requires_explicit_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    root.mkdir()
    lock = root / "audit.lock"
    lock.write_text(json.dumps({
        "run_id": "stale-run",
        "pid": 987654321,
        "hostname": audit.socket.gethostname(),
    }), encoding="utf-8")
    monkeypatch.setattr(audit.os, "kill", lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError()))

    with pytest.raises(ValueError, match="recover_lock=True"):
        audit._RunLock(root, recover=False).__enter__()
    assert lock.is_file()
    with audit._RunLock(root, recover=True):
        assert lock.is_file()
    assert not lock.exists()
