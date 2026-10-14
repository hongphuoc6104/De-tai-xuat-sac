"""Real-producer E2E checks for immutable whole-image execution resume."""
from __future__ import annotations

import json
import shutil
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from histology_data import (
    image_classification as core,
    image_classification_audit as audit,
    image_classification_execution as execution,
    pathology_encoder,
)
from histology_data.io import file_hash, fingerprint
from tests.image_classification_fixtures import OriginalImageFixture, make_original_image_fixture
from tests.test_image_classification import TinyPixelEncoder


@dataclass(frozen=True)
class ProducedArtifacts:
    fixture: OriginalImageFixture
    inventory_path: Path
    feature_root: Path
    encoder: TinyPixelEncoder
    config: core.FeatureExtractionConfig
    feature_rows: list[dict[str, Any]]
    audit_root: Path
    audit_id: str
    audit_commit_id: str
    audit_commit_sha256: str


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = f"symlink:{path.readlink()}"
        elif path.is_file():
            snapshot[relative] = file_hash(path)
        elif path.is_dir():
            snapshot[relative] = "directory"
    return snapshot


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _produce_artifacts(root: Path) -> ProducedArtifacts:
    fixture = make_original_image_fixture(root / "source", patients=12)
    inventory_path = core.build_original_image_inventory(
        core.OriginalImageInventoryConfig(
            metadata=fixture.metadata,
            archives=fixture.archives,
            output_root=root / "inventory",
        )
    )
    encoder = TinyPixelEncoder()
    config = core.FeatureExtractionConfig(
        batch_size=4,
        shard_size=8,
        max_long_edge=48,
        device="cpu",
        budget_minutes=5,
        reserve_minutes=0,
    )
    feature_root = core.extract_original_image_features(
        inventory_path,
        root / "features" / "A1",
        root / "weights",
        encoder=encoder,
        config=config,
    )
    _features, feature_rows, _release = core._load_feature_matrix(feature_root)
    audited = audit.audit_original_images(
        inventory_path,
        fixture.archives,
        root / "audit",
        verified_feature_rows=feature_rows,
        budget_minutes=5,
        reserve_minutes=0,
    )
    commit_path = audited.audit_root / "audit.commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    return ProducedArtifacts(
        fixture=fixture,
        inventory_path=inventory_path,
        feature_root=feature_root,
        encoder=encoder,
        config=config,
        feature_rows=feature_rows,
        audit_root=audited.audit_root,
        audit_id=audited.audit_root.name,
        audit_commit_id=commit["commit_id"],
        audit_commit_sha256=file_hash(commit_path),
    )


def _load_pinned_audit(produced: ProducedArtifacts, *, root: Path | None = None,
                       feature_rows: list[dict[str, Any]] | None = None) -> execution.VerifiedAudit:
    return execution.load_verified_audit(
        produced.inventory_path,
        root or produced.audit_root,
        expected_audit_id=produced.audit_id,
        expected_commit_id=produced.audit_commit_id,
        expected_commit_sha256=produced.audit_commit_sha256,
        verified_feature_rows=produced.feature_rows if feature_rows is None else feature_rows,
    )


def test_e2e_producer_to_readonly_n0_and_completed_cache_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inventory→frozen vectors→audit→pinned read→cache select does no rewrite."""
    produced = _produce_artifacts(tmp_path / "producer")
    audit_before = _snapshot_tree(produced.audit_root)
    feature_before = _snapshot_tree(produced.feature_root)

    def forbidden_writer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("read-only resume helper invoked a producer")

    monkeypatch.setattr(audit, "audit_original_images", forbidden_writer)
    monkeypatch.setattr(audit, "atomic_json", forbidden_writer)
    monkeypatch.setattr(core, "extract_original_image_features", forbidden_writer)

    verified = _load_pinned_audit(produced)
    selected = execution.select_completed_feature_release(
        produced.inventory_path,
        produced.feature_root.parents[1],
        TinyPixelEncoder(),
        produced.config,
    )
    assert selected is not None
    assert verified.receipt["audit_id"] == produced.audit_id
    assert verified.receipt["audit_commit_id"] == produced.audit_commit_id
    assert verified.receipt["image_count"] == 36
    assert verified.receipt["chunk_count"] == 2
    assert len(verified.representative_image_ids) == 36
    assert selected.run_root == produced.feature_root
    assert len(selected.rows) == 36 and selected.features.shape == (36, TinyPixelEncoder.feature_dim)

    reports = tmp_path / "execution"
    reports.mkdir()
    previous_report = reports / "attempt-old" / "report" / "result.json"
    previous_report.parent.mkdir(parents=True)
    previous_report.write_text('{"state":"complete"}\n', encoding="utf-8")
    old_report_hash = file_hash(previous_report)
    persistent_cache = tmp_path / "persistent"
    first_attempt = execution.create_execution_attempt(
        reports, persistent_cache, verified, attempt_id="gpu-a"
    )
    second_attempt = execution.create_execution_attempt(
        reports, persistent_cache, verified, attempt_id="gpu-b"
    )
    assert first_attempt.attempt_root != second_attempt.attempt_root
    assert first_attempt.n0_audit_root == second_attempt.n0_audit_root == produced.audit_root
    assert first_attempt.feature_roots == second_attempt.feature_roots == {
        "A1": persistent_cache / "features" / "A1",
        "B1": persistent_cache / "features" / "B1",
    }
    assert first_attempt.benchmark_root == second_attempt.benchmark_root == persistent_cache / "benchmarks"
    assert file_hash(previous_report) == old_report_hash
    assert _snapshot_tree(produced.audit_root) == audit_before
    assert _snapshot_tree(produced.feature_root) == feature_before


@pytest.mark.parametrize("pin_field", ("expected_audit_id", "expected_commit_id", "expected_commit_sha256"))
def test_pinned_audit_mismatch_is_rejected_without_writes(
    tmp_path: Path, pin_field: str,
) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    before = _snapshot_tree(produced.audit_root)
    values = {
        "expected_audit_id": produced.audit_id,
        "expected_commit_id": produced.audit_commit_id,
        "expected_commit_sha256": produced.audit_commit_sha256,
        "verified_feature_rows": produced.feature_rows,
    }
    values[pin_field] = "f" * 64
    with pytest.raises(ValueError):
        execution.load_verified_audit(produced.inventory_path, produced.audit_root, **values)
    assert _snapshot_tree(produced.audit_root) == before


def test_audit_feature_binding_mismatch_is_rejected_without_writes(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    before = _snapshot_tree(produced.audit_root)
    changed_rows = [dict(row) for row in produced.feature_rows]
    changed_rows[0]["original_image_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="feature-row identity"):
        _load_pinned_audit(produced, feature_rows=changed_rows)
    assert _snapshot_tree(produced.audit_root) == before


def test_audit_incomplete_chunk_is_rejected_without_loader_cleanup(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    partial_root = tmp_path / "partial-audit" / produced.audit_id
    partial_root.parent.mkdir()
    shutil.copytree(produced.audit_root, partial_root)
    (partial_root / "chunks" / "chunk-000000" / "commit.json").unlink()
    before = _snapshot_tree(partial_root)

    with pytest.raises(ValueError, match="incomplete or unexpected entries"):
        _load_pinned_audit(produced, root=partial_root)
    assert _snapshot_tree(partial_root) == before


def test_audit_source_semantics_are_checked_even_for_resigned_chunks(tmp_path: Path) -> None:
    """A re-fingerprinted chunk cannot detach a committed row from its ZIP member."""
    produced = _produce_artifacts(tmp_path / "producer")
    forged_root = tmp_path / "forged-audit" / produced.audit_id
    forged_root.parent.mkdir()
    shutil.copytree(produced.audit_root, forged_root)

    chunk_path = forged_root / "chunks" / "chunk-000000"
    records_path = chunk_path / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0]["source_member"] = "forged/member.tif"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    chunk_commit_path = chunk_path / "commit.json"
    chunk_commit = json.loads(chunk_commit_path.read_text(encoding="utf-8"))
    chunk_body = {key: value for key, value in chunk_commit.items() if key != "commit_id"}
    chunk_body["records_sha256"] = file_hash(records_path)
    chunk_commit = {**chunk_body, "commit_id": fingerprint(chunk_body)}
    _write_json(chunk_commit_path, chunk_commit)

    final_commit_path = forged_root / "audit.commit.json"
    final_commit = json.loads(final_commit_path.read_text(encoding="utf-8"))
    final_body = {key: value for key, value in final_commit.items() if key != "commit_id"}
    final_body["chunk_commit_ids"]["0"] = chunk_commit["commit_id"]
    final_commit = {**final_body, "commit_id": fingerprint(final_body)}
    _write_json(final_commit_path, final_commit)
    forged_pins = {
        "expected_audit_id": produced.audit_id,
        "expected_commit_id": final_commit["commit_id"],
        "expected_commit_sha256": file_hash(final_commit_path),
        "verified_feature_rows": produced.feature_rows,
    }
    before = _snapshot_tree(forged_root)

    with pytest.raises(ValueError, match="source field 'source_member'"):
        execution.load_verified_audit(produced.inventory_path, forged_root, **forged_pins)
    assert _snapshot_tree(forged_root) == before


def test_tampered_completed_feature_cache_fails_closed_without_rewrite(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    manifest_path = produced.feature_root / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["descriptor"]["preprocessing"]["max_long_edge"] = 999
    _write_json(manifest_path, manifest)
    before = _snapshot_tree(produced.feature_root)

    with pytest.raises(ValueError, match="feature release manifest"):
        execution.select_completed_feature_release(
            produced.inventory_path,
            produced.feature_root.parents[1],
            TinyPixelEncoder(),
            produced.config,
        )
    assert _snapshot_tree(produced.feature_root) == before


def test_feature_selector_does_not_fall_back_to_another_encoder_release(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    before = _snapshot_tree(produced.feature_root)

    class ChangedTinyEncoder(TinyPixelEncoder):
        @property
        def descriptor(self) -> dict[str, Any]:
            return {"name": "qa-tiny-pixel-statistics", "version": 2, "feature_dim": self.feature_dim}

    selected = execution.select_completed_feature_release(
        produced.inventory_path,
        produced.feature_root.parents[1],
        ChangedTinyEncoder(),
        produced.config,
    )
    assert selected is None
    assert _snapshot_tree(produced.feature_root) == before


def test_feature_selector_rejects_symlinked_matching_release(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    release_id = json.loads((produced.feature_root / "release.json").read_text())["release_id"]
    symlink_output = tmp_path / "symlink-output"
    runs_root = symlink_output / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / release_id).symlink_to(produced.feature_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        execution.select_completed_feature_release(
            produced.inventory_path, symlink_output, TinyPixelEncoder(), produced.config
        )


def test_attempt_creation_refuses_a_changed_verified_n0_commit(tmp_path: Path) -> None:
    produced = _produce_artifacts(tmp_path / "producer")
    verified = _load_pinned_audit(produced)
    commit_path = produced.audit_root / "audit.commit.json"
    commit_path.write_text("{}\n", encoding="utf-8")
    before = _snapshot_tree(produced.audit_root)
    attempts = tmp_path / "attempts"

    with pytest.raises(ValueError, match="N0 audit changed"):
        execution.create_execution_attempt(attempts, tmp_path / "cache", verified, attempt_id="resume")
    assert not attempts.exists()
    assert _snapshot_tree(produced.audit_root) == before


def test_retccl_policy_uses_real_download_and_encoder_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only remote transport/schema failures are optional; local integrity is fatal."""
    monkeypatch.setattr(
        pathology_encoder,
        "probe_official_weights",
        lambda **_kwargs: {"download_ready": True},
    )

    def fail_remote(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("synthetic offline endpoint")

    monkeypatch.setattr(pathology_encoder.urllib.request, "urlopen", fail_remote)
    with pytest.raises(urllib.error.URLError) as remote_error:
        pathology_encoder.download_official_weights(
            tmp_path / "weights" / "best_ckpt.pth", expected_sha256="a" * 64
        )
    assert execution.classify_retccl_failure(remote_error.value) is execution.RetCCLFailureKind.REMOTE_UNAVAILABLE
    assert not (tmp_path / "weights" / "best_ckpt.pth").exists()
    assert list((tmp_path / "weights").iterdir()) == []

    monkeypatch.setattr(
        pathology_encoder,
        "probe_official_weights",
        lambda **_kwargs: {"download_ready": False, "readiness_status": "remote_probe_failed"},
    )
    with pytest.raises(RuntimeError, match="not ready for download") as probe_error:
        pathology_encoder.download_official_weights(
            tmp_path / "unavailable" / "best_ckpt.pth", expected_sha256="a" * 64
        )
    assert execution.classify_retccl_failure(probe_error.value) is execution.RetCCLFailureKind.REMOTE_UNAVAILABLE
    assert not (tmp_path / "unavailable").exists()

    class TorchSchemaStub:
        @staticmethod
        def is_tensor(_value: Any) -> bool:
            return True

    with pytest.raises(ValueError) as schema_error:
        pathology_encoder._validate_state_dict_schema({}, {"weight": object()}, TorchSchemaStub)
    assert execution.classify_retccl_failure(schema_error.value) is execution.RetCCLFailureKind.CHECKPOINT_SCHEMA_INCOMPATIBLE

    checkpoint = tmp_path / "bad-pinned-checkpoint.pth"
    checkpoint.write_bytes(b"wrong local checkpoint")
    with pytest.raises(ValueError, match="full SHA-256 pin") as sha_error:
        pathology_encoder.RetCCLResNet50Encoder(checkpoint, expected_sha256="0" * 64)
    assert execution.classify_retccl_failure(sha_error.value) is execution.RetCCLFailureKind.FATAL
    assert execution.classify_retccl_failure(OSError("Drive I/O error")) is execution.RetCCLFailureKind.FATAL
    assert execution.classify_retccl_failure(AssertionError("failed invariant")) is execution.RetCCLFailureKind.FATAL
    assert execution.classify_retccl_failure(RuntimeError("CUDA out of memory")) is execution.RetCCLFailureKind.FATAL
