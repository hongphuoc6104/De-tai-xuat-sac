"""Drive-safe feature staging contracts using a tiny governed test release."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from histology_data import mil_runtime
from histology_data.readiness import build_pretrain_bundle


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


@pytest.fixture(scope="module")
def governed_source(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from test_pretrain import make_feature_fixture, make_reviewed_governance

    root = tmp_path_factory.mktemp("mil-runtime-source")
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
    assert result["training_ready"] is True
    return {
        "bundle": Path(result["bundle_path"]),
        "feature_root": source["feature_root"],
        "source_root": source["root"],
    }


def test_staging_verifies_original_and_staged_bundle_without_mutating_drive_source(
    governed_source: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = governed_source["bundle"]
    feature_root = governed_source["feature_root"]
    before_bundle = _tree_hashes(bundle)
    before_features = _tree_hashes(feature_root)

    staged = mil_runtime.stage_training_inputs(
        bundle, feature_root, tmp_path / "ssd", deadline=time.monotonic() + 120,
    )
    assert staged["reused"] is False
    assert staged["bundle_path"].is_dir()
    assert (staged["bundle_path"] / "bundle.json").is_file()
    assert staged["feature_id"]
    assert staged["staged_bytes"] > 0
    staged_files = [path for path in staged["stage_root"].rglob("*") if path.is_file()]
    assert staged_files
    assert all(path.suffix.lower() not in {".png", ".zip", ".jpg", ".tif", ".pth", ".pt"}
               for path in staged_files)

    # The public verifier normally emits a fresh feature audit. Calling it on
    # the committed stage must not make that stage appear reusable under a
    # different source identity.
    from histology_data.readiness import verify_pretrain_bundle

    checked = verify_pretrain_bundle(staged["bundle_path"], staged["feature_root"])
    assert checked["training_ready"] is True
    reused = mil_runtime.stage_training_inputs(
        bundle, feature_root, tmp_path / "ssd", deadline=time.monotonic() + 120,
    )
    assert reused["reused"] is True
    assert reused["bundle_path"] == staged["bundle_path"]
    assert _tree_hashes(bundle) == before_bundle
    assert _tree_hashes(feature_root) == before_features


def test_staging_rejects_tampered_reuse_and_deadline_before_work(
    governed_source: dict[str, Path], tmp_path: Path,
) -> None:
    bundle = governed_source["bundle"]
    feature_root = governed_source["feature_root"]
    target = tmp_path / "staged"
    staged = mil_runtime.stage_training_inputs(bundle, feature_root, target, time.monotonic() + 120)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    part_id = manifest["feature_id"]
    feature_version = staged["feature_root"] / part_id
    feature_part = next((feature_version / "parts").iterdir())
    matrix = feature_part / "features.npy"
    matrix.write_bytes(matrix.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        mil_runtime.stage_training_inputs(bundle, feature_root, target, time.monotonic() + 120)

    untouched = tmp_path / "not-created"
    with pytest.raises(TimeoutError, match="deadline"):
        mil_runtime.stage_training_inputs(bundle, feature_root, untouched, time.monotonic() - 1)
    assert not untouched.exists()


def test_disk_preflight_enforces_four_gib_cap_before_staging(
    governed_source: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "disk-capped"
    monkeypatch.setattr(mil_runtime, "MAX_STAGE_BYTES", 1)
    with pytest.raises(OSError, match="above the"):
        mil_runtime.stage_training_inputs(
            governed_source["bundle"], governed_source["feature_root"], target,
            time.monotonic() + 120,
        )
    assert not any(path.is_dir() and len(path.name) == 64 for path in target.iterdir())
