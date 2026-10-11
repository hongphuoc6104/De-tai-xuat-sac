"""E2E artifact publication and weighted train-scope contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from histology_data.dual_common import (
    ArtifactStore,
    fit_scope_transforms,
    fit_weighted_pca,
    normalize_rows,
    photo_sample_indices,
    project_case,
)


def test_committed_job_resume_and_corruption(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", {"source": "verified", "protocol": "v1"})
    store.publish("A/shard.bin", b"complete")
    recovered = ArtifactStore(tmp_path / "run", {"source": "verified", "protocol": "v1"})
    assert recovered.read("A/shard.bin") == b"complete"
    with pytest.raises(ValueError, match="overwrite"):
        recovered.publish("A/shard.bin", b"other")
    (tmp_path / "run/A/shard.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        recovered.read("A/shard.bin")


def test_uncommitted_payload_never_advances_and_scope_mismatch(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", {"scope": "inner0"})
    (tmp_path / "run/incomplete.bin").write_bytes(b"partial")
    assert store.read("incomplete.bin") is None
    store.publish("incomplete.bin", b"complete")
    assert store.read("incomplete.bin") == b"complete"
    with pytest.raises(ValueError, match="identity"):
        ArtifactStore(tmp_path / "run", {"scope": "outer_test"})
    with pytest.raises(ValueError, match="relative"):
        store.publish("../escape.bin", b"invalid")


def test_weights_change_actual_pca_fit_and_project_queries_without_refit() -> None:
    class Source:
        case_ids = ["a", "b"]
        identity = {"mode": "smoke", "source": "toy"}

        def loaded_tile_order(self, case):
            return orders[case]

        records = {"a": {"patient_id": "a", "target": 0}, "b": {"patient_id": "b", "target": 1}}

        def load_case(self, case: str) -> SimpleNamespace:
            values = np.tile(
                np.array([1, 0] if case == "a" else [0, 1], dtype=np.float32), (2 if case == "a" else 6, 1)
            )
            return SimpleNamespace(
                features={lens: values for lens in ["4", "10", "40"]},
                patient_id=case,
                target=self.records[case]["target"],
            )

    rows, orders = {}, {}
    for case in ["a", "b"]:
        orders[case] = {}
        for lens in ["4", "10", "40"]:
            tiles = [f"{case}_{lens}_{i}" for i in range(2 if case == "a" else 6)]
            orders[case][lens] = tiles
            for tile in tiles:
                rows[tile] = {"photo_id": case, "tile_id": tile}
    source = Source()
    transforms = fit_scope_transforms(
        source,
        {"rows": rows, "case_tile_order": orders, "source_identity": source.identity},
        ["a", "b"],
        authorized_train_ids=["a", "b"],
        dimensions=1,
    )
    assert np.allclose(transforms["lenses"]["4"]["pca"].mean_, [0.5, 0.5], atol=1e-6)
    projected = project_case(source, "b", transforms, chunk=2)
    assert projected["4"].shape == (6, 1)
    assert np.isfinite(projected["4"]).all()
    with pytest.raises(ValueError, match="authorized"):
        fit_scope_transforms(
            source,
            {"rows": rows, "case_tile_order": orders, "source_identity": source.identity},
            ["a"],
            authorized_train_ids=["a", "b"],
            dimensions=1,
        )


def test_photo_sampling_without_padding_and_weighted_transform() -> None:
    ids = ["photo0"] * 10 + ["photo1"] * 2
    selected = photo_sample_indices(ids, cap=4, seed=42)
    assert len(selected) == len(set(selected)) == 4
    assert sum(ids[int(i)] == "photo1" for i in selected) == 2
    assert len(photo_sample_indices(ids, cap=100, seed=42)) == 12
    x = np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32)
    pca = fit_weighted_pca(x, np.array([0.5, 0.25, 0.25]), 1, 42)
    assert np.allclose(pca.mean_, [0.5, 0.5])
    assert pca.transform(x).shape == (3, 1)


def test_invalid_norm_and_weight_do_not_silently_repair() -> None:
    with pytest.raises(ValueError, match="norm"):
        normalize_rows(np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid"):
        fit_weighted_pca(np.ones((3, 4), dtype=np.float32), np.array([0, 1, 1]), 2, 42)
    with pytest.raises(ValueError, match="variance"):
        fit_weighted_pca(np.ones((3, 4), dtype=np.float32), np.ones(3), 2, 42)


def test_existing_committed_store_reader_does_not_take_writer_lease(tmp_path):
    from histology_data.mil_checkpoint import FitLock

    root = tmp_path / "shared"
    ArtifactStore(root, {"scope": "shared"}).publish("ready.bin", b"ready")
    with FitLock(root):
        reader = ArtifactStore(root, {"scope": "shared"})
        assert reader.read("ready.bin") == b"ready"
