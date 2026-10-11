"""Tiny verified-shape fixture tests for shared B member preparation/resume."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from histology_data.dual_common import ArtifactStore, WeightedPCA, normalize_rows
from histology_data.dual_member import prepare_member_states
from histology_data.dual_novelty import NegativePrototypeBank, RobustNoveltyScale

FEATURE_DIM = 2048
LENSES = ("4", "10", "40")


class TinyMemberDataset:
    def __init__(self, cases: dict[str, Any], records: dict[str, dict[str, Any]], case_ids: Sequence[str],
                 tile_order: dict[str, Any] | None = None) -> None:
        self._cases = cases
        self.records = {case_id: records[case_id] for case_id in case_ids}
        self.case_ids = tuple(case_ids)
        self._tile_order = tile_order or {}
        self.identity = {
            "mode": "smoke",
            "feature_id": "feature-fixture",
            "bundle_id": "bundle-fixture",
            "split_id": "split-fixture",
            "source_manifest_sha256": "a" * 64,
        }

    def load_case(self, case_id: str) -> Any:
        if case_id not in self.case_ids:
            raise ValueError("Case is outside this dataset view.")
        return self._cases[case_id]

    def loaded_tile_order(self, case_id: str) -> dict[str, list[str]]:
        return {lens: list(tile_ids) for lens, tile_ids in self._tile_order[case_id].items()}


def _fixture() -> tuple[TinyMemberDataset, Any, dict[str, Any], dict[str, Any], tuple[str, ...], str]:
    rng = np.random.default_rng(20261011)
    case_ids = ("negative-anchor", "negative-b", "negative-c", "positive-a", "positive-b")
    targets = (0, 0, 0, 1, 1)
    records = {
        case_id: {"patient_id": f"patient-{index}", "target": target}
        for index, (case_id, target) in enumerate(zip(case_ids, targets, strict=True))
    }
    cases: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    case_tile_order = {case_id: {lens: [] for lens in LENSES} for case_id in case_ids}
    sample_vectors = {lens: [] for lens in LENSES}
    sample_refs = {lens: [] for lens in LENSES}
    components = np.zeros((32, FEATURE_DIM), dtype=np.float32)
    components[:, :32] = np.eye(32, dtype=np.float32)
    for lens_index, lens in enumerate(LENSES):
        pca = WeightedPCA(
            mean_=np.zeros(FEATURE_DIM, dtype=np.float32),
            components_=components.copy(),
            explained_variance_ratio_=np.ones(32, dtype=np.float32) / 32,
            n_components_=32,
        )
        for case_index, case_id in enumerate(case_ids):
            count = 64 if case_index == 0 else 8
            matrix = rng.normal(size=(count, FEATURE_DIM)).astype(np.float32)
            if targets[case_index]:
                matrix[:, 0] += np.float32(0.4)
            features = {key: np.empty((0, FEATURE_DIM), dtype=np.float32) for key in LENSES}
            if case_id not in cases:
                cases[case_id] = SimpleNamespace(
                    case_id=case_id,
                    patient_id=records[case_id]["patient_id"],
                    target=targets[case_index],
                    features=features,
                    presence_mask=(False, False, False),
                )
            cases[case_id].features[lens] = matrix
            cases[case_id].presence_mask = tuple(key in {"4", "10", "40"} for key in LENSES)
            normalized = normalize_rows(matrix)
            projected = pca.transform(normalized)
            photo_ids = [f"{case_id}-{lens}-photo-{index % 4}" for index in range(count)]
            for row_index, photo_id in enumerate(photo_ids):
                tile_id = f"{case_id}-{lens}-tile-{row_index}"
                case_tile_order[case_id][lens].append(tile_id)
                rows[tile_id] = {"case_id": case_id, "lens": lens, "photo_id": photo_id, "tile_id": tile_id}
                sample_vectors[lens].append(projected[row_index])
                sample_refs[lens].append({
                    "case_id": case_id,
                    "lens": lens,
                    "photo_id": photo_id,
                    "tile_id": tile_id,
                    "source_case_label": targets[case_index],
                })
        sample_arrays = np.stack(sample_vectors[lens]).astype(np.float32)
        sample_vectors[lens] = sample_arrays
        sample_refs[lens] = sample_refs[lens]
        if lens_index == 0:
            pcas = {}
        pcas[lens] = pca
    dataset_all = TinyMemberDataset(cases, records, case_ids, case_tile_order)
    transforms = {
        "members": {case_id: records[case_id] for case_id in case_ids},
        "seed": 42,
        "dimensions": 32,
        "lenses": {
            lens: {
                "pca": pcas[lens],
                "sample_projected": sample_vectors[lens],
                "sample_refs": sample_refs[lens],
            }
            for lens in LENSES
        },
    }
    source_map = {
        "source_identity": dataset_all.identity,
        "rows": rows,
        "case_tile_order": case_tile_order,
    }

    def dataset_factory(ids: Sequence[str]) -> TinyMemberDataset:
        return TinyMemberDataset(cases, records, ids, case_tile_order)

    return dataset_all, dataset_factory, transforms, source_map, case_ids, "negative-anchor"


def test_member_state_commits_anchor_banks_and_train_only_scales_then_resumes(tmp_path) -> None:
    dataset_all, dataset_factory, transforms, source_map, train_ids, anchor = _fixture()
    store = ArtifactStore(tmp_path / "branch-b-store", {"study": "tiny-member-state-v1"})
    deadline = time.monotonic() + 120

    first = prepare_member_states(
        dataset_all, dataset_factory, transforms, source_map, train_ids, anchor,
        store, "outer_0/inner_0", deadline,
    )
    second = prepare_member_states(
        dataset_all, dataset_factory, transforms, source_map, train_ids, anchor,
        store, "outer_0/inner_0", deadline,
    )

    assert first["head_train_ids"] == tuple(case for case in train_ids if case != anchor)
    assert first["member_id"] == second["member_id"]
    assert first["anchor_token"] != anchor and len(first["anchor_token"]) == 64
    assert set(first["lens_states"]) == set(LENSES)
    assert {lens: state.bank.bank_id for lens, state in first["lens_states"].items()} == {
        lens: state.bank.bank_id for lens, state in second["lens_states"].items()
    }
    assert all(isinstance(state.bank, NegativePrototypeBank) for state in first["lens_states"].values())
    assert all(state.bank.prototypes.shape == (32, 32) for state in first["lens_states"].values())
    assert all(state.bank.anchor_case_id == anchor for state in first["lens_states"].values())
    assert all(isinstance(state.scale, RobustNoveltyScale) for state in first["lens_states"].values())
    assert first["member_identity_id"] == second["member_identity_id"]
    assert store.read(f"scopes/outer_0/inner_0/members/{first['anchor_token']}/state.pkl") is not None


def test_member_state_rejects_positive_anchor_and_changed_transform_before_publish(tmp_path) -> None:
    dataset_all, dataset_factory, transforms, source_map, train_ids, anchor = _fixture()
    store = ArtifactStore(tmp_path / "branch-b-store", {"study": "tiny-member-state-v1"})
    deadline = time.monotonic() + 120

    with pytest.raises(ValueError, match="reviewed negative"):
        prepare_member_states(
            dataset_all, dataset_factory, transforms, source_map, train_ids, "positive-a",
            store, "outer_0/inner_0", deadline,
        )

    prepare_member_states(
        dataset_all, dataset_factory, transforms, source_map, train_ids, anchor,
        store, "outer_0/inner_0", deadline,
    )
    changed = dict(transforms)
    changed_lenses = dict(transforms["lenses"])
    changed_entry = dict(changed_lenses["4"])
    changed_pca = changed_entry["pca"]
    changed_entry["pca"] = WeightedPCA(
        mean_=changed_pca.mean_.copy(),
        components_=changed_pca.components_.copy(),
        explained_variance_ratio_=changed_pca.explained_variance_ratio_.copy(),
        n_components_=32,
    )
    changed_entry["pca"].mean_[0] = np.float32(0.01)
    changed_lenses["4"] = changed_entry
    changed["lenses"] = changed_lenses
    with pytest.raises(ValueError, match="another scope/source/transform/membership"):
        prepare_member_states(
            dataset_all, dataset_factory, changed, source_map, train_ids, anchor,
            store, "outer_0/inner_0", deadline,
        )
