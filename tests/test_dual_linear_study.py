"""Tiny end-to-end contract tests for the nested dual-track linear runner."""
from __future__ import annotations

import hashlib
import pickle
import time
from typing import Any

import pytest

from histology_data.dual_common import ArtifactStore, fit_scope_transforms
from histology_data.dual_linear_study import _make_dataset, run_linear_study
from histology_data.mil_data import FixtureDataset, create_toy_fixture
from histology_data.splits import create_nested_patient_splits


class _SubsetDataset:
    """Exact-membership fixture adapter with deterministic tile order."""

    def __init__(self, base: FixtureDataset, case_ids: list[str], source_map: dict[str, Any]) -> None:
        self._base = base
        self.case_ids = list(case_ids)
        self.records = {case: base.records[case] for case in case_ids}
        self.identity = base.identity
        self.mode = base.mode
        self._source_map = source_map

    def load_case(self, case_id: str) -> Any:
        if case_id not in self.case_ids:
            raise KeyError(case_id)
        return self._base.load_case(case_id)

    def loaded_tile_order(self, case_id: str) -> dict[str, list[str]]:
        if case_id not in self.case_ids:
            raise KeyError(case_id)
        return {
            lens: list(self._source_map["case_tile_order"][case_id][lens])
            for lens in ("4", "10", "40")
        }


@pytest.fixture(scope="module")
def study_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("dual-linear-e2e")
    base = FixtureDataset(create_toy_fixture(root / "fixture", seed=29, case_count=16, rows_per_lens=32))
    cases = [
        {"case_id": case_id, "patient_id": base.records[case_id]["patient_id"],
         "case_label": base.records[case_id]["target"]}
        for case_id in base.case_ids
    ]
    splits = create_nested_patient_splits(cases, seed=42)
    source_rows: dict[str, dict[str, Any]] = {}
    case_tile_order: dict[str, dict[str, list[str]]] = {}
    for case_id in base.case_ids:
        case_tile_order[case_id] = {}
        case = base.load_case(case_id)
        for lens in ("4", "10", "40"):
            tiles: list[str] = []
            for row_index in range(len(case.features[lens])):
                tile_id = f"{case_id}-{lens}-{row_index:03d}"
                photo_id = f"{case_id}-{lens}-photo-{row_index // 32}"
                tiles.append(tile_id)
                source_rows[tile_id] = {
                    "case_id": case_id,
                    "lens": lens,
                    "photo_id": photo_id,
                    "tile_id": tile_id,
                    "original_crop_geometry": "unverified",
                }
            case_tile_order[case_id][lens] = tiles
    source_map = {
        "source_identity": base.identity,
        "rows": source_rows,
        "case_tile_order": case_tile_order,
    }

    def dataset_factory(ids: list[str]) -> _SubsetDataset:
        return _SubsetDataset(base, ids, source_map)

    shared_store = ArtifactStore(root / "shared", {"test": "dual-linear", "source": base.identity_hash})
    for outer in splits["outer"]:
        scopes = [
            (f"outer_{outer['fold']}/inner_{inner['fold']}", inner["train_cases"])
            for inner in outer["inner_folds"]
        ]
        scopes.append((f"outer_{outer['fold']}/refit", outer["train_cases"]))
        for scope, train_ids in scopes:
            transform = fit_scope_transforms(
                dataset_factory(list(train_ids)),
                source_map,
                list(train_ids),
                authorized_train_ids=list(train_ids),
                cap=64,
                dimensions=32,
                seed=42,
            )
            shared_store.publish(
                f"scopes/{scope}/transform.pkl",
                pickle.dumps(transform, protocol=5),
            )
    return {
        "root": root,
        "dataset": base,
        "dataset_factory": dataset_factory,
        "splits": splits,
        "source_map": source_map,
        "shared_store": shared_store,
    }


def _branch_store(inputs: dict[str, Any], name: str) -> ArtifactStore:
    return ArtifactStore(
        inputs["root"] / name,
        {"test": "dual-linear", "branch": name, "source": inputs["dataset"].identity_hash,
         "split_id": inputs["splits"]["split_id"]},
    )


def _run(inputs: dict[str, Any], branch: str, store: ArtifactStore) -> dict[str, Any]:
    return run_linear_study(
        branch,
        inputs["dataset"],
        inputs["dataset_factory"],
        inputs["splits"],
        inputs["source_map"],
        inputs["shared_store"],
        store,
        time.monotonic() + 180.0,
    )


def test_branch_a_runs_atlas_nested_selection_oof_and_source_assignments(study_inputs: dict[str, Any]) -> None:
    store = _branch_store(study_inputs, "branch-a")
    result = _run(study_inputs, "A", store)

    assert result["phase"] == "complete"
    assert result["a1"]["logical_head_fits"] == result["a1"]["expected_head_fits"] == 39
    assert result["a1"]["oof_count"] == 16
    assert len({row["case_id"] for row in result["a1"]["oof_predictions"]}) == 16
    assert result["counts_only_control"]["oof_count"] == 16
    assert result["a0"]["scope"] == "outer_0/inner_0"
    assert set(result["a0"]["lenses"]) == {"4", "10", "40"}
    for lens_report in result["a0"]["lenses"].values():
        assert len(lens_report["hdbscan_grid"]) == 16
        assert len(lens_report["ward_multicut"]) == 4
        assert len(lens_report["kmeans_partitions"]) == 3
        assert len(lens_report["kmeans_seed_stability"]) == 3
        assert len(lens_report["hdbscan_patient_resampling_stability"]) == 4
    case_id = study_inputs["splits"]["outer"][0]["inner_folds"][0]["train_cases"][0]
    assignment = store.read(
        f"assignments/outer_0_inner_0/k4/lens_4/"
        f"{hashlib.sha256(case_id.encode()).hexdigest()}.csv.gz"
    )
    assert assignment is not None and assignment.startswith(b"\x1f\x8b")
    assert store.read("results/a1_oof.json") is not None
    resumed = _run(study_inputs, "A", store)
    assert resumed["a1"]["oof_predictions"] == result["a1"]["oof_predictions"]


def test_branch_b_uses_anchor_exclusion_member_ensembles_and_oof(study_inputs: dict[str, Any]) -> None:
    pytest.importorskip("histology_data.dual_member")
    store = _branch_store(study_inputs, "branch-b")
    result = _run(study_inputs, "B", store)

    assert result["phase"] == "complete"
    assert result["b0"]["logical_head_fits"] == result["b0"]["expected_head_fits"]
    assert result["b0"]["logical_head_fits"] > 0
    assert result["b0"]["oof_count"] == 16
    assert len({row["case_id"] for row in result["b0"]["oof_predictions"]}) == 16
    assert result["counts_only_control"]["oof_count"] == 16
    assert store.read("results/b0_oof.json") is not None
    resumed = _run(study_inputs, "B", store)
    assert resumed["b0"]["oof_predictions"] == result["b0"]["oof_predictions"]


def test_runner_requires_all_shared_transforms_before_branch_writes(study_inputs: dict[str, Any]) -> None:
    empty_shared = ArtifactStore(
        study_inputs["root"] / "missing-shared",
        {"test": "dual-linear", "source": study_inputs["dataset"].identity_hash},
    )
    store = _branch_store(study_inputs, "missing-transform-branch")
    with pytest.raises(FileNotFoundError, match="Required committed train-only transform"):
        run_linear_study(
            "A",
            study_inputs["dataset"],
            study_inputs["dataset_factory"],
            study_inputs["splits"],
            study_inputs["source_map"],
            empty_shared,
            store,
            time.monotonic() + 30.0,
        )
    assert store.read("atlas/outer_0_inner_0/summary.json") is None


def test_dataset_factory_cannot_silently_sample_membership(study_inputs: dict[str, Any]) -> None:
    requested = study_inputs["dataset"].case_ids[:4]

    def truncated_factory(ids: list[str]) -> _SubsetDataset:
        return _SubsetDataset(study_inputs["dataset"], ids[:-1], study_inputs["source_map"])

    with pytest.raises(ValueError, match="silent sampling is forbidden"):
        _make_dataset(truncated_factory, requested)
