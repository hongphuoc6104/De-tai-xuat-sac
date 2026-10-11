"""Bounded synthetic loop/resume checks for B1 nested member orchestration."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from histology_data import dual_neural_study
from histology_data.dual_common import ArtifactStore, WeightedPCA, normalize_rows
from histology_data.dual_neural_study import outer_epoch_budget, run_neural_study, select_beta
from histology_data.mil_checkpoint import CheckpointStore

FEATURE_DIM = 2048
LENSES = ("4", "10", "40")


class TinyDataset:
    def __init__(self, cases: dict[str, Any], records: dict[str, dict[str, Any]],
                 tile_order: dict[str, Any], case_ids: Sequence[str]) -> None:
        self._cases = cases
        self.records = {case_id: records[case_id] for case_id in case_ids}
        self._tile_order = tile_order
        self.case_ids = list(case_ids)
        self.identity = {
            "mode": "smoke",
            "feature_id": "tiny-feature-id",
            "bundle_id": "tiny-bundle-id",
            "split_id": "tiny-split-id",
            "source_manifest_sha256": "f" * 64,
        }

    def load_case(self, case_id: str) -> Any:
        if case_id not in self.records:
            raise ValueError("Case is outside this dataset view.")
        return self._cases[case_id]

    def loaded_tile_order(self, case_id: str) -> dict[str, list[str]]:
        return {lens: list(values) for lens, values in self._tile_order[case_id].items()}


def _tiny_study() -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_case_ids = ("n0", "n1", "n2", "p0", "p1", "n3", "p2")
    labels = (0, 0, 0, 1, 1, 0, 1)
    outer_train = ("n0", "n1", "n2", "p0", "p1")
    inner_train = ("n0", "n1", "p0", "p1")
    records = {
        case_id: {"patient_id": f"patient-{index}", "target": labels[index]}
        for index, case_id in enumerate(all_case_ids)
    }
    cases: dict[str, Any] = {}
    tile_order = {case_id: {lens: [] for lens in LENSES} for case_id in all_case_ids}
    source_rows: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(5521)
    pca_components = np.zeros((32, FEATURE_DIM), dtype=np.float32)
    pca_components[:, :32] = np.eye(32, dtype=np.float32)
    pcas: dict[str, WeightedPCA] = {}
    samples: dict[str, list[np.ndarray]] = {lens: [] for lens in LENSES}
    sample_refs: dict[str, list[dict[str, Any]]] = {lens: [] for lens in LENSES}
    for lens in LENSES:
        pcas[lens] = WeightedPCA(
            mean_=np.zeros(FEATURE_DIM, dtype=np.float32),
            components_=pca_components.copy(),
            explained_variance_ratio_=np.ones(32, dtype=np.float32) / 32,
            n_components_=32,
        )
        for case_index, case_id in enumerate(all_case_ids):
            count = 64 if case_id in {"n0", "n1"} else 8
            matrix = rng.normal(size=(count, FEATURE_DIM)).astype(np.float32)
            if labels[case_index] == 1:
                matrix[:, 0] += np.float32(0.5)
            if case_id not in cases:
                cases[case_id] = SimpleNamespace(
                    case_id=case_id,
                    patient_id=records[case_id]["patient_id"],
                    target=labels[case_index],
                    features={key: np.empty((0, FEATURE_DIM), dtype=np.float32) for key in LENSES},
                    presence_mask=(True, True, True),
                )
            cases[case_id].features[lens] = matrix
            projected = pcas[lens].transform(normalize_rows(matrix))
            photo_ids = [f"{case_id}-{lens}-photo-{row % 4}" for row in range(count)]
            for row_index, photo_id in enumerate(photo_ids):
                tile_id = f"{case_id}-{lens}-tile-{row_index}"
                tile_order[case_id][lens].append(tile_id)
                source_rows[tile_id] = {"case_id": case_id, "lens": lens, "photo_id": photo_id,
                                        "tile_id": tile_id}
            if case_id in inner_train:
                samples[lens].extend(projected)
                sample_refs[lens].extend(
                    {"case_id": case_id, "lens": lens, "photo_id": photo_id,
                     "tile_id": f"{case_id}-{lens}-tile-{row_index}",
                     "source_case_label": labels[case_index]}
                    for row_index, photo_id in enumerate(photo_ids)
                )
    dataset_all = TinyDataset(cases, records, tile_order, all_case_ids)

    def dataset_factory(ids: Sequence[str]) -> TinyDataset:
        return TinyDataset(cases, records, tile_order, ids)

    transforms = {
        "members": {case_id: records[case_id] for case_id in inner_train},
        "seed": 42,
        "dimensions": 32,
        "lenses": {
            lens: {
                "pca": pcas[lens],
                "sample_projected": np.asarray(samples[lens], dtype=np.float32),
                "sample_refs": sample_refs[lens],
            }
            for lens in LENSES
        },
    }
    source_map = {
        "source_identity": dataset_all.identity,
        "rows": source_rows,
        "case_tile_order": tile_order,
    }
    splits = {
        "split_id": "tiny-locked-split",
        "seed": 42,
        "outer_folds": 1,
        "inner_folds": 1,
        "case_ids": list(all_case_ids),
        "outer": [{
            "fold": 0,
            "train_cases": list(outer_train),
            "test_cases": ["n3", "p2"],
            "inner_folds": [{
                "fold": 0,
                "train_cases": list(inner_train),
                "validation_cases": ["n2"],
            }],
        }],
    }
    return dataset_all, dataset_factory, transforms, source_map, splits


def _publish_inner_transform(shared_store: ArtifactStore, transforms: dict[str, Any]) -> None:
    shared_store.publish("scopes/outer_0/inner_0/transform.pkl", pickle.dumps(transforms, protocol=5))


def test_beta_selection_and_outer_epoch_median_policy() -> None:
    selected = select_beta({0.0: [0.52, 0.48], 0.5: [0.50, 0.47]})
    retained = select_beta({0.0: [0.52, 0.48], 0.5: [0.515, 0.475]})

    assert selected["selected_beta"] == 0.5
    assert selected["improvement_from_beta_0"] == pytest.approx(0.015)
    assert retained["selected_beta"] == 0.0
    assert outer_epoch_budget([[1, 3, 5], [2, 5]]) == 4
    with pytest.raises(ValueError, match="both beta"):
        select_beta({0.0: [0.5]})


def test_neural_inner_loop_pauses_without_oof_then_resumes_from_fit_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_all, dataset_factory, transforms, source_map, splits = _tiny_study()
    shared_store = ArtifactStore(tmp_path / "shared", {"study": "tiny-b1-shared"})
    b_store = ArtifactStore(tmp_path / "branch-b", {"study": "tiny-b1"})
    _publish_inner_transform(shared_store, transforms)
    output = tmp_path / "neural-output"
    deadline = time.monotonic() + 300
    real_run_fit = dual_neural_study.run_fit
    interrupted = False

    def pause_first_beta_half_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal interrupted
        fit_identity = kwargs["fit_identity"]
        if fit_identity["beta"] == 0.5 and not interrupted:
            interrupted = True
            return {"phase": "paused_steps", "epoch": 0}
        return real_run_fit(*args, **kwargs)

    monkeypatch.setattr(dual_neural_study, "run_fit", pause_first_beta_half_fit)
    paused = run_neural_study(
        dataset_all,
        dataset_factory,
        splits,
        source_map,
        shared_store,
        b_store,
        output,
        deadline,
        device="cpu",
        max_epochs=2,
        smoke=True,
    )
    assert paused["phase"] == "paused_steps"
    assert paused["completed_fits"] == 2
    assert not (output / "evaluation" / "oof_predictions.json").exists()
    assert not b_store.read("B1/neural/" + paused["study_id"] + "/smoke_result.json")

    monkeypatch.setattr(dual_neural_study, "run_fit", real_run_fit)
    completed = run_neural_study(
        dataset_all,
        dataset_factory,
        splits,
        source_map,
        shared_store,
        b_store,
        output,
        deadline,
        device="cpu",
        max_epochs=2,
        smoke=True,
    )

    assert completed["phase"] == "smoke_complete"
    assert completed["outer_refits_performed"] is False
    assert completed["oof_predictions"] == []
    assert completed["selection"]["selected_beta"] in {0.0, 0.5}
    assert completed["inner_scopes"]["0"]["beta_0"]["member_best_epochs"]
    assert not (output / "evaluation" / "oof_predictions.json").exists()
    fit_dir = output / "fits" / "outer_0" / "inner_0" / "inner" / "beta_0" / "members"
    fit_dirs = list(fit_dir.iterdir())
    assert len(fit_dirs) == 2
    for member_fit in fit_dirs:
        identity = json.loads((member_fit / "resolved_config.json").read_text(encoding="utf-8"))
        state = CheckpointStore(member_fit / "checkpoints", identity).load_latest()
        assert state["finished"] is True and state["epoch"] == 2

    rerun = run_neural_study(
        dataset_all,
        dataset_factory,
        splits,
        source_map,
        shared_store,
        b_store,
        output,
        deadline,
        device="cpu",
        max_epochs=2,
        smoke=True,
    )
    assert rerun == completed
