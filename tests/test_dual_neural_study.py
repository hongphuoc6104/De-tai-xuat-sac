"""Bounded synthetic loop/resume checks for B1 nested member orchestration."""
from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from histology_data import dual_neural_study
from histology_data.dual_common import ArtifactStore, WeightedPCA, normalize_rows
from histology_data.dual_neural_study import outer_epoch_budget, run_neural_study, select_beta
from histology_data.io import fingerprint
from histology_data.mil_checkpoint import ActiveFitLockError, CheckpointStore, FitLock

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
    output = b_store.root / "neural-output"
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

    beta0_members = output / "fits" / "outer_0" / "inner_0" / "inner" / "beta_0" / "members"
    stale_fit = sorted(beta0_members.iterdir())[0]
    stale_run = subprocess.run(
        [
            sys.executable,
            "-c",
            "from histology_data.mil_checkpoint import FitLock; import os,sys; "
            "FitLock(sys.argv[1], owner_id='killed-B1-worker').acquire(); os._exit(0)",
            str(stale_fit),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert stale_run.returncode == 0, stale_run.stdout + stale_run.stderr
    stale_metadata = json.loads((stale_fit / ".fit.lock").read_text(encoding="utf-8"))
    assert stale_metadata["pid"] != os.getpid()
    with pytest.raises(ProcessLookupError):
        os.kill(stale_metadata["pid"], 0)
    resolved_before_resume = (stale_fit / "resolved_config.json").read_bytes()

    remote_fit = sorted(beta0_members.iterdir())[1]
    remote_run = subprocess.run(
        [
            sys.executable,
            "-c",
            "from histology_data.mil_checkpoint import FitLock; import os,sys; "
            "FitLock(sys.argv[1], owner_id='remote-killed-B1-worker').acquire(); os._exit(0)",
            str(remote_fit),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert remote_run.returncode == 0, remote_run.stdout + remote_run.stderr
    remote_lock_path = remote_fit / ".fit.lock"
    remote_metadata = json.loads(remote_lock_path.read_text(encoding="utf-8"))
    remote_metadata["host"] = "stopped-remote-host"
    remote_lock_path.write_text(json.dumps(remote_metadata), encoding="utf-8")
    relative_fit_path = remote_fit.absolute().relative_to(b_store.root.absolute()).as_posix()
    recovery_evidence = {
        relative_fit_path: {
            "token": remote_metadata["token"],
            "host": remote_metadata["host"],
            "boot_id": remote_metadata["boot_id"],
            "session_id": remote_metadata["session_id"],
            "old_runtime_stopped": True,
            "evidence": "synthetic test receipt binds the stopped remote owner",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    remote_resolved_before_resume = (remote_fit / "resolved_config.json").read_bytes()

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
        recover_stale_lock=True,
        recovery_evidence_map=recovery_evidence,
    )

    assert completed["phase"] == "smoke_complete"
    assert completed["outer_refits_performed"] is False
    assert completed["oof_predictions"] == []
    assert completed["selection"]["selected_beta"] in {0.0, 0.5}
    assert completed["inner_scopes"]["0"]["beta_0"]["member_best_epochs"]
    assert not (output / "evaluation" / "oof_predictions.json").exists()
    assert not (stale_fit / ".fit.lock").exists()
    assert (stale_fit / "resolved_config.json").read_bytes() == resolved_before_resume
    assert not remote_lock_path.exists()
    assert (remote_fit / "resolved_config.json").read_bytes() == remote_resolved_before_resume
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


def test_production_fit_count_matches_beta_zero_and_selected_beta_outer_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    negative_ids = [f"n{index}" for index in range(9)]
    positive_ids = [f"p{index}" for index in range(9)]
    all_case_ids = negative_ids + positive_ids
    labels = {case_id: int(case_id.startswith("p")) for case_id in all_case_ids}
    records = {
        case_id: {"patient_id": f"patient-{case_id}", "target": labels[case_id]}
        for case_id in all_case_ids
    }
    dataset_all = SimpleNamespace(
        case_ids=all_case_ids,
        records=records,
        identity={"mode": "production", "feature_id": "fit-count-feature",
                  "bundle_id": "fit-count-bundle", "split_id": "fit-count-split"},
    )

    def dataset_factory(ids: Sequence[str]) -> Any:
        return SimpleNamespace(
            case_ids=list(ids),
            records={case_id: records[case_id] for case_id in ids},
            identity=dataset_all.identity,
        )

    outer_folds = []
    test_groups = [
        (negative_ids[0:3], positive_ids[0:3]),
        (negative_ids[3:6], positive_ids[3:6]),
        (negative_ids[6:9], positive_ids[6:9]),
    ]
    for fold, (test_negatives, test_positives) in enumerate(test_groups):
        test_ids = test_negatives + test_positives
        train_negatives = [case for case in negative_ids if case not in test_negatives]
        train_positives = [case for case in positive_ids if case not in test_positives]
        inner_train0 = train_negatives[:3] + train_positives[:3]
        inner_val0 = train_negatives[3:] + train_positives[3:]
        inner_train1 = inner_val0
        inner_val1 = inner_train0
        outer_folds.append({
            "fold": fold,
            "train_cases": train_negatives + train_positives,
            "test_cases": test_ids,
            "inner_folds": [
                {"fold": 0, "train_cases": inner_train0, "validation_cases": inner_val0},
                {"fold": 1, "train_cases": inner_train1, "validation_cases": inner_val1},
            ],
        })
    splits = {
        "split_id": "fit-count-split",
        "seed": 42,
        "outer_folds": 3,
        "inner_folds": 2,
        "case_ids": all_case_ids,
        "outer": outer_folds,
    }
    source_map = {"source_identity": dataset_all.identity}
    shared = ArtifactStore(tmp_path / "fit-count-shared", {"store": "shared"})
    branch_b = ArtifactStore(tmp_path / "fit-count-b", {"store": "branch-b"})
    empty_transform = {"dimensions": 32, "members": {}, "seed": 42, "lenses": {}}
    for outer in outer_folds:
        for inner in outer["inner_folds"]:
            scope = f"outer_{outer['fold']}/inner_{inner['fold']}"
            shared.publish(f"scopes/{scope}/transform.pkl", pickle.dumps(empty_transform, protocol=5))
        scope = f"outer_{outer['fold']}/refit"
        shared.publish(f"scopes/{scope}/transform.pkl", pickle.dumps(empty_transform, protocol=5))

    monkeypatch.setattr(dual_neural_study.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dual_neural_study.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        dual_neural_study,
        "prepare_member_states",
        lambda _dataset, _factory, _transforms, _map, ids, anchor, _store, scope, _deadline: {
            "head_train_ids": tuple(case for case in ids if case != anchor),
            "lens_states": {},
            "member_id": fingerprint({"scope": scope, "anchor": anchor}),
            "member_identity_id": fingerprint({"scope": scope, "anchor": anchor, "state": 1}),
            "anchor_token": fingerprint({"anchor": anchor}),
        },
    )
    monkeypatch.setattr(
        dual_neural_study,
        "make_novelty_model_factory",
        lambda _states, *, beta, member_id: (lambda _config: None),
    )
    monkeypatch.setattr(
        dual_neural_study,
        "novelty_model_identity",
        lambda _config, _states, *, beta, member_id: {"beta": beta, "member_id": member_id},
    )

    def fake_inner_fit(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        beta = kwargs["beta"]
        validation_ids = kwargs["validation_ids"]
        validation = dataset_factory(validation_ids)

        def probability(target: int) -> float:
            return 0.5 if beta == 0.0 else (0.1 if target == 0 else 0.9)

        predictions = [
            {"case_id": case_id, "target": records[case_id]["target"],
             "probability": probability(records[case_id]["target"])}
            for case_id in validation.case_ids
        ]
        return {"phase": "complete", "best_epoch": 1,
                "identity_id": fingerprint(kwargs["fit_identity"])}, predictions

    monkeypatch.setattr(dual_neural_study, "_fit_validation_member", fake_inner_fit)
    monkeypatch.setattr(
        dual_neural_study,
        "run_fit",
        lambda *_args, **kwargs: {"phase": "complete", "identity_id": fingerprint(kwargs["fit_identity"])},
    )
    monkeypatch.setattr(
        dual_neural_study,
        "_load_model",
        lambda _fit_dir, _config, _factory, identity, *, best: SimpleNamespace(beta=identity["beta"]),
    )

    def fake_evaluate(model: Any, dataset: Any, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        def probability(target: int) -> float:
            return 0.5 if model.beta == 0.0 else (0.1 if target == 0 else 0.9)

        return {"predictions": [
            {"case_id": case_id, "target": records[case_id]["target"],
             "probability": probability(records[case_id]["target"])}
            for case_id in dataset.case_ids
        ]}

    monkeypatch.setattr(dual_neural_study, "evaluate", fake_evaluate)
    owner_lock = FitLock(dual_neural_study._gpu_lock_path(0), owner_id="existing-B1-study")
    owner_lock.acquire()
    locked_output = tmp_path / "concurrent-output"
    try:
        with pytest.raises(ActiveFitLockError):
            run_neural_study(
                dataset_all,
                dataset_factory,
                splits,
                source_map,
                shared,
                branch_b,
                locked_output,
                time.monotonic() + 300,
                device="cuda",
                max_epochs=1,
                smoke=False,
            )
    finally:
        owner_lock.release()
    locked_status = json.loads((locked_output / "status.json").read_text(encoding="utf-8"))
    assert locked_status["phase"] == "error"
    assert locked_status["error_category"] == "concurrency"

    fit_count_output = tmp_path / "fit-count-output"
    result = run_neural_study(
        dataset_all,
        dataset_factory,
        splits,
        source_map,
        shared,
        branch_b,
        fit_count_output,
        time.monotonic() + 300,
        device="cuda",
        max_epochs=1,
        smoke=False,
    )

    assert result["phase"] == "complete"
    assert all(value == 0.5 for value in result["selected_beta_by_outer"].values())
    assert result["completed_fits"] == 72
    assert result["expected_fits"] == 72
    assert result["expected_fits_max"] == 72

    identity = json.loads((fit_count_output / "resolved_config.json").read_text(encoding="utf-8"))
    dual_neural_study.verify_neural_result(result, dataset_all, splits, identity)
    artifact_name = f"B1/neural/{result['study_id']}/final_result.json"
    corruptions = ("duplicate_case", "drop_case", "wrong_target", "wrong_patient", "wrong_metric")
    for corruption in corruptions:
        forged = copy.deepcopy(result)
        primary = forged["oof_predictions"]["B1_selected"]
        if corruption == "duplicate_case":
            primary.append(copy.deepcopy(primary[0]))
        elif corruption == "drop_case":
            primary.pop()
        elif corruption == "wrong_target":
            primary[0]["target"] = 1 - primary[0]["target"]
        elif corruption == "wrong_patient":
            primary[0]["patient_id"] = "wrong-patient"
        else:
            forged["metrics"]["B1_selected"]["pooled_oof"]["cases"] += 1

        # ArtifactStore verifies the checksum and receipt; the consumer must
        # still reject semantically corrupt but correctly committed metadata.
        corrupt_store = ArtifactStore(tmp_path / f"trusted-cache-{corruption}", branch_b.identity)
        corrupt_store.json(artifact_name, forged)
        cached_output = tmp_path / f"cached-output-{corruption}"
        with pytest.raises(ValueError):
            run_neural_study(
                dataset_all,
                dataset_factory,
                splits,
                source_map,
                shared,
                corrupt_store,
                cached_output,
                time.monotonic() + 300,
                device="cuda",
                max_epochs=1,
                smoke=False,
            )
        status = json.loads((cached_output / "status.json").read_text(encoding="utf-8"))
        assert status["phase"] == "error" and status["error_category"] == "integrity"


@pytest.mark.parametrize(
    ("failure", "category"),
    [(OSError("simulated artifact I/O failure"), "io"),
     (MemoryError("simulated allocation failure"), "resource")],
)
def test_io_and_resource_failures_write_study_error_without_erasing_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    category: str,
) -> None:
    dataset_all, dataset_factory, transforms, source_map, splits = _tiny_study()
    shared = ArtifactStore(tmp_path / "failure-shared", {"study": "failure-shared"})
    branch_b = ArtifactStore(tmp_path / "failure-b", {"study": "failure-b"})
    _publish_inner_transform(shared, transforms)
    branch_b.json("preserve/last-good.json", {"committed": True})

    def fail_member_preparation(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(dual_neural_study, "prepare_member_states", fail_member_preparation)
    output = tmp_path / "failed-neural"
    with pytest.raises(type(failure)):
        run_neural_study(
            dataset_all,
            dataset_factory,
            splits,
            source_map,
            shared,
            branch_b,
            output,
            time.monotonic() + 120,
            device="cpu",
            max_epochs=1,
            smoke=True,
        )

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "error" and status["error_category"] == category
    assert json.loads(branch_b.read("preserve/last-good.json")) == {"committed": True}
    assert not (output / "evaluation" / "oof_predictions.json").exists()
