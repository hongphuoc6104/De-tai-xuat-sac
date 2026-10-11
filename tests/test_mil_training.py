"""End-to-end CPU contracts for bounded, resumable case-level MIL fitting."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from histology_data import mil_training
from histology_data.mil_checkpoint import CheckpointStore
from histology_data.mil_data import FixtureDataset, create_toy_fixture
from histology_data.mil_model import LENS_KEYS, MILModel, ModelConfig
from histology_data.mil_training import TrainConfig, evaluate, run_fit

FEATURE_DIM = 2048
ROWS_PER_LENS = 16
MEMORY_LIMIT_BYTES = 2 * 1024**3


class InMemoryDataset:
    """Tiny case source with the same identity/records/load contract as a bundle."""

    def __init__(
        self,
        name: str,
        labels: tuple[int, ...] = (0, 1),
        *,
        seed: int = 20261011,
        patient_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.case_ids = [f"{name}-case-{index}" for index in range(len(labels))]
        patients = patient_ids or tuple(f"{name}-patient-{index}" for index in range(len(labels)))
        if len(patients) != len(labels):
            raise ValueError("patient_ids must have one entry per case")
        self.records = {
            case_id: {"patient_id": patients[index], "target": target}
            for index, (case_id, target) in enumerate(zip(self.case_ids, labels, strict=True))
        }
        self.identity = {"mode": "smoke", "kind": "in_memory_test", "dataset_id": name}
        self.load_calls: list[str] = []
        rng = np.random.default_rng(seed)
        self._cases: dict[str, SimpleNamespace] = {}
        for index, (case_id, target) in enumerate(zip(self.case_ids, labels, strict=True)):
            features = {}
            for lens_index, lens in enumerate(LENS_KEYS):
                matrix = rng.normal(
                    loc=0.0,
                    scale=0.04,
                    size=(ROWS_PER_LENS, FEATURE_DIM),
                ).astype(np.float32)
                matrix[:, 0] += (target * 2 - 1) * 0.15 + lens_index * 0.01
                features[lens] = matrix
            self._cases[case_id] = SimpleNamespace(
                case_id=case_id,
                patient_id=patients[index],
                target=target,
                features=features,
                presence_mask=(True, True, True),
            )

    def load_case(self, case_id: str) -> SimpleNamespace:
        self.load_calls.append(case_id)
        return self._cases[case_id]


def _config(
    *,
    mode: str = "smoke",
    max_epochs: int = 2,
    max_steps: int | None = 20,
    dropout: float = 0.35,
    patience: int = 5,
    device: str = "cpu",
) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(variant="E4", dropout=dropout),
        mode=mode,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=5e-4,
        weight_decay=1e-3,
        seed=71,
        device=device,
        max_steps=max_steps,
        memory_limit_bytes=MEMORY_LIMIT_BYTES,
        num_threads=2,
        deterministic=True,
    )


def _run(
    train: InMemoryDataset,
    validation: InMemoryDataset | None,
    output: Path,
    config: TrainConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_fit(
        train,
        validation,
        output,
        config,
        fit_identity={"fit_id": "mil-training-e2e", "protocol": "case-level-smoke-v1"},
        deadline=time.monotonic() + 300,
        **kwargs,
    )


def _checkpoint_store(output: Path) -> CheckpointStore:
    identity = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    return CheckpointStore(output / "checkpoints", identity)


def _committed_epoch_numbers(output: Path) -> list[int]:
    manifests = sorted((output / "checkpoints" / "versions").glob("*/commit.json"))
    return [json.loads(path.read_text(encoding="utf-8"))["epoch"] for path in manifests]


def _assert_state_close(left: Any, right: Any, *, key: str | None = None) -> None:
    if key == "duration_seconds":
        return
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for child_key in left:
            _assert_state_close(left[child_key], right[child_key], key=str(child_key))
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_state_close(left_value, right_value)
    elif isinstance(left, float):
        assert isinstance(right, (float, int))
        assert math.isclose(left, float(right), rel_tol=1e-6, abs_tol=1e-8)
    else:
        assert left == right


def test_fit_runs_two_case_level_epochs_and_activates_dropout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = InMemoryDataset("fit-train")
    validation = InMemoryDataset("fit-valid", seed=20261012)
    assert {row["patient_id"] for row in train.records.values()}.isdisjoint(
        row["patient_id"] for row in validation.records.values()
    )
    observed_dropout: list[tuple[torch.Tensor, torch.Tensor]] = []
    instances: list[MILModel] = []
    real_model = MILModel

    def recording_model(config: ModelConfig) -> MILModel:
        model = real_model(config)
        instances.append(model)
        model.dropout.register_forward_hook(
            lambda _module, arguments, output: observed_dropout.append(
                (arguments[0].detach().clone(), output.detach().clone()),
            ),
        )
        return model

    monkeypatch.setattr(mil_training, "MILModel", recording_model)
    result = _run(train, validation, tmp_path / "fit", _config())

    assert result["phase"] == "complete"
    assert result["epoch"] == 2 and result["global_step"] == 4
    assert len(result["history"]) == 2
    assert all(row["validation_loss"] is not None for row in result["history"])
    assert len(train.load_calls) == 4
    assert len(validation.load_calls) == 4
    assert instances and instances[0].dropout.p == 0.35
    assert any(not torch.equal(before, after) for before, after in observed_dropout)


def test_stop_after_epoch_one_resume_matches_continuous_fit_including_optimizer_and_rng(
    tmp_path: Path,
) -> None:
    train = InMemoryDataset("resume-train")
    validation = InMemoryDataset("resume-valid", seed=20261012)
    config = _config()
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"

    whole_result = _run(train, validation, continuous, config)
    stop_calls = 0

    def stop_before_second_epoch() -> bool:
        nonlocal stop_calls
        stop_calls += 1
        return stop_calls == 2

    paused_result = _run(
        train,
        validation,
        split,
        config,
        stop_requested=stop_before_second_epoch,
    )
    assert paused_result["phase"] == "paused_budget"
    assert paused_result["epoch"] == 1
    assert _checkpoint_store(split).load_latest()["epoch"] == 1

    resumed_result = _run(train, validation, split, config)
    assert whole_result["phase"] == resumed_result["phase"] == "complete"
    whole_state = _checkpoint_store(continuous).load_latest()
    resumed_state = _checkpoint_store(split).load_latest()
    _assert_state_close(whole_state, resumed_state)


def test_exception_after_optimizer_step_discards_uncommitted_epoch_then_resumes(
    tmp_path: Path,
) -> None:
    train = InMemoryDataset("fault-train")
    validation = InMemoryDataset("fault-valid", seed=20261012)
    output = tmp_path / "faulted"
    injected = False

    def fail_after_first_step(stage: str, _details: dict[str, Any]) -> None:
        nonlocal injected
        if stage == "after_step" and not injected:
            injected = True
            raise RuntimeError("injected interruption after optimizer step")

    with pytest.raises(RuntimeError, match="injected interruption"):
        _run(train, validation, output, _config(), event_hook=fail_after_first_step)

    committed = _checkpoint_store(output).load_latest()
    assert committed["epoch"] == 0
    assert committed["global_step"] == 0
    assert committed["history"] == []
    assert _committed_epoch_numbers(output) == [0]
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["phase"] == "error"

    resumed = _run(train, validation, output, _config())
    assert resumed["phase"] == "complete"
    assert resumed["epoch"] == 2 and resumed["global_step"] == 4


def test_patient_leakage_is_rejected_before_any_fit_output_is_created(tmp_path: Path) -> None:
    train = InMemoryDataset("leak-train")
    validation = InMemoryDataset(
        "leak-valid",
        patient_ids=("leak-train-patient-0", "leak-valid-patient-1"),
    )
    output = tmp_path / "leak"

    with pytest.raises(ValueError, match="Patient leakage"):
        _run(train, validation, output, _config())

    assert not output.exists()
    assert train.load_calls == [] and validation.load_calls == []


def test_requested_cuda_fails_closed_without_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mil_training.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "cuda-requested"

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _run(
            InMemoryDataset("cuda-train"),
            InMemoryDataset("cuda-valid"),
            output,
            _config(device="cuda"),
        )

    assert not output.exists()


def test_nonfinite_feature_guard_preserves_only_initial_checkpoint(tmp_path: Path) -> None:
    train = InMemoryDataset("nan-train")
    validation = InMemoryDataset("nan-valid", seed=20261012)
    for case in train._cases.values():
        case.features["4"][0, 0] = np.nan
    output = tmp_path / "nan"

    with pytest.raises(FloatingPointError, match="non-finite"):
        _run(train, validation, output, _config())

    state = _checkpoint_store(output).load_latest()
    assert state["epoch"] == state["global_step"] == 0
    assert _committed_epoch_numbers(output) == [0]
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["phase"] == "error"


def test_step_budget_pauses_without_marking_an_unrun_fit_complete(tmp_path: Path) -> None:
    train = InMemoryDataset("budget-train")
    validation = InMemoryDataset("budget-valid", seed=20261012)
    output = tmp_path / "step-budget"

    result = _run(train, validation, output, _config(max_steps=1))

    assert result["phase"] == "paused_steps"
    assert result["epoch"] == result["global_step"] == 0
    assert result.get("finished") is not True
    assert _checkpoint_store(output).load_latest()["finished"] is False
    assert _committed_epoch_numbers(output) == [0]
    assert train.load_calls == [] and validation.load_calls == []


def test_outer_refit_rejects_validation_partition(tmp_path: Path) -> None:
    output = tmp_path / "refit-with-validation"

    with pytest.raises(ValueError, match="refit cannot early-stop"):
        _run(
            InMemoryDataset("refit-train"),
            InMemoryDataset("refit-valid", seed=20261012),
            output,
            _config(mode="refit", max_epochs=1),
        )

    assert not output.exists()


def test_production_inner_fit_refuses_verified_smoke_fixture(tmp_path: Path) -> None:
    fixture_path = create_toy_fixture(
        tmp_path / "fixture",
        seed=2026,
        case_count=4,
        rows_per_lens=ROWS_PER_LENS,
    )
    fixture = FixtureDataset(fixture_path)
    output = tmp_path / "production-fit"

    with pytest.raises(ValueError, match="Smoke fixtures cannot be used for production"):
        _run(fixture, fixture, output, _config(mode="inner", max_epochs=1))

    assert not output.exists()
    assert fixture.identity["mode"] == "smoke"


def test_checkpoint_reload_preserves_predictions_and_completed_rerun_does_no_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = InMemoryDataset("reload-train")
    validation = InMemoryDataset("reload-valid", seed=20261012)
    output = tmp_path / "reload"
    instances: list[MILModel] = []
    real_model = MILModel

    def capturing_model(config: ModelConfig) -> MILModel:
        model = real_model(config)
        instances.append(model)
        return model

    monkeypatch.setattr(mil_training, "MILModel", capturing_model)
    result = _run(train, validation, output, _config())
    assert result["phase"] == "complete" and len(instances) == 1

    before_reload = evaluate(instances[0], validation, "cpu")
    state = _checkpoint_store(output).load_latest()
    restored = MILModel(ModelConfig(**json.loads(
        (output / "resolved_config.json").read_text(encoding="utf-8"),
    )["config"]["model"]))
    restored.load_state_dict(state["model"])
    after_reload = evaluate(restored, validation, "cpu")
    assert before_reload["loss"] == pytest.approx(after_reload["loss"], rel=1e-7, abs=1e-8)
    assert len(before_reload["predictions"]) == len(after_reload["predictions"]) == 2
    for before, after in zip(before_reload["predictions"], after_reload["predictions"], strict=True):
        assert before["case_id"] == after["case_id"]
        assert before["patient_id"] == after["patient_id"]
        assert before["target"] == after["target"]
        assert before["probability"] == pytest.approx(after["probability"], rel=1e-7, abs=1e-8)
        assert before["logit"] == pytest.approx(after["logit"], rel=1e-7, abs=1e-8)

    prior_load_count = len(train.load_calls) + len(validation.load_calls)
    prior_commit_count = len(_committed_epoch_numbers(output))
    rerun = _run(train, validation, output, _config())

    assert rerun["phase"] == "complete" and rerun["resumed"] is True
    assert len(train.load_calls) + len(validation.load_calls) == prior_load_count
    assert len(_committed_epoch_numbers(output)) == prior_commit_count
