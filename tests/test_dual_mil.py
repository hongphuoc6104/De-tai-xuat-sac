"""CPU E2E contracts for novelty-guided full-bag MIL and committed resume."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from histology_data.dual_mil import (
    NoveltyLensState,
    NoveltyMILModel,
    make_novelty_model_factory,
    novelty_model_identity,
)
from histology_data.dual_novelty import NegativePrototypeBank, RobustNoveltyScale
from histology_data.mil_checkpoint import CheckpointIdentityError, CheckpointStore
from histology_data.mil_model import MILModel, ModelConfig
from histology_data.mil_training import TrainConfig, evaluate, run_fit

FEATURE_DIM = 32


def _state(*, anchor: str = "anchor-token", zero_iqr: bool = False) -> NoveltyLensState:
    bank = NegativePrototypeBank(
        lens="4",
        anchor_case_id=anchor,
        prototypes=np.eye(32, dtype=np.float32),
        prototype_row_ids=tuple(f"prototype-row-{index}" for index in range(32)),
        prototype_photo_ids=tuple(f"source-photo-{index % 4}" for index in range(32)),
    )
    scale = RobustNoveltyScale(
        median=0.0,
        iqr=0.0 if zero_iqr else 1.0,
        epsilon=1e-6,
        degenerate=zero_iqr,
    )
    return NoveltyLensState(
        mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        components=np.eye(FEATURE_DIM, dtype=np.float32),
        bank=bank,
        scale=scale,
    )


def test_beta_zero_matches_full_bag_baseline_outputs_and_parameter_gradients() -> None:
    config = ModelConfig(variant="E4", dropout=0.0, feature_dim=FEATURE_DIM)
    torch.manual_seed(91)
    baseline = MILModel(config).eval()
    novelty = NoveltyMILModel(config, {"4": _state()}, beta=0.0, member_id="member-a").eval()
    novelty.core.load_state_dict(baseline.state_dict())
    features = {"4": torch.rand(6, FEATURE_DIM, dtype=torch.float32)}

    baseline_output = baseline(features, presence_mask=(True, False, False))
    novelty_output = novelty(features, presence_mask=(True, False, False))
    torch.testing.assert_close(novelty_output["logit"], baseline_output["logit"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        novelty_output["attention"]["4"], baseline_output["attention"]["4"], rtol=0.0, atol=0.0,
    )
    assert novelty_output["attention"]["4"].shape == (6,)
    assert novelty_output["novelty_z"]["4"].shape == (6,)
    assert torch.isfinite(novelty_output["novelty_z"]["4"]).all()

    baseline_output["logit"].backward()
    novelty_output["logit"].backward()
    novelty_parameters = dict(novelty.core.named_parameters())
    for name, parameter in baseline.named_parameters():
        novelty_gradient = novelty_parameters[name].grad
        assert (parameter.grad is None) == (novelty_gradient is None)
        if parameter.grad is not None:
            assert novelty_gradient is not None
            torch.testing.assert_close(parameter.grad, novelty_gradient, rtol=0.0, atol=0.0)


def test_positive_beta_biases_attention_over_the_complete_bag() -> None:
    config = ModelConfig(variant="E4", dropout=0.0, feature_dim=FEATURE_DIM)
    model = NoveltyMILModel(config, {"4": _state()}, beta=0.5, member_id="member-a").eval()
    with torch.no_grad():
        for parameter in model.core.attention_heads["4"].parameters():
            parameter.zero_()
    features = {"4": torch.zeros(3, FEATURE_DIM, dtype=torch.float32)}
    features["4"][0, 0] = 1.0
    features["4"][1, 1] = 1.0
    features["4"][2, 0] = -1.0

    output = model(features, presence_mask=(True, False, False))

    weights = output["attention"]["4"]
    novelty = output["novelty_z"]["4"]
    assert weights.shape == (3,)
    assert weights.sum().item() == pytest.approx(1.0)
    assert novelty.shape == (3,)
    assert torch.all((novelty >= -3.0) & (novelty <= 3.0))
    assert weights[2] > weights[0] and weights[2] > weights[1]


def test_zero_iqr_stays_finite_and_zero_norm_input_fails_closed() -> None:
    config = ModelConfig(variant="E4", dropout=0.0, feature_dim=FEATURE_DIM)
    model = NoveltyMILModel(config, {"4": _state(zero_iqr=True)}, beta=0.5, member_id="member-a")
    features = {"4": torch.ones(2, FEATURE_DIM, dtype=torch.float32)}
    output = model(features, presence_mask=(True, False, False))
    torch.testing.assert_close(output["novelty_z"]["4"], torch.zeros(2))
    assert torch.isfinite(output["logit"])

    with pytest.raises(ValueError, match="Zero-norm"):
        model({"4": torch.zeros(1, FEATURE_DIM)}, presence_mask=(True, False, False))


class TinyNoveltyDataset:
    """Small case-level source for trainer checkpoint/resume integration."""

    def __init__(self, name: str, labels: tuple[int, ...], seed: int) -> None:
        self.case_ids = [f"{name}-case-{index}" for index in range(len(labels))]
        self.records = {
            case_id: {"patient_id": f"{name}-patient-{index}", "target": label}
            for index, (case_id, label) in enumerate(zip(self.case_ids, labels, strict=True))
        }
        self.identity = {"mode": "smoke", "kind": "dual_b_in_memory", "dataset_id": name}
        rng = np.random.default_rng(seed)
        self._cases: dict[str, Any] = {}
        for index, (case_id, label) in enumerate(zip(self.case_ids, labels, strict=True)):
            matrix = rng.normal(0.0, 0.2, size=(5, FEATURE_DIM)).astype(np.float32)
            matrix[:, 0] += 0.25 if label else -0.1
            self._cases[case_id] = SimpleNamespace(
                case_id=case_id,
                patient_id=self.records[case_id]["patient_id"],
                target=label,
                features={"4": matrix},
                presence_mask=(True, False, False),
            )

    def load_case(self, case_id: str) -> Any:
        return self._cases[case_id]


def _train_config() -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(variant="E4", dropout=0.1, feature_dim=FEATURE_DIM),
        mode="smoke",
        max_epochs=2,
        patience=5,
        learning_rate=5e-4,
        weight_decay=1e-3,
        seed=103,
        device="cpu",
        max_steps=20,
        memory_limit_bytes=2 * 1024**3,
        num_threads=1,
        deterministic=True,
    )


def test_custom_model_identity_save_resume_and_predict_e2e(tmp_path: Path) -> None:
    config = _train_config()
    lens_states = {"4": _state()}
    factory = make_novelty_model_factory(lens_states, beta=0.5, member_id="member-a")
    model_identity = novelty_model_identity(config.model, lens_states, beta=0.5, member_id="member-a")
    train = TinyNoveltyDataset("train", (0, 0, 1, 1), seed=33)
    validation = TinyNoveltyDataset("validation", (0, 1), seed=34)
    fit_identity = {"fit_id": "dual-b-tiny-e2e", "protocol": "member-anchor-exclusion-v1"}
    output = tmp_path / "member-fit"

    stop_checks = 0

    def pause_after_first_epoch() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks == 2

    paused = run_fit(
        train,
        validation,
        output,
        config,
        fit_identity=fit_identity,
        deadline=time.monotonic() + 300,
        stop_requested=pause_after_first_epoch,
        model_factory=factory,
        model_identity=model_identity,
    )
    assert paused["phase"] == "paused_budget" and paused["epoch"] == 1
    identity_path = output / "resolved_config.json"
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    assert identity["model_identity"] == model_identity
    assert identity["source_fingerprint"]
    store = CheckpointStore(output / "checkpoints", identity)
    assert store.load_latest()["epoch"] == 1

    completed = run_fit(
        train,
        validation,
        output,
        config,
        fit_identity=fit_identity,
        deadline=time.monotonic() + 300,
        model_factory=factory,
        model_identity=model_identity,
    )
    assert completed["phase"] == "complete" and completed["epoch"] == 2
    loaded_state = store.load_latest()
    assert loaded_state["epoch"] == 2 and loaded_state["finished"]
    restored_model = factory(config).eval()
    restored_model.load_state_dict(loaded_state["model"])
    case = validation.load_case(validation.case_ids[1])
    tensors = {lens: torch.from_numpy(value.copy()) for lens, value in case.features.items()}
    with torch.no_grad():
        restored_output = restored_model(tensors, presence_mask=case.presence_mask)
        probability = torch.sigmoid(restored_output["logit"])
    assert torch.isfinite(probability) and 0.0 <= probability.item() <= 1.0
    assert torch.isfinite(restored_output["attention"]["4"]).all()

    beta_zero_factory = make_novelty_model_factory({"4": _state()}, beta=0.0, member_id="member-a")
    beta_zero_identity = novelty_model_identity(
        config.model, {"4": _state()}, beta=0.0, member_id="member-a",
    )
    with pytest.raises(CheckpointIdentityError, match="another config/data/fold/environment"):
        run_fit(
            train,
            validation,
            output,
            config,
            fit_identity=fit_identity,
            deadline=time.monotonic() + 300,
            model_factory=beta_zero_factory,
            model_identity=beta_zero_identity,
        )
    assert identity_path.read_bytes() == identity_bytes


def test_trainer_rejects_member_anchor_before_creating_output(tmp_path: Path) -> None:
    config = _train_config()
    lens_states = {"4": _state()}
    factory = make_novelty_model_factory(lens_states, beta=0.5, member_id="member-a")
    model_identity = novelty_model_identity(config.model, lens_states, beta=0.5, member_id="member-a")
    train = TinyNoveltyDataset("anchor-train", (0, 0, 1, 1), seed=40)
    previous_id = train.case_ids[0]
    case = train._cases.pop(previous_id)
    record = train.records.pop(previous_id)
    train.case_ids[0] = "anchor-token"
    train.records["anchor-token"] = record
    case.case_id = "anchor-token"
    train._cases["anchor-token"] = case
    validation = TinyNoveltyDataset("anchor-validation", (0, 1), seed=41)
    output = tmp_path / "rejected-anchor-fit"

    with pytest.raises(ValueError, match="excluded from supervised head training"):
        run_fit(
            train,
            validation,
            output,
            config,
            fit_identity={"fit_id": "dual-b-anchor-rejection"},
            deadline=time.monotonic() + 300,
            model_factory=factory,
            model_identity=model_identity,
        )
    assert not output.exists()


def test_evaluator_rejects_bank_anchor_as_query_case() -> None:
    config = _train_config()
    model = make_novelty_model_factory(
        {"4": _state()}, beta=0.5, member_id="member-a",
    )(config)
    query = TinyNoveltyDataset("anchor-query", (0, 1), seed=50)
    previous_id = query.case_ids[0]
    case = query._cases.pop(previous_id)
    record = query.records.pop(previous_id)
    query.case_ids[0] = "anchor-token"
    query.records["anchor-token"] = record
    case.case_id = "anchor-token"
    query._cases["anchor-token"] = case

    with pytest.raises(ValueError, match="cannot be self-scored"):
        evaluate(model, query, "cpu")
