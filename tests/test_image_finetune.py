"""Synthetic whole-frame tests for the conditional F0/F1 fold trainer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from histology_data import image_finetune as finetune
from histology_data.image_finetune import ImageFinetuneConfig, train_one_image_fold
from histology_data.mil_checkpoint import (
    CheckpointIdentityError,
    CheckpointIntegrityError,
    CheckpointStore,
)


class TinyWholeFrameCNN(nn.Module):
    """Small stem/layer1-4/fc network with BatchNorm for mask and resume tests."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        values = self.stem(frames)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        return self.fc(self.pool(values).flatten(1))


class TinyFrameDataset:
    def __init__(self) -> None:
        self.labels = [0, 1, 0, 1, 0, 1, 0, 1]
        self.forbidden_indices: set[int] = set()
        self.loaded_indices: list[int] = []
        self.frames = []
        for index, label in enumerate(self.labels):
            generator = torch.Generator().manual_seed(1000 + index)
            noise = torch.rand((3, 13, 17), generator=generator) * 0.15
            signal = torch.zeros((3, 13, 17))
            signal[label] = 0.65
            self.frames.append((signal + noise).clamp(0, 1))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        if index in self.forbidden_indices:
            raise AssertionError(f"Outer-test frame was loaded: {index}")
        self.loaded_indices.append(index)
        return self.frames[index], self.labels[index], f"image-{index}"


def _rows() -> list[dict[str, Any]]:
    rows = []
    for index, target in enumerate((0, 1, 0, 1, 0, 1, 0, 1)):
        rows.append({
            "image_id": f"image-{index}",
            "patient_id": f"patient-{index}",
            "image_target": target,
            "objective_lens": (4, 10, 40)[index % 3],
            "source_archive_sha256": hashlib.sha256(f"archive-{index % 3}".encode()).hexdigest(),
            "original_image_sha256": hashlib.sha256(f"original-{index}".encode()).hexdigest(),
            "raw_glade": str(0 if target == 0 else 3),
        })
    return rows


def _model(seed: int = 800) -> TinyWholeFrameCNN:
    torch.manual_seed(seed)
    return TinyWholeFrameCNN()


def _config(mode: str = "F0", **kwargs: Any) -> ImageFinetuneConfig:
    values = {
        "mode": mode,
        "image_size": 16,
        "batch_size": 1,
        "accumulation_steps": 2,
        "max_epochs": 3,
        "patience": 3,
        "seed": 212,
        "checkpoint_interval_steps": 100,
        "augment_whole_frame": True,
        "amp": False,
        "budget_minutes": 10,
        "reserve_minutes": 0,
    }
    values.update(kwargs)
    return ImageFinetuneConfig(**values)


def _train(
    model: TinyWholeFrameCNN,
    dataset: TinyFrameDataset,
    output_root: Path,
    *,
    config: ImageFinetuneConfig | None = None,
    train_ids: list[str] | None = None,
    validation_ids: list[str] | None = None,
    rows: list[dict[str, Any]] | None = None,
    source_binding: dict[str, Any] | None = None,
    outer_test_image_ids: list[str] | None = None,
    max_optimizer_steps_per_call: int | None = None,
) -> dict[str, Any]:
    return train_one_image_fold(
        model,
        dataset,
        rows or _rows(),
        train_ids if train_ids is not None else [f"image-{index}" for index in range(6)],
        validation_ids if validation_ids is not None else ["image-6", "image-7"],
        config or _config(),
        output_root,
        source_binding=source_binding or {"binding_id": "synthetic-source-binding"},
        outer_test_image_ids=outer_test_image_ids or [],
        device="cpu",
        max_optimizer_steps_per_call=max_optimizer_steps_per_call,
    )


def _checkpoint_state(result: dict[str, Any]) -> dict[str, Any]:
    checkpoint_root = Path(result["checkpoint_root"])
    commit_paths = sorted((checkpoint_root / "versions").glob("*/commit.json"))
    assert len(commit_paths) >= 2
    manifest = json.loads(commit_paths[-1].read_text(encoding="utf-8"))
    return CheckpointStore(checkpoint_root, manifest["identity"]).load_latest()


def _assert_nested_equal(left: Any, right: Any) -> None:
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    elif hasattr(left, "shape") and hasattr(right, "shape"):
        assert (left == right).all()
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["F0", "F1"])
def test_synthetic_fold_updates_only_approved_parameters_and_freezes_batch_norm(
    tmp_path: Path, mode: str,
) -> None:
    dataset = TinyFrameDataset()
    model = _model()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    result = _train(
        model, dataset, tmp_path / mode,
        config=_config(mode, max_epochs=1, augment_whole_frame=True),
    )
    assert result["status"]["state"] == "complete"
    assert result["ready_for_fit"] is True
    assert result["finite_gradient_checks"] > 0
    assert result["status"]["class_weight_basis"] == {
        "negative_class_patients": 3,
        "positive_class_patients": 3,
    }
    state = model.state_dict()
    updated = {key for key in state if not torch.equal(initial[key], state[key])}
    assert any(key.startswith("fc.") for key in updated)
    if mode == "F0":
        assert updated <= {key for key in state if key.startswith("fc.")}
        assert set(result["trainable_parameter_names"]) == {"fc.weight", "fc.bias"}
    else:
        assert any(key.startswith("layer4.0.") for key in updated)
        assert all(key.startswith(("layer4.0.", "fc.")) for key in updated)
        assert set(result["trainable_parameter_names"]) == {
            "layer4.0.weight", "fc.weight", "fc.bias",
        }
    for key in state:
        if ".1." in key and any(name in key for name in ("stem", "layer1", "layer4")):
            assert torch.equal(initial[key], state[key])
    assert result["status"]["checkpoint_count"] >= 2
    state = _checkpoint_state(result)
    expected_rates = [1e-3] if mode == "F0" else [1e-5, 1e-3]
    assert [group["lr"] for group in state["optimizer_state"]["param_groups"]] == expected_rates


def test_interruption_resume_matches_uninterrupted_model_optimizer_scheduler_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TinyFrameDataset()
    initial = _model()
    initial_state = copy.deepcopy(initial.state_dict())
    uninterrupted_model = _model()
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted = _train(uninterrupted_model, dataset, tmp_path / "uninterrupted")
    assert uninterrupted["status"]["state"] == "complete"

    interrupted_model = _model()
    interrupted_model.load_state_dict(initial_state)
    fake_clock = iter([0.0, 0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(finetune.time, "monotonic", lambda: next(fake_clock, 1.0))
    partial = _train(
        interrupted_model, dataset, tmp_path / "resumed",
        config=_config(budget_minutes=0.001),
    )
    monkeypatch.undo()
    assert partial["status"]["state"] == "partial"
    assert partial["ready_for_fit"] is False
    assert partial["status"]["stop_reason"] == "budget"
    assert partial["global_step"] == 1
    partial_state = _checkpoint_state(partial)
    assert partial_state["train_cursor"] == 2
    assert partial_state["epoch"] == 0
    assert partial_state["optimizer_state"]
    assert partial_state["scheduler_state"]
    assert partial_state["scaler_state"] == {}
    assert partial_state["rng"]["python"]
    assert partial_state["rng"]["numpy"]
    assert partial_state["rng"]["torch_cpu"].numel() > 0
    assert partial_state["rng"]["shuffle_generator"].numel() > 0

    resumed_model = _model()
    resumed = _train(resumed_model, dataset, tmp_path / "resumed")
    assert resumed["status"]["state"] == "complete"
    assert resumed["global_step"] == uninterrupted["global_step"]
    _assert_nested_equal(uninterrupted_model.state_dict(), resumed_model.state_dict())
    uninterrupted_state = _checkpoint_state(uninterrupted)
    resumed_state = _checkpoint_state(resumed)
    for key in (
        "model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng",
        "train_permutation", "shuffle_generator_state", "history", "global_step",
        "epoch", "best_metric", "best_epoch", "patience_count",
    ):
        _assert_nested_equal(uninterrupted_state[key], resumed_state[key])
    frames = torch.stack([
        finetune._frame_to_tensor(dataset[index][0], image_size=16)
        for index in range(len(dataset))
    ])
    uninterrupted_model.eval()
    resumed_model.eval()
    with torch.inference_mode():
        uninterrupted_probabilities = torch.sigmoid(uninterrupted_model(frames)).flatten()
        resumed_probabilities = torch.sigmoid(resumed_model(frames)).flatten()
    assert torch.equal(uninterrupted_probabilities, resumed_probabilities)


@pytest.mark.parametrize("mismatch", ["split", "source", "model"])
def test_resume_rejects_changed_split_or_source_binding(tmp_path: Path, mismatch: str) -> None:
    dataset = TinyFrameDataset()
    _train(_model(), dataset, tmp_path / "fold", config=_config(max_epochs=1))
    options: dict[str, Any] = {}
    if mismatch == "split":
        options["train_ids"] = [f"image-{index}" for index in range(5)]
        options["validation_ids"] = ["image-5", "image-6", "image-7"]
    else:
        options["source_binding"] = {"binding_id": "different-source"}
    resume_model = _model(seed=801) if mismatch == "model" else _model()
    with pytest.raises(CheckpointIdentityError, match="identity mismatch"):
        _train(resume_model, dataset, tmp_path / "fold", **options)


def test_resume_rejects_tampered_latest_payload_without_fallback(tmp_path: Path) -> None:
    dataset = TinyFrameDataset()
    result = _train(_model(), dataset, tmp_path / "fold", config=_config(max_epochs=1))
    pointer = json.loads((Path(result["checkpoint_root"]) / "latest.json").read_text(encoding="utf-8"))
    payload = Path(result["checkpoint_root"]) / "versions" / pointer["version"] / "state.pt"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        _train(_model(), dataset, tmp_path / "fold", config=_config(max_epochs=1))


def test_fold_rejects_patient_overlap_before_creating_checkpoint(tmp_path: Path) -> None:
    dataset = TinyFrameDataset()
    rows = _rows()
    rows[5]["patient_id"] = rows[4]["patient_id"]
    with pytest.raises(ValueError, match="Patient-level leakage"):
        _train(
            _model(), dataset, tmp_path / "leak",
            train_ids=["image-0", "image-1", "image-2", "image-3", "image-4"],
            validation_ids=["image-5", "image-6", "image-7"],
            rows=rows,
        )
    assert not (tmp_path / "leak" / "checkpoints").exists()


def test_fold_excludes_explicit_outer_test_images_from_training_and_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Outer-test images must be excluded"):
        _train(
            _model(), TinyFrameDataset(), tmp_path / "outer-leak",
            validation_ids=["image-6", "image-7"],
            outer_test_image_ids=["image-7"],
        )
    assert not (tmp_path / "outer-leak" / "checkpoints").exists()


def test_fixed_epoch_outer_refit_never_reads_outer_test_or_uses_validation(tmp_path: Path) -> None:
    dataset = TinyFrameDataset()
    dataset.forbidden_indices = {6, 7}
    result = _train(
        _model(), dataset, tmp_path / "outer-refit",
        config=_config(fixed_epoch_count=2),
        train_ids=[f"image-{index}" for index in range(6)],
        validation_ids=[],
        outer_test_image_ids=["image-6", "image-7"],
    )
    assert result["status"]["state"] == "complete"
    assert result["ready_for_fit"] is True
    assert result["epoch_completed"] == 2
    assert result["best_epoch"] == 2
    assert result["best_validation_loss"] is None
    assert result["selection_policy"] == "fixed_epoch_count_no_validation"
    assert len(result["history"]) == 2
    assert all(row["validation_loss"] is None and row["validation_accuracy"] is None
               for row in result["history"])
    assert set(dataset.loaded_indices) == set(range(6))
    changed_outer_rows = _rows()
    changed_outer_rows[6]["image_target"] = 1
    changed_outer_rows[6]["raw_glade"] = "3"
    replay = _train(
        _model(), dataset, tmp_path / "outer-refit",
        config=_config(fixed_epoch_count=2),
        train_ids=[f"image-{index}" for index in range(6)],
        validation_ids=[],
        outer_test_image_ids=["image-6", "image-7"],
        rows=changed_outer_rows,
    )
    assert replay["fit_id"] == result["fit_id"]
    assert set(dataset.loaded_indices) == set(range(6))
