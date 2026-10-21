"""Synthetic before-fix reproduction for patient weights and epoch selection."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

PROOF_PATH = Path(__file__).with_name("image_finetune_pre_fix.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from histology_data import image_classification_studies as studies, image_finetune as finetune  # noqa: E402


class TinyTenImageDataset:
    """Ten constant whole-frame pixels, with unequal image counts per patient-class."""

    def __init__(self) -> None:
        self.labels = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
        self.frames = [torch.full((3, 8, 8), (index + 1) / 20.0) for index in range(10)]
        self.last_index: int | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        self.last_index = index
        return self.frames[index], self.labels[index], f"proof-image-{index}"


class ScriptedValidationTinyCNN(nn.Module):
    """Tiny injected model with deliberately scripted validation probabilities."""

    def __init__(self, train_images_per_epoch: int, dataset: TinyTenImageDataset) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)
        self.train_images_per_epoch = train_images_per_epoch
        self.dataset = dataset
        self.training_forward_count = 0
        self.validation_probabilities = (
            (0.49, 0.50, 0.51, 0.52),  # AUROC 1.00, BCE about 0.693.
            (0.001, 0.60, 0.40, 0.999),  # AUROC 0.75, lower BCE about 0.459.
            (0.50, 0.50, 0.50, 0.50),  # AUROC 0.50, BCE about 0.693.
        )
        self.validation_trace: list[list[tuple[int, float]]] = [[], [], []]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.training_forward_count += len(frames)
            return self.fc(frames.mean(dim=(2, 3)))
        epoch = min(
            max(self.training_forward_count // self.train_images_per_epoch - 1, 0),
            len(self.validation_probabilities) - 1,
        )
        recovered = frames[:, 0].mean(dim=(1, 2)) * 0.229 + 0.485
        indices = torch.round(recovered * 20).to(torch.int64) - 1
        local_indices = indices - 6
        probabilities = torch.tensor(self.validation_probabilities[epoch], dtype=torch.float32)
        selected = probabilities[local_indices]
        self.validation_trace[epoch].extend(
            (int(index), float(probability))
            for index, probability in zip(local_indices.tolist(), selected.tolist(), strict=True)
        )
        return torch.logit(selected).reshape(-1, 1)


def _row(index: int, target: int, patient: str) -> dict[str, Any]:
    return {
        "image_id": f"proof-image-{index}",
        "patient_id": patient,
        "image_target": target,
        "objective_lens": 4,
        "source_archive_sha256": hashlib.sha256(b"synthetic-4x-zip").hexdigest(),
        "original_image_sha256": hashlib.sha256(f"synthetic-original-{index}".encode()).hexdigest(),
        "raw_glade": "0" if target == 0 else "3",
    }


def _run_reproduction() -> dict[str, Any]:
    dataset = TinyTenImageDataset()
    rows = [
        _row(0, 0, "negative-multi"),
        _row(1, 0, "negative-multi"),
        _row(2, 0, "negative-single"),
        _row(3, 1, "positive-multi"),
        _row(4, 1, "positive-multi"),
        _row(5, 1, "positive-single"),
        _row(6, 0, "validation-negative-a"),
        _row(7, 0, "validation-negative-b"),
        _row(8, 1, "validation-positive-a"),
        _row(9, 1, "validation-positive-b"),
    ]
    labels = np.asarray([rows[index]["image_target"] for index in range(6)], dtype=np.int64)
    patients = np.asarray([rows[index]["patient_id"] for index in range(6)], dtype=object)
    expected_patient_weights = studies.sample_weights(labels, patients, "patient_class")

    original_loss_class = finetune.nn.BCEWithLogitsLoss
    observed_loss_calls: list[dict[str, Any]] = []

    class RecordingBCE:
        def __init__(self, *, pos_weight: torch.Tensor, reduction: str) -> None:
            self.pos_weight = pos_weight.detach().cpu().tolist()
            self.reduction = reduction
            self.inner = original_loss_class(pos_weight=pos_weight, reduction=reduction)

        def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if model.training and dataset.last_index is not None and dataset.last_index < 6:
                target = int(targets.item())
                multiplier = self.pos_weight[0] if target == 1 else 1.0
                observed_loss_calls.append({
                    "image_id": f"proof-image-{dataset.last_index}",
                    "patient_id": rows[dataset.last_index]["patient_id"],
                    "target": target,
                    "pos_weight": self.pos_weight,
                    "reduction": self.reduction,
                    "effective_image_multiplier": multiplier,
                })
            return self.inner(logits, targets)

    finetune.nn.BCEWithLogitsLoss = RecordingBCE
    try:
        torch.manual_seed(20261014)
        model = ScriptedValidationTinyCNN(train_images_per_epoch=6, dataset=dataset)
        config = finetune.ImageFinetuneConfig(
            mode="F0", image_size=8, batch_size=1, accumulation_steps=1,
            max_epochs=3, patience=3, seed=17,
            augment_whole_frame=False, amp=False,
            budget_minutes=10, reserve_minutes=0,
        )
        with tempfile.TemporaryDirectory(prefix="finetune-before-fix-") as temporary_root:
            result = finetune.train_one_image_fold(
                model, dataset, rows,
                [f"proof-image-{index}" for index in range(6)],
                [f"proof-image-{index}" for index in range(6, 10)],
                config,
                Path(temporary_root) / "fold",
                source_binding={"binding_id": "pre-fix-synthetic-binding"},
                device="cpu",
            )
            val_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
            validation_auroc = [
                float(roc_auc_score(val_labels, [probability for _, probability in trace]))
                for trace in model.validation_trace
            ]
            losses = [float(row["validation_loss"]) for row in result["history"]]
            expected_patient_mass = defaultdict(float)
            for index, weight in enumerate(expected_patient_weights):
                expected_patient_mass[rows[index]["patient_id"]] += float(weight)
            observed_patient_mass_per_epoch = defaultdict(float)
            for sample in observed_loss_calls:
                observed_patient_mass_per_epoch[sample["patient_id"]] += sample["effective_image_multiplier"] / 3
            proof = {
                "phase": "before_fix",
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "training_image_count": len(labels),
                "validation_image_count": len(val_labels),
                "train_patient_image_counts": {
                    patient: dict(Counter(rows[index]["image_target"] for index in range(6)
                                          if rows[index]["patient_id"] == patient))
                    for patient in sorted(set(patients.tolist()))
                },
                "expected_patient_class_weight_scheme": {
                    "name": "studies.sample_weights(patient_class), mass N",
                    "image_weights": {
                        rows[index]["image_id"]: float(weight)
                        for index, weight in enumerate(expected_patient_weights)
                    },
                    "patient_mass_per_epoch": dict(expected_patient_mass),
                    "total_mass": float(expected_patient_weights.sum()),
                },
                "observed_train_loop_loss_objective": {
                    "status_policy": result["status"]["class_weight_policy"],
                    "patient_class_counts": result["status"]["class_weight_basis"],
                    "distinct_training_loss_calls": len(observed_loss_calls),
                    "first_calls": observed_loss_calls[:6],
                    "effective_patient_mass_per_epoch": dict(observed_patient_mass_per_epoch),
                },
                "epoch_selection": {
                    "current_policy": result["selection_policy"],
                    "validation_loss_by_epoch": losses,
                    "deliberate_validation_auroc_by_epoch": validation_auroc,
                    "selected_best_epoch": result["best_epoch"],
                    "best_validation_loss_epoch": int(np.argmin(losses)) + 1,
                    "best_validation_auroc_epoch": int(np.argmax(validation_auroc)) + 1,
                },
            }
            assert len(observed_loss_calls) == 18
            assert set(sample["effective_image_multiplier"] for sample in observed_loss_calls) == {1.0}
            assert expected_patient_mass["negative-multi"] == expected_patient_mass["negative-single"]
            assert expected_patient_mass["positive-multi"] == expected_patient_mass["positive-single"]
            assert result["best_epoch"] == int(np.argmin(losses)) + 1 == 2
            assert int(np.argmax(validation_auroc)) + 1 == 1
            assert validation_auroc == [1.0, 0.75, 0.5]
            return proof
    finally:
        finetune.nn.BCEWithLogitsLoss = original_loss_class


if __name__ == "__main__":
    result = _run_reproduction()
    PROOF_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
