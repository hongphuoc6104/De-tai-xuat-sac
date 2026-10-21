"""Same synthetic whole-image E2E after patient-class/AUROC corrections."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

PROOF_PATH = Path(__file__).with_name("image_finetune_post_fix.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from histology_data import image_classification_studies as studies, image_finetune as finetune  # noqa: E402
from histology_data.mil_checkpoint import CheckpointStore  # noqa: E402


class TinyTenImageDataset:
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
    def __init__(self, dataset: TinyTenImageDataset) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)
        self.dataset = dataset
        self.training_forward_count = 0
        self.validation_probabilities = (
            (0.49, 0.50, 0.51, 0.52),
            (0.001, 0.60, 0.40, 0.999),
            (0.50, 0.50, 0.50, 0.50),
        )
        self.validation_trace: list[list[tuple[int, float]]] = [[], [], []]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.training_forward_count += len(frames)
            return self.fc(frames.mean(dim=(2, 3)))
        epoch = min(max(self.training_forward_count // 6 - 1, 0), 2)
        recovered = frames[:, 0].mean(dim=(1, 2)) * 0.229 + 0.485
        local_indices = torch.round(recovered * 20).to(torch.int64) - 1 - 6
        probabilities = torch.tensor(self.validation_probabilities[epoch], dtype=torch.float32)[local_indices]
        self.validation_trace[epoch].extend(
            (int(index), float(probability))
            for index, probability in zip(local_indices.tolist(), probabilities.tolist(), strict=True)
        )
        return torch.logit(probabilities).reshape(-1, 1)


def _rows() -> list[dict[str, Any]]:
    patients = [
        "negative-multi", "negative-multi", "negative-single",
        "positive-multi", "positive-multi", "positive-single",
        "validation-negative-a", "validation-negative-b",
        "validation-positive-a", "validation-positive-b",
    ]
    labels = (0, 0, 0, 1, 1, 1, 0, 0, 1, 1)
    return [{
        "image_id": f"proof-image-{index}",
        "patient_id": patients[index],
        "image_target": target,
        "objective_lens": 4,
        "source_archive_sha256": hashlib.sha256(b"synthetic-4x-zip").hexdigest(),
        "original_image_sha256": hashlib.sha256(f"synthetic-original-{index}".encode()).hexdigest(),
        "raw_glade": "0" if target == 0 else "3",
    } for index, target in enumerate(labels)]


def _run_reproduction() -> dict[str, Any]:
    dataset = TinyTenImageDataset()
    rows = _rows()
    train_ids = [f"proof-image-{index}" for index in range(6)]
    train_labels = np.asarray([rows[index]["image_target"] for index in range(6)], dtype=np.int64)
    train_patients = np.asarray([rows[index]["patient_id"] for index in range(6)], dtype=object)
    expected_weights = studies.sample_weights(train_labels, train_patients, "patient_class")
    observed: dict[str, Any] = {"weight_gradient_calls": []}
    original_weight_function = finetune._patient_class_sample_weights

    def capture_weights(rows_by_id: Any, image_ids: Any, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
        weights, diagnostics = original_weight_function(rows_by_id, image_ids, device)
        observed["train_image_weights"] = weights.detach().cpu().tolist()
        weights.requires_grad_(True)
        weights.register_hook(
            lambda gradient: observed["weight_gradient_calls"].append(gradient.detach().cpu().tolist())
        )
        return weights, diagnostics

    original_bce = finetune.nn.BCEWithLogitsLoss

    class RecordingBCE(original_bce):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("reduction") != "none" or kwargs.get("pos_weight") is not None:
                raise AssertionError("Trainer must use unreduced BCE element losses without pos_weight.")
            observed["bce_reduction"] = kwargs["reduction"]
            observed["bce_pos_weight"] = kwargs.get("pos_weight")
            super().__init__(*args, **kwargs)

    finetune._patient_class_sample_weights = capture_weights
    finetune.nn.BCEWithLogitsLoss = RecordingBCE
    try:
        torch.manual_seed(20261014)
        model = ScriptedValidationTinyCNN(dataset)
        config = finetune.ImageFinetuneConfig(
            mode="F0", image_size=8, batch_size=1, accumulation_steps=1,
            max_epochs=3, patience=3, seed=17,
            augment_whole_frame=False, amp=False,
            budget_minutes=10, reserve_minutes=0,
        )
        with tempfile.TemporaryDirectory(prefix="finetune-after-fix-") as temporary_root:
            result = finetune.train_one_image_fold(
                model, dataset, rows, train_ids,
                [f"proof-image-{index}" for index in range(6, 10)],
                config, Path(temporary_root) / "fold",
                source_binding={"binding_id": "synthetic-source-binding"},
                device="cpu",
            )
            validation_labels = [0, 0, 1, 1]
            auc_by_epoch = [
                float(roc_auc_score(validation_labels, [probability for _, probability in trace]))
                for trace in model.validation_trace
            ]
            saved_auc_by_epoch = [
                float(roc_auc_score(
                    [row["image_target"] for row in epoch["validation_predictions"]],
                    [row["probability"] for row in epoch["validation_predictions"]],
                ))
                for epoch in result["history"]
            ]
            losses = [float(row["validation_loss"]) for row in result["history"]]
            weight_values = observed["train_image_weights"]
            patient_contribution = Counter()
            patient_exposure = Counter()
            for gradient in observed["weight_gradient_calls"]:
                active = [index for index, value in enumerate(gradient) if abs(value) > 1e-12]
                if len(active) != 1:
                    raise AssertionError(f"Expected one selected image per optimizer step; got {active}.")
                index = active[0]
                patient = rows[index]["patient_id"]
                patient_exposure[patient] += 1
                patient_contribution[patient] += weight_values[index] * gradient[index]

            pointer = json.loads((Path(result["checkpoint_root"]) / "best.json").read_text(encoding="utf-8"))
            manifest_path = Path(result["checkpoint_root"]) / "versions" / pointer["version"] / "commit.json"
            identity = json.loads(manifest_path.read_text(encoding="utf-8"))["identity"]
            best_state = CheckpointStore(result["checkpoint_root"], identity).load_best()
            diagnostics = result["status"]["sample_weight_diagnostics"]
            proof = {
                "phase": "after_fix",
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "patient_class_weights": {
                    "scheme": diagnostics["scheme"],
                    "normalization": diagnostics["normalization"],
                    "image_weights": {
                        train_ids[index]: float(weight)
                        for index, weight in enumerate(weight_values)
                    },
                    "diagnostic_patient_class_mass": diagnostics["patient_class_weight_sums"],
                    "weight_sum": diagnostics["weight_sum"],
                    "mean_weight": diagnostics["weight_mean"],
                    "effective_loss_formula": result["status"]["loss_policy"],
                    "bce_reduction": observed["bce_reduction"],
                    "bce_pos_weight": observed["bce_pos_weight"],
                    "weight_gradient_calls": len(observed["weight_gradient_calls"]),
                    "patient_optimizer_step_exposures": dict(patient_exposure),
                    "weighted_loss_contributions": dict(patient_contribution),
                },
                "epoch_selection": {
                    "policy": result["selection_policy"],
                    "validation_loss_by_epoch": losses,
                    "validation_auroc_by_epoch": auc_by_epoch,
                    "saved_probability_label_auroc_by_epoch": saved_auc_by_epoch,
                    "loss_best_epoch": int(np.argmin(losses)) + 1,
                    "auroc_best_epoch": int(np.argmax(auc_by_epoch)) + 1,
                    "selected_best_epoch": result["best_epoch"],
                    "best_checkpoint_epoch": best_state["epoch"],
                    "best_checkpoint_auroc": best_state["best_metric"],
                },
            }
            assert observed["train_image_weights"] == pytest.approx(expected_weights.tolist())
            assert diagnostics["weight_sum"] == pytest.approx(len(train_ids))
            assert diagnostics["weight_mean"] == pytest.approx(1.0)
            assert len(observed["weight_gradient_calls"]) == 18
            assert patient_exposure == {
                "negative-multi": 6,
                "negative-single": 3,
                "positive-multi": 6,
                "positive-single": 3,
            }
            assert auc_by_epoch == pytest.approx([1.0, 0.75, 0.5])
            assert saved_auc_by_epoch == pytest.approx(auc_by_epoch)
            assert losses[1] < losses[0]
            assert result["best_epoch"] == best_state["epoch"] == 1
            assert result["best_validation_auroc"] == best_state["best_metric"] == pytest.approx(1.0)
            return proof
    finally:
        finetune._patient_class_sample_weights = original_weight_function
        finetune.nn.BCEWithLogitsLoss = original_bce


if __name__ == "__main__":
    proof_result = _run_reproduction()
    PROOF_PATH.write_text(json.dumps(proof_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(proof_result, indent=2, sort_keys=True))
