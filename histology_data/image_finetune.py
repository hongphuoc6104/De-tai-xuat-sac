"""Conditional F0/F1 whole-frame fine-tuning with resumable fold checkpoints.

This module accepts an already-created pretrained Torch module and a source
dataset that yields complete frames. It never constructs a backbone, fetches
weights, or chooses patients for the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .image_classification import resize_pad_rgb
from .io import atomic_json, file_hash, fingerprint
from .mil_checkpoint import CheckpointIdentityError, CheckpointStore, FitLock

FINETUNE_SCHEMA_VERSION = 1
FINETUNE_PROTOCOL = "conditional-whole-frame-finetune-v1"
MODES = {"F0", "F1"}
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageFinetuneConfig:
    """Fixed fold-level optimization controls for the conditional trainer."""

    mode: str = "F0"
    image_size: int = 512
    batch_size: int = 8
    accumulation_steps: int = 1
    max_epochs: int = 10
    fixed_epoch_count: int | None = None
    patience: int = 3
    min_delta: float = 0.0
    seed: int = 17
    head_learning_rate: float = 1e-3
    layer4_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    checkpoint_interval_steps: int = 10
    augment_whole_frame: bool = True
    amp: bool = True
    budget_minutes: float = 240.0
    reserve_minutes: float = 20.0
    recover_stale_lock: bool = False

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError("mode must be 'F0' or 'F1'.")
        for name in ("image_size", "batch_size", "accumulation_steps", "max_epochs", "patience",
                     "checkpoint_interval_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_epochs > 10:
            raise ValueError("max_epochs may not exceed the approved 10-epoch limit.")
        if self.fixed_epoch_count is not None and (
            isinstance(self.fixed_epoch_count, bool)
            or not isinstance(self.fixed_epoch_count, int)
            or not 1 <= self.fixed_epoch_count <= 10
        ):
            raise ValueError("fixed_epoch_count must be an integer in [1,10] or None.")
        if self.fixed_epoch_count is not None and self.fixed_epoch_count > self.max_epochs:
            raise ValueError("fixed_epoch_count may not exceed max_epochs.")
        if self.patience != 3:
            raise ValueError("The conditional F0/F1 protocol fixes early-stop patience at 3 epochs.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        for name in ("min_delta", "head_learning_rate", "layer4_learning_rate", "weight_decay",
                     "budget_minutes", "reserve_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.min_delta < 0 or self.weight_decay < 0:
            raise ValueError("min_delta and weight_decay must be nonnegative.")
        if self.head_learning_rate <= 0 or self.layer4_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.head_learning_rate != 1e-3 or self.layer4_learning_rate != 1e-5:
            raise ValueError("The conditional protocol fixes head/layer4 learning rates at 1e-3/1e-5.")
        if self.budget_minutes <= 0 or self.reserve_minutes < 0 or self.reserve_minutes >= self.budget_minutes:
            raise ValueError("budget_minutes must be positive and exceed reserve_minutes.")
        for name in ("augment_whole_frame", "amp", "recover_stale_lock"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON-compatible mapping.")
    try:
        result = json.loads(json.dumps(dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON-compatible values.") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = ("image_id", "patient_id", "image_target", "objective_lens",
                "source_archive_sha256", "original_image_sha256")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every source row must be a mapping.")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Source row is missing required binding fields: {missing}")
        image_id = str(row["image_id"])
        patient_id = str(row["patient_id"])
        if not image_id or not patient_id or image_id in seen:
            raise ValueError("Source image IDs and patient IDs must be nonempty; image IDs must be unique.")
        raw_target = row["image_target"]
        if isinstance(raw_target, bool):
            raise ValueError(f"Image target must be binary for {image_id}.")
        try:
            target = int(raw_target)
            lens = int(row["objective_lens"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Image target and objective lens must be integers for {image_id}.") from exc
        if target not in (0, 1) or str(target) != str(raw_target).strip():
            raise ValueError(f"Image target must be binary for {image_id}.")
        if isinstance(row["objective_lens"], bool) or lens not in (4, 10, 40):
            raise ValueError(f"Objective lens must be 4, 10, or 40 for {image_id}.")
        for key in ("source_archive_sha256", "original_image_sha256"):
            digest = row[key]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"Source row has an invalid {key} for {image_id}.")
        normalized_row = {
            "image_id": image_id,
            "patient_id": patient_id,
            "image_target": int(target),
            "objective_lens": int(lens),
            "source_archive_sha256": row["source_archive_sha256"],
            "original_image_sha256": row["original_image_sha256"],
        }
        if "raw_glade" in row:
            normalized_row["raw_glade"] = str(row["raw_glade"])
        normalized.append(normalized_row)
        seen.add(image_id)
    if not normalized:
        raise ValueError("At least one bound original-image source row is required.")
    return normalized


def _model_schema(model: nn.Module) -> dict[str, Any]:
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in model.state_dict().items()
        ],
    }


def _model_state_sha256(model: nn.Module) -> str:
    """Bind resume identity to the injected pretrained weights and seeded head."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value) or value.layout != torch.strided:
            raise ValueError("Injected model state must contain only dense Torch tensors.")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _configure_trainability(model: nn.Module, mode: str) -> tuple[list[str], list[dict[str, Any]]]:
    fc = getattr(model, "fc", None)
    layer4 = getattr(model, "layer4", None)
    if not isinstance(fc, nn.Module):
        raise ValueError("Injected backbone must expose an nn.Module named 'fc'.")
    if mode == "F1" and not isinstance(layer4, nn.Module):
        raise ValueError("F1 requires the injected backbone to expose an nn.Module named 'layer4'.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[nn.Module] = [fc] if mode == "F0" else [layer4, fc]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    batch_norm_names: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
            batch_norm_names.append({
                "name": name,
                "running_mean": module.running_mean is not None,
                "running_var": module.running_var is not None,
            })
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise ValueError(f"No trainable parameters remain for {mode}.")
    return trainable_names, batch_norm_names


def _optimizer(model: nn.Module, mode: str, config: ImageFinetuneConfig) -> torch.optim.Optimizer:
    fc = getattr(model, "fc")
    fc_ids = {id(parameter) for parameter in fc.parameters() if parameter.requires_grad}
    groups: list[dict[str, Any]] = []
    if mode == "F1":
        layer4 = getattr(model, "layer4")
        layer4_parameters = [parameter for parameter in layer4.parameters()
                             if parameter.requires_grad and id(parameter) not in fc_ids]
        if layer4_parameters:
            groups.append({"params": layer4_parameters, "lr": config.layer4_learning_rate,
                           "group_name": "layer4"})
    fc_parameters = [parameter for parameter in model.parameters()
                      if parameter.requires_grad and id(parameter) in fc_ids]
    if fc_parameters:
        groups.append({"params": fc_parameters, "lr": config.head_learning_rate,
                       "group_name": "fc"})
    if not groups:
        raise ValueError("No optimizer parameter groups were created.")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _train_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng(shuffle_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "shuffle_generator": shuffle_generator.get_state().cpu(),
    }


def _restore_rng(state: Mapping[str, Any], shuffle_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    shuffle_generator.set_state(state["shuffle_generator"].cpu())


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    return _cpu_tree(state)


def _frame_to_tensor(frame: Any, image_size: int) -> torch.Tensor:
    """Resize and pad a complete RGB frame; never crop or select patches."""
    if isinstance(frame, Image.Image):
        prepared = resize_pad_rgb(frame, max_long_edge=image_size)
        array = np.asarray(prepared, dtype=np.uint8).copy()
        prepared.close()
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
    else:
        if torch.is_tensor(frame):
            tensor = frame.detach().to(device="cpu", dtype=torch.float32)
        else:
            array = np.asarray(frame)
            tensor = torch.from_numpy(np.array(array, copy=True)).to(torch.float32)
        if tensor.ndim != 3:
            raise ValueError("A source frame must have exactly three dimensions.")
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor[:3]
        else:
            tensor = tensor.permute(2, 0, 1)[:3]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] != 3 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
            raise ValueError("A source frame must contain nonempty RGB pixels.")
        if tensor.numel() and float(tensor.max()) > 1.0:
            tensor.div_(255.0)
        if not torch.isfinite(tensor).all() or float(tensor.min()) < 0 or float(tensor.max()) > 1.0:
            raise ValueError("Source frame values must be finite raw pixels in [0,1] or [0,255].")
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
        scale = image_size / max(height, width)
        new_width = max(1, min(image_size, int(round(width * scale))))
        new_height = max(1, min(image_size, int(round(height * scale))))
        resized = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width),
                                mode="bilinear", align_corners=False).squeeze(0)
        canvas = torch.empty((3, image_size, image_size), dtype=torch.float32)
        canvas[0].fill_(124.0 / 255.0)
        canvas[1].fill_(116.0 / 255.0)
        canvas[2].fill_(104.0 / 255.0)
        top = (image_size - new_height) // 2
        left = (image_size - new_width) // 2
        canvas[:, top:top + new_height, left:left + new_width] = resized
        tensor = canvas
    mean = torch.tensor(_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _augment_whole_frame(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    if bool(torch.rand(()) < 0.5):
        tensor = torch.flip(tensor, dims=(-1,))
    rotations = int(torch.randint(0, 4, ()).item())
    return torch.rot90(tensor, k=rotations, dims=(-2, -1))


def _dataset_batch(
    dataset: Any,
    dataset_rows: list[dict[str, Any]],
    positions: Sequence[int],
    *,
    image_size: int,
    augment: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames: list[torch.Tensor] = []
    labels: list[int] = []
    for position in positions:
        expected = dataset_rows[position]
        item = dataset[position]
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("source_dataset[index] must return (whole_frame, label, image_id).")
        frame, label_value, image_id_value = item
        if str(image_id_value) != expected["image_id"]:
            raise ValueError(f"Source dataset image identity changed at row {position}.")
        label = int(label_value.item()) if torch.is_tensor(label_value) else int(label_value)
        if label != expected["image_target"]:
            raise ValueError(f"Source dataset label changed for {expected['image_id']}.")
        tensor = _frame_to_tensor(frame, image_size)
        frames.append(_augment_whole_frame(tensor, augment))
        labels.append(label)
    return torch.stack(frames).to(device), torch.tensor(labels, dtype=torch.float32, device=device)


def _validate_logits(logits: torch.Tensor, batch_count: int) -> torch.Tensor:
    values = logits.reshape(-1)
    if values.shape != (batch_count,) or not torch.isfinite(values).all():
        raise ValueError("The injected model must return one finite binary logit per whole frame.")
    return values


def _validation_metrics(
    model: nn.Module,
    dataset: Any,
    dataset_rows: list[dict[str, Any]],
    dataset_positions: dict[str, int],
    image_ids: Sequence[str],
    *,
    config: ImageFinetuneConfig,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(image_ids), config.batch_size):
            batch_ids = image_ids[start:start + config.batch_size]
            positions = [dataset_positions[image_id] for image_id in batch_ids]
            frames, labels = _dataset_batch(
                dataset, dataset_rows, positions, image_size=config.image_size,
                augment=False, device=device,
            )
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast:
                logits = _validate_logits(model(frames), len(batch_ids))
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite.")
            total_loss += float(loss.item()) * len(batch_ids)
            correct += int(((logits >= 0) == (labels >= 0.5)).sum().item())
            count += len(batch_ids)
    return total_loss / count, correct / count


def _class_weight(rows_by_id: Mapping[str, dict[str, Any]], train_ids: Sequence[str],
                  device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    patient_classes: dict[str, set[int]] = defaultdict(set)
    for image_id in train_ids:
        row = rows_by_id[image_id]
        patient_classes[row["patient_id"]].add(row["image_target"])
    negative_patients = sum(0 in classes for classes in patient_classes.values())
    positive_patients = sum(1 in classes for classes in patient_classes.values())
    if negative_patients == 0 or positive_patients == 0:
        raise ValueError("The training patients must represent both binary image classes.")
    pos_weight = negative_patients / positive_patients
    return torch.tensor([pos_weight], dtype=torch.float32, device=device), {
        "negative_class_patients": negative_patients,
        "positive_class_patients": positive_patients,
    }


def _validate_split(
    rows: list[dict[str, Any]], train_image_ids: Sequence[str],
    validation_image_ids: Sequence[str], outer_test_image_ids: Sequence[str], dataset: Any,
    *, allow_empty_validation: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str], list[str], dict[str, Any]]:
    if len(dataset) != len(rows):
        raise ValueError("source_dataset and rows must have identical image counts and order.")
    row_by_id = {row["image_id"]: row for row in rows}
    dataset_positions = {row["image_id"]: index for index, row in enumerate(rows)}
    train_ids = [str(value) for value in train_image_ids]
    validation_ids = [str(value) for value in validation_image_ids]
    outer_test_ids = [str(value) for value in outer_test_image_ids]
    if (not train_ids or (not validation_ids and not allow_empty_validation)
            or len(set(train_ids)) != len(train_ids)
            or len(set(validation_ids)) != len(validation_ids)):
        raise ValueError("Training IDs must be nonempty and both split ID lists must be unique.")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation image IDs overlap.")
    if len(set(outer_test_ids)) != len(outer_test_ids):
        raise ValueError("Outer-test image IDs must be unique.")
    if set(outer_test_ids) & (set(train_ids) | set(validation_ids)):
        raise ValueError("Outer-test images must be excluded from training and validation.")
    unknown = (set(train_ids) | set(validation_ids) | set(outer_test_ids)) - set(row_by_id)
    if unknown:
        raise ValueError(f"Fold split references unknown source image IDs: {sorted(unknown)[:5]}")
    train_patients = {row_by_id[image_id]["patient_id"] for image_id in train_ids}
    validation_patients = {row_by_id[image_id]["patient_id"] for image_id in validation_ids}
    outer_test_patients = {row_by_id[image_id]["patient_id"] for image_id in outer_test_ids}
    overlap = (train_patients & validation_patients) | (outer_test_patients & (train_patients | validation_patients))
    if overlap:
        raise ValueError("Patient-level leakage across train, validation, and outer-test groups.")
    split_binding = {
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids,
        "outer_test_image_ids": outer_test_ids,
        "train_patient_sha256": sorted(_sha256_text(patient) for patient in train_patients),
        "validation_patient_sha256": sorted(_sha256_text(patient) for patient in validation_patients),
        "outer_test_patient_sha256": sorted(_sha256_text(patient) for patient in outer_test_patients),
    }
    return row_by_id, dataset_positions, train_ids, validation_ids, split_binding


def _identity(
    model: nn.Module,
    config: ImageFinetuneConfig,
    rows: list[dict[str, Any]],
    source_binding: dict[str, Any],
    split_binding: dict[str, Any],
    device_type: str,
    initial_model_state_sha256: str,
) -> dict[str, Any]:
    semantic_config = asdict(config)
    for key in ("budget_minutes", "reserve_minutes", "recover_stale_lock"):
        semantic_config.pop(key)
    fitted_ids = set(split_binding["train_image_ids"]) | set(split_binding["validation_image_ids"])
    row_binding = [
        {
            "image_id": row["image_id"],
            "patient_sha256": _sha256_text(row["patient_id"]),
            "image_target": row["image_target"],
            "objective_lens": row["objective_lens"],
            "source_archive_sha256": row["source_archive_sha256"],
            "original_image_sha256": row["original_image_sha256"],
            "raw_glade": row.get("raw_glade"),
        }
        for row in rows if row["image_id"] in fitted_ids
    ]
    return {
        "schema_version": FINETUNE_SCHEMA_VERSION,
        "protocol": FINETUNE_PROTOCOL,
        "mode": config.mode,
        "device_type": device_type,
        "config": semantic_config,
        "model_schema": _model_schema(model),
        "initial_model_state_sha256": initial_model_state_sha256,
        "torch_version": str(torch.__version__),
        "trainer_implementation_sha256": file_hash(Path(__file__)),
        "whole_frame_preprocessing_sha256": file_hash(Path(resize_pad_rgb.__code__.co_filename)),
        "source_binding": source_binding,
        "source_rows_sha256": fingerprint(row_binding),
        "split_binding": split_binding,
    }


def _new_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    shuffle_generator: torch.Generator,
    identity: dict[str, Any],
    train_count: int,
    seed: int,
    class_weight_basis: dict[str, int],
) -> dict[str, Any]:
    _seed_all(seed)
    shuffle_generator.manual_seed(seed)
    permutation = torch.randperm(train_count, generator=shuffle_generator).tolist()
    state = {
        "schema_version": FINETUNE_SCHEMA_VERSION,
        "identity_sha256": fingerprint(identity),
        "identity": identity,
        "source_binding": identity["source_binding"],
        "split_binding": identity["split_binding"],
        "config": identity["config"],
        "class_weight_basis": class_weight_basis,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": 0,
        "global_step": 0,
        "train_permutation": permutation,
        "train_cursor": 0,
        "epoch_loss_sum": 0.0,
        "epoch_sample_count": 0,
        "epoch_step_count": 0,
        "shuffle_generator_state": shuffle_generator.get_state(),
        "best_metric": math.inf,
        "best_epoch": 0,
        "patience_count": 0,
        "history": [],
        "finished": False,
        "rng": _capture_rng(shuffle_generator),
    }
    return _snapshot_state(state)


def _restore_state(
    state: dict[str, Any],
    identity: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    shuffle_generator: torch.Generator,
    train_count: int,
    class_weight_basis: dict[str, int],
) -> dict[str, Any]:
    if (state.get("schema_version") != FINETUNE_SCHEMA_VERSION
            or state.get("identity_sha256") != fingerprint(identity)
            or state.get("identity") != identity
            or state.get("source_binding") != identity["source_binding"]
            or state.get("split_binding") != identity["split_binding"]
            or state.get("config") != identity["config"]
            or state.get("class_weight_basis") != class_weight_basis):
        raise CheckpointIdentityError("Fine-tune checkpoint state does not match source, split, or config identity.")
    permutation = state.get("train_permutation")
    if (not isinstance(permutation, list) or len(permutation) != train_count
            or sorted(permutation) != list(range(train_count))):
        raise ValueError("Checkpoint shuffle permutation is invalid for this training split.")
    cursor = state.get("train_cursor")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= train_count:
        raise ValueError("Checkpoint training cursor is invalid.")
    if (not isinstance(state.get("epoch_loss_sum"), (int, float))
            or not math.isfinite(float(state["epoch_loss_sum"]))
            or isinstance(state.get("epoch_step_count"), bool)
            or not isinstance(state.get("epoch_step_count"), int)
            or state["epoch_step_count"] < 0
            or isinstance(state.get("epoch_sample_count"), bool)
            or not isinstance(state.get("epoch_sample_count"), int)
            or state["epoch_sample_count"] < 0):
        raise ValueError("Checkpoint epoch loss counters are invalid.")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    shuffle_generator.set_state(state["shuffle_generator_state"].cpu())
    _restore_rng(state["rng"], shuffle_generator)
    return state


def _status_write(path: Path, *, state: str, fit_id: str, epoch: int, global_step: int,
                  train_cursor: int, best_metric: float, best_epoch: int,
                  patience_count: int, checkpoint_count: int, class_weight_basis: dict[str, int],
                  ready_for_fit: bool, selection_policy: str,
                  fixed_epoch_count: int | None,
                  stop_reason: str | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": FINETUNE_SCHEMA_VERSION,
        "protocol": FINETUNE_PROTOCOL,
        "state": state,
        "fit_id": fit_id,
        "epoch_completed": epoch,
        "global_step": global_step,
        "train_cursor": train_cursor,
        "best_validation_loss": None if not math.isfinite(best_metric) else best_metric,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "checkpoint_count": checkpoint_count,
        "class_weight_basis": class_weight_basis,
        "class_weight_policy": "BCE pos_weight from train-only patient-class counts; no balanced sampler",
        "optimizer_boundary_only_checkpoints": True,
        "ready_for_fit": ready_for_fit,
        "selection_policy": selection_policy,
        "fixed_epoch_count": fixed_epoch_count,
    }
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    atomic_json(path, payload)
    return payload


def train_one_image_fold(
    model: nn.Module,
    source_dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    train_image_ids: Sequence[str],
    validation_image_ids: Sequence[str],
    config: ImageFinetuneConfig,
    output_root: str | Path,
    *,
    source_binding: Mapping[str, Any],
    outer_test_image_ids: Sequence[str] = (),
    device: str = "auto",
    max_optimizer_steps_per_call: int | None = None,
    notify: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one explicit patient-disjoint whole-image fold using injected weights.

    The injected fresh model must expose ``fc`` and, for F1, ``layer4``; resume
    calls must use the same initial state, including a deterministically seeded
    classifier head. The dataset must yield ``(whole_frame, binary_label,
    image_id)`` at each row index.
    Frames are resized proportionally, center-padded to ``image_size``,
    ImageNet-normalized, and augmented only by whole-canvas flips and quarter-turn
    rotations. Outer-test IDs can be supplied to assert they are excluded from
    both explicit groups; the trainer never loads those IDs. Setting
    ``fixed_epoch_count`` requires empty validation IDs and trains every source
    row outside the explicit outer-test group without validation or early stop.
    """
    config.validate()
    if max_optimizer_steps_per_call is not None and (
        isinstance(max_optimizer_steps_per_call, bool)
        or not isinstance(max_optimizer_steps_per_call, int)
        or max_optimizer_steps_per_call < 1
    ):
        raise ValueError("max_optimizer_steps_per_call must be a positive integer or None.")
    if notify is not None and not callable(notify):
        raise TypeError("notify must be callable or None.")
    normalized_rows = _normalize_rows(rows)
    source_binding_json = _json_mapping(source_binding, "source_binding")
    if not isinstance(source_binding_json.get("binding_id"), str) or not source_binding_json["binding_id"]:
        raise ValueError("source_binding must contain a nonempty binding_id.")
    if config.fixed_epoch_count is not None:
        if validation_image_ids:
            raise ValueError("Fixed-epoch outer refit must use an empty validation image set.")
        if not outer_test_image_ids:
            raise ValueError("Fixed-epoch outer refit requires explicit outer_test_image_ids.")
    row_by_id, dataset_positions, train_ids, validation_ids, split_binding = _validate_split(
        normalized_rows, train_image_ids, validation_image_ids, outer_test_image_ids, source_dataset,
        allow_empty_validation=config.fixed_epoch_count is not None,
    )
    if config.fixed_epoch_count is not None and set(train_ids) != (
        set(row_by_id) - set(split_binding["outer_test_image_ids"])
    ):
        raise ValueError("Fixed-epoch outer refit must train on every outer-train image.")
    model_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    try:
        resolved_device = torch.device(model_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid training device: {device!r}.") from exc
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("Fine-tune device must resolve to CPU or CUDA.")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model.to(resolved_device)
    initial_model_state_sha256 = _model_state_sha256(model)
    trainable_names, batch_norm_names = _configure_trainability(model, config.mode)
    identity = _identity(
        model, config, normalized_rows, source_binding_json, split_binding,
        resolved_device.type, initial_model_state_sha256,
    )
    epoch_limit = config.fixed_epoch_count or config.max_epochs
    selection_policy = (
        "fixed_epoch_count_no_validation"
        if config.fixed_epoch_count is not None
        else "validation_loss_with_patience"
    )
    fit_id = fingerprint(identity)
    fit_root = Path(output_root).expanduser()
    fit_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = fit_root / "checkpoints"
    store = CheckpointStore(checkpoint_root, identity)
    optimizer = _optimizer(model, config.mode, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7,
    )
    amp_enabled = config.amp and resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    shuffle_generator = torch.Generator(device="cpu")
    pos_weight, class_weight_basis = _class_weight(row_by_id, train_ids, resolved_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    deadline = time.monotonic() + (config.budget_minutes - config.reserve_minutes) * 60.0
    invocation_steps = 0
    finite_gradient_checks = 0
    started = time.monotonic()

    with FitLock(fit_root, owner_id=fit_id, recover_stale=config.recover_stale_lock):
        loaded = store.load_latest(map_location="cpu")
        if loaded is None:
            state = _new_state(
                model, optimizer, scheduler, scaler, shuffle_generator, identity,
                len(train_ids), config.seed, class_weight_basis,
            )
            receipt = store.publish(state, epoch=0, is_best=False)
            _restore_rng(state["rng"], shuffle_generator)
            last_checkpoint_sha256 = receipt.sha256
        else:
            state = _restore_state(
                loaded, identity, model, optimizer, scheduler, scaler,
                shuffle_generator, len(train_ids), class_weight_basis,
            )
            pointer = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
            last_checkpoint_sha256 = pointer["sha256"]

        checkpoint_count = len(list((checkpoint_root / "versions").glob("*/commit.json")))
        status_path = fit_root / "status.json"
        if state["finished"]:
            best_state = store.load_best(map_location="cpu")
            if best_state is None:
                raise ValueError("Completed fold has no committed best checkpoint.")
            model.load_state_dict(best_state["model_state"], strict=True)
            model.eval()
            result_status = _status_write(
                status_path, state="complete", fit_id=fit_id,
                epoch=state["epoch"], global_step=state["global_step"],
                train_cursor=state["train_cursor"], best_metric=state["best_metric"],
                best_epoch=state["best_epoch"], patience_count=state["patience_count"],
                checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
                ready_for_fit=True, selection_policy=selection_policy,
                fixed_epoch_count=config.fixed_epoch_count,
            )
            return _result_payload(fit_root, checkpoint_root, result_status, state,
                                   finite_gradient_checks, started, fit_id, last_checkpoint_sha256,
                                   trainable_names, batch_norm_names)

        _status_write(
            status_path, state="running", fit_id=fit_id,
            epoch=state["epoch"], global_step=state["global_step"],
            train_cursor=state["train_cursor"], best_metric=state["best_metric"],
            best_epoch=state["best_epoch"], patience_count=state["patience_count"],
            checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
            ready_for_fit=False, selection_policy=selection_policy,
            fixed_epoch_count=config.fixed_epoch_count,
        )

        def publish(*, is_best: bool = False) -> None:
            nonlocal last_checkpoint_sha256, checkpoint_count
            state["model_state"] = model.state_dict()
            state["optimizer_state"] = optimizer.state_dict()
            state["scheduler_state"] = scheduler.state_dict()
            state["scaler_state"] = scaler.state_dict()
            state["shuffle_generator_state"] = shuffle_generator.get_state()
            state["rng"] = _capture_rng(shuffle_generator)
            receipt = store.publish(_snapshot_state(state), epoch=state["epoch"], is_best=is_best)
            _restore_rng(state["rng"], shuffle_generator)
            last_checkpoint_sha256 = receipt.sha256
            checkpoint_count = len(list((checkpoint_root / "versions").glob("*/commit.json")))

        def finish_epoch() -> dict[str, Any] | None:
            if config.fixed_epoch_count is None:
                validation_loss, validation_accuracy = _validation_metrics(
                    model, source_dataset, normalized_rows, dataset_positions, validation_ids,
                    config=config, device=resolved_device, amp_enabled=amp_enabled,
                )
                scheduler.step(validation_loss)
                improved = validation_loss < state["best_metric"] - config.min_delta
            else:
                validation_loss, validation_accuracy = None, None
                improved = True
            if improved:
                if validation_loss is not None:
                    state["best_metric"] = validation_loss
                state["best_epoch"] = state["epoch"] + 1
                state["patience_count"] = 0
            elif config.fixed_epoch_count is None:
                state["patience_count"] += 1
            state["history"].append({
                "epoch": state["epoch"] + 1,
                "train_loss": state["epoch_loss_sum"] / max(1, state["epoch_sample_count"]),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            })
            state["epoch"] += 1
            state["train_cursor"] = 0
            state["epoch_loss_sum"] = 0.0
            state["epoch_sample_count"] = 0
            state["epoch_step_count"] = 0
            stop_early = (
                config.fixed_epoch_count is None
                and state["patience_count"] >= config.patience
            )
            state["finished"] = state["epoch"] >= epoch_limit or stop_early
            if not state["finished"]:
                state["train_permutation"] = torch.randperm(
                    len(train_ids), generator=shuffle_generator,
                ).tolist()
            state["shuffle_generator_state"] = shuffle_generator.get_state()
            publish(is_best=improved)
            state_label = "complete" if state["finished"] else "running"
            result_status = _status_write(
                status_path, state=state_label, fit_id=fit_id,
                epoch=state["epoch"], global_step=state["global_step"],
                train_cursor=state["train_cursor"], best_metric=state["best_metric"],
                best_epoch=state["best_epoch"], patience_count=state["patience_count"],
                checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
                ready_for_fit=state["finished"], selection_policy=selection_policy,
                fixed_epoch_count=config.fixed_epoch_count,
            )
            if notify is not None:
                notify({"state": state_label, "epoch_completed": state["epoch"],
                        "global_step": state["global_step"], "fit_root": str(fit_root)})
            if not state["finished"]:
                return None
            best = store.load_best(map_location="cpu")
            if best is not None:
                model.load_state_dict(best["model_state"], strict=True)
                model.eval()
            return _result_payload(
                fit_root, checkpoint_root, result_status, state,
                finite_gradient_checks, started, fit_id, last_checkpoint_sha256,
                trainable_names, batch_norm_names,
            )

        while state["epoch"] < epoch_limit and not state["finished"]:
            permutation = state["train_permutation"]
            if state["train_cursor"] == len(permutation):
                result = finish_epoch()
                if result is not None:
                    return result
                continue
            if time.monotonic() >= deadline or (
                max_optimizer_steps_per_call is not None
                and invocation_steps >= max_optimizer_steps_per_call
            ):
                stop_reason = "budget" if time.monotonic() >= deadline else "optimizer_step_cap"
                _status_write(
                    status_path, state="partial", fit_id=fit_id,
                    epoch=state["epoch"], global_step=state["global_step"],
                    train_cursor=state["train_cursor"], best_metric=state["best_metric"],
                    best_epoch=state["best_epoch"], patience_count=state["patience_count"],
                    checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
                    ready_for_fit=False, selection_policy=selection_policy,
                    fixed_epoch_count=config.fixed_epoch_count, stop_reason=stop_reason,
                )
                return _result_payload(fit_root, checkpoint_root, json.loads(status_path.read_text()),
                                       state, finite_gradient_checks, started, fit_id, last_checkpoint_sha256,
                                       trainable_names, batch_norm_names)

            _train_mode(model)
            cursor = state["train_cursor"]
            window_capacity = config.batch_size * config.accumulation_steps
            end_cursor = min(len(permutation), cursor + window_capacity)
            selected_positions = permutation[cursor:end_cursor]
            batches = [selected_positions[start:start + config.batch_size]
                       for start in range(0, len(selected_positions), config.batch_size)]
            if not batches:
                raise ValueError("Checkpoint cursor points past the training permutation.")
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            for batch_positions in batches:
                image_ids = [train_ids[position] for position in batch_positions]
                dataset_indices = [dataset_positions[image_id] for image_id in image_ids]
                frames, labels = _dataset_batch(
                    source_dataset, normalized_rows, dataset_indices,
                    image_size=config.image_size, augment=config.augment_whole_frame,
                    device=resolved_device,
                )
                autocast = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
                with autocast:
                    logits = _validate_logits(model(frames), len(image_ids))
                    batch_loss_sum = criterion(logits, labels)
                    loss = batch_loss_sum / len(selected_positions)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite.")
                step_loss_sum += float(batch_loss_sum.detach().item())
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            checked = [parameter.grad for parameter in model.parameters()
                       if parameter.requires_grad and parameter.grad is not None]
            if not checked or any(not torch.isfinite(gradient).all() for gradient in checked):
                optimizer.zero_grad(set_to_none=True)
                raise ValueError("Trainable parameters produced missing or non-finite gradients.")
            finite_gradient_checks += len(checked)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["train_cursor"] = end_cursor
            state["global_step"] += 1
            state["epoch_loss_sum"] += step_loss_sum
            state["epoch_sample_count"] += len(selected_positions)
            state["epoch_step_count"] += 1
            invocation_steps += 1
            should_stop = (
                time.monotonic() >= deadline
                or (max_optimizer_steps_per_call is not None
                    and invocation_steps >= max_optimizer_steps_per_call)
            )
            due_checkpoint = (
                state["global_step"] % config.checkpoint_interval_steps == 0
                or should_stop
                or state["train_cursor"] == len(permutation)
            )
            if due_checkpoint:
                publish()
                _status_write(
                    status_path, state="running", fit_id=fit_id,
                    epoch=state["epoch"], global_step=state["global_step"],
                    train_cursor=state["train_cursor"], best_metric=state["best_metric"],
                    best_epoch=state["best_epoch"], patience_count=state["patience_count"],
                    checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
                    ready_for_fit=False, selection_policy=selection_policy,
                    fixed_epoch_count=config.fixed_epoch_count,
                )
                if notify is not None:
                    notify({"state": "running", "epoch_completed": state["epoch"],
                            "global_step": state["global_step"], "fit_root": str(fit_root)})
            if should_stop and state["train_cursor"] < len(permutation):
                stop_reason = "budget" if time.monotonic() >= deadline else "optimizer_step_cap"
                _status_write(
                    status_path, state="partial", fit_id=fit_id,
                    epoch=state["epoch"], global_step=state["global_step"],
                    train_cursor=state["train_cursor"], best_metric=state["best_metric"],
                    best_epoch=state["best_epoch"], patience_count=state["patience_count"],
                    checkpoint_count=checkpoint_count, class_weight_basis=class_weight_basis,
                    ready_for_fit=False, selection_policy=selection_policy,
                    fixed_epoch_count=config.fixed_epoch_count, stop_reason=stop_reason,
                )
                if notify is not None:
                    notify({"state": "partial", "epoch_completed": state["epoch"],
                            "global_step": state["global_step"], "fit_root": str(fit_root)})
                return _result_payload(fit_root, checkpoint_root, json.loads(status_path.read_text()),
                                       state, finite_gradient_checks, started, fit_id, last_checkpoint_sha256,
                                       trainable_names, batch_norm_names)

            if state["train_cursor"] == len(permutation):
                result = finish_epoch()
                if result is not None:
                    return result

    raise RuntimeError("Fine-tune fold left its lock without a terminal or partial result.")


def _result_payload(fit_root: Path, checkpoint_root: Path, status: dict[str, Any],
                    state: dict[str, Any], finite_gradient_checks: int,
                    started: float, fit_id: str, checkpoint_sha256: str | None,
                    trainable_names: list[str], batch_norm_names: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fit_root": str(fit_root.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "fit_id": fit_id,
        "status": status,
        "history": list(state["history"]),
        "global_step": int(state["global_step"]),
        "epoch_completed": int(state["epoch"]),
        "best_epoch": int(state["best_epoch"]),
        "best_validation_loss": None if not math.isfinite(state["best_metric"]) else float(state["best_metric"]),
        "finite_gradient_checks": finite_gradient_checks,
        "latest_checkpoint_sha256": checkpoint_sha256,
        "trainable_parameter_names": trainable_names,
        "frozen_batch_norm_modules": batch_norm_names,
        "selection_policy": status["selection_policy"],
        "fixed_epoch_count": status["fixed_epoch_count"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "ready_for_fit": bool(status["ready_for_fit"]),
    }
