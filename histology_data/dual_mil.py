"""Full-bag MIL with frozen, train-scope reference-novelty attention bias."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .dual_novelty import NegativePrototypeBank, RobustNoveltyScale
from .mil_model import LENS_KEYS, MILModel, ModelConfig


def _float32_array(value: Any, name: str, *, ndim: int) -> NDArray[np.float32]:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32; received {array.dtype}.")
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NoveltyLensState:
    """Frozen L2/PCA32 transform, member bank, and train-only robust scale."""

    mean: NDArray[np.float32]
    components: NDArray[np.float32]
    bank: NegativePrototypeBank
    scale: RobustNoveltyScale

    def __post_init__(self) -> None:
        mean = _float32_array(self.mean, "mean", ndim=1).copy()
        components = _float32_array(self.components, "components", ndim=2).copy()
        if components.shape != (32, mean.size):
            raise ValueError("Frozen PCA components must have shape [32, raw_feature_dim].")
        if not isinstance(self.bank, NegativePrototypeBank):
            raise TypeError("bank must be a NegativePrototypeBank.")
        if self.bank.prototypes.shape != (32, 32):
            raise ValueError("B1 requires exactly 32 distinct prototypes in the PCA32 space per lens.")
        if not isinstance(self.scale, RobustNoveltyScale):
            raise TypeError("scale must be a RobustNoveltyScale fit from T-minus-anchor training rows.")
        mean.setflags(write=False)
        components.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "components", components)


class _FrozenNoveltyLens(nn.Module):
    """Torch buffers for one lens; no transform, bank, or scale parameter is trainable."""

    def __init__(self, state: NoveltyLensState) -> None:
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(state.mean.copy()))
        self.register_buffer("components", torch.from_numpy(state.components.copy()))
        self.register_buffer("prototypes", torch.from_numpy(state.bank.prototypes.copy()))
        self.register_buffer("scale_median", torch.tensor(state.scale.median, dtype=torch.float32))
        self.register_buffer("scale_iqr", torch.tensor(state.scale.iqr, dtype=torch.float32))
        self.register_buffer("scale_epsilon", torch.tensor(state.scale.epsilon, dtype=torch.float32))
        self.register_buffer("scale_degenerate", torch.tensor(state.scale.degenerate, dtype=torch.bool))
        self.bank_id = state.bank.bank_id
        self.anchor_case_id = state.bank.anchor_case_id
        self.prototype_row_ids = state.bank.prototype_row_ids
        self.prototype_photo_ids = state.bank.prototype_photo_ids

    def forward(self, features: Tensor, *, epsilon: float = 1e-12) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("Novelty input must have shape [instances, raw_feature_dim].")
        if features.dtype != self.mean.dtype or features.device != self.mean.device:
            raise TypeError("Novelty input dtype/device must match the frozen transform buffers.")
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("Novelty input contains non-finite values.")
        norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        if bool(torch.any(norms <= epsilon).item()):
            raise ValueError("Zero-norm feature rows cannot enter the L2/PCA novelty transform.")
        unit = features / norms
        projected = (unit - self.mean) @ self.components.transpose(0, 1)
        distances = torch.cdist(projected, self.prototypes, p=2).amin(dim=1)
        if bool(self.scale_degenerate.item()):
            normalized = torch.zeros_like(distances)
        else:
            normalized = (distances - self.scale_median) / self.scale_iqr
        normalized = torch.clamp(normalized, min=-3.0, max=3.0).detach()
        if not bool(torch.isfinite(normalized).all().item()):
            raise FloatingPointError("Normalized novelty scores contain non-finite values.")
        return normalized


def novelty_model_identity(
    config: ModelConfig,
    lens_states: Mapping[str, NoveltyLensState],
    *,
    beta: float,
    member_id: str,
) -> dict[str, Any]:
    """Build the custom model identity without initializing a trainable model."""
    if not isinstance(config, ModelConfig) or config.variant != "E4":
        raise ValueError("NoveltyMILModel requires the E4 multi-lens architecture.")
    if isinstance(beta, bool) or not isinstance(beta, (float, int)) or not math.isfinite(beta):
        raise ValueError("beta must be a finite number.")
    if float(beta) not in (0.0, 0.5):
        raise ValueError("The approved B1 candidates are beta=0.0 and beta=0.5.")
    if not isinstance(member_id, str) or not member_id.strip():
        raise ValueError("member_id must be a non-empty opaque member token.")
    if not isinstance(lens_states, Mapping) or not lens_states:
        raise ValueError("At least one frozen per-lens novelty state is required.")
    if set(lens_states) - set(LENS_KEYS):
        raise ValueError(f"Novelty states contain unsupported lenses: {sorted(set(lens_states) - set(LENS_KEYS))}.")
    for lens, state in lens_states.items():
        if not isinstance(state, NoveltyLensState):
            raise TypeError(f"lens_states[{lens!r}] must be a NoveltyLensState.")
        if state.bank.lens != lens:
            raise ValueError(f"Novelty bank lens {state.bank.lens!r} does not match key {lens!r}.")
    anchor_ids = {state.bank.anchor_case_id for state in lens_states.values()}
    if len(anchor_ids) != 1:
        raise ValueError("All per-lens banks in one model member must share the same negative anchor.")
    raw_dims = {state.mean.size for state in lens_states.values()}
    if raw_dims != {config.feature_dim}:
        raise ValueError("Every novelty transform raw dimension must match ModelConfig.feature_dim.")

    anchor_token = hashlib.sha256(next(iter(anchor_ids)).encode("utf-8")).hexdigest()
    lens_identity = {
        lens: {
            "bank_id": lens_states[lens].bank.bank_id,
            "prototype_row_ids_sha256": hashlib.sha256(
                json.dumps(lens_states[lens].bank.prototype_row_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "mean_sha256": _array_hash(lens_states[lens].mean),
            "components_sha256": _array_hash(lens_states[lens].components),
            "prototypes_sha256": _array_hash(lens_states[lens].bank.prototypes),
            "scale": {
                "median": lens_states[lens].scale.median,
                "iqr": lens_states[lens].scale.iqr,
                "epsilon": lens_states[lens].scale.epsilon,
                "degenerate": lens_states[lens].scale.degenerate,
            },
        }
        for lens in LENS_KEYS if lens in lens_states
    }
    return {
        "type": "NoveltyMILModel",
        "core": config.to_dict(),
        "member_id": member_id,
        "anchor_token": anchor_token,
        "beta": float(beta),
        "novelty_policy": {
            "transform": "l2_then_frozen_centered_pca32_v1",
            "l2_epsilon": 1e-12,
            "pca_components": 32,
            "prototypes_per_lens": 32,
            "distance": "nearest_prototype_euclidean",
            "standardization": "train_minus_anchor_median_iqr_v1",
            "clip": [-3.0, 3.0],
            "patch_selection": "none_full_bag",
        },
        "lenses": lens_identity,
    }


class NoveltyMILModel(nn.Module):
    """E4-size learned attention with ``beta * novelty`` added over each full bag.

    The frozen transform is L2 normalization followed by the supplied PCA32
    projection. Novelty is the nearest Euclidean distance to one member's
    negative bank, standardized with its T-minus-anchor median/IQR and clipped
    to [-3, 3]. No instances are selected or removed.
    """

    def __init__(
        self,
        config: ModelConfig,
        lens_states: Mapping[str, NoveltyLensState],
        *,
        beta: float,
        member_id: str,
    ) -> None:
        super().__init__()
        identity = novelty_model_identity(config, lens_states, beta=beta, member_id=member_id)

        self.config = config
        self.beta = identity["beta"]
        self.member_id = member_id
        self._model_identity = identity
        self.core = MILModel(config)
        self.novelty_lenses = nn.ModuleDict({
            lens: _FrozenNoveltyLens(lens_states[lens]) for lens in LENS_KEYS if lens in lens_states
        })

    def get_config(self) -> dict[str, Any]:
        """Return stable identity for trainer resume and member-bank ownership."""
        return json.loads(json.dumps(self._model_identity))

    def validate_fit_scope(self, train: Any, validation: Any | None, *, mode: str) -> None:
        """Enforce anchor exclusion and production member class counts pre-write."""
        if mode not in {"smoke", "inner", "refit"}:
            raise ValueError("mode must be smoke, inner or refit.")
        anchor = next(iter(self.novelty_lenses.values())).anchor_case_id
        train_ids = tuple(train.case_ids)
        if anchor in train_ids:
            raise ValueError("Negative bank anchor must be excluded from supervised head training.")
        if validation is not None and anchor in validation.case_ids:
            raise ValueError("Negative bank anchor cannot be self-scored in validation/query data.")
        minimum = 1 if mode == "smoke" else 2
        counts = {
            target: sum(train.records[case_id]["target"] == target for case_id in train_ids)
            for target in (0, 1)
        }
        if any(counts[target] < minimum for target in (0, 1)):
            raise ValueError(
                f"Novelty MIL member needs at least {minimum} supervised cases from each class; got {counts}."
            )

    def validate_query_scope(self, dataset: Any) -> None:
        """Reject any direct evaluation request that would self-match its bank anchor."""
        anchor = next(iter(self.novelty_lenses.values())).anchor_case_id
        if anchor in dataset.case_ids:
            raise ValueError("Negative bank anchor cannot be self-scored as validation/query data.")

    def forward(
        self,
        features: Mapping[str, Tensor],
        presence_mask: tuple[bool, bool, bool] | list[bool] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a lens-keyed mapping.")
        novelty: dict[str, Tensor] = {}
        for lens, matrix in features.items():
            if not isinstance(matrix, Tensor):
                raise TypeError(f"features[{lens!r}] must be a torch.Tensor.")
            if lens not in self.novelty_lenses:
                if matrix.shape[0] > 0:
                    raise ValueError(f"No frozen novelty state exists for non-empty {lens!r}x bag.")
                continue
            if matrix.shape[0] > 0:
                novelty[lens] = self.novelty_lenses[lens](matrix)
        biases = {lens: self.beta * values for lens, values in novelty.items()}
        output = self.core(features, presence_mask=presence_mask, attention_bias=biases)
        output["novelty_z"] = novelty
        output["attention_bias"] = biases
        return output


def make_novelty_model_factory(
    lens_states: Mapping[str, NoveltyLensState],
    *,
    beta: float,
    member_id: str,
) -> Callable[[Any], NoveltyMILModel]:
    """Create the explicit ``model_factory(TrainConfig)`` accepted by ``run_fit``."""
    frozen_states = dict(lens_states)

    def factory(train_config: Any) -> NoveltyMILModel:
        model_config = getattr(train_config, "model", None)
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_factory expects a TrainConfig with a ModelConfig in .model.")
        return NoveltyMILModel(model_config, frozen_states, beta=beta, member_id=member_id)

    return factory


__all__ = ["NoveltyLensState", "NoveltyMILModel", "make_novelty_model_factory", "novelty_model_identity"]
