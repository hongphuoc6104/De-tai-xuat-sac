"""Native full-bag multiple-instance learning models for histology features.

The module consumes frozen, per-patch feature matrices and learns only MIL
pooling/classification layers. It does not load an image encoder or use
patch-level targets. ``E0`` is the unprojected mean-pooling baseline, so its
returned embeddings have ``feature_dim`` elements; attention variants return
``projection_dim`` elements per lens.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LENS_KEYS: tuple[str, str, str] = ("4", "10", "40")
_VARIANTS = {"E0", "E1", "E2", "E3", "E4"}
_SINGLE_LENS_VARIANTS = {"E1": "4", "E2": "10", "E3": "40"}


@dataclass(frozen=True)
class ModelConfig:
    """Serializable configuration for one MIL model variant."""

    variant: str = "E4"
    projection_dim: int = 128
    attention_dim: int = 64
    dropout: float = 0.25
    feature_dim: int = 2048

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}.")
        for name in ("projection_dim", "attention_dim", "feature_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (float, int))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) <= 1.0
        ):
            raise ValueError("dropout must be a finite number between 0 and 1 inclusive.")
        object.__setattr__(self, "dropout", float(self.dropout))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this configuration."""
        return asdict(self)


class _GatedAttention(nn.Module):
    """Ilse-style gated attention scorer for one lens."""

    def __init__(self, projection_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.tanh_projection = nn.Linear(projection_dim, attention_dim)
        self.sigmoid_projection = nn.Linear(projection_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, instances: Tensor) -> Tensor:
        gated = torch.tanh(self.tanh_projection(instances)) * torch.sigmoid(
            self.sigmoid_projection(instances),
        )
        return self.score(gated).squeeze(-1)


class MILModel(nn.Module):
    """Classify one case from its per-lens frozen feature bags.

    ``features`` maps the string lens keys ``"4"``, ``"10"`` and ``"40"``
    to ``[instances, feature_dim]`` floating-point tensors. Empty matrices of
    shape ``[0, feature_dim]`` represent a missing lens for E0/E4. Single-lens
    variants require a non-empty matrix for their selected lens. If supplied,
    ``presence_mask`` follows ``LENS_KEYS`` order and must match the actual
    non-empty feature matrices.

    The returned attention and embedding mappings contain only the lenses used
    by the configured variant. E0 returns uniform mean-pooling weights and
    unprojected means; attention variants return learned weights and projected
    bag embeddings. Missing E0/E4 lenses have empty attention and zero vectors.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig instance.")
        self.config = config
        self.dropout = nn.Dropout(p=config.dropout)

        if config.variant == "E0":
            self.classifier = nn.Linear(config.feature_dim * 3 + 3, 1)
            self.projection = None
            self.attention_heads = nn.ModuleDict()
            self._used_lenses = LENS_KEYS
        elif config.variant == "E4":
            self.classifier = nn.Linear(config.projection_dim * 3 + 3, 1)
            self.projection = nn.Linear(config.feature_dim, config.projection_dim)
            self.attention_heads = nn.ModuleDict({
                lens: _GatedAttention(config.projection_dim, config.attention_dim)
                for lens in LENS_KEYS
            })
            self._used_lenses = LENS_KEYS
        else:
            lens = _SINGLE_LENS_VARIANTS[config.variant]
            self.classifier = nn.Linear(config.projection_dim, 1)
            self.projection = nn.Linear(config.feature_dim, config.projection_dim)
            self.attention_heads = nn.ModuleDict({
                lens: _GatedAttention(config.projection_dim, config.attention_dim),
            })
            self._used_lenses = (lens,)

    def get_config(self) -> dict[str, Any]:
        """Return the model build configuration in a JSON-safe form."""
        return self.config.to_dict()

    def forward(
        self,
        features: Mapping[str, Tensor],
        presence_mask: Sequence[bool] | None = None,
        *,
        attention_bias: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """Return a case-level logit, per-lens attention, and bag embeddings.

        ``attention_bias`` optionally adds a fixed per-instance logit to the
        learned attention score before softmax. Omitting it preserves the
        baseline forward path exactly.
        """
        matrices, actual_mask = self._validate_inputs(features, presence_mask)
        if self.config.variant == "E0" and attention_bias is not None:
            raise ValueError("E0 mean pooling does not support attention_bias.")
        biases = self._validate_attention_bias(attention_bias, matrices)
        missing_required = [
            lens for lens in self._used_lenses
            if lens not in matrices or matrices[lens].shape[0] == 0
        ]
        if missing_required and self.config.variant not in {"E0", "E4"}:
            raise ValueError(
                f"variant {self.config.variant} requires a non-empty {missing_required[0]}x feature matrix.",
            )

        if self.config.variant == "E0":
            embeddings, attention = self._mean_pool(matrices)
            mask = self._mask_tensor(actual_mask, self.classifier.weight)
            joined = torch.cat([*(embeddings[lens] for lens in LENS_KEYS), mask])
            logit = self.classifier(self.dropout(joined)).squeeze(0)
        else:
            embeddings, attention = self._attentive_pool(matrices, biases)
            if self.config.variant == "E4":
                mask = self._mask_tensor(actual_mask, self.classifier.weight)
                joined = torch.cat([*(embeddings[lens] for lens in LENS_KEYS), mask])
            else:
                joined = embeddings[self._used_lenses[0]]
            logit = self.classifier(self.dropout(joined)).squeeze(0)

        self._require_finite(logit, "logit")
        for lens, weights in attention.items():
            self._require_finite(weights, f"{lens}x attention")
        for lens, embedding in embeddings.items():
            self._require_finite(embedding, f"{lens}x embedding")
        return {"logit": logit, "attention": attention, "embeddings": embeddings}

    def _validate_attention_bias(
        self,
        attention_bias: Mapping[str, Tensor] | None,
        matrices: Mapping[str, Tensor],
    ) -> dict[str, Tensor] | None:
        if attention_bias is None:
            return None
        if not isinstance(attention_bias, Mapping):
            raise TypeError("attention_bias must be a lens-keyed mapping of per-instance tensors.")
        unknown = set(attention_bias) - set(self._used_lenses)
        if unknown:
            raise ValueError(f"attention_bias contains unused lens keys: {sorted(map(str, unknown))}.")
        biases: dict[str, Tensor] = {}
        for lens, bias in attention_bias.items():
            matrix = matrices.get(lens)
            if matrix is None or matrix.shape[0] == 0:
                raise ValueError(f"attention_bias[{lens!r}] was supplied for a missing lens bag.")
            if not isinstance(bias, Tensor):
                raise TypeError(f"attention_bias[{lens!r}] must be a torch.Tensor.")
            if bias.ndim != 1 or bias.shape[0] != matrix.shape[0]:
                raise ValueError(f"attention_bias[{lens!r}] must match the full instance count.")
            if bias.dtype != matrix.dtype or bias.device != matrix.device:
                raise ValueError(f"attention_bias[{lens!r}] must match feature dtype and device.")
            self._require_finite(bias, f"{lens}x attention bias")
            biases[lens] = bias
        return biases

    def _validate_inputs(
        self,
        features: Mapping[str, Tensor],
        presence_mask: Sequence[bool] | None,
    ) -> tuple[dict[str, Tensor], tuple[bool, bool, bool]]:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping keyed by '4', '10', and/or '40'.")
        unknown = set(features) - set(LENS_KEYS)
        if unknown:
            raise ValueError(f"features contains unknown lens keys: {sorted(map(str, unknown))}.")

        parameter = self.classifier.weight
        matrices: dict[str, Tensor] = {}
        for lens, matrix in features.items():
            if not isinstance(matrix, Tensor):
                raise TypeError(f"features[{lens!r}] must be a torch.Tensor.")
            if matrix.ndim != 2:
                raise ValueError(f"features[{lens!r}] must have shape [instances, feature_dim].")
            if matrix.shape[1] != self.config.feature_dim:
                raise ValueError(
                    f"features[{lens!r}] has feature dimension {matrix.shape[1]}, "
                    f"expected {self.config.feature_dim}.",
                )
            if not torch.is_floating_point(matrix):
                raise TypeError(f"features[{lens!r}] must use a floating-point dtype.")
            if matrix.dtype != parameter.dtype:
                raise TypeError(
                    f"features[{lens!r}] dtype {matrix.dtype} does not match model dtype {parameter.dtype}.",
                )
            if matrix.device != parameter.device:
                raise ValueError(
                    f"features[{lens!r}] device {matrix.device} does not match model device {parameter.device}.",
                )
            self._require_finite(matrix, f"features[{lens!r}]")
            matrices[lens] = matrix

        actual_mask = tuple(
            lens in matrices and matrices[lens].shape[0] > 0
            for lens in LENS_KEYS
        )
        if self.config.variant in {"E0", "E4"} and not any(actual_mask):
            raise ValueError("E0 and E4 require at least one non-empty lens feature matrix.")
        if presence_mask is not None:
            if not isinstance(presence_mask, (tuple, list)):
                raise TypeError("presence_mask must be a tuple or list of three booleans.")
            if len(presence_mask) != len(LENS_KEYS):
                raise ValueError("presence_mask must contain exactly three booleans in 4x/10x/40x order.")
            if any(type(value) is not bool for value in presence_mask):
                raise TypeError("presence_mask values must be booleans.")
            if tuple(presence_mask) != actual_mask:
                raise ValueError("presence_mask does not match the non-empty feature matrices.")
        return matrices, actual_mask  # type: ignore[return-value]

    def _mean_pool(
        self,
        matrices: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        reference = self.classifier.weight
        embeddings: dict[str, Tensor] = {}
        attention: dict[str, Tensor] = {}
        for lens in LENS_KEYS:
            matrix = matrices.get(lens)
            if matrix is None or matrix.shape[0] == 0:
                embeddings[lens] = reference.new_zeros(self.config.feature_dim)
                attention[lens] = reference.new_empty((0,))
            else:
                embeddings[lens] = matrix.mean(dim=0)
                attention[lens] = matrix.new_full((matrix.shape[0],), 1.0 / matrix.shape[0])
        return embeddings, attention

    def _attentive_pool(
        self,
        matrices: Mapping[str, Tensor],
        attention_bias: Mapping[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        assert self.projection is not None
        embeddings: dict[str, Tensor] = {}
        attention: dict[str, Tensor] = {}
        reference = self.classifier.weight
        for lens in self._used_lenses:
            matrix = matrices.get(lens)
            if matrix is None or matrix.shape[0] == 0:
                embeddings[lens] = reference.new_zeros(self.config.projection_dim)
                attention[lens] = reference.new_empty((0,))
                continue
            projected = F.relu(self.projection(matrix))
            self._require_finite(projected, f"{lens}x projected features")
            projected = self.dropout(projected)
            scores = self.attention_heads[lens](projected)
            self._require_finite(scores, f"{lens}x attention scores")
            if attention_bias is not None and lens in attention_bias:
                scores = scores + attention_bias[lens]
                self._require_finite(scores, f"{lens}x biased attention scores")
            weights = torch.softmax(scores, dim=0)
            self._require_finite(weights, f"{lens}x attention")
            embeddings[lens] = torch.sum(weights.unsqueeze(-1) * projected, dim=0)
            attention[lens] = weights
        return embeddings, attention

    @staticmethod
    def _mask_tensor(mask: tuple[bool, bool, bool], reference: Tensor) -> Tensor:
        return reference.new_tensor(mask)

    @staticmethod
    def _require_finite(value: Tensor, name: str) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values.")


__all__ = ["LENS_KEYS", "MILModel", "ModelConfig"]
