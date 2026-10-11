"""Contract tests for native full-bag MIL model variants."""
from __future__ import annotations

import json

import pytest
import torch

from histology_data.mil_model import LENS_KEYS, MILModel, ModelConfig

VARIANTS = ("E0", "E1", "E2", "E3", "E4")


def _features(*, feature_dim: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    return {
        "4": torch.randn((5, feature_dim), generator=generator),
        "10": torch.randn((3, feature_dim), generator=generator),
        "40": torch.randn((4, feature_dim), generator=generator),
    }


def _config(variant: str, *, dropout: float = 0.1) -> ModelConfig:
    return ModelConfig(
        variant=variant,
        projection_dim=4,
        attention_dim=3,
        dropout=dropout,
        feature_dim=6,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_returns_finite_case_logit_and_expected_bag_shapes(variant: str) -> None:
    model = MILModel(_config(variant)).eval()
    result = model(_features())
    used_lenses = LENS_KEYS if variant in {"E0", "E4"} else {
        "E1": ("4",), "E2": ("10",), "E3": ("40",),
    }[variant]

    assert set(result) == {"logit", "attention", "embeddings"}
    assert result["logit"].shape == torch.Size([])
    assert torch.isfinite(result["logit"])
    assert set(result["attention"]) == set(used_lenses)
    assert set(result["embeddings"]) == set(used_lenses)
    expected_dim = 6 if variant == "E0" else 4
    assert all(result["embeddings"][lens].shape == (expected_dim,) for lens in used_lenses)
    assert all(torch.isfinite(value).all() for value in result["embeddings"].values())
    assert all(weights.shape == (_features()[lens].shape[0],) for lens, weights in result["attention"].items())


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_supports_case_level_gradient_updates(variant: str) -> None:
    torch.manual_seed(8)
    model = MILModel(_config(variant))
    result = model(_features())
    result["logit"].square().backward()

    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
    if variant != "E0":
        assert model.projection is not None
        assert model.projection.weight.grad is not None
        lens = {"E1": "4", "E2": "10", "E3": "40", "E4": "4"}[variant]
        assert model.attention_heads[lens].score.weight.grad is not None


@pytest.mark.parametrize("variant", VARIANTS)
def test_eval_predictions_are_invariant_to_patch_order(variant: str) -> None:
    torch.manual_seed(21)
    model = MILModel(_config(variant)).eval()
    inputs = _features()
    permuted = {
        lens: matrix[torch.randperm(matrix.shape[0], generator=torch.Generator().manual_seed(i + 40))]
        for i, (lens, matrix) in enumerate(inputs.items())
    }

    original_logit = model(inputs)["logit"]
    permuted_logit = model(permuted)["logit"]
    torch.testing.assert_close(permuted_logit, original_logit, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_present_lens_attention_weights_sum_to_one(variant: str) -> None:
    result = MILModel(_config(variant)).eval()(_features())
    for weights in result["attention"].values():
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))


def test_e4_accepts_missing_lens_and_uses_zero_embedding_and_empty_attention() -> None:
    model = MILModel(_config("E4")).eval()
    inputs = {"4": torch.randn(2, 6), "10": torch.empty((0, 6)), "40": torch.randn(3, 6)}
    result = model(inputs, presence_mask=(True, False, True))

    assert result["embeddings"]["10"].shape == (4,)
    assert torch.equal(result["embeddings"]["10"], torch.zeros(4))
    assert result["attention"]["10"].shape == (0,)
    assert torch.isfinite(result["logit"])
    with pytest.raises(ValueError, match="does not match"):
        model(inputs, presence_mask=(True, True, True))


def test_e0_uses_unprojected_means_and_uniform_pooling_weights() -> None:
    result = MILModel(_config("E0")).eval()(_features())

    for lens, matrix in _features().items():
        torch.testing.assert_close(result["embeddings"][lens], matrix.mean(dim=0))
        torch.testing.assert_close(
            result["attention"][lens],
            torch.full((matrix.shape[0],), 1 / matrix.shape[0]),
        )


@pytest.mark.parametrize("variant", ("E0", "E4"))
def test_fusion_variants_reject_a_case_with_no_features(variant: str) -> None:
    empty = {lens: torch.empty((0, 6)) for lens in LENS_KEYS}
    with pytest.raises(ValueError, match="at least one non-empty"):
        MILModel(_config(variant))(empty, presence_mask=(False, False, False))


@pytest.mark.parametrize(
    ("variant", "required_lens"),
    (("E1", "4"), ("E2", "10"), ("E3", "40")),
)
def test_single_lens_variants_require_their_selected_bag(variant: str, required_lens: str) -> None:
    inputs = {lens: torch.randn(2, 6) for lens in LENS_KEYS if lens != required_lens}
    with pytest.raises(ValueError, match=f"requires a non-empty {required_lens}x"):
        MILModel(_config(variant))(inputs)


@pytest.mark.parametrize(
    ("inputs", "exception", "message"),
    [
        ({"4": torch.ones(6)}, ValueError, "shape \\[instances"),
        ({"4": torch.ones((2, 7))}, ValueError, "feature dimension 7"),
        ({"4": torch.ones((2, 6), dtype=torch.int64)}, TypeError, "floating-point dtype"),
        ({"4": torch.full((2, 6), float("nan"))}, FloatingPointError, "non-finite"),
        ({"4X": torch.ones((2, 6))}, ValueError, "unknown lens keys"),
    ],
)
def test_invalid_feature_payloads_fail_closed(
    inputs: dict[str, torch.Tensor],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        MILModel(_config("E4"))(inputs)


def test_presence_mask_requires_three_actual_boolean_values() -> None:
    model = MILModel(_config("E4"))
    inputs = {"4": torch.randn(2, 6)}

    with pytest.raises(ValueError, match="exactly three"):
        model(inputs, presence_mask=(True, False))
    with pytest.raises(TypeError, match="values must be booleans"):
        model(inputs, presence_mask=(1, False, False))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        model(inputs, presence_mask=(True, True, False))


def test_config_is_json_serializable_and_classifier_keeps_intercept() -> None:
    model = MILModel(_config("E4"))
    serialized = json.dumps(model.get_config(), sort_keys=True)

    assert json.loads(serialized) == _config("E4").to_dict()
    assert model.classifier.bias is not None


def test_default_feature_and_classifier_dimensions_match_model_contract() -> None:
    baseline = MILModel(ModelConfig(variant="E0"))
    fusion = MILModel(ModelConfig())

    assert baseline.classifier.in_features == 6147
    assert fusion.classifier.in_features == 387
    assert fusion.projection is not None and fusion.projection.in_features == 2048
    assert fusion.projection.out_features == 128


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "E5"},
        {"projection_dim": 0},
        {"attention_dim": -1},
        {"feature_dim": 1.5},
        {"dropout": float("nan")},
        {"dropout": 1.1},
    ],
)
def test_invalid_model_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)  # type: ignore[arg-type]
