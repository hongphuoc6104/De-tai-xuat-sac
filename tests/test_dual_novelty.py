"""Synthetic contracts for B0 negative-reference banks and descriptors."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from histology_data.dual_novelty import (
    NegativePrototypeBank,
    equal_weight_member_ensemble,
    fit_linear_novelty_head,
    fit_negative_prototype_bank,
    fit_robust_novelty_scale,
    photo_balanced_descriptors,
    predict_linear_novelty_head,
    query_novelty,
    standardize_novelty,
    validate_anchor_head_membership,
)


def _bank(anchor: str = "negative-anchor") -> NegativePrototypeBank:
    vectors = np.eye(32, dtype=np.float32)
    return NegativePrototypeBank(
        lens="4",
        anchor_case_id=anchor,
        prototypes=vectors,
        prototype_row_ids=tuple(f"row-{index}" for index in range(32)),
        prototype_photo_ids=tuple(f"photo-{index % 4}" for index in range(32)),
    )


def _bank_metadata(size: int, *, anchor: str = "negative-anchor") -> list[dict[str, Any]]:
    return [
        {"case_id": anchor, "target": 0, "photo_id": f"photo-{index % 4}", "row_id": f"row-{index}"}
        for index in range(size)
    ]


def test_negative_bank_is_fixed_unique_photo_balanced_and_provenance_bound() -> None:
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(64, 32)).astype(np.float32)
    metadata = _bank_metadata(64)

    bank = fit_negative_prototype_bank(vectors, metadata, lens="4", anchor_case_id="negative-anchor")

    assert bank.prototypes.shape == (32, 32)
    assert len(set(bank.prototype_row_ids)) == 32
    assert len(bank.bank_id) == 64
    assert bank.lens == "4" and bank.anchor_case_id == "negative-anchor"
    with pytest.raises(ValueError, match="does not match"):
        NegativePrototypeBank(
            lens=bank.lens,
            anchor_case_id=bank.anchor_case_id,
            prototypes=bank.prototypes.copy(),
            prototype_row_ids=bank.prototype_row_ids,
            prototype_photo_ids=bank.prototype_photo_ids,
            bank_id="forged",
        )


def test_bank_fails_closed_on_mixed_member_positive_or_duplicate_rows() -> None:
    vectors = np.eye(32, dtype=np.float32)
    mixed: list[Mapping[str, Any]] = _bank_metadata(32)
    mixed[0] = {**mixed[0], "case_id": "another-case"}
    with pytest.raises(ValueError, match="anchor case only"):
        fit_negative_prototype_bank(vectors, mixed, lens="4", anchor_case_id="negative-anchor")

    positive = _bank_metadata(32)
    positive[0] = {**positive[0], "target": 1}
    with pytest.raises(ValueError, match="target 0"):
        fit_negative_prototype_bank(vectors, positive, lens="4", anchor_case_id="negative-anchor")

    repeated = np.eye(32, dtype=np.float32)
    repeated[31] = repeated[0]
    with pytest.raises(ValueError, match="distinct vectors"):
        fit_negative_prototype_bank(repeated, _bank_metadata(32), lens="4", anchor_case_id="negative-anchor")


def test_chunked_nearest_euclidean_scores_match_exact_reference_and_permutation() -> None:
    bank = _bank()
    queries = np.vstack([bank.prototypes[:5], np.full((3, 32), 0.125, dtype=np.float32)])
    direct = np.linalg.norm(queries[:, None, :] - bank.prototypes[None, :, :], axis=2)

    whole = query_novelty(queries, bank, chunk_size=8)
    chunked = query_novelty(queries, bank, chunk_size=2)
    permutation = np.asarray([6, 1, 4, 0, 7, 2, 5, 3])
    permuted = query_novelty(queries[permutation], bank, chunk_size=1)

    np.testing.assert_allclose(whole.distances, direct.min(axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunked.distances, whole.distances, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(chunked.nearest_prototype_indices, whole.nearest_prototype_indices)
    np.testing.assert_allclose(permuted.distances[np.argsort(permutation)], whole.distances, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        permuted.nearest_prototype_indices[np.argsort(permutation)], whole.nearest_prototype_indices,
    )


def test_photo_balanced_descriptors_give_each_source_photo_equal_weight() -> None:
    scores = np.asarray([0.0] * 100 + [100.0], dtype=np.float32)
    photos = ["large-photo"] * 100 + ["small-photo"]

    descriptors = photo_balanced_descriptors(scores, photos)

    assert descriptors.median == pytest.approx(50.0)
    assert descriptors.q90 == pytest.approx(50.0)
    assert descriptors.q95 == pytest.approx(50.0)
    assert descriptors.tail5_mean == pytest.approx(50.0)
    assert np.isfinite(descriptors.as_array()).all()


def test_train_only_t_minus_anchor_scale_has_finite_zero_iqr_policy() -> None:
    constant = np.asarray([2.0] * 4, dtype=np.float32)
    photos = ["a", "a", "b", "b"]
    training_metadata = [
        {"case_id": "negative-2", "partition": "train"},
        {"case_id": "negative-2", "partition": "train"},
        {"case_id": "positive-1", "partition": "train"},
        {"case_id": "positive-1", "partition": "train"},
    ]
    scale = fit_robust_novelty_scale(
        constant,
        photos,
        training_metadata,
        anchor_case_id="negative-anchor",
        allowed_train_case_ids=("negative-2", "positive-1"),
    )

    assert scale.degenerate and scale.iqr == 0.0
    np.testing.assert_array_equal(standardize_novelty(constant, scale), np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="Anchor rows"):
        fit_robust_novelty_scale(
            constant,
            photos,
            [{"case_id": "negative-anchor", "partition": "train"}] * 4,
            anchor_case_id="negative-anchor",
            allowed_train_case_ids=("negative-2", "positive-1"),
        )
    with pytest.raises(ValueError, match="training rows only"):
        fit_robust_novelty_scale(
            constant,
            photos,
            [{"case_id": "held-out", "partition": "validation"}] * 4,
            anchor_case_id="negative-anchor",
            allowed_train_case_ids=("held-out",),
        )

    with pytest.raises(ValueError, match="outside the explicit"):
        fit_robust_novelty_scale(
            constant,
            photos,
            [{"case_id": "held-out", "partition": "train"}] * 4,
            anchor_case_id="negative-anchor",
            allowed_train_case_ids=("negative-2", "positive-1"),
        )


def test_robust_scale_balances_case_then_photo_then_rows() -> None:
    values = np.asarray([0.0, 10.0, 10.0, 10.0], dtype=np.float32)
    photos = ["a-photo", "b-photo-1", "b-photo-2", "b-photo-3"]
    metadata = [
        {"case_id": "case-a", "partition": "train"},
        {"case_id": "case-b", "partition": "train"},
        {"case_id": "case-b", "partition": "train"},
        {"case_id": "case-b", "partition": "train"},
    ]

    scale = fit_robust_novelty_scale(
        values,
        photos,
        metadata,
        anchor_case_id="negative-anchor",
        allowed_train_case_ids=("case-a", "case-b"),
    )

    assert scale.median == 0.0
    assert scale.iqr == 10.0


def test_head_training_excludes_anchor_enforces_class_gate_and_binds_member_bank() -> None:
    bank = _bank()
    descriptors = np.asarray(
        [[0.0, 0.1, 0.2], [0.2, 0.1, 0.0], [1.0, 1.2, 0.8], [0.8, 1.0, 1.1]],
        dtype=np.float32,
    )
    case_ids = ("negative-2", "negative-3", "positive-1", "positive-2")
    targets = (0, 0, 1, 1)
    head = fit_linear_novelty_head(
        descriptors,
        targets,
        case_ids,
        bank=bank,
        anchor_case_id=bank.anchor_case_id,
        c=0.1,
    )

    probability = predict_linear_novelty_head(
        head, descriptors[:2], case_ids[:2], bank=bank,
    )
    assert probability.shape == (2,)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    with pytest.raises(ValueError, match="excluded from"):
        validate_anchor_head_membership(
            (*case_ids, bank.anchor_case_id), (*targets, 0), anchor_case_id=bank.anchor_case_id,
        )
    with pytest.raises(ValueError, match="at least 2"):
        validate_anchor_head_membership(
            case_ids[:3], targets[:3], anchor_case_id=bank.anchor_case_id,
        )
    with pytest.raises(ValueError, match="does not belong"):
        predict_linear_novelty_head(head, descriptors[:1], ("query",), bank=_bank("other-anchor"))
    with pytest.raises(ValueError, match="self-scored"):
        predict_linear_novelty_head(
            head, descriptors[:1], (bank.anchor_case_id,), bank=bank,
        )


def test_member_ensemble_is_equal_weight_and_requires_aligned_queries() -> None:
    values = equal_weight_member_ensemble([
        np.asarray([0.0, 0.2], dtype=np.float32),
        np.asarray([1.0, 0.8], dtype=np.float32),
    ], member_query_case_ids=[("case-a", "case-b"), ("case-a", "case-b")])
    np.testing.assert_allclose(values, np.asarray([0.5, 0.5], dtype=np.float32))
    with pytest.raises(ValueError, match="same ordered query case IDs"):
        equal_weight_member_ensemble([
            np.asarray([0.1, 0.2], dtype=np.float32), np.asarray([0.2, 0.3], dtype=np.float32),
        ], member_query_case_ids=[("case-a", "case-b"), ("case-b", "case-a")])
