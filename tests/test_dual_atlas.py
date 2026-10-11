"""Contract tests for train-scoped clustering and atlas descriptors."""
from __future__ import annotations

import numpy as np
import pytest

from histology_data.dual_atlas import (
    assign_to_centers,
    cluster_case_label_crosstab,
    compare_clusterings,
    fit_case_classifier,
    fit_kmeans_dictionary,
    fit_ward_tree,
    guard_train_scope,
    hdbscan_resampling_stability,
    kmeans_seed_stability,
    photo_balanced_histogram,
    run_hdbscan_grid,
    ward_multicut,
)


def _toy_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(37)
    centers = np.asarray([
        [-3.0, 0.0, 0.0],
        [-2.5, 0.2, 0.0],
        [2.5, 0.0, 0.0],
        [3.0, 0.2, 0.0],
    ], dtype=np.float32)
    features = np.concatenate([
        center + rng.normal(0.0, 0.08, size=(4, 3)).astype(np.float32)
        for center in centers
    ])
    patient_ids = np.repeat(np.asarray(["p0", "p1", "p2", "p3"]), 4)
    photo_ids = np.concatenate([
        np.repeat(np.asarray([f"photo-{patient}-a", f"photo-{patient}-b"]), 2)
        for patient in ("p0", "p1", "p2", "p3")
    ])
    return features, patient_ids, photo_ids, centers


def test_toy_dictionary_assignment_histogram_and_case_head_form_a_complete_path() -> None:
    features, patient_ids, photo_ids, _ = _toy_rows()
    dictionary = fit_kmeans_dictionary(
        features,
        patient_ids,
        ["p0", "p1", "p2", "p3"],
        n_clusters=4,
        sample_weight=np.full(len(features), 1.0, dtype=np.float64),
        random_state=42,
        n_init=3,
        batch_size=8,
    )

    assigned, distances, margins = assign_to_centers(features, dictionary.cluster_centers_, chunk_size=3)
    histogram = photo_balanced_histogram(
        assigned,
        photo_ids,
        patient_ids,
        n_clusters=4,
    )
    targets = np.asarray([0, 0, 1, 1], dtype=np.int32)
    head = fit_case_classifier(
        histogram.descriptors,
        targets,
        histogram.patient_ids,
        ["p0", "p1", "p2", "p3"],
        regularization_c=0.1,
        random_state=42,
    )

    predictions = head.predict_proba(histogram.descriptors)
    assert dictionary.cluster_centers_.shape == (4, features.shape[1])
    assert distances.shape == margins.shape == (len(features),)
    assert np.isfinite(distances).all() and np.all(margins >= 0)
    assert histogram.full_histograms.shape == (4, 4)
    assert histogram.descriptors.shape == (4, 3)
    np.testing.assert_allclose(histogram.full_histograms.sum(axis=1), 1.0, atol=2e-6)
    assert predictions.shape == (4, 2)
    assert np.isfinite(predictions).all()


def test_photo_balancing_gives_each_source_photo_equal_weight() -> None:
    result = photo_balanced_histogram(
        np.asarray([0] * 10 + [1] * 6, dtype=np.int32),
        np.asarray(["p-a"] * 10 + ["p-b"] * 3 + ["p-c"] * 3),
        np.asarray(["case-a"] * 13 + ["case-b"] * 3),
        n_clusters=2,
    )

    assert result.patient_ids.tolist() == ["case-a", "case-b"]
    np.testing.assert_allclose(result.full_histograms, [[0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(result.descriptors, [[0.5], [0.0]])
    assert result.photo_counts.tolist() == [2, 1]
    permutation = np.random.default_rng(71).permutation(16)
    permuted = photo_balanced_histogram(
        np.asarray([0] * 10 + [1] * 6, dtype=np.int32)[permutation],
        np.asarray(["p-a"] * 10 + ["p-b"] * 3 + ["p-c"] * 3)[permutation],
        np.asarray(["case-a"] * 13 + ["case-b"] * 3)[permutation],
        n_clusters=2,
    )
    np.testing.assert_allclose(permuted.full_histograms, result.full_histograms)


def test_train_scope_guard_rejects_heldout_rows_and_overlapping_manifests() -> None:
    with pytest.raises(ValueError, match="outside the training scope"):
        guard_train_scope(["train", "heldout"], ["train"], context="dictionary")
    with pytest.raises(ValueError, match="forbidden held-out"):
        guard_train_scope(["train"], ["train", "heldout"], forbidden_patient_ids=["heldout"])
    with pytest.raises(TypeError, match="contain strings"):
        guard_train_scope([7], ["train"])


def test_kmeans_and_classifier_reject_out_of_scope_training_examples() -> None:
    features, patient_ids, _, _ = _toy_rows()
    expanded_patients = patient_ids.copy()
    expanded_patients[-1] = "heldout"
    with pytest.raises(ValueError, match="outside the training scope"):
        fit_kmeans_dictionary(
            features,
            expanded_patients,
            ["p0", "p1", "p2", "p3"],
            n_clusters=4,
        )
    with pytest.raises(ValueError, match="outside the training scope"):
        fit_case_classifier(
            np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            np.asarray([0, 1, 0], dtype=np.int32),
            ["p0", "p1", "heldout"],
            ["p0", "p1"],
            regularization_c=0.1,
        )


def test_kmeans_fails_closed_on_nonfinite_or_insufficient_distinct_rows() -> None:
    patients = ["p0", "p0", "p1"]
    repeated = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="distinct training vectors"):
        fit_kmeans_dictionary(repeated, patients, ["p0", "p1"], n_clusters=3)
    invalid = repeated.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_kmeans_dictionary(invalid, patients, ["p0", "p1"], n_clusters=2)


def test_center_assignment_is_chunk_invariant_and_margin_is_second_nearest_gap() -> None:
    centers = np.asarray([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    features = np.asarray([[0.5, 0.0], [3.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    small = assign_to_centers(features, centers, chunk_size=1)
    large = assign_to_centers(features, centers, chunk_size=10)

    for actual, expected in zip(small, large, strict=True):
        np.testing.assert_allclose(actual, expected)
    assert small[0].tolist() == [0, 1, 0]
    np.testing.assert_allclose(small[1], [0.5, 1.0, np.sqrt(5.0)], rtol=1e-6)
    assert np.all(small[2] >= 0)


def test_classifier_scaler_is_fit_only_on_the_declared_training_cases() -> None:
    descriptors = np.asarray([[-4.0, 1.0], [-2.0, 2.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    targets = np.asarray([0, 0, 1, 1], dtype=np.int32)
    model = fit_case_classifier(
        descriptors,
        targets,
        ["p0", "p1", "p2", "p3"],
        ["p0", "p1", "p2", "p3"],
        regularization_c=1.0,
    )

    np.testing.assert_allclose(model.named_steps["scaler"].mean_, descriptors.mean(axis=0))
    assert model.named_steps["classifier"].class_weight is None
    assert model.named_steps["classifier"].l1_ratio == 0.0


def test_density_grid_and_ward_multicut_report_geometry_case_and_photo_metrics() -> None:
    features, patient_ids, photo_ids, _ = _toy_rows()
    density_runs = run_hdbscan_grid(
        features,
        patient_ids,
        photo_ids,
        ["p0", "p1", "p2", "p3"],
        min_cluster_sizes=(3,),
        min_samples_values=(2,),
        selection_methods=("eom", "leaf"),
    )
    assert [run.config.selection_method for run in density_runs] == ["eom", "leaf"]
    for run in density_runs:
        assert run.summary.row_count == len(features)
        assert run.summary.coverage + run.summary.noise_fraction == pytest.approx(1.0)
        assert all(cluster.patient_count <= 4 and cluster.photo_count <= 8 for cluster in run.summary.clusters)

    tree = fit_ward_tree(
        features,
        patient_ids,
        ["p0", "p1", "p2", "p3"],
        photo_ids=photo_ids,
        maximum_rows=32,
    )
    cuts = ward_multicut(tree, features, [2, 4], photo_ids, patient_ids)
    assert [count for count, _, _ in cuts] == [2, 4]
    assert all(summary.coverage == 1.0 and summary.noise_fraction == 0.0 for _, _, summary in cuts)
    assert len(cuts[0][2].clusters) == 2
    assert all(cluster.mean_radius >= 0 and cluster.q90_radius >= cluster.mean_radius * 0 for cluster in cuts[1][2].clusters)


def test_partition_stability_reports_noise_excluded_ari_and_joint_coverage() -> None:
    comparisons = compare_clusterings({
        "first": np.asarray([-1, -1, 0, 0, 1, 1], dtype=np.int32),
        "second": np.asarray([-1, -1, 9, 9, 7, 7], dtype=np.int32),
    })
    comparison = comparisons[0]

    assert comparison.full_probe_ari == pytest.approx(1.0)
    assert comparison.jointly_assigned_ari == pytest.approx(1.0)
    assert comparison.jointly_assigned_count == 4
    assert comparison.jointly_assigned_fraction == pytest.approx(4 / 6)
    assert comparison.left_coverage == comparison.right_coverage == pytest.approx(4 / 6)
    all_noise = compare_clusterings({
        "one": np.asarray([-1, -1], dtype=np.int32),
        "two": np.asarray([-1, -1], dtype=np.int32),
    })[0]
    assert all_noise.jointly_assigned_ari is None
    assert all_noise.jointly_assigned_fraction == 0.0


def test_case_label_crosstab_keeps_noise_and_counts_cases_separately_from_rows() -> None:
    crosstab = cluster_case_label_crosstab(
        np.asarray([0, 0, 0, 1, -1, -1], dtype=np.int32),
        ["p0", "p0", "p1", "p1", "p2", "p3"],
        np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32),
    )

    assert [row.cluster_id for row in crosstab] == [-1, 0, 1]
    assert crosstab[1].row_counts_by_target == (2, 1)
    assert crosstab[1].patient_counts_by_target == (1, 1)
    assert crosstab[0].row_counts_by_target == (1, 1)
    assert crosstab[0].patient_counts_by_target == (1, 1)
    with pytest.raises(ValueError, match="conflicting source case labels"):
        cluster_case_label_crosstab(
            np.asarray([0, 1], dtype=np.int32),
            ["p0", "p0"],
            np.asarray([0, 1], dtype=np.int32),
        )


def test_seed_and_patient_resampling_stability_use_fixed_train_probe() -> None:
    features, patient_ids, photo_ids, _ = _toy_rows()
    seed_assignments, seed_scores = kmeans_seed_stability(
        features,
        patient_ids,
        ["p0", "p1", "p2", "p3"],
        features[:8],
        patient_ids[:8],
        n_clusters=4,
        seeds=(42, 43, 44),
        n_init=3,
        batch_size=8,
    )
    assert set(seed_assignments) == {"seed-42", "seed-43", "seed-44"}
    assert len(seed_scores) == 3
    assert all(score.jointly_assigned_count == 8 for score in seed_scores)

    density_assignments, density_scores = hdbscan_resampling_stability(
        features,
        patient_ids,
        photo_ids,
        ["p0", "p1", "p2", "p3"],
        probe_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        min_cluster_size=2,
        min_samples=1,
        selection_method="eom",
        seeds=(42, 43, 44),
        patient_fraction=0.5,
    )
    assert set(density_assignments) == {
        "patient-resample-42", "patient-resample-43", "patient-resample-44",
    }
    assert len(density_scores) == 3
    assert all(score.left_coverage <= 1 and score.right_coverage <= 1 for score in density_scores)


def test_photo_lineage_and_cluster_noise_fail_closed() -> None:
    with pytest.raises(ValueError, match="multiple patients"):
        photo_balanced_histogram(
            np.asarray([0, 1], dtype=np.int32),
            ["same-photo", "same-photo"],
            ["p0", "p1"],
            n_clusters=2,
        )
    with pytest.raises(ValueError, match="noise is not a KMeans histogram bin"):
        photo_balanced_histogram(
            np.asarray([-1, 0], dtype=np.int32),
            ["photo-a", "photo-b"],
            ["p0", "p1"],
            n_clusters=2,
        )


def test_ward_enforces_the_configured_quadratic_memory_cap() -> None:
    features, patient_ids, photo_ids, _ = _toy_rows()
    with pytest.raises(ValueError, match="above the configured cap"):
        fit_ward_tree(
            features,
            patient_ids,
            ["p0", "p1", "p2", "p3"],
            photo_ids=photo_ids,
            maximum_rows=8,
        )


def test_ward_tree_rejects_reordered_feature_rows_before_multicut() -> None:
    features, patient_ids, photo_ids, _ = _toy_rows()
    tree = fit_ward_tree(
        features,
        patient_ids,
        ["p0", "p1", "p2", "p3"],
        photo_ids=photo_ids,
    )

    with pytest.raises(ValueError, match="row order"):
        ward_multicut(
            tree,
            features[::-1].copy(),
            [2],
            photo_ids[::-1].copy(),
            patient_ids[::-1].copy(),
        )
