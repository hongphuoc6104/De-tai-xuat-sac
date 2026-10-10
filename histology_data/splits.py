"""Deterministic nested grouped splits with patient and duplicate isolation."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from .io import fingerprint

LABELS = (0, 1)


class SplitError(ValueError):
    """A nested patient-grouped split cannot satisfy the locked protocol."""


class _UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        first, second = sorted((a, b))
        self.parent[second] = first


def _case_counts(cases: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for case in cases:
        label = case.get("case_label")
        if isinstance(label, bool):
            raise SplitError(f"Case {case.get('case_id')} has boolean label; expected reviewed integer 0 or 1.")
        if isinstance(label, str) and label in {"0", "1"}:
            label = int(label)
        if label not in LABELS:
            raise SplitError(f"Case {case.get('case_id')} has no reviewed binary case label.")
        counts[label] += 1
    return counts


def _group_records(
    cases: list[dict[str, Any]], duplicate_patient_links: list[list[str]],
) -> list[dict[str, Any]]:
    patients = {str(case["patient_id"]) for case in cases}
    union = _UnionFind(patients)
    for link in duplicate_patient_links:
        members = sorted({str(patient) for patient in link if str(patient) in patients})
        for member in members[1:]:
            union.union(members[0], member)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    patient_members: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        patient = str(case["patient_id"])
        root = union.find(patient)
        grouped[root].append(case)
        patient_members[root].add(patient)
    result = []
    for root, members in grouped.items():
        component_patients = sorted(patient_members[root])
        counts = _case_counts(members)
        result.append({
            "split_group_id": "group-" + fingerprint(component_patients)[:20],
            "patients": component_patients,
            "cases": sorted(str(case["case_id"]) for case in members),
            "case_labels": {str(case["case_id"]): int(case["case_label"]) for case in members},
            "class_counts": {str(label): counts[label] for label in LABELS},
            "case_count": len(members),
        })
    return sorted(result, key=lambda group: group["split_group_id"])


def _assign_group_folds(groups: list[dict[str, Any]], folds: int, seed: int) -> list[list[dict[str, Any]]]:
    """Use sklearn's global stratified-group assignment over case rows.

    Each case keeps its own label while `split_group_id` joins all cases of a
    patient and any explicitly preserved cross-patient duplicate component.
    """
    if folds < 2:
        raise SplitError("At least two folds are required.")
    if len(groups) < folds:
        raise SplitError(f"Only {len(groups)} independent patient/duplicate groups for {folds} folds.")
    total_counts: Counter[int] = Counter()
    for group in groups:
        total_counts.update({int(key): value for key, value in group["class_counts"].items()})
    group_support = {
        label: sum(int(group["class_counts"].get(str(label), 0)) > 0 for group in groups)
        for label in LABELS
    }
    for label in LABELS:
        if total_counts[label] < folds or group_support[label] < folds:
            raise SplitError(
                f"Class {label} cannot appear in every one of {folds} folds "
                f"(cases={total_counts[label]}, independent_groups={group_support[label]})."
            )
    case_ids = [case_id for group in groups for case_id in group["cases"]]
    labels = [int(group["case_labels"][case_id]) for group in groups for case_id in group["cases"]]
    group_ids = [group["split_group_id"] for group in groups for _ in group["cases"]]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_groups: list[list[dict[str, Any]]] = []
    case_to_group = {case_id: group["split_group_id"] for group in groups for case_id in group["cases"]}
    group_by_id = {group["split_group_id"]: group for group in groups}
    try:
        for _, fold_indices in splitter.split(np.zeros(len(case_ids)), labels, group_ids):
            selected_group_ids = {case_to_group[case_ids[index]] for index in fold_indices}
            fold_groups.append([group_by_id[group_id] for group_id in sorted(selected_group_ids)])
    except ValueError as exc:
        raise SplitError(f"Stratified patient-group split failed: {exc}") from exc
    if len(fold_groups) != folds or any(not fold for fold in fold_groups):
        raise SplitError("Stratified patient-group split produced an empty fold.")
    for index, fold in enumerate(fold_groups):
        fold_labels = {
            int(label)
            for group in fold
            for label in group["case_labels"].values()
        }
        if fold_labels != set(LABELS):
            raise SplitError(f"Fold {index} lacks one or more case classes.")
    return fold_groups


def _cases_for_groups(groups: list[dict[str, Any]]) -> list[str]:
    return sorted(case for group in groups for case in group["cases"])


def _patients_for_groups(groups: list[dict[str, Any]]) -> list[str]:
    return sorted({patient for group in groups for patient in group["patients"]})


def create_nested_patient_splits(
    cases: list[dict[str, Any]],
    *,
    duplicate_patient_links: list[list[str]] | None = None,
    seed: int = 42,
    outer_folds: int = 3,
    inner_folds: int = 2,
) -> dict[str, Any]:
    """Create 3x2 grouped CV or fail closed without removing patients/cases."""
    if not cases:
        raise SplitError("Cannot split an empty cohort.")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise SplitError("Case IDs must be present and unique in the selected cohort.")
    if any(not str(case.get("patient_id", "")).strip() for case in cases):
        raise SplitError("Every selected case requires a reviewed patient_id.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SplitError("seed must be a nonnegative integer.")
    for name, value in (("outer_folds", outer_folds), ("inner_folds", inner_folds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise SplitError(f"{name} must be an integer of at least 2.")
    groups = _group_records(cases, duplicate_patient_links or [])
    _case_counts(cases)
    outer_assignment = _assign_group_folds(groups, outer_folds, seed)
    outer_rows = []
    for outer_index, test_groups in enumerate(outer_assignment):
        test_group_ids = {group["split_group_id"] for group in test_groups}
        train_groups = [group for group in groups if group["split_group_id"] not in test_group_ids]
        if not train_groups:
            raise SplitError(f"Outer fold {outer_index} has no training groups.")
        train_cases = _cases_for_groups(train_groups)
        test_cases = _cases_for_groups(test_groups)
        _require_classes(cases, train_cases, f"outer {outer_index} train")
        _require_classes(cases, test_cases, f"outer {outer_index} test")
        inner_assignment = _assign_group_folds(train_groups, inner_folds, seed)
        inner_rows = []
        for inner_index, validation_groups in enumerate(inner_assignment):
            val_group_ids = {group["split_group_id"] for group in validation_groups}
            inner_train_groups = [group for group in train_groups if group["split_group_id"] not in val_group_ids]
            inner_train_cases = _cases_for_groups(inner_train_groups)
            validation_cases = _cases_for_groups(validation_groups)
            _require_classes(cases, inner_train_cases, f"outer {outer_index} inner {inner_index} train")
            _require_classes(cases, validation_cases, f"outer {outer_index} inner {inner_index} validation")
            inner_rows.append({
                "fold": inner_index,
                "train_cases": inner_train_cases,
                "validation_cases": validation_cases,
                "train_patients": _patients_for_groups(inner_train_groups),
                "validation_patients": _patients_for_groups(validation_groups),
                "train_split_groups": sorted(group["split_group_id"] for group in inner_train_groups),
                "validation_split_groups": sorted(group["split_group_id"] for group in validation_groups),
            })
        outer_rows.append({
            "fold": outer_index,
            "train_cases": train_cases,
            "test_cases": test_cases,
            "train_patients": _patients_for_groups(train_groups),
            "test_patients": _patients_for_groups(test_groups),
            "train_split_groups": sorted(group["split_group_id"] for group in train_groups),
            "test_split_groups": sorted(test_group_ids),
            "inner_folds": inner_rows,
        })
    assignments = {}
    group_for_patient = {patient: group["split_group_id"] for group in groups for patient in group["patients"]}
    for case in sorted(cases, key=lambda row: str(row["case_id"])):
        patient = str(case["patient_id"])
        test_fold = next(index for index, fold in enumerate(outer_rows) if str(case["case_id"]) in fold["test_cases"])
        assignments[str(case["case_id"])] = {
            "patient_id": patient,
            "split_group_id": group_for_patient[patient],
            "outer_test_fold": test_fold,
        }
    body = {
        "schema_version": 1,
        "method": "deterministic_patient_and_duplicate_group_stratification_v1",
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "splitter_version": sklearn.__version__,
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "case_ids": sorted(case_ids),
        "patient_ids": sorted({str(case["patient_id"]) for case in cases}),
        "independent_split_group_count": len(groups),
        "assignments": assignments,
        "outer": outer_rows,
        "class_case_counts": {str(label): _case_counts(cases)[label] for label in LABELS},
        "duplicate_patient_links": sorted(sorted(set(link)) for link in (duplicate_patient_links or [])),
    }
    body["split_id"] = fingerprint(body)
    validate_split_structure(body, cases)
    return body


def _require_classes(cases: list[dict[str, Any]], case_ids: list[str], name: str) -> None:
    by_id = {str(case["case_id"]): case for case in cases}
    labels = {int(by_id[case_id]["case_label"]) for case_id in case_ids}
    if labels != set(LABELS):
        raise SplitError(f"{name} lacks both case classes; refusing to alter the cohort or seed.")


def validate_split_structure(splits: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """Validate assignments independently of how they were produced."""
    _case_counts(cases)
    expected_cases = {str(case["case_id"]) for case in cases}
    if set(splits.get("case_ids", [])) != expected_cases:
        raise SplitError("Split case set differs from the selected cohort.")
    assignment = splits.get("assignments", {})
    if set(assignment) != expected_cases:
        raise SplitError("Split assignment rows do not match the selected cohort.")
    if len(splits.get("outer", [])) != splits.get("outer_folds"):
        raise SplitError("Outer split count differs from the locked protocol.")
    if any(len(outer.get("inner_folds", [])) != splits.get("inner_folds") for outer in splits.get("outer", [])):
        raise SplitError("Inner split count differs from the locked protocol.")
    test_seen: list[str] = []
    for outer in splits.get("outer", []):
        train, test = set(outer["train_cases"]), set(outer["test_cases"])
        if train & test or train | test != expected_cases:
            raise SplitError(f"Outer fold {outer.get('fold')} case partition is invalid.")
        _require_classes(cases, sorted(train), f"outer {outer.get('fold')} train")
        _require_classes(cases, sorted(test), f"outer {outer.get('fold')} test")
        if set(outer["train_patients"]) & set(outer["test_patients"]):
            raise SplitError(f"Outer fold {outer.get('fold')} leaks patients.")
        if set(outer["train_split_groups"]) & set(outer["test_split_groups"]):
            raise SplitError(f"Outer fold {outer.get('fold')} leaks duplicate-linked groups.")
        test_seen.extend(outer["test_cases"])
        inner_seen: list[str] = []
        for inner in outer.get("inner_folds", []):
            inner_train = set(inner["train_cases"])
            validation = set(inner["validation_cases"])
            if inner_train & validation or inner_train | validation != train:
                raise SplitError(f"Outer {outer['fold']} inner {inner.get('fold')} case partition is invalid.")
            _require_classes(cases, sorted(inner_train), f"outer {outer['fold']} inner {inner.get('fold')} train")
            _require_classes(cases, sorted(validation), f"outer {outer['fold']} inner {inner.get('fold')} validation")
            if set(inner["train_patients"]) & set(inner["validation_patients"]):
                raise SplitError(f"Outer {outer['fold']} inner {inner.get('fold')} leaks patients.")
            if set(inner["train_split_groups"]) & set(inner["validation_split_groups"]):
                raise SplitError(f"Outer {outer['fold']} inner {inner.get('fold')} leaks duplicate-linked groups.")
            inner_seen.extend(inner["validation_cases"])
        if Counter(inner_seen) != Counter(outer["train_cases"]):
            raise SplitError(f"Outer {outer['fold']} inner validation folds do not partition outer train exactly once.")
    if Counter(test_seen) != Counter(expected_cases):
        raise SplitError("Outer test folds do not predict each selected case exactly once.")
    patient_folds: dict[str, set[int]] = defaultdict(set)
    patient_groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        case_id = str(case["case_id"])
        row = assignment[case_id]
        if row.get("patient_id") != str(case["patient_id"]):
            raise SplitError(f"Split patient mapping changed for case {case_id}.")
        matching = [outer["fold"] for outer in splits["outer"] if case_id in outer["test_cases"]]
        if matching != [row.get("outer_test_fold")]:
            raise SplitError(f"Split outer fold assignment mismatch for case {case_id}.")
        patient_folds[str(case["patient_id"])].add(int(row["outer_test_fold"]))
        patient_groups[str(case["patient_id"])].add(str(row.get("split_group_id", "")))
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise SplitError("Cases from one patient occur in multiple outer test folds.")
    if any(len(groups) != 1 or "" in groups for groups in patient_groups.values()):
        raise SplitError("Cases from one patient do not share a split-group assignment.")
    for link in splits.get("duplicate_patient_links", []):
        linked = [patient for patient in link if patient in patient_folds]
        if len(linked) > 1 and len({next(iter(patient_folds[patient])) for patient in linked}) != 1:
            raise SplitError("Duplicate-linked patients occur in different outer test folds.")


def validate_content_isolation(
    splits: dict[str, Any],
    cases: list[dict[str, Any]],
    instance_refs: list[dict[str, Any]],
) -> None:
    """Check patient, case, image, tile, PNG-byte and RGB-pixel isolation."""
    validate_split_structure(splits, cases)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in instance_refs:
        by_case[str(ref["case_id"])].append(ref)
    def validate_pair(left_cases: set[str], right_cases: set[str], name: str) -> None:
        if left_cases & right_cases:
            raise SplitError(f"{name}: case overlap.")
        left_refs = [ref for case_id in left_cases for ref in by_case.get(case_id, [])]
        right_refs = [ref for case_id in right_cases for ref in by_case.get(case_id, [])]
        for key in ("patient_id", "image_id", "tile_id", "source_png_sha256", "rgb_pixel_sha256"):
            left_values = {str(ref[key]) for ref in left_refs if ref.get(key) not in {None, ""}}
            right_values = {str(ref[key]) for ref in right_refs if ref.get(key) not in {None, ""}}
            overlap = left_values & right_values
            if overlap:
                raise SplitError(f"{name}: {key} overlap ({len(overlap)} values).")
    for outer in splits["outer"]:
        validate_pair(set(outer["train_cases"]), set(outer["test_cases"]), f"outer {outer['fold']}")
        for inner in outer["inner_folds"]:
            validate_pair(set(inner["train_cases"]), set(inner["validation_cases"]),
                          f"outer {outer['fold']} inner {inner['fold']}")
