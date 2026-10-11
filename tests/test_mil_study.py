"""End-to-end tests for nested patient-grouped MIL study orchestration."""
from __future__ import annotations

import csv
import json
import math
import runpy
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from histology_data import mil_study
from histology_data.mil_study import StudyConfig, metrics, run_study


@pytest.fixture(scope="module")
def governed_bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build tiny, governed 2048-D bundles via the project's real pretrain helpers."""
    root = tmp_path_factory.mktemp("mil-study-governed")
    helper_path = Path(__file__).resolve().parents[1] / "tests" / "test_pretrain.py"
    helpers = runpy.run_path(str(helper_path))
    fixture = helpers["make_feature_fixture"](root / "source", patients=6)
    reviewed = helpers["make_reviewed_governance"](fixture, root / "governance")
    governance = reviewed["governance"]
    build_bundle = helpers["build_pretrain_bundle"]
    common = build_bundle(
        Path(governance["path"]),
        fixture["feature_root"],
        root / "bundles",
        cohort_mode="common",
        expected_sources=1,
        expected_vectors=fixture["png_count"],
    )
    all_cases = build_bundle(
        Path(governance["path"]),
        fixture["feature_root"],
        root / "bundles",
        cohort_mode="all",
        expected_sources=1,
        expected_vectors=fixture["png_count"],
    )
    return {
        "root": root,
        "feature_root": fixture["feature_root"],
        "common": Path(common["bundle_path"]),
        "all": Path(all_cases["bundle_path"]),
        "case_count": len(fixture["case_labels"]),
        "patient_count": len(set(fixture["case_patients"].values())),
    }


def _study_config(**overrides: Any) -> StudyConfig:
    defaults: dict[str, Any] = {
        "variants": ("E0",),
        "max_epochs": 1,
        "patience": 1,
        "seed": 42,
        "device": "cpu",
        "memory_limit_bytes": 2 * 1024**3,
        "projection_dim": 128,
        "attention_dim": 64,
        "dropout": 0.25,
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _run(
    bundle: Path,
    feature_root: Path,
    output: Path,
    config: StudyConfig,
    *,
    allow_synthetic_cpu: bool = True,
) -> dict[str, Any]:
    return run_study(
        bundle,
        feature_root,
        output,
        config,
        deadline=time.monotonic() + 600,
        allow_synthetic_cpu=allow_synthetic_cpu,
    )


def _fit_dirs(output: Path) -> list[Path]:
    return sorted(path.parent for path in (output / "fits" / "E0").glob("outer_*/**/resolved_config.json"))


def _fit_snapshot(output: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"fits": {}, "predictions": {}, "evaluation": {}}
    for fit_dir in _fit_dirs(output):
        state_identity = json.loads((fit_dir / "resolved_config.json").read_text(encoding="utf-8"))
        checkpoint_root = fit_dir / "checkpoints"
        manifests = sorted((checkpoint_root / "versions").glob("*/commit.json"))
        state = mil_study.CheckpointStore(checkpoint_root, state_identity).load_latest()
        snapshot["fits"][str(fit_dir.relative_to(output))] = {
            "global_step": state["global_step"],
            "epoch": state["epoch"],
            "finished": state["finished"],
            "commits": [(path.parent.name, path.read_bytes()) for path in manifests],
        }
        prediction_path = fit_dir / "predictions.json"
        commit_path = fit_dir / "predictions.commit.json"
        if prediction_path.exists() or commit_path.exists():
            snapshot["predictions"][str(fit_dir.relative_to(output))] = (
                prediction_path.read_bytes(), commit_path.read_bytes(),
            )
    for path in sorted((output / "evaluation").glob("*")):
        if path.is_file():
            snapshot["evaluation"][path.name] = path.read_bytes()
    return snapshot


def test_run_study_on_synthetic_governed_bundle_e2e(
    governed_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise 3×2 inner fits, outer refits, OOF and resume through real bundle APIs."""
    bundle = governed_bundles["common"]
    feature_root = governed_bundles["feature_root"]
    output = tmp_path / "study"
    split = json.loads((bundle / "splits.json").read_text(encoding="utf-8"))
    assert split["outer_folds"] == 3 and split["inner_folds"] == 2
    assert len(split["outer"]) == 3
    for outer in split["outer"]:
        assert len(outer["inner_folds"]) == 2
        assert set(outer["train_patients"]).isdisjoint(outer["test_patients"])
        for inner in outer["inner_folds"]:
            assert set(inner["train_patients"]).isdisjoint(inner["validation_patients"])

    load_count = 0
    original_load_case = mil_study.BundleDataset.load_case
    original_run_fit = mil_study.run_fit
    fit_calls: list[dict[str, Any]] = []
    outer_by_fold = {outer["fold"]: outer for outer in split["outer"]}

    def count_case_loads(dataset: Any, case_id: str) -> Any:
        nonlocal load_count
        load_count += 1
        return original_load_case(dataset, case_id)

    def audit_fit_partitions(
        train_dataset: Any,
        validation_dataset: Any,
        fit_dir: Path,
        fit_config: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fit_id = kwargs["fit_identity"]
        outer = outer_by_fold[fit_id["outer"]]
        train_ids = set(train_dataset.case_ids)
        if fit_id.get("refit"):
            assert validation_dataset is None
            assert train_ids == set(outer["train_cases"])
        else:
            inner = next(row for row in outer["inner_folds"] if row["fold"] == fit_id["inner"])
            assert validation_dataset is not None
            assert train_ids == set(inner["train_cases"])
            assert set(validation_dataset.case_ids) == set(inner["validation_cases"])
            assert fit_config.mode == "inner"
        assert train_ids.isdisjoint(outer["test_cases"])
        if validation_dataset is not None:
            assert set(validation_dataset.case_ids).isdisjoint(outer["test_cases"])
        result = original_run_fit(
            train_dataset,
            validation_dataset,
            fit_dir,
            fit_config,
            **kwargs,
        )
        fit_calls.append(fit_id)
        return result

    monkeypatch.setattr(mil_study.BundleDataset, "load_case", count_case_loads)
    monkeypatch.setattr(mil_study, "run_fit", audit_fit_partitions)
    result = _run(bundle, feature_root, output, _study_config())

    assert result["phase"] == "complete"
    assert result["completed_fits"] == result["expected_fits"] == 9
    assert result["oof_rows"] == governed_bundles["case_count"] == 12
    assert load_count > 0
    assert len(fit_calls) == 9
    fit_dirs = _fit_dirs(output)
    assert len(fit_dirs) == 9
    for fit_dir in fit_dirs:
        state_identity = json.loads((fit_dir / "resolved_config.json").read_text(encoding="utf-8"))
        state = mil_study.CheckpointStore(fit_dir / "checkpoints", state_identity).load_latest()
        assert state["finished"] is True and state["epoch"] == 1
        if fit_dir.name == "refit":
            inner_epochs = []
            outer_dir = fit_dir.parent
            for inner_index in (0, 1):
                inner_dir = outer_dir / f"inner_{inner_index}"
                inner_identity = json.loads((inner_dir / "resolved_config.json").read_text(encoding="utf-8"))
                inner_state = mil_study.CheckpointStore(
                    inner_dir / "checkpoints", inner_identity,
                ).load_latest()
                inner_epochs.append(inner_state["best_epoch"])
            assert state_identity["config"]["max_epochs"] == max(
                1,
                math.ceil(float(statistics.median(inner_epochs))),
            )

    with (output / "evaluation" / "oof_predictions.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert len({row["case_id"] for row in rows}) == 12
    assert {row["variant"] for row in rows} == {"E0"}
    patient_by_case = split["assignments"]
    for outer in split["outer"]:
        fold_rows = [row for row in rows if int(row["outer_fold"]) == outer["fold"]]
        assert {row["case_id"] for row in fold_rows} == set(outer["test_cases"])
        for row in fold_rows:
            assert row["patient_id"] == patient_by_case[row["case_id"]]["patient_id"]
            assert row["patient_id"] in outer["test_patients"]
            assert row["patient_id"] not in outer["train_patients"]

    metrics_payload = json.loads((output / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    pooled = metrics_payload["E0"]["pooled_oof"]
    assert pooled["cases"] == 12 and pooled["patients"] == 6
    assert pooled["undefined_reason"] is None
    assert set(metrics_payload["E0"]["outer"]) == {"0", "1", "2"}
    for fold in metrics_payload["E0"]["outer"].values():
        assert fold["cases"] == 4 and fold["patients"] == 2
        assert fold["undefined_reason"] is None
        assert fold["AUROC"] is not None and fold["AP"] is not None
    assert (output / "evaluation" / "commit.json").is_file()

    before_rerun = _fit_snapshot(output)
    before_resolved_config = (output / "resolved_config.json").read_bytes()
    before_case_load_count = load_count
    rerun = _run(bundle, feature_root, output, _study_config())
    assert rerun["phase"] == "complete" and rerun["completed_fits"] == 9
    assert load_count == before_case_load_count
    assert _fit_snapshot(output) == before_rerun

    with pytest.raises(ValueError, match="different protocol/cache/bundle"):
        _run(bundle, feature_root, output, _study_config(dropout=0.3))
    assert (output / "resolved_config.json").read_bytes() == before_resolved_config


def test_study_rejects_all_cohort_even_when_every_case_has_all_lenses(
    governed_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    output = tmp_path / "all-cohort"
    with pytest.raises(ValueError, match="common full-lens cohort"):
        _run(
            governed_bundles["all"],
            governed_bundles["feature_root"],
            output,
            _study_config(),
        )
    assert not (output / "resolved_config.json").exists()


def test_cpu_study_requires_explicit_synthetic_fixture_override(
    governed_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpu-production"
    with pytest.raises(ValueError, match="Full real-data study runs on Colab CUDA"):
        _run(
            governed_bundles["common"],
            governed_bundles["feature_root"],
            output,
            _study_config(),
            allow_synthetic_cpu=False,
        )
    assert not output.exists()


def test_attention_callback_failure_cannot_commit_partial_outer_predictions_then_resume(
    governed_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = governed_bundles["common"]
    feature_root = governed_bundles["feature_root"]
    output = tmp_path / "callback-failure"
    splits = json.loads((bundle / "splits.json").read_text(encoding="utf-8"))
    first_outer = splits["outer"][0]
    refit_dir = output / "fits" / "E0" / "outer_0" / "refit"
    original_publish = mil_study.publish_case_attention
    injected = False

    def publish_then_fail(*args: Any, **kwargs: Any) -> None:
        nonlocal injected
        original_publish(*args, **kwargs)
        if not injected:
            injected = True
            raise RuntimeError("synthetic attention publication interruption")

    monkeypatch.setattr(mil_study, "publish_case_attention", publish_then_fail)
    with pytest.raises(RuntimeError, match="attention publication interruption"):
        _run(bundle, feature_root, output, _study_config())

    assert injected is True
    assert not (refit_dir / "predictions.json").exists()
    assert not (refit_dir / "predictions.commit.json").exists()
    attention_receipts = list((refit_dir / "attention").glob("*/commit.json"))
    assert len(attention_receipts) == 1
    first_phase_fits = [
        output / "fits" / "E0" / "outer_0" / "inner_0",
        output / "fits" / "E0" / "outer_0" / "inner_1",
        refit_dir,
    ]
    committed_before_resume = {}
    for fit_dir in first_phase_fits:
        identity = json.loads((fit_dir / "resolved_config.json").read_text(encoding="utf-8"))
        store = mil_study.CheckpointStore(fit_dir / "checkpoints", identity)
        state = store.load_latest()
        assert state["finished"] is True
        committed_before_resume[fit_dir] = sorted(
            path.parent.name for path in (fit_dir / "checkpoints" / "versions").glob("*/commit.json")
        )

    monkeypatch.setattr(mil_study, "publish_case_attention", original_publish)
    result = _run(bundle, feature_root, output, _study_config())

    assert result["phase"] == "complete"
    assert result["oof_rows"] == len(splits["case_ids"])
    assert sorted(row["case_id"] for row in json.loads(
        (refit_dir / "predictions.json").read_text(encoding="utf-8"),
    )) == sorted(first_outer["test_cases"])
    assert (refit_dir / "predictions.commit.json").is_file()
    assert len(list((refit_dir / "attention").glob("*/commit.json"))) == len(first_outer["test_cases"])
    for fit_dir, prior_commits in committed_before_resume.items():
        current_commits = sorted(
            path.parent.name for path in (fit_dir / "checkpoints" / "versions").glob("*/commit.json")
        )
        assert current_commits == prior_commits


def test_metrics_report_single_class_partitions_as_undefined_and_reject_bad_targets() -> None:
    single_class = [
        {"case_id": "case-a", "patient_id": "patient-a", "target": 0, "probability": 0.2},
        {"case_id": "case-b", "patient_id": "patient-b", "target": 0, "probability": 0.7},
    ]
    report = metrics(single_class)
    assert report["AUROC"] is None
    assert report["AP"] is None
    assert report["balanced_accuracy"] is None
    assert report["undefined_reason"] == "single_class_partition"

    invalid_target = [{"case_id": "case-c", "patient_id": "patient-c", "target": 2, "probability": 0.4}]
    with pytest.raises(ValueError, match="Invalid case prediction labels"):
        metrics(invalid_target)
