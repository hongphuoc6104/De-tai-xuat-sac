"""Locked nested patient CV orchestration; outer predictions never choose parameters."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .io import atomic_json, file_hash, fingerprint
from .mil_checkpoint import CheckpointStore, FitLock
from .mil_data import BundleDataset
from .mil_model import MILModel, ModelConfig
from .mil_outputs import publish_case_attention, verify_attention_outputs
from .mil_training import TrainConfig, evaluate, run_fit, source_fingerprint
from .readiness import verify_pretrain_bundle


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    """Predeclared control study; only scheduling deadlines may change on resume."""

    variants: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4")
    max_epochs: int = 30
    patience: int = 5
    seed: int = 42
    device: str = "cuda"
    memory_limit_bytes: int = 8 * 1024**3
    projection_dim: int = 128
    attention_dim: int = 64
    dropout: float = 0.25


def metrics(rows: list[dict[str, Any]], *, bootstrap_samples: int = 0) -> dict[str, Any]:
    """Describe case OOF predictions with counts and optional fixed-OOF uncertainty."""
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Metric rows must contain unique cases.")
    labels = np.array([r["target"] for r in rows], dtype=int)
    probs = np.array([r["probability"] for r in rows], dtype=float)
    if not set(labels).issubset({0, 1}) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Invalid case prediction labels/probabilities.")
    pred = probs >= 0.5
    tp = int(((labels == 1) & pred).sum()); fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum()); fp = int(((labels == 0) & pred).sum())
    both = len(set(labels)) == 2
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result: dict[str, Any] = {
        "cases": len(rows), "patients": len({r["patient_id"] for r in rows}),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
        "AUROC": float(roc_auc_score(labels, probs)) if both else None,
        "AP": float(average_precision_score(labels, probs)) if both else None,
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
        "Brier": float(brier_score_loss(labels, probs)),
        "threshold": 0.5,
        "undefined_reason": None if both else "single_class_partition",
    }
    if bootstrap_samples:
        patients = sorted({r["patient_id"] for r in rows})
        rng = np.random.default_rng(42)
        values: dict[str, list[float]] = {key: [] for key in ("AUROC", "AP", "balanced_accuracy")}
        missing_class = 0
        for _ in range(bootstrap_samples):
            selected = rng.choice(patients, size=len(patients), replace=True)
            selected_rows = [r for patient in selected for r in rows if r["patient_id"] == patient]
            y = np.array([r["target"] for r in selected_rows]); p = np.array([r["probability"] for r in selected_rows])
            if len(set(y)) != 2:
                missing_class += 1; continue
            values["AUROC"].append(float(roc_auc_score(y, p)))
            values["AP"].append(float(average_precision_score(y, p)))
            values["balanced_accuracy"].append(float(((p[y == 1] >= .5).mean() + (p[y == 0] < .5).mean()) / 2))
        result["bootstrap"] = {
            "samples": bootstrap_samples, "missing_class_samples": missing_class,
            "intervals95": {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in values.items()},
            "scope": "fixed OOF predictions; excludes uncertainty from retraining/fold selection",
        }
    return result


def _load_fit_model(fit_dir: Path, config: TrainConfig) -> MILModel:
    identity = json.loads((fit_dir / "resolved_config.json").read_text())
    state = CheckpointStore(fit_dir / "checkpoints", identity).load_latest(map_location="cpu")
    if state is None or not state["finished"]:
        raise ValueError("Outer prediction requires a committed completed refit.")
    model = MILModel(config.model).to(config.device)
    model.load_state_dict(state["model"])
    return model


def run_pilot(
    bundle: Path, feature_root: Path, output: Path, *, deadline: float,
    variant: str = "E4", epochs: int = 2, recover_stale_lock: bool = False,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Benchmark one locked inner fit with full real bags, without touching outer test.

    Only inner-train/validation labels are read by the model. Weights are a pilot
    artifact and are never reused by the production study.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Pilot requires actual CUDA on Colab; refusing CPU fallback.")
    if type(epochs) is not int or not 1 <= epochs <= 2:
        raise ValueError("Pilot is bounded to one or two epochs.")
    verify_pretrain_bundle(Path(bundle), Path(feature_root), deadline=deadline)
    splits = json.loads((Path(bundle) / "splits.json").read_text())
    outer = splits["outer"][0]
    counts = {case_id: 0 for case_id in outer["train_cases"]}
    with (Path(bundle) / "bags.jsonl").open() as stream:
        for line in stream:
            bag = json.loads(line)
            if bag["case_id"] in counts:
                counts[bag["case_id"]] += int(bag["instance_count"])
    largest_case = max(counts, key=lambda case: (counts[case], case))
    inner = next(row for row in outer["inner_folds"] if largest_case in row["train_cases"])
    config = TrainConfig(
        model=ModelConfig(variant=variant, dropout=0.0 if variant == "E0" else .25),
        mode="inner", max_epochs=epochs, seed=42, device="cuda", max_steps=None,
        memory_limit_bytes=8 * 1024**3, learning_rate=1e-3 if variant == "E0" else 1e-4,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = run_fit(
        BundleDataset(bundle, feature_root, inner["train_cases"]),
        BundleDataset(bundle, feature_root, inner["validation_cases"]), Path(output) / "fit", config,
        fit_identity={"mode": "pilot", "outer": 0, "inner": inner["fold"], "split_id": splits["split_id"]},
        deadline=deadline, recover_stale_lock=recover_stale_lock, stop_requested=stop_requested,
    )
    report = {
        **result, "mode": "pilot", "clinical_metrics": False, "outer_test_used": False,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "full_bags": True, "largest_outer_train_case": largest_case,
        "largest_case_instances": counts[largest_case], "inner_fold": inner["fold"],
    }
    atomic_json(Path(output) / "benchmark.json", report)
    return report


def run_study(
    bundle: Path, feature_root: Path, output: Path, config: StudyConfig,
    *, deadline: float, recover_stale_lock: bool = False, allow_synthetic_cpu: bool = False,
    recovery_evidence: dict[str, dict[str, Any]] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute/reconcile nested fits without ever reusing head weights across folds.

    CPU production is deliberately blocked. ``allow_synthetic_cpu`` exists only
    for tiny end-to-end test bundles whose encoder declares synthetic-fixture.
    """
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError("A future finite deadline is required.")
    if not config.variants or len(set(config.variants)) != len(config.variants):
        raise ValueError("Study variants must be nonempty and unique.")
    for variant in config.variants:
        ModelConfig(variant=variant)
    if not 1 <= config.max_epochs <= 30 or config.patience < 1:
        raise ValueError("Study epoch/patience protocol is invalid.")
    bundle, feature_root, output = Path(bundle), Path(feature_root), Path(output)
    verified = verify_pretrain_bundle(bundle, feature_root, deadline=deadline)
    if not verified["training_ready"]:
        raise ValueError("Bundle failed production readiness.")
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest.get("cohort_mode") != "common":
        raise ValueError("Control study requires the common full-lens cohort; all cohort is a separate study.")
    feature_config = json.loads((feature_root / manifest["feature_id"] / "feature_config.json").read_text())
    synthetic = "synthetic-fixture" in json.dumps(feature_config)
    if config.device != "cuda" and not (allow_synthetic_cpu and synthetic and config.device == "cpu"):
        raise ValueError("Full real-data study runs on Colab CUDA; local CPU is for bounded smoke only.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent fallback or second runtime.")
    splits = json.loads((bundle / "splits.json").read_text())
    if splits["outer_folds"] != 3 or splits["inner_folds"] != 2 or splits["seed"] != 42:
        raise ValueError("Control study requires the locked 3x2 patient folds with seed42.")
    identity = {
        "schema": 1, "bundle_manifest_sha256": file_hash(bundle / "bundle.json"),
        "split_id": splits["split_id"], "feature_id": manifest["feature_id"],
        "governance_id": manifest["governance_id"], "config": dataclasses.asdict(config),
        "source_fingerprint": source_fingerprint(),
    }
    # JSON normalization avoids tuple/list differences when resuming from disk.
    identity = json.loads(json.dumps(identity))
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / "resolved_config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != identity:
        raise ValueError("Study output already belongs to a different protocol/cache/bundle.")
    rows: list[dict[str, Any]] = []
    completed = 0
    total = len(config.variants) * 9

    def status(phase: str, **kwargs: Any) -> dict[str, Any]:
        result = {"phase": phase, "completed_fits": completed, "expected_fits": total,
                  "study_id": fingerprint(identity), "training_ready": True, **kwargs}
        atomic_json(output / "status.json", result)
        return result

    recovery = recovery_evidence or {}
    with FitLock(output, recover_stale=recover_stale_lock, recovery_evidence=recovery.get("study")):
        atomic_json(cfg_path, identity)
        for variant in config.variants:
            for outer in splits["outer"]:
                outer_index = outer["fold"]
                model_config = ModelConfig(variant=variant, projection_dim=config.projection_dim,
                                           attention_dim=config.attention_dim,
                                           dropout=0.0 if variant == "E0" else config.dropout)
                train_config = TrainConfig(
                    model=model_config, mode="inner", max_epochs=config.max_epochs, patience=config.patience,
                    learning_rate=1e-3 if variant == "E0" else 1e-4, seed=config.seed + outer_index * 100,
                    device=config.device, max_steps=None, memory_limit_bytes=config.memory_limit_bytes,
                )
                best_epochs = []
                for inner in outer["inner_folds"]:
                    fit_id = {"variant": variant, "outer": outer_index, "inner": inner["fold"], "split_id": splits["split_id"]}
                    fit_dir = output / "fits" / variant / f"outer_{outer_index}" / f"inner_{inner['fold']}"
                    if time.monotonic() >= deadline or (stop_requested and stop_requested()):
                        return status("paused_budget", next_fit=fit_id)
                    train_dataset = BundleDataset(bundle, feature_root, inner["train_cases"])
                    validation_dataset = BundleDataset(bundle, feature_root, inner["validation_cases"])
                    fit_config = dataclasses.replace(train_config, seed=train_config.seed + inner["fold"] * 10)
                    fit = run_fit(train_dataset, validation_dataset, fit_dir, fit_config, fit_identity=fit_id,
                                  deadline=deadline, recover_stale_lock=recover_stale_lock,
                                  lock_recovery_evidence=recovery.get(str(fit_dir.relative_to(output))),
                                  stop_requested=stop_requested)
                    if fit["phase"] != "complete":
                        return status(fit["phase"], next_fit=fit_id)
                    completed += 1
                    best_epochs.append(fit["best_epoch"])
                epoch_count = max(1, math.ceil(float(np.median(best_epochs))))
                refit_config = dataclasses.replace(train_config, mode="refit", max_epochs=epoch_count,
                                                   seed=train_config.seed + 99)
                refit_id = {"variant": variant, "outer": outer_index, "refit": True, "split_id": splits["split_id"]}
                refit_dir = output / "fits" / variant / f"outer_{outer_index}" / "refit"
                if time.monotonic() >= deadline or (stop_requested and stop_requested()):
                    return status("paused_budget", next_fit=refit_id)
                refit = run_fit(BundleDataset(bundle, feature_root, outer["train_cases"]), None,
                                refit_dir, refit_config, fit_identity=refit_id, deadline=deadline,
                                recover_stale_lock=recover_stale_lock,
                                lock_recovery_evidence=recovery.get(str(refit_dir.relative_to(output))),
                                stop_requested=stop_requested)
                if refit["phase"] != "complete":
                    return status(refit["phase"], next_fit=refit_id)
                completed += 1
                prediction_path = refit_dir / "predictions.json"
                commit_path = refit_dir / "predictions.commit.json"
                if commit_path.exists():
                    commit = json.loads(commit_path.read_text())
                    if commit.get("sha256") != file_hash(prediction_path) or commit.get("fit") != refit_id:
                        raise ValueError("Outer prediction commit is corrupt or from another fit.")
                    predictions = json.loads(prediction_path.read_text())
                    attention_commits = verify_attention_outputs(refit_dir / "attention", outer["test_cases"])
                    if attention_commits != commit.get("attention_commits"):
                        raise ValueError("Attention receipts differ from the locked prediction commit.")
                else:
                    if time.monotonic() >= deadline or (stop_requested and stop_requested()):
                        return status("paused_budget", next_fit={**refit_id, "prediction_pending": True})
                    test_dataset = BundleDataset(bundle, feature_root, outer["test_cases"])
                    model = _load_fit_model(refit_dir, refit_config)
                    predictions = evaluate(model, test_dataset, config.device, config=refit_config,
                                           deadline=deadline,
                                           case_callback=lambda case, result: publish_case_attention(
                                               bundle, test_dataset, case, result, refit_dir / "attention"
                                           ))["predictions"]
                    attention_commits = verify_attention_outputs(refit_dir / "attention", outer["test_cases"])
                    atomic_json(prediction_path, predictions)
                    atomic_json(commit_path, {"fit": refit_id, "sha256": file_hash(prediction_path),
                                              "attention_commits": attention_commits})
                if sorted(r["case_id"] for r in predictions) != sorted(outer["test_cases"]):
                    raise ValueError("Outer predictions have missing or extra cases.")
                rows.extend({**r, "variant": variant, "outer_fold": outer_index,
                             "threshold": .5, "predicted_label": int(r["probability"] >= .5)} for r in predictions)
                status("running", last_completed=refit_id)
        evaluation = output / "evaluation"
        evaluation.mkdir(exist_ok=True)
        reports = {}
        for variant in config.variants:
            selected = [r for r in rows if r["variant"] == variant]
            if sorted(r["case_id"] for r in selected) != sorted(splits["case_ids"]):
                raise ValueError("OOF coverage is not exactly one prediction per case/variant.")
            reports[variant] = {
                "pooled_oof": metrics(selected, bootstrap_samples=1000),
                "outer": {str(k): metrics([r for r in selected if r["outer_fold"] == k]) for k in range(3)},
            }
        atomic_json(evaluation / "metrics.json", reports)
        with (evaluation / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        atomic_json(evaluation / "commit.json", {
            "study_id": fingerprint(identity), "rows": len(rows),
            "metrics_sha256": file_hash(evaluation / "metrics.json"),
            "oof_sha256": file_hash(evaluation / "oof_predictions.csv"),
        })
        report_path = output / "REPORT.md"
        report_path.write_text(
            "# MIL nested study\n\n"
            f"Completed fits: {completed}/{total}. Case OOF rows: {len(rows)}.\n\n"
            "Encoder: frozen ResNet50 ImageNet V1; no CNN fine-tuning or patch-level supervision.\n\n"
            "Protocol: patient grouped 3 outer x 2 inner, inner BCE chooses epochs; fresh outer refits.\n\n"
            "See evaluation/metrics.json for fold/pooled counts and fixed-OOF patient bootstrap. "
            "Intervals do not include retraining uncertainty.\n\n"
            "Attention artifacts contain relative bag weights and source-coordinate point maps, "
            "not pixel cancer probabilities or segmentation truth. No new-patient external validation.\n"
        )
        atomic_json(output / "model_card.json", {
            "study_identity": identity, "variants": list(config.variants), "case_endpoint": "benign_vs_cancer",
            "CNN_frozen": True, "patch_ground_truth": False, "stain_normalization_added": False,
            "weights": "fits/<variant>/outer_<fold>/refit/checkpoints/versions/<version>/state.pt",
            "evaluation": "patient grouped OOF; not an independent test of a full-cohort demo model",
            "external_validation": False,
        })
        return status("complete", oof_rows=len(rows), evaluation_commit=str(evaluation / "commit.json"))
