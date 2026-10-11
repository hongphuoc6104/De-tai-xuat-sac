"""Bounded Colab-only entrypoint for shared preprocessing and parallel A/B studies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .dual_common import ArtifactStore, build_source_photo_map, check_deadline, fit_scope_transforms
from .io import atomic_json, file_hash
from .mil_case_cache import CacheDataset
from .mil_training import source_fingerprint

EXPECTED_FEATURE = "7301faeb7b193fb077451a9f2da44bbb609b69d3d7647261df73c351590cbe04"
EXPECTED_BUNDLE = "6e412d65e429ad19c3bcec645a19d7ee06fe75e560ef67149233f14d4175b0ff"
EXPECTED_SPLIT = "1feba2fd0694aa691682ead89121927062680f04351b6c0d4feaf6e2594c5501"
MODES = ("preprocess", "cpu", "worker_A", "worker_B", "gpu_smoke", "neural")


def parse_deadline(text: str) -> float:
    """Translate a future explicit UTC budget into a monotonic deadline."""
    instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Deadline must be timezone-aware.")
    remaining = (instant - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("No remaining work budget.")
    return time.monotonic() + remaining


def validate_config(config: dict[str, Any]) -> None:
    """Reject incompatible source, scope or resource settings before creating writers."""
    required = {"study_id", "setup_path", "output_root", "source_code_sha256"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Dual-track config is missing mandatory settings.")
    name = config["study_id"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("study_id must be a safe path component.")
    code_hash = config["source_code_sha256"]
    if not isinstance(code_hash, str) or len(code_hash) != 64 or any(c not in "0123456789abcdef" for c in code_hash):
        raise ValueError("source_code_sha256 must be a full pinned SHA-256.")
    if type(config.get("recover_stale_locks", False)) is not bool:
        raise ValueError("recover_stale_locks must be an explicit boolean.")
    if not isinstance(config.get("recovery_evidence", {}), dict):
        raise ValueError("Recovery evidence must be a branch/lease mapping.")
    if config.get("cohort", "common16") != "common16":
        raise ValueError("Primary release supports common16; all18 masked protocol is not yet released.")
    if sklearn.__version__ != "1.8.0":
        raise ValueError("The governed split/runtime requires scikit-learn==1.8.0.")


def validate_runtime_code(config: dict[str, Any]) -> None:
    """Bind the invoked module tree to the actual pinned code-only archive."""
    archive_path = Path(config.get("runtime_archive_path", ""))
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("Runtime archive must be a regular verified file.")
    if file_hash(archive_path) != config["source_code_sha256"]:
        raise ValueError("Runtime archive SHA does not match the pinned code release.")
    code_root = Path(__file__).resolve().parent.parent
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("runtime_manifest.json"))
        for name, expected in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe runtime manifest path.")
            actual = code_root / relative
            if actual.is_symlink() or not actual.is_file() or file_hash(actual) != expected:
                raise ValueError("Invoked code differs from the pinned runtime archive.")


def _load_inputs(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    setup = json.loads(Path(config["setup_path"]).read_text())
    bundle = Path(setup["bundle_path"])
    features = Path(setup["feature_root"])
    cache = Path(setup["case_cache"])
    manifest = json.loads((bundle / "bundle.json").read_text())
    if (
        manifest["feature_id"] != EXPECTED_FEATURE
        or manifest["bundle_id"] != EXPECTED_BUNDLE
        or manifest["split_id"] != EXPECTED_SPLIT
    ):
        raise ValueError("Source manifest does not match approved common16 feature/bundle/splits.")
    splits = json.loads((bundle / "splits.json").read_text())
    all_ids = manifest["cohort_case_ids"]
    if len(all_ids) != 16 or splits["split_id"] != EXPECTED_SPLIT:
        raise ValueError("Primary cohort/split cardinality mismatch.")

    def factory(ids: list[str]) -> CacheDataset:
        return CacheDataset(bundle, features, cache, list(ids))

    dataset = factory(all_ids)
    counts = {target: sum(r["target"] == target for r in dataset.records.values()) for target in [0, 1]}
    if counts != {0: 9, 1: 7}:
        raise ValueError("Primary case labels/counts differ from approved governance.")
    return dataset, factory, splits, setup


def _stores(config: dict[str, Any], dataset: Any) -> tuple[ArtifactStore, ArtifactStore, ArtifactStore]:
    protocol = {
        "version": 1,
        "study_id": config["study_id"],
        "cohort": "common16",
        "feature_id": EXPECTED_FEATURE,
        "bundle_id": EXPECTED_BUNDLE,
        "split_id": EXPECTED_SPLIT,
        "code_sha256": config["source_code_sha256"],
        "source_fingerprint": source_fingerprint(),
        "numeric_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "source_cache_id": dataset.identity["cache_id"],
        "pca_dimensions": 32,
        "sample_cap_per_patient_lens": 256,
        "seed": 42,
        "A": {"k": [4, 8, 16], "C": [0.1, 1.0], "descriptor": "photo_balanced_last_bin_drop"},
        "B": {
            "anchor_per_member": 1,
            "prototypes_per_lens": 32,
            "C": [0.1, 1.0],
            "beta": [0.0, 0.5],
            "minimum_inner_BCE_gain": 0.01,
            "descriptors": "mean_of_per_photo_median_q90_q95_top5mean",
        },
        "inner_score_aggregation": "equal_weight_mean_of_fold_mean_BCE",
        "threshold": 0.5,
        "full_bags": True,
        "semantic_cluster_labels": "unknown",
    }
    root = Path(config["output_root"])
    for source in (dataset.bundle, dataset.feature_root, dataset.cache_path):
        source = Path(source).resolve()
        target = root.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("Output must be separate from immutable source/cache trees.")
    return tuple(
        ArtifactStore(
            root / branch,
            {**protocol, "artifact_branch": branch},
            recover_stale=bool(config.get("recover_stale_locks", False)),
            recovery_evidence=config.get("recovery_evidence", {}).get(branch, {}),
        )
        for branch in ["shared", "A", "B"]
    )


def run_preprocess(config: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Publish the immutable photo map and all nine train-only PCA scopes once."""
    dataset, factory, splits, setup = _load_inputs(config)
    shared, _, _ = _stores(config, dataset)
    photo_map = build_source_photo_map(
        dataset, Path(setup["bundle_path"]), Path(setup["feature_root"]), shared, deadline
    )
    scopes = []
    for outer in splits["outer"]:
        for inner in outer["inner_folds"]:
            scopes.append((f"outer_{outer['fold']}/inner_{inner['fold']}", inner["train_cases"]))
        scopes.append((f"outer_{outer['fold']}/refit", outer["train_cases"]))
    for scope, ids in scopes:
        check_deadline(deadline)
        shared.object(
            f"scopes/{scope}/transform.pkl",
            lambda ids=ids: fit_scope_transforms(
                factory(ids),
                photo_map,
                ids,
                authorized_train_ids=ids,
            ),
        )
        atomic_json(
            Path(config["output_root"]) / "scheduler/status.json",
            {"phase": "preprocessing", "last_committed_scope": scope, "total_scopes": len(scopes)},
        )
    receipt = {
        "phase": "complete",
        "stage": "shared_preprocessing",
        "scopes": len(scopes),
        "tiles": len(photo_map["rows"]),
        "common_identity": shared.identity_id,
        "new_training_started": False,
    }
    shared.json("preprocessing_complete.json", receipt)
    return receipt


def run_worker(branch: str, config: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Execute one branch over committed shared data, with an exclusive branch lease."""
    from .dual_linear_study import run_linear_study
    from .mil_checkpoint import FitLock

    dataset, factory, splits, _ = _load_inputs(config)
    shared, a_store, b_store = _stores(config, dataset)
    if shared.read("preprocessing_complete.json") is None:
        raise ValueError("Shared source preprocessing is not committed; workers may not fit it themselves.")
    data = shared.read("source_photo_map.pkl")
    if data is None:
        raise ValueError("Source-photo map is not committed.")
    photo_map = pickle.loads(data)
    store = a_store if branch == "A" else b_store
    lock_root = Path(config["output_root"]) / "scheduler" / ("worker_" + branch)
    lock_root.mkdir(parents=True, exist_ok=True)
    with FitLock(
        lock_root,
        recover_stale=bool(config.get("recover_stale_locks", False)),
        recovery_evidence=config.get("recovery_evidence", {}).get("scheduler", {}).get("worker_" + branch),
    ):
        try:
            result = run_linear_study(branch, dataset, factory, splits, photo_map, shared, store, deadline)
        except TimeoutError:
            result = {"phase": "paused_budget", "branch": branch, "new_oof_committed": False}
        atomic_json(Path(config["output_root"]) / "scheduler" / ("worker_" + branch + "_result.json"), result)
        return result


def _run_cpu(config: dict[str, Any], config_path: Path, deadline: float, deadline_text: str) -> dict[str, Any]:
    """Launch A and B CPU workers together and join/clean up owned child processes."""
    processes = {}
    result = {"phase": "preflight", "branches": {}}
    root = Path(config["output_root"])
    log_handles = []
    try:
        preflight = run_preprocess(config, deadline)
        result.update(phase="running", preprocessing=preflight)
        for branch in ["A", "B"]:
            check_deadline(deadline)
            path = root / "scheduler" / ("worker_" + branch + ".log")
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a")
            log_handles.append(stream)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            processes[branch] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "histology_data.dual_cli",
                    "--mode",
                    "worker_" + branch,
                    "--config",
                    str(config_path),
                    "--deadline-utc",
                    deadline_text,
                ],
                cwd=Path(__file__).resolve().parent.parent,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        while any(p.poll() is None for p in processes.values()):
            check_deadline(deadline)
            result["branches"] = {
                branch: {"returncode": p.poll(), "running": p.poll() is None} for branch, p in processes.items()
            }
            atomic_json(root / "scheduler/status.json", result)
            time.sleep(2)
        result["branches"] = {branch: {"returncode": p.returncode} for branch, p in processes.items()}
        phases = []
        for branch in processes:
            child_result_path = root / "scheduler" / ("worker_" + branch + "_result.json")
            child = json.loads(child_result_path.read_text()) if child_result_path.is_file() else {"phase": "error"}
            result["branches"][branch]["phase"] = child["phase"]
            phases.append(child["phase"])
        if any(p.returncode != 0 for p in processes.values()) or any(p == "error" for p in phases):
            result["phase"] = "error"
        elif all(p == "complete" for p in phases):
            result["phase"] = "cpu_complete"
            result["pending_stages"] = ["T4_smoke", "B1", "final_report"]
        elif "paused_budget" in phases:
            result["phase"] = "paused_budget"
        else:
            result["phase"] = "error"
        atomic_json(root / "scheduler/status.json", result)
        return result
    except TimeoutError:
        result["phase"] = "paused_budget"
        atomic_json(root / "scheduler/status.json", result)
        return result
    except Exception as exc:
        result.update(phase="error", error_type=type(exc).__name__, error=str(exc)[:500])
        atomic_json(root / "scheduler/status.json", result)
        raise
    finally:
        # Children have the same cooperative deadline; retain a grace period for commits.
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        for stream in log_handles:
            stream.close()


def run_cpu(config: dict[str, Any], config_path: Path, deadline: float, deadline_text: str) -> dict[str, Any]:
    """Reject duplicate controllers while retaining independent A/B worker leases."""
    from .mil_checkpoint import FitLock

    lease = Path(config["output_root"]) / "scheduler/controller"
    lease.mkdir(parents=True, exist_ok=True)
    proof = config.get("recovery_evidence", {}).get("scheduler", {}).get("controller")
    with FitLock(lease, recover_stale=config.get("recover_stale_locks", False), recovery_evidence=proof):
        return _run_cpu(config, config_path, deadline, deadline_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="preprocess")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    try:
        colab_available = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        colab_available = False
    if not colab_available:
        raise RuntimeError("Full-source dual-track processing is Colab-only; local fixtures use tested APIs.")
    validate_runtime_code(config)
    deadline = parse_deadline(args.deadline_utc)
    with threadpool_limits(limits=1):
        if args.mode == "preprocess":
            from .mil_checkpoint import FitLock

            lease = Path(config["output_root"]) / "scheduler/controller"
            lease.mkdir(parents=True, exist_ok=True)
            proof = config.get("recovery_evidence", {}).get("scheduler", {}).get("controller")
            with FitLock(lease, recover_stale=config.get("recover_stale_locks", False), recovery_evidence=proof):
                result = run_preprocess(config, deadline)
        elif args.mode == "cpu":
            result = run_cpu(config, args.config.resolve(), deadline, args.deadline_utc)
        elif args.mode.startswith("worker_"):
            result = run_worker(args.mode[-1], config, deadline)
        else:
            import torch

            from .dual_neural_study import run_neural_study

            if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0).upper():
                raise RuntimeError("Neural production requires the authorized Colab T4; no CPU fallback.")
            dataset, factory, splits, _ = _load_inputs(config)
            shared, a_store, b_store = _stores(config, dataset)
            if shared.read("preprocessing_complete.json") is None:
                raise ValueError("Shared preprocessing is not committed.")
            photo_map = pickle.loads(shared.read("source_photo_map.pkl"))
            for store in (a_store, b_store):
                raw = store.read("results/linear_study.json")
                if raw is None or json.loads(raw).get("phase") != "complete":
                    raise ValueError("Both CPU studies must be committed before the T4 stage.")
            if args.mode == "gpu_smoke":
                from .dual_gpu_smoke import run_gpu_smoke

                result = run_gpu_smoke(
                    dataset,
                    factory,
                    splits,
                    photo_map,
                    shared,
                    b_store,
                    Path(config["output_root"]) / "B/gpu_smoke",
                    deadline,
                )
            else:
                if b_store.read("gpu_smoke/new_process_resume.json") is None:
                    raise ValueError("Verified T4 smoke/new-process resume is required before B1.")
                result = run_neural_study(
                    dataset,
                    factory,
                    splits,
                    photo_map,
                    shared,
                    b_store,
                    Path(config["output_root"]) / "B/neural",
                    deadline,
                    device="cuda",
                    recover_stale_lock=config.get("recover_stale_locks", False),
                    recovery_evidence_map=config.get("recovery_evidence", {}).get("B", {}),
                )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get("phase") in {"complete", "cpu_complete", "paused_budget"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
