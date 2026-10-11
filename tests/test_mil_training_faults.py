"""End-to-end fault reproduction for resumable MIL training transactions."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from histology_data.mil_checkpoint import CheckpointStore
from histology_data.mil_data import FixtureDataset, create_toy_fixture
from histology_data.mil_model import LENS_KEYS, ModelConfig
from histology_data.mil_training import TrainConfig, run_fit

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIM = 2048
ROWS_PER_LENS = 16
MEMORY_LIMIT_BYTES = 2 * 1024**3
FIT_IDENTITY = {"fit_id": "mil-fault-e2e", "protocol": "smoke-resume-v1"}


class MemoryDataset:
    """Small writable in-memory cases used for loop-level fault injection."""

    def __init__(self, name: str, seed: int) -> None:
        self.case_ids = [f"{name}-case-{index}" for index in range(2)]
        self.records = {
            case_id: {"patient_id": f"{name}-patient-{index}", "target": index}
            for index, case_id in enumerate(self.case_ids)
        }
        self.identity = {"mode": "smoke", "kind": "fault-test-memory", "dataset_id": name}
        rng = np.random.default_rng(seed)
        self._cases: dict[str, SimpleNamespace] = {}
        for index, case_id in enumerate(self.case_ids):
            features = {
                lens: rng.normal(0, 0.04, (ROWS_PER_LENS, FEATURE_DIM)).astype(np.float32)
                for lens in LENS_KEYS
            }
            for values in features.values():
                values[:, 0] += (index * 2 - 1) * 0.2
            self._cases[case_id] = SimpleNamespace(
                case_id=case_id,
                patient_id=self.records[case_id]["patient_id"],
                target=index,
                features=features,
                presence_mask=(True, True, True),
            )

    def load_case(self, case_id: str) -> SimpleNamespace:
        return self._cases[case_id]


class Partition:
    """Patient-disjoint view with the CaseDataset contract used by run_fit."""

    def __init__(self, dataset: Any, case_ids: list[str], name: str) -> None:
        self.dataset = dataset
        self.case_ids = list(case_ids)
        self.records = {case_id: dataset.records[case_id] for case_id in self.case_ids}
        self.identity = {**dataset.identity, "partition": name}

    def load_case(self, case_id: str) -> Any:
        if case_id not in self.records:
            raise KeyError(f"Case is outside partition: {case_id}")
        return self.dataset.load_case(case_id)


def _memory_partitions() -> tuple[MemoryDataset, MemoryDataset]:
    return MemoryDataset("fault-train", 20261011), MemoryDataset("fault-valid", 20261012)


def _config(
    *,
    seed: int = 71,
    max_epochs: int = 2,
    max_steps: int | None = 20,
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(variant="E4", dropout=0.35),
        mode="smoke",
        max_epochs=max_epochs,
        patience=5,
        learning_rate=5e-4,
        weight_decay=1e-3,
        seed=seed,
        device="cpu",
        max_steps=max_steps,
        memory_limit_bytes=memory_limit_bytes,
        num_threads=2,
        deterministic=True,
    )


def _run(
    train: Any,
    validation: Any,
    output: Path,
    config: TrainConfig,
    *,
    deadline: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_fit(
        train,
        validation,
        output,
        config,
        fit_identity=FIT_IDENTITY,
        deadline=deadline if deadline is not None else time.monotonic() + 300,
        **kwargs,
    )


def _state(output: Path) -> dict[str, Any] | None:
    identity = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    return CheckpointStore(output / "checkpoints", identity).load_latest()


def _committed_epochs(output: Path) -> list[int]:
    manifests = sorted((output / "checkpoints" / "versions").glob("*/commit.json"))
    return [json.loads(path.read_text(encoding="utf-8"))["epoch"] for path in manifests]


def _assert_state_equal(left: Any, right: Any, *, key: str | None = None) -> None:
    if key == "duration_seconds":
        return
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-8)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for child_key in left:
            _assert_state_equal(left[child_key], right[child_key], key=str(child_key))
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_state_equal(left_value, right_value)
    elif isinstance(left, float):
        assert isinstance(right, (float, int))
        assert math.isclose(left, float(right), rel_tol=1e-6, abs_tol=1e-8)
    else:
        assert left == right


def _run_cli(mode: str, config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_mil.py"), "--mode", mode, "--config", str(config_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_public_cli_rejects_seed_mismatch_without_overwriting_original_config(tmp_path: Path) -> None:
    fixture = create_toy_fixture(tmp_path / "fixture", seed=2026, case_count=4, rows_per_lens=ROWS_PER_LENS)
    output = tmp_path / "same-fit"
    config_path = tmp_path / "smoke.json"
    config = {
        "fixture": str(fixture),
        "output": str(output),
        "epochs": 2,
        "seed": 71,
        "device": "cpu",
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    first = _run_cli("smoke", config_path)

    assert first.returncode == 0, first.stdout + first.stderr
    assert json.loads((output / "smoke_report.json").read_text(encoding="utf-8"))["phase"] == "complete"
    original_config = (output / "resolved_config.json").read_bytes()
    original_identity = json.loads(original_config)
    assert _committed_epochs(output) == [0, 1, 2]

    config["seed"] = 72
    config_path.write_text(json.dumps(config), encoding="utf-8")
    second = _run_cli("smoke", config_path)

    assert second.returncode == 1, second.stdout + second.stderr
    assert json.loads(second.stdout)["error_type"] == "CheckpointIdentityError"
    assert (output / "resolved_config.json").read_bytes() == original_config
    assert CheckpointStore(output / "checkpoints", original_identity).load_latest()["epoch"] == 2


def test_real_readonly_fixture_arrays_do_not_warn_or_change_during_torch_conversion(tmp_path: Path) -> None:
    fixture_path = create_toy_fixture(tmp_path / "fixture", seed=30, case_count=4, rows_per_lens=ROWS_PER_LENS)
    dataset = FixtureDataset(fixture_path)
    train_ids = dataset.case_ids[:2]
    validation_ids = dataset.case_ids[2:]
    train = Partition(dataset, train_ids, "train")
    validation = Partition(dataset, validation_ids, "validation")
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    for case_id in dataset.case_ids:
        case = dataset.load_case(case_id)
        assert all(not values.flags.writeable for values in case.features.values())
        snapshots[case_id] = {lens: np.array(values, copy=True) for lens, values in case.features.items()}

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _run(train, validation, tmp_path / "readonly-fit", _config(max_epochs=1, max_steps=4))

    readonly_warnings = [
        str(item.message) for item in captured
        if "not writable" in str(item.message).casefold() or "non-writable" in str(item.message).casefold()
    ]
    assert readonly_warnings == []
    for case_id, by_lens in snapshots.items():
        case = dataset.load_case(case_id)
        for lens, expected in by_lens.items():
            np.testing.assert_array_equal(case.features[lens], expected)


class ControlledOutOfMemory(MemoryError):
    """Deterministic stand-in for an allocation failure at case load time."""


class OutOfMemoryDataset(MemoryDataset):
    def load_case(self, case_id: str) -> SimpleNamespace:
        raise ControlledOutOfMemory("controlled MIL bag allocation failure")


def test_controlled_memoryerror_subclass_reports_last_committed_epoch(tmp_path: Path) -> None:
    train = OutOfMemoryDataset("oom-train", 8)
    validation = MemoryDataset("oom-valid", 9)
    output = tmp_path / "controlled-oom"

    with pytest.raises(ControlledOutOfMemory, match="controlled MIL bag allocation"):
        _run(train, validation, output, _config())

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "error"
    assert status["error_type"] == "ControlledOutOfMemory"
    assert status["last_committed_epoch"] == 0
    assert _committed_epochs(output) == [0]
    assert _state(output)["global_step"] == 0


def test_process_memory_cap_fails_closed_before_any_optimizer_step(tmp_path: Path) -> None:
    train, validation = _memory_partitions()
    output = tmp_path / "resource-cap"

    with pytest.raises(MemoryError, match="exceeded memory_limit_bytes"):
        _run(train, validation, output, _config(memory_limit_bytes=1))

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "error"
    assert status["error_type"] == "MemoryError"
    assert status["last_committed_epoch"] == 0
    assert _committed_epochs(output) == [0]
    assert _state(output)["global_step"] == 0


def test_deadline_interrupt_inside_epoch_keeps_previous_commit_and_replays_exactly(tmp_path: Path) -> None:
    train, validation = _memory_partitions()
    config = _config()
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    _run(train, validation, continuous, config)
    deadline = time.monotonic() + 2.5
    injected = False

    def expire_after_first_step(stage: str, _details: dict[str, Any]) -> None:
        nonlocal injected
        if stage == "after_step" and not injected:
            injected = True
            time.sleep(max(0.0, deadline - time.monotonic() + 0.02))

    paused = _run(train, validation, interrupted, config, deadline=deadline, event_hook=expire_after_first_step)

    assert injected
    assert paused["phase"] == "paused_budget"
    assert paused["interrupted_epoch"] is True
    assert paused["epoch"] == paused["global_step"] == 0
    assert _committed_epochs(interrupted) == [0]
    assert _state(interrupted)["history"] == []
    resumed = _run(train, validation, interrupted, config)
    assert resumed["phase"] == "complete"
    assert _committed_epochs(interrupted) == [0, 1, 2]
    _assert_state_equal(_state(continuous), _state(interrupted))


def test_deadline_expiring_after_final_train_step_pauses_validation_without_committing(tmp_path: Path) -> None:
    train, validation = _memory_partitions()
    output = tmp_path / "deadline-validation"
    deadline = time.monotonic() + 2.5
    observed_steps: list[str] = []

    def expire_after_final_train_step(stage: str, details: dict[str, Any]) -> None:
        if stage == "after_step":
            observed_steps.append(details["case_id"])
            if len(observed_steps) == len(train.case_ids):
                time.sleep(max(0.0, deadline - time.monotonic() + 0.1))

    result = _run(train, validation, output, _config(), deadline=deadline, event_hook=expire_after_final_train_step)

    assert len(observed_steps) == len(train.case_ids)
    assert result["phase"] == "paused_budget"
    assert result.get("interrupted_epoch") is True
    assert result["epoch"] == result["global_step"] == 0
    assert _committed_epochs(output) == [0]
    assert _state(output)["history"] == []
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["phase"] == "paused_budget"


def test_commit_survives_pointer_write_failure_and_resume_reconciles_it(tmp_path: Path) -> None:
    train, validation = _memory_partitions()
    config = _config()
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "pointer-failure"
    _run(train, validation, continuous, config)
    latest_pointer = interrupted / "checkpoints" / "latest.json"

    def obstruct_pointer(stage: str, details: dict[str, Any]) -> None:
        if stage == "before_checkpoint" and details["epoch"] == 1:
            latest_pointer.unlink()
            latest_pointer.mkdir()

    with pytest.raises(OSError):
        _run(train, validation, interrupted, config, event_hook=obstruct_pointer)

    assert _committed_epochs(interrupted) == [0, 1]
    status = json.loads((interrupted / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "error"
    assert status["last_committed_epoch"] == 1
    read_only = CheckpointStore(
        interrupted / "checkpoints",
        json.loads((interrupted / "resolved_config.json").read_text(encoding="utf-8")),
    ).load_latest(reconcile_pointers=False)
    assert read_only["epoch"] == 1
    assert latest_pointer.is_dir()
    latest_pointer.rmdir()
    recovered = _state(interrupted)
    assert recovered["epoch"] == 1
    assert latest_pointer.is_file()

    resumed = _run(train, validation, interrupted, config)
    assert resumed["phase"] == "complete"
    assert _committed_epochs(interrupted) == [0, 1, 2]
    _assert_state_equal(_state(continuous), _state(interrupted))


def test_abrupt_child_exit_recovers_stale_lock_and_matches_continuous_resume(tmp_path: Path) -> None:
    fixture_path = create_toy_fixture(tmp_path / "fixture", seed=17, case_count=4, rows_per_lens=ROWS_PER_LENS)
    dataset = FixtureDataset(fixture_path)
    train_ids = [dataset.case_ids[0], dataset.case_ids[1]]
    validation_ids = [dataset.case_ids[2], dataset.case_ids[3]]
    train = Partition(dataset, train_ids, "train")
    validation = Partition(dataset, validation_ids, "validation")
    continuous = tmp_path / "continuous"
    crashed = tmp_path / "crashed"
    config = _config()
    _run(train, validation, continuous, config)

    child_script = tmp_path / "abrupt_worker.py"
    child_script.write_text(
        """from __future__ import annotations
import os, sys, time
from pathlib import Path
from histology_data.mil_data import FixtureDataset
from histology_data.mil_model import ModelConfig
from histology_data.mil_training import TrainConfig, run_fit

class Part:
    def __init__(self, dataset, case_ids, name):
        self.dataset = dataset
        self.case_ids = list(case_ids)
        self.records = {case: dataset.records[case] for case in self.case_ids}
        self.identity = {**dataset.identity, 'partition': name}
    def load_case(self, case_id):
        if case_id not in self.records:
            raise KeyError(case_id)
        return self.dataset.load_case(case_id)

fixture, output, train_csv, valid_csv = sys.argv[1:]
dataset = FixtureDataset(Path(fixture))
train = Part(dataset, train_csv.split(','), 'train')
validation = Part(dataset, valid_csv.split(','), 'validation')
config = TrainConfig(
    model=ModelConfig(variant='E4', dropout=0.35), mode='smoke', max_epochs=2,
    patience=5, learning_rate=5e-4, weight_decay=1e-3, seed=71,
    device='cpu', max_steps=20, memory_limit_bytes=2 * 1024**3,
    num_threads=2, deterministic=True,
)
def crash_after_optimizer_step(stage, details):
    if stage == 'after_step':
        os._exit(23)
run_fit(
    train, validation, Path(output), config,
    fit_identity={'fit_id': 'mil-fault-e2e', 'protocol': 'smoke-resume-v1'},
    deadline=time.monotonic() + 300, event_hook=crash_after_optimizer_step,
)
""",
        encoding="utf-8",
    )
    child = subprocess.run(
        [
            sys.executable,
            str(child_script),
            str(fixture_path),
            str(crashed),
            ",".join(train_ids),
            ",".join(validation_ids),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert child.returncode == 23, child.stdout + child.stderr
    assert _committed_epochs(crashed) == [0]
    stale_lock = json.loads((crashed / ".fit.lock").read_text(encoding="utf-8"))
    assert stale_lock["pid"] != os.getpid()
    with pytest.raises(ProcessLookupError):
        os.kill(stale_lock["pid"], 0)

    resumed = _run(train, validation, crashed, config, recover_stale_lock=True)

    assert resumed["phase"] == "complete"
    assert _committed_epochs(crashed) == [0, 1, 2]
    final_state = _state(crashed)
    assert final_state["epoch"] == 2 and final_state["global_step"] == 4
    assert [len(row["case_order"]) for row in final_state["history"]] == [2, 2]
    _assert_state_equal(_state(continuous), final_state)
