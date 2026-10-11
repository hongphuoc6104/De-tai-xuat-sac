"""Filesystem-backed transaction and fit-lock tests for MIL checkpoints."""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from histology_data.mil_checkpoint import (
    ActiveFitLockError,
    CheckpointIdentityError,
    CheckpointIntegrityError,
    CheckpointStore,
    FitLock,
    UnsafeStaleLockError,
)

IDENTITY = {
    "run_mode": "smoke",
    "fit_id": "synthetic-fit-a",
    "data_digest": "sha256:synthetic-data",
    "resume_contract": {"model": "tiny-mil-v1", "rng": "python-torch-v1"},
}


def _state(epoch: int, *, best: bool = False) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": {"weight": torch.tensor([float(epoch), 2.0])},
        "rng_state": {"python": (3, (1, 2, 3), None), "marker": "best" if best else "regular"},
        "numpy_rng_state": np.random.RandomState(epoch).get_state(),
    }


class _InjectedFailure(RuntimeError):
    pass


def _raise_at(expected_stage: str):
    def inject(stage: str) -> None:
        if stage == expected_stage:
            raise _InjectedFailure(stage)

    return inject


def test_publish_writes_immutable_versions_and_loads_latest_and_best(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "run", IDENTITY)
    initial = store.publish(_state(0), epoch=0)
    best = store.publish(_state(2, best=True), epoch=2, is_best=True)
    latest = store.publish(_state(3), epoch=3)
    latest_state = store.load_latest()

    assert initial.epoch == 0
    assert best.epoch == 2
    assert latest.epoch == 3
    assert Path(best.state_path).is_file()
    assert torch.equal(latest_state["model_state"]["weight"], torch.tensor([3.0, 2.0]))
    assert latest_state["rng_state"]["marker"] == "regular"
    assert np.array_equal(
        latest_state["numpy_rng_state"][1],
        _state(3)["numpy_rng_state"][1],
    )
    assert store.load_best()["rng_state"]["marker"] == "best"
    assert json.loads((tmp_path / "run" / "latest.json").read_text())[
        "version"
    ] == latest.version
    assert json.loads((tmp_path / "run" / "best.json").read_text())[
        "version"
    ] == best.version
    assert len(list((tmp_path / "run" / "versions").glob("*/commit.json"))) == 3
    assert json.loads(json.dumps(dataclasses.asdict(latest)))["state_path"] == latest.state_path


def test_latest_uses_highest_epoch_even_when_a_lower_epoch_was_published_later(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "run", IDENTITY)
    store.publish(_state(9), epoch=9)
    store.publish(_state(2), epoch=2)

    assert store.load_latest()["epoch"] == 9


def test_load_best_returns_none_until_a_best_version_is_committed(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "run", IDENTITY)
    assert store.load_latest() is None
    assert store.load_best() is None

    store.publish(_state(0), epoch=0)

    assert store.load_best() is None


@pytest.mark.parametrize("stage", ["after_payload", "after_copy_before_commit"])
def test_incomplete_publish_is_ignored_and_preserves_last_valid_state(tmp_path: Path, stage: str) -> None:
    root = tmp_path / "run"
    CheckpointStore(root, IDENTITY).publish(_state(0), epoch=0)
    broken_store = CheckpointStore(root, IDENTITY, fault_injector=_raise_at(stage))

    with pytest.raises(_InjectedFailure, match=stage):
        broken_store.publish(_state(1), epoch=1)

    assert CheckpointStore(root, IDENTITY).load_latest()["epoch"] == 0
    assert len(list((root / "versions").glob("*/commit.json"))) == 1
    assert len(list((root / "versions").glob("*/state.pt"))) == (1 if stage == "after_payload" else 2)


def test_committed_publish_is_reconciled_after_pointer_stage_failure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    store = CheckpointStore(root, IDENTITY, fault_injector=_raise_at("after_commit_before_pointer"))

    with pytest.raises(_InjectedFailure, match="after_commit_before_pointer"):
        store.publish(_state(0), epoch=0)

    assert not (root / "latest.json").exists()
    assert len(list((root / "versions").glob("*/commit.json"))) == 1
    assert CheckpointStore(root, IDENTITY).load_latest()["epoch"] == 0
    assert (root / "latest.json").is_file()


def test_corrupt_committed_payload_fails_closed_instead_of_falling_back(tmp_path: Path) -> None:
    root = tmp_path / "run"
    store = CheckpointStore(root, IDENTITY)
    store.publish(_state(0), epoch=0)
    latest = store.publish(_state(1), epoch=1)
    Path(latest.state_path).write_bytes(Path(latest.state_path).read_bytes() + b"tampered")

    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        store.load_latest()


def test_identity_mismatch_rejects_resume_before_deserialization(tmp_path: Path) -> None:
    root = tmp_path / "run"
    CheckpointStore(root, IDENTITY).publish(_state(0), epoch=0)
    incompatible = {**IDENTITY, "data_digest": "sha256:other-data"}

    with pytest.raises(CheckpointIdentityError, match="identity mismatch"):
        CheckpointStore(root, incompatible).load_latest()


def test_malformed_commit_record_is_treated_as_committed_corruption(tmp_path: Path) -> None:
    root = tmp_path / "run"
    receipt = CheckpointStore(root, IDENTITY).publish(_state(0), epoch=0)
    (Path(receipt.state_path).parent / "commit.json").write_text("{truncated", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="Unreadable committed checkpoint manifest"):
        CheckpointStore(root, IDENTITY).load_latest()


def test_fit_lock_blocks_a_second_owner_in_the_same_process_and_releases_on_exit(tmp_path: Path) -> None:
    root = tmp_path / "fit"
    with FitLock(root, owner_id="first") as lock:
        assert lock.metadata["pid"] == os.getpid()
        assert lock.metadata["host"] == socket.gethostname()
        assert "boot_id" in lock.metadata
        assert "session_id" in lock.metadata
        assert lock.metadata["process_start"]
        with pytest.raises(ActiveFitLockError, match="locked"):
            FitLock(root, owner_id="second").acquire()
        assert (root / ".fit.lock").is_file()

    assert not (root / ".fit.lock").exists()
    with FitLock(root, owner_id="second") as lock:
        assert lock.metadata["owner_id"] == "second"


def test_stale_same_host_lock_from_a_crashed_process_can_be_recovered(tmp_path: Path) -> None:
    root = tmp_path / "fit"
    code = (
        "from histology_data.mil_checkpoint import FitLock; import os, sys; "
        "lock = FitLock(sys.argv[1], owner_id='crashed-worker'); lock.acquire(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code, str(root)], check=True, cwd=Path(__file__).parents[1], timeout=20)
    stale = json.loads((root / ".fit.lock").read_text(encoding="utf-8"))

    assert stale["owner_id"] == "crashed-worker"
    assert stale["host"] == socket.gethostname()
    with pytest.raises(ActiveFitLockError, match="locked"):
        FitLock(root, owner_id="no-recovery-default").acquire()
    with FitLock(root, owner_id="recovery-worker", recover_stale=True) as lock:
        assert lock.metadata["owner_id"] == "recovery-worker"

    assert not (root / ".fit.lock").exists()


def test_cross_host_lock_is_never_recovered_without_external_receipt(tmp_path: Path) -> None:
    root = tmp_path / "fit"
    root.mkdir()
    lock_path = root / ".fit.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "token": "foreign-token",
                "owner_id": "remote-worker",
                "pid": 987654321,
                "host": "another-host",
                "boot_id": "unknown",
                "session_id": None,
                "process_start": None,
                "acquired_at_ns": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsafeStaleLockError, match="another host"):
        FitLock(root, recover_stale=True).acquire()
    assert lock_path.is_file()


def test_cross_host_lock_recovery_requires_exact_operator_receipt(tmp_path: Path) -> None:
    root = tmp_path / "fit"
    root.mkdir()
    lock_path = root / ".fit.lock"
    old_metadata = {
        "schema_version": 1,
        "token": "remote-owner-token",
        "owner_id": "remote-worker",
        "pid": 987654321,
        "host": "another-host",
        "boot_id": "remote-boot-17",
        "session_id": 36,
        "process_start": "101",
        "acquired_at_ns": 1,
    }
    lock_path.write_text(json.dumps(old_metadata), encoding="utf-8")
    evidence = {
        "token": old_metadata["token"],
        "host": old_metadata["host"],
        "boot_id": old_metadata["boot_id"],
        "session_id": old_metadata["session_id"],
        "old_runtime_stopped": True,
        "evidence": "Colab runtime stop confirmed in manager session runtime-17",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    wrong_owner_evidence = {**evidence, "token": "different-lock-token"}

    with pytest.raises(UnsafeStaleLockError, match="does not match.*token"):
        FitLock(root, recover_stale=True, recovery_evidence=wrong_owner_evidence).acquire()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == old_metadata["token"]

    with FitLock(root, owner_id="new-local-worker", recover_stale=True, recovery_evidence=evidence) as lock:
        assert lock.metadata["owner_id"] == "new-local-worker"
        assert lock.metadata["host"] == socket.gethostname()
    assert not lock_path.exists()


def test_live_current_pid_lock_is_not_taken_over_even_when_recovery_is_enabled(tmp_path: Path) -> None:
    root = tmp_path / "fit"
    with FitLock(root, owner_id="live-owner"):
        with pytest.raises(ActiveFitLockError, match="locked"):
            FitLock(root, owner_id="intruder", recover_stale=True).acquire()
