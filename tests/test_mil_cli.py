"""Real subprocess contracts for the bounded MIL operator entry point."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from histology_data.mil_checkpoint import CheckpointStore

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_mil.py"
SUBPROCESS_TIMEOUT_SECONDS = 60


def _write_config(path: Path, values: dict[str, Any]) -> Path:
    path.write_text(json.dumps(values, sort_keys=True), encoding="utf-8")
    return path


def _invoke(
    mode: str,
    config: Path,
    *,
    deadline_utc: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CLI), "--mode", mode, "--config", str(config)]
    if deadline_utc is not None:
        command.extend(["--deadline-utc", deadline_utc])
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        env=env,
    )


def _json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"CLI emitted no JSON output. stderr={result.stderr!r}"
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def toy_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mil-cli")
    fixture_path = root / "toy-fixture"
    config_path = _write_config(
        root / "toy-config.json",
        {"output": str(fixture_path), "seed": 913, "case_count": 4, "rows": 16},
    )
    result = _invoke("toy", config_path)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    response = _json_stdout(result)
    assert response["phase"] == "fixture_complete"
    assert Path(response["path"]) == fixture_path
    return {"root": root, "path": fixture_path, "manifest": json.loads(
        (fixture_path / "fixture.json").read_text(encoding="utf-8"),
    )}


def test_toy_cli_creates_four_case_fixture_with_case_level_only_targets(
    toy_fixture: dict[str, Any],
) -> None:
    manifest = toy_fixture["manifest"]
    assert manifest["mode"] == "smoke"
    assert manifest["case_count"] == 4
    assert manifest["production_eligible"] is False
    assert {row["target"] for row in manifest["cases"]} == {0, 1}

    for case in manifest["cases"]:
        assert set(case) == {"case_id", "patient_id", "target", "arrays"}
        assert set(case["arrays"]) == {"4", "10", "40"}
        for entry in case["arrays"].values():
            assert set(entry) == {"path", "sha256", "shape", "dtype"}
            assert entry["shape"] == [16, 2048]
            assert entry["dtype"] == "float32"
            array = np.load(toy_fixture["path"] / entry["path"], allow_pickle=False)
            assert array.shape == (16, 2048)
            assert array.dtype == np.float32
            assert hashlib.sha256((toy_fixture["path"] / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_smoke_cli_runs_two_epochs_and_completed_rerun_adds_no_steps_or_versions(
    toy_fixture: dict[str, Any],
) -> None:
    fit_output = toy_fixture["root"] / "smoke-fit"
    config_path = _write_config(
        toy_fixture["root"] / "smoke-config.json",
        {
            "fixture": str(toy_fixture["path"]),
            "output": str(fit_output),
            "variant": "E4",
            "epochs": 2,
            "dropout": 0.2,
            "seed": 31,
            "device": "cpu",
        },
    )
    before_hashes = {
        entry["path"]: entry["sha256"]
        for case in toy_fixture["manifest"]["cases"]
        for entry in case["arrays"].values()
    }

    first = _invoke("smoke", config_path)
    assert first.returncode == 0, f"stdout={first.stdout}\nstderr={first.stderr}"
    first_result = _json_stdout(first)
    assert first_result["phase"] == "complete"
    assert first_result["epoch"] == 2 and first_result["global_step"] == 4

    report = json.loads((fit_output / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["phase"] == "complete"
    assert report["mode"] == "smoke"
    assert report["clinical_metrics"] is False
    assert len(report["history"]) == 2
    resolved = json.loads((fit_output / "resolved_config.json").read_text(encoding="utf-8"))
    state = CheckpointStore(fit_output / "checkpoints", resolved).load_latest()
    assert state["finished"] is True
    assert state["global_step"] == 4
    assert state["epoch"] == 2
    versions = sorted(path.parent.name for path in (fit_output / "checkpoints" / "versions").glob("*/commit.json"))
    assert len(versions) == 3  # Initial state plus completed epochs one and two.

    for case in toy_fixture["manifest"]["cases"]:
        assert type(case["target"]) is int
        first_lens_entry = next(iter(case["arrays"].values()))
        assert set(first_lens_entry) == {"path", "sha256", "shape", "dtype"}
    after_hashes = {
        entry["path"]: hashlib.sha256((toy_fixture["path"] / entry["path"]).read_bytes()).hexdigest()
        for case in toy_fixture["manifest"]["cases"]
        for entry in case["arrays"].values()
    }
    assert after_hashes == before_hashes

    second = _invoke("smoke", config_path)
    assert second.returncode == 0, f"stdout={second.stdout}\nstderr={second.stderr}"
    second_result = _json_stdout(second)
    assert second_result["phase"] == "complete" and second_result["resumed"] is True
    rerun_state = CheckpointStore(fit_output / "checkpoints", resolved).load_latest()
    rerun_versions = sorted(
        path.parent.name for path in (fit_output / "checkpoints" / "versions").glob("*/commit.json")
    )
    assert rerun_state["global_step"] == state["global_step"] == 4
    assert rerun_state["history"] == state["history"]
    assert rerun_versions == versions


def test_production_without_deadline_fails_before_cuda_or_vm_operations(tmp_path: Path) -> None:
    guard = tmp_path / "python-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import torch\n"
        "def reject_cuda_probe():\n"
        "    raise AssertionError('CUDA probe happened before deadline validation')\n"
        "torch.cuda.is_available = reject_cuda_probe\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
    output = tmp_path / "must-not-be-created"
    config = _write_config(
        tmp_path / "production-config.json",
        {"output": str(output), "bundle": str(tmp_path / "missing-bundle"),
         "feature_root": str(tmp_path / "missing-features"), "variants": ["E0"]},
    )

    result = _invoke("production", config, env=env)
    assert result.returncode == 1, f"stdout={result.stdout}\nstderr={result.stderr}"
    response = _json_stdout(result)
    assert response["phase"] == "error"
    assert response["error_type"] == "ValueError"
    assert "explicit bounded --deadline-utc" in response["error"]
    assert not output.exists()

    tree = ast.parse(CLI.read_text(encoding="utf-8"), filename=str(CLI))
    forbidden_vm_calls = {
        "create_vm", "create_instance", "allocate_vm", "launch_vm",
        "create_virtual_machine", "create_compute_instance",
    }
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert calls.isdisjoint(forbidden_vm_calls)
    imported_modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported_modules.update(
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(name.startswith(("google.colab", "google.cloud.compute")) for name in imported_modules)


def test_pilot_cli_requires_cuda_and_never_falls_back_to_cpu(tmp_path: Path) -> None:
    guard = tmp_path / "python-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import torch\n"
        "torch.cuda.is_available = lambda: False\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
    output = tmp_path / "pilot-must-not-run"
    config = _write_config(
        tmp_path / "pilot-config.json",
        {"output": str(output), "bundle": str(tmp_path / "missing-bundle"),
         "feature_root": str(tmp_path / "missing-features"), "variant": "E0", "epochs": 1},
    )
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    result = _invoke("pilot", config, deadline_utc=deadline, env=env)

    assert result.returncode == 1, f"stdout={result.stdout}\nstderr={result.stderr}"
    response = _json_stdout(result)
    assert response["phase"] == "error"
    assert response["error_type"] == "RuntimeError"
    assert "Pilot requires actual CUDA" in response["error"]
    assert not output.exists()
