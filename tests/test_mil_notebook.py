"""Execute the actual notebook bootstrap cells with only root/hash overrides."""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import zipfile
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts.build_mil_runtime import _runtime_files, build_runtime_archive

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Colab_MIL_Training.ipynb"


def _load_notebook() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return notebook


def _code(notebook: dict[str, Any]) -> list[tuple[int, str]]:
    return [(index, "".join(cell["source"])) for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"]


def test_runtime_builder_is_reproducible_and_contains_only_code_and_requirements(tmp_path: Path) -> None:
    first = build_runtime_archive(ROOT, tmp_path / "first.zip")
    second = build_runtime_archive(ROOT, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == len(first["files"])
    assert "histology_data/mil_case_cache.py" in first["files"]
    assert "histology_data/mil_runtime.py" in first["files"]
    assert "scripts/run_mil.py" in first["files"]
    assert "requirements-mil.txt" in first["files"]
    assert "requirements-pretrain.txt" in first["files"]
    assert "requirements-data.txt" in first["files"]
    assert all(
        (name.startswith("histology_data/") and name.endswith(".py"))
        or name == "scripts/run_mil.py"
        or (name.startswith("requirements") and name.endswith(".txt"))
        for name in first["files"]
    )
    assert not any(part in {"data", "weights", "tests", ".git"} for name in first["files"]
                   for part in Path(name).parts)
    with zipfile.ZipFile(first["path"]) as archive:
        assert sorted(archive.namelist()) == first["files"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())

    settings = next(text for _, text in _code(_load_notebook()) if "PINNED_RUNTIME_SHA256" in text)
    match = re.search(r'PINNED_RUNTIME_SHA256\s*=\s*"([0-9a-f]{64})"', settings)
    assert match is not None
    assert match.group(1) == first["sha256"]


def test_unrelated_encoder_modules_cannot_change_the_pinned_mil_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for path, relative in _runtime_files(ROOT):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    first = build_runtime_archive(source, tmp_path / "first.zip")
    (source / "histology_data/unrelated_encoder.py").write_text("# unrelated future encoder\n")
    (source / "requirements-unrelated.txt").write_text("huge-library==1.0\n")
    second = build_runtime_archive(source, tmp_path / "second.zip")
    assert second["sha256"] == first["sha256"]
    assert "histology_data/unrelated_encoder.py" not in second["files"]
    assert "requirements-unrelated.txt" not in second["files"]


def test_actual_notebook_bootstrap_cells_use_only_root_and_sha_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "synthetic-drive-root"
    runtime_zip = root / "runtime" / "histology-mil-runtime.zip"
    runtime_zip.parent.mkdir(parents=True)
    build = build_runtime_archive(ROOT, runtime_zip)
    monkeypatch.setenv("HISTOLOGY_ROOT", str(root))
    monkeypatch.setenv("HISTOLOGY_MIL_RUNTIME_SHA256", str(build["sha256"]))

    notebook = _load_notebook()
    cells = dict(_code(notebook))
    environment_cell = next(text for text in cells.values() if "GOOGLE_PARENT_AVAILABLE" in text)
    settings_cell = next(text for text in cells.values() if "RUN_MODE =" in text)
    bootstrap_cell = next(text for text in cells.values() if "def bootstrap_runtime" in text)
    namespace: dict[str, Any] = {"__name__": "mil_notebook_test"}
    previous_path = list(sys.path)
    try:
        exec(compile(environment_cell, "notebook-environment", "exec"), namespace)
        assert namespace["IS_COLAB"] is False
        assert namespace["ROOT"] == root
        first_session_start = namespace["SESSION_STARTED_UTC"]
        exec(compile(settings_cell, "notebook-settings", "exec"), namespace)
        assert namespace["RUN_MODE"] == "smoke"
        assert namespace["BUNDLE_PATH"] == root / "bundles" / namespace["BUNDLE_ID"]
        assert namespace["FEATURE_ROOT"] == root / "features" / "v001"
        assert namespace["RUNTIME_ZIP"] == runtime_zip
        assert namespace["EXPECTED_RUNTIME_SHA256"] == build["sha256"]
        exec(compile(bootstrap_cell, "notebook-runtime-bootstrap", "exec"), namespace)
        runtime_path = namespace["RUNTIME_PATH"]
        assert runtime_path.is_dir()
        assert (runtime_path / "histology_data" / "mil_case_cache.py").is_file()
        assert (runtime_path / "histology_data" / "mil_runtime.py").is_file()
        assert (runtime_path / "scripts" / "run_mil.py").is_file()
        assert namespace["bootstrap_runtime"](
            runtime_zip, build["sha256"], namespace["RUNTIME_ROOT"],
        ) == runtime_path
        exec(compile(environment_cell, "notebook-environment-rerun", "exec"), namespace)
        assert namespace["SESSION_STARTED_UTC"] == first_session_start
        assert namespace["SESSION_STOP_UTC"] == first_session_start + timedelta(minutes=300)
        malicious = zipfile.ZipInfo("../escape.py")
        with pytest.raises(ValueError, match="unsafe path"):
            namespace["_safe_runtime_member"](malicious)
        with pytest.raises(ValueError, match="SHA-256"):
            namespace["bootstrap_runtime"](runtime_zip, "0" * 64, namespace["RUNTIME_ROOT"])
    finally:
        sys.path[:] = previous_path


def test_notebook_cells_have_safe_defaults_and_no_remote_runtime_controls() -> None:
    code_cells = _code(_load_notebook())
    combined = "\n".join(source for _, source in code_cells)
    for index, source in code_cells:
        ast.parse(source, filename=f"Colab_MIL_Training.ipynb:cell-{index}")
    assert 'RUN_MODE = "smoke"' in combined
    assert "SESSION_BUDGET_MINUTES = 330" in combined
    assert "SESSION_RESERVE_MINUTES = 30" in combined
    assert "SESSION_WORK_MINUTES = SESSION_BUDGET_MINUTES - SESSION_RESERVE_MINUTES" in combined
    assert "SMOKE_EPOCHS = 2" in combined
    assert "MAX_SMOKE_CASES = 4" in combined
    assert "MAX_SMOKE_ROWS_PER_LENS = 64" in combined
    assert "drive.mount(str(DRIVE_MOUNT))" in combined
    assert "if IS_COLAB:" in combined
    lowered = combined.casefold()
    for forbidden in ("autossh", "keepalive", "create_vm", "create_instance", "google.cloud.compute"):
        assert forbidden not in lowered
    assert "pip install torch" not in lowered
    assert not re.search(r"(?:AIza[0-9A-Za-z_-]{30,}|ghp_[0-9A-Za-z]{30,}|-----BEGIN .*PRIVATE KEY-----)", combined)


def test_actual_cache_stage_cell_and_pilot_production_configs_use_materialized_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_cells = dict(_code(_load_notebook()))
    stage_source = next(source for source in code_cells.values() if "case_cache_path = prepare_case_cache" in source)
    run_source = next(source for source in code_cells.values() if 'if RUN_MODE == "smoke":' in source)

    calls: dict[str, Any] = {}
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _index: "Tesla T4",
    )
    fake_cache = ModuleType("histology_data.mil_case_cache")
    fake_runtime = ModuleType("histology_data.mil_runtime")

    staged_bundle = tmp_path / "stage" / "bundle"
    staged_features = tmp_path / "stage" / "features"
    expected_cache = tmp_path / "stage" / "case-caches" / "cache-identity"

    def stage_inputs(bundle: Path, feature_root: Path, stage_root: Path, *, deadline: float) -> dict[str, Any]:
        calls["stage"] = (bundle, feature_root, stage_root, deadline)
        return {
            "bundle_path": staged_bundle,
            "feature_root": staged_features,
            "staged_bytes": 123,
            "bundle_id": "b" * 64,
        }

    def prepare_cache(bundle: Path, feature_root: Path, cache_root: Path, deadline: float) -> Path:
        calls["cache"] = (bundle, feature_root, cache_root, deadline)
        return expected_cache

    fake_runtime.stage_training_inputs = stage_inputs
    fake_cache.prepare_case_cache = prepare_cache
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "histology_data.mil_runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "histology_data.mil_case_cache", fake_cache)
    namespace: dict[str, Any] = {
        "BUNDLE_PATH": tmp_path / "drive" / "bundle",
        "FEATURE_ROOT": tmp_path / "drive" / "features",
        "STAGE_ROOT": tmp_path / "stage",
        "_monotonic_deadline": lambda: 9876.5,
    }
    exec(compile(stage_source, "notebook-cache-stage", "exec"), namespace)

    assert namespace["case_cache_path"] == expected_cache
    assert calls["stage"] == (
        namespace["BUNDLE_PATH"], namespace["FEATURE_ROOT"], namespace["STAGE_ROOT"], 9876.5,
    )
    assert calls["cache"] == (staged_bundle, staged_features, namespace["STAGE_ROOT"] / "case-caches", 9876.5)

    def run_branch(mode: str) -> tuple[str, dict[str, Any]]:
        captured: list[tuple[str, dict[str, Any], int]] = []
        branch_namespace: dict[str, Any] = {
            "RUN_MODE": mode,
            "RUN_OUTPUT_ROOT": tmp_path / "drive" / "runs",
            "staged": {"bundle_id": "b" * 64, "bundle_path": staged_bundle, "feature_root": staged_features},
            "EXPECTED_RUNTIME_SHA256": "a" * 64,
            "case_cache_path": expected_cache,
            "PILOT_EPOCHS": 1,
            "PRODUCTION_VARIANTS": ["E0"],
            "PRODUCTION_EPOCHS": 1,
            "_seconds_until_stop": lambda: 120.0,
            "_run_mil_cli": lambda name, config, *, timeout_cap_seconds: (
                captured.append((name, config, timeout_cap_seconds)) or {"phase": "complete"}
            ),
        }
        exec(compile(run_source, f"notebook-{mode}-config", "exec"), branch_namespace)
        assert len(captured) == 1
        name, config, _timeout = captured[0]
        return name, config

    pilot_name, pilot_config = run_branch("pilot")
    production_name, production_config = run_branch("production")
    assert pilot_name == "pilot" and pilot_config["case_cache"] == str(expected_cache)
    assert production_name == "production" and production_config["case_cache"] == str(expected_cache)
