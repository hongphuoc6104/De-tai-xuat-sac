"""Execute the actual notebook bootstrap cells with only root/hash overrides."""
from __future__ import annotations

import ast
import json
import re
import sys
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.build_mil_runtime import build_runtime_archive

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
