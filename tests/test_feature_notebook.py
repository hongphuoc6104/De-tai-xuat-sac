"""Structural checks for the operator-facing feature extraction notebook."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Colab_Feature_Extraction.ipynb"
CONFIG_PATH = ROOT / "configs" / "features_colab.json"
GUIDE_PATH = ROOT / "docs" / "FEATURE_EXTRACTION.md"


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cells_of_type(notebook: dict[str, object], cell_type: str) -> list[dict[str, object]]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return [cell for cell in cells if cell["cell_type"] == cell_type]


def test_notebook_schema_and_code_cells_are_valid_python() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 4
    code_cells = _cells_of_type(notebook, "code")
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_first_cell_short_circuits_when_google_parent_is_missing(monkeypatch) -> None:
    notebook = _load_notebook()
    first_code = _cells_of_type(notebook, "code")[0]
    source = "".join(first_code["source"])
    real_find_spec = importlib.util.find_spec
    calls = []

    def find_spec_without_google_parent(name: str, package: str | None = None):
        calls.append(name)
        if name == "google":
            return None
        if name == "google.colab":
            raise AssertionError("google.colab must not be checked without its parent package")
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_google_parent)
    namespace: dict[str, object] = {}
    exec(compile(source, f"{NOTEBOOK_PATH.name}:first-cell", "exec"), namespace)
    assert namespace["IS_COLAB"] is False
    assert calls == ["google"]


def test_runtime_is_loaded_before_project_import_without_reinstalling_gpu_stack() -> None:
    notebook = _load_notebook()
    code = "\n".join("".join(cell["source"]) for cell in _cells_of_type(notebook, "code"))
    extraction = code.index("with zipfile.ZipFile(RUNTIME_ZIP)")
    path_insert = code.index("sys.path.insert(0, str(RUNTIME_ROOT))")
    feature_import = code.index("importlib.import_module('histology_data.features')")
    assert extraction < path_insert < feature_import
    assert "histology-feature-runtime.zip" in code
    assert "histology-data-runtime.zip" in code
    assert "subprocess" not in code
    assert "pip install" not in code


def test_config_is_a_complete_drive_bound_feature_run_config() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    from histology_data.features import FeatureRunConfig

    parsed = FeatureRunConfig.from_dict(config)
    required = {
        "metadata", "source_root", "source_kind", "output_root", "work_root", "weights_dir",
        "batch_size", "device", "precision", "expected_archives", "expected_pngs",
        "directory_part_size", "max_new_parts", "max_patches_per_part", "budget_minutes",
        "reserve_minutes", "retries", "aliases", "recover_lock",
    }
    assert set(config) == required
    assert config["source_kind"] == "zip"
    assert config["source_root"] == "/content/drive/MyDrive/histology/source/archives"
    assert config["output_root"] == "/content/drive/MyDrive/histology/features/v001"
    assert config["expected_archives"] == 25
    assert config["expected_pngs"] == 148991
    assert config["budget_minutes"] == 240
    assert config["reserve_minutes"] == 30
    assert config["recover_lock"] is False
    assert parsed.source_kind == "zip"


def test_smoke_and_production_have_separate_namespaces_and_limits() -> None:
    notebook = _load_notebook()
    code = "\n".join("".join(cell["source"]) for cell in _cells_of_type(notebook, "code"))
    assert "OUTPUT_ROOT = FEATURES_ROOT / 'smoke_v001'" in code
    assert "OUTPUT_ROOT = FEATURES_ROOT / 'v001'" in code
    assert "MAX_NEW_PARTS = 1" in code
    assert "MAX_PATCHES_PER_PART = 6" in code
    assert "MAX_NEW_PARTS = PRODUCTION_MAX_NEW_PARTS" in code
    assert "MAX_PATCHES_PER_PART = None" in code


def test_notebook_supports_partial_zip_uploads_and_single_source_mode() -> None:
    notebook = _load_notebook()
    code = "\n".join("".join(cell["source"]) for cell in _cells_of_type(notebook, "code"))
    assert "SOURCE_KIND = 'zip'" in code
    assert "SOURCE_KIND not in {'zip', 'directory'}" in code
    assert "Thiếu ZIP không chặn lượt chạy từng phần." in code
    assert "assert len(available_sources)" not in code
    assert "source_kind=SOURCE_KIND" in code
    assert "output_root=str(OUTPUT_ROOT)" in code


def test_notebook_passes_budget_and_requires_explicit_stale_lock_recovery() -> None:
    notebook = _load_notebook()
    code = "\n".join("".join(cell["source"]) for cell in _cells_of_type(notebook, "code"))
    module = ast.parse(code)
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    run_config = next(
        node for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_dict"
    )
    assert isinstance(run_config.func.value, ast.Name)
    assert run_config.func.value.id == "FeatureRunConfig"
    assert ast.unparse(run_config.args[0]) == "run_values"
    assert "budget_minutes" in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "reserve_minutes" in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for setting in (
        "BATCH_SIZE = 32",
        "DEVICE = 'cuda'",
        "PRECISION = 'fp32'",
        "BUDGET_MINUTES = 240",
        "RESERVE_MINUTES = 30",
        "RECOVER_STALE_LOCK = False",
        "PRODUCTION_MAX_NEW_PARTS = None",
        "batch_size=BATCH_SIZE",
        "device=DEVICE",
        "precision=PRECISION",
        "budget_minutes=BUDGET_MINUTES",
        "reserve_minutes=RESERVE_MINUTES",
        "recover_lock=RECOVER_STALE_LOCK",
        "if DEVICE.startswith('cuda')",
    ):
        assert setting in code
    assert "RECOVER_STALE_LOCK = CONFIG.get" not in code


def test_operator_guide_covers_resume_runtime_budget_and_medical_scope() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    required_phrases = (
        "awaiting_sources",
        "budget_exhausted",
        "T4",
        "330 phút/ngày",
        "danh tính bệnh nhân chưa được xác minh",
        "training_ready",
        "recover_lock=false",
        "WEIGHTS_DIR",
        "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
        "partial_release.json",
        "release_snapshot.json",
        "không stage lại",
    )
    assert all(phrase in guide for phrase in required_phrases)
