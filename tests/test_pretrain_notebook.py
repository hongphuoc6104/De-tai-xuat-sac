"""Check the CPU-only Colab pretrain-readiness wrapper without touching Drive."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import types
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Colab_PreTrain_Readiness.ipynb"
REVIEW_FIELDS = (
    "case_id", "patient_id", "identity_verified", "identity_evidence", "case_label",
    "label_verified", "label_evidence", "raw_glade_values_json",
    "raw_conclusion_values_json", "candidate_label_codes_json", "metadata_case_sha256",
)


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _cell_with(notebook: dict[str, Any], marker: str) -> str:
    for cell in _code_cells(notebook):
        source = "".join(cell["source"])
        if marker in source:
            return source
    raise AssertionError(f"Notebook code cell containing {marker!r} was not found.")


def _exec(source: str, namespace: dict[str, Any], name: str) -> None:
    exec(compile(source, f"{NOTEBOOK_PATH.name}:{name}", "exec"), namespace)


def _runtime_namespace() -> tuple[dict[str, Any], dict[str, Any]]:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {}
    helper_source = _cell_with(notebook, "def validate_and_extract_runtime")
    _exec(helper_source, namespace, "runtime-helpers")
    return notebook, namespace


def _make_runtime_zip(path: Path, *, cli_source: str | None = None, extra: dict[str, bytes] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "histology_data/__init__.py": b"\"\"\"Synthetic package for notebook tests.\"\"\"\n",
        "histology_data/pretrain_cli.py": (cli_source or "print('{}')\n").encode("utf-8"),
        "requirements-pretrain.txt": b"",
    }
    files.update(extra or {})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_notebook_schema_and_code_cells_are_valid_python() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 4
    cells = _code_cells(notebook)
    assert cells
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:cell-{index}")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_google_colab_probe_does_not_check_child_when_google_parent_is_absent(monkeypatch) -> None:
    notebook = _load_notebook()
    source = _code_cells(notebook)[0]
    calls: list[str] = []

    def find_spec(name: str, package: str | None = None):
        calls.append(name)
        if name == "google":
            return None
        if name == "google.colab":
            raise AssertionError("google.colab must not be checked without the google parent")
        return None

    monkeypatch.setattr(importlib.machinery, "PathFinder", importlib.machinery.PathFinder)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    namespace: dict[str, Any] = {}
    _exec("".join(source["source"]), namespace, "environment")
    assert namespace["IS_COLAB"] is False
    assert calls == ["google"]


def _fake_colab(monkeypatch, mount_calls: list[str]) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.__path__ = []
    drive = types.ModuleType("google.colab.drive")
    drive.mount = lambda path: mount_calls.append(path)
    colab.drive = drive
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, package=None: importlib.machinery.ModuleSpec(name, loader=None)
        if name in {"google", "google.colab"} else None,
    )


def test_drive_mount_is_skipped_when_the_colab_mount_is_already_present(monkeypatch) -> None:
    notebook = _load_notebook()
    mount_calls: list[str] = []
    _fake_colab(monkeypatch, mount_calls)
    monkeypatch.setattr(os.path, "ismount", lambda path: True)
    namespace: dict[str, Any] = {}
    _exec("".join(_code_cells(notebook)[0]["source"]), namespace, "environment")
    assert namespace["IS_COLAB"] is True
    assert mount_calls == []


def test_drive_mount_runs_only_when_colab_exists_and_drive_is_not_mounted(monkeypatch) -> None:
    notebook = _load_notebook()
    mount_calls: list[str] = []
    _fake_colab(monkeypatch, mount_calls)
    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    original_is_dir = Path.is_dir

    def is_dir(path: Path) -> bool:
        if str(path) == "/content/drive/MyDrive":
            return False
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    namespace: dict[str, Any] = {}
    _exec("".join(_code_cells(notebook)[0]["source"]), namespace, "environment")
    assert mount_calls == ["/content/drive"]


def test_runtime_sha_guard_and_safe_extraction_keep_bundle_separate_from_encoder(tmp_path: Path) -> None:
    _, namespace = _runtime_namespace()
    bundle = _make_runtime_zip(tmp_path / "runtime.zip")
    runtime_root = tmp_path / "content" / "histology-pretrain-runtime"
    encoder_root = tmp_path / "drive" / "encoders"
    extract = namespace["validate_and_extract_runtime"]

    with pytest.raises(ValueError, match="reviewed 64-character SHA-256"):
        extract(bundle, "SET_AFTER_BUILD", runtime_root, encoder_root)
    with pytest.raises(ValueError, match="does not match"):
        extract(bundle, "0" * 64, runtime_root, encoder_root)

    result = extract(bundle, _sha256(bundle), runtime_root, encoder_root)
    assert result["sha256"] == _sha256(bundle)
    assert (runtime_root / "histology_data" / "pretrain_cli.py").is_file()
    assert (runtime_root / "requirements-pretrain.txt").is_file()
    assert not encoder_root.exists()
    assert not runtime_root.resolve().is_relative_to(encoder_root.resolve())


def test_runtime_rejects_untrusted_paths_zip_bombs_and_missing_cli_package(tmp_path: Path) -> None:
    _, namespace = _runtime_namespace()
    extract = namespace["validate_and_extract_runtime"]
    encoder_root = tmp_path / "encoders"

    unsafe = _make_runtime_zip(
        tmp_path / "unsafe.zip",
        extra={"../escaped.py": b"nope"},
    )
    with pytest.raises(ValueError, match="unsafe member path"):
        extract(unsafe, _sha256(unsafe), tmp_path / "unsafe-runtime", encoder_root)
    assert not (tmp_path / "escaped.py").exists()

    bomb = _make_runtime_zip(
        tmp_path / "oversized.zip",
        extra={"large.py": b"x" * (2 * 1024 * 1024 + 1)},
    )
    with pytest.raises(ValueError, match="2 MiB"):
        extract(bomb, _sha256(bomb), tmp_path / "oversized-runtime", encoder_root)

    missing_cli = tmp_path / "missing-cli.zip"
    with zipfile.ZipFile(missing_cli, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("histology_data/__init__.py", b"\n")
        archive.writestr("requirements-pretrain.txt", b"\n")
    with pytest.raises(RuntimeError, match="missing histology_data.pretrain_cli"):
        extract(missing_cli, _sha256(missing_cli), tmp_path / "missing-cli-runtime", encoder_root)


def _review_rows(count: int = 18) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        rows.append({
            "case_id": f"CASE_{index:02d}",
            "patient_id": f"PATIENT_{index:02d}",
            "identity_verified": "true",
            "identity_evidence": "reviewed source record",
            "case_label": "0" if index < 10 else "1",
            "label_verified": "true",
            "label_evidence": "reviewed final conclusion",
            "raw_glade_values_json": "[]",
            "raw_conclusion_values_json": "[]",
            "candidate_label_codes_json": "[]",
            "metadata_case_sha256": "a" * 64,
        })
    return rows


def _write_review(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_feature_audit(root: Path, *, verified_vectors: int = 148991, audit_complete: bool = True) -> Path:
    feature_id = "f" * 64
    release_root = root / feature_id
    release_root.mkdir(parents=True)
    (root / "status.json").write_text(json.dumps({
        "status": "complete", "feature_complete": True, "feature_id": feature_id,
        "audit_path": "/stale/Colab/absolute/path/dataset_audit.json",
    }), encoding="utf-8")
    (release_root / "release.json").write_text(json.dumps({
        "feature_id": feature_id, "feature_complete": True, "source_complete": True, "audit_complete": audit_complete,
    }), encoding="utf-8")
    (release_root / "dataset_audit.json").write_text(json.dumps({
        "feature_id": feature_id, "feature_complete": True, "source_complete": True,
        "verified_vectors": verified_vectors,
    }), encoding="utf-8")
    return root


def test_build_preflight_requires_reviewed_rows_and_a_complete_relative_feature_audit(tmp_path: Path) -> None:
    _, namespace = _runtime_namespace()
    review = _write_review(tmp_path / "reviews" / "case_review.csv", _review_rows())
    inspect_review = namespace["inspect_case_review_file"]
    inspect_feature = namespace["inspect_feature_release"]

    accepted_review = inspect_review(review, 18, 18, {"0": 10, "1": 8})
    assert accepted_review["ready"] is True
    assert accepted_review["case_count"] == 18
    assert accepted_review["unique_patient_count"] == 18
    assert accepted_review["label_counts"] == {"0": 10, "1": 8}
    assert not any("PATIENT_" in str(value) for value in accepted_review.values())

    feature_root = _write_feature_audit(tmp_path / "features")
    accepted_feature = inspect_feature(feature_root, 148991)
    assert accepted_feature["ready"] is True
    assert accepted_feature["verified_vectors"] == 148991

    incomplete = _write_feature_audit(tmp_path / "incomplete-features", audit_complete=False)
    blocked_feature = inspect_feature(incomplete, 148991)
    assert blocked_feature["ready"] is False
    assert "feature_release_audit_marker_missing" in blocked_feature["blockers"]

    modified_rows = _review_rows()
    modified_rows[0]["identity_verified"] = "false"
    blocked_review = inspect_review(_write_review(tmp_path / "bad-review.csv", modified_rows), 18, 18,
                                   {"0": 10, "1": 8})
    assert blocked_review["ready"] is False
    assert "identity_review_or_evidence_incomplete" in blocked_review["blockers"]


def test_build_mode_returns_blocked_without_invoking_cli_if_prerequisites_are_missing(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    notebook = _load_notebook()
    root = tmp_path / "histology"
    root.mkdir()
    namespace: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    config_cell = _cell_with(notebook, "RUN_MODE = 'draft'")
    _exec(config_cell, namespace, "configuration")
    namespace.update({
        "RUN_MODE": "build",
        "EXPECTED_CASES": 18, "EXPECTED_PATIENTS": 18,
        "EXPECTED_LABEL_COUNTS": {"0": 10, "1": 8}, "EXPECTED_VECTORS": 148991,
    })
    helper = _cell_with(notebook, "def validate_and_extract_runtime")
    _exec(helper, namespace, "helpers")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("CLI must not run while blocked"))
    execute_cell = _cell_with(notebook, "def execute_readiness_mode")
    _exec(execute_cell, namespace, "execute")
    result = namespace["RUN_STATUS"]
    assert result["status"] == "blocked"
    assert result["training_ready"] is False
    assert "case_review_missing" in result["blockers"]
    assert "feature_status_missing" in result["blockers"]
    assert "PATIENT_" not in capsys.readouterr().out


def _synthetic_cli_source() -> str:
    return """\
import json
import os
import sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['PRETRAIN_TEST_ARGS_FILE']).write_text(json.dumps(args), encoding='utf-8')
if args and args[0] == 'draft':
    output_root = Path(args[args.index('--output-root') + 1])
    folder = output_root / 'governance' / 'drafts' / 'synthetic-draft'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'case_review.csv').write_text('synthetic review template\\n', encoding='utf-8')
if args and args[0] == 'build':
    payload = {
        'governance': {'case_count': 18, 'unique_patient_count': 18},
        'bundle': {
            'status': 'blocked', 'training_ready': False, 'blockers': ['PRIVATE_BLOCKER_SENTINEL'],
            'case_count': 16, 'patient_count': 16,
        },
        'candidate_case_ids': ['PRIVATE_CASE_SENTINEL'],
        'review_rows': [{'patient_id': 'PRIVATE_PATIENT_SENTINEL'}],
        'diagnosis_preview': 'PRIVATE_CONCLUSION_SENTINEL',
    }
else:
    payload = {
        'status': 'draft_created', 'training_ready': False, 'case_count': 18,
        'candidate_case_ids': ['PRIVATE_CASE_SENTINEL'],
        'review_rows': [{'patient_id': 'PRIVATE_PATIENT_SENTINEL'}],
        'diagnosis_preview': 'PRIVATE_CONCLUSION_SENTINEL',
    }
print(json.dumps(payload))
if args and args[0] == 'build':
    sys.exit(2)
"""


def _exec_notebook_mode(notebook: dict[str, Any], namespace: dict[str, Any]) -> str:
    _exec(_cell_with(notebook, "def validate_and_extract_runtime"), namespace, "helpers")
    output = StringIO()
    with redirect_stdout(output):
        _exec(_cell_with(notebook, "def execute_readiness_mode"), namespace, "execute")
    return output.getvalue()


def _configure_synthetic_draft(notebook: dict[str, Any], tmp_path: Path) -> tuple[dict[str, Any], Path]:
    root = tmp_path / "histology"
    root.mkdir()
    namespace: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    _exec(_cell_with(notebook, "RUN_MODE = 'draft'"), namespace, "configuration")
    namespace["SOURCE_ROOT"].mkdir(parents=True)
    namespace["METADATA"].parent.mkdir(parents=True, exist_ok=True)
    namespace["METADATA"].write_bytes(b"synthetic metadata placeholder")
    bundle = _make_runtime_zip(namespace["RUNTIME_ZIP"], cli_source=_synthetic_cli_source())
    namespace["EXPECTED_RUNTIME_SHA256"] = _sha256(bundle)
    return namespace, bundle


def test_synthetic_notebook_draft_runs_cli_and_suppresses_case_and_patient_previews(
    tmp_path: Path, monkeypatch,
) -> None:
    notebook = _load_notebook()
    namespace, _ = _configure_synthetic_draft(notebook, tmp_path)
    captured_args = tmp_path / "cli-args.json"
    monkeypatch.setenv("PRETRAIN_TEST_ARGS_FILE", str(captured_args))
    output = _exec_notebook_mode(notebook, namespace)
    status = namespace["RUN_STATUS"]
    args = json.loads(captured_args.read_text(encoding="utf-8"))

    assert status["mode"] == "draft"
    assert status["status"] == "draft_created"
    assert status["training_ready"] is False
    assert status["review_csv_path"].endswith("/synthetic-draft/case_review.csv")
    assert args[0] == "draft"
    assert args[args.index("--source-kind") + 1] == "zip"
    assert args[args.index("--expected-sources") + 1] == "25"
    assert args[args.index("--expected-patches") + 1] == "148991"
    assert args[args.index("--output-root") + 1] == str(namespace["ROOT"])
    assert "PRIVATE_CASE_SENTINEL" not in output
    assert "PRIVATE_PATIENT_SENTINEL" not in output
    assert "PRIVATE_CONCLUSION_SENTINEL" not in output
    assert namespace["RUNTIME_ROOT"].is_dir()
    assert not namespace["ENCODER_ROOT"].exists()


def test_build_command_uses_approved_arguments_and_verify_has_one_subcommand() -> None:
    notebook = _load_notebook()
    root = Path('/drive/histology')
    namespace: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    _exec(_cell_with(notebook, "RUN_MODE = 'draft'"), namespace, "configuration")
    _exec(_cell_with(notebook, "def validate_and_extract_runtime"), namespace, "helpers")
    build = namespace["build_pretrain_command"]
    namespace["BUNDLE_PATH"] = root / 'bundles' / 'bundle-1'
    args = build('build')
    assert args[3] == 'build'
    assert args[args.index('--case-review') + 1] == '/drive/histology/governance/reviews/case_review.csv'
    assert args[args.index('--feature-root') + 1] == '/drive/histology/features/v001'
    assert args[args.index('--output-root') + 1] == '/drive/histology'
    assert args[args.index('--cohort') + 1] == 'common'
    assert args[args.index('--expected-sources') + 1] == '25'
    assert args[args.index('--expected-vectors') + 1] == '148991'
    verify = build('verify')
    assert verify[3] == 'verify'
    assert verify.count('verify') == 1
    assert verify[verify.index('--bundle') + 1] == '/drive/histology/bundles/bundle-1'


def test_cli_blocked_status_cannot_report_training_ready() -> None:
    _, namespace = _runtime_namespace()
    namespace["COHORT"] = "common"
    summary = namespace["_safe_cli_summary"](
        ["python", "-m", "histology_data.pretrain_cli", "build"], 0,
        json.dumps({"bundle": {"status": "blocked", "training_ready": True, "blockers": ["case-x"]}}),
        "build", Path("/safe/output"),
    )
    assert summary["status"] == "blocked"
    assert summary["training_ready"] is False
    assert summary["cli_blocker_count"] == 1
    assert "case-x" not in json.dumps(summary)


def test_draft_and_unknown_cli_status_never_claim_training_ready() -> None:
    _, namespace = _runtime_namespace()
    namespace["COHORT"] = "common"
    summarize = namespace["_safe_cli_summary"]
    draft = summarize(
        ["python", "-m", "histology_data.pretrain_cli", "draft"], 0,
        json.dumps({"status": "draft_created", "training_ready": True}),
        "draft", Path("/safe/drafts"),
    )
    unknown = summarize(
        ["python", "-m", "histology_data.pretrain_cli", "build"], 0,
        json.dumps({"status": "unrecognized", "training_ready": True}),
        "build", Path("/safe/output"),
    )
    assert draft["training_ready"] is False
    assert unknown["training_ready"] is False


def test_runtime_requirements_keep_compatible_distributions_and_refuse_torch(tmp_path: Path, monkeypatch) -> None:
    _, namespace = _runtime_namespace()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    requirements = runtime_root / "requirements-pretrain.txt"
    requirements.write_text("Pillow>=10\nnumpy>=1.24\n", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: object())
    installed_versions = {"pillow": "10.4.0", "numpy": "2.1.3"}
    monkeypatch.setattr(namespace["importlib_metadata"], "version", lambda name: installed_versions[name])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("present modules must not be reinstalled"))
    assert namespace["install_missing_runtime_requirements"](runtime_root) == []

    requirements.write_text("torch>=2.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not install PyTorch or torchvision"):
        namespace["install_missing_runtime_requirements"](runtime_root)


def test_runtime_requirements_fail_closed_for_incompatible_installed_pillow(
    tmp_path: Path, monkeypatch,
) -> None:
    _, namespace = _runtime_namespace()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "requirements-pretrain.txt").write_text("Pillow>=10\n", encoding="utf-8")
    module_spec = importlib.machinery.ModuleSpec("PIL", loader=None)
    monkeypatch.setattr(namespace["importlib_util"], "find_spec", lambda name: module_spec if name == "PIL" else None)
    monkeypatch.setattr(namespace["importlib_metadata"], "version", lambda name: "9.5.0")
    installer_calls = []

    def capture_install(command, **kwargs):
        installer_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", capture_install)
    with pytest.raises(ValueError, match="Installed Pillow version 9.5.0.*Pillow>=10.*refusing to replace Pillow in-place"):
        namespace["install_missing_runtime_requirements"](runtime_root)

    assert installer_calls == []


def test_runtime_requirements_can_install_missing_pillow_without_dependencies(
    tmp_path: Path, monkeypatch,
) -> None:
    _, namespace = _runtime_namespace()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "requirements-pretrain.txt").write_text("Pillow>=10\n", encoding="utf-8")
    monkeypatch.setattr(namespace["importlib_util"], "find_spec", lambda name: None)

    def missing_distribution(name):
        raise namespace["importlib_metadata"].PackageNotFoundError(name)

    monkeypatch.setattr(namespace["importlib_metadata"], "version", missing_distribution)
    installer_calls = []

    def capture_install(command, **kwargs):
        installer_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", capture_install)
    missing = namespace["install_missing_runtime_requirements"](runtime_root)

    assert missing == ["pillow"]
    assert len(installer_calls) == 1
    assert installer_calls[0][-2:] == ["--no-deps", "Pillow>=10"]


def test_runtime_requirement_reinstalls_present_module_when_installed_version_is_incompatible(
    tmp_path: Path, monkeypatch,
) -> None:
    _, namespace = _runtime_namespace()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "requirements-pretrain.txt").write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    module_spec = importlib.machinery.ModuleSpec("sklearn", loader=None)
    monkeypatch.setattr(namespace["importlib_util"], "find_spec", lambda name: module_spec if name == "sklearn" else None)
    monkeypatch.setattr(
        namespace["importlib_metadata"], "version",
        lambda name: "1.6.1" if name == "scikit-learn" else pytest.fail(f"unexpected distribution lookup: {name}"),
    )
    installer_calls = []

    def capture_install(command, **kwargs):
        installer_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", capture_install)
    missing = namespace["install_missing_runtime_requirements"](runtime_root)

    assert missing == ["scikit-learn"]
    assert len(installer_calls) == 1
    assert installer_calls[0][-2:] == ["--no-deps", "scikit-learn==1.8.0"]


def test_notebook_does_not_train_or_control_the_colab_vm() -> None:
    code = "\n".join("".join(cell["source"]) for cell in _code_cells(_load_notebook()))
    lowered = code.lower()
    assert "import torch" not in lowered
    assert "import torchvision" not in lowered
    assert "drive.unmount" not in lowered
    assert "colab.runtime" not in lowered
    assert "fit(" not in lowered
    assert "cuda_visible_devices'] = ''" in lowered
    assert "--no-deps" in code


def test_notebook_paths_derive_from_one_root_and_match_colab_runtime_layout(tmp_path: Path) -> None:
    notebook = _load_notebook()
    configuration = _cell_with(notebook, "RUN_MODE = 'draft'")
    root = tmp_path / "drive" / "histology"
    local: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    _exec(configuration, local, "configuration-local")
    assert local["RUNTIME_ZIP"] == root / "runtime" / "histology-pretrain-runtime.zip"
    assert local["BUNDLE_OUTPUT_ROOT"] == root
    assert local["METADATA"] == root / "source" / "Metadata.xlsx"
    assert local["CASE_REVIEW"] == root / "governance" / "reviews" / "case_review.csv"
    assert local["RUNTIME_ROOT"] == root / ".runtime" / "histology-pretrain-runtime"

    colab: dict[str, Any] = {"IS_COLAB": True, "ROOT": root}
    _exec(configuration, colab, "configuration-colab")
    assert colab["RUNTIME_ROOT"] == Path("/content/histology-pretrain-runtime")


def test_synthetic_notebook_build_runs_only_after_review_and_full_audit(
    tmp_path: Path, monkeypatch,
) -> None:
    notebook = _load_notebook()
    root = tmp_path / "histology"
    root.mkdir()
    namespace: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    _exec(_cell_with(notebook, "RUN_MODE = 'draft'"), namespace, "configuration")
    namespace["RUN_MODE"] = "build"
    namespace["METADATA"].parent.mkdir(parents=True, exist_ok=True)
    namespace["METADATA"].write_bytes(b"synthetic metadata placeholder")
    _write_review(namespace["CASE_REVIEW"], _review_rows())
    _write_feature_audit(namespace["FEATURE_ROOT"])
    runtime_zip = _make_runtime_zip(namespace["RUNTIME_ZIP"], cli_source=_synthetic_cli_source())
    namespace["EXPECTED_RUNTIME_SHA256"] = _sha256(runtime_zip)

    captured_args = tmp_path / "build-cli-args.json"
    monkeypatch.setenv("PRETRAIN_TEST_ARGS_FILE", str(captured_args))
    output = _exec_notebook_mode(notebook, namespace)
    status = namespace["RUN_STATUS"]
    args = json.loads(captured_args.read_text(encoding="utf-8"))

    assert status["status"] == "blocked"
    assert status["training_ready"] is False
    assert status["reviewed_case_count"] == 18
    assert status["reviewed_patient_count"] == 18
    assert status["cohort_case_count"] == 16
    assert status["cohort_patient_count"] == 16
    assert status["verified_vectors"] == 148991
    assert status["cli_blocker_count"] == 1
    assert status["cohort"] == "common"
    assert args[0] == "build"
    assert args[args.index("--metadata") + 1] == str(namespace["METADATA"])
    assert args[args.index("--case-review") + 1] == str(namespace["CASE_REVIEW"])
    assert args[args.index("--feature-root") + 1] == str(namespace["FEATURE_ROOT"])
    assert args[args.index("--output-root") + 1] == str(root)
    assert args[args.index("--cohort") + 1] == "common"
    assert args[args.index("--expected-sources") + 1] == "25"
    assert args[args.index("--expected-vectors") + 1] == "148991"
    assert "PRIVATE_CASE_SENTINEL" not in output
    assert "PRIVATE_PATIENT_SENTINEL" not in output
    assert "PRIVATE_CONCLUSION_SENTINEL" not in output
    assert "PRIVATE_BLOCKER_SENTINEL" not in output


def test_synthetic_notebook_draft_runs_the_real_bundled_cli_without_decoding_images(
    tmp_path: Path,
) -> None:
    notebook = _load_notebook()
    root = tmp_path / "histology"
    root.mkdir()
    namespace: dict[str, Any] = {"IS_COLAB": False, "ROOT": root}
    _exec(_cell_with(notebook, "RUN_MODE = 'draft'"), namespace, "configuration")
    namespace["SOURCE_ROOT"].mkdir(parents=True)
    namespace["METADATA"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "Ten_File": "IMG_SYNTH.tif", "Ma_Nam": "YCT 26", "Ma_So": "CASE_SENTINEL",
            "Do_Phong_Dai": "4X", "Glade": "0",
            "Ket_Luan": "TĂNG SẢN SENTINEL_CONCLUSION", "Ten_Slide": "Slide-1",
        },
    ]).to_excel(namespace["METADATA"], index=False)
    with zipfile.ZipFile(namespace["SOURCE_ROOT"] / "Tiles-001.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as archive:
        # The draft builder reads ZIP metadata only; it never decodes this placeholder.
        archive.writestr("Tiles/IMG_SYNTH/0_0.png", b"not a decoded image")
    namespace["RUNTIME_ZIP"].parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(namespace["RUNTIME_ZIP"], "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for module_path in sorted((PROJECT_ROOT / "histology_data").glob("*.py")):
            bundle.write(module_path, f"histology_data/{module_path.name}")
        bundle.writestr("requirements-pretrain.txt", "")
    namespace["EXPECTED_RUNTIME_SHA256"] = _sha256(namespace["RUNTIME_ZIP"])
    namespace["EXPECTED_SOURCES"] = 1
    namespace["EXPECTED_PATCHES"] = 1

    output = _exec_notebook_mode(notebook, namespace)
    status = namespace["RUN_STATUS"]

    assert status["status"] == "draft_created"
    assert status["training_ready"] is False
    assert status["case_count"] == 1
    assert status["review_csv_path"] is not None
    assert Path(status["review_csv_path"]).is_file()
    assert "CASE_SENTINEL" not in output
    assert "SENTINEL_CONCLUSION" not in output
