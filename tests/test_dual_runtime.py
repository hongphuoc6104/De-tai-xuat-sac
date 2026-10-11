"""Deterministic runtime excludes patient data and binds all code files."""

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.build_dual_runtime import build_dual_runtime


def test_dual_code_only_runtime_is_deterministic(tmp_path):
    source = Path(__file__).resolve().parent.parent
    first = build_dual_runtime(source, tmp_path / "a.zip")
    second = build_dual_runtime(source, tmp_path / "b.zip")
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(first["path"]) as archive:
        names = archive.namelist()
        assert "scripts/run_dual.py" in names and "histology_data/dual_cli.py" in names
        assert not any(n.startswith(("data/", "Results/", "tests/", "weights/")) for n in names)
        assert not any(n.endswith((".pkl", ".npy", ".pt", ".xlsx")) for n in names)
        manifest = json.loads(archive.read("runtime_manifest.json"))
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in manifest["files"].items()
        )
