#!/usr/bin/env python3
"""Build a deterministic code-only dual-track runtime from verified source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path

from scripts.build_mil_runtime import _runtime_files


def build_dual_runtime(root: Path, output: Path) -> dict[str, object]:
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output.suffix != ".zip" or output.is_symlink():
        raise ValueError("Runtime output must be a regular ZIP path.")
    files = _runtime_files(root)
    entry = root / "scripts/run_dual.py"
    if entry.is_symlink() or not entry.is_file():
        raise ValueError("Dual-track entrypoint is missing or symlinked.")
    files.append((entry, "scripts/run_dual.py"))
    if not (root / "histology_data/dual_cli.py").is_file():
        raise ValueError("Dual-track controller is not present.")
    manifest = {
        "schema": 1,
        "purpose": "code_only_dual_track_v1",
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in sorted(files)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".zip", dir=output.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with zipfile.ZipFile(temp, "w") as archive:
            payloads = [(name, path.read_bytes()) for path, name in files]
            payloads.append(("runtime_manifest.json", json.dumps(manifest, sort_keys=True).encode()))
            for name, data in sorted(payloads):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        os.replace(temp, output)
        output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n")
    finally:
        temp.unlink(missing_ok=True)
    return {"sha256": digest, "bytes": output.stat().st_size, "path": str(output), "code_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_dual_runtime(args.source_root, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
