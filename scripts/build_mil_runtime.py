#!/usr/bin/env python3
"""Build a deterministic code-only MIL runtime ZIP and SHA-256 sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_REQUIREMENTS = {"requirements-mil.txt", "requirements-pretrain.txt", "requirements-data.txt"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_files(source_root: Path) -> list[tuple[Path, str]]:
    source_root = Path(source_root).resolve(strict=True)
    # Original-image classification has its own runtime and dependency graph.
    # Its implementation must not invalidate existing MIL notebook pins.
    candidates = [path for path in (source_root / "histology_data").glob("*.py")
                  if not path.name.startswith("image_classification")]
    candidates.append(source_root / "scripts" / "run_mil.py")
    candidates.extend(path for path in sorted(source_root.glob("requirements*.txt"))
                      if path.name != "requirements-image-classification.txt")
    required = {
        source_root / "histology_data" / "__init__.py",
        source_root / "scripts" / "run_mil.py",
        *(source_root / name for name in REQUIRED_REQUIREMENTS),
    }
    if not required.issubset(set(candidates)):
        missing = sorted(path.relative_to(source_root).as_posix() for path in required - set(candidates))
        raise FileNotFoundError(f"MIL runtime is missing required source files: {missing}")
    files: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        if (path.is_symlink() or not path.is_file() or not (relative.startswith("histology_data/")
                or relative == "scripts/run_mil.py" or relative.startswith("requirements"))):
            raise ValueError(f"Unsafe or out-of-scope runtime source: {relative}")
        if relative.startswith("histology_data/") and not relative.endswith(".py"):
            raise ValueError(f"Runtime module is not a Python source file: {relative}")
        if relative.startswith("requirements") and not relative.endswith(".txt"):
            raise ValueError(f"Unexpected requirements artifact: {relative}")
        if relative in seen:
            raise ValueError(f"Duplicate runtime archive path: {relative}")
        seen.add(relative)
        files.append((path, relative))
    if any("tests/" in name or name.startswith(("data/", "weights/", ".git/")) for name in seen):
        raise ValueError("Runtime archive contains a forbidden data/test/weights path.")
    return files


def build_runtime_archive(source_root: Path, output_path: Path) -> dict[str, object]:
    """Write sorted, timestamp-free source files; omit data, weights and tests."""
    source_root = Path(source_root).resolve(strict=True)
    output_path = Path(output_path).expanduser().absolute()
    if output_path.suffix.lower() != ".zip":
        raise ValueError("Runtime output path must end in .zip.")
    if output_path.is_symlink():
        raise ValueError("Refusing to replace a symlink runtime output.")
    sidecar = output_path.with_name(output_path.name + ".sha256")
    if sidecar.is_symlink():
        raise ValueError("Refusing to replace a symlink runtime checksum sidecar.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _runtime_files(source_root)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".mil-runtime-build-", dir=output_path.parent))
    temporary_zip = temporary_dir / output_path.name
    temporary_sidecar = temporary_dir / sidecar.name
    try:
        with zipfile.ZipFile(temporary_zip, "w", allowZip64=True) as archive:
            for source, relative in files:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.file_size = source.stat().st_size
                with source.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        digest = _sha256_file(temporary_zip)
        temporary_sidecar.write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
        os.replace(temporary_zip, output_path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return {
        "path": str(output_path),
        "sha256": digest,
        "sidecar": str(sidecar),
        "file_count": len(files),
        "size_bytes": output_path.stat().st_size,
        "files": [relative for _, relative in files],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "histology-mil-runtime.zip")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = build_runtime_archive(args.source_root, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
