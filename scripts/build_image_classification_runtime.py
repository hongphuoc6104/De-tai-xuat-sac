#!/usr/bin/env python3
"""Build a deterministic, code-only original-image classification runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path


def _requirements_files(source_root: Path) -> list[Path]:
    """Resolve the runtime's local requirements closure without outside paths."""
    pending = [source_root / "requirements-image-classification.txt"]
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe runtime requirements: {path.name}")
        resolved.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            tokens = line.strip().split()
            if not tokens or tokens[0] not in {"-r", "--requirement"}:
                continue
            if len(tokens) != 2:
                raise ValueError("Requirements include must specify one local filename.")
            name = tokens[1]
            if Path(name).name != name or not name.startswith("requirements") or not name.endswith(".txt"):
                raise ValueError(f"Unsafe requirements include: {name}")
            pending.append(source_root / name)
    return sorted(resolved)


def build_runtime(source_root: Path, output: Path) -> dict[str, object]:
    """Package source modules and the image CLI, excluding datasets and secrets."""
    source_root = source_root.resolve(strict=True)
    files = sorted((source_root / "histology_data").glob("*.py"))
    files.append(source_root / "scripts/run_image_classification.py")
    files.extend(_requirements_files(source_root))
    required = {"histology_data/image_classification.py", "histology_data/image_classification_cli.py"}
    names = {p.relative_to(source_root).as_posix() for p in files}
    if not required.issubset(names):
        raise FileNotFoundError("Original-image runtime implementation is incomplete.")
    for path in files:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source_root):
            raise ValueError(f"Unsafe runtime source: {path.name}")
    if output.is_symlink() or output.suffix != ".zip":
        raise ValueError("Runtime output must be a regular .zip path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(files, key=lambda p: p.relative_to(source_root).as_posix()):
                name = path.relative_to(source_root).as_posix()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, path.read_bytes())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, output)
        output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "sha256": digest, "size_bytes": output.stat().st_size,
            "file_count": len(files), "source_files": sorted(names)}


def main() -> None:
    """Build the code archive and print its integrity receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_runtime(args.source_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
