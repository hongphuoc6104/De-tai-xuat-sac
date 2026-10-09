"""Create a deterministic, data-free ZIP runtime for Google Drive/Colab."""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "requirements-data.txt",
    "docs/DATA_SHARDS.md",
    "docs/DATA_SHARDS_CONTRACT.md",
    "configs/data_shards_smoke.json",
    "configs/data_shards_colab.json",
)


def _runtime_files() -> list[Path]:
    files = [path for path in (ROOT / "histology_data").rglob("*.py") if path.is_file()]
    files.extend(ROOT / name for name in RUNTIME_FILES)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime source file is missing: {missing[0].relative_to(ROOT)}")
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def build_runtime(output: Path) -> Path:
    """Write a stable ZIP containing code, dependency declarations, and docs only."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _runtime_files()
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                name = path.relative_to(ROOT).as_posix()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "histology-data-runtime.zip")
    args = parser.parse_args()
    try:
        result = build_runtime(args.output)
    except (OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
