"""Content fingerprints and verified publication shared by preparation modules."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    """Hash a deterministic JSON representation, independent of output paths."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Compute SHA-256 using a bounded streaming buffer."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish one local JSON file via a unique temporary file and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verified_copy(source: Path, target: Path, expected_sha256: str) -> None:
    """Copy to a new local/Drive file and verify its bytes before commit markers."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        shutil.copyfile(source, temp)
        if file_hash(temp) != expected_sha256:
            raise ValueError(f"Copy checksum mismatch: {target.name}")
        os.replace(temp, target)
        if file_hash(target) != expected_sha256:
            raise ValueError(f"Published checksum mismatch: {target.name}")
    finally:
        temp.unlink(missing_ok=True)
