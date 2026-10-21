"""Explicit future-runtime resource smoke; never a quality experiment."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import image_classification as core, image_classification_r0_r3 as units
from .io import atomic_json, file_hash, fingerprint
from .uni_cache import _write_arrays, validate_rows
from .uni_views import ViewEncoder, embed_image


def run(rows: list[dict[str, Any]], archives: dict[int, Path], output: Path, encoder: ViewEncoder, *,
        batch_size: int = 4) -> dict[str, Any]:
    """Select largest original per lens by geometry only, then check batch parity."""
    validate_rows(rows)
    selected = [max((r for r in rows if r["objective_lens"] == lens),
                    key=lambda r: (r["original_width"] * r["original_height"], r["image_id"])) for lens in (4, 10, 40)]
    for row in selected:
        if file_hash(archives[row["objective_lens"]]) != row["source_archive_sha256"]:
            raise ValueError("Smoke ZIP differs from baseline.")
    descriptor = {"protocol": "uni-resource-smoke-v1", "encoder": encoder.descriptor(),
                  "image_sha256": [r["original_image_sha256"] for r in selected], "batch_size": batch_size,
                  "implementation_sha256": file_hash(Path(__file__))}
    root = output / fingerprint(descriptor)
    root.mkdir(parents=True, exist_ok=True)
    # This small independent check may run again explicitly; it never touches feature caches.
    with core._RunLock(root, recover=False):
        binding = {"descriptor": descriptor}
        if units._verify(root, binding):
            return json_load(root / "receipt.json")
        checks, vectors = [], []
        torch = getattr(encoder, "torch", None)
        cuda = torch is not None and getattr(encoder, "device", None) == "cuda"
        for row in selected:
            with zipfile.ZipFile(archives[row["objective_lens"]]) as archive:
                photo, sha, _, _ = core._load_member_image(archive, row,
                    max_image_bytes=512 * 1024**2, max_image_pixels=178_000_000)
            try:
                if sha != row["original_image_sha256"]:
                    raise ValueError("Smoke image checksum mismatch.")
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                value, _, coords = embed_image(photo, encoder, batch_size=batch_size)
                if cuda:
                    torch.cuda.synchronize()
                seconds = time.monotonic() - start
                peak = torch.cuda.max_memory_allocated() if cuda else None
                independent, _, _ = embed_image(photo, encoder, batch_size=1)
            finally:
                photo.close()
            delta = float(np.max(np.abs(value - independent)))
            checks.append({"objective_lens": row["objective_lens"], "image_sha256": sha,
                           "views": len(coords), "seconds": seconds, "peak_vram_bytes": peak,
                           "batch_parity_max_absolute_delta": delta, "passed": delta <= .005})
            vectors.append(value)
        receipt = {"technical_smoke_passed": all(c["passed"] for c in checks), "quality_evaluated": False,
                   "GPU_checked": cuda, "encoder_kind": encoder.descriptor()["kind"],
                   "checks": checks, "smoke_root": str(root), "descriptor": descriptor}
        atomic_json(root / "receipt.json", receipt)
        if receipt["technical_smoke_passed"]:
            _write_arrays(root / "features.npz", features=np.asarray(vectors, dtype=np.float32))
            units._publish(root, binding, ["receipt.json", "features.npz"])
        return receipt


def json_load(path: Path) -> dict[str, Any]:
    """Read a verified smoke receipt without importing encoder dependencies."""
    import json
    return json.loads(path.read_text())
