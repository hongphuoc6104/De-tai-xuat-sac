"""Explicit source-verified comparisons across frozen whole-image releases."""
from __future__ import annotations

import json
import time
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import image_classification as core, image_classification_studies as studies
from .io import atomic_json, file_hash, fingerprint

SOURCE_FIELDS = (
    "image_id", "patient_id", "objective_lens", "image_target", "raw_glade", "file_name",
    "source_archive_sha256", "source_member", "member_crc32", "member_byte_size",
    "original_image_sha256", "original_width", "original_height", "source_format",
)


def validate_source_lineage(original: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> str:
    """Only geometry/encoder-derived fields may differ between compared releases."""
    old = {row["image_id"]: row for row in original}
    new = {row["image_id"]: row for row in incoming}
    if (len(old) != len(original) or len(new) != len(incoming) or not old or set(old) != set(new)):
        raise ValueError("Comparison source image coverage is not identical and unique.")
    for image_id, row in old.items():
        if any(key not in row or key not in new[image_id] or row[key] != new[image_id][key]
               for key in SOURCE_FIELDS):
            raise ValueError("Comparison source lineage differs from the original baseline.")
    return fingerprint([{key: old[image_id][key] for key in SOURCE_FIELDS} for image_id in sorted(old)])


def _load(new_run: Path, original_run: Path, baseline_root: Path) -> tuple[studies.StudyData, dict[str, Any]]:
    original = studies.load_study_data(original_run, baseline_root)
    features, rows, release = core._load_feature_matrix(new_run)
    core._audit_original_image_content(new_run, rows)
    lineage = validate_source_lineage(original.rows, rows)
    if (release["inventory_id"] != original.release["inventory_id"]
            or release["descriptor"]["source_archive_sha256"] != original.release["descriptor"]["source_archive_sha256"]):
        raise ValueError("Comparison inventory/source lineage identity differs.")
    positions = {row["image_id"]: i for i, row in enumerate(rows)}
    outer = [(np.array([positions[original.rows[i]["image_id"]] for i in train], dtype=np.int64),
              np.array([positions[original.rows[i]["image_id"]] for i in test], dtype=np.int64))
             for train, test in original.outer]
    binding = {
        "protocol": "same-original-source-cross-frozen-release-v1",
        "source_feature_release_id": original.release["release_id"],
        "comparison_feature_release_id": release["release_id"],
        "baseline_study_id": original.baseline["study_id"],
        "baseline_commit_sha256": file_hash(Path(baseline_root) / "study.commit.json"),
        "source_lineage_sha256": lineage, "assignment_sha256": original.assignment_sha256,
        "comparison_loader_sha256": file_hash(Path(__file__)),
        "image_count": len(rows), "patient_count": len({r["patient_id"] for r in rows}),
        "same_source_lineage": True,
        "images_by_lens": dict(sorted(Counter(str(r["objective_lens"]) for r in rows).items())),
    }
    data = studies.StudyData(Path(new_run).resolve(), features, rows, release, original.baseline,
                            outer, original.assignment_sha256, source_comparison=binding)
    return data, binding


def load_comparison_data(new_run: Path, original_run: Path, baseline_root: Path) -> studies.StudyData:
    """Replay old patient folds only after verifying every original source field."""
    return _load(new_run, original_run, baseline_root)[0]


def comparison_receipt(new_run: Path, original_run: Path, baseline_root: Path) -> dict[str, Any]:
    """Return an immutable provenance binding, without source/patient identifiers."""
    return _load(new_run, original_run, baseline_root)[1]


def benchmark_whole_frames(
    inventory_path: Path, verified_rows: list[dict[str, Any]], archives: Mapping[int, Path],
    selected_ids: list[str], variants: Mapping[str, tuple[core.ImageFeatureEncoder, int]],
    output_root: Path, *, deadline: float | None = None,
) -> dict[str, Any]:
    """Measure bounded whole-frame inference and publish source-bound smoke vectors.

    No fitting or metric-based model selection occurs. At most four decoded
    originals are live;1024 is measured at batch1,512 at batch4 when possible.
    """
    inventory = core._load_inventory(inventory_path)
    expected = {row["image_id"]: row for row in verified_rows}
    if (not selected_ids or len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids) <= set(expected) or set(archives) != {4, 10, 40}
            or not variants or any(not name.isidentifier() or isinstance(size, bool)
                                  or not isinstance(size, int) or size < 1
                                  for name, (_, size) in variants.items())):
        raise ValueError("Invalid benchmark selection, archives or variant names/geometry.")
    for lens, archive in archives.items():
        if deadline is not None:
            core._check_deadline(deadline)
        if file_hash(archive) != inventory["sources"][str(lens)]["archive_sha256"]:
            raise ValueError("Benchmark source archive differs from inventoried bytes.")
    descriptor = {"protocol": "whole-original-image-resource-benchmark-v1",
                  "inventory_id": inventory["inventory_id"], "selection": selected_ids,
                  "source_sha256": {key: expected[key]["original_image_sha256"] for key in selected_ids},
                  "variants": {name: {"encoder": encoder.descriptor, "max_long_edge": size}
                               for name, (encoder, size) in variants.items()},
                  "implementation_sha256": file_hash(Path(__file__))}
    identity = fingerprint(descriptor)
    root = Path(output_root) / identity
    root.mkdir(parents=True, exist_ok=True)
    with core._RunLock(root, recover=False):
        if (root / "benchmark.commit.json").is_file():
            from .image_classification_lab import _verify_file_manifest
            _verify_file_manifest(root, "benchmark.commit.json")
            return json.loads((root / "resource_benchmark.json").read_text())
        vectors = {name: [] for name in variants}
        times = {name: [] for name in variants}
        preprocess = {name: [] for name in variants}
        peaks = {name: 0 for name in variants}
        decode_seconds = 0.0
        with ExitStack() as stack:
            zipped = {lens: stack.enter_context(zipfile.ZipFile(path)) for lens, path in archives.items()}
            for offset in range(0, len(selected_ids), 4):
                if deadline is not None:
                    core._check_deadline(deadline)
                with ExitStack() as images_stack:
                    originals = []
                    for image_id in selected_ids[offset:offset + 4]:
                        row = expected[image_id]
                        start = time.perf_counter()
                        photo, sha, _, _ = core._load_member_image(
                            zipped[row["objective_lens"]], row, max_image_bytes=512 * 1024**2,
                            max_image_pixels=178_000_000)
                        images_stack.callback(photo.close)
                        if sha != row["original_image_sha256"]:
                            raise ValueError("Benchmark image source lineage changed.")
                        decode_seconds += time.perf_counter() - start
                        originals.append(photo)
                    for name, (encoder, size) in variants.items():
                        torch = getattr(encoder, "_torch", None)
                        cuda = torch is not None and getattr(getattr(encoder, "_device", None), "type", None) == "cuda"
                        start = time.perf_counter()
                        processed = [core.resize_pad_rgb(photo, size) for photo in originals]
                        preprocess[name].append(time.perf_counter() - start)
                        try:
                            if cuda:
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            batch = 1 if size >= 1024 else 4
                            matrix, _ = core._encode_adaptive(processed, encoder, batch)
                            if cuda:
                                torch.cuda.synchronize()
                                peaks[name] = max(peaks[name], torch.cuda.max_memory_allocated())
                            times[name].append(time.perf_counter() - start)
                            vectors[name].extend(matrix)
                        finally:
                            for photo in processed:
                                photo.close()
        result: dict[str, Any] = {"benchmark_root": str(root), "selected_images": len(selected_ids),
                                 "training_ready": False, "descriptor": descriptor,
                                 "decode_seconds": decode_seconds, "variants": {}}
        files = {}
        for name, (_, size) in variants.items():
            matrix = np.asarray(vectors[name], dtype=np.float32)
            seconds = sum(times[name])
            result["variants"][name] = {"max_long_edge": size, "feature_shape": list(matrix.shape),
                "inference_seconds": seconds, "seconds_per_image": seconds / len(selected_ids),
                "cold_chunk_seconds": times[name][0], "steady_chunk_seconds": times[name][1:],
                "preprocessing_seconds": sum(preprocess[name]), "peak_vram_allocated_bytes": peaks[name]}
            path = root / f"{name}.smoke_features.npy"
            np.save(path, matrix, allow_pickle=False)
            files[path.name] = file_hash(path)
        atomic_json(root / "resource_benchmark.json", result)
        files["resource_benchmark.json"] = file_hash(root / "resource_benchmark.json")
        commit = {"complete": True, "benchmark_id": identity, "files": files}
        commit["commit_id"] = fingerprint(commit)
        atomic_json(root / "benchmark.commit.json", commit)
        return result
