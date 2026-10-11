"""Traceable relative-attention outputs; never claim patch or pixel cancer truth."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, file_hash


def publish_case_attention(bundle: Path, dataset: Any, case: Any, output: dict[str, Any], target: Path) -> None:
    """Store weights and verified references; plot one source-image coordinate map per lens.

    Maps are standalone SVGs of patch origins on their original coordinate system,
    not segmentations or overlays on unavailable pixels. Top source-image selection
    is only for display, after model selection/refit is locked.
    """
    key = hashlib.sha256(case.case_id.encode()).hexdigest()
    directory = Path(target) / key
    directory.mkdir(parents=True, exist_ok=True)
    tile_order = dataset.loaded_tile_order(case.case_id)
    with (Path(bundle) / "bags.jsonl").open() as stream:
        bags = [json.loads(line) for line in stream if json.loads(line)["case_id"] == case.case_id]
    bag_by_lens = {str(bag["lens"]): bag for bag in bags}
    files: dict[str, str] = {}
    for lens, weights in output["attention"].items():
        array = weights.detach().cpu().numpy().astype(np.float32)
        if len(array) != len(tile_order[lens]) or not np.isfinite(array).all():
            raise ValueError("Attention does not match verified tile order.")
        if (array < 0).any():
            raise ValueError("Attention weights must be nonnegative.")
        if len(array) and not np.isclose(array.sum(dtype=np.float64), 1.0, atol=1e-5):
            raise ValueError("Attention is not normalized over the full lens bag.")
        np.save(directory / f"attention_{lens}.npy", array, allow_pickle=False)
        bag = bag_by_lens[lens]
        with (Path(bundle) / "instance_refs.jsonl").open("rb") as stream:
            stream.seek(int(bag["ref_start_byte"]))
            payload = stream.read(int(bag["ref_end_byte"]) - int(bag["ref_start_byte"]))
        refs = [json.loads(line) for line in payload.splitlines()]
        if [r["tile_id"] for r in refs] != tile_order[lens]:
            raise ValueError("Attention references differ from loaded source tile order.")
        atomic_json(directory / f"references_{lens}.json", refs)
        if len(refs):
            top_ref = refs[int(array.argmax())]
            selected = [(r, float(a)) for r, a in zip(refs, array, strict=True) if r["image_id"] == top_ref["image_id"]]
            xmax = max(float(r["x"]) for r, _ in selected) + 1
            ymax = max(float(r["y"]) for r, _ in selected) + 1
            scale = max(float(array.max()), 1e-12)
            circles = []
            for ref, score in selected:
                strength = score / scale
                colour = f"rgb({int(255 * strength)},70,{int(230 * (1-strength))})"
                x = 30 + float(ref["x"]) / xmax * 540
                y = 75 + float(ref["y"]) / ymax * 440
                circles.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="8" fill="{colour}"><title>'
                               f'x={ref["x"]}, y={ref["y"]}, relative attention={score:.8g}</title></circle>')
            title = html.escape(f"{lens}x · {top_ref['image_id']}")
            svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 570">'
                   '<rect width="600" height="570" fill="white"/>'
                   f'<text x="20" y="28" font-family="sans-serif" font-size="15">{title}</text>'
                   '<text x="20" y="53" font-family="sans-serif" font-size="12">'
                   'Patch origins in source coordinates; relative attention, not cancer probability.</text>'
                   + ''.join(circles) + '</svg>')
            (directory / f"coordinates_{lens}.svg").write_text(svg)
        for name in (f"attention_{lens}.npy", f"references_{lens}.json", f"coordinates_{lens}.svg"):
            path = directory / name
            if path.exists():
                files[name] = file_hash(path)
    atomic_json(directory / "commit.json", {
        "case_id": case.case_id, "patient_id": case.patient_id,
        "kind": "relative_attention_with_source_coordinate_maps", "files": files,
        "pixel_ground_truth": False, "registration_between_lenses": False,
    })


def verify_attention_outputs(root: Path, expected_case_ids: list[str]) -> dict[str, str]:
    """Validate exact case coverage and artifact hashes before trusting prediction receipts."""
    commits: dict[str, str] = {}
    for case_id in expected_case_ids:
        directory = Path(root) / hashlib.sha256(case_id.encode()).hexdigest()
        receipt = directory / "commit.json"
        record = json.loads(receipt.read_text())
        if record["case_id"] != case_id:
            raise ValueError("Attention receipt case mismatch.")
        for name, digest in record["files"].items():
            if Path(name).name != name or file_hash(directory / name) != digest:
                raise ValueError("Attention output path/checksum mismatch.")
        commits[case_id] = file_hash(receipt)
    return commits
