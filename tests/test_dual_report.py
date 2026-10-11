"""Synthetic receipt-to-report tests; no real cohort identifiers or images."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from histology_data.dual_atlas import cluster_case_label_crosstab, summarize_partition
from histology_data.dual_common import ArtifactStore, gzip_csv_bytes
from scripts.render_dual_report import render_dual_report

_SAMPLE_FIELDS = [
    "scope", "case_id", "patient_id", "target", "lens", "tile_id", "photo_id", "cluster_id",
]


def _rows_for_partition(labels: np.ndarray, refs: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    return [
        {
            "scope": scope,
            "case_id": ref["case_id"],
            "patient_id": ref["patient_id"],
            "target": ref["source_case_label"],
            "lens": ref["lens"],
            "tile_id": ref["tile_id"],
            "photo_id": ref["photo_id"],
            "cluster_id": int(label),
        }
        for label, ref in zip(labels, refs, strict=True)
    ]


def _publish_partition(
    store: ArtifactStore,
    base: str,
    cluster_label: int | None,
    labels: np.ndarray,
    features: np.ndarray,
    refs: list[dict[str, Any]],
    scope: str,
) -> None:
    rows = _rows_for_partition(labels, refs, scope)
    summary = summarize_partition(
        features,
        labels,
        [ref["photo_id"] for ref in refs],
        [ref["patient_id"] for ref in refs],
    )
    crosstab = cluster_case_label_crosstab(
        labels,
        [ref["patient_id"] for ref in refs],
        [ref["source_case_label"] for ref in refs],
    )
    store.publish(f"{base}/assignments.csv.gz", gzip_csv_bytes(rows, _SAMPLE_FIELDS))
    metrics: dict[str, Any] = {
        "partition": asdict(summary),
        "source_case_label_crosstab": [asdict(row) for row in crosstab],
        "assignment_rows": len(rows),
    }
    if cluster_label is None:
        size, samples, method = base.rsplit("/", 1)[1].split("_")
        metrics["config"] = {
            "min_cluster_size": int(size.replace("mcs", "")),
            "min_samples": int(samples.replace("ms", "")),
            "selection_method": method,
        }
    else:
        metrics["cluster_count"] = cluster_label
    store.json(f"{base}/metrics.json", metrics)


def _fixture_stores(root: Path) -> tuple[ArtifactStore, ArtifactStore, ArtifactStore, dict[str, Any]]:
    scope = "outer_0/inner_0"
    case_ids = [f"synthetic-case-{index}" for index in range(5)]
    patient_ids = [f"synthetic-patient-{index}" for index in range(5)]
    targets = [0, 1, 0, 1, 1]
    source_rows: dict[str, dict[str, str]] = {}
    case_tile_order: dict[str, dict[str, list[str]]] = {}
    members = {case: {"patient_id": patient_ids[index], "target": targets[index]}
               for index, case in enumerate(case_ids)}
    rng = np.random.default_rng(908)
    transforms: dict[str, Any] = {"members": members, "dimensions": 32, "lenses": {}}
    for lens in ("4", "10", "40"):
        features = rng.normal(size=(20, 32)).astype(np.float32)
        features[:, :2] += np.asarray([[targets[i // 4], i // 4] for i in range(20)], dtype=np.float32)
        refs: list[dict[str, Any]] = []
        for case_index, case_id in enumerate(case_ids):
            case_tile_order.setdefault(case_id, {})[lens] = []
            for offset in range(4):
                tile_id = f"{case_id}-{lens}-tile-{offset}"
                photo_id = f"{case_id}-{lens}-photo-{offset // 2}"
                case_tile_order[case_id][lens].append(tile_id)
                part_id = f"part-{lens}"
                row_index = case_index * 4 + offset
                source_rows[tile_id] = {"case_id": case_id, "lens": lens, "photo_id": photo_id,
                                        "part_id": part_id, "row_index": row_index}
                refs.append({
                    "case_id": case_id,
                    "patient_id": patient_ids[case_index],
                    "source_case_label": targets[case_index],
                    "lens": lens,
                    "tile_id": tile_id,
                    "photo_id": photo_id,
                    "part_id": part_id,
                    "row_index": row_index,
                })
        transforms["lenses"][lens] = {"sample_projected": features, "sample_refs": refs}
    source_identity = {"kind": "synthetic", "fixture_id": "fixture-identity", "case_ids": case_ids}
    source_map = {"source_identity": source_identity, "rows": source_rows, "case_tile_order": case_tile_order}
    shared = ArtifactStore(root / "shared", {"kind": "synthetic-report", "source": "fixture"})
    shared.publish("source_photo_map.pkl", pickle.dumps(source_map, protocol=5))
    shared.publish(f"scopes/{scope}/transform.pkl", pickle.dumps(transforms, protocol=5))
    a_store = ArtifactStore(root / "A", {"branch": "A", "source": "fixture"})
    b_store = ArtifactStore(root / "B", {"branch": "B", "source": "fixture"})
    refs_by_lens = transforms["lenses"]
    for lens in ("4", "10", "40"):
        features = refs_by_lens[lens]["sample_projected"]
        refs = refs_by_lens[lens]["sample_refs"]
        for size in (8, 16, 32, 64):
            for samples in (4, 8):
                for method in ("eom", "leaf"):
                    labels = np.asarray([-1 if index % 9 == 0 else index % 3 for index in range(20)], dtype=np.int32)
                    base = (f"atlas/outer_0_inner_0/hdbscan/lens_{lens}/"
                            f"mcs{size}_ms{samples}_{method}")
                    _publish_partition(a_store, base, None, labels, features, refs, scope)
        for count in (2, 4, 8, 16):
            labels = np.asarray([index % count for index in range(20)], dtype=np.int32)
            _publish_partition(a_store, f"atlas/outer_0_inner_0/ward/lens_{lens}/k{count}",
                               count, labels, features, refs, scope)
        for count in (4, 8, 16):
            labels = np.asarray([index % count for index in range(20)], dtype=np.int32)
            _publish_partition(a_store, f"atlas/outer_0_inner_0/kmeans/lens_{lens}/k{count}",
                               count, labels, features, refs, scope)
    main_rows: list[dict[str, Any]] = []
    counts_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    b_count_rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        base = {"case_id": case_id, "patient_id": patient_ids[index], "target": targets[index],
                "scope": f"outer_{index % 3}/refit", "outer_fold": index % 3}
        main_rows.append({**base, "probability": 0.15 + 0.7 * targets[index], "selected_k": 4, "selected_c": 0.1})
        counts_rows.append({**base, "probability": 0.25 + 0.5 * targets[index], "selected_k": None, "selected_c": 0.1})
        b_rows.append({**base, "probability": 0.2 + 0.6 * targets[index], "selected_c": 0.1, "member_count": 2})
        b_count_rows.append({**base, "probability": 0.3 + 0.4 * targets[index], "selected_k": None, "selected_c": 0.1})
    a_store.json("results/a1_oof.json", main_rows)
    a_store.json("results/counts_only_oof.json", counts_rows)
    b_store.json("results/b0_oof.json", b_rows)
    b_store.json("results/counts_only_oof.json", b_count_rows)
    return shared, a_store, b_store, source_map


def test_renderer_verifies_lineage_anonymizes_ids_and_builds_self_contained_report(tmp_path: Path) -> None:
    shared, a_store, b_store, source_map = _fixture_stores(tmp_path)
    output = tmp_path / "report"
    private_map = tmp_path / "private" / "id-map.json"

    manifest = render_dual_report(a_store, b_store, shared, source_map, output, private_map)

    assert manifest["status"] == "partial"  # B1 is intentionally pending.
    assert manifest["a0"]["status"] == "complete"
    assert manifest["a0"]["development_case_count"] == 5
    assert manifest["matched_oof"]["status"] == "matched_full_cohort"
    assert manifest["b1"]["status"] == "pending"
    assert (output / "report.md").is_file()
    html_text = (output / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "source case label" in html_text.lower()
    assert "http://" not in html_text and "https://" not in html_text
    assert (output / "a0_scatter_lens_4.svg").is_file()
    assert (output / "oof_case_scatter.png").is_file()
    lineage = list(csv.DictReader((output / "point_lineage.csv").open(encoding="utf-8")))
    assert len(lineage) == 20 * 3
    assert all(row["point_token"].startswith("POINT-") for row in lineage)
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output / "report.md", output / "report.html", output / "report_manifest.json",
                     output / "point_lineage.csv", output / "oof_lineage.csv",
                     output / "a0_scatter_lens_4.svg")
    )
    assert "synthetic-case-" not in public_text
    assert "synthetic-patient-" not in public_text
    private = json.loads(private_map.read_text(encoding="utf-8"))
    assert len(private["cases"]) == 5
    assert len(private["points"]) == 60
    assert private_map.stat().st_mode & 0o777 == 0o600
    representatives = private_map.with_name("id-map.representatives.csv")
    selected = list(csv.DictReader(representatives.open(encoding="utf-8")))
    assert 0 < len(selected) <= 96
    assert len({row["algorithm"] for row in selected}) == 6
    assert len({(row["lens"], row["algorithm"]) for row in selected}) == 18
    assert set(selected[0]) == {
        "tile_id", "lens", "source_case_label", "algorithm", "cluster_id",
        "part_id", "row_index", "prototype_role",
    }
    assert representatives.stat().st_mode & 0o777 == 0o600


def test_b1_is_partial_until_both_candidates_match_a_and_b(tmp_path: Path) -> None:
    shared, a_store, b_store, source_map = _fixture_stores(tmp_path)
    cases = list(source_map["case_tile_order"])
    targets = [0, 1, 0, 1, 1]
    records = [{"case_id": case, "patient_id": f"synthetic-patient-{index}", "target": targets[index],
                "probability": 0.2 + 0.6 * targets[index]} for index, case in enumerate(cases)]
    b1_store = ArtifactStore(tmp_path / "B1", {"branch": "B1", "source": "fixture"})
    b1_store.json("oof/beta0.json", records)
    b1_store.json("oof/beta05.json", records[:-1])
    args = {"beta=0.0": "oof/beta0.json", "beta=0.5": "oof/beta05.json"}
    partial = render_dual_report(
        a_store, b_store, shared, source_map, tmp_path / "partial", tmp_path / "private" / "map.json",
        b1=b1_store, b1_artifacts=args,
    )
    assert partial["b1"]["status"] == "partial"
    b1_store.json("oof/beta05-complete.json", records)
    complete = render_dual_report(
        a_store, b_store, shared, source_map, tmp_path / "complete", tmp_path / "private" / "map.json",
        b1=b1_store, b1_artifacts={"beta=0.0": "oof/beta0.json", "beta=0.5": "oof/beta05-complete.json"},
    )
    assert complete["b1"]["status"] == "complete"
    assert complete["status"] == "complete"
