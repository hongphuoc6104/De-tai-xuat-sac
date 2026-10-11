"""Receipt validation, tokenized report data and private image-review refs."""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import hmac
import io
import json
import math
import os
import pickle
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from histology_data.dual_common import ArtifactStore

LENSES = ("4", "10", "40")
HDB = [(size, samples, method) for size in (8, 16, 32, 64)
       for samples in (4, 8) for method in ("eom", "leaf")]
REPS = [("hdb_eom_16_4", "HDB EOM 16/4"),
        ("hdb_leaf_16_4", "HDB leaf 16/4"), ("ward_k4", "Ward k4"),
        ("kmeans_k4", "KMeans k4"), ("kmeans_k8", "KMeans k8"),
        ("kmeans_k16", "KMeans k16")]
REPRESENTATIVE_FIELDS = ["tile_id", "lens", "source_case_label", "algorithm", "cluster_id",
                         "part_id", "row_index", "prototype_role"]


def open_store(root: Path) -> ArtifactStore:
    """Open a committed store using its recorded identity."""
    root = Path(root).expanduser().absolute()
    identity = root / "identity.json"
    if root.is_symlink() or not root.is_dir() or identity.is_symlink() or not identity.is_file():
        raise ValueError(f"Artifact store identity is missing: {root}")
    return ArtifactStore(root, json.loads(identity.read_text(encoding="utf-8")))


def read(store: ArtifactStore, name: str, parser: Any = json.loads, *, required: bool = False) -> Any:
    payload = store.read(name)
    if payload is None:
        if required:
            raise FileNotFoundError(f"Missing committed report input: {name}")
        return None
    return parser(payload)


class PrivateTokens:
    """Stable opaque report IDs with an owner-only reverse map."""

    def __init__(self, path: Path, identity: Mapping[str, Any], output: Path) -> None:
        self.path = Path(path).expanduser().absolute()
        if self.path.is_symlink() or self.path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Private mapping must be a regular file outside the report directory.")
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            if not self.path.is_file():
                raise ValueError("Private mapping path must be a regular file.")
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != 1 or self.data.get("source_sha256") != digest:
                raise ValueError("Private mapping belongs to another source identity.")
            self.salt = base64.b64decode(self.data["salt"], validate=True)
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        else:
            self.salt = secrets.token_bytes(32)
            self.data = {"schema": 1, "source_sha256": digest,
                         "salt": base64.b64encode(self.salt).decode(), "cases": {}, "points": {}}
        self.case_tokens = {row["case_id"]: token for token, row in self.data["cases"].items()}
        self.point_tokens = {row["tile_id"]: token for token, row in self.data["points"].items()}

    def token(self, kind: str, raw: str) -> str:
        suffix = hmac.new(self.salt, f"{kind}\0{raw}".encode(), hashlib.sha256).hexdigest()[:16]
        return kind.upper() + "-" + suffix

    def case(self, case_id: str, patient_id: str, target: int) -> str:
        token = self.case_tokens.get(case_id) or self.token("case", case_id)
        row = {"case_id": case_id, "patient_id": patient_id, "target": int(target)}
        if token in self.data["cases"] and self.data["cases"][token] != row:
            raise ValueError("Case token has conflicting patient/label metadata.")
        self.data["cases"][token] = row
        self.case_tokens[case_id] = token
        return token

    def point(self, ref: Mapping[str, Any], scope: str) -> str:
        tile = str(ref["tile_id"])
        token = self.point_tokens.get(tile) or self.token("point", tile)
        row = {"tile_id": tile, "photo_id": ref["photo_id"], "case_id": ref["case_id"],
               "lens": ref["lens"], "scope": scope}
        if token in self.data["points"] and self.data["points"][token] != row:
            raise ValueError("Point token has conflicting source lineage.")
        self.data["points"][token] = row
        self.point_tokens[tile] = token
        return token

    def save(self) -> None:
        _write_private(self.path, json.dumps(self.data, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False).encode("utf-8"))


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        temp.unlink(missing_ok=True)


def _oof(store: ArtifactStore, name: str, cohort: set[str], tokens: PrivateTokens) -> list[dict[str, Any]] | None:
    rows = read(store, name)
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError(f"OOF artifact must be a list: {name}")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid OOF row in {name}")
        case, patient, target, probability = (row.get(key) for key in ("case_id", "patient_id", "target", "probability"))
        if (not isinstance(case, str) or case not in cohort or case in seen or not isinstance(patient, str)
                or type(target) is not int or target not in (0, 1) or isinstance(probability, bool)
                or not isinstance(probability, (int, float)) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError(f"Invalid OOF row in {name}")
        seen.add(case)
        row["case_token"] = tokens.case(case, patient, target)
    return rows


def _assignment_rows(store: ArtifactStore, base: str, refs: list[Mapping[str, Any]],
                     source: Mapping[str, Any], lens: str, scope: str) -> list[dict[str, Any]] | None:
    payload = store.read(f"{base}/assignments.csv.gz")
    if payload is None:
        return None
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(payload).decode("utf-8"))))
    if len(rows) != len(refs) or [row.get("tile_id") for row in rows] != [ref["tile_id"] for ref in refs]:
        raise ValueError(f"Assignment table differs from committed tile order: {base}")
    for row, ref in zip(rows, refs, strict=True):
        tile, case, photo = ref["tile_id"], ref["case_id"], ref["photo_id"]
        source_row = source["rows"].get(tile)
        try:
            label = int(row["cluster_id"])
            target = int(row["target"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid assignment metadata in {base}") from exc
        if (not isinstance(source_row, Mapping) or source_row.get("case_id") != case
                or source_row.get("photo_id") != photo or str(source_row.get("lens")) != lens
                or row.get("scope") != scope or row.get("case_id") != case
                or row.get("patient_id") != ref["patient_id"] or row.get("photo_id") != photo
                or row.get("lens") != lens or target != int(ref["source_case_label"])):
            raise ValueError(f"Assignment/source lineage mismatch for {tile}")
        if label < -1:
            raise ValueError(f"Cluster id below noise sentinel in {base}")
    return rows


def _partition_metrics(metrics: Mapping[str, Any], labels: list[int], base: str) -> None:
    partition = metrics.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError(f"Partition summary missing: {base}")
    count = len(labels)
    assigned = sum(label >= 0 for label in labels)
    clusters = len({label for label in labels if label >= 0})
    coverage = assigned / count if count else 0.0
    if (metrics.get("assignment_rows") != count or partition.get("row_count") != count
            or partition.get("assigned_count") != assigned or partition.get("cluster_count") != clusters
            or not math.isclose(float(partition.get("coverage", -1)), coverage, abs_tol=1e-7)
            or not math.isclose(float(partition.get("noise_fraction", -1)), 1.0 - coverage, abs_tol=1e-7)):
        raise ValueError(f"Committed cluster metrics disagree with assignments: {base}")


def _point_records(rows: list[dict[str, Any]], refs: list[Mapping[str, Any]], features: np.ndarray,
                   lens: str, scope: str, target_by_case: Mapping[str, int],
                   tokens: PrivateTokens, algorithm: str) -> list[dict[str, Any]]:
    points = []
    for index, (row, ref) in enumerate(zip(rows, refs, strict=True)):
        target = int(ref["source_case_label"])
        if target_by_case[ref["case_id"]] != target:
            raise ValueError("A0 source case label differs from verified transform membership.")
        points.append({
            "scope": scope, "algorithm": algorithm, "point_token": tokens.point(ref, scope),
            "case_token": tokens.case(ref["case_id"], ref["patient_id"], target),
            "patient_token": tokens.token("patient", ref["patient_id"]),
            "photo_token": tokens.token("photo", ref["photo_id"]),
            "tile_id": ref["tile_id"], "part_id": ref["part_id"], "row_index": int(ref["row_index"]),
            "lens": lens, "x": float(features[index, 0]), "y": float(features[index, 1]),
            "target": target, "cluster_id": int(row["cluster_id"]),
        })
    return points


def _atlas(a: ArtifactStore, shared: ArtifactStore, source: Mapping[str, Any],
           tokens: PrivateTokens) -> dict[str, Any]:
    scope = "outer_0/inner_0"
    transforms = read(shared, f"scopes/{scope}/transform.pkl", pickle.loads)
    if not isinstance(transforms, dict):
        return {"status": "pending", "scope": scope, "development_cases": 5,
                "lenses": {}, "missing": ["A0 development transform"]}
    members = transforms.get("members")
    if not isinstance(members, Mapping) or len(members) != 5:
        raise ValueError("A0 report requires the predeclared five-case development scope.")
    patient_ids = [member.get("patient_id") for member in members.values() if isinstance(member, Mapping)]
    if len(patient_ids) != 5 or len(set(patient_ids)) != 5 or transforms.get("dimensions") != 32:
        raise ValueError("A0 development membership or PCA32 metadata is invalid.")
    target_by_case: dict[str, int] = {}
    for case, member in members.items():
        if (not isinstance(case, str) or not isinstance(member, Mapping)
                or not isinstance(member.get("patient_id"), str) or type(member.get("target")) is not int
                or member["target"] not in (0, 1)):
            raise ValueError("A0 development membership has invalid case metadata.")
        target_by_case[case] = int(member["target"])
    result: dict[str, Any] = {"status": "complete", "scope": scope, "development_cases": 5,
                              "lenses": {}, "missing": []}
    for lens in LENSES:
        layer = transforms.get("lenses", {}).get(lens)
        if not isinstance(layer, Mapping):
            raise ValueError(f"A0 transform is missing lens {lens}.")
        features = np.asarray(layer.get("sample_projected"))
        refs = layer.get("sample_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"Invalid A0 development references for lens {lens}")
        if (features.dtype != np.float32 or features.ndim != 2 or features.shape != (len(refs), 32)
                or not np.isfinite(features).all()):
            raise ValueError(f"Invalid A0 development sample for lens {lens}")
        seen_tiles: set[str] = set()
        for ref in refs:
            case = ref.get("case_id") if isinstance(ref, Mapping) else None
            source_row = source["rows"].get(ref.get("tile_id")) if isinstance(ref, Mapping) else None
            if (case not in members or ref.get("patient_id") != members[case]["patient_id"]
                    or ref.get("source_case_label") != target_by_case[case] or ref.get("lens") != lens
                    or not isinstance(ref.get("part_id"), str) or type(ref.get("row_index")) is not int
                    or ref.get("tile_id") in seen_tiles or not isinstance(source_row, Mapping)
                    or source_row.get("case_id") != case or source_row.get("photo_id") != ref.get("photo_id")
                    or source_row.get("part_id") != ref.get("part_id")
                    or source_row.get("row_index") != ref.get("row_index")):
                raise ValueError("A0 sample refs do not match the committed train-only source lineage.")
            seen_tiles.add(ref["tile_id"])
        if {ref["case_id"] for ref in refs} != set(members):
            raise ValueError("A0 reservoir must cover all five development cases per lens.")

        grid: list[dict[str, Any]] = []
        representatives: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for size, samples, method in HDB:
            base = f"atlas/outer_0_inner_0/hdbscan/lens_{lens}/mcs{size}_ms{samples}_{method}"
            metrics = read(a, f"{base}/metrics.json")
            if metrics is None:
                missing.append(base)
                continue
            if metrics.get("config") != {"min_cluster_size": size, "min_samples": samples,
                                          "selection_method": method}:
                raise ValueError(f"A0 HDBSCAN config metadata mismatch: {base}")
            rows = _assignment_rows(a, base, refs, source, lens, scope)
            if rows is None:
                missing.append(base + "/assignments")
                continue
            labels = [int(row["cluster_id"]) for row in rows]
            _partition_metrics(metrics, labels, base)
            part = metrics["partition"]
            grid.append({"method": method, "min_cluster_size": size, "min_samples": samples,
                         "coverage": float(part["coverage"]), "noise": float(part["noise_fraction"]),
                         "clusters": int(part["cluster_count"])})
            if (size, samples) == (16, 4):
                algorithm = f"hdbscan_{method}_mcs{size}_ms{samples}"
                points = _point_records(rows, refs, features, lens, scope, target_by_case, tokens, algorithm)
                key = f"hdb_{method}_16_4"
                representatives[key] = {"title": f"HDBSCAN {method} mcs16/ms4", "algorithm": algorithm,
                                         "points": points, "metrics": metrics}

        for kind, configs in (("ward", (2, 4, 8, 16)), ("kmeans", (4, 8, 16))):
            for k in configs:
                base = (f"atlas/outer_0_inner_0/ward/lens_{lens}/k{k}" if kind == "ward"
                        else f"atlas/outer_0_inner_0/kmeans/lens_{lens}/k{k}")
                metrics = read(a, f"{base}/metrics.json")
                if metrics is None:
                    missing.append(base)
                    continue
                rows = _assignment_rows(a, base, refs, source, lens, scope)
                if rows is None:
                    missing.append(base + "/assignments")
                    continue
                labels = [int(row["cluster_id"]) for row in rows]
                _partition_metrics(metrics, labels, base)
                if metrics.get("cluster_count") != k or int(metrics["partition"]["cluster_count"]) != k:
                    raise ValueError(f"A0 {kind} K/cut metadata mismatch: {base}")
                if (kind == "ward" and k == 4) or kind == "kmeans":
                    algorithm = f"{kind}_k{k}"
                    key = f"{kind}_k{k}"
                    points = _point_records(rows, refs, features, lens, scope, target_by_case, tokens, algorithm)
                    representatives[key] = {"title": f"{kind.upper()} k{k}", "algorithm": algorithm,
                                             "points": points, "metrics": metrics}
                grid.append({"method": kind, "k": k, "coverage": float(metrics["partition"]["coverage"]),
                             "noise": float(metrics["partition"]["noise_fraction"]), "clusters": k})
        result["missing"].extend(missing)
        if missing:
            result["status"] = "partial"
        result["lenses"][lens] = {
            "sample_rows": len(refs), "sample_cases": len({ref["case_id"] for ref in refs}),
            "configs": grid, "representatives": representatives, "status": "partial" if missing else "complete",
        }
    return result


def _public_atlas(atlas: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": atlas["status"], "scope": atlas["scope"],
            "development_case_count": atlas["development_cases"], "missing": list(atlas["missing"]),
            "lenses": {lens: {"sample_rows": data["sample_rows"], "sample_cases": data["sample_cases"],
                              "status": data["status"], "configs": data["configs"]}
                       for lens, data in atlas["lenses"].items()}}


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def point_lineage_csv(atlas: Mapping[str, Any]) -> str:
    fields = ["scope", "lens", "point_token", "case_token", "patient_token", "photo_token",
              "source_case_label", "x", "y", *(f"cluster_{key}" for key, _ in REPS)]
    points: dict[tuple[str, str], dict[str, Any]] = {}
    for lens, data in atlas.get("lenses", {}).items():
        for key, item in data["representatives"].items():
            for row in item["points"]:
                point = points.setdefault((lens, row["point_token"]), {field: "" for field in fields})
                for field in fields[:9]:
                    point[field] = row.get(field, "")
                point[f"cluster_{key}"] = row["cluster_id"]
    return _csv_text(list(points.values()), fields)


def oof_lineage_csv(models: Mapping[str, list[dict[str, Any]]]) -> str:
    def column_name(name: str) -> str:
        return name.lower().replace(" ", "_").replace("=", "").replace(".", "_").replace("-", "_")

    rows: dict[str, dict[str, Any]] = {}
    for name, values in models.items():
        for row in values:
            point = rows.setdefault(row["case_token"], {"case_token": row["case_token"],
                                                         "source_case_label": row["target"]})
            point[column_name(name)] = row["probability"]
            if name == "B1 selected":
                point["outer_fold"] = row["outer_fold"]
                point["selected_beta"] = row["beta"]
    fields = ["case_token", "source_case_label", "outer_fold", "selected_beta",
              "a1", "a_counts", "b0", "b_counts"]
    fields.extend(column_name(name) for name in models if name.startswith("B1 "))
    return _csv_text(list(rows.values()), fields)


def _validate_controls(main: list[dict[str, Any]], control: list[dict[str, Any]], name: str) -> None:
    if not main or not control:
        return
    model, counts = ({row["case_id"]: row for row in rows} for rows in (main, control))
    if set(model) != set(counts) or any((model[key]["patient_id"], model[key]["target"])
                                        != (counts[key]["patient_id"], counts[key]["target"]) for key in model):
        raise ValueError(f"{name} primary and counts OOF rows do not align.")


def _metadata_by_case(models: Mapping[str, list[dict[str, Any]]]) -> tuple[dict[str, str], dict[str, int]]:
    patients: dict[str, str] = {}
    targets: dict[str, int] = {}
    for rows in models.values():
        for row in rows:
            case = row["case_id"]
            metadata = (row["patient_id"], row["target"])
            if case in patients and (patients[case], targets[case]) != metadata:
                raise ValueError("A/B OOF metadata differs for a shared case.")
            patients[case], targets[case] = metadata
    return patients, targets


def _b1_prediction_rows(
    rows: Any,
    *,
    name: str,
    cohort: set[str],
    expected_cases: set[str],
    patient_by_case: Mapping[str, str],
    target_by_case: Mapping[str, int],
    tokens: PrivateTokens,
    beta_by_outer: Mapping[str, float],
    fixed_beta: float | None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"B1 {name} predictions must be a row list.")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"B1 {name} prediction rows must be objects.")
        case, target, probability = (row.get(key) for key in ("case_id", "target", "probability"))
        fold, beta = row.get("outer_fold"), row.get("beta")
        if (not isinstance(case, str) or case not in cohort or case in seen
                or case not in patient_by_case or case not in target_by_case
                or type(target) is not int or target != target_by_case[case]
                or isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1
                or type(fold) is not int or str(fold) not in beta_by_outer
                or isinstance(beta, bool) or not isinstance(beta, (int, float))
                or float(beta) != (fixed_beta if fixed_beta is not None else beta_by_outer[str(fold)])
                or (fixed_beta is not None and float(beta) != fixed_beta)
                or ("patient_id" in row and row["patient_id"] != patient_by_case[case])):
            raise ValueError(f"Invalid B1 {name} OOF metadata for case {case!r}.")
        seen.add(case)
        normalized.append({"case_id": case, "patient_id": patient_by_case[case], "target": target,
                           "probability": float(probability), "outer_fold": fold, "beta": float(beta),
                           "case_token": tokens.case(case, patient_by_case[case], target)})
    if seen != expected_cases:
        raise ValueError(f"B1 {name} predictions cover {len(seen)} rows; expected {len(expected_cases)}.")
    return normalized


def _read_b1_result(store: ArtifactStore | None, artifact: str | None, cohort: set[str],
                    base_models: Mapping[str, list[dict[str, Any]]],
                    tokens: PrivateTokens) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if store is None or artifact is None:
        return {}, {"status": "pending", "source": "run_neural_study final_result.json"}
    result = read(store, artifact)
    if not isinstance(result, Mapping):
        return {}, {"status": "pending", "source": "run_neural_study final_result.json",
                    "missing": [artifact]}
    case_ids = result.get("case_ids")
    expected_variant_names = {"B1_beta0_control", "B1_selected", "B1_beta05_selected_folds"}
    if (result.get("phase") != "complete" or result.get("software_only") is not False
            or result.get("outer_refits_performed") is not True
            or not isinstance(result.get("study_id"), str) or not result["study_id"]
            or not isinstance(result.get("protocol"), str) or not result["protocol"]
            or result.get("oof_rows") != len(cohort)
            or not isinstance(case_ids, list) or any(not isinstance(case, str) for case in case_ids)
            or len(case_ids) != len(cohort) or set(case_ids) != cohort):
        raise ValueError("B1 final_result must be a complete full-cohort outer-refit result.")
    predictions = result.get("oof_predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != expected_variant_names:
        raise ValueError("B1 final_result must contain the three committed prediction groups.")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"B1_beta0_control", "B1_selected"}:
        raise ValueError("B1 final_result must contain metrics for the two full-cohort variants.")
    selection = result.get("selected_beta_by_outer")
    if (not isinstance(selection, Mapping) or not selection
            or any(not isinstance(key, str) or not key.isdecimal()
                   or type(value) not in (int, float) or float(value) not in (0.0, 0.5)
                   for key, value in selection.items())):
        raise ValueError("B1 selected_beta_by_outer must map fold ids to beta 0.0 or 0.5.")
    beta_by_outer = {key: float(value) for key, value in selection.items()}
    patient_by_case, target_by_case = _metadata_by_case(base_models)
    full = _b1_prediction_rows(
        predictions["B1_beta0_control"], name="beta0_control", cohort=cohort,
        expected_cases=cohort, patient_by_case=patient_by_case, target_by_case=target_by_case,
        tokens=tokens, beta_by_outer=beta_by_outer, fixed_beta=0.0,
    )
    selected = _b1_prediction_rows(
        predictions["B1_selected"], name="selected", cohort=cohort,
        expected_cases=cohort, patient_by_case=patient_by_case, target_by_case=target_by_case,
        tokens=tokens, beta_by_outer=beta_by_outer, fixed_beta=None,
    )
    control_by_case = {row["case_id"]: row for row in full}
    selected_by_case = {row["case_id"]: row for row in selected}
    if any(control_by_case[case]["outer_fold"] != selected_by_case[case]["outer_fold"] for case in cohort):
        raise ValueError("B1 control and selected predictions disagree on outer-fold assignment.")
    folds = {str(row["outer_fold"]) for row in selected}
    if folds != set(beta_by_outer) or {row["beta"] for row in selected} - {0.0, 0.5}:
        raise ValueError("B1 selected OOF rows do not cover the declared outer folds.")
    beta05_cases = {case for case, row in selected_by_case.items() if row["beta"] == 0.5}
    beta05 = _b1_prediction_rows(
        predictions["B1_beta05_selected_folds"], name="beta05_selected_folds", cohort=cohort,
        expected_cases=beta05_cases, patient_by_case=patient_by_case, target_by_case=target_by_case,
        tokens=tokens, beta_by_outer=beta_by_outer, fixed_beta=0.5,
    )
    for row in beta05:
        selected_row = selected_by_case[row["case_id"]]
        if (row["outer_fold"], row["target"], row["probability"]) != (
                selected_row["outer_fold"], selected_row["target"], selected_row["probability"]):
            raise ValueError("B1 beta .5 subset differs from the matching selected-fold predictions.")
    models = {"B1 beta0 control": full, "B1 selected": selected,
              "B1 beta0.5 selected-fold subset": beta05}
    return models, {
        "status": "complete", "source": "run_neural_study final_result.json",
        "control_cases": len(full), "selected_cases": len(selected),
        "selected_beta_by_outer": beta_by_outer,
        "beta05_selected_folds": {"status": "subset_only" if beta05 else "not_selected",
                                  "cases": len(beta05), "cohort_cases": len(cohort),
                                  "coverage": len(beta05) / len(cohort),
                                  "outer_folds": sorted({row["outer_fold"] for row in beta05})},
    }


def prepare_report_data(a: ArtifactStore, b: ArtifactStore, shared: ArtifactStore,
                        source: Mapping[str, Any], output: Path, private_map: Path, *,
                        b1: ArtifactStore | None = None,
                        b1_result_artifact: str | None = None) -> dict[str, Any]:
    committed_source = read(shared, "source_photo_map.pkl", pickle.loads, required=True)
    if committed_source != source:
        raise ValueError("Source-photo map differs from the committed shared artifact.")
    cohort = set(source.get("case_tile_order", {}))
    if not cohort or not isinstance(source.get("source_identity"), Mapping):
        raise ValueError("Committed source-photo map has no cohort or dataset identity.")
    tokens = PrivateTokens(private_map, source["source_identity"], output)
    models: dict[str, list[dict[str, Any]]] = {
        "A1": _oof(a, "results/a1_oof.json", cohort, tokens) or [],
        "A counts": _oof(a, "results/counts_only_oof.json", cohort, tokens) or [],
        "B0": _oof(b, "results/b0_oof.json", cohort, tokens) or [],
        "B counts": _oof(b, "results/counts_only_oof.json", cohort, tokens) or [],
    }
    _validate_controls(models["A1"], models["A counts"], "A1")
    _validate_controls(models["B0"], models["B counts"], "B0")
    a_by, b_by = ({row["case_id"]: row for row in models[name]} for name in ("A1", "B0"))
    matched = sorted(set(a_by) & set(b_by))
    if any((a_by[key]["patient_id"], a_by[key]["target"])
           != (b_by[key]["patient_id"], b_by[key]["target"]) for key in matched):
        raise ValueError("A1/B0 matched cases disagree on patient or source case label.")
    matched_status = "matched_full_cohort" if (
        all({row["case_id"] for row in models[name]} == cohort
            for name in ("A1", "A counts", "B0", "B counts"))
        and matched and {a_by[key]["target"] for key in matched} == {0, 1}
    ) else "partial"

    b1_models, b1_manifest = _read_b1_result(
        b1, b1_result_artifact, cohort, models, tokens,
    )
    models.update(b1_models)
    atlas = _atlas(a, shared, source, tokens)
    tokens.save()
    manifest = {
        "status": "complete" if atlas["status"] == "complete"
        and matched_status == "matched_full_cohort" and b1_manifest["status"] == "complete" else "partial",
        "a0": _public_atlas(atlas),
        "matched_oof": {"status": matched_status, "matched_cases": len(matched), "cohort_cases": len(cohort)},
        "b1": b1_manifest,
    }
    return {"manifest": manifest, "atlas": atlas, "models": models, "matched": matched,
            "a_by": a_by, "b_by": b_by, "point_csv": point_lineage_csv(atlas),
            "oof_csv": oof_lineage_csv(models), "tokens": tokens}


def select_representatives(atlas: Mapping[str, Any], maximum: int = 96) -> list[dict[str, Any]]:
    """Select up to five deterministic, case/photo-diverse candidates per plot config."""
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for lens in LENSES:
        data = atlas.get("lenses", {}).get(lens, {})
        for key, _ in REPS:
            item = data.get("representatives", {}).get(key)
            if item:
                groups.append((lens, item["algorithm"], item["points"]))
    if not groups or maximum < 1:
        return []
    per_group = max(1, min(5, maximum // len(groups)))
    selected: list[dict[str, Any]] = []
    for lens, algorithm, points in groups:
        remaining = list(points)
        chosen: list[dict[str, Any]] = []
        cluster_seen: set[int] = set()
        case_seen: set[str] = set()
        photo_seen: set[str] = set()
        target_count = {0: 0, 1: 0}
        if algorithm.startswith("hdbscan"):
            noise = sorted((p for p in remaining if p["cluster_id"] == -1), key=lambda p: p["point_token"])
            if noise:
                chosen.append(noise[0])
                remaining.remove(noise[0])
        while remaining and len(chosen) < per_group:
            point = max(remaining, key=lambda p: (
                p["cluster_id"] >= 0 and p["cluster_id"] not in cluster_seen,
                p["case_token"] not in case_seen,
                p["photo_token"] not in photo_seen,
                -target_count[p["target"]],
                p["point_token"],
            ))
            chosen.append(point)
            remaining.remove(point)
            if point["cluster_id"] >= 0:
                cluster_seen.add(point["cluster_id"])
            case_seen.add(point["case_token"])
            photo_seen.add(point["photo_token"])
            target_count[point["target"]] += 1
        for point in chosen:
            ref_role = "noise_review" if point["cluster_id"] == -1 else "cluster_review"
            selected.append({"tile_id": point["tile_id"], "lens": lens,
                             "source_case_label": point["target"], "algorithm": algorithm,
                             "cluster_id": point["cluster_id"], "part_id": point["part_id"],
                             "row_index": point["row_index"], "prototype_role": ref_role})
    return selected[:maximum]


def representative_csv(rows: list[dict[str, Any]]) -> str:
    return _csv_text(rows, REPRESENTATIVE_FIELDS)


def concentration_rows(
    representatives: Mapping[str, Any],
) -> list[tuple[int, str, int, int, int, float, float, int]]:
    """Summarize cluster counts and largest patient/photo row shares for plotted partitions."""
    result = []
    for index, (key, _) in enumerate(REPS):
        item = representatives.get(key)
        if item is None:
            continue
        clusters: dict[int, list[dict[str, Any]]] = {}
        for point in item["points"]:
            clusters.setdefault(point["cluster_id"], []).append(point)
        for cluster, rows in sorted(clusters.items()):
            patients = [row["patient_token"] for row in rows]
            photos = [row["photo_token"] for row in rows]
            result.append((index, item["title"], cluster,
                           sum(row["target"] == 0 for row in rows),
                           sum(row["target"] == 1 for row in rows),
                           max(patients.count(value) for value in set(patients)) / len(rows),
                           max(photos.count(value) for value in set(photos)) / len(rows), len(rows)))
    return result
