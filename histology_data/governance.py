"""Reviewed case identity and label governance for downstream MIL bundles.

This module never turns metadata heuristics into patient identities or labels.
It prepares a review template and accepts only explicit, evidenced review rows.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_sources import build_source_plan
from .io import file_hash, fingerprint
from .precut import REQUIRED_METADATA, _metadata_rows

CASE_REVIEW_FIELDS = (
    "case_id",
    "patient_id",
    "identity_verified",
    "identity_evidence",
    "case_label",
    "label_verified",
    "label_evidence",
    "raw_glade_values_json",
    "raw_conclusion_values_json",
    "candidate_label_codes_json",
    "metadata_case_sha256",
)

CASE_FIELDS = (
    "case_id",
    "patient_id",
    "case_label",
    "identity_evidence",
    "label_evidence",
    "raw_glade_values_json",
    "raw_conclusion_values_json",
    "candidate_label_codes_json",
    "candidate_label_status",
    "objective_lenses_json",
    "image_ids_json",
    "metadata_case_sha256",
)


class GovernanceError(ValueError):
    """Input review data is incomplete, inconsistent, or not tied to metadata."""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _csv_bytes(fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return stream.getvalue().encode("utf-8")


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Publish content at a fingerprinted path, rejecting any changed overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Immutable governance artifact differs: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_context(metadata_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Return the established metadata mapping plus exact per-case source rows."""
    metadata_path = Path(metadata_path)
    frame = (pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
             if metadata_path.suffix.lower() == ".csv"
             else pd.read_excel(metadata_path, dtype=str, keep_default_na=False))
    missing = REQUIRED_METADATA - set(frame.columns)
    if missing:
        raise GovernanceError(f"Metadata missing required fields: {sorted(missing)}")
    context = _metadata_rows(metadata_path)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_stems: set[str] = set()
    for record in frame.to_dict("records"):
        name = str(record["Ten_File"]).strip()
        stem = Path(name).stem
        if stem in seen_stems:
            raise GovernanceError(f"Metadata image stems are ambiguous: {stem}")
        seen_stems.add(stem)
        row_context = context.get(stem)
        if row_context is None:
            raise GovernanceError(f"Metadata image is not present in the canonical map: {name}")
        exact = {
            "image_id": stem,
            "Ten_File": str(record["Ten_File"]),
            "Ma_Nam": str(record["Ma_Nam"]),
            "Ma_So": str(record["Ma_So"]),
            "Do_Phong_Dai": str(record["Do_Phong_Dai"]),
            "Glade": str(record["Glade"]),
            "Ket_Luan": str(record["Ket_Luan"]),
            "Ten_Slide": str(record["Ten_Slide"]),
            "objective_lens": int(row_context["objective_lens"]),
            "candidate_label_code": row_context["candidate_label_from_conclusion"],
        }
        by_case[str(row_context["candidate_case_id"])].append(exact)
    for rows in by_case.values():
        rows.sort(key=lambda row: row["image_id"])
    return context, by_case


def _case_review_source_row(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_glade = [{"image_id": row["image_id"], "value": row["Glade"]} for row in rows]
    conclusions = [{"image_id": row["image_id"], "value": row["Ket_Luan"]} for row in rows]
    candidate_codes = [{"image_id": row["image_id"], "value": row["candidate_label_code"]} for row in rows]
    metadata_digest = fingerprint({"case_id": case_id, "rows": rows})
    return {
        "case_id": case_id,
        "raw_glade_values_json": _json_text(raw_glade),
        "raw_conclusion_values_json": _json_text(conclusions),
        "candidate_label_codes_json": _json_text(candidate_codes),
        "objective_lenses_json": _json_text(sorted({row["objective_lens"] for row in rows})),
        "image_ids_json": _json_text([row["image_id"] for row in rows]),
        "metadata_case_sha256": metadata_digest,
    }


def create_case_review_draft(
    metadata: Path,
    source_root: Path,
    *,
    source_kind: str = "zip",
    expected_sources: int | None = 25,
    expected_patches: int | None = 148991,
    directory_part_size: int = 1024,
) -> dict[str, Any]:
    """Build a no-pixel-read case-review draft from metadata and source inventory."""
    metadata = Path(metadata)
    _, by_case = _metadata_context(metadata)
    plan = build_source_plan(
        metadata,
        Path(source_root),
        source_kind,
        directory_part_size=directory_part_size,
        expected_archives=expected_sources,
        expected_pngs=expected_patches,
    )
    observed_case_ids = sorted({
        row.get("candidate_case_id")
        for part in plan["parts"]
        for row in part["records"]
        if row.get("metadata_match") and row.get("candidate_case_id")
    })
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not observed_case_ids:
        blockers.append("no_patch_covered_cases_observed")
    for case_id in observed_case_ids:
        source_rows = by_case.get(case_id)
        if not source_rows:
            blockers.append(f"metadata_case_missing:{case_id}")
            continue
        rows.append({
            **_case_review_source_row(case_id, source_rows),
            "patient_id": "",
            "identity_verified": "false",
            "identity_evidence": "",
            "case_label": "",
            "label_verified": "false",
            "label_evidence": "",
        })
    if not plan["source_complete"]:
        blockers.append("feature_source_incomplete")
    if plan["source_errors"]:
        blockers.append("feature_source_inventory_errors")
    blockers.extend(("patient_identity_review_required", "case_label_review_required", "feature_release_not_audited"))
    blocker_list = sorted(set(blockers))
    compact_source = {
        "metadata_sha256": plan["metadata_sha256"],
        "source_kind": plan["source_kind"],
        "observed_sources": plan["observed_sources"],
        "expected_sources": plan["expected_sources"],
        "observed_pngs": plan["observed_pngs"],
        "expected_pngs": plan["expected_pngs"],
        "parts": [{key: part.get(key) for key in ("part_id", "source_kind", "source_name", "source_signature")}
                  for part in plan["parts"]],
    }
    body = {
        "schema_version": 1,
        "metadata_sha256": file_hash(metadata),
        "source_kind": source_kind,
        "source_plan_id": fingerprint(compact_source),
        "observed_sources": plan["observed_sources"],
        "expected_sources": plan["expected_sources"],
        "observed_patches": plan["observed_pngs"],
        "expected_patches": plan["expected_pngs"],
        "source_complete": bool(plan["source_complete"]),
        "source_errors": plan["source_errors"],
        "case_count": len(rows),
        "candidate_case_ids": [row["case_id"] for row in rows],
        "candidate_label_status": "unreviewed_metadata_heuristic_only",
        "raw_glade_policy": "preserved_as_metadata_provenance_not_used_as_target",
        "training_ready": False,
        "blockers": blocker_list,
    }
    body["draft_id"] = fingerprint(body)
    return {**body, "review_rows": rows, "source_plan": plan}


def write_case_review_draft(draft: dict[str, Any], output_root: Path) -> Path:
    """Publish the review CSV and its immutable inventory/blocker report."""
    draft_id = str(draft["draft_id"])
    if len(draft_id) != 64 or fingerprint({key: value for key, value in draft.items()
                                            if key not in {"draft_id", "review_rows", "source_plan"}}) != draft_id:
        raise ValueError("Case review draft fingerprint is invalid.")
    root = Path(output_root) / "governance" / "drafts" / draft_id
    rows = draft["review_rows"]
    review_payload = _csv_bytes(CASE_REVIEW_FIELDS, rows)
    _publish_immutable(root / "case_review.csv", review_payload)
    summary = {key: value for key, value in draft.items() if key not in {"review_rows", "source_plan"}}
    summary["case_review_sha256"] = hashlib.sha256(review_payload).hexdigest()
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _publish_immutable(root / "draft.json", summary_payload)
    return root


def _read_review(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CASE_REVIEW_FIELDS:
            raise GovernanceError(f"case_review.csv columns must exactly match: {CASE_REVIEW_FIELDS}")
        rows = list(reader)
    if not rows:
        raise GovernanceError("case_review.csv is empty.")
    return rows


def validate_case_review(metadata: Path, case_review: Path) -> list[dict[str, Any]]:
    """Validate explicit rowwise identity/label attestations against source metadata."""
    _, by_case = _metadata_context(Path(metadata))
    review_rows = _read_review(Path(case_review))
    seen: set[str] = set()
    canonical: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row_number, review in enumerate(review_rows, start=2):
        case_id = str(review.get("case_id", "")).strip()
        if not case_id:
            blockers.append(f"row_{row_number}:missing_case_id")
            continue
        if case_id in seen:
            blockers.append(f"row_{row_number}:duplicate_case_id:{case_id}")
            continue
        seen.add(case_id)
        source_rows = by_case.get(case_id)
        if not source_rows:
            blockers.append(f"row_{row_number}:case_not_in_metadata:{case_id}")
            continue
        source = _case_review_source_row(case_id, source_rows)
        for field in ("raw_glade_values_json", "raw_conclusion_values_json", "candidate_label_codes_json",
                      "metadata_case_sha256"):
            if review.get(field) != source[field]:
                blockers.append(f"row_{row_number}:metadata_provenance_changed:{case_id}:{field}")
        patient_id = str(review.get("patient_id", "")).strip()
        if not patient_id or str(review.get("identity_verified", "")).strip().casefold() != "true":
            blockers.append(f"row_{row_number}:patient_identity_not_reviewed:{case_id}")
        identity_evidence = str(review.get("identity_evidence", "")).strip()
        if not identity_evidence:
            blockers.append(f"row_{row_number}:patient_identity_evidence_missing:{case_id}")
        label_text = str(review.get("case_label", "")).strip()
        if label_text not in {"0", "1"} or str(review.get("label_verified", "")).strip().casefold() != "true":
            blockers.append(f"row_{row_number}:case_label_not_reviewed_binary:{case_id}")
        label_evidence = str(review.get("label_evidence", "")).strip()
        if not label_evidence:
            blockers.append(f"row_{row_number}:case_label_evidence_missing:{case_id}")
        if (patient_id and identity_evidence and label_text in {"0", "1"} and label_evidence
                and str(review.get("identity_verified", "")).strip().casefold() == "true"
                and str(review.get("label_verified", "")).strip().casefold() == "true"
                and all(review.get(field) == source[field] for field in (
                    "raw_glade_values_json", "raw_conclusion_values_json", "candidate_label_codes_json",
                    "metadata_case_sha256"))):
            metadata_rows = source_rows
            canonical.append({
                **source,
                "patient_id": patient_id,
                "case_label": int(label_text),
                "identity_evidence": identity_evidence,
                "label_evidence": label_evidence,
                "candidate_label_status": "unreviewed_metadata_heuristic_only",
                "objective_lenses_json": source["objective_lenses_json"],
                "image_ids_json": source["image_ids_json"],
                "metadata_case_sha256": source["metadata_case_sha256"],
                "_raw_rows": metadata_rows,
            })
    if blockers:
        raise GovernanceError("Case review is blocked: " + "; ".join(sorted(set(blockers))))
    return sorted(canonical, key=lambda row: row["case_id"])


def build_governance(metadata: Path, case_review: Path, output_root: Path) -> dict[str, Any]:
    """Create a versioned immutable governance artifact from reviewed case rows."""
    metadata = Path(metadata)
    case_review = Path(case_review)
    cases = validate_case_review(metadata, case_review)
    public_cases = [{key: row[key] for key in CASE_FIELDS} for row in cases]
    case_payload = _csv_bytes(CASE_FIELDS, public_cases)
    body = {
        "schema_version": 1,
        "metadata_sha256": file_hash(metadata),
        "case_review_sha256": file_hash(case_review),
        "case_count": len(public_cases),
        "unique_patient_count": len({row["patient_id"] for row in public_cases}),
        "case_labels": {str(label): sum(row["case_label"] == label for row in public_cases) for label in (0, 1)},
        "raw_glade_policy": "preserved_as_metadata_provenance_not_used_as_target",
        "candidate_label_policy": "source_heuristic_preserved_separately_never_used_as_target",
        "governance_complete": True,
        "cases_sha256": hashlib.sha256(case_payload).hexdigest(),
    }
    governance_id = fingerprint({**body, "cases": public_cases})
    body["governance_id"] = governance_id
    root = Path(output_root) / "governance" / governance_id
    _publish_immutable(root / "cases.csv", case_payload)
    _publish_immutable(root / "case_review.csv", case_review.read_bytes())
    governance_payload = json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _publish_immutable(root / "governance.json", governance_payload)
    return {**body, "path": str(root), "cases_path": str(root / "cases.csv"),
            "case_review_path": str(root / "case_review.csv")}


def load_governance(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify a governance directory and return its canonical case rows."""
    root = Path(path)
    governance_path = root / "governance.json" if root.is_dir() else root
    body = json.loads(governance_path.read_text(encoding="utf-8"))
    case_path = governance_path.parent / "cases.csv"
    review_path = governance_path.parent / "case_review.csv"
    if file_hash(case_path) != body.get("cases_sha256"):
        raise ValueError("Governance cases.csv checksum mismatch.")
    if file_hash(review_path) != body.get("case_review_sha256"):
        raise ValueError("Governance case_review.csv checksum mismatch.")
    with case_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CASE_FIELDS:
            raise ValueError("Governance cases.csv schema mismatch.")
        rows = list(reader)
    normalized = []
    for row in rows:
        normalized.append({
            **row,
            "case_label": int(row["case_label"]),
            "raw_glade_values_json": row["raw_glade_values_json"],
        })
    expected_id = fingerprint({**{key: value for key, value in body.items() if key != "governance_id"},
                               "cases": normalized})
    # JSON CSV values remain strings, so reconstruct the exact reviewed case payload for the fingerprint.
    if expected_id != body.get("governance_id"):
        public_cases = [{key: row[key] for key in CASE_FIELDS} for row in normalized]
        if fingerprint({**{key: value for key, value in body.items() if key != "governance_id"},
                       "cases": public_cases}) != body.get("governance_id"):
            raise ValueError("Governance fingerprint mismatch.")
    if len(rows) != body.get("case_count") or not body.get("governance_complete"):
        raise ValueError("Governance case count/completion gate failed.")
    return body, rows
