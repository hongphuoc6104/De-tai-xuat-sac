"""Authoritative metadata catalog; weak case labels never become patch grades."""
from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .io import file_hash, fingerprint

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def safe_member(name: str) -> str:
    """Reject traversal, absolute, Windows and ambiguous archive names."""
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError(f"Unsafe source member: {name!r}")
    if name != path.as_posix():
        raise ValueError(f"Noncanonical source member: {name!r}")
    return name


def source_inventory(source: Path, source_id: str) -> list[dict[str, Any]]:
    """Inspect directory files/ZIP central records without decoding all images."""
    source = Path(source).resolve()
    entries: list[dict[str, Any]] = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.is_symlink() or not path.resolve().is_relative_to(source):
                    raise ValueError("Source images must be regular files within their source root.")
                entries.append(dict(source_id=source_id, source_member=safe_member(path.relative_to(source).as_posix()),
                                    file_name=path.name, byte_size=path.stat().st_size,
                                    source_signature=None))
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            seen: set[str] = set()
            for info in archive.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                member = safe_member(info.filename)
                if member in seen or info.flag_bits & 1:
                    raise ValueError("Duplicate or encrypted source image in ZIP.")
                seen.add(member)
                entries.append(dict(source_id=source_id, source_member=member,
                                    file_name=PurePosixPath(member).name, byte_size=info.file_size,
                                    source_signature=f"zip-crc32:{info.CRC:08x}"))
    else:
        raise ValueError(f"Expected image directory or readable ZIP: {source}")
    return entries


def build_catalog(metadata: Path, sources: list[Path], lenses: list[int] | None = None,
                  per_lens: int | None = None, identity_map: Path | None = None,
                  labels_reviewed: bool = False) -> dict[str, Any]:
    """Build portable provenance and deterministic balanced small smoke subsets.

    Full preparation can use candidate case identity; scientific training requires
    an explicit case-to-patient mapping and reviewed case-level binary labels.
    """
    metadata = Path(metadata)
    raw = pd.read_csv(metadata, dtype=str, keep_default_na=False) if metadata.suffix.lower() == ".csv" else pd.read_excel(metadata, dtype=str, keep_default_na=False)
    required = {"Ten_File", "Ma_Nam", "Ma_So", "Do_Phong_Dai", "Glade", "Ket_Luan", "Ten_Slide"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Authoritative metadata missing fields: {sorted(required - set(raw.columns))}")
    selected_lenses = sorted(set(lenses or [4, 10, 40]))
    if any(lens not in {4, 10, 40} for lens in selected_lenses) or (per_lens is not None and per_lens < 1):
        raise ValueError("Use lenses 4/10/40 and positive per_lens smoke limits.")
    if raw.Ten_File.duplicated().any():
        raise ValueError("Duplicate filenames in authoritative metadata.")
    identities: dict[str, str] = {}
    if identity_map is not None:
        table = pd.read_csv(identity_map, dtype=str, keep_default_na=False)
        if not {"case_id", "patient_id"}.issubset(table.columns) or table.case_id.duplicated().any():
            raise ValueError("Identity map requires unique case_id and patient_id columns.")
        if table.patient_id.str.strip().eq("").any():
            raise ValueError("Identity map contains empty patient identifiers.")
        identities = dict(zip(table.case_id, table.patient_id.str.strip(), strict=True))
    inventory: dict[str, dict[str, Any]] = {}
    definitions = []
    for i, source in enumerate(sources):
        source_id = f"source-{i:03d}"
        definitions.append(dict(source_id=source_id, path=str(Path(source).resolve()),
                                kind="directory" if Path(source).is_dir() else "zip"))
        for entry in source_inventory(source, source_id):
            if entry["file_name"] in inventory:
                raise ValueError(f"Image present in multiple sources: {entry['file_name']}")
            inventory[entry["file_name"]] = entry
    by_lens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    expected: dict[int, int] = defaultdict(int)
    missing = []
    for record in raw.to_dict("records"):
        name = str(record["Ten_File"]).strip()
        if PurePosixPath(safe_member(name)).name != name:
            raise ValueError("Metadata file names must be basenames.")
        match = re.fullmatch(r"(4|10|40)[xX×]?", str(record["Do_Phong_Dai"]).strip())
        if not match:
            raise ValueError(f"Unknown objective lens for {name}")
        lens = int(match.group(1))
        if lens not in selected_lenses:
            continue
        expected[lens] += 1
        year = re.sub(r"[\s-]+", "", str(record["Ma_Nam"])).upper()
        number = str(record["Ma_So"]).strip()
        if not year or not number:
            raise ValueError(f"Missing case identity: {name}")
        case = f"{year}_{number}"
        if name not in inventory:
            missing.append(name)
            continue
        report = str(record["Ket_Luan"]).strip().upper()
        if "CARCIN" in report:
            label: int | None = 1
        elif "TĂNG SẢN" in report or "HYPERPLASIA" in report:
            label = 0
        else:
            label = None
        patient = identities.get(case)
        image_id = Path(name).stem
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", image_id):
            raise ValueError(f"Image ID cannot safely name a shard member: {name}")
        by_lens[lens].append(dict(inventory[name], image_id=image_id, objective_lens=lens,
                                 case_id=case, patient_id=patient, slide_group_id=str(record["Ten_Slide"]),
                                 raw_glade=str(record["Glade"]), grade_semantics="unconfirmed",
                                 case_label=label, label_level="case", label_source="Ket_Luan",
                                 identity_status="verified_mapping" if patient else "candidate_case"))
    images = []
    for lens in selected_lenses:
        records = sorted(by_lens[lens], key=lambda x: (x["case_id"], x["image_id"]))
        if not records:
            raise ValueError(f"No available source images at {lens}X.")
        if per_lens is not None:
            # Deterministic alternation of both labels and cases; limits are technical smoke only.
            groups: dict[tuple[int | None, str], list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                groups[(record["case_label"], record["case_id"])].append(record)
            labels = sorted({key[0] for key in groups}, key=lambda value: (value is None, str(value)))
            class_records = {}
            for label in labels:
                keys = sorted(key for key in groups if key[0] == label)
                class_records[label] = [groups[key][offset] for offset in range(max(len(groups[key]) for key in keys))
                                        for key in keys if offset < len(groups[key])]
            order = [class_records[label][offset] for offset in range(max(map(len, class_records.values())))
                     for label in labels if offset < len(class_records[label])]
            records = order[:per_lens]
        images.extend(records)
    roots = {source["source_id"]: Path(source["path"]) for source in definitions}
    for image in images:
        if image["source_signature"] is None:
            image["source_signature"] = file_hash(roots[image["source_id"]] / image["source_member"])
    ids = [image["image_id"] for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs collide across magnifications.")
    case_labels: dict[str, set[int | None]] = defaultdict(set)
    for image in images:
        case_labels[image["case_id"]].add(image["case_label"])
    if any(len(values) > 1 for values in case_labels.values()):
        raise ValueError("Conflicting case-level conclusions; review labels before preparation.")
    mode = "smoke" if per_lens is not None else "full"
    ready = mode == "full" and not missing and labels_reviewed and all(image["patient_id"] and image["case_label"] is not None for image in images)
    body = dict(schema_version=1, mode=mode, metadata_sha256=file_hash(metadata),
                lenses=selected_lenses, images=images, missing_images=missing,
                labels_reviewed=labels_reviewed, training_ready=ready,
                coverage={str(lens): {"expected": expected[lens], "selected": len([x for x in images if x["objective_lens"] == lens])} for lens in selected_lenses})
    body["catalog_id"] = fingerprint(body)
    return dict(body, sources=definitions)


def require_training_ready(catalog: dict[str, Any]) -> None:
    """Reject smoke or unreviewed identity/labels at the training boundary."""
    if catalog.get("mode") != "full" or not catalog.get("training_ready"):
        raise ValueError("Training requires full coverage, verified patient mapping and reviewed case labels.")
