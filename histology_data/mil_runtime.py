"""Verified SSD staging for governed MIL training inputs.

The source bundle and feature release are read-only inputs. The readiness
verifier writes audit files while it runs, so verification uses a temporary
hard-link mirror of staged feature files; atomic verifier outputs then stay in
that mirror and never change Drive or the immutable staged files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .features import PART_FILES
from .io import fingerprint, verified_copy
from .readiness import verify_pretrain_bundle

MAX_STAGE_BYTES = 4 * 1024**3
VERIFY_TEMP_HEADROOM_BYTES = 256 * 1024**2
STAGE_MANIFEST = "stage_manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PART_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORBIDDEN_SUFFIXES = {".zip", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svs", ".pth", ".pt"}


def _check_deadline(deadline: float) -> None:
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic timestamp.")
    if time.monotonic() >= deadline:
        raise TimeoutError("MIL input staging deadline reached.")


def _sha256_file(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            _check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, name: str, deadline: float) -> dict[str, Any]:
    _check_deadline(deadline)
    if Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {name}: {path}") from exc
    _check_deadline(deadline)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {constant}")


def _safe_relative(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe relative path for {name}.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts
            or any(part in {"", "."} for part in posix.parts) or posix.as_posix() != value):
        raise ValueError(f"Unsafe relative path for {name}: {value!r}")
    if Path(value).suffix.lower() in _FORBIDDEN_SUFFIXES:
        raise ValueError(f"Raw image/archive/weight file is outside the MIL runtime staging contract: {value}")
    return value


def _source_file(root: Path, relative: str, name: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Symlink source directory is not allowed for {name}: {cursor}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlink source file for {name}: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source file escapes its root for {name}: {path}")
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"Source path is not a regular file for {name}: {path}")
    return path


def _record(
    source_root: Path,
    source_relative: str,
    destination_relative: str,
    expected_sha256: str | None,
    deadline: float,
) -> dict[str, Any]:
    source_relative = _safe_relative(source_relative, "source")
    destination_relative = _safe_relative(destination_relative, "destination")
    source = _source_file(source_root, source_relative, source_relative)
    size = source.stat().st_size
    if size < 0:
        raise ValueError(f"Negative source file size: {source}")
    if expected_sha256 is None:
        expected_sha256 = _sha256_file(source, deadline)
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError(f"Invalid expected SHA-256 for {source_relative}.")
    return {
        "source": source,
        "source_relative": source_relative,
        "destination": destination_relative,
        "sha256": expected_sha256,
        "size": size,
    }


def _verify_file_hash(record: dict[str, Any], deadline: float) -> None:
    actual = _sha256_file(record["source"], deadline)
    if actual != record["sha256"]:
        raise ValueError(f"Source checksum mismatch: {record['source_relative']}")


def _inventory_bundle(bundle_root: Path, deadline: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle_root = bundle_root.resolve(strict=True)
    manifest = _json_object(bundle_root / "bundle.json", "bundle manifest", deadline)
    bundle_id = manifest.get("bundle_id")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if (not isinstance(bundle_id, str) or not _SHA256.fullmatch(bundle_id)
            or bundle_id != fingerprint(unsigned) or manifest.get("training_ready") is not True):
        raise ValueError("Source bundle fingerprint/readiness check failed.")
    feature_id = manifest.get("feature_id")
    if not isinstance(feature_id, str) or not _SHA256.fullmatch(feature_id):
        raise ValueError("Source bundle feature_id is invalid.")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Source bundle artifact checksum table is missing.")
    files: list[dict[str, Any]] = []
    bundle_json_sha = _sha256_file(bundle_root / "bundle.json", deadline)
    files.append(_record(bundle_root, "bundle.json", "bundle/bundle.json", bundle_json_sha, deadline))
    readiness_path = _source_file(bundle_root, "training_readiness.json", "training readiness")
    readiness_sha = _sha256_file(readiness_path, deadline)
    files.append(_record(bundle_root, "training_readiness.json", "bundle/training_readiness.json", readiness_sha, deadline))
    for relative, expected in sorted(artifact_hashes.items()):
        _check_deadline(deadline)
        relative = _safe_relative(relative, "bundle artifact")
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError(f"Invalid bundle artifact SHA-256: {relative}")
        files.append(_record(bundle_root, relative, f"bundle/{relative}", expected, deadline))
    names = {row["source_relative"] for row in files}
    if len(names) != len(files):
        raise ValueError("Source bundle file inventory contains duplicate paths.")
    _check_deadline(deadline)
    return manifest, files


def _inventory_features(
    feature_root: Path,
    bundle_manifest: dict[str, Any],
    deadline: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_root = feature_root.resolve(strict=True)
    feature_id = str(bundle_manifest["feature_id"])
    version = feature_root / feature_id
    for relative in ("status.json", f"{feature_id}/feature_config.json", f"{feature_id}/release.json"):
        _source_file(feature_root, relative, "feature control")
    status = _json_object(feature_root / "status.json", "feature status", deadline)
    config = _json_object(version / "feature_config.json", "feature configuration", deadline)
    release = _json_object(version / "release.json", "feature release", deadline)
    if status.get("feature_id") != feature_id or fingerprint(config) != feature_id:
        raise ValueError("Feature status/configuration identity differs from the source bundle.")
    if (release.get("feature_id") != feature_id or release.get("source_complete") is not True
            or release.get("feature_complete") is not True or release.get("audit_complete") is not True):
        raise ValueError("Feature release is incomplete or differs from its feature_id.")
    if (_sha256_file(version / "feature_config.json", deadline) != bundle_manifest.get("feature_config_sha256")
            or _sha256_file(version / "release.json", deadline) != bundle_manifest.get("feature_release_sha256")):
        raise ValueError("Feature config/release checksum differs from source bundle.")

    files: list[dict[str, Any]] = []
    status_sha = _sha256_file(feature_root / "status.json", deadline)
    files.append(_record(feature_root, "status.json", "features/status.json", status_sha, deadline))
    config_sha = str(bundle_manifest["feature_config_sha256"])
    release_sha = str(bundle_manifest["feature_release_sha256"])
    files.append(_record(feature_root, f"{feature_id}/feature_config.json",
                         f"features/{feature_id}/feature_config.json", config_sha, deadline))
    files.append(_record(feature_root, f"{feature_id}/release.json",
                         f"features/{feature_id}/release.json", release_sha, deadline))

    source_plan_id = release.get("source_plan_id")
    if not isinstance(source_plan_id, str) or not _SHA256.fullmatch(source_plan_id):
        raise ValueError("Feature release source_plan_id is invalid.")
    source_plan_relative = f"{feature_id}/source_plans/{source_plan_id}.json"
    _source_file(feature_root, source_plan_relative, "feature source plan")
    source_plan = _json_object(feature_root / source_plan_relative, "feature source plan", deadline)
    if fingerprint(source_plan) != source_plan_id:
        raise ValueError("Feature source plan fingerprint mismatch.")
    source_plan_sha = _sha256_file(feature_root / source_plan_relative, deadline)
    files.append(_record(feature_root, source_plan_relative,
                         f"features/{source_plan_relative}", source_plan_sha, deadline))

    audit_relative = _safe_relative(release.get("audit_file"), "feature audit")
    audit_sha = release.get("audit_sha256")
    if not isinstance(audit_sha, str) or not _SHA256.fullmatch(audit_sha):
        raise ValueError("Feature release audit SHA-256 is invalid.")
    if audit_sha != bundle_manifest.get("feature_audit_sha256"):
        raise ValueError("Feature audit checksum differs from source bundle.")
    audit_source_relative = f"{feature_id}/{audit_relative}"
    _source_file(feature_root, audit_source_relative, "feature audit")
    audit = _json_object(feature_root / audit_source_relative, "feature audit", deadline)
    files.append(_record(feature_root, audit_source_relative,
                         f"features/{audit_source_relative}", audit_sha, deadline))
    duplicate_relative = _safe_relative(audit.get("duplicate_group_file"), "duplicate group audit")
    duplicate_sha = audit.get("duplicate_group_file_sha256")
    if not isinstance(duplicate_sha, str) or not _SHA256.fullmatch(duplicate_sha):
        raise ValueError("Duplicate group audit SHA-256 is invalid.")
    duplicate_source_relative = f"{feature_id}/{duplicate_relative}"
    files.append(_record(feature_root, duplicate_source_relative,
                         f"features/{duplicate_source_relative}", duplicate_sha, deadline))

    parts = release.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Feature release has no committed feature parts.")
    seen_parts: set[str] = set()
    for part in parts:
        _check_deadline(deadline)
        if not isinstance(part, dict):
            raise ValueError("Malformed feature part entry in release.")
        part_id = part.get("part_id")
        commit_id = part.get("commit_id")
        if (not isinstance(part_id, str) or not _SAFE_PART_ID.fullmatch(part_id)
                or part_id in seen_parts or not isinstance(commit_id, str) or not _SHA256.fullmatch(commit_id)):
            raise ValueError("Invalid or duplicate feature part identity in release.")
        seen_parts.add(part_id)
        part_relative = f"{feature_id}/parts/{part_id}"
        commit_path = feature_root / part_relative / "commit.json"
        _source_file(feature_root, f"{part_relative}/commit.json", "feature part commit")
        commit = _json_object(commit_path, "feature part commit", deadline)
        commit_body = {key: value for key, value in commit.items() if key != "commit_id"}
        if (commit.get("commit_id") != commit_id or fingerprint(commit_body) != commit_id
                or commit.get("feature_id") != feature_id or commit.get("part_id") != part_id
                or commit.get("complete") is not True):
            raise ValueError(f"Feature part commit identity mismatch: {part_id}")
        commit_files = commit.get("files")
        if not isinstance(commit_files, dict) or set(commit_files) != set(PART_FILES):
            raise ValueError(f"Feature part file checksum schema mismatch: {part_id}")
        for filename in PART_FILES:
            digest = commit_files.get(filename)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(f"Invalid committed file checksum: {part_id}/{filename}")
            relative = f"{part_relative}/{filename}"
            files.append(_record(feature_root, relative, f"features/{relative}", digest, deadline))
        commit_sha = _sha256_file(commit_path, deadline)
        files.append(_record(feature_root, f"{part_relative}/commit.json",
                             f"features/{part_relative}/commit.json", commit_sha, deadline))
    destinations = [row["destination"] for row in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Feature file inventory contains duplicate destinations.")
    _check_deadline(deadline)
    return {"feature_id": feature_id, "parts": len(parts)}, files


def _disk_preflight(inventory: list[dict[str, Any]], target_parent: Path) -> int:
    total_bytes = sum(int(row["size"]) for row in inventory)
    largest_file = max((int(row["size"]) for row in inventory), default=0)
    peak_estimate = total_bytes + largest_file + VERIFY_TEMP_HEADROOM_BYTES
    if peak_estimate > MAX_STAGE_BYTES:
        raise OSError(
            f"Staged inputs need an estimated {peak_estimate} bytes, above the {MAX_STAGE_BYTES}-byte cap."
        )
    free_bytes = shutil.disk_usage(target_parent).free
    if free_bytes < peak_estimate:
        raise OSError(f"Insufficient local disk: need about {peak_estimate} bytes, have {free_bytes}.")
    return total_bytes


def _copy_inventory(inventory: list[dict[str, Any]], stage: Path, deadline: float) -> None:
    for row in inventory:
        _check_deadline(deadline)
        destination = stage / row["destination"]
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Unexpected existing staged path: {destination}")
        verified_copy(row["source"], destination, row["sha256"])
        _check_deadline(deadline)


def _hardlink_verification_mirror(stage_features: Path, inventory: list[dict[str, Any]], mirror: Path,
                                  deadline: float) -> Path:
    if mirror.exists() or mirror.is_symlink():
        raise FileExistsError(f"Verification mirror path already exists: {mirror}")
    mirror_root = mirror / "features"
    mirror_root.mkdir(parents=True)
    for row in inventory:
        _check_deadline(deadline)
        relative = row["destination"].removeprefix("features/")
        source = stage_features.joinpath(*PurePosixPath(relative).parts)
        destination = mirror_root.joinpath(*PurePosixPath(relative).parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Staged feature file is missing or symlinked: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError("Local staged inputs must support hard links for read-only verification.") from exc
    _check_deadline(deadline)
    return mirror_root


def _verify_bundle_with_mirror(bundle: Path, features_root: Path, inventory: list[dict[str, Any]],
                               mirror: Path, deadline: float) -> dict[str, Any]:
    mirror_features = _hardlink_verification_mirror(features_root, inventory, mirror, deadline)
    try:
        return verify_pretrain_bundle(bundle, mirror_features, deadline=deadline)
    finally:
        shutil.rmtree(mirror, ignore_errors=True)


def _stage_manifest_body(bundle_id: str, feature_id: str, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "feature_id": feature_id,
        "files": [
            {"path": row["destination"], "sha256": row["sha256"], "size": int(row["size"])}
            for row in sorted(inventory, key=lambda item: item["destination"])
        ],
    }


def _is_verified_audit_output(path: Path, relative: str, feature_id: str, deadline: float) -> bool:
    parts = PurePosixPath(relative).parts
    if len(parts) == 3 and parts == ("features", feature_id, "dataset_audit.json"):
        report = _json_object(path, "generated feature audit", deadline)
        if report.get("feature_id") != feature_id:
            raise ValueError("Generated feature audit belongs to a different feature release.")
        return True
    if len(parts) == 4 and parts[:3] == ("features", feature_id, "audits"):
        if not parts[3].endswith(".json") or not _SHA256.fullmatch(parts[3][:-5]):
            return False
        report = _json_object(path, "generated feature audit", deadline)
        if fingerprint(report) != parts[3][:-5]:
            raise ValueError(f"Generated feature audit fingerprint mismatch: {relative}")
        return True
    if len(parts) == 4 and parts[:3] == ("features", feature_id, "duplicate_groups"):
        if not parts[3].endswith(".jsonl") or not _SHA256.fullmatch(parts[3][:-6]):
            return False
        if _sha256_file(path, deadline) != parts[3][:-6]:
            raise ValueError(f"Generated duplicate-group checksum mismatch: {relative}")
        return True
    return False


def _verify_stage_files(stage_root: Path, expected_body: dict[str, Any], deadline: float) -> None:
    manifest_path = stage_root / STAGE_MANIFEST
    marker = _json_object(manifest_path, "stage manifest", deadline)
    marker_id = marker.get("stage_id")
    body = {key: value for key, value in marker.items() if key != "stage_id"}
    if marker_id != fingerprint(body) or body != expected_body:
        raise ValueError("Existing staged inputs belong to a different immutable source inventory.")
    expected_files = {row["path"]: row for row in expected_body["files"]}
    found_files: set[str] = set()
    feature_id = str(expected_body["feature_id"])
    for path in stage_root.rglob("*"):
        _check_deadline(deadline)
        relative = path.relative_to(stage_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symlink found in staged input tree: {relative}")
        if path.is_file() and relative != STAGE_MANIFEST:
            if relative in expected_files:
                found_files.add(relative)
            elif not _is_verified_audit_output(path, relative, feature_id, deadline):
                raise ValueError(f"Unexpected file in staged input tree: {relative}")
    if found_files != set(expected_files):
        raise ValueError("Existing staged input file set differs from its immutable manifest.")
    for relative, row in expected_files.items():
        _check_deadline(deadline)
        path = stage_root.joinpath(*PurePosixPath(relative).parts)
        if path.stat().st_size != row["size"] or _sha256_file(path, deadline) != row["sha256"]:
            raise ValueError(f"Existing staged input checksum mismatch: {relative}")


def _write_stage_manifest(stage_root: Path, body: dict[str, Any]) -> None:
    marker = {**body, "stage_id": fingerprint(body)}
    path = stage_root / STAGE_MANIFEST
    path.write_text(json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _ensure_no_source_nesting(target_root: Path, bundle_root: Path, feature_root: Path) -> None:
    if (target_root == bundle_root or target_root.is_relative_to(bundle_root)
            or target_root == feature_root or target_root.is_relative_to(feature_root)):
        raise ValueError("target_root must be separate from the immutable source bundle and feature root.")


def stage_training_inputs(
    bundle: Path,
    feature_root: Path,
    target_root: Path,
    deadline: float,
) -> dict[str, Any]:
    """Stage a verified bundle and its committed feature parts onto local SSD.

    Source files are never written. Verification audit output is redirected to
    ephemeral hard-link mirrors, and the published stage is immutable by bundle
    ID. A prior stage is reused only when its full file set and hashes match.
    """
    _check_deadline(deadline)
    source_bundle = Path(bundle).expanduser().resolve(strict=True)
    source_features = Path(feature_root).expanduser().resolve(strict=True)
    destination_root = Path(target_root).expanduser().absolute()
    if destination_root.is_symlink():
        raise ValueError("target_root must not be a symlink.")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root = destination_root.resolve(strict=True)
    _ensure_no_source_nesting(destination_root, source_bundle, source_features)

    bundle_manifest, bundle_inventory = _inventory_bundle(source_bundle, deadline)
    feature_summary, feature_inventory = _inventory_features(source_features, bundle_manifest, deadline)
    inventory = bundle_inventory + feature_inventory
    total_bytes = sum(int(row["size"]) for row in inventory)
    bundle_id = str(bundle_manifest["bundle_id"])
    feature_id = str(bundle_manifest["feature_id"])
    stage_root = destination_root / bundle_id
    body = _stage_manifest_body(bundle_id, feature_id, inventory)

    if stage_root.exists() or stage_root.is_symlink():
        if stage_root.is_symlink() or not stage_root.is_dir():
            raise ValueError("Existing stage path is not a regular directory.")
        _verify_stage_files(stage_root, body, deadline)
        _verify_bundle_with_mirror(source_bundle, stage_root / "features", feature_inventory,
                                   stage_root / ".verify-source", deadline)
        staged_bundle = stage_root / "bundle"
        _verify_bundle_with_mirror(staged_bundle, stage_root / "features", feature_inventory,
                                   stage_root / ".verify-staged", deadline)
        return {
            "stage_root": stage_root,
            "bundle_path": staged_bundle,
            "feature_root": stage_root / "features",
            "bundle_id": bundle_id,
            "feature_id": feature_summary["feature_id"],
            "staged_bytes": total_bytes,
            "reused": True,
        }

    _disk_preflight(inventory, destination_root)

    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=destination_root))
    try:
        features_dir = temporary / "features"
        bundle_dir = temporary / "bundle"
        for row in feature_inventory:
            _check_deadline(deadline)
            relative = row["destination"].removeprefix("features/")
            destination_row = {**row, "destination": f"features/{relative}"}
            _copy_inventory([destination_row], temporary, deadline)

        _verify_bundle_with_mirror(source_bundle, features_dir, feature_inventory,
                                   temporary / ".verify-source", deadline)

        for row in bundle_inventory:
            _check_deadline(deadline)
            _copy_inventory([row], temporary, deadline)

        _verify_bundle_with_mirror(bundle_dir, features_dir, feature_inventory,
                                   temporary / ".verify-staged", deadline)
        _check_deadline(deadline)
        _write_stage_manifest(temporary, body)
        _verify_stage_files(temporary, body, deadline)
        try:
            os.rename(temporary, stage_root)
        except FileExistsError:
            # A concurrent worker may have published the same complete stage.
            _verify_stage_files(stage_root, body, deadline)
            shutil.rmtree(temporary, ignore_errors=True)
            _verify_bundle_with_mirror(source_bundle, stage_root / "features", feature_inventory,
                                       stage_root / ".verify-source", deadline)
            _verify_bundle_with_mirror(stage_root / "bundle", stage_root / "features", feature_inventory,
                                       stage_root / ".verify-staged", deadline)
            return {
                "stage_root": stage_root,
                "bundle_path": stage_root / "bundle",
                "feature_root": stage_root / "features",
                "bundle_id": bundle_id,
                "feature_id": feature_summary["feature_id"],
                "staged_bytes": total_bytes,
                "reused": True,
            }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "stage_root": stage_root,
        "bundle_path": stage_root / "bundle",
        "feature_root": stage_root / "features",
        "bundle_id": bundle_id,
        "feature_id": feature_summary["feature_id"],
        "staged_bytes": total_bytes,
        "reused": False,
    }
