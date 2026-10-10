"""CPU-only orchestration for pre-MIL governance and bundle readiness.

This runner assumes Drive is already mounted. It never allocates a VM, mounts
Drive, constructs an encoder, trains a model, or shuts down a runtime.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from histology_data.features import BudgetExhausted  # noqa: E402
from histology_data.governance import (  # noqa: E402
    CASE_REVIEW_FIELDS,
    build_governance,
    create_case_review_draft,
    load_governance,
    write_case_review_draft,
)
from histology_data.io import atomic_json, file_hash  # noqa: E402
from histology_data.readiness import (  # noqa: E402
    DUPLICATE_REVIEW_FIELDS,
    build_pretrain_bundle,
    verify_pretrain_bundle,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNNER_LOCK_NAME = "runner.lock"
_FEATURE_LOCK_NAME = "extractor.lock"
_WAITABLE_FEATURE_STATUSES = {
    "running",
    "auditing",
    "awaiting_sources",
    "budget_exhausted",
    "part_limit_reached",
}
_ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "drive_root",
    "metadata",
    "source_root",
    "source_kind",
    "feature_root",
    "case_review",
    "user_confirmation",
    "duplicate_dispositions",
    "cohorts",
    "expected_sources",
    "expected_vectors",
    "outer_folds",
    "inner_folds",
    "seed",
    "allow_training",
    "source_pixels",
    "encoder_preprocessing",
    "runtime_policy",
}


class RunnerError(RuntimeError):
    """Invalid runner configuration or inconsistent persistent state."""


class AttentionRequired(RunnerError):
    """A human or a later feature-extraction run must resolve a readiness gate."""

    def __init__(self, blockers: Sequence[str], **details: Any) -> None:
        self.blockers = sorted({str(value) for value in blockers if str(value)})
        self.details = details
        super().__init__("; ".join(self.blockers) or "Pretrain readiness needs attention.")


@dataclass(frozen=True)
class RunnerConfig:
    """Validated, absolute paths and immutable run settings."""

    config_path: Path
    config_sha256: str
    drive_root: Path
    metadata: Path
    source_root: Path
    source_kind: str
    feature_root: Path
    case_review: Path
    user_confirmation: Path
    duplicate_dispositions: Path | None
    cohorts: tuple[str, ...]
    expected_sources: int
    expected_vectors: int
    outer_folds: int
    inner_folds: int
    seed: int

    def resolved_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "drive_root": str(self.drive_root),
            "metadata": str(self.metadata),
            "source_root": str(self.source_root),
            "source_kind": self.source_kind,
            "feature_root": str(self.feature_root),
            "case_review": str(self.case_review),
            "user_confirmation": str(self.user_confirmation),
            "duplicate_dispositions": str(self.duplicate_dispositions) if self.duplicate_dispositions else None,
            "cohorts": list(self.cohorts),
            "expected_sources": self.expected_sources,
            "expected_vectors": self.expected_vectors,
            "outer_folds": self.outer_folds,
            "inner_folds": self.inner_folds,
            "seed": self.seed,
            "allow_training": False,
            "runtime_policy": "CPU only; no VM allocation, encoder invocation, training, or shutdown.",
        }


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_text(value: dt.datetime | None = None) -> str:
    moment = value or _utc_now()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("UTC timestamps must be timezone-aware.")
    return moment.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Could not read JSON object at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"Expected a JSON object at {path}.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunnerError(f"{name} must be a positive integer.")
    return value


def _resolve_drive_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError(f"Config field {name!r} must be a nonempty path string.")
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RunnerError(f"Config path {name!r} escapes drive_root: {resolved}") from exc
    return resolved


def load_config(config_path: Path) -> RunnerConfig:
    """Load a root-relative JSON config and reject unsafe or training settings."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=True)
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RunnerError("Config root must be a JSON object.")
    unknown = set(raw) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise RunnerError(f"Unknown config fields: {sorted(unknown)}")
    if raw.get("schema_version") != 1:
        raise RunnerError("Only pretrain config schema_version=1 is supported.")
    drive_text = raw.get("drive_root")
    if not isinstance(drive_text, str) or not drive_text.strip():
        raise RunnerError("drive_root must be an absolute path supplied by the operator.")
    drive_root = Path(drive_text).expanduser().resolve(strict=False)
    if not drive_root.is_absolute() or not drive_root.is_dir():
        raise RunnerError(f"Drive root is unavailable; mount it before starting this runner: {drive_root}")
    paths = {
        name: _resolve_drive_path(drive_root, raw.get(name), name)
        for name in ("metadata", "source_root", "feature_root", "case_review", "user_confirmation")
    }
    duplicate_value = raw.get("duplicate_dispositions")
    duplicate_path = (
        _resolve_drive_path(drive_root, duplicate_value, "duplicate_dispositions")
        if duplicate_value not in (None, "") else None
    )
    source_kind = raw.get("source_kind")
    if source_kind != "zip":
        raise RunnerError("This pretrain runner accepts only source_kind='zip'.")
    if not paths["metadata"].is_file():
        raise RunnerError(f"Metadata file is missing: {paths['metadata']}")
    if not paths["source_root"].is_dir():
        raise RunnerError(f"ZIP source directory is missing: {paths['source_root']}")
    cohorts_raw = raw.get("cohorts")
    if (not isinstance(cohorts_raw, list) or len(cohorts_raw) != 2
            or set(cohorts_raw) != {"common", "all"}):
        raise RunnerError("cohorts must contain exactly 'common' and 'all'.")
    allow_training = raw.get("allow_training")
    if allow_training is not False:
        raise RunnerError("This runner is preparation-only; allow_training must be false.")
    outer_folds = _positive_int(raw.get("outer_folds"), "outer_folds")
    inner_folds = _positive_int(raw.get("inner_folds"), "inner_folds")
    if (outer_folds, inner_folds) != (3, 2):
        raise RunnerError("The readiness builder currently supports the locked 3 outer × 2 inner protocol.")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RunnerError("seed must be a nonnegative integer.")
    return RunnerConfig(
        config_path=path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        drive_root=drive_root,
        metadata=paths["metadata"],
        source_root=paths["source_root"],
        source_kind=source_kind,
        feature_root=paths["feature_root"],
        case_review=paths["case_review"],
        user_confirmation=paths["user_confirmation"],
        duplicate_dispositions=duplicate_path,
        cohorts=tuple(cohorts_raw),
        expected_sources=_positive_int(raw.get("expected_sources"), "expected_sources"),
        expected_vectors=_positive_int(raw.get("expected_vectors"), "expected_vectors"),
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )


class RunState:
    """Atomic status and append-only event records for one orchestration run."""

    def __init__(self, run_dir: Path, run_id: str, mode: str, config: RunnerConfig) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status_path = self.run_dir / "status.json"
        self.events_path = self.run_dir / "events.jsonl"
        atomic_json(self.run_dir / "resolved_config.json", config.resolved_json())
        self.status: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "mode": mode,
            "status": "running",
            "started_at": _utc_text(),
            "updated_at": _utc_text(),
            "training_started": False,
            "allow_training": False,
            "config_sha256": config.config_sha256,
        }
        self.save()
        self.event("run_started", mode=mode)

    def save(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = _utc_text()
        atomic_json(self.status_path, self.status)

    def event(self, name: str, **values: Any) -> None:
        row = {"event": name, "run_id": self.status["run_id"], "timestamp": _utc_text(), **values}
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class RunnerLock:
    """Exclusive orchestration lock; stale recovery is explicit and audited."""

    def __init__(self, path: Path, run_id: str, recover_stale: bool) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.recover_stale = recover_stale
        self.recovered: dict[str, Any] | None = None
        self.acquired = False

    @staticmethod
    def _known_live(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            raise RunnerError(f"Cannot establish whether runner PID {pid} is alive: {exc}") from exc
        return True

    def _recover_if_requested(self) -> None:
        if not self.path.exists() and not self.path.is_symlink():
            return
        if not self.recover_stale:
            raise RunnerError(f"Pretrain runner lock exists: {self.path}; use explicit recovery only after stop proof.")
        if self.path.is_symlink() or not self.path.is_file():
            raise RunnerError("Refusing to recover a symlink or non-file runner lock.")
        raw = self.path.read_bytes()
        try:
            previous = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerError("Runner lock is unreadable; refusing stale-lock recovery.") from exc
        if not isinstance(previous, dict):
            raise RunnerError("Runner lock schema is invalid; refusing stale-lock recovery.")
        host = previous.get("hostname")
        pid = previous.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 or not isinstance(host, str) or not host:
            raise RunnerError("Runner lock lacks a valid host/PID; refusing stale-lock recovery.")
        if host == socket.gethostname() and self._known_live(pid):
            raise RunnerError(f"Runner lock belongs to a live process (host={host}, pid={pid}); refusing recovery.")
        # Re-read before unlinking so a changed marker is never silently replaced.
        if self.path.read_bytes() != raw:
            raise RunnerError("Runner lock changed during recovery check; retry after inspecting the active run.")
        self.path.unlink()
        self.recovered = {
            "previous_run_id": previous.get("run_id"),
            "previous_hostname": host,
            "previous_pid": pid,
            "operator_recovery_requested": True,
        }

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover_if_requested()
        descriptor = {
            "schema_version": 1,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _utc_text(),
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RunnerError(f"Another runner acquired the lock: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(descriptor, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = _json_object(self.path)
        except RunnerError:
            return
        if current.get("run_id") == self.run_id:
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> RunnerLock:
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


class PretrainWriterLock:
    """Share the CLI's `.pretrain.lock` so direct API calls cannot race its writers."""

    def __init__(self, output_root: Path) -> None:
        self.path = Path(output_root) / ".pretrain.lock"
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": self.token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _utc_text(),
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise AttentionRequired(["pretrain_writer_lock_active"], lock_path=str(self.path)) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = _json_object(self.path)
        except RunnerError:
            return
        if current.get("token") == self.token:
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> PretrainWriterLock:
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def _new_run_id() -> str:
    return "pretrain-" + _utc_now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]


def _timestamp(value: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError("--deadline-utc must be an aware ISO-8601 timestamp.")
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise RunnerError(f"Invalid --deadline-utc timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunnerError("--deadline-utc must include a timezone offset.")
    return parsed.astimezone(dt.timezone.utc)


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RunnerError(f"CSV schema mismatch at {path}; expected columns {expected_fields}.")
        rows = list(reader)
    if not rows:
        raise RunnerError(f"CSV is empty: {path}")
    return rows


def _csv_bytes(fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    output = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return output.read().encode("utf-8")


def _atomic_publish_immutable(path: Path, payload: bytes) -> str:
    """Atomically publish a file, accepting identical bytes and rejecting changed ones."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.is_symlink():
        raise RunnerError(f"Refusing to publish through a symlink: {path}")
    if path.exists():
        if not path.is_file() or file_hash(path) != digest:
            raise AttentionRequired(["case_review_conflict"], case_review_path=str(path))
        return digest
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            if path.is_file() and not path.is_symlink() and file_hash(path) == digest:
                return digest
            raise AttentionRequired(["case_review_conflict"], case_review_path=str(path))
        os.replace(temporary, path)
        if file_hash(path) != digest:
            raise RunnerError(f"Published case review checksum mismatch: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _confirm_case_review(
    template_path: Path,
    confirmation_path: Path,
    output_path: Path,
    *,
    metadata_sha256: str,
    expected_case_ids: Sequence[str],
    drive_root: Path,
) -> dict[str, Any]:
    """Apply only explicit user-provided mappings to the exact reviewed template."""
    if not confirmation_path.is_file():
        raise AttentionRequired(["user_confirmation_missing"], path=str(confirmation_path))
    confirmation = _json_object(confirmation_path)
    required = {
        "schema_version", "confirmed_at_utc", "scope", "identity_confirmation_quote",
        "identity_semantics", "label_confirmation_quote", "binary_mapping",
        "mixed_benign_malignant_case_policy", "raw_glade_policy", "case_scope_template_sha256",
        "cases", "patient_pseudonym_policy", "approved_cohort_plan", "metadata_sha256",
        "case_labels", "patient_map",
    }
    missing = sorted(required - set(confirmation))
    if missing:
        raise AttentionRequired(["user_confirmation_schema_incomplete"], missing_fields=missing)
    if confirmation.get("schema_version") != 1:
        raise AttentionRequired(["user_confirmation_schema_unsupported"])
    confirmed_at = _timestamp(str(confirmation.get("confirmed_at_utc", "")))
    for field in (
        "scope", "identity_confirmation_quote", "identity_semantics", "label_confirmation_quote",
        "mixed_benign_malignant_case_policy", "raw_glade_policy", "patient_pseudonym_policy",
    ):
        if not isinstance(confirmation.get(field), str) or not confirmation[field].strip():
            raise AttentionRequired([f"user_confirmation_{field}_missing"])
    if confirmation.get("metadata_sha256") != metadata_sha256:
        raise AttentionRequired(["user_confirmation_metadata_sha256_mismatch"])
    template_sha256 = file_hash(template_path)
    if confirmation.get("case_scope_template_sha256") != template_sha256:
        raise AttentionRequired(["user_confirmation_case_scope_template_mismatch"],
                                expected_template_sha256=template_sha256)
    template_rows = _read_csv(template_path, CASE_REVIEW_FIELDS)
    template_ids = [row["case_id"] for row in template_rows]
    confirmation_cases = confirmation.get("cases")
    if (not isinstance(confirmation_cases, list)
            or any(not isinstance(case_id, str) or not case_id.strip() for case_id in confirmation_cases)
            or len(confirmation_cases) != len(set(confirmation_cases))):
        raise AttentionRequired(["user_confirmation_case_scope_invalid"])
    expected_ids = [str(case_id) for case_id in expected_case_ids]
    if len(expected_ids) != len(set(expected_ids)) or set(template_ids) != set(expected_ids):
        raise AttentionRequired(["draft_case_scope_inconsistent"])
    if set(confirmation_cases) != set(expected_ids):
        raise AttentionRequired(["user_confirmation_case_scope_mismatch"],
                                expected_case_count=len(expected_ids),
                                confirmed_case_count=len(confirmation_cases))
    case_labels = confirmation.get("case_labels")
    patient_map = confirmation.get("patient_map")
    if not isinstance(case_labels, dict) or not isinstance(patient_map, dict):
        raise AttentionRequired(["user_confirmation_case_maps_invalid"])
    if set(case_labels) != set(expected_ids) or set(patient_map) != set(expected_ids):
        raise AttentionRequired(["user_confirmation_case_map_scope_mismatch"])
    normalized_labels: dict[str, int] = {}
    for case_id, label in case_labels.items():
        if isinstance(label, bool) or label not in (0, 1, "0", "1"):
            raise AttentionRequired(["user_confirmation_label_not_binary"])
        normalized_labels[case_id] = int(label)
    if set(normalized_labels.values()) != {0, 1}:
        raise AttentionRequired(["user_confirmation_binary_classes_incomplete"])
    normalized_patients: dict[str, str] = {}
    for case_id, patient_id in patient_map.items():
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise AttentionRequired(["user_confirmation_patient_map_incomplete"])
        normalized_patients[case_id] = patient_id.strip()
    semantics = " ".join(confirmation["identity_semantics"].casefold().split())
    asserts_distinct = ("distinct patient per released case" in semantics
                        or "no shared patients between" in semantics)
    if asserts_distinct and len(set(normalized_patients.values())) != len(expected_ids):
        raise AttentionRequired(["user_confirmation_identity_semantics_conflict"])
    binary_mapping = confirmation.get("binary_mapping")
    if not isinstance(binary_mapping, dict) or not binary_mapping:
        raise AttentionRequired(["user_confirmation_binary_mapping_missing"])
    mapped_labels = set()
    for value in binary_mapping.values():
        if isinstance(value, bool) or value not in (0, 1, "0", "1"):
            raise AttentionRequired(["user_confirmation_binary_mapping_invalid"])
        mapped_labels.add(int(value))
    if mapped_labels != {0, 1}:
        raise AttentionRequired(["user_confirmation_binary_mapping_incomplete"])
    confirmation_sha256 = file_hash(confirmation_path)
    try:
        confirmation_relative = confirmation_path.resolve().relative_to(drive_root.resolve()).as_posix()
    except ValueError:
        confirmation_relative = confirmation_path.name
    identity_evidence = (
        f"{confirmation_relative}#sha256={confirmation_sha256}; "
        f"confirmed_at_utc={_utc_text(confirmed_at)}; "
        f"quote={confirmation['identity_confirmation_quote'].strip()}; "
        f"semantics={confirmation['identity_semantics'].strip()}; "
        f"pseudonym_policy={confirmation['patient_pseudonym_policy'].strip()}"
    )
    label_evidence = (
        f"{confirmation_relative}#sha256={confirmation_sha256}; "
        f"confirmed_at_utc={_utc_text(confirmed_at)}; "
        f"quote={confirmation['label_confirmation_quote'].strip()}; "
        f"binary_mapping={json.dumps(binary_mapping, ensure_ascii=False, sort_keys=True)}; "
        f"mixed_case_policy={confirmation['mixed_benign_malignant_case_policy'].strip()}; "
        f"raw_glade_policy={confirmation['raw_glade_policy'].strip()}"
    )
    rows: list[dict[str, Any]] = []
    for template in template_rows:
        case_id = template["case_id"]
        row = dict(template)
        # Keep every metadata provenance cell byte-for-byte as emitted in the draft.
        row.update(
            patient_id=normalized_patients[case_id],
            identity_verified="true",
            identity_evidence=identity_evidence,
            case_label=str(normalized_labels[case_id]),
            label_verified="true",
            label_evidence=label_evidence,
        )
        rows.append(row)
    payload = _csv_bytes(CASE_REVIEW_FIELDS, rows)
    digest = _atomic_publish_immutable(output_path, payload)
    return {
        "case_review_sha256": digest,
        "confirmation_sha256": confirmation_sha256,
        "case_count": len(rows),
        "patient_count": len(set(normalized_patients.values())),
        "label_counts": {str(label): sum(value == label for value in normalized_labels.values())
                         for label in (0, 1)},
    }


def _create_case_review_draft(
    config: RunnerConfig,
    state: RunState,
    *,
    require_complete: bool,
) -> tuple[dict[str, Any], Path]:
    state.save(status="inventorying_source")
    state.event("case_review_draft_started")
    draft = create_case_review_draft(
        config.metadata,
        config.source_root,
        source_kind=config.source_kind,
        expected_sources=config.expected_sources,
        expected_patches=config.expected_vectors,
    )
    with PretrainWriterLock(config.drive_root):
        draft_dir = write_case_review_draft(draft, config.drive_root)
    template_path = draft_dir / "case_review.csv"
    state.event(
        "case_review_draft_written",
        draft_id=draft["draft_id"],
        case_count=draft["case_count"],
        observed_sources=draft["observed_sources"],
        observed_vectors=draft["observed_patches"],
        source_complete=bool(draft["source_complete"]),
        draft_path=str(draft_dir),
    )
    state.save(
        draft_id=draft["draft_id"],
        draft_path=str(draft_dir),
        draft_case_count=draft["case_count"],
        observed_sources=draft["observed_sources"],
        observed_vectors=draft["observed_patches"],
        source_complete=bool(draft["source_complete"]),
    )
    incomplete = (draft.get("observed_sources") != config.expected_sources
                  or draft.get("observed_patches") != config.expected_vectors
                  or not draft.get("source_complete") or draft.get("source_errors"))
    if require_complete and incomplete:
        raise AttentionRequired(["source_inventory_incomplete"],
                                observed_sources=draft.get("observed_sources"),
                                expected_sources=config.expected_sources,
                                observed_vectors=draft.get("observed_patches"),
                                expected_vectors=config.expected_vectors,
                                source_errors=draft.get("source_errors", []))
    return draft, template_path


def _create_review_and_governance(
    config: RunnerConfig,
    state: RunState,
    *,
    draft: dict[str, Any] | None = None,
    template_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if draft is None or template_path is None:
        draft, template_path = _create_case_review_draft(config, state, require_complete=True)
    state.save(status="applying_user_confirmation")
    with PretrainWriterLock(config.drive_root):
        review = _confirm_case_review(
            template_path,
            config.user_confirmation,
            config.case_review,
            metadata_sha256=file_hash(config.metadata),
            expected_case_ids=draft["candidate_case_ids"],
            drive_root=config.drive_root,
        )
        state.event("user_confirmation_applied", **review)
        state.save(
            case_review_path=str(config.case_review),
            case_review_sha256=review["case_review_sha256"],
            confirmation_sha256=review["confirmation_sha256"],
            governed_case_count=review["case_count"],
            governed_patient_count=review["patient_count"],
            label_counts=review["label_counts"],
            status="building_governance",
        )
        governance = build_governance(config.metadata, config.case_review, config.drive_root)
        body, rows = load_governance(Path(governance["path"]))
    if (not body.get("governance_complete") or len(rows) != review["case_count"]
            or body.get("metadata_sha256") != file_hash(config.metadata)
            or body.get("case_review_sha256") != file_hash(config.case_review)):
        raise AttentionRequired(["governance_artifact_verification_failed"])
    expected_labels = review["label_counts"]
    actual_labels = {str(key): value for key, value in body.get("case_labels", {}).items()}
    if actual_labels != expected_labels or body.get("unique_patient_count") != review["patient_count"]:
        raise AttentionRequired(["governance_counts_differ_from_confirmation"])
    state.event("governance_ready", governance_id=body["governance_id"], case_count=len(rows))
    state.save(governance_id=body["governance_id"], governance_path=str(governance["path"]))
    return draft, governance


def _feature_snapshot(feature_root: Path, expected_sources: int, expected_vectors: int) -> dict[str, Any]:
    """Read only status/release markers; the builder performs checksum/global audit."""
    root = Path(feature_root)
    status_path = root / "status.json"
    lock_path = root / _FEATURE_LOCK_NAME
    lock_present = lock_path.exists() or lock_path.is_symlink()
    if not status_path.is_file():
        return {"ready": False, "reason": "feature_status_missing", "lock_present": lock_present}
    try:
        status = _json_object(status_path)
    except RunnerError as exc:
        return {"ready": False, "reason": "feature_status_unreadable", "detail": str(exc),
                "lock_present": lock_present}
    status_name = status.get("status")
    if status_name in {"error", "interrupted"}:
        return {"ready": False, "terminal_error": status_name, "reason": f"feature_run_{status_name}",
                "status": status, "lock_present": lock_present}
    if status_name != "complete" or lock_present:
        return {"ready": False, "reason": "feature_writer_active_or_incomplete", "status": status,
                "lock_present": lock_present}
    feature_id = status.get("feature_id")
    if not isinstance(feature_id, str) or not _SHA256.fullmatch(feature_id):
        return {"ready": False, "reason": "feature_id_missing_or_invalid", "status": status,
                "lock_present": lock_present}
    release_path = root / feature_id / "release.json"
    try:
        release = _json_object(release_path)
    except RunnerError as exc:
        return {"ready": False, "reason": "feature_release_missing_or_unreadable", "detail": str(exc),
                "status": status, "feature_id": feature_id, "lock_present": lock_present}
    expected_status = (
        status.get("feature_complete") is True
        and status.get("source_complete") is True
        and status.get("expected_sources") == expected_sources
        and status.get("available_parts") == expected_sources
        and status.get("observed_pngs") == expected_vectors
        and status.get("committed_vectors") == expected_vectors
    )
    expected_release = (
        release.get("feature_id") == feature_id
        and release.get("source_complete") is True
        and release.get("feature_complete") is True
        and release.get("audit_complete") is True
        and release.get("observed_sources") == expected_sources
        and release.get("expected_sources") == expected_sources
        and release.get("observed_pngs") == expected_vectors
        and release.get("expected_pngs") == expected_vectors
        and release.get("committed_vectors") == expected_vectors
    )
    if not (expected_status and expected_release):
        return {"ready": False, "reason": "feature_completion_contract_mismatch", "status": status,
                "release": release, "feature_id": feature_id, "lock_present": lock_present}
    return {"ready": True, "status": status, "release": release,
            "feature_id": feature_id, "lock_present": False}


def _require_feature_ready_now(config: RunnerConfig) -> dict[str, Any]:
    snapshot = _feature_snapshot(config.feature_root, config.expected_sources, config.expected_vectors)
    if snapshot.get("terminal_error"):
        raise AttentionRequired([snapshot["reason"]], feature_status=snapshot.get("status"))
    if not snapshot.get("ready"):
        reason = snapshot.get("reason", "feature_release_incomplete")
        raise AttentionRequired([reason], lock_present=snapshot.get("lock_present", False))
    return snapshot


def _wait_for_feature_release(
    config: RunnerConfig,
    state: RunState,
    deadline: dt.datetime,
    monotonic_deadline: float,
    *,
    poll_seconds: int,
    now_fn: Callable[[], dt.datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for an audited production release and a vanished extractor lock."""
    if not 30 <= poll_seconds <= 60:
        raise RunnerError("poll_seconds must be between 30 and 60 seconds.")
    last_signature: tuple[Any, ...] | None = None
    while True:
        if (now_fn().astimezone(dt.timezone.utc) >= deadline
                or time.monotonic() >= monotonic_deadline):
            raise AttentionRequired(["feature_wait_deadline_reached"])
        snapshot = _feature_snapshot(config.feature_root, config.expected_sources, config.expected_vectors)
        status = snapshot.get("status", {})
        feature_status = status.get("status") if isinstance(status, dict) else None
        if snapshot.get("ready"):
            state.event("feature_release_ready", feature_id=snapshot["feature_id"])
            return snapshot
        if snapshot.get("terminal_error") and not snapshot.get("lock_present"):
            raise AttentionRequired([snapshot["reason"]], feature_status=feature_status)
        if feature_status == "complete" and not snapshot.get("lock_present"):
            raise AttentionRequired([snapshot.get("reason", "feature_completion_contract_mismatch")])
        if snapshot.get("reason") in {"feature_status_missing", "feature_status_unreadable"} \
                and not snapshot.get("lock_present"):
            raise AttentionRequired([snapshot["reason"]], detail=snapshot.get("detail"))
        if feature_status not in _WAITABLE_FEATURE_STATUSES and feature_status not in {None, "complete"}:
            if not snapshot.get("lock_present"):
                raise AttentionRequired(["feature_status_unexpected"], feature_status=feature_status)
        now = now_fn().astimezone(dt.timezone.utc)
        remaining = min((deadline - now).total_seconds(), monotonic_deadline - time.monotonic())
        signature = (feature_status, snapshot.get("reason"), snapshot.get("lock_present"),
                     status.get("observed_sources") if isinstance(status, dict) else None,
                     status.get("observed_pngs") if isinstance(status, dict) else None)
        if signature != last_signature:
            state.save(
                status="waiting_for_feature",
                feature_status=feature_status,
                feature_wait_reason=snapshot.get("reason"),
                feature_lock_present=snapshot.get("lock_present", False),
                observed_sources=signature[3],
                observed_vectors=signature[4],
                profile_deadline_utc=_utc_text(deadline),
            )
            state.event("waiting_for_feature", **{
                "feature_status": feature_status,
                "reason": snapshot.get("reason"),
                "lock_present": snapshot.get("lock_present", False),
                "observed_sources": signature[3],
                "observed_vectors": signature[4],
            })
            last_signature = signature
        if remaining <= 0:
            raise AttentionRequired(["feature_wait_deadline_reached"],
                                    last_feature_status=feature_status,
                                    last_reason=snapshot.get("reason"))
        sleep_fn(min(float(poll_seconds), remaining))


def _write_duplicate_review_template(feature_root: Path, feature_id: str, run_dir: Path) -> Path:
    """Persist an explicit blank review row for every audited exact-content group."""
    release_root = Path(feature_root) / feature_id
    audit_path = release_root / "dataset_audit.json"
    audit = _json_object(audit_path)
    if audit.get("feature_id") != feature_id:
        raise RunnerError("Dataset audit feature_id differs from the selected release.")
    relative = Path(str(audit.get("duplicate_group_file", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise RunnerError("Dataset audit does not identify a safe duplicate-group artifact.")
    group_path = release_root / relative
    if not group_path.is_file() or file_hash(group_path) != audit.get("duplicate_group_file_sha256"):
        raise RunnerError("Duplicate-group artifact is missing or fails its checksum.")
    rows = []
    with group_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            group = json.loads(line)
            kind = group.get("kind")
            sha256 = group.get("sha256")
            members = group.get("members")
            if (kind not in {"source_png_sha256", "rgb_pixel_sha256"}
                    or not isinstance(sha256, str) or not _SHA256.fullmatch(sha256)
                    or not isinstance(members, list) or len(members) < 2):
                raise RunnerError(f"Malformed duplicate group at line {line_number}.")
            rows.append({
                "kind": kind,
                "sha256": sha256,
                "keep_tile_ids_json": "",
                "exclude_tile_ids_json": "",
                "canonical_tile_id": "",
                "reviewer": "",
                "reviewed_at": "",
                "evidence": "",
            })
    payload = _csv_bytes(DUPLICATE_REVIEW_FIELDS, rows)
    path = Path(run_dir) / "duplicate_dispositions_template.csv"
    _atomic_publish_immutable(path, payload)
    if not rows:
        raise RunnerError("Duplicate review was requested, but the audited group file is empty.")
    return path


def _is_duplicate_review_blocker(blockers: Sequence[str]) -> bool:
    return any("duplicate" in blocker.casefold()
               and any(term in blocker.casefold() for term in ("review", "disposition", "exact"))
               for blocker in blockers)


def _ensure_before_deadline(
    deadline: dt.datetime | None,
    monotonic_deadline: float | None,
    now_fn: Callable[[], dt.datetime],
    phase: str,
) -> None:
    if deadline is not None and now_fn().astimezone(dt.timezone.utc) >= deadline:
        raise AttentionRequired(["profile_deadline_reached"], phase=phase)
    if monotonic_deadline is not None and time.monotonic() >= monotonic_deadline:
        raise AttentionRequired(["profile_deadline_reached"], phase=phase)


def _build_and_verify_bundles_locked(
    config: RunnerConfig,
    governance: dict[str, Any],
    state: RunState,
    run_dir: Path,
    *,
    duplicate_dispositions: Path | None,
    deadline: dt.datetime | None,
    monotonic_deadline: float | None,
    now_fn: Callable[[], dt.datetime],
) -> dict[str, Any]:
    built: dict[str, dict[str, Any]] = {}
    blocked: dict[str, list[str]] = {}
    feature_id: str | None = None
    for cohort in config.cohorts:
        _ensure_before_deadline(deadline, monotonic_deadline, now_fn, f"build:{cohort}")
        _require_feature_ready_now(config)
        state.save(status="building_bundles", current_cohort=cohort)
        state.event("bundle_build_started", cohort=cohort)
        try:
            result = build_pretrain_bundle(
                Path(governance["path"]),
                config.feature_root,
                config.drive_root,
                cohort_mode=cohort,
                duplicate_dispositions=duplicate_dispositions,
                expected_sources=config.expected_sources,
                expected_vectors=config.expected_vectors,
                seed=config.seed,
                deadline=monotonic_deadline,
            )
        except BudgetExhausted as exc:
            raise AttentionRequired(["profile_deadline_reached_during_bundle_build"],
                                    cohort=cohort, detail=str(exc)) from exc
        if result.get("status") != "ready" or result.get("training_ready") is not True:
            blockers = [str(value) for value in result.get("blockers", [])]
            if _is_duplicate_review_blocker(blockers):
                duplicate_template = _write_duplicate_review_template(
                    config.feature_root, str(result.get("feature_id") or _require_feature_ready_now(config)["feature_id"]),
                    run_dir,
                )
                raise AttentionRequired(["duplicate-review-required", *blockers],
                                        duplicate_dispositions_template=str(duplicate_template),
                                        cohort=cohort)
            blocked[cohort] = blockers or ["bundle_builder_returned_blocked"]
            state.event("bundle_build_blocked", cohort=cohort, blockers=blocked[cohort])
            continue
        current_feature_id = result.get("feature_id")
        if feature_id is None:
            feature_id = str(current_feature_id)
        elif current_feature_id != feature_id:
            raise AttentionRequired(["cohort_feature_id_mismatch"])
        built[cohort] = result
        state.event("bundle_built", cohort=cohort, bundle_id=result.get("bundle_id"),
                    case_count=result.get("case_count"), instance_count=result.get("instance_count"))
        state.save(built_bundles={key: value.get("bundle_id") for key, value in built.items()})

    verified: dict[str, dict[str, Any]] = {}
    for cohort in config.cohorts:
        result = built.get(cohort)
        if result is None:
            continue
        _ensure_before_deadline(deadline, monotonic_deadline, now_fn, f"verify:{cohort}")
        _require_feature_ready_now(config)
        bundle_path = Path(str(result["bundle_path"]))
        try:
            verification = verify_pretrain_bundle(
                bundle_path,
                config.feature_root,
                deadline=monotonic_deadline,
            )
        except BudgetExhausted as exc:
            raise AttentionRequired(["profile_deadline_reached_during_bundle_verification"],
                                    cohort=cohort, detail=str(exc)) from exc
        if (verification.get("status") != "ready" or verification.get("training_ready") is not True
                or verification.get("feature_id") != result.get("feature_id")
                or verification.get("governance_id") != governance.get("governance_id")):
            blocked[cohort] = ["bundle_verification_mismatch"]
            state.event("bundle_verification_failed", cohort=cohort, blockers=blocked[cohort])
            continue
        verified[cohort] = verification
        state.event("bundle_verified", cohort=cohort, bundle_id=verification.get("bundle_id"),
                    feature_id=verification.get("feature_id"))
    if blocked or set(verified) != set(config.cohorts):
        raise AttentionRequired(
            [f"bundle_{cohort}_blocked" for cohort in config.cohorts if cohort not in verified],
            cohort_blockers=blocked,
            verified_cohorts=sorted(verified),
        )
    verified_ids = {item.get("feature_id") for item in verified.values()}
    if verified_ids != {feature_id}:
        raise AttentionRequired(["cohort_feature_id_mismatch"])
    return {
        "feature_id": feature_id,
        "governance_id": governance["governance_id"],
        "cohorts": {
            cohort: {
                "bundle_id": verified[cohort]["bundle_id"],
                "bundle_path": str(Path(built[cohort]["bundle_path"])),
                "case_count": verified[cohort]["case_count"],
                "patient_count": verified[cohort]["patient_count"],
                "instance_count": verified[cohort]["instance_count"],
                "training_ready": True,
            }
            for cohort in config.cohorts
        },
    }


def _build_and_verify_bundles(
    config: RunnerConfig,
    governance: dict[str, Any],
    state: RunState,
    run_dir: Path,
    *,
    duplicate_dispositions: Path | None,
    deadline: dt.datetime | None,
    monotonic_deadline: float | None,
    now_fn: Callable[[], dt.datetime],
) -> dict[str, Any]:
    # Match histology_data.pretrain_cli's `.pretrain.lock` so direct public API
    # calls cannot overlap a concurrent CLI build.
    with PretrainWriterLock(config.drive_root):
        return _build_and_verify_bundles_locked(
            config,
            governance,
            state,
            run_dir,
            duplicate_dispositions=duplicate_dispositions,
            deadline=deadline,
            monotonic_deadline=monotonic_deadline,
            now_fn=now_fn,
        )


def run_pretrain(
    config_path: Path,
    *,
    mode: str = "auto",
    deadline_utc: str | None = None,
    poll_seconds: int = 45,
    recover_stale_runner_lock: bool = False,
    duplicate_dispositions: Path | None = None,
    now_fn: Callable[[], dt.datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run draft, build, or unattended pre-MIL data-readiness orchestration."""
    if mode not in {"draft", "auto", "build"}:
        raise RunnerError("mode must be draft, auto, or build.")
    if mode in {"auto", "build"}:
        if deadline_utc is None:
            raise RunnerError("auto and build modes require --deadline-utc from the active profile budget.")
        deadline = _timestamp(deadline_utc)
        if mode == "auto" and not 30 <= poll_seconds <= 60:
            raise RunnerError("poll_seconds must be between 30 and 60 seconds.")
    else:
        deadline = _timestamp(deadline_utc) if deadline_utc else None
    config = load_config(Path(config_path))
    monotonic_deadline: float | None = None
    if deadline is not None:
        remaining_seconds = max(0.0, (deadline - now_fn().astimezone(dt.timezone.utc)).total_seconds())
        monotonic_deadline = time.monotonic() + remaining_seconds
    selected_dispositions = config.duplicate_dispositions
    if duplicate_dispositions is not None:
        candidate = Path(duplicate_dispositions).expanduser()
        selected_dispositions = (candidate if candidate.is_absolute() else config.drive_root / candidate).resolve()
        try:
            selected_dispositions.relative_to(config.drive_root)
        except ValueError as exc:
            raise RunnerError("Duplicate dispositions path must be inside drive_root.") from exc
    if selected_dispositions is not None and not selected_dispositions.is_file():
        raise RunnerError(f"Duplicate disposition file does not exist: {selected_dispositions}")
    run_id = _new_run_id()
    run_root = config.drive_root / "pretrain_runs"
    run_dir = run_root / run_id
    lock = RunnerLock(run_root / _RUNNER_LOCK_NAME, run_id, recover_stale_runner_lock)
    with lock:
        state = RunState(run_dir, run_id, mode, config)
        if deadline is not None:
            state.save(profile_deadline_utc=_utc_text(deadline))
        if lock.recovered:
            state.event("stale_runner_lock_recovered", **lock.recovered)
        try:
            _ensure_before_deadline(deadline, monotonic_deadline, now_fn, "start")
            if mode == "build":
                _require_feature_ready_now(config)
            if mode == "draft":
                draft, draft_path = _create_case_review_draft(config, state, require_complete=False)
                state.save(
                    status="draft_ready",
                    draft_id=draft["draft_id"],
                    draft_path=str(draft_path.parent),
                    source_complete=bool(draft["source_complete"]),
                    finished_at=_utc_text(),
                )
                state.event("draft_run_finished")
                return dict(state.status)
            draft, governance = _create_review_and_governance(config, state)
            if mode == "auto":
                assert deadline is not None
                if deadline <= now_fn().astimezone(dt.timezone.utc):
                    raise AttentionRequired(["profile_deadline_already_reached"])
                _wait_for_feature_release(
                    config, state, deadline, monotonic_deadline or time.monotonic(),
                    poll_seconds=poll_seconds,
                    now_fn=now_fn, sleep_fn=sleep_fn,
                )
            else:
                _require_feature_ready_now(config)
            state.save(status="feature_release_verified_for_bundle_build")
            state.event("feature_release_gate_complete")
            bundle_result = _build_and_verify_bundles(
                config, governance, state, run_dir,
                duplicate_dispositions=selected_dispositions,
                deadline=deadline,
                monotonic_deadline=monotonic_deadline,
                now_fn=now_fn,
            )
            state.save(
                status="data_ready",
                feature_id=bundle_result["feature_id"],
                governance_id=bundle_result["governance_id"],
                cohorts=bundle_result["cohorts"],
                data_ready=True,
                training_started=False,
                finished_at=_utc_text(),
            )
            state.event("data_readiness_complete", feature_id=bundle_result["feature_id"],
                        cohort_ids={key: value["bundle_id"] for key, value in bundle_result["cohorts"].items()})
            return dict(state.status)
        except AttentionRequired as exc:
            state.save(
                status="attention_required",
                attention_required=True,
                data_ready=False,
                training_started=False,
                blockers=exc.blockers,
                details=exc.details,
                finished_at=_utc_text(),
            )
            state.event("attention_required", blockers=exc.blockers, details=exc.details)
            return dict(state.status)
        except KeyboardInterrupt:
            state.save(status="interrupted", interrupted=True, finished_at=_utc_text())
            state.event("run_interrupted")
            raise
        except Exception as exc:
            state.save(
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
                finished_at=_utc_text(),
            )
            state.event("run_failed", error_type=type(exc).__name__, error_message=str(exc))
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and audit pre-MIL governance/bundles on an already mounted Drive; never trains models."
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON config path relative to the project root")
    parser.add_argument("--mode", choices=("draft", "auto", "build"), default="auto")
    parser.add_argument("--deadline-utc",
                        help="Aware ISO-8601 action cutoff, already adjusted for the profile reserve; required for auto/build")
    parser.add_argument("--poll-seconds", type=int, default=45, help="Wait interval, bounded to 30–60 seconds")
    parser.add_argument("--duplicate-dispositions", type=Path,
                        help="Reviewed duplicate dispositions CSV; required when the feature audit finds duplicates")
    parser.add_argument("--recover-stale-runner-lock", action="store_true",
                        help="Recover only after confirming the previous runner stopped; live same-host PIDs are refused")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_pretrain(
            args.config,
            mode=args.mode,
            deadline_utc=args.deadline_utc,
            poll_seconds=args.poll_seconds,
            recover_stale_runner_lock=args.recover_stale_runner_lock,
            duplicate_dispositions=args.duplicate_dispositions,
        )
    except KeyboardInterrupt:
        print("Pretrain runner interrupted; inspect the persisted run status before restarting.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Pretrain runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in {"draft_ready", "data_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
