"""Crash-safe checkpoints and single-writer locks for MIL fitting.

Checkpoint payloads are immutable once committed. A manifest is the commit
record; pointer files are only an index and can always be rebuilt from valid
manifests. Payloads are hashed before deserialization. ``torch.load`` uses
``weights_only=False`` because training states may contain optimizer and RNG
objects; only load files created and committed by this store. Never point this
store at checkpoints from an untrusted source.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

_LOGGER = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_BLOCK_SIZE = 1024 * 1024


class CheckpointError(RuntimeError):
    """Base class for checkpoint storage failures."""


class CheckpointIntegrityError(CheckpointError):
    """A committed checkpoint or its manifest is incomplete or corrupted."""


class CheckpointIdentityError(CheckpointError):
    """A committed checkpoint belongs to a different training identity."""


class FitLockError(RuntimeError):
    """Base class for fit lock failures."""


class ActiveFitLockError(FitLockError):
    """The fit directory is already owned by a running process."""


class UnsafeStaleLockError(FitLockError):
    """A lock cannot be proven stale under the supported takeover policy."""


class LockOwnershipError(FitLockError):
    """The current process no longer owns the lock it is releasing."""


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """Identity and immutable payload location of a committed checkpoint."""

    version: str
    epoch: int
    timestamp_ns: int
    sha256: str
    is_best: bool
    state_path: str


@dataclass(frozen=True, slots=True)
class _Commit:
    version: str
    epoch: int
    timestamp_ns: int
    sha256: str
    is_best: bool
    manifest_path: Path
    state_path: Path

    def pointer(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "version": self.version,
            "epoch": self.epoch,
            "timestamp_ns": self.timestamp_ns,
            "sha256": self.sha256,
        }


def _canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible metadata deterministically and reject NaN."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint identity must contain only finite JSON values.") from exc


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX filesystems."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def _process_stat_tokens(pid: int) -> tuple[str, str] | None:
    """Return Linux process state and start-time ticks, distinguishing PID reuse."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    # The comm field may contain spaces and parentheses; the final ')' marks
    # its end. Field 22 (starttime) is field 20 after the state field.
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        return None
    fields_after_comm = raw[closing_paren + 1 :].split()
    if len(fields_after_comm) <= 19:
        return None
    return fields_after_comm[0], fields_after_comm[19]


def _process_start_token(pid: int) -> str | None:
    """Return Linux /proc start-time ticks, which distinguish PID reuse."""
    tokens = _process_stat_tokens(pid)
    return tokens[1] if tokens is not None else None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def _current_session_id() -> int | None:
    try:
        return os.getsid(0)
    except (AttributeError, OSError):
        return None


class CheckpointStore:
    """Publish, verify, and load immutable training checkpoints.

    Args:
        root: Checkpoint directory. It is created if it does not exist.
        identity: JSON-compatible run identity, for example data/split,
            model/config, and seed fingerprints. Every committed manifest must
            match this identity exactly.
        fault_injector: Test-only hook invoked at the documented transaction
            boundaries (``after_payload``, ``after_copy_before_commit``, and
            ``after_commit_before_pointer``).
    """

    def __init__(
        self,
        root: str | Path,
        identity: dict[str, Any],
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(identity, dict):
            raise TypeError("identity must be a dictionary.")
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.identity = json.loads(_canonical_json(identity).decode("utf-8"))
        self._identity_bytes = _canonical_json(self.identity)
        self._fault_injector = fault_injector

    def publish(self, state: dict[str, Any], epoch: int, is_best: bool = False) -> CheckpointReceipt:
        """Durably commit a new version and refresh latest/best pointers."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary.")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if not isinstance(is_best, bool):
            raise TypeError("is_best must be a boolean.")

        self._ensure_directories()
        version = f"{time.time_ns()}-{uuid.uuid4().hex}"
        version_dir = self.versions_dir / version
        version_dir.mkdir(mode=0o700)
        _fsync_directory(self.versions_dir)
        state_path = version_dir / "state.pt"
        staged_payload = version_dir / ".state.pt.tmp"

        try:
            import torch

            with staged_payload.open("xb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            sha256 = self._hash_path(staged_payload)
            self._inject("after_payload")

            os.replace(staged_payload, state_path)
            _fsync_directory(version_dir)
            self._inject("after_copy_before_commit")

            timestamp_ns = time.time_ns()
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "version": version,
                "epoch": epoch,
                "timestamp_ns": timestamp_ns,
                "sha256": sha256,
                "is_best": is_best,
                "identity": self.identity,
            }
            self._write_json_commit_atomic(version_dir / "commit.json", manifest)
            _fsync_directory(version_dir)
            self._inject("after_commit_before_pointer")

            commits = self._scan_commits()
            self._reconcile_pointers(commits)
            committed = next(commit for commit in commits if commit.version == version)
            _LOGGER.info(
                "Committed checkpoint epoch=%d version=%s best=%s",
                committed.epoch,
                committed.version,
                committed.is_best,
            )
            return CheckpointReceipt(
                version=committed.version,
                epoch=committed.epoch,
                timestamp_ns=committed.timestamp_ns,
                sha256=committed.sha256,
                is_best=committed.is_best,
                state_path=str(committed.state_path),
            )
        finally:
            # A directory without commit.json is intentionally ignored during
            # recovery. Keep it as transaction evidence; never erase a commit.
            staged_payload.unlink(missing_ok=True)

    def load_latest(
        self,
        map_location: Any = "cpu",
        *,
        reconcile_pointers: bool = True,
    ) -> dict[str, Any] | None:
        """Load the highest-epoch committed state, breaking ties by time.

        Set ``reconcile_pointers=False`` for a read-only recovery inspection
        after a pointer write failed. Commits and payloads are still fully
        verified; no pointer files are read or changed.
        """
        if not isinstance(reconcile_pointers, bool):
            raise TypeError("reconcile_pointers must be a boolean.")
        commits = self._scan_commits()
        if reconcile_pointers:
            self._reconcile_pointers(commits)
        if not commits:
            return None
        selected = max(commits, key=lambda item: (item.epoch, item.timestamp_ns, item.version))
        return self._load_commit(selected, map_location=map_location)

    def load_best(self, map_location: Any = "cpu") -> dict[str, Any] | None:
        """Load the most recent committed checkpoint marked as best."""
        commits = self._scan_commits()
        self._reconcile_pointers(commits)
        best_commits = [commit for commit in commits if commit.is_best]
        if not best_commits:
            return None
        selected = max(best_commits, key=lambda item: (item.epoch, item.timestamp_ns, item.version))
        return self._load_commit(selected, map_location=map_location)

    def _ensure_directories(self) -> None:
        if self.root.is_symlink():
            raise CheckpointIntegrityError(f"Checkpoint root cannot be a symbolic link: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.versions_dir.is_symlink():
            raise CheckpointIntegrityError(f"Checkpoint versions directory cannot be a symbolic link: {self.versions_dir}")
        self.versions_dir.mkdir(exist_ok=True)
        if not self.versions_dir.is_dir():
            raise CheckpointIntegrityError(f"Checkpoint versions path is not a directory: {self.versions_dir}")

    @staticmethod
    def _hash_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_json_commit_atomic(path: Path, value: dict[str, Any]) -> None:
        """Install a complete commit record atomically as the commit point."""
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # This version directory is uniquely created by this transaction,
            # so same-directory rename gives an atomic commit point without
            # requiring hard-link support from mounted Colab filesystems.
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _inject(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _scan_commits(self) -> list[_Commit]:
        """Validate every committed version; never mask corruption by fallback."""
        if not self.root.exists():
            return []
        self._ensure_directories()
        commits: list[_Commit] = []
        try:
            entries = sorted(self.versions_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise CheckpointIntegrityError(f"Cannot enumerate checkpoint versions: {self.versions_dir}") from exc

        for version_dir in entries:
            if version_dir.is_symlink():
                raise CheckpointIntegrityError(f"Symbolic link found in checkpoint versions: {version_dir}")
            if not version_dir.is_dir():
                continue
            manifest_path = version_dir / "commit.json"
            if not manifest_path.exists() and not manifest_path.is_symlink():
                # A crash before the commit record leaves an uncommitted
                # payload. It is invisible and does not replace prior state.
                continue
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise CheckpointIntegrityError(f"Invalid checkpoint commit record: {manifest_path}")
            manifest = self._read_manifest(manifest_path)
            commit = self._validate_manifest(manifest, version_dir, manifest_path)
            self._verify_payload(commit)
            commits.append(commit)
        return commits

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointIntegrityError(f"Unreadable committed checkpoint manifest: {path}") from exc
        if not isinstance(value, dict):
            raise CheckpointIntegrityError(f"Checkpoint manifest must be an object: {path}")
        return value

    def _validate_manifest(self, manifest: dict[str, Any], version_dir: Path, manifest_path: Path) -> _Commit:
        expected_fields = {
            "schema_version", "version", "epoch", "timestamp_ns", "sha256", "is_best", "identity",
        }
        if set(manifest) != expected_fields:
            raise CheckpointIntegrityError(f"Checkpoint manifest has an unexpected schema: {manifest_path}")
        schema_version = manifest["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != _SCHEMA_VERSION:
            raise CheckpointIntegrityError(f"Unsupported checkpoint manifest version: {manifest_path}")
        if not isinstance(manifest["version"], str) or manifest["version"] != version_dir.name:
            raise CheckpointIntegrityError(f"Checkpoint manifest version does not match its directory: {manifest_path}")
        epoch = manifest["epoch"]
        timestamp_ns = manifest["timestamp_ns"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise CheckpointIntegrityError(f"Checkpoint epoch is invalid: {manifest_path}")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
            raise CheckpointIntegrityError(f"Checkpoint timestamp is invalid: {manifest_path}")
        if not isinstance(manifest["is_best"], bool):
            raise CheckpointIntegrityError(f"Checkpoint best marker is invalid: {manifest_path}")
        sha256 = manifest["sha256"]
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise CheckpointIntegrityError(f"Checkpoint SHA-256 is invalid: {manifest_path}")
        identity = manifest["identity"]
        if not isinstance(identity, dict):
            raise CheckpointIntegrityError(f"Checkpoint identity is invalid: {manifest_path}")
        try:
            identity_bytes = _canonical_json(identity)
        except ValueError as exc:
            raise CheckpointIntegrityError(f"Checkpoint identity is invalid: {manifest_path}") from exc
        if identity_bytes != self._identity_bytes:
            raise CheckpointIdentityError(
                f"Checkpoint identity mismatch in committed version {version_dir.name}; refusing resume."
            )

        state_path = version_dir / "state.pt"
        if state_path.is_symlink() or not state_path.is_file():
            raise CheckpointIntegrityError(f"Committed checkpoint payload is missing or unsafe: {state_path}")
        return _Commit(
            version=version_dir.name,
            epoch=epoch,
            timestamp_ns=timestamp_ns,
            sha256=sha256,
            is_best=manifest["is_best"],
            manifest_path=manifest_path,
            state_path=state_path,
        )

    def _verify_payload(self, commit: _Commit) -> None:
        try:
            actual_sha256 = self._hash_path(commit.state_path)
        except OSError as exc:
            raise CheckpointIntegrityError(f"Cannot read committed checkpoint payload: {commit.state_path}") from exc
        if actual_sha256 != commit.sha256:
            raise CheckpointIntegrityError(
                f"Committed checkpoint checksum mismatch for version {commit.version}; refusing fallback."
            )

    def _reconcile_pointers(self, commits: list[_Commit]) -> None:
        latest = max(commits, key=lambda item: (item.epoch, item.timestamp_ns, item.version)) if commits else None
        best_candidates = [commit for commit in commits if commit.is_best]
        best = max(best_candidates, key=lambda item: (item.epoch, item.timestamp_ns, item.version)) if best_candidates else None
        self._write_pointer(self.root / "latest.json", latest)
        self._write_pointer(self.root / "best.json", best)

    @staticmethod
    def _write_pointer(path: Path, commit: _Commit | None) -> None:
        desired = commit.pointer() if commit is not None else None
        if path.is_file() and not path.is_symlink() and desired is not None:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                existing = None
            if existing == desired:
                return
        if desired is None:
            if path.exists() or path.is_symlink():
                path.unlink()
                _fsync_directory(path.parent)
            return

        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("x", encoding="utf-8") as stream:
                json.dump(desired, stream, sort_keys=True, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            _fsync_directory(path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load_commit(self, commit: _Commit, *, map_location: Any) -> dict[str, Any]:
        import torch

        try:
            with commit.state_path.open("rb") as stream:
                actual_sha256 = _hash_stream(stream)
                if actual_sha256 != commit.sha256:
                    raise CheckpointIntegrityError(
                        f"Committed checkpoint checksum mismatch for version {commit.version}; refusing load."
                    )
                stream.seek(0)
                # Deliberately use weights_only=False only for own files after
                # validating the manifest identity and payload checksum.
                state = torch.load(stream, map_location=map_location, weights_only=False)
        except CheckpointIntegrityError:
            raise
        except Exception as exc:
            raise CheckpointIntegrityError(f"Cannot deserialize committed checkpoint {commit.version}.") from exc
        if not isinstance(state, dict):
            raise CheckpointIntegrityError(f"Checkpoint state must be a dictionary: {commit.state_path}")
        return state


class FitLock:
    """Exclusive process lock for a single MIL fit directory.

    Same-host stale takeover is opt-in and requires proof that the recorded
    process/session is gone. Cross-host takeover requires an explicit,
    operator-verified stopped-runtime receipt bound to the exact persisted
    lock identity.
    """

    def __init__(
        self,
        root: str | Path,
        owner_id: str | None = None,
        *,
        recover_stale: bool = False,
        recovery_evidence: dict[str, Any] | None = None,
    ) -> None:
        if owner_id is not None and not isinstance(owner_id, str):
            raise TypeError("owner_id must be a string or None.")
        if not isinstance(recover_stale, bool):
            raise TypeError("recover_stale must be a boolean.")
        if recovery_evidence is not None and not isinstance(recovery_evidence, dict):
            raise TypeError("recovery_evidence must be a dictionary or None.")
        self.root = Path(root)
        self.owner_id = owner_id
        self.recover_stale = recover_stale
        self.recovery_evidence = dict(recovery_evidence) if recovery_evidence is not None else None
        self.lock_path = self.root / ".fit.lock"
        self.guard_path = self.root / ".fit.lock.guard"
        self._token: str | None = None
        self._metadata: dict[str, Any] | None = None

    def __enter__(self) -> FitLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.release()
        return False

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        """Metadata written for this owner's lock, if acquired."""
        return self._metadata

    def acquire(self) -> None:
        if self._token is not None:
            raise FitLockError("This FitLock instance already owns a lock.")
        if self.root.is_symlink():
            raise FitLockError(f"Fit lock root cannot be a symbolic link: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

        # A stable advisory guard serializes stale removal and O_EXCL creation.
        # The kernel releases flock automatically if a process exits.
        with self.guard_path.open("a+b") as guard:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
            try:
                if self.lock_path.exists() or self.lock_path.is_symlink():
                    self._handle_existing_lock()
                self._create_lock()
            finally:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)

    def _handle_existing_lock(self) -> None:
        existing = self._read_lock_metadata()
        if not self.recover_stale:
            raise ActiveFitLockError(self._describe_lock(existing))
        if existing.get("host") != socket.gethostname():
            if not self._has_matching_recovery_evidence(existing):
                raise UnsafeStaleLockError(
                    "Cannot recover a fit lock from another host without matching operator-verified "
                    "stopped-runtime evidence."
                )
            _LOGGER.warning("Recovering remote fit lock with explicit stopped-runtime evidence at %s", self.lock_path)
        elif not self._is_proven_stale(existing):
            raise ActiveFitLockError(self._describe_lock(existing))
        self.lock_path.unlink()
        _fsync_directory(self.root)
        if existing.get("host") == socket.gethostname():
            _LOGGER.warning("Recovered stale same-host fit lock at %s", self.lock_path)

    def _has_matching_recovery_evidence(self, metadata: Mapping[str, Any]) -> bool:
        """Validate manager-checked proof for takeover of a remote runtime."""
        evidence = self.recovery_evidence
        if evidence is None:
            return False
        required_fields = {
            "token", "host", "boot_id", "session_id", "old_runtime_stopped", "evidence", "verified_at_utc",
        }
        if set(evidence) != required_fields:
            raise UnsafeStaleLockError("Recovery evidence has an unexpected schema.")
        for field in ("token", "host", "boot_id", "session_id"):
            expected = metadata.get(field)
            actual = evidence[field]
            if type(actual) is not type(expected) or actual != expected:
                raise UnsafeStaleLockError(f"Recovery evidence does not match the old lock's {field}.")
        if evidence["old_runtime_stopped"] is not True:
            raise UnsafeStaleLockError("Recovery evidence must confirm old_runtime_stopped=true.")
        proof = evidence["evidence"]
        if not isinstance(proof, str) or not proof.strip():
            raise UnsafeStaleLockError("Recovery evidence must include a non-empty operator/manager proof.")
        verified_at = evidence["verified_at_utc"]
        if not isinstance(verified_at, str):
            raise UnsafeStaleLockError("Recovery evidence must include a timezone-aware verified_at_utc timestamp.")
        try:
            parsed_at = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise UnsafeStaleLockError("Recovery evidence timestamp is not valid ISO-8601.") from exc
        if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
            raise UnsafeStaleLockError("Recovery evidence timestamp must include a timezone.")
        if parsed_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
            raise UnsafeStaleLockError("Recovery evidence timestamp cannot be in the future.")
        return True

    def _read_lock_metadata(self) -> dict[str, Any]:
        try:
            if self.lock_path.is_symlink() or not self.lock_path.is_file():
                raise UnsafeStaleLockError(f"Fit lock is not a regular file: {self.lock_path}")
            value = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UnsafeStaleLockError(f"Cannot verify existing fit lock: {self.lock_path}") from exc
        if not isinstance(value, dict):
            raise UnsafeStaleLockError(f"Cannot verify existing fit lock: {self.lock_path}")
        if not isinstance(value.get("pid"), int) or isinstance(value.get("pid"), bool) or value["pid"] <= 0:
            raise UnsafeStaleLockError(f"Existing fit lock has invalid process metadata: {self.lock_path}")
        return value

    @staticmethod
    def _describe_lock(metadata: Mapping[str, Any]) -> str:
        return (
            "Fit directory is locked by "
            f"pid={metadata.get('pid')} host={metadata.get('host')} owner={metadata.get('owner_id')!r}."
        )

    @staticmethod
    def _is_proven_stale(metadata: Mapping[str, Any]) -> bool:
        pid = metadata["pid"]
        current_boot_id = _boot_id()
        recorded_boot_id = metadata.get("boot_id")
        if current_boot_id and recorded_boot_id and current_boot_id != recorded_boot_id:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False

        process_tokens = _process_stat_tokens(pid)
        if process_tokens is not None and process_tokens[0] in {"Z", "X"}:
            return True

        recorded_start = metadata.get("process_start")
        current_start = process_tokens[1] if process_tokens is not None else None
        if recorded_start is not None and current_start is not None and recorded_start != current_start:
            return True  # PID has been reused; its recorded owner is gone.
        return False

    def _create_lock(self) -> None:
        token = uuid.uuid4().hex
        metadata = {
            "schema_version": 1,
            "token": token,
            "owner_id": self.owner_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "boot_id": _boot_id(),
            "session_id": _current_session_id(),
            "process_start": _process_start_token(os.getpid()),
            "acquired_at_ns": time.time_ns(),
        }
        payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            existing = self._read_lock_metadata()
            raise ActiveFitLockError(self._describe_lock(existing)) from exc

        try:
            payload_bytes = payload.encode("utf-8")
            remaining = memoryview(payload_bytes)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("Could not finish writing fit lock metadata.")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            _fsync_directory(self.root)
            self._token = token
            self._metadata = metadata
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            # This process created the O_EXCL file while holding the guard, so
            # remove it if a synchronous write/sync failure leaves it partial.
            self.lock_path.unlink(missing_ok=True)
            _fsync_directory(self.root)
            raise

    def release(self) -> None:
        if self._token is None:
            return
        with self.guard_path.open("a+b") as guard:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
            try:
                current = self._read_lock_metadata()
                if current.get("token") != self._token:
                    raise LockOwnershipError("Fit lock ownership changed; refusing to remove another owner's lock.")
                self.lock_path.unlink()
                _fsync_directory(self.root)
                self._token = None
                self._metadata = None
            finally:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


__all__ = [
    "ActiveFitLockError",
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointIntegrityError",
    "CheckpointReceipt",
    "CheckpointStore",
    "FitLock",
    "FitLockError",
    "LockOwnershipError",
    "UnsafeStaleLockError",
]
