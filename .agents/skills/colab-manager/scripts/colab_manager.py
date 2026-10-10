#!/usr/bin/env python3
"""Manage saved Colab CLI profiles, tracked sessions, and local time estimates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

SKILL_SCRIPT = Path(__file__).resolve()
LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
SUPPORTED_GPUS = ("T4", "L4", "G4", "H100", "A100")
DEFAULT_T4_BUDGET_MINUTES = 350
ALERT_THRESHOLDS_MINUTES = (30, 10, 5, 0)
SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

CAPACITY_ERROR_PATTERNS = (
    re.compile(r"\bresource[_ -]exhausted\b", re.IGNORECASE),
    re.compile(r"\bquota[_ -]exceeded\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:gpu|accelerator|runtime|capacity)\s+available\b", re.IGNORECASE),
    re.compile(
        r"\b(?:gpu|accelerator|runtime)\s+(?:is\s+)?(?:currently\s+)?unavailable\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:capacity|availability)\s+(?:is\s+)?(?:temporarily\s+)?unavailable\b", re.IGNORECASE),
    re.compile(r"\bresource\s+limit\s+(?:reached|exceeded)\b", re.IGNORECASE),
    re.compile(r"\b(?:gpu|accelerator)\s+quota\s+(?:reached|exceeded)\b", re.IGNORECASE),
    re.compile(r"\b(?:insufficient|not enough|no)\s+compute units\b", re.IGNORECASE),
    re.compile(r"\bcompute units?\s+(?:depleted|exhausted|exceeded)\b", re.IGNORECASE),
    re.compile(r"\bmaximum\s+(?:number\s+of\s+)?(?:sessions|runtimes)\s+reached\b", re.IGNORECASE),
)
NON_RETRYABLE_ERROR_PATTERN = re.compile(
    r"\b(?:"
    r"unauthenticated|unauthorized|401|invalid_grant|invalid credentials?|"
    r"permission denied|insufficient scopes?|"
    r"connection (?:error|refused|reset|timed out)|"
    r"network (?:error|unreachable)|could not resolve|"
    r"timed? out|timeout|"
    r"invalid argument|unknown option"
    r")\b",
    re.IGNORECASE,
)
SESSION_NOT_FOUND_PATTERN = re.compile(
    r"(?:\b404\b|\bsession\b.{0,100}\bnot found\b|"
    r"\bno\s+(?:active\s+)?sessions?\b.{0,40}\b(?:found|exists?)\b)",
    re.IGNORECASE,
)


class ManagerError(Exception):
    """An expected operational failure with a user-readable message."""


@dataclass(frozen=True)
class Profile:
    """A named Colab CLI profile with non-secret readiness metadata."""

    alias: str
    token_path: Path
    session_config: Path

    @property
    def ready(self) -> bool:
        """Return whether both local files required by the wrapper exist."""
        return self.token_path.is_file() and self.session_config.is_file()


@dataclass(frozen=True)
class CommandResult:
    """Captured result from a child CLI command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        """Return output suitable for classifying a CLI failure."""
        return f"{self.stdout}\n{self.stderr}".strip()


def default_path(environment_key: str, fallback: Path) -> Path:
    """Resolve an optional path override while keeping defaults user-local."""
    value = os.environ.get(environment_key)
    return Path(value).expanduser() if value else fallback


def default_profiles_path() -> Path:
    """Return the Colab CLI profile registry path."""
    return default_path(
        "COLAB_MANAGER_PROFILES_FILE",
        Path.home() / ".config" / "colab-cli" / "profiles.json",
    )


def default_state_path() -> Path:
    """Return the local timer state path; it contains no credential material."""
    return default_path(
        "COLAB_MANAGER_STATE_FILE",
        Path.home() / ".local" / "state" / "colab-manager" / "state.json",
    )


def default_settings_path() -> Path:
    """Return the user-editable Colab Manager settings path."""
    return default_path(
        "COLAB_MANAGER_SETTINGS_FILE",
        Path.home() / ".config" / "colab-manager" / "settings.json",
    )


def default_unit_dir() -> Path:
    """Return the per-user systemd unit directory."""
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return Path(os.environ.get("COLAB_MANAGER_UNIT_DIR", config_home / "systemd" / "user"))


def default_profile_command() -> str:
    """Return the installed profile wrapper command."""
    return os.environ.get("COLAB_MANAGER_PROFILE_COMMAND", "colab-profile")


def default_systemctl_command() -> str:
    """Return the systemctl executable, with a test override."""
    return os.environ.get("COLAB_MANAGER_SYSTEMCTL_COMMAND", "systemctl")


def default_notification_command() -> str:
    """Return the desktop notification executable, if configured."""
    return os.environ.get("COLAB_MANAGER_NOTIFY_COMMAND", "notify-send")


def parse_datetime(value: str) -> datetime:
    """Parse ISO timestamps, treating old naive values as UTC."""
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_datetime(value: datetime) -> str:
    """Format an aware timestamp in stable UTC ISO-8601 form."""
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def local_day_bounds(value: datetime) -> tuple[date, datetime, datetime]:
    """Return the local date and its UTC start/end instants."""
    local_value = value.astimezone(LOCAL_TIMEZONE)
    local_date = local_value.date()
    local_start = datetime.combine(local_date, time.min, tzinfo=LOCAL_TIMEZONE)
    next_start = datetime.combine(local_date + timedelta(days=1), time.min, tzinfo=LOCAL_TIMEZONE)
    return local_date, local_start.astimezone(timezone.utc), next_start.astimezone(timezone.utc)


def read_json(path: Path) -> Any:
    """Read JSON with a concise error for malformed user state."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ManagerError(f"Cannot read JSON file {path}: {error}") from error


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist local state with owner-only permissions."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ManagerError(f"Cannot save local state to {path}: {error}") from error


class ProfileRegistry:
    """Read the profile order and file readiness without opening token files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[Profile]:
        """Load aliases in configured order; reject ambiguous registry entries."""
        try:
            data = read_json(self.path)
        except FileNotFoundError as error:
            raise ManagerError(f"Colab profile registry not found: {self.path}") from error
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
            raise ManagerError("Colab profile registry must contain a profiles list.")

        profiles: list[Profile] = []
        seen: set[str] = set()
        for index, entry in enumerate(data["profiles"], start=1):
            if not isinstance(entry, dict):
                raise ManagerError(f"Profile entry {index} is not an object.")
            alias = entry.get("alias")
            token_value = entry.get("token_path")
            session_value = entry.get("session_config")
            if not isinstance(alias, str) or not SESSION_NAME_PATTERN.fullmatch(alias):
                raise ManagerError(f"Profile entry {index} has an invalid alias.")
            if alias in seen:
                raise ManagerError(f"Duplicate profile alias in registry: {alias}")
            if not isinstance(token_value, str) or not isinstance(session_value, str):
                raise ManagerError(f"Profile {alias} is missing path metadata.")
            seen.add(alias)
            profiles.append(
                Profile(
                    alias=alias,
                    token_path=Path(os.path.expandvars(token_value)).expanduser(),
                    session_config=Path(os.path.expandvars(session_value)).expanduser(),
                )
            )
        return profiles


def empty_state() -> dict[str, Any]:
    """Create a versioned metadata-only state document."""
    return {
        "version": 1,
        "last_successful_profile": None,
        "profile_activity": {},
        "sessions": [],
        "alerts": {},
    }


class StateStore:
    """Serialize concurrent timer and CLI updates with an advisory file lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            data = read_json(self.path)
        except FileNotFoundError:
            return empty_state()
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ManagerError(f"Unsupported or invalid Colab Manager state file: {self.path}")
        if not isinstance(data.get("sessions"), list):
            raise ManagerError(f"Invalid sessions list in state file: {self.path}")
        if not isinstance(data.get("profile_activity"), dict) or not isinstance(data.get("alerts"), dict):
            raise ManagerError(f"Invalid profile or alert state in: {self.path}")
        return data

    def load(self) -> dict[str, Any]:
        """Read a consistent state snapshot."""
        self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            return self._read_unlocked()

    def update(self, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Apply one state update under an exclusive lock and atomic replace."""
        self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = self._read_unlocked()
            mutate(state)
            atomic_write_json(self.path, state)
            return state


class SettingsStore:
    """Read and update the editable local runtime budget settings."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def defaults() -> dict[str, Any]:
        return {
            "version": 1,
            "default_daily_budget_minutes": {"T4": DEFAULT_T4_BUDGET_MINUTES},
            "profile_daily_budget_minutes": {},
        }

    def load(self) -> dict[str, Any]:
        """Load settings, supplying the plan's default T4 estimate."""
        try:
            data = read_json(self.path)
        except FileNotFoundError:
            return self.defaults()
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ManagerError(f"Unsupported or invalid Colab Manager settings file: {self.path}")
        defaults = self.defaults()
        default_budgets = data.get("default_daily_budget_minutes", {})
        profile_budgets = data.get("profile_daily_budget_minutes", {})
        if not isinstance(default_budgets, dict) or not isinstance(profile_budgets, dict):
            raise ManagerError(f"Invalid budget settings in: {self.path}")
        defaults["default_daily_budget_minutes"].update(default_budgets)
        defaults["profile_daily_budget_minutes"] = profile_budgets
        return defaults

    def get_budget(self, profile: str, gpu: str) -> int | None:
        """Return a profile override or the GPU-wide local estimate."""
        data = self.load()
        profile_budgets = data["profile_daily_budget_minutes"].get(profile, {})
        if not isinstance(profile_budgets, dict):
            raise ManagerError(f"Invalid profile budget for {profile}.")
        value = profile_budgets.get(gpu, data["default_daily_budget_minutes"].get(gpu))
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ManagerError(f"Invalid daily budget for {profile}/{gpu}.")
        return value

    def set_budget(self, gpu: str, minutes: int, profile: str | None = None) -> None:
        """Persist one positive per-day estimate."""
        if gpu not in SUPPORTED_GPUS:
            raise ManagerError(f"Unsupported GPU {gpu}; choose from {', '.join(SUPPORTED_GPUS)}.")
        if minutes < 1 or minutes > 1440:
            raise ManagerError("Daily budget must be between 1 and 1440 minutes.")
        data = self.load()
        if profile is None:
            data["default_daily_budget_minutes"][gpu] = minutes
        else:
            overrides = data["profile_daily_budget_minutes"].setdefault(profile, {})
            if not isinstance(overrides, dict):
                raise ManagerError(f"Invalid profile budget for {profile}.")
            overrides[gpu] = minutes
        atomic_write_json(self.path, data)


def daily_usage_seconds(
    state: dict[str, Any],
    profile: str,
    gpu: str,
    now: datetime,
) -> float:
    """Sum tracked runtime seconds within the current Ho Chi Minh calendar day."""
    _, day_start, next_day_start = local_day_bounds(now)
    total = 0.0
    for session in state["sessions"]:
        if session.get("profile") != profile or session.get("gpu") != gpu:
            continue
        try:
            start = parse_datetime(session["started_at"])
            end = parse_datetime(session["ended_at"]) if session.get("ended_at") else now
        except (KeyError, TypeError, ValueError) as error:
            raise ManagerError("Invalid session timestamps in Colab Manager state.") from error
        interval_start = max(start, day_start)
        interval_end = min(end, next_day_start, now)
        if interval_end > interval_start:
            total += (interval_end - interval_start).total_seconds()
    return total


def active_sessions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sessions whose local tracking interval is still open."""
    return [session for session in state["sessions"] if session.get("ended_at") is None]


def human_minutes(seconds: float) -> str:
    """Format seconds as a one-decimal minute estimate."""
    return f"{seconds / 60:.1f} min"


def is_capacity_unavailable(message: str) -> bool:
    """Only recognize explicit quota/capacity errors for automatic failover."""
    if NON_RETRYABLE_ERROR_PATTERN.search(message):
        return False
    return any(pattern.search(message) for pattern in CAPACITY_ERROR_PATTERNS)


def is_session_missing(result: CommandResult) -> bool:
    """Distinguish an explicit expired session from auth or network failures."""
    if NON_RETRYABLE_ERROR_PATTERN.search(result.combined_output):
        return False
    return result.returncode in (404, 44) or bool(SESSION_NOT_FOUND_PATTERN.search(result.combined_output))


def rotate_profiles(profiles: Sequence[Profile], start_alias: str | None) -> list[Profile]:
    """Rotate the registry order from an alias, retaining incomplete slots as skips."""
    if not profiles:
        return []
    if start_alias is None:
        start_index = 0
    else:
        indexes = [index for index, profile in enumerate(profiles) if profile.alias == start_alias]
        start_index = indexes[0] if indexes else 0
    return list(profiles[start_index:]) + list(profiles[:start_index])


def validate_session_name(value: str) -> str:
    """Reject session names that could be ambiguous in local state or CLI output."""
    if not SESSION_NAME_PATTERN.fullmatch(value):
        raise ManagerError("Session name must be 1–64 letters, digits, dots, underscores, or hyphens.")
    return value


def new_session_name(gpu: str) -> str:
    """Generate a unique, CLI-safe session name."""
    stamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"cm-{gpu.lower()}-{stamp}-{suffix}"


class ColabProfileRunner:
    """Invoke the existing wrapper without shell evaluation or credential access."""

    def __init__(self, command: str) -> None:
        self.command = command

    def run(self, alias: str, arguments: Sequence[str], timeout: float = 120.0) -> CommandResult:
        """Call colab-profile and capture only transient command output."""
        try:
            result = subprocess.run(
                [self.command, alias, *arguments],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
            return CommandResult(124, stdout, stderr or "Command timed out.", timed_out=True)
        except FileNotFoundError as error:
            raise ManagerError(f"Profile wrapper not found: {self.command}") from error
        except OSError as error:
            raise ManagerError(f"Cannot run profile wrapper {self.command}: {error}") from error


def print_command_output(result: CommandResult) -> None:
    """Forward transient CLI output to the caller without writing it to state."""
    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")


def ask_confirmation(prompt: str, approved: bool) -> None:
    """Require explicit approval in direct CLI use; automation must pass --yes."""
    if approved:
        return
    if not sys.stdin.isatty():
        raise ManagerError(f"{prompt} Run interactively or pass --yes after user authorization.")
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise ManagerError("Cancelled by user.")


def resolve_start_alias(
    profiles: Sequence[Profile],
    state: dict[str, Any],
    requested_alias: str | None,
) -> str | None:
    """Choose the explicit start profile, last successful alias, or first slot."""
    if requested_alias:
        if not any(profile.alias == requested_alias for profile in profiles):
            raise ManagerError(f"Profile alias not found: {requested_alias}")
        return requested_alias
    previous = state.get("last_successful_profile")
    if isinstance(previous, str) and any(profile.alias == previous for profile in profiles):
        return previous
    return profiles[0].alias if profiles else None


def record_profile_attempt(
    store: StateStore,
    alias: str,
    result: str,
    timestamp: datetime,
) -> None:
    """Persist non-sensitive last-attempt metadata for one profile."""
    def mutate(state: dict[str, Any]) -> None:
        activity = state["profile_activity"].setdefault(alias, {})
        activity["last_attempt_at"] = format_datetime(timestamp)
        activity["last_attempt_result"] = result

    store.update(mutate)


def record_created_session(
    store: StateStore,
    alias: str,
    session_name: str,
    gpu: str,
    timestamp: datetime,
) -> None:
    """Record a successfully created VM and advance the round-robin cursor."""
    def mutate(state: dict[str, Any]) -> None:
        if any(item.get("session") == session_name for item in state["sessions"]):
            raise ManagerError(f"Session name is already tracked: {session_name}")
        state["sessions"].append(
            {
                "profile": alias,
                "session": session_name,
                "gpu": gpu,
                "started_at": format_datetime(timestamp),
                "ended_at": None,
            }
        )
        state["last_successful_profile"] = alias
        activity = state["profile_activity"].setdefault(alias, {})
        activity["last_attempt_at"] = format_datetime(timestamp)
        activity["last_attempt_result"] = "created"
        activity["last_used_at"] = format_datetime(timestamp)

    store.update(mutate)


def finish_session(store: StateStore, profile: str, session_name: str, timestamp: datetime) -> bool:
    """Set an end time once; return whether a tracked session was changed."""
    changed = False

    def mutate(state: dict[str, Any]) -> None:
        nonlocal changed
        for session in state["sessions"]:
            if session.get("profile") == profile and session.get("session") == session_name:
                if session.get("ended_at") is None:
                    session["ended_at"] = format_datetime(timestamp)
                    changed = True
                break

    store.update(mutate)
    return changed


def systemd_quote(value: str) -> str:
    """Quote one systemd ExecStart argument, including percent specifiers."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def systemd_escape_path(value: str) -> str:
    """Escape path whitespace for systemd path-valued directives."""
    return value.replace("\\", "\\x5c").replace("%", "%%").replace(" ", "\\x20").replace("\t", "\\x09")


class UserTimer:
    """Install and control a user-level systemd timer only while sessions exist."""

    def __init__(
        self,
        unit_dir: Path,
        systemctl_command: str,
        profile_command: str,
        profiles_path: Path,
        state_path: Path,
        settings_path: Path,
        notification_command: str,
    ) -> None:
        self.unit_dir = unit_dir
        self.systemctl_command = systemctl_command
        self.profile_command = profile_command
        self.profiles_path = profiles_path
        self.state_path = state_path
        self.settings_path = settings_path
        self.notification_command = notification_command

    @property
    def service_path(self) -> Path:
        return self.unit_dir / "colab-manager.service"

    @property
    def timer_path(self) -> Path:
        return self.unit_dir / "colab-manager.timer"

    def _systemctl(self, *arguments: str) -> CommandResult:
        try:
            result = subprocess.run(
                [self.systemctl_command, "--user", *arguments],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return CommandResult(result.returncode, result.stdout, result.stderr)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
            return CommandResult(127, "", str(error))

    def unit_contents(self) -> tuple[str, str]:
        """Render units that invoke this exact skill helper and local state."""
        command = [
            sys.executable,
            str(SKILL_SCRIPT),
            "--profiles-file",
            str(self.profiles_path),
            "--state-file",
            str(self.state_path),
            "--settings-file",
            str(self.settings_path),
            "--unit-dir",
            str(self.unit_dir),
            "--profile-command",
            self.profile_command,
            "--systemctl-command",
            self.systemctl_command,
            "--notification-command",
            self.notification_command,
            "watch",
        ]
        exec_start = " ".join(systemd_quote(argument) for argument in command)
        workspace_root = SKILL_SCRIPT.parents[4]
        service = (
            "[Unit]\n"
            "Description=Check tracked Google Colab session time\n\n"
            "[Service]\n"
            "Type=oneshot\n"
            f"WorkingDirectory={systemd_escape_path(str(workspace_root))}\n"
            f"ExecStart={exec_start}\n"
        )
        timer = (
            "[Unit]\n"
            "Description=Track Google Colab session time every minute\n\n"
            "[Timer]\n"
            "OnCalendar=*-*-* *:*:00\n"
            "Persistent=true\n"
            "AccuracySec=2s\n"
            "Unit=colab-manager.service\n\n"
            "[Install]\n"
            "WantedBy=timers.target\n"
        )
        return service, timer

    def install(self) -> None:
        """Write user unit files and reload systemd without starting the timer."""
        self.unit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        service, timer = self.unit_contents()
        atomic_write_text(self.service_path, service)
        atomic_write_text(self.timer_path, timer)
        result = self._systemctl("daemon-reload")
        if result.returncode != 0:
            raise ManagerError(
                "Could not reload the user systemd manager. "
                f"Check that systemd --user is available. {result.stderr.strip()}"
            )

    def start(self) -> None:
        """Install and enable the timer while at least one session is tracked."""
        self.install()
        result = self._systemctl("enable", "--now", "colab-manager.timer")
        if result.returncode != 0:
            raise ManagerError(f"Could not start the Colab timer: {result.stderr.strip()}")

    def stop_when_idle(self) -> None:
        """Disable the timer after all tracked sessions have ended."""
        result = self._systemctl("disable", "--now", "colab-manager.timer")
        if result.returncode not in (0, 1):
            print(
                f"Timer cleanup warning: {result.stderr.strip() or 'systemd returned an error.'}",
                file=sys.stderr,
            )

    def status(self) -> tuple[bool, str, str]:
        """Return whether unit files exist plus systemd enabled/active states."""
        installed = self.service_path.is_file() and self.timer_path.is_file()
        if not installed:
            return False, "not installed", "inactive"
        active_result = self._systemctl("is-active", "colab-manager.timer")
        enabled_result = self._systemctl("is-enabled", "colab-manager.timer")
        active = "active" if active_result.returncode == 0 else "inactive"
        if active_result.returncode not in (0, 3):
            active = "unknown"
        enabled = "enabled" if enabled_result.returncode == 0 else "disabled"
        if enabled_result.returncode not in (0, 1):
            enabled = "unknown"
        return installed, enabled, active

    def uninstall(self) -> None:
        """Stop the timer and remove its units after callers verify no active sessions."""
        active_result = self._systemctl("is-active", "colab-manager.timer")
        if active_result.returncode not in (0, 3, 4):
            raise ManagerError(
                "Could not verify that the Colab timer is inactive; unit files were preserved. "
                f"{active_result.stderr.strip()}"
            )
        disable_result = self._systemctl("disable", "--now", "colab-manager.timer")
        if disable_result.returncode != 0:
            raise ManagerError(
                "Could not disable the Colab timer; unit files were preserved. "
                f"{disable_result.stderr.strip()}"
            )
        self.service_path.unlink(missing_ok=True)
        self.timer_path.unlink(missing_ok=True)
        result = self._systemctl("daemon-reload")
        if result.returncode != 0:
            raise ManagerError(f"Could not reload systemd after removing units: {result.stderr.strip()}")


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write a unit file with owner-only permissions."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ManagerError(f"Cannot write systemd unit {path}: {error}") from error


def build_timer(args: argparse.Namespace) -> UserTimer:
    """Construct the user timer from resolved CLI paths and executables."""
    profile_command = shutil.which(args.profile_command) or args.profile_command
    systemctl_command = shutil.which(args.systemctl_command) or args.systemctl_command
    notification_command = (
        shutil.which(args.notification_command) or ""
        if args.notification_command
        else ""
    )
    return UserTimer(
        unit_dir=args.unit_dir,
        systemctl_command=systemctl_command,
        profile_command=profile_command,
        profiles_path=args.profiles_file,
        state_path=args.state_file,
        settings_path=args.settings_file,
        notification_command=notification_command,
    )


def print_profiles(args: argparse.Namespace) -> None:
    """Display numbered profile order without exposing account identity or tokens."""
    profiles = ProfileRegistry(args.profiles_file).load()
    state = StateStore(args.state_file).load()
    print("Profiles (configured order; readiness checks file presence only):")
    for index, profile in enumerate(profiles, start=1):
        ready = "ready by file check" if profile.ready else "skipped: missing token/session file"
        current = " [last successful]" if state.get("last_successful_profile") == profile.alias else ""
        activity = state["profile_activity"].get(profile.alias, {})
        last_used = activity.get("last_used_at")
        last_used_text = f"; last used {format_local(last_used)}" if isinstance(last_used, str) else ""
        print(f"{index}. {profile.alias}: {ready}{last_used_text}{current}")
    print("A file check does not verify that Colab authentication is still valid.")


def format_local(timestamp: str) -> str:
    """Format a stored UTC timestamp in the configured local timezone."""
    try:
        return parse_datetime(timestamp).astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
    except (ValueError, TypeError):
        return "unknown"


def print_status(args: argparse.Namespace) -> None:
    """Show local usage estimates and active tracked sessions."""
    profiles = ProfileRegistry(args.profiles_file).load()
    state = StateStore(args.state_file).load()
    settings = SettingsStore(args.settings_file)
    now = datetime.now(timezone.utc)
    current_alias = state.get("last_successful_profile")
    print(f"Local estimate date: {now.astimezone(LOCAL_TIMEZONE).date().isoformat()} (Asia/Ho_Chi_Minh)")
    print("Daily budget is an estimate; Colab does not expose remaining GPU hours through usage.")
    print("Profile status:")
    for index, profile in enumerate(profiles, start=1):
        readiness = "ready by file check" if profile.ready else "skipped: missing token/session file"
        marker = " *" if profile.alias == current_alias else ""
        budget = settings.get_budget(profile.alias, "T4")
        if budget is None:
            budget_text = "T4: no daily estimate"
        else:
            used = daily_usage_seconds(state, profile.alias, "T4", now)
            remaining = max(0.0, budget * 60 - used)
            budget_text = f"T4: {human_minutes(used)} / {budget} min; about {human_minutes(remaining)} remain"
        activity = state["profile_activity"].get(profile.alias, {})
        last_used = activity.get("last_used_at")
        last_used_text = f"; last used {format_local(last_used)}" if isinstance(last_used, str) else "; never used by manager"
        print(f"  {index}. {profile.alias}{marker}: {readiness}; {budget_text}{last_used_text}")
    sessions = active_sessions(state)
    if not sessions:
        print("Tracked active sessions: none")
        return
    print("Tracked active sessions:")
    for session in sessions:
        started = parse_datetime(session["started_at"])
        elapsed = max(0.0, (now - started).total_seconds())
        print(
            f"  {session['session']} — {session['profile']} / {session['gpu']}, "
            f"elapsed {human_minutes(elapsed)}"
        )


def profile_start_index(profiles: Sequence[Profile], alias: str | None) -> int:
    """Return the configured position for a starting alias."""
    if not profiles:
        return 0
    if alias is None:
        return 0
    for index, profile in enumerate(profiles):
        if profile.alias == alias:
            return index
    return 0


def available_for_budget(profile: Profile, gpu: str, state: dict[str, Any], settings: SettingsStore) -> tuple[bool, str]:
    """Check local configured budget before attempting a new remote runtime."""
    budget = settings.get_budget(profile.alias, gpu)
    if budget is None:
        return True, ""
    used_seconds = daily_usage_seconds(state, profile.alias, gpu, datetime.now(timezone.utc))
    remaining_seconds = max(0.0, budget * 60 - used_seconds)
    if remaining_seconds <= 0:
        return False, f"local {gpu} estimate reached ({budget} min/day)"
    return True, ""


def _update_last_attempt(store: StateStore, alias: str, result: str) -> None:
    timestamp = datetime.now(timezone.utc)
    record_profile_attempt(store, alias, result, timestamp)


def start_timer_safely(timer: UserTimer) -> None:
    """Start background monitoring without undoing a successfully created VM."""
    try:
        timer.start()
    except ManagerError as error:
        print(
            "Session is tracked locally, but background notifications could not start. "
            f"Run timer install after resolving this issue: {error}",
            file=sys.stderr,
        )


def create_session(args: argparse.Namespace) -> int:
    """Create a session with bounded round-robin failover on explicit resource errors."""
    gpu = args.gpu.upper()
    if gpu not in SUPPORTED_GPUS:
        raise ManagerError(f"Unsupported GPU {args.gpu}; choose from {', '.join(SUPPORTED_GPUS)}.")
    session_name = validate_session_name(args.session) if args.session else new_session_name(gpu)
    ask_confirmation(f"Create a Colab {gpu} session named {session_name}?", args.yes)

    profiles = ProfileRegistry(args.profiles_file).load()
    ready_profiles = [profile for profile in profiles if profile.ready]
    if not profiles:
        raise ManagerError("No saved Colab profiles are configured.")
    if not ready_profiles:
        raise ManagerError("No profile has both its token and session configuration files.")

    store = StateStore(args.state_file)
    state = store.load()
    if any(session.get("session") == session_name for session in state["sessions"]):
        raise ManagerError(f"Session name is already tracked: {session_name}")
    start_alias = resolve_start_alias(profiles, state, args.start_profile)
    ordered = rotate_profiles(profiles, start_alias)
    settings = SettingsStore(args.settings_file)
    runner = ColabProfileRunner(args.profile_command)
    timer = build_timer(args)
    skipped: list[str] = []

    for profile in ordered:
        if not profile.ready:
            skipped.append(f"{profile.alias}: missing token/session file")
            continue
        state = store.load()
        allowed, reason = available_for_budget(profile, gpu, state, settings)
        if not allowed:
            skipped.append(f"{profile.alias}: {reason}")
            _update_last_attempt(store, profile.alias, "local_budget_reached")
            continue

        print(f"Trying profile {profile.alias} for {gpu} session {session_name}...")
        result = runner.run(
            profile.alias,
            ["new", "--session", session_name, "--gpu", gpu],
            timeout=600,
        )
        if result.returncode == 0:
            print_command_output(result)
            try:
                record_created_session(
                    store,
                    profile.alias,
                    session_name,
                    gpu,
                    datetime.now(timezone.utc),
                )
            except ManagerError as error:
                raise ManagerError(
                    f"Colab created {session_name}, but local tracking could not be saved: {error}. "
                    "Do not retry creation until checking Colab sessions."
                ) from error
            start_timer_safely(timer)
            print(f"Tracked {session_name} under profile {profile.alias}.")
            return 0

        print_command_output(result)
        if result.timed_out:
            status = runner.run(profile.alias, ["status", "--session", session_name], timeout=60)
            if is_session_missing(status):
                raise ManagerError(
                    f"Creation timed out on profile {profile.alias}; Colab reported that session "
                    f"{session_name} does not exist. No session was tracked and no other profile "
                    "was tried. Check Colab sessions before retrying."
                )
            if status.returncode == 0:
                try:
                    record_created_session(store, profile.alias, session_name, gpu, datetime.now(timezone.utc))
                except ManagerError as error:
                    raise ManagerError(
                        f"Session status succeeded after a create timeout, but tracking failed: {error}. "
                        "Do not retry creation until checking Colab sessions."
                    ) from error
                start_timer_safely(timer)
                print(f"Create command timed out, but {session_name} is active and now tracked.")
                return 0
            raise ManagerError(
                f"Creation timed out on profile {profile.alias}; the remote result is uncertain. "
                "No other profile was tried to avoid creating duplicate VMs. Check Colab sessions first."
            )

        _update_last_attempt(
            store,
            profile.alias,
            "unavailable" if is_capacity_unavailable(result.combined_output) else "failed",
        )
        if is_capacity_unavailable(result.combined_output):
            skipped.append(f"{profile.alias}: Colab reported quota/capacity unavailable")
            print(f"Profile {profile.alias} is unavailable for this request; trying the next profile.")
            continue
        raise ManagerError(
            f"Creation failed on profile {profile.alias}; the error was not a recognized quota/capacity denial. "
            "Automatic failover stopped."
        )

    skipped_text = "; ".join(skipped) if skipped else "no eligible profile"
    raise ManagerError(f"No profile could create the requested session. Checked: {skipped_text}")


def find_tracked_session(state: dict[str, Any], session_name: str) -> dict[str, Any]:
    """Find a uniquely tracked session by its explicit local name."""
    matches = [session for session in state["sessions"] if session.get("session") == session_name]
    if not matches:
        raise ManagerError(f"Session is not tracked by this skill: {session_name}")
    if len(matches) > 1:
        raise ManagerError(f"Session name is ambiguous in local state: {session_name}")
    return matches[0]


def connect_session(args: argparse.Namespace) -> int:
    """Open the Colab page only for a tracked session that still exists."""
    session_name = validate_session_name(args.session)
    store = StateStore(args.state_file)
    session = find_tracked_session(store.load(), session_name)
    if session.get("ended_at") is not None:
        raise ManagerError(f"Tracked session has ended: {session_name}; no replacement VM was created.")
    runner = ColabProfileRunner(args.profile_command)
    status = runner.run(session["profile"], ["status", "--session", session_name], timeout=60)
    if is_session_missing(status):
        finish_session(store, session["profile"], session_name, datetime.now(timezone.utc))
        raise ManagerError(f"Colab session {session_name} no longer exists; no replacement VM was created.")
    if status.returncode != 0:
        raise ManagerError(
            f"Could not verify session {session_name} due to an authentication or network error. "
            "No replacement VM was created."
        )
    result = runner.run(session["profile"], ["url", "--session", session_name, "--open"], timeout=60)
    print_command_output(result)
    if result.returncode != 0:
        raise ManagerError(f"Could not open the Colab session {session_name}.")
    return 0


def stop_session(args: argparse.Namespace) -> int:
    """Ask before terminating the tracked VM, then close its local timer interval."""
    session_name = validate_session_name(args.session)
    store = StateStore(args.state_file)
    session = find_tracked_session(store.load(), session_name)
    if session.get("ended_at") is not None:
        raise ManagerError(f"Session is already marked ended: {session_name}")
    ask_confirmation(
        f"Stop Colab VM {session_name} on profile {session['profile']}? This terminates the VM.",
        args.yes,
    )
    runner = ColabProfileRunner(args.profile_command)
    result = runner.run(session["profile"], ["stop", "--session", session_name], timeout=120)
    print_command_output(result)
    if result.returncode != 0:
        raise ManagerError(f"Colab did not confirm that session {session_name} stopped.")
    finish_session(store, session["profile"], session_name, datetime.now(timezone.utc))
    if not active_sessions(store.load()):
        build_timer(args).stop_when_idle()
    return 0


def run_usage(args: argparse.Namespace) -> int:
    """Show upstream compute-unit usage without interpreting it as remaining time."""
    profiles = ProfileRegistry(args.profiles_file).load()
    ready = [profile for profile in profiles if profile.ready]
    if not ready:
        raise ManagerError("No profile has both its token and session configuration files.")
    state = StateStore(args.state_file).load()
    alias = args.profile or resolve_start_alias(profiles, state, None)
    profile = next((item for item in ready if item.alias == alias), None)
    if profile is None:
        raise ManagerError(f"Profile is not ready by file check: {alias}")
    result = ColabProfileRunner(args.profile_command).run(profile.alias, ["usage"], timeout=60)
    print_command_output(result)
    if result.returncode == 0:
        print("Colab usage reports compute units, not remaining free GPU hours.")
    return result.returncode


def set_budget(args: argparse.Namespace) -> int:
    """Set a default or per-profile daily estimate."""
    if args.profile:
        profiles = ProfileRegistry(args.profiles_file).load()
        if not any(profile.alias == args.profile for profile in profiles):
            raise ManagerError(f"Profile alias not found: {args.profile}")
    SettingsStore(args.settings_file).set_budget(args.gpu.upper(), args.minutes, args.profile)
    scope = f"profile {args.profile}" if args.profile else "all profiles by default"
    print(f"Set {args.gpu.upper()} daily estimate to {args.minutes} minutes for {scope}.")
    print("This is a local estimate, not a Colab-reported quota.")
    return 0


def show_budgets(args: argparse.Namespace) -> int:
    """Display configured daily estimates."""
    data = SettingsStore(args.settings_file).load()
    print("Default daily estimates:")
    for gpu, minutes in sorted(data["default_daily_budget_minutes"].items()):
        print(f"  {gpu}: {minutes} minutes")
    overrides = data["profile_daily_budget_minutes"]
    if overrides:
        print("Profile overrides:")
        for profile, budgets in overrides.items():
            for gpu, minutes in sorted(budgets.items()):
                print(f"  {profile}/{gpu}: {minutes} minutes")
    print("These values estimate tracked runtime only; they are not official Colab quotas.")
    return 0


def poll_tracked_sessions(
    store: StateStore,
    runner: ColabProfileRunner,
    now: datetime,
) -> tuple[dict[str, Any], list[str]]:
    """Close timers only after an explicit remote missing-session response."""
    state = store.load()
    ended: list[str] = []
    for session in active_sessions(state):
        result = runner.run(
            session["profile"],
            ["status", "--session", session["session"]],
            timeout=60,
        )
        if is_session_missing(result):
            if finish_session(store, session["profile"], session["session"], now):
                ended.append(session["session"])
            continue
        if result.returncode == 0:
            continue
        print(
            f"Timer could not verify {session['profile']}/{session['session']}; "
            "it remains tracked until Colab can be checked.",
            file=sys.stderr,
        )
    return store.load(), ended


def notification_message(profile: str, gpu: str, remaining_seconds: float, budget: int) -> tuple[str, str]:
    """Build a concise notification for the next threshold reached."""
    if remaining_seconds <= 0:
        return f"{gpu} estimate reached", f"{profile}: tracked runtime reached the {budget}-minute daily estimate."
    remaining_minutes = max(1, int((remaining_seconds + 59) // 60))
    return (
        f"{gpu} time estimate",
        f"{profile}: about {remaining_minutes} tracked minutes remain today (estimate).",
    )


def send_notification(command: str, title: str, body: str) -> None:
    """Send a desktop notification, falling back to the service journal."""
    executable = shutil.which(command) if command else None
    if executable is None:
        print(f"{title}: {body}")
        return
    try:
        result = subprocess.run(
            [executable, "--app-name=Colab Manager", title, body],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Desktop notification failed ({error}); {title}: {body}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"Desktop notification failed; {title}: {body}", file=sys.stderr)


def notify_reached_thresholds(
    store: StateStore,
    settings: SettingsStore,
    notification_command: str,
    now: datetime,
) -> None:
    """Notify once per threshold crossed for each profile/GPU/local day."""
    local_day = now.astimezone(LOCAL_TIMEZONE).date().isoformat()
    state = store.load()
    known_profiles = {
        session.get("profile")
        for session in state["sessions"]
        if isinstance(session.get("profile"), str)
    }
    for profile in sorted(known_profiles):
        known_gpus = {
            session.get("gpu")
            for session in state["sessions"]
            if session.get("profile") == profile and isinstance(session.get("gpu"), str)
        }
        for gpu in sorted(known_gpus):
            budget = settings.get_budget(profile, gpu)
            if budget is None:
                continue
            used = daily_usage_seconds(state, profile, gpu, now)
            remaining = max(0.0, budget * 60 - used)
            reached = [threshold for threshold in ALERT_THRESHOLDS_MINUTES if remaining <= threshold * 60]
            if not reached:
                continue
            alert_key = f"{profile}|{gpu}|{local_day}"
            old_alerts = state["alerts"].get(alert_key, [])
            if not isinstance(old_alerts, list):
                old_alerts = []
            new_alerts = [threshold for threshold in reached if threshold not in old_alerts]
            if not new_alerts:
                continue
            most_urgent = min(new_alerts)

            def mark_alerts(current: dict[str, Any]) -> None:
                values = current["alerts"].setdefault(alert_key, [])
                for threshold in reached:
                    if threshold not in values:
                        values.append(threshold)
                values.sort()

            store.update(mark_alerts)
            title, body = notification_message(profile, gpu, remaining, budget)
            if most_urgent != 0:
                title = f"{gpu}: {most_urgent}-minute warning"
            send_notification(notification_command, title, body)
            state = store.load()


def watch_sessions(args: argparse.Namespace) -> int:
    """Run one background monitoring tick and disable the timer when idle."""
    store = StateStore(args.state_file)
    runner = ColabProfileRunner(args.profile_command)
    now = datetime.now(timezone.utc)
    state, ended = poll_tracked_sessions(store, runner, now)
    for session_name in ended:
        print(f"Colab session ended: {session_name}")
    notify_reached_thresholds(
        store,
        SettingsStore(args.settings_file),
        args.notification_command,
        now,
    )
    if not active_sessions(state):
        build_timer(args).stop_when_idle()
        print("No tracked active Colab sessions; background timer stopped.")
    return 0


def timer_command(args: argparse.Namespace) -> int:
    """Install, remove, or inspect the per-user systemd timer."""
    timer = build_timer(args)
    if args.timer_action == "install":
        timer.install()
        print(f"Installed timer units in {timer.unit_dir}; they are not running until a session is tracked.")
        return 0
    if args.timer_action == "uninstall":
        if active_sessions(StateStore(args.state_file).load()):
            ask_confirmation(
                "Remove the background timer while tracked sessions are active? "
                "The VMs will keep running without notifications.",
                args.yes,
            )
        timer.uninstall()
        print("Removed Colab Manager timer units; tracked VMs were not stopped.")
        return 0
    installed, enabled, active = timer.status()
    state = StateStore(args.state_file).load()
    if not installed:
        print(f"Timer units: not installed; tracked sessions: {len(active_sessions(state))}")
        return 0
    print(
        f"Timer units: {'installed' if installed else 'not installed'}; "
        f"{enabled}; {active}; tracked sessions: {len(active_sessions(state))}"
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    """Build the standalone helper CLI."""
    parser = argparse.ArgumentParser(
        prog="colab-manager",
        description="Manage saved Colab profiles and local runtime-time estimates.",
    )
    parser.add_argument("--profiles-file", type=Path, default=default_profiles_path())
    parser.add_argument("--state-file", type=Path, default=default_state_path())
    parser.add_argument("--settings-file", type=Path, default=default_settings_path())
    parser.add_argument("--unit-dir", type=Path, default=default_unit_dir())
    parser.add_argument("--profile-command", default=default_profile_command())
    parser.add_argument("--systemctl-command", default=default_systemctl_command())
    parser.add_argument("--notification-command", default=default_notification_command())
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("profiles", help="List profile order and local file readiness.")
    commands.add_parser("status", help="Show tracked elapsed time and per-profile daily estimates.")

    usage_parser = commands.add_parser("usage", help="Show Colab compute-unit usage for one saved profile.")
    usage_parser.add_argument("--profile", help="Profile alias; defaults to the last successful profile.")

    create_parser = commands.add_parser("create", help="Create a new GPU session with bounded profile failover.")
    create_parser.add_argument("--gpu", default="T4", choices=SUPPORTED_GPUS, type=str.upper)
    create_parser.add_argument("--session", help="Session name; a unique name is generated if omitted.")
    create_parser.add_argument("--start-profile", help="Begin profile rotation from this alias.")
    create_parser.add_argument("--yes", action="store_true", help="Confirm creation after user authorization.")

    connect_parser = commands.add_parser("connect", help="Open the web UI for a tracked active session.")
    connect_parser.add_argument("--session", required=True)

    stop_parser = commands.add_parser("stop", help="Terminate a tracked session after confirmation.")
    stop_parser.add_argument("--session", required=True)
    stop_parser.add_argument("--yes", action="store_true", help="Confirm VM termination after user authorization.")

    budget_parser = commands.add_parser("budget", help="Show or change local daily runtime estimates.")
    budget_commands = budget_parser.add_subparsers(dest="budget_action", required=True)
    budget_commands.add_parser("show", help="Show configured budgets.")
    set_parser = budget_commands.add_parser("set", help="Set the default or profile-specific estimate.")
    set_parser.add_argument("--gpu", required=True, choices=SUPPORTED_GPUS, type=str.upper)
    set_parser.add_argument("--minutes", required=True, type=int)
    set_parser.add_argument("--profile", help="Set an override for a single profile alias.")

    timer_parser = commands.add_parser("timer", help="Manage the user-level background timer.")
    timer_parser.add_argument("timer_action", choices=("status", "install", "uninstall"))
    timer_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm removal while tracked sessions are active.",
    )

    watch_parser = commands.add_parser("watch", help=argparse.SUPPRESS)
    watch_parser.set_defaults(command="watch")
    return parser


def run_command(args: argparse.Namespace) -> int:
    """Dispatch one public helper command."""
    if args.command == "profiles":
        print_profiles(args)
        return 0
    if args.command == "status":
        print_status(args)
        return 0
    if args.command == "usage":
        return run_usage(args)
    if args.command == "create":
        return create_session(args)
    if args.command == "connect":
        return connect_session(args)
    if args.command == "stop":
        return stop_session(args)
    if args.command == "budget":
        if args.budget_action == "show":
            return show_budgets(args)
        return set_budget(args)
    if args.command == "timer":
        return timer_command(args)
    if args.command == "watch":
        return watch_sessions(args)
    raise ManagerError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Colab Manager helper."""
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except ManagerError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
