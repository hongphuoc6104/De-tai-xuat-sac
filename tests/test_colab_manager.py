"""Unit and local end-to-end checks for the Colab Manager helper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "colab-manager"
SCRIPT_PATH = SKILL_ROOT / "scripts" / "colab_manager.py"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import colab_manager  # noqa: E402


def make_registry(tmp_path: Path, aliases: list[str], incomplete: set[str] | None = None) -> Path:
    """Create isolated profile files containing dummy, non-secret test data."""
    incomplete_aliases = incomplete or set()
    profiles = []
    for alias in aliases:
        token_path = tmp_path / f"{alias}.token"
        session_path = tmp_path / f"{alias}.sessions.json"
        if alias not in incomplete_aliases:
            token_path.write_text("dummy test credential", encoding="utf-8")
            session_path.write_text("{}", encoding="utf-8")
        profiles.append(
            {
                "alias": alias,
                "token_path": str(token_path),
                "session_config": str(session_path),
            }
        )
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")
    return path


def write_executable(path: Path, content: str) -> Path:
    """Write an isolated executable stub for local end-to-end tests."""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)
    return path


def fake_profile_wrapper(path: Path) -> Path:
    """Create a colab-profile stub controlled entirely by environment variables."""
    content = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

alias, *args = sys.argv[1:]
with Path(os.environ["FAKE_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps([alias, *args]) + "\\n")

command = args[0] if args else ""
failures = json.loads(os.environ.get("FAKE_CAPACITY_FAILURES", "[]"))
if command == "new" and alias in failures:
    print("T4 accelerator is currently unavailable for this account", file=sys.stderr)
    raise SystemExit(1)
if command == "new" and os.environ.get("FAKE_NEW_ERROR"):
    print(os.environ["FAKE_NEW_ERROR"], file=sys.stderr)
    raise SystemExit(1)
if command == "status" and alias in json.loads(os.environ.get("FAKE_MISSING_SESSIONS", "[]")):
    print("Session not found (404)", file=sys.stderr)
    raise SystemExit(1)
if command == "new":
    print("created " + args[args.index("--session") + 1])
elif command == "usage":
    print("Current balance: 0.00 compute units")
elif command == "url":
    print("opened session")
elif command == "stop":
    print("stopped session")
elif command == "status":
    print("Session is active")
raise SystemExit(0)
"""
    return write_executable(path, content)


def fake_systemctl(path: Path) -> Path:
    """Create a systemctl stub that accepts user-timer lifecycle calls."""
    content = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

with Path(os.environ["FAKE_SYSTEMCTL_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(" ".join(sys.argv[1:]) + "\\n")
raise SystemExit(0)
    """
    return write_executable(path, content)


def fake_notification(path: Path) -> Path:
    """Create a notification stub so tests never reach the desktop."""
    content = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["FAKE_NOTIFICATIONS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
raise SystemExit(0)
"""
    return write_executable(path, content)


def run_manager(
    tmp_path: Path,
    *arguments: str,
    profiles_path: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real helper against isolated state and fake external commands."""
    profile_path = profiles_path or make_registry(tmp_path, ["primary", "alternate"])
    fake = fake_profile_wrapper(tmp_path / "colab-profile")
    systemctl = fake_systemctl(tmp_path / "systemctl")
    notification = fake_notification(tmp_path / "notify-send")
    calls_path = tmp_path / "profile-calls.jsonl"
    systemctl_calls = tmp_path / "systemctl-calls.txt"
    notification_calls = tmp_path / "notification-calls.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_CALLS": str(calls_path),
            "FAKE_SYSTEMCTL_CALLS": str(systemctl_calls),
            "FAKE_NOTIFICATIONS": str(notification_calls),
            "PATH": f"{tmp_path}:{env.get('PATH', '')}",
        }
    )
    if environment:
        env.update(environment)
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--profiles-file",
        str(profile_path),
        "--state-file",
        str(tmp_path / "state" / "state.json"),
        "--settings-file",
        str(tmp_path / "config" / "settings.json"),
        "--unit-dir",
        str(tmp_path / "systemd"),
        "--profile-command",
        str(fake),
        "--systemctl-command",
        str(systemctl),
        "--notification-command",
        str(notification),
        *arguments,
    ]
    return subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=15)


def read_profile_calls(path: Path) -> list[list[str]]:
    """Load fake wrapper invocations in their observed order."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rotation_wraps_in_registry_order_and_keeps_incomplete_slot_skipped(tmp_path: Path):
    """The account ring follows the registry and excludes incomplete profiles."""
    registry = make_registry(tmp_path, ["primary", "alternate", "backup", "newaccount"], {"backup"})
    profiles = colab_manager.ProfileRegistry(registry).load()
    rotated = colab_manager.rotate_profiles(profiles, "newaccount")
    ready = [profile.alias for profile in rotated if profile.ready]
    assert ready == ["newaccount", "primary", "alternate"]
    assert [profile.alias for profile in rotated][3] == "backup"


def test_daily_usage_splits_runtime_at_local_midnight():
    """Runtime that crosses midnight is counted only after the current local midnight."""
    now = datetime(2026, 10, 9, 17, 30, tzinfo=timezone.utc)
    state = {
        "sessions": [
            {
                "profile": "primary",
                "session": "night-run",
                "gpu": "T4",
                "started_at": "2026-10-09T16:30:00Z",
                "ended_at": None,
            }
        ]
    }
    seconds = colab_manager.daily_usage_seconds(state, "primary", "T4", now)
    assert seconds == 30 * 60


def test_capacity_classifier_only_accepts_explicit_resource_denials():
    """Authentication, network, and unknown errors must not trigger account rotation."""
    assert colab_manager.is_capacity_unavailable("T4 accelerator is currently unavailable")
    assert colab_manager.is_capacity_unavailable("RESOURCE_EXHAUSTED: daily quota exceeded")
    assert not colab_manager.is_capacity_unavailable("401 Unauthorized: invalid credentials")
    assert not colab_manager.is_capacity_unavailable("401: resource capacity unavailable")
    assert not colab_manager.is_capacity_unavailable("connection timed out")
    assert not colab_manager.is_capacity_unavailable("unexpected traceback")


def test_recording_success_persists_last_profile_use_without_credentials(tmp_path: Path):
    """Successful session metadata is persisted without copying token contents."""
    token_path = tmp_path / "token.json"
    token_path.write_text("do-not-copy-this-secret", encoding="utf-8")
    registry = make_registry(tmp_path, ["primary"])
    store = colab_manager.StateStore(tmp_path / "state" / "state.json")
    started = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    colab_manager.record_created_session(store, "primary", "train-run", "T4", started)
    raw_state = store.path.read_text(encoding="utf-8")
    state = store.load()
    assert state["last_successful_profile"] == "primary"
    assert state["profile_activity"]["primary"]["last_used_at"] == "2026-10-09T12:00:00Z"
    assert state["sessions"][0]["session"] == "train-run"
    assert "do-not-copy-this-secret" not in raw_state
    assert colab_manager.ProfileRegistry(registry).load()[0].ready


def test_create_fails_over_on_capacity_and_tracks_only_successful_profile(tmp_path: Path):
    """A local E2E flow tries aliases in order, skips incomplete files, then starts timer."""
    registry = make_registry(tmp_path, ["primary", "alternate", "backup", "newaccount"], {"backup"})
    result = run_manager(
        tmp_path,
        "create",
        "--gpu",
        "T4",
        "--session",
        "t4-proof",
        "--yes",
        profiles_path=registry,
        environment={"FAKE_CAPACITY_FAILURES": json.dumps(["primary"])},
    )
    assert result.returncode == 0, result.stderr
    assert "Tracked t4-proof under profile alternate" in result.stdout
    calls = read_profile_calls(tmp_path / "profile-calls.jsonl")
    create_aliases = [call[0] for call in calls if call[1] == "new"]
    assert create_aliases == ["primary", "alternate"]
    state_path = tmp_path / "state" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_successful_profile"] == "alternate"
    assert state["sessions"][0]["profile"] == "alternate"
    assert (tmp_path / "systemd" / "colab-manager.timer").is_file()
    assert "colab_manager.py" in (tmp_path / "systemd" / "colab-manager.service").read_text(encoding="utf-8")
    systemctl_calls = (tmp_path / "systemctl-calls.txt").read_text(encoding="utf-8")
    assert "enable --now colab-manager.timer" in systemctl_calls


def test_create_stops_on_auth_or_unknown_error_without_trying_next_profile(tmp_path: Path):
    """A failed auth/config request cannot silently switch identities."""
    result = run_manager(
        tmp_path,
        "create",
        "--gpu",
        "T4",
        "--session",
        "auth-failure",
        "--yes",
        environment={"FAKE_NEW_ERROR": "401 Unauthorized: invalid credentials"},
    )
    assert result.returncode == 2
    calls = read_profile_calls(tmp_path / "profile-calls.jsonl")
    assert [call[0] for call in calls if call[1] == "new"] == ["primary"]
    state_path = tmp_path / "state" / "state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["sessions"] == []


def test_connect_missing_session_never_creates_replacement(tmp_path: Path):
    """Reconnect reports a disappeared VM and does not allocate a replacement."""
    state_path = tmp_path / "state" / "state.json"
    store = colab_manager.StateStore(state_path)
    colab_manager.record_created_session(
        store,
        "primary",
        "gone-session",
        "T4",
        datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
    )
    registry = make_registry(tmp_path, ["primary"])
    result = run_manager(
        tmp_path,
        "connect",
        "--session",
        "gone-session",
        profiles_path=registry,
        environment={"FAKE_MISSING_SESSIONS": json.dumps(["primary"])},
    )
    assert result.returncode == 2
    assert "no replacement VM was created" in result.stderr
    calls = read_profile_calls(tmp_path / "profile-calls.jsonl")
    assert [call[1] for call in calls] == ["status"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["sessions"][0]["ended_at"] is not None


def test_stop_requires_confirmation_before_remote_command(tmp_path: Path):
    """A non-interactive unapproved stop must never reach colab stop."""
    state_path = tmp_path / "state" / "state.json"
    store = colab_manager.StateStore(state_path)
    colab_manager.record_created_session(
        store,
        "primary",
        "active-session",
        "T4",
        datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
    )
    registry = make_registry(tmp_path, ["primary"])
    result = run_manager(
        tmp_path,
        "stop",
        "--session",
        "active-session",
        profiles_path=registry,
    )
    assert result.returncode == 2
    assert "after user authorization" in result.stderr
    assert read_profile_calls(tmp_path / "profile-calls.jsonl") == []


def test_timer_warns_once_at_nearest_daily_threshold_without_stopping_vm(tmp_path: Path):
    """The timer sends a local warning at five minutes remaining and never stops a VM."""
    state_path = tmp_path / "state" / "state.json"
    store = colab_manager.StateStore(state_path)
    now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    started = now - timedelta(minutes=345)
    colab_manager.record_created_session(store, "primary", "long-run", "T4", started)
    notifications: list[tuple[str, str]] = []
    original_which = colab_manager.shutil.which
    original_run = colab_manager.subprocess.run

    def fake_which(command: str) -> str | None:
        return "/fake/notify-send" if command == "notify-send" else original_which(command)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        notifications.append((command[-2], command[-1]))
        return subprocess.CompletedProcess(command, 0, "", "")

    colab_manager.shutil.which = fake_which
    colab_manager.subprocess.run = fake_run
    try:
        colab_manager.notify_reached_thresholds(
            store,
            colab_manager.SettingsStore(tmp_path / "config" / "settings.json"),
            "notify-send",
            now,
        )
        colab_manager.notify_reached_thresholds(
            store,
            colab_manager.SettingsStore(tmp_path / "config" / "settings.json"),
            "notify-send",
            now,
        )
    finally:
        colab_manager.shutil.which = original_which
        colab_manager.subprocess.run = original_run

    assert len(notifications) == 1
    assert notifications[0][0] == "T4: 5-minute warning"
    assert "about 5 tracked minutes remain" in notifications[0][1]
    calls_path = tmp_path / "profile-calls.jsonl"
    assert not calls_path.exists()


def test_budget_override_is_persisted_per_profile(tmp_path: Path):
    """A profile-specific budget overrides the default without changing it."""
    settings = colab_manager.SettingsStore(tmp_path / "config" / "settings.json")
    settings.set_budget("T4", 240, "alternate")
    assert settings.get_budget("alternate", "T4") == 240
    assert settings.get_budget("primary", "T4") == 350


def test_timer_install_is_idempotent_and_does_not_start_timer(tmp_path: Path):
    """Installing user units reloads systemd but leaves the timer inactive."""
    registry = make_registry(tmp_path, ["primary"])
    result = run_manager(
        tmp_path,
        "timer",
        "install",
        profiles_path=registry,
    )
    assert result.returncode == 0, result.stderr
    assert "not running until a session is tracked" in result.stdout
    repeat = run_manager(
        tmp_path,
        "timer",
        "install",
        profiles_path=registry,
    )
    assert repeat.returncode == 0, repeat.stderr
    calls = (tmp_path / "systemctl-calls.txt").read_text(encoding="utf-8")
    assert calls.splitlines() == ["--user daemon-reload", "--user daemon-reload"]
    unit_text = (tmp_path / "systemd" / "colab-manager.timer").read_text(encoding="utf-8")
    assert "OnCalendar=*-*-* *:*:00" in unit_text
    assert "Persistent=true" in unit_text


def test_watch_notifies_without_stopping_an_active_vm(tmp_path: Path):
    """A reached local estimate sends a warning but leaves the remote VM running."""
    registry = make_registry(tmp_path, ["primary"])
    store = colab_manager.StateStore(tmp_path / "state" / "state.json")
    now = datetime.now(timezone.utc)
    _, local_day_start, _ = colab_manager.local_day_bounds(now)
    colab_manager.SettingsStore(tmp_path / "config" / "settings.json").set_budget("T4", 1, "primary")
    colab_manager.record_created_session(
        store,
        "primary",
        "monitored",
        "T4",
        local_day_start,
    )
    result = run_manager(tmp_path, "watch", profiles_path=registry)
    assert result.returncode == 0, result.stderr
    assert [call[1] for call in read_profile_calls(tmp_path / "profile-calls.jsonl")] == ["status"]
    notification_lines = (tmp_path / "notification-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(notification_lines) == 1
    assert not (tmp_path / "systemctl-calls.txt").exists()
    assert store.load()["sessions"][0]["ended_at"] is None


def test_timer_unit_resolves_commands_for_systemd_environment(tmp_path: Path):
    """Generated user units store the absolute helper and profile-wrapper paths."""
    timer = colab_manager.UserTimer(
        unit_dir=tmp_path / "units",
        systemctl_command="systemctl",
        profile_command="colab-profile",
        profiles_path=tmp_path / "profiles.json",
        state_path=tmp_path / "state.json",
        settings_path=tmp_path / "settings.json",
        notification_command="notify-send",
    )
    service, _ = timer.unit_contents()
    assert str(SCRIPT_PATH) in service
    assert "colab-profile" in service
    assert "WorkingDirectory=" in service
    workspace_path = str(SCRIPT_PATH.parents[4]).replace(" ", "\\x20")
    assert f"WorkingDirectory={workspace_path}" in service


def test_build_timer_resolves_profile_and_systemd_commands(monkeypatch, tmp_path: Path):
    """The installed user unit does not depend on an interactive shell PATH."""
    monkeypatch.setattr(colab_manager.shutil, "which", lambda command: f"/opt/bin/{command}")
    args = Namespace(
        unit_dir=tmp_path / "units",
        systemctl_command="systemctl",
        profile_command="colab-profile",
        profiles_file=tmp_path / "profiles.json",
        state_file=tmp_path / "state.json",
        settings_file=tmp_path / "settings.json",
        notification_command="notify-send",
    )
    timer = colab_manager.build_timer(args)
    assert timer.profile_command == "/opt/bin/colab-profile"
    assert timer.systemctl_command == "/opt/bin/systemctl"
    assert timer.notification_command == "/opt/bin/notify-send"


def test_timer_uninstall_needs_confirmation_when_a_tracked_vm_is_active(tmp_path: Path):
    """Removing background monitoring never stops or silently abandons a tracked VM."""
    registry = make_registry(tmp_path, ["primary"])
    state = colab_manager.StateStore(tmp_path / "state" / "state.json")
    colab_manager.record_created_session(
        state,
        "primary",
        "keep-running",
        "T4",
        datetime.now(timezone.utc),
    )
    refused = run_manager(tmp_path, "timer", "uninstall", profiles_path=registry)
    assert refused.returncode == 2
    assert "after user authorization" in refused.stderr
    assert not (tmp_path / "systemctl-calls.txt").exists()

    approved = run_manager(tmp_path, "timer", "uninstall", "--yes", profiles_path=registry)
    assert approved.returncode == 0, approved.stderr
    assert "tracked VMs were not stopped" in approved.stdout
    assert state.load()["sessions"][0]["ended_at"] is None
    calls = (tmp_path / "systemctl-calls.txt").read_text(encoding="utf-8")
    assert "--user disable --now colab-manager.timer" in calls


def test_systemd_units_validate_with_analyze(tmp_path: Path):
    """The generated unit accepts the real workspace path, including spaces."""
    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        pytest.skip("systemd-analyze is unavailable on this host")
    timer = colab_manager.UserTimer(
        unit_dir=tmp_path,
        systemctl_command="systemctl",
        profile_command="colab-profile",
        profiles_path=Path.home() / ".config" / "colab-cli" / "profiles.json",
        state_path=tmp_path / "state.json",
        settings_path=tmp_path / "settings.json",
        notification_command="notify-send",
    )
    service, timer_unit = timer.unit_contents()
    service_path = tmp_path / "colab-manager.service"
    timer_path = tmp_path / "colab-manager.timer"
    service_path.write_text(service, encoding="utf-8")
    timer_path.write_text(timer_unit, encoding="utf-8")
    result = subprocess.run(
        [systemd_analyze, "--user", "verify", str(service_path), str(timer_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
