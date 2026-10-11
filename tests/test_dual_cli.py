"""Controller gates and real-child lifecycle, without large source processing."""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from histology_data import dual_cli


def config(tmp_path):
    return {
        "study_id": "test-study",
        "source_code_sha256": "a" * 64,
        "setup_path": str(tmp_path / "setup.json"),
        "output_root": str(tmp_path / "results"),
    }


def test_config_and_deadline_fail_closed(tmp_path):
    c = config(tmp_path)
    dual_cli.validate_config(c)
    assert dual_cli.parse_deadline((datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()) > time.monotonic()
    with pytest.raises(ValueError, match="common16"):
        dual_cli.validate_config({**c, "cohort": "all18"})
    with pytest.raises(ValueError, match="safe"):
        dual_cli.validate_config({**c, "study_id": "../escape"})
    with pytest.raises(TimeoutError):
        dual_cli.parse_deadline("2000-01-01T00:00:00+00:00")


def test_sources_cannot_be_output_roots(tmp_path):
    source = tmp_path / "features"
    source.mkdir()
    data = SimpleNamespace(
        bundle=tmp_path / "bundle", feature_root=source, cache_path=tmp_path / "cache", identity={"cache_id": "c"}
    )
    with pytest.raises(ValueError, match="separate"):
        dual_cli._stores({**config(tmp_path), "output_root": str(source / "results")}, data)
    assert not (source / "results").exists()


@pytest.mark.parametrize("child_phase,expected", [("complete", "cpu_complete"), ("paused_budget", "paused_budget")])
def test_two_real_children_do_not_turn_pause_into_complete(tmp_path, monkeypatch, child_phase, expected):
    c = config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    monkeypatch.setattr(dual_cli, "run_preprocess", lambda *_: {"phase": "complete", "stage": "shared"})
    real_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        branch = command[command.index("--mode") + 1][-1]
        result = Path(c["output_root"]) / "scheduler" / f"worker_{branch}_result.json"
        code = (
            "import json;from pathlib import Path;Path("
            + repr(str(result))
            + ').write_text(json.dumps({"phase":'
            + repr(child_phase)
            + "}))"
        )
        process = real_popen([sys.executable, "-c", code], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(dual_cli.subprocess, "Popen", launch)
    result = dual_cli.run_cpu(
        c, path, time.monotonic() + 20, (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat()
    )
    assert result["phase"] == expected
    assert len(children) == 2 and all(p.poll() == 0 for p in children)
    if expected == "cpu_complete":
        assert result["pending_stages"] == ["T4_smoke", "B1", "final_report"]


def test_preflight_failure_is_error_not_running(tmp_path, monkeypatch):
    c = config(tmp_path)

    def fail(*args):
        raise OSError("fixture I/O failure")

    monkeypatch.setattr(dual_cli, "run_preprocess", fail)
    with pytest.raises(OSError, match="fixture"):
        dual_cli.run_cpu(c, tmp_path / "config.json", time.monotonic() + 20, "2099-01-01T00:00:00+00:00")
    result = json.loads((Path(c["output_root"]) / "scheduler/status.json").read_text())
    assert result["phase"] == "error" and result["error_type"] == "OSError"


def test_runtime_archive_pin_is_checked_against_actual_code(tmp_path):
    from scripts.build_dual_runtime import build_dual_runtime

    source = Path(dual_cli.__file__).resolve().parent.parent
    result = build_dual_runtime(source, tmp_path / "runtime.zip")
    c = {**config(tmp_path), "runtime_archive_path": result["path"], "source_code_sha256": result["sha256"]}
    dual_cli.validate_runtime_code(c)
    with pytest.raises(ValueError, match="SHA"):
        dual_cli.validate_runtime_code({**c, "source_code_sha256": "f" * 64})
    assert result["code_only"] is True
