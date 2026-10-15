"""End-to-end tests for the official public Google Drive checkpoint flow."""

from __future__ import annotations

import hashlib
import importlib
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from histology_data import image_classification_execution as execution, pathology_encoder

_CONFIRMATION_HTML = (
    '<form id="download-form" action="https://drive.google.com/uc?export=download">'
    '<input type="hidden" name="id" value="1Sn0kYwkXp7eqdIPUTFtK_ilz5ThApOsn">'
    '<input type="hidden" name="confirm" value="t">'
    '<input type="hidden" name="uuid" value="fixture-confirmation">'
    "</form>"
)


class _Response:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        text: str = "",
        chunks: tuple[bytes, ...] = (),
        status_code: int = 200,
        target: Path | None = None,
    ) -> None:
        self.url = url
        self.headers = headers
        self.text = text
        self._chunks = chunks
        self.status_code = status_code
        self._target = target
        self.iter_content_called = False
        self.closed = False

    def iter_content(self, *, chunk_size: int) -> Any:
        self.iter_content_called = True
        assert chunk_size > 0
        for chunk in self._chunks:
            if self._target is not None:
                assert not self._target.exists(), "final target appeared before verification"
            yield chunk

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(
        self,
        *,
        payload: bytes,
        behavior: str,
        target: Path,
        reported_filename: str = "best_ckpt.pth",
    ) -> None:
        self.payload = payload
        self.behavior = behavior
        self.target = target
        self.reported_filename = reported_filename
        self.cookies: list[object] = []
        self.requests: list[tuple[str, dict[str, str] | None]] = []
        self.responses: list[_Response] = []
        self._body_requested = False
        self.closed = False

    def get(
        self,
        url: str,
        *,
        stream: bool,
        verify: bool,
        headers: dict[str, str] | None = None,
    ) -> _Response:
        assert stream is True
        assert verify is True
        self.requests.append((url, headers))
        if "confirm=" not in url:
            response = _Response(
                url=url,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=_CONFIRMATION_HTML,
            )
            self.responses.append(response)
            return response

        self._body_requested = True
        disposition = f'attachment; filename="{self.reported_filename}"'
        if headers is not None and "Range" in headers:
            start = int(headers["Range"].removeprefix("bytes=").removesuffix("-"))
            response = _Response(
                url=url,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": disposition,
                    "Content-Length": str(len(self.payload) - start),
                },
                chunks=(self.payload[start:],),
                status_code=206,
                target=self.target,
            )
            self.responses.append(response)
            return response

        if self.behavior == "truncated":
            prefix = self.payload[: max(1, len(self.payload) // 2)]
            response = _Response(
                url=url,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": disposition,
                    "Content-Length": str(len(self.payload)),
                },
                chunks=(prefix,),
                target=self.target,
            )
            self.responses.append(response)
            return response
        response = _Response(
            url=url,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": disposition,
                "Content-Length": str(len(self.payload)),
            },
            chunks=(self.payload,),
            target=self.target,
        )
        self.responses.append(response)
        return response

    def close(self) -> None:
        self.closed = True


def _install_gdown_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: bytes,
    target: Path,
    behaviors: list[str],
    reported_filename: str = "best_ckpt.pth",
) -> list[_Session]:
    gdown = pytest.importorskip("gdown")
    assert gdown.__version__ == "6.2.0"
    gdown_module = importlib.import_module("gdown.download")
    behavior_queue = deque(behaviors)
    sessions: list[_Session] = []

    def fake_get_session(
        *,
        proxy: str | None,
        use_cookies: bool,
        user_agent: str,
        cookies_file: str | None,
    ) -> tuple[_Session, str]:
        assert proxy is None
        assert use_cookies is False
        assert user_agent
        session = _Session(
            payload=payload,
            behavior=behavior_queue.popleft(),
            target=target,
            reported_filename=reported_filename,
        )
        sessions.append(session)
        return session, str(cookies_file or "unused-cookies.txt")

    def fail_if_cookies_are_read(**_kwargs: Any) -> None:
        pytest.fail("public checkpoint acquisition must not read cookies")

    monkeypatch.setattr(gdown_module, "_get_session", fake_get_session)
    monkeypatch.setattr(gdown_module, "_load_cookies", fail_if_cookies_are_read)
    return sessions


def _acquire_fixture(
    target: Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int,
) -> dict[str, Any]:
    return pathology_encoder._acquire_official_weights_with_gdown(
        target,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )


def test_official_confirmation_download_returns_bound_receipt_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"tiny public RetCCL checkpoint fixture"
    # Fixture-only size injection keeps the production API under test without
    # fetching the author's 94 MB checkpoint.
    monkeypatch.setattr(pathology_encoder, "OFFICIAL_WEIGHTS_SIZE_BYTES", len(payload))
    target = tmp_path / "weights" / "best_ckpt.pth"
    sessions = _install_gdown_transport(
        monkeypatch,
        payload=payload,
        target=target,
        behaviors=["full", "full"],
    )

    receipt = pathology_encoder.acquire_official_weights_via_gdown(target)

    assert target.read_bytes() == payload
    assert receipt["source_file_id"] == pathology_encoder.OFFICIAL_WEIGHTS_FILE_ID
    assert receipt["source_url"] == pathology_encoder.OFFICIAL_WEIGHTS_URL
    assert receipt["source_folder_url"] == pathology_encoder.RETCCL_CHECKPOINT_FOLDER_URL
    assert receipt["source_revision"] == pathology_encoder.RETCCL_SOURCE_REVISION
    assert receipt["filename"] == "best_ckpt.pth"
    assert receipt["size_bytes"] == len(payload)
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["sha256_verification"] == "first_acquisition_tofu"
    assert receipt["acquisition"] == "official_public_drive_gdown"
    assert receipt["transport"] == {"name": "gdown", "version": "6.2.0"}
    assert receipt["cookies_used"] is False
    assert receipt["resume_enabled"] is True
    assert datetime.fromisoformat(receipt["receipt_created_at_utc"].replace("Z", "+00:00"))
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert all(
        session.requests[0][0]
        == f"https://drive.google.com/uc?id={pathology_encoder.OFFICIAL_WEIGHTS_FILE_ID}"
        for session in sessions
    )
    assert all(any("confirm=" in url for url, _headers in session.requests) for session in sessions)
    assert all(session.responses[-1].iter_content_called is False for session in sessions[:1])
    assert sessions[-1].responses[-1].iter_content_called is True
    assert list(target.parent.iterdir()) == [target]


def test_truncated_public_download_is_resumable_without_publishing_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"a public checkpoint large enough to resume"
    target = tmp_path / "best_ckpt.pth"
    sessions = _install_gdown_transport(
        monkeypatch,
        payload=payload,
        target=target,
        behaviors=["full", "truncated", "full", "full"],
    )

    with pytest.raises(URLError, match="Download is incomplete") as error:
        _acquire_fixture(target, expected_size_bytes=len(payload))
    assert execution.classify_retccl_failure(error.value) is execution.RetCCLFailureKind.REMOTE_UNAVAILABLE
    assert not target.exists()
    partials = list(target.parent.glob(".best_ckpt.pth.retccl-gdown*.part"))
    assert len(partials) == 1
    prefix_length = partials[0].stat().st_size
    assert 0 < prefix_length < len(payload)

    receipt = _acquire_fixture(target, expected_size_bytes=len(payload))

    assert target.read_bytes() == payload
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert any(
        headers == {"Range": f"bytes={prefix_length}-"}
        for session in sessions
        for _url, headers in session.requests
        if headers is not None
    )
    assert not list(target.parent.glob("*.part"))


def test_permission_confirmation_failure_is_optional_remote_and_creates_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("gdown")
    gdown_module = importlib.import_module("gdown.download")
    target = tmp_path / "missing" / "best_ckpt.pth"

    class PermissionSession:
        cookies: list[object] = []

        def get(self, url: str, *, stream: bool, verify: bool, headers: Any = None) -> _Response:
            assert stream is True and verify is True and headers is None
            return _Response(
                url=url,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="<html><body>Access denied</body></html>",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        gdown_module,
        "_get_session",
        lambda **kwargs: (PermissionSession(), "unused-cookies.txt"),
    )

    with pytest.raises(URLError) as error:
        _acquire_fixture(target, expected_size_bytes=10)

    assert execution.classify_retccl_failure(error.value) is execution.RetCCLFailureKind.REMOTE_UNAVAILABLE
    assert not target.parent.exists()


@pytest.mark.parametrize(
    "local_error",
    [OSError("atomic hard-link promotion is unsupported"), TimeoutError("local promotion timed out")],
)
def test_local_promotion_failure_remains_fatal_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_error: Exception,
) -> None:
    payload = b"small verified body"
    target = tmp_path / "best_ckpt.pth"
    _install_gdown_transport(
        monkeypatch,
        payload=payload,
        target=target,
        behaviors=["full", "full"],
    )

    def local_timeout(*_args: Any, **_kwargs: Any) -> None:
        raise local_error

    monkeypatch.setattr(pathology_encoder.os, "link", local_timeout)
    with pytest.raises(type(local_error), match=str(local_error)) as error:
        _acquire_fixture(target, expected_size_bytes=len(payload))

    assert execution.classify_retccl_failure(error.value) is execution.RetCCLFailureKind.FATAL
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_bad_existing_pin_is_never_replaced_or_adopted(tmp_path: Path) -> None:
    target = tmp_path / "best_ckpt.pth"
    original = b"local file with an untrusted digest"
    target.write_bytes(original)

    with pytest.raises(ValueError, match="caller-supplied SHA-256 pin"):
        _acquire_fixture(target, expected_sha256="0" * 64, expected_size_bytes=len(original))
    assert target.read_bytes() == original

    with pytest.raises(ValueError, match="requires its caller-supplied SHA-256 pin"):
        _acquire_fixture(target, expected_size_bytes=len(original))
    assert target.read_bytes() == original


def test_wrong_public_filename_fails_before_any_checkpoint_body_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "best_ckpt.pth"
    sessions = _install_gdown_transport(
        monkeypatch,
        payload=b"not the RetCCL checkpoint",
        target=target,
        behaviors=["full"],
        reported_filename="other.bin",
    )

    with pytest.raises(ValueError, match="unexpected filename"):
        _acquire_fixture(target, expected_size_bytes=25)

    assert not target.exists()
    assert len(sessions) == 1
    assert all(not response.iter_content_called for response in sessions[0].responses)
