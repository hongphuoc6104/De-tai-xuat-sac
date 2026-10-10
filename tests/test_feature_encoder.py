"""Focused tests for pretrained weight integrity and frozen feature inference."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from histology_data import feature_encoder


def test_import_keeps_torch_dependencies_lazy() -> None:
    code = (
        "import sys; import histology_data.feature_encoder; "
        "assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (MemoryError("allocation failed"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 8 GiB"), True),
        (RuntimeError("ordinary model failure"), False),
        (ValueError("invalid image"), False),
    ],
)
def test_is_oom_classifies_memory_failures(exception: BaseException, expected: bool) -> None:
    assert feature_encoder.is_oom(exception) is expected


def test_missing_weights_with_download_disabled_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download=True"):
        feature_encoder._ensure_weights(SimpleNamespace(), tmp_path, download=False)


def test_checkpoint_download_is_hash_checked_and_cached_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = b"synthetic cache fixture; not model weights"
    expected_sha256 = hashlib.sha256(checkpoint).hexdigest()
    monkeypatch.setattr(
        feature_encoder,
        "OFFICIAL_WEIGHTS_HASH_PREFIX",
        expected_sha256[:8],
    )
    # The fixture exercises cache mechanics only; it is not a model checkpoint.
    monkeypatch.setattr(feature_encoder, "OFFICIAL_WEIGHTS_SHA256", expected_sha256)
    download_calls: list[tuple[str, str, str | None]] = []

    def download_url_to_file(
        url: str,
        destination: str,
        *,
        hash_prefix: str | None,
        progress: bool,
    ) -> None:
        assert progress is True
        download_calls.append((url, destination, hash_prefix))
        Path(destination).write_bytes(checkpoint)

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(download_url_to_file=download_url_to_file),
    )
    weights_path, sha256 = feature_encoder._ensure_weights(fake_torch, tmp_path, download=True)
    assert sha256 == expected_sha256
    assert weights_path.read_bytes() == checkpoint
    manifest = json.loads(
        (weights_path.parent / feature_encoder.WEIGHTS_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["sha256"] == expected_sha256
    assert manifest["url"] == feature_encoder.OFFICIAL_WEIGHTS_URL
    assert len(download_calls) == 1
    download_url, download_destination, download_prefix = download_calls[0]
    assert download_url == feature_encoder.OFFICIAL_WEIGHTS_URL
    assert Path(download_destination).name == feature_encoder.WEIGHTS_FILENAME
    assert download_prefix == expected_sha256[:8]

    def unexpected_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("a verified immutable cache must not be downloaded again")

    cached_path, cached_sha256 = feature_encoder._ensure_weights(
        SimpleNamespace(hub=SimpleNamespace(download_url_to_file=unexpected_download)),
        tmp_path,
        download=True,
    )
    assert cached_path == weights_path
    assert cached_sha256 == expected_sha256


def test_corrupt_cache_fails_closed_without_redownload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = b"synthetic cache fixture; not model weights"
    expected_sha256 = hashlib.sha256(checkpoint).hexdigest()
    monkeypatch.setattr(
        feature_encoder,
        "OFFICIAL_WEIGHTS_HASH_PREFIX",
        expected_sha256[:8],
    )
    # The fixture exercises corruption handling only; it is not model weights.
    monkeypatch.setattr(feature_encoder, "OFFICIAL_WEIGHTS_SHA256", expected_sha256)
    cache_dir = tmp_path / feature_encoder.WEIGHTS_CACHE_NAME
    cache_dir.mkdir()
    weights_path = cache_dir / feature_encoder.WEIGHTS_FILENAME
    weights_path.write_bytes(checkpoint)
    manifest = {
        "schema_version": feature_encoder.WEIGHTS_MANIFEST_VERSION,
        "model": "resnet50",
        "weights": feature_encoder.WEIGHTS_ENUM,
        "url": feature_encoder.OFFICIAL_WEIGHTS_URL,
        "filename": feature_encoder.WEIGHTS_FILENAME,
        "sha256": expected_sha256,
    }
    (cache_dir / feature_encoder.WEIGHTS_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    weights_path.write_bytes(checkpoint + b" corrupted")

    def unexpected_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("corrupt cache must fail closed, not silently replace weights")

    with pytest.raises(ValueError, match="manifest SHA-256"):
        feature_encoder._ensure_weights(
            SimpleNamespace(hub=SimpleNamespace(download_url_to_file=unexpected_download)),
            tmp_path,
            download=True,
        )


def test_cache_rejects_alternate_full_sha_with_official_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a self-consistent manifest whose hash only matches the 8-char prefix."""
    cache_dir = tmp_path / feature_encoder.WEIGHTS_CACHE_NAME
    cache_dir.mkdir()
    weights_path = cache_dir / feature_encoder.WEIGHTS_FILENAME
    weights_path.write_bytes(b"synthetic cache bytes; not model weights")
    alternate_sha256 = feature_encoder.OFFICIAL_WEIGHTS_HASH_PREFIX + "0" * 56
    assert alternate_sha256 != feature_encoder.OFFICIAL_WEIGHTS_SHA256
    manifest = {
        "schema_version": feature_encoder.WEIGHTS_MANIFEST_VERSION,
        "model": "resnet50",
        "weights": feature_encoder.WEIGHTS_ENUM,
        "url": feature_encoder.OFFICIAL_WEIGHTS_URL,
        "filename": feature_encoder.WEIGHTS_FILENAME,
        "sha256": alternate_sha256,
    }
    (cache_dir / feature_encoder.WEIGHTS_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    # Stub the file hash so this models a file/manifest pair that agrees with
    # each other. The test does not assert that such a real checkpoint exists.
    monkeypatch.setattr(feature_encoder, "_sha256_file", lambda _path: alternate_sha256)

    with pytest.raises(ValueError, match="pinned official checkpoint"):
        feature_encoder._read_verified_cache(tmp_path)


def test_amp_fp16_requires_cuda_before_loading_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: SimpleNamespace(type=name),
    )
    fake_weights = SimpleNamespace(IMAGENET1K_V1=SimpleNamespace(url=feature_encoder.OFFICIAL_WEIGHTS_URL))
    monkeypatch.setattr(
        feature_encoder,
        "_load_ml_dependencies",
        lambda: (fake_torch, SimpleNamespace(__version__="fake"), None, fake_weights),
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        feature_encoder.ResNet50Encoder(
            Path("unused"),
            device="cpu",
            precision="amp_fp16",
            download=False,
        )


def test_official_checkpoint_produces_frozen_finite_features() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    weights_dir = Path(
        os.environ.get("HISTOLOGY_FEATURE_TEST_WEIGHTS", "/tmp/histology-feature-weights"),
    )
    cache_dir = weights_dir / feature_encoder.WEIGHTS_CACHE_NAME
    if not cache_dir.is_dir():
        pytest.skip("official ResNet-50 V1 checkpoint is not cached")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        encoder = feature_encoder.ResNet50Encoder(
            weights_dir,
            device="cpu",
            precision="fp32",
            download=False,
        )
        from PIL import Image

        inputs = [
            Image.new("RGB", (240, 180), color=(150, 80, 110)),
            Image.new("L", (260, 220), color=180),
        ]
        grad_modes: list[bool] = []
        handle = encoder._model.register_forward_hook(
            lambda _module, _inputs, _output: grad_modes.append(torch.is_grad_enabled()),
        )
        try:
            features = encoder.encode(inputs)
        finally:
            handle.remove()

        assert features.shape == (2, 2048)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
        assert grad_modes == [False]
        assert encoder._model.training is False
        assert all(not parameter.requires_grad for parameter in encoder._model.parameters())
        descriptor = encoder.descriptor
        assert descriptor["name"] == "resnet50"
        assert descriptor["weights"] == feature_encoder.WEIGHTS_ENUM
        assert descriptor["weights_sha256"].startswith(feature_encoder.OFFICIAL_WEIGHTS_HASH_PREFIX)
        assert descriptor["preprocessing"] == {
            "input_color": "RGB",
            "transform": f"{feature_encoder.WEIGHTS_ENUM}.transforms()",
            "resize_size": [256],
            "crop_size": [224],
            "crop": "center",
            "interpolation": "bilinear",
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        assert descriptor["pillow_version"] == feature_encoder.PILLOW_VERSION
        assert "weights_dir" not in descriptor
        assert "device" not in descriptor
        assert "batch_size" not in descriptor
    finally:
        torch.set_num_threads(previous_threads)
