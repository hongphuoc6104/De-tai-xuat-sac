"""Safe checkpoint and tiny-model contracts for the RetCCL image encoder."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from histology_data import pathology_encoder


def test_import_keeps_torch_dependencies_lazy() -> None:
    code = (
        "import sys; import histology_data.pathology_encoder; "
        "assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_official_metadata_records_author_revision_and_head_size() -> None:
    metadata = pathology_encoder.official_weights_metadata()
    assert metadata["source_revision"] == "a85b97226b62853b64143e8d535f5e10dae6cc89"
    assert metadata["source_files"]["get_feature.py"].endswith("/get_feature.py")
    assert metadata["weights_file_id"] == "1Sn0kYwkXp7eqdIPUTFtK_ilz5ThApOsn"
    assert metadata["weights_filename"] == "best_ckpt.pth"
    assert metadata["weights_size_bytes"] == 94_367_005
    assert metadata["remote_available_at_head_check"] is True
    assert metadata["weights_sha256"] is None
    assert metadata["weights_sha256_author_published"] is False


def test_head_probe_reports_binary_availability_without_fetching_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HeadResponse:
        status = 200
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="best_ckpt.pth"',
            "Content-Length": str(pathology_encoder.OFFICIAL_WEIGHTS_SIZE_BYTES),
        }

        def __enter__(self) -> HeadResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    observed_methods: list[str] = []

    def open_head(request: Any, *, timeout: float) -> HeadResponse:
        assert timeout == 5.0
        observed_methods.append(request.get_method())
        return HeadResponse()

    monkeypatch.setattr(pathology_encoder.urllib.request, "urlopen", open_head)
    report = pathology_encoder.probe_official_weights(timeout_seconds=5.0)
    assert observed_methods == ["HEAD"]
    assert report["remote_accessible"] is True
    assert report["remote_size_matches"] is True
    assert report["download_ready"] is True
    assert report["inference_ready"] is False
    assert report["readiness_status"] == "sha256_pin_required"


def test_local_readiness_requires_full_sha_pin_and_checks_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"synthetic RetCCL fixture; not pretrained weights"
    weights_path = tmp_path / "best_ckpt.pth"
    weights_path.write_bytes(content)
    monkeypatch.setattr(pathology_encoder, "OFFICIAL_WEIGHTS_SIZE_BYTES", len(content))
    expected_sha256 = hashlib.sha256(content).hexdigest()

    unpinned = pathology_encoder.checkpoint_readiness(
        weights_path,
        expected_sha256=None,
    )
    assert unpinned["ready"] is False
    assert unpinned["readiness_status"] == "sha256_pin_required"
    assert unpinned["observed_sha256"] == expected_sha256

    verified = pathology_encoder.checkpoint_readiness(
        weights_path,
        expected_sha256=expected_sha256,
    )
    assert verified["ready"] is True
    assert verified["readiness_status"] == "ready"
    assert verified["weights_sha256"] == expected_sha256

    mismatch = pathology_encoder.checkpoint_readiness(
        weights_path,
        expected_sha256="0" * 64,
    )
    assert mismatch["ready"] is False
    assert mismatch["readiness_status"] == "sha256_mismatch"


class _TinyEncoderModel:
    """Small torch model used to exercise safe load and feature inference."""

    def __new__(cls, torch: Any) -> Any:
        nn = torch.nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(3, pathology_encoder.FEATURE_DIM)
                self.fc = nn.Linear(pathology_encoder.FEATURE_DIM, 128)
                self.force_nonfinite = False
                self.grad_modes: list[bool] = []

            def forward(self, batch: Any) -> Any:
                self.grad_modes.append(torch.is_grad_enabled())
                if self.force_nonfinite:
                    return torch.full(
                        (batch.shape[0], pathology_encoder.FEATURE_DIM),
                        float("nan"),
                        dtype=torch.float32,
                        device=batch.device,
                    )
                pooled = batch.mean(dim=(2, 3))
                return self.fc(self.projection(pooled))

        return Model()


def _save_encoder_only_checkpoint(torch: Any, path: Path) -> str:
    torch.manual_seed(123)
    model = _TinyEncoderModel(torch)
    model.fc = torch.nn.Identity()
    torch.save(model.state_dict(), path)
    return pathology_encoder.sha256_file(path)


@pytest.fixture
def tiny_torchvision_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        pathology_encoder,
        "_load_ml_dependencies",
        lambda: (torch, SimpleNamespace(__version__="test-torchvision"), object()),
    )
    monkeypatch.setattr(
        pathology_encoder,
        "_build_model",
        lambda _torch, _models: _build_tiny_model(torch),
    )
    return torch


def _build_tiny_model(torch: Any) -> Any:
    model = _TinyEncoderModel(torch)
    model.fc = torch.nn.Identity()
    return model


def test_verified_checkpoint_reconstructs_frozen_finite_full_canvas_features(
    tiny_torchvision_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = tiny_torchvision_runtime
    weights_path = tmp_path / "best_ckpt.pth"
    expected_sha256 = _save_encoder_only_checkpoint(torch, weights_path)
    load_options: list[dict[str, Any]] = []
    strict_options: list[bool] = []
    original_torch_load = torch.load

    def checked_torch_load(*args: Any, **kwargs: Any) -> Any:
        load_options.append(dict(kwargs))
        return original_torch_load(*args, **kwargs)

    def build_test_model(_torch: Any, _models: Any) -> Any:
        model = _build_tiny_model(torch)
        original_load_state_dict = model.load_state_dict

        def checked_load_state_dict(state_dict: Any, *, strict: bool = True) -> Any:
            strict_options.append(strict)
            return original_load_state_dict(state_dict, strict=strict)

        model.load_state_dict = checked_load_state_dict
        return model

    monkeypatch.setattr(torch, "load", checked_torch_load)
    monkeypatch.setattr(pathology_encoder, "_build_model", build_test_model)
    encoder = pathology_encoder.RetCCLResNet50Encoder(
        weights_path,
        expected_sha256=expected_sha256,
        device="cpu",
        canvas_size=8,
    )
    images = [
        Image.new("RGB", (8, 8), color=(128, 64, 32)),
        Image.new("L", (8, 8), color=180),
    ]
    captured_inputs: list[Any] = []
    grad_modes: list[bool] = []
    handle = encoder._model.register_forward_pre_hook(
        lambda _model, inputs: captured_inputs.append(inputs[0].detach().clone()),
    )
    hook = encoder._model.register_forward_hook(
        lambda _model, _inputs, _output: grad_modes.append(torch.is_grad_enabled()),
    )
    try:
        features = encoder.encode(images)
    finally:
        handle.remove()
        hook.remove()

    assert features.shape == (2, pathology_encoder.FEATURE_DIM)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert grad_modes == [False]
    assert encoder._model.grad_modes == [False]
    assert encoder._model.training is False
    assert all(not parameter.requires_grad for parameter in encoder._model.parameters())
    assert load_options == [{"map_location": "cpu", "weights_only": True}]
    assert strict_options == [True]
    expected_pixel = (
        np.asarray((128, 64, 32), dtype=np.float32) / 255.0
        - np.asarray(
            pathology_encoder.IMAGENET_MEAN,
            dtype=np.float32,
        )
    ) / np.asarray(pathology_encoder.IMAGENET_STD, dtype=np.float32)
    np.testing.assert_allclose(captured_inputs[0][0, :, 0, 0].numpy(), expected_pixel, rtol=1e-6)
    descriptor = encoder.descriptor
    assert descriptor["name"] == "retccl_resnet50"
    assert descriptor["source_revision"] == pathology_encoder.RETCCL_SOURCE_REVISION
    assert descriptor["weights_sha256"] == expected_sha256
    assert descriptor["runtime_architecture"].startswith("torchvision.models.resnet50")
    assert descriptor["preprocessing"]["canvas_size"] == [8, 8]
    assert descriptor["preprocessing"]["internal_crop"] is None
    assert descriptor["preprocessing"]["mean"] == list(pathology_encoder.IMAGENET_MEAN)
    descriptor["source_revision"] = "changed"
    assert encoder.descriptor["source_revision"] == pathology_encoder.RETCCL_SOURCE_REVISION
    assert encoder.encode([]).shape == (0, pathology_encoder.FEATURE_DIM)
    encoder.release_memory()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing=.*projection.bias"),
        ("classifier", "unexpected=.*fc.weight"),
        ("shape", "shape"),
    ],
)
def test_checkpoint_schema_requires_all_encoder_tensors_and_no_classifier(
    tiny_torchvision_runtime: Any,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    torch = tiny_torchvision_runtime
    weights_path = tmp_path / "bad-schema.pth"
    torch.manual_seed(123)
    model = _build_tiny_model(torch)
    state = model.state_dict()
    if mutation == "missing":
        state.pop("projection.bias")
    elif mutation == "classifier":
        state["fc.weight"] = torch.zeros((128, pathology_encoder.FEATURE_DIM))
    else:
        state["projection.weight"] = state["projection.weight"][:-1]
    torch.save(state, weights_path)

    with pytest.raises(ValueError, match=message):
        pathology_encoder.RetCCLResNet50Encoder(
            weights_path,
            expected_sha256=pathology_encoder.sha256_file(weights_path),
            device="cpu",
            canvas_size=8,
        )


def test_checkpoint_hash_mismatch_fails_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "untrusted.pth"
    weights_path.write_bytes(b"not the pinned checkpoint")

    def unexpected_import() -> None:
        raise AssertionError("hash mismatch should be rejected before loading frameworks")

    monkeypatch.setattr(pathology_encoder, "_load_ml_dependencies", unexpected_import)
    with pytest.raises(ValueError, match="full SHA-256 pin"):
        pathology_encoder.RetCCLResNet50Encoder(
            weights_path,
            expected_sha256="0" * 64,
        )


def test_constructor_rejects_bad_sha_format() -> None:
    with pytest.raises(ValueError, match="full 64-character"):
        pathology_encoder.RetCCLResNet50Encoder(
            Path("missing.pth"),
            expected_sha256="abcd1234",
        )


def test_constructor_rejects_invalid_canvas_size_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="canvas_size must be a positive integer"):
        pathology_encoder.RetCCLResNet50Encoder(
            Path("missing.pth"),
            expected_sha256="0" * 64,
            canvas_size=0,
        )


def test_encoder_rejects_non_square_or_wrong_sized_canvas(
    tiny_torchvision_runtime: Any,
    tmp_path: Path,
) -> None:
    torch = tiny_torchvision_runtime
    weights_path = tmp_path / "best_ckpt.pth"
    expected_sha256 = _save_encoder_only_checkpoint(torch, weights_path)
    encoder = pathology_encoder.RetCCLResNet50Encoder(
        weights_path,
        expected_sha256=expected_sha256,
        device="cpu",
        canvas_size=8,
    )
    with pytest.raises(ValueError, match="full square resize-pad canvas"):
        encoder.encode([Image.new("RGB", (8, 10))])
    with pytest.raises(ValueError, match="full square resize-pad canvas"):
        encoder.encode([Image.new("RGB", (4, 4))])
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        encoder.encode([object()])  # type: ignore[list-item]


def test_encoder_rejects_nonfinite_model_output(
    tiny_torchvision_runtime: Any,
    tmp_path: Path,
) -> None:
    torch = tiny_torchvision_runtime
    weights_path = tmp_path / "best_ckpt.pth"
    expected_sha256 = _save_encoder_only_checkpoint(torch, weights_path)
    encoder = pathology_encoder.RetCCLResNet50Encoder(
        weights_path,
        expected_sha256=expected_sha256,
        device="cpu",
        canvas_size=8,
    )
    encoder._model.force_nonfinite = True
    with pytest.raises(RuntimeError, match="non-finite"):
        encoder.encode([Image.new("RGB", (8, 8), color=(1, 2, 3))])
