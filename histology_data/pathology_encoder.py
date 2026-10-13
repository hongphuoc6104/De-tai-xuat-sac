"""Verified RetCCL ResNet-50 features for full-canvas histology images.

The public RetCCL checkpoint is fetched only after a caller supplies its full
SHA-256 pin.  Torch and torchvision stay optional until the encoder is created;
checkpoint inspection and readiness checks do not import either framework.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

FEATURE_DIM = 2048
CANVAS_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RETCCL_REPOSITORY_URL = "https://github.com/Xiyue-Wang/RetCCL"
RETCCL_SOURCE_REVISION = "a85b97226b62853b64143e8d535f5e10dae6cc89"
RETCCL_SOURCE_FILES = {
    name: f"{RETCCL_REPOSITORY_URL}/blob/{RETCCL_SOURCE_REVISION}/{name}"
    for name in ("README.md", "ResNet.py", "get_feature.py")
}
RETCCL_CHECKPOINT_FOLDER_URL = "https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL?usp=sharing"
OFFICIAL_WEIGHTS_FILE_ID = "1Sn0kYwkXp7eqdIPUTFtK_ilz5ThApOsn"
OFFICIAL_WEIGHTS_FILENAME = "best_ckpt.pth"
OFFICIAL_WEIGHTS_URL = f"https://drive.usercontent.google.com/download?id={OFFICIAL_WEIGHTS_FILE_ID}&export=download"
OFFICIAL_WEIGHTS_SIZE_BYTES = 94_367_005
OFFICIAL_WEIGHTS_HEAD_VERIFIED_UTC = "2026-10-13T21:47:39Z"
ENCODER_IMPLEMENTATION_VERSION = "1.0.0"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file using bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("expected_sha256 must be a full 64-character SHA-256 hex digest.")
    return value.lower()


def official_weights_metadata() -> dict[str, Any]:
    """Return verified source and acquisition metadata without network access."""
    return {
        "repository_url": RETCCL_REPOSITORY_URL,
        "source_revision": RETCCL_SOURCE_REVISION,
        "source_files": dict(RETCCL_SOURCE_FILES),
        "checkpoint_folder_url": RETCCL_CHECKPOINT_FOLDER_URL,
        "weights_url": OFFICIAL_WEIGHTS_URL,
        "weights_file_id": OFFICIAL_WEIGHTS_FILE_ID,
        "weights_filename": OFFICIAL_WEIGHTS_FILENAME,
        "weights_size_bytes": OFFICIAL_WEIGHTS_SIZE_BYTES,
        "weights_size_source": "HTTP HEAD Content-Length",
        "remote_available_at_head_check": True,
        "head_http_status": 200,
        "head_content_type": "application/octet-stream",
        "weights_sha256": None,
        "weights_sha256_author_published": False,
        "weights_sha256_status": (
            "authors publish no checksum; compute at first acquisition and pin the observed full digest"
        ),
        "head_verified_utc": OFFICIAL_WEIGHTS_HEAD_VERIFIED_UTC,
    }


def probe_official_weights(*, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Probe only HTTP headers for the author-hosted checkpoint download."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive.")

    report = official_weights_metadata()
    try:
        request = urllib.request.Request(OFFICIAL_WEIGHTS_URL, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            headers = response.headers
            content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip()
            content_disposition = str(headers.get("Content-Disposition", ""))
            raw_size = headers.get("Content-Length")
            size_bytes = int(raw_size) if raw_size is not None else None
            is_checkpoint = OFFICIAL_WEIGHTS_FILENAME in content_disposition
            accessible = status == 200 and content_type == "application/octet-stream" and is_checkpoint
            report.update(
                {
                    "http_status": status,
                    "content_type": content_type,
                    "content_disposition": content_disposition,
                    "remote_size_bytes": size_bytes,
                    "remote_size_matches": size_bytes == OFFICIAL_WEIGHTS_SIZE_BYTES,
                    "remote_accessible": accessible,
                    "download_ready": accessible and size_bytes == OFFICIAL_WEIGHTS_SIZE_BYTES,
                    "inference_ready": False,
                    "readiness_status": "sha256_pin_required",
                }
            )
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        report.update(
            {
                "remote_accessible": False,
                "download_ready": False,
                "inference_ready": False,
                "readiness_status": "remote_probe_failed",
                "probe_error_type": type(exc).__name__,
            }
        )
    return report


def checkpoint_readiness(
    weights_path: Path,
    *,
    expected_sha256: str | None = None,
    require_official_size: bool = True,
) -> dict[str, Any]:
    """Check a local checkpoint against its caller-supplied full digest pin."""
    if not isinstance(require_official_size, bool):
        raise ValueError("require_official_size must be a boolean.")
    expected = _expected_sha256(expected_sha256) if expected_sha256 is not None else None
    path = Path(weights_path)
    report: dict[str, Any] = {
        **official_weights_metadata(),
        "path": str(path),
        "expected_sha256": expected,
        "ready": False,
    }
    if path.is_symlink():
        report["readiness_status"] = "unsafe_symlink"
        return report
    if not path.is_file():
        report["readiness_status"] = "checkpoint_missing"
        return report

    size_bytes = path.stat().st_size
    report["local_size_bytes"] = size_bytes
    report["local_size_matches"] = size_bytes == OFFICIAL_WEIGHTS_SIZE_BYTES
    if require_official_size and not report["local_size_matches"]:
        report["readiness_status"] = "checkpoint_size_mismatch"
        return report

    actual_sha256 = sha256_file(path)
    report["observed_sha256"] = actual_sha256
    if expected is None:
        report["readiness_status"] = "sha256_pin_required"
        return report
    if actual_sha256 != expected:
        report["readiness_status"] = "sha256_mismatch"
        return report

    report.update(
        {
            "weights_sha256": actual_sha256,
            "ready": True,
            "readiness_status": "ready",
        }
    )
    return report


def download_official_weights(
    destination: Path,
    *,
    expected_sha256: str,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download the public checkpoint atomically after HEAD and SHA pin checks.

    The full expected digest is mandatory.  This function is opt-in and is
    never called by encoder construction.
    """
    expected = _expected_sha256(expected_sha256)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive.")
    remote = probe_official_weights(timeout_seconds=min(timeout_seconds, 20.0))
    if not remote.get("download_ready"):
        raise RuntimeError("Official RetCCL checkpoint is not ready for download according to its HEAD metadata.")

    target = Path(destination)
    if target.is_symlink():
        raise ValueError(f"Refusing to write checkpoint through a symlink: {target}")
    if target.exists():
        readiness = checkpoint_readiness(target, expected_sha256=expected)
        if readiness["ready"]:
            return target
        raise ValueError("An existing RetCCL checkpoint failed size or SHA-256 verification; refusing to replace it.")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = urllib.request.Request(OFFICIAL_WEIGHTS_URL, method="GET")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if int(response.status) != 200:
                    raise RuntimeError(f"Official RetCCL download returned HTTP {response.status}.")
                while block := response.read(1024 * 1024):
                    total_bytes += len(block)
                    if total_bytes > OFFICIAL_WEIGHTS_SIZE_BYTES:
                        raise ValueError("Official RetCCL download exceeded its pinned HEAD size.")
                    digest.update(block)
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if total_bytes != OFFICIAL_WEIGHTS_SIZE_BYTES:
            raise ValueError("Official RetCCL download size differs from the verified HEAD size.")
        if digest.hexdigest() != expected:
            raise ValueError("Official RetCCL download failed its caller-supplied SHA-256 pin.")
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    _LOGGER.info("Downloaded and SHA-256 verified official RetCCL weights to %s", target)
    return target


def _load_ml_dependencies() -> tuple[Any, Any, Any]:
    """Load optional inference dependencies only when an encoder is created."""
    try:
        import torch
        import torchvision
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("RetCCLResNet50Encoder requires the optional PyTorch and torchvision dependencies.") from exc
    return torch, torchvision, models


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return tuple(int(item) for item in value)


def _validate_torchvision_resnet50(model: Any) -> None:
    """Fail closed unless torchvision exposes RetCCL's verified ResNet-50 trunk."""
    required_attributes = (
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
        "fc",
    )
    if any(not hasattr(model, attribute) for attribute in required_attributes):
        raise RuntimeError("Runtime ResNet is missing part of the RetCCL ResNet-50 architecture.")
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    if tuple(len(stage) for stage in stages) != (3, 4, 6, 3):
        raise RuntimeError("Runtime ResNet-50 must use RetCCL's [3, 4, 6, 3] Bottleneck stages.")
    if _pair(model.conv1.kernel_size) != (7, 7) or _pair(model.conv1.stride) != (2, 2):
        raise RuntimeError("Runtime ResNet stem differs from the official RetCCL architecture.")
    if _pair(model.conv1.padding) != (3, 3):
        raise RuntimeError("Runtime ResNet stem padding differs from the official RetCCL architecture.")
    if _pair(model.maxpool.kernel_size) != (3, 3) or _pair(model.maxpool.stride) != (2, 2):
        raise RuntimeError("Runtime ResNet max-pooling differs from the official RetCCL architecture.")
    if tuple(model.avgpool.output_size) != (1, 1):
        raise RuntimeError("Runtime ResNet must use 1x1 adaptive average pooling.")
    if int(model.fc.in_features) != FEATURE_DIM:
        raise RuntimeError("Runtime ResNet classifier input does not produce 2048 features.")
    if getattr(model, "groups", None) != 1 or getattr(model, "base_width", None) != 64:
        raise RuntimeError("Runtime ResNet width/groups differ from the official RetCCL architecture.")
    if not _is_default_batch_norm(model.bn1):
        raise RuntimeError("Runtime ResNet stem BatchNorm differs from the official RetCCL architecture.")

    for stage_index, stage in enumerate(stages):
        for block_index, block in enumerate(stage):
            if not all(hasattr(block, name) for name in ("conv1", "conv2", "conv3", "bn1", "bn2", "bn3")):
                raise RuntimeError("Runtime ResNet block is not a standard Bottleneck block.")
            expected_stride = 2 if stage_index > 0 and block_index == 0 else 1
            if (
                _pair(block.conv1.kernel_size) != (1, 1)
                or _pair(block.conv1.stride) != (1, 1)
                or _pair(block.conv2.kernel_size) != (3, 3)
                or _pair(block.conv2.stride) != (expected_stride, expected_stride)
                or _pair(block.conv2.padding) != (1, 1)
                or _pair(block.conv2.dilation) != (1, 1)
                or block.conv2.groups != 1
                or _pair(block.conv3.kernel_size) != (1, 1)
                or block.conv1.out_channels != 64 * (2**stage_index)
                or block.conv3.out_channels != 256 * (2**stage_index)
            ):
                raise RuntimeError("Runtime ResNet Bottleneck stride or convolution layout differs from RetCCL.")
            if not all(_is_default_batch_norm(getattr(block, name)) for name in ("bn1", "bn2", "bn3")):
                raise RuntimeError("Runtime ResNet Bottleneck BatchNorm differs from RetCCL.")


def _is_default_batch_norm(layer: Any) -> bool:
    return getattr(layer, "momentum", None) == 0.1 and getattr(layer, "eps", None) == 1e-5


def _build_model(torch: Any, models: Any) -> Any:
    """Build the matching modern torchvision ResNet-50 and remove its head."""
    try:
        model = models.resnet50(weights=None)
    except TypeError as exc:
        raise RuntimeError("RetCCL requires torchvision's modern resnet50(weights=None) API.") from exc
    _validate_torchvision_resnet50(model)
    model.fc = torch.nn.Identity()
    return model


def _validate_state_dict_schema(
    state_dict: Any,
    expected_state: Any,
    torch: Any,
) -> None:
    """Require exact encoder-only keys and tensor shapes before strict loading."""
    if not isinstance(state_dict, dict):
        raise ValueError("RetCCL checkpoint must contain a direct state dictionary.")
    expected_keys = set(expected_state)
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "RetCCL checkpoint keys differ from the complete encoder-only schema "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]})."
        )
    for name, expected_tensor in expected_state.items():
        actual_tensor = state_dict[name]
        if not torch.is_tensor(actual_tensor):
            raise ValueError(f"RetCCL checkpoint value {name!r} is not a tensor.")
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"RetCCL checkpoint tensor {name!r} has shape {tuple(actual_tensor.shape)}, "
                f"expected {tuple(expected_tensor.shape)}."
            )


class RetCCLResNet50Encoder:
    """Extract frozen 2048-dimensional RetCCL descriptors from square canvases.

    Args:
        weights_path: Local copy of the official ``best_ckpt.pth`` file.
        expected_sha256: Required full digest, pinned after an independent
            download/checksum step. Prefix hashes are rejected.
        device: ``"auto"``, ``"cpu"``, or a CUDA device such as ``"cuda:0"``.
        canvas_size: Side length expected from the upstream full-image
            resize-and-pad preprocessing. RetCCL's image-classification path
            uses the complete canvas; this adapter does not crop or resize it.
    """

    feature_dim = FEATURE_DIM

    def __init__(
        self,
        weights_path: Path,
        *,
        expected_sha256: str,
        device: str = "auto",
        canvas_size: int = CANVAS_SIZE,
    ) -> None:
        self.weights_path = Path(weights_path)
        expected = _expected_sha256(expected_sha256)
        if isinstance(canvas_size, bool) or not isinstance(canvas_size, int) or canvas_size < 1:
            raise ValueError("canvas_size must be a positive integer.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string.")
        if self.weights_path.is_symlink() or not self.weights_path.is_file():
            raise FileNotFoundError(f"RetCCL checkpoint is not a regular local file: {self.weights_path}")

        actual_sha256 = sha256_file(self.weights_path)
        if actual_sha256 != expected:
            raise ValueError("RetCCL checkpoint failed the required full SHA-256 pin.")

        self._torch, self._torchvision, models = _load_ml_dependencies()
        self._device = self._resolve_device(device)
        try:
            state_dict = self._torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "RetCCL requires torch.load(..., weights_only=True) support for safe checkpoint loading."
            ) from exc
        model = _build_model(self._torch, models)
        _validate_state_dict_schema(state_dict, model.state_dict(), self._torch)
        model.load_state_dict(state_dict, strict=True)
        del state_dict

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(device=self._device, dtype=self._torch.float32)
        self._canvas_size = canvas_size
        self._mean = self._torch.tensor(
            IMAGENET_MEAN,
            dtype=self._torch.float32,
            device=self._device,
        ).view(1, 3, 1, 1)
        self._std = self._torch.tensor(
            IMAGENET_STD,
            dtype=self._torch.float32,
            device=self._device,
        ).view(1, 3, 1, 1)
        self._weights_sha256 = actual_sha256
        self._descriptor = self._build_descriptor()
        _LOGGER.info("Initialized frozen RetCCL ResNet-50 on %s in FP32", self._device)

    def _resolve_device(self, requested: str) -> Any:
        cuda_available = bool(self._torch.cuda.is_available())
        if requested == "auto":
            return self._torch.device("cuda" if cuda_available else "cpu")
        try:
            resolved = self._torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device string: {requested!r}.") from exc
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("device must resolve to CPU or CUDA.")
        if resolved.type == "cuda" and not cuda_available:
            raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable.")
        return resolved

    def _build_descriptor(self) -> dict[str, Any]:
        return {
            "name": "retccl_resnet50",
            "feature_dim": self.feature_dim,
            "source_url": RETCCL_REPOSITORY_URL,
            "source_revision": RETCCL_SOURCE_REVISION,
            "source_files": dict(RETCCL_SOURCE_FILES),
            "weights_url": OFFICIAL_WEIGHTS_URL,
            "weights_file_id": OFFICIAL_WEIGHTS_FILE_ID,
            "weights_filename": OFFICIAL_WEIGHTS_FILENAME,
            "weights_size_bytes": OFFICIAL_WEIGHTS_SIZE_BYTES,
            "weights_sha256": self._weights_sha256,
            "runtime_architecture": (
                "torchvision.models.resnet50(weights=None), RetCCL Bottleneck stages "
                "[3, 4, 6, 3], conv2 stride, fc=Identity"
            ),
            "torch_version": str(self._torch.__version__),
            "torchvision_version": str(self._torchvision.__version__),
            "pillow_version": str(PILLOW_VERSION),
            "precision": "fp32",
            "model_mode": "eval; parameters frozen; torch.inference_mode",
            "pooling": "layer4 + adaptive_avg_pool2d(1x1) + flatten; 2048 values",
            "preprocessing": {
                "input_color": "RGB",
                "input_canvas": "square full-image resize-pad canvas",
                "canvas_size": [self._canvas_size, self._canvas_size],
                "internal_resize": None,
                "internal_crop": None,
                "normalization": "(pixel / 255 - mean) / std",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
            "implementation_version": ENCODER_IMPLEMENTATION_VERSION,
        }

    @property
    def descriptor(self) -> dict[str, Any]:
        """Return a detached copy of this feature identity."""
        return json.loads(json.dumps(self._descriptor))

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode the complete square resize-pad canvases without cropping."""
        if not isinstance(images, list):
            raise TypeError("images must be a list of PIL.Image.Image instances.")
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        if any(not isinstance(image, Image.Image) for image in images):
            raise TypeError("every input must be a PIL.Image.Image instance.")

        tensors = []
        expected_size = (self._canvas_size, self._canvas_size)
        for image in images:
            if image.size != expected_size:
                raise ValueError(
                    f"RetCCL expects the full square resize-pad canvas {expected_size}; received {image.size}."
                )
            rgb = image.convert("RGB")
            try:
                array = np.asarray(rgb, dtype=np.float32) / 255.0
            finally:
                rgb.close()
            tensor = self._torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            tensors.append(tensor)

        batch = self._torch.stack(tensors, dim=0).to(device=self._device, dtype=self._torch.float32)
        batch = (batch - self._mean) / self._std
        with self._torch.inference_mode():
            features = self._model(batch)
        if not self._torch.is_tensor(features) or tuple(features.shape) != (len(images), self.feature_dim):
            shape = tuple(features.shape) if hasattr(features, "shape") else type(features).__name__
            raise RuntimeError(f"RetCCL returned shape {shape}, expected ({len(images)}, {self.feature_dim}).")
        result = features.to(device="cpu", dtype=self._torch.float32).numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("RetCCL returned non-finite feature values.")
        return np.asarray(result, dtype=np.float32)

    def release_memory(self) -> None:
        """Collect Python garbage and unused CUDA allocator blocks."""
        gc.collect()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
