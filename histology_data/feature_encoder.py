"""Frozen ImageNet ResNet-50 feature extraction for histology tiles.

The pretrained checkpoint is fetched only from the official torchvision model
URL and kept in a checksum-verified cache. Torch is imported lazily so data
preparation and cache inspection can still run without the optional feature
dependencies installed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

OFFICIAL_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
OFFICIAL_WEIGHTS_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
OFFICIAL_WEIGHTS_HASH_PREFIX = OFFICIAL_WEIGHTS_SHA256[:8]
WEIGHTS_ENUM = "ResNet50_Weights.IMAGENET1K_V1"
WEIGHTS_CACHE_NAME = "resnet50_imagenet1k_v1"
WEIGHTS_FILENAME = "resnet50_imagenet1k_v1.pth"
WEIGHTS_MANIFEST_NAME = "manifest.json"
WEIGHTS_MANIFEST_VERSION = 1
ENCODER_IMPLEMENTATION_VERSION = "1.0.0"
FEATURE_DIM = 2048

_LOGGER = logging.getLogger(__name__)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    """Hash a file with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ml_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import the optional torch/torchvision stack when an encoder is created."""
    try:
        import torch
        import torchvision
        from torchvision import models
        from torchvision.models import ResNet50_Weights
    except ImportError as exc:
        raise RuntimeError(
            "ResNet50Encoder requires the optional feature dependencies; "
            "install them with `pip install -r requirements-features.txt`."
        ) from exc
    return torch, torchvision, models, ResNet50_Weights


def _cache_directory(weights_dir: Path) -> Path:
    return weights_dir / WEIGHTS_CACHE_NAME


def _read_verified_cache(weights_dir: Path) -> tuple[Path, str]:
    """Return a cached checkpoint only when its manifest and full hash agree."""
    cache_dir = _cache_directory(weights_dir)
    manifest_path = cache_dir / WEIGHTS_MANIFEST_NAME
    weights_path = cache_dir / WEIGHTS_FILENAME
    if cache_dir.is_symlink() or manifest_path.is_symlink() or weights_path.is_symlink():
        raise ValueError(f"Unsafe symlink in ResNet-50 weight cache: {cache_dir}")
    if not cache_dir.is_dir() or not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Incomplete ResNet-50 weight cache: {cache_dir}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable ResNet-50 weight manifest: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("ResNet-50 weight manifest must be a JSON object.")
    expected_fields = {
        "schema_version", "model", "weights", "url", "filename", "sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("ResNet-50 weight manifest has an unexpected schema.")
    if (
        manifest["schema_version"] != WEIGHTS_MANIFEST_VERSION
        or manifest["model"] != "resnet50"
        or manifest["weights"] != WEIGHTS_ENUM
        or manifest["url"] != OFFICIAL_WEIGHTS_URL
        or manifest["filename"] != WEIGHTS_FILENAME
    ):
        raise ValueError("ResNet-50 weight manifest does not identify the official V1 checkpoint.")

    expected_sha256 = manifest["sha256"]
    if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("ResNet-50 weight manifest has an invalid full SHA-256.")
    if expected_sha256 != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("ResNet-50 weight manifest SHA-256 does not match the pinned official checkpoint.")
    if not expected_sha256.startswith(OFFICIAL_WEIGHTS_HASH_PREFIX):
        raise ValueError("ResNet-50 weight manifest SHA-256 does not match the official hash prefix.")
    actual_sha256 = _sha256_file(weights_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Cached ResNet-50 checkpoint failed its manifest SHA-256 check.")
    if not actual_sha256.startswith(OFFICIAL_WEIGHTS_HASH_PREFIX):
        raise ValueError("Cached ResNet-50 checkpoint failed its official hash-prefix check.")
    return weights_path, actual_sha256


def _write_manifest(path: Path, sha256: str) -> None:
    """Write the compact, path-independent cache manifest."""
    manifest = {
        "schema_version": WEIGHTS_MANIFEST_VERSION,
        "model": "resnet50",
        "weights": WEIGHTS_ENUM,
        "url": OFFICIAL_WEIGHTS_URL,
        "filename": WEIGHTS_FILENAME,
        "sha256": sha256,
    }
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _ensure_weights(torch: Any, weights_dir: Path, download: bool) -> tuple[Path, str]:
    """Validate or atomically populate the immutable official checkpoint cache."""
    cache_dir = _cache_directory(weights_dir)
    if cache_dir.exists() or cache_dir.is_symlink():
        return _read_verified_cache(weights_dir)
    if not download:
        raise FileNotFoundError(
            f"Official ResNet-50 V1 weights are not cached at {cache_dir}; "
            "set download=True to fetch them."
        )

    weights_dir.mkdir(parents=True, exist_ok=True)
    staging_dir: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{WEIGHTS_CACHE_NAME}.", dir=weights_dir)
    )
    staged_weights = staging_dir / WEIGHTS_FILENAME
    try:
        _LOGGER.info("Downloading official ResNet-50 V1 weights from %s", OFFICIAL_WEIGHTS_URL)
        torch.hub.download_url_to_file(
            OFFICIAL_WEIGHTS_URL,
            str(staged_weights),
            hash_prefix=OFFICIAL_WEIGHTS_HASH_PREFIX,
            progress=True,
        )
        downloaded_sha256 = _sha256_file(staged_weights)
        if downloaded_sha256 != OFFICIAL_WEIGHTS_SHA256:
            raise ValueError("Downloaded ResNet-50 checkpoint failed its pinned official SHA-256 check.")
        _write_manifest(staging_dir / WEIGHTS_MANIFEST_NAME, downloaded_sha256)
        try:
            os.rename(staging_dir, cache_dir)
            staging_dir = None
        except OSError:
            # Another process may have published the same immutable cache while
            # this process downloaded it. Accept only a complete verified copy.
            if not cache_dir.exists():
                raise
            _read_verified_cache(weights_dir)
            _LOGGER.info("Using concurrently published verified ResNet-50 checkpoint")
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return _read_verified_cache(weights_dir)


def is_oom(exc: BaseException) -> bool:
    """Identify common framework and allocator out-of-memory exceptions."""
    if isinstance(exc, MemoryError):
        return True
    for exception_type in type(exc).__mro__:
        if (
            exception_type.__name__ == "OutOfMemoryError"
            and exception_type.__module__.startswith("torch")
        ):
            return True
    return "out of memory" in str(exc).casefold()


class ResNet50Encoder:
    """Extract frozen, 2048-dimensional ImageNet ResNet-50 descriptors.

    Args:
        weights_dir: Persistent directory used for the verified checkpoint cache.
        device: ``"auto"``, ``"cpu"``, or a CUDA device such as ``"cuda:0"``.
        precision: ``"fp32"`` or CUDA-only ``"amp_fp16"`` inference.
        download: Whether a missing official checkpoint may be downloaded.
    """

    feature_dim = FEATURE_DIM

    def __init__(
        self,
        weights_dir: Path,
        *,
        device: str = "auto",
        precision: str = "fp32",
        download: bool = True,
    ) -> None:
        if precision not in {"fp32", "amp_fp16"}:
            raise ValueError("precision must be 'fp32' or 'amp_fp16'.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device string.")

        self.weights_dir = Path(weights_dir)
        self._torch, self._torchvision, models, weights_enum = _load_ml_dependencies()
        self._weights = weights_enum.IMAGENET1K_V1
        if self._weights.url != OFFICIAL_WEIGHTS_URL:
            raise RuntimeError(
                "Installed torchvision maps IMAGENET1K_V1 to an unexpected URL; "
                "refusing to load a different checkpoint."
            )

        self._device = self._resolve_device(device)
        if precision == "amp_fp16" and self._device.type != "cuda":
            raise ValueError("precision='amp_fp16' requires a CUDA device.")
        self.precision = precision

        weights_path, weights_sha256 = _ensure_weights(
            self._torch, self.weights_dir, download=download,
        )
        self._weights_sha256 = weights_sha256
        self._transform = self._weights.transforms()
        self._validate_official_transform(self._transform)

        try:
            state_dict = self._torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "ResNet50Encoder requires torch.load(..., weights_only=True) support "
                "to load pretrained weights safely."
            ) from exc
        if not isinstance(state_dict, dict):
            raise ValueError("Official ResNet-50 checkpoint must contain a state dictionary.")

        model = models.resnet50(weights=None)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.fc = self._torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(self._device)
        self._descriptor = self._build_descriptor()
        _LOGGER.info(
            "Initialized frozen ResNet-50 encoder on %s with %s precision",
            self._device,
            self.precision,
        )

    def _resolve_device(self, requested: str) -> Any:
        """Resolve and validate supported CPU/CUDA execution devices."""
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

    @staticmethod
    def _validate_official_transform(transform: Any) -> None:
        """Guard the documented IMAGENET1K_V1 resize/crop/normalization contract."""
        expected = {
            "resize_size": [256],
            "crop_size": [224],
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        actual = {
            "resize_size": list(transform.resize_size),
            "crop_size": list(transform.crop_size),
            "antialias": transform.antialias,
            "mean": list(transform.mean),
            "std": list(transform.std),
        }
        interpolation = getattr(transform.interpolation, "name", str(transform.interpolation))
        if actual != expected or interpolation.casefold() != "bilinear":
            raise RuntimeError(
                "Installed torchvision's IMAGENET1K_V1 preprocessing does not match "
                "the recorded 256-resize/224-center-crop bilinear contract."
            )

    def _build_descriptor(self) -> dict[str, Any]:
        """Build path- and device-independent feature identity metadata."""
        preprocessing = {
            "input_color": "RGB",
            "transform": f"{WEIGHTS_ENUM}.transforms()",
            "resize_size": [256],
            "crop_size": [224],
            "crop": "center",
            "interpolation": "bilinear",
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        return {
            "name": "resnet50",
            "weights": WEIGHTS_ENUM,
            "weights_sha256": self._weights_sha256,
            "preprocessing": preprocessing,
            "torch_version": str(self._torch.__version__),
            "torchvision_version": str(self._torchvision.__version__),
            "pillow_version": str(PILLOW_VERSION),
            "precision": self.precision,
            "implementation_version": ENCODER_IMPLEMENTATION_VERSION,
        }

    @property
    def descriptor(self) -> dict[str, Any]:
        """Return feature identity metadata without mutable shared containers."""
        return json.loads(json.dumps(self._descriptor))

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Transform RGB images and return finite float32 ``N x 2048`` features."""
        if not isinstance(images, list):
            raise TypeError("images must be a list of PIL.Image.Image instances.")
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        if any(not isinstance(image, Image.Image) for image in images):
            raise TypeError("every input must be a PIL.Image.Image instance.")

        batch = self._torch.stack([
            self._transform(image.convert("RGB")) for image in images
        ]).to(self._device)
        with self._torch.inference_mode():
            if self.precision == "amp_fp16":
                with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                    features = self._model(batch)
            else:
                features = self._model(batch)

        if features.ndim != 2 or tuple(features.shape) != (len(images), self.feature_dim):
            raise RuntimeError(
                f"ResNet-50 returned shape {tuple(features.shape)}, expected "
                f"({len(images)}, {self.feature_dim})."
            )
        result = features.to(device="cpu", dtype=self._torch.float32).numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("ResNet-50 returned non-finite feature values.")
        return np.asarray(result, dtype=np.float32)

    def release_memory(self) -> None:
        """Release Python garbage and unused CUDA allocator blocks."""
        gc.collect()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
