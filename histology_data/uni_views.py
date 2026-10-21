"""Frozen UNI global/local features with original-pixel coordinates.

No dependency downloads or model construction occur on import. Clinical labels
belong to original images only; views never acquire classification targets.
"""
from __future__ import annotations

import importlib.metadata
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
from PIL import Image

from .io import file_hash

PROTOCOL = "uni-original-image-global-mean-max-v1"
UNI_DIM = 1024
IMAGE_DIM = 3 * UNI_DIM
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ViewConfig:
    """Fixed geometry; pixel sizes do not imply equal micrometres per pixel."""

    tile_size: int = 512
    input_size: int = 224
    padding_rgb: tuple[int, int, int] = (255, 255, 255)
    max_image_pixels: int = 178_000_000

    def validate(self) -> None:
        """Refuse silent changes to the registered image geometry."""
        if self.tile_size != 512 or self.input_size != 224 or self.padding_rgb != (255, 255, 255):
            raise ValueError("UNI v1 fixes tile512/input224/white padding.")
        if type(self.max_image_pixels) is not int or self.max_image_pixels < 1:
            raise ValueError("max_image_pixels must be a positive integer.")


class ViewEncoder(Protocol):
    """Small injectable interface; fixture encoders never masquerade as UNI."""

    feature_dim: int

    def descriptor(self) -> dict[str, Any]:
        """Return a JSON-compatible identity including weights and numerics."""
        ...

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Return one finite float32 vector per prepared RGB view."""
        ...


def canvas(image: Image.Image, size: int) -> Image.Image:
    """Resize the complete RGB field with bilinear interpolation and white pad."""
    scale = size / max(image.size)
    dimensions = tuple(max(1, min(size, round(v * scale))) for v in image.size)
    resized = image.resize(dimensions, Image.Resampling.BILINEAR)
    result = Image.new("RGB", (size, size), (255, 255, 255))
    result.paste(resized, ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
    resized.close()
    return result


def views(image: Image.Image, config: ViewConfig = ViewConfig()) -> Iterator[tuple[dict[str, Any], Image.Image]]:
    """Yield global then row-major tiles; caller owns each prepared PIL image."""
    config.validate()
    width, height = image.size
    if image.mode != "RGB" or width * height > config.max_image_pixels:
        raise ValueError("Views require a bounded, orientation-corrected RGB image.")
    yield {"kind": "global", "x": 0, "y": 0, "width": width, "height": height}, canvas(image, config.input_size)
    for y in range(0, height, config.tile_size):
        for x in range(0, width, config.tile_size):
            valid_w, valid_h = min(config.tile_size, width - x), min(config.tile_size, height - y)
            with image.crop((x, y, x + valid_w, y + valid_h)) as cropped:
                with Image.new("RGB", (config.tile_size, config.tile_size), config.padding_rgb) as padded:
                    padded.paste(cropped, (0, 0))
                    prepared = padded.resize((config.input_size, config.input_size), Image.Resampling.BILINEAR)
            yield {"kind": "local", "x": x, "y": y, "width": valid_w, "height": valid_h}, prepared


def pool(view_features: np.ndarray) -> np.ndarray:
    """Concatenate independently L2-normalized global, local mean and local max.

    Pool raw local embeddings before normalizing each block. Max is elementwise
    feature pooling, never an inferred cancer label or lesion segmentation.
    """
    values = np.asarray(view_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != UNI_DIM or not np.isfinite(values).all():
        raise ValueError("Expected finite global+local UNI1024 features.")
    blocks = np.stack((values[0], values[1:].mean(axis=0, dtype=np.float64), values[1:].max(axis=0))).astype(np.float64)
    norms = np.linalg.norm(blocks, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero pooling block; refusing undefined normalization.")
    return (blocks / norms[:, None]).reshape(IMAGE_DIM).astype(np.float32)


def embed_image(image: Image.Image, encoder: ViewEncoder, *, config: ViewConfig = ViewConfig(),
                batch_size: int = 4) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Bound view-image memory by batch size and return complete view lineage."""
    if type(batch_size) is not int or batch_size < 1 or encoder.feature_dim != UNI_DIM:
        raise ValueError("Positive batch size and a UNI1024 encoder are required.")
    features, records, pending = [], [], []

    def flush() -> None:
        try:
            result = np.asarray(encoder.encode(pending), dtype=np.float32)
            if result.shape != (len(pending), UNI_DIM) or not np.isfinite(result).all():
                raise ValueError("Encoder output shape/finiteness mismatch.")
            features.extend(result)
        finally:
            for photo in pending:
                photo.close()
            pending.clear()

    try:
        for record, prepared in views(image, config):
            records.append(record)
            pending.append(prepared)
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()
    finally:
        for photo in pending:
            photo.close()
    matrix = np.asarray(features, dtype=np.float32)
    return pool(matrix), matrix, records


def weight_readiness(weights: Path, sha256: str) -> dict[str, Any]:
    """Inspect an explicitly provided local checkpoint; never authenticate/download."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("A lowercase full checkpoint SHA256 pin is required.")
    if not weights.is_file() or weights.is_symlink():
        return {"ready": False, "reason": "local_weights_missing", "download_attempted": False}
    if file_hash(weights) != sha256:
        raise ValueError("UNI checkpoint checksum mismatch.")
    return {"ready": True, "sha256": sha256, "download_attempted": False,
            "scope": "bytes verified only; model load and GPU not yet checked"}


class UniEncoder:
    """Official UNI ViT-L/16 loaded strictly from an access-approved local file."""

    feature_dim = UNI_DIM

    def __init__(self, weights: Path, sha256: str, *, device: str = "cuda", precision: str = "float16"):
        if not weight_readiness(weights, sha256)["ready"]:
            raise FileNotFoundError("Provide access-approved UNI weights before creating an encoder.")
        if device not in {"cuda", "cpu"} or precision not in {"float16", "float32"}:
            raise ValueError("Explicit CPU/CUDA and float16/float32 precision required.")
        if device == "cpu" and precision != "float32":
            raise ValueError("CPU UNI requires float32.")
        import timm
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; refusing silent CPU fallback.")
        model = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224,
                                  patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False).eval()
        self.model = model.to(device)
        self.torch, self.device, self.precision = torch, device, precision
        self._descriptor = {"kind": "UNI", "architecture": "vit_large_patch16_224",
                            "weights_sha256": sha256, "feature_dim": UNI_DIM, "frozen": True,
                            "device": device, "precision": precision, "views": asdict(ViewConfig()),
                            "normalization": {"mean": MEAN, "std": STD},
                            "implementation_sha256": file_hash(Path(__file__)),
                            "versions": {name: importlib.metadata.version(name)
                                         for name in ("torch", "torchvision", "timm", "Pillow", "numpy")}}

    def descriptor(self) -> dict[str, Any]:
        """Return explicit geometry, precision, libraries and weight identity."""
        import json
        return json.loads(json.dumps(self._descriptor))

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Inference only; input views are already 224 square with no centre crop."""
        torch = self.torch
        if not images or any(photo.mode != "RGB" or photo.size != (224, 224) for photo in images):
            raise ValueError("UNI accepts a nonempty batch of RGB224 prepared views.")
        values = np.stack([np.asarray(photo, dtype=np.float32).transpose(2, 0, 1) / 255 for photo in images])
        values = (values - np.asarray(MEAN, dtype=np.float32)[None, :, None, None]) / np.asarray(STD, dtype=np.float32)[None, :, None, None]
        tensor = torch.from_numpy(values).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16,
                                                   enabled=self.precision == "float16"):
            features = self.model(tensor)
        return features.float().cpu().numpy()
