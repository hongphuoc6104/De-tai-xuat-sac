"""Command-line entry point for bounded histology data preparation."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Sequence

LOGGER = logging.getLogger("histology_data")
DEFAULT_MAX_MIB = 512.0
PROCESSING_KEYS = (
    "tile_size",
    "stride",
    "min_tissue",
    "max_tiles_per_image",
    "blur_threshold",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m histology_data",
        description="Build, process, and verify restartable histology data shards.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build a catalog and pack raw image shards")
    prepare.add_argument("--metadata", required=True, type=Path)
    prepare.add_argument("--source", required=True, action="append", type=Path,
                         help="image directory or ZIP; repeat to include multiple sources")
    prepare.add_argument("--release", required=True, type=Path)
    prepare.add_argument("--config", type=Path, help="JSON config with a prepare section")
    prepare.add_argument("--lenses", type=int, nargs="+")
    prepare.add_argument("--per-lens", type=int, help="deterministic smoke limit per objective lens")
    prepare.add_argument("--identity-map", type=Path)
    prepare.add_argument("--labels-reviewed", action="store_true", default=None)
    prepare.add_argument("--max-mib", type=float)
    prepare.add_argument("--max-shards", type=int)

    process = commands.add_parser("process", help="produce QC and lossless tiles from committed shards")
    process.add_argument("--release", required=True, type=Path)
    process.add_argument("--work-root", required=True, type=Path, help="temporary SSD work directory")
    process.add_argument("--output", required=True, type=Path, help="verified output location")
    process.add_argument("--config", type=Path, help="JSON config with a process section")
    process.add_argument("--shard", help="process one committed shard ID")
    process.add_argument("--max-shards", type=int, help="limit newly processed shards in this invocation")
    process.add_argument("--tile-size", type=int)
    process.add_argument("--stride", type=int)
    process.add_argument("--min-tissue", type=float)
    process.add_argument("--max-tiles-per-image", type=int)
    process.add_argument("--blur-threshold", type=float)

    verify = commands.add_parser("verify", help="verify a raw shard release and its contents")
    verify.add_argument("--release", required=True, type=Path)
    return parser


def _load_config(path: Path | None, section: str) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object.")
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Config section {section!r} must be a JSON object.")
    return values


def _configured(value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    """Choose an explicit CLI value before config, then the documented default."""
    return value if value is not None else config.get(key, default)


def _optional_integer(value: Any, name: str, *, allow_zero: bool = False) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return value


def _positive_number(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def _prepare(args: argparse.Namespace) -> int:
    from .catalog import build_catalog
    from .shards import pack_catalog

    config = _load_config(args.config, "prepare")
    max_mib = float(_configured(args.max_mib, config, "max_mib", DEFAULT_MAX_MIB))
    max_shards = _optional_integer(_configured(args.max_shards, config, "max_shards", None), "--max-shards")
    per_lens = _optional_integer(_configured(args.per_lens, config, "per_lens", None), "--per-lens")
    lenses = _configured(args.lenses, config, "lenses", None)
    identity_map = args.identity_map or (Path(config["identity_map"]) if config.get("identity_map") else None)
    labels_reviewed = args.labels_reviewed or bool(config.get("labels_reviewed", False))
    _positive_number(max_mib, "--max-mib")
    max_bytes = int(max_mib * 1024 * 1024)
    if max_bytes < 1:
        raise ValueError("--max-mib is too small to produce a positive byte limit.")
    if lenses is not None:
        if (
            not isinstance(lenses, list)
            or not lenses
            or any(isinstance(lens, bool) or not isinstance(lens, int) for lens in lenses)
        ):
            raise ValueError("--lenses must contain one or more of 4, 10, and 40.")
        if any(lens not in {4, 10, 40} for lens in lenses):
            raise ValueError("--lenses must contain one or more of 4, 10, and 40.")

    catalog = build_catalog(
        metadata=args.metadata,
        sources=args.source,
        lenses=lenses,
        per_lens=int(per_lens) if per_lens is not None else None,
        identity_map=identity_map,
        labels_reviewed=labels_reviewed,
    )
    release = pack_catalog(
        catalog,
        output=args.release,
        max_bytes=max_bytes,
        max_shards=max_shards,
    )
    LOGGER.info(
        "Prepared catalog %s (%s mode): %d selected images, %d committed shards; release complete=%s.",
        catalog["catalog_id"],
        catalog["mode"],
        len(catalog["images"]),
        len(release.get("shards", [])),
        release.get("complete", False),
    )
    return 0


def _read_release_shards(release_root: Path) -> tuple[bool, list[str]]:
    release_path = Path(release_root) / "release.json"
    with release_path.open(encoding="utf-8") as stream:
        descriptor = json.load(stream)
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("shards"), list):
        raise ValueError("release.json must contain a shards list.")
    shard_ids: list[str] = []
    for entry in descriptor["shards"]:
        shard_id = entry.get("shard_id") if isinstance(entry, dict) else None
        if (
            not isinstance(shard_id, str)
            or not shard_id
            or shard_id in {".", ".."}
            or "/" in shard_id
            or "\\" in shard_id
        ):
            raise ValueError("release.json contains an invalid shard_id.")
        shard_ids.append(shard_id)
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("release.json contains duplicate shard IDs.")
    return bool(descriptor.get("complete", False)), shard_ids


def _process(args: argparse.Namespace) -> int:
    from .processing import process_shard

    config = _load_config(args.config, "process")
    max_shards = _optional_integer(_configured(args.max_shards, config, "max_shards", None), "--max-shards")
    process_config: dict[str, Any] = {}
    defaults: dict[str, Any] = {
        "tile_size": 256,
        "stride": 256,
        "min_tissue": 0.2,
        "max_tiles_per_image": 0,
        "blur_threshold": 0,
    }
    for key in PROCESSING_KEYS:
        value = _configured(getattr(args, key), config, key, defaults[key])
        if value is not None:
            process_config[key] = value
    process_config["tile_size"] = _optional_integer(process_config["tile_size"], "--tile-size")
    process_config["stride"] = _optional_integer(process_config["stride"], "--stride")
    process_config["max_tiles_per_image"] = _optional_integer(
        process_config["max_tiles_per_image"], "--max-tiles-per-image", allow_zero=True
    )
    min_tissue = float(process_config["min_tissue"])
    if not math.isfinite(min_tissue) or not 0 <= min_tissue <= 1:
        raise ValueError("--min-tissue must be between 0 and 1.")
    blur_threshold = float(process_config["blur_threshold"])
    if not math.isfinite(blur_threshold) or blur_threshold < 0:
        raise ValueError("--blur-threshold must be finite and nonnegative.")
    process_config["min_tissue"] = min_tissue
    process_config["blur_threshold"] = blur_threshold

    release_complete, shard_ids = _read_release_shards(args.release)
    if args.shard:
        if args.shard not in shard_ids:
            raise ValueError(f"Shard {args.shard!r} is not committed in this release.")
        shard_ids = [args.shard]
    if not shard_ids:
        raise ValueError("Release has no committed shards to process.")

    new_count = 0
    reused_count = 0
    selected_count = 0
    for shard_id in shard_ids:
        result = process_shard(
            release_root=args.release,
            shard_id=shard_id,
            work_root=args.work_root,
            output_root=args.output,
            config=process_config,
        )
        selected_count += 1
        output_path = result.get("output_path")
        location = f" at {output_path}" if output_path is not None else ""
        if result.get("reused", False):
            reused_count += 1
            LOGGER.info("Verified existing processed shard %s%s.", shard_id, location)
        else:
            new_count += 1
            LOGGER.info("Committed processed shard %s%s.", shard_id, location)
        if max_shards is not None and new_count >= max_shards:
            break

    LOGGER.info(
        "Processing pass finished: %d new, %d verified existing, %d visited. Raw release complete=%s.",
        new_count,
        reused_count,
        selected_count,
        release_complete,
    )
    if not release_complete:
        LOGGER.warning("The raw release is partial; this processing pass cannot represent full cohort coverage.")
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .shards import verify_release

    release = verify_release(args.release)
    LOGGER.info(
        "Verified release %s: %d committed shards, %s.",
        release.get("catalog_id", "unknown"),
        len(release.get("shards", [])),
        "complete" if release.get("complete", False) else "partial",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and keep operational failures concise and nonzero."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "process":
            return _process(args)
        return _verify(args)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:  # Keep unexpected library failures actionable without a long traceback.
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
