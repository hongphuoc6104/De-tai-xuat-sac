"""Explicit UNI preparation/extraction/training commands; never create a VM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import image_classification_r0_r3 as baseline, uni_cache, uni_smoke, uni_study
from .uni_views import UniEncoder, weight_readiness


def main() -> int:
    """Dispatch only the operation explicitly requested by the operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-weights", help="Verify local checkpoint bytes without downloading.")
    extract = commands.add_parser("extract", help="GPU extraction; explicit local checkpoint required.")
    smoke = commands.add_parser("smoke", help="Future-runtime technical resource check; no quality tuning.")
    train = commands.add_parser("train", help="CPU head only; exact baseline patient splits.")
    predict = commands.add_parser("predict", help="One original image with a verified research deployment.")
    for command in (check, extract, smoke, predict):
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--weights-sha256", required=True)
    for command in (extract, smoke, train):
        command.add_argument("--baseline-release", type=Path, required=True)
        command.add_argument("--baseline-study", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--budget-minutes", type=float, default=240)
        command.add_argument("--reserve-minutes", type=float, default=30)
    for command in (extract, smoke, predict):
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--precision", choices=("float16", "float32"), default="float16")
        command.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--archives-json", type=Path, required=True, help='JSON mapping lens strings to ZIP paths.')
    smoke.add_argument("--archives-json", type=Path, required=True)
    extract.add_argument("--max-new-shards", type=int)
    extract.add_argument("--recover-lock", action="store_true", help="Same-host dead-owner recovery only.")
    train.add_argument("--uni-cache", type=Path, required=True)
    train.add_argument("--max-new-fits", type=int)
    train.add_argument("--refit", action="store_true", help="Also create research all-data model; not independent validation.")
    predict.add_argument("--study-root", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-weights":
        result = weight_readiness(args.weights, args.weights_sha256)
    elif args.command == "predict":
        from PIL import Image, ImageOps

        encoder = UniEncoder(args.weights, args.weights_sha256, device=args.device, precision=args.precision)
        with Image.open(args.image) as original:
            if original.width * original.height > 178_000_000:
                raise ValueError("Original inference image exceeds the pixel limit.")
            with ImageOps.exif_transpose(original).convert("RGB") as image:
                result = uni_study.predict_image(args.study_root, image, encoder, batch_size=args.batch_size)
    else:
        if not 0 <= args.reserve_minutes < args.budget_minutes:
            parser.error("Require 0 <= reserve < budget minutes.")
        source = baseline.load_source(args.baseline_release, args.baseline_study)
        if args.command in {"extract", "smoke"}:
            if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.baseline_release, args.baseline_study)):
                parser.error("Output cannot be inside immutable baseline artifacts.")
            archives = {int(lens): Path(path) for lens, path in json.loads(args.archives_json.read_text()).items()}
            encoder = UniEncoder(args.weights, args.weights_sha256, device=args.device, precision=args.precision)
            if args.command == "smoke":
                result = uni_smoke.run(source.rows, archives, args.output, encoder, batch_size=args.batch_size)
            else:
                result = uni_cache.extract(source.rows, archives, args.output, encoder, batch_size=args.batch_size,
                    budget_seconds=(args.budget_minutes - args.reserve_minutes) * 60,
                    max_new_shards=args.max_new_shards, recover_lock=args.recover_lock)
        else:
            incoming = uni_study.source_from_cache(args.uni_cache, source)
            config = baseline.Config(budget_minutes=args.budget_minutes, reserve_minutes=args.reserve_minutes)
            result = uni_study.run_study(incoming, args.output, config, max_new_fits=args.max_new_fits, refit=args.refit)
    # Avoid dumping private image/patient-level records or transport exceptions.
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False))
    return 0 if (result.get("state") == "complete" or result.get("ready") is True
                 or result.get("technical_smoke_passed") is True or "score" in result) else 2


if __name__ == "__main__":
    raise SystemExit(main())
