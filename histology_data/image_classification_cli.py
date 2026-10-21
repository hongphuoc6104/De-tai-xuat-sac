"""CLI for inventory, frozen feature extraction, grouped training, and reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .image_classification import (
    FeatureExtractionConfig,
    OriginalImageInventoryConfig,
    TrainingConfig,
    build_original_image_inventory,
    extract_original_image_features,
    render_image_classification_report,
    train_original_image_classifier,
)


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="audit original image metadata and three source ZIPs")
    inventory.add_argument("--metadata", type=Path, required=True)
    inventory.add_argument("--archive-4x", type=Path, required=True)
    inventory.add_argument("--archive-10x", type=Path, required=True)
    inventory.add_argument("--archive-40x", type=Path, required=True)
    inventory.add_argument("--output-root", type=Path, required=True)
    inventory.add_argument("--exclude-missing", action="store_true",
                           help="record metadata rows with no matching source image as explicit exclusions")
    inventory.add_argument("--exclude-unknown-grade", action="store_true",
                           help="record missing/unsupported Glade values as explicit exclusions")

    extract = commands.add_parser("extract", help="extract checksum-bound frozen features from original images")
    extract.add_argument("--inventory", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--weights-dir", type=Path, required=True)
    extract.add_argument("--archive-4x", type=Path, help="runtime SSD path override for the inventoried 4X ZIP")
    extract.add_argument("--archive-10x", type=Path, help="runtime SSD path override for the inventoried 10X ZIP")
    extract.add_argument("--archive-40x", type=Path, help="runtime SSD path override for the inventoried 40X ZIP")
    extract.add_argument("--batch-size", type=int, default=2)
    extract.add_argument("--shard-size", type=int, default=32)
    extract.add_argument("--max-long-edge", type=int, default=512)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--precision", choices=["fp32", "amp_fp16"], default="fp32")
    extract.add_argument("--budget-minutes", type=float, default=240.0)
    extract.add_argument("--reserve-minutes", type=float, default=30.0)
    extract.add_argument("--smoke-max-images-per-lens", type=int,
                         help="explicitly cap each lens for an isolated, non-training-ready smoke release")
    extract.add_argument("--recover-lock", action="store_true",
                         help="recover a same-host run lock only when its recorded PID is verified dead")

    train = commands.add_parser("train", help="run three-fold patient-grouped nested logistic regression")
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--inner-folds", type=int, default=3)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--c-values", default="0.01,0.1,1,10,100")
    train.add_argument("--max-iter", type=int, default=2000)
    train.add_argument("--budget-minutes", type=float, default=240.0)
    train.add_argument("--reserve-minutes", type=float, default=15.0)
    train.add_argument("--no-balanced-patient-weights", action="store_true")
    train.add_argument("--recover-lock", action="store_true")

    report = commands.add_parser("report", help="verify and display the saved OOF image-level report")
    report.add_argument("--study-root", type=Path, required=True)

    lab = commands.add_parser("head-lab", help="fixed-patient cached weight/head/PCA studies and source review")
    lab.add_argument("--run-root", type=Path, required=True)
    lab.add_argument("--baseline-root", type=Path, required=True)
    lab.add_argument("--output-root", type=Path, required=True)
    lab.add_argument("--variants", default="patient_lr,class_lr,uniform_lr,patient_pca32,patient_pca64,per_lens_lr,per_lens_pca32")
    lab.add_argument("--budget-minutes", type=float, default=120)
    lab.add_argument("--reserve-minutes", type=float, default=15)
    for lens in (4, 10, 40):
        lab.add_argument(f"--archive-{lens}x", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "head-lab":
            from .image_classification_lab import run_lab

            archives = {lens: getattr(args, f"archive_{lens}x") for lens in (4, 10, 40)
                        if getattr(args, f"archive_{lens}x") is not None}
            result = run_lab(args.run_root, args.baseline_root, args.output_root,
                             variants=tuple(x.strip() for x in args.variants.split(",") if x.strip()),
                             budget_minutes=args.budget_minutes, reserve_minutes=args.reserve_minutes,
                             archives=archives or None, notify=_json_print)
            _json_print({k: v for k, v in result.items() if k != "studies"})
            return 0 if result["state"] == "complete" else 2
        if args.command == "inventory":
            manifest = build_original_image_inventory(OriginalImageInventoryConfig(
                metadata=args.metadata,
                archives={4: args.archive_4x, 10: args.archive_10x, 40: args.archive_40x},
                output_root=args.output_root,
                exclude_missing=args.exclude_missing,
                exclude_unknown_grade=args.exclude_unknown_grade,
            ))
            content = json.loads(manifest.read_text(encoding="utf-8"))
            _json_print({"phase": "inventory_complete", "inventory_id": content["inventory_id"],
                         "manifest": str(manifest), "image_count": len(content["images"]),
                         "coverage": content["coverage"], "exclusions": len(content["exclusions"])})
            return 0
        if args.command == "extract":
            overrides = {lens: path for lens, path in ((4, args.archive_4x), (10, args.archive_10x), (40, args.archive_40x))
                         if path is not None}
            run_root = extract_original_image_features(
                args.inventory, args.output_root, args.weights_dir,
                archive_paths=overrides or None,
                config=FeatureExtractionConfig(
                    batch_size=args.batch_size,
                    shard_size=args.shard_size,
                    max_long_edge=args.max_long_edge,
                    device=args.device,
                    precision=args.precision,
                    budget_minutes=args.budget_minutes,
                    reserve_minutes=args.reserve_minutes,
                    smoke_max_images_per_lens=args.smoke_max_images_per_lens,
                    recover_lock=args.recover_lock,
                ),
            )
            status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
            _json_print({"phase": status["state"], "run_root": str(run_root), **status})
            return 0 if status["state"] == "complete" else 2
        if args.command == "train":
            try:
                c_values = tuple(float(value.strip()) for value in args.c_values.split(",") if value.strip())
            except ValueError as exc:
                raise ValueError("--c-values must be comma-separated positive numbers.") from exc
            result = train_original_image_classifier(args.run_root, config=TrainingConfig(
                inner_folds=args.inner_folds,
                seed=args.seed,
                c_values=c_values,
                balanced_patient_weights=not args.no_balanced_patient_weights,
                max_iter=args.max_iter,
                budget_minutes=args.budget_minutes,
                reserve_minutes=args.reserve_minutes,
                recover_lock=args.recover_lock,
            ))
            _json_print({"phase": result["state"], **result})
            return 0 if result["state"] == "complete" else 2
        result = render_image_classification_report(args.study_root)
        _json_print({"phase": "report_complete", **result})
        return 0
    except Exception as exc:
        _json_print({"phase": "error", "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
