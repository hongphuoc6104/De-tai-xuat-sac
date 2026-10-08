import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"
METHOD_CONFIGS = {
    "A0_baseline": {"apply_clahe": False, "use_percentile_norm": True},
    "A1_clahe_on": {"apply_clahe": True, "use_percentile_norm": True},
    "A2_no_percentile": {"apply_clahe": False, "use_percentile_norm": False},
    "A3_clahe_no_percentile": {"apply_clahe": True, "use_percentile_norm": False},
}


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5: Image pre-processing based on patient-level split.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Usage examples:\n"
            "  1) Basic run\n"
            "     pixi run python data/Real_data/Script/5_Images_Pre_Processing.py\n"
            "\n"
            "  2) Run with a named method\n"
            "     pixi run python data/Real_data/Script/5_Images_Pre_Processing.py \\\n"
            "       --method A1_clahe_on\n"
            "\n"
            "  3) Run all four ablation methods\n"
            "     pixi run python data/Real_data/Script/5_Images_Pre_Processing.py \\\n"
            "       --run_all_methods\n"
            "\n"
            "  4) Run and save processed images\n"
            "     pixi run python data/Real_data/Script/5_Images_Pre_Processing.py \\\n"
            "       --method A0_baseline \\\n"
            "       --save_images\n"
            "\n"
            "  5) Backward-compatible flag run\n"
            "     pixi run python data/Real_data/Script/5_Images_Pre_Processing.py \\\n"
            "       --apply_clahe --no_percentile_norm\n"
            "\n"
            "Outputs:\n"
            "  ./Results/Step5_Images_Pre_Processing_A0_baseline/log_step5.log\n"
            "  ./Results/Step5_Images_Pre_Processing_A1_clahe_on/log_step5.log\n"
            "  ./Results/Step5_Images_Pre_Processing_A2_no_percentile/log_step5.log\n"
            "  ./Results/Step5_Images_Pre_Processing_A3_clahe_no_percentile/log_step5.log"
        ),
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results root. Defaults to data/Real_data/Results.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional custom output suffix. Normally leave empty to use the A0-A3 method folder.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_CONFIGS),
        default="",
        help="Predefined Step 5 method. Defaults to A0_baseline unless legacy flags imply another method.",
    )
    parser.add_argument(
        "--run_all_methods",
        action="store_true",
        help="Run A0_baseline, A1_clahe_on, A2_no_percentile, and A3_clahe_no_percentile.",
    )
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--apply_clahe", action="store_true", help="Backward-compatible flag.")
    parser.add_argument("--no_percentile_norm", action="store_true", help="Backward-compatible flag.")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--sample_preview", type=int, default=6)
    parser.add_argument(
        "--max_tiles_per_split",
        type=int,
        default=0,
        help="Optional smoke-test limit per split. Use 0 to process all tiles.",
    )
    return parser.parse_args()


def get_method_name(apply_clahe: bool, use_percentile_norm: bool) -> str:
    if apply_clahe and use_percentile_norm:
        return "A1_clahe_on"
    if apply_clahe:
        return "A3_clahe_no_percentile"
    if use_percentile_norm:
        return "A0_baseline"
    return "A2_no_percentile"


def resolve_requested_methods(args):
    if args.run_all_methods:
        return [
            (method_name, config["apply_clahe"], config["use_percentile_norm"])
            for method_name, config in METHOD_CONFIGS.items()
        ]
    if args.method:
        config = METHOD_CONFIGS[args.method]
        return [(args.method, config["apply_clahe"], config["use_percentile_norm"])]

    use_percentile_norm = not args.no_percentile_norm
    method_name = get_method_name(args.apply_clahe, use_percentile_norm)
    return [(method_name, args.apply_clahe, use_percentile_norm)]


def log(msg: str):
    print(msg)


def normalize_percentile_rgb(img_rgb: np.ndarray) -> np.ndarray:
    out = img_rgb.astype(np.float32).copy()
    for c in range(3):
        low = np.percentile(out[..., c], 1)
        high = np.percentile(out[..., c], 99)
        if high - low < 1e-6:
            continue
        out[..., c] = np.clip((out[..., c] - low) * 255.0 / (high - low), 0, 255)
    return out.astype(np.uint8)


def preprocess_tile(
    img_bgr: np.ndarray,
    target_size: int,
    apply_clahe: bool,
    use_percentile_norm: bool,
) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if use_percentile_norm:
        img_rgb = normalize_percentile_rgb(img_rgb)

    if apply_clahe:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img_rgb


def save_preview_grid(df: pd.DataFrame, split: str, src_tiles_root: Path, dst_dir: Path, n_show: int):
    if len(df) == 0 or n_show <= 0:
        return
    sample_df = df.sample(n=min(n_show, len(df)), random_state=42)
    n = len(sample_df)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        img_path = src_tiles_root / row["tile_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(f"{split} | {row['slide_id']} | isup={row['isup']}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(dst_dir / f"preview_original_{split}.png", dpi=200)
    plt.close()


def save_processed_preview_grid(
    df: pd.DataFrame,
    split: str,
    src_tiles_root: Path,
    dst_dir: Path,
    n_show: int,
    target_size: int,
    apply_clahe: bool,
    use_percentile_norm: bool,
):
    if len(df) == 0 or n_show <= 0:
        return
    sample_df = df.sample(n=min(n_show, len(df)), random_state=42)
    n = len(sample_df)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        img_path = src_tiles_root / row["tile_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = preprocess_tile(
            img_bgr=img,
            target_size=target_size,
            apply_clahe=apply_clahe,
            use_percentile_norm=use_percentile_norm,
        )
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(f"{split} | {row['slide_id']} | isup={row['isup']}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(dst_dir / f"preview_processed_{split}.png", dpi=200)
    plt.close()


def plot_distribution(df: pd.DataFrame, col: str, out_png: Path, title: str):
    if len(df) == 0 or col not in df.columns:
        return
    counts = df[col].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([str(x) for x in counts.index], counts.values, color="#4C72B0")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def process_split(
    split_df: pd.DataFrame,
    split_name: str,
    src_tiles_root: Path,
    processed_tiles_root: Path,
    target_size: int,
    apply_clahe: bool,
    use_percentile_norm: bool,
    save_images: bool,
    log,
) -> pd.DataFrame:
    records = []
    brightness_values = []

    iterable = tqdm(split_df.itertuples(index=False), total=len(split_df), desc=f"{split_name} preprocessing")
    for row in iterable:
        src_path = src_tiles_root / row.tile_path
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            continue

        out_rgb = preprocess_tile(
            img_bgr=img_bgr,
            target_size=target_size,
            apply_clahe=apply_clahe,
            use_percentile_norm=use_percentile_norm,
        )
        gray = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY)
        brightness_values.append(float(gray.mean()))

        processed_rel = Path(split_name) / row.tile_path
        if save_images:
            processed_abs = processed_tiles_root / processed_rel
            processed_abs.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(processed_abs), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

        records.append(
            {
                "split": split_name,
                "slide_id": row.slide_id,
                "image_id": getattr(row, "image_id", row.slide_id),
                "patient_id": getattr(row, "patient_id", ""),
                "tile_path": row.tile_path,
                "processed_tile_path": str(processed_rel),
                "isup": int(row.isup),
                "label": int(row.isup > 0),
                "objective_lens": row.objective_lens,
                "objective_label": getattr(row, "objective_label", str(row.objective_lens)),
                "brightness_mean": float(gray.mean()),
                "brightness_std": float(gray.std()),
                "tissue_ratio": float(row.tissue_ratio),
            }
        )

    log(f"{split_name}: processed tiles = {len(records)}")
    if brightness_values:
        log(
            f"{split_name}: brightness mean={np.mean(brightness_values):.2f}, "
            f"std={np.std(brightness_values):.2f}"
        )

    return pd.DataFrame(records)


def run_method(args, results_root, step4_tables, step3_tiles_root, method_name, apply_clahe, use_percentile_norm):
    exp_name = args.exp_name.strip()
    if exp_name and not args.run_all_methods:
        safe_exp = re.sub(r"[^A-Za-z0-9._-]+", "_", exp_name)
        out_root = results_root / f"Step5_Images_Pre_Processing_{safe_exp}"
    else:
        out_root = results_root / f"Step5_Images_Pre_Processing_{method_name}"
    plots_dir = out_root / "Plots"
    tables_dir = out_root / "Tables"
    processed_tiles_dir = out_root / "Processed_Tiles"

    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_tiles_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_root / "log_step5.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_file.open("w", encoding="utf-8") as log_stream:
        sys.stdout = TeeLogger(original_stdout, log_stream)
        sys.stderr = TeeLogger(original_stderr, log_stream)
        try:
            log("STEP 5 - IMAGE PRE-PROCESSING")
            log(f"results_root: {results_root}")
            log(f"step4_tables: {step4_tables}")
            log(f"step3_tiles_root: {step3_tiles_root}")
            log(f"method_name: {method_name}")
            log(f"output_dir: {out_root}")
            log(f"exp_name: {exp_name if exp_name else '(auto method name)'}")
            log(f"target_size: {args.target_size}")
            log(f"apply_clahe: {apply_clahe}")
            log(f"use_percentile_norm: {use_percentile_norm}")
            log(f"save_images: {args.save_images}")

            train_tiles_csv = step4_tables / "train_tiles.csv"
            test_tiles_csv = step4_tables / "test_tiles.csv"
            if not train_tiles_csv.exists() or not test_tiles_csv.exists():
                raise FileNotFoundError(
                    "Missing train/test tiles from Step 4. "
                    "Please run 4_Patient_level_Split.py first."
                )

            train_df = pd.read_csv(train_tiles_csv)
            test_df = pd.read_csv(test_tiles_csv)
            log(f"train tiles loaded: {len(train_df)}")
            log(f"test tiles loaded: {len(test_df)}")
            if args.max_tiles_per_split > 0:
                train_df = train_df.head(args.max_tiles_per_split).copy()
                test_df = test_df.head(args.max_tiles_per_split).copy()
                log(f"max_tiles_per_split: {args.max_tiles_per_split}")
                log(f"train tiles after limit: {len(train_df)}")
                log(f"test tiles after limit: {len(test_df)}")

            save_preview_grid(
                train_df, "train", step3_tiles_root, plots_dir, n_show=args.sample_preview
            )
            save_preview_grid(
                test_df, "test", step3_tiles_root, plots_dir, n_show=args.sample_preview
            )
            save_processed_preview_grid(
                train_df,
                "train",
                step3_tiles_root,
                plots_dir,
                n_show=args.sample_preview,
                target_size=args.target_size,
                apply_clahe=apply_clahe,
                use_percentile_norm=use_percentile_norm,
            )
            save_processed_preview_grid(
                test_df,
                "test",
                step3_tiles_root,
                plots_dir,
                n_show=args.sample_preview,
                target_size=args.target_size,
                apply_clahe=apply_clahe,
                use_percentile_norm=use_percentile_norm,
            )

            train_processed = process_split(
                split_df=train_df,
                split_name="train",
                src_tiles_root=step3_tiles_root,
                processed_tiles_root=processed_tiles_dir,
                target_size=args.target_size,
                apply_clahe=apply_clahe,
                use_percentile_norm=use_percentile_norm,
                save_images=args.save_images,
                log=log,
            )
            test_processed = process_split(
                split_df=test_df,
                split_name="test",
                src_tiles_root=step3_tiles_root,
                processed_tiles_root=processed_tiles_dir,
                target_size=args.target_size,
                apply_clahe=apply_clahe,
                use_percentile_norm=use_percentile_norm,
                save_images=args.save_images,
                log=log,
            )

            train_processed.to_csv(tables_dir / "train_tiles_preprocessed.csv", index=False)
            test_processed.to_csv(tables_dir / "test_tiles_preprocessed.csv", index=False)
            all_df = pd.concat([train_processed, test_processed], ignore_index=True)
            all_df.to_csv(tables_dir / "all_tiles_preprocessed.csv", index=False)

            summary = []
            for split_name, split_df in [("train", train_processed), ("test", test_processed)]:
                summary.append(
                    {
                        "split": split_name,
                        "method": method_name,
                        "target_size": args.target_size,
                        "apply_clahe": apply_clahe,
                        "use_percentile_norm": use_percentile_norm,
                        "n_tiles": int(len(split_df)),
                        "n_slides": int(split_df["slide_id"].nunique()) if len(split_df) else 0,
                        "n_patients": int(split_df["patient_id"].nunique()) if len(split_df) else 0,
                        "n_isup0": int((split_df["label"] == 0).sum()) if len(split_df) else 0,
                        "n_isup_pos": int((split_df["label"] == 1).sum()) if len(split_df) else 0,
                        "brightness_mean": float(split_df["brightness_mean"].mean()) if len(split_df) else np.nan,
                        "brightness_std": float(split_df["brightness_mean"].std()) if len(split_df) else np.nan,
                    }
                )
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(tables_dir / "preprocessing_summary.csv", index=False)

            plot_distribution(
                all_df, "label", plots_dir / "label_distribution_preprocessed.png",
                "Label Distribution (0 vs >0)"
            )
            plot_distribution(
                all_df, "objective_label", plots_dir / "objective_distribution_preprocessed.png",
                "Objective Lens Distribution"
            )

            if len(all_df):
                plt.figure(figsize=(6, 4))
                for split_name, split_df in [("train", train_processed), ("test", test_processed)]:
                    if len(split_df):
                        plt.hist(
                            split_df["brightness_mean"],
                            bins=40,
                            alpha=0.5,
                            label=split_name,
                        )
                plt.xlabel("Tile mean brightness")
                plt.ylabel("Count")
                plt.title("Brightness Distribution After Pre-processing")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "brightness_distribution_preprocessed.png", dpi=200)
                plt.close()

            log("\nPreprocessing summary:")
            log(str(summary_df))
            log(f"Saved tables to: {tables_dir}")
            log(f"Saved plots to: {plots_dir}")
            if args.save_images:
                log(f"Saved processed images to: {processed_tiles_dir}")
            else:
                log("Processed images were not saved (--save_images not set).")
            log("STEP 5 COMPLETED.")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def main():
    args = parse_args()
    results_root = args.results_root.resolve()
    step4_tables = results_root / "Step4_Patient_level_Split" / "Tables"
    step3_tiles_root = results_root / "Step3_Tissue_detection_tiles" / "Tiles"

    for method_name, apply_clahe, use_percentile_norm in resolve_requested_methods(args):
        run_method(
            args=args,
            results_root=results_root,
            step4_tables=step4_tables,
            step3_tiles_root=step3_tiles_root,
            method_name=method_name,
            apply_clahe=apply_clahe,
            use_percentile_norm=use_percentile_norm,
        )


if __name__ == "__main__":
    main()
