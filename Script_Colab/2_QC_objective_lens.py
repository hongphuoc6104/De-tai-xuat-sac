import argparse
import os
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
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
QC_THRESHOLDS = {
    "4X": {"tissue": 0.03, "blur": 25},
    "10X": {"tissue": 0.05, "blur": 40},
    "40X": {"tissue": 0.05, "blur": 20},
    "DEFAULT": {"tissue": 0.05, "blur": 35},
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
        description="Run objective-lens-aware QC for real microscopy images."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing train.csv and train_images.",
    )
    parser.add_argument(
        "--results_root",
        "--subset_root",
        dest="results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for Step 2 outputs.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata CSV. Defaults to raw_root/train.csv.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit for quick test runs. Default processes all images.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate plots from existing QC_results_objective_lens_aware.csv without rerunning QC.",
    )
    return parser.parse_args()


def image_inventory(image_dir):
    if not image_dir.exists():
        return {}
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def tissue_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    tissue_mask = (s > 20) & (v < 240)
    return float(tissue_mask.mean())


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_lens(lens):
    text = str(lens).strip().upper().replace(" ", "")
    if text.endswith(".0"):
        text = text[:-2]
    if text and not text.endswith("X") and text.isdigit():
        text = f"{text}X"
    return text


def pass_qc_by_lens(tissue_value, blur_value, lens):
    thr = QC_THRESHOLDS.get(normalize_lens(lens), QC_THRESHOLDS["DEFAULT"])
    return (tissue_value > thr["tissue"]) and (blur_value > thr["blur"])


def load_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    required_cols = ["image_id", "isup_grade", "objective_lens"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in metadata CSV: {missing}")

    df["image_id"] = df["image_id"].astype(str).str.strip()
    if "file_name" not in df.columns:
        df["file_name"] = ""
    df["objective_lens"] = df["objective_lens"].map(normalize_lens)
    return df


def resolve_image_path(row, images_by_stem, image_dir):
    file_name = str(getattr(row, "file_name", "")).strip()
    if file_name:
        direct = image_dir / file_name
        if direct.exists():
            return direct

    image_id = str(row.image_id).strip()
    if image_id in images_by_stem:
        return images_by_stem[image_id]

    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_dir / f"{image_id}.tiff"


def run_qc(raw_root, metadata_path, image_dir, qc_root, log_path, max_images=None):
    print("Log file:", log_path)
    print("Raw root:", raw_root)
    print("Metadata:", metadata_path)
    print("Image directory:", image_dir)
    print("Output directory:", qc_root)

    df = load_metadata(metadata_path)
    if max_images is not None:
        df = df.head(max_images).copy()
    images_by_stem = image_inventory(image_dir)
    qc_records = []

    print("\nTotal metadata rows:", len(df))
    print("Total image files found:", len(images_by_stem))
    print("\nObjective lens distribution before QC:")
    print(df["objective_lens"].value_counts().sort_index())

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="QC images",
        mininterval=30,
        miniters=100,
    ):
        img_path = resolve_image_path(row, images_by_stem, image_dir)
        error = ""

        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot read image")

            tr = tissue_ratio(img)
            blur = blur_score(img)
            passed = pass_qc_by_lens(tr, blur, row.objective_lens)

        except Exception as exc:
            tr = np.nan
            blur = np.nan
            passed = False
            error = str(exc)

        qc_records.append(
            {
                "image_id": row.image_id,
                "file_name": getattr(row, "file_name", ""),
                "image_path": str(img_path),
                "isup": row.isup_grade,
                "objective_lens": row.objective_lens,
                "tissue_ratio": tr,
                "blur": blur,
                "pass_qc": passed,
                "error": error,
            }
        )

    qc_df = pd.DataFrame(qc_records)
    qc_results_path = qc_root / "QC_results_objective_lens_aware.csv"
    qc_df.to_csv(qc_results_path, index=False)

    overall_pass_rate = round(qc_df["pass_qc"].mean() * 100, 2)
    failed_read_count = int(qc_df["error"].ne("").sum())
    print("\nFull QC results saved to:", qc_results_path)
    print("Overall QC pass rate:", overall_pass_rate, "%")
    print("Images with read/QC errors:", failed_read_count)

    qc_by_lens = (
        qc_df.groupby("objective_lens")
        .agg(
            total_images=("image_id", "count"),
            pass_qc_images=("pass_qc", "sum"),
            failed_or_unreadable=("error", lambda x: int(x.ne("").sum())),
            mean_tissue_ratio=("tissue_ratio", "mean"),
            median_blur=("blur", "median"),
            pass_rate=("pass_qc", "mean"),
        )
        .reset_index()
    )
    qc_by_lens["pass_rate_%"] = (qc_by_lens["pass_rate"] * 100).round(2)
    qc_by_lens_path = qc_root / "QC_by_objective_lens.csv"
    qc_by_lens.to_csv(qc_by_lens_path, index=False)

    print("\nQC summary by objective lens:")
    print(qc_by_lens)

    qc_by_isup_after_qc = (
        qc_df[qc_df["pass_qc"]]
        .groupby("isup")
        .agg(images_after_qc=("image_id", "count"))
        .reset_index()
    )
    qc_by_isup_path = qc_root / "QC_images_after_qc_by_isup.csv"
    qc_by_isup_after_qc.to_csv(qc_by_isup_path, index=False)

    print("\nImages after QC by ISUP:")
    print(qc_by_isup_after_qc)

    summary_rows = [
        ["Total metadata rows", float(len(df))],
        ["Total image files found", float(len(images_by_stem))],
        ["Overall QC pass rate (%)", overall_pass_rate],
        ["Images passing QC", float(qc_df["pass_qc"].sum())],
        ["Images failing QC", float((~qc_df["pass_qc"]).sum())],
        ["Images with read/QC errors", float(failed_read_count)],
    ]
    for _, row in qc_by_lens.iterrows():
        lens = row["objective_lens"]
        summary_rows.append([f"Lens {lens} total images", float(row["total_images"])])
        summary_rows.append([f"Lens {lens} pass QC images", float(row["pass_qc_images"])])
        summary_rows.append([f"Lens {lens} pass rate (%)", float(row["pass_rate_%"])])

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    summary_path = qc_root / "Step2_QC_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary table saved to:", summary_path)
    print(summary_df)

    plot_qc(qc_df, qc_root)

    print("\n===== STEP 2 FINISHED SUCCESSFULLY =====")


def plot_qc(qc_df, qc_root):
    qc_df = qc_df.copy()
    qc_df["objective_lens"] = qc_df["objective_lens"].map(normalize_lens)
    lens_order = [lens for lens in ("4X", "10X", "40X") if lens in set(qc_df["objective_lens"])]
    if not lens_order:
        lens_order = sorted(qc_df["objective_lens"].dropna().unique().tolist())

    pass_color = "#1b9e77"
    fail_color = "#d95f02"
    fig, axes = plt.subplots(1, len(lens_order), figsize=(6 * len(lens_order), 5), sharey=True)
    if len(lens_order) == 1:
        axes = [axes]

    legend_handles = None
    for ax, lens in zip(axes, lens_order):
        lens_df = qc_df[qc_df["objective_lens"] == lens].copy()
        passed = lens_df[lens_df["pass_qc"]]
        failed = lens_df[~lens_df["pass_qc"]]
        thr = QC_THRESHOLDS.get(lens, QC_THRESHOLDS["DEFAULT"])

        h1 = ax.scatter(
            passed["tissue_ratio"],
            passed["blur"],
            s=20,
            alpha=0.35,
            color=pass_color,
            label="Pass QC",
        )
        h2 = ax.scatter(
            failed["tissue_ratio"],
            failed["blur"],
            s=20,
            alpha=0.35,
            color=fail_color,
            label="Fail QC",
        )
        h3 = ax.axvline(
            thr["tissue"],
            linestyle="--",
            linewidth=2,
            color="blue",
            label="Tissue threshold",
        )
        h4 = ax.axhline(
            thr["blur"],
            linestyle="--",
            linewidth=2,
            color="red",
            label="Blur threshold",
        )
        legend_handles = [h1, h2, h3, h4]

        ax.set_title(f"{lens} | n={len(lens_df)}")
        ax.set_xlabel("Tissue ratio")
        ax.set_xlim(-0.01, 1.02)
        ax.grid(alpha=0.15)
        ax.text(
            0.98,
            0.98,
            f"tissue > {thr['tissue']:.2f}\nblur > {thr['blur']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0].set_ylabel("Blur score")
    fig.suptitle("QC Scatter by Objective Lens", fontsize=18, y=0.98)
    if legend_handles is not None:
        axes[0].legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(qc_root / "QC_scatter_objective_lens.png", dpi=300)
    plt.close(fig)

    fig_t, axes_t = plt.subplots(1, len(lens_order), figsize=(6 * len(lens_order), 4), sharey=True)
    if len(lens_order) == 1:
        axes_t = [axes_t]
    for ax, lens in zip(axes_t, lens_order):
        lens_df = qc_df[qc_df["objective_lens"] == lens]
        ax.hist(
            lens_df.loc[lens_df["pass_qc"], "tissue_ratio"].dropna(),
            bins=30,
            alpha=0.65,
            label="Pass",
            color=pass_color,
        )
        ax.hist(
            lens_df.loc[~lens_df["pass_qc"], "tissue_ratio"].dropna(),
            bins=30,
            alpha=0.65,
            label="Fail",
            color=fail_color,
        )
        ax.set_title(lens)
        ax.set_xlabel("Tissue ratio")
        ax.grid(alpha=0.15)
    axes_t[0].set_ylabel("Count")
    axes_t[0].legend()
    fig_t.suptitle("Tissue Ratio Distribution by Objective Lens", fontsize=18, y=0.98)
    fig_t.tight_layout(rect=(0, 0, 1, 0.92))
    fig_t.savefig(qc_root / "QC_hist_tissue_ratio.png", dpi=300)
    plt.close(fig_t)

    fig_b, axes_b = plt.subplots(1, len(lens_order), figsize=(6 * len(lens_order), 4), sharey=True)
    if len(lens_order) == 1:
        axes_b = [axes_b]
    for ax, lens in zip(axes_b, lens_order):
        lens_df = qc_df[qc_df["objective_lens"] == lens]
        ax.hist(
            lens_df.loc[lens_df["pass_qc"], "blur"].dropna(),
            bins=30,
            alpha=0.65,
            label="Pass",
            color=pass_color,
        )
        ax.hist(
            lens_df.loc[~lens_df["pass_qc"], "blur"].dropna(),
            bins=30,
            alpha=0.65,
            label="Fail",
            color=fail_color,
        )
        ax.set_title(lens)
        ax.set_xlabel("Blur score")
        ax.grid(alpha=0.15)
    axes_b[0].set_ylabel("Count")
    axes_b[0].legend()
    fig_b.suptitle("Blur Score Distribution by Objective Lens", fontsize=18, y=0.98)
    fig_b.tight_layout(rect=(0, 0, 1, 0.92))
    fig_b.savefig(qc_root / "QC_hist_blur.png", dpi=300)
    plt.close(fig_b)


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    metadata_path = (args.metadata or (raw_root / "train.csv")).resolve()
    image_dir = raw_root / "train_images"
    qc_root = args.results_root.resolve() / "Step2_QC_objective_lens_results"
    qc_root.mkdir(parents=True, exist_ok=True)
    log_path = qc_root / "log_step2.log"

    if args.plot_only:
        qc_results_path = qc_root / "QC_results_objective_lens_aware.csv"
        if not qc_results_path.exists():
            raise FileNotFoundError(
                f"Missing existing QC results for plot-only mode: {qc_results_path}"
            )
        qc_df = pd.read_csv(qc_results_path)
        plot_qc(qc_df, qc_root)
        print("Regenerated QC plots from existing QC results:", qc_results_path)
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeLogger(original_stdout, log_file)
        sys.stderr = TeeLogger(original_stderr, log_file)
        try:
            run_qc(raw_root, metadata_path, image_dir, qc_root, log_path, args.max_images)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
