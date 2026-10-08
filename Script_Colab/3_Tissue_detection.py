import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
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

TILE_DIR = None
IMAGE_DIR = None
TILE_SIZE = 512
MAX_TILES_PER_SLIDE = 150


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
        description="Extract tissue-containing tiles from QC-passing microscopy images."
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
        help="Root directory containing Step 2 outputs and Step 3 outputs.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata CSV. Defaults to raw_root/train.csv.",
    )
    parser.add_argument(
        "--qc_results",
        type=Path,
        default=None,
        help="Step 2 QC CSV. Defaults to results_root/Step2_QC_objective_lens_results/QC_results_objective_lens_aware.csv.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size in pixels.",
    )
    parser.add_argument(
        "--max_tiles_per_slide",
        type=int,
        default=150,
        help="Maximum saved tissue tiles per slide.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 16),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--max_slides",
        type=int,
        default=None,
        help="Optional limit for quick test runs. Default processes all QC-passing slides.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tile coordinate shuffling.",
    )
    return parser.parse_args()


def normalize_lens(lens):
    text = str(lens).strip().upper().replace(" ", "")
    if text.endswith(".0"):
        text = text[:-2]
    if text and not text.endswith("X") and text.isdigit():
        text = f"{text}X"
    return text


def objective_to_int(lens):
    text = normalize_lens(lens).replace("X", "")
    return int(float(text)) if text else 10


def image_inventory(image_dir):
    if not image_dir.exists():
        return {}
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def resolve_image_path(row, images_by_stem, image_dir):
    image_path = str(row.get("image_path", "")).strip()
    if image_path and Path(image_path).exists():
        return Path(image_path)

    file_name = str(row.get("file_name", "")).strip()
    if file_name:
        direct = image_dir / file_name
        if direct.exists():
            return direct

    image_id = str(row["image_id"]).strip()
    if image_id in images_by_stem:
        return images_by_stem[image_id]

    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_dir / f"{image_id}.tiff"


def load_slide_table(metadata_path, qc_results_path, image_dir, max_slides=None):
    meta_df = pd.read_csv(metadata_path)
    qc_df = pd.read_csv(qc_results_path)

    if "pass_qc" not in qc_df.columns:
        raise RuntimeError("QC results must contain a pass_qc column.")

    passed_qc = qc_df[qc_df["pass_qc"] == True].copy()
    passed_qc["objective_lens"] = passed_qc["objective_lens"].map(normalize_lens)

    keep_meta = [col for col in ["image_id", "data_provider", "isup_grade"] if col in meta_df.columns]
    slide_df = passed_qc.merge(meta_df[keep_meta], on="image_id", how="left", suffixes=("_qc", ""))

    if "isup_grade" not in slide_df.columns:
        if "isup" in slide_df.columns:
            slide_df["isup_grade"] = slide_df["isup"]
        else:
            raise RuntimeError("Cannot resolve ISUP label from metadata or QC results.")

    if "data_provider" not in slide_df.columns:
        slide_df["data_provider"] = "unknown"

    images_by_stem = image_inventory(image_dir)
    slide_df["resolved_image_path"] = slide_df.apply(
        lambda row: str(resolve_image_path(row, images_by_stem, image_dir)), axis=1
    )
    slide_df["image_found"] = slide_df["resolved_image_path"].map(lambda p: Path(p).exists())
    slide_df = slide_df[slide_df["image_found"]].copy()
    slide_df["objective_int"] = slide_df["objective_lens"].map(objective_to_int)
    slide_df["isup_grade"] = slide_df["isup_grade"].astype(int)

    if max_slides is not None:
        slide_df = slide_df.head(max_slides).copy()

    return slide_df


def get_tissue_mask(img, objective):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    if objective <= 10:
        s_thr, v_thr = 20, 240
    else:
        s_thr, v_thr = 15, 245

    return ((s > s_thr) & (v < v_thr)).astype(np.uint8)


def white_ratio(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return float((gray > 220).mean())


def get_min_tissue_ratio(objective):
    if objective <= 10:
        return 0.03
    if objective <= 20:
        return 0.02
    return 0.04


def init_worker(tile_dir, image_dir, tile_size, max_tiles_per_slide):
    global TILE_DIR, IMAGE_DIR, TILE_SIZE, MAX_TILES_PER_SLIDE
    TILE_DIR = Path(tile_dir)
    IMAGE_DIR = Path(image_dir)
    TILE_SIZE = int(tile_size)
    MAX_TILES_PER_SLIDE = int(max_tiles_per_slide)


def extract_tiles(task):
    slide_id = str(task["image_id"])
    isup = int(task["isup_grade"])
    provider = task.get("data_provider", "unknown")
    objective = int(task.get("objective_int", 10))
    img_path = Path(task["resolved_image_path"])
    has_cancer = bool(isup > 0)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return [], {"slide_id": slide_id, "status": "read_error", "tiles": 0}

    h, w, _ = img.shape
    slide_tile_dir = TILE_DIR / slide_id
    slide_tile_dir.mkdir(exist_ok=True, parents=True)

    stride = TILE_SIZE // 2
    coords = [
        (x, y)
        for y in range(0, h - TILE_SIZE + 1, stride)
        for x in range(0, w - TILE_SIZE + 1, stride)
    ]

    rng = np.random.default_rng(int(task.get("seed", 42)))
    rng.shuffle(coords)

    records = []
    min_tissue_ratio = get_min_tissue_ratio(objective)

    for x, y in coords:
        if len(records) >= MAX_TILES_PER_SLIDE:
            break

        tile = img[y : y + TILE_SIZE, x : x + TILE_SIZE]
        if tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
            continue

        if white_ratio(tile) > 0.995:
            continue

        tissue_mask = get_tissue_mask(tile, objective)
        tissue_ratio = float(tissue_mask.mean())
        if tissue_ratio < min_tissue_ratio:
            continue

        tile_name = f"{x}_{y}.png"
        tile_rel_path = f"{slide_id}/{tile_name}"
        cv2.imwrite(str(slide_tile_dir / tile_name), tile)

        records.append(
            {
                "slide_id": slide_id,
                "image_id": slide_id,
                "tile_path": tile_rel_path,
                "x": x,
                "y": y,
                "isup": isup,
                "has_cancer": has_cancer,
                "provider": provider,
                "objective_lens": objective,
                "objective_label": f"{objective}X",
                "tissue_ratio": tissue_ratio,
            }
        )

    return records, {"slide_id": slide_id, "status": "ok", "tiles": len(records)}


def save_plots(tiles_df, slide_summary_df, plot_dir):
    if tiles_df.empty:
        return

    tiles_per_slide = tiles_df.groupby("slide_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(tiles_per_slide, bins=50, edgecolor="black")
    plt.axvline(tiles_per_slide.mean(), linestyle="--", color="black")
    plt.xlabel("Tiles per slide")
    plt.ylabel("Count")
    plt.title("Tile count per slide")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_tiles_per_slide.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    tiles_df["tissue_ratio"].hist(bins=50, edgecolor="black")
    plt.xlabel("Tile tissue ratio")
    plt.ylabel("Count")
    plt.title("Tile tissue ratio distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_tile_tissue_ratio.png", dpi=300)
    plt.close()

    tiles_by_objective = tiles_df.groupby("objective_label").size().sort_index()
    plt.figure(figsize=(6, 4))
    tiles_by_objective.plot(kind="bar", colormap="Dark2")
    plt.xlabel("Objective lens")
    plt.ylabel("Number of tiles")
    plt.title("Tiles by objective lens")
    plt.tight_layout()
    plt.savefig(plot_dir / "tiles_by_objective_lens.png", dpi=300)
    plt.close()

    if not slide_summary_df.empty:
        slide_summary_df["tiles"].hist(bins=50, edgecolor="black")
        plt.xlabel("Tiles per QC-passing slide")
        plt.ylabel("Count")
        plt.title("Extracted tiles per slide")
        plt.tight_layout()
        plt.savefig(plot_dir / "hist_extracted_tiles_per_slide.png", dpi=300)
        plt.close()


def run_step3(args, raw_root, results_root, metadata_path, qc_results_path, out_dir, log_path):
    print("Log file:", log_path)
    print("Raw root:", raw_root)
    print("Metadata:", metadata_path)
    print("QC results:", qc_results_path)
    print("Output directory:", out_dir)

    image_dir = raw_root / "train_images"
    tile_dir = out_dir / "Tiles"
    plot_dir = out_dir / "Plots"
    tile_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    slide_df = load_slide_table(metadata_path, qc_results_path, image_dir, args.max_slides)
    print("\nQC-passing slides with image files:", len(slide_df))
    print("\nSlides by ISUP:")
    print(slide_df.groupby("isup_grade").size())
    print("\nSlides by objective lens:")
    print(slide_df.groupby("objective_lens").size())

    slide_list_path = out_dir / "slides_used_for_tissue_detection.csv"
    slide_df.to_csv(slide_list_path, index=False)
    print("\nSlide list saved to:", slide_list_path)

    tasks = slide_df[
        ["image_id", "isup_grade", "data_provider", "objective_int", "resolved_image_path"]
    ].to_dict("records")
    for index, task in enumerate(tasks):
        task["seed"] = args.seed + index

    all_records = []
    slide_summaries = []
    workers = max(1, min(int(args.workers), cpu_count()))
    print("Workers:", workers)
    print("Tile size:", args.tile_size)
    print("Max tiles per slide:", args.max_tiles_per_slide)

    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(tile_dir, image_dir, args.tile_size, args.max_tiles_per_slide),
    ) as pool:
        for records, summary in tqdm(
            pool.imap_unordered(extract_tiles, tasks),
            total=len(tasks),
            desc="Extract tissue tiles",
            mininterval=30,
            miniters=10,
        ):
            all_records.extend(records)
            slide_summaries.append(summary)

    tiles_df = pd.DataFrame(all_records)
    slide_summary_df = pd.DataFrame(slide_summaries)

    tiles_path = out_dir / "tiles_metadata.csv"
    slide_summary_path = out_dir / "slide_tile_summary.csv"
    summary_path = out_dir / "Step3_tissue_detection_summary.csv"

    tiles_df.to_csv(tiles_path, index=False)
    slide_summary_df.to_csv(slide_summary_path, index=False)

    print("\nTotal tiles:", len(tiles_df))
    if not tiles_df.empty:
        print("\nTiles per ISUP:")
        print(tiles_df.groupby("isup").size())
        print("\nTiles by has_cancer:")
        print(tiles_df["has_cancer"].value_counts())
        print("\nTiles by objective:")
        print(tiles_df.groupby("objective_lens").size())

    summary_rows = [
        ["QC-passing slides with image files", float(len(slide_df))],
        ["Slides with at least one tile", float((slide_summary_df["tiles"] > 0).sum())],
        ["Slides with zero tiles", float((slide_summary_df["tiles"] == 0).sum())],
        ["Total tiles", float(len(tiles_df))],
        ["Mean tiles per slide", round(float(slide_summary_df["tiles"].mean()), 3) if len(slide_summary_df) else 0],
        ["Median tiles per slide", round(float(slide_summary_df["tiles"].median()), 3) if len(slide_summary_df) else 0],
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    summary_df.to_csv(summary_path, index=False)

    save_plots(tiles_df, slide_summary_df, plot_dir)

    print("\nTiles metadata saved to:", tiles_path)
    print("Slide tile summary saved to:", slide_summary_path)
    print("Summary saved to:", summary_path)
    print(summary_df)
    print("\n===== STEP 3 FINISHED SUCCESSFULLY =====")


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    results_root = args.results_root.resolve()
    metadata_path = (args.metadata or (raw_root / "train.csv")).resolve()
    qc_results_path = (
        args.qc_results
        or (results_root / "Step2_QC_objective_lens_results" / "QC_results_objective_lens_aware.csv")
    ).resolve()
    out_dir = results_root / "Step3_Tissue_detection_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log_step3.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeLogger(original_stdout, log_file)
        sys.stderr = TeeLogger(original_stderr, log_file)
        try:
            run_step3(args, raw_root, results_root, metadata_path, qc_results_path, out_dir, log_path)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
