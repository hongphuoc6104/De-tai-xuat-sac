import argparse
from pathlib import Path

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--raw_root", type=Path, required=True,
                    help="Root directory of raw dataset")
parser.add_argument("--subset_root", type=Path, required=True,
                    help="Root directory for processed subset / outputs")

args = parser.parse_args()

RAW_ROOT = args.raw_root
SUBSET_ROOT = args.subset_root

# =========================
# Dataset paths
# =========================
TRAIN_CSV = RAW_ROOT / "train.csv"
IMAGE_DIR = RAW_ROOT / "train_images"

# Mask directory (OPTIONAL)
# - PANDA: exists
# - Real microscopy data: usually NOT exists
MASK_DIR = RAW_ROOT / "train_label_masks"
HAS_MASK = MASK_DIR.exists()

# =========================
# Subset selection config
# =========================
N_SLIDES_PER_CLASS = 40        # ~40 × 6 = 240 slides total
RANDOM_SEED = 42

# =========================
# Tile extraction config
# =========================
TILE_SIZE = 512               # 512 × 512 pixels
MAX_TILES_PER_SLIDE = 150     # cap to avoid slide dominance

# =========================
# QC / Filtering thresholds
# =========================
MIN_TISSUE_RATIO = 0.05       # ≥ 5% tissue required in tile
BLUR_THRESHOLD = 50.0         # variance of Laplacian (focus check)
