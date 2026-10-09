import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Dict, List, Tuple

from data_integrity import assert_disjoint, identity_map, strict_grade, strict_lens, verify_against_metadata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"
EXCEL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")
METHOD_CONFIGS = {
    "A0_baseline": {"apply_clahe": False, "use_percentile_norm": True},
    "A1_clahe_on": {"apply_clahe": True, "use_percentile_norm": True},
    "A2_no_percentile": {"apply_clahe": False, "use_percentile_norm": False},
    "A3_clahe_no_percentile": {"apply_clahe": True, "use_percentile_norm": False},
}

try:
    from torchvision.models import convnext_tiny, efficientnet_b0, vit_b_16
except Exception:  # pragma: no cover
    convnext_tiny = None
    efficientnet_b0 = None
    vit_b_16 = None


MODEL_ORDER = {"efficientnet": 0, "convnext": 1, "vit": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 7 - Evaluate shortlisted models from Step 6 on test set.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Usage examples:
  1) Default shortlist from Step 6 (A0_baseline + EfficientNet/ViT, all magnifications)
     pixi run python data/Real_data/Script/7_Evaluate_Model.py

  2) Custom Step 6 folder and threshold
     pixi run python data/Real_data/Script/7_Evaluate_Model.py \\
       --step6_dir data/Real_data/Results/Step6_Strategy_A_training \\
       --threshold 0.5

Outputs:
  ./Results/Step7_Evaluate_Model/log_step7.log
  ./Results/Step7_Evaluate_Model/Tables/*.csv
  ./Results/Step7_Evaluate_Model/Plots/*.png""",
    )
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--metadata_xlsx", type=Path, default=None)
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Override img_size. If omitted, read from Step6 config.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--step6_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--exp_names", nargs="+", default=["A0_baseline"])
    parser.add_argument("--models", nargs="+", default=["efficientnet", "vit"])
    parser.add_argument("--magnifications", type=int, nargs="+", default=[4, 10, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--patient_id_columns", nargs="+", default=["Ma_Nam", "Ma_So"],
                        help="Confirmed patient identity columns; no automatic year/case inference.")
    return parser.parse_args()


def make_logger(log_file: Path):
    def _log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return _log


def col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch) - 64)
    return value - 1


def norm_numeric_str(value: str) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except ValueError:
        pass
    return s


def read_first_sheet_rows(xlsx_path: Path) -> List[List[str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{EXCEL_NS}si"):
                text = "".join((t.text or "") for t in si.findall(f".//{EXCEL_NS}t"))
                shared_strings.append(text)

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        max_col = 0
        for row in sheet_root.findall(f".//{EXCEL_NS}sheetData/{EXCEL_NS}row"):
            row_map = {}
            for cell in row.findall(f"{EXCEL_NS}c"):
                ref = cell.attrib.get("r", "")
                match = CELL_REF_RE.match(ref)
                if not match:
                    continue
                col_idx = col_to_index(match.group(1))
                ctype = cell.attrib.get("t")
                v_node = cell.find(f"{EXCEL_NS}v")
                is_node = cell.find(f"{EXCEL_NS}is")

                if ctype == "s" and v_node is not None and v_node.text is not None:
                    value = shared_strings[int(v_node.text)]
                elif ctype == "inlineStr" and is_node is not None:
                    value = "".join((t.text or "") for t in is_node.findall(f".//{EXCEL_NS}t"))
                else:
                    value = v_node.text if v_node is not None and v_node.text else ""

                row_map[col_idx] = value
                max_col = max(max_col, col_idx)
            rows.append([row_map.get(i, "") for i in range(max_col + 1)])
    return rows


def build_slide_to_patient_map(metadata_xlsx, patient_columns=None):
    return identity_map(metadata_xlsx, patient_columns)


def resolve_exp_root(results_root: Path, exp_name: str) -> Path:
    if exp_name.strip().lower() in ("", "default", "base"):
        return results_root / "Step5_Images_Pre_Processing"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", exp_name.strip())
    return results_root / f"Step5_Images_Pre_Processing_{safe}"


def resolve_step6_root(results_root: Path, step6_dir: Path | None) -> Path:
    if step6_dir is not None:
        return step6_dir
    return results_root / "Step6_Strategy_A_training"


def get_method_config(exp_name: str) -> Dict[str, bool]:
    if exp_name in METHOD_CONFIGS:
        return METHOD_CONFIGS[exp_name]
    return {"apply_clahe": False, "use_percentile_norm": False}


def attach_image_path(df, exp_root, results_root, exp_name):
    # Always start from the same raw tiles; never mix cached Step5 images and on-load preprocessing.
    df = df.copy()
    raw_root = results_root / "Step3_Tissue_detection_tiles" / "Tiles"
    method = get_method_config(exp_name)
    df["image_path"] = df["tile_path"].map(lambda p: str(raw_root / p))
    df["apply_preprocess_on_load"] = True
    df["apply_clahe"] = method["apply_clahe"]
    df["use_percentile_norm"] = method["use_percentile_norm"]
    return df


def normalize_percentile_rgb(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    for c in range(3):
        low = np.percentile(out[..., c], 1)
        high = np.percentile(out[..., c], 99)
        if high - low < 1e-6:
            continue
        out[..., c] = np.clip((out[..., c] - low) * 255.0 / (high - low), 0, 255)
    return out.astype(np.uint8)


def apply_clahe_rgb(arr: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for CLAHE preprocessing.") from exc

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_rgb_array(arr: np.ndarray, apply_clahe: bool, use_percentile_norm: bool) -> np.ndarray:
    if use_percentile_norm:
        arr = normalize_percentile_rgb(arr)
    if apply_clahe:
        arr = apply_clahe_rgb(arr)
    return arr


class TileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB").resize((self.img_size, self.img_size))
        if bool(row.get("apply_preprocess_on_load", False)):
            arr = np.asarray(img, dtype=np.uint8)
            arr = preprocess_rgb_array(
                arr,
                apply_clahe=bool(row.get("apply_clahe", False)),
                use_percentile_norm=bool(row.get("use_percentile_norm", False)),
            )
            img = Image.fromarray(arr)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr).float()
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


def prepare_test_table(
    df: pd.DataFrame,
    exp_root: Path,
    results_root: Path,
    exp_name: str,
    slide_to_patient: Dict[str, str],
    log,
) -> pd.DataFrame:
    required = ["slide_id", "tile_path", "isup", "objective_lens"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in Step 5 test table: {missing}")

    df = df.copy()
    if slide_to_patient or "patient_id" not in df.columns or df["patient_id"].isna().any():
        if not slide_to_patient:
            raise RuntimeError("Step 5 test table is missing patient_id and Metadata fallback is unavailable.")
        df["patient_id"] = df["slide_id"].map(slide_to_patient)
    if df["patient_id"].isna().any():
        raise ValueError("Missing patient identity; fix metadata instead of dropping tiles.")
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = (df["isup"].map(strict_grade) > 0).astype(int)
    df["objective_lens"] = df["objective_lens"].map(strict_lens)
    df = attach_image_path(df, exp_root, results_root, exp_name)

    missing_images = int((~df["image_path"].map(lambda p: Path(p).exists())).sum())
    if missing_images:
        log(f"[WARN] Missing image files after path resolution: {missing_images}. Stopping.")
        raise FileNotFoundError("Missing tile files; restore the complete dataset before running.")
    n_on_load = int(df["apply_preprocess_on_load"].sum())
    if n_on_load:
        log(
            f"{exp_name}: processed image files not found for {n_on_load} test tiles; "
            "preprocessing will be applied on-the-fly from Step 3 raw tiles."
        )
    return df


class EfficientNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        if efficientnet_b0 is None:
            raise ImportError("torchvision missing for EfficientNet.")
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class ConvNeXtBinary(nn.Module):
    def __init__(self):
        super().__init__()
        if convnext_tiny is None:
            raise ImportError("torchvision missing for ConvNeXt.")
        self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class ViTBinary(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        if vit_b_16 is None:
            raise ImportError("torchvision missing for ViT.")
        self.backbone = vit_b_16(weights=None, image_size=image_size)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def build_model(name: str, img_size: int):
    if name == "efficientnet":
        return EfficientNetBinary()
    if name == "convnext":
        return ConvNeXtBinary()
    if name == "vit":
        if img_size % 16 != 0:
            raise ValueError(f"ViT requires img_size divisible by 16, got {img_size}")
        return ViTBinary(image_size=img_size)
    raise ValueError(f"Unsupported model: {name}")


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iters: int = 1000,
    seed: int = 42,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    auc_samples = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_samples.append(binary_auc(yt, yp))

    if len(auc_samples) == 0:
        return float("nan"), float("nan")

    low_q = (1.0 - alpha) / 2.0
    high_q = 1.0 - low_q
    return float(np.quantile(auc_samples, low_q)), float(np.quantile(auc_samples, high_q))


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "acc": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(1e-8, precision + recall)),
        "auc": binary_auc(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    return calc_metrics(y_true, y_prob, threshold=threshold), y_true, y_prob


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray):
    thresholds = np.r_[1.0, np.sort(np.unique(y_prob))[::-1], 0.0]
    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int64)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))
    return np.array(fpr_list), np.array(tpr_list), thresholds


def calibration_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    pred_mean = []
    obs_freq = []
    counts = []
    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        pred_mean.append(float(y_prob[mask].mean()))
        obs_freq.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(pred_mean), np.array(obs_freq), np.array(counts)


def aggregate_patient_level(df: pd.DataFrame, threshold: float):
    out = (
        df.groupby("patient_id", as_index=False)
        .agg(
            y_true=("patient_label" if "patient_label" in df else "label", "max"),
            y_prob=("y_prob", "mean"),
            n_tiles=("label", "size"),
        )
    )
    out["y_pred"] = (out["y_prob"] >= threshold).astype(int)
    return out


def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, title: str, out_file: Path):
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def safe_name(*parts: str):
    raw = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def fmt_mean_std(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.4f} ± {std_val:.4f}"


def sort_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_model_order"] = out["model_name"].map(lambda x: MODEL_ORDER.get(str(x), 99))
    return out.sort_values(
        by=["exp_name", "_model_order", "magnification", "seed"],
        ascending=[True, True, True, True],
    ).drop(columns=["_model_order"])


def build_group_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "tile_auc_roc",
        "tile_sensitivity",
        "tile_specificity",
        "tile_f1",
        "tile_accuracy",
        "patient_auc_roc",
        "patient_sensitivity",
        "patient_specificity",
        "patient_f1",
        "patient_accuracy",
    ]
    agg = {"n_runs": ("seed", "size")}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    out = df.groupby(group_cols, as_index=False).agg(**agg)
    for col in metric_cols:
        out[f"{col}_std"] = out[f"{col}_std"].fillna(0.0)
        out[f"{col}_mean_std"] = out.apply(
            lambda r: fmt_mean_std(r[f"{col}_mean"], r[f"{col}_std"]),
            axis=1,
        )
    out["_model_order"] = out["model_name"].map(lambda x: MODEL_ORDER.get(str(x), 99)) if "model_name" in out.columns else 0
    sort_cols = [c for c in ["exp_name", "_model_order", "magnification"] if c in out.columns]
    out = out.sort_values(by=sort_cols) if sort_cols else out
    if "_model_order" in out.columns:
        out = out.drop(columns=["_model_order"])
    return out


def save_overall_plot(df: pd.DataFrame, out_file: Path):
    if df.empty:
        return
    labels = [f"{m} | {mag}X" for m, mag in zip(df["model_name"], df["magnification"])]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, df["patient_auc_roc_mean"], color="#4C72B0")
    plt.errorbar(
        x=np.arange(len(df)),
        y=df["patient_auc_roc_mean"],
        yerr=df["patient_auc_roc_std"],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    plt.ylim(0, 1)
    plt.xlabel("Model | Magnification")
    plt.ylabel("Patient-level AUC ROC")
    plt.title("Step 7 Test Evaluation (mean ± std across seeds)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    results_root = args.results_root.resolve()
    metadata_xlsx = (args.metadata_xlsx or (raw_root / "Metadata.xlsx")).resolve()

    step6_root = resolve_step6_root(results_root, args.step6_dir).resolve()
    output_root = (args.output_dir or (results_root / "Step7_Evaluate_Model")).resolve()
    tables_dir = output_root / "Tables"
    plots_dir = output_root / "Plots"
    per_run_tables = tables_dir / "Per_Run"
    per_run_plots = plots_dir / "Per_Run"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    per_run_tables.mkdir(parents=True, exist_ok=True)
    per_run_plots.mkdir(parents=True, exist_ok=True)

    log_file = output_root / "log_step7.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("STEP 7 - Evaluate models from Step 6\n")
    log = make_logger(log_file)

    cfg = {}
    cfg_path = step6_root / "training_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    img_size = args.img_size if args.img_size is not None else int(cfg.get("img_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    log(f"raw_root: {raw_root}")
    log(f"results_root: {results_root}")
    log(f"metadata_xlsx: {metadata_xlsx}")
    log(f"Device: {device}")
    log(f"step6_root: {step6_root}")
    log(f"img_size: {img_size}")
    log(f"threshold: {args.threshold}")
    log(f"bootstrap_iters: {args.bootstrap_iters}")
    log(f"bootstrap_seed: {args.bootstrap_seed}")
    log(f"exp_names: {args.exp_names}")
    log(f"models: {args.models}")
    log(f"magnifications: {args.magnifications}")
    log(f"seeds: {args.seeds if args.seeds is not None else 'all from Step6 summary'}")

    summary_path = step6_root / "Tables" / "summary_all_runs.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary from Step6: {summary_path}")
    runs = pd.read_csv(summary_path)
    if "seed" not in runs.columns:
        raise RuntimeError("Step6 summary_all_runs.csv must contain a 'seed' column.")

    runs = runs[
        runs["exp_name"].isin(args.exp_names)
        & runs["model_name"].isin(args.models)
        & runs["magnification"].astype(int).isin(args.magnifications)
    ].copy()
    if args.seeds is not None:
        runs = runs[runs["seed"].astype(int).isin(args.seeds)].copy()
    runs = sort_runs(runs)

    if runs.empty:
        raise RuntimeError("No Step6 runs matched the requested exp/model/magnification/seed filters.")

    slide_to_patient = {}
    if metadata_xlsx.exists():
        slide_to_patient = build_slide_to_patient_map(metadata_xlsx, args.patient_id_columns)
        log(f"metadata slide->patient mappings loaded: {len(slide_to_patient)}")
    else:
        log("[WARN] Metadata.xlsx not found. This is OK if Step 5 test table already contains patient_id.")
    rows = []
    for rec in runs.to_dict("records"):
        exp = rec["exp_name"]
        model_name = rec["model_name"]
        mag = int(rec["magnification"])
        seed = int(rec["seed"])

        model_file = step6_root / exp / model_name / f"seed_{seed}" / f"{mag}X" / "best_model.pt"
        if not model_file.exists():
            log(f"[WARN] Missing model: {model_file}")
            continue

        exp_root = resolve_exp_root(results_root, exp)
        te_csv = exp_root / "Tables" / "test_tiles_preprocessed.csv"
        if not te_csv.exists():
            log(f"[WARN] Missing test table: {te_csv}")
            continue
        te_df = pd.read_csv(te_csv)
        te_df = prepare_test_table(
            te_df,
            exp_root=exp_root,
            results_root=results_root,
            exp_name=exp,
            slide_to_patient=slide_to_patient,
            log=log,
        )
        te_df = verify_against_metadata(te_df, metadata_xlsx, args.patient_id_columns)
        te_df = te_df[te_df["objective_lens"] == mag].copy()
        if te_df.empty or te_df["label"].nunique() < 2:
            log(f"[WARN] Invalid test set for {exp}|{model_name}|{mag}X|seed={seed}")
            continue

        run_folder = model_file.parent
        audits = []
        for split_name in ["train", "val"]:
            audit_path = run_folder / f"{split_name}_manifest.csv"
            if not audit_path.exists():
                raise FileNotFoundError(f"Missing leakage audit manifest: {audit_path}; rerun corrected Step6.")
            audits.append(pd.read_csv(audit_path, dtype={"patient_id":str}))
        assert_disjoint(*audits, te_df)
        model = build_model(model_name, img_size=img_size).to(device)
        state = torch.load(model_file, map_location=device, weights_only=True)
        model.load_state_dict(state)

        ds = TileDataset(te_df, img_size=img_size)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
        )
        tile_metrics, y_true_tile, y_prob_tile = evaluate(model, dl, device, threshold=args.threshold)
        te_df = te_df.reset_index(drop=True).copy()
        te_df["y_prob"] = y_prob_tile
        te_df["y_pred"] = (te_df["y_prob"] >= args.threshold).astype(int)

        patient_df = aggregate_patient_level(te_df, threshold=args.threshold)
        patient_metrics = calc_metrics(
            patient_df["y_true"].to_numpy(dtype=np.int64),
            patient_df["y_prob"].to_numpy(dtype=np.float64),
            threshold=args.threshold,
        )
        tile_auc_ci_low, tile_auc_ci_high = bootstrap_auc_ci(
            y_true_tile,
            y_prob_tile,
            n_iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            alpha=0.95,
        )
        patient_auc_ci_low, patient_auc_ci_high = bootstrap_auc_ci(
            patient_df["y_true"].to_numpy(dtype=np.int64),
            patient_df["y_prob"].to_numpy(dtype=np.float64),
            n_iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            alpha=0.95,
        )

        tag = safe_name(exp, model_name, f"seed_{seed}", f"{mag}X")
        te_df.to_csv(per_run_tables / f"{tag}_tile_predictions.csv", index=False)
        patient_df.to_csv(per_run_tables / f"{tag}_patient_predictions.csv", index=False)

        fpr_t, tpr_t, _ = roc_curve_points(y_true_tile, y_prob_tile)
        fpr_p, tpr_p, _ = roc_curve_points(
            patient_df["y_true"].to_numpy(dtype=np.int64),
            patient_df["y_prob"].to_numpy(dtype=np.float64),
        )
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_t, tpr_t, label=f"Tile ROC (AUC={tile_metrics['auc']:.3f})")
        plt.plot(fpr_p, tpr_p, label=f"Patient ROC (AUC={patient_metrics['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve | {exp} | {model_name} | seed={seed} | {mag}X")
        plt.legend()
        plt.tight_layout()
        plt.savefig(per_run_plots / f"{tag}_roc_curve.png", dpi=200)
        plt.close()

        pm_t, of_t, _ = calibration_curve_points(y_true_tile, y_prob_tile, n_bins=10)
        pm_p, of_p, _ = calibration_curve_points(
            patient_df["y_true"].to_numpy(dtype=np.int64),
            patient_df["y_prob"].to_numpy(dtype=np.float64),
            n_bins=10,
        )
        plt.figure(figsize=(6, 5))
        if len(pm_t):
            plt.plot(pm_t, of_t, marker="o", label="Tile calibration")
        if len(pm_p):
            plt.plot(pm_p, of_p, marker="s", label="Patient calibration")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive fraction")
        plt.title(f"Calibration Curve | {exp} | {model_name} | seed={seed} | {mag}X")
        plt.legend()
        plt.tight_layout()
        plt.savefig(per_run_plots / f"{tag}_calibration_curve.png", dpi=200)
        plt.close()

        plot_confusion_matrix(
            tp=tile_metrics["tp"],
            tn=tile_metrics["tn"],
            fp=tile_metrics["fp"],
            fn=tile_metrics["fn"],
            title=f"Tile CM | {exp} | {model_name} | seed={seed} | {mag}X",
            out_file=per_run_plots / f"{tag}_tile_confusion_matrix.png",
        )
        plot_confusion_matrix(
            tp=patient_metrics["tp"],
            tn=patient_metrics["tn"],
            fp=patient_metrics["fp"],
            fn=patient_metrics["fn"],
            title=f"Patient CM | {exp} | {model_name} | seed={seed} | {mag}X",
            out_file=per_run_plots / f"{tag}_patient_confusion_matrix.png",
        )

        rows.append(
            {
                "exp_name": exp,
                "model_name": model_name,
                "magnification": mag,
                "seed": seed,
                "threshold": float(args.threshold),
                "tile_auc_roc": tile_metrics["auc"],
                "tile_auc_ci95_low": tile_auc_ci_low,
                "tile_auc_ci95_high": tile_auc_ci_high,
                "tile_auc_ci95": f"[{tile_auc_ci_low:.4f}, {tile_auc_ci_high:.4f}]",
                "tile_sensitivity": tile_metrics["sensitivity"],
                "tile_specificity": tile_metrics["specificity"],
                "tile_f1": tile_metrics["f1"],
                "tile_accuracy": tile_metrics["acc"],
                "tile_precision": tile_metrics["precision"],
                "tile_tp": tile_metrics["tp"],
                "tile_tn": tile_metrics["tn"],
                "tile_fp": tile_metrics["fp"],
                "tile_fn": tile_metrics["fn"],
                "patient_auc_roc": patient_metrics["auc"],
                "patient_auc_ci95_low": patient_auc_ci_low,
                "patient_auc_ci95_high": patient_auc_ci_high,
                "patient_auc_ci95": f"[{patient_auc_ci_low:.4f}, {patient_auc_ci_high:.4f}]",
                "patient_sensitivity": patient_metrics["sensitivity"],
                "patient_specificity": patient_metrics["specificity"],
                "patient_f1": patient_metrics["f1"],
                "patient_accuracy": patient_metrics["acc"],
                "patient_precision": patient_metrics["precision"],
                "patient_tp": patient_metrics["tp"],
                "patient_tn": patient_metrics["tn"],
                "patient_fp": patient_metrics["fp"],
                "patient_fn": patient_metrics["fn"],
                "n_test_tiles": int(len(te_df)),
                "n_test_patients": int(len(patient_df)),
            }
        )
        log(
            f"{exp} | {model_name} | seed={seed} | {mag}X | "
            f"tile_auc={tile_metrics['auc']:.4f} "
            f"CI95=[{tile_auc_ci_low:.4f},{tile_auc_ci_high:.4f}] "
            f"patient_auc={patient_metrics['auc']:.4f} "
            f"CI95=[{patient_auc_ci_low:.4f},{patient_auc_ci_high:.4f}] "
            f"patient_acc={patient_metrics['acc']:.4f}"
        )

    if not rows:
        log("No evaluation result was produced.")
        return

    result_df = sort_runs(pd.DataFrame(rows))
    result_df.to_csv(tables_dir / "summary_test_eval_all_runs.csv", index=False)
    result_df.to_csv(tables_dir / "summary_test_eval.csv", index=False)

    by_exp_model_mag = build_group_summary(result_df, ["exp_name", "model_name", "magnification"])
    by_exp_model_mag.to_csv(tables_dir / "summary_test_eval_mean_std_by_exp_model_mag.csv", index=False)

    by_model_mag = build_group_summary(result_df, ["model_name", "magnification"])
    by_model_mag.to_csv(tables_dir / "summary_test_eval_mean_std_by_model_mag.csv", index=False)

    by_model = build_group_summary(result_df, ["model_name"])
    by_model.to_csv(tables_dir / "summary_test_eval_mean_std_by_model.csv", index=False)

    save_overall_plot(by_model_mag, plots_dir / "step7_patient_auc_by_model_mag.png")

    log(f"Saved: {tables_dir / 'summary_test_eval_all_runs.csv'}")
    log(f"Saved: {tables_dir / 'summary_test_eval_mean_std_by_exp_model_mag.csv'}")
    log(f"Saved: {tables_dir / 'summary_test_eval_mean_std_by_model_mag.csv'}")
    log(f"Saved: {tables_dir / 'summary_test_eval_mean_std_by_model.csv'}")
    log(f"Saved: {plots_dir / 'step7_patient_auc_by_model_mag.png'}")


if __name__ == "__main__":
    main()
