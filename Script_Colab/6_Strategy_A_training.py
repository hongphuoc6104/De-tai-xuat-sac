import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Dict, List

from data_integrity import (
    assert_disjoint,
    identity_map,
    lock_config,
    patient_predictions,
    split_patients as strict_split,
    strict_grade,
    strict_lens,
    verify_against_metadata,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"
PATIENT_CV_HELPER = SCRIPT_DIR / "8_Patient_Level_CV_Colab.py"
EXCEL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")
METHOD_CONFIGS = {
    "A0_baseline": {"apply_clahe": False, "use_percentile_norm": True},
    "A1_clahe_on": {"apply_clahe": True, "use_percentile_norm": True},
    "A2_no_percentile": {"apply_clahe": False, "use_percentile_norm": False},
    "A3_clahe_no_percentile": {"apply_clahe": True, "use_percentile_norm": False},
}
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from torchvision.models import convnext_tiny, efficientnet_b0, vit_b_16
except Exception:  # pragma: no cover
    convnext_tiny = None
    efficientnet_b0 = None
    vit_b_16 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 6 - Strategy A training with multi-experiment comparison "
            "(A0/A1/A2/A3), supporting EfficientNet/ConvNeXt/ViT."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Usage examples (train + val only, no test in Step 6):\n"
            "  1) Run default A0-A3 with EfficientNet + ConvNeXt + ViT and 5 seeds\n"
            "     pixi run python data/Real_data/Script/6_Strategy_A_training.py\n"
            "\n"
            "  2) Run selected experiments only\n"
            "     pixi run python data/Real_data/Script/6_Strategy_A_training.py \\\n"
            "       --exp_name A0_baseline A1_clahe_on \\\n"
            "       --models efficientnet vit\n"
            "\n"
            "  3) Custom seeds + model-specific LR + early stopping\n"
            "     pixi run python data/Real_data/Script/6_Strategy_A_training.py \\\n"
            "       --seeds 42 52 62 72 82 \\\n"
            "       --lr_efficientnet 1e-3 --lr_convnext 1e-3 --lr_vit 1e-4 \\\n"
            "       --epochs 40 --patience 8\n"
        ),
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results root. Defaults to data/Real_data/Results.",
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Raw data root. Defaults to data/Real_data.",
    )
    parser.add_argument("--metadata_xlsx", type=Path, default=None)
    parser.add_argument(
        "--exp_name",
        nargs="+",
        default=[
            "A0_baseline",
            "A1_clahe_on",
            "A2_no_percentile",
            "A3_clahe_no_percentile",
        ],
        help="One or more Step5 experiment names.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet", "convnext", "vit"],
        choices=["efficientnet", "convnext", "vit"],
    )
    parser.add_argument("--magnifications", type=int, nargs="+", default=[4, 10, 40])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_efficientnet", type=float, default=1e-4)
    parser.add_argument("--lr_convnext", type=float, default=1e-4)
    parser.add_argument("--lr_vit", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--val_ratio_from_train", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--balance",
        choices=["weighted_sampler", "class_weight", "none"],
        default="weighted_sampler",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--patient_id_columns", nargs="+", default=["Ma_Nam", "Ma_So"],
                        help="Confirmed patient identity columns; no automatic year/case inference.")
    return parser.parse_args()


def run_patient_cv_mode_if_requested() -> bool:
    """Delegate Colab patient-level CV mode before the classic Step 6 parser runs."""
    argv = sys.argv[1:]
    if "--mode" not in argv:
        return False

    mode_idx = argv.index("--mode")
    if mode_idx + 1 >= len(argv) or argv[mode_idx + 1] != "patient_cv":
        return False

    filtered_argv = argv[:mode_idx] + argv[mode_idx + 2 :]
    if not PATIENT_CV_HELPER.exists():
        raise FileNotFoundError(f"Missing patient CV helper: {PATIENT_CV_HELPER}")

    spec = importlib.util.spec_from_file_location("patient_level_cv_colab", PATIENT_CV_HELPER)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + filtered_argv
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = original_argv
    return True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": binary_auc(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def split_train_val_by_patient(train_df, val_ratio, seed):
    return strict_split(train_df, val_ratio, seed)


def get_model_lr(model_name: str, args):
    if model_name == "efficientnet":
        return args.lr_efficientnet
    if model_name == "convnext":
        return args.lr_convnext
    if model_name == "vit":
        return args.lr_vit
    raise ValueError(f"Unsupported model: {model_name}")


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
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _augment(self, img: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        last_error = None
        for attempt in range(5):
            try:
                with Image.open(image_path) as im:
                    img = im.convert("RGB").resize((self.img_size, self.img_size))
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        else:
            raise OSError(f"Could not read image after retries: {image_path}") from last_error
        if bool(row.get("apply_preprocess_on_load", False)):
            arr = np.asarray(img, dtype=np.uint8)
            arr = preprocess_rgb_array(
                arr,
                apply_clahe=bool(row.get("apply_clahe", False)),
                use_percentile_norm=bool(row.get("use_percentile_norm", False)),
            )
            img = Image.fromarray(arr)
        if self.train:
            img = self._augment(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr).float()
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


class EfficientNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        if efficientnet_b0 is None:
            raise ImportError("torchvision is required for EfficientNet model.")
        self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class ConvNeXtBinary(nn.Module):
    def __init__(self):
        super().__init__()
        if convnext_tiny is None:
            raise ImportError("torchvision is required for ConvNeXt model.")
        self.backbone = convnext_tiny(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class ViTBinary(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        if vit_b_16 is None:
            raise ImportError("torchvision is required for ViT model. Please install torchvision.")
        if image_size != 224:
            raise ValueError("Pretrained ViT requires --img_size 224.")
        self.backbone = vit_b_16(weights="IMAGENET1K_V1", image_size=image_size)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def build_model(model_name: str, img_size: int):
    if model_name == "efficientnet":
        return EfficientNetBinary()
    if model_name == "convnext":
        return ConvNeXtBinary()
    if model_name == "vit":
        if img_size % 16 != 0:
            raise ValueError(
                f"ViT-B/16 requires img_size divisible by 16. Got img_size={img_size}."
            )
        return ViTBinary(image_size=img_size)
    raise ValueError(f"Unsupported model: {model_name}")


def make_train_loader(train_df, dataset, batch_size, num_workers, balance, pin_memory):
    sampler = None
    if balance == "weighted_sampler":
        counts = train_df["label"].value_counts().to_dict()
        weights = train_df["label"].map(lambda v: 1.0 / max(1, counts.get(int(v), 1))).values
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, frame=None):
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    out = binary_metrics(y_true, y_prob) if len(y_true) else {}
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    if frame is not None:
        predictions = frame.reset_index(drop=True).copy()
        predictions["y_prob"] = y_prob
        pat = patient_predictions(predictions)
        out["tile_auc"] = out["auc"]
        out.update(binary_metrics(pat.y_true.to_numpy(), pat.y_prob.to_numpy()))
    return out


def resolve_exp_root(results_root: Path, exp_name: str) -> Path:
    if exp_name.strip().lower() in ("", "default", "base"):
        return results_root / "Step5_Images_Pre_Processing"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", exp_name.strip())
    return results_root / f"Step5_Images_Pre_Processing_{safe}"


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


def prepare_training_table(
    df: pd.DataFrame,
    exp_root: Path,
    results_root: Path,
    exp_name: str,
    slide_to_patient: Dict[str, str],
    log,
):
    required = ["slide_id", "tile_path", "isup", "objective_lens"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in Step 5 table: {missing}")

    df = df.copy()
    if slide_to_patient or "patient_id" not in df.columns or df["patient_id"].isna().any():
        if not slide_to_patient:
            raise RuntimeError("Step 5 table is missing patient_id and Metadata fallback is unavailable.")
        df["patient_id"] = df["slide_id"].map(slide_to_patient)
    if df["patient_id"].isna().any():
        raise ValueError("Missing patient identity; fix metadata instead of dropping tiles.")
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = (df["isup"].map(strict_grade) > 0).astype(int)
    df["objective_lens"] = df["objective_lens"].map(strict_lens)
    df["patient_label"] = df.groupby("patient_id")["label"].transform("max")
    df = attach_image_path(df, exp_root, results_root, exp_name)

    missing_images = int((~df["image_path"].map(lambda p: Path(p).exists())).sum())
    if missing_images:
        log(f"[WARN] Missing image files after path resolution: {missing_images}. Stopping.")
        raise FileNotFoundError("Missing tile files; restore the complete dataset before running.")
    n_on_load = int(df["apply_preprocess_on_load"].sum())
    if n_on_load:
        log(
            f"{exp_name}: processed image files not found for {n_on_load} tiles; "
            "preprocessing will be applied on-the-fly from Step 3 raw tiles."
        )
    return df


def run_one(
    exp_name: str,
    model_name: str,
    mag: int,
    seed: int,
    train_df: pd.DataFrame,
    out_dir: Path,
    args,
    device,
    use_pin_memory: bool,
    log,
):
    set_seed(seed)
    run_name = f"{exp_name} | {model_name} | {mag}X | seed={seed}"
    log(f"\n{'='*80}\nRun: {run_name}\n{'='*80}")

    train_m = train_df[train_df["objective_lens"] == mag].copy()
    if train_m.empty:
        log("[WARN] Empty train data for this magnification. Skip.")
        return None

    tr_df = va_df = None
    split_ok = False
    for retry in range(50):
        tr_df, va_df = split_train_val_by_patient(
            train_m, args.val_ratio_from_train, seed + retry
        )
        if tr_df["label"].nunique() >= 2 and va_df["label"].nunique() >= 2:
            split_ok = True
            break
    if not split_ok:
        log("[WARN] Could not obtain valid train/val split with both classes. Skip.")
        return None

    run_dir = out_dir / exp_name / model_name / f"seed_{seed}" / f"{mag}X"
    run_dir.mkdir(parents=True, exist_ok=True)

    assert_disjoint(tr_df, va_df)
    lock_config(run_dir, dict(vars(args), exp=exp_name, model=model_name, mag=mag, seed=seed,
                             weights="IMAGENET1K_V1", version="strict-v2",
                             train=tr_df[["slide_id","tile_path","patient_id","label"]].to_dict("records"),
                             val=va_df[["slide_id","tile_path","patient_id","label"]].to_dict("records")))
    tr_df.to_csv(run_dir / "train_manifest.csv", index=False)
    va_df.to_csv(run_dir / "val_manifest.csv", index=False)
    ds_tr = TileDataset(tr_df, img_size=args.img_size, train=True)
    ds_va = TileDataset(va_df, img_size=args.img_size, train=False)

    dl_tr = make_train_loader(
        tr_df, ds_tr, args.batch_size, args.num_workers, args.balance, use_pin_memory
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin_memory
    )

    model = build_model(model_name, img_size=args.img_size).to(device)
    model_lr = get_model_lr(model_name, args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_lr, weight_decay=args.weight_decay
    )

    pos_weight = None
    if args.balance == "class_weight":
        n_pos = int((tr_df["label"] == 1).sum())
        n_neg = int((tr_df["label"] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_auc = -1.0
    best_epoch = -1
    best_f1_at_best_auc = float("nan")
    best_f1_max = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(dl_tr, desc=f"{run_name} | epoch {epoch}/{args.epochs}", leave=False)
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")

        va = evaluate(model, dl_va, criterion, device, va_df)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        for k, v in va.items():
            record[f"val_{k}"] = v
        history.append(record)

        val_auc = va.get("auc", np.nan)
        val_f1 = va.get("f1", np.nan)
        if not np.isnan(val_f1):
            best_f1_max = max(best_f1_max, float(val_f1))
        if not np.isnan(val_auc) and val_auc > (best_auc + args.min_delta):
            best_auc = va["auc"]
            best_epoch = epoch
            best_f1_at_best_auc = float(val_f1) if not np.isnan(val_f1) else float("nan")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        log(
            f"{run_name} | epoch={epoch} "
            f"train_loss={record['train_loss']:.4f} "
            f"val_auc={va.get('auc', np.nan):.4f} "
            f"val_f1={va.get('f1', np.nan):.4f}"
        )

        if no_improve >= args.patience:
            log(
                f"{run_name} | Early stopping at epoch {epoch} "
                f"(patience={args.patience}, min_delta={args.min_delta})"
            )
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint; refusing to save a last-epoch fallback.")
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_model.pt")

    log(f"{run_name} | Training completed (test is reserved for Step 7)")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_name": exp_name,
                "model_name": model_name,
                "magnification": mag,
                "seed": seed,
                "lr": model_lr,
                "best_epoch": int(best_epoch),
                "best_val_auc": float(best_auc),
                "best_val_f1_at_best_auc": float(best_f1_at_best_auc),
                "best_val_f1_max": float(best_f1_max if best_f1_max >= 0 else float("nan")),
                "n_train": int(len(tr_df)),
                "n_val": int(len(va_df)),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    plt.figure(figsize=(7, 4))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    if "val_auc" in hist_df.columns:
        plt.plot(hist_df["epoch"], hist_df["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=200)
    plt.close()

    return {
        "exp_name": exp_name,
        "model_name": model_name,
        "magnification": mag,
        "seed": seed,
        "lr": model_lr,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "best_val_f1_at_best_auc": float(best_f1_at_best_auc),
        "best_val_f1_max": float(best_f1_max if best_f1_max >= 0 else float("nan")),
        "n_train": int(len(tr_df)),
        "n_val": int(len(va_df)),
    }


def main():
    if run_patient_cv_mode_if_requested():
        return

    args = parse_args()
    set_seed(min(args.seeds))
    results_root = args.results_root.resolve()
    raw_root = args.raw_root.resolve()

    metadata_xlsx = (args.metadata_xlsx or (raw_root / "Metadata.xlsx")).resolve()

    output_root = (args.output_dir or (results_root / "Step6_Strategy_A_training")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "Tables"
    plots_dir = output_root / "Plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_root / "log_step6.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("STEP 6 - Strategy A (EfficientNet + ConvNeXt + ViT) with A0-A3 comparison\n")
    log = make_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training is configured for Colab GPU; CPU training is disabled.")
    use_pin_memory = device.type == "cuda"
    log(f"raw_root: {raw_root}")
    log(f"results_root: {results_root}")
    log(f"metadata_xlsx: {metadata_xlsx}")
    log(f"output_root: {output_root}")
    log(f"Device: {device}")
    log(f"pin_memory: {use_pin_memory}")
    log(f"seeds: {args.seeds}")
    log(
        f"model_lrs: efficientnet={args.lr_efficientnet}, "
        f"convnext={args.lr_convnext}, vit={args.lr_vit}"
    )
    log(f"early_stopping: patience={args.patience}, min_delta={args.min_delta}")
    log(f"Experiments: {args.exp_name}")
    log(f"Models: {args.models}")
    log(f"Magnifications: {args.magnifications}")

    slide_to_patient = {}
    if metadata_xlsx.exists():
        slide_to_patient = build_slide_to_patient_map(metadata_xlsx, args.patient_id_columns)
        log(f"metadata slide->patient mappings loaded: {len(slide_to_patient)}")
    else:
        log("[WARN] Metadata.xlsx not found. This is OK if Step 5 tables already contain patient_id.")
    all_rows = []

    for exp_name in args.exp_name:
        exp_root = resolve_exp_root(results_root, exp_name)
        tr_csv = exp_root / "Tables" / "train_tiles_preprocessed.csv"
        if not tr_csv.exists():
            log(f"[WARN] Missing preprocessed train table for exp={exp_name}. Skip.")
            continue

        tr_df = pd.read_csv(tr_csv)
        tr_df = prepare_training_table(
            tr_df,
            exp_root=exp_root,
            results_root=results_root,
            exp_name=exp_name,
            slide_to_patient=slide_to_patient,
            log=log,
        )
        tr_df = verify_against_metadata(tr_df, metadata_xlsx, args.patient_id_columns)
        test_csv = exp_root / "Tables" / "test_tiles_preprocessed.csv"
        if not test_csv.exists():
            raise FileNotFoundError("Need the held-out test manifest to audit leakage before training.")
        test_audit = pd.read_csv(test_csv)
        if slide_to_patient:
            test_audit["patient_id"] = test_audit["slide_id"].map(slide_to_patient)
        test_audit["label"] = (test_audit["isup"].map(strict_grade) > 0).astype(int)
        test_audit = verify_against_metadata(test_audit, metadata_xlsx, args.patient_id_columns)
        assert_disjoint(tr_df, test_audit)
        log(
            f"{exp_name}: train tiles={len(tr_df)}, "
            f"slides={tr_df['slide_id'].nunique()}, patients={tr_df['patient_id'].nunique()}, "
            f"positive_tiles={int((tr_df['label'] == 1).sum())}"
        )

        for model_name in args.models:
            if model_name == "efficientnet" and efficientnet_b0 is None:
                log("[WARN] torchvision missing. Skip EfficientNet.")
                continue
            if model_name == "convnext" and convnext_tiny is None:
                log("[WARN] torchvision missing. Skip ConvNeXt.")
                continue
            if model_name == "vit" and vit_b_16 is None:
                log("[WARN] torchvision missing. Skip ViT.")
                continue
            for mag in args.magnifications:
                for seed in args.seeds:
                    row = run_one(
                        exp_name=exp_name,
                        model_name=model_name,
                        mag=mag,
                        seed=seed,
                        train_df=tr_df,
                        out_dir=output_root,
                        args=args,
                        device=device,
                        use_pin_memory=use_pin_memory,
                        log=log,
                    )
                    if row is not None:
                        all_rows.append(row)

    if not all_rows:
        log("No successful run.")
        return

    summary = pd.DataFrame(all_rows).sort_values(by=["best_val_auc"], ascending=False)
    summary.to_csv(tables_dir / "summary_all_runs.csv", index=False)

    agg_exp_model_mag = (
        summary.groupby(["exp_name", "model_name", "magnification"], as_index=False)
        .agg(
            n_runs=("best_val_auc", "size"),
            best_val_auc_mean=("best_val_auc", "mean"),
            best_val_auc_std=("best_val_auc", "std"),
            best_val_f1_at_best_auc_mean=("best_val_f1_at_best_auc", "mean"),
            best_val_f1_at_best_auc_std=("best_val_f1_at_best_auc", "std"),
            best_val_f1_max_mean=("best_val_f1_max", "mean"),
            best_val_f1_max_std=("best_val_f1_max", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_std=("best_epoch", "std"),
        )
        .sort_values(by=["best_val_auc_mean"], ascending=False)
    )
    agg_exp_model_mag["best_val_auc_std"] = agg_exp_model_mag["best_val_auc_std"].fillna(0.0)
    agg_exp_model_mag["best_val_f1_at_best_auc_std"] = agg_exp_model_mag["best_val_f1_at_best_auc_std"].fillna(0.0)
    agg_exp_model_mag["best_val_f1_max_std"] = agg_exp_model_mag["best_val_f1_max_std"].fillna(0.0)
    agg_exp_model_mag["best_epoch_std"] = agg_exp_model_mag["best_epoch_std"].fillna(0.0)
    agg_exp_model_mag["best_val_auc_mean_std"] = agg_exp_model_mag.apply(
        lambda r: f"{r['best_val_auc_mean']:.4f} ± {r['best_val_auc_std']:.4f}", axis=1
    )
    agg_exp_model_mag["best_val_f1_at_best_auc_mean_std"] = agg_exp_model_mag.apply(
        lambda r: f"{r['best_val_f1_at_best_auc_mean']:.4f} ± {r['best_val_f1_at_best_auc_std']:.4f}", axis=1
    )
    agg_exp_model_mag["best_val_f1_max_mean_std"] = agg_exp_model_mag.apply(
        lambda r: f"{r['best_val_f1_max_mean']:.4f} ± {r['best_val_f1_max_std']:.4f}", axis=1
    )
    agg_exp_model_mag.to_csv(tables_dir / "summary_mean_std_by_exp_model_mag.csv", index=False)

    agg_exp_model = (
        summary.groupby(["exp_name", "model_name"], as_index=False)
        .agg(
            n_runs=("best_val_auc", "size"),
            best_val_auc_mean=("best_val_auc", "mean"),
            best_val_auc_std=("best_val_auc", "std"),
            best_val_f1_at_best_auc_mean=("best_val_f1_at_best_auc", "mean"),
            best_val_f1_at_best_auc_std=("best_val_f1_at_best_auc", "std"),
            best_val_f1_max_mean=("best_val_f1_max", "mean"),
            best_val_f1_max_std=("best_val_f1_max", "std"),
        )
        .sort_values(by=["best_val_auc_mean"], ascending=False)
    )
    agg_exp_model["best_val_auc_std"] = agg_exp_model["best_val_auc_std"].fillna(0.0)
    agg_exp_model["best_val_f1_at_best_auc_std"] = agg_exp_model["best_val_f1_at_best_auc_std"].fillna(0.0)
    agg_exp_model["best_val_f1_max_std"] = agg_exp_model["best_val_f1_max_std"].fillna(0.0)
    agg_exp_model["best_val_auc_mean_std"] = agg_exp_model.apply(
        lambda r: f"{r['best_val_auc_mean']:.4f} ± {r['best_val_auc_std']:.4f}", axis=1
    )
    agg_exp_model["best_val_f1_at_best_auc_mean_std"] = agg_exp_model.apply(
        lambda r: f"{r['best_val_f1_at_best_auc_mean']:.4f} ± {r['best_val_f1_at_best_auc_std']:.4f}", axis=1
    )
    agg_exp_model["best_val_f1_max_mean_std"] = agg_exp_model.apply(
        lambda r: f"{r['best_val_f1_max_mean']:.4f} ± {r['best_val_f1_max_std']:.4f}", axis=1
    )
    agg_exp_model.to_csv(tables_dir / "summary_by_exp_model.csv", index=False)

    agg_exp = (
        summary.groupby("exp_name", as_index=False)
        .agg(
            n_runs=("best_val_auc", "size"),
            best_val_auc_mean=("best_val_auc", "mean"),
            best_val_auc_std=("best_val_auc", "std"),
            best_val_f1_at_best_auc_mean=("best_val_f1_at_best_auc", "mean"),
            best_val_f1_at_best_auc_std=("best_val_f1_at_best_auc", "std"),
            best_val_f1_max_mean=("best_val_f1_max", "mean"),
            best_val_f1_max_std=("best_val_f1_max", "std"),
        )
        .sort_values(by="best_val_auc_mean", ascending=False)
    )
    agg_exp["best_val_auc_std"] = agg_exp["best_val_auc_std"].fillna(0.0)
    agg_exp["best_val_f1_at_best_auc_std"] = agg_exp["best_val_f1_at_best_auc_std"].fillna(0.0)
    agg_exp["best_val_f1_max_std"] = agg_exp["best_val_f1_max_std"].fillna(0.0)
    agg_exp["best_val_auc_mean_std"] = agg_exp.apply(
        lambda r: f"{r['best_val_auc_mean']:.4f} ± {r['best_val_auc_std']:.4f}", axis=1
    )
    agg_exp["best_val_f1_at_best_auc_mean_std"] = agg_exp.apply(
        lambda r: f"{r['best_val_f1_at_best_auc_mean']:.4f} ± {r['best_val_f1_at_best_auc_std']:.4f}", axis=1
    )
    agg_exp["best_val_f1_max_mean_std"] = agg_exp.apply(
        lambda r: f"{r['best_val_f1_max_mean']:.4f} ± {r['best_val_f1_max_std']:.4f}", axis=1
    )
    agg_exp.to_csv(tables_dir / "summary_compare_A0_A3.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(agg_exp["exp_name"], agg_exp["best_val_auc_mean"], color="#4C72B0")
    plt.ylim(0, 1)
    plt.xlabel("Preprocess experiment")
    plt.ylabel("Mean val AUC")
    plt.title("A0-A3 Comparison (Mean val AUC over seeds/models/magnifications)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_A0_A3_mean_auc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(agg_exp["exp_name"], agg_exp["best_val_f1_at_best_auc_mean"], color="#55A868")
    plt.ylim(0, 1)
    plt.xlabel("Preprocess experiment")
    plt.ylabel("Mean val F1 (at best AUC epoch)")
    plt.title("A0-A3 Comparison (Mean val F1 @ best AUC epoch)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_A0_A3_mean_f1_at_best_auc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(agg_exp["exp_name"], agg_exp["best_val_f1_max_mean"], color="#C44E52")
    plt.ylim(0, 1)
    plt.xlabel("Preprocess experiment")
    plt.ylabel("Mean max val F1")
    plt.title("A0-A3 Comparison (Mean max val F1)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_A0_A3_mean_f1_max.png", dpi=200)
    plt.close()

    log(f"Saved: {tables_dir / 'summary_all_runs.csv'}")
    log(f"Saved: {tables_dir / 'summary_mean_std_by_exp_model_mag.csv'}")
    log(f"Saved: {tables_dir / 'summary_by_exp_model.csv'}")
    log(f"Saved: {tables_dir / 'summary_compare_A0_A3.csv'}")
    log(f"Saved: {plots_dir / 'comparison_A0_A3_mean_auc.png'}")
    log(f"Saved: {plots_dir / 'comparison_A0_A3_mean_f1_at_best_auc.png'}")
    log(f"Saved: {plots_dir / 'comparison_A0_A3_mean_f1_max.png'}")
    log("\nTop runs:")
    log(summary.head(10).to_string(index=False))

    with open(output_root / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "img_size": args.img_size,
                "weights": "IMAGENET1K_V1",
                "selection_metric": "val_patient_auc",
                "patient_id_columns": args.patient_id_columns,
                "models": args.models,
                "magnifications": args.magnifications,
                "exp_name": args.exp_name,
                "seeds": args.seeds,
                "lr_efficientnet": args.lr_efficientnet,
                "lr_convnext": args.lr_convnext,
                "lr_vit": args.lr_vit,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "step6_mode": "train_val_only",
                "tracked_metrics": ["best_val_auc", "best_val_f1_at_best_auc", "best_val_f1_max"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
