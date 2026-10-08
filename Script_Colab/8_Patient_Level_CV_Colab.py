import argparse
import importlib.util
import json
import os
import random
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_integrity import validate_frame, assert_disjoint, lock_config, patient_predictions, verify_against_metadata
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "Step6_CV_patient_level_colab"
STEP6_PATH = SCRIPT_DIR / "6_Strategy_A_training.py"


def load_step6_module():
    spec = importlib.util.spec_from_file_location("step6_training_colab", STEP6_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step6 = load_step6_module()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Patient-level cross-validation training for Colab. "
            "Each outer fold is a held-out patient test fold; threshold is optimized "
            "on validation patients and then locked for that fold's test patients."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Recommended Colab smoke test:
  python data/Real_data/Script_Colab/8_Patient_Level_CV_Colab.py \\
    --exp_name A2_no_percentile \\
    --models efficientnet \\
    --magnifications 40 \\
    --seeds 42 \\
    --n_splits 3 \\
    --epochs 2

Recommended serious run on Colab:
  python data/Real_data/Script_Colab/8_Patient_Level_CV_Colab.py \\
    --exp_name A2_no_percentile A3_clahe_no_percentile \\
    --models efficientnet \\
    --magnifications 40 \\
    --seeds 42 52 62 \\
    --n_splits 5 \\
    --epochs 25 \\
    --patience 5""",
    )
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--metadata_xlsx", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exp_name", nargs="+", default=["A2_no_percentile"])
    parser.add_argument("--models", nargs="+", default=["efficientnet"], choices=["efficientnet", "convnext", "vit"])
    parser.add_argument("--magnifications", type=int, nargs="+", default=[40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--cv_source", choices=["train_only", "train_plus_test"], default="train_only")
    parser.add_argument("--val_ratio_from_train", type=float, default=0.2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr_efficientnet", type=float, default=1e-4)
    parser.add_argument("--lr_convnext", type=float, default=1e-4)
    parser.add_argument("--lr_vit", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--balance", choices=["weighted_sampler", "class_weight", "none"], default="weighted_sampler")
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "topk_mean", "softmax_attention"],
        choices=["mean", "topk_mean", "softmax_attention"],
    )
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--attention_temperature", type=float, default=0.10)
    parser.add_argument("--threshold_metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--patient_id_columns", nargs="+", default=["Ma_Nam", "Ma_So"])
    return parser.parse_args()


def make_logger(log_file: Path):
    def _log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return _log


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


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * sensitivity / max(1e-8, precision + sensitivity)
    return {
        "auc": binary_auc(y_true, y_prob),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str, step: float) -> Tuple[float, Dict[str, float]]:
    thresholds = np.arange(0.0, 1.0 + step / 2.0, step)
    rows = []
    for thr in thresholds:
        m = calc_metrics(y_true, y_prob, threshold=float(thr))
        rows.append((float(thr), m))
    rows = sorted(rows, key=lambda x: (-x[1][metric], abs(x[0] - 0.5)))
    return rows[0]


def aggregate_patient_probs(tile_df: pd.DataFrame, method: str, topk: int, temperature: float) -> pd.DataFrame:
    rows = []
    for patient_id, g in tile_df.groupby("patient_id"):
        probs = g["y_prob"].to_numpy(dtype=np.float64)
        y_true = int(g["patient_label" if "patient_label" in g else "label"].max())
        if method == "mean":
            y_prob = float(probs.mean())
        elif method == "topk_mean":
            k = max(1, min(int(topk), len(probs)))
            y_prob = float(np.sort(probs)[-k:].mean())
        elif method == "softmax_attention":
            temp = max(float(temperature), 1e-6)
            z = probs / temp
            z = z - z.max()
            weights = np.exp(z)
            weights = weights / max(weights.sum(), 1e-12)
            y_prob = float((weights * probs).sum())
        else:
            raise ValueError(f"Unsupported aggregation: {method}")
        rows.append({"patient_id": patient_id, "y_true": y_true, "y_prob": y_prob, "n_tiles": int(len(g))})
    return pd.DataFrame(rows)


def stratified_patient_folds(df: pd.DataFrame, n_splits: int, seed: int) -> List[set]:
    labels = validate_frame(df)
    if n_splits < 2 or labels.value_counts().min() < n_splits:
        raise ValueError("n_splits must be >=2 and <= number of patients in the smallest class.")
    patient_label = labels.rename("label").reset_index()
    pos = patient_label[patient_label["label"] == 1]["patient_id"].tolist()
    neg = patient_label[patient_label["label"] == 0]["patient_id"].tolist()
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [set() for _ in range(n_splits)]
    for i, patient_id in enumerate(pos):
        folds[i % n_splits].add(patient_id)
    for i, patient_id in enumerate(neg):
        folds[i % n_splits].add(patient_id)
    return folds


def split_inner_train_val(train_df: pd.DataFrame, val_ratio: float, seed: int):
    split_ok = False
    tr_df = va_df = None
    for retry in range(100):
        tr_df, va_df = step6.split_train_val_by_patient(train_df, val_ratio, seed + retry)
        if tr_df["label"].nunique() >= 2 and va_df["label"].nunique() >= 2:
            split_ok = True
            break
    if not split_ok:
        raise RuntimeError("Could not obtain inner train/validation split with both classes.")
    return tr_df, va_df


@torch.no_grad()
def predict_tiles(model, df: pd.DataFrame, args, device, use_pin_memory: bool) -> pd.DataFrame:
    ds = step6.TileDataset(df, img_size=args.img_size, train=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    model.eval()
    ys, ps = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    out = df.reset_index(drop=True).copy()
    out["y_prob"] = np.concatenate(ps).astype(np.float64)
    out["label"] = np.concatenate(ys).astype(np.int64)
    return out


def train_one_fold(exp_name, model_name, mag, seed, fold_id, train_df, val_df, test_df, out_dir, args, device, use_pin_memory, log):
    step6.set_seed(seed + fold_id * 1000)
    assert_disjoint(train_df, val_df, test_df)
    tag = f"{exp_name} | {model_name} | {mag}X | seed={seed} | fold={fold_id}"
    log(f"\n{'=' * 88}\n{tag}\n{'=' * 88}")
    run_dir = out_dir / exp_name / model_name / f"seed_{seed}" / f"{mag}X" / f"fold_{fold_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lock_config(run_dir, dict(vars(args), exp=exp_name, model=model_name, mag=mag, seed=seed,
                             fold=fold_id, version="strict-v2", weights="IMAGENET1K_V1",
                             train=train_df[["slide_id","tile_path","patient_id","label"]].to_dict("records"),
                             val=val_df[["slide_id","tile_path","patient_id","label"]].to_dict("records")))
    for split_name, frame in [("train",train_df),("val",val_df),("test",test_df)]:
        frame.to_csv(run_dir / f"{split_name}_manifest.csv", index=False)
    ds_tr = step6.TileDataset(train_df, img_size=args.img_size, train=True)
    ds_va = step6.TileDataset(val_df, img_size=args.img_size, train=False)
    dl_tr = step6.make_train_loader(train_df, ds_tr, args.batch_size, args.num_workers, args.balance, use_pin_memory)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin_memory)

    model = step6.build_model(model_name, img_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=step6.get_model_lr(model_name, args),
        weight_decay=args.weight_decay,
    )
    pos_weight = None
    if args.balance == "class_weight":
        n_pos = int((train_df["label"] == 1).sum())
        n_neg = int((train_df["label"] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(dl_tr, desc=f"{tag} | epoch {epoch}/{args.epochs}", leave=False)
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

        val_pred = predict_tiles(model, val_df, args, device, use_pin_memory)
        val_patients = patient_predictions(val_pred)
        val_auc = binary_auc(val_patients["y_true"].to_numpy(), val_patients["y_prob"].to_numpy())
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_patient_auc": float(val_auc),
        }
        history.append(record)
        log(f"{tag} | epoch={epoch} train_loss={record['train_loss']:.4f} val_patient_auc={val_auc:.4f}")
        if not np.isnan(val_auc) and val_auc > best_auc + args.min_delta:
            best_auc = float(val_auc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            log(f"{tag} | Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint.")
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    val_tiles = predict_tiles(model, val_df, args, device, use_pin_memory)
    test_tiles = predict_tiles(model, test_df, args, device, use_pin_memory)
    tile_threshold, val_tile_threshold_metrics = optimize_threshold(
        val_tiles["label"].to_numpy(),
        val_tiles["y_prob"].to_numpy(),
        metric=args.threshold_metric,
        step=args.threshold_step,
    )
    test_tile_metrics = calc_metrics(
        test_tiles["label"].to_numpy(),
        test_tiles["y_prob"].to_numpy(),
        threshold=tile_threshold,
    )

    tile_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{exp_name}_{model_name}_seed_{seed}_{mag}X_fold_{fold_id}")
    val_tiles.to_csv(run_dir / f"{tile_tag}_val_tile_predictions.csv", index=False)
    test_tiles.to_csv(run_dir / f"{tile_tag}_test_tile_predictions.csv", index=False)

    rows = []
    for agg_name in args.aggregations:
        val_pat = aggregate_patient_probs(val_tiles, agg_name, args.topk, args.attention_temperature)
        test_pat = aggregate_patient_probs(test_tiles, agg_name, args.topk, args.attention_temperature)
        patient_threshold, val_patient_threshold_metrics = optimize_threshold(
            val_pat["y_true"].to_numpy(),
            val_pat["y_prob"].to_numpy(),
            metric=args.threshold_metric,
            step=args.threshold_step,
        )
        test_patient_metrics = calc_metrics(
            test_pat["y_true"].to_numpy(),
            test_pat["y_prob"].to_numpy(),
            threshold=patient_threshold,
        )
        val_pat["y_pred"] = (val_pat["y_prob"] >= patient_threshold).astype(int)
        test_pat["y_pred"] = (test_pat["y_prob"] >= patient_threshold).astype(int)
        val_pat.to_csv(run_dir / f"{tile_tag}_{agg_name}_val_patient_predictions.csv", index=False)
        test_pat.to_csv(run_dir / f"{tile_tag}_{agg_name}_test_patient_predictions.csv", index=False)

        row = {
            "exp_name": exp_name,
            "model_name": model_name,
            "magnification": int(mag),
            "seed": int(seed),
            "fold": int(fold_id),
            "aggregation": agg_name,
            "best_epoch": int(best_epoch),
            "best_val_patient_auc": float(best_auc),
            "tile_threshold_from_val": float(tile_threshold),
            "patient_threshold_from_val": float(patient_threshold),
            "val_tile_threshold_metric": float(val_tile_threshold_metrics[args.threshold_metric]),
            "val_patient_threshold_metric": float(val_patient_threshold_metrics[args.threshold_metric]),
            "test_tile_auc": test_tile_metrics["auc"],
            "test_tile_f1": test_tile_metrics["f1"],
            "test_tile_accuracy": test_tile_metrics["accuracy"],
            "test_tile_sensitivity": test_tile_metrics["sensitivity"],
            "test_tile_specificity": test_tile_metrics["specificity"],
            "test_patient_auc": test_patient_metrics["auc"],
            "test_patient_f1": test_patient_metrics["f1"],
            "test_patient_accuracy": test_patient_metrics["accuracy"],
            "test_patient_sensitivity": test_patient_metrics["sensitivity"],
            "test_patient_specificity": test_patient_metrics["specificity"],
            "n_train_patients": int(train_df["patient_id"].nunique()),
            "n_val_patients": int(val_df["patient_id"].nunique()),
            "n_test_patients": int(test_df["patient_id"].nunique()),
            "n_train_tiles": int(len(train_df)),
            "n_val_tiles": int(len(val_df)),
            "n_test_tiles": int(len(test_df)),
        }
        rows.append(row)
        log(
            f"{tag} | agg={agg_name} | tile_thr={tile_threshold:.2f} "
            f"patient_thr={patient_threshold:.2f} | test_tile_auc={test_tile_metrics['auc']:.4f} "
            f"test_patient_auc={test_patient_metrics['auc']:.4f} test_patient_f1={test_patient_metrics['f1']:.4f}"
        )
    return rows


def build_group_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "test_tile_auc",
        "test_tile_f1",
        "test_tile_accuracy",
        "test_patient_auc",
        "test_patient_f1",
        "test_patient_accuracy",
        "patient_threshold_from_val",
    ]
    agg = {"n_runs": ("fold", "size")}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    out = df.groupby(group_cols, as_index=False).agg(**agg)
    for col in metric_cols:
        out[f"{col}_std"] = out[f"{col}_std"].fillna(0.0)
        out[f"{col}_mean_std"] = out.apply(
            lambda r: f"{r[f'{col}_mean']:.4f} +/- {r[f'{col}_std']:.4f}",
            axis=1,
        )
    return out.sort_values(by=["test_patient_auc_mean", "test_tile_auc_mean"], ascending=False)


def save_summary_plot(summary: pd.DataFrame, out_file: Path):
    if summary.empty:
        return
    labels = [
        f"{r.exp_name}\n{r.model_name} {int(r.magnification)}X\n{r.aggregation}"
        for r in summary.itertuples()
    ]
    plt.figure(figsize=(max(8, 0.8 * len(labels)), 5))
    plt.bar(labels, summary["test_patient_auc_mean"], color="#4C72B0")
    plt.errorbar(
        np.arange(len(summary)),
        summary["test_patient_auc_mean"],
        yerr=summary["test_patient_auc_std"],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    plt.ylim(0, 1)
    plt.ylabel("Patient-level AUC")
    plt.title("Patient-level CV test AUC by aggregation")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if args.cv_source != "train_only":
        raise ValueError("The independent test set must remain held out. Use --cv_source train_only.")
    step6.set_seed(min(args.seeds))
    raw_root = args.raw_root.resolve()
    results_root = args.results_root.resolve()
    output_root = args.output_dir.resolve()
    tables_dir = output_root / "Tables"
    plots_dir = output_root / "Plots"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_root / "log_patient_level_cv.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("PATIENT-LEVEL CROSS-VALIDATION FOR COLAB\n")
    log = make_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use Colab GPU; CPU training is disabled.")
    use_pin_memory = device.type == "cuda"
    metadata_xlsx = (args.metadata_xlsx or (raw_root / "Metadata.xlsx")).resolve()
    slide_to_patient = step6.build_slide_to_patient_map(metadata_xlsx, args.patient_id_columns) if metadata_xlsx.exists() else {}

    log(f"raw_root: {raw_root}")
    log(f"results_root: {results_root}")
    log(f"output_root: {output_root}")
    log(f"device: {device}")
    log(f"cv_source: {args.cv_source}")
    log(f"exp_name: {args.exp_name}")
    log(f"models: {args.models}")
    log(f"magnifications: {args.magnifications}")
    log(f"seeds: {args.seeds}")
    log(f"n_splits: {args.n_splits}")
    log(f"aggregations: {args.aggregations}")
    log(f"threshold_metric: {args.threshold_metric}")

    all_rows = []
    for exp_name in args.exp_name:
        exp_root = step6.resolve_exp_root(results_root, exp_name)
        csv_paths = [exp_root / "Tables" / "train_tiles_preprocessed.csv"]
        if args.cv_source == "train_plus_test":
            csv_paths.append(exp_root / "Tables" / "test_tiles_preprocessed.csv")
        frames = []
        for csv_path in csv_paths:
            if not csv_path.exists():
                log(f"[WARN] Missing table for exp={exp_name}: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            df = step6.prepare_training_table(df, exp_root, results_root, exp_name, slide_to_patient, log)
            df = verify_against_metadata(df, metadata_xlsx, args.patient_id_columns)
            frames.append(df)
        if not frames:
            log(f"[WARN] No data for exp={exp_name}. Skip.")
            continue
        full_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tile_path", "slide_id"])
        heldout_path = exp_root / "Tables" / "test_tiles_preprocessed.csv"
        if not heldout_path.exists():
            raise FileNotFoundError("Need independent test manifest for leakage audit.")
        heldout = verify_against_metadata(pd.read_csv(heldout_path), metadata_xlsx, args.patient_id_columns)
        heldout["label"] = (heldout["isup"].astype(int) > 0).astype(int)
        assert_disjoint(full_df, heldout)
        log(
            f"{exp_name}: cv tiles={len(full_df)}, slides={full_df['slide_id'].nunique()}, "
            f"patients={full_df['patient_id'].nunique()}, positive_tiles={int((full_df['label'] == 1).sum())}"
        )

        for model_name in args.models:
            for mag in args.magnifications:
                df_m = full_df[full_df["objective_lens"] == int(mag)].copy()
                if df_m.empty or df_m["label"].nunique() < 2:
                    log(f"[WARN] Invalid data for {exp_name}|{model_name}|{mag}X. Skip.")
                    continue
                for seed in args.seeds:
                    folds = stratified_patient_folds(df_m, args.n_splits, seed)
                    for fold_idx, test_patients in enumerate(folds, start=1):
                        test_df = df_m[df_m["patient_id"].isin(test_patients)].copy()
                        outer_train_df = df_m[~df_m["patient_id"].isin(test_patients)].copy()
                        if test_df["label"].nunique() < 2 or outer_train_df["label"].nunique() < 2:
                            raise ValueError(f"Fold {fold_idx} lacks both classes; reduce n_splits.")
                        train_df, val_df = split_inner_train_val(outer_train_df, args.val_ratio_from_train, seed + fold_idx * 1000)
                        rows = train_one_fold(
                            exp_name,
                            model_name,
                            mag,
                            seed,
                            fold_idx,
                            train_df,
                            val_df,
                            test_df,
                            output_root,
                            args,
                            device,
                            use_pin_memory,
                            log,
                        )
                        all_rows.extend(rows)

    if not all_rows:
        log("No CV result was produced.")
        return

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(tables_dir / "cv_all_runs.csv", index=False)
    summary = build_group_summary(result_df, ["exp_name", "model_name", "magnification", "aggregation"])
    summary.to_csv(tables_dir / "cv_summary_by_exp_model_mag_aggregation.csv", index=False)
    summary_model = build_group_summary(result_df, ["model_name", "magnification", "aggregation"])
    summary_model.to_csv(tables_dir / "cv_summary_by_model_mag_aggregation.csv", index=False)
    save_summary_plot(summary, plots_dir / "cv_patient_auc_by_aggregation.png")

    with open(output_root / "cv_training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)

    log(f"Saved: {tables_dir / 'cv_all_runs.csv'}")
    log(f"Saved: {tables_dir / 'cv_summary_by_exp_model_mag_aggregation.csv'}")
    log(f"Saved: {tables_dir / 'cv_summary_by_model_mag_aggregation.csv'}")
    log(f"Saved: {plots_dir / 'cv_patient_auc_by_aggregation.png'}")
    log("\nTop CV summaries:")
    log(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
