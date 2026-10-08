import argparse
import os
import random
import re
import sys
import zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_integrity import (metadata_table, identity_map, split_patients as strict_split,
                            assert_disjoint, validate_frame, strict_grade, strict_lens,
                            lock_config, fingerprint, patient_predictions, file_hash, verify_against_metadata)

import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"

EXCEL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")


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
        description="Step 4: patient-level train/test split for real microscopy data."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing Metadata.xlsx and train.csv.",
    )
    parser.add_argument(
        "--results_root",
        "--subset_root",
        dest="results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing Step 3 outputs and Step 4 outputs.",
    )
    parser.add_argument(
        "--metadata_xlsx",
        type=Path,
        default=None,
        help="Metadata Excel file. Defaults to raw_root/Metadata.xlsx.",
    )
    parser.add_argument(
        "--slides_csv",
        type=Path,
        default=None,
        help="Slide table to split. Defaults to Step3_Tissue_detection_tiles/slides_used_for_tissue_detection.csv.",
    )
    parser.add_argument(
        "--tiles_csv",
        type=Path,
        default=None,
        help="Tile metadata CSV. Defaults to Step3_Tissue_detection_tiles/tiles_metadata.csv.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=500)
    parser.add_argument("--patient_id_columns", nargs="+", default=["Ma_Nam", "Ma_So"],
                        help="Confirmed patient identity columns; no automatic year/case inference.")
    return parser.parse_args()


def cell_to_col_index(cell_ref):
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_first_sheet_df(xlsx_path):
    with zipfile.ZipFile(xlsx_path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{EXCEL_NS}si"):
                shared_strings.append("".join(t.text or "" for t in item.findall(f".//{EXCEL_NS}t")))

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}
        first_sheet = workbook_root.find(f".//{EXCEL_NS}sheet")
        if first_sheet is None:
            raise RuntimeError(f"No sheets found in {xlsx_path}")

        rel_id = first_sheet.attrib[f"{REL_NS}id"]
        sheet_path = "xl/" + rel_targets[rel_id].lstrip("/")
        sheet_root = ET.fromstring(workbook.read(sheet_path))

        rows = []
        for row in sheet_root.findall(f".//{EXCEL_NS}sheetData/{EXCEL_NS}row"):
            values = []
            for cell in row.findall(f"{EXCEL_NS}c"):
                col_idx = cell_to_col_index(cell.attrib.get("r", "A1"))
                while len(values) <= col_idx:
                    values.append("")

                value_node = cell.find(f"{EXCEL_NS}v")
                inline_node = cell.find(f"{EXCEL_NS}is/{EXCEL_NS}t")
                if inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is None:
                    value = ""
                else:
                    value = value_node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = shared_strings[int(value)]
                values[col_idx] = value
            rows.append(values)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = [clean_text(col) for col in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def clean_text(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def find_col(df, candidates):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    return None


def normalize_image_id(value):
    text = clean_text(value)
    if not text:
        return ""
    return Path(text).stem


def build_slide_to_patient_map(metadata_xlsx, patient_columns=None):
    meta = metadata_table(metadata_xlsx, patient_columns)
    mapping = meta.set_index('slide_id').patient_id.to_dict()
    return mapping, meta.rename(columns={'slide_id':'image_id'})


def normalize_lens(value):
    text = clean_text(value).upper().replace(" ", "")
    if text.endswith(".0"):
        text = text[:-2]
    if text and text.isdigit():
        return f"{text}X"
    return text


def load_slide_table(slides_csv, slide_to_patient):
    df = pd.read_csv(slides_csv)
    required = ["image_id", "isup_grade", "objective_lens"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in slide table: {missing}")

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["patient_id"] = df["image_id"].map(slide_to_patient)
    df["isup_grade"] = df["isup_grade"].map(strict_grade)
    df["label"] = (df["isup_grade"] > 0).astype(int)
    df["objective_lens"] = df["objective_lens"].map(normalize_lens)
    if "data_provider" not in df.columns:
        df["data_provider"] = "unknown"
    return df


def split_patients(slide_df, test_ratio, seed, max_retries):
    train, test = strict_split(slide_df, test_ratio, seed)
    return train, test, 1


def add_tile_split(tiles_csv, train_df, test_df, slide_to_patient, tables_dir):
    if not tiles_csv.exists():
        return None, None

    tiles_df = pd.read_csv(tiles_csv)
    if "slide_id" not in tiles_df.columns:
        raise RuntimeError("tiles_metadata.csv must contain slide_id.")

    tiles_df["patient_id"] = tiles_df["slide_id"].map(slide_to_patient)
    if tiles_df["patient_id"].isna().any():
        raise ValueError("Tiles missing patient mapping.")

    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    train_tiles = tiles_df[tiles_df["patient_id"].isin(train_patients)].copy()
    test_tiles = tiles_df[tiles_df["patient_id"].isin(test_patients)].copy()

    train_tiles.to_csv(tables_dir / "train_tiles.csv", index=False)
    test_tiles.to_csv(tables_dir / "test_tiles.csv", index=False)
    return train_tiles, test_tiles


def save_summary_tables(train_df, test_df, train_tiles, test_tiles, patient_map_df, tables_dir):
    train_df.to_csv(tables_dir / "train_slides.csv", index=False)
    test_df.to_csv(tables_dir / "test_slides.csv", index=False)
    patient_map_df.to_csv(tables_dir / "slide_patient_map.csv", index=False)

    all_slides = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
    patient_split = (
        all_slides.groupby("patient_id", as_index=False)
        .agg(
            split=("split", "first"),
            patient_label=("label", "max"),
            max_isup=("isup_grade", "max"),
            n_slides=("image_id", "count"),
        )
        .sort_values("patient_id")
    )
    if train_tiles is not None and test_tiles is not None:
        all_tiles = pd.concat([train_tiles, test_tiles], ignore_index=True)
        tile_counts = all_tiles.groupby("patient_id").size().rename("n_tiles").reset_index()
        patient_split = patient_split.merge(tile_counts, on="patient_id", how="left")
        patient_split["n_tiles"] = patient_split["n_tiles"].fillna(0).astype(int)
    else:
        patient_split["n_tiles"] = 0
    train_patients = set(train_df["patient_id"])
    patient_split["split"] = patient_split["patient_id"].map(lambda x: "train" if x in train_patients else "test")
    patient_split.to_csv(tables_dir / "patient_split.csv", index=False)

    rows = []
    for split, df, tiles in [
        ("train", train_df, train_tiles),
        ("test", test_df, test_tiles),
    ]:
        rows.append(
            {
                "split": split,
                "n_slides": len(df),
                "n_patients": df["patient_id"].nunique(),
                "n_isup0": int((df["isup_grade"] == 0).sum()),
                "n_isup3": int((df["isup_grade"] == 3).sum()),
                "n_isup4": int((df["isup_grade"] == 4).sum()),
                "n_isup5": int((df["isup_grade"] == 5).sum()),
                "n_isup_pos": int((df["isup_grade"] > 0).sum()),
                "n_tiles": len(tiles) if tiles is not None else 0,
            }
        )
    split_summary = pd.DataFrame(rows)
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)
    return split_summary, patient_split


def plot_bar_counts(series_by_split, title, ylabel, path):
    labels = sorted(set().union(*[set(s.index.astype(str)) for s in series_by_split.values()]))
    x = list(range(len(labels)))
    width = 0.38
    cmap = plt.get_cmap("Dark2")
    plt.figure(figsize=(8, 4.5))
    for idx, (split, series) in enumerate(series_by_split.items()):
        offset = -width / 2 if idx == 0 else width / 2
        plt.bar(
            [i + offset for i in x],
            [int(series.get(label, 0)) for label in labels],
            width=width,
            color=cmap.colors[idx],
            label=split,
        )
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_plots(train_df, test_df, train_tiles, test_tiles, plots_dir):
    cmap = plt.get_cmap("Dark2")

    plt.figure(figsize=(7, 4))
    x = [0, 1]
    width = 0.38
    plt.bar([i - width / 2 for i in x], [len(train_df), len(test_df)], width=width, color=cmap.colors[0], label="Slides")
    plt.bar(
        [i + width / 2 for i in x],
        [train_df["patient_id"].nunique(), test_df["patient_id"].nunique()],
        width=width,
        color=cmap.colors[1],
        label="Patients",
    )
    plt.xticks(x, ["train", "test"])
    plt.ylabel("Count")
    plt.title("Patient-level Split Overview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "split_overview_slides_patients.png", dpi=220)
    plt.close()

    plot_bar_counts(
        {
            "train": train_df["isup_grade"].astype(str).value_counts(),
            "test": test_df["isup_grade"].astype(str).value_counts(),
        },
        "Slide ISUP Distribution by Split",
        "Slides",
        plots_dir / "slide_isup_distribution_by_split.png",
    )

    plot_bar_counts(
        {
            "train": train_df["objective_lens"].astype(str).value_counts(),
            "test": test_df["objective_lens"].astype(str).value_counts(),
        },
        "Slide Objective Lens Distribution by Split",
        "Slides",
        plots_dir / "slide_objective_distribution_by_split.png",
    )

    patient_train = train_df.groupby("patient_id")["label"].max().astype(str).value_counts()
    patient_test = test_df.groupby("patient_id")["label"].max().astype(str).value_counts()
    plot_bar_counts(
        {"train": patient_train, "test": patient_test},
        "Patient Label Distribution by Split",
        "Patients",
        plots_dir / "patient_label_distribution_by_split.png",
    )

    if train_tiles is not None and test_tiles is not None:
        plot_bar_counts(
            {
                "train": train_tiles["objective_label"].astype(str).value_counts(),
                "test": test_tiles["objective_label"].astype(str).value_counts(),
            },
            "Tile Distribution by Magnification",
            "Tiles",
            plots_dir / "tile_distribution_by_magnification.png",
        )
        plot_bar_counts(
            {
                "train": train_tiles["isup"].astype(str).value_counts(),
                "test": test_tiles["isup"].astype(str).value_counts(),
            },
            "Tile ISUP Distribution by Split",
            "Tiles",
            plots_dir / "tile_isup_distribution_by_split.png",
        )


def run_step4(args, raw_root, results_root, metadata_xlsx, slides_csv, tiles_csv, step4_root):
    tables_dir = step4_root / "Tables"
    plots_dir = step4_root / "Plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Raw root:", raw_root)
    print("Metadata:", metadata_xlsx)
    print("Slides CSV:", slides_csv)
    print("Tiles CSV:", tiles_csv)
    print("Output directory:", step4_root)
    print("Test ratio:", args.test_ratio)
    print("Seed:", args.seed)

    slide_to_patient, patient_map_df = build_slide_to_patient_map(metadata_xlsx, args.patient_id_columns)
    slide_df = load_slide_table(slides_csv, slide_to_patient)
    slide_df = verify_against_metadata(slide_df, metadata_xlsx, args.patient_id_columns)
    missing = int(slide_df["patient_id"].isna().sum())
    if missing:
        print(f"[WARN] Missing patient_id for {missing} slides. Dropping them.")
        raise ValueError("Slides missing patient mapping; refusing to silently drop them.")

    print("\nSlides available for split:", len(slide_df))
    print("Unique patients:", slide_df["patient_id"].nunique())
    print("\nSlides by ISUP:")
    print(slide_df.groupby("isup_grade").size())
    print("\nSlides by objective lens:")
    print(slide_df.groupby("objective_lens").size())

    train_df, test_df, retries_used = split_patients(slide_df, args.test_ratio, args.seed, args.max_retries)
    print("\nRetries used:", retries_used)
    print("Train slides:", len(train_df), "patients:", train_df["patient_id"].nunique())
    print("Test slides:", len(test_df), "patients:", test_df["patient_id"].nunique())

    overlap = set(train_df["patient_id"]) & set(test_df["patient_id"])
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {len(overlap)} overlapping patients")
    print("Patient leakage check: PASS")

    train_tiles, test_tiles = add_tile_split(tiles_csv, train_df, test_df, slide_to_patient, tables_dir)
    if train_tiles is not None:
        print("Train tiles:", len(train_tiles))
        print("Test tiles:", len(test_tiles))

    split_summary, patient_split = save_summary_tables(
        train_df, test_df, train_tiles, test_tiles, patient_map_df, tables_dir
    )
    save_plots(train_df, test_df, train_tiles, test_tiles, plots_dir)

    print("\nSplit summary:")
    print(split_summary)
    print("\nSaved tables to:", tables_dir)
    print("Saved plots to:", plots_dir)
    print("\n===== STEP 4 FINISHED SUCCESSFULLY =====")


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    results_root = args.results_root.resolve()
    metadata_xlsx = (args.metadata_xlsx or (raw_root / "Metadata.xlsx")).resolve()
    slides_csv = (
        args.slides_csv
        or (results_root / "Step3_Tissue_detection_tiles" / "slides_used_for_tissue_detection.csv")
    ).resolve()
    tiles_csv = (
        args.tiles_csv
        or (results_root / "Step3_Tissue_detection_tiles" / "tiles_metadata.csv")
    ).resolve()
    step4_root = results_root / "Step4_Patient_level_Split"
    step4_root.mkdir(parents=True, exist_ok=True)
    log_path = step4_root / "log_step4.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeLogger(original_stdout, log_file)
        sys.stderr = TeeLogger(original_stderr, log_file)
        try:
            print("Log file:", log_path)
            run_step4(args, raw_root, results_root, metadata_xlsx, slides_csv, tiles_csv, step4_root)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
