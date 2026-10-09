import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = DEFAULT_RAW_ROOT / "Results"

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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
        description="Review metadata and generate Step 1 overview outputs."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing Metadata.xlsx/train.csv and train_images.",
    )
    parser.add_argument(
        "--results_root",
        "--subset_root",
        dest="results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for Step 1 outputs.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata file. Defaults to Metadata.xlsx, then train.csv under raw_root.",
    )
    return parser.parse_args()


def cell_to_col_index(cell_ref):
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_xlsx_stdlib(path):
    """Read the first worksheet in a simple .xlsx file without openpyxl."""
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    with zipfile.ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                text = "".join(t.text or "" for t in item.findall(".//main:t", ns))
                shared_strings.append(text)

        wb_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        first_sheet = wb_root.find(".//main:sheet", ns)
        if first_sheet is None:
            raise ValueError(f"No sheets found in {path}")

        rel_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        sheet_path = "xl/" + rel_targets[rel_id].lstrip("/")
        sheet_root = ET.fromstring(workbook.read(sheet_path))

        rows = []
        for row in sheet_root.findall(".//main:sheetData/main:row", ns):
            values = []
            for cell in row.findall("main:c", ns):
                col_index = cell_to_col_index(cell.attrib.get("r", "A1"))
                while len(values) <= col_index:
                    values.append("")

                value_node = cell.find("main:v", ns)
                inline_node = cell.find("main:is/main:t", ns)
                if inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is None:
                    value = ""
                else:
                    value = value_node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = shared_strings[int(value)]
                    elif cell.attrib.get("t") == "b":
                        value = "TRUE" if value == "1" else "FALSE"
                values[col_index] = value
            rows.append(values)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = [str(col).strip() for col in rows[0]]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def read_metadata(path):
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            return read_xlsx_stdlib(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def first_existing_column(df, candidates):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    return None


def clean_text(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def normalize_grade(value):
    text = clean_text(value)
    if not text:
        return np.nan
    match = re.search(r"\d+", text)
    if match is None:
        return np.nan
    return int(match.group(0))


def normalize_lens(value):
    text = clean_text(value).upper().replace(" ", "")
    match = re.search(r"\d+", text)
    if match is None:
        return ""
    return f"{int(match.group(0))}X"


def normalize_metadata(raw_df):
    image_col = first_existing_column(raw_df, ["image_id", "Ten_File", "filename", "file_name"])
    provider_col = first_existing_column(raw_df, ["data_provider", "Data_Provider", "provider"])
    grade_col = first_existing_column(raw_df, ["isup_grade", "Grade", "grade", "Glade", "isup"])
    lens_col = first_existing_column(raw_df, ["objective_lens", "Do_Phong_Dai", "vat_kinh", "vật kính"])

    required = {
        "image_id/Ten_File": image_col,
        "isup_grade/Grade": grade_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise RuntimeError(f"Missing required metadata columns: {missing}")

    df = pd.DataFrame()
    image_names = raw_df[image_col].map(clean_text)
    df["file_name"] = image_names
    df["image_id"] = image_names.map(lambda name: Path(name).stem if name else "")
    df["data_provider"] = (
        raw_df[provider_col].map(clean_text) if provider_col is not None else "Unknown"
    )
    df["isup_grade"] = raw_df[grade_col].map(normalize_grade)
    df["objective_lens"] = (
        raw_df[lens_col].map(normalize_lens) if lens_col is not None else ""
    )

    keep_cols = [
        col for col in ["Id", "Ma_So", "Ma_Nam", "Ket_Luan", "Ten_Slide"] if col in raw_df.columns
    ]
    for col in keep_cols:
        df[col] = raw_df[col].map(clean_text)

    df = df[df["image_id"] != ""].copy()
    df["isup_grade"] = df["isup_grade"].astype("Int64")
    return df


def image_inventory(image_dir):
    if not image_dir.exists():
        return {}
    return {
        path.stem: path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def save_bar(series, path, xlabel, ylabel, title, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    series.plot(kind="bar", colormap="Dark2")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = raw_root / "Metadata.xlsx"
        if not metadata_path.exists():
            metadata_path = raw_root / "train.csv"
    metadata_path = metadata_path.resolve()

    image_dir = raw_root / "train_images"
    mask_dir = raw_root / "train_label_masks"
    plots_dir = args.results_root.resolve() / "Step1_Overview_results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_path = plots_dir / "log_step1.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeLogger(original_stdout, log_file)
        sys.stderr = TeeLogger(original_stderr, log_file)
        try:
            run_step1(raw_root, metadata_path, image_dir, mask_dir, plots_dir, log_path)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_step1(raw_root, metadata_path, image_dir, mask_dir, plots_dir, log_path):
    print("Log file:", log_path)

    print("Raw root:", raw_root)
    print("Metadata:", metadata_path)
    print("Image directory:", image_dir)
    print("Output directory:", plots_dir)

    raw_df = read_metadata(metadata_path)
    df = normalize_metadata(raw_df)
    total_wsi = len(df)
    if total_wsi == 0:
        raise RuntimeError("No valid metadata rows found after normalization.")

    print("\nTotal WSI:", total_wsi)

    isup_counts = df["isup_grade"].dropna().value_counts().sort_index()
    print("\nISUP distribution:")
    print(isup_counts)

    provider_counts = df["data_provider"].replace("", "Unknown").value_counts()
    print("\nData provider distribution:")
    print(provider_counts)

    lens_counts = df["objective_lens"].replace("", "Unknown").value_counts().sort_index()
    print("\nObjective lens distribution:")
    print(lens_counts)

    images = image_inventory(image_dir)
    df["image_found"] = df["image_id"].isin(images)
    df["actual_file_name"] = df["image_id"].map(images).fillna("")
    slides_with_image = int(df["image_found"].sum())
    slides_without_image = total_wsi - slides_with_image
    print("\nSlides with image files:", slides_with_image)
    print("Slides without image files:", slides_without_image)

    if mask_dir.exists():
        mask_ids = {p.stem for p in mask_dir.glob("*.tif*")}
        slides_with_mask = int(df["image_id"].isin(mask_ids).sum())
    else:
        slides_with_mask = 0
    slides_without_mask = total_wsi - slides_with_mask
    print("\nSlides with masks:", slides_with_mask)
    print("Slides without masks:", slides_without_mask)

    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / total_wsi * 100).round(2)
    print("\nMissing metadata (% per column if any):")
    print(missing_pct)

    missing_grade_count = int(df["isup_grade"].isna().sum())
    problem_rows = df[df["isup_grade"].isna() | ~df["image_found"]].copy()
    problem_rows_path = plots_dir / "Step1_problem_rows.csv"
    problem_rows.to_csv(problem_rows_path, index=False)
    print("\nRows with missing ISUP grade:", missing_grade_count)
    print("Problem rows saved to:", problem_rows_path)

    max_class = int(isup_counts.max()) if not isup_counts.empty else 0
    min_class = int(isup_counts.min()) if not isup_counts.empty else 0
    imbalance_ratio = round(max_class / min_class, 2) if min_class else np.nan

    if len(isup_counts) > 1:
        isup_probs = isup_counts / isup_counts.sum()
        isup_entropy = float(-np.sum(isup_probs * np.log2(isup_probs)))
        isup_normalized_entropy = float(isup_entropy / np.log2(len(isup_counts)))
    else:
        isup_entropy = 0.0
        isup_normalized_entropy = 0.0

    print("\nISUP imbalance ratio (max/min):", imbalance_ratio)
    print("ISUP entropy:", round(isup_entropy, 3))
    print("ISUP normalized entropy:", round(isup_normalized_entropy, 3))

    save_bar(
        isup_counts,
        plots_dir / "isup_distribution.png",
        "ISUP Grade",
        "Number of slides",
        "ISUP Grade Distribution",
    )

    isup_provider = (
        df.assign(data_provider=df["data_provider"].replace("", "Unknown"))
        .groupby(["isup_grade", "data_provider"])
        .size()
        .unstack(fill_value=0)
    )
    isup_provider.plot(kind="bar", stacked=True, figsize=(7, 5), colormap="Dark2")
    plt.xlabel("ISUP Grade")
    plt.ylabel("Number of slides")
    plt.title("ISUP x Data Provider Distribution")
    plt.legend(title="Provider")
    plt.tight_layout()
    plt.savefig(plots_dir / "isup_provider_stacked.png", dpi=300)
    plt.close()

    save_bar(
        lens_counts,
        plots_dir / "objective_lens_distribution.png",
        "Objective lens",
        "Number of slides",
        "Objective Lens Distribution",
    )

    isup_lens = (
        df.assign(objective_lens=df["objective_lens"].replace("", "Unknown"))
        .groupby(["isup_grade", "objective_lens"])
        .size()
        .unstack(fill_value=0)
    )
    isup_lens.plot(kind="bar", stacked=True, figsize=(7, 5), colormap="Dark2")
    plt.xlabel("ISUP Grade")
    plt.ylabel("Number of slides")
    plt.title("ISUP x Objective Lens Distribution")
    plt.legend(title="Objective lens")
    plt.tight_layout()
    plt.savefig(plots_dir / "isup_objective_lens_stacked.png", dpi=300)
    plt.close()

    summary_rows = [
        ["Total WSI", float(total_wsi)],
        ["Slides with image files", float(slides_with_image)],
        ["Slides without image files", float(slides_without_image)],
        ["Slides with masks", float(slides_with_mask)],
        ["Slides without masks", float(slides_without_mask)],
        ["Rows with missing ISUP grade", float(missing_grade_count)],
        ["ISUP max class count", float(max_class)],
        ["ISUP min class count", float(min_class)],
        ["ISUP imbalance ratio (max/min)", imbalance_ratio],
        ["ISUP entropy", round(isup_entropy, 3)],
        ["ISUP normalized entropy", round(isup_normalized_entropy, 3)],
    ]

    for grade, count in isup_counts.items():
        summary_rows.append([f"ISUP {grade} slides", float(count)])
        summary_rows.append([f"ISUP {grade} (%)", round(count / total_wsi * 100, 2)])

    for lens, count in lens_counts.items():
        summary_rows.append([f"Lens {lens} slides", float(count)])
        summary_rows.append([f"Lens {lens} (%)", round(count / total_wsi * 100, 2)])

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    summary_path = plots_dir / "Step1_summary_table.csv"
    normalized_path = plots_dir / "Step1_metadata_normalized.csv"
    train_csv_path = raw_root / "train.csv"

    summary_df.to_csv(summary_path, index=False)
    df.to_csv(normalized_path, index=False)
    df[["image_id", "file_name", "data_provider", "isup_grade", "objective_lens"]].to_csv(
        train_csv_path, index=False
    )

    print("\nSummary table saved to:", summary_path)
    print("Normalized metadata saved to:", normalized_path)
    print("train.csv saved to:", train_csv_path)
    print(summary_df)

    print("\n===== STEP 1 FINISHED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
