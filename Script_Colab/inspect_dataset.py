from pathlib import Path

import pandas as pd
from PIL import Image


def analyze_dataset():
    metadata_path = Path("/data/newProject_HE_classification/Script_Colab/Metadata.xlsx")
    image_dir = Path("/data/newProject_HE_classification/data/train_images")

    print("=== READING METADATA ===")
    df = pd.read_excel(metadata_path)
    print(f"Total rows in Metadata: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    meta_files = set(df['Ten_File'].dropna().astype(str).str.strip())
    print(f"Unique filenames in Metadata: {len(meta_files)}")

    # Check duplicate filenames in metadata
    dup_meta = df[df.duplicated(subset=['Ten_File'], keep=False)]
    if not dup_meta.empty:
        print(f"WARNING: Found {len(dup_meta)} rows with duplicate Ten_File in Metadata:")
        print(dup_meta[['Id', 'Ten_File', 'Ma_So', 'Ma_Nam', 'Glade']].head(10))
    else:
        print("✓ No duplicate Ten_File found in Metadata.")

    print("\n=== SCANNING IMAGE DIRECTORY ===")
    if not image_dir.exists():
        print(f"Directory {image_dir} does not exist yet!")
        return

    disk_files = {p.name: p for p in image_dir.iterdir() if p.is_file()}
    print(f"Total files in {image_dir}: {len(disk_files)}")

    # Compare sets
    in_meta_not_disk = meta_files - set(disk_files.keys())
    in_disk_not_meta = set(disk_files.keys()) - meta_files
    matched = meta_files.intersection(set(disk_files.keys()))

    print("\n=== MATCHING RESULTS ===")
    print(f"Exact match (in both Metadata and disk): {len(matched)} / {len(meta_files)} ({len(matched)/len(meta_files)*100:.2f}%)")
    print(f"Files in Metadata but NOT on disk: {len(in_meta_not_disk)}")
    if in_meta_not_disk:
        print("Sample missing files:", list(in_meta_not_disk)[:10])
    print(f"Files on disk but NOT in Metadata: {len(in_disk_not_meta)}")
    if in_disk_not_meta:
        print("Sample unmapped files on disk:", list(in_disk_not_meta)[:10])

    print("\n=== INSPECTING IMAGE SIZES & CHARACTERISTICS (RAW VS TILES) ===")
    sample_files = list(disk_files.values())[:30]
    sizes = set()
    modes = set()

    for p in sample_files:
        try:
            with Image.open(p) as img:
                sizes.add(img.size) # (width, height)
                modes.add(img.mode)
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

    print("Sampled 30 images:")
    print(f"Distinct Dimensions (Width x Height): {sizes}")
    print(f"Color Modes: {modes}")

    # Check all images size distribution
    all_sizes = {}
    for p in list(disk_files.values())[:500]:
        try:
            with Image.open(p) as img:
                all_sizes[img.size] = all_sizes.get(img.size, 0) + 1
        except Exception:
            pass
    print("Size distribution (first 500 images):", all_sizes)

if __name__ == "__main__":
    analyze_dataset()
