# Data preparation contract v1

Package: `histology_data` (Python 3.10+, NumPy, Pillow, pandas/openpyxl; QC may use OpenCV).

`catalog.build_catalog(metadata: Path, sources: list[Path], lenses: list[int] | None = None, per_lens: int | None = None, identity_map: Path | None = None, labels_reviewed: bool = False) -> dict`

Catalog: `schema_version`, `catalog_id`, `mode` smoke/full, `sources` (source_id/path/kind), `images`, `coverage`, `training_ready`. Images contain `image_id`, `file_name`, `source_id`, `source_member`, `byte_size`, `source_signature`, `objective_lens`, `case_id`, `patient_id` optional, `slide_group_id`, `raw_glade`, `grade_semantics=unconfirmed`, `case_label` optional, `label_level=case`, `label_source=Ket_Luan`, `identity_status`. Paths in source definitions are runtime mappings, excluded from catalog fingerprint. Keep labels at case level; no automatic ISUP conversion or patch labels.

Storage API to implement in `histology_data/shards.py`:

- `pack_catalog(catalog: dict, output: Path, max_bytes: int, max_shards: int | None = None) -> dict`. Raw TAR files and individual committed manifest files. Release descriptor `release.json`: `catalog_id`, `catalog` (catalog including source mapping), `complete`, `shards` list. Each committed shard manifest: `shard_id`, `catalog_id`, `tar_name`, `tar_sha256`, `images` list (all catalog fields plus `member=images/<image_id><suffix>`, `sha256`). Preserve size limit including TAR overhead; reject a single member that cannot fit. Source stream ZIP CRC must be consumed; verify immutable commits on resume. Source reads and file hashing use bounded buffers.
- `verify_release(release_root: Path) -> dict`: verifies descriptor and all referenced committed shard archives and member contents; returns descriptor.
- `stage_shard(release_root: Path, shard_id: str, work_root: Path) -> tuple[Path, dict]`: verifies checksum, regular whitelist members, free disk, extracts safely into a shard-specific working directory and returns stage root plus manifest. Never trust TAR traversal/symlinks/duplicate/unexpected members. Safe marker only after verified extraction. Partial releases can stage their committed shards; cannot be reported complete.

Processing API to implement in `histology_data/processing.py`:

- `process_shard(release_root: Path, shard_id: str, work_root: Path, output_root: Path, config: dict | None = None) -> dict`.
- Config defaults: `tile_size=256`, `stride=256`, `min_tissue=0.2`, `max_tiles_per_image=0` (all), `blur_threshold=0` (score/review only). Capped extraction is smoke and never becomes training-ready.
- Output per shard: lossless `patches.tar`, `tiles.jsonl`, `qc.jsonl`, optional preview images, final `commit.json` with catalog/config IDs, file checksums, counts, complete status. Verified on resume, corrupted output fails without silent success. Use SSD work-root then verified publication output-root (may be Drive).
- Coordinate origin: upper-left, x/y/w/h in original image pixels; include valid_w/valid_h for white padding at borders. `tile_id` hashes raw image SHA + coordinate + processing config. Same inputs produce same IDs across shards/profile paths. Tile rows reference case/bag provenance but must not propagate case cancer label into supervised patch target.
- Fully decode one field at a time; bounded patch buffer, no entire-cohort pixel load. Per-image QC records decode/geometry/mode/lens/tissue/focus and reason. Empty/no-tissue fields explicitly logged for review. Fail closed for corrupt source/output; no empty success commits.

CLI to implement in `histology_data/__main__.py`:

- `prepare --metadata PATH --source PATH [--source PATH ...] --release PATH --lenses 4 10 40 [--per-lens N] [--identity-map CSV] [--labels-reviewed] --max-mib 512 [--max-shards N]` -> build_catalog then pack_catalog. Explicit local small tests use `--per-lens`; large production runs on Colab.
- `process --release PATH --work-root PATH --output PATH [--shard SHARD_ID] [--max-shards N] --tile-size 256 --stride 256 --min-tissue 0.2 [--max-tiles-per-image N] [--blur-threshold VALUE]` -> process committed shards sequentially. Do not equate shards with train/val/test splits or bags; those remain global identity contracts.
- `verify --release PATH` -> verify_release.

Notebook is a thin entry point calling this API, no quota rotation/VM creation and no fake model/encoder placeholder success. It prepares raw/processed shards for subsequent feature/MIL modules. Runtime source ZIP import may initially read a large original, but new committed processing units are independent bounded shards and resume never redoes all completed shards.
