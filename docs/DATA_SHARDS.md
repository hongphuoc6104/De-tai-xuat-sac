# Restartable histology data shards

This workflow inventories microscope images, commits bounded raw shards, and processes each shard into QC records and lossless image patches. It does not train a model, extract encoder features, create MIL bags, or assign patch-level cancer targets. Those steps require a separately reviewed patient mapping and later feature/MIL adapters.

`Metadata.xlsx` (or an explicitly supplied metadata CSV) remains authoritative; source ZIPs provide image files only and never replace metadata. The catalog preserves the original `Glade` value as `raw_glade`, labels conclusions at case level, and marks grade semantics as unconfirmed. A `candidate_case` identity is not a verified patient identity. Do not use shard boundaries as train, validation, test, or patient boundaries. Any future evaluation split must be constructed globally from verified patient identities.

## Local smoke run

Use a tiny representative sample first. The smoke config selects at most three source images per lens, caps each image at four tiles, and keeps raw shards to 32 MiB. A smoke result is for exercising the pipeline and reviewing QC only.

```bash
python -m histology_data prepare \
  --metadata /path/to/Metadata.xlsx \
  --source /path/to/tiny_image_directory \
  --release /tmp/histology-smoke/raw \
  --config configs/data_shards_smoke.json

python -m histology_data verify --release /tmp/histology-smoke/raw

python -m histology_data process \
  --release /tmp/histology-smoke/raw \
  --work-root /tmp/histology-smoke/work \
  --output /tmp/histology-smoke/processed \
  --config configs/data_shards_smoke.json
```

`--source` accepts an image directory or a ZIP archive and can be repeated. Explicit command-line options override their matching values from the JSON config. Without a config, `prepare` defaults to 512 MiB maximum raw shard size, and `process` uses 256-pixel tiles with a 256-pixel stride, 0.2 minimum tissue fraction, all tiles, and no focus rejection threshold.

## Full preparation and Colab processing

Build the portable runtime bundle from a clean checkout and copy it, the metadata file, and the source directory or original ZIP archives to Google Drive:

```bash
python scripts/build_data_runtime.py --output dist/histology-data-runtime.zip
```

Open [Colab_Data_Shards.ipynb](../notebooks/Colab_Data_Shards.ipynb), set the Drive paths in its configuration cell, and run the notebook in one Colab session. The notebook mounts Drive only in Colab. It places shard staging and tile extraction under `/content` (the VM's temporary disk) and stores committed raw and processed outputs on Drive so they survive a session restart. Set `WORK_ROOT` to a local SSD path with enough free space for one raw shard plus its temporary patches. At the 512 MiB raw-shard default, budget for at least that source shard plus processing scratch. By default, each successfully processed part is removed from SSD before the next begins.

The Colab source adapter accepts a directory together with one or more ZIP archives, such as 4X images in a directory and 10X/40X images in separate ZIPs. The CLI equivalent repeats `--source` for each path. The catalog reports colliding image basenames across inputs. The first catalog/packing pass must be able to read every original source archive; packing streams selected members into new bounded shards and does not extract the entire source ZIP to disk. Keep the original source available until the raw release is complete. Completed shard manifests and checksums allow later runs to resume without rebuilding committed parts.

The Colab config uses all available metadata rows for the selected lenses, 512 MiB raw shards, and uncapped patch enumeration. It does not mark labels reviewed or training ready. For a long run, set `max_shards` in the config to a positive value to cap newly processed shards in one invocation; rerunning the same command verifies completed outputs and advances to the next unprocessed shards.

Typical CLI calls, if running outside the notebook, are:

```bash
python -m histology_data prepare \
  --metadata /content/drive/MyDrive/histology/Metadata.xlsx \
  --source /content/drive/MyDrive/histology/source.zip \
  --release /content/drive/MyDrive/histology/raw-release \
  --config configs/data_shards_colab.json

python -m histology_data verify \
  --release /content/drive/MyDrive/histology/raw-release

python -m histology_data process \
  --release /content/drive/MyDrive/histology/raw-release \
  --work-root /content/histology-work \
  --output /content/drive/MyDrive/histology/processed \
  --config configs/data_shards_colab.json
```

Add `--keep-work` to retain a verified staged source shard on SSD after a new processing run for inspection. The notebook's `KEEP_WORK` setting defaults to `False`; set it to `True` to enable the same behavior there. Reusing an existing output verifies its commit without reading the raw TAR or rebuilding a missing stage; with the default retention setting, any existing stage is cleaned. The default cleans each new staged shard before advancing. Retaining staged shards grows SSD use by roughly one raw shard per retained part; clear those staged directories when review is finished.

## Outputs and resume behavior

- `prepare` writes an authoritative catalog and committed raw TAR shards with per-image checksums. `--max-shards` limits newly committed shards in that invocation; repeat the same command to continue.
- `verify` checks the release descriptor, committed shard archives, and member contents.
- `process` visits committed raw shards in order, stages one shard under the SSD work root, and publishes per-shard QC, tile metadata, lossless patches, and a verified commit under `<output>/<shard_id>/<processing_id>/`. Existing commits for the same processing settings are verified before they count as reused. `--max-shards` counts newly processed results, so reruns pass already completed work and continue forward; changing processing settings creates a separate result.
- Tile coordinates use the upper-left origin and refer to original image pixels. Border tiles record valid dimensions when white padding is needed. `tile_id` includes the canonical catalog `image_id`, raw image checksum, coordinates, and versioned processing settings, so separate source instances remain distinct and IDs do not depend on workspace paths. `content_tile_id` excludes the source identity and identifies matching raw-file bytes and coordinates for duplicate review; it is not a perceptual near-duplicate detector.
- QC reports decode, geometry, image mode, objective lens, tissue, and focus observations for review. Empty/no-tissue fields remain explicit in QC output.

Partial raw releases can be verified and processed, but their committed subset does not represent complete cohort coverage. A `complete` release means the catalog's planned raw shards were committed; it does not certify clinical label correctness, patient identity, or readiness for model training.

## Real-data E2E proof

Run the standalone bounded smoke driver against the actual source folder, keeping all outputs separate from raw inputs:

```bash
python scripts/run_data_smoke.py \
  --data-root /path/to/data \
  --output /path/to/Results/data_smoke/multi3 \
  --lenses 4 10 40 --per-lens 3
```

The driver limits local work to 1–5 images per lens, commits at most one raw shard per invocation, resumes packaging and processing, verifies that a final rerun leaves committed outputs unchanged, and checks that staged raw TIFFs were cleaned. It writes `execution_actions.json` and `summary.json` with command exit codes, counts and timing. A second trial can use `--lenses 4 --per-lens 2`. Real source ZIPs are read only for selected members, never fully decoded by this driver.

The catalog records `training_blockers` and exact-SHA duplicate groups. Smoke, unknown labels, unverified patients, unreviewed labels, exact raw duplicates, or ZIP content whose per-image SHA audit is still pending never pass the catalog training gate. Full scientific readiness needs a separate cohort-wide SHA audit after import and reviewed identity/label/bag construction. This preparation package does not promote a CRC-only catalog to scientific training approval.
