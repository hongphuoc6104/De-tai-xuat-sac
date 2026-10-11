# Dual-track private report

`scripts/render_dual_report.py` is a postprocessor for committed A0/A1/B0 artifacts. It does not enter the model runtime and does not read whole-slide images or feature bundles. It validates assignment row order and source-photo lineage before rendering any point-level plot.

## Inputs

Call `render_dual_report(a_store, b_store, shared_store, source_photo_map, output_dir, private_map, ...)` or use the CLI. Each store is an existing `ArtifactStore` opened from its root and `identity.json`. The supplied source map must equal the committed `source_photo_map.pkl` in the shared store.

The renderer reads these committed artifacts:

| Store | Artifact path | Use |
|---|---|---|
| Shared | `source_photo_map.pkl` | Cohort and verified tile/photo/feature-part lineage |
| Shared | `scopes/outer_0/inner_0/transform.pkl` | Five-patient train-only PCA32 sample and ordered row references for A0 plots |
| A | `atlas/outer_0_inner_0/hdbscan/lens_<lens>/mcs<size>_ms<samples>_<eom-or-leaf>/{metrics.json,assignments.csv.gz}` | EOM/leaf coverage, noise, cluster count, and selected scatter partitions |
| A | `atlas/outer_0_inner_0/ward/lens_<lens>/k<k>/{metrics.json,assignments.csv.gz}` | Ward cuts and selected Ward scatter |
| A | `atlas/outer_0_inner_0/kmeans/lens_<lens>/k<k>/{metrics.json,assignments.csv.gz}` | KMeans controls and selected KMeans scatters |
| A | `results/a1_oof.json`, `results/counts_only_oof.json` | A1 and its separate count-only OOF control |
| B | `results/b0_oof.json`, `results/counts_only_oof.json` | B0 and its separate count-only OOF control |
| Optional B1 | Paths supplied by `b1_artifacts` | Candidate OOF lists keyed only by `beta=0.0` and `beta=0.5` |

The A0 scope is deliberately fixed to `outer_0/inner_0`, with exactly five distinct patients. A0 cluster IDs are local to each lens/configuration panel; the report does not join or compare cluster IDs across folds. Source case labels appear only in descriptive row-count/concentration plots and are not patch truth.

OOF files are lists of records containing `case_id`, `patient_id`, integer `target`, and `probability` in `[0,1]`. The renderer checks uniqueness and cohort membership, aligns each count-only control to its primary model, and labels A/B results `matched_full_cohort` only when all four files cover the same cases. B1 is complete only when both named candidates cover that same full cohort.

## Outputs and privacy

`output_dir` contains PNG/SVG plots, a self-contained `report.html`, portable `report.md`, `report_manifest.json`, and tokenized `point_lineage.csv` / `oof_lineage.csv`. The HTML embeds its PNGs and has no remote dependencies. Point rows link back to opaque case, patient, photo, and point tokens; the report does not contain source IDs.

The reverse token map is written to the supplied `private_map`, outside `output_dir`, with owner-only permissions. A second owner-only file is written beside it as `<private-map-stem>.representatives.csv` unless `representatives_private_path` is explicitly supplied. This file contains at most 96 source retrieval candidates (at most five per plotted configuration), with the columns:

`tile_id,lens,source_case_label,algorithm,cluster_id,part_id,row_index,prototype_role`

The selection includes HDBSCAN noise when present, then favors unseen clusters and distinct source cases/photos. Algorithm names are configuration-specific (`hdbscan_eom_mcs16_ms4`, `hdbscan_leaf_mcs16_ms4`, `ward_k4`, and `kmeans_k4/k8/k16`). These are retrieval candidates for later image review, not biological prototypes. Keep this raw-reference CSV with the private mapping; do not publish it with the tokenized report.

## Verification limits

Synthetic fixtures verify receipt checks, A/B alignment, B1 completion rules, tokenization, private file permissions, report rendering, and the 96-row representative cap. The report summarizes committed results; it does not establish morphology, patch-level truth, clinical utility, or full-cohort image interpretation.
