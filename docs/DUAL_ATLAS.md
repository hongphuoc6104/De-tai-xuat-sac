# Dual-atlas primitives

`histology_data.dual_atlas` provides the train-scoped, in-memory primitives for
the exploratory atlas (A0) and the KMeans histogram classifier (A1). It accepts
already transformed `float32` vectors and explicit patient/photo IDs. It does
not read bundles, infer lineage from filenames, persist artifacts, select a
CV fold, or assign biological meaning to an unsupervised cluster. The caller
owns those boundaries and must pass only the appropriate reservoir for each
fit.

## Scope and input contract

Every fitting function receives `patient_ids` aligned to feature rows and an
explicit `train_patient_ids` allow-list. Rows outside that list raise before
fitting. Pass held-out IDs as `forbidden_patient_ids` as a second guard against
an accidentally widened training manifest. Patient IDs and photo IDs are
checked as metadata only; neither is concatenated with numeric features.

Feature matrices and centers must be finite two-dimensional NumPy `float32`
arrays. The module does not silently cast feature precision. L2 normalization,
PCA, lens selection, balanced reservoir sampling, and transforms must be fit
upstream on the same training scope. Metadata arrays must have one ID per row.
Photo IDs must identify one patient consistently.

## A1 dictionary and case head

```python
dictionary = fit_kmeans_dictionary(
    train_vectors,
    train_patient_ids_per_row,
    train_patient_ids,
    n_clusters=8,
    sample_weight=case_balanced_row_weights,
    random_state=42,
    n_init=10,
)
cluster_id, center_distance, center_margin = assign_to_centers(
    query_vectors,
    dictionary.cluster_centers_,
    chunk_size=512,
)
histograms = photo_balanced_histogram(
    cluster_id,
    source_photo_ids,
    patient_ids_per_row,
    n_clusters=8,
)
head = fit_case_classifier(
    histograms.descriptors,
    case_targets,
    histograms.patient_ids,
    train_patient_ids,
    regularization_c=0.1,
)
```

The dictionary defaults are deliberately explicit and reproducible: `random_state=42`,
`n_init=10`, and one native numerical thread during fit. Callers should construct
sample weights so each patient has equal total weight (and, where appropriate,
each source photo has equal weight within a patient). KMeans rejects fewer than
K distinct training vectors.

Assignment keeps only a `chunk_size × K` temporary distance matrix. It returns
the nearest center ID, Euclidean distance to that center, and the gap between
the second-nearest and nearest Euclidean distances. These are geometric
descriptors; they are not cancer probabilities.

For each source photo, `photo_balanced_histogram` first divides cluster counts
by that photo's tile count. It then averages photos within each patient, so a
large photo does not dominate a small photo. The returned `full_histograms`
retain K bins for inspection. `descriptors` drop the fixed final bin (`K-1`),
leaving K−1 dimensions per lens. Combine three lenses in the caller in fixed
`4x, 10x, 40x` order to obtain 9, 21, or 45 dimensions for K=4, 8, or 16.
If photo lineage is unavailable, this primary descriptor cannot be computed;
do not relabel a tile-balanced histogram as the same study.

`fit_case_classifier` requires one row per unique patient/case and both
case-level classes. It fits `StandardScaler` followed by unweighted,
intercept-enabled L2 `LogisticRegression` (`lbfgs`). Fit a fresh pipeline within
each inner training scope, choose settings from inner predictions, and create a
new pipeline on outer-train for the final outer prediction. The function never
receives patient IDs or case counts as numeric model features.

## A0 atlas exploration

`run_hdbscan_grid` runs the requested cross-product of `min_cluster_sizes`,
`min_samples_values`, and `selection_methods`; defaults are 8/16/32/64, 4/8,
and EOM/leaf respectively. It returns every configuration with labels and a
summary. Invalid input sizes fail instead of silently removing a candidate.
The caller must provide a capped train-only reservoir; HDBSCAN has no
out-of-sample prediction behavior here.

`fit_ward_tree` computes a Ward linkage once and defaults to a 5,000-row cap
because linkage memory grows quadratically. `ward_multicut` cuts that tree at
several cluster counts while preserving the input row order. The fitted tree
stores SHA-256 digests for the exact feature, patient, and photo arrays; a
same-length reordered or mismatched input is rejected before metrics are
attached to rows.

Both density and Ward outputs use `summarize_partition`, which reports overall
coverage/noise and, for every non-noise cluster, row fraction, patient/photo
counts, largest single-patient and single-photo row shares, and mean/90th
percentile distance to the cluster centroid. A large noise fraction reduces
how much of the reservoir the partition describes; it does not show that the
underlying images are invalid.

`cluster_case_label_crosstab` can add a descriptive source-case-label view for
reservoir rows. It reports row counts and distinct patient counts for both
case targets per cluster, including noise as `-1`, and rejects conflicting
targets for one patient. Targets are consumed only after clustering for this
summary and never enter the fit or assignment.

No cluster ID is a reviewed tissue or cancer label. Source concentration and
geometry support retrieval and prioritization for image review only.

## Stability

`kmeans_seed_stability` fits independent seeded dictionaries and assigns the
same caller-provided probe. `hdbscan_resampling_stability` uses a fixed probe
whose patient groups are retained in each fit, then samples other training
patients without replacement. This keeps probe rows identical while changing
the surrounding patient mixture; it is not an HDBSCAN prediction API.

Both return pairwise comparisons from `compare_clusterings`. Each comparison
includes ARI on the full probe, ARI on rows assigned by both fits, and the
fraction jointly assigned. For HDBSCAN, inspect all of these together: if a
large part of the probe is noise in both fits, a seemingly strong full-probe
ARI can reflect shared noise rather than stable, informative clusters. Cluster
numbers may be permuted across fits; ARI accounts for that permutation.

## Local verification

The local tests use small synthetic vectors (at most 64 rows per patient) and
verify contracts, numerical invariants, leakage guards, photo balancing,
noise-aware stability, and an end-to-end dictionary-to-case-head path. They do
not validate morphology, clinical performance, production data lineage, or
full-cohort resource use.
