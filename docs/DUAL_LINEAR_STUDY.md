# Dual-track linear study runner

`histology_data.dual_linear_study.run_linear_study` executes the train-only
atlas/classifier path (A0+A1) or the negative-reference linear ensemble (B0)
over the supplied nested patient splits. It uses the pure primitives in
`dual_atlas.py` and `dual_novelty.py`; it does not create a bundle, infer source
photos, fit transforms, or publish anything to a remote service.

```python
result = run_linear_study(
    "A", dataset_all, dataset_factory, original_splits, verified_source_map,
    shared_store, branch_store, deadline,
)
```

The shared store is read-only. Before the runner starts any fit or branch
artifact, it verifies every required transform at
`scopes/<scope>/transform.pkl`, including membership, patient/label metadata,
PCA32 shape, row counts, and source-photo lineage. For the production 3×2
protocol that means all six inner train transforms and three outer-train
transforms. A missing or mismatched transform stops the run. The caller's
`dataset_factory(ids)` must return exactly those cases; it may not silently
sample or widen the requested membership.

The `splits` argument is the original nested split manifest. Every outer test
case must appear once; each inner validation set must partition its
outer-train cases once; and patient IDs may not cross the relevant train/query
boundary. Production runs require the locked three outer folds, two inner
folds, and seed 42. Tiny fixture datasets carry the explicit `smoke` marker
and still pass the same scope and lineage checks.

## Branch A

A0 runs the 16-setting HDBSCAN EOM/leaf grid for each lens on the
`outer_0/inner_0` train reservoir, builds one Ward tree per lens with cuts at
2/4/8/16, and reports KMeans K=4/8/16 partitions. Every assignment table stays
aligned to the committed reservoir refs and includes its source case label,
patient and photo counts, geometry, and noise coverage. A0 runs three seeded
KMeans comparisons and three patient resamples for four predeclared HDBSCAN
shortlist configurations per lens. Stability artifacts preserve the fixed
probe refs and report full ARI, jointly-assigned ARI, and coverage.

A1 fits weighted MiniBatchKMeans dictionaries per lens and K=4/8/16 in every
inner train scope. It assigns every train and validation tile in chunks,
retaining tile ID, cluster ID, nearest-center distance, and second-minus-nearest
margin in compressed per-case artifacts. Per-photo normalized histograms are
averaged within each case, the fixed last bin is dropped, and a fresh
train-only-scaled L2 logistic head evaluates C=0.1/1.0. Selection uses pooled
case log loss across inner validation predictions. Candidates within 0.001 of
the best loss tie-break toward smaller K and smaller C. The selected dictionary
and head are refit from outer-train before producing one outer-test prediction
per case. The logical head count is computed from the supplied split manifest;
the locked 3×2 study has 39 A1 head fits.

The counts-only control has six log1p inputs: tile count and source-photo count
for each lens. It uses fixed C=0.1 and is reported separately from A1 selection.

## Branch B

B0 calls `dual_member.prepare_member_states` once per negative anchor and
train scope. Each state binds all three lens banks to the same anchor and scope.
The anchor is excluded from its member head's train and query rows. Per-case
descriptors use photo-balanced novelty distance summaries for all lenses; each
linear head is trained at C=0.1 and C=1.0, and member probabilities are averaged
with equal weights after query-order verification. C is selected by pooled
inner ensemble log loss. Outer-train members are rebuilt before the outer-test
ensemble. The number of member-head fits is calculated from the actual split
anchors; for three inner and six outer anchors per fold it is 54.

The same separate counts-only control is produced for B0. OOF manifests retain
one probability, reviewed target, patient ID, outer fold, and selected setting
per cohort case. These are internal artifacts and contain case metadata; keep
the branch store private.

## Resume and limits

Models, member states, descriptors, tile assignments, metrics, and OOF output
use immutable per-job `ArtifactStore` receipts. Existing commits are checksum
and store-identity verified; conflicting rewrites fail. Shared transforms are
never created by this runner. A monotonic deadline is checked between jobs and
cases; an interrupted run retains completed artifacts for a later invocation.

The synthetic E2E tests use 16 cases and 32 rows per lens per case. They verify
scope boundaries, source order, atlas metrics/assignments, nested selection,
anchor exclusion, OOF uniqueness, and resumable artifacts. They do not validate
morphology, clinical performance, or full-cohort resource use.
