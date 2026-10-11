# B member preparation and novelty-guided MIL

`dual_member.prepare_member_states` constructs the single source of truth for one negative-anchor member. B0 and B1 use these same per-lens banks and robust scales, so the two branches cannot silently fit different anchors or scopes.

```python
member = prepare_member_states(
    dataset_all,
    dataset_factory,
    transforms,
    source_map,
    train_ids,
    anchor_case_id,
    b_store,
    "outer_0/inner_0",
    deadline,
)
```

The transform must be committed for exactly `train_ids`. The anchor must be reviewed class 0 and must appear in that transform scope. The helper filters its PCA32 reservoir rows, verifies each tile/photo against the source map, fits one fixed 32-prototype bank per lens, then scores a photo-balanced reservoir of at most 256 rows per remaining training case/lens. It fits the median/IQR scale from those T-minus-anchor rows using equal weight per case, then per photo, then per row. Production heads require at least two cases from each class after anchor removal; smoke mode accepts one.

The helper commits its immutable state under `scopes/<scope>/members/<anchor_token>/state.pkl`. Reuse checks the ArtifactStore receipt, source identity, exact training membership, PCA state, tile/photo provenance, bank IDs, and scale samples before returning it. `member_id` and `anchor_token` are opaque fingerprints. `head_train_ids` preserves the supplied training order minus the anchor.

For B0, project each train/query case with the same transform, score every row against the matching lens bank, and aggregate `photo_balanced_descriptors` over its verified source-photo IDs. Fit/predict each logistic head with that exact bank; `fit_linear_novelty_head` and `predict_linear_novelty_head` enforce anchor and bank ownership.

## Nested B1 run

The shared stage commits transforms as `scopes/<scope>/transform.pkl` and supplies the verified source-photo map. `run_neural_study(dataset_all, dataset_factory, splits, source_map, shared_store, b_store, output, deadline, device="cuda", max_epochs=30, smoke=False)` loads those committed transforms and visits inner scopes, members, betas, then outer refits in sequence. It requires the locked 3×2 patient folds in production and refuses CPU fallback.

For every inner scope, each negative anchor gets a fresh head trained on T-minus-anchor. Beta 0.0 and 0.5 fits use the same seed per scope/member, but each starts from fresh E4 weights. The study loads each completed member's best checkpoint, predicts all inner-validation cases with that member's bank, checks ordered case IDs, and averages member probabilities equally. It selects beta 0.5 only when its mean inner ensemble BCE improves on beta 0 by at least 0.01.

Outer refit epochs are derived separately per beta: take the median member best epoch in each inner scope, take the median of those scope medians, then round up with a minimum of one. Every outer fold receives a fresh beta-zero refit as the matched control. If inner validation selects beta 0.5, the study also fits fresh beta-0.5 outer members. Both predictions are equal-weight member ensembles; no patches are removed and no outer-test labels select settings.

The public model factory is `make_novelty_model_factory(lens_states, beta=..., member_id=...)`; build its identity without constructing a throwaway model using `novelty_model_identity(config.model, lens_states, beta=..., member_id=...)`. `run_fit` checks that identity, the member anchor is absent from training and query partitions, and the production class gate passes before writing. Prediction reloads only a fully committed checkpoint whose trainer identity matches the bank, transform, scale, member, and beta.

Each member fit and prediction shard is committed to `b_store`. A paused fit returns its committed progress and no outer OOF output; resume reuses verified member state and the trainer's epoch checkpoints. The final production result has one beta-zero control and one nested-selected prediction per case. The optional beta-0.5 fold predictions are separately recorded when selected.

`smoke=True` accepts one synthetic outer/inner scope and at most two CPU epochs. It exercises bank preparation, both candidate fits, best-checkpoint prediction, beta selection, and resume, then returns a software-only result with an empty OOF list. Smoke results cannot be promoted to production metrics.

`tests/test_dual_member.py` covers committed member reuse and transform/anchor guards. `tests/test_dual_neural_study.py` checks beta and epoch selection and runs a bounded synthetic pause/resume through both candidates without creating OOF predictions.
