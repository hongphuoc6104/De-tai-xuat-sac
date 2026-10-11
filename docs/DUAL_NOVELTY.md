# Negative reference novelty (branch B)

This module provides the bank, novelty, descriptor, and full-bag MIL primitives for the negative-reference branch. It does not load clinical bundles or decide patient folds; the study runner supplies already verified vectors, source-photo IDs, and metadata.

## B0 reference bank and descriptors

For one member and lens, call `fit_negative_prototype_bank` with transformed `float32` vectors from one reviewed negative anchor. Each row's metadata must include `case_id`, integer `target=0`, `photo_id`, and unique `row_id`. The function rejects mixed cases, positive rows, fewer than 32 distinct vectors, empty clusters, and duplicate prototypes. KMeans weights each source photo equally and stores a distinct source row for every prototype.

Query data with `query_novelty(query_vectors, bank, chunk_size=...)`. The maximum chunk size is 512; results include nearest Euclidean distance and the winning prototype index. The prototype row/photo can be recovered from the same bank object's provenance fields. Distances are reference novelty scores, not cancer probabilities.

For one case, `photo_balanced_descriptors(scores, photo_ids)` computes the median, q90, q95, and mean of each photo's upper five percent, then gives every source photo equal weight. `descriptor_vector_by_lens` gives a stable lens-major feature order. `fit_robust_novelty_scale` requires every row to be marked `partition="train"`, requires the explicit `allowed_train_case_ids` membership, and rejects the anchor. Its weighted median/IQR gives equal total weight to each case, then equal weight to each photo within a case, then equal weight to rows within a photo. Freeze it with `standardize_novelty`. A zero IQR maps values to zero and is recorded as a degenerate scale.

`fit_linear_novelty_head` fits a train-only `StandardScaler` and logistic regression over case descriptors. Its production gate requires at least two non-anchor cases of each class. Prediction checks the exact bank identity and refuses to score the anchor. `equal_weight_member_ensemble` requires each member's explicit, identical ordered query case IDs before averaging probabilities.

## B1 novelty-guided MIL

Build one `NoveltyLensState` per available lens from the shared train-only L2/PCA32 transform, that member's 32-prototype bank, and its `T-minus-anchor` robust novelty scale. The transform arrays and prototypes must be `float32`; raw feature dimension must match the E4 `ModelConfig`. All lens banks in a model must belong to the same anchor.

`NoveltyMILModel` keeps the E4 projection, gated attention, classifier, and full bags. For every patch it computes a frozen L2/PCA32 representation and nearest-bank Euclidean distance, applies the frozen robust scale, clips it to `[-3, 3]`, and adds `beta*z` to the learned attention logit before softmax. Supported candidates are `beta=0.0` and `beta=0.5`; no patch is removed. Beta zero is the matched full-bag control.

The transform, bank, and scale are registered as non-trainable buffers and are included in the checkpoint state. `get_config()` contains their hashes, schema, member token, bank IDs, scale policy, and beta. Create the model factory accepted by `run_fit` with `make_novelty_model_factory(lens_states, beta=..., member_id=...)`; compute its identity with `novelty_model_identity(config.model, lens_states, beta=..., member_id=...)`. This lets the trainer construct the seeded model once and still verify its actual configuration.

```python
factory = make_novelty_model_factory(lens_states, beta=0.5, member_id=member_token)
model_identity = novelty_model_identity(
    train_config.model, lens_states, beta=0.5, member_id=member_token,
)
result = run_fit(
    train_dataset,
    validation_dataset,
    output_dir,
    train_config,
    fit_identity=fit_identity,
    deadline=deadline,
    model_factory=factory,
    model_identity=model_identity,
)
```

The trainer verifies that the declared identity equals the factory model's `get_config()`, adds it to the fit/checkpoint identity, and fingerprints both novelty modules. Before creating fit output, it calls the model's explicit `validate_fit_scope` contract when available; `NoveltyMILModel` uses it to reject the anchor in train or validation and enforce the production two-per-class gate. To resume or load predictions, construct the same factory from the same verified transform, bank, scale, member, and beta. A changed bank, transform, scale, or beta fails the existing output identity guard. The default trainer still constructs `MILModel(config.model)` and emits its existing prediction fields.

## Scope and verification

These primitives do not implement nested-fold scheduling, verified data loading, prototype-bank persistence, source-image retrieval, OOF evaluation, or study-level candidate selection. The caller must fit every transform, bank, scale, and standardizer from the correct training scope, keep the anchor out of its supervised head, and use the same member bank for head training and query inference. A software pass does not establish predictive benefit or pathology meaning.

The synthetic tests in `tests/test_dual_novelty.py` and `tests/test_dual_mil.py` cover bank/member guards, chunk and permutation invariance, photo balancing, degenerate scale handling, beta-zero output/gradient parity, full-bag beta bias, and a tiny checkpoint/resume/predict run.
