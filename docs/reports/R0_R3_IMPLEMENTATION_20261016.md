# R0–R3 implementation — 16/10/2026

CPU-only implementation of [the current plan](../plans/IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md), in `histology_data/image_classification_r0_r3.py`. It reads the completed B1 release/study without writes, replays its actual outer/inner records, and publishes a separate v2 namespace.

R1 uses lens-restricted shared inner assignments. R2 adds sample L2 normalization before the train-only scaler. R3 jointly tunes lambda/ElasticNet ratio, with a separate SAGA-L2 numerical control. Patient-class weights, inner-only thresholds, deterministic ties and baseline provenance remain explicit. Models, validation predictions and receipts commit per fit; completed reports verify their full fit graph before reuse or deployment.

Each fit has a cancellable worker with a 900-second total cap across iteration retries and a 20-minute session reserve. Two timeouts on the same unit require attention. Failed/nonconverged fits never contribute to results. Synthetic producer→CLI→partial→resume→refit→save/load→tamper tests cover the real paths; clinical data is not trained locally.

```bash
python -m histology_data.image_classification_r0_r3 \
  --release-root /path/to/verified/B1/release \
  --baseline-root /path/to/verified/B1/head_study \
  --output /path/to/new/head_ablation_r0_r3_v001 \
  --max-new-fits 2
```

The two-fit cap is the technical L-BFGS/SAGA pilot. Remove it to resume the same scientific study. `--refit` additionally publishes full-data models; they do not supply an independent performance estimate. Required Python packages: pinned scikit-learn 1.8.0, NumPy/SciPy/Pillow/joblib and Matplotlib; the Colab preparation receipt records exact installed versions.

This document records implementation, not completion of real-data experiments. Production results and runtime/Drive receipts belong in the private execution directory. No GPU, encoder extraction, source ZIP staging, patch labels or shutdown of the local machine is part of this run.

Local verification:498 tests passed,1 optional skipped;422 assertion-bearing test methods. Proof:Results/proofs/test_evidence_20261016_014233.log. Critique/docs/lint/compile passed. CI and real-data CPU pilot remain separate gates.
