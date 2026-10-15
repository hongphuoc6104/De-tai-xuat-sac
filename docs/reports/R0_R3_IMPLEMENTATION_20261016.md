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

## Numerical protocol correction

The first real-data v2 run stopped at its matched-solver gate after114committed fits, preserving them. A same-source diagnostic found identical preprocessing but differing precision and stopping accuracy. The corrected protocol `original-image-retccl-head-ablation-v3-float64` optimizes both solvers in float64 and uses normalized solver tolerances1e−8. It preserves source images/features, labels, folds, patient weighting, lambda/ratio grids and score/objective control thresholds. No outer performance metrics were used for this correction.

Updated verification:499local tests passed,1optional skipped;423active-assertion methods. Evidence:Results/proofs/test_evidence_20261016_022457.log. Corrected code CI and real-data control remain pending; the old partial run is not a completed R0–R3 result.

## Conditioning follow-up
V3 preserved121fits before a900s SAGA-control timeout. Same-scope Colab E2E validates objective-preserving weighted centering/curvature conditioning:3809iterations/120.84s, no warnings, score delta5.07e−6, objective delta1.45e−11. Conditioning receipt records logical and solver parameters; new study identity and strict R1/R2-only cache import preserve108verified L-BFGS fits. No outer labels/metrics used to design the fix. Full validation/CI required before deployment.

## Adaptive conditioning follow-up
Universal conditioning slowed the known easy pilot beyond600s (activeCPU, RSS260MB) versus63.68s for the unconditioned fit. Pilot stopped safely; no outer metrics used. Solver now tries the unconditioned first iteration budget and conditions only nonconverged retries, retaining objective and gates. Operator must isolate process group and verify no grandchildren after interrupt. New source identity/test evidence/CI must be bound before the next attempt.

## Warm retry follow-up
V5 preserved115fits after a900s weak-control timeout. Numerical E2E now keeps SAGA iterates between normal2000/10000 retries and reserves conditioning for the final50000 retry. Intercept offsetdelta preserves logits, C×weights preserves objective. CLI worker continuity test failedbefore/passedafter. Colab inner-only weak scope:176.72s/2000+4608iterations, validationdelta8.18e−6/objectivedelta3.84e−12; difficult scope validation remains required before production. No quality gate/folds/grids/labels/tolerance relaxed.
