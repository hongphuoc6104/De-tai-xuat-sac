# Fixed-patient image-head studies

Purpose: reuse completed whole-image frozen features, preserve the original baseline, and compare weighting/head/PCA choices with identical outer patient assignments. This is original-image classification, not case MIL or patch classification.

## Preregistered cached experiments

`patient_lr`, `class_lr`, `uniform_lr`, `patient_pca32`, `patient_pca64`, `per_lens_lr`, `per_lens_pca32`. Weight objectives are explicit; each has total weight N. Tune penalty lambda in {50,5,0.5,0.05}; sklearn C=1/(lambda*sum(weights)). For patient-class weighting this reproduces the old sum-2 objective/C-grid, not a claimed improvement by changing units.

Load and verify the exact completed feature release and baseline study once. Copy its original image-to-outer-fold assignments; never recompute independent outer folds for individual lenses. Inner scaler/PCA fitting and lambda/threshold selection use only the outer training data. PCA uses seeded randomized SVD. Infeasible lens/PCA support returns attention before training, not a silently missing cohort.

## Colab command

CPU is sufficient for fitting cached heads; source TIFF decode/review also runs in Colab. Pin sklearn1.8.0. Install needed runtime dependencies without installing a new GPU encoder for cache-only training.

```bash
python scripts/run_image_classification.py head-lab \
  --run-root /content/drive/MyDrive/histology/image_classification/v001/runs/<feature_release> \
  --baseline-root /content/drive/MyDrive/histology/image_classification/v001/runs/<feature_release>/training/<baseline_study> \
  --output-root /content/drive/MyDrive/histology/image_classification/v001/head_labs \
  --archive-4x /content/drive/MyDrive/NCKH_Duy/data_4x.zip \
  --archive-10x /content/drive/MyDrive/NCKH_Duy/data_10x.zip \
  --archive-40x /content/drive/MyDrive/NCKH_Duy/data_40x.zip \
  --budget-minutes 120 --reserve-minutes 15
```

Feature matrices remain immutable. New artifacts use `runs/<release>/head_studies/<study_id>/` and `head_labs/<lab_id>/`. Every fold commits model dictionary, full train-only audits and OOF predictions. Source, baseline assignments, weight/head/PCA choices, implementation hashes and math-library versions enter identity. Resume verifies receipts and ownership; cross-host stale locks are not auto-deleted.

Review preserves exact bytes for a bounded sample of originals in a private directory, verifies source ZIP/image hashes and writes anonymous whole-frame montages. They select examples for review, not tumor-location annotations or new training labels. Models use image targets from original Glade; case reports remain unused.

Reports contain pooled/per-lens metrics, patient-balanced image metrics and paired patient-bootstrap deltas conditional on saved OOF. Scores from outer folds already examined for research are exploratory, not an untouched external test. All model choices still use inner validation.

## One-image inference

`image_classification_studies.predict_image(study_root, image_path, weights_dir, objective_lens=40)` uses the original release's encoder/preprocessing. A per-lens model requires the known lens. Global models accept one image without lens metadata. CPU cached-head training does not prove raw-image encoder compatibility in a different framework build; match the original encoder descriptor when running raw-image inference.

## Scope

No CNN fine-tuning, stain normalization or1024 feature extraction is started by this lab. Those need separate benchmarks/protocols after reviewing these results. Local tests use only small synthetic originals. Full data/preview generation/fitting runs on Colab.
