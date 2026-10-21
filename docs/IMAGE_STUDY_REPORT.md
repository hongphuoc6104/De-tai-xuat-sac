# Verified original-image OOF reporting

The frozen-feature trainer produces one patient-grouped outer-test prediction per original image. This report verifies those persisted outputs and adds AUROC, average precision (AUPRC), fixed/validation-selected confusion matrices, lens breakdowns and optional patient-group bootstrap intervals. It does not refit a model or evaluate deployment training predictions.

## Inputs and execution

Install the image-classification dependencies and `matplotlib>=3.9,<4`. Copy these four artifacts from a completed, verified study into an owner-only directory (0700; files 0600): `study.json`, `oof_predictions.csv`, `training_config.json`, `study.commit.json`. Keep patient/image identifiers private.

```bash
python scripts/analyze_image_study.py \
  --input-dir /path/to/private-study-copy \
  --report-dir /path/to/aggregate-report \
  --bootstrap-replicates 1000
```

The default expected cohort is 2,802 images / 18 patients, 2,298 negatives / 504 positives, with lens counts 364 / 830 / 1,608. For another reviewed cohort, set the explicit `--expected-*` arguments; a mismatch stops report creation.

## Integrity contract

- Require complete, valid study commit identity and matching study/OOF file hashes before computing metrics.
- Verify config identity, unique image rows, grade-target mapping, both classes, patient-disjoint outer/inner splits, scaler train-only IDs and validation-only C/threshold selection.
- Version-1 trainer audits serialize scaler scales but do not require a separate outer-fold scale hash. The committed study hash binds those scale values; dimension/finite/positive-scale checks still apply. Any supplied per-array scale hash must also match. Never rewrite a completed study to invent fields for a reporter.
- Recompute every producer metric from saved OOF CSV and compare it with `study.json`; compute AUPRC as average precision.

Outputs: `verification.json`, `report_vi.md`, `roc_oof.png`, `precision_recall_oof.png`, `confusion_selected_threshold.png`. Reports contain aggregate metrics and counts, with no source filenames or patient IDs. Bootstrap resamples patients using the saved OOF predictions and thresholds; it does not retrain the models and does not establish external clinical validity.

## Regression evidence

`tests/test_image_study_report.py` includes actual inventory → tiny encoder → production trainer → private export → report CLI E2E, plus tamper, invalid scale, counts, leakage and aggregate-only output checks. The historical failure and exact real-data reproduction are documented in `.agents/skills/image-classification/references/failures.md`.
