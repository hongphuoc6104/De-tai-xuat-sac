# Original-image classification QA

## Result

The focused local QA suite passes: 16 tests passed in 2.29 seconds using synthetic Metadata.xlsx, three synthetic original-photo ZIPs, and an injected four-value pixel encoder. No pretrained checkpoint was downloaded. Two additional targeted regressions also pass: duplicate image bytes with conflicting labels across patients are rejected before training, and a same-size payload rebind that preserves member size and CRC is rejected by the inventoried archive hash before any feature is committed.

Coverage includes the 0→no-cancer and 3/4/5→cancer image-level mapping; mixed labels within a patient; exact filename/lens matching; duplicate, missing, and unknown-grade failures; ZIP/member receipts; rejection of coordinate-patch members; full-frame aspect-preserving resize-pad with edge landmarks retained; vector-to-row and shard commit hashes; interruption, resume, tamper rejection, live/cross-host lock handling, and deadline partial status; patient-grouped outer and inner folds; train-only scaler fitting; persisted deployment reload and repeat single-image prediction; and deterministic code-only runtime packaging that includes the actual transitive requirements closure, excludes data, rejects symlinks, and fails on missing or traversal includes.

The before-fix reproductions are recorded in [image_before_fix_original_image_qa.log](proofs/image_before_fix_original_image_qa.log) and [image_before_fix_duplicate_content_and_rebind.log](proofs/image_before_fix_duplicate_content_and_rebind.log). Passing runs are recorded in [image_after_fix_targeted_qa.log](proofs/image_after_fix_targeted_qa.log) and [image_after_fix_duplicate_content_and_rebind.log](proofs/image_after_fix_duplicate_content_and_rebind.log).

## Scope limit

These tests establish the synthetic local software contract only. They do not establish performance on clinical images, validate archive contents from the mounted source, verify ResNet-50 weights on Colab, or establish production readiness. No patient identifiers or real image data were used in this QA run.
