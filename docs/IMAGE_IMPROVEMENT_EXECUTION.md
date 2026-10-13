# Original-image improvement execution

The approved sequence is technical audit → resource smoke → frozen1024 and pathology-pretrained512 → conditional supervised adaptation. Labels stay original-image Glade0→0,3/4/5→1. Full frames remain intact; all views of one patient retain the baseline partition.

`image_classification_audit.audit_original_images` streams inventoried source ZIPs with full archive/member hashes, RGB-pixel duplicate checks, pHash candidates and descriptive QC. Its typed result exposes audit root/status/binding and48 representative image IDs. Chunk32 receipts support verified resume. Cross-patient candidates and exact label conflicts require review before fitting; pHash alone is not a medical label error or automatic exclusion. Dead same-host locks require explicit recovery; cross-host locks are not reclaimed.

`image_classification_improvement.load_comparison_data(new_run, original_run, baseline_root)` is the explicit cross-release path. The old same-release loader remains strict. The new path independently verifies both releases, matches every source image/patient/lens/grade/target/archive/member/hash/dimension, and replays the exact baseline outer assignments. Its provenance binding includes both release IDs, baseline receipt, source-lineage/assignment and loader hashes in study identity. Scaler/tuning remain train-only. Benchmark smoke vectors are marked not training-ready.

`benchmark_whole_frames` uses bounded decode, existing full-image resize/pad and adaptive encoder batching; measures decode/preprocessing/inference/cold vs warm timing and allocated CUDA memory. It binds source bytes, selection, encoder/geometry/code to checksummed smoke files.1024 is measured at batch1,512 at batch4. These resource measurements do not estimate accuracy.

`pathology_encoder.RetCCLResNet50Encoder` requires the observed full SHA pin, safe weights-only loading, exact encoder-only key/shape schema and strict loading. It uses frozen/eval ResNet50 GAP/2048D, full square input and ImageNet normalization. The authors publish a download endpoint but no checksum. First acquisition must be recorded as trust-on-first-download using the author URL/revision, then hashes pinned on every reuse. Do not bypass checkpoint schema mismatches or silently crop inputs.

Heavy data processing/weights acquisition run in Colab only. Prepare code/tests/CI before creating T4; mount through the user's official consent. Stage source ZIPs one at a time to SSD, hash/read back and reuse locally for audit/bench/extraction, rather than unpacking all TIFFs. Store clinical artifacts outside Git, separate from the existing feature/baseline studies.

Fine-tuning remains a conditional later stage. Existing LR fold resume and frozen-vector caching do not implement CNN optimizer/epoch/RNG/sampler resume. No full CNN job is launched until its own tiny E2E/checkpoint tests and Colab gradient smoke pass.

Synthetic evidence: `scripts/reproduce_cross_release_comparison.py`, `tests/test_image_classification_improvement.py`, `tests/test_image_classification_audit.py`, `tests/test_pathology_encoder.py`. No real clinical model is fitted locally.
