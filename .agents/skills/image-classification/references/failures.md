# Lỗi đã tái hiện — original-image classifier

## [ERR-20261012-28] TIFF decode dùng file đã đóng
- **Triệu chứng (Symptom):** E2E metadata→inventory→extract trên TIFF thất bại `ValueError` khi EXIF transpose truy cập file đóng.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_image_classification.py::test_e2e_tiny_frozen_encoder_vectors_keep_row_identity_and_hashes`; log `Results/proofs/image_before_fix_original_image_qa.log`.
- **Nguyên nhân cốt lõi (Root Cause):** PIL parser trả ảnh cần lazy file access sau khi member stream không còn sống.
- **Giả định sai lầm (Wrong Assumptions):** Hoàn tất parser đồng nghĩa mọi TIFF metadata/pixel đã được materialize.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Bounded member bytes → BytesIO sống trong decode/transpose; giới hạn byte/pixel giữ nguyên. E2E TIFF qua trong `image_after_fix_targeted_qa.log`.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Test TIFF thực trong fixture toàn pipeline, không chỉ PNG; giữ nguồn sống đến khi pixel/EXIF đã được đọc.

## [ERR-20261012-29] Inventory ảnh gốc chấp nhận coordinate patches
- **Triệu chứng (Symptom):** ZIP `Tiles/.../x_y.png` được nhận như nguồn original-photo nếu metadata giả khớp tên.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_image_classification.py::test_inventory_rejects_coordinate_patch_archive_members`; before log ở trên.
- **Nguyên nhân cốt lõi (Root Cause):** Basename match chỉ chứng minh liên kết, không chứng minh cấp đơn vị ảnh.
- **Giả định sai lầm (Wrong Assumptions):** Ảnh có dòng metadata tương ứng luôn là original image.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Guard archive coordinate-patch schema trước tạo inventory; regression qua trong after log.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Phân biệt original-photo pipeline và patch pipeline bằng contract rõ ràng, không đổi target khi nhập dữ liệu.

## [ERR-20261012-30] Release khai training_ready trước khi đủ features
- **Triệu chứng (Symptom):** Deadline trả status partial nhưng release.json ban đầu khai training_ready=true.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_image_classification.py::test_deadline_returns_partial_status_and_never_a_complete_release`; before log ở trên.
- **Nguyên nhân cốt lõi (Root Cause):** Readiness suy ra từ mode/config thay vì coverage đã commit.
- **Giả định sai lầm (Wrong Assumptions):** Không phải smoke đồng nghĩa dữ liệu đã đủ để train.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Release chưa ready khi khởi tạo; chỉ promote sau verify đủ parts. Trainer từ chối partial/smoke; E2E regression qua.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Readiness là bằng chứng hậu điều kiện, không phải ý định cấu hình.

## [ERR-20261012-31] Patient split vẫn rò rỉ ảnh trùng nội dung
- **Triệu chứng (Symptom):** Cùng bytes ảnh được gán tên/người/target khác nhau vẫn được train trong synthetic E2E.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `Results/proofs/image_before_fix_duplicate_content_and_rebind.log`; fixture inventory→extract→train có hai patient IDs cho cùng ảnh.
- **Nguyên nhân cốt lõi (Root Cause):** Gate chỉ kiểm image_id và patient partition, không kiểm content SHA.
- **Giả định sai lầm (Wrong Assumptions):** Different filenames/patient IDs bảo đảm nội dung độc lập.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Audit content groups và từ chối repeated original SHA trước fit; lưu cross-patient/target-conflict counts. After E2E log `image_after_fix_duplicate_content_and_rebind.log` xác nhận bị chặn.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm cả identity và content duplication trước grouped evaluation.

## [ERR-20261012-32] ZIP rebinding giữ size/CRC nhưng đổi bytes
- **Triệu chứng (Symptom):** Staged ZIP sửa nội dung, giữ tổng size/member CRC, qua testzip nhưng vẫn được encode.
- **Đường dẫn tái hiện E2E (Reproduction Path):** Before duplicate/rebind log ở trên; synthetic payload collision giữ CRC nhưng SHA khác.
- **Nguyên nhân cốt lõi (Root Cause):** Runtime chỉ kiểm size/central CRC, gắn vector vào source archive SHA cũ mà không rehash staged file.
- **Giả định sai lầm (Wrong Assumptions):** ZIP CRC đủ làm checksum toàn vẹn nguồn.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Verify staged archive SHA trước encode/commit, cache binding theo path/stat identity trong cùng runtime; regression chặn trước encoder trong after log.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** CRC dùng phát hiện lỗi ZIP, SHA dùng bind nguồn; đổi runtime path phải verify bytes.

## [ERR-20261012-33] Runtime bỏ requirements được include gián tiếp
- **Triệu chứng (Symptom):** ZIP code thật chỉ có requirements-image-classification.txt, thiếu requirements-features.txt được tham chiếu bằng `-r`.
- **Đường dẫn tái hiện E2E (Reproduction Path):** Build actual source ZIP rồi kiểm include closure; `Results/proofs/image_runtime_requirements_before_after.json` lưu before missing/after đủ; regression `tests/test_image_classification_runtime_builder.py`.
- **Nguyên nhân cốt lõi (Root Cause):** Whitelist builder chỉ gồm entry requirements, không resolve local dependency closure.
- **Giả định sai lầm (Wrong Assumptions):** File requirements duy nhất đủ cho pip install trên VM mới.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Recursive closure local-only, reject missing/traversal/symlink; actual archive verification và regression transitive includes qua.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm package bằng requirements thật của runtime; fixture tối giản không được che dependency graph.

## [ERR-20261012-34] Module ảnh gốc làm đổi pin runtime MIL lịch sử
- **Triệu chứng (Symptom):** Full conveyor trả 401 passed/1 failed/1 skipped; MIL notebook pin không khớp ZIP mới dù không sửa trainer MIL.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_mil_notebook.py::test_runtime_builder_is_reproducible_and_contains_only_code_and_requirements`; `Results/proofs/test_evidence_20261012_021304.log` chạy actual builder và notebook hiện có.
- **Nguyên nhân cốt lõi (Root Cause):** MIL builder gom mọi module và requirements trong repository, gồm pipeline ảnh gốc mới có runtime độc lập.
- **Giả định sai lầm (Wrong Assumptions):** Thêm pipeline độc lập không ảnh hưởng identity của code archive dùng glob toàn repository.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** Scope MIL archive loại namespace/dependency riêng image_classification; actual pin cũ được phục hồi. MIL notebook và image runtime packaging suite qua 10/10.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Runtime từng domain có phạm vi nguồn rõ ràng; extension pipeline khác không đổi cache/notebook pin ngoài ý muốn.


### [ERR-20261014-01] Reporter required a scaler hash absent from the producer schema
- **Triệu chứng (Symptom):** Real 2,802-image study completed, but analyzer stopped: “Fold 1 scaler scale hash does not match its serialized audit vector.”
- **Tái hiện E2E (Reproduction Path):** Exact Colab analyzer invocation persisted `Results/image_classification/execution_bop_20261014/report_e2e_reproduction.json`; returncode 2. Synthetic actual-producer reproduction: `tests/test_image_study_report.py::test_analyzer_cli_accepts_an_actual_v1_producer_study`.
- **Nguyên nhân cốt lõi (Root Cause):** Version-1 trainer stores `scaler_scale` in committed study audits without a separate `scaler_scale_sha256`; the old report-only fixture invented that field and hid producer/consumer drift.
- **Giả định sai lầm (Wrong Assumptions):** A hand-built report fixture represents the real trainer's persisted schema. Adding missing hashes to an existing study would be safe; it would instead mutate committed evidence.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Reporter requires and verifies the producer study commit binding study/OOF file bytes; validates dimensions, finite positive scales and optional per-array hashes when present. Existing study/model/cache remain immutable. Actual producer→private export→report CLI E2E and tamper/nonpositive-scale regressions verify the boundary.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Test reporting against real producer output. Do not invent audit fields or rewrite committed studies to satisfy a consumer; bind serialization through its declared commit contract.


### [ERR-20261014-02..05] Fixed-patient head-study replay and routing contracts
- **Triệu chứng (Symptom):** Tiny real-producer E2E initially failed identical-study resume (tuple/list config mismatch), budget stop (nonexistent BudgetStop), per-lens threshold reports (one arbitrary threshold per fold), and wrong-head row rejection.
- **Tái hiện E2E (Reproduction Path):** `tests/test_image_classification_studies.py`: inventory → frozen tiny encoder → legacy baseline → new studies → reload/resume/CLI/source review. Before proof `Results/proofs/image_studies_before_20261014.log` (4 failed/4 passed).
- **Nguyên nhân cốt lõi (Root Cause):** Python-native configuration was compared with decoded JSON; caller guessed the exception name; reused pooled-report metadata assumed one head per fold; file integrity alone did not bind per-row decisions to head tuning.
- **Giả định sai lầm (Wrong Assumptions):** JSON preserves tuples; deadline exception names are interchangeable; a global threshold map describes lens-specific heads; checksum-valid predictions imply correct routing semantics.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Canonical JSON config; catch the defined BudgetExhausted; explicit fold×head threshold map; bind every OOF head/lambda/threshold to its committed tuning audit. Whole-image and packaged CLI E2E, source-byte replay, weight-scale invariance and baseline/cache preservation tests pass. No clinical artifacts were changed while fixing these prototype issues.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Replay stored JSON through the actual writer/reader, test budget interruption with the real exception owner, and validate semantic routing in addition to hashes. Encode head/weight/PCA/implementation/environment in identity before creating experiments.


### [ERR-20261014-06] Lab source review was outside complete-artifact coverage
- **Triệu chứng (Symptom):** E2E showed partial archive arguments trained a model before failing review, review/private symlink was followed, and completed lab manifest omitted diagnostics/review evidence.
- **Tái hiện E2E (Reproduction Path):** `test_partial_archive_map_fails_before_training`, `test_source_review_refuses_existing_private_symlink`, `test_complete_lab_uses_cache_and_publishes_source_review`; before log `Results/proofs/image_lab_before_20261014.log` (3 failed/12 passed).
- **Nguyên nhân cốt lõi (Root Cause):** Source preflight was deferred until after fitting; publication wrote directly into a possibly existing directory; lab completeness only covered its comparison table, not all linked deliverables.
- **Giả định sai lầm (Wrong Assumptions):** A successful head fit implies its review source is ready; mkdir(exist_ok=True) creates a fresh private directory; comparison hashes prove all source review evidence is durable.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** All-or-none archive arguments and full source hash bindings before training; same-process signatures avoid rehashing unchanged archives; staged review with atomic rename and minimized identifier references; lab commit includes diagnostics, every review file and all sub-study commit hashes. Budget checks cover bounded reads and use the lab's absolute deadline. E2E plus tamper/budget-resume regressions pass before clinical data execution.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Preflight prerequisites before fits and define completeness over the entire artifact graph. Never follow preexisting private symlinks; keep source review atomic and distinct from training labels.
# [ERR-20261014-07] Comparing new frozen releases using an old bound baseline

- Symptom: the existing loader rejects a geometry-changed feature release despite identical original sources.
- Reproduction Path: `scripts/reproduce_cross_release_comparison.py`; before proof `Results/proofs/image_cross_release_e2e_before_20261014.log` uses real producer flow with synthetic original TIFFs48→64.
- Root Cause: baseline/source release identity is deliberately bound; geometry or encoder changes create another feature identity.
- Wrong Assumptions: identical image counts alone justify reusing an unrelated baseline fold model; relaxing the old release check would be safe.
- Resolution & Proof: explicit `load_comparison_data` verifies both releases and every source lineage field, replays fixed patient assignments, and fingerprints that binding in new study identity; producer→study→resume E2E and source/label/coverage rejection tests pass. Old strict loader remains unchanged.
- Golden Prevention Rule: preserve existing release guards; compare new frozen representations only through verified source-cohort/assignment bindings.

# [ERR-20261014-08] Notebook source-bundle pin drift after module integration

- Symptom: full tests fail because the reference MIL notebook pin differs from the deterministic package of current source modules.
- Reproduction Path: `tests/test_mil_notebook.py::test_runtime_builder_is_reproducible_and_contains_only_code_and_requirements`; full producer/notebook E2E failure is recorded in `Results/proofs/test_evidence_20261014_050054.log`.
- Root Cause: shared code-only bundles include the package's modules; adding modules changes the bundle SHA while a checked-in notebook still has the prior source pin.
- Wrong Assumptions: a new image module cannot affect a shared source-bundle receipt.
- Resolution & Proof: rebuild the deterministic package and synchronize the notebook's generated pin; targeted27 tests and full448 passed/1optional skipped in `Results/proofs/test_evidence_20261014_050607.log`. The new bundle is not uploaded over old deployed bundles or model/data releases.
- Golden Prevention Rule: keep source/notebook pins coherent per revision and preserve deployed release identities; never weaken integrity assertions to bypass a changed bundle.

## [ERR-20261014-09] Resume tái gọi producer và làm thay đổi bằng chứng đã hoàn tất
- **Triệu chứng (Symptom):** Một lượt tiếp tục frozen run có thể gọi lại audit hoặc extraction cho release đã complete, chạm `status.json`/`release.json`, đồng thời ghi report mới lên cùng execution directory. Bắt lỗi RetCCL bằng `OSError`/`AssertionError` hoặc timeout chung có thể che lỗi đọc Drive/ghi đĩa hoặc sai SHA cục bộ như thể chỉ là nhánh B1 chưa sẵn sàng.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_image_classification_execution.py::test_e2e_producer_to_readonly_n0_and_completed_cache_selection` tạo TIFF fixture thật → inventory → cache frozen → audit → resume read → chọn completed cache; các bài tamper/binding/chunk/symlink kiểm tra đường từ chối.
- **Nguyên nhân cốt lõi (Root Cause):** Producer API hỗ trợ tiếp tục tính toán nhưng không phải API đọc chỉ-đọc: extractor cập nhật trạng thái/manifest kể cả khi các shard đã có; loader audit có nhánh dọn chunk chưa commit. Orchestration cũng dùng một thư mục cho log nhiều lần chạy và phân loại lỗi tùy chọn quá rộng.
- **Giả định sai lầm (Wrong Assumptions):** Gọi producer lần nữa để “xác minh” không thể đổi artifact hoàn tất; một cache có đủ file là cache tương thích; mọi `ValueError`/`OSError` trong nhánh trọng số đều có thể bỏ qua.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** `load_verified_audit` yêu cầu pin audit/commit, kiểm binding, chunk, metadata nguồn, coverage và selection mà không gọi writer; `select_completed_feature_release` dùng descriptor thật và `_load_feature_matrix`, trả cache chính xác hoặc từ chối cache lỗi; attempt có thư mục duy nhất còn A1/B1 ở cache root ổn định. RetCCL chỉ đánh dấu pending cho `URLError`/HTTP transport và schema `ValueError` đã nhận diện; caller phải bọc timeout phát sinh tại `urlopen`/`response.read` thành lỗi mạng, còn timeout từ ghi file cục bộ, sai SHA, local I/O, assertion và lỗi runtime là fatal. E2E dùng downloader thật xác nhận timeout đọc từ xa đã bọc là optional và `TimeoutError` lúc ghi file là fatal. 45 test audit/cache/encoder/execution qua trong QA CPU; E2E xác nhận hash toàn bộ N0/release/report cũ không đổi.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm tra artifact hoàn tất bằng reader chỉ-đọc có pin trước khi quyết định gọi producer; mỗi lượt chạy ghi vào thư mục mới, còn cache tiếp tục dùng identity/root cố định. Chỉ đánh dấu lỗi từ biên mạng hoặc schema checkpoint đã nhận diện là optional; không phân loại `TimeoutError`/`ConnectionError` trần từ khối tải-xử-lý chung.

## [ERR-20261015-01] Public RetCCL checkpoint needs Google's confirmation flow
- **Triệu chứng (Symptom):** A bounded request to the public Drive download URL returned an HTML virus-scan confirmation form for `best_ckpt.pth`, so the raw binary downloader could not start the 90 MiB transfer.
- **Đường dẫn tái hiện E2E (Reproduction Path):** `tests/test_retccl_public_download.py` runs the pinned gdown 6.2.0 implementation against a stubbed public confirmation form, then exercises complete, interrupted/resumed, denied, wrong-name, and local-promotion cases without downloading the checkpoint.
- **Nguyên nhân cốt lõi (Root Cause):** A public Drive link can require a confirmation request before it serves the binary; a successful HEAD/direct URL probe does not mean the first GET body is the checkpoint.
- **Giả định sai lầm (Wrong Assumptions):** Treating the first HTML response as the checkpoint or as proof that the file is private; retrying through an arbitrary mirror or reusing browser credentials would bypass the declared public-source contract.
- **Giải pháp & Kết quả kiểm chứng (Resolution & Proof):** The opt-in helper uses pinned `gdown==6.2.0` by the author's file ID, resolves and checks the exact filename before downloading, disables cookies, resumes its own partial, verifies the pinned source size and full SHA-256, then atomically publishes without overwriting an existing path. The destination must be local storage supporting same-directory hard links; a separate caller-owned, readback-verified copy is required for Drive FUSE. First-acquisition SHA, author repository/revision, file ID, source URL, and size are returned as a receipt for immutable caller-side recording. Typed remote request/confirmation failures remain optional; local write/promotion errors remain fatal. Seven new download E2E cases plus twelve execution regression tests pass with transport fixtures; no checkpoint bytes were fetched.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Use the official public confirmation flow with cookies disabled; record the first observed full SHA as a local pin with source revision and file ID, and keep local I/O/integrity failures fatal.
