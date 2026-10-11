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
