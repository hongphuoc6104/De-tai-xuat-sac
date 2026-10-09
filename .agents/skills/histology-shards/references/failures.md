# E2E failures và bài học về histology shards

### [ERR-20261009-03] Tập smoke đọc toàn bộ ảnh nguồn và bỏ sót lớp dương
- **Triệu chứng (Symptom):** Public catalog API với sáu TIFF thật và `per_lens=2` đọc checksum cả sáu ảnh, đồng thời chọn hai ca lành dù nguồn có ca ung thư.
- **Tái hiện E2E (Reproduction Path):** `pytest tests/test_data_catalog.py::test_smoke_alternates_available_case_labels tests/test_data_catalog.py::test_smoke_hashes_only_selected_files`; trước sửa, kết quả là labels `[0,0]` và sáu lượt hash TIFF cho hai ảnh được chọn. Fixture chạy inventory → metadata → selection trên file ảnh/CSV thật.
- **Nguyên nhân cốt lõi (Root Cause):** Inventory directory hash ảnh trước selection; selector đi qua toàn bộ ca của lớp 0 trước lớp 1.
- **Giả định sai (Wrong Assumptions):** Giả định giới hạn sau inventory cũng giới hạn I/O; round-robin theo ca được coi là cân bằng giữa lớp.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Inventory chỉ liệt kê path/size, hash sau khi chọn; xen kẽ lớp và ca. Cả tám kiểm thử catalog pass, chỉ hai TIFF được hash và nhãn `{0,1}` có mặt. Smoke luôn `training_ready=false` dù có identity map và nhãn đã review.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Áp dụng giới hạn kỹ thuật trước đọc/decode/hash ảnh; smoke phải đo coverage của ca/lớp và không được báo là release khoa học đầy đủ.

### [ERR-20261009-04] Colab Adapter Rejected Mixed Directory and ZIP Sources
- **Triệu chứng (Symptom):** Notebook stopped when a 4X source directory and 10X/40X ZIP archives were both configured, although the CLI and catalog support multiple source paths.
- **Tái hiện E2E (Reproduction Path):** Execute the marked source-selection block in `notebooks/Colab_Data_Shards.ipynb` against temporary 4X directory plus 10X and 40X ZIP fixtures. Before the fix, it raised `Choose a source directory or source ZIPs for this run; do not mix both.` The regression path is `pytest -q tests/test_data_shards_notebook.py`.
- **Nguyên nhân cốt lõi (Root Cause):** The notebook adapter imposed a single-source-format rule that did not exist in the CLI or catalog contracts.
- **Giả định sai (Wrong Assumptions):** All magnifications would be stored using the same source packaging format.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** The adapter now appends an optional directory and all sorted ZIP archives, then passes them together to catalog preparation. A notebook-source E2E test confirms mixed inputs work and that duplicate basenames are still rejected by the catalog.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Keep notebook source selection aligned with the CLI's repeated `--source` contract; let the catalog enforce cross-source filename integrity.


### [ERR-20261009-05] Stage một phần lại quét mọi TAR và trùng cache giữa catalog
- **Triệu chứng (Symptom):** Stage shard đầu thất bại vì TAR cuối không liên quan bị hỏng; mỗi shard đọc lại toàn release. Hai release dùng chung work-root bị đụng staged/shard-000001.
- **Tái hiện E2E (Reproduction Path):** Sáu TIFF 32×32 đóng thành ba shard 10 KiB; sửa byte cuối TAR thứ ba rồi stage shard đầu. Chạy tests/test_data_shards_storage.py; trước sửa first shard bị chặn bởi checksum shard-000003.
- **Nguyên nhân cốt lõi (Root Cause):** Stage gọi deep verifier toàn release; cache path chỉ có shard ID.
- **Giả định sai (Wrong Assumptions):** Kiểm tra toàn bộ raw ở mọi thao tác được coi là an toàn mà không ảnh hưởng độc lập/I/O; shard ID được coi là định danh toàn cục.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Tách loader metadata và targeted TAR verification; explicit verify_release vẫn quét toàn bộ. Stage cache gồm catalog_id/shard_id. 19 tests storage pass, instrumentation xác nhận chỉ TAR mục tiêu được đọc khi stage.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Scope kiểm tra byte theo đơn vị xử lý; kiểm tra toàn cohort là thao tác riêng. Namespace cache theo phiên bản dữ liệu.

### [ERR-20261009-06] Nội dung ảnh trùng làm mất định danh tile instance
- **Triệu chứng (Symptom):** Hai trường nhìn có bytes giống nhau nhưng image_id khác tạo tile_id giống nhau, làm TAR patch trùng member và fail commit.
- **Tái hiện E2E (Reproduction Path):** Hai TIFF 32×32 cùng nội dung, image_id riêng → build_catalog → pack_catalog → process_shard(tile_size=16,stride=16); trước sửa báo Generated patch archive has an invalid member. tests/test_data_shards_processing.py có regression file thật.
- **Nguyên nhân cốt lõi (Root Cause):** Dùng content fingerprint làm instance identity của ảnh/tile.
- **Giả định sai (Wrong Assumptions):** Hai ảnh byte-identical sẽ không xuất hiện hoặc có cùng provenance.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Tile ID có image_id; content_tile_id giữ identity nội dung. Duplicate source được flag review và giữ cả lineage. 13 tests processing pass, bao gồm duplicate, coordinate/padding và identity ổn định qua output paths.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Tách instance identity với content identity; duplicate review không được biến thành bỏ ảnh im lặng.

### [ERR-20261009-07] Resume vẫn stage raw và cache không có algorithm version
- **Triệu chứng (Symptom):** Verified resume và retry output hỏng đều stage lại raw; thay thuật toán nhưng config giữ nguyên có thể reuse output cũ. Staging tích lũy trên SSD.
- **Tái hiện E2E (Reproduction Path):** process_shard lần đầu, lần reuse, rồi sau corrupt output; đếm stage call thật. Trước sửa đếm ba lần. Regression trong tests/test_data_shards_processing.py.
- **Nguyên nhân cốt lõi (Root Cause):** Stage trước kiểm tra processed commit; fingerprint chỉ có tham số; cache raw giữ vô hạn.
- **Giả định sai (Wrong Assumptions):** Mọi resume cần raw; tham số đủ mô tả thuật toán; giữ cache tất cả shard luôn có lợi.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Load manifest có checksum trước, verify output rồi reuse không stage. PROCESSING_VERSION nằm trong fingerprint/commit. Default cleanup staging sau publication, keep_staged chỉ là runtime option. 13 processing tests pass; verified resume không tăng stage call.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Reuse theo artifact đã committed và implementation version; giữ footprint của working cache theo phần đang xử lý.

### [ERR-20261009-08] Training gate bỏ qua raw trùng giữa người bệnh
- **Triệu chứng (Symptom):** Hai TIFF cùng SHA thuộc p1/p2, nhãn và mapping đã review, vẫn được catalog báo training_ready=true.
- **Tái hiện E2E (Reproduction Path):** Hai TIFF 16×16 giống bytes và identity CSV có p1/p2; build_catalog(labels_reviewed=True) trả true trước sửa. pytest tests/test_data_catalog.py::test_duplicate_content_cannot_pass_patient_training_gate.
- **Nguyên nhân cốt lõi (Root Cause):** Gate chỉ xem coverage, mapping và nhãn, không xét identity nội dung. Config string false còn có thể bị bool coercion thành true.
- **Giả định sai (Wrong Assumptions):** Hai patient_id riêng đồng nghĩa dữ liệu độc lập; truthiness tương đương xác nhận review.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Catalog ghi duplicate_source_groups/training_blockers, duplicate raw không training-ready; CRC ZIP chưa là SHA toàn cohort cũng pending import audit. Labels_reviewed bắt buộc boolean. 10 catalog tests pass; CLI regression từ chối string false.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Kiểm tra content leakage trước scientific training; approval fields phải có kiểu tường minh và bằng chứng đúng cấp nhãn.

### [ERR-20261009-09] Lint toàn repo phát hiện import notebook bị bỏ sót
- **Triệu chứng (Symptom):** Kiểm tra Python module riêng pass nhưng Trạm 2 thất bại ở test Trạm 4 vì Ruff báo hai I001 trong cell notebook.
- **Tái hiện E2E (Reproduction Path):** `./pipeline station 2`; log `test_evidence_20261009_161806.log` ghi 83 tests pass, một test lint fail và hai import-order errors ở Colab_Data_Shards.ipynb.
- **Nguyên nhân cốt lõi (Root Cause):** Lint của subtask chỉ quét file Python, trong khi Ruff toàn repo cũng hỗ trợ notebook.
- **Giả định sai (Wrong Assumptions):** Cell compile được đồng nghĩa notebook qua static checks của dự án.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Sắp import và khoảng trắng theo Ruff trên chính notebook; chạy lại `ruff check .` và toàn bộ conveyor station 2 sau sửa.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Kiểm tra phạm vi lint thực của dự án, gồm notebook; parse/compile là một lớp kiểm chứng riêng.

### [ERR-20261009-10] Saturation mask nhận nền xám ám màu là mô
- **Triệu chứng (Symptom):** Preview thật giữ patch nền 4×; ảnh nền thuần RGB(140,150,135) tạo bốn tissue tiles và không báo review.
- **Tái hiện E2E (Reproduction Path):** TIFF/PNG nền 64×64 → build_catalog → pack_catalog → process_shard(tile_size=32,stride=32,min_tissue=0.2); trước sửa tiles_written=4. Thêm ảnh nền cùng ROI tím để chứng minh cả giữ mô lẫn loại nền; regression trong tests/test_data_shards_processing.py.
- **Nguyên nhân cốt lõi (Root Cause):** Saturation >20 và độ sáng thấp được coi là mô, bỏ qua ánh sáng/nền kính hiển vi và đặc điểm stain.
- **Giả định sai (Wrong Assumptions):** Nền luôn trắng/trung tính; pass test chức năng là đủ cho QC ngữ nghĩa.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Bounded background sampling, background-normalized OD và H&E green-absorbance contrast; thresholds/config/version lưu rõ, raw patch pixels giữ nguyên. cpu-tiles-v3 làm cache cũ không bị reuse. 19 processing tests pass; nền không tạo tile, ROI tím đúng tọa độ và đúng pixels.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Xem preview thật trước modeling; calibrate/review mask theo stain và môi trường chụp, không báo threshold heuristic là chuẩn lâm sàng.
