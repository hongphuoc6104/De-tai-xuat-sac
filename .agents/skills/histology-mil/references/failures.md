# Lỗi MIL đã tái hiện và kiểm chứng — 11/10/2026

## [ERR-20261011-07] Study đọc sai tên manifest của bundle đã nghiệm thu
- **Symptom:** Nested study dừng FileNotFoundError khi bundle đã verify thành công.
- **Reproduction Path:** `tests/test_mil_study.py` tạo 6 người/12 ca/36 vector tổng hợp qua feature→governance→common bundle rồi gọi `run_study`; before trace tại `Results/proofs/mil_study_manifest_repro_20261011.log`.
- **Root Cause:** Orchestrator mới giả định `manifest.json`, trong khi contract artifact của builder/loader là `bundle.json`.
- **Wrong Assumptions:** Tên generic “manifest” trong tài liệu được coi là tên file thực.
- **Resolution & Proof:** Đọc/fingerprint cùng `bundle.json` với verified loader; actual E2E chạy 9 fits, 12 OOF, cùng patient folds; rerun không phát sinh fits/predictions mới.
- **Golden Prevention Rule:** Tích hợp bằng artifact schema của producer, kiểm bằng governed bundle thực trong E2E, không dùng file fixture khác contract production.

## [ERR-20261011-08] Tensor CPU dùng trực tiếp storage feature chỉ đọc
- **Symptom:** Study E2E phát PyTorch warning do NumPy memmap không writable.
- **Reproduction Path:** `tests/test_mil_study.py` qua public case loader; regression `tests/test_mil_training_faults.py` kiểm CaseData readonly.
- **Root Cause:** `from_numpy` chia sẻ storage; `.to(cpu)` có thể không copy, tạo tensor với nền không writable.
- **Wrong Assumptions:** Đổi device luôn tách tensor khỏi memmap nguồn.
- **Resolution & Proof:** Copy writable đúng một ca trước tensor conversion; warning không xuất hiện, features nguồn giữ nguyên, smoke/full synthetic study qua.
- **Golden Prevention Rule:** Bảo vệ nguồn read-only tại ranh giới NumPy→Torch, hạch toán copy trong memory cap; không bật writable hoặc sửa cache gốc.

## [ERR-20261011-09] Resume mismatch ghi đè cấu hình trước khi bị từ chối
- **Symptom:** Đổi seed bị checkpoint identity gate chặn nhưng resolved config cũ đã bị thay.
- **Reproduction Path:** `tests/test_mil_training_faults.py` chạy fit thật rồi resume khác seed; initial failure log `Results/proofs/mil_training_faults_repro_20261011.log`.
- **Root Cause:** Trainer ghi resolved config trước khi xác minh danh tính run đã tồn tại.
- **Wrong Assumptions:** Checkpoint gate cuối đủ bảo vệ mọi artifact run.
- **Resolution & Proof:** So identity/config trước mọi config write; mismatch giữ nguyên config và checkpoint; regression qua.
- **Golden Prevention Rule:** Validate identity trước mutation; việc load thất bại không được làm mất bằng chứng run tốt trước đó.

## [ERR-20261011-10] Deadline hết trong validation bị báo thành lỗi training
- **Symptom:** Validation timeout sau train step cuối ném TimeoutError thay vì paused_budget.
- **Reproduction Path:** `tests/test_mil_training_faults.py` dùng deadline thật và bounded sleep trong test hook; before/after log cùng fault proof.
- **Root Cause:** Budget check chỉ xử lý trong train loop; evaluation có deadline nhưng generic exception path báo error.
- **Wrong Assumptions:** Xong train steps nghĩa là còn đủ thời gian validation/checkpoint.
- **Resolution & Proof:** Deadline-caused evaluation timeout phục hồi committed state và trả paused_budget; epoch dở không commit; regression qua.
- **Golden Prevention Rule:** Deadline phải bao trùm train/validation/publish; resume theo committed epoch, không theo số step đã chạy trong RAM.

## [ERR-20261011-11] Pointer lỗi sau commit làm báo thiếu epoch bền
- **Symptom:** Có commits epoch 0/1 nhưng error status báo last_committed_epoch 0.
- **Reproduction Path:** `tests/test_mil_training_faults.py` tạo pointer obstruction sau commit trong filesystem sandbox; before/after fault proof.
- **Root Cause:** publish payload/receipt thành công nhưng reconcile pointer lỗi; trainer giữ state trước publish làm nguồn báo cáo.
- **Wrong Assumptions:** publish raise nghĩa là mọi bước transaction chưa commit.
- **Resolution & Proof:** `load_latest(reconcile_pointers=False)` vẫn verify tất cả commits/payload và đọc tiến độ bền mà không ghi pointer; báo error rõ nhưng epoch 1 đúng; không xóa obstruction tự động.
- **Golden Prevention Rule:** Receipt/bytes là nguồn tiến độ; pointer/status là index. Phân biệt lỗi trước commit và sau commit, không che corruption bằng fallback âm thầm.

## [ERR-20261011-12] Attention tổng bằng một nhưng có giá trị âm vẫn được publish
- **Symptom:** Publisher nhận [-0,25; 1,25] và commit dù đó không là trọng số softmax hợp lệ.
- **Reproduction Path:** `tests/test_mil_outputs.py::test_publisher_rejects_negative_attention_even_when_sum_is_one`; fixture source refs và dtype/order hợp lệ, trước sửa pytest báo DID NOT RAISE.
- **Root Cause:** Output boundary kiểm finite/length/sum nhưng thiếu bất đẳng thức không âm.
- **Wrong Assumptions:** Tổng một và source model softmax đủ bảo vệ mọi caller/artifact attention.
- **Resolution & Proof:** Chặn phần tử âm trước ghi weights; chạy lại cùng E2E cùng real governed artifact tests và callback-failure study.
- **Golden Prevention Rule:** Artifact boundary phải kiểm đủ miền giá trị, không chỉ shape và tổng; không coi output ngoài model là đáng tin mặc định.

## [ERR-20261011-13] Cache loader chưa khởi tạo hợp đồng số lượng nguồn
- **Symptom:** CacheDataset.load_case ném AttributeError ở `_source_instance_counts`.
- **Reproduction Path:** `tests/test_mil_case_cache_safety.py`; trace trước sửa tại `Results/proofs/mil_case_cache_safety_20261011.log`.
- **Root Cause:** Constructor xác minh cache nhưng chưa lưu bảng số lượng ca/vật kính cho load boundary.
- **Wrong Assumptions:** Kiểm commit cache thay được contract count của nguồn đã verify.
- **Resolution & Proof:** Khởi tạo bảng count từ bundle đã verify; chặn bags bị cắt ngắn dù đã tính lại checksum. 14 cache tests và full conveyor 327 passed/1 skipped.
- **Golden Prevention Rule:** Mọi invariant được kiểm tại load boundary phải khởi tạo từ nguồn độc lập ngay ở constructor.

## [ERR-20261011-14] Consumer verifier ghi audit vào dataset nguồn
- **Symptom:** E2E cached/uncached study làm thay đổi file audit nguồn dù training chỉ đọc dữ liệu.
- **Reproduction Path:** `tests/test_mil_cached_study.py::test_cached_study_matches_uncached_weights_oof_and_verified_callback_order`; assertion tree hashes trước/sau consumer.
- **Root Cause:** Public pretrain verifier có trách nhiệm xuất audit; study gọi trực tiếp trên source tree.
- **Wrong Assumptions:** Verify là thao tác read-only ở mọi API.
- **Resolution & Proof:** Verify đầy đủ trên mirror tạm bằng `verify_source_readonly`; giữ source tree hashes nguyên vẹn. Cached và uncached study cho cùng weights, OOF, tile order; toàn bộ 327 tests pass.
- **Golden Prevention Rule:** Phân biệt verifier có side effects với consumer chỉ đọc; kiểm immutability trên E2E, không chỉ array contents.

## [ERR-20261011-15] Báo cáo phân cụm mất định danh cấu hình và truy hồi điểm mẫu
- **Symptom:** Kết quả v001 thiếu algorithm/config trong từng metric; v002 có điểm biểu đồ nhưng chưa có khóa nối từng điểm đến tile nguồn.
- **Reproduction Path:** `Results/training_preflight/clustering_comparison_full/metric_schema_reproduction.json` và `sample_trace_reproduction.json`, kiểm trực tiếp artifact full Colab trước mỗi sửa.
- **Root Cause:** Serializer chỉ xuất số đo và tọa độ, chưa coi định danh cấu hình và lineage là phần bắt buộc của output contract.
- **Wrong Assumptions:** Thứ tự danh sách đủ để xác định cấu hình; prototype refs đủ để truy hồi mọi điểm.
- **Resolution & Proof:** Mỗi metric có algorithm/config_id/config; sample assignments có sample_row_index, sidecar nối (lens,index) đến tile/part/row. Full v003: 36 candidates, 109.248 assignments nối 9.104 provenance rows; 48 shards đủ 146.506 dòng, hashes/CRC và kết quả v002 khớp. Proof: `Results/training_preflight/clustering_comparison_full/verification_full_v003.json`; synthetic E2E: `clustering_design/synthetic_schema_smoke.json`.
- **Golden Prevention Rule:** Báo cáo thực nghiệm phải tự định danh cấu hình và truy hồi từng observation; kiểm join một-một và hash sau serialization, không chỉ trong RAM.

## [ERR-20261011-16] Tên ảnh cluster noise va chạm với cluster dương
- **Symptom:** Slug('-1') và slug('1') cùng thành '1', có thể ghi đè montage noise bằng montage cluster 1.
- **Reproduction Path:** `Results/training_preflight/clustering_comparison_full/RETRIEVAL_AUDIT.md`, tái hiện file output va chạm trước sửa trong helper phân tích riêng.
- **Root Cause:** Chuẩn hóa filename bỏ dấu âm, không kiểm tính duy nhất theo khóa group.
- **Wrong Assumptions:** Slug của khóa group là ánh xạ một-một.
- **Resolution & Proof:** Thêm digest theo nguyên group, chặn collision và hàng khác group. Hai E2E regression pass; ruff/compile pass. 30 ảnh đã bàn giao không có group noise nên không bị ảnh hưởng.
- **Golden Prevention Rule:** Tên file chuẩn hóa phải kèm định danh không va chạm và guard trước ghi; không dùng slug làm khóa dữ liệu.

## [ERR-20261011-17] Trình vẽ không nhận trạng thái noeligible
- **Symptom:** Vẽ so sánh full v003 ném StopIteration khi DBSCAN 4x không có cấu hình đạt điều kiện.
- **Reproduction Path:** `Results/training_preflight/clustering_comparison_full/chart_reproduction.json`; chạy render_report_charts.py với metrics full v003 trước sửa.
- **Root Cause:** Consumer coi selection luôn là config_id hoặc null; producer dùng sentinel noeligible có nghĩa rõ ràng.
- **Wrong Assumptions:** Mọi thuật toán đều có cấu hình được chọn.
- **Resolution & Proof:** Render trạng thái noeligible thành ô giải thích, không tìm metric giả. Chạy lại cùng full metrics tạo đủ ba PNG/SVG, kiểm hình trực tiếp.
- **Golden Prevention Rule:** Consumer phải xử lý mọi trạng thái hợp lệ của output contract; không biến không đủ điều kiện thành lỗi hoặc kết quả giả.
