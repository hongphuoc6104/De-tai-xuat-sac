# Dual-track đã tái hiện và kiểm chứng


### [ERR-20261011-18] PCA không dùng trọng số cân bằng ca/ảnh
- **Triệu chứng (Symptom):** Hai ca có2 và6 rows làm mean PCA thành[0,25;0,75], dù trọng số được tính để mỗi ca đóng góp ngang nhau.
- **Tái hiện E2E (Reproduction Path):** `Results/clustering_dual/execution_20261011/reproductions/pca_weighting.py`; actual `fit_scope_transforms`, before/after receipts.
- **Nguyên nhân cốt lõi (Root Cause):** Reservoir tính weights nhưng PCA sklearn.fit không nhận chúng.
- **Giả định sai (Wrong Assumptions):** Cân bằng cap/sampling thay được weighted covariance khi số rows khả dụng khác nhau.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Weighted mean và sqrt(weight)-scaled randomized SVD; cùng E2E cho mean[0,5;0,5]. Rootcommon5tests và QAweighted checks qua.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm trọng số tác động vào phép fit thực, không chỉ có mặt trong metadata.

### [ERR-20261011-19] Ward multicut chưa bind đúng row order
- **Triệu chứng (Symptom):** Đảo cùng số rows có thể gắn labels cây cũ vào ảnh khác.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_atlas.py::test_ward_tree_rejects_reordered_feature_rows_before_multicut`; pytest.raises fail trước sửa.
- **Nguyên nhân cốt lõi (Root Cause):** WardTree chỉ giữ row count.
- **Giả định sai (Wrong Assumptions):** Cùng độ dài đảm bảo cùng observation/order.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Bind SHA của features/patient/photo arrays; reject reorder trước attach labels; A14tests pass.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Clustering lineage phải bind exact ordered observations.

### [ERR-20261011-20] Direct MIL evaluator cho phép bank-anchor self-query
- **Triệu chứng (Symptom):** evaluate có thể score chính anchor bằng bank của nó dù trainer đã có scope guard.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_mil.py::test_evaluator_rejects_bank_anchor_as_query_case`, actualdirectevaluate failed trước guard.
- **Nguyên nhân cốt lõi (Root Cause):** Public inference không kiểm member ownership của custom model.
- **Giả định sai (Wrong Assumptions):** Training validation đủ bảo vệ mọi inference entrypoint.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Explicit validate_query_scope trước case load trong evaluate; same E2E và B14tests/core78tests qua.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm anchor/scope ở cả training và public evaluation boundary.

### [ERR-20261011-21] Reader artifact lấy khóa writer và producer chạy trước khóa
- **Triệu chứng (Symptom):** Reader của store có identity hợp lệ bị chặn bởi writer khác; hai process cùng chạy producer cho một artifact chưa tồn tại.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_common.py::test_existing_committed_store_reader_does_not_take_writer_lease`; `Results/clustering_dual/execution_20261011/qa/tests/test_final_adversarial_review.py::test_single_artifact_lease_prevents_duplicate_producers` dùng hai process thật.
- **Nguyên nhân cốt lõi (Root Cause):** Constructor lấy write lease dù chỉ đọc; object thực hiện producer trước khi lấy per-name lease.
- **Giả định sai (Wrong Assumptions):** Lock tại thời điểm publish đủ tránh tính hai lần và mọi reader cần writer lease.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Constructor đọc lại identity đã commit mà không khóa writer; object lấy compute lease rồi đọc lại trước producer. QA xác nhận một producer, một payload commit; test reader pass, release suite 379 pass.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Tách read/compute/publish; khóa trước tính toán và kiểm lại commit sau lấy khóa.

### [ERR-20261011-22] Lỗi worker bị che bởi trạng thái pause
- **Triệu chứng (Symptom):** Một child pause và child khác exit lỗi khiến controller báo paused_budget; lỗi preflight có thể để status running.
- **Tái hiện E2E (Reproduction Path):** QA `test_mixed_pause_and_nonzero_child_exit_is_error` dùng child processes thật; `tests/test_dual_cli.py::test_preflight_failure_is_error_not_running` gọi controller với lỗi I/O.
- **Nguyên nhân cốt lõi (Root Cause):** Nhánh pause được ưu tiên trước error, và preflight nằm ngoài handler ghi terminal state.
- **Giả định sai (Wrong Assumptions):** Exit/status của một worker đại diện trạng thái toàn study.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Kiểm nonzero/error trước pause, bao preflight trong error handler; cả hai E2E pass, controller không gọi partial là complete.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Error ưu tiên pause; trạng thái tổng phải xét mọi child và mọi giai đoạn.

### [ERR-20261011-23] Worker bị dừng để lại lock chưa có đường resume an toàn
- **Triệu chứng (Symptom):** Worker chết khi giữ lease làm lần chạy tiếp theo bị khóa.
- **Tái hiện E2E (Reproduction Path):** QA `test_stale_worker_lock_requires_explicit_same_host_recovery` tạo lock từ process đã dừng rồi gọi public acquisition.
- **Nguyên nhân cốt lõi (Root Cause):** Runner giữ lock nhưng không truyền chính sách xác minh recovery từ config tới từng lease.
- **Giả định sai (Wrong Assumptions):** Kết thúc child tự xóa lease bền vững hoặc có thể dùng TTL để giành lại lock.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Opt-in recovery, xác minh PID cùng host đã chết; cross-host cần receipt runtime cũ đã dừng; truyền chính sách đến shared, worker và từng B1 fit. QA xác nhận default reject và proven-dead recovery pass.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Chỉ thu hồi khi chứng minh owner đã dừng, không dùng tuổi lock thay bằng chứng.

## [ERR-20261011-24] Upper bound đếm thừa fit beta khi chọn beta .5
- **Triệu chứng (Symptom):** Orchestration hoàn tất 72 fit nhưng báo cần 90 fit, khiến study không thể hoàn tất dù đã chạy đủ control beta-zero và các outer beta-.5 được chọn.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_neural_study.py::test_production_fit_count_matches_beta_zero_and_selected_beta_outer_fits` chạy `run_neural_study` với fixture 3 outer × 2 inner. Trước sửa, QA tái hiện `completed_fits=72` nhưng `expected_fits=90`; test sau sửa khẳng định exact/max đều là 72.
- **Nguyên nhân cốt lõi (Root Cause):** Công thức cũ nhân số outer beta-zero member với ba, gộp beta-zero control, beta-selected output và số outer folds thành ba nhóm fit riêng dù beta-zero được dùng chung khi fold chọn beta 0.
- **Giả định sai (Wrong Assumptions):** Mỗi fold luôn cần ba nhóm outer refit; lựa chọn beta không làm thay đổi số fit bắt buộc.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Tách upper bound khỏi exact requirement. `expected_fits_max = inner_fits + 2 * outer_beta0_fits`; sau chọn beta, `expected_fits = inner_fits + outer_beta0_fits + selected_outer_beta05_fits`. E2E orchestration xác nhận 36 fit inner + 18 beta-zero outer + 18 beta-.5 outer = 72; focused B1 tests pass.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Đếm fit theo cặp outer-fold/member/beta thực sự được huấn luyện; tính upper bound trước selection và exact requirement sau selection, rồi buộc complete phải khớp cả hai.

## [ERR-20261011-25] Study B1 thứ hai không bị chặn trên cùng GPU
- **Triệu chứng (Symptom):** Hai B1 study dùng output roots/fit paths khác nhau có thể đồng thời chiếm cùng một CUDA device; per-fit locks không va chạm.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_neural_study.py::test_production_fit_count_matches_beta_zero_and_selected_beta_outer_fits` giữ device lease bằng `FitLock(_gpu_lock_path(0))`, rồi gọi public `run_neural_study` với CUDA và output root khác. Trước sửa, B1 không có study-wide device owner; sau sửa, lần gọi thứ hai bị `ActiveFitLockError` trước fit và ghi `phase=error`, `error_category=concurrency`.
- **Nguyên nhân cốt lõi (Root Cause):** Lock chỉ bảo vệ từng fit directory, trong khi GPU là tài nguyên dùng chung giữa toàn bộ study và output roots.
- **Giả định sai (Wrong Assumptions):** Khóa fit riêng đủ để tuần tự hóa nhiều B1 study trên một device.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Giữ một `FitLock` dùng chung theo device tại `/tmp/histology-dual-gpu-<device_index>` trong toàn bộ B1 study; chỉ thu hồi stale lock khi FitLock xác minh PID cùng host đã dừng. E2E lock-contention assertion pass trong test nêu trên.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Khóa tài nguyên ở đúng phạm vi dùng chung của nó; GPU lease phải bao trùm toàn study và không được lấy lại theo TTL.

## [ERR-20261011-26] Cache OOF có checksum hợp lệ nhưng nội dung sai vẫn được xuất
- **Triệu chứng (Symptom):** Một `final_result.json` đã commit đúng ArtifactStore receipt/checksum nhưng chứa case lặp, case thiếu, target/patient sai hoặc metric không khớp vẫn có thể được tin cậy từ cache.
- **Tái hiện E2E (Reproduction Path):** `tests/test_dual_neural_study.py::test_production_fit_count_matches_beta_zero_and_selected_beta_outer_fits` dùng `ArtifactStore.json` để commit từng payload sai với checksum/receipt hợp lệ, sau đó gọi lại `run_neural_study` ở cache path. Các biến thể duplicate case, dropped case, wrong target, wrong patient và wrong metric đều bị từ chối; status ghi `phase=error`, `error_category=integrity` trước khi sao chép kết quả ra output.
- **Nguyên nhân cốt lõi (Root Cause):** Cache consumer chỉ so `study_id`; checksum chứng minh bytes không đổi kể từ lúc commit, không chứng minh OOF phù hợp với split và dataset công khai.
- **Giả định sai (Wrong Assumptions):** Artifact có checksum hợp lệ thì có thể phát metrics/OOF mà không kiểm tra lại ngữ nghĩa.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** `verify_neural_result(result, dataset, splits, identity)` chạy cho kết quả mới và cache trước publish/copy. Verifier kiểm tra nhóm patient và fold, coverage từng case đúng một lần, target, beta và xác suất hữu hạn trong [0,1], fit counts, subset beta-.5 đã chọn; sau đó tính lại hai metric chính theo pooled OOF và từng outer fold, sai số tuyệt đối tối đa `1e-12`. Smoke result bị chặn nếu có OOF/metrics. Năm payload checksum-valid bị từ chối; focused B1 tests pass.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Kiểm chứng checksum và ngữ nghĩa ở mỗi consumer boundary; metrics phải được tính lại từ prediction rows gắn với split/data hiện hành trước khi xuất hoặc resume cache.
