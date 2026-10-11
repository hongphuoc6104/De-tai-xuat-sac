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
