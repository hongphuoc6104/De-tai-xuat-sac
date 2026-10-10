# Các lỗi đã tái hiện khi phát triển pretrain

## [ERR-20261010-09] Chấm điểm fold riêng làm nhóm bệnh nhân có hai nhãn dồn vào một fold
- **Symptom:** Cohort hợp lệ12người, mỗi người có hai ca nhãn0/1, bị báo fold rỗng.
- **Reproduction Path:** Direct public split API với24case records vàseed42; traceback và script ở `Results/proofs/feature_pretrain_split_repro_20261010.log`. Đây là repro luồng split, không phải full bundle trước sửa.
- **Root Cause:** Greedy score đo khoảng cách của riêng candidate fold; các nhóm mixed-label bằng nhau tie và chọnfold0 lặp lại.
- **Wrong Assumptions:** Mục tiêu gần target ở mỗi fold tương đương cân bằng toàn partition; bệnh nhân luôn chỉ có một nhãn.
- **Resolution & Proof:** Dùng `StratifiedGroupKFold` trên từng ca với group patient/duplicate component, giữ seed và toàn dữ liệu. Full synthetic feature→governance→common/all→vector round-trip và CLI draft→build→verify qua; `Results/proofs/feature_pretrain_bundle_after_20261010.log`.
- **Golden Prevention Rule:** Split có group và mixed-class phải được kiểm trên toàn partition; không đổi nhãn người, bỏ ca hoặc reseed dựa trên test performance.

## [ERR-20261010-11] Loader bag không tiêu thụ đúng schema và phạm vi stream đã commit
- **Symptom:** Full round-trip dừng ở KeyError case_count, sau sửa schema tiếp tục dừng ở stream.tell trên file đã đóng.
- **Reproduction Path:** `tests/test_pretrain.py::test_full_feature_to_common_and_all_bundle_e2e_roundtrip`; dataset synthetic12người/24ca, cache2048-D và hai bundle.
- **Root Cause:** Reader dùng case_count thay cho cohort_case_count trong manifest; byte-range postcondition được chạy sau khi context file đóng.
- **Wrong Assumptions:** Count field của toàn governance và count của cohort là một hợp đồng; trạng thái stream còn dùng sau context.
- **Resolution & Proof:** Reader dùng trường cohort_case_count, kiểm byte range khi stream vẫn mở. Full E2E đọc đúng từng vector/row/lens/mask và checksums, qua trong proof log bundle_after.
- **Golden Prevention Rule:** Kiểm cả writer→serialized artifact→loader→vector round-trip; compile/unit array không đủ chứng minh persistence contract.

## [ERR-20261010-12] CLI pretrain import constant từ module không sở hữu nó
- **Symptom:** `python -m histology_data.pretrain_cli --help` chết trước parser vì bags import FEATURE_DIM từ features.
- **Reproduction Path:** CLI help→readiness→bags→features trên checkout mới; root quan sát actual ImportError trước sửa. Full CLI regression nằm trong tests/test_pretrain.py.
- **Root Cause:** features nhận feature_dim qua encoder protocol, không export constant FEATURE_DIM; baseline constant thuộc feature_encoder.
- **Wrong Assumptions:** Mọi module xử lý feature đều sở hữu cùng constant.
- **Resolution & Proof:** Import baseline dimension từ feature_encoder, verifier từ features; help và full CLI draft→build→verify qua, không tải weights cho help.
- **Golden Prevention Rule:** Chạy executable CLI cold từ môi trường thực và import API từ module sở hữu; không dùng stub wrapper thay kiểm integration package.
