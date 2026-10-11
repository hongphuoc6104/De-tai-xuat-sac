---
name: histology-mil
description: Phát triển, kiểm thử và chạy MIL cấp ca từ frozen feature bundles đã verify; dùng cho model E0–E4, nested folds, checkpoint/resume và Colab smoke.
---

# Histology MIL

Đọc [kế hoạch train](../../../docs/plans/MIL_TRAINING_PLAN.md) và [reliability gates](../../../docs/plans/COLAB_TRAINING_RELIABILITY.md) cho run hiện tại. Đọc [lỗi đã tái hiện](references/failures.md) khi chỉnh trainer, tích hợp bundle hoặc recovery; không coi các rủi ro dự đoán là lỗi đã sửa.

- Nhãn supervised ở cấp ca. Encoder ImageNet V1/cache cố định; không có nhãn patch hoặc gradient CNN trong baseline. Mọi split theo người; outer test không dùng điều chỉnh kiến trúc/epoch/threshold.
- Bundle manifest là `bundle.json`. Dùng `BundleDataset`/public verified loader, không bypass checksum hoặc tự truncate production bags. Exported fixtures có mode smoke và không đủ điều kiện production.
- Run local chỉ có fixture/cap nhỏ; dùng cùng trainer cho Colab, pin môi trường và identity. Không reuse smoke weights hoặc head của fold khác. Không đòi CPU/GPU training bit-identical.
- Checkpoint phiên bản bất biến với commit receipt; chỉ committed epoch được coi là tiến độ. Pointer là index có thể lỗi sau commit. Resume từ verified state, không từ `status.json` đơn lẻ. Config khác phải fail trước khi ghi đè config run cũ.
- Active lock không được lấy lại theo TTL. Recovery khác host cần receipt gắn chính xác lock và bằng chứng runtime cũ đã dừng. Không tạo VM thay thế, keep-alive giả hoặc đổi account để vượt giới hạn tài nguyên.
- Deadline giữa validation phải trả paused, không publish epoch dở. Lỗi I/O/integrity vẫn báo rõ và giữ dữ liệu. `complete` chỉ ứng với toàn fit/study đã verify, không phải hết budget.
- Code/CLI: `histology_data/mil_*.py`, `scripts/run_mil.py`; test: `tests/test_mil_*.py`. New runtime/notebook phải pin ZIP SHA; hoàn tất conveyor/CI trước dùng full study.
