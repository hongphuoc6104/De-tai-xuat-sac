# Hướng nghiên cứu hiện hành cho phân loại ảnh

Cập nhật: **16/10/2026**. Thay hàng đợi nghiên cứu ngày 14/10 bằng **[kế hoạch R0–R3](IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md)**.

Ưu tiên hiện tại là thử routing, feature normalization và sparsity trên cache RetCCL512 bằng Colab CPU. Giữ B1 làm mốc bất biến, patient-disjoint assignments và nhãn toàn ảnh. Không trích lại ảnh hoặc mở rộng model-grid trong lượt này.

Hai nhánh ImageNet1024/RetCCL512 của kế hoạch cũ đã hoàn tất ngày 15/10: AUROC gộp lần lượt 0,787 và 0,870. N3 fine-tune có điều kiện chưa được kích hoạt; không coi đây là công việc đang chờ tự chạy. Xem [báo cáo kết quả](../reports/IMAGE_FROZEN_COMPARISON_20261015.md).

Sau R0–R3, RetCCL1024 toàn khung hoặc thí nghiệm màu có đối chứng có thể được thiết kế riêng. External test trên bệnh nhân mới/công khai cần protocol về nhãn, scale và tính độc lập; AUROC nội bộ cao không thay thế bước đó.

[Thiết kế ngày 14/10 được giữ làm lịch sử](IMAGE_CLASSIFICATION_FEASIBLE_DIRECTIONS_20261014_ARCHIVE.md). Seed, grid và công việc kế tiếp phải lấy theo kế hoạch R0–R3; không thực thi gợi ý thử seed khác trong bản lịch sử sau khi xem điểm.

[Kế hoạch điều phối](IMAGE_CLASSIFICATION_NEXT_STEPS.md) · [Mục lục](README.md).
