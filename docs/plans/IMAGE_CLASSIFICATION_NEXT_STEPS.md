# Kế hoạch tiếp tục phân loại từng ảnh gốc

Cập nhật: **16/10/2026**. Kế hoạch thực thi hiện hành là **[R0–R3 trên cache RetCCL512](IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md)**.

## Công việc kế tiếp

1. **P1 — Phát triển và kiểm thử:** tái sử dụng study runner, bổ sung shared inner assignments cho R1, Normalizer cho R2, ElasticNet và solver control cho R3; phiên bản hóa config/loader/checkpoint. Qua conveyor và CI trước dữ liệu thật.
2. **P2–P3 — Một phiên Colab CPU:** gắn Drive, verify cache B1 và R0, preflight từng vật kính rồi benchmark L-BFGS/SAGA. Không tạo T4 hoặc trích lại features.
3. **P4–P6 — Head studies:** R1 routing, R2 normalization, SAGA L2 control, rồi R3 sparsity; cùng nhóm người bệnh và patient-class weights. Lưu/commit từng đơn vị để resume.
4. **P7–P8 — Đánh giá và bàn giao:** OOF đầy đủ, so sánh theo người/vật kính, paired CI và gate ưu tiên; refit phương án cần giữ, verify model/backup, flush và dừng CPU.

| Nhánh | Nội dung |
|---|---|
| R0 | B1 hiện có, bất biến; không train lại |
| R1 | RetCCL512 + ba head riêng theo vật kính |
| R2 | RetCCL512 + Normalizer → scaler → global L2 LR |
| R3 | Cùng preprocessing R2 + ElasticNet; có SAGA L2 control riêng |

Nguồn: 2.802 ảnh gốc, 18 người; nhãn toàn ảnh Glade0→0, Glade3/4/5→1. B1 đã đủ features và models; AUROC OOF 0,870, AP 0,624. R1–R3 chưa triển khai/chạy theo protocol mới. Kết quả trên 18 người là nghiên cứu nội bộ, chưa phải kiểm định độc lập.

Lượt cập nhật này **chỉ thay kế hoạch trong dự án**, không tạo phiên hoặc chạy training. RetCCL1024, màu nhuộm, multiview, encoder mới và fine-tune là các nhánh sau, không tự kích hoạt.

## Lịch sử

[Bản kế hoạch ngày 13/10 và snapshot nghiệm thu cũ](IMAGE_CLASSIFICATION_NEXT_STEPS_20261013_ARCHIVE.md) được giữ làm lịch sử. Các câu “chưa train”, “2.570/2.802 vector” và đề xuất T4 trong bản đó là trạng thái lịch sử, không phải việc cần chạy lại.

[Mục lục kế hoạch](README.md) · [Báo cáo frozen ngày 15/10](../reports/IMAGE_FROZEN_COMPARISON_20261015.md).
