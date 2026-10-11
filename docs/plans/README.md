# Kế hoạch dự án

Thư mục này lưu kế hoạch tham chiếu, lộ trình chuẩn bị dữ liệu và huấn luyện MIL. Các tài liệu phân biệt dữ kiện đã xác nhận, tác vụ đang chạy và đề xuất chưa thực hiện.

## Kế hoạch hiện tại

- [Hai nhánh song song A/B](CLUSTERING_DUAL_TRACK_EXECUTION.md): atlas/histogram và negative-reference novelty/MIL; [runbook](../DUAL_TRACK_RUN.md). Source đang được phát triển/kiểm thử, full study mới chưa được báo nghiệm thu.

- **Thực thi tiếp — kế hoạch train:** [`MIL_TRAINING_PLAN.md`](MIL_TRAINING_PLAN.md) — v1.1, thiết kế E0–E4, nested CV 3×2, các bước T0–T7, local→Colab, dry-run/checkpoint, benchmark T4, lưu trữ/resume, đánh giá và checklist. Đã lưu kế hoạch; chưa chạy train.
- **Phụ lục độ tin cậy:** [`COLAB_TRAINING_RELIABILITY.md`](COLAB_TRAINING_RELIABILITY.md) — train loop nhẹ local, L0–L3/C0–C3, các rủi ro Colab Free, test lỗi giả lập F01–F12, checkpoint commit và giới hạn tài nguyên.
- **Phụ lục nguyên lý:** [`MIL_MODEL_EXPLAINED.md`](MIL_MODEL_EXPLAINED.md) — giải thích CNN frozen, projection, gated attention, fusion, loss/gradient, custom và bài báo; [bản HTML độc lập](../../.lavish/MIL_MODEL_EXPLAINED.html).
- **Đã nghiệm thu trước train:** [`PRETRAIN_FINALIZATION.md`](PRETRAIN_FINALIZATION.md) — CPU, duplicate review, hai cohort, bags/splits, verify và bàn giao.

- [`MIL_MULTISCALE_REFERENCE.md`](MIL_MULTISCALE_REFERENCE.md): kế hoạch phân loại mô học tuyến tiền liệt bằng MIL ở 4×, 10×, 40×; gồm dữ liệu, encoder pretrained/frozen, features, bags, protocol và đánh giá dự kiến.
- [`PRETRAIN_EXECUTION.md`](PRETRAIN_EXECUTION.md): lộ trình G0–G5 lịch sử v1.1; trạng thái tại thời điểm trích xuất chưa xong. Đọc kế hoạch tiếp tục phía trên cho việc còn lại.

Bằng chứng kiểm kê và hình minh họa được lưu trong `Results/planning_20261010/`.

## Dữ liệu patch và extractor

- [Hướng dẫn import](../PRECUT_IMPORT.md) và [Notebook import](../../notebooks/Colab_PreCut_Import.ipynb): inventory, smoke, import và audit patch; bản toolkit hiện đi cùng nhánh training, chưa merge vào main.
- [Hướng dẫn extractor](../../dist/HUONG_DAN_CHAY.md) và [Notebook features bàn giao](../../dist/Colab_Feature_Extraction.ipynb): ResNet50 ImageNet V1 frozen chạy theo ZIP hoặc thư mục PNG.

Snapshot nghiệm thu 11/10/2026: nguồn 25 ZIP/148.991 vector giữ nguyên; governance 18 người, nhãn 10/8; duplicate review đã xong. Hai bundle common16/all18 đã build/verify `data_ready` trên CPU thiha; loader 34/34 lượt ca qua. Xem [báo cáo hoàn tất](../../Results/planning_20261010/pretrain/final_verified_20261011/BAO_CAO_HOAN_TAT.md). Chưa train MIL; tự tắt máy vẫn đã hủy.

- [MIL_T4_OPTIMIZATION_PLAN.md](MIL_T4_OPTIMIZATION_PLAN.md): nghiên cứu cache GPU/AMP/nén head, benchmark và cổng đánh giá; chưa triển khai vào baseline.
