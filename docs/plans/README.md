# Kế hoạch dự án

Mục tiêu hiện tại: **một ảnh gốc 4×, 10× hoặc 40× → có/không ung thư**, dùng nhãn bác sĩ của từng ảnh. Patient ID dùng chia dữ liệu, không làm mục tiêu chẩn đoán ca. MIL cấp ca và clustering patch là lịch sử tham khảo.

## Kế hoạch hiện hành — 21/10/2026

**[UNI_FINAL_PREPARATION.md](UNI_FINAL_PREPARATION.md)** là kế hoạch chốt mới: frozenUNI + global/mean/max views + L2LR, phân loại từng ảnh gốc. **Hiện chỉ chuẩn bị phần mềm và kiểm thử synthetic; chưa train dữ liệu thật, chưa có quyền/weightsUNI hoặcGPU smoke đã xác minh.**

[Hướng dẫn vận hành](../UNI_RUN_GUIDE.md). Không tự tạoColab, trích lạiB1 hay chạyLoRA. R0–R3 dưới đây là kết quả hoàn tất và nguồn đối chứng, không phải hàng đợi cần chạy lại.

## Lượt đã hoàn tất — 16/10/2026

**[IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md](IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md)** là nguồn điều phối duy nhất cho lượt R0–R3:

| Nhánh | Công việc |
|---|---|
| R0 | Giữ B1 RetCCL512 + global L2 LR làm đối chứng bất biến |
| R1 | Cùng cache, head riêng theo vật kính |
| R2 | Sample L2 Normalizer → StandardScaler → global L2 LR |
| R3 | Cùng preprocessing R2 + ElasticNet, có certified full-batch L2 solver control |

Kế hoạch chốt nguồn/identities, shared patient assignments, weights và grids, 186 nested fits trước refit, checkpoint/resume, kiểm thử, báo cáo, gate và ranh giới external test. **Đã hoàn tất186 fit và27controls trên Colab CPU; giữR0 theo gate.** [Báo cáo kết quả](../reports/R0_R3_RESULTS_20261016.md), Drive/local handoff đã verify. Việc tiếp theo là thiết kế kiểm định ngoài; không tự mở rộng grid trên cùng OOF.

- [Đường dẫn kế hoạch chính](IMAGE_CLASSIFICATION_NEXT_STEPS.md) và [hướng nghiên cứu](IMAGE_CLASSIFICATION_FEASIBLE_DIRECTIONS.md) đã chuyển sang R0–R3.
- [Báo cáo frozen ngày 15/10](../reports/IMAGE_FROZEN_COMPARISON_20261015.md): 2.802 ảnh/18 người; ImageNet512 AUROC 0,729, ImageNet1024 0,787, RetCCL512 0,870. Cache/model/OOF đã lưu bền vững, T4 đã dừng; đây là kết quả nội bộ.
- Phân tích overfitting/test độc lập và phản biện khuyến nghị được giữ local ở Results/image_classification/research_validation_20261016/.

## Kế hoạch phân loại ảnh đã được thay thế

- [Snapshot tiếp tục ngày 13/10](IMAGE_CLASSIFICATION_NEXT_STEPS_20261013_ARCHIVE.md): giữ thiết kế và các trạng thái ở thời điểm viết; không phải hàng đợi chạy hiện tại.
- [Hai hướng nghiên cứu ngày 14/10](IMAGE_CLASSIFICATION_FEASIBLE_DIRECTIONS_20261014_ARCHIVE.md): lịch sử N0–N3; hai nhánh frozen đã hoàn tất. Không tự chạy fine-tune hoặc thử seed khác theo bản này.
- [Lần thử nhãn mạnh đầu tiên](IMAGE_STRONG_LABEL_COLAB_EXECUTION.md): lịch sử ngày12/10; không lấy tên runtime/mount snapshot làm trạng thái hiện tại.

## Lịch sử và tài liệu tham khảo của mục tiêu cũ

- [CLUSTERING_DUAL_TRACK_EXECUTION.md](CLUSTERING_DUAL_TRACK_EXECUTION.md): atlas/histogram và negative-reference novelty/MIL; không thuộc luồng R0–R3.
- [CLUSTERING_WEAK_LABEL_STRATEGY.md](CLUSTERING_WEAK_LABEL_STRATEGY.md): nền lý luận nhãn yếu và truy hồi patch; không thay nhãn toàn ảnh đã chốt.
- [MIL_TRAINING_PLAN.md](MIL_TRAINING_PLAN.md): trainer E0–E4 và đánh giá cấp ca, giữ làm tham khảo.
- [COLAB_TRAINING_RELIABILITY.md](COLAB_TRAINING_RELIABILITY.md): rủi ro ngắt phiên, checkpoint và thử lỗi; áp dụng nguyên tắc phù hợp, không tự chạy lại MIL.
- [MIL_MODEL_EXPLAINED.md](MIL_MODEL_EXPLAINED.md): nguyên lý mô hình cấp ca cũ.
- [PRETRAIN_FINALIZATION.md](PRETRAIN_FINALIZATION.md), [PRETRAIN_EXECUTION.md](PRETRAIN_EXECUTION.md): lịch sử chuẩn bị patch/bags.
- [MIL_MULTISCALE_REFERENCE.md](MIL_MULTISCALE_REFERENCE.md), [MIL_T4_OPTIMIZATION_PLAN.md](MIL_T4_OPTIMIZATION_PLAN.md): các kế hoạch tham chiếu trước khi chuyển mục tiêu.

Giữ dữ liệu/nhãn/bằng chứng lịch sử, không xóa hoặc ghi đè để cập nhật kế hoạch. Không có lịch tự tắt máy Na trong công việc hiện hành.
