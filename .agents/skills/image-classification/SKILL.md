---
name: image-classification
description: Huấn luyện phân loại từng ảnh mô học gốc có nhãn bác sĩ, cache toàn ảnh, grouped validation và chạy Colab có resume.
---

# Original-image classification

Đọc `docs/plans/IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md` cho kế hoạch hiện hành: R0 bất biến, R1 routing, R2 Normalizer, R3 ElasticNet trên cache RetCCL512/Colab CPU. R1–R3 chưa chạy theo protocol mới; không tự tạo T4 hoặc trích lại features. `IMAGE_STRONG_LABEL_COLAB_EXECUTION.md` là lịch sử lần chạy đầu. Xem [các lỗi đã tái hiện](references/failures.md) khi sửa pipeline.

- Target riêng từng ảnh từ Glade theo xác nhận người dùng: 0→0, 3/4/5→1. Không dùng Ket_Luan/case_label cho mọi ảnh. Giữ raw Glade, không tự diễn giải ISUP/Gleason.
- Patient ID chỉ là group split. Mọi ảnh/vật kính cùng người cùng partition. Ảnh trùng nội dung hoặc nhãn xung đột phải được disposition trước train; không dựa unique filename để kết luận độc lập.
- Đây là ảnh gốc, không coordinate patches. Inventory phải đối chiếu đúng basename và lens; source ZIP immutable/hash-bound, từ chối lẫn archive tile.
- Whole-frame aspect resize + pad, không crop/patch. Frozen ResNet50 phải eval/inference_mode, pinned ImageNet V1 weights, fc Identity → 2048D. Không dùng default center-crop preprocessing mù quáng.
- TIFF phải decode từ stream còn sống, kiểm byte/pixel caps, CRC và content SHA. Local chỉ fixture nhỏ; full data/extraction trên Colab.
- Vector cache theo source/encoder/preprocessing identity. Commit shard chứa vectors/refs/order/hash; resume verified parts, không từ status.json đơn lẻ. training_ready chỉ sau đủ coverage và integrity.
- Active/unreadable/cross-host lock không được lấy lại theo TTL hoặc cờ recover đơn lẻ. Same-host recovery chỉ khi PID đã được xác minh chết. Nếu cần cross-host recovery, phải thiết kế proof gắn đúng lock và runtime trước mở lại.
- Scaler và C/threshold selection chỉ dùng train/inner validation. Outer test giữ kín. Pooled/per-lens OOF là kết quả từng ảnh, không đổi thành kết quả người bệnh. Deployment refit trên toàn dữ liệu phải tách khỏi đánh giá OOF.
- Cache bị cap smoke không được vào production training. Deadline/source chưa staged = partial, không complete. Config identity khác dùng cache/run mới.
- Runtime ZIP chỉ có code và closure requirements; không workbook/ảnh/credentials. Package hash pin và CI trước full run. CLI auth/Drive consent do người dùng hoàn tất theo official flow; không SSH/keepalive giả.
- Sau job: verify Drive artifacts, ngừng writers, official flush/unmount, dừng runtime được phép. Không shutdown máy local hoặc tạo VM thay thế tự động.
