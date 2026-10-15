# Thử phân loại ảnh bằng nhãn mạnh trên Colab — 12/10/2026

> **Lịch sử ngày 12/10/2026.** Các trạng thái “chưa train/chưa mount” bên dưới thuộc lần chạy đầu. Kế hoạch hiện hành ngày 16/10 là [R0–R3 trên cache RetCCL512 và Colab CPU](IMAGE_CLASSIFICATION_R0_R3_EXECUTION.md); B1 đã hoàn tất, không chạy lại extraction hoặc tạo T4 theo snapshot này.

## Mục tiêu và nguồn

Một ảnh gốc 4×, 10× hoặc 40× → xác suất ảnh có ung thư. Người dùng xác nhận bác sĩ gán Glade cho từng ảnh: 0 = không ung thư, 3/4/5 = có ung thư. Mã người bệnh chỉ dùng chia dữ liệu và truy hồi, không làm target. Không tạo patch mới.

Profile và Drive folder do người dùng chọn được lưu trong receipt private local, không đưa account hoặc folder ID vào Git. Nguồn gồm `data_4x.zip`, `data_10x.zip`, `data_40x.zip`, `Metadata.xlsx`; vị trí MyDrive/shortcut phải kiểm sau mount. Ảnh chụp màn hình không chứng minh số ảnh/kích thước/chất lượng ZIP; cần kiểm thật.

Một runtime T4 duy nhất, tên được lưu trong receipt private. Xác thực CLI và Drive đã kiểm trong đúng kernel; chưa trích feature hoặc train khi tạo kế hoạch.

## I0 — Inventory và kiểm nhãn

Trong chính kernel CLI: kiểm T4, đọc được MyDrive, tìm đúng thư mục nguồn, đọc metadata và so SHA với bản đã audit. Nếu workbook khác, báo khác biệt và kiểm lại trước lập target.

Kiểm central directory ba ZIP: original filename khớp duy nhất Ten_File, đúng lens, không trùng basename/ảnh, không path traversal/symlink, không nhầm archive coordinate patches. Báo ảnh thiếu/ngoài metadata và phân bố người/lớp/lens. Không âm thầm lấy ảnh không có nhãn. Kiểm decode toàn ảnh trong extraction, giữ checksum nội dung và phát hiện ảnh giống nhau vượt patient boundaries trước split.

Đầu ra: versioned image manifest, archive inventory/hash receipts, label_evidence, exclusion/attention report. Chưa gọi data_ready chỉ từ screenshot.

## I1 — Smoke nhỏ và benchmark

Fixture tổng hợp nhỏ local kiểm phép ghép nhãn/split/cache/resume; mã mới qua conveyor/CI trước full run.

Trên T4, lấy mẫu thật ở cả ba vật kính và hai lớp khi nguồn cho phép. Decode một ảnh tại một thời điểm, full-image resize giữ tỉ lệ rồi padding, không center/random crop. Bắt đầu long edge 512; ghi kích thước ảnh gốc, thời gian đọc/forward, RAM/VRAM và mẫu preview nhỏ. 512 là baseline thử, chưa khẳng định bảo toàn chi tiết; thử 1024 là cấu hình khác nếu benchmark/budget cho phép.

Encoder: ResNet50 ImageNet V1, frozen + eval + inference_mode, global average pooling → 2.048 giá trị/ảnh. Không dùng đầu phân loại ImageNet 1.000 lớp làm feature. Lưu weights identity/checksum và toàn bộ preprocessing trong feature_id.

Đầu ra smoke riêng, không dùng smoke weights/features làm production. Kiểm forward thực và row–vector mapping, không NaN/Inf, save/load/resume.

## I2 — Trích đặc trưng toàn ảnh theo phần

Làm từng ZIP/lens: copy ZIP sang SSD Colab khi dung lượng đủ, kiểm hash; đọc ảnh qua ZIP stream, không giải nén toàn bộ ra hàng nghìn file. Sau phần đã commit và read-back đúng hash, dọn bản SSD của đúng ZIP đó. Không sao chép dữ liệu lớn về máy cá nhân.

Lưu vector và refs theo chunk nhỏ bất biến; row có image_id, patient_id, lens, raw_glade, image_target, ZIP/member/content hash. Config, source manifest, encoder identity khác phải dùng version khác. Resume chỉ từ commits/checksum hợp lệ; không dùng status.json như bằng chứng hoàn tất. Lock đang hoạt động không bị lấy lại theo TTL.

Output trên Drive dự kiến: `MyDrive/histology/image_classification/v001/<run_id>/` gồm inventory, features/parts, commits, config, status và events. Đây là vùng mới của tài khoản bop; không ghi đè nguồn NCKH_Duy hoặc features patch cũ.

## I3 — Train baseline nhanh

Train bộ phân loại Logistic Regression trên vector toàn ảnh, nhãn riêng từng ảnh. Chuẩn hóa vector fit chỉ trên train. Thử grid C nhỏ; chọn C và threshold bằng validation, không bằng outer test. Không fine-tune CNN trong vòng đầu. Nếu thêm MLP/fine-tune phải có đối chứng và config/run riêng.

Outer cross-validation grouped theo người bệnh; inner validation grouped. Ưu tiên 3 outer folds nếu mỗi train/test chứa hai lớp và phân bố nguồn hợp lệ; kiểm feasibility bằng dữ liệu thực. Ảnh mọi lens của cùng người luôn cùng fold. Duplicate content giữa người phải được disposition trước fit; không để vượt train/test. Báo số người/ảnh/lớp/lens mỗi partition. Sampling/weighting theo người là đối chứng được ghi config, không đổi nhãn ảnh thành nhãn ca.

Đầu ra: classifier/scaler theo fold, selected hyperparameters, OOF dự đoán từng ảnh với image/lens/fold lineage, metrics pooled và theo lens. Chọn threshold trong validation; báo thêm fixed threshold 0.5. Fit được publish atomically và verify trước resume.

## I4 — Báo cáo, suy luận và kết thúc phiên

Báo AUROC, AUPRC, sensitivity/specificity/confusion matrix, số ảnh và số người. 18 người là giới hạn số người độc lập; không tuyên bố hiệu quả lâm sàng từ số ảnh. Chỉ báo OOF cho ảnh thực sự có prediction hợp lệ. Mô hình nhị phân chỉ trả nhãn ảnh, không xác định vị trí ung thư.

Kiểm inference một ảnh mới → preprocessing đúng phiên bản → encoder → classifier/scaler; không cần case label, nhiều ảnh hoặc đủ ba lens. Kiểm prediction trước/sau lưu nạp trong sai số cho phép. Encoder frozen vẫn cần chạy khi suy luận ảnh mới.

Lưu report cuối trên Drive và bản nhỏ local, read-back checksum, dừng writers, official Drive flush/unmount và dừng T4 khi hoàn tất hoặc không thể tiếp tục. Không dùng SSH cho managed Colab Free, không keepalive giả, không tự đổi account/tạo VM thay thế nếu phiên mất. Không shutdown máy cá nhân.

## Ngân sách và trạng thái

5 giờ 30 phút là ngân sách vận hành người dùng đưa ra, không phải số giờ còn lại Colab xác nhận. Có budget stop chủ động trước giới hạn, giữ phần đã commit; hết budget = partial/paused, không complete. Thời gian dự kiến được tính sau smoke theo ảnh/lens, gồm copy ZIP và I/O Drive, không đoán từ dung lượng file.

Giai đoạn hiện tại: CLI và Drive đã xác minh; header inventory có đủ 2.802 TIFF khớp metadata (4× 364, 10× 830, 40× 1.608), không thiếu/trùng basename. Sáu header mẫu đều 3.840×2.160; chưa suy rộng kích thước cho toàn bộ ảnh. Đang staging ZIP và chạy kiểm thử/CI; chưa chạy training mới.

## Tham khảo

- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html — frozen feature extractor và fine-tuning.
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html — pretrained ResNet50 và cấu hình weights.
- Metadata và xác nhận người dùng đã audit tại `Results/label_audit_20261012/` của workspace chính; giữ raw Glade, không chuyển sang nhãn patch.
