# Dự án phân loại mô học bằng MIL nhiều độ phóng đại

- [Kế hoạch tham chiếu](docs/plans/MIL_MULTISCALE_REFERENCE.md): nguồn dữ liệu, nhãn yếu, features, bags, train/validation/test và đánh giá.
- [Trích vector trên Colab](docs/FEATURE_EXTRACTION.md): chạy độc lập theo từng phần từ ZIP hoặc thư mục PNG, kiểm tra và resume.
- [Notebook feature extractor](notebooks/Colab_Feature_Extraction.ipynb): smoke sáu patch trước khi production.
- [Import patch đã duyệt](docs/PRECUT_IMPORT.md): inventory/QC không tạo features.
- [Hợp đồng xử lý ảnh gốc](docs/DATA_SHARDS_CONTRACT.md): raw/processed tools trước đây.

Extractor tải ResNet50 ImageNet V1, đóng băng và lưu features 2.048 chiều. Chưa huấn luyện MIL hoặc suy luận ung thư; cần xác nhận người bệnh, nhãn ca và splits trước modeling.
