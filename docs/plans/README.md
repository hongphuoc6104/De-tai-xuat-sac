# Kế hoạch dự án

Thư mục này lưu kế hoạch tham chiếu và lộ trình trước huấn luyện MIL. Các tài liệu phân biệt dữ kiện đã xác nhận, tác vụ đang chạy và đề xuất chưa thực hiện.

## Kế hoạch hiện tại

- [`MIL_MULTISCALE_REFERENCE.md`](MIL_MULTISCALE_REFERENCE.md): kế hoạch phân loại mô học tuyến tiền liệt bằng MIL ở 4×, 10×, 40×; gồm dữ liệu, encoder pretrained/frozen, features, bags, protocol và đánh giá dự kiến.
- [`PRETRAIN_EXECUTION.md`](PRETRAIN_EXECUTION.md): lộ trình G0–G5 và readiness audit trước MIL, cùng đường dẫn, đầu vào/đầu ra và trạng thái hiện tại. Không bao gồm huấn luyện MIL.

Bằng chứng kiểm kê và hình minh họa được lưu trong `Results/planning_20261010/`.

## Dữ liệu patch và extractor

- [`../PRECUT_IMPORT.md`](../PRECUT_IMPORT.md) và [`../../notebooks/Colab_PreCut_Import.ipynb`](../../notebooks/Colab_PreCut_Import.ipynb): inventory, smoke, import và audit dữ liệu patch đã duyệt.
- [`../FEATURE_EXTRACTION.md`](../FEATURE_EXTRACTION.md) và [`../../notebooks/Colab_Feature_Extraction.ipynb`](../../notebooks/Colab_Feature_Extraction.ipynb): extractor ResNet50 ImageNet V1 frozen chạy theo ZIP hoặc thư mục PNG.

Hiện Drive có đủ 25/25 ZIP. Smoke T4 đã tạo sáu vector thật và hoàn tất trong 112,616 giây. Production writer `histology-features-resume-20261010` vẫn chạy với source plan 24 ZIP/142.791 PNG; sau khi lượt này kết thúc, cần resume để inventory ZIP thứ 25. Full feature cache và audit chưa hoàn tất. Governance, split và bag tooling đang được phát triển; chưa có bundle đã kiểm toán và chưa huấn luyện MIL.
