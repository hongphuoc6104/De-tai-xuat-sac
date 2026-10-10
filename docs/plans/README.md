# Kế hoạch dự án

Thư mục này lưu kế hoạch tham chiếu để phát triển, theo dõi tiến độ và đối chiếu kết quả. Tài liệu phân biệt thiết kế dự kiến với phần đã thực thi; các thay đổi được ghi theo phiên bản.

## Kế hoạch hiện tại

- `MIL_MULTISCALE_REFERENCE.md`: kế hoạch phân loại mô học tuyến tiền liệt bằng MIL ở 4×, 10×, 40×; gồm dữ liệu, pretrained/freeze, features, bags, train/validation/test, đánh giá, lưu trữ Colab và các trạm giám sát.

Các bản render HTML/PNG/SVG và bằng chứng kiểm kê theo ngày được lưu trong `Results/planning_20261010/`. Bản tham chiếu chính nằm trong thư mục này.

## Tài liệu triển khai dữ liệu patch đã duyệt

- `../PRECUT_IMPORT.md`: quy trình và lệnh inventory, smoke, import từng ZIP, audit trùng nội dung.
- `../../notebooks/Colab_PreCut_Import.ipynb`: notebook Colab preflight và import có thể tiếp tục sau khi ngắt.
- Trạng thái thực thi mới nhất được ghi trong phiên bản kế hoạch 1.2; ghi tài khoản/Drive đã thiết lập, cách chuẩn bị từng phần và ngân sách T4 330 phút/ngày. Trạng thái đầy đủ phụ thuộc source release và audit; chưa có features/train.
