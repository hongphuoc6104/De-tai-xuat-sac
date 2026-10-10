# Import ảnh patch đã duyệt

Module histology_data.precut đọc trực tiếp PNG từ ZIP hiện có. ZIP nguồn được giữ nguyên; bộ import không cắt lại, lọc, đổi màu, resize hoặc tạo nhãn patch. Thay đổi này chỉ lập chỉ mục, kiểm CRC/SHA và ghi QC. Feature vectors và bag MIL là bước sau.

Mỗi hàng trong tile_index.jsonl nối tile_id với ZIP/member, image ID, mã ca ứng viên nếu ghép được metadata, vật kính, tọa độ x/y, Glade nguyên gốc, checksum PNG, checksum pixel RGB và tình trạng nhãn. Patch không khớp metadata vẫn được ghi và chặn gate train. patient_id và bag_id để trống cho đến khi được xác minh. Kích thước PNG đang lưu được ghi riêng; kích thước crop trong ảnh nguồn vẫn là chưa xác minh vì tên x_y.png chỉ cung cấp tọa độ.

Tên x_y(1).png được hiểu là một instance khác cùng tọa độ. Mọi instance vẫn được giữ; kiểm kê tạo coordinate_collisions để người phụ trách xem checksum và nội dung trước modeling.

## Thử nhỏ tại máy phát triển

precut-inventory chỉ đọc danh mục ZIP (central directory) và metadata; chưa giải mã pixel. precut-smoke giải mã mặc định hai patch cho từng vật kính và lưu báo cáo JSON. Dùng thư mục ZIP nguồn hiện có:

    python -m histology_data precut-inventory --metadata Metadata.xlsx \
      --source /path/to/Tiles-001.zip --source /path/to/Tiles-002.zip \
      --output /tmp/precut/inventory.json

    python -m histology_data precut-smoke --inventory /tmp/precut/inventory.json \
      --output /tmp/precut/smoke.json --per-lens 2

Kiểm tra coverage, coordinate_collisions, sáu hàng checked, vật kính, case ID, x/y, kích thước, checksum và training_ready=false. Smoke này không đại diện cho toàn bộ dữ liệu.

## Nhập theo từng ZIP và tiếp tục sau ngắt

Sau khi runtime code, metadata và ZIP nguồn nằm trên Drive, trên Colab tạo inventory ngay tại đường dẫn Drive để các nguồn trong inventory tiếp tục truy cập được qua lần chạy sau. Gọi precut-import với --max-parts 1 trong preflight; full pass có thể bỏ giới hạn hoặc lặp lệnh. Mỗi ZIP được giải mã tuần tự và commit riêng:

    python -m histology_data precut-inventory \
      --metadata /content/drive/MyDrive/histology/source/Metadata.xlsx \
      --source /content/drive/MyDrive/histology/source/archives/Tiles-001.zip \
      --source /content/drive/MyDrive/histology/source/archives/Tiles-002.zip \
      --output /content/drive/MyDrive/histology/precut/v001/inventory.json

    python -m histology_data precut-import \
      --inventory /content/drive/MyDrive/histology/precut/v001/inventory.json \
      --output /content/drive/MyDrive/histology/precut/v001/parts \
      --work-root /content/histology-work --max-parts 1

Rerun xác minh part đã commit rồi tiếp tục part kế tiếp. Một ZIP được chép lên SSD của Colab, đọc/giải mã tại đó, rồi bản tạm được dọn sau khi commit trên Drive. precut-audit chỉ chạy khi đủ part; nó xác minh commit/checksum/toàn bộ hàng và quét trùng SHA trên toàn release:

    python -m histology_data precut-audit \
      --inventory /content/drive/MyDrive/histology/precut/v001/inventory.json \
      --parts-root /content/drive/MyDrive/histology/precut/v001/parts \
      --work-root /content/histology-work \
      --output /content/drive/MyDrive/histology/precut/v001/dataset_audit.json

Part gồm tile_index.jsonl, qc.json, commit.json. Commit chỉ xuất hiện sau khi index và QC được ghi/kiểm tra; rerun kiểm hash trước khi tái sử dụng. Audit toàn release dùng SQLite tạm trên SSD của Colab và bỏ file tạm sau khi xong. Không chạy audit trên sample để kết luận duplicate toàn bộ.

## Phạm vi và gate

- Mỗi patch được đọc với giới hạn kích thước byte/pixel, CRC ZIP, kiểm định PNG và giải mã đầy đủ. Ghi SHA byte PNG và SHA của pixel RGB.
- Thống kê pixel không cắt patch; tissue fraction hiện tại chỉ là chỉ dấu quan sát đơn giản RGB max < 245, chưa hiệu chuẩn và không dùng để loại ảnh.
- raw_glade được giữ nguyên. Kết luận metadata chỉ xuất hiện dưới trường candidate_label_from_conclusion, có trạng thái chưa review, để hỗ trợ chọn mẫu kỹ thuật; không phải target patch hoặc nhãn train.
- training_ready luôn false: cần patient mapping, nhãn ca được xác nhận, global duplicate review, encoder features và bag builder.
- Inventory chứa đường dẫn ZIP của runtime đã tạo. Hãy tạo inventory ở đúng môi trường/đường dẫn lưu trữ nơi sẽ chạy Colab; không chuyển inventory local sang Colab với đường dẫn máy local.
- Không có kết quả encode, feature cache, train, accuracy hay benchmark T4 trong module này.
