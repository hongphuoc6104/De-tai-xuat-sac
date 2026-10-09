---
name: histology-shards
description: Chuẩn bị ảnh mô học thành shard có checksum, QC và tọa độ patch; dùng khi phát triển smoke nhỏ tại local hoặc xử lý từng phần có resume trên Colab.
---

# Histology shards

Đọc `docs/DATA_SHARDS.md` cho CLI/notebook đang triển khai và `docs/DATA_SHARDS_CONTRACT.md` khi sửa schema/API. Kiến trúc hiện dùng `histology_data`, không ghép lại các thuật toán trong notebook.

- Local smoke phải giới hạn ảnh trước hash/decode; dùng vài ảnh mỗi vật kính và đánh dấu smoke. Full dữ liệu chạy trên Colab theo yêu cầu hiện tại.
- Shard là đơn vị vận chuyển/xử lý; patient fold và bag là đơn vị học. Một người nằm cùng fold dù ảnh ở nhiều shard/vật kính.
- Giữ `raw_glade` và nguồn nhãn yếu; không đổi pattern thành ISUP, không truyền nhãn ca xuống supervised patch target.
- Mỗi raw/processed shard chỉ completed sau kiểm tra checksum và commit; phần hỏng không ngăn xử lý phần tốt không liên quan. Resume processed output xác minh metadata/outputs trước, không stage lại raw đã xong.
- Cache staging phải có catalog ID; processing fingerprint có implementation version. Runtime-only cleanup không đổi tile IDs.
- `tile_id` phân biệt instance theo image ID + raw SHA + tọa độ + config/version. `content_tile_id` phục vụ phát hiện nội dung trùng. Giữ provenance và flag review, không âm thầm bỏ bản trùng.
- Candidate case ID, nhãn chưa review, smoke hoặc SHA ZIP chưa audit không được báo training-ready. Trước modeling, kiểm tra duplicate SHA toàn cohort và mapping người bệnh.
- Profile đổi đường dẫn/runtime preset, giữ cùng schema và metadata/fold version. Không mở hoặc copy credential files. Không tạo VM trong tác vụ chỉ phát triển local hoặc chuẩn bị notebook.

Đọc [references/failures.md](references/failures.md) khi sửa resume, metadata gate, ZIP adapter hoặc tile identity; file lưu E2E repro và kết quả từng lỗi đã được sửa.
