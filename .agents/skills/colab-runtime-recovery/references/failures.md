# Bằng chứng lỗi vận hành Colab

Các thao tác SSH trong entries là lịch sử pretrain, không là cách chạy mặc định cho Colab Free.

### [ERR-20261011-03] Colab Monaco nối code khi chỉ fill textarea
- **Triệu chứng (Symptom):** Thay code trong ô Colab đang có nội dung bằng `fill()` nối script mới vào cuối model, gây `SyntaxError` tại `print(...)import csv`.
- **Tái hiện E2E (Reproduction Path):** Chrome scratchpad, Cell 2: chạy setup; fill script review vào cùng textarea; Run cell; DOM output iframe ghi lỗi cú pháp trước khi review thực thi.
- **Nguyên nhân cốt lõi (Root Cause):** Textarea của Monaco là bộ đệm input, không đại diện toàn bộ model code. Locator fill không bảo đảm thay toàn bộ tài liệu.
- **Giả định sai (Wrong Assumptions):** Xem textarea Editor content như một ô nhập HTML thông thường.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Focus editor, ControlOrMeta+A, Backspace; kiểm DOM xác nhận model trống, rồi fill và chạy. Review thật báo `review_complete=true`, hai group và một tile loại; runner mới đi vào `building_bundles`. Proof: `Results/planning_20261010/pretrain/ui_editor_recovery.json`. Không sửa mã pipeline dữ liệu.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Dùng cell mới hoặc xóa toàn model bằng phím editor trước thay code; đọc DOM và output iframe, không coi output region trống là thành công.


### [ERR-20261011-04] Hết deadline trong mount tương tác không chứng minh Drive đã mount
- **Triệu chứng (Symptom):** CLI báo credentials propagated rồi timeout; SSH kiểm tra mountpoint/MyDrive đều chưa có.
- **Tái hiện E2E (Reproduction Path):** Mount thiha qua CLI, chờ người dùng duyệt Google; xem log `/tmp/histology-thiha-drivemount.log` và probe SSH trước retry.
- **Nguyên nhân cốt lõi (Root Cause):** Automation có deadline 600 giây bao gồm thời gian duyệt; quyền VM và FUSE mount là hai trạng thái khác nhau.
- **Giả định sai (Wrong Assumptions):** Dòng credentials propagated hoặc status BUSY cached đủ để kết luận mount filesystem thành công.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Xác minh đúng VM, kết thúc client lỗi, retry mount trên cùng VM dùng quyền đã cấp; thành công, MyDrive có đủ 25 ZIP và metadata SHA đúng. Receipt `Results/planning_20261010/pretrain/profile_handoff_20261011.json`.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Kiểm mountpoint và artifact thật; không tạo VM thay thế hoặc xin OAuth lại chỉ vì client timeout.


### [ERR-20261011-05] Brace expansion không hoạt động với SCP dùng SFTP
- **Triệu chứng (Symptom):** Remote path `{status.json,...}` bị coi là tên file literal, trả No such file.
- **Tái hiện E2E (Reproduction Path):** `/tmp/histology-thiha-final-download.log` ghi lỗi tải bốn report bằng SCP brace.
- **Nguyên nhân cốt lõi (Root Cause):** SFTP backend không thực thi shell expansion của remote path.
- **Giả định sai (Wrong Assumptions):** SCP mới có cùng expansion semantics với remote shell cũ.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Dùng một SSH tar stream với bốn tên tường minh; local giải nén whitelist, chặn links/path traversal và cap 2 MiB; final-report SHA khớp loader report.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Dùng tên tường minh hoặc stream bounded; không phụ thuộc shell glob/brace của SFTP.


### [ERR-20261011-06] Drive flush qua SSH thiếu môi trường Colab
- **Triệu chứng (Symptom):** API chính thức `flush_and_unmount` báo KeyError CLOUDSDK_CONFIG dù artifacts đã verify và không còn writer.
- **Tái hiện E2E (Reproduction Path):** `/tmp/histology-thiha-cleanup-run.log`; chạy cleanup bằng Python SSH.
- **Nguyên nhân cốt lõi (Root Cause):** SSH shell không kế thừa CLOUDSDK_CONFIG mà SDK dùng để xác định root chứa Drive binary.
- **Giả định sai (Wrong Assumptions):** Môi trường SSH tương đương kernel notebook.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Đối chiếu source API, kiểm `/opt/google/drive/drive`, truyền CLOUDSDK_CONFIG=/content/.config cho child rồi dùng cùng API; Drive unmount thành công, CPU stop, server sessions rỗng. Proof `Results/planning_20261010/pretrain/final_verified_20261011/cleanup_verified.json`.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Cung cấp môi trường SDK hợp lệ; không đọc/copy credentials, không force-unmount khi còn writer.

## V. Bài học đã chuyển thành skill

- [ERR-20261009-03] Smoke I/O và cân bằng lớp: xem histology-shards/references/failures.md.
- [ERR-20261009-04] Directory/ZIP mixed sources: xem histology-shards/references/failures.md.
- [ERR-20261009-05] Targeted staging và catalog cache: xem histology-shards/references/failures.md.
- [ERR-20261009-06] Instance/content tile identity: xem histology-shards/references/failures.md.
- [ERR-20261009-07] Resume/version/SSD cleanup: xem histology-shards/references/failures.md.
- [ERR-20261009-08] Raw duplicate leakage và typed review: xem histology-shards/references/failures.md.
- [ERR-20261009-09] Notebook import order bị bỏ sót ở lint riêng: xem histology-shards/references/failures.md.
- [ERR-20261009-10] Nền ám màu bị giữ như mô: xem histology-shards/references/failures.md.
- [ERR-20261009-11] Local pytest wrapper che lỗi import CI: xem histology-shards/references/failures.md.

Chi tiết: [.agents/skills/histology-shards/references/failures.md](.agents/skills/histology-shards/references/failures.md).

### [ERR-20261012-01] File API upload không tạo thư mục cha
- **Triệu chứng (Symptom):** Upload hai artifact báo cáo vào thư mục mới trả HTTP500; file local vẫn nguyên vẹn.
- **Tái hiện E2E (Reproduction Path):** Native upload tới reports/cpu_preliminary trước khi thư mục tồn tại; proof `Results/clustering_dual/execution_20261011/operations/upload_parent_directory_20261012.json`.
- **Nguyên nhân cốt lõi (Root Cause):** File API tạo file nhưng không tạo cây thư mục đích chưa tồn tại.
- **Giả định sai (Wrong Assumptions):** Upload API hoạt động như mkdir -p rồi copy.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Tạo đúng parent trong kernel Colab đã mount, retry cùng hai payload; hai upload exit0. Không đổi VM, mount hay payload.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Xác nhận thư mục cha trước transfer; lọc proxy-token khỏi lỗi CLI trước khi đưa output vào báo cáo.
