---
name: colab-runtime-recovery
description: Khôi phục lỗi mount Drive, notebook editor và bàn giao artifact Colab khi đã có phiên; dùng khi filesystem/kernel không khớp hoặc flush thất bại.
---

# Colab runtime recovery

Đọc [lỗi đã tái hiện](references/failures.md) theo tình huống thực tế. Các log SSH trong đó là bằng chứng lịch sử; managed Colab Free chạy qua notebook/CLI notebook execution, không dùng SSH để tránh giới hạn dịch vụ.

- Xác minh đúng account/session/kernel và đọc được filesystem trong chính kernel sẽ chạy. Trang cấp quyền thành công không chứng minh Drive đã mount.
- Timeout mount không tạo VM mới: kiểm session hiện có, kiểm directory/root và retry official mount khi consent đã đủ. Người dùng tự nhập credentials/CAPTCHA/duyệt quyền trên Chrome của họ khi cần. Không đọc/copy/edit cached auth files.
- Thay code notebook: select all, clear, xác minh rỗng rồi fill; đọc lại nội dung đúng một bản trước Run.
- Chuyển artifact: danh sách file chính xác và kiểm hash; không dựa brace expansion trong SFTP. Không in proxy credentials/URL secrets.
- Khi kết thúc, dừng writers trước official Drive flush/unmount; môi trường API phải là môi trường Colab thật. Chỉ kết thúc runtime thuộc công việc khi có quyền; không force hoặc tự shutdown máy.
