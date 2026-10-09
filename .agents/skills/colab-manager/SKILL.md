---
name: colab-manager
description: Quản lý profile và phiên Google Colab, theo dõi thời gian GPU cục bộ, và luân phiên profile khi gặp lỗi quota/capacity đã nhận diện.
---

# Colab Manager

Dùng skill này khi cần kiểm tra profile đã lưu, xem mức dùng, tạo hoặc kết nối phiên Colab, luân phiên tài khoản trong một lần tạo phiên, hay theo dõi thời gian T4. Helper dựa vào colab-profile và CLI colab đang cài trên máy.

## Giới hạn cần biết

- colab usage báo compute units và tốc độ sử dụng, không báo số giờ GPU miễn phí còn lại. Ngưỡng 350 phút T4/profile/ngày là ước tính cục bộ có thể chỉnh, không phải quota do Colab xác nhận.
- Chỉ tính phiên được tạo qua helper này. Phiên tạo trên Colab web hoặc gọi colab trực tiếp không được tự phát hiện hay tính vào bộ đếm.
- Profile chỉ được coi là sẵn sàng sơ bộ khi cả tệp token và tệp session của nó tồn tại. Không mở, in, sao chép hoặc chỉnh các tệp xác thực; sự tồn tại của tệp không chứng minh phiên OAuth còn hiệu lực.
- Bộ đếm dùng đồng hồ UTC đã ghi và ngày địa phương Asia/Ho_Chi_Minh. Các phiên T4 chạy đồng thời được cộng dồn theo phút runtime.

## Thao tác

Chạy helper từ thư mục gốc dự án:

    python3 .agents/skills/colab-manager/scripts/colab_manager.py profiles
    python3 .agents/skills/colab-manager/scripts/colab_manager.py status

- profiles hiển thị thứ tự trong ~/.config/colab-cli/profiles.json, số thứ tự, trạng thái tệp cấu hình và lần dùng gần nhất. Giữ thứ tự đó; bỏ qua profile thiếu một trong hai tệp.
- status hiển thị phiên do helper theo dõi, thời gian đã chạy, tổng phút T4 trong ngày, ngưỡng và phần ước tính còn lại.
- usage [--profile ALIAS] gọi colab usage qua alias đã lưu. Không diễn giải compute units thành giờ còn lại.
- budget show xem ngưỡng; budget set --gpu T4 --minutes N [--profile ALIAS] đổi mặc định 350 phút hoặc đặt riêng cho một profile.

### Tạo và luân phiên profile

Chỉ tạo VM khi người dùng yêu cầu rõ ràng hoặc vừa xác nhận. Khi được yêu cầu tạo GPU mà không nêu loại, dùng T4. Bắt đầu từ profile tạo phiên thành công gần nhất; lần đầu dùng profile khả dụng đầu tiên theo thứ tự registry.

Gọi helper với create --gpu T4 [--session NAME] --yes sau yêu cầu rõ ràng của người dùng. Trong đúng yêu cầu đó, nếu Colab trả lỗi quota/capacity rõ ràng thì thử profile kế tiếp theo vòng tròn, bỏ qua profile thiếu tệp và profile đã chạm ngưỡng ngày cục bộ. Chỉ chuyển con trỏ và ghi lần dùng sau khi tạo phiên thành công.

Không tự luân phiên khi có lỗi OAuth, thiếu scope, lỗi mạng, timeout chưa xác định được kết quả, tham số sai hoặc lỗi không nhận diện. Với timeout, kiểm tra đúng profile và session name đã gửi; nếu chưa xác minh được VM, dừng để tránh tạo trùng. Nếu tất cả profile hết ngân sách ước tính hoặc không khả dụng, báo kết quả và chờ người dùng chỉnh ngân sách hoặc chọn cách khác.

### Kết nối, dừng và timer

- connect --session NAME chỉ mở lại phiên đang được helper theo dõi và còn tồn tại. Nếu phiên đã mất, báo trạng thái; không tự tạo VM thay thế.
- stop --session NAME --yes kết thúc VM thật và ghi thời điểm dừng. Chỉ dùng sau yêu cầu rõ ràng hoặc xác nhận của người dùng. Đóng CLI/websocket không đồng nghĩa với dừng VM.
- Sau khi tạo thành công, helper cài và bật timer user-level nếu cần. Timer kiểm tra phiên mỗi phút, gửi thông báo desktop ở các mốc còn 30, 10, 5 phút và khi chạm ngưỡng ngày. Timer chỉ đếm và thông báo; khi hết giờ phải hỏi trước khi dừng VM.
- Khi không còn phiên đang theo dõi, timer tự disable. timer status xem trạng thái; timer install cài unit nhưng không bật timer khi đang rảnh; timer uninstall gỡ unit nền, không dừng VM và sẽ hỏi xác nhận nếu còn phiên đang theo dõi.
- Nếu systemd user hoặc thông báo desktop không khả dụng, giữ phiên và bản ghi local, báo rằng timer nền chưa chạy. Không tự dừng VM để che lỗi timer.

Trạng thái và settings nằm trong ~/.local/state/colab-manager/ và ~/.config/colab-manager/; chỉ lưu alias, tên phiên, mốc thời gian và ngân sách, không lưu token hay nội dung đầu ra CLI.
