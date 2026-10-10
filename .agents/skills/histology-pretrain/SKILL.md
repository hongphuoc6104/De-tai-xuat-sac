---
name: histology-pretrain
description: Tạo governance, split theo người bệnh và bag tham chiếu từ feature mô học đã commit; dùng khi chuẩn bị hoặc kiểm bundle MIL trước train trên Colab.
---

# Chuẩn bị dữ liệu MIL trước train

Đọc `docs/PRETRAIN_READINESS.md` cho cách chạy, `docs/plans/PRETRAIN_EXECUTION.md` cho trạng thái và `histology_data/{governance,splits,bags,readiness}.py` cho schema. Khi sửa lỗi, đọc [references/failures.md](references/failures.md).

- Phân biệt image metadata, ca và người bệnh. Scope review theo các ca thực sự có patch; 990 ảnh master không có trong release hiện tại không tự trở thành lỗi mất dữ liệu.
- Candidate case ID không phải patient ID nếu chưa có xác nhận. Bằng chứng phải đi cùng từng mapping/nhãn và metadata SHA; thay cách serialize review CSV không được thay sự thật nghiệp vụ. Nếu migrate binding, giữ hash artifact trước và chứng minh cùng metadata/case scope.
- Nhãn binary nằm ở cấp ca; ca có vùng carcinoma và lành vẫn là ca dương khi endpoint đã xác nhận. Giữ raw Glade, không tạo supervised patch targets hoặc suy ISUP.
- Feature cache luôn có `training_ready=false` vì chưa chứa governance; downstream bundle xét full release/audit, encoder/preprocess/commit identity và governance riêng. Smoke/partial/capped cache không được qua gate.
- Baseline hiện tại dùng frozen ResNet50 ImageNet V1, RGB resize cạnh ngắn256 + center-crop224; PNG nguồn không đổi. Không sửa ba module feature tạo fingerprint để bổ sung governance; preprocessing khác cần feature namespace riêng.
- Bag là ca×vật kính, gồm references đến part/row/tile. Không dùng Ten_Slide, ZIP hoặc patient làm bag ID. Lens thiếu là mask với0instance, không tạo vector giả. Một lần đọc một ca, finite/row audits phải có giới hạn bộ nhớ.
- Chia nhóm theo patient map đã xác minh và duplicate components đã review; giữ nhiều ca/nhãn khác nhau của cùng người cùng phía. Nested3×2 chỉ qua khi mọi outer/inner partition có đủ lớp; không tách patch để cứu split.
- Duplicate dispositions phải bao phủ đúng members và có evidence. Không silently drop hoặc để cùng nội dung đi sang hai split. PNG và cache vẫn giữ để truy ngược dù bundle loại reference đã review.
- Deadline từ ngân sách profile phải áp dụng cả wait, hash, audit, build và verify; dùng deadline UTC cho vận hành và monotonic cho vòng I/O. Không đặt lại330phút sau resume.
- Một writer feature/output và một writer pretrain/output. Chỉ recover lock sau khi writer/runtime trước đã dừng; không chiếm PID sống. Mọi công việc dài/full chạy Colab; local chỉ metadata nhẹ và synthetic/smoke nhỏ.
