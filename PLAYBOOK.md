# PLAYBOOK.md — Sổ Tay Lỗi & Bài Học Kinh Nghiệm Dự Án

Sổ tay này lưu trữ toàn bộ các bài học sau khi fix bug thành công trong dự án.
**Mục tiêu:** Lưu vết nguyên nhân cốt lõi, phòng ngừa tái phạm, biến mỗi sai sót thành năng lực kỹ thuật lâu dài cho Agent.

---

## I. Hướng Dẫn Đóng Góp Bài Học (Contribution Protocol)

1. **Thời điểm thêm:** Ngay sau khi giải quyết triệt để lỗi và đã qua bước kiểm chứng E2E thành công.
2. **Cơ chế tách Skill (Progressive Disclosure):**
   - Khi một chủ đề tích lũy từ 3 bài học trở lên hoặc sổ tay vượt quá 200 dòng, nội dung chi tiết sẽ được chuyển thành Skill độc lập tại `.agents/skills/<skill-name>/SKILL.md`.
   - Trong file này chỉ giữ lại 1 dòng mục lục kèm đường dẫn tham chiếu.

---

## II. Danh Mục Kỹ Năng Đã Module Hóa (Modularized Skills Index)

*Khi cần xử lý vấn đề thuộc các chuyên môn dưới đây, Agent chỉ đọc lướt mô tả ngắn; nếu đúng bài toán mới mở toàn bộ `SKILL.md`.*

| Tên Skill | Mô tả ngắn kích hoạt | Đường dẫn |
| :--- | :--- | :--- |
| colab-manager | Quản lý profile/phiên Colab, bộ đếm GPU cục bộ và luân phiên profile khi gặp lỗi quota/capacity | [.agents/skills/colab-manager/SKILL.md](file:///data/đề tài xuất sắc/.agents/skills/colab-manager/SKILL.md) |
| `histology-shards` | QC, packing, instance coordinates và verified resume trên local smoke/Colab | [.agents/skills/histology-shards/SKILL.md](.agents/skills/histology-shards/SKILL.md) |
| `gh-axi` | Thao tác GitHub qua CLI tối ưu token TOON (issues, PRs, CI, repo, release) | [.agents/skills/gh-axi/SKILL.md](file:///data/đề tài xuất sắc/.agents/skills/gh-axi/SKILL.md) |
| `treehouse` | Quản lý Git worktree cô lập cho đa Agent song song (.worktrees) | [.agents/skills/treehouse/SKILL.md](file:///data/đề tài xuất sắc/.agents/skills/treehouse/SKILL.md) |
| `lavish` | Báo cáo trực quan HTML tương tác và giao diện phản biện đa năng | [.agents/skills/lavish/SKILL.md](file:///data/đề tài xuất sắc/.agents/skills/lavish/SKILL.md) |

---

## III. Mẫu Ghi Chép Bài Học Chuẩn (Entry Template)

```markdown
### [ERR-YYYYMMDD-01] <Tiêu đề ngắn gọn bản chất lỗi>
- **Triệu chứng (Symptom):** Mô tả hiện tượng lỗi gặp phải trong thực tế.
- **Tái hiện E2E (Reproduction Path):** Lệnh hoặc script test tái hiện bug trước khi fix.
- **Nguyên nhân cốt lõi (Root Cause):** Cơ chế kỹ thuật sâu dẫn đến lỗi (không đổ lỗi triệu chứng).
- **Giả định sai (Wrong Assumptions):** Giả định chủ quan ban đầu gây ra thiết kế sai sót.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Phương án xử lý chuẩn kiến trúc và kết quả chạy lại bài test E2E.
- **Quy tắc phòng ngừa vàng (Golden Rule):** 1-2 quy tắc hành động cụ thể để vĩnh viễn không tái diễn.
```

---

## IV. Nhật Ký Lỗi & Bài Học (Active Error Log)

### [ERR-20261009-01] Đệ quy Pytest và Nhận diện Thiếu Khuyết Của Phân Tích Cú Pháp AST Trong Trạm 2
- **Triệu chứng (Symptom):** Khi chạy `pytest tests`, test suite bị treo do test station gọi đệ quy toàn bộ `pytest tests`, đồng thời AST Integrity Check ban đầu coi `with pytest.raises` là test rỗng vì thiếu node `ast.Assert`.
- **Tái hiện E2E (Reproduction Path):** Chạy `./pipeline run` khi `test_pipeline_stations.py` gọi trực tiếp `run_station_2_tests()`.
- **Nguyên nhân cốt lõi (Root Cause):** (1) Test integration gọi lại runner của chính nó tạo vòng lặp đệ quy. (2) Trình phân tích AST chỉ quét `ast.Assert` và `ast.Call(assert*)` mà bỏ sót context manager `ast.With` của `pytest.raises`.
- **Giả định sai (Wrong Assumptions):** Giả định rằng mọi bài kiểm tra ngoại lệ đều sinh node `ast.Assert` thông thường; giả định runner cấp 2 có thể tự gọi runner tổng mà không giới hạn scope.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Thêm tham số `target_tests` cô lập scope cho integration test; mở rộng parser AST nhận diện cả `pytest.raises`; thêm kiểm tra assert trực tiếp trên exception value. Toàn bộ 19 tests pass 100% trong 3.34s, proof log xuất tại `Results/proofs/test_evidence_20261009_070701.log`.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Các bài test kiểm thử chính pipeline runner phải luôn chạy trên target con biệt lập, không bao giờ được quét lại chính thư mục chứa bài test đó. Cú pháp AST cho assertion phải bao phủ cả `ast.Assert`, `assert*` call và context manager ngoại lệ.


### [ERR-20261009-02] systemd Không Chấp Nhận Đường Dẫn WorkingDirectory Có Dấu Cách Khi Được Quote
- **Triệu chứng (Symptom):** systemd-analyze --user verify từ chối unit timer Colab vì đường dẫn workspace /data/đề tài xuất sắc/... bị coi là không tuyệt đối.
- **Tái hiện E2E (Reproduction Path):** Chạy pytest -q tests/test_colab_manager.py::test_systemd_units_validate_with_analyze; test tạo unit với đường dẫn workspace thật rồi gọi systemd-analyze --user verify.
- **Nguyên nhân cốt lõi (Root Cause):** WorkingDirectory= là giá trị đường dẫn của systemd, không phải đối số ExecStart; quote kiểu đối số khiến dấu quote bị giữ như một phần của đường dẫn.
- **Giả định sai (Wrong Assumptions):** Giả định cùng một quy tắc quote có thể dùng cho mọi giá trị trong unit file.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Escape dấu cách của WorkingDirectory= thành \x20, giữ quote riêng cho từng đối số ExecStart; kiểm tra lại unit bằng systemd-analyze --user verify và test hồi quy E2E, cả hai đều pass.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Dùng cú pháp escape theo loại directive của systemd; không áp dụng shell quoting vào giá trị đường dẫn trong unit file.

### [ERR-20261010-01] Colab CLI Trả Mã 0 Cho Phiên Không Còn Tồn Tại
- **Triệu chứng (Symptom):** `connect` để phiên đã mất ở trạng thái đang chạy; timer tiếp tục tính phút, và kiểm tra sau timeout của `create` có thể ghi nhận phiên không tồn tại là tạo thành công.
- **Tái hiện E2E (Reproduction Path):** Gọi `colab-profile ad8410552 status --session <tên-test-ngẫu-nhiên>` với một tên chưa tồn tại: CLI trả mã 0 cùng `[colab] Session '<tên>' not found.` hoặc `No active sessions found on server.`. Lặp lại luồng `connect`/`watch` với state và profile registry tạm; hồi quy được giữ tại `tests/test_colab_manager.py`.
- **Nguyên nhân cốt lõi (Root Cause):** Các consumer chỉ gọi bộ phân loại missing-session sau khi kiểm tra mã thoát khác 0; mẫu nhận diện cũng không khớp thông báo “No active sessions found”.
- **Giả định sai (Wrong Assumptions):** Mã thoát 0 của lệnh `status` luôn đồng nghĩa phiên còn hoạt động.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Phân loại thông điệp missing bất kể mã thoát, nhận diện tên phiên hợp lệ dài đến 64 ký tự cùng mẫu “no active sessions”, đóng timer interval ở `connect`/poll và từ chối ghi nhận thành công sau timeout khi status báo không có phiên. E2E read-only với CLI thật xác nhận `connect` và poll đều đóng state tạm; 20 test Colab Manager và toàn bộ 96 tests pass. Proof: `Results/proofs/test_evidence_20261010_144242.log`; CI chạy bộ Colab Manager ở bước riêng.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Dựa trên nội dung rõ ràng của phản hồi status để xác định phiên có tồn tại; không suy ra từ mã thoát riêng lẻ.


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
