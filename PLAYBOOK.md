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

### [ERR-20261009-03] Tập smoke đọc toàn bộ ảnh nguồn và bỏ sót lớp dương
- **Triệu chứng (Symptom):** Public catalog API với sáu TIFF thật và `per_lens=2` đọc checksum cả sáu ảnh, đồng thời chọn hai ca lành dù nguồn có ca ung thư.
- **Tái hiện E2E (Reproduction Path):** `pytest tests/test_data_catalog.py::test_smoke_alternates_available_case_labels tests/test_data_catalog.py::test_smoke_hashes_only_selected_files`; trước sửa, kết quả là labels `[0,0]` và sáu lượt hash TIFF cho hai ảnh được chọn. Fixture chạy inventory → metadata → selection trên file ảnh/CSV thật.
- **Nguyên nhân cốt lõi (Root Cause):** Inventory directory hash ảnh trước selection; selector đi qua toàn bộ ca của lớp 0 trước lớp 1.
- **Giả định sai (Wrong Assumptions):** Giả định giới hạn sau inventory cũng giới hạn I/O; round-robin theo ca được coi là cân bằng giữa lớp.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Inventory chỉ liệt kê path/size, hash sau khi chọn; xen kẽ lớp và ca. Cả tám kiểm thử catalog pass, chỉ hai TIFF được hash và nhãn `{0,1}` có mặt. Smoke luôn `training_ready=false` dù có identity map và nhãn đã review.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Áp dụng giới hạn kỹ thuật trước đọc/decode/hash ảnh; smoke phải đo coverage của ca/lớp và không được báo là release khoa học đầy đủ.

### [ERR-20261009-04] Colab Adapter Rejected Mixed Directory and ZIP Sources
- **Triệu chứng (Symptom):** Notebook stopped when a 4X source directory and 10X/40X ZIP archives were both configured, although the CLI and catalog support multiple source paths.
- **Tái hiện E2E (Reproduction Path):** Execute the marked source-selection block in `notebooks/Colab_Data_Shards.ipynb` against temporary 4X directory plus 10X and 40X ZIP fixtures. Before the fix, it raised `Choose a source directory or source ZIPs for this run; do not mix both.` The regression path is `pytest -q tests/test_data_shards_notebook.py`.
- **Nguyên nhân cốt lõi (Root Cause):** The notebook adapter imposed a single-source-format rule that did not exist in the CLI or catalog contracts.
- **Giả định sai (Wrong Assumptions):** All magnifications would be stored using the same source packaging format.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** The adapter now appends an optional directory and all sorted ZIP archives, then passes them together to catalog preparation. A notebook-source E2E test confirms mixed inputs work and that duplicate basenames are still rejected by the catalog.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Keep notebook source selection aligned with the CLI's repeated `--source` contract; let the catalog enforce cross-source filename integrity.
