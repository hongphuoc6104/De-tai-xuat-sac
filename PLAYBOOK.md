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

### [ERR-20261010-01] Smoke precut lấy nhãn đầu tiên nhiều lần thay vì luân phiên lớp
- **Triệu chứng (Symptom):** Mẫu vài patch mỗi vật kính chỉ chọn case thuộc một nhóm kết luận dù cả hai nhóm có ảnh.
- **Tái hiện E2E (Reproduction Path):** `python -m pytest tests/test_precut.py::test_precut_smoke_reads_examples_per_lens_and_keeps_case_labels_weak -q`; trước sửa chỉ nhận `YCT26_1` thay vì cả hai case.
- **Nguyên nhân cốt lõi (Root Cause):** Danh sách case được ghép hết trong một nhóm nhãn trước khi chuyển nhóm; giới hạn mẫu cắt danh sách quá sớm.
- **Giả định sai (Wrong Assumptions):** Sắp xếp theo nhãn rồi giới hạn được coi là đủ đại diện cho smoke kỹ thuật.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Chọn luân phiên giữa nhóm nhãn và case cho từng vật kính. Test tái hiện và toàn bộ `tests/test_precut.py` qua; smoke thật giữ hai patch mỗi vật kính và hai nhóm nhãn ứng viên.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Khi giới hạn smoke có yếu tố phân nhóm, chọn vòng qua nhóm trước khi áp giới hạn; kiểm tra cả coverage lẫn định danh mẫu.

Chi tiết đã chuyển vào [.agents/skills/histology-shards/references/failures.md](.agents/skills/histology-shards/references/failures.md).

### [ERR-20261010-02] Notebook Colab import package trước khi giải nén runtime
- **Triệu chứng (Symptom):** Trạm 2 chạy 98/99 test, nhưng test lint notebook fail vì import sau setup, import trùng giữa cell và nhiều import một dòng.
- **Tái hiện E2E (Reproduction Path):** `./pipeline station 2`; bằng chứng `Results/proofs/test_evidence_20261010_131312.log` ghi 10 lỗi Ruff trong notebook.
- **Nguyên nhân cốt lõi (Root Cause):** Notebook load package sau khi cài runtime; Ruff kiểm tra các cell và phát hiện thứ tự/import lặp.
- **Giả định sai (Wrong Assumptions):** Import sau cài package được coi là ngoại lệ khỏi lint cell.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Dùng `importlib.import_module` sau khi thêm runtime vào đường dẫn, sắp imports và dùng chung symbols giữa cell. Ruff notebook và test Trạm 4 qua; chạy lại toàn bộ station 2.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Lint toàn notebook; import runtime muộn qua importlib và tránh khai báo lại imports giữa các cell.

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


### [ERR-20261010-03] Báo cáo thời gian giả định commit cũ có trường tùy chọn mới
- **Triệu chứng (Symptom):** Lệnh thống kê thời gian đóng ZIP trên Colab gặp `KeyError: prefetch_workers`; công việc đóng ZIP vẫn tiếp tục bình thường.
- **Tái hiện E2E (Reproduction Path):** Chạy `histology_repack_timing_summary.py` trên runtime đã có commit phần 001 từ lệnh benchmark đầu tiên; commit này chưa có trường `prefetch_workers`.
- **Nguyên nhân cốt lõi (Root Cause):** Helper báo cáo đọc nhiều phiên bản commit vận hành như cùng một schema.
- **Giả định sai (Wrong Assumptions):** Trường đo tối ưu I/O được coi là bắt buộc ở mọi artifact đã hoàn tất trước đó.
- **Giải pháp & Kiểm chứng (Resolution & Proof):** Đọc trường tùy chọn bằng `get`, chỉ tổng hợp nhóm có số luồng được ghi rõ. Chạy lại cùng runtime trả thống kê 6 phần và median 32 luồng 151,585 giây; không sửa commit nguồn hoặc khởi động lại job. Lệnh và kết quả lưu trong `Results/planning_20261010/colab_setup/commands/`.
- **Quy tắc phòng ngừa vàng (Golden Prevention Rule):** Tách trường kiểm toàn vẹn bắt buộc khỏi trường đo hiệu năng tùy chọn; không suy diễn giá trị thiếu và không sửa artifact cũ chỉ để báo cáo chạy được.

- [ERR-20261010-04..07] Feature integrity/resume, budget/completeness, cold notebook và ZIP lookup: xem [.agents/skills/histology-features/SKILL.md](.agents/skills/histology-features/SKILL.md).


### [ERR-20261010-08] Hai kết nối SSH đồng thời bị Colab từ chối
- **Triệu chứng (Symptom):** Kết nối rsync thứ hai trả HTTP429 Already-active SSH session; lần truyền thứ nhất vẫn thành công.
- **Tái hiện E2E (Reproduction Path):** Hai lệnh rsync tới cùng VM histology-features-resume-20261010 chạy chồng nhau khi tải các artifact review; shell session26741 trả255, session65589 hoàn tất.
- **Nguyên nhân cốt lõi (Root Cause):** SSH endpoint của runtime chỉ cho một kết nối active tại một thời điểm; các file đích khác nhau không làm kết nối trở thành độc lập.
- **Giả định sai (Wrong Assumptions):** Upload các file khác nhau qua SSH có thể song song như file API.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Chờ kết nối đầu đóng rồi chạy lại rsync thứ hai; session76582 exit0. Không dừng VM/encoder hoặc thay key. Xác nhận artifacts nhỏ trên Drive sau transfer.
- **Quy tắc phòng ngừa vàng (Golden Rule):** Mỗi Colab endpoint chỉ mở một SSH; thao tác nhiều file trong một lượt hoặc dùng file API cho artifact nhỏ, tránh in proxy URL khi backend trả lỗi.

### [ERR-20261010-10] Test ghi đè derived defaults làm che đường dẫn notebook sai
- **Triệu chứng (Symptom):** Notebook tìm runtime/runtime thay vì runtime; build dùng thêm pretrain dù builder tự nối bundles.
- **Tái hiện E2E (Reproduction Path):** Actual config/helper/main cells chỉ bind ROOT và expectedSHA vào fixture có ZIP ởROOT/runtime; before trace `Results/proofs/pretrain_notebook_defaults_repro_20261010.json`.
- **Nguyên nhân cốt lõi (Root Cause):** Default folder bị nối hai lần; test helper ghi đè RUNTIME_ZIP/BUNDLE_OUTPUT_ROOT nên không kiểm derivation thực tế.
- **Giả định sai (Wrong Assumptions):** Kiểm wrapper sau khi thay mọi đường dẫn tương đương chạy cấu hình người dùng.
- **Giải pháp & Kiểm chứng (Resolution & E2E Proof):** Một runtime segment, build output-rootROOT; tests dùng actual defaults với ROOT/SHA binding, actual CLI draft E2E và17 notebook tests qua.
- **Quy tắc phòng ngừa vàng (Golden Rule):** E2E giữ derived defaults của notebook; override chỉ những input môi trường cần thiết, không sửa giá trị muốn kiểm.

- [ERR-20261010-09,11,12] Split mixed-label patients, bag persistence contract và CLI constant ownership: xem [.agents/skills/histology-pretrain/references/failures.md](.agents/skills/histology-pretrain/references/failures.md).
