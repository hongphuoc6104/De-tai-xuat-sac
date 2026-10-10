---
name: histology-features
description: Trích vector từ PNG đã duyệt bằng encoder frozen, có kiểm checksum, source snapshot, budget và resume; dùng khi sửa extractor hoặc hướng dẫn Colab.
---

# Histology feature extraction

Đọc `docs/FEATURE_EXTRACTION.md` để vận hành và `histology_data/features.py` cho schema thực tế. Nguồn có một mode ZIP hoặc directory; không tạo lại ZIP dữ liệu. Frozen ResNet50 ImageNet V1 phải có full SHA `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`. RGB/resize256/center-crop224/ImageNet mean/std là preprocessing cố định cho tensor, không đổi PNG nguồn hoặc fit stain statistics.

- Feature identity gồm weights SHA, preprocessing, precision, Torch/Torchvision/Pillow versions và source-code SHA; runtime paths/batch/time budget đi riêng.
- Part ID độc lập với inventory tổng/đường dẫn mount; thêm ZIP không làm tên part cũ thay đổi. Mỗi source snapshot compact có fingerprint và danh mục parts; không ghi toàn pixel/cohort vào RAM.
- Publish features/index/QC bằng copy verified, commit cuối cùng. Resume xác minh outputs và SHA nguồn, không stage/decode/encode lại part đã commit. Budget phải kiểm trong hash prepass và final audit, không chỉ trước part mới.
- `complete` chỉ sau final audit; trạng thái `auditing` chưa phải hoàn tất. Partial run không được ghi đè manifest complete cũ; giữ snapshot từng run và source-plan snapshot referenced.
- Smoke chọn tối đa số parts đã khóa và interleave vật kính; rerun không được tăng phạm vi smoke. Capped output luôn training-ready false.
- Một writer/output. Foreign-runtime stale lock chỉ recover sau khi operator đã dừng runtime cũ; không dùng recover để chiếm writer sống. File guard không thay thế quy trình một writer khi dùng nhiều Colab.
- Giữ raw_glade/case candidates/patient_id=None/bag_id=None. Không supervised patch targets, không chuyển tự động thành ISUP. Scientific train cần patient mapping, nhãn cấp ca và splits riêng.
- Full compute chỉ Colab. Local pixel smoke giới hạn trước bytes/hash/decode; dùng đúng sáu PNG thật hoặc fixture nhỏ. Không tạo GPU VM khi chỉ phát triển/test.

## [ERR-20261010-04] Resume mất completeness hoặc tái sử dụng nguồn thay đổi

- **Symptom:** E2E complete→partial ghi đè release cũ; PNG khác bytes nhưng cùng size/mtime được reuse; smoke lặp tăng part count.
- **Reproduction Path:** `python -m pytest tests/test_features.py` với `test_complete_release_survives_temporarily_missing_source`, `test_directory_same_size_and_mtime_change_is_not_reused`, `test_smoke_repeat_stays_bounded_and_does_not_become_full`; before log `Results/proofs/feature_reproduction_before.log`.
- **Root Cause:** Aggregate plan được dùng như phạm vi release hiện tại, source reuse chỉ tin stat, cap mới theo từng invocation.
- **Wrong Assumptions:** Metadata filesystem chứng minh byte identity; subset hiện tại có thể thay thế manifest hoàn chỉnh; cap per-run tương đương cap smoke tổng.
- **Resolution & Proof:** Source SHA trên resume, immutable complete manifest + partial/run snapshots, selection smoke persisted. Các E2E hồi quy qua. Root/ZIP migration không làm encode lại khi byte/config giữ nguyên.
- **Golden Prevention Rule:** Identity/source integrity/completeness/cap là bốn hợp đồng riêng; không suy diễn từ số file hoặc mtime.

## [ERR-20261010-05] Hoàn tất/budget/khóa writer chưa được kiểm tới cuối luồng

- **Symptom:** Resume SHA vượt deadline; status complete xuất hiện trước final audit; manifest tự nhất quán nhưng thiếu parts vẫn được báo complete; recover có thể chiếm PID đang sống.
- **Reproduction Path:** `test_resume_hash_validation_obeys_deadline`, `test_audit_failure_never_publishes_terminal_complete`, `test_consistent_truncated_release_cannot_claim_complete`, `test_recovery_cannot_take_lock_from_known_live_process`; before log `Results/proofs/feature_completion_reproduction_before.log`.
- **Root Cause:** Deadline chỉ bọc new computation, audit tin count của release, recovery không phân biệt writer có thể xác minh đang sống.
- **Wrong Assumptions:** Bước hash/audit không đáng kể; manifest complete tự chứng minh phạm vi; recovery flag có thể bỏ mọi lock.
- **Resolution & Proof:** Bounded hashing/deadline trong prepass/audit, source-plan referenced và count/IDs cross-check, auditing phase và commit terminal sau audit, refuse known live writer. Suite E2E mới qua; metadata/data không bị sửa khi stopped.
- **Golden Prevention Rule:** Mọi I/O dài thuộc budget; marker terminal là giao dịch cuối sau bằng chứng, không phải trước nó.

## [ERR-20261010-06] Notebook cold environment bị lỗi namespace Google

- **Symptom:** Thực thi tất cả code cells trong CPU venv dừng ngay với `ModuleNotFoundError: google` dù hỗ trợ local.
- **Reproduction Path:** Notebook real-six fixture; `Results/proofs/feature_notebook_e2e.log`; regression `test_first_cell_short_circuits_when_google_parent_is_missing` trong notebook suite.
- **Root Cause:** `find_spec('google.colab')` cần parent namespace tồn tại.
- **Wrong Assumptions:** Tìm module con luôn trả None nếu parent không có.
- **Resolution & Proof:** Short-circuit kiểm `find_spec('google')` trước child. Toàn bộ notebook code cells chạy trên sáu PNG thật; after log `Results/proofs/feature_notebook_e2e_after.log`.
- **Golden Prevention Rule:** Kiểm parent namespace trước optional child; compile/AST không thay actual notebook execution.

## [ERR-20261010-07] Lookup ZIP theo mỗi patch quét lại central directory

- **Symptom:** `infolist()` lặp cho mọi payload làm độ phức tạp O(N²)/ZIP.
- **Reproduction Path:** Instrumentation E2E trong `tests/test_feature_sources.py` forbids central scans during staged payload reads; trước đổi helper cũ tái hiện fail.
- **Root Cause:** Tìm member bằng list scan dù central directory đã kiểm uniqueness.
- **Wrong Assumptions:** Lookup vài nghìn member không ảnh hưởng budget của 149k patch.
- **Resolution & Proof:** `ZipFile.getinfo()` trên handle đã mở; CRC/size vẫn checked. Source suite qua.
- **Golden Prevention Rule:** Kiểm directory một lần; payload lookup O(1), bounded read/decode từng batch.

- Upload chưa đủ/ZIP chưa có central directory là `awaiting_sources` khi không có lỗi khác; test `test_only_incomplete_uploads_wait_without_loading_or_calling_encoder` tái hiện `error` trước sửa và qua sau sửa. Không load weights/encoder khi chưa có part hợp lệ. Khi upload đã hoàn tất mà vẫn như vậy, kiểm file nguồn thay vì ép complete.
