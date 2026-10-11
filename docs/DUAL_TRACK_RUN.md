# Chạy hai study A/B có resume

Dữ liệu nguồn: common16 đã verify, frozen feature v001; không cần nén lại hoặc tải toàn bộ PNG. Source pipeline và notebook mặc định không chạy CNN hoặc fine-tune encoder.

1. Dùng runtime code-only có SHA đã pin và notebook `notebooks/Colab_Dual_Track.ipynb`. Chọn CPU cho preprocessing/A0/A1/B0, gắn Drive trong đúng kernel; người dùng tự duyệt Google trong Chrome của mình.
2. Stage bundle/features/case cache lên SSD Colab một lần. Khai báo config gồm study_id, setup_path, output_root, source_code_sha256, runtime_archive_path. Controller kiểm archive bytes/extracted files, cohort/folds/labels, output tách source/cache.
3. Chạy `preprocess` để commit source-photo map và chín PCA training scopes. Chạy `cpu` để hai worker A/B cùng xử lý, mỗi worker một thread. Stage CPU chỉ là `cpu_complete` khi hai nhánh có final artifacts; B1/báo cáo còn riêng.
4. Sau source/CPU artifacts verified, dừng CPU đúng công việc rồi tạo T4 khi cần. Stage lại cache immutable; giữ đúng study/code/config để dùng model/PCA/bank đã commit. Gắn Drive cho runtime mới nếu Google yêu cầu.
5. Chạy `gpu_smoke` hai lần từ hai process mới. Gate gồm beta-zero parity, full-bag thật thuộc inner-train, checkpoint hoàn tất và new-process resume. Weights smoke không dùng cho study.
6. Chạy `neural`: B1 beta-zero control và beta=.5 theo inner selection. Một lease cho CUDA device, một fit tại một thời điểm, giữ toàn bag. Đánh giá cấp ca và giữ nguyên splits.

Entry point:

```text
python -m histology_data.dual_cli --mode cpu --config /content/dual-config.json --deadline-utc <UTC>
python -m histology_data.dual_cli --mode gpu_smoke --config /content/dual-config.json --deadline-utc <UTC>
python -m histology_data.dual_cli --mode neural --config /content/dual-config.json --deadline-utc <UTC>
```

Config/runtime changes phải dùng identity mới hoặc fail trước ghi. Rerun đọc hash/receipt trước khi bỏ qua việc đã hoàn tất; logistic jobs replay toàn fit nếu chưa commit, MIL tiếp từ epoch đã commit. Không coi status.json đơn lẻ là weights/state hợp lệ.

Recovery cùng host chỉ khi bật `recover_stale_locks=true`, process/session cũ được chứng minh đã chết. Recovery khác VM phải có `recovery_evidence` theo nhánh/lock path, đúng token/host/boot_id/session_id, old_runtime_stopped, bằng chứng và verified_at_utc. Không reclaim lock đang sống hoặc dùng TTL để giả định owner chết.

Nếu worker lỗi, nhánh còn lại có thể tiếp tục khi shared sources vẫn valid; parent báo error trước paused nếu có cả hai. Deadline thiếu thì paused_budget, không complete; giữ committed jobs/checkpoints. Không auto-click/keep-alive giả/SSH/account switching để vượt quota.

Sau khi đã verify artifacts: dừng writers, official Drive flush/unmount, dừng đúng runtime qua colab-manager, kiểm server. Không shutdown máy cá nhân; yêu cầu tự tắt cũ đã hủy.

Đọc [kế hoạch đầy đủ](plans/CLUSTERING_DUAL_TRACK_EXECUTION.md), [A APIs](DUAL_ATLAS.md), [linear study](DUAL_LINEAR_STUDY.md), [B bank/MIL](DUAL_NOVELTY.md) và [B1 neural study](DUAL_NEURAL_STUDY.md).
