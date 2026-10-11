# Vận hành MIL cấp ca

Thiết kế: [kế hoạch train](plans/MIL_TRAINING_PLAN.md), [nguyên lý](plans/MIL_MODEL_EXPLAINED.md), [reliability gates](plans/COLAB_TRAINING_RELIABILITY.md). Code là trainer mới từ frozen feature bundles, không là trainer patch-level cũ.

## Các mode và phạm vi

`python scripts/run_mil.py --mode <mode> --config <file.json> [--deadline-utc <ISO UTC>]`

- `toy`: tạo fixture tổng hợp có signal, nhãn ca, tối đa 16 ca/64 rows/lens; fixture chưa đủ điều kiện production.
- `export`: export 2–4 ca thuộc một inner-train partition, không lấy outer test; public loader/full bundle verification và provenance refs giữ nguyên.
- `smoke`: một train/validation loop có cap, default 2 epochs, max20 steps, max10 phút, một ca/batch; CPU local hoặc CUDA. Validation smoke có thể một lớp để kiểm cơ chế trên fixture nhỏ, không tính clinical metrics. Production vẫn yêu cầu cả hai lớp.
- `pilot`: full-bag CUDA, một/two epoch của inner fit chứa ca lớn nhất outer0-train; không dùng outer-test và không tái dùng weights vào production. Deadline bắt buộc.
- `production`: nested common 3 outer ×2 inner, E0–E4 mặc định, 45 fits, CUDA, deadline bắt buộc. Không dùng smoke fixtures hoặc CPU real full study. Study all chưa là control hiện hành.

Exit 0: complete/fixture_complete; exit 2: dừng budget/step cap; exit 1: lỗi. Các mode không tạo/mount/tắt runtime hoặc tự đổi account; notebook/kernel và lifecycle cần điều phối bên ngoài theo quyền người dùng.

## Model và preprocessing

Không tải CNN trong MIL fit. E0: mean từng lens 2048 chiều + ba mask, linear6147→1, dropout0 trong study. E1/E2/E3: shared projection2048→128, gated attention64 của lens chọn, linear128→1. E4: projection dùng chung, ba attention riêng, concat387→1. Dropout0,25 ở vector pooled trước classifier của attention variants; score cuối attention không bias. Schema model/run lưu đầy đủ; attention không có Transformer pairwise hoặc spatial positional encoding.

Đầu vào float32, đúng lens/mask, nhãn ca0/1. Nguồn read-only được copy đúng một ca tại boundary NumPy→Torch. Không thay stain/geometry/Glade, không broadcast target thành patch labels. Study dựa cache resize256/center224 ImageNetV1 đã verify; ảnh source không sửa.

## Config tối thiểu để thử local

Toy: `{"output":"/tmp/mil-toy","case_count":4,"rows":16,"seed":42}`.

Smoke: `{"fixture":"/tmp/mil-toy","output":"/tmp/mil-smoke","device":"cpu","variant":"E4","epochs":2}`.

Run toy một lần rồi smoke. Chạy lại cùng smoke config phải load completed fit, không tạo epoch/optimizer step mới. Đổi model/seed/data/code/environment phải dùng output mới; mismatch không được ghi đè config cũ.

Real export config: bundle path, feature_root, output fixture path, outer0/inner0, max_cases4/max_rows64. Fixture dùng smoke flag và source hashes/IDs/tile refs. Không đưa báo cáo ca hoặc identifier vào tensor.

Production config: `bundle`, `feature_root`, `output`, `variants` (defaultE0..E4), `epochs`30, `patience`5; luôn có deadline. Output phải ở Drive. Stage input features lên SSD qua `stage_training_inputs`, không stage PNG/ZIP.

## Cache bag đầy đủ theo ca sau benchmark

T4 pilot thực tế: hai epoch full bag khoảng 57 giây/epoch; peak VRAM reserved0,64 GiB, RSS1,66 GiB. Profile một ca inner-train 19.736 instances: public verified loader9,437 giây, copy/transfer0,094 giây, forward/backward/step0,028 giây. Đây là số đo kỹ thuật, không là clinical performance.

`prepare_case_cache` kiểm nguồn đầy đủ rồi dùng public loader cho từng ca một lần để materialize ba ma trận và thứ tự tile. Cache ở SSD Colab, bound với bundle/feature/split/source hashes, giữ mọi vector và dtype/tile order; không sampling, re-encode hoặc học tham số từ test. `CacheDataset` kiểm manifest/commit/checksum/schema/case identity từng lần đọc; study vẫn full-verify nguồn khi mở run. Cache là snapshot bất biến của phiên bản đã verify, không phản ánh thay đổi nguồn sau snapshot.

CLI pilot/production nhận `case_cache` path tùy chọn; notebook tự chuẩn bị cache. Namespace outputs thêm runtime SHA để không trộn checkpoint code cũ/mới. Cache chỉ giảm kiểm lặp toàn feature parts; không bỏ kiểm toàn vẹn payload được train. Phải qua byte-equality và adversarial tests, CI, rồi benchmark lại trên T4 trước full study.

## Checkpoint và resume

Lưu baseline epoch0 trước optimizer step đầu. Mỗi epoch train/validation hoàn tất lưu state model+optimizer+RNG+global step+history+early-stop; receipt xác nhận SHA/identity mới là commit. Giữ tất cả phiên bản committed trong run; `best` để chọn inner epoch, `latest` để tiếp tục. State schema bound với data/fold/config/source code/environment.

Ngắt giữa epoch: process mới load committed epoch và replay epoch dở. Không tiếp tục optimizer state đã cập nhật một phần trong RAM. Deadline validation báo paused_budget và không commit epoch dở. Nếu payload đã commit nhưng pointer ghi lỗi, báo error cùng epoch bền đọc từ manifests; không gọi committed epoch là mất.

Lock lưu PID/start/host/boot/session/token. `recover_stale_lock` chỉ cho phép takeover owner đã chứng minh chết trên cùng host. Khác host cần `lock_recovery_evidence` (smoke) hoặc mapping `recovery_evidence` (production, keys `study` / relative fit directory), gắn chính xác metadata lock và bằng chứng runtime cũ đã dừng. Không dùng TTL hoặc giả rằng mất websocket nghĩa là VM đã dừng.

Số học CPU/GPU có thể khác; verify resume trong cùng environment. Model eval trước/sau save-load phải khớp. Thay environment bị identity gate từ chối; cần compatibility review/run version riêng, không coi reset RNG hay load weights đơn thuần là resume đầy đủ.

## Output

- Per-fit: resolved config, events/status, immutable checkpoints. Inner `best_epoch`; outer refit số epoch median làm tròn lên, weights mới.
- Per-outer refit: committed predictions và attention arrays/tile references theo lens. SVG minh họa điểm gốc patch trong hệ tọa độ ảnh nguồn, không phải pixel overlay/segmentation; weights là tương đối trong bag.
- Study: OOF CSV, từng fold và pooled metrics/counts, fixed-OOF patient bootstrap, report/model card. Không dùng outer scores để chọn winner rồi gọi chính điểm đó là đánh giá độc lập.
- Study `complete` khác với nghiệm thu GPU/Drive: phải có bằng chứng full real-data run; local synthetic 9-fit E2E không là clinical performance.

## Môi trường và vận hành

Local dùng venv CPU, Torch2.9.0 được kiểm thử; sklearn1.8.0 giữ contract splits. Runtime Colab giữ Torch CUDA sẵn có, ghi environment và kiểm GPU smoke trước full run; không tự cài đè Torch/CUDA. Nếu dependency không đáp ứng thì preflight dừng rõ ràng.

Native notebook là đường chạy cho Colab Free, không SSH/proxy hoặc keep-alive giả. Mount đúng kernel và user duyệt quyền khi Google yêu cầu. CLI chỉ điều phối notebook/kernel; không bảo đảm runtime sống hoặc đủ quota. Stage feature parts, record deadline từ thời điểm runtime, chừa 30 phút trong budget330 của dự án. Run all không được reset ngân sách session. Chưa benchmark thật thì không cam kết fits/giờ.

## Kiểm chứng

Tests `tests/test_mil_*.py`: model contracts; filesystem-backed checkpoint/lock failures; synthetic governed bundle/export; CPU loop; actual subprocess CLI; actual abrupt child exit23 và resume; nested 9-fit synthetic study. Reproduction logs trong Results/proofs (không commit data/weights).

Code sửa lỗi theo E2E protocol và [bài học MIL](../.agents/skills/histology-mil/references/failures.md); conveyor/CI trước release. Không suy từ CPU/synthetic tests rằng CUDA, Google Drive/FUSE hoặc full cohort đã qua.
