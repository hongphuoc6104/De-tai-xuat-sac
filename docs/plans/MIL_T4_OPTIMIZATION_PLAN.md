# Thiết kế tối ưu MIL trên Colab T4

> Nghiên cứu đề xuất · 11/10/2026. Chỉ đọc mã nguồn và thiết kế thí nghiệm; chưa sửa code, chưa chạy benchmark/training. Các tốc độ và ảnh hưởng accuracy dưới đây là giả thuyết cho tới khi đo trên cùng T4 và cùng fold.

## Kết luận ưu tiên

Phần đáng tối ưu trước là đường dữ liệu và truyền tensor, không phải nén trọng số MIL: E4 chỉ có **312.388 tham số FP32 (1,192 MiB)**, còn feature common là **146.506 × 2.048 × 4 = 1.200.177.152 byte (~1,118 GiB)** trước index; bộ staged cache đo **1,382 GiB** gồm index. Do đó nén model vài trăm KiB khó làm thay đổi đáng kể VRAM hoặc thời gian tổng.

Thứ tự thí nghiệm đề xuất: (1) cache bag đầy đủ trên SSD đã qua integrity gate; đo lại với **cache FP32 resident trên GPU theo fit/fold**; (2) AMP FP16 có vùng tính toán nhạy giữ FP32; (3) giảm chiều E4 128→64 riêng một nhánh; (4) thử cache feature FP16 như một phiên bản dữ liệu mới. Chỉ thử một yếu tố mỗi lần. Không áp dụng thay đổi vào 45-fit baseline đang được quản lý.

## Bằng chứng và giới hạn hiện tại

- Mã nguồn tại commit được chỉ định khai báo E0=6.148, E1–E3=278.977 và E4=312.388 tham số. E4 đang chiếu 2.048→128, attention hidden=64; các feature là frozen ResNet50, không train CNN.
- Có **một pilot cache full-bag gồm hai epoch** trên cùng inner fold có ca lớn nhất (19.736 instances): thời lượng từng epoch là 4,026 s và 3,272 s; peak reserved 0,641 GiB. Pilot cũ không dùng cache, cùng fold/ca lớn nhất, mất 57,104 s và 56,594 s cho hai epoch tương ứng — mức giảm thời gian quan sát khoảng 14,2× và 17,3×. Đây là so sánh cache, không phải kết quả AMP hay hai lượt chạy cache độc lập.
- Profile riêng trước cache của một ca 19.736 instance ghi loader 9,437 s và GPU step 0,028 s. Không gộp phép đo này vào phép so sánh hai pilot ở trên; profile GPU compute của cache mới chưa được tách/đo.
- T4 là Turing, compute capability 7.5. Tensor Core FP16 là mục tiêu phù hợp; BF16/TF32 không có trong ma trận hỗ trợ cho CC 7.5. Structured sparsity 2:4 là tính năng Ampere, không phải đường tăng tốc trên T4. Công suất đỉnh của datasheet không dự đoán tốc độ thực của bag nhỏ và nhiều thao tác điều phối.

## Các nhánh tối ưu

| Ưu tiên | Thử nghiệm | Lý do / rủi ro | Điều kiện giữ lại |
|---|---|---|---|
| P0 | Nạp tensor **FP32 đầy đủ** của đúng fit lên GPU một lần, tái sử dụng qua các epoch; chỉ nạp train và inner-validation. Outer test chỉ được nạp sau khi lựa chọn/weights đã khóa. | Loại copy host→device lặp lại, không đổi phép tính, feature hay số instance. Tổng common raw khoảng 1,118 GiB; có khả năng vừa T4 nhưng cần đo cả context, workspace, allocation/reserved và bộ nhớ còn lại thực tế. | Hash/ID/tile order/count giữ nguyên; so tensor với nguồn đã verify; full-bag, không sampling; peak và headroom qua gate ở dưới; có cải thiện thời gian fit. |
| P1 | AMP `autocast(float16)` cho các Linear/projection/attention; giữ model weights và AdamW state FP32, dùng `GradScaler`. | T4 có FP16 Tensor Cores; phép chiếu 2.048→128 là phần tính toán lớn nhất theo số instance. AMP có thể không tăng tốc nếu sau cache workload bị giới hạn bởi launch/sync hoặc bags quá ngắn. | Có finite checks, scaler hoạt động và resume đúng; time-to-fit giảm trên cùng cache và paired fold; inner validation nằm trong tolerance đã khóa trước. |
| P2 | E4 projection 128→64 và attention hidden 64→32, thành biến thể/identity riêng. | Giảm xấp xỉ 54% trọng số E4 (312.388→143.908; FP32 khoảng 1,192→0,549 MiB) và gần một nửa phép chiếu theo instance. Đây là đổi kiến trúc nên có thể đổi accuracy. | Đánh giá bằng inner folds; full outer test vẫn khóa đến khi chọn xong. Báo đầy đủ từng fold, không suy luận “bảo toàn accuracy” từ một smoke. |
| P3 | Phiên bản feature cache FP16, chỉ sau khi đã có mốc GPU-resident FP32. | Raw feature footprint giảm một nửa (ước tính common ~0,559 GiB), nhưng lượng tử hóa embedding có thể đổi attention/logit. Không sửa bundle FP32 nguồn. | Cache có schema/hash/version riêng; so từng feature sau chuyển đổi và output eval với cùng weights; paired train/validation; chỉ giữ nếu lợi ích đáng kể và validation trong tolerance. |

**Cổng VRAM cần bổ sung trước khi triển khai:** giới hạn `memory_limit_bytes` hiện kiểm RSS của process, không chặn GPU allocation. Runtime tương lai cần preflight `mem_get_info` và giới hạn VRAM riêng được pin vào fit identity; từ chối nạp cache nếu dung lượng trống sau khi tính cache/workspace không đạt headroom đã định. Đây là yêu cầu thiết kế, chưa triển khai.

## Cơ hội riêng cho E0–E3 (chưa đo)

- **E0:** `_mean_pool` hiện tính `matrix.mean(dim=0)` trên GPU mỗi lần lặp. Có thể tính đúng phép giảm FP32 này một lần cho mỗi case/lens trong fold, rồi cache **3 × 2.048 chiều = 6.144D means + presence mask 3 lens** để head tuyến tính dùng lại. Lens thiếu giữ zero mean, count=0, mask=false và attention rỗng như hiện tại. Giữ số instance và tile refs/counts ở metadata để dựng attention uniform `1/N` và bản đồ đầu ra như cũ; không coi means cache là thay thế provenance. Cổng parity phải so cached means với kết quả `_mean_pool` trên cùng FP32 input/GPU, với tolerance số học đã chốt trước. Không precompute outer-test trước khi model/config được khóa.
- **E1/E2/E3:** các biến thể chỉ dùng một lens, nhưng `_case_tensors` hiện chuyển mọi lens của ca sang device. Có thể giữ bước đọc/xác minh ca đầy đủ, chỉ chuyển tensor lens mà biến thể cần qua H2D. Hợp đồng `MILModel` cho phép map chỉ chứa lens được dùng nếu bỏ `presence_mask` toàn ca khi gọi model; model vẫn tự kiểm dtype/shape/finiteness và yêu cầu lens đó không rỗng. Giữ mask gốc, counts và refs trong metadata của case; kiểm thử adapter để chứng minh tensor còn lại không lên GPU, không bỏ integrity gate hoặc làm mất khả năng dựng outputs.

Đây là đề xuất từ cấu trúc mã (`mil_model.py`, `mil_training.py`), chưa có benchmark hay thay đổi source/cache.

### Vì sao chưa ưu tiên pruning, INT8/INT4 và distillation

- **Weight pruning/Deep Compression:** giảm file trọng số chủ yếu có ích với mạng lớn. Ở đây E4 chỉ 1,192 MiB; pruning thưa không tự tăng tốc phép nhân dense và 2:4 không có Sparse Tensor Core trên T4. Không ưu tiên cho thời gian train.
- **INT8/INT4 hoặc quantization-aware training:** T4 có các chế độ Tensor Core cho suy luận lượng tử, nhưng model nhỏ và baseline đang train FP32/AMP; bước chuẩn bị/QAT có thể tốn hơn phần tiết kiệm. Chỉ xem xét cho triển khai inference sau khi accuracy đã khóa.
- **Knowledge distillation:** có thể chuyển hành vi teacher sang head nhỏ, nhưng phải train teacher trước; với 16 bệnh nhân, soft target từ cùng ít ca dễ thêm phương sai và không rút ngắn baseline 45 fits. Chỉ nghiên cứu riêng nếu deployment cần một head nhỏ hơn, dùng teacher/out-of-fold protocol hoàn toàn trong training folds.

## Benchmark có giới hạn và cổng quyết định

Không chạy trên T4 đang phục vụ baseline. Sau khi chủ sở hữu baseline chọn cửa sổ tài nguyên, pilot sàng lọc dùng **một inner fold đại diện**, chỉ dùng train/validation đã khóa; tối đa 2 epoch cho mỗi cấu hình. Pilot này chỉ sàng lọc integrity, số học, memory và throughput; không đủ để chấp nhận chất lượng hoặc khẳng định accuracy. Không đọc outer-test để điều chỉnh. Lệnh/runner phải dùng fit identity và output tạm riêng, không ghi vào checkpoint baseline.

1. **Mốc FP32:** cùng source, cache, fold, case order, seed và model E4 hiện tại; đo riêng thời gian stage, load, H2D, forward/backward, validation và checkpoint. Warm-up tách khỏi thời gian. Đồng bộ CUDA quanh timer. Lặp ba lượt ngắn nếu budget cho phép; báo median và từng lượt.
2. **P0 FP32 GPU cache:** cùng dữ liệu/mô hình/số bước; so ID, số instance, tile order, masks và tensor values với loader đã verify. Đo `max_memory_allocated`, `max_memory_reserved`, bộ nhớ trống và RSS. Chỉ lưu nếu peak ổn định, còn tối thiểu 25% VRAM của thiết bị được cấp làm dự phòng, không OOM, và median epoch nhanh hơn ít nhất 15% so mốc. Nếu chưa đạt ngưỡng, giữ cache SSD.
3. **P1 AMP:** pilot paired dùng cùng seed, fold, full bags và cache FP32 để sàng lọc số học/throughput. Chỉ cho phép FP16 ở phép nhân tuyến tính được autocast; đưa softmax attention, tổng có trọng số/reduction, logit cuối và BCEWithLogits về FP32. Sau backward phải unscale gradient trước finite check; scaler step/update và state phải được checkpoint/resume cùng optimizer. Ngưỡng pilot về throughput là nhanh hơn ít nhất 15%; đây không phải cổng accuracy.
4. **P2/P3:** chỉ thử từng nhánh một khi P0/P1 qua sàng lọc kỹ thuật. Sau đó, đánh giá mọi ứng viên có khả năng được chọn bằng các lượt dài theo protocol trên **tất cả inner folds**. So sánh chất lượng chỉ trên các validation folds; đề xuất ngưỡng engineering là không quá **+0,01 BCE tuyệt đối ở bất kỳ inner fold nào** so với FP32 control, khóa trước khi xem kết quả. Nếu chọn architecture/precision/cache, lựa chọn đó chỉ dùng inner results; outer-test/OOF không được dùng để tune và chỉ mở sau khi khóa cấu hình. Với P3, lưu cache mới bên cạnh bản FP32 và giữ nguồn FP32 làm chuẩn. Không đổi epochs, sampling, cohort, preprocessing hoặc folds để bù một nhánh chậm/giảm điểm.

Mỗi nhánh phải dừng ngay nếu thiếu case/patch, mismatch hash/schema/fold, NaN/Inf, scaler bỏ optimizer step liên tục, OOM hoặc peak vượt headroom. Pilot hai epoch chỉ là evidence kỹ thuật/số học. Cổng quality yêu cầu các lượt dài qua toàn bộ inner folds; nested CV với outer-test chỉ đánh giá biến thể đã khóa, không quay lại dùng OOF làm lựa chọn. Trên 16 bệnh nhân, không có cổng nhỏ nào bảo đảm accuracy ngoài mẫu.

## Nguồn chính

- [Micikevicius et al., *Mixed Precision Training* (2017)](https://arxiv.org/abs/1710.03740): FP32 master weights/loss scaling và giới hạn động của FP16.
- [PyTorch AMP reference](https://docs.pytorch.org/docs/stable/amp.html) và [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html): autocast, GradScaler, op policy; scaler state cần được lưu cùng model/optimizer khi resume.
- [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) và [CUDA compute capability table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html): T4/Turing, CC 7.5, FP16 Tensor Core.
- [NVIDIA Ampere structured sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/): 2:4 sparse Tensor Core là Ampere.
- [Jacob et al., CVPR 2018 quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html), [Han et al., Deep Compression](https://arxiv.org/abs/1510.00149), [Hinton et al., distillation](https://arxiv.org/abs/1503.02531): nguồn phương pháp; các kết quả của họ không phải bảo đảm cho MIL/histology cohort này.
