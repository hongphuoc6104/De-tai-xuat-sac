# Mô hình MIL đa vật kính: thiết kế và nguyên lý huấn luyện

Ngày đối chiếu: 11/10/2026. Đây là phụ lục kỹ thuật của [kế hoạch train MIL](MIL_TRAINING_PLAN.md), giải thích thiết kế vòng đầu, không phải báo cáo mô hình đã train. Tham chiếu: [kế hoạch tổng quát](MIL_MULTISCALE_REFERENCE.md) và [báo cáo nghiệm thu](../../Results/planning_20261010/pretrain/final_verified_20261011/BAO_CAO_HOAN_TAT.md).

## 1. Chính xác ta sẽ train mô hình gì?

**Frozen ResNet50 ImageNet V1 + shared projection + gated Attention-MIL riêng cho 4×/10×/40× + linear fusion cấp ca.** Đây là mô hình học có giám sát yếu: biết nhãn ca nhưng không biết nhãn từng patch.

- Encoder và 148.991 vectors 2.048 chiều đã có; chưa triển khai/nghiệm thu trainer MIL hoặc tạo weights MIL đã học.
- Nghiên cứu chính: common, 16 người/16 ca đủ ba vật kính, 9 lành và 7 ung thư; 146.506 instance được chọn.
- Nghiên cứu bổ sung: all, 18 người/18 ca, 10 lành và 8 ung thư; 148.990 instance; có vật kính thiếu.
- Một patch trùng bị bỏ khỏi tham chiếu bag; file và vector gốc vẫn giữ.
- Target: có CARCINÔM → 1; chỉ tăng sản lành tính/kèm viêm → 0. Glade giữ nguyên, không làm target của nghiên cứu này.

148.991 patch không phải 148.991 nhãn độc lập. Năng lực học có giám sát hiện bị giới hạn bởi 16 hoặc 18 người bệnh.

## 2. Tại sao là MIL, tại sao đóng băng?

Trong ca ung thư có thể có mô lành, mô đệm và vùng tổn thương. Gán nhãn 1 cho tất cả patch của ca sẽ tạo nhãn patch sai. MIL học quan hệ giữa tập patch và nhãn chung. Một instance là một patch/vector; một bag nhánh là tất cả vector của một ca tại một vật kính; một mẫu supervised của mô hình fusion là toàn ca.

Đóng băng CNN và tái sử dụng features giúp mỗi lượt thử không phải đọc lại hàng chục GB PNG và chạy lại CNN. Phần MIL nhỏ hơn có thể dùng một ca mỗi batch trên T4. Cách này hợp giới hạn phiên Colab và số người ít. Đây là lựa chọn baseline thực dụng, chưa phải bằng chứng encoder ImageNet tốt nhất cho mô học.

Các patch được đưa vào phải chứa thông tin liên quan tới kết luận ca. Nếu các trường nhìn đã chọn bỏ hết tổn thương thì nhãn ca không đủ giúp mô hình khôi phục thông tin không có trong ảnh.

## 3. ResNet50 thực sự làm gì?

Patch → RGB → resize cạnh ngắn 256 → center crop 224×224 → scale [0,1] và ImageNet mean/std → ResNet50 bỏ classifier 1.000 lớp → global average pooling → 2.048 số.

CNN dùng convolution để biến các mẫu màu/cạnh/kết cấu thành biểu diễn phức tạp hơn. Khối residual có dạng y = F(x) + x, hoặc cộng với projection khi kích thước thay đổi, giúp truyền thông tin và gradient qua mạng sâu. Không được khẳng định một chiều feature cụ thể là “độ ung thư” hay “số nhân”: mỗi chiều là đáp ứng đã học; kiến thức từ ImageNet chưa đồng nghĩa đã hiểu bệnh học.

Global average pooling trung bình từng kênh trên không gian của một patch. MIL pooling sau đó tổng hợp nhiều patch của một ca; hai phép pooling nằm ở hai cấp khác nhau.

PNG gốc trong bộ kiểm thử là 512×512. 224×224 là tensor encoder, không phải cắt lại file nguồn; x/y vẫn ở hệ tọa độ nguồn. Crop ở tâm bỏ rìa trong tensor: nếu muốn giữ toàn patch, phải tạo feature version khác và so sánh riêng.

Nguồn: [ResNet, He et al.](https://arxiv.org/abs/1512.03385), [weights và transforms chính thức](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

## 4. Đầu vào của một ca

Ví dụ giả định, không phải số đo một ca thật:

```
F4   : [100, 2048]
F10  : [300, 2048]
F40  : [700, 2048]
mask : [1, 1, 1]
y    : 1
```

N thay đổi theo ca và lens. Không cần pad mọi ca thành số patch bằng nhau khi batch một ca. Case/patient ID dùng truy vết, chọn fold và lưu kết quả; không đưa mã bệnh nhân, filename, Glade hay kết luận văn bản vào tensor phân loại.

Tọa độ dùng truy xuất ngược/heatmap, không được dùng làm positional encoding trong baseline. Baseline coi các vector trong mỗi lens là một tập không có thứ tự và không mô hình hóa quan hệ không gian giữa patch.

## 5. Projection: từ 2.048 xuống 128 chiều

Với mỗi vector fᵢ:

```
hᵢ = ReLU(P fᵢ + b), hᵢ ∈ R¹²⁸
```

P khởi tạo ngẫu nhiên và được học bằng nhãn ca. Đây là lớp neural network học theo mục tiêu, không phải PCA. P dùng chung cả ba lens để giảm số tham số và tạo biểu diễn cùng kích thước. Dùng chung là giả định thiết kế; hiệu quả so với projection riêng cần thí nghiệm, chưa được chứng minh.

Dropout 0,25 là thiết lập regularization khởi đầu; vị trí đặt chính xác cần chốt trong model config khi triển khai. Dropout phần tử embedding khác với instance dropout bỏ cả patch.

## 6. Gated attention: chọn trọng số tổng hợp trong từng lens

Mỗi lens m có các tham số riêng Vₘ, Uₘ, wₘ; hidden size dự kiến 64:

```
tᵢ = tanh(Vₘ hᵢ + bV)
gᵢ = sigmoid(Uₘ hᵢ + bU)
sᵢ = wₘᵀ(tᵢ ⊙ gᵢ) + bs
aᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)
zₘ = Σᵢ aᵢ hᵢ             # 128 chiều
```

Nhánh sigmoid điều tiết các thành phần của nhánh tanh. Softmax cho trọng số không âm có tổng 1 trên toàn bag của lens đó. Nếu attention bằng nhau thì pooling trở thành trung bình. Với trọng số minh họa 0,1/0,2/0,7, vector thứ ba đóng góp mạnh nhất vào tổng; bác sĩ không cung cấp các trọng số này, mô hình học chúng gián tiếp từ loss cấp ca.

Đây là attention pooling, không phải Transformer self-attention: không tính ma trận quan hệ tất cả cặp patch. Tính score mỗi patch rồi tổng hợp giúp tính toán tăng chủ yếu tuyến tính theo số patch, phù hợp baseline trên T4.

Attention cao không tự đồng nghĩa patch ung thư. Vùng lành cũng có thể hữu ích cho quyết định ca lành. Vì softmax phụ thuộc toàn bag, không so trực tiếp trọng số thô giữa các ca có số patch khác nhau; số lượng patch và tỷ lệ tổn thương vẫn có thể ảnh hưởng.

Nguồn công thức: [Ilse, Tomczak & Welling, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html), [model.py chính thức](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py).

## 7. Fusion: kết hợp thông tin ở cấp ca

```
q = concat(z4, z10, z40, mask)   # 128×3 + 3 = 387 chiều
logit = W q + b                 # một số thực
p = sigmoid(logit)             # điểm lớp ung thư trong [0,1]
```

Mỗi lens cung cấp 128 số dù có ít hay nhiều patch. Nhờ đó, 40× không chiếm phần lớn input chỉ vì nhiều patch hơn; vẫn phải kiểm shortcut theo quy trình chụp/số ảnh. Linear head học hệ số cho từng thành phần ba lens; baseline không có module riêng xuất “trọng số độ quan trọng của lens” và không có tương tác chéo phi tuyến sau nối.

4× có thể cung cấp bối cảnh bố cục mô, 10× cấu trúc tuyến, 40× chi tiết tế bào. Đây là lý do thiết kế hợp lý để thử, không phải cam kết mô hình sẽ dùng đúng hoặc fusion sẽ thắng.

Nếu thiếu 4×: z4 = vector zero 128 chiều, mask4 = 0; không tạo ảnh trắng và không tính softmax trên bag rỗng. Study common đủ lens là đối chứng chính. Study all cần kiểm model chỉ đọc mask/số ảnh để xem quy trình thu thập có vô tình tiết lộ nhãn.

Chưa xác minh registration giữa lens, nên không ghép patch 4× và 40× có cùng x/y thành một vùng. Fusion diễn ra sau khi tóm tắt từng lens của cùng ca.

Với toàn bộ Linear có bias như công thức trên, khoảng 312.391 tham số học: projection 262.272, ba attention 49.731, classifier 388. Đây là tính toán theo thiết kế, không phải số đo model đã triển khai.

## 8. Trọng số nào có sẵn, trọng số nào được train?

| Khối | Khởi tạo | Cập nhật vòng đầu? |
|---|---|---|
| ResNet50 | ImageNet IMAGENET1K_V1 đã tải | Không |
| Classifier ImageNet | Bỏ khỏi forward | Không dùng |
| Projection 2048→128 | Random theo seed | Có |
| Ba gated attention | Random theo seed | Có |
| Classifier 387→1 | Random theo seed | Có |

Pretraining đã diễn ra trên ImageNet trước dự án; feature extraction là inference encoder; training sắp tới học phần MIL; fine-tuning CNN là thí nghiệm khác. Freeze đúng gồm requires_grad=False, eval() và không cập nhật BatchNorm statistics. MIL train() khi học, eval() khi validation/test để tắt dropout. [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

Không có MIL weights đã pretrain cho đúng bài toán này để load ở baseline. Resume sau ngắt phiên mới load weights/optimizer/RNG/checkpoint của lượt MIL tương ứng. Mỗi fold mới khởi tạo MIL và optimizer mới.

## 9. Một lượt học diễn ra như thế nào?

1. Lấy một ca thuộc train của fit hiện tại, đọc ba ma trận feature, mask và y.
2. Chạy projection → attention từng lens → pooling → fusion → logit.
3. Tính BCEWithLogitsLoss trực tiếp từ logit và y. Không sigmoid hai lần; sigmoid chỉ dùng xuất điểm.
4. backward tính gradient qua classifier, attention và projection; optimizer AdamW cập nhật các tham số này.
5. Lặp các ca train với thứ tự được quản lý bằng seed/RNG. Qua hết ca là một epoch, không phải qua hết patch thành các mẫu có nhãn riêng.
6. Chạy validation không gradient; tính loss trung bình theo ca, chọn checkpoint/epoch trong inner fold.

Viết theo xác suất, BCE = −[y ln(p) + (1−y) ln(1−p)]. Ví dụ y=1,p=0,2 thì loss≈1,609; p=0,8 thì loss≈0,223. Gradient của BCE theo logit là p−y; tín hiệu sai ở cấp ca lan ngược vào cách tóm tắt patch. Không có phép đổi loss này thành nhãn thật từng patch.

Điểm sigmoid chưa mặc nhiên là xác suất đã được calibration tốt. Nếu cần calibration/threshold tối ưu, phải fit trong inner data.

## 10. Train, validation, test: dùng folds đã khóa

Đã tạo và kiểm chứng nested 3 outer × 2 inner folds theo người bệnh. Với common 16 ca, outer test mỗi lượt khoảng 5–6 người, outer-train 10–11 người.

1. Giữ riêng outer test.
2. Trong outer-train, chạy hai inner fits để train và validate.
3. Chọn best epoch bằng inner validation BCE, tối đa 30 epoch, patience 5.
4. Lấy median hai best epochs, làm tròn lên; khởi tạo MIL mới và refit toàn outer-train đúng số epoch đã chọn.
5. Chạy outer test sau khi đã khóa cấu hình, weights và threshold; không dùng test early-stop.
6. Lặp ba outer folds, lưu mỗi ca một dự đoán OOF từ model không được train trên người đó.

Cấu hình khởi đầu: AdamW LR 1e-4 cho Attention-MIL, mean-linear LR 1e-3, weight decay 1e-3, projection 128, attention hidden 64, dropout 0,25, head FP32, batch một ca. Đây là thiết kế cần benchmark, không phải thông số đã tối ưu.

Nguồn phương pháp đánh giá: [Varma & Simon 2006 — thiên lệch khi chọn model và báo điểm trên cùng CV](https://link.springer.com/article/10.1186/1471-2105-7-91).

## 11. So sánh gì để biết mô hình có ích?

| Biến thể | Nội dung | Câu hỏi |
|---|---|---|
| E0 | Mean từng lens + linear, 6147→1 | Attention có thêm giá trị so với tóm tắt đơn giản? |
| E1 | Attention chỉ 4× | Độ thấp có đủ thông tin? |
| E2 | Attention chỉ 10× | Baseline vật kính trung gian ra sao? |
| E3 | Attention chỉ 40× | Chi tiết tế bào có ích thế nào? |
| E4 | Attention riêng + fusion ba lens | Kết hợp có ích hơn single-lens? |

Cả năm dùng cùng common cohort và cùng folds. 5×3×(2 inner + 1 refit) = 45 fits, không phải 45 epochs. E4 so với E2 là primary contrast dự kiến. Không chọn winner bằng outer score rồi gọi chính score đó là đánh giá độc lập của mô hình được chọn.

Lưu OOF probability/nhãn/threshold/fold, AUROC, AP, sensitivity, specificity, balanced accuracy, TP/FP/TN/FN, log loss và Brier. Báo mẫu số và từng fold; 16 người rất nhỏ nên một người làm chỉ số đổi mạnh. Không có patch truth để báo accuracy patch hoặc Dice vùng ung thư.

## 12. Tham khảo bài nào và custom gì?

**Attention-MIL, Ilse et al., ICML 2018:** nguồn trực tiếp của gated pooling. Dự án thêm encoder frozen/cache, nhánh theo lens, shared projection và fusion cấp ca. Không dùng nguyên CNN demo MNIST của tác giả.

**MRMIL, Li et al., 2021:** sát chủ đề tuyến tiền liệt, nhãn yếu và nhiều độ phóng đại. Bài dùng WSI, 5× chọn vùng rồi lấy cùng vùng tại 10× để phân loại; mình chưa có registration và target là binary cấp ca, nên dùng fusion cấp ca thay luồng zoom vùng. [Bản tác giả](https://arxiv.org/html/2011.02679), [bài xuất bản](https://doi.org/10.1016/j.compbiomed.2021.104253).

**CLAM, Lu et al., Nature Biomedical Engineering 2021:** tham chiếu tổ chức feature/coordinate cache và học nhãn lam. CLAM thêm clustering constraint cấp instance; baseline dự án chưa có loss đó, nên không gọi mô hình hiện tại là CLAM. [Mã và giải thích chính thức](https://github.com/mahmoodlab/CLAM).

Những điều chỉnh này là lựa chọn kiến trúc/kỹ thuật cho dữ liệu của mình. Chưa có thực nghiệm chứng minh cải thiện accuracy, chưa đủ cơ sở gọi là đóng góp thuật toán mới.

## 13. Có chuẩn hóa màu, augmentation, fine-tuning không?

Baseline hiện tại dùng ảnh đã được duyệt và ImageNet normalization cố định. ImageNet mean/std không đồng bộ thuốc nhuộm về một stain template; chưa tự thêm Macenko/Vahadane. Đổi cách xử lý màu cần một cache version khác; mọi template/fit từ dữ liệu chỉ được chọn trên train của fit tương ứng.

Training từ cache không có xoay/lật/jitter màu RGB tự động. Dropout embedding là regularization khác với augmentation hình ảnh. Đổi encoder hoặc fine-tune cần đọc lại RGB; cache cố định không có computational graph để cập nhật CNN.

Ưu tiên mở thí nghiệm sau control: (1) encoder pathology pretrained vẫn frozen; (2) giữ toàn patch thay center crop; (3) stain normalization/augmentation train; (4) CLAM; (5) fine-tune block cuối CNN. Chỉ đổi một yếu tố mỗi control; nếu lựa chọn bằng dữ liệu thì dùng inner folds. Encoder pathology chính thức có trong hệ sinh thái CLAM, nhưng phải kiểm weights/license/transform và VRAM. Với 18 người, mở toàn CNN ngay có rủi ro overfit lớn.

## 14. Những việc cần hoàn tất trước lần train thật

- Triển khai model E0–E4 và trainer dựa đúng bundle/folds đã verify.
- Dry-run train-only: input/shape, lens thiếu, finite loss, gradient đúng khối, đảo thứ tự patch cho prediction gần như không đổi trong eval.
- Kiểm save/load và resume model, optimizer, epoch/RNG; prediction trước/sau load khớp.
- Benchmark một fit và peak RAM/VRAM T4; full-bag trước, không tự giảm patch khi hết budget.
- Nếu chunk attention, chuẩn hóa softmax toàn bag; không trung bình kết quả các chunk đã normalize riêng.
- Stage features cần dùng sang SSD Colab; Drive giữ config, fingerprint, logs, last/best/final checkpoints và OOF. Một fit/epoch cần có boundary để resume khi ngắt.
- Ngân sách dự án tối đa 330 phút T4/ngày, dành 30 phút dự phòng; quota Google thực tế có thể khác và phải kiểm phiên. Chưa đo thời gian fit nên chưa cam kết 45 fits xong một phiên.

Khi inference ca mới: cùng preprocessing/encoder → ba bags → MIL weights đã học → điểm cấp ca, nhãn theo threshold và attention để truy vết. Không cần nhãn ca mới để predict, nhưng cần kết luận độc lập để đánh giá độ đúng.
