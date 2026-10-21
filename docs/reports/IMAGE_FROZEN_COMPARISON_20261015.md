# Kết quả phân loại từng ảnh gốc — 15/10/2026

**Ưu tiên B1: RetCCL ResNet50 frozen512 + Logistic Regression.** Cả hai thử nghiệm mới đã hoàn tất; N3 fine-tune có điều kiện không được kích hoạt vì cả A1/B1 đã qua gate ưu tiên.

## Dữ liệu và cách làm

- 2.802 ảnh gốc:4×364,10×830,40×1.608;18 người bệnh;2.298 ảnh lành,504 ảnh ung thư.
- Nhãn bác sĩ theo ảnh:Glade0→0;Glade3/4/5→1; giữ nguyên Glade gốc. Patient ID chỉ dùng nhóm split.
- Giữ toàn khung, resize theo tỷ lệ và pad; không cắt patch, không PCA, không stain normalization trong lượt này.
- Encoder pretrained được đóng băng, vector2.048D; train StandardScaler/LR, chọn lambda và threshold trong inner validation. Ba outer folds giữ nguyên theo người bệnh.

## Kết quả OOF

|Phương án|AUROC4×|AUROC10×|AUROC40×|AUROC gộp|AP|Accuracy tại0,5|
|---|---:|---:|---:|---:|---:|---:|
|A0 — ImageNet512|0.591|0.687|0.773|0.729|0.398|77.3%|
|A1 — ImageNet1024|0.658|0.746|0.828|0.787|0.520|78.6%|
|B1 — RetCCL512|0.883|0.883|0.857|0.870|0.624|84.9%|

AUROC là khả năng xếp hạng, không phải accuracy. Dự đoán toàn bộ ảnh là lành đã đạt82,0% accuracy nhưng độ nhạy0%; do đó cần xem đồng thời AUROC,AP và độ nhạy/đặc hiệu.

## Hai cách đặt ngưỡng cho B1

|Cách đánh giá|Accuracy|Độ nhạy|Đặc hiệu|TP|FN|FP|TN|
|---|---:|---:|---:|---:|---:|---:|---:|
|Ngưỡng cố định0,5|84.9%|71.4%|87.9%|360|144|278|2020|
|Ngưỡng chọn từ inner validation|77.5%|80.2%|76.9%|404|100|530|1768|

Ngưỡng validation có thể khác giữa các fold. Mô hình deployment được refit trên toàn dữ liệu và lưu ngưỡng riêng trong study; kết quả fit-all không được dùng làm ước lượng accuracy. Score logistic chưa được xác minh như xác suất nguy cơ lâm sàng đã hiệu chuẩn.

## Mức cải thiện và giới hạn

- B1 so với A0:ΔAUROC=0.140, khoảng bootstrap95%=[0.086;0.194].
- ΔAP=0.226, khoảng95%=[0.108;0.386].
- Bootstrap500lần theo người bệnh từ OOF đã lưu, không refit mô hình; không bao phủ toàn bộ bất định huấn luyện.
- Outer folds đã được xem qua nhiều thử nghiệm: đây là kết quả thăm dò nội bộ trên18người, không phải test lâm sàng độc lập.
- Cải thiện ở4×/10× gợi ý encoder pretrained mô học phù hợp hơn trong bộ này; chưa đủ suy rộng sang cơ sở, cách nhuộm hoặc thiết bị khác.

## Đã kiểm tra và lưu gì

- N0 audit giữ nguyên;492vector A1 cũ được fullverify rồi dùng lại, chỉ trích phần thiếu.
- A1 vàB1 đều đủ2.802vector, source/encoder/preprocessing identity và checksum hợp lệ; folds không trùng người bệnh.
- Pilot48ảnh/3encoder pass, đối chứng512 khớp cache chính xác; peakVRAM khoảng422MB.
- Sáu kiểm tra suy luận:3ảnh gốc đủ4×/10×/40× qua2encoder, kết quả khớp đường dùng vector cache; đây là kiểm tra đường xử lý, không đánh giá accuracy mới.
- Gói bàn giao281.353.760byte,1.038file; có code,weights,features,models/folds/OOF,inventory,audit,reports. Không chứa ba ZIP ảnh nguồn.
- Gói trên Drive và bản local đều đã verifySHA; local dùng9phần≤32MiB khi truyền, các phần tạm local đã xóa sau ghép đúng.

### Vị trí

Drive: `MyDrive/histology/image_classification/improvement_v002/executions/attempt-resume4-20261015-a65e760b3a/`.

- `report/BAO_CAO.md`, `report/auroc_comparison.png`, `report/gate_decision.json`
- `single_image_smoke/receipt.json`
- `handoff/histology-frozen-handoff.zip`, `handoff/verification.json`, `handoff/README_VI.md`
- `comparison.json`, `result.json`; study_root/release_root trong đó chỉ tới models và cache gốc.

Bản local: `private/histology-frozen-handoff.zip`; hướng dẫn nạp một ảnh nằm trongREADME_VI.md của gói.

SHA-256 gói: `0e244bdd3db024ea186b21109ee780c6d9f5fd183670fa2d090859652c88a540`.

## Tham chiếu

Kế hoạch `docs/plans/IMAGE_CLASSIFICATION_FEASIBLE_DIRECTIONS.md`; runtime source commit2e6feac;487tests local pass vàCI12/12 trước chạy. RetCCL dùng checkpoint tác giả tại [repositoryRetCCL](https://github.com/Xiyue-Wang/RetCCL), revisiona85b97226b62853b64143e8d535f5e10dae6cc89; receipt nguồn/SHA được giữ trong gói.

## Kiểm chứng so sánh bổ sung

- B1 so với A1:ΔAUROC=0.083, bootstrap95%=[-0.008;0.186]. Khoảng này còn chứa0: B1 có điểm ước lượng cao nhất, chưa đủ khẳng định chắc chắn hơn A1 ngoài bộ dữ liệu này.
- Cached controlA0 có cùng cấu hình LR, inner folds và assignment theo người nhưA1/B1. Các nhãn/fold/quyết định nhị phân trùng khớp baseline lịch sử; xác suất chênh tối đa3.47e-09 do số học, không phải bit-identical. Không train lại hoặc trích lại đối chứng512.
