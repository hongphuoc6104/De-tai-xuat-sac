# Chuẩn bị bộ dữ liệu MIL trước train

Toolkit này nối feature cache đã commit với nhãn cấp ca, người bệnh và split. Không cắt/nhuộm lại PNG, không chạy encoder lần nữa, không huấn luyện MIL và không tạo VM.

## Luồng và điều kiện

1. ZIP đã duyệt + Metadata.xlsx → inventory và review draft theo các ca thực sự có patch.
2. Xác nhận người bệnh/nhãn → case_review.csv có evidence và governance có fingerprint.
3. Feature cache complete + final audit → bag tham chiếu theo ca × vật kính.
4. Patient/duplicate groups → ba outer folds, hai inner folds bên trong mỗi outer-train.
5. Kiểm source/version/vector-row/bag/split → bundle dữ liệu sẵn sàng cho bước train sau này.

Dữ liệu hiện tại đã được người dùng xác nhận: 18 ca là 18 người khác nhau; CARCINÔM ở kết luận → ca dương, kể cả có vùng lành; chỉ tăng sản lành/kèm viêm → ca âm. Không chuyển nhãn ca thành nhãn patch. `Glade` giữ nguyên, không suy Grade Group. Xác nhận chỉ áp dụng cho cohort release hiện tại; dữ liệu bổ sung phải có mapping riêng. Sửa nhãn hoặc patient mapping trong artifact governance/review có phiên bản, giữ snapshot Metadata.xlsx gốc; không cần encode lại khi pixels/encoder/preprocessing vẫn giữ nguyên.

Cohort `common` là các ca đủ 4×/10×/40×, dùng cho E0–E4 trên cùng fold/cohort; `all` là bộ bổ sung có mask lens thiếu. Dữ liệu hiện tại dự kiến lần lượt 16 và 18 ca; công cụ tính lại từ feature index và ghi các ca/vector bị loại khỏi cohort chính. Không tạo ảnh hay vector trắng cho lens thiếu.

## Vị trí dữ liệu

```text
MyDrive/histology/
  source/Metadata.xlsx
  source/archives/Tiles-20261009T164818Z-1-001.zip ... -025.zip
  runtime/histology-pretrain-runtime.zip
  governance/reviews/user_confirmation.json
  governance/reviews/case_review.csv
  governance/drafts/<draft_id>/case_review.csv, draft.json
  governance/<governance_id>/governance.json, cases.csv, case_review.csv
  features/v001/<feature_id>/parts/<part_id>/
    features.npy, tile_index.jsonl, qc.json, commit.json
  bundles/<bundle_id>/
    bags.jsonl, instance_refs.jsonl, splits.json, bundle.json, training_readiness.json
  pretrain_runs/<run_id>/
    status.json, events.jsonl, resolved_config.json
```

`/content` dùng cho code đã giải nén và scratch, không phải bản bền vững. Bundle chỉ tham chiếu part/row của features, không sao chép 1,22 GB vectors thành nhiều bản. Code của encoder, weights SHA và preprocessing của cache hiện tại được giữ nguyên.

## Chạy trên Colab

Dùng [Colab_PreTrain_Readiness.ipynb](../notebooks/Colab_PreTrain_Readiness.ipynb) hoặc CLI bên dưới. Runtime CPU đủ cho metadata, split, bag và verify; nếu chạy tiếp trong phiên T4 đang encode thì không mở GPU mới. Khi bootstrap, notebook đọc từng requirement theo cú pháp PEP 508 và so cả module lẫn phiên bản distribution đã cài bằng `importlib.metadata`. Module có thể import được nhưng vẫn phải cài nếu phiên bản không thỏa specifier; ví dụ `sklearn` 1.6.1 không thỏa `scikit-learn==1.8.0`. Package thiếu được cài theo đúng requirement với `pip --no-deps`. Nếu Pillow đã cài nhưng không thỏa specifier, notebook dừng và báo cần chạy trên runtime mới có Pillow tương thích thay vì nâng Pillow đang dùng; Pillow chỉ được cài riêng nếu còn thiếu. Torch và torchvision không bao giờ được cài hoặc nâng cấp bởi bootstrap. Nếu package đã đúng phiên bản, notebook giữ nguyên và bỏ qua pip.

Sau giải nén runtime và thêm vào PYTHONPATH, kiểm kê nhẹ để tạo mẫu review:

```bash
python -m histology_data.pretrain_cli draft \
  --metadata /content/drive/MyDrive/histology/source/Metadata.xlsx \
  --source-root /content/drive/MyDrive/histology/source/archives \
  --output-root /content/drive/MyDrive/histology \
  --source-kind zip --expected-sources 25 --expected-patches 148991
```

Review CSV yêu cầu đúng các cột do draft tạo, nhãn `case_label` 0/1, `identity_verified=true`, `label_verified=true`, `patient_id` và evidence không trống. Giữ nguyên các trường raw metadata và `metadata_case_sha256`. Xác nhận đã ghi của người dùng có thể được runner áp dụng sau khi kiểm metadata SHA và đúng tập ca; không tự suy patient ID từ mã ca cho dữ liệu chưa xác minh.

Chỉ tạo bundle khi release feature đầy đủ và audit qua:

```bash
python -m histology_data.pretrain_cli build \
  --metadata /content/drive/MyDrive/histology/source/Metadata.xlsx \
  --case-review /content/drive/MyDrive/histology/governance/reviews/case_review.csv \
  --feature-root /content/drive/MyDrive/histology/features/v001 \
  --output-root /content/drive/MyDrive/histology \
  --cohort common --expected-sources 25 --expected-vectors 148991
```

Chạy `--cohort all` riêng để tạo bundle bổ sung; hai bundle cùng tham chiếu cache gốc nhưng có cohort và protocol riêng. Nếu có duplicate audit groups, cung cấp `--duplicate-dispositions PATH` với phân xử rõ giữ/loại, canonical tile và evidence. Giữ nguyên PNG/cache; loại tham chiếu khỏi bundle phải có log. Nhóm trùng được giữ giữa nhiều người phải cùng split group, và split phải vẫn khả thi; không lặng lẽ để cùng nội dung sang train và test.

```bash
python -m histology_data.pretrain_cli verify \
  --bundle /content/drive/MyDrive/histology/bundles/<bundle_id> \
  --feature-root /content/drive/MyDrive/histology/features/v001
```

## Chạy tự động và tiếp tục sau ngắt

`scripts/run_pretrain_colab.py` dùng `configs/pretrain_colab.json` để lấy đường dẫn. Chế độ `auto` tạo draft/governance từ bằng chứng xác nhận, chờ release feature và writer cũ hoàn tất, rồi build/verify hai cohort. Nó không khởi chạy encoder thứ hai. Deadline phải được truyền từ ngân sách thật còn lại của profile; không bắt đầu lại một ngân sách 330 phút sau mỗi lần resume.

Ví dụ chạy tự động sau khi đã gắn Drive và giải nén runtime:

```bash
python scripts/run_pretrain_colab.py --config configs/pretrain_colab.json \
  --mode auto --deadline-utc 2026-10-10T14:00:00+00:00 --poll-seconds 45
```

Mốc deadline trên chỉ minh họa; thay bằng cutoff được tính từ ngân sách profile còn lại và 30 phút dự phòng. Không dùng nguyên mốc này cho ngày/phiên khác. Runner truyền cùng monotonic deadline cho hash, audit, build và verify; không báo sẵn sàng nếu hết thời gian giữa hai cohort.

Runner ghi trạng thái vào `pretrain_runs/<run_id>/status.json` và sự kiện vào `events.jsonl`. Đọc trạng thái/checksum/commit mới quyết định hoàn tất, không chỉ dựa trên sự tồn tại của file. Sau runtime mất, chạy lại cùng source/config/version; chỉ phục hồi lock cũ khi phiên trước đã dừng, không chiếm writer sống.

## Kiểm tra trước khi nhận bundle

- Feature release đủ 25 nguồn/148.991 vectors, finite float32, chiều 2048 và audit_complete/feature_complete; smoke hoặc release partial không đủ.
- Mapping và nhãn có evidence, đúng metadata hash và đúng ca; chỉ nhãn cấp ca được đưa vào model sau này.
- Mỗi instance reference trỏ đúng part/row/tile/lens/x/y/hash; đọc bag trả đúng ma trận Nx2048 và mask.
- Mọi người bệnh, ca, ảnh và duplicate content group không giao hai phía của outer/inner split; mỗi tập phải đủ lớp theo protocol.
- Duplicate dispositions bao phủ đúng các members, không tạo dữ liệu mới hoặc bỏ qua nhóm chưa review.
- `training_readiness.json` của bundle phải qua toàn bộ gates. Feature cache riêng vẫn có `training_ready=false` vì chưa chứa governance; đó không phải lỗi encoder.

Bundle sẵn sàng là kết quả chuẩn bị dữ liệu. MIL weights, checkpoint fit, OOF và đánh giá ngoài fold thuộc G6–G8, chưa thực thi trong bước này.
