# UNI: hướng dẫn chuẩn bị và chạy về sau

**Ngày21/10/2026: chỉ chuẩn bị phần mềm. Chưa được phép chạy cohort thật ở lượt này.** Không mởGPU để chờ xin quyền hoặc viết mã.

Người dùng sẽ cung cấp một tài khoảnColabPro khác sau. Mã không gắn với tài khoảnGoogle cụthể; khi chọnprofilemới, chia sẻ/shortcutDrive và xác minh quyềnđọc/ghi trongđúngkernel. HFaccessUNI độc lập với tài khoảnGoogle/ColabPro.

## A. Local: chỉ fixture giả, không tải model

```bash
uv venv /tmp/histology-uni-prepare --python 3.12
uv pip install --python /tmp/histology-uni-prepare/bin/python -r requirements-uni-local.txt
/tmp/histology-uni-prepare/bin/python -m pytest tests/test_uni_preparation.py
python3 scripts/build_image_classification_runtime.py --requirements requirements-uni.txt --output dist/histology-uni-runtime.zip
```

Không cài requirements-image-classification/requirements-uni trên local để chạy fixture: chúng kéo torch/CUDA không cần cho nhiệm vụ này. RuntimeZIP là code-only, không có ảnh, weights hoặc credentials.

## B. Những thứ cần có trước ColabGPU

1. Tài khoảnHuggingFace có quyềnUNI; người dùng tự duyệt điều kiện chính thức. Không gửi token/password trong chat. Chưa có quyền thì dừng ởpreparation.
2. CheckpointUNI tải hợp lệ vào Drive, ví dụ`histology/models/UNI/pytorch_model.bin`. TínhSHA256 và lưu receipt; không phát hành lại weights.
3. BaZIP gốc4×/10×/40×; không dùng bộ patch cũ thay ảnh gốc.
4. B1 release và baseline study đã verify, vẫn giữ tại`histology/image_classification/`.
5. RuntimeZIP đã qua tests/CI. Python3.12local chưa chứng minh ColabPython3.13/GPU tương thích; phải smoke thật sau khi được phép.

## C. Đường dẫn chuẩn (không sao chép tên VM cũ)

```python
from pathlib import Path
ROOT = Path('/content/drive/MyDrive/histology')
PROJECT = ROOT / 'image_classification'
BASELINE_RELEASE = PROJECT / 'improvement_v002/features/B1/runs/c294914d1c5d00d3da82e40007988ad65170338af19914a1489aa76611d9f955'
BASELINE_STUDY = PROJECT / 'head_studies/16d92369288182d8cf5165bbad27f7f582e33ddbc985f043c4628a1db7622b5b'
OUTPUT = PROJECT / 'final_uni_v001'
```

Mount bằngCLI trên đúng runtime khi được yêu cầu, tự duyệtGoogle bằng đúngprofileChrome. Webmount không chứng minh CLI kernel thấyDrive. Xác minh MyDrive và các đường dẫn trên trong chínhkernel chạyjob. Không lưu linkOAuth hoặc credential vào runtimeZIP.

## D. Những lệnh dành cho giai đoạn đã được phép chạy

Giải nén runtimeZIP vào`/content/histology-uni-runtime`, cài`requirements-uni.txt` trongColab. Không chạy lệnh install hoặcextract ởlocal trên dữ liệu thật.

CLI help không khởi tạo model:

```bash
python -m histology_data.uni_cli --help
python -m histology_data.uni_cli check-weights --weights "$UNI_WEIGHTS" --weights-sha256 "$UNI_SHA256"
```

`check-weights` chỉ chứng minh bytes đúng pin do operator cung cấp; không chứng minh checkpoint làUNI chính thức hoặcmodel/GPUload được. Identity của nguồn download phải được lưu riêng.

Tạo`archives.json` cục bộ trênColab, không đưa vàoGitHub, dùng đường dẫnZIP thực tế:

```json
{"4":"/content/drive/MyDrive/NCKH_Duy/data_4x.zip","10":"/content/drive/MyDrive/NCKH_Duy/data_10x.zip","40":"/content/drive/MyDrive/NCKH_Duy/data_40x.zip"}
```

Trước extraction, chạy smoke kỹ thuật trên ảnh lớn nhất mỗi lens (không chọn theo nhãn/score):

```bash
python -m histology_data.uni_cli smoke --baseline-release "$BASELINE_RELEASE" --baseline-study "$BASELINE_STUDY" --archives-json /content/archives.json --weights "$UNI_WEIGHTS" --weights-sha256 "$UNI_SHA256" --output "$OUTPUT/resource_smoke"
```

Receipt phải có`technical_smoke_passed=true`, `GPU_checked=true`, `encoder_kind=UNI`. Lưu thời gian,peakVRAM và batchparity (max absolute delta vector≤0,005). Đây không phải chất lượng phân loại. NếuGPU smoke không qua thì giữbằngchứng và điều tra, không chạyfullcohort.

Trích sau khi đã được phép và córesource smoke:

```bash
python -m histology_data.uni_cli extract --baseline-release "$BASELINE_RELEASE" --baseline-study "$BASELINE_STUDY" --archives-json /content/archives.json --weights "$UNI_WEIGHTS" --weights-sha256 "$UNI_SHA256" --output "$OUTPUT/features" --budget-minutes 240 --reserve-minutes 30
```

Lệnh trả`cache_root`; lần chạy sau giữ nguyênconfig đểresume. `--max-new-shards 1` là giới hạn kỹ thuật cho lượt kiểm tra, không là bảnfeatures hoàn tất. Không tự đặt`--recover-lock` khi chưa chứng minh owner cũ chết; recoverycross-host chưa được tự động hóa.

TrênColabCPU, verifycache rồitrainhead:

```bash
python -m histology_data.uni_cli train --baseline-release "$BASELINE_RELEASE" --baseline-study "$BASELINE_STUDY" --uni-cache "$UNI_CACHE_ROOT" --output "$OUTPUT/studies" --budget-minutes 240 --reserve-minutes 30
```

Không dùng`--refit` cho lượt đánh giá đầu tiên. Sau khi báo cáo/gate được chốt, thêm`--refit` nếu cần modelUNI fit-all đểnghiên cứu/bàn giao. Nó tạo9fits thêm và không tạo validation độc lập. Nếu gatefail, R0 vẫn là phương ánưu tiên; không chọnUNI chỉ vì đã cómodelrefit.

Singleimage sau researchrefit:

```bash
python -m histology_data.uni_cli predict --study-root "$UNI_STUDY_ROOT" --image /content/one_original.tiff --weights "$UNI_WEIGHTS" --weights-sha256 "$UNI_SHA256"
```

Giữ nguyên device/precision/versions/weights nhưcache; loader từ chối descriptor khác. Score chưa được hiệu chuẩn lâm sàng. CLI đọc modeljoblib chỉ từstudy do dự án tạo vàverify; không dùngjoblib người lạ gửi.

## E. Trạng thái và lỗi

- Exit0: command hoàn tất hoặc đã tạo mộtprediction; exit2: missingweights/partial. `status.json` cóstate`partial` phảiresume, không báocomplete.
- Extraction thiếuRAM/VRAM hoặc lỗi khác: lưuattention_required, giữ shardcommit. Không âm thầm giảmprecision hoặcbatch đểkhớpcache cũ; thaycấu hình tạoidentity mới.
- Checksum/lineage/labels/folds sai: dừng, audit nguyên nhân; không xóa commit để bỏqua.
- Mất VM: artifacts đãcommit trênDrive giữ được. Kiểmtra writer cũ vàlock trướcresume; không xóaDrive hoặc chạyencoder thứhai.
- Completion phần mềm/synthetic không thay choGPU smoke, kết quả train thật hoặcexternaltest.

[Kế hoạch chi tiết](plans/UNI_FINAL_PREPARATION.md)
