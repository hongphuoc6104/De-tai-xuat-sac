# UNI preparation readiness — 21/10/2026

**Trạng thái: phần mềm chuẩn bị đã có và được kiểm chứng bằng synthetic; chưa đủ điều kiện chạy cohort thật.** Người dùng sẽ cung cấp tài khoảnColabPro khác sau. Không tạoVM, không tải trọng sốUNI, không đọcảnh bệnhnhân đểtrích features hoặctrain trong lượtchuẩnbị.

## Đã chuẩn bị

- Kế hoạch chốt và CLI dùngđườngdẫn tham số, không gắn với profilebop/thiha hay mộtVM cũ.
- UNI adapter lazy-load, chỉ nạp checkpoint local cóSHA256pin và strictstate; frozen/eval/inference_mode. Không tự xin quyền hoặc tảiweights.
- Global224 giữtrọnảnh và lưới512 trên tọađộgốc, cạnhcuối padtrắng; rawembeddings1024D/view → global/mean/max3072D/image.
- Cache32ảnh/shard, viewcoordinates/offsets/rawfeatures, ZIP/member/imagechecksums, commit cuối, strictresume và lease mộtwriter.
- HeadUNI duy nhất dùng workerbounded hiệncó, sharedpatientfolds, train-onlyscaler, patient-classweights, L2LR, innerselection/ngưỡng, reportOOF và gate vớiR0.
- Inference từmộtảnh hoặcvector dùng đúngdescriptor; graphverified trước khi nạp model do dựán tạo. Score chưa được hiệu chuẩn lâm sàng.
- Resource smoke command chờColab: ảnhlớn nhất mỗilens, đopeakVRAM/thờigian và batchparity; không đánhgiá chấtlượng hoặc chọnrecipe.
- Runtimecode-only, requirementsColabriêng và requirementslocalnhẹ khôngtorch/CUDA. Không phát hành ảnh, patientrecords, weightsUNI hoặccredential.

## Bằng chứng phần mềm

- **132passed,6skipped** trong12testfiles liênquan;123giây. Gồm12testUNI mới: geometry/pooling, ZIPsource và metadata, partial/resume, tamper, actualbaselineproducerbridge, patientleakage, train-onlyscaler, modelrestore, runtimeunzipCLI, missingweights và resource-smokesynthetic.
-6skip là các kiểm traencoder cũ cầnTorch khôngcài trong môi trườnglocalnhẹ. Không coi chúng làpass hoặc bằngchứngUNI thật.
- Một warningPillow deprecation trongtest cũ, không phải lintwarning; không thaycode cũ đểche cảnhbáo.
- AST kiểm445testmethods cóactiveassertion. Conveyor1/3/4 qua; ruff vàcompileall qua; local links kếhoạch/guide hợp lệ.
- Proof riêngtư: `Results/proofs/uni_preparation_20261021_144042.log` tạiworktree. Proof lưucommands/duration/exitcode/sourcehashes.
- Dependencyresolution choPython3.13 thànhcông; chưa cài/chạy UNI trênkernelColab3.13 hoặcGPU.
- Lỗi rawGlade dạngdecimalExcel đã táihiện bằngZIP E2E, sửa bằngparserbaseline, chạy lạiqua và ghiPLAYBOOK.

## Điều kiện chưa đạt trước chạy dữ liệu thật

| Điều kiện | Bằng chứng còn thiếu |
|---|---|
| ColabProprofile | Người dùng cung cấp tài khoản sau; cần mount đúngkernel và quyềnDrive |
| UNIaccess | Người dùng tự xin quyềnHF chínhthức; không gửi token/password vàochat |
| Checkpoint | Filetảihợp lệ, receipt nguồn vàSHA256; chưa cóbytesverified/modelstrictload |
| DữliệuDrive | Verify3ZIPoriginals+B1release/study đúnglineage; không dựa vào đườngdẫnmẫu |
| GPU | UNI thật1024D/view,3072D/image,finite, VRAM/time vàparity; chưa chạy |
| Training | Chưa có UNIcohortfeatures/27nestedfits/OOF; không có sốAUROC mới |
| Externaltest | SICAPv2 cần audit riêng và khóaquytrình trước đánhgiá; chưacó kếtquả |

Sau khi người dùng cung cấpPro, quyềnUNI vàtrọngsố cóthể đượcchuẩn bị trướcGPU. Không đốt phiênGPU đểchờ quyền/tảidữliệucôngviệcCPU. GPUchỉbắtđầu sauchecksource/software, rồiresource smoke trướcfullcohort. Nếu bộsmoke hoặcgate khôngqua, giữbằngchứng vàbáocáo đúng, không tựmởnhánhmodelkhác.

Đây là báo cáo chuẩn bị, không phải kết quảthựcnghiệm UNI hoặc xác nhận môhình vượtR0. CI của commitchuẩnbị được theo dõi riêng trênPR; bảnreceiptlocal cậpnhật saukhi checks hoàn tất.

[Kế hoạch](../plans/UNI_FINAL_PREPARATION.md) · [Hướng dẫn](../UNI_RUN_GUIDE.md)
