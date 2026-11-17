# 12 - BÁO CÁO CHUẨN HÓA DỮ LIỆU GROUND TRUTH (SPRINT 4B.4)

**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Tập dữ liệu nguồn:** `demo/annotations_ground_truth_review_draft.json`  
**Tập dữ liệu chuẩn hóa mới:** `demo/annotations_ground_truth_clean.json`  
**Tài liệu tham chiếu:** `docs/sprint4a_validation/11_ground_truth_evaluator_compatibility.md`  
**Thời điểm thực hiện:** 2026-11-12T23:48:00+07:00  

---

## 1. TỔNG QUAN VÀ KẾT QUẢ ĐÁNH GIÁ (VERDICT)

### Kết luận phân loại kỹ thuật: `PASS` (Chuẩn hóa) kèm `BLOCKED` (Vận hành đánh giá)

- **Chuẩn hóa cấu trúc (Cleaning & Schema): `PASS`**  
  Tập dữ liệu `demo/annotations_ground_truth_clean.json` đã được tái cấu trúc thành công: 100% các annotation sai lệch hoặc thiếu ngữ cảnh (`exclude_*`) đã được bóc tách và cách ly vào khu vực lưu trữ lịch sử (`excluded_annotations`). Danh sách Ground Truth hoạt động (`parts_ground_truth` và `damages_ground_truth`) hoàn toàn sạch, ID duy nhất và tương thích tuyệt đối với `IndependentEvaluator`.
- **Trạng thái xác minh (Verification Status): `BLOCKED` (Chủ động theo quy chuẩn an toàn)**  
  Toàn bộ 19 annotation ứng viên trong danh sách hoạt động tiếp tục giữ nguyên `is_verified: false` và `verification_status: "pending_review"`. Hệ thống kiên quyết không tự động đánh dấu verified, ngăn ngừa tuyệt đối việc tạo ra các chỉ số accuracy giả mạo.

---

## 2. BẢNG THỐNG KÊ CHI TIẾT CHUẨN HÓA DỮ LIỆU

| Chỉ Số Kiểm Kê | Số Lượng | Tỷ Lệ / Chi Tiết |
| :--- | :---: | :--- |
| **Tổng số ảnh xử lý** | **8 ảnh** | Đủ 8 ảnh trong `demo/input/`, ID 1..8 duy nhất |
| **Tổng số annotation ban đầu (Draft)** | **28** | 14 bộ phận xe + 14 vùng tổn thất |
| **1. Số annotation bị loại khỏi tập hoạt động** | **9** | **8 bộ phận + 1 tổn thất** (cách ly vào `excluded_annotations`) |
| - *Bộ phận gán sai nhận diện (`exclude_candidate_wrong_part`)* | 5 | Ảnh #5 (cua_truoc, nap_capo, kinh_xe), Ảnh #7 (kinh_xe), Ảnh #8 (can_truoc) |
| - *Bộ phận thiếu ngữ cảnh macro (`exclude_part_mapping_pending_context`)* | 2 | Ảnh #4 (can_truoc), Ảnh #6 (ve_che_banh) |
| - *Bộ phận nghi ngờ góc chụp (`exclude_candidate_unless_confirmed`)* | 1 | Ảnh #2 (nap_capo) |
| - *Tổn thất nghi ngờ bóng phản chiếu (`exclude_candidate_unless_confirmed`)* | 1 | Ảnh #1 (crack d5, conf 0.272) |
| **2. Số annotation bộ phận cần chỉnh polygon** | **6** | 100% bộ phận hoạt động đều cần chỉnh lại đường biên mềm mại |
| **3. Số annotation tổn thất cần chỉnh bounding box** | **9** | Cần thu hẹp hộp ôm sát hoặc chia nhỏ theo vết xước thực tế |
| **4. Số annotation nghi trùng lặp (Duplicate Detections)** | **2** | Ảnh #8: `dent` #1 và `dent` #2 (chồng lấn nhau trên cùng vết móp cửa) |
| **5. Số đối tượng còn thiếu tọa độ** | **9** | Ghi nhận trong `review_candidates_pending` (chưa có bbox/polygon) |
| **Tổng số annotation hoạt động được giữ lại (Active Candidates)** | **19** | **6 bộ phận + 13 tổn thất** (Tất cả đều `pending_review`) |
| **Tổng số annotation đã được xác minh (`verified`)** | **0** | **0 / 19 (0.0%) — Nghiêm cấm tự động verified** |

---

## 3. PHÂN LOẠI CHI TIẾT THEO HÀNH ĐỘNG THẨM ĐỊNH (`review_action`)

### 3.1. Danh sách 9 Annotation bị loại trừ khỏi tập hoạt động (`excluded_annotations`)
Các annotation này được chuyển sang mảng `excluded_annotations` trong từng ảnh để phục vụ đối soát, không tham gia vào quá trình tính IoU hay Precision/Recall của Evaluator:
1. `image copy 4.png` (Ảnh #2) — Part #3 `nap_capo`: Góc chụp hông xe không thể hiện rõ nắp capo (`exclude_candidate_unless_confirmed`).
2. `image copy 6.png` (Ảnh #4) — Part #1 `can_truoc`: Ảnh macro siêu cận cảnh, thiếu mốc nhận dạng để xác định cản trước (`exclude_part_mapping_pending_context`).
3. `image copy 7.png` (Ảnh #5) — Part #1 `cua_truoc`: AI nhận nhầm góc thân/vòm bánh xe phía sau thành cửa trước (`exclude_candidate_wrong_part`).
4. `image copy 7.png` (Ảnh #5) — Part #2 `nap_capo`: Không có nắp capo trong vùng ảnh sườn sau (`exclude_candidate_wrong_part`).
5. `image copy 7.png` (Ảnh #5) — Part #3 `kinh_xe`: Không có kính xe trong khung hình này (`exclude_candidate_wrong_part`).
6. `image copy 8.png` (Ảnh #6) — Part #1 `ve_che_banh`: Ảnh macro hẹp, không đủ căn cứ khẳng định vè che bánh (`exclude_part_mapping_pending_context`).
7. `image copy.png` (Ảnh #7) — Part #2 `kinh_xe`: AI gán nhầm vùng choá đèn pha vỡ thành kính xe (`exclude_candidate_wrong_part`).
8. `image.png` (Ảnh #8) — Part #1 `can_truoc`: Ảnh thể hiện cánh cửa hông xe bị móp, không phải cản trước (`exclude_candidate_wrong_part`).
9. `image copy 3.png` (Ảnh #1) — Damage #5 `crack` (conf 0.272): Khe nối panel và bóng phản chiếu, không phải vết nứt cơ học (`exclude_candidate_unless_confirmed`).

---

### 3.2. Danh sách 6 Bộ phận hoạt động cần chỉnh Polygon (`parts_ground_truth`)
Toàn bộ 6 bộ phận giữ lại đều có `action_required: "adjust_polygon_boundary"`:
1. Ảnh #1 (`image copy 3.png`) — `nap_capo` (31 đỉnh): Cần vẽ lại viền panel tránh tràn xuống mép vè xe.
2. Ảnh #2 (`image copy 4.png`) — `kinh_xe` (36 đỉnh): Cần cắt tỉa ranh giới kính, không để lấn khung kim loại.
3. Ảnh #2 (`image copy 4.png`) — `cua_truoc` (12 đỉnh): Cần bo sát khe cửa (door gap), tránh tràn sang cửa sau.
4. Ảnh #2 (`image copy 4.png`) — `ve_che_banh` (10 đỉnh): Cần vẽ lại theo đường cong vòm bánh sau.
5. Ảnh #3 (`image copy 5.png`) — `ve_che_banh` (40 đỉnh): Cần kiểm tra độ phủ trọn vẹn của vè xe.
6. Ảnh #7 (`image copy.png`) — `nap_capo` (4 đỉnh): Đa giác 4 điểm thô sơ, vi phạm quy tắc $\ge 16$ đỉnh; cần vẽ lại viền cong khí động học.

---

### 3.3. Danh sách 13 Tổn thất hoạt động (`damages_ground_truth`)
- **Tổn thất cần chia nhỏ hộp bao (Split Oversized Box — 1 vết):**
  - Ảnh #1 `scratch` #2 (`[10.5, 464.6, 667.0, 698.2]`): Hộp bao chiếm >30% diện tích ảnh, trùm cả mặt đường. Cần chia nhỏ thành các hộp bám theo từng vết xước thực tế.
- **Tổn thất cần vẽ lại hộp bao ôm sát (Redraw Tight Box — 6 vết):**
  - Ảnh #1 `crack` #1 (Đèn pha vỡ): Bounding box bao quá rộng phần cản xung quanh.
  - Ảnh #2 `dent` #1 (Cửa trước móp sâu): Bounding box trùm rộng, cần chỉnh sát mép gân tôn bị lõm.
  - Ảnh #4 `crack` #1: Cần vẽ hộp hẹp bám theo đường nứt cản.
  - Ảnh #5 `scratch` #1 và `scratch` #2: Cần co hẹp hộp bao ôm sát vệt cào.
  - Ảnh #7 `crack` #1 (Cụm đèn pha vỡ): Cần thu hẹp ranh giới, không trùm cản nhựa.
- **Tổn thất cần phân biệt loại hoặc kiểm tra bóng đổ (Verify Artefacts — 3 vết):**
  - Ảnh #1 `scratch` #3 & `dent` #4: Cần xác nhận có phải bóng phản chiếu hoặc đường gân dập nắp capo không.
  - Ảnh #6 `scratch` #1: Cần phân biệt vệt cọ trầy xước với vết bùn đất bám trên ốp sườn.
  - Ảnh #5 `scratch` #3: Cần kiểm tra có bị đè lấn với vùng scratch #1/#2 hay không.
- **Tổn thất nghi trùng lặp cần hợp nhất (Suspected Duplicate — 2 vết):**
  - Ảnh #8 `dent` #1 (`[421.7, 83.3, 757.8, 347.7]`) và `dent` #2 (`[138.6, 86.9, 811.0, 381.1]`): Hai hộp bao này mô tả cùng một vùng móp lớn trên cửa hông xe. Cần giám định viên hợp nhất thành 1 hộp bao duy nhất.

---

## 4. DANH SÁCH 9 ĐỐI TƯỢNG CÒN THIẾU TỌA ĐỘ (`review_candidates_pending`)

Đúng theo nguyên tắc **không tự suy đoán tọa độ khi chưa có công cụ chú thích**, các đối tượng sau được lưu giữ dưới dạng ghi chú nghiệp vụ để chờ con người vẽ bounding box / polygon:
1. **Ảnh #1:** Đèn xe (`den_xe`) bị vỡ nát, nhưng mô hình phân đoạn thiếu class này $\rightarrow$ Cần bổ sung polygon cho cụm đèn.
2. **Ảnh #2:** Vết nứt/vỡ kính nhỏ gần cột B trụ giữa $\rightarrow$ Cần phóng to kiểm tra và vẽ hộp bao bổ sung nếu có.
3. **Ảnh #3:** Vết cào xước (`scratch`) ở mép vòm bánh xe bị AI bỏ sót $\rightarrow$ Cần chuyên viên định vị và vẽ bbox.
4. **Ảnh #4:** Vết nứt cản macro $\rightarrow$ Cần ảnh góc rộng hơn để xác định vị trí cản trước hay cản sau.
5. **Ảnh #5:** Vùng tổn thất nằm ở vè bánh sau hoặc panel hông sau $\rightarrow$ Cần vẽ lại polygon cho vè sau thay vì cửa trước.
6. **Ảnh #6:** Vết trầy trên ốp sườn $\rightarrow$ Cần xác định là cửa dưới, ốp lườn hay bậc lên xuống.
7. **Ảnh #7:** Cụm đèn trước (`den_xe`) bị vỡ vỡ vụn $\rightarrow$ Cần vẽ polygon cụm đèn thay thế cho mask `kinh_xe` đã bị loại bỏ.
8. **Ảnh #8:** Cánh cửa (`cua_truoc`) bị móp $\rightarrow$ Cần vẽ polygon cánh cửa thay thế cho mask `can_truoc` đã bị loại bỏ.
9. **Ảnh #8:** Vết xước nhỏ xung quanh vùng móp $\rightarrow$ Cần kiểm tra xem có cần bổ sung annotation `scratch` hay không.

---

## 5. NHỮNG VẤN ĐỀ CHƯA THỂ GIẢI QUYẾT BẰNG DỮ LIỆU HIỆN CÓ

1. **Thiếu Lớp `den_xe` Trong Mô Hình Phân Đoạn (Stage 2 Segmentation Gap):**  
   Mô hình hiện tại chỉ huấn luyện 8 lớp nhưng không có cụm đèn xe trong tập segmentation, buộc pipeline phải dựa vào rule ngữ nghĩa `lamp_broken`. Khi làm Ground Truth chuẩn, cần có mask đa giác cho `den_xe` nhưng hiện chưa có công cụ dán nhãn giao diện trực quan trong terminal.
2. **Ảnh Siêu Cận Cảnh Mất Mốc Tham Chiếu (Context Truncation):**  
   Ảnh #4, #6, #8 là các ảnh chụp macro cận cảnh một vết nứt hoặc vết móp hẹp. Không có bất kỳ thuật toán CV hay con người nào có thể khẳng định chắc chắn 100% chi tiết kim loại trần đó thuộc về cản trước hay cản sau nếu không được cung cấp thêm ảnh chụp góc rộng của cùng chiếc xe đó.
3. **Hiện Tượng Trùng Ảnh Do Hệ Điều Hành Đặt Tên (`image copy...`):**  
   Tên file phản ánh việc thu thập ảnh demo chưa có quy trình quản lý định danh xe (Vehicle ID).

---

## 6. KẾT QUẢ KIỂM THỬ KỸ THUẬT TỰ ĐỘNG

Script kiểm tra độc lập `scratch/validate_clean_ground_truth.py` đã xác minh thành công 5 tiêu chí bắt buộc:

```text
=== VALIDATING CLEAN GROUND TRUTH FILE ===
[PASS] JSON parsed successfully.
[PASS] 8 images with unique IDs: [1, 2, 3, 4, 5, 6, 7, 8]
[PASS] Active parts: 6, Active damages: 13
[PASS] Excluded annotations quarantined: 9
[PASS] Verified count: 0 (100% pending_review)
[PASS] 0 exclude_* annotations found in active ground truth sets.
[PASS] Original draft and ground truth files are present and intact.
[PASS] Evaluator integration test: Evaluator safely BLOCKS execution with 0 verified annotations.
=== ALL VALIDATION CHECKS PASSED PERFECTLY! ===
```

---

## 7. DANH SÁCH VIỆC CẦN CON NGƯỜI THỰC HIỆN TIẾP THEO

Để hoàn tất quy trình từ bộ dữ liệu Clean sang Báo Cáo Đánh Giá Độ Chính Xác Chính Thức, chuyên viên giám định con người cần thực hiện chính xác 4 công việc sau:

1. **Vẽ Lại 6 Đa Giác Bộ Phận:** Dùng công cụ chú thích hình ảnh (ví dụ: Labelme, CVAT hoặc chỉnh tọa độ) vẽ lại đường biên polygon cho 6 bộ phận còn lại trong `parts_ground_truth` (đặc biệt là nắp capo Ảnh #7 và cửa trước Ảnh #2).
2. **Chia Nhỏ & Định Biên Lại Hộp Tổn Thất:**  
   - Cắt hộp bao `scratch` Ảnh #1 thành 2-3 hộp nhỏ bám theo đường xước cản thực tế.
   - Hợp nhất 2 hộp `dent` trong Ảnh #8 thành 1 hộp duy nhất.
3. **Bổ Sung Tọa Độ Cho 2 Cụm Đèn Xe (`den_xe`):** Thêm annotation polygon `den_xe` cho Ảnh #1 và Ảnh #7 để đánh giá công bằng cho tổn thất vỡ đèn pha.
4. **Ký Duyệt Từng Mục (`is_verified: true`):**  
   Sau khi hoàn tất chỉnh sửa tọa độ trực quan trên ảnh, đổi trường `"is_verified": true` và `"verification_status": "verified"` cho từng annotation đã duyệt trong `demo/annotations_ground_truth_clean.json`.
5. **Kích Hoạt Công Cụ Đánh Giá Độc Lập:**  
   ```bash
   python evaluation/run_evaluation.py --gt demo/annotations_ground_truth_clean.json
   ```
   Hệ thống sẽ tự động mở khóa trạng thái `BLOCKED` và tính toán chính xác mAP50, mIoU và Damage-to-Part Mapping Accuracy.
