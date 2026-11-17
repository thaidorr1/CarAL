# BÁO CÁO KIỂM TOÁN ĐỘC LẬP: PILOT DATASET STAGE 2
**Mã kiểm toán:** P1.1.1 — Independent Pilot Dataset Audit  
**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Ngày thực hiện:** 13/11/2026  
**Chế độ:** READ-ONLY INDEPENDENT AUDIT  

---

## 1. KẾT QUẢ KIỂM TOÁN TỔNG THỂ (AUDIT VERDICT)

| STT | Hạng mục kiểm toán | Tiêu chí đánh giá | Kết quả | Ghi chú kỹ thuật |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Tính toàn vẹn Manifest** | Đủ 80 dòng, đường dẫn ảnh tồn tại 80/80, 0 bản ghi thiếu, 80 SHA256 độc nhất. | <span style="color:green">**PASS**</span> | Không có trùng lặp byte (0 duplicate SHA256). Tất cả ảnh đều đọc được qua Pillow và OpenCV. |
| **2** | **Định danh xe (Vehicle Identity)** | Đối chiếu `vehicle_id` với `base_vehicle_stem`, kiểm tra khả năng trùng xe. | <span style="color:orange">**PASS WITH OBSERVATION**</span> | 80 ảnh thuộc 80 stem khác nhau. Quét 3.160 cặp ảnh phát hiện 1 cặp nghi vấn cùng xe/cùng bối cảnh (#65 vs #67). |
| **3** | **Phân chia Train / Val** | 60 Train, 20 Val, không rò rỉ (Zero Leakage) theo xe thực tế. | <span style="color:orange">**CONDITIONAL PASS**</span> | Tỷ lệ 75/25 chính xác. Cần xử lý hoán đổi split cho cặp nghi vấn (#65 vs #67) trước khi gán nhãn. |
| **4** | **Kiểm tra Class Taxonomy** | Đối chiếu 8 lớp Pilot với 7 lớp mô hình `parts_seg.pt` và cấu hình YAML. | <span style="color:blue">**REQUIRES ALIGNMENT**</span> | Phát hiện độ lệch taxonomy: Tách `cua_hong` thành 2 cửa, mở rộng `kinh_lai` thành `kinh_xe`, bổ sung `den_xe`. |
| **5** | **Nguồn gốc Annotation** | Không dùng pseudo-label làm Ground Truth, không có polygon giả. | <span style="color:green">**PASS**</span> | Thư mục `ai/data/parts_pilot/` hoàn toàn sạch, chưa có file nhãn `.txt` nào được sinh tự động. |

> [!IMPORTANT]
> **KẾT LUẬN CHUNG:**
> Bộ dữ liệu 80 ảnh Pilot Dataset đạt trạng thái **READY FOR ANNOTATION (CÓ ĐIỀU KIỆN)**: Người phụ trách thẩm định chỉ cần xác nhận phương án xử lý đối với cặp ảnh nghi vấn `#65 vs #67` trước khi đưa vào công cụ gán nhãn (X-AnyLabeling / CVAT).

---

## 2. KIỂM TRA ĐỊNH DANH XE VÀ RÒ RỈ DỮ LIỆU (VEHICLE IDENTITY & LEAKAGE REPORT)

### 2.1. Phương pháp kiểm tra thực nghiệm
Do toàn bộ tập dữ liệu gốc CarDD đã bị xóa sạch thẻ EXIF (không có số VIN, không có thời gian chụp, biển số xe bị làm mờ), kiểm toán độc lập đã áp dụng thuật toán đối chiếu thị giác đa chiều trên toàn bộ $\binom{80}{2} = 3.160$ cặp ảnh:
1. **Perceptual Hash & Difference Hash (dHash):** So khớp tương quan hình học cấu trúc ảnh.
2. **HSV Color Histogram Cosine Distance:** So khớp tông màu sơn xe và ánh sáng môi trường.
3. **ORB Feature Matching (500 Keypoints):** Trích xuất và so khớp các điểm đặc trưng hình học giữa các cặp ảnh.

### 2.2. Kết quả phát hiện cặp nghi vấn
Trong 3.160 cặp so sánh, thuật toán phát hiện **duy nhất 1 cặp ảnh có chỉ số tương đồng vượt ngưỡng cảnh báo**:

| Ảnh 1 | Split 1 | Ảnh 2 | Split 2 | Điểm ORB Matches | Độ tương đồng màu | Đánh giá rủi ro |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#65** (`003947.jpg`) | **VAL** | **#67** (`003930.jpg`) | **TRAIN** | **62 matches** | **0.965** | **RỦI RO RÒ RỈ DỮ LIỆU (LEAKAGE RISK)** |

- **Phân tích chi tiết:**
  - Cả 2 ảnh đều là góc xiên $45^\circ$ sau (Rear-3/4 View).
  - Cả 2 ảnh đều có nhãn hư hỏng gốc CarDD là `glass_shatter` (vỡ rạn kính sau).
  - Tông màu xe, góc chiếu sáng và hậu cảnh có độ tương đồng màu sắc lên tới 96.5% và có 62 điểm đặc trưng trùng khớp.
  - Hiện tại, `#65` đang nằm ở tập `VAL`, còn `#67` lại nằm ở tập `TRAIN`. Nếu hai ảnh này là cùng một chiếc xe chụp ở hai góc chếch nhẹ trong cùng một vụ giám định, việc giữ nguyên split sẽ gây rò rỉ thông tin nền và bề mặt sơn giữa Train và Val.

### 2.3. Đề xuất khắc phục (2 phương án)
- **Phương án A (Khuyến nghị):** Chuyển `#67` sang tập `VAL` (cùng xe cùng tập), đồng thời hoán đổi một ảnh khác cùng nhóm góc từ `VAL` sang `TRAIN` (ví dụ `#65` và `#67` đều nằm ở `VAL`, chuyển `#70 (000072.jpg)` sang `TRAIN`).
- **Phương án B:** Thay thế ảnh `#67` bằng một ảnh góc xiên $45^\circ$ sau hoàn toàn độc lập khác từ kho 281 ảnh Rear-3/4 của CarDD.

---

## 3. ĐỐI CHIẾU VÀ ĐỀ XUẤT THỐNG NHẤT CLASS TAXONOMY

### 3.1. Bảng đối chiếu thực trạng
Hiện tại có sự khác biệt giữa mô hình cũ Stage 2 và bộ dữ liệu Pilot 8 lớp mới:

| ID | Class Model Hiện Tại (`parts_seg.pt` & `parts.yaml`) | Class Mục Tiêu Pilot Dataset (8 Lớp) | Bản chất thay đổi | Đánh giá kỹ thuật |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `can_truoc` (Cản trước) | `can_truoc` (Cản trước) | Giữ nguyên 1:1 | Đồng bộ hoàn hảo. |
| 1 | `can_sau` (Cản sau) | `can_sau` (Cản sau) | Giữ nguyên 1:1 | Đồng bộ hoàn hảo. |
| 2 | `nap_capo` (Nắp capo) | `nap_capo` (Nắp capo) | Giữ nguyên 1:1 | Đồng bộ hoàn hảo. |
| 3 | `ve_che_banh` (Vè che bánh) | `ve_che_banh` (Vè che bánh) | Giữ nguyên 1:1 | Đồng bộ hoàn hảo. |
| 4 | `cua_hong` (Cửa hông nói chung) | `cua_truoc` (Cửa trước)<br>`cua_sau` (Cửa sau) | **Tách 1 thành 2** | Khắc phục triệt để lỗi gộp cửa, cho phép phân định chính xác chi phí bồi thường cửa trước/sau. |
| 5 | `kinh_chieu_hau` (Gương chiếu hậu) | *(Không có trong 8 lớp Pilot)* | **Chưa có trong Pilot** | Cần quyết định xem `kinh_chieu_hau` có gán nhãn thành lớp thứ 9 hay gộp vào thân vỏ. |
| 6 | `kinh_lai` (Kính chắn gió trước) | `kinh_xe` (Toàn bộ bề mặt kính) | **Mở rộng phạm vi** | Hợp lý hóa vì tai nạn vỡ kính thường xảy ra trên cả kính lái, kính sau và kính cửa sổ. |
| 7 | *(Không tồn tại trong model cũ)* | `den_xe` (Đèn pha, đèn hậu) | **Bổ sung mới 100%** | Khắc phục thiếu sót nghiêm trọng nhất của Stage 2 cũ (đèn xe là bộ phận tổn thất đắt tiền nhất). |

### 3.2. Đề xuất Taxonomy Thống Nhất cho Giai Đoạn Huấn Luyện (Stage 2 Retrain)
Để pipeline đồng bộ và tương thích mượt mà giữa mô hình và API giám định, đề xuất ontology chuẩn hóa **8 lớp hoặc 9 lớp** như sau:

```yaml
# Đề xuất cấu hình chuẩn hóa Stage 2 (ai/configs/parts_pilot.yaml)
nc: 8
names:
  0: can_truoc       # Cản trước (Front Bumper)
  1: can_sau         # Cản sau (Rear Bumper)
  2: nap_capo        # Nắp capo (Hood)
  3: ve_che_banh     # Vè che bánh (Fender / Quarter Panel)
  4: cua_truoc       # Cửa trước (Front Door)
  5: cua_sau         # Cửa sau (Rear Door)
  6: kinh_xe         # Kính xe (Windshield, Rear & Side Windows)
  7: den_xe          # Đèn xe (Headlights, Taillights, Fog lamps)

# Tùy chọn Class 8: kinh_chieu_hau (Gương chiếu hậu) nếu muốn giữ lại
```

*(Lưu ý: Không chỉnh sửa các file config hiện có trong giai đoạn kiểm toán theo đúng ràng buộc).*

---

## 4. KIỂM SOÁT NGUỒN GỐC ANNOTATION (PROVENANCE & INTEGRITY)

- **Kiểm tra thư mục `ai/data/parts_pilot/`:**
  - Tệp tồn tại: `pilot_manifest.csv` (33.5 KB) và `pilot_contact_sheet.jpg` (1.55 MB).
  - Hoàn toàn **KHÔNG CÓ** tệp nhãn `.txt` hay `.json` nào được tự động tạo ra trong thư mục này.
- **Kiểm tra liên kết với nhãn cũ trong repo:**
  - Trong 80 ảnh được chọn, chỉ có 5 ảnh từng xuất hiện trong tập `ai/data/parts/labels/` (chứa polygon 4 đỉnh chữ nhật sai quy chuẩn).
  - Toàn bộ 80 ảnh đều có nhãn hộp tổn thất trong `ai/data/yolo/labels/` (nhãn CarDD), phục vụ rất tốt cho việc định vị hư hỏng khi con người gán nhãn đa giác bộ phận xe.
- **Kết luận:** Hoàn toàn không có hiện tượng dùng pseudo-label làm Ground Truth. Dữ liệu thuần khiết ở trạng thái ảnh thô (Raw Images) chờ chuyên gia con người thẩm định và gán nhãn.

---

## 5. NHỮNG VẤN ĐỀ CHƯA THỂ XÁC MINH (UNVERIFIED ISSUES)

1. **Lịch sử hồ sơ tai nạn gốc của CarDD:**
   - Do CarDD là tập dữ liệu mở đã ẩn danh hóa, không thể truy xuất biên bản bồi thường thực tế hay góc chụp thứ hai được xác thực bằng metadata cứng (EXIF Timestamp/GPS) của từng xe.
2. **Độ biến dạng quá mức ở các vụ va chạm nặng:**
   - Trong một số ảnh trung cảnh va chạm dập nát, ranh giới giữa vè xe (`ve_che_banh`) và cản trước (`can_truoc`) bị rách toác dính liền. Khi gán nhãn con người cần có quy ước phân định đường biên dựa trên cấu trúc khung gầm nguyên bản.

---

## 6. DANH MỤC CHI TIẾT TOÀN BỘ 80 ẢNH TRONG PILOT DATASET

Dưới đây là bảng trích xuất toàn bộ 80 bản ghi từ [`ai/data/parts_pilot/pilot_manifest.csv`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_manifest.csv):

| ID | File | Vehicle ID | Split | Viewpoint | Visible Parts | Image Quality | Selection Rationale |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| #1 | `003848.jpg` | `VEH_003848` | **VAL** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1291.6, Brightness=142.1 | Ảnh góc trước trực diện |
| #2 | `001977.jpg` | `VEH_001977` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1263.8, Brightness=122.9 | Ảnh góc trước trực diện |
| #3 | `000049.jpg` | `VEH_000049` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1259.0, Brightness=110.2 | Ảnh góc trước trực diện |
| #4 | `003664.jpg` | `VEH_003664` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1240.2, Brightness=137.9 | Ảnh góc trước trực diện |
| #5 | `001007.jpg` | `VEH_001007` | **VAL** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1230.1, Brightness=106.3 | Ảnh góc trước trực diện |
| #6 | `003977.jpg` | `VEH_003977` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1224.2, Brightness=126.8 | Ảnh góc trước trực diện |
| #7 | `002661.jpg` | `VEH_002661` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1189.7, Brightness=114.7 | Ảnh góc trước trực diện |
| #8 | `002733.jpg` | `VEH_002733` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1173.2, Brightness=75.7 | Ảnh góc trước trực diện |
| #9 | `003126.jpg` | `VEH_003126` | **VAL** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1164.5, Brightness=98.9 | Ảnh góc trước trực diện |
| #10 | `001550.jpg` | `VEH_001550` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1152.0, Brightness=142.0 | Ảnh góc trước trực diện |
| #11 | `000452.jpg` | `VEH_000452` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1148.6, Brightness=128.5 | Ảnh góc trước trực diện |
| #12 | `003463.jpg` | `VEH_003463` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1146.9, Brightness=139.7 | Ảnh góc trước trực diện |
| #13 | `001479.jpg` | `VEH_001479` | **VAL** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 669x1000, Sharpness=1143.0, Brightness=162.2 | Ảnh góc trước trực diện |
| #14 | `000084.jpg` | `VEH_000084` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1138.3, Brightness=122.9 | Ảnh góc trước trực diện |
| #15 | `000780.jpg` | `VEH_000780` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1134.4, Brightness=127.7 | Ảnh góc trước trực diện |
| #16 | `002131.jpg` | `VEH_002131` | **TRAIN** | Góc trước | `can_truoc, nap_capo, kinh_xe, den_xe` | 1000x667, Sharpness=1132.8, Brightness=110.1 | Ảnh góc trước trực diện |
| #17 | `001489.jpg` | `VEH_001489` | **VAL** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1420.7, Brightness=112.5 | Ảnh góc sau trực diện |
| #18 | `000109.jpg` | `VEH_000109` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1220.4, Brightness=119.5 | Ảnh góc sau trực diện |
| #19 | `002875.jpg` | `VEH_002875` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1219.0, Brightness=120.3 | Ảnh góc sau trực diện |
| #20 | `003359.jpg` | `VEH_003359` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1214.3, Brightness=120.7 | Ảnh góc sau trực diện |
| #21 | `001221.jpg` | `VEH_001221` | **VAL** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1189.6, Brightness=130.6 | Ảnh góc sau trực diện |
| #22 | `001356.jpg` | `VEH_001356` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1164.7, Brightness=119.7 | Ảnh góc sau trực diện |
| #23 | `000980.jpg` | `VEH_000980` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1163.7, Brightness=115.5 | Ảnh góc sau trực diện |
| #24 | `003177.jpg` | `VEH_003177` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1150.3, Brightness=114.9 | Ảnh góc sau trực diện |
| #25 | `001183.jpg` | `VEH_001183` | **VAL** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1147.2, Brightness=128.8 | Ảnh góc sau trực diện |
| #26 | `000788.jpg` | `VEH_000788` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1146.5, Brightness=130.3 | Ảnh góc sau trực diện |
| #27 | `003665.jpg` | `VEH_003665` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1143.9, Brightness=136.2 | Ảnh góc sau trực diện |
| #28 | `001537.jpg` | `VEH_001537` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1128.0, Brightness=123.6 | Ảnh góc sau trực diện |
| #29 | `001948.jpg` | `VEH_001948` | **VAL** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1122.9, Brightness=115.3 | Ảnh góc sau trực diện |
| #30 | `002761.jpg` | `VEH_002761` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1122.1, Brightness=118.9 | Ảnh góc sau trực diện |
| #31 | `003341.jpg` | `VEH_003341` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1114.2, Brightness=190.1 | Ảnh góc sau trực diện |
| #32 | `002394.jpg` | `VEH_002394` | **TRAIN** | Góc sau | `can_sau, kinh_xe, den_xe` | 1000x667, Sharpness=1110.8, Brightness=117.8 | Ảnh góc sau trực diện |
| #33 | `000001.jpg` | `VEH_000001` | **VAL** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x750, Sharpness=1647.7, Brightness=134.1 | Ảnh góc nhìn ngang thân xe |
| #34 | `001391.jpg` | `VEH_001391` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1374.9, Brightness=137.4 | Ảnh góc nhìn ngang thân xe |
| #35 | `001076.jpg` | `VEH_001076` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1224.2, Brightness=126.7 | Ảnh góc nhìn ngang thân xe |
| #36 | `002636.jpg` | `VEH_002636` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1184.2, Brightness=125.6 | Ảnh góc nhìn ngang thân xe |
| #37 | `001081.jpg` | `VEH_001081` | **VAL** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x686, Sharpness=1179.3, Brightness=112.2 | Ảnh góc nhìn ngang thân xe |
| #38 | `000736.jpg` | `VEH_000736` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1142.1, Brightness=118.0 | Ảnh góc nhìn ngang thân xe |
| #39 | `000602.jpg` | `VEH_000602` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1141.0, Brightness=79.1 | Ảnh góc nhìn ngang thân xe |
| #40 | `000305.jpg` | `VEH_000305` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1129.8, Brightness=142.6 | Ảnh góc nhìn ngang thân xe |
| #41 | `000728.jpg` | `VEH_000728` | **VAL** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1127.4, Brightness=130.9 | Ảnh góc nhìn ngang thân xe |
| #42 | `001012.jpg` | `VEH_001012` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1126.9, Brightness=129.8 | Ảnh góc nhìn ngang thân xe |
| #43 | `003265.jpg` | `VEH_003265` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1107.0, Brightness=112.5 | Ảnh góc nhìn ngang thân xe |
| #44 | `001925.jpg` | `VEH_001925` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1105.7, Brightness=131.7 | Ảnh góc nhìn ngang thân xe |
| #45 | `002166.jpg` | `VEH_002166` | **VAL** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1105.5, Brightness=137.9 | Ảnh góc nhìn ngang thân xe |
| #46 | `001202.jpg` | `VEH_001202` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1097.2, Brightness=119.0 | Ảnh góc nhìn ngang thân xe |
| #47 | `002170.jpg` | `VEH_002170` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1095.5, Brightness=115.1 | Ảnh góc nhìn ngang thân xe |
| #48 | `000734.jpg` | `VEH_000734` | **TRAIN** | Góc bên hông | `cua_truoc, cua_sau, ve_che_banh, kinh_xe` | 1000x667, Sharpness=1090.9, Brightness=127.2 | Ảnh góc nhìn ngang thân xe |
| #49 | `002936.jpg` | `VEH_002936` | **VAL** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1420.2, Brightness=126.9 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #50 | `001716.jpg` | `VEH_001716` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1380.0, Brightness=128.5 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #51 | `001323.jpg` | `VEH_001323` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1374.6, Brightness=117.8 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #52 | `000645.jpg` | `VEH_000645` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1349.5, Brightness=123.6 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #53 | `002288.jpg` | `VEH_002288` | **VAL** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1334.3, Brightness=134.1 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #54 | `002577.jpg` | `VEH_002577` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1327.9, Brightness=127.9 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #55 | `001715.jpg` | `VEH_001715` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1308.2, Brightness=124.6 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #56 | `001742.jpg` | `VEH_001742` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1303.4, Brightness=123.9 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #57 | `002425.jpg` | `VEH_002425` | **VAL** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1295.5, Brightness=124.8 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #58 | `000009.jpg` | `VEH_000009` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1210.0, Brightness=132.0 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #59 | `000077.jpg` | `VEH_000077` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1198.8, Brightness=126.3 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #60 | `001284.jpg` | `VEH_001284` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x750, Sharpness=1186.7, Brightness=134.5 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #61 | `003882.jpg` | `VEH_003882` | **VAL** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1184.2, Brightness=123.5 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #62 | `001552.jpg` | `VEH_001552` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1182.2, Brightness=127.3 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #63 | `000454.jpg` | `VEH_000454` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1164.2, Brightness=124.7 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #64 | `000762.jpg` | `VEH_000762` | **TRAIN** | Góc xiên 45 độ trước | `can_truoc, nap_capo, ve_che_banh, cua_truoc, den_xe, kinh_xe` | 1000x667, Sharpness=1158.4, Brightness=127.7 | Ảnh góc xiên 45 độ trước thể hiện liên hoàn 6 bộ phận: cản trước |
| #65 | `003947.jpg` | `VEH_003947` | **VAL** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x676, Sharpness=1384.6, Brightness=116.4 | Ảnh góc xiên 45 độ sau |
| #66 | `000026.jpg` | `VEH_000026` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1264.4, Brightness=123.6 | Ảnh góc xiên 45 độ sau |
| #67 | `003930.jpg` | `VEH_003930` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x602, Sharpness=1255.4, Brightness=102.5 | Ảnh góc xiên 45 độ sau |
| #68 | `003239.jpg` | `VEH_003239` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1214.3, Brightness=124.8 | Ảnh góc xiên 45 độ sau |
| #69 | `000962.jpg` | `VEH_000962` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1211.2, Brightness=123.9 | Ảnh góc xiên 45 độ sau |
| #70 | `000072.jpg` | `VEH_000072` | **VAL** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1128.5, Brightness=126.8 | Ảnh góc xiên 45 độ sau |
| #71 | `001389.jpg` | `VEH_001389` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1126.4, Brightness=130.4 | Ảnh góc xiên 45 độ sau |
| #72 | `001059.jpg` | `VEH_001059` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1104.9, Brightness=123.8 | Ảnh góc xiên 45 độ sau |
| #73 | `002503.jpg` | `VEH_002503` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1103.7, Brightness=123.6 | Ảnh góc xiên 45 độ sau |
| #74 | `003714.jpg` | `VEH_003714` | **TRAIN** | Góc xiên 45 độ sau | `can_sau, ve_che_banh, cua_sau, den_xe, kinh_xe` | 1000x667, Sharpness=1098.2, Brightness=125.1 | Ảnh góc xiên 45 độ sau |
| #75 | `000052.jpg` | `VEH_000052` | **VAL** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 1000x750, Sharpness=1351.4, Brightness=100.9 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |
| #76 | `003862.jpg` | `VEH_003862` | **TRAIN** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 750x1000, Sharpness=1295.6, Brightness=107.8 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |
| #77 | `001157.jpg` | `VEH_001157` | **TRAIN** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 1000x667, Sharpness=1285.5, Brightness=128.6 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |
| #78 | `003095.jpg` | `VEH_003095` | **VAL** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 1000x750, Sharpness=1284.2, Brightness=99.2 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |
| #79 | `003450.jpg` | `VEH_003450` | **TRAIN** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 1000x667, Sharpness=1264.8, Brightness=122.9 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |
| #80 | `000039.jpg` | `VEH_000039` | **TRAIN** | Trung cảnh | `can_truoc, nap_capo, ve_che_banh, den_xe` | 1000x667, Sharpness=1255.4, Brightness=125.3 | Ảnh trung cảnh chi tiết góc cụm đầu xe/đuôi xe |

---

## 7. KẾT LUẬN VÀ KIẾN NGHỊ PHÊ DUYỆT

Kiểm toán độc lập xác nhận:
1. Manifest và hình ảnh đáp ứng đầy đủ tính toàn vẹn vật lý và độ nét cao ($100\%$).
2. Không có nhãn giả dạng Canny hay hình chữ nhật 4 đỉnh bị đưa vào bộ Ground Truth chuẩn.
3. Độ bao phủ 8 lớp đồng đều, giải quyết triệt để vấn đề thiếu hụt đèn xe và gộp cửa hông.

**Kiến nghị hành động trước khi gán nhãn:**
- Thực hiện xử lý cặp nghi vấn `#65 vs #67` theo Phương án A (chuyển `#67` sang `VAL` và đưa `#70` sang `TRAIN` để bảo đảm tỷ lệ $60/20$ và triệt tiêu nguy cơ rò rỉ).
- Sau khi được người dùng phê duyệt phương án trên, bộ dữ liệu chính thức chuyển sang giai đoạn **P1.2: Human Annotation with MobileSAM Assistance**.
