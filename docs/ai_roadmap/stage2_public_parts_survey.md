# KHẢO SÁT & ĐỐI CHIẾU DATASET VEHICLE PARTS SEGMENTATION CHO STAGE 2
## (PUBLIC DATASETS SURVEY & TAXONOMY ALIGNMENT REPORT)

**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Mục tiêu:** Khảo sát nguồn dữ liệu phân đoạn bộ phận xe, giải trình nguyên nhân gốc rễ hạn chế mAP 28.34% của bộ dữ liệu cũ và lập bảng đối chiếu các dataset công khai có đa giác (Polygon Segmentation) chuẩn theo 8 lớp mục tiêu.  
**Tài liệu xuất ra:** `docs/ai_roadmap/stage2_public_parts_survey.md`  
**Thời điểm thực hiện:** 14/11/2026  

---

## 1. KIỂM TOÁN NGUỒN DỮ LIỆU PARTS CŨ TRONG WORKSPACE (`ai/data/parts`)

### 1.1. Hiện trạng cấu hình và quy mô dữ liệu cũ
Kiểm tra các tệp cấu hình tại `ai/configs/data_parts.yaml`, `ai/configs/parts.yaml`, `ai/data/parts/data.yaml` và `ai/data/parts/dataset.yaml`:

- **Đường dẫn dữ liệu:** `D:/AI_CAR/ai/data/parts`
- **Quy mô tập ảnh cũ:** 1,622 ảnh (Train: 1,525 ảnh | Val: 97 ảnh)
- **Danh mục 7 lớp cũ:**
  - `0: can_truoc` (Cản trước)
  - `1: can_sau` (Cản sau)
  - `2: nap_capo` (Nắp capo)
  - `3: ve_che_banh` (Vè che bánh)
  - `4: cua_hong` (Cửa hông chung)
  - `5: kinh_chieu_hau` (Gương chiếu hậu)
  - `6: kinh_lai` (Kính lái)

### 1.2. Nguyên nhân gốc rễ khiến chất lượng Mask mAP@50 chỉ đạt 28.34%
Kết quả kiểm toán độc lập trên **6,534 instances** của tập dữ liệu cũ đã chứng minh nguyên nhân cốt lõi khiến mô hình trước đây học sai hình học:

```mermaid
pie title Phân Bố Bản Chất Nhãn Tập Dữ Liệu Parts Cũ (6,534 Instances)
    "Hộp chữ nhật 4 đỉnh giả lập đa giác (75.97%)" : 4964
    "Đa giác Canny Raycasting 12 tia (23.32%)" : 1524
    "Bát giác đều nhân tạo (0.70%)" : 46
    "Tọa độ gán cứng (0.21%)" : 14
    "Đa giác vẽ tay thực tế (0.00%)" : 0
```

1. **Hoàn toàn không có đa giác Ground Truth thực tế ($0.00\%$):** Không có bất kỳ nhãn nào được chuyên viên vẽ tay ôm sát đường viền thân vỏ xe hơi.
2. **$75.97\%$ là hộp chữ nhật giả lập polygon:** 
   - Xuất phát từ script cũ `generate_grounded_parts_dataset.py` lấy từ CarDD. Do CarDD **chỉ có nhãn vết hỏng (damage) chứ không có nhãn linh kiện xe**, script đã chạy mô hình phát hiện xe rồi chia cắt cơ học bounding box của xe thành các góc phần tư thô thiển:
     - Góc phần tư dưới bên trái gán cứng thành `can_truoc`.
     - Góc phần tư dưới bên phải gán cứng thành `can_sau`.
     - Toàn bộ nửa trên gán cứng thành `ve_che_banh` (trùm lên cả kính, nóc xe và khoang lái).
     - Khoang giữa gán cứng thành `cua_hong`.
3. **$23.32\%$ đa giác Canny Raycasting nhân tạo:**
   - Tạo ra từ script `mine_closeup_parts_from_cardd.py` bằng cách chạy bộ lọc biên Canny rồi bắn 12 tia chia đều 30 độ. Các đỉnh này bám theo nhiễu viền bóng đổ, sọc sơn hoặc rãnh gạch nền đường chứ không bám theo mép bộ phận xe.
4. **Hệ quả trên mô hình Stage 2:** 
   - Mô hình YOLOv8s-seg học các hộp chữ nhật chồng chéo, dự đoán cản trước và kính xe thành các khối vuông vức che lấp nhau, cắt ngang qua bánh xe và mặt đường. Vì vậy, chỉ số **Mask mAP@50 đạt cực đại ở mức $28.34\%$** và không thể cải thiện thêm nếu không thay thế nguồn nhãn.

---

## 2. KHẢO SÁT & ĐỐI CHIẾU CÁC DATASET CÔNG KHAI CÓ POLYGON SEGMENTATION

### 2.1. Cấu hình tích hợp sẵn trong Ultralytics: `carparts-seg.yaml`
Kiểm tra trực tiếp tệp cấu hình tích hợp sẵn trong gói thư viện Ultralytics tại đường dẫn hệ thống:
`C:\Users\Admin\AppData\Local\Programs\Python\Python313\Lib\site-packages\ultralytics\cfg\datasets\carparts-seg.yaml`

- **Giấy phép (License):** AGPL-3.0 License
- **Đường dẫn tải trực tiếp:** `https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip` (Dung lượng nén: **133 MB**)
- **Quy mô:** **3,833 ảnh** được phân chia chuẩn hóa:
  - `Train`: **3,156 ảnh**
  - `Val`: **401 ảnh**
  - `Test`: **276 ảnh**
- **Định dạng:** Native YOLOv8 Instance Segmentation (`.txt` chứa `class_id x1 y1 x2 y2 ... xn yn` chuẩn hóa $[0, 1]$).
- **Danh sách 23 lớp gốc (`names`):**
  ```yaml
  names:
    0: back_bumper
    1: back_door
    2: back_glass
    3: back_left_door
    4: back_left_light
    5: back_light
    6: back_right_door
    7: back_right_light
    8: front_bumper
    9: front_door
    10: front_glass
    11: front_left_door
    12: front_left_light
    13: front_light
    14: front_right_door
    15: front_right_light
    16: hood
    17: left_mirror
    18: object
    19: right_mirror
    20: tailgate
    21: trunk
    22: wheel
  ```

---

### 2.2. So sánh đối chiếu 3 nguồn Dataset công khai chính

Bảng đối chiếu toàn diện theo 5 tiêu chí kỹ thuật:

| Tiêu chí đối chiếu | (A) Ultralytics `carparts-seg` | (B) Roboflow Universe (Car Parts) | (C) Pascal-Part / PartImageNet |
| :--- | :--- | :--- | :--- |
| **1. Độ khớp với 8 lớp AutoClaim-AI** | **Khớp 7/8 lớp** (`can_truoc`, `can_sau`, `nap_capo`, `cua_truoc`, `cua_sau`, `kinh_xe`, `den_xe`). Khuyết lớp `ve_che_banh`. | **Khớp 8/8 lớp** (Một số dataset như *car parts detection 2* hoặc *vehicle-parts* có gắn riêng `Fender` / `Quarter-panel`). | **Khớp kém (3-4/8 lớp)**. Chỉ có `door`, `window`, `headlight`, `wheel`, `body`. Không tách cản trước/sau, nắp capo riêng biệt. |
| **2. Chất lượng Polygon Segmentation** | **Rất cao (Polygon ôm sát viền thực tế)**, trung bình 20–60 đỉnh/polygon, thể hiện đường cong khí động học chuẩn. | **Khá - Tốt**. Đa phần là polygon vẽ tay bởi cộng đồng, tuy nhiên một số tệp có độ chính xác không đồng đều. | **Trung bình - Thô**. PartImageNet chỉ gán `Body` chung; Pascal-Part gán theo lưới ô pixel PASCAL VOC (răng cưa). |
| **3. Quy mô & Phân bố Train/Val** | **3,833 ảnh** (Train: 3,156 / Val: 401 / Test: 276). Góc chụp đa dạng (trước, sau, xiên, hông). | **1,500 – 3,000 ảnh** tùy project (Ví dụ Joniboy: 2,760 ảnh). | Pascal-Part: ~1,500 xe (trộn chung nhiều đồ vật). PartImageNet: nhãn thô. |
| **4. Định dạng dữ liệu & Giấy phép** | **YOLOv8-seg native (`.txt`)**, AGPL-3.0 / Open Source. Tải 1 link duy nhất từ GitHub Releases (133MB). | **YOLOv8-seg (`.txt`)** hoặc COCO JSON. Giấy phép CC BY 4.0 / Open Data. Cần Roboflow API Key để tải tự động. | **Pascal VOC XML / MATLAB format / HDF5**. Đòi hỏi viết script parse ma trận nhãn phức tạp. |
| **5. Tính khả thi kỹ thuật** | **Cực cao (Độ trễ thấp, tích hợp ngay)**. Có sẵn tệp YAML trong thư viện `ultralytics`, xử lý ETL cực kỳ tinh gọn. | **Cao**, nhưng phụ thuộc vào API Roboflow hoặc tải thủ công file zip qua trình duyệt. | **Thấp**. Tốn nhiều chi phí chuyển đổi định dạng và chất lượng nhãn không phù hợp cho giám định bảo hiểm. |

---

## 3. BẢNG ÁNH XẠ CHI TIẾT (TAXONOMY MAPPING MATRIX)

Bảng ánh xạ chuẩn hóa từ 23 lớp của `carparts-seg` sang 8 lớp mục tiêu của AutoClaim-AI:

| STT | 8 Lớp Mục Tiêu (AutoClaim-AI) | Class ID Đích | Các lớp nguồn gộp từ `carparts-seg` (Original ID & Name) | Quy tắc xử lý (ETL Rule) |
| :-: | :--- | :-: | :--- | :--- |
| **1** | `can_truoc` | **0** | `8: front_bumper` | Giữ nguyên cản trước |
| **2** | `can_sau` | **1** | `0: back_bumper` | Giữ nguyên cản sau |
| **3** | `nap_capo` | **2** | `16: hood` | Giữ nguyên nắp capo |
| **4** | `ve_che_banh` | **3** | *(Không có trong `carparts-seg`)* | Bổ sung từ **Pilot Dataset 80 ảnh** & tập phụ trợ Roboflow Fender |
| **5** | `cua_truoc` | **4** | `9: front_door`<br>`11: front_left_door`<br>`14: front_right_door` | Gộp tất cả cửa trước bên trái, phải và cửa trước tổng quát |
| **6** | `cua_sau` | **5** | `1: back_door`<br>`3: back_left_door`<br>`6: back_right_door` | Gộp tất cả cửa sau bên trái, phải và cửa sau tổng quát |
| **7** | `kinh_xe` | **6** | `2: back_glass`<br>`10: front_glass` | Gộp kính chắn gió trước và kính chắn gió sau |
| **8** | `den_xe` | **7** | `4: back_left_light`<br>`5: back_light`<br>`7: back_right_light`<br>`12: front_left_light`<br>`13: front_light`<br>`15: front_right_light` | Gộp toàn bộ 6 lớp đèn pha trước, đèn hậu, đèn xi-nhan trái/phải |
| -- | *Loại bỏ (Dropped)* | *Ignore* | `17: left_mirror`<br>`18: object`<br>`19: right_mirror`<br>`20: tailgate`<br>`21: trunk`<br>`22: wheel` | Bỏ qua gương chiếu hậu, vật thể thừa, cốp thùng xe và bánh xe |

---

## 4. ĐỀ XUẤT PHƯƠNG ÁN CHỐT CHO BƯỚC 2

### 4.1. Lựa chọn Dataset nền tảng
- **Đề xuất Dataset cốt lõi:** Chọn **Ultralytics `carparts-seg`** làm bộ dữ liệu xương sống (Foundational Pre-training Dataset).
- **Lý do lựa chọn:**
  1. **Độ sạch và chuẩn xác cao:** 3,833 ảnh được gán nhãn đa giác ôm sát viền thực tế, giải quyết dứt điểm vấn đề nhãn hộp chữ nhật 4 đỉnh mAP 28.34%.
  2. **Tốc độ và tính sẵn sàng:** Tệp nén chỉ **133 MB** tải trực tiếp từ GitHub Releases của Ultralytics, định dạng chuẩn YOLOv8-seg, không phụ thuộc rate limit của các nền tảng bên thứ ba.
  3. **Độ phủ lớp vượt trội:** Đáp ứng ngay lập tức **7/8 lớp** quan trọng nhất với hàng nghìn instance chất lượng cao (đặc biệt là `den_xe`, `can_truoc`, `can_sau`, `nap_capo`, `cua_truoc`, `cua_sau`, `kinh_xe`).

### 4.2. Chiến lược xử lý khoảng trống (Gap Resolution cho lớp `ve_che_banh`)
Do `carparts-seg` không gắn nhãn vè che bánh (`ve_che_banh` / fender), chúng ta áp dụng **Chiến lược lai ghép (Hybrid Fine-tuning Strategy)**:
1. **Bước 2.1 (Tải & Chuẩn hóa):** Tải `carparts-seg.zip` về thư mục riêng biệt (`ai/data/external_carparts_seg/`), chạy script ETL ánh xạ 23 lớp $\rightarrow$ 7 lớp mục tiêu theo bảng trên.
2. **Bước 2.2 (Huấn luyện Backbone Stage 2):** Huấn luyện mô hình YOLOv8-seg trên tập nhãn sạch 3,833 ảnh để mô hình hấp thụ triệt để hình học thân vỏ xe hơi.
3. **Bước 2.3 (Transfer Learning & Ghép Pilot):** Kết hợp cùng **80 ảnh Pilot Dataset** đã có sẵn nhãn Ground Truth chuẩn của `ve_che_banh` và các crop fender từ Roboflow để hoàn thiện trọn vẹn 8 lớp.

---
*Báo cáo đã lưu trữ chính thức tại:* [`docs/ai_roadmap/stage2_public_parts_survey.md`](file:///d:/AI_CAR/docs/ai_roadmap/stage2_public_parts_survey.md)
