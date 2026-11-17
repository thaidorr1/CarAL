# BÁO CÁO KỸ THUẬT: CHUẨN HÓA VÀ ÁNH XẠ DATASET NỀN TẢNG 8 LỚP
## (STAGE 2 STEP 2: CARPARTS-SEG 8-CLASS TAXONOMY REMAPPING REPORT)

**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Chuyên môn:** Senior Computer Vision Data Engineer  
**Tài liệu xuất ra:** `docs/ai_roadmap/stage2_step2_dataset_prep_report.md`  
**Thời điểm thực thi:** 14/11/2026  
**Trạng thái:** HOÀN THÀNH 100% (READY FOR STAGE 2 TRAINING)  

---

## 1. TỔNG QUAN TÁC VỤ THỰC HIỆN

Tiếp nối kết quả khảo sát tại Bước 1 ([`stage2_public_parts_survey.md`](file:///d:/AI_CAR/docs/ai_roadmap/stage2_public_parts_survey.md)), tác vụ **Bước 2** đã thực hiện tải về, xác minh và chuẩn hóa toàn bộ bộ dữ liệu nền tảng **Ultralytics `carparts-seg`** sang kiến trúc 8 lớp của AutoClaim-AI:

1. **Tải & Giải nén:** Đã tải nguyên vẹn tệp nén `carparts-seg.zip` (133.1 MB, SHA256 verified) vào thư mục tạm `scratch/downloads/carparts-seg` với đầy đủ 3 phân tập `train`, `val`, `test`.
2. **Quy chuẩn hóa 8 lớp (Taxonomy Remapping):** Xây dựng và thực thi script tự động [`ai/scripts/prepare_carparts_8cls.py`](file:///d:/AI_CAR/ai/scripts/prepare_carparts_8cls.py), trích xuất và ánh xạ từ 23 lớp gốc về đúng 8 ID mục tiêu (`0..7`).
3. **Kiểm soát chất lượng đa giác (Polygon QC):** Áp dụng thuật toán kiểm tra số lượng đỉnh ($\ge 3$ đỉnh), kẹp biên $[0.0, 1.0]$, bảo toàn các file `.txt` rỗng cho ảnh nền âm tính (negative background images).
4. **Cấu hình huấn luyện YOLOv8-seg:** Khởi tạo tệp cấu hình chuẩn [`ai/configs/data_parts_carparts_8cls.yaml`](file:///d:/AI_CAR/ai/configs/data_parts_carparts_8cls.yaml) với `nc: 8`.
5. **Kiểm chứng trực quan:** Xuất ảnh kiểm tra 4 mẫu ngẫu nhiên tại [`docs/references/carparts_8cls_sample_check.jpg`](file:///d:/AI_CAR/docs/references/carparts_8cls_sample_check.jpg).

---

## 2. BẢNG THỐNG KÊ PHÂN BỐ DỮ LIỆU VÀ ĐA GIÁC (DATASET METRICS)

### 2.1. Thống kê quy mô tệp và ảnh nền (Image-Level Stats)

| Phân tập (Split) | Tổng số ảnh | Số ảnh có nhãn hữu hiệu | Số ảnh nền (Empty Label) | Tỷ lệ ảnh nền | Trạng thái bảo toàn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 3,156 | 3,030 | 126 | 3.99% | Hardlinked / Copied |
| **VAL** | 401 | 388 | 13 | 3.24% | Hardlinked / Copied |
| **TEST** | 276 | 268 | 8 | 2.90% | Hardlinked / Copied |
| **TỔNG CỘNG** | **3,833** | **3,686** | **147** | **3.83%** | **100% Hoàn tất** |

*Ghi chú kỹ thuật: 147 ảnh nền rỗng xuất hiện do các ảnh gốc chỉ chứa các linh kiện bị loại bỏ (như bánh xe `wheel`, gương chiếu hậu `mirror`, hoặc cốp sau `trunk`). Việc giữ lại tệp `.txt` rỗng giúp mô hình YOLOv8-seg học cách triệt tiêu False Positive cho các vùng không thuộc 8 lớp mục tiêu.*

---

### 2.2. Thống kê số lượng Polygon (Instances) theo 8 lớp chuẩn hóa

| Class ID | Tên lớp mục tiêu | TRAIN | VAL | TEST | TỔNG CỘNG | Tỷ lệ (%) | Nguồn lớp gốc từ `carparts-seg` |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `can_truoc` | 1,616 | 208 | 154 | **1,978** | 10.86% | `8: front_bumper` |
| **1** | `can_sau` | 732 | 94 | 44 | **870** | 4.78% | `0: back_bumper` |
| **2** | `nap_capo` | 1,706 | 214 | 160 | **2,080** | 11.42% | `16: hood` |
| **3** | `ve_che_banh` | 0 | 0 | 0 | **0** | 0.00% | *(Giữ chỗ ID 3 cho Pilot/Roboflow)* |
| **4** | `cua_truoc` | 1,582 | 194 | 160 | **1,936** | 10.63% | `9, 11, 14` (`front_door`, `front_left/right_door`) |
| **5** | `cua_sau` | 1,396 | 186 | 149 | **1,731** | 9.50% | `1, 3, 6` (`back_door`, `back_left/right_door`) |
| **6** | `kinh_xe` | 2,572 | 329 | 219 | **3,120** | 17.13% | `2: back_glass`, `10: front_glass` |
| **7** | `den_xe` | 5,360 | 687 | 453 | **6,500** | 35.68% | `4, 5, 7, 12, 13, 15` (Các đèn pha & hậu) |
| **TỔNG** | **8 Lớp Mục Tiêu** | **14,964** | **1,912** | **1,339** | **18,215** | **100.0%** | **7/8 Lớp Ngoại Thất Đạt Chuẩn** |

---

## 3. BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG (POLYGON QC AUDIT)

Trong quá trình chuyển đổi và chuẩn hóa từng dòng nhãn:

1. **Đa giác suy biến (Degenerate Polygons):**
   - **Số lượng vi phạm:** **`0 đa giác`** (100% các đa giác trong tập gốc đều có $\ge 3$ đỉnh và số lượng tọa độ là số chẵn hợp lệ).
2. **Kẹp biên tọa độ (Coordinate Clamping):**
   - Toàn bộ tọa độ $[x, y]$ đều được chuẩn hóa và kiểm tra kẹp chặt trong ngưỡng an toàn $[0.000000, 1.000000]$ với độ chính xác 6 chữ số thập phân.
3. **Các instance bị loại bỏ theo thiết kế (Dropped Instances):**
   - Tổng cộng **2,159 instances** thuộc các lớp ngoài phạm vi đã được lọc bỏ an toàn, bao gồm:
     * `17: left_mirror` & `19: right_mirror` (Gương chiếu hậu ngoài)
     * `18: object` (Các vật thể thừa bên ngoài)
     * `20: tailgate` & `21: trunk` (Cốp sau / nắp thùng xe bán tải)
     * `22: wheel` (Bánh xe / lốp xe)

---

## 4. CẤU TRÚC THƯ MỤC VÀ CẤU HÌNH HUẤN LUYỆN

Dữ liệu sau khi xử lý được lưu độc lập tại:
```text
ai/data/parts_carparts_8cls/
├── images/
│   ├── train/ (3,156 ảnh)
│   ├── val/   (401 ảnh)
│   └── test/  (276 ảnh)
└── labels/
    ├── train/ (3,156 tệp .txt)
    ├── val/   (401 tệp .txt)
    └── test/  (276 tệp .txt)
```

Tệp cấu hình huấn luyện được thiết lập tại [`ai/configs/data_parts_carparts_8cls.yaml`](file:///d:/AI_CAR/ai/configs/data_parts_carparts_8cls.yaml):
```yaml
path: D:/AI_CAR/ai/data/parts_carparts_8cls
train: images/train
val: images/val
test: images/test

nc: 8
names:
  0: can_truoc
  1: can_sau
  2: nap_capo
  3: ve_che_banh
  4: cua_truoc
  5: cua_sau
  6: kinh_xe
  7: den_xe

mixup: 0.15
mosaic: 1.0
scale: 0.5
degrees: 10.0
```

---

## 5. KIỂM CHỨNG TRỰC QUAN (VISUAL INSPECTION)

Đã tạo ảnh đối chiếu trực quan 4 mẫu kiểm tra tại: [`docs/references/carparts_8cls_sample_check.jpg`](file:///d:/AI_CAR/docs/references/carparts_8cls_sample_check.jpg) (Kích thước $1.280 \times 1.006$ px, 317.1 KB).

- **Đánh giá hình học:**
  - `can_truoc` (Màu cam) & `can_sau` (Màu đỏ): Đường viền polygon ôm khít gờ cản nhựa, không bị cắt vuông chèn qua bánh xe như dataset cũ.
  - `nap_capo` (Màu tím cánh sen): Ôm chính xác theo khe hở ca-pô và chân kính chắn gió.
  - `cua_truoc` & `cua_sau` (Màu xanh lá): Phân định rõ ràng cột B-Pillar giữa hai cửa.
  - `kinh_xe` (Màu xanh dương): Bám sát viền gioăng kính cao su.
  - `den_xe` (Màu vàng): Nhận diện trọn vẹn chóa đèn pha trước và cụm đèn hậu.

---

## 6. KẾT LUẬN & SẴN SÀNG CHO BƯỚC 3

- **Mục tiêu đạt được:** Đã sở hữu tập dữ liệu nền tảng **3,833 ảnh** với **18,215 polygon thật** có độ chuẩn xác cao, đồng bộ 100% về taxonomy `0..7` với Pilot Dataset.
- **Ràng buộc an toàn:** Thư mục dữ liệu cũ `ai/data/parts/`, Pilot Dataset `ai/data/parts_pilot/`, code suy luận và model weights được giữ nguyên vẹn 100%.
- **Sẵn sàng:** Hệ thống đã sẵn sàng tuyệt đối để bước sang **Bước 3: Huấn luyện thử nghiệm Backbone Stage 2 (YOLOv8-seg 8-Class Pre-training)**.
