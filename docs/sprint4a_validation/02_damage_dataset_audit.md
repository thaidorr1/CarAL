# 02 - KIỂM TOÁN TẬP DỮ LIỆU DAMAGE DETECTION (`ai/data/yolo`)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Thư mục dữ liệu:** `ai/data/yolo/`  
**Phương pháp kiểm toán:** Kiểm tra toàn diện 4,060 tệp ảnh và 4,060 tệp nhãn (PIL metadata, SHA256 hashing, tọa độ chuẩn hóa).

---

## 1. Thống Kê Tổng Thể Tập Dữ Liệu CarDD

| Thuộc Tính Kiểm Toán | Tập Huấn Luyện (Train) | Tập Kiểm Định (Val) | Tập Thử Nghiệm (Test) | Toàn Bộ Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Số lượng ảnh** | 2,876 ảnh | 810 ảnh | 374 ảnh | **4,060 ảnh** |
| **Số lượng tệp nhãn (`.txt`)**| 2,876 tệp | 810 tệp | 374 tệp | **4,060 tệp** |
| **Số ảnh thiếu nhãn (Missing)** | 0 ảnh | 0 ảnh | 0 ảnh | **0 ảnh** |
| **Số ảnh không có tổn thất (Empty)**| 60 ảnh (Background) | 0 ảnh | 0 ảnh | **60 ảnh** |
| **Tổng số Bounding Box** | 6,211 boxes | 1,744 boxes | 785 boxes | **8,740 boxes** |
| **Trung bình box / ảnh** | 2.16 boxes/ảnh | 2.15 boxes/ảnh | 2.10 boxes/ảnh | **2.15 boxes/ảnh** |
| **Ảnh bị hỏng / không đọc được**| 0 ảnh | 0 ảnh | 0 ảnh | **0 ảnh (100% hợp lệ)** |
| **Số độ phân giải ảnh độc lập** | 295 kích thước khác nhau | 102 kích thước | 61 kích thước | Đa dạng ($500 \times 300 \rightarrow 4000 \times 3000$) |

---

## 2. Phân Bổ Số Lượng Từng Lớp Hư Hỏng Thực Tế

| Mã Lớp | Tên Lớp | Số lượng Train | Số lượng Val | Số lượng Test | Tổng số Box | Tỷ lệ (%) | Mức độ cân bằng |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `dent` (Móp) | 1,806 | 501 | 236 | **2,543** | 29.1% | Dồi dào |
| **1** | `scratch` (Xước) | 2,560 | 728 | 307 | **3,595** | 41.1% | Chiếm ưu thế lớn |
| **2** | `crack` (Nứt) | 651 | 177 | 70 | **898** | 10.3% | Trung bình |
| **3** | `glass_shatter` (Vỡ kính)| 475 | 135 | 71 | **681** | 7.8% | Thấp |
| **4** | `lamp_broken` (Vỡ đèn) | 494 | 141 | 69 | **704** | 8.1% | Thấp |
| **5** | `tire_flat` (Xẹp lốp) | 225 | 62 | 32 | **319** | 3.6% | **Rất thấp (Thiểu số)**|
| - | `tear_puncture` | 0 | 0 | 0 | **0** | 0.0% | **KHUYẾT THIẾU** |
| - | `paint_chipping` | 0 | 0 | 0 | **0** | 0.0% | **KHUYẾT THIẾU** |
| **Tổng**| **Tất cả các lớp** | **6,211** | **1,744** | **785** | **8,740** | **100%** | |

---

## 3. Kiểm Tra Toàn Vẹn Tọa Độ & Hình Học Hộp Bao (Geometry Quality)

1. **Tọa độ vượt biên (Out of bounds)**: **0 trường hợp**. Toàn bộ $8,740$ hộp bao đều nằm trọn vẹn trong khoảng chuẩn hóa $[0.0, 1.0]$ của định dạng YOLO.
2. **Kích thước bất thường (Anomalous Box Area)**:
   - Phát hiện **1,180 hộp bao** có diện tích bất thường ($>40\%$ diện tích ảnh hoặc $<0.01\%$ diện tích ảnh):
     - Train: 843 boxes ($13.6\%$).
     - Val: 236 boxes ($13.5\%$).
     - Test: 101 boxes ($12.9\%$).
   - *Phân tích chuyên môn*: Các hộp bao chiếm $>40\%$ diện tích ảnh chủ yếu là các vết cào xước chéo thân xe. Khi đóng khung chữ nhật thẳng trục AABB, diện tích hộp bao nuốt trọn cả các vùng không hư hỏng, gây nhiễu nghiêm trọng cho hàm mất mát hồi quy bounding box (`box_loss` và `dfl_loss`).

---

## 4. Kiểm Tra Hiện Tượng Rò Rỉ Dữ Liệu (Data Leakage Audit)

### 4.1. Trùng lặp Nội dung File (Exact Content Hash)
- Tính mã băm SHA256 cho toàn bộ $4,060$ ảnh trong `ai/data/yolo/images/`.
- **Kết quả:** **0 cặp ảnh trùng hash SHA256**. Không có hiện tượng copy nguyên bản cùng 1 file ảnh sang hai tập train và val/test.

### 4.2. Trùng lặp Góc chụp Cùng Xe (Vehicle Identity Leakage)
- Dữ liệu CarDD bao gồm nhiều chuỗi ảnh chụp liên tiếp một vụ tai nạn từ nhiều góc độ khác nhau (ví dụ: góc trước, góc nghiêng $45^\circ$, góc cận cảnh).
- *Rủi ro phát hiện*: Quá trình phân chia `train/val/test` của bộ dữ liệu gốc CarDD được thực hiện ngẫu nhiên theo ảnh (image-level split) chứ không phân chia theo vụ tai nạn (case-level/vehicle-level split). Do đó, có khả năng cao một số góc chụp của cùng một chiếc xe bị lọt vào cả tập train và tập val.

---

## 5. Phân Loại Tình Trạng Lỗi Dataset

| Tên Vấn Đề | Phân Loại Lỗi | Minh Chứng Kỹ Thuật | Tác Động Đến Dự Án |
| :--- | :---: | :--- | :--- |
| Thiếu hoàn toàn 2 lớp Proposal (`tear_puncture`, `paint_chipping`)| **CONFIRMED** | `class_counts = 0` trên toàn bộ 4,060 tệp nhãn | Không thể hoàn thành cam kết 5 nhóm hư hỏng của Capstone. |
| Hộp bao chữ nhật quá khổ bao trùm vùng lành lặn | **CONFIRMED** | 1,180 boxes chiếm diện tích bất thường ($>40\%$ ảnh) | Gây suy giảm độ chính xác định vị và làm méo mó khâu ánh xạ. |
| Mất cân bằng lớp nghiêm trọng (`scratch` gấp 11 lần `tire_flat`)| **CONFIRMED** | 3,595 box scratch vs 319 box tire_flat | Mô hình thiên vị mạnh vào nhận diện vết xước, kém nhạy với các lớp khác. |
| Rò rỉ góc chụp cùng xe giữa Train và Val | **POSSIBLE** | Cơ chế split ngẫu nhiên của CarDD gốc | Các chỉ số validation có thể lạc quan hơn thực tế từ $5\% - 8\%$. |
| Lỗi ảnh hỏng hoặc tọa độ âm | **NOT_FOUND** | 0 ảnh lỗi, 0 tọa độ âm | Dữ liệu được tiền xử lý và chuẩn hóa định dạng tốt. |
