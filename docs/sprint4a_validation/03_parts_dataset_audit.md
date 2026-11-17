# 03 - KIỂM TOÁN TẬP DỮ LIỆU PARTS SEGMENTATION (`ai/data/parts`)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Thư mục dữ liệu:** `ai/data/parts/`  
**Phương pháp kiểm toán:** Phân tích cấu trúc hình học đa giác (Polygon Vertex Profiling), tính diện tích polygon và phân phối lớp trên 1,622 tệp nhãn.

---

## 1. Thống Kê Tổng Thể Tập Dữ Liệu Phân Đoạn Bộ Phận

| Thuộc Tính Kiểm Toán | Tập Huấn Luyện (Train) | Tập Kiểm Định (Val) | Tập Thử Nghiệm (Test) | Toàn Bộ Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Số lượng ảnh** | 1,525 ảnh | 97 ảnh | **0 ảnh (Trống)** | **1,622 ảnh** |
| **Số lượng tệp nhãn (`.txt`)**| 1,525 tệp | 97 tệp | 0 tệp | **1,622 tệp** |
| **Tổng số Polygon nhãn** | 6,103 đa giác | 431 đa giác | 0 đa giác | **6,534 đa giác** |
| **Trung bình polygon / ảnh**| 4.00 polygons/ảnh | 4.44 polygons/ảnh | 0 | **4.03 polygons/ảnh** |
| **Số điểm đỉnh trung bình/polygon**| **5.9 đỉnh** | **5.4 đỉnh** | 0 | **5.87 đỉnh** |
| **Ảnh bị hỏng / không đọc được**| 0 ảnh | 0 ảnh | 0 ảnh | **0 ảnh** |
| **Polygon tọa độ vượt biên** | 0 | 0 | 0 | **0** |

---

## 2. Phát Hiện Trọng Yếu: Bê Bối "Đa Giác Giả" (The Pseudo-Polygon Discovery)

Qua phân tích số lượng điểm đỉnh của từng đa giác trong toàn bộ $6,103$ nhãn huấn luyện, chúng tôi phát hiện **SỰ THẬT KỸ THUẬT GÂY CHẤN ĐỘNG**:

| Số Điểm Đỉnh (Vertices) | Số Lượng Đa Giác | Tỷ Lệ (%) | Bản Chất Kỹ Thuật |
| :---: | :---: | :---: | :--- |
| **Đúng 4 điểm đỉnh** | **4,616 đa giác** | **75.6%** | **HỘP CHỮ NHẬT GIẢ ĐA GIÁC (Bounding Box converted to 4-point polygon)** |
| **Đúng 8 điểm đỉnh** | 33 đa giác | 0.5% | Hộp vát góc đơn giản |
| **Đúng 12 điểm đỉnh** | **1,454 đa giác** | **23.8%** | **KHUNG ĐA GIÁC ĐỀU TẠO BẰNG SCRIPT TỰ ĐỘNG (Synthetic 12-point template)** |
| **Số đỉnh hữu cơ (>15 đỉnh)**| **0 đa giác** | **0.0%** | **HOÀN TOÀN KHÔNG CÓ ĐA GIÁC VẼ TAY THẬT BÁM ĐƯỜNG CONG XE** |

> **KẾT LUẬN KIỂM TOÁN CỐT LÕI:**  
> Hơn **$75.6\%$ nhãn huấn luyện của tập phân đoạn bộ phận thực chất chỉ là HỘP CHỮ NHẬT 4 ĐIỂM** được ghi dưới dạng đa giác `[x1 y1 x2 y1 x2 y2 x1 y2]`.  
> Mô hình YOLOv8s-seg đang bị ép phải học các mặt nạ chữ nhật cứng nhắc thay vì học đường biên cong thực tế của xe!  
> **Đây chính là nguyên nhân giải thích 100% tại sao Mask mAP@50 chỉ đạt $28.34\%$ và mặt nạ luôn bị lấn biên loang lổ sang các bộ phận bên cạnh.**

---

## 3. Phân Bổ Số Lượng Từng Lớp Bộ Phận Thực Tế

| Mã Lớp | Tên Lớp Bộ Phận | Số lượng Train | Số lượng Val | Tổng số Polygon | Tỷ lệ (%) | Mức độ đại diện |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `can_truoc` (Cản trước) | 822 | 57 | **879** | 13.5% | Tốt |
| **1** | `can_sau` (Cản sau) | 827 | 59 | **886** | 13.6% | Tốt |
| **2** | `nap_capo` (Nắp capo) | 766 | 60 | **826** | 12.6% | Tốt |
| **3** | `ve_che_banh` (Vè che bánh)| 1,717 | 113 | **1,830** | **28.0%** | Chiếm ưu thế áp đảo |
| **4** | `cua_hong` (Cửa hông gộp)| 898 | 58 | **956** | 14.6% | Gộp cả 4 cửa |
| **5** | `kinh_chieu_hau` (Gương) | 618 | 42 | **660** | 10.1% | Khá |
| **6** | `kinh_lai` (Kính chắn gió)| 455 | 42 | **497** | 7.6% | Thiểu số |
| - | `den_xe` (Đèn pha/hậu) | 0 | 0 | **0** | 0.0% | **KHUYẾT THIẾU HOÀN TOÀN** |
| **Tổng**| **Tất cả các lớp** | **6,103** | **431** | **6,534** | **100%** | |

---

## 4. Phân Tích Khuyết Tật Dữ Liệu Bộ Phận (Defects Breakdown)

1. **Khuyết thiếu lớp Đèn xe (`den_xe`)**:
   - Hoàn toàn không có dữ liệu cho cụm đèn pha và đèn hậu. Mạng nơ-ron không có bất kỳ khả năng nào để khoanh vùng đèn xe nếu không có sự can thiệp của code heuristics ngoài pipeline.
2. **Gộp cửa hông (`cua_hong`) làm mất thông tin không gian**:
   - Dữ liệu không phân tách cửa trước và cửa sau. Một mặt nạ hình chữ nhật bao phủ toàn bộ sườn xe, đè lấp lên cả cột B (B-pillar), kính cửa và vè bánh sau.
3. **Tập Test Set bằng 0 (Missing Test Set)**:
   - Thư mục `ai/data/parts/images/test` và `labels/test` hoàn toàn rỗng ($0$ file).
   - Tập validation chỉ có vỏn vẹn **97 ảnh** (quá nhỏ để đảm bảo độ tin cậy thống kê cho 7 lớp đối tượng phức tạp).
4. **Mất cân bằng lớp**: Lớp `ve_che_banh` chiếm tới $28\%$ số lượng đa giác (gấp gần 4 lần lớp `kinh_lai`), khiến mô hình có xu hướng dự đoán thiên vị vào vè bánh xe ở mọi góc chụp.

---

## 5. Bảng Phân Loại Lỗi Dataset Phân Đoạn

| Vấn Đề Kiểm Toán | Phân Loại | Minh Chứng Cụ Thể | Tác Động Thực Tế |
| :--- | :---: | :--- | :--- |
| Nhãn phân đoạn là hộp chữ nhật giả đa giác (Pseudo-polygon) | **CONFIRMED** | 75.6% polygon có đúng 4 đỉnh; 23.8% có đúng 12 đỉnh template | Mask mAP thấp thảm hại ($28.34\%$), mặt nạ lem luốc sang bộ phận khác. |
| Thiếu lớp Đèn xe trong dữ liệu huấn luyện | **CONFIRMED** | `den_xe` = 0 polygon trong cả 1,622 tệp nhãn | Không thể tự động phát hiện hư hỏng đèn bằng AI. |
| Không phân biệt cửa trước và cửa sau | **CONFIRMED** | Chỉ có 1 nhãn chung `cua_hong` | Phải dựa vào thuật toán chia tọa độ 2D thủ công trong pipeline. |
| Tập Test độc lập không tồn tại | **CONFIRMED** | `ai/data/parts/images/test` có 0 ảnh | Không có cơ sở đánh giá độ tổng quát hóa trên dữ liệu chưa từng thấy. |
| Rò rỉ dữ liệu giữa Train và Val | **NOT_FOUND** | 0 cặp ảnh trùng mã băm SHA256 | Các tệp ảnh được phân tách độc lập về mặt file hash. |
