# 07 - PHƯƠNG PHÁP ĐÁNH GIÁ ĐỘC LẬP CHO DAMAGE-TO-PART MAPPING
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Mục tiêu:** Tách bạch và chuẩn hóa phương pháp đo lường độ chính xác của giải thuật ánh xạ tổn thất vào bộ phận xe.

---

## 1. Phân Tích Logic Hàm `map_damages_to_parts()` Hiện Tại

Hàm `map_damages_to_parts()` (từ dòng 1,905 đến 2,363 trong [pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py)) gồm 458 dòng mã với các tầng điều kiện cụ thể:

### 1.1. Các Ngưỡng & Điều Kiện Hình Học Phụ Thuộc Tọa Độ
1. **Ngưỡng diện tích đè phủ (`overlap_ratio`)**:
   - Giá trị cứng: $0.35$ ($35\%$ diện tích vết tổn thất nằm trong mặt nạ bộ phận).
   - Nếu $\ge 0.35 \rightarrow$ Nhận diện là `match_method = 'mask_overlap'`.
2. **Ngưỡng khoảng cách tâm hình học (`center_distance`)**:
   - Nếu không đủ overlap, tính khoảng cách Euclidean giữa tâm hộp tổn thất và tâm mặt nạ bộ phận gần nhất.
   - Khoảng cách tối đa cho phép: $120$ pixels (hoặc $0.18 \times \text{Chiều rộng ảnh}$).
3. **Thuật toán chia cột B thủ công (B-Pillar Split)**:
   - Dựa vào tọa độ hoành độ X trung bình của cụm cửa để nội suy vị trí cột ngăn giữa cửa trước và cửa sau:
     `b_pillar_x = (left_door_cx + right_door_cx) / 2.0`.
4. **Bộ lọc màu đèn hậu HSV**:
   - Quét không gian màu HSV lọc dải màu đỏ: `(H in [0..10] or [160..180]) and S > 70 and V > 50`.
   - Nếu tỷ lệ diện tích đỏ $> 8\%$, kết luận là đuôi xe $\rightarrow$ ép cản sau.

---

## 2. Công Thức Đo Lường Ba Chỉ Số Độc Lập

Tuyệt đối **KHÔNG ĐƯỢC GỘP** ba chỉ số thành một con số "Accuracy chung chung" không có cơ sở khoa học. Bắt buộc phải tách thành 3 chỉ số riêng biệt:

### 2.1. Chỉ số 1: Damage Detection Accuracy
- Đo lường bằng chuẩn COCO:
  $$\text{mAP@50 (Box)}, \quad \text{Precision (Box)}, \quad \text{Recall (Box)}$$
- Đánh giá: Khả năng tìm đúng tọa độ và đúng loại tổn thất của Stage 1.

### 2.2. Chỉ số 2: Part Segmentation Accuracy
- Đo lường bằng chuẩn COCO Segmentation:
  $$\text{Mask mAP@50}, \quad \text{Mask mIoU}, \quad \text{Mask Recall}$$
- Đánh giá: Khả năng bám sát đường cong và đúng tên bộ phận của Stage 2.

### 2.3. Chỉ số 3: Damage-to-Part Mapping Accuracy (Đo riêng cho khâu Ghép Cặp)
- Chỉ đánh giá trên các vết hư hỏng mà Stage 1 đã phát hiện đúng (True Positive Bounding Box):
  $$\text{Mapping Accuracy} = \frac{\text{Số vết hư hỏng ghép ĐÚNG bộ phận thực tế}}{\text{Tổng số vết hư hỏng được phát hiện đúng (TP)}}$$
- **Chỉ số An toàn (Safety Rejection Precision)**:
  $$\text{Safety Precision} = \frac{\text{Số ca từ chối ĐÚNG (Ảnh cận cảnh/không rõ)}}{\text{Tổng số ca trả về 'khong_xac_dinh' hoặc 'insufficient_data'}}$$

---

## 3. Kế Hoạch Đánh Giá Thực Nghiệm Trong Sprint 4B

1. **Chuẩn bị 50 cặp ảnh Before/After và 100 ảnh đơn lẻ có Ground Truth**.
2. Chạy pipeline qua toàn bộ tập dữ liệu, xuất file `evaluation_results.json`.
3. So sánh trực tiếp từng trường `located_on_part` của AI với `true_part_name` của Ground Truth.
4. Lập **Ma Trận Nhầm Lẫn Ánh Xạ (Mapping Confusion Matrix)**:
   - Hàng: Bộ phận thực tế.
   - Cột: Bộ phận do AI ánh xạ.
   - Ô chéo chính là tỷ lệ ghép đúng; các ô lệch chéo chỉ ra chính xác cặp bộ phận nào đang bị gán nhầm nhiều nhất (ví dụ: Cửa trước bị gán nhầm sang Cửa sau bao nhiêu %).
