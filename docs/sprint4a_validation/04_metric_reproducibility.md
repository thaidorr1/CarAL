# 04 - XÁC MINH KHẢ NĂNG TÁI LẬP CHỈ SỐ ĐỘ CHÍNH XÁC (METRIC REPRODUCIBILITY AUDIT)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Đối tượng kiểm toán:** Báo cáo các chỉ số hiệu năng trong Sprint 3 và đối chiếu nhật ký huấn luyện gốc.

---

## 1. Đối Chiếu Nguồn Dữ Liệu Của Từng Chỉ Số Báo Cáo

### 1.1. Phân hệ Damage Detection
- **Chỉ số đã báo cáo:**
  - Precision(B): **$72.88\%$**
  - Recall(B): **$67.82\%$**
  - mAP@50(B): **$70.94\%$**
  - mAP@50-95(B): **$55.96\%$**
- **Nguồn dữ liệu gốc xác minh:** Tệp `ai/runs/damage/cardd_detect_30epochs/results.csv`, dòng epoch thứ 30.
- **Tập dữ liệu dùng để tính:** Tập kiểm định CarDD (`ai/data/yolo/images/val` - 810 ảnh, 1,744 boxes).
- **Trọng số tương ứng:** Checkpoint `ai/runs/damage/cardd_detect_30epochs/weights/best.pt` (SHA256 trùng khớp 100% với `ai/models/weights/damage_best.pt`).
- **Xác minh trong metadata Checkpoint:** Trường `train_metrics` trong header của file `.pt` lưu chính xác:
  `{'metrics/precision(B)': 0.72884, 'metrics/recall(B)': 0.67824, 'metrics/mAP50(B)': 0.7094, 'metrics/mAP50-95(B)': 0.55964}`.
- **Đánh giá tái lập:** **VERIFIED (Xác nhận khớp 100% với nhật ký huấn luyện trên tập Validation)**.

### 1.2. Phân hệ Parts Segmentation
- **Chỉ số đã báo cáo:**
  - Mask Precision(M): **$37.90\%$**
  - Mask Recall(M): **$38.22\%$**
  - Mask mAP@50(M): **$28.34\%$**
  - Mask mAP@50-95(M): **$13.10\%$**
  - Box mAP@50(B): **$37.81\%$**
- **Nguồn dữ liệu gốc xác minh:** Tệp `ai/runs/parts/parts_yolov8s_seg_800/results.csv`, dòng epoch thứ 20.
- **Tập dữ liệu dùng để tính:** Tập kiểm định Parts (`ai/data/parts/images/val` - chỉ 97 ảnh, 431 polygons).
- **Trọng số tương ứng:** Checkpoint `ai/runs/parts/parts_yolov8s_seg_800/weights/best.pt` (SHA256 trùng khớp 100% với `ai/models/weights/parts_seg.pt`).
- **Xác minh trong metadata Checkpoint:** Trường `train_metrics` lưu chính xác:
  `{'metrics/precision(M)': 0.37902, 'metrics/recall(M)': 0.3822, 'metrics/mAP50(M)': 0.28338, 'metrics/mAP50-95(M)': 0.13101}`.
- **Đánh giá tái lập:** **VERIFIED (Xác nhận khớp 100% với nhật ký huấn luyện trên tập Validation 97 ảnh)**.

---

## 2. Phân Loại Mức Độ Đáng Tin Cậy Của Từng Chỉ Số (Trustworthiness Assessment)

| Chỉ Số Đánh Giá | Giá Trị Báo Cáo | Nguồn Đánh Giá | Khả Năng Tái Lập Lịch Sử | Mức Độ Tin Cậy Thực Tế | Lý Do & Hạn Chế Kỹ Thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Damage mAP@50** | $70.94\%$ | Val set (810 ảnh) | **VERIFIED** | **PARTIALLY_VERIFIED** | Đánh giá trên tập validation cùng phân phối với train, chưa phải test set độc lập hoàn toàn. |
| **Damage Precision** | $72.88\%$ | Val set (810 ảnh) | **VERIFIED** | **PARTIALLY_VERIFIED** | Bị ảnh hưởng bởi 1,180 hộp bao diện tích quá lớn làm sai lệch IoU biên. |
| **Damage Recall** | $67.82\%$ | Val set (810 ảnh) | **VERIFIED** | **PARTIALLY_VERIFIED** | Bỏ sót các vết móp nhỏ dưới ngưỡng conf 0.25. |
| **Parts Mask mAP@50** | $28.34\%$ | Val set (97 ảnh) | **VERIFIED** | **INVALID (Về mặt ngữ nghĩa Seg)** | **Nhãn Val là hộp chữ nhật 4 đỉnh giả đa giác**. Con số $28.3\%$ phản ánh sự so sánh giữa mask hình chữ nhật giả với mask dự đoán, không phản ánh chất lượng phân đoạn bề mặt xe thực tế. |
| **Parts Mask Recall** | $38.22\%$ | Val set (97 ảnh) | **VERIFIED** | **INVALID** | Kích thước tập val quá nhỏ (97 ảnh) và nhãn không đúng biên bộ phận. |

---

## 3. Các Sai Lầm Cần Khắc Phục Trong Phương Pháp Đánh Giá

1. **Lấy kết quả Validation làm kết quả Test Set**:
   - Cả hai mô hình hiện tại đều chỉ được ghi nhận chỉ số trên tập `val`.
   - Đối với Parts Segmentation, tập `test` hoàn toàn không có ảnh ($0$ ảnh). Do đó, con số $28.34\%$ mAP chưa từng được kiểm chứng trên một tập dữ liệu bên ngoài độc lập (Unseen Out-of-Distribution Data).
2. **Đánh giá Mask mAP trên nhãn Hộp chữ nhật**:
   - Việc tính toán Intersection over Union trên mặt nạ (Mask IoU) giả định rằng ground truth là biên cong chuẩn. Khi ground truth là hình chữ nhật 4 đỉnh, một dự đoán bám sát đường cong xe thật sẽ bị phạt điểm IoU nặng nề vì không phủ kín 4 góc vuông của hộp chữ nhật!
   - $\Rightarrow$ **Chỉ số Mask mAP $28.34\%$ bị bóp méo bởi chất lượng nhãn tồi của tập kiểm định.**
