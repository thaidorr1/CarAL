# BÁO CÁO TỔNG KẾT KIỂM TOÁN DỮ LIỆU & ĐÁNH GIÁ BASELINE AI (FINAL SPRINT 4A REPORT)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Ngày hoàn tất:** 2026-11-12  
**Phiên bản:** Sprint 4A Baseline Report  
**Vai trò thực hiện:** Senior Computer Vision Engineer / AI Dataset Auditor / ML Evaluation Engineer  
**Trạng thái tuân thủ:** **READ-ONLY 100%**. Không có bất kỳ file mã nguồn, checkpoint model hoặc tệp dữ liệu nào bị thay đổi.

---

## 1. Trả Lời 10 Câu Hỏi Trọng Tâm Của Đợt Kiểm Toán

### 1. Model hiện tại thực sự nhận diện được những class nào?
- **Damage Detection (`damage_best.pt`)**: Thực tế nhận diện được **6 lớp gốc**: `dent`, `scratch`, `crack`, `glass_shatter`, `lamp_broken`, `tire_flat`. Khi qua `pipeline.py`, code gộp `glass_shatter` và `lamp_broken` vào `crack`, và bỏ `tire_flat`. Vì vậy, hệ thống **chỉ có thể xuất ra 3 nhóm nhãn**: `dent`, `scratch`, `crack`.
- **Parts Segmentation (`parts_seg.pt`)**: Thực tế nhận diện được **7 lớp**: `can_truoc`, `can_sau`, `nap_capo`, `ve_che_banh`, `cua_hong`, `kinh_chieu_hau`, `kinh_lai`. **KHÔNG CÓ LỚP ĐÈN XE** và gộp chung cả 4 cửa vào 1 nhãn `cua_hong`.

### 2. Chỉ số accuracy nào đã được xác minh?
- Các chỉ số ghi nhận trong `results.csv` và header metadata của checkpoint đã được **XÁC MINH (VERIFIED khớp 100%)** trên tập Validation của từng mô hình:
  - Damage Detection: Precision **$72.88\%$**, Recall **$67.82\%$**, mAP@50 **$70.94\%$** (trên tập val 810 ảnh CarDD).
  - Parts Segmentation: Mask Precision **$37.90\%$**, Mask Recall **$38.22\%$**, Mask mAP@50 **$28.34\%$** (trên tập val 97 ảnh Parts).
- **Lưu ý chuyên môn**: Các chỉ số này **CHƯA ĐƯỢC KIỂM CHỨNG TRÊN TẬP TEST ĐỘC LẬP** (tập test của Parts hiện có 0 ảnh).

### 3. Dataset có đủ chất lượng để huấn luyện tiếp không?
- **Damage Dataset (`ai/data/yolo`)**: **ĐỦ CHẤT LƯỢNG CHO 3 LỚP CƠ BẢN** (dent, scratch, crack với 4,060 ảnh hợp lệ, 0 ảnh lỗi), nhưng **chưa đủ để đáp ứng 5 lớp của Capstone Proposal** vì thiếu hẳn 2 lớp rách và tróc sơn.
- **Parts Dataset (`ai/data/parts`)**: **KHÔNG ĐỦ CHẤT LƯỢNG ĐỂ HUẤN LUYỆN TIẾP**. $75.6\%$ nhãn là hình chữ nhật 4 đỉnh giả đa giác. Nếu tiếp tục train trên dữ liệu này, model sẽ chỉ học ra các khối hộp vuông vức, không thể cải thiện độ chính xác phân đoạn.

### 4. Parts Segmentation đang yếu vì dữ liệu, nhãn, model hay chưa đủ bằng chứng?
- **Sự thật đã xác minh**: Parts Segmentation yếu **CHỦ YẾU DO CHẤT LƯỢNG NHÃN (Label Quality)**. Việc biến hộp bao bounding box thành đa giác 4 đỉnh giả đã "đầu độc" hàm mất mát phân đoạn mặt nạ (`seg_loss`), khiến mô hình YOLOv8s-seg không thể học được đường biên cong vật lý của xe ô tô.

### 5. Damage Detection có cần thêm dữ liệu không?
- **CÓ**. Cần bổ sung dữ liệu cho 2 lớp còn thiếu (`tear_puncture` và `paint_chipping`) và bổ sung thêm các mẫu vết xước nhỏ có bounding box ôm sát để giảm thiểu 1,180 hộp bao khổng lồ đang chiếm diện tích bất thường.

### 6. Những class nào cần bổ sung?
1. Lớp tổn thất: `tear_puncture` (Rách thủng) và `paint_chipping` (Tróc sơn).
2. Lớp bộ phận: `den_xe` (Cụm đèn pha và đèn hậu).
3. Tách lớp bộ phận: Tách `cua_hong` thành `cua_truoc` và `cua_sau`.

### 7. Cần bao nhiêu ảnh kiểm thử có ground truth để đánh giá có ý nghĩa?
- Để đánh giá có ý nghĩa thống kê (Statistical Significance với khoảng tin cậy $95\%$):
  - Cần tối thiểu **150 đến 200 ảnh xe thực tế có nhãn Ground Truth chuẩn** (bao gồm đa giác thật $>15$ đỉnh/polygon và hộp bao tổn thất ôm sát).
  - Bộ 8 ảnh demo hiện tại chỉ có ý nghĩa như một bài Smoke Test ban đầu.

### 8. Mapping hiện tại có đáng tin cậy không?
- **Đáng tin cậy về mặt An toàn (Safety Reliability: Cao)**: Khi gặp ảnh cận cảnh hoặc không đủ bằng chứng hình học, hệ thống chủ động trả về `khong_xac_dinh` / `insufficient_data` thay vì gán bừa.
- **Kém tin cậy về mặt Mô hình AI thuần túy**: Do model Parts Seg yếu, hàm ánh xạ đang phải phụ thuộc vào hàng ngàn dòng code heuristics chắp vá (HSV đỏ cho đèn hậu, chia đôi cột B bằng tọa độ X 2D).

### 9. Những lỗi nào cần ưu tiên sửa ở Sprint 4B?
1. Thay thế nhãn 4 đỉnh giả bằng tập đa giác thật bám sát thân vỏ xe.
2. Xây dựng tệp `demo/annotations_ground_truth.json` cho 8 ảnh mẫu.
3. Đồng bộ hóa các tệp cấu hình YAML phân mảnh.
4. Bổ sung nhãn `den_xe` và tách 2 cửa trước/sau.

### 10. Có đủ cơ sở để bắt đầu cải thiện AI chưa?
- **ĐÃ ĐỦ CƠ SỞ VỀ MẶT ĐỊNH BỆNH (Diagnosis Complete)**.
- Nhưng **CHƯA ĐƯỢC PHÉP BẮT ĐẦU HUẤN LUYỆN NGAY** cho đến khi:
  - Tập dữ liệu đa giác thật (True Polygon Dataset) được chuẩn bị xong.
  - Bộ ground truth kiểm định được thiết lập để có thước đo so sánh trước/sau.

---

## 2. Bảng Xếp Hạng 5 Vấn Đề Nghiêm Trọng Nhất Phát Hiện

| STT | Tên Vấn Đề | Vị Trí / Bằng Chứng | Mức Độ | Độ Tin Cậy | Tác Động Thực Tế | Đề Xuất Xử Lý |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **1** | **75.6% nhãn phân đoạn là hộp chữ nhật giả đa giác** | `ai/data/parts/labels/train/*.txt` (4,616/6,103 polygon có đúng 4 đỉnh) | **CRITICAL** | **CONFIRMED** | Nguyên nhân gốc rễ khiến Mask mAP chỉ đạt 28.34% và mask loang lổ | Thay thế bằng bộ nhãn phân đoạn đa giác thật bám viền cong |
| **2** | **Mô hình Parts hoàn toàn không có lớp 'Đèn xe'** | Checkpoint metadata `parts_seg.pt` & `ai/data/parts/` | **HIGH** | **CONFIRMED** | Bỏ sót hoàn toàn cụm đèn, phải dùng rule ảo để cứu | Bổ sung lớp `den_xe` vào dataset phân đoạn |
| **3** | **Khuyết thiếu 2/5 lớp hư hỏng cam kết trong Proposal** | `ai/data/yolo/labels/` (0 box `tear_puncture` & `paint_chipping`) | **HIGH** | **CONFIRMED** | Không đạt tiêu chuẩn cam kết nghiệm thu đề tài | Thu thập bổ sung 300-500 ảnh cho 2 lớp này |
| **4** | **Gộp 4 cánh cửa thành một nhãn `cua_hong`** | `ai/configs/parts.yaml` & dữ liệu parts | **HIGH** | **CONFIRMED** | Gây nhầm lẫn giữa cửa trước và cửa sau ở các góc chụp xiên 3D | Tách thành `cua_truoc` và `cua_sau` trong dataset |
| **5** | **Tập Test của Parts Seg hoàn toàn rỗng (0 ảnh)** | `ai/data/parts/images/test/` (0 files) | **HIGH** | **CONFIRMED** | Không có cơ sở đánh giá độ tổng quát hóa trên dữ liệu độc lập | Thiết lập tập test tối thiểu 150 ảnh độc lập |

---

## 3. Phân Biệt Giữa Sự Thật, Nhận Định Kỹ Thuật Và Giả Thuyết

1. **Sự thật đã xác minh (Verified Facts)**:
   - Checkpoint `damage_best.pt` có 6 lớp; `parts_seg.pt` có 7 lớp.
   - Nhãn Parts Seg có $75.6\%$ là 4 đỉnh chữ nhật.
   - Tập `ai/data/parts/images/test` có $0$ ảnh.
   - Thư mục `demo/input/` có đúng 8 ảnh, định dạng thực tế là JPEG lưu dưới đuôi `.png`.
   - Các chỉ số accuracy trong Sprint 3 trùng khớp $100\%$ với header metadata và nhật ký `results.csv` trên tập validation.
2. **Nhận định kỹ thuật (Engineering Assessments)**:
   - Hàm `map_damages_to_parts()` hoạt động an toàn khi từ chối ảnh cận cảnh, nhưng quá phụ thuộc vào quy tắc hình học 2D thủ công.
   - Không nên kết luận mô hình damage có accuracy $70.9\%$ trên thực tế vì chỉ số này được tính trên tập val CarDD cùng phân phối và có $13.5\%$ hộp bao quá khổ.
3. **Giả thuyết cần kiểm tra tiếp trong Sprint 4B (Hypotheses to Test)**:
   - Nếu thay thế nhãn 4 đỉnh bằng đa giác thật $\ge 16$ đỉnh và huấn luyện 100 epochs, Mask mAP@50 của Parts Segmentation có thể tăng từ $28.3\%$ lên $\ge 60\%$.
   - Giả thuyết có sự rò rỉ góc chụp cùng xe (vehicle identity leakage) giữa train và val trong tập CarDD gốc cần được kiểm tra bằng trích xuất biển số hoặc màu xe.
