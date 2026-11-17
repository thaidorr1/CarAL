# 02 - KIỂM TOÁN CHUYÊN SÂU MÔ HÌNH DAMAGE DETECTION (`damage_best.pt`)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Đánh giá năng lực phát hiện, độ chính xác phân loại, kích thước bounding box và độ tin cậy confidence của mô hình Stage 1.

---

## 1. Kiến Trúc & Cấu Hình Huấn Luyện

- **Tệp trọng số:** `ai/models/weights/damage_best.pt`
- **Kiến trúc cốt lõi:** YOLOv8n (Nano Detection) - Ultralytics.
- **Tham số mạng:** 3,133,922 tham số, 8.6 GFLOPs.
- **Tập dữ liệu gốc:** CarDD Dataset (chuyển đổi từ COCO format qua `ai/scripts/coco_to_yolo.py`).
  - Phân chia: 2,876 ảnh huấn luyện (Train), 810 ảnh kiểm định (Val), 374 ảnh thử nghiệm (Test).
- **Lịch sử huấn luyện:** Ghi nhận tại `ai/runs/damage/cardd_detect_30epochs/`:
  - Số epoch: 30 epochs.
  - Image size: 640x640 BCHW.
  - Optimizer: SGD/AdamW theo mặc định Ultralytics.

---

## 2. Chi Tiết Lớp Đối Tượng & Xung Đột Định Nghĩa (Class Definition Drift)

Qua kiểm toán các tệp cấu hình `ai/configs/classes.yaml`, `ai/configs/damage.yaml`, `ai/configs/data.yaml` và `ai/configs/classes.md`, chúng tôi phát hiện **SỰ BẤT ĐỒNG BỘ NGHIÊM TRỌNG** giữa lý thuyết Proposal và mô hình thực tế:

| Tên lớp trong Model (`damage_best.pt`) | Tên lớp trong Capstone Proposal | Có dữ liệu train trong CarDD không? | Cơ chế xử lý hiện tại trong `pipeline.py` |
| :--- | :--- | :---: | :--- |
| `0: dent` | Vết móp / lõm (Dent) | **CÓ** | Giữ nguyên nhãn `dent`. |
| `1: scratch` | Vết cào xước (Scratch) | **CÓ** | Giữ nguyên nhãn `scratch`. |
| `2: crack` | Rạn nứt kính/đèn (Crack) | **CÓ** | Giữ nguyên nhãn `crack`. |
| `3: glass_shatter` | Rạn nứt kính/đèn (Crack) | **CÓ** | Gộp (merge) về nhãn `crack`. |
| `4: lamp_broken` | Rạn nứt kính/đèn (Crack) | **CÓ** | Gộp (merge) về nhãn `crack` (nhưng giữ `raw_damage_type`). |
| `5: tire_flat` | *Không nằm trong Scope 5 lớp* | **CÓ** | Bị loại bỏ khỏi báo cáo hoặc fallback. |
| *Không có trong model* | Rách thủng cản nhựa (`tear_puncture`) | **KHÔNG** | Chưa từng được huấn luyện (Zero shot). |
| *Không có trong model* | Bong tróc lớp sơn (`paint_chipping`) | **KHÔNG** | Chưa từng được huấn luyện (Zero shot). |

> **KẾT LUẬN QUAN TRỌNG:**  
> Hệ thống hiện tại **thực chất chỉ nhận diện được 3/5 nhóm tổn thất** của Capstone Proposal (`dent`, `scratch`, `crack`). Hai nhóm tổn thất `tear_puncture` và `paint_chipping` hiện đang bị khuyết thiếu hoàn toàn trong dữ liệu huấn luyện, mô hình không có khả năng phát hiện hai loại này.

---

## 3. Phân Tích Các Vấn Đề Lỗi Kỹ Thuật (Defect Analysis)

### 3.1. Hộp bao tổn thất quá rộng (Abnormal Giant Bounding Box)
- **Hiện tượng**: Trên ảnh `demo/input/image copy 3.png`, mô hình dự đoán một vết `scratch` với bounding box:
  `[x1=10.5, y1=464.6, x2=667.0, y2=698.2]`.
  Diện tích hộp bao này chiếm tới **34.2% toàn bộ khung hình ảnh**.
- **Nguyên nhân**: Bản chất mô hình Stage 1 là Object Detection dạng hình hộp chữ nhật thẳng trục (Axis-Aligned Bounding Box - AABB). Khi vết trầy xước kéo dài chéo từ đầu xe xuống gầm xe, hộp bao AABB buộc phải bao trùm toàn bộ khoảng cách chéo, dẫn đến việc nuốt trọn cả các phần thân vỏ và bánh xe lành lặn.
- **Hệ lụy**: Hộp bao quá rộng làm sai lệch hoàn toàn phép tính diện tích đè phủ (Overlap Ratio) và tính tâm hình học (Center Distance) trong bước ánh xạ sau đó.

### 3.2. Bỏ sót tổn thất do chênh lệch độ tương phản (False Negatives)
- Trên `image copy 5.png`, mô hình dự đoán 0 tổn thất khi đặt ngưỡng `conf = 0.25`. Tuy nhiên khi hạ ngưỡng xuống `0.20`, xuất hiện các vết móp nhỏ với điểm số dao động $0.22 - 0.24$.
- Ngưỡng confidence mặc định $0.25$ là sự đánh đổi: nếu hạ thấp hơn sẽ sinh ra rất nhiều vết xước "ma" do bóng cây hoặc phản chiếu ánh sáng trên vỏ xe; nếu giữ nguyên thì bỏ sót các vết móp nhẹ (minor dents).

### 3.3. Độ tin cậy Confidence chưa được hiệu chuẩn (Uncalibrated Confidence)
- Các điểm số confidence trả về từ hàm Softmax của YOLOv8n chưa qua bước hiệu chuẩn nhiệt độ (Temperature Scaling / Platt Scaling).
- Một số vết nứt kính có confidence rất cao ($0.88 - 0.89$), trong khi các vết móp kim loại có độ sâu lớn đôi khi chỉ đạt confidence $0.46 - 0.50$. Không nên sử dụng điểm số này làm thước đo mức độ nghiêm trọng vật lý của vụ tai nạn.

---

## 4. Đề Xuất Cải Tiến Kỹ Thuật Cho Sprint Tiếp Theo

1. **Chuyển đổi Stage 1 sang Damage Segmentation (YOLOv8-seg)**:
   - Thay vì dùng Box Detection, huấn luyện mô hình phân đoạn tổn thất dạng polygon đa giác.
   - Khi đó vết trầy chéo chỉ là một đa giác mảnh ôm sát vết xước, triệt tiêu 100% hiện tượng "hộp bao khổng lồ nuốt trọn thân xe".
2. **Bổ sung tập dữ liệu cho 2 lớp còn thiếu**:
   - Thu thập bổ sung tối thiểu 300 ảnh cho `tear_puncture` (rách cản) và `paint_chipping` (tróc sơn) từ các nguồn mở (như Roboflow Car Damage, OpenImages) để hoàn thành cam kết 5 lớp của Capstone Proposal.
3. **Chuẩn hóa tệp cấu hình**:
   - Xóa bỏ sự mâu thuẫn giữa `classes.yaml`, `damage.yaml` và `data.yaml`. Thống nhất một cấu hình duy nhất phản ánh trung thực số lớp của mô hình đang nạp.
