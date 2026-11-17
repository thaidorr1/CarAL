# 03 - KIỂM TOÁN CHUYÊN SÂU MÔ HÌNH PARTS SEGMENTATION (`parts_seg.pt`)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Phân tích cấu trúc phân đoạn đa giác, độ chính xác mặt nạ, hiện tượng lấn biên và nguyên nhân gây lỗi nhận diện bộ phận ngoại thất xe.

---

## 1. Thông Số Huấn Luyện & Chỉ Số Kỹ Thuật

- **Tệp trọng số:** `ai/models/weights/parts_seg.pt`
- **Kiến trúc:** YOLOv8s-seg (Small Instance Segmentation) - Ultralytics.
- **Tham số mạng:** 11,782,309 tham số, 39.9 GFLOPs.
- **Tập dữ liệu:** 1,525 ảnh train, 97 ảnh validation (ghi nhận tại `ai/configs/data_parts.yaml`).
- **Chỉ số kiểm định thực tế (Epoch 20 trong `ai/runs/parts/parts_yolov8s_seg_800/results.csv`):**
  - **Mask Precision (M):** $37.90\%$
  - **Mask Recall (M):** $38.22\%$
  - **Mask mAP@50 (M):** **$28.34\%$**
  - **Mask mAP@50-95 (M):** **$13.10\%$**

---

## 2. Phát Hiện 3 Điểm Khiếm Khuyết Kiến Trúc Nghiêm Trọng

### 2.1. Thiếu hẳn lớp "Đèn xe" trong Mô hình AI (The Missing Headlamp Class)
- **Sự thật kiểm toán**: Danh mục 7 lớp của mô hình `parts_seg.pt` gồm:
  `{0: can_truoc, 1: can_sau, 2: nap_capo, 3: ve_che_banh, 4: cua_hong, 5: kinh_chieu_hau, 6: kinh_lai}`.
  $\Rightarrow$ **Mô hình AI KHÔNG CÓ LỚP "Đèn xe" (`den_xe` / `headlamp`)**.
- **Hệ quả trong code**: Khi xe bị vỡ đèn pha hoặc đèn hậu, mô hình phân đoạn hoàn toàn không nhận diện được cụm đèn. Để khắc phục, lập trình viên trong Sprint trước đã phải tạo ra hàm cứu hộ:
  `semantic_hard_rule_lamp_broken_strictly_belongs_to_headlight()`
  nhằm tự động gán nhãn ảo `Đèn xe` khi mô hình Stage 1 bắt được vết nứt đèn pha. Đây là giải pháp chắp vá bằng quy tắc cứng (hard-coded rule) chứ không phải do mô hình AI tự nhận thức được.

### 2.2. Gộp 4 Cửa Hông thành một lớp `cua_hong` (Aggregated Door Mask)
- **Vấn đề**: Mô hình được huấn luyện với nhãn gộp `cua_hong` đại diện cho cả 4 cửa hông (cửa trước trái, cửa sau trái, cửa trước phải, cửa sau phải).
- **Hệ quả**:
  - Khi xe nhìn từ góc nghiêng thân, mô hình sinh ra một dải mặt nạ (mask) dài ngoằng bao phủ cả hai cánh cửa.
  - Tệp [pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py) phải tốn hơn **350 dòng mã giải thuật** (`resolve_four_side_doors`, `resolve_door_position_from_topology`) để tìm tọa độ cột B (B-pillar), sau đó "chém đôi" mặt nạ dựa trên tỷ lệ tọa độ X hình học.
  - Khi xe đỗ ở góc xiên (phối cảnh 3D), phép chia đôi tọa độ 2D này thường xuyên gán nhầm: vết xước ở mép sau cánh cửa trước bị gán thành cửa sau.

### 2.3. Lấn Biên Mặt Nạ Sang Vè Sau (Mask Bleed onto Quarter Panel)
- **Hiện tượng**: Mặt nạ của cánh cửa sau thường bị loang (bleed) đè lên phần vè bánh sau (Rear Quarter Panel) và cản sau.
- **Minh chứng**: Trong unit test `test_quarter_panel_priority_over_door`, hệ thống phải cài đặt một quy tắc heuristics riêng biệt để ép hạ ưu tiên của mặt nạ cửa nếu tâm vết móp nằm lùi về phía bánh sau.
- **Nguyên nhân**: Dữ liệu huấn luyện 1,525 ảnh có chất lượng nhãn phân đoạn đa giác chưa sắc nét (noisy ground truth polygons), dẫn đến việc mạng nơ-ron học nhầm đường chỉ gân xe thành ranh giới cửa.

---

## 3. Đánh Giá Khả Năng Định Hướng Đầu - Đuôi Xe (Car Orientation Confusion)

- Khi chụp từ góc hông phẳng (side view 90 độ), mô hình thường xuyên nhầm lẫn giữa **Cản trước** (`can_truoc`) và **Cản sau** (`can_sau`).
- Hệ thống hiện tại phải dùng giải thuật `_calculate_hsv_red_ratio()`: quét dải màu Đỏ trong không gian màu HSV tại hai đầu xe. Nếu đầu nào có tỷ lệ màu đỏ cao (đèn hậu), hệ thống sẽ cưỡng chế ép đầu đó là Cản sau.
- *Hạn chế*: Nếu chiếc xe có màu sơn đỏ nguyên bản (Red Car Body), giải thuật HSV này sẽ bị tê liệt và nhận diện sai cả hai đầu xe (như đã phản ánh trong test case `test_front_broken_headlamp_red_car_forces_front_bumper_and_disables_can_sau`).

---

## 4. Đề Xuất Kỹ Thuật Trọng Tâm Cho Sprint 4

1. **Chuẩn hóa lại Bộ Nhãn Phân Đoạn (8 Lớp Ngoại Thất Độc Lập)**:
   - Tách dứt điểm `cua_truoc` và `cua_sau` ngay từ khâu gán nhãn dataset.
   - Bổ sung lớp `den_xe` (Đèn pha / Đèn hậu) vào mô hình phân đoạn để AI tự nhận diện trực quan thay vì dùng rule chắp vá.
2. **Nâng cấp Backbone & Tăng số Epoch Huấn Luyện**:
   - Tăng số epoch từ 20 lên 100 epochs với kỹ thuật Augmentation phù hợp (Mosaic, Mixup, Random Perspective).
   - Mục tiêu bắt buộc: Đưa **Mask mAP@50 từ 28.34% lên $\ge 65.0\%$**, giải phóng `pipeline.py` khỏi hàng ngàn dòng code heuristics.
