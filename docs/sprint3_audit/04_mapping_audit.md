# 04 - KIỂM TOÁN GIẢI THUẬT ÁNH XẠ HƯ HỎNG - BỘ PHẬN (DAMAGE-TO-PART MAPPING)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Đánh giá tính khoa học, độ tin cậy và sự cân bằng giữa mô hình AI và các tầng quy tắc heuristics trong hàm `map_damages_to_parts()`.

---

## 1. Kiến Trúc Giải Thuật Ánh Xạ Hiện Tại

Hàm `map_damages_to_parts()` trong [ai/inference/pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py) hoạt động theo kiến trúc xếp tầng 5 giai đoạn (Cascade Fallback Pipeline):

```
[Hộp bao tổn thất Stage 1] + [Mặt nạ bộ phận Stage 2]
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │ 1. Mask Overlap & Polygon IoU       │ ──> (Overlap >= 0.35) ──> Ánh xạ thành công
  └─────────────────────────────────────┘
                     │ (Không đủ overlap)
                     ▼
  ┌─────────────────────────────────────┐
  │ 2. Spatial Distance (Tâm hình học)  │ ──> (Khoảng cách < d_max) ──> Ánh xạ gần nhất
  └─────────────────────────────────────┘
                     │ (Tâm cách quá xa)
                     ▼
  ┌─────────────────────────────────────┐
  │ 3. Semantic Hard-Rules (Luật cứng)  │ ──> (Vỡ kính/vỡ đèn/thủng lốp) ──> Ép bộ phận
  └─────────────────────────────────────┘
                     │ (Không khớp luật cứng)
                     ▼
  ┌─────────────────────────────────────┐
  │ 4. Shot-type & Close-up Guard       │ ──> (Ảnh cận cảnh/mất bối cảnh) ──> Từ chối
  └─────────────────────────────────────┘
                     │ (Không đủ bằng chứng)
                     ▼
  ┌─────────────────────────────────────┐
  │ 5. Unmapped Topology Fallback       │ ──> Trả về 'khong_xac_dinh' & 'needs_review'
  └─────────────────────────────────────┘
```

---

## 2. Đánh Giá Độ Chính Xác & Phân Loại Các Trường Hợp Thực Tế

Dựa trên kết quả chạy trên 8 ảnh mẫu tại `demo/results/predictions.json`:

### 2.1. Nhóm Ghép Đúng Hoàn Toàn (True Matches - Đạt 100%)
- **`image copy 4.png`**: Vết móp lớn trên cửa trước bên phụ.
  - Phương thức: `mask_overlap` ($\text{Overlap Ratio} = 0.87$).
  - Kết quả: `dent -> Cửa trước [0.87]`.
  - Phân tích: Mặt nạ cửa bao trọn vết móp, thuật toán gán chính xác tuyệt đối mà không cần can thiệp heuristics.
- **`image copy 7.png`**: Cụm 3 vết xước trên cánh cửa trước.
  - 2 vết xước khớp qua `mask_overlap` ($0.93$ và $0.75$).
  - 1 vết xước nhỏ ở mép khớp qua `spatial_distance` ($0.87$).
  - Kết quả: Toàn bộ 3 vết đều được gom đúng vào `Cửa trước`.

### 2.2. Nhóm Được Cứu Bởi Quy Tắc Ngữ Nghĩa (Rule-Rescued Matches)
- **`image copy.png` & `image copy 3.png`**: Vết vỡ đèn pha.
  - Do mô hình Stage 2 không có nhãn `den_xe`, mặt nạ phân đoạn tại vùng đèn hoàn toàn bằng 0.
  - Cơ chế `semantic_hard_rule_lamp` phát hiện `raw_damage_type == 'lamp_broken'` đã tự động tạo một hộp bao đại diện và gán vào `Đèn xe` với độ tin cậy $0.88 - 0.89$.
  - *Đánh giá*: Hành vi này giúp hệ thống cứu được một ca lỗi do model seg gây ra, tuy nhiên làm giảm tính thuần khiết của mô hình AI học sâu.

### 2.3. Nhóm Kích Hoạt Chốt Chặn An Toàn (Safety Guards - Trả về `khong_xac_dinh`)
- **`image copy 6.png`**, **`image copy 8.png`**, **`image.png`**:
  - Cả 3 ảnh này đều là ảnh chụp macro/cận cảnh một mảng sơn hoặc mép gầm xe.
  - Hàm `classify_shot_type()` phát hiện tỷ lệ chiếm dụng khung hình của thân xe vượt ngưỡng, xác định `shot_type = 'close_up'`.
  - Thay vì cố ép vết xước vào nắp capo hay cản trước, hệ thống gán nhãn:
    `located_on_part: "khong_xac_dinh"`
    kèm theo trạng thái tổng thể: `status: "insufficient_data"`, yêu cầu người dùng chụp lại ảnh ở cự ly toàn cảnh hoặc trung cảnh ($1.5\text{m} - 2.5\text{m}$).
  - *Đánh giá*: **RẤT TỐT (High Reliability)**. Đây là hành vi mẫu mực của một hệ thống AI phục vụ ngành bảo hiểm: thà báo thiếu thông tin để con người kiểm tra lại, còn hơn tự động gán bừa dẫn đến bồi thường sai.

---

## 3. Các Điểm Hạn Chế Cần Tinh Gọn Trong Khâu Mapping

1. **Sự phụ thuộc vào hộp bao AABB thô**:
   - Khi tính toán IoU đè phủ, nếu vết hư hỏng không có đa giác phân đoạn mà chỉ có hộp chữ nhật `[x1, y1, x2, y2]`, góc thừa của hình chữ nhật có thể chọc sang mặt nạ của bộ phận lân cận (ví dụ góc hộp xước cửa lấn sang vè bánh).
2. **Quy tắc cứng (Hard-coded heuristics) quá dày đặc**:
   - Khoảng 1,200 dòng mã trong `pipeline.py` chỉ phục vụ việc kiểm tra tọa độ pixel tương đối giữa các bộ phận (ví dụ: `pcx < door_cx - 0.06 * w`). Các con số hằng số này ($0.06, 0.40, 0.15$) được tinh chỉnh thủ công (manual fine-tuning) trên một số ít ảnh, không đảm bảo tính tổng quát khi gặp các dòng xe có thiết kế đặc thù (xe bán tải, xe Hatchback đuôi cụt, xe SUV gầm cao).
