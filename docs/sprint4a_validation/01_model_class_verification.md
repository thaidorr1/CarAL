# 01 - XÁC MINH MODEL VÀ ĐỐI CHIẾU DANH MỤC LỚP (MODEL & CLASS MAPPING VERIFICATION)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Phương pháp kiểm toán:** Trích xuất siêu dữ liệu nhị phân (PyTorch checkpoint headers, TorchScript metadata, YAML AST).  
**Trạng thái kiểm toán:** READ-ONLY. Xác minh thực nghiệm 100%.

---

## 1. Xác Minh Trực Tiếp Hai Checkpoint Chính Thức

### 1.1. Mô Hình Damage Detection (`damage_best.pt`)
- **Đường dẫn tệp thực tế:** `ai/models/weights/damage_best.pt` *(Lưu ý: Không tồn tại tệp `ai/models/damage_best.pt`, chỉ tồn tại trong thư mục con `weights/`)*.
- **Mã băm SHA256:** `6870716d82c99e6ef7fc55e39202786b44c898218133fcbb70093b77845874f0`.
- **Nguồn gốc huấn luyện xác minh:** Trùng khớp 100% (cùng hash SHA256) với `ai/runs/damage/cardd_detect_30epochs/weights/best.pt`.
- **Thời điểm huấn luyện:** `2026-10-30T19:41:01.794959+07:00`.
- **Kiến trúc mạng:** `YOLOv8n` (Object Detection) - 73 layers, 3,133,922 parameters, 8.6 GFLOPs.
- **Kích thước đầu vào:** `640x640` BCHW `(1, 3, 640, 640)`.
- **Số lượng lớp trong Checkpoint:** **6 lớp**.
  - Metadata `model.names`: `{0: 'dent', 1: 'scratch', 2: 'crack', 3: 'glass_shatter', 4: 'lamp_broken', 5: 'tire_flat'}`.
- **Định dạng ONNX tương ứng:** `ai/models/weights/damage_best.onnx` (12.19 MB, input: `['images']`, output: `['output0'] [1, 10, 8400]`).

### 1.2. Mô Hình Parts Segmentation (`parts_seg.pt`)
- **Đường dẫn tệp thực tế:** `ai/models/weights/parts_seg.pt`.
- **Mã băm SHA256:** `c7d68e20377f6baade7ed54e2cc0f37109cbd9efd59ea1bc7c5166d4c17b430e`.
- **Nguồn gốc huấn luyện xác minh:** Trùng khớp 100% (cùng hash SHA256) với `ai/runs/parts/parts_yolov8s_seg_800/weights/best.pt`.
- **Thời điểm huấn luyện:** `2026-11-03T19:11:06.008441+07:00`.
- **Kiến trúc mạng:** `YOLOv8s-seg` (Instance Segmentation) - 85 layers, 11,782,309 parameters, 39.9 GFLOPs.
- **Kích thước đầu vào:** `640x640` BCHW `(1, 3, 640, 640)`.
- **Số lượng lớp trong Checkpoint:** **7 lớp**.
  - Metadata `model.names`: `{0: 'can_truoc', 1: 'can_sau', 2: 'nap_capo', 3: 've_che_banh', 4: 'cua_hong', 5: 'kinh_chieu_hau', 6: 'kinh_lai'}`.
- **Định dạng ONNX tương ứng:** `ai/models/weights/parts_seg.onnx` (45.19 MB, input: `['images']`, output: `['output0', 'output1']`).

---

## 2. Giải Mã Mâu Thuẫn Giữa 3 Lớp, 5 Lớp Và 6 Lớp Trong Damage Detection

Trong Sprint 3, báo cáo kỹ thuật nêu "mô hình chỉ nhận diện được 3 lớp", trong khi checkpoint lại chứa 6 lớp và Proposal yêu cầu 5 nhóm. Dưới đây là phân tích đối chiếu sự thật dựa trên dữ liệu và mã nguồn:

1. **Thực tế Checkpoint (`damage_best.pt`)**: Checkpoint chứa đủ trọng số cho **6 lớp gốc của CarDD**:
   `0: dent, 1: scratch, 2: crack, 3: glass_shatter, 4: lamp_broken, 5: tire_flat`.
2. **Thực tế Proposal yêu cầu**: Proposal đặt mục tiêu **5 nhóm hư hỏng chuẩn**:
   `1. scratch, 2. dent, 3. crack, 4. tear_puncture, 5. paint_chipping`.
3. **Thực tế trong `pipeline.py` (Hàm `normalize_damage_class()`)**:
   - `dent` $\rightarrow$ giữ nguyên `dent`.
   - `scratch` $\rightarrow$ giữ nguyên `scratch`.
   - `crack`, `glass_shatter`, `lamp_broken` $\rightarrow$ cả 3 lớp này bị code gộp (merged) về chung một nhãn đại diện là `crack`.
   - `tire_flat` $\rightarrow$ bị lọc bỏ khỏi danh mục kết quả bảo hiểm (hoặc fallback).
   - Hai lớp `tear_puncture` và `paint_chipping` hoàn toàn không có trong checkpoint, dẫn đến việc trên thực tế hệ thống **chỉ có thể xuất ra 3 nhãn phân loại**: `dent`, `scratch`, `crack`.

$\Rightarrow$ **KẾT LUẬN KIỂM TOÁN:** Không có sự tương đương giữa 3, 5 và 6 lớp. Mô hình được train trên 6 lớp của CarDD, nhưng pipeline đã gộp 3 lớp nứt/vỡ lại và bỏ lớp xẹp lốp, trong khi bỏ quên 2 lớp rách/tróc sơn của Proposal.

---

## 3. Bảng Đối Chiếu Ma Trận Lớp Toàn Diện (Class Mapping Matrix)

### 3.1. Phân hệ Damage Detection

| Tên Lớp Đối Tượng | Dataset (`ai/data/yolo`) | Checkpoint (`damage_best.pt`) | Pipeline (`pipeline.py`) | Proposal (Capstone SRS) | Trạng Thái Kiểm Toán |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **dent** (Vết móp) | CÓ (2,543 box) | CÓ (id: 0) | CÓ (`dent`) | CÓ (Nhóm 2) | **MATCH** |
| **scratch** (Vết xước) | CÓ (3,595 box) | CÓ (id: 1) | CÓ (`scratch`) | CÓ (Nhóm 1) | **MATCH** |
| **crack** (Vết nứt) | CÓ (898 box) | CÓ (id: 2) | CÓ (`crack`) | CÓ (Nhóm 3) | **MATCH** |
| **glass_shatter** (Vỡ kính) | CÓ (681 box) | CÓ (id: 3) | GỘP VÀO `crack` | GỘP VÀO `crack` | **MATCH (Merged)** |
| **lamp_broken** (Vỡ đèn) | CÓ (704 box) | CÓ (id: 4) | GỘP VÀO `crack` | GỘP VÀO `crack` | **MATCH (Merged)** |
| **tire_flat** (Xẹp lốp) | CÓ (319 box) | CÓ (id: 5) | BỎ / FALLBACK | NGOÀI SCOPE | **MISMATCH** |
| **tear_puncture** (Rách thủng) | **KHÔNG** (0 box) | **KHÔNG** | Khai báo enum | CÓ (Nhóm 4) | **MISSING** |
| **paint_chipping** (Tróc sơn) | **KHÔNG** (0 box) | **KHÔNG** | Khai báo enum | CÓ (Nhóm 5) | **MISSING** |

### 3.2. Phân hệ Parts Segmentation

| Tên Bộ Phận Ngoại Thất | Dataset (`ai/data/parts`) | Checkpoint (`parts_seg.pt`) | Pipeline (`pipeline.py`) | Proposal (8 Bộ Phận Chuẩn) | Trạng Thái Kiểm Toán |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **can_truoc** (Cản trước) | CÓ (879 poly) | CÓ (id: 0) | CÓ (`Cản trước`) | CÓ (`Cản trước`) | **MATCH** |
| **can_sau** (Cản sau) | CÓ (886 poly) | CÓ (id: 1) | CÓ (`Cản sau`) | CÓ (`Cản sau`) | **MATCH** |
| **nap_capo** (Nắp capo) | CÓ (826 poly) | CÓ (id: 2) | CÓ (`Nắp capo`) | CÓ (`Nắp capo`) | **MATCH** |
| **ve_che_banh** (Vè che bánh) | CÓ (1,830 poly) | CÓ (id: 3) | CÓ (`Vè che bánh`) | CÓ (`Vè che bánh`) | **MATCH** |
| **cua_hong** (Cửa hông gộp) | CÓ (956 poly) | CÓ (id: 4) | TÁCH BẰNG CODE THỦ CÔNG | TÁCH THÀNH 4 CỬA | **MISMATCH (Coarse)** |
| **kinh_chieu_hau** (Gương chiếu hậu)| CÓ (660 poly) | CÓ (id: 5) | GỘP VÀO `Kính xe` | CÓ (`Kính xe`) | **MATCH (Merged)** |
| **kinh_lai** (Kính chắn gió) | CÓ (497 poly) | CÓ (id: 6) | GỘP VÀO `Kính xe` | CÓ (`Kính xe`) | **MATCH (Merged)** |
| **den_xe** (Đèn pha / Đèn hậu) | **KHÔNG** (0 poly)| **KHÔNG** | TẠO BẰNG RULE ẢO | CÓ (`Đèn xe`) | **MISSING** |
