# BÁO CÁO TỔNG KẾT SPRINT 2 - AUTOCLAIM-AI CORE MODULE
**Thời điểm thực hiện**: 2026-11-12  
**Dự án**: `AI_CAR` (Hệ thống Giám định & Đánh giá Tổn thất Xe Cơ giới Tự động)  
**Phạm vi**: Đối soát Before/After (`change_detection.py`), Xuất mô hình ONNX & Benchmark Hiệu năng, Kiểm thử Đơn vị & Cập nhật Knowledge Graph.

---

## 1. Kết Quả Kiểm Thử Mô-đun Đối Soát Before/After (`change_detection.py`)

### 1.1. Kiến trúc & Tính An Toàn Thuật Toán
- **Căn chỉnh góc chụp & Homography RANSAC**:
  - Tích hợp phương pháp trích xuất đặc trưng `SIFT` (mặc định) và cơ chế tự động chuyển đổi sang `ORB` nếu môi trường không hỗ trợ hoặc SIFT bị lỗi.
  - Xử lý an toàn các trường hợp biên: bắt ngoại lệ `cv2.error` trong `cv2.findHomography` và từ chối ma trận khi số inliers $< 10$, tự động chuyển sang chế độ fallback so sánh tọa độ trực tiếp mà không làm sập tiến trình.
- **Tính toán ma trận IoU và phân loại tổn thất**:
  - Tích hợp thư viện `supervision` (`sv.box_iou_batch`) để tính toán ma trận IoU hàng loạt giữa các hộp bao đã nắn chỉnh của ảnh Check-out (After) và ảnh Check-in (Before).
  - Ngưỡng phân loại chuẩn xác theo thiết kế:
    - $\text{IoU} \ge 0.60$: Phân loại là **"Tổn thất cũ"** (`old_damage`) - đã ghi nhận trước khi bàn giao xe.
    - $\text{IoU} < 0.60$ hoặc không tìm thấy vết khớp: Phân loại là **"Tổn thất mới phát sinh"** (`new_damage`) kèm theo bộ phận định vị tương ứng (ví dụ: `Nắp capo`, `Cửa trước`, `Cản trước`...).
  - **Giữ nguyên 100% chữ ký & kiểu trả về**: Đối tượng trả về tuân thủ cấu trúc dữ liệu `ChangeDetectionResult` và `DamageComparisonItem`.

### 1.2. Thử nghiệm End-to-End trên Ảnh Thực tế
- **Dữ liệu thực nghiệm**: Ảnh `demo/input/image copy 3.png` (Check-in / Before) được ghép với ảnh Check-out (After) qua biến đổi xoay góc ($3.5^\circ$), biến dạng phối cảnh nhẹ và vẽ thêm 1 vết xước giả lập mới.
- **Kết quả đối soát**:
  - Trạng thái căn chỉnh: `aligned=True`, thuật toán `SIFT`, tìm được 697 điểm tương đồng (matches) với **664 inliers RANSAC**.
  - Tổng số vết phát hiện: 5 vết cũ (`old_damage`, IoU từ $0.841$ đến $0.926$) và 1 vết mới phát sinh (`new_damage`, `scratch` tại vị trí `Nắp capo`, $\text{IoU} = 0.016$).
  - **Tệp trực quan kết quả**: Đã tạo thành công ảnh side-by-side tại `demo/results/change_detection_sample.jpg` hiển thị rõ viền Xanh lá (`old_damage`) và Đỏ (`new_damage`).

---

## 2. Bảng So Sánh Benchmark PyTorch (`.pt`) vs ONNX (`.onnx`)

### 2.1. Thông Số Xuất Mô Hình ONNX (`ai/scripts/export_onnx.py`)
| Tên Mô Hình | Định dạng Gốc (`.pt`) | Dung lượng `.pt` | Dung lượng `.onnx` (opset=12) | Trạng thái Nạp (`onnxruntime`) |
| :--- | :---: | :---: | :---: | :---: |
| **Damage Detection** (`damage_best`) | YOLOv8n Detect | 6.19 MB | **12.19 MB** | `['images']` $\rightarrow$ `['output0']` (OK) |
| **Parts Segmentation** (`parts_seg`) | YOLOv8s-seg Segment | 22.73 MB | **45.19 MB** | `['images']` $\rightarrow$ `['output0', 'output1']` (OK) |

*Ghi chú*: Cả hai mô hình `.onnx` đều có kích thước $< 50\text{ MB}$ (dưới mức trần $90\text{ MB}$ quy định).

### 2.2. Đo Lường Độ Trễ Suy Luận (Inference Latency trên 8 ảnh `demo/input/`)
*(Đã loại bỏ ảnh warm-up đầu tiên để đảm bảo độ chính xác)*

| Metric / Tiêu chí | PyTorch Backend (`.pt` trên GPU) | ONNX Runtime (`.onnx` trên CPU) | Đánh giá so với KPI Mentor ($\le 3.0\text{s}$/ảnh) |
| :--- | :---: | :---: | :---: |
| **Thời gian nạp model** | $2.842\text{s}$ | **$0.004\text{s}$** *(Cực nhanh)* | Đạt |
| **Độ trễ trung bình (Mean)** | **$0.141\text{s}$/ảnh** | **$0.728\text{s}$/ảnh** | **VƯỢT XA KPI** (Nhanh hơn từ $4.1\times$ đến $21.2\times$) |
| **Độ trễ thấp nhất (Min)** | $0.102\text{s}$ | $0.551\text{s}$ | Đạt |
| **Độ trễ cao nhất (Max)** | $0.183\text{s}$ | $0.939\text{s}$ | Đạt |
| **Tỷ lệ đạt KPI $\le 3.0\text{s}$** | **100% (8/8 ảnh)** | **100% (8/8 ảnh)** | **XUẤT SẮC** |

---

## 3. Kết Quả Kiểm Thử Đơn Vị (`pytest`)
- **Lệnh thực thi**: `python -m pytest ai/tests/test_pipeline_unit.py -v`
- **Kết quả tổng hợp**:
  - Tổng số test case: **46 test cases**
  - **Passed: 46 / 46 (100% PASS)**
  - Failed: 0
  - Thời gian thực thi: **2.52 giây**
- **Chi tiết các nhóm kiểm thử**:
  - `TestPipelinePostProcessing` (28/28 tests passed): Đảm bảo các quy tắc an toàn vật lý, hợp nhất nhãn tổn thất, giải quyết chồng lấn 4 cửa bên hông, phân loại góc chụp cận cảnh/toàn cảnh.
  - `TestChangeDetectionSuite` (8/8 tests passed): Đảm bảo tính toán IoU, ma trận nắn Homography, phân loại Before/After, chuẩn hóa 8 bộ phận chuẩn.
  - `TestPhase2OptimizationSuite` (10/10 tests passed): Đảm bảo khử nhiễu bộ phận theo hướng đầu xe, nhận diện đèn hậu đỏ bằng không gian màu HSV, định dạng phản hồi 3 trạng thái.

- **Kiểm tra luồng suy luận chính (`random_demo.py`)**:
  - Chạy thành công toàn bộ 8 ảnh mẫu, xuất file JSON chuẩn tại `demo/results/predictions.json` và ảnh minh họa tại `demo/results/`.

---

## 4. Trạng Thái Git & Kiểm Soát Tệp Nặng
- **Cấu hình `.gitignore`**:
  - Đã bổ sung các đường dẫn trọng yếu: `/scratch/`, `graphify-out/`, `/demo/results/`, `/ai/data/`, `/ai/runs/`, `*.onnx`, `*.pt`.
- **Rà soát dung lượng (`git status -s`)**:
  - Không có bất kỳ tệp dữ liệu huấn luyện hoặc trọng số lớn $> 90\text{ MB}$ nào lọt vào staging của Git (`Files >90MB: 0`).
  - Toàn bộ kết quả demo nặng (`demo/results/`) được giữ tại máy cục bộ và được gitignore đúng chuẩn.

---

## 5. Cập Nhật Knowledge Graph (Graphify)
- Đã chạy lại Graphify quét trên các module cốt lõi (`ai/inference/`, `ai/tests/`, `random_demo.py`, `README.md`).
- **Thống kê đồ thị tri thức mới nhất**:
  - **240 nodes**, **555 edges**, **20 communities** (tăng thêm các kết nối trực tiếp với `supervision` và mô-đun Change Detection).
  - Không phát hiện vòng lặp import (Import Cycles: None).
  - Đã đồng bộ tài liệu kiến trúc sang `docs/GRAPH_REPORT.md`.
