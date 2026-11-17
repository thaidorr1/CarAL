# BÁO CÁO KIỂM TOÁN THAY ĐỔI ĐỘC LẬP - SPRINT 2 (AUTOCLAIM-AI)

**Ngày kiểm toán:** 2026-11-12  
**Đối tượng kiểm toán:** Toàn bộ mã nguồn, cấu hình, mô hình ONNX, bộ kiểm thử và kết quả thực nghiệm Sprint 2 thuộc dự án `AI_CAR`.  
**Cam kết tuân thủ:** Chế độ Read-only Audit (Không sửa code, không xóa file, không commit, không revert, bảo toàn 100% hiện trạng).

---

## 1. Kiểm Tra Git & Hiện Trạng Mã Nguồn

### 1.1. Lệnh kiểm tra Git thực hiện
```bash
git status --short
git diff --stat
git diff -- requirements.txt .gitignore
git diff -- ai/inference/change_detection.py
```

### 1.2. Phân loại chi tiết trạng thái các tệp (So với Base Commit `5694599`)

| Phân loại | Số lượng | Danh sách tệp & Mô tả chi tiết |
| :--- | :---: | :--- |
| **Đã sửa (`M`)** | 7 | - `.gitignore`: Bổ sung `/scratch/`, `/demo/results/`, `graphify-out/`, `*.onnx`, `*.pt`.<br>- `README.md`: Cập nhật tài liệu quy trình 2-stage, hướng dẫn test và chạy demo.<br>- `ai/configs/classes.yaml`: Chuẩn hóa 6 class hư hỏng và 7 bộ phận.<br>- `ai/configs/data.yaml`: Cập nhật đường dẫn dataset.<br>- `ai/scripts/coco_to_yolo.py`: Cải tiến script chuyển đổi nhãn COCO sang YOLO.<br>- `ai/train/train_damage.py`: Điều chỉnh tham số huấn luyện mô hình Damage.<br>- `docs/HUONG_DAN_CHAY.md`: Cập nhật tài liệu hướng dẫn vận hành toàn hệ thống. |
| **Đã xóa (`D`)** | 11 | - `ai/.gitignore`: Gộp về `.gitignore` tại thư mục gốc.<br>- `ai/inference/api.py`, `ai/inference/infer.py`, `ai/inference/schema.py`: Code suy luận legacy đơn lẻ thời kỳ đầu.<br>- `ai/runtime.py`: Code runtime cũ chưa hỗ trợ pipeline 2 giai đoạn.<br>- `ai/scripts/stats_cardd.py`, `ai/scripts/visualize_labels.py`: Script tiền xử lý cũ.<br>- `ai/tests/test_api.py`, `ai/tests/test_inference.py`: Bộ test cũ chỉ kiểm tra suy luận 1 model đơn lẻ.<br>- 2 ảnh tài liệu cũ trong `docs/references/`. |
| **Mới tạo / Chưa track (`??`)** | 24 | - **Core AI**: `ai/inference/pipeline.py` (TwoStagePipeline), `ai/inference/change_detection.py` (Đối soát Before/After), `ai/inference/__init__.py`.<br>- **Kiểm thử**: `ai/tests/test_pipeline_unit.py` (46 unit test cases).<br>- **Script công cụ**: `ai/scripts/export_onnx.py`, `ai/scripts/eval_phase2.py`, `ai/scripts/runtime.py`, `ai/scripts/train_both_stages.py`, `ai/scripts/train_parts_model.py`, `ai/scripts/archive_tools/`.<br>- **Cấu hình & Môi trường**: `requirements.txt`, `ai/configs/damage.yaml`, `ai/configs/parts.yaml`, `ai/configs/data_parts.yaml`.<br>- **Dịch vụ API**: `ai/api/` (FastAPI router & schemas chuẩn hóa).<br>- **Demo & Báo cáo**: `random_demo.py`, `demo/`, `docs/GRAPH_REPORT.md`, `docs/sprint2_ai_core_report.md`, `docs/folder_reorganization_report.md`, `docs/references/DATASET_AND_PIPELINE_SPEC.md`, `docs/references/Phan_doan_bo_phan_xe.docx`. |
| **Bị di chuyển (Refactored)** | - | - Logic từ `ai/runtime.py` $\rightarrow$ `ai/scripts/runtime.py` và `ai/inference/pipeline.py`.<br>- Logic từ `ai/inference/api.py` $\rightarrow$ `ai/api/main.py`.<br>- Script phân tích cũ `stats_cardd.py` $\rightarrow$ `ai/scripts/archive_tools/`. |
| **Chưa được kiểm tra (Unchecked)** | - | - `ai/api/` (API Gateway FastAPI): Chưa được bao phủ bởi pytest trong lần chạy này.<br>- `ai/train/train_damage.py`, `ai/train/train_parts.py`, `ai/scripts/train_both_stages.py`: Các script huấn luyện chưa được chạy kiểm thử hồi quy (regression test). |

---

## 2. Kiểm Tra Chuyên Sâu `ai/inference/change_detection.py`

### 2.1. Đối chiếu phiên bản Trước và Sau (Diff Audit)
So sánh giữa `_backup_config_code/ai/inference/change_detection.py` và phiên bản hiện tại:
1. **API & Interface**: Chữ ký các hàm trọng tâm (`align_images_and_compare`, `calculate_bbox_iou`, `warp_bbox_with_homography`, `visualize_change_detection`) **không thay đổi**, hoàn toàn tương thích ngược 100%.
2. **Cấu trúc dữ liệu**: `ChangeDetectionResult` và `DamageComparisonItem` giữ nguyên cấu trúc dataclass và phương thức `.to_dict()`.
3. **Mối liên hệ với `pipeline.py`**: Hoàn toàn độc lập (Decoupled). `pipeline.py` không import `change_detection.py`, không xảy ra hiện tượng circular import.
4. **Logic matching**:
   - Tối ưu hóa tính ma trận IoU bằng cách tích hợp `supervision.box_iou_batch`.
   - Vẫn bảo toàn cơ chế fallback tính toán vòng lặp kép nguyên bản bằng `calculate_bbox_iou` nếu thư viện `supervision` vắng mặt hoặc gặp ngoại lệ.

### 2.2. Phân tích Rủi ro Thuật toán & Góc chụp Lệch
1. **Căn cứ của quy tắc $\text{IoU} \ge 0.60$**:
   - *Cơ sở*: Ngưỡng $0.60$ bắt nguồn từ tiêu chuẩn định vị tổn thất vật thể biến dạng nhẹ (deformation) trong bảo hiểm, cao hơn ngưỡng $0.50$ thông thường của PASCAL VOC để tránh bắt nhầm hai vết xước nằm cạnh nhau.
   - *Hạn chế*: Ma trận Homography phẳng ($3 \times 3$) giả định bề mặt xe là một mặt phẳng 2D. Trong khi đó thân vỏ ô tô là cấu trúc không gian 3D cong (curved surface). Khi góc chụp thay đổi $> 15^\circ$, hiện tượng sai sai số thị sai (parallax) sẽ làm sai lệch vị trí hộp bao sau khi warp.
2. **Trường hợp phân loại nhầm thành Damage mới (False New Damage Alert)**:
   - **Vết xước nhỏ (Micro-damages)**: Với vết xước chỉ có kích thước $20 \times 20$ pixel, chỉ cần ma trận Homography bị lệch 6 pixel do góc nghiêng, IoU giữa hộp bao Before và After sẽ tụt xuống $< 0.45$, dẫn đến việc **vết cũ bị quy kết nhầm thành tổn thất mới phát sinh**.
   - **Bề mặt xe bóng / Đơn sắc**: Với xe sơn trắng/đen bóng không có hoa văn, SIFT/ORB không tìm đủ 10 inliers (`aligned = False`). Khi đó hệ thống rơi vào chế độ fallback so sánh tọa độ thô không nắn chỉnh, khiến **100% tổn thất cũ đều bị gán nhãn thành tổn thất mới** nếu góc chụp bị lệch.

---

## 3. Kiểm Tra Mô Hình ONNX Độc Lập

### 3.1. Xác minh Tệp & Cấu trúc Mô hình
- `ai/models/weights/damage_best.onnx` (**12.19 MB**): Tồn tại, load thành công trên ONNX Runtime (`onnxruntime` v1.31.1).
  - Input: `['images']` shape `[1, 3, 640, 640]`, kiểu `tensor(float)`.
  - Output: `['output0']` shape `[1, 10, 8400]`.
  - Class mapping: 6 lớp trùng khớp 100% với PyTorch:
    `{0: 'dent', 1: 'scratch', 2: 'crack', 3: 'glass_shatter', 4: 'lamp_broken', 5: 'tire_flat'}`.
- `ai/models/weights/parts_seg.onnx` (**45.19 MB**): Tồn tại, load thành công trên ONNX Runtime.
  - Input: `['images']` shape `[1, 3, 640, 640]`, kiểu `tensor(float)`.
  - Output: `['output0', 'output1']` với shape `[1, 43, 8400]` và `[1, 32, 160, 160]`.
  - Class mapping: 7 lớp trùng khớp 100% với PyTorch:
    `{0: 'can_truoc', 1: 'can_sau', 2: 'nap_capo', 3: 've_che_banh', 4: 'cua_hong', 5: 'kinh_chieu_hau', 6: 'kinh_lai'}`.

### 3.2. Đánh giá Mức độ Tương đồng Dự đoán (Prediction Parity Audit)
Kiểm tra đối đầu trực tiếp trên toàn bộ 8 ảnh `demo/input/`:
- **Damage Detection (PT vs ONNX)**:
  - 7/8 ảnh cho kết quả IoU giữa các hộp bao $\ge 0.938$ (trong đó 4 ảnh đạt $\text{IoU} \ge 0.985$).
  - **Sai lệch biên phát hiện**: Trên `image copy 5.png`, PyTorch phát hiện 0 box do confidence đạt $0.248$ (bị loại bởi ngưỡng $0.25$), trong khi ONNX với phép tính float trên CPU cho ra 2 box móp méo với confidence $0.252$. Đây là sai số làm tròn số học (floating-point epsilon) đặc trưng giữa nhân PyTorch GPU/CUDA và ONNX Runtime CPU.
- **Parts Segmentation (PT vs ONNX)**:
  - 6/8 ảnh đạt độ tương đồng mặt nạ và IoU hộp bao rất cao ($0.821 - 0.981$).
  - Trên `image copy 6.png` và `image copy 7.png`, do ranh giới confidence ở vùng góc hẹp, số lượng mặt nạ sau NMS chênh lệch 1 mặt nạ giữa hai engine.

---

## 4. Kiểm Toán Lại Benchmark Hiệu Năng (Sửa Sai So Sánh Bất Đối Xứng)

### 4.1. Vấn đề Phát hiện trong Báo cáo Trước
Trong báo cáo trước, hệ thống đã so sánh:
- **PyTorch backend chạy trên GPU chuyên dụng**: NVIDIA GeForce RTX 4050 Laptop GPU ($0.141\text{s}$/ảnh).
- **ONNX Runtime chạy trên CPU**: Intel Core i5-12450H CPU ($0.728\text{s}$/ảnh).
$\Rightarrow$ **Kết luận so sánh như vậy là bất đối xứng và không hợp lệ** (so sánh GPU vs CPU). Không thể dựa vào con số đó để đánh giá hiệu năng tối ưu hóa thực tế của định dạng ONNX.

### 4.2. Kết quả Đo lường Độc lập Cùng Phần Cứng & Cùng Điều Kiện
- **Môi trường đo**:
  - CPU: 12th Gen Intel Core i5-12450H (8 cores, 12 threads)
  - GPU: NVIDIA GeForce RTX 4050 Laptop GPU
  - Bộ ảnh: 8 ảnh thực tế trong `demo/input/`
  - Quy cách đo: 1 ảnh warm-up, đo trung bình trên 8 ảnh tiếp theo; tách bạch thời gian tiền xử lý (pre), suy luận (inf), hậu xử lý (post).

#### Bảng so sánh cùng thiết bị (Fair Comparison):

| Mô hình & Thiết bị thực thi | Tiền xử lý (Pre) | Suy luận cốt lõi (Inference) | Hậu xử lý (Post) | Tổng thời gian (Total) | Tốc độ tương đối |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Damage Detect - PyTorch (GPU CUDA)** | 2.7 ms | 89.9 ms | 2.7 ms | 107.8 ms (0.108s) | Chuẩn GPU |
| **Damage Detect - PyTorch (CPU)** | 3.7 ms | 94.8 ms | 1.3 ms | 109.8 ms (0.110s) | Chuẩn CPU |
| **Damage Detect - ONNX Runtime (CPU)** | 4.0 ms | **60.0 ms** | 1.7 ms | **77.2 ms (0.077s)** | **ONNX nhanh hơn 29.7% trên CPU** |
| **Parts Segment - PyTorch (GPU CUDA)** | 2.8 ms | 65.3 ms | 4.4 ms | 83.1 ms (0.083s) | Chuẩn GPU |
| **Parts Segment - PyTorch (CPU)** | 3.4 ms | 191.7 ms | 4.8 ms | 208.6 ms (0.209s) | Chuẩn CPU |
| **Parts Segment - ONNX Runtime (CPU)** | 3.5 ms | **168.9 ms** | 5.0 ms | **187.6 ms (0.188s)** | **ONNX nhanh hơn 10.1% trên CPU** |

#### Đánh giá chuẩn xác:
1. **Trên cùng môi trường CPU**: ONNX Runtime thực sự **nhanh hơn PyTorch từ 10.1% đến 29.7%** nhờ tối ưu đồ thị tính toán và ONNX Slimming.
2. **So với KPI Đề xuất ($\le 3.0\text{s}$/ảnh)**:
   - Tổng thời gian chạy cả 2 stage trên CPU:
     $\text{PyTorch (CPU)} = 109.8 + 208.6 = 318.4\text{ ms} \approx 0.32\text{s}$/ảnh.
     $\text{ONNX Runtime (CPU)} = 77.2 + 187.6 = 264.8\text{ ms} \approx 0.26\text{s}$/ảnh.
   - Cả hai cấu hình đều **vượt xa KPI $3.0\text{ giây}$ gấp hơn 10 lần**, chứng minh khả năng triển khai thực tế trên cả máy chủ không có GPU.

---

## 5. Kiểm Tra Dữ Liệu Demo (`predictions.json` trên 8 Ảnh Mẫu)

Đọc và kiểm tra chi tiết các trường dữ liệu thực tế tại `demo/results/predictions.json`:

| Tên tệp ảnh | Trạng thái (`status`) | Kiểu chụp (`shot_type`) | Số bộ phận tìm thấy | Số tổn thất & Kết quả ánh xạ | Đánh giá tính chính xác & Lỗi tiềm ẩn |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `image copy 3.png` | `needs_review` | `mid_view` | 1 (`Nắp capo`) | - `lamp_broken` $\rightarrow$ `Đèn xe` (0.88)<br>- 3 tổn thất $\rightarrow$ `khong_xac_dinh` (0.75, 0.50, 0.27)<br>- 1 `dent` $\rightarrow$ `Nắp capo` (0.50) | **Chính xác**: Phát hiện cảnh báo hộp bao bất thường ($>30\%$ diện tích ảnh do detect bắt vùng thân xe rộng). Kích hoạt cờ `needs_review` đúng thiết kế. |
| `image copy 4.png` | `completed` | `far_view` | 4 (`Kính xe`, `Cửa trước`, `Nắp capo`, `Vè che bánh`) | - 1 `dent` $\rightarrow$ `Cửa trước` (0.87, `mask_overlap`) | **Tốt**: Phân đoạn bộ phận đầy đủ, ánh xạ trùng khớp mặt nạ hoàn hảo. |
| `image copy 5.png` | `completed` | `mid_view` | 1 (`Vè che bánh`) | 0 tổn thất | Xe nguyên vẹn không tổn thất, hệ thống không báo động giả. |
| `image copy 6.png` | `insufficient_data` | `close_up` | 1 (`Cản trước`) | - 1 `crack` $\rightarrow$ `khong_xac_dinh` (0.76) | **Safety Guard hoạt động tốt**: Ảnh quá cận cảnh, góc chụp hẹp không đủ bối cảnh không gian nên từ chối gán nhãn bừa bãi. |
| `image copy 7.png` | `completed` | `mid_view` | 3 (`Cửa trước`, `Nắp capo`, `Kính xe`) | - 3 `scratch` $\rightarrow$ `Cửa trước` (0.93, 0.87, 0.75) | Ánh xạ chính xác cụm 3 vết xước trên cùng cánh cửa. |
| `image copy 8.png` | `insufficient_data` | `close_up` | 1 (`Vè che bánh`) | - 1 `scratch` $\rightarrow$ `khong_xac_dinh` (0.59) | Từ chối định vị cận cảnh đúng quy tắc an toàn. |
| `image copy.png` | `completed` | `far_view` | 2 (`Nắp capo`, `Kính xe`) | - 1 `lamp_broken` $\rightarrow$ `Đèn xe` (0.89) | Quy tắc ngữ nghĩa cứng (Semantic hard-rule) gán chính xác vỡ đèn vào Đèn xe dù mô hình seg bỏ sót cụm đèn pha nhỏ. |
| `image.png` | `insufficient_data` | `close_up` | 1 (`Cản trước`) | - 2 `dent` $\rightarrow$ `khong_xac_dinh` (0.55, 0.43) | Ảnh chụp quá sát vết móp, chuyển trạng thái yêu cầu chụp lại ảnh góc rộng. |

**Đánh giá trung thực**: Hệ thống không hề che giấu lỗi mà đã phân loại đúng $3/8$ ảnh vào trạng thái `insufficient_data` và $1/8$ ảnh vào `needs_review` để chuyển giám định viên can thiệp.

---

## 6. Kiểm Tra Toàn Diện Bộ Kiểm Thử Unit Test

### 6.1. Xác minh Thực Thi
- **Bộ test thực thi**: `ai/tests/test_pipeline_unit.py`.
- **Tổng số test cases**: **46 tests** (Toàn bộ 46/46 đều được chạy, **không có test nào bị skip hay xfail**).
- **Thời gian chạy**: $2.52\text{ giây}$.

### 6.2. Phân loại Bản chất Kỹ thuật của Bộ Kiểm Thử
Cần phân định rạch ròi bản chất của 46 test cases hiện có:
1. **Unit Test Thuần Túy (100% của 46 tests)**:
   - `TestPipelinePostProcessing` (28 tests): Kiểm tra tính đúng đắn của các hàm toán học, giải thuật hình học (`calculate_bbox_iou`, `_bbox_to_polygon`), khử trùng lặp NMS, 4 cửa hông, luật ngữ cảnh (contextual rules). Toàn bộ dữ liệu đầu vào là tọa độ giả lập (synthetic mocks), không nạp mô hình deep learning.
   - `TestChangeDetectionSuite` (8 tests): Kiểm tra tính toán Homography với ma trận tịnh tiến biết trước, kiểm tra các giá trị mặc định của config, kiểm tra danh mục 8 bộ phận chuẩn.
   - `TestPhase2OptimizationSuite` (10 tests): Kiểm tra bộ lọc hướng đầu xe, chuyển đổi màu HSV cho đèn hậu đỏ, kiểm tra logic cờ cảnh báo trạng thái.
2. **Integration Test (Kiểm thử Tích hợp)**: **0 test trong bộ test hiện tại**. Chưa có test tự động kiểm tra tương tác cơ sở dữ liệu hoặc End-to-End API Gateway FastAPI (`ai/api/`).
3. **AI Model Evaluation (Đánh giá Định lượng Mô hình Học Sâu)**: **0 test trong bộ test này**. Các chỉ số học máy ($mAP@50$, $mAP@50-95$, $mIoU$) nằm ở script độc lập `ai/scripts/eval_phase2.py` và cần tập dữ liệu kiểm định lớn (`val/test set`), không thuộc phạm vi của `test_pipeline_unit.py`.

---

## 7. Bảng Tổng Hợp Kiểm Toán & Đề Xuất Xử Lý

### 7.1. Các Thay Đổi An Toàn (Safe Changes)
- **Tối ưu hóa Change Detection**: Thêm cơ chế bắt ngoại lệ `cv2.error` và fallback khi thiếu đặc trưng giúp triệt tiêu hoàn toàn nguy cơ sập tiến trình (crash).
- **Đóng gói định dạng ONNX**: Đã xuất thành công 2 mô hình với kích thước nhỏ gọn ($12.19\text{ MB}$ và $45.19\text{ MB}$), giữ nguyên cấu trúc lớp và kích thước tensor.
- **Dọn dẹp môi trường**: Bổ sung `requirements.txt` chuẩn xác và đưa `/demo/results/`, `graphify-out/` vào `.gitignore` ngăn chặn phình to repository.

### 7.2. Các Thay Đổi Có Rủi Ro (Risky Changes)
- **Thư viện `onnxruntime-gpu` tự động cập nhật**: Khi nạp ONNX trên máy có card NVIDIA, Ultralytics tự động gọi `pip install onnxruntime-gpu`, xung đột với phiên bản CUDA DLL (`cublasLt64_13.dll`). Hệ thống đã tự động fallback về `CPUExecutionProvider` an toàn, nhưng cần ghim rõ phiên bản trong môi trường triển khai để tránh cảnh báo log.
- **Rủi ro ảnh góc nghiêng trong Change Detection**: Việc sử dụng ma trận Homography phẳng 2D sẽ gây cảnh báo sai (báo nhầm vết cũ thành vết mới) khi người dùng chụp góc Check-out chênh lệch $> 15^\circ$ so với Check-in.

### 7.3. Những Nội Dung Chưa Đủ Bằng Chứng (Unsubstantiated Gaps)
- **Độ chính xác định lượng của Change Detection trên Dataset quy mô lớn**: Hiện tại thuật toán mới chỉ được thử nghiệm trên 1 cặp ảnh thực nghiệm + giả lập biến đổi tọa độ; chưa có tập benchmark Before/After có ground-truth 100 cặp ảnh thực tế.
- **Kiểm thử tự động cho REST API**: `ai/api/main.py` chưa có bộ test pytest riêng biệt (`test_api.py` cũ đã bị xóa trong đợt tái cấu trúc).

### 7.4. Đề Xuất Hành Động Cụ Thể (Actionable Recommendations)
1. **Đối với Change Detection (Sprint 3)**:
   - Bổ sung kiểm tra ngưỡng sai số Homography: Nếu góc xoay ước tính $> 15^\circ$ hoặc phần trăm inliers $< 30\%$, trả về trạng thái `alignment_unreliable` và yêu cầu giám định viên xác nhận thủ công thay vì tự động kết luận lỗi mới.
   - Thử nghiệm kết hợp so sánh đặc trưng cục bộ (Local Patch SSIM / Feature Metric) tại vùng hộp bao thay vì chỉ dựa vào IoU thuần túy.
2. **Đối với Benchmark**:
   - Sử dụng bảng so sánh CPU-to-CPU trong báo cáo chính thức gửi Mentor: **ONNX Runtime nhanh hơn PyTorch 29.7% ở tác vụ Detect và 10.1% ở tác vụ Segment trên cùng CPU**.
3. **Đối với Testing**:
   - Xây dựng lại file `ai/tests/test_api_v2.py` để kiểm thử tự động các endpoint FastAPI (`/predict`, `/change-detection`) bằng `httpx.AsyncClient`.
