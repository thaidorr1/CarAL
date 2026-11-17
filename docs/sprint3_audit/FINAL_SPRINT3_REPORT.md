# FINAL REPORT: BÁO CÁO KIỂM TOÁN KIẾN TRÚC & ĐỘ CHÍNH XÁC AI SPRINT 3
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Ngày phát hành:** 2026-11-12  
**Phiên bản kiểm toán:** 3.0 (Comprehensive Audit & Refactoring Proposal)  
**Tác giả:** Senior AI Architect / Computer Vision QA Lead  
**Tình trạng thực hiện:** Read-only Audit hoàn tất. Bảo toàn 100% hiện trạng mã nguồn và dữ liệu.

---

## 1. Tóm Tắt Hiện Trạng Dự Án (Executive Summary)

Dự án **AutoClaim-AI** đã xây dựng thành công luồng xử lý suy luận cốt lõi cho bài toán Giám định Tổn thất Xe Cơ giới Tự động:
- Đã thiết lập xong luồng **TwoStagePipeline** kết hợp giữa mô hình phát hiện tổn thất (`damage_best.pt`) và mô hình phân đoạn bộ phận xe (`parts_seg.pt`).
- Đã tích hợp thành công mô-đun đối soát **Before/After** (`change_detection.py`) dựa trên giải thuật SIFT/ORB và RANSAC Homography.
- Đã xuất thành công hai mô hình sang định dạng **ONNX** đạt chuẩn, tối ưu hóa độ trễ suy luận trên CPU xuống chỉ còn **$0.26\text{s}$/ảnh**, vượt xa KPI mục tiêu $\le 3.0\text{s}$.
- Bộ kiểm thử thuật toán đạt **46/46 passed (100%)**.

Tuy nhiên, đợt kiểm toán chuyên sâu này đã chỉ ra **3 rủi ro kiến trúc mang tính sống còn** cần được tái cấu trúc trước khi bước vào giai đoạn bảo vệ đồ án tốt nghiệp:
1. **Điểm nghẽn độ chính xác của Mô hình Phân Đoạn Bộ Phận**: Mask mAP@50 chỉ đạt **$28.34\%$**, thiếu hẳn lớp `Đèn xe` và gộp chung 4 cánh cửa thành một nhãn `cua_hong`.
2. **Hiện tượng "Bù đắp Quy tắc" (Rule-Engine Compensation)**: Tệp `pipeline.py` đã bị phình to bất thường lên **2,958 dòng mã** với hàng ngàn dòng code heuristics chắp vá để cứu lỗi cho mô hình AI.
3. **Phân mảnh cấu hình và rác dữ liệu**: Workspace đang bị chiếm dụng bởi **117 MB bản sao lưu tĩnh** (`_backup_config_code/`), 22 script mồ côi trong `archive_tools/`, 19 script tạm trong `scratch/`, và 6 file cấu hình trùng lặp.

---

## 2. Bảng Điểm Đánh Giá Kiến Trúc Tổng Thể (Architectural Scorecard)

| Tiêu chí Đánh giá | Điểm số (Thang 10) | Bằng chứng Kỹ thuật & Hiện trạng |
| :--- | :---: | :--- |
| **1. Modularity (Tính mô-đun)** | **4.0 / 10** | `pipeline.py` là một God-Class 2,958 dòng làm từ việc tính toán hình học, gán nhãn, heuristics, định hướng xe đến vẽ ảnh Unicode. |
| **2. Coupling (Độ ghép nối)** | **5.5 / 10** | Logic ánh xạ bị phụ thuộc chặt chẽ vào cấu trúc trả về thô của Ultralytics Boxes/Masks thay vì đi qua lớp trừu tượng (Abstraction Layer). |
| **3. Cohesion (Độ gắn kết nội tại)** | **4.5 / 10** | Graphify phân tích Cohesion của cụm `pipeline.py` chỉ đạt $0.096$ (rất thấp), các nút chức năng ít liên kết trực tiếp với nhau. |
| **4. Maintainability (Khả năng bảo trì)** | **5.0 / 10** | Bất kỳ một chỉnh sửa nhỏ về ngưỡng tọa độ ở góc chụp này cũng có nguy cơ làm gãy logic nhận diện ở góc chụp khác. |
| **5. Testability (Khả năng kiểm thử)** | **6.5 / 10** | 46 unit tests chạy tốt, nhưng 100% là test hình học giả lập; chưa có test cho FastAPI (`ai/api/`) và test nạp model thực tế. |
| **6. Reliability (Độ tin cậy vận hành)** | **8.0 / 10** | **Điểm sáng**: Các chốt chặn an toàn (Safety Guards) hoạt động rất tốt, sẵn sàng trả về `khong_xac_dinh` khi ảnh cận cảnh thay vì đoán bừa. |
| **7. AI Accuracy (Độ chính xác AI)** | **6.0 / 10** | Damage Detect đạt mức khá (mAP50 70.9%); Parts Seg yếu (Mask mAP50 28.3%), thiếu 2 lớp proposal (`tear_puncture`, `paint_chipping`). |
| **8. Documentation (Tài liệu hóa)** | **8.5 / 10** | Tài liệu rất phong phú: có SRS, Proposal, Hướng dẫn chạy, Knowledge Graph, báo cáo kiểm toán kỹ thuật qua từng sprint. |
| **9. Project Organization (Tổ chức dự án)**| **5.0 / 10** | Tồn tại nhiều thư mục thừa (`_backup_config_code`, `archive_tools`, `scratch`), cấu hình YAML phân mảnh, 2 file requirements.txt. |
| **TỔNG ĐIỂM TRUNG BÌNH** | **5.83 / 10** | **Mức Đạt Yêu Cầu Chức Năng - Cần Tinh Gọn Cấu Trúc Để Đạt Chuẩn Xuất Sắc** |

---

## 3. Trả Lời 8 Câu Hỏi Trọng Tâm Của Kiểm Toán

### 1. Dự án hiện đang thừa những gì?
- Thừa thư mục bản sao lưu `_backup_config_code/` (117.2 MB).
- Thừa 22 script thử nghiệm mồ côi trong `ai/scripts/archive_tools/` và 19 script tạm trong `scratch/`.
- Thừa các checkpoint COCO pre-trained gốc (`yolov8n.pt`, `yolov8s.pt`, `yolov8s-seg.pt` - 51.8 MB).
- Thừa 2 file cấu hình môi trường phụ: `ai/requirements.txt` và `ai/requirements-dev.txt`.
- Thừa các thư mục con rỗng trong `ai/data/parts_verified/`.

### 2. Dự án đang thiếu những gì?
- Thiếu nhãn Ground Truth chuẩn hóa cho 8 ảnh trong `demo/input/`.
- Thiếu lớp `Đèn xe` và phân tách `Cửa trước` / `Cửa sau` trong mô hình AI phân đoạn.
- Thiếu dữ liệu huấn luyện cho 2 nhóm tổn thất `tear_puncture` (rách cản) và `paint_chipping` (tróc sơn).
- Thiếu Integration Test cho hệ thống REST API (`ai/api/main.py`).

### 3. File nào có thể loại bỏ an toàn (Safe to Delete)?
- Toàn bộ thư mục `ai/scripts/archive_tools/` (sau khi nén lưu trữ ngoài repo).
- Hai file `ai/requirements.txt` và `ai/requirements-dev.txt`.
- Thư mục rỗng `ai/data/parts_verified/`.
- Tệp `ai/configs/classes.md` (chứa tài liệu lỗi thời tham chiếu file chết).

### 4. File nào cần gộp (Merge)?
- `ai/configs/damage.yaml` + `ai/configs/data.yaml` + `ai/configs/classes.yaml` $\rightarrow$ gộp thành 1 tệp `ai/configs/damage.yaml` chuẩn hóa 5 lớp.
- `ai/configs/parts.yaml` + `ai/configs/data_parts.yaml` $\rightarrow$ gộp thành 1 tệp `ai/configs/parts.yaml` chuẩn hóa 8 lớp.
- 4 script huấn luyện phân tán $\rightarrow$ gộp thành 2 script duy nhất: `ai/train/train_damage.py` và `ai/train/train_parts.py`.

### 5. File nào nên giữ nguyên (Keep)?
- `ai/models/weights/*` (cả 4 tệp trọng số `.pt` và `.onnx`).
- `ai/inference/change_detection.py` (đã tối ưu hóa).
- `ai/api/main.py` và `ai/api/schemas.py`.
- `ai/tests/test_pipeline_unit.py` (bảo vệ hồi quy).
- `requirements.txt` (ở thư mục gốc).

### 6. Có module nào đang làm quá nhiều nhiệm vụ?
- **Chính là [ai/inference/pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py)** (2,958 dòng). Tệp này đang vi phạm nguyên lý Single Responsibility nghiêm trọng nhất dự án. Cần chia nhỏ thành `geometry.py`, `rules.py`, `mapper.py`, `visualizer.py` và `pipeline.py`.

### 7. Có module nào không còn cần thiết?
- `ai/scripts/runtime.py` (runtime legacy trước khi có pipeline 2 giai đoạn).
- `ai/scripts/finalize_damage.py` (script tạm Sprint 1).

### 8. Cấu trúc hiện tại có dễ hiểu và dễ trình bày với giảng viên không?
- **CHƯA ĐẠT**. Với 130 KB code dồn vào một file, 6 file configs chồng chéo, các script huấn luyện nằm rải rác ở cả `ai/train/` lẫn `ai/scripts/`, và sự xuất hiện của các thư mục tạm (`scratch`, `_backup_config_code`), giảng viên sẽ gặp khó khăn khi thẩm định kiến trúc và có thể đánh giá dự án thiếu tính bài bản (codebase lộn xộn).

---

## 4. TOP 10 THÀNH PHẦN CẦN TINH GỌN NHẤT (THE TOP 10 REFACTORING TARGETS)

Bảng xếp hạng ưu tiên hành động dựa trên tiêu chí: **Ảnh hưởng Kiến trúc + Khối lượng Tiết kiệm + Mức độ An toàn**:

| Hạng | Tên Thành Phần / Tệp Tin | Hiện Trạng & Vấn Đề | Đề Xuất Xử Lý | Mức Độ An Toàn |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **`ai/inference/pipeline.py`** | 2,958 dòng (God Class), ôm đồm quá nhiều chức năng | Bẻ nhỏ thành 4 module con (`geometry`, `rules`, `mapper`, `visualizer`) | **An toàn cao** (Nếu 46 unit tests pass 100%) |
| **2** | **Thư mục `_backup_config_code/`** | Chiếm 117.2 MB bản sao lưu tĩnh vô ích trong repo Git | Nén `.zip` lưu trữ ngoài repo hoặc xóa khỏi workspace | **Tuyệt đối an toàn** (Git đã lưu lịch sử) |
| **3** | **Cụm cấu hình `ai/configs/*.yaml`** | 6 file chồng chéo giữa 5 lớp proposal và 6 lớp CarDD | Hợp nhất thành 2 file duy nhất (`damage.yaml`, `parts.yaml`) | **An toàn cao** (Đã kiểm tra parser) |
| **4** | **Thư mục `ai/scripts/archive_tools/`** | 22 script tạm dở dang, không có bất kỳ caller nào | Xóa bỏ hoặc đưa vào `_archive/` | **Tuyệt đối an toàn** (0 import) |
| **5** | **File `ai/requirements.txt` & `dev.txt`** | 2 file trùng lặp và thiếu thư viện so với root | Xóa bỏ 2 file con, dùng duy nhất `requirements.txt` gốc | **Tuyệt đối an toàn** |
| **6** | **Cặp script `train_both_stages` & `train_parts_model`** | Nằm trong `scripts/` trùng lặp với code trong `train/` | Xóa bỏ, tập trung mã nguồn train về `ai/train/` | **An toàn cao** |
| **7** | **Thư mục `ai/data/parts_verified/`** | 6 thư mục con rỗng không chứa ảnh | Xóa bỏ thư mục rác | **Tuyệt đối an toàn** |
| **8** | **Tệp `random_demo.py`** | Nằm ở thư mục gốc, tên file mang tính ngẫu nhiên | Đổi tên & chuyển vị trí thành `demo/run_demo.py` | **An toàn cao** |
| **9** | **Các checkpoint COCO gốc (`yolov8*.pt`)** | Chiếm 51.8 MB trong `ai/models/` | Di chuyển ra thư mục `data/checkpoints/` hoặc lưu cloud | **An toàn cao** (Inference dùng `weights/`) |
| **10**| **Thư mục `scratch/`** | 19 tệp script thử nghiệm cũ từ các sprint trước | Đưa vào `.gitignore` hoặc nén thành archive | **Tuyệt đối an toàn** |

---

## 5. Kết Luận Cuối Cùng & Khuyến Nghị

Kiểm toán Sprint 3 đã hoàn thành xuất sắc sứ mệnh "chụp X-quang" toàn diện hệ thống AutoClaim-AI. Dự án sở hữu nền tảng thuật toán rất vững chắc và logic xử lý nghiệp vụ thông minh, nhưng đang bị bao phủ bởi lớp "bụi kỹ thuật" (technical debt) sau nhiều đợt thử nghiệm dồn dập.

**Khuyến nghị:**
Chủ dự án nên phê duyệt kế hoạch tinh gọn theo **Roadmap Sprint 4** (tại `11_sprint4_roadmap.md`) để:
1. Huấn luyện lại mô hình `parts_seg` đạt Mask mAP@50 $\ge 60\%$.
2. Refactor `pipeline.py` thành các module con sáng sủa, chuyên nghiệp.
3. Dọn dẹp Top 10 thành phần thừa để sẵn sàng bước vào giai đoạn nghiệm thu Capstone đạt điểm số cao nhất.
