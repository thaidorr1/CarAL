# HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH HỆ THỐNG AI_CAR (AUTOCLAIM AI - C1SE.79)

Tài liệu này hướng dẫn chi tiết quy trình cài đặt môi trường, chạy suy luận (**Two-Stage Context-Aware Pipeline**), khởi chạy API & Web UI, huấn luyện lại (**Retrain**) hai mô hình YOLOv8-seg, cũng như kiểm toán dữ liệu (**Dataset Audit**) và chạy bộ kiểm thử đơn vị (**Unit Tests**) cho dự án Capstone **C1SE.79 - AutoClaimAI**.

---

## 1. Kiến trúc Pipeline 2 Giai Đoạn (Two-Stage Pipeline)

Hệ thống phân tích hư hại xe ô tô hoạt động theo kiến trúc **2 mô hình YOLOv8-seg chạy song song** kết hợp **Context & Physical Semantic Rules** (`ai/inference/pipeline.py`):

1. **Stage 1 — Nhận diện Hư hại (`ai/models/weights/damage_best.pt`)**:
   - Phân đoạn 6 lớp hư hại chuẩn CarDD:
     - `0: dent` (Móp, lõm thân vỏ)
     - `1: scratch` (Trầy, xước sơn)
     - `2: crack` (Nứt, rách phần nhựa/thân vỏ)
     - `3: glass_shatter` (Vỡ kính)
     - `4: lamp_broken` (Vỡ đèn xe)
     - `5: tire_flat` (Xẹp/hỏng lốp xe)

2. **Stage 2 — Phân đoạn Bộ phận xe (`ai/models/weights/parts_seg.pt`)**:
   - Phân đoạn **7 bộ phận xe chuẩn hóa** (theo tài liệu đặc tả *Phân đoạn bộ phận xe*):
     - `0: can_truoc` (`cản trước`)
     - `1: can_sau` (`cản sau`)
     - `2: nap_capo` (`nắp capo`)
     - `3: ve_che_banh` (`vè che bánh`)
     - `4: cua_hong` (`cửa hông`)
     - `5: kinh_chieu_hau` (`kính chiếu hậu`)
     - `6: kinh_lai` (`kính lái`)

3. **Stage 3 — Context Pipeline & Khóa Vật Lý (`map_damages_to_parts`)**:
   - **Hard Semantic Rules**: Khóa cứng `lamp_broken -> đèn xe`, `tire_flat -> Bánh xe / Lốp xe`, cấm tuyệt đối gán `dent`/`scratch`/`crack` lên bề mặt kính hoặc đèn.
   - **Headlamp Shatter Guard**: Bảo vệ chóa đèn pha vỡ (`glass_shatter` chồng lấn `IoU > 0.20` với cụm đèn hoặc nằm ở chóa đèn đầu xe) bắt buộc gán về `đèn xe`, cấm đè nhãn `kính lái`.
   - **Headlamp-Grille Anchor Lock**: Khi phát hiện đèn pha (`lamp_broken`/`den_xe`) hoặc lưới tản nhiệt (`grille`), khóa cứng 100% các vết hư hại tôn/nhựa phía dưới và ôm quanh cụm đèn về `cản trước` (`can_truoc`).
   - **Rear Bumper Seam Heuristic**: Ở góc chụp hông/đuôi xe có bánh xe bên trái (`x < 0.35`), tự động phân tách vết trầy trên vòm lốp về `vè che bánh` (`ve_che_banh`) và vết trầy phía sau khe nối về `cản sau` (`can_sau`).
   - **Loại bỏ HoughCircles giả**: Chỉ công nhận bánh xe (`wheel`) khi có mask AI hợp lệ hoặc vùng cao su lốp đen thực tế sát mặt đất.

---

## 2. Yêu cầu hệ thống & Cài đặt môi trường

### 2.1. Yêu cầu phần cứng & phần mềm
- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux (Ubuntu 20.04+).
- **Python**: `>= 3.10` (Khuyến nghị Python 3.11 – 3.13).
- **GPU (Khuyến nghị)**: NVIDIA GPU hỗ trợ CUDA (ví dụ RTX 3050/4050 trở lên) để đạt tốc độ suy luận `< 150ms/ảnh`. Hệ thống tự động fallback về CPU nếu không có GPU.

### 2.2. Cài đặt thư viện
Mở terminal (PowerShell / Bash) tại thư mục gốc `AI_CAR`:

```bash
# 1. Tạo và kích hoạt môi trường ảo (khuyến nghị)
python -m venv .venv
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên macOS / Linux:
# source .venv/bin/activate

# 2. Cài đặt các gói phụ thuộc
pip install -r ai/requirements.txt
```

---

## 3. Chạy Suy luận Two-Stage Pipeline (Inference)

### 3.1. Chạy trình diễn trên tập ảnh mẫu (`random_demo.py`)
Sử dụng script `random_demo.py` ở thư mục gốc để chạy song song `damage_best.pt` + `parts_seg.pt` trên 8 ảnh trong `demo/input/` và xuất ảnh kết quả kèm file JSON vào `demo/results/`:

```bash
python random_demo.py
```

### 3.2. Sử dụng `TwoStagePipeline` trực tiếp trong Python

```python
from ai.inference.pipeline import TwoStagePipeline

pipeline = TwoStagePipeline(
    damage_model_path="ai/models/weights/damage_best.pt",
    parts_model_path="ai/models/weights/parts_seg.pt",
    damage_conf=0.25,
    parts_conf=0.30,
    device="auto",  # Tự động chọn GPU (CUDA/MPS) hoặc CPU
)

response = pipeline.run("demo/input/image copy 4.png")
print(response.to_dict())
```

---

## 4. Khởi chạy API Server & Giao diện Web (FastAPI)

Khởi động máy chủ FastAPI phục vụ suy luận và giao diện trực quan:

```bash
python -m uvicorn ai.inference.api:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi server khởi chạy:
- **Giao diện Web Demo**: Truy cập `http://localhost:8000/`
- **Swagger UI (Tài liệu API tương tác)**: Truy cập `http://localhost:8000/docs`
- **Health Check Endpoint**: `GET http://localhost:8000/health`
- **Damage Detection Endpoint (v1)**: `POST http://localhost:8000/api/v1/damage/detect`

Ví dụ gọi API bằng `curl`:

```bash
curl -X POST "http://localhost:8000/api/v1/damage/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ai/data/yolo/images/val/001541.jpg"
```

---

## 5. Huấn luyện lại Mô hình (Retrain) & Kiểm toán Dữ liệu (Dataset Audit)

### 5.1. Huấn luyện Mô hình Bộ phận xe (Stage 2 - 7 Classes)
Tập dữ liệu bộ phận xe được cấu hình tại `ai/configs/parts_seg.yaml` (`ai/data/parts`). Để huấn luyện lại và tự động cập nhật trọng số `ai/models/parts_seg.pt`:

```bash
python ai/scripts/train_parts_model.py
```

### 5.2. Huấn luyện Mô hình Hư hại xe (Stage 1 - 6 Classes CarDD)
Tập dữ liệu hư hại được cấu hình tại `ai/configs/cardd_seg.yaml` (`ai/data/yolo`). Để huấn luyện mô hình nhận diện hư hại:

```bash
python ai/train/train_damage.py \
  --data ai/configs/cardd_seg.yaml \
  --model yolov8s-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### 5.3. Kiểm toán Dữ liệu & Đánh giá Batch (Dataset Audit)
- Kiểm tra tính toàn vẹn của nhãn phân đoạn bộ phận xe (7 classes `0..6`):
  ```bash
  python ai/scripts/audit_dataset.py
  ```
- Chạy đánh giá tổng quát hóa trên tập ảnh Validation ngẫu nhiên:
  ```bash
  python ai/scripts/test_extended_batch.py
  ```

---

## 6. Chạy Bộ Kiểm thử Tự động (Unit & Integration Tests)

Hệ thống đi kèm bộ 66 unit tests kiểm tra toàn diện hình học polygon, tiền xử lý, hậu xử lý, mức độ nghiêm trọng, API endpoints và toàn bộ quy tắc vật lý của Context Pipeline (`test_pipeline_unit.py`):

```bash
python -m unittest discover -s ai/tests -v
```

Tất cả 66 bài kiểm thử phải đạt trạng thái `OK` (100% PASS) trước khi đóng gói hoặc triển khai.

---

## 7. Cấu trúc Thư mục Dự án

```text
ai/
+-- requirements.txt
+-- requirements-dev.txt
+-- runtime.py
+-- api/
|   +-- main.py
|   +-- schemas.py
|   \-- __init__.py
+-- configs/
|   +-- classes.yaml
|   +-- damage.yaml
|   +-- data.yaml
|   +-- data_parts.yaml
|   \-- parts.yaml
+-- inference/
|   +-- pipeline.py
|   \-- change_detection.py
+-- models/
|   +-- weights/
|   |   +-- damage_best.pt
|   |   \-- parts_seg.pt
|   +-- yolov8n.pt
|   +-- yolov8s.pt
|   \-- yolov8s-seg.pt
+-- reports/                     (Sẵn sàng cho việc xuất kết quả mới)
+-- scripts/
|   +-- coco_to_yolo.py
|   +-- evaluate_damage.py
|   +-- finalize_damage.py
|   +-- train_both_stages.py
|   +-- train_parts_model.py
|   \-- archive_tools/           (Thư mục lưu trữ các công cụ dữ liệu lịch sử)
+-- tests/
|   \-- test_pipeline_unit.py    (Bộ test suite thống nhất duy nhất)
\-- train/
    +-- train_damage.py
    \-- train_parts.py
```
