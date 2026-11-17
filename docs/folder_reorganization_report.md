# BÁO CÁO TÁI TỔ CHỨC CẤU TRÚC THƯ MỤC & ĐỒNG BỘ HỆ THỐNG AUTOCLAIM-AI

## 1. Cấu trúc Thư mục Trước và Sau khi Sắp xếp

### 1.1. Cấu trúc trước khi sắp xếp
```text
AI_CAR/
├── .vscode/
├── ai/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── configs/
│   ├── data/
│   ├── inference/
│   │   ├── change_detection.py
│   │   ├── pipeline.py
│   │   ├── pipeline.py.bak          # File backup tạm thời nằm lẫn trong source
│   │   └── pipeline.py.bak2         # File backup tạm thời nằm lẫn trong source
│   ├── models/
│   │   ├── damage_best.pt           # Trọng số chính nằm chung với pretrained backbones
│   │   ├── parts_seg.pt             # Trọng số chính nằm chung với pretrained backbones
│   │   ├── yolov8n.pt
│   │   ├── yolov8s.pt
│   │   └── yolov8s-seg.pt
│   ├── reports/
│   ├── runs/
│   ├── scripts/
│   ├── tests/
│   │   ├── test_pipeline_unit.py
│   │   └── test_pipeline_unit.py.bak
│   ├── train/
│   ├── requirements.txt
│   └── requirements-dev.txt
├── demo/
│   ├── input/                       # Có 8 ảnh .png kèm 5 ảnh .jpg rời rạc
│   └── results/
├── docs/
├── scratch/
├── .gitignore
├── random_demo.py
└── README.md                        # Chưa đồng bộ với cấu trúc thực tế và đường dẫn mới
```

### 1.2. Cấu trúc sau khi sắp xếp (Chuẩn hóa)
```text
AI_CAR/
├── ai/
│   ├── __init__.py                  # [MỚI] Đóng gói chuẩn Python package
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── configs/
│   ├── data/                        # Giữ nguyên toàn bộ dữ liệu huấn luyện & kiểm thử
│   ├── inference/
│   │   ├── __init__.py              # [MỚI] Export TwoStagePipeline & các lớp dữ liệu chính
│   │   ├── pipeline.py
│   │   └── change_detection.py
│   ├── models/
│   │   ├── weights/                 # [MỚI] Tách biệt trọng số chính thức của pipeline
│   │   │   ├── damage_best.pt
│   │   │   └── parts_seg.pt
│   │   ├── yolov8n.pt
│   │   ├── yolov8s.pt
│   │   └── yolov8s-seg.pt
│   ├── reports/
│   ├── runs/
│   ├── scripts/
│   ├── tests/
│   │   └── test_pipeline_unit.py
│   └── train/
├── demo/
│   ├── input/                       # Đúng 8 ảnh đầu vào chuẩn phục vụ trình diễn
│   └── results/                     # Kết quả trực quan hóa + predictions.json
├── docs/
│   ├── folder_reorganization_report.md
│   ├── HUONG_DAN_CHAY.md
│   └── references/
├── random_demo.py
├── requirements.txt                 # [MỚI] File cài đặt phụ thuộc tại thư mục gốc
├── .gitignore                       # [CẬP NHẬT] Chặn cache, .vscode, .bak và thư mục backup
└── README.md                        # [CẬP NHẬT] Đồng bộ cây thư mục & hướng dẫn mới
```

---

## 2. Danh sách File / Thư mục đã Di chuyển & Tạo mới

| Thao tác | Nguồn (`Source`) | Đích (`Destination`) | Mục đích |
| :--- | :--- | :--- | :--- |
| **Sao lưu an toàn** | Tất cả file `.py`, `.md`, `.txt`, `.yaml` | `_backup_config_code/` | Đảm bảo an toàn mã nguồn và cấu hình trước khi tái cấu trúc (không nhân bản `ai/data/` và `.pt`) |
| **Di chuyển Model 1** | `ai/models/damage_best.pt` | `ai/models/weights/damage_best.pt` | Gom trọng số chính thức Stage 1 vào `ai/models/weights/` |
| **Di chuyển Model 2** | `ai/models/parts_seg.pt` | `ai/models/weights/parts_seg.pt` | Gom trọng số chính thức Stage 2 vào `ai/models/weights/` |
| **Làm sạch `.bak`** | `ai/inference/pipeline.py.bak`, `pipeline.py.bak2`, `ai/tests/test_pipeline_unit.py.bak` | `_backup_config_code/` | Giữ thư mục `ai/inference/` và `ai/tests/` sạch sẽ theo đúng thiết kế |
| **Chuẩn hóa `demo/input/`** | 5 file `.jpg` rời (`000470.jpg`, `001139.jpg`, `001288.jpg`, `003682.jpg`, `closeup_crop_000154_v1.jpg`) | `_backup_config_code/demo_extra_jpgs/` | Giữ đúng 8 ảnh `.png` chuẩn trong `demo/input/` mà không xóa bất kỳ file nào |
| **Tạo Package Init** | *(Tạo mới)* | `ai/__init__.py`, `ai/inference/__init__.py` | Chuẩn hóa cấu trúc import module Python |
| **Tạo Requirements gốc** | *(Tạo mới)* | `requirements.txt` | Cho phép cài đặt nhanh từ thư mục gốc `pip install -r requirements.txt` |

---

## 3. Danh sách Đường dẫn Code & Tài liệu đã Cập nhật

| Tên file | Dòng cập nhật | Nội dung thay đổi |
| :--- | :--- | :--- |
| [`ai/inference/pipeline.py`](file:///d:/AI_CAR/ai/inference/pipeline.py#L2367-L2400) | Dòng `2367–2400` | Cập nhật đường dẫn mặc định của `TwoStagePipeline.__init__` sang `"ai/models/weights/damage_best.pt"` và `"ai/models/weights/parts_seg.pt"`, đồng thời hỗ trợ tự động phân giải vào `weights/` nếu caller truyền đường dẫn cũ |
| [`random_demo.py`](file:///d:/AI_CAR/random_demo.py#L28-L31) | Dòng `28–31` | Cập nhật `damage_model_path="ai/models/weights/damage_best.pt"` và `parts_model_path="ai/models/weights/parts_seg.pt"` |
| [`ai/api/main.py`](file:///d:/AI_CAR/ai/api/main.py#L81-L82) | Dòng `81–82` | Cập nhật `default_dmg` và `default_parts` trỏ vào `AI_ROOT / "models" / "weights"` |
| [`ai/scripts/eval_phase2.py`](file:///d:/AI_CAR/ai/scripts/eval_phase2.py#L21-L22) | Dòng `21–22` | Cập nhật `damage_model` và `parts_model` trỏ vào `ai/models/weights/` |
| [`ai/scripts/train_both_stages.py`](file:///d:/AI_CAR/ai/scripts/train_both_stages.py#L52-L55) | Dòng `52–55` | Cập nhật `STAGE1_EXPORT_TARGET`, `STAGE1_SEG_TARGET`, `STAGE2_EXPORT_TARGET` vào `ai/models/weights/` |
| [`ai/scripts/train_parts_model.py`](file:///d:/AI_CAR/ai/scripts/train_parts_model.py#L237-L265) | Dòng `240, 264` | Cập nhật `--export-target` và `model_eval_path` sang `AI_ROOT / "models/weights/parts_seg.pt"` |
| [`ai/train/train_parts.py`](file:///d:/AI_CAR/ai/train/train_parts.py#L78-L135) | Dòng `81, 133` | Cập nhật danh sách `candidates` và `--export-target` sang `AI_ROOT / "models/weights/parts_seg.pt"` |
| [`README.md`](file:///d:/AI_CAR/README.md#L1-L108) | Toàn bộ file (`1–108`) | Vẽ lại cây thư mục chuẩn xác, cập nhật câu lệnh chạy pytest, chạy demo và ví dụ Python với `ai/models/weights/` |
| [`docs/HUONG_DAN_CHAY.md`](file:///d:/AI_CAR/docs/HUONG_DAN_CHAY.md#L11-L188) | Dòng `11, 20, 65–88, 181–188` | Đồng bộ đường dẫn `ai/models/weights/damage_best.pt` và `ai/models/weights/parts_seg.pt` |
| [`docs/references/DATASET_AND_PIPELINE_SPEC.md`](file:///d:/AI_CAR/docs/references/DATASET_AND_PIPELINE_SPEC.md#L6-L23) | Dòng `6, 23` | Đồng bộ đường dẫn trọng số Stage 1 và Stage 2 sang `ai/models/weights/` |
| [`.gitignore`](file:///d:/AI_CAR/.gitignore#L1-L32) | Dòng `1–32` | Chặn `__pycache__/`, `.pytest_cache/`, `*.pyc`, `.vscode/`, `*.bak`, `/scratch/` và `/_backup_config_code/` |

---

## 4. Kết quả Kiểm thử sau khi Sắp xếp

### 4.1. Kết quả Unit Tests (`python -m pytest ai/tests/test_pipeline_unit.py -v`)
* **Trạng thái**: **46 / 46 PASSED (100%)** trong `1.07s`.
* Không phát sinh bất kỳ lỗi import hay lỗi đường dẫn nào.

### 4.2. Kết quả chạy Demo (`python random_demo.py`)
* Hai model `ai/models/weights/damage_best.pt` và `ai/models/weights/parts_seg.pt` nạp thành công trên thiết bị GPU (`device='0'`).
* Xử lý đủ **8/8 ảnh** trong `demo/input/` và xuất đầy đủ ảnh trực quan hóa cùng file [`demo/results/predictions.json`](file:///d:/AI_CAR/demo/results/predictions.json):
  1. `[1/8] image copy 3.png`: `scratch -> khong_xac_dinh [0.88], scratch -> khong_xac_dinh [0.58]`
  2. `[2/8] image copy 4.png`: `scratch -> khong_xac_dinh [0.79], dent -> khong_xac_dinh [0.72], scratch -> Cản sau [0.69], scratch -> Cản sau [0.51], dent -> Vè che bánh [0.38]`
  3. `[3/8] image copy 5.png`: `glass_shatter -> Kính xe [0.94]`
  4. `[4/8] image copy 6.png`: `dent -> Cửa trước [0.92]`
  5. `[5/8] image copy 7.png`: `lamp_broken -> Đèn xe [0.90], scratch -> khong_xac_dinh [0.51]`
  6. `[6/8] image copy 8.png`: *(Không có vết hư hỏng vượt ngưỡng)*
  7. `[7/8] image copy.png`: `glass_shatter -> Kính xe [0.93]`
  8. `[8/8] image.png`: `lamp_broken -> Đèn xe [0.88], crack -> khong_xac_dinh [0.85], crack -> khong_xac_dinh [0.76], scratch -> khong_xac_dinh [0.54]`

---

## 5. Các File / Thư mục Giữ lại để Bảo toàn Lịch sử Dự án

Tuân thủ nguyên tắc **không xóa các file chưa xác định công dụng hoặc phục vụ huấn luyện/kiểm toán lịch sử**, các thành phần sau được giữ nguyên vị trí:
1. **`ai/models/yolov8n.pt`, `yolov8s.pt`, `yolov8s-seg.pt`**: Các trọng số nền (pretrained backbones) được sử dụng bởi các script huấn luyện trong `ai/train/` và `ai/scripts/`.
2. **`ai/api/`, `ai/configs/`, `ai/scripts/`, `ai/train/`, `ai/runs/`, `ai/reports/`**: Các module phục vụ FastAPI Gateway, cấu hình YAML, script huấn luyện và báo cáo kiểm toán dữ liệu qua các giai đoạn (Phase G0, G1, Phase 2, Phase F).
3. **`scratch/` và `_backup_config_code/`**: Chứa các script kiểm toán tạm thời và bản sao lưu an toàn trước khi tái cấu trúc (đã được đưa vào `.gitignore` để không làm bẩn Git).
