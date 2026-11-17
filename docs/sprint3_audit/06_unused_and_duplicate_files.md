# 06 - DANH MỤC THÀNH PHẦN THỪA, TRÙNG LẶP & LỖI THỜI
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Rà soát và chỉ định chính xác các tệp tin, đoạn mã, cấu hình và thư mục không còn giá trị sử dụng trong hệ thống.

---

## 1. Các Tệp Tin Trùng Lặp Chức Năng (Duplicate Files)

### 1.1. Trùng lặp Cấu hình Môi trường (Requirements)
- **`requirements.txt` (gốc)**: Chứa đầy đủ 18 thư viện (gồm cả `supervision`, `onnx`, `onnxruntime`, `pydantic`, `graphifyy`).
- **`ai/requirements.txt`**: Chứa phiên bản cũ 9 thư viện.
- **`ai/requirements-dev.txt`**: Chỉ chứa 2 dòng (`pytest`, `httpx`).
$\Rightarrow$ **Đề xuất:** Giữ duy nhất `requirements.txt` ở thư mục gốc, xóa bỏ 2 tệp phụ trong `ai/`.

### 1.2. Trùng lặp & Xung đột Cấu hình Dataset (Config Drift)
- **Cặp 1 (Damage Dataset)**:
  - `ai/configs/damage.yaml`: Định nghĩa `nc: 5`, đường dẫn `ai/data/yolo`.
  - `ai/configs/data.yaml`: Định nghĩa `nc: 6` (CarDD gốc), cùng đường dẫn `ai/data/yolo`.
  $\Rightarrow$ **Đề xuất:** Hợp nhất thành 1 tệp `ai/configs/damage_data.yaml` chuẩn hóa 5 lớp theo Capstone Proposal.
- **Cặp 2 (Parts Dataset)**:
  - `ai/configs/parts.yaml`: Định nghĩa 7 lớp ngoại thất xe.
  - `ai/configs/data_parts.yaml`: Định nghĩa cùng 7 lớp với thông số đường dẫn tương tự.
  $\Rightarrow$ **Đề xuất:** Hợp nhất thành 1 tệp `ai/configs/parts_data.yaml`.

### 1.3. Trùng lặp Script Huấn Luyện (Training Scripts Redundancy)
Hiện dự án tồn tại 4 script huấn luyện phân tán tại hai thư mục khác nhau:
1. `ai/train/train_damage.py` (3.1 KB)
2. `ai/train/train_parts.py` (8.2 KB)
3. `ai/scripts/train_both_stages.py` (17.0 KB)
4. `ai/scripts/train_parts_model.py` (14.6 KB)
$\Rightarrow$ **Đề xuất:** Đưa toàn bộ việc huấn luyện về duy nhất thư mục `ai/train/`, chuẩn hóa thành `train_damage.py` và `train_parts.py`, loại bỏ hai script trùng lặp trong `ai/scripts/`.

---

## 2. Các Tệp Tin Không Còn Được Sử Dụng (Dead / Obsolete Code)

### 2.1. Thư mục `ai/scripts/archive_tools/` (22 tệp tin mồ côi)
Bao gồm các script dùng một lần từ các giai đoạn thử nghiệm sơ khai:
- `analyze_pipeline_reports.py`, `apply_pipeline_optimization.py`, `audit_full_dataset.py`, `audit_full_set.py`, `audit_parts_dataset.py`, `check_filenames.py`, `clean_and_standardize_parts_dataset.py`, `clean_noisy_closeup_labels.py`, `download_cardd.py`, `download_wild_test.py`, `fix_var.py`, `generate_closeup_dataset.py`, `generate_grounded_parts_dataset.py`, `mine_closeup_parts_from_cardd.py`, `patch_pipeline_logic.py`, `refine_headlamp_override.py`, `standardize_7_parts.py`, `stats_cardd.py`, `supplement_data.py`, `update_unit_tests.py`, `visualize_labels.py`.
- **Thực trạng**: Không có bất kỳ file nào trong hệ thống core hoặc test import hay gọi đến 22 scripts này.

### 2.2. Thư mục `scratch/` (19 tệp tin tạm thời)
- Chứa các script audit nhanh, kiểm tra bounding box và benchmark cũ (`audit_datasets_g0.py`, `check_bboxes.py`, `count_classes.py`, `inspect_img3.py`...).
- Ngoại trừ các script mới dùng kiểm toán gần đây, các script còn lại đều không còn giá trị vận hành.

### 2.3. Tệp tin tham chiếu chết (Dangling References)
- `ai/configs/classes.md`: Chứa hướng dẫn "Chỉ gộp/lọc khi xuất JSON trong `inference/infer.py`" $\rightarrow$ File `inference/infer.py` đã bị xóa hoàn toàn từ Base Commit.

---

## 3. Thư Mục và Trọng Số Dư Thừa Dung Lượng Lớn

| Tên Thư Mục / Tệp | Dung lượng | Lý do dư thừa | Rủi ro nếu xóa bỏ | Đề xuất |
| :--- | :---: | :--- | :---: | :--- |
| `_backup_config_code/` | **117.2 MB** | Chứa bản sao lưu tĩnh của code cũ (`pipeline.py.bak`, `test_pipeline_unit.py.bak`...). Git đã có tính năng version control đầy đủ. | **KHÔNG CÓ** (Nếu đã có Git commit) | **ARCHIVE / DELETE** (Đưa ra khỏi repo làm việc) |
| `ai/models/yolov8s-seg.pt` | 23.4 MB | Checkpoint COCO gốc chưa fine-tune. | Thấp (Chỉ cần khi train lại từ đầu) | Giữ tại cloud storage hoặc `data/checkpoints/` |
| `ai/models/yolov8s.pt` | 22.1 MB | Checkpoint COCO gốc chưa fine-tune. | Thấp | Lưu trữ ngoài repo |
| `ai/models/yolov8n.pt` | 6.4 MB | Checkpoint COCO gốc chưa fine-tune. | Thấp | Lưu trữ ngoài repo |
| `ai/data/parts_verified/` | 2.5 MB | 6 thư mục con trống (`train`, `val`, `test` = 0 files), chỉ có 3 file audit cũ. | **KHÔNG CÓ** | **DELETE** |
