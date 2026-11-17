# 11 - ROADMAP HÀNH ĐỘNG SPRINT 4 & SPRINT 5 (ACTIONABLE ROADMAP)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu:** Kế hoạch hành động cụ thể, phân kỳ theo thứ tự ưu tiên nhằm khắc phục triệt để các tồn đọng phát hiện trong đợt kiểm toán.

---

## 1. Nguyên Tắc Ưu Tiên Hành Động

> **NGUYÊN TẮC CỐT LÕI:** "Độ chính xác AI và Sự ổn định Pipeline là số 1. Tinh gọn cấu trúc mã nguồn là số 2. Không tối ưu hóa những phần chưa bị lỗi."

Toàn bộ lộ trình được thiết kế theo 7 bước tuần tự:

```
┌─────────────────────────────────────────────────────────────────┐
│ Bước 1: Tạo Bộ Ground Truth & Đo Lường Định Lượng AI             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 2: Huấn Luyện Nâng Cấp Model Parts Seg (Thêm Đèn, Tách Cửa)│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 3: Đơn Giản Hóa Giải Thuật Ánh Xạ (Giảm bớt Heuristics)    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 4: Tái Cấu Trúc 'pipeline.py' Thành 4 Module Con           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 5: Loại Bỏ & Lưu Trữ Các File Thừa (Sau khi User Phê Duyệt)│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 6: Bổ Sung Test Hồi Quy & Integration Test Cho REST API    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Bước 7: Hoàn Thiện Demo 1-Click Chuẩn Bị Báo Cáo Capstone       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Kế Hoạch Chi Tiết Từng Bước

### Giai Đoạn 1: Cải Thiện AI & Dữ Liệu (Trọng tâm Sprint 4)

#### Bước 1: Xây dựng Bộ Ground Truth cho 8 Ảnh Demo
- **Công việc:** Tạo file `demo/annotations_ground_truth.json` gán nhãn thủ công chuẩn xác cho 8 ảnh `demo/input/`.
- **Mục tiêu:** Cho phép tự động tính ma trận nhầm lẫn (Confusion Matrix), Precision, Recall và IoU thực tế của pipeline trên bộ ảnh demo sau mỗi lần cập nhật.

#### Bước 2: Huấn Luyện Lại Mô Hình Phân Đoạn Bộ Phận (`parts_seg`)
- **Vấn đề cần giải quyết:** Khắc phục tình trạng Mask mAP@50 chỉ đạt $28.34\%$.
- **Hành động cụ thể:**
  1. Tách lớp `cua_hong` thành `cua_truoc` và `cua_sau` trong dataset `ai/data/parts`.
  2. Bổ sung nhãn `den_xe` (Đèn pha / Đèn hậu) vào dữ liệu huấn luyện.
  3. Huấn luyện lại với YOLOv8s-seg trong 100 epochs với Data Augmentation (Mosaic, Mixup).
- **KPI mục tiêu:** Đạt **Mask mAP@50 $\ge 60.0\%$**.

#### Bước 3: Tinh Giản Khâu Ánh Xạ (Refactoring Mapping Engine)
- Khi mô hình Parts Segmentation đã tự nhận diện được Đèn xe và phân biệt 2 cửa hông:
  - Loại bỏ các hàm heuristic phức tạp: `_calculate_hsv_red_ratio`, `detect_micro_features`, `resolve_four_side_doors`.
  - Thuật toán `map_damages_to_parts()` chuyển về cơ chế chuẩn: $90\%$ dựa vào Mask Overlap IoU thuần túy, $10\%$ dựa vào cự ly tâm và chốt chặn an toàn cận cảnh.

---

### Giai Đoạn 2: Tái Cấu Trúc Mã Nguồn & Dọn Dẹp (Trọng tâm Sprint 5)

#### Bước 4: Module hóa Tệp `pipeline.py` (2,958 dòng)
- Tách thành 4 module con độc lập trong `ai/inference/`:
  - `geometry.py`: Chứa các hàm toán học thuần túy (`calculate_center`, `calculate_polygon_overlap_and_iou`, `_bbox_to_polygon`).
  - `rules.py`: Chứa các bộ lọc an toàn (`classify_shot_type`, `filter_parts_with_contextual_rules`).
  - `mapper.py`: Chứa logic ghép cặp (`map_damages_to_parts`).
  - `visualizer.py`: Chứa logic vẽ nhãn tiếng Việt (`draw_annotations`, `_get_unicode_font`).
  - `pipeline.py`: Giữ lại lớp `TwoStagePipeline` tinh gọn dưới 250 dòng đóng vai trò Facade / Orchestrator.
- **Tiêu chuẩn nghiệm thu:** Chạy `pytest ai/tests/test_pipeline_unit.py -v` đạt 46/46 passed mà không cần sửa bất kỳ test case nào.

#### Bước 5: Dọn dẹp Thành phần Thừa (Sau khi có Phê Duyệt từ User)
- Di chuyển `_backup_config_code/` (117 MB) ra khỏi workspace Git hoặc nén thành `.zip` lưu trữ riêng.
- Xóa bỏ thư mục `ai/scripts/archive_tools/` (22 file thừa).
- Dọn sạch thư mục rỗng `ai/data/parts_verified/`.
- Xóa 2 file thừa `ai/requirements.txt` và `ai/requirements-dev.txt`.
- Gộp các file cấu hình phân mảnh thành 2 file duy nhất: `ai/configs/damage.yaml` và `ai/configs/parts.yaml`.

#### Bước 6: Hoàn Thiện Kiểm Thử Tích Hợp
- Viết tệp `ai/tests/test_api.py` kiểm thử toàn diện các endpoint FastAPI trong `ai/api/main.py`.
- Viết tệp `ai/tests/test_model_sanity.py` kiểm tra nạp trọng số thực tế trên cả 2 backend PyTorch và ONNX.

#### Bước 7: Chuẩn Hóa Trình Diễn Đồ Án (Capstone Showcase)
- Đổi tên `random_demo.py` thành `demo/run_demo.py`.
- Tích hợp giao diện FastAPI Swagger UI (`http://127.0.0.1:8000/docs`) sẵn sàng để demo trực tiếp với Giảng viên và Hội đồng chấm điểm.
