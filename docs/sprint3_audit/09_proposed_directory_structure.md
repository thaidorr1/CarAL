# 09 - ĐỀ XUẤT CẤU TRÚC THƯ MỤC TỐI ƯU (PROPOSED DIRECTORY STRUCTURE)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu:** Thiết kế lại cây thư mục tinh gọn, dễ bảo trì, đạt chuẩn công nghiệp và thân thiện khi thuyết trình tốt nghiệp với Giảng viên / Hội đồng.

---

## 1. Cấu Trúc Đề Xuất Tinh Gọn (Target Architecture)

```
AutoClaim-AI/
│
├── .gitignore                     # Cấu hình chặn file rác, dataset và cache
├── README.md                      # Tài liệu tổng quan, kiến trúc và cách chạy nhanh
├── requirements.txt               # Thư viện phụ thuộc chuẩn nhất (đã chuẩn hóa version)
│
├── ai/                            # Toàn bộ mã nguồn cốt lõi AI
│   ├── __init__.py
│   ├── configs/                   # Chỉ giữ 2 tệp cấu hình chuẩn hóa duy nhất
│   │   ├── damage.yaml            # Cấu hình 5 lớp hư hỏng (Stage 1)
│   │   └── parts.yaml             # Cấu hình 8 lớp bộ phận xe (Stage 2)
│   │
│   ├── inference/                 # Bẻ nhỏ 'pipeline.py' monolithic thành các module con
│   │   ├── __init__.py
│   │   ├── pipeline.py            # [Tinh gọn < 250 dòng] Điều phối luồng TwoStagePipeline
│   │   ├── change_detection.py    # So sánh Before/After qua Homography & IoU
│   │   ├── geometry.py            # Tính toán tâm, polygon overlap, IoU và khoảng cách
│   │   ├── rules.py               # Luật an toàn vật lý, hướng đầu xe, shot-type guard
│   │   ├── mapper.py              # Thuật toán ghép cặp hư hỏng vào bộ phận
│   │   └── visualizer.py          # Render nhãn Unicode tiếng Việt và side-by-side
│   │
│   ├── models/                    # Quản lý trọng số
│   │   └── weights/               # Trọng số chính thức phục vụ suy luận
│   │       ├── damage_best.pt / .onnx
│   │       └── parts_seg.pt / .onnx
│   │
│   ├── api/                       # REST API Gateway (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py                # Server FastAPI với endpoints (/predict, /change-detection)
│   │   └── schemas.py             # Pydantic schemas chuẩn hóa dữ liệu
│   │
│   ├── train/                     # Tập trung toàn bộ script huấn luyện
│   │   ├── train_damage.py        # Huấn luyện mô hình phát hiện tổn thất
│   │   └── train_parts.py         # Huấn luyện mô hình phân đoạn bộ phận
│   │
│   ├── evaluation/                # Phân hệ đánh giá định lượng AI
│   │   ├── evaluate_damage.py     # Tính mAP@50-95 cho mô hình Damage
│   │   ├── evaluate_parts.py      # Tính Mask mAP/mIoU cho mô hình Parts
│   │   └── benchmark_latency.py   # Đo độ trễ suy luận PyTorch vs ONNX
│   │
│   ├── data/                      # Dữ liệu phục vụ huấn luyện (được Gitignore)
│   │   ├── yolo/                  # Dữ liệu CarDD
│   │   └── parts/                 # Dữ liệu phân đoạn bộ phận
│   │
│   └── tests/                     # Bộ kiểm thử chia theo tầng rõ rệt
│       ├── test_pipeline_unit.py  # Unit tests cho các hàm toán học & rules
│       ├── test_change_detection.py# Unit tests cho module Before/After
│       ├── test_api.py            # Integration tests cho FastAPI endpoints
│       └── test_model_sanity.py   # Sanity tests nạp trọng số thực tế
│
├── demo/                          # Khu vực trình diễn trực quan cho Giảng viên / Khách hàng
│   ├── input/                     # 8 ảnh mẫu đầu vào
│   ├── results/                   # Kết quả ảnh vẽ nhãn và JSON predictions
│   └── run_demo.py                # [Đổi tên từ random_demo.py] Entrypoint 1-click chạy demo
│
└── docs/                          # Kho lưu trữ tài liệu báo cáo & kiểm toán
    ├── C1SE_79_SRS_AutoClaimAI_v1.0.docx
    ├── PROPOSAL_Ver2.docx
    ├── HUONG_DAN_CHAY.md
    └── sprint3_audit/             # Trọn bộ 12 báo cáo kiểm toán Sprint 3
```

---

## 2. So Sánh Trước và Sau Tinh Gọn

| Tiêu chí | Cấu trúc Hiện tại (Trước Audit) | Cấu trúc Đề xuất (Sau Audit) | Lợi ích Đem lại |
| :--- | :--- | :--- | :--- |
| **Độ dài `pipeline.py`** | 2,958 dòng (God file) | $< 250$ dòng (Chỉ điều phối) | Đọc hiểu và debug cực kỳ dễ dàng. |
| **Số file configs** | 6 file trùng lặp, mâu thuẫn | 2 file rõ ràng (`damage`, `parts`) | Chấm dứt hiện tượng config drift. |
| **Thư mục rác/tạm** | Chứa `scratch/`, `_backup_config_code/` (117 MB), `archive_tools/` | Loại bỏ/Lưu trữ riêng biệt | Giảm hơn 150 MB rác khỏi workspace. |
| **Phân hệ Evaluation** | Phân tán trong `scripts/` và `scratch/` | Tập trung tại `ai/evaluation/` | Giảng viên dễ dàng thẩm định độ chính xác AI. |
| **Entrypoint Demo** | `random_demo.py` ở thư mục gốc | `demo/run_demo.py` | Đúng chuẩn quy ước thiết kế demo. |
| **Testing Coverage** | Chỉ có 1 file unit test đơn lẻ | Phân tách Unit, API và Model Sanity | Đảm bảo chất lượng toàn diện trước khi bàn giao. |
