# 05 - KIỂM TOÁN CẤU TRÚC TOÀN BỘ DỰ ÁN (PROJECT STRUCTURE AUDIT)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Quét, phân tích và đánh giá chức năng của từng thư mục và tệp tin trong toàn bộ workspace.

---

## 1. Sơ Đồ Cấu Trúc Thư Mục Hiện Tại

```
D:\AI_CAR\
├── .git/                      # Git repository (Base commit 5694599)
├── .pytest_cache/             # Pytest cache cục bộ
├── .vscode/                   # Cấu hình IDE VS Code (launch.json, settings.json)
├── README.md                  # Hướng dẫn tổng quan dự án (5.6 KB)
├── requirements.txt           # Danh mục thư viện toàn dự án (18 dòng)
├── random_demo.py             # Script chạy demo 8 ảnh chính thức (3.0 KB)
│
├── ai/                        # Gốc mô-đun AI của dự án
│   ├── __init__.py            # Khởi tạo package
│   ├── requirements.txt       # [TRÙNG LẶP] Phiên bản cũ hơn của requirements
│   ├── requirements-dev.txt   # [DƯ THỪA] Chỉ chứa 2 dòng công cụ dev
│   ├── api/                   # REST API Gateway (FastAPI)
│   │   ├── main.py            # Entrypoint FastAPI server (13.6 KB)
│   │   └── schemas.py         # Pydantic models cho request/response
│   ├── configs/               # Chứa 6 file cấu hình và tài liệu lớp
│   │   ├── classes.md         # Tài liệu mô tả mapping CarDD (tham chiếu file cũ)
│   │   ├── classes.yaml       # Định nghĩa 5 lớp proposal & merge map
│   │   ├── damage.yaml        # [CHỒNG CHÉO] Cấu hình YOLO detect 5 lớp
│   │   ├── data.yaml          # [CHỒNG CHÉO] Cấu hình YOLO detect 6 lớp CarDD
│   │   ├── data_parts.yaml    # [CHỒNG CHÉO] Cấu hình dataset 7 lớp bộ phận
│   │   └── parts.yaml         # [CHỒNG CHÉO] Cấu hình mô hình phân đoạn 7 lớp
│   ├── data/                  # Thư mục dữ liệu huấn luyện (Tổng ~3.3 GB)
│   │   ├── yolo/              # Dataset CarDD (images/labels train, val, test)
│   │   ├── parts/             # Dataset phân đoạn bộ phận xe (1,525 train, 97 val)
│   │   ├── parts_verified/    # [RÁC / TRỐNG] Chứa 6 thư mục con 0 files
│   │   └── wild_test/         # 6 ảnh thử nghiệm thực tế (1.64 MB)
│   ├── inference/             # Lõi suy luận của hệ thống
│   │   ├── pipeline.py        # [MONOLITHIC] TwoStagePipeline (2,958 dòng, 130 KB)
│   │   └── change_detection.py# So sánh Before/After qua Homography (513 dòng)
│   ├── models/                # Trọng số mô hình
│   │   ├── yolov8n.pt         # Trọng số pre-trained gốc từ COCO (6.4 MB)
│   │   ├── yolov8s.pt         # Trọng số pre-trained gốc từ COCO (22.1 MB)
│   │   ├── yolov8s-seg.pt     # Trọng số pre-trained seg gốc từ COCO (23.4 MB)
│   │   └── weights/           # Trọng số fine-tuned chính thức của dự án
│   │       ├── damage_best.pt / .onnx   # Model phát hiện tổn thất (6.2 MB / 12.2 MB)
│   │       └── parts_seg.pt / .onnx     # Model phân đoạn bộ phận (22.7 MB / 45.2 MB)
│   ├── reports/               # Chứa 19 báo cáo thử nghiệm lịch sử & ảnh tạm
│   ├── runs/                  # Nhật ký huấn luyện Ultralytics (damage & parts)
│   ├── scripts/               # Các script huấn luyện, xuất mô hình và lưu trữ
│   │   ├── coco_to_yolo.py    # Script chuyển đổi format dữ liệu
│   │   ├── evaluate_damage.py # Đánh giá mô hình damage
│   │   ├── eval_phase2.py     # Script tính mAP Phase 2
│   │   ├── export_onnx.py     # Xuất mô hình sang định dạng ONNX
│   │   ├── finalize_damage.py # Script kiểm định cuối mô hình damage
│   │   ├── runtime.py         # [LEGACY] Script suy luận cũ
│   │   ├── train_damage.py    # Script train stage 1
│   │   ├── train_parts.py     # Script train stage 2
│   │   ├── train_both_stages.py   # Script train kết hợp
│   │   ├── train_parts_model.py   # Script train parts độc lập
│   │   └── archive_tools/     # [DƯ THỪA] Chứa 22 script tạm/thử nghiệm cũ
│   └── tests/                 # Bộ kiểm thử
│       └── test_pipeline_unit.py  # [MONOLITHIC] 46 unit test cases (1,445 dòng)
│
├── demo/                      # Thư mục trình diễn
│   ├── input/                 # 8 ảnh mẫu đầu vào (`.png`)
│   └── results/               # Kết quả xuất tự động (ảnh vẽ nhãn & JSON)
│
├── docs/                      # Tài liệu dự án
│   ├── C1SE_79_SRS_AutoClaimAI_v1.0.docx  # Tài liệu SRS chính thức (12.4 MB)
│   ├── PROPOSAL_Ver2.docx                 # Đề xuất Capstone Proposal (0.46 MB)
│   ├── HUONG_DAN_CHAY.md                  # Hướng dẫn cài đặt & vận hành
│   ├── GRAPH_REPORT.md                    # Báo cáo Knowledge Graph từ Graphify
│   ├── sprint2_ai_core_report.md          # Báo cáo kỹ thuật Sprint 2
│   ├── sprint2_change_audit.md            # Báo cáo kiểm toán Sprint 2
│   └── references/                        # Tài liệu đặc tả kỹ thuật và file word
│
├── scratch/                   # [TẠM BỢ] 19 file script audit, kiểm tra nhanh
├── _backup_config_code/       # [DƯ THỪA LỚN] 117 MB bản sao lưu toàn bộ code cũ
└── graphify-out/              # [DỮ LIỆU ĐỒ THỊ] JSON & HTML trích xuất đồ thị tri thức
```

---

## 2. Đánh Giá Hiện Trạng Từng Phân Hệ

| Phân hệ / Thư mục | Tình trạng | Mức độ nghiêm trọng | Vấn đề chính phát hiện |
| :--- | :---: | :---: | :--- |
| **Thư mục gốc** | Khá gọn | Thấp | Tồn tại 2 file `requirements.txt` (gốc và trong `ai/`). |
| **`ai/inference/`** | Khổng lồ | **RẤT CAO** | `pipeline.py` là một God-Class (2,958 dòng) gộp chung quá nhiều trách nhiệm (hình học, heuristic, rendering, I/O). |
| **`ai/configs/`** | Phân mảnh | **CAO** | 6 file cấu hình chồng chéo, mâu thuẫn giữa 5 lớp proposal và 6 lớp CarDD. |
| **`ai/scripts/`** | Rác & Trùng lặp | **CAO** | Có 4 script huấn luyện khác nhau (`train_damage`, `train_parts`, `train_both_stages`, `train_parts_model`) và 22 script con trong `archive_tools/`. |
| **`scratch/`** | Tạm bợ | Trung bình | Chứa 19 file thử nghiệm dở dang từ Sprint 1 & 2. |
| **`_backup_config_code/`** | Dư thừa | **CAO** | Bản sao lưu chiếm 117 MB trong Git workspace, tạo nguy cơ chỉnh sửa nhầm file backup thay vì file chính thức. |
| **`ai/data/parts_verified/`**| Thư mục chết | Thấp | 6 thư mục con rỗng không chứa ảnh, tốn dung lượng quản lý. |
