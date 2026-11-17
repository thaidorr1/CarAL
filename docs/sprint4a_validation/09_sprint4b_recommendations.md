# 09 - ĐỀ XUẤT HÀNH ĐỘNG CHO SPRINT 4B (SPRINT 4B RECOMMENDATIONS)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Mục tiêu:** Định hướng hành động kỹ thuật chuẩn xác, ngăn ngừa rủi ro lãng phí tài nguyên tính toán trước khi bắt đầu cải thiện AI.

---

## 1. Cảnh Báo "Đỏ" Trước Khi Bắt Đầu Sprint 4B

> **CẢNH BÁO QUAN TRỌNG NHẤT:**  
> **TUYỆT ĐỐI KHÔNG HUẤN LUYỆN LẠI MÔ HÌNH `parts_seg.pt` TRÊN TẬP DỮ LIỆU HIỆN TẠI (`ai/data/parts`)!**  
> Việc huấn luyện lại trên tập dữ liệu này sẽ hoàn toàn vô ích vì $75.6\%$ nhãn trong đó vốn là hộp chữ nhật 4 đỉnh giả đa giác. Mạng nơ-ron dù train 100 hay 200 epochs cũng chỉ học cách sinh ra các mặt nạ hình hộp chữ nhật!

---

## 2. Danh Mục 5 Hành Động Ưu Tiên Cần Thực Hiện Ở Sprint 4B

### Ưu tiên 1: Chuẩn Bị Bộ Dữ Liệu Phân Đoạn Bộ Phận Thật (True Polygon Dataset)
- Thu thập lại hoặc làm sạch tập dữ liệu nhãn đa giác cho bộ phận xe:
  - Tối thiểu 1,000 ảnh có đa giác vẽ tay thật ôm sát mép cửa, cản, nắp capo và vè bánh (tối thiểu 15 - 30 điểm đỉnh/polygon).
  - Bổ sung dứt điểm lớp `den_xe` (Đèn pha và Đèn hậu).
  - Tách nhãn `cua_hong` thành `cua_truoc` và `cua_sau`.

### Ưu tiên 2: Xây Dựng File Ground Truth Cho 8 Ảnh Demo
- Tạo tệp `demo/annotations_ground_truth.json` theo đúng cấu trúc Schema đã thiết kế trong báo cáo `06_ground_truth_design.md`.
- Nhờ 2 thành viên trong nhóm giám định độc lập để thẩm định tính khách quan của nhãn trước khi đưa vào kiểm thử hồi quy.

### Ưu tiên 3: Xây Dựng Kịch Bản Đánh Giá Hồi Quy Tự Động (`ai/evaluation/eval_regression.py`)
- Viết 1 script chạy đánh giá độc lập so sánh kết quả suy luận của `TwoStagePipeline` với file `demo/annotations_ground_truth.json`.
- Script tự động tính và in ra:
  - Số lượng tổn thất phát hiện đúng (TP), sai (FP), bỏ sót (FN).
  - Tỷ lệ gán đúng bộ phận (Mapping Accuracy).
  - Tỷ lệ kích hoạt đúng chốt chặn an toàn (Safety Rejection Rate).

### Ưu tiên 4: Đồng Bộ Hóa Cấu Hình Lớp (Config Consolidation)
- Hợp nhất 6 tệp cấu hình phân mảnh thành 2 tệp duy nhất:
  - `ai/configs/damage.yaml`: Cấu hình 6 lớp CarDD gốc + bản ánh xạ 5 nhóm Proposal.
  - `ai/configs/parts.yaml`: Cấu hình 8 lớp bộ phận ngoại thất xe chuẩn.

### Ưu tiên 5: Thu Thập Dữ Liệu Bổ Sung Cho 2 Lớp Tổn Thất Còn Thiếu
- Tìm kiếm từ các nguồn dữ liệu mở (Roboflow Car Damage Dataset, VOC-Car) khoảng $300 - 500$ ảnh cho hai dạng tổn thất:
  - `tear_puncture`: Rách thủng cản xe hoặc tai xe.
  - `paint_chipping`: Tróc mảng sơn kim loại.
- Bổ sung vào tập `ai/data/yolo/` trước khi tiến hành huấn luyện mô hình Damage v2.
