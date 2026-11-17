# 01 - BÁO CÁO KIỂM ĐỊNH TỔNG THỂ ĐỘ CHÍNH XÁC AI (AI ACCURACY AUDIT)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Người thực hiện:** Senior AI Architect / Computer Vision QA  
**Phạm vi:** Mô hình Damage Detection (`damage_best.pt`), Parts Segmentation (`parts_seg.pt`) và Giải thuật Ánh xạ Hư hỏng - Bộ phận.

---

## 1. Phương Pháp Luận & Nguyên Tắc Kiểm Định

### 1.1. Phân biệt Rạch ròi giữa Unit Test và Độ Chính Xác AI
Trong các Sprint trước, toàn bộ **46 test cases** trong `ai/tests/test_pipeline_unit.py` đều đạt kết quả **PASS (100%)**. Tuy nhiên, qua kiểm toán kiến trúc độc lập, chúng tôi khẳng định:
> **CẢNH BÁO KIẾN TRÚC:** Unit Test PASS **KHÔNG PHẢI** là bằng chứng cho thấy mô hình AI hoạt động chính xác.

- **Bản chất của bộ Unit Test hiện tại**: Toàn bộ 46 tests chỉ kiểm tra các hàm xử lý hình học giả lập (synthetic mocks) như `calculate_bbox_iou`, thuật toán NMS tọa độ cố định, và logic câu lệnh điều kiện `if/else` của bộ quy tắc ngữ cảnh.
- **Độ chính xác AI thực tế**: Phải được đo lường bằng các chỉ số định lượng ($mAP@50$, $mAP@50-95$, $Precision$, $Recall$, $mIoU$) trên tập kiểm định độc lập (Ground Truth Validation/Test Set) chứa hình ảnh xe ô tô thực tế.

---

## 2. Tổng Hợp Chỉ Số Hiệu Năng Thực Tế Của Hai Mô Hình

Dựa trên dữ liệu huấn luyện lịch sử được ghi nhận trong các tệp nhật ký `ai/runs/damage/cardd_detect_30epochs/results.csv` và `ai/runs/parts/parts_yolov8s_seg_800/results.csv`:

| Tiêu chí Đánh giá | Stage 1: Damage Detection (`damage_best.pt`) | Stage 2: Parts Segmentation (`parts_seg.pt`) | Nhận xét Chuyên môn |
| :--- | :---: | :---: | :--- |
| **Kiến trúc mạng** | YOLOv8n (Object Detection) | YOLOv8s-seg (Instance Segmentation) | Kiến trúc chuẩn của Ultralytics |
| **Kích thước mô hình** | 6.19 MB (3.13M tham số) | 22.73 MB (11.78M tham số) | Nhẹ, phù hợp triển khai Edge/Server |
| **Tập dữ liệu huấn luyện** | CarDD (2,876 train / 810 val) | Parts Custom (1,525 train / 97 val) | Chênh lệch lớn về quy mô dữ liệu |
| **Precision (Hộp bao - B)** | **72.88%** | **41.42%** | Stage 2 có tỷ lệ báo động giả cao ($>58\%$) |
| **Recall (Hộp bao - B)** | **67.82%** | **42.94%** | Stage 2 bỏ sót hơn $57\%$ bộ phận xe |
| **mAP@50 (Hộp bao - B)** | **70.94%** | **37.81%** | Stage 1 đạt mức khá, Stage 2 yếu |
| **mAP@50-95 (Hộp bao - B)**| **55.96%** | **18.82%** | Tọa độ Stage 2 định vị thiếu chính xác |
| **Mask Precision (Mặt nạ - M)**| *Không áp dụng (Bbox only)* | **37.90%** | Biên mặt nạ bộ phận lẹm ngoài thực tế |
| **Mask Recall (Mặt nạ - M)** | *Không áp dụng* | **38.22%** | Bỏ sót phần lớn diện tích bề mặt |
| **Mask mAP@50 (M)** | *Không áp dụng* | **28.34%** | **Điểm nghẽn cốt lõi của toàn hệ thống** |
| **Mask mAP@50-95 (M)** | *Không áp dụng* | **13.10%** | Chất lượng phân đoạn rất thô |

---

## 3. Phát Hiện Trọng Yếu: "Bù Đắp Quy Tắc" (Rule-Engine Compensation)

Chỉ số **Mask mAP@50 chỉ đạt 28.34%** của mô hình phân đoạn bộ phận là nguyên nhân trực tiếp dẫn tới hiện tượng:
1. Tệp [pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py) đã bị phình to bất thường lên tới **2,958 dòng mã**.
2. Đội ngũ phát triển đã phải viết hàng chục hàm lọc heuristics phức tạp (`filter_parts_with_contextual_rules`, `_calculate_hsv_red_ratio`, `detect_micro_features`, `resolve_four_side_doors`, `infer_part_from_spatial_coordinates`) để "chữa cháy" cho việc mô hình AI phân đoạn sai hoặc bỏ sót bộ phận.
3. Việc lạm dụng rule-based engine để vá lỗi cho model tạo ra sự giòn gãy (brittleness): một quy tắc sửa được góc chụp này sẽ gây lỗi ở góc chụp khác.

---

## 4. Checklist Thẩm Định Thủ Công Cho Chuyên Viên & Giảng Viên

Do bộ dữ liệu demo gồm 8 ảnh trong `demo/input/` hiện chưa có nhãn Ground Truth (file annotation `.json` hoặc `.txt` độc lập đi kèm), bảng checklist sau đây được thiết kế để con người thẩm định trực quan:

| Ảnh Demo | Vết Hư Hỏng Thực Tế | Bộ Phận Thực Tế | AI Dự Đoán Hư Hỏng | AI Phân Đoạn Bộ Phận | Mapping AI | Đánh Giá Độ Chính Xác |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `image copy 3.png` | Đèn vỡ, xước cản/thân, móp capo | Đèn pha, Cản trước, Capo | `lamp_broken`, `scratch`, `dent`, `crack` | Chỉ thấy `Nắp capo` | Đèn $\rightarrow$ Đèn xe; Xước $\rightarrow$ `khong_xac_dinh` | **Một phần**: Bỏ sót cản trước, kích hoạt `needs_review`. |
| `image copy 4.png` | Móp cửa trước bên phụ | Cửa trước, Cửa sau, Vè | `dent` (conf 0.87) | Cửa trước, Nắp capo, Kính, Vè | `dent` $\rightarrow$ `Cửa trước` | **Chính xác (Đạt)**: Bắt đúng vết và đúng cửa. |
| `image copy 5.png` | Không có tổn thất | Thân xe / Vè | Không có detection | Vè che bánh | 0 tổn thất | **Chính xác (Đạt)**: Không báo động giả. |
| `image copy 6.png` | Vết nứt cận cảnh | Cận cảnh góc đèn/cản | `crack` (conf 0.76) | Bắt nhầm `Cản trước` | `crack` $\rightarrow$ `khong_xac_dinh` | **An toàn**: Rule cận cảnh chặn gán nhãn sai. |
| `image copy 7.png` | Cụm xước dài cửa trước | Cửa trước | 3 vết `scratch` | Cửa trước, Capo, Kính | 3 xước $\rightarrow$ `Cửa trước` | **Chính xác (Đạt)**: Bắt trọn cụm xước. |
| `image copy 8.png` | Vết xước cận cảnh | Góc vè/hông | `scratch` (conf 0.59) | Vè che bánh | `scratch` $\rightarrow$ `khong_xac_dinh` | **An toàn**: Thuật toán từ chối ép gán. |
| `image copy.png` | Đèn pha trái vỡ nứt | Cụm đèn trước | `lamp_broken` (conf 0.89) | Bỏ sót đèn, chỉ thấy Capo & Kính | `lamp_broken` $\rightarrow$ `Đèn xe` | **Đạt nhờ Rule**: Hard-rule cứu lỗi model seg. |
| `image.png` | 2 vết móp cận cảnh | Bề mặt kim loại hẹp | 2 vết `dent` (0.55, 0.43) | Bắt nhầm `Cản trước` | 2 dent $\rightarrow$ `khong_xac_dinh` | **An toàn**: Không ép gán nhãn bừa bãi. |

---

## 5. Kết Luận Giai Đoạn Kiểm Định Độ Chính Xác AI
- **Mô hình Damage Detection (`damage_best.pt`)**: Đạt yêu cầu cơ bản ($mAP@50 = 70.94\%$), khả năng phát hiện vết móp, vết xước và vỡ đèn khá nhạy. Điểm yếu là kích thước bounding box đôi khi bao trùm quá rộng vùng xung quanh.
- **Mô hình Parts Segmentation (`parts_seg.pt`)**: **Dưới mức kỳ vọng triển khai thực tế** ($mAP@50 = 28.34\%$). Mô hình thường xuyên bỏ sót đèn xe, cản trước khi chụp chéo, và không phân biệt được ranh giới giữa 2 cửa hông liền kề.
- **Hệ thống Ánh xạ (Mapping)**: Nhờ có các "chốt chặn an toàn" (Safety Guards), hệ thống không bị lỗi ngụy tạo kết quả (hallucination) mà chủ động trả về `khong_xac_dinh` hoặc `needs_review` khi không đủ bằng chứng.
