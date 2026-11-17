# 07 - DANH MỤC CÁC THÀNH PHẦN CÒN THIẾU (MISSING COMPONENTS)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Nhận diện các lỗ hổng kỹ thuật, thành phần kiểm định và tài liệu còn thiếu để đưa dự án lên chuẩn chất lượng đồ án tốt nghiệp xuất sắc (Capstone Ready).

---

## 1. Thiếu Sót Trong Phân Hệ Suy Luận (Inference Architecture Gaps)

### 1.1. Thiếu Tách Bạch Trách Nhiệm (Separation of Concerns)
Hiện tại toàn bộ logic suy luận đang bị nhồi nhét vào duy nhất tệp [ai/inference/pipeline.py](file:///d:/AI_CAR/ai/inference/pipeline.py) (2,958 dòng). Để đạt chuẩn SOLID và Clean Architecture, hệ thống đang **THIẾU** các mô-đun chuyên biệt:
- **`ai/inference/geometry.py`**: Chuyên trách tính toán tâm, khoảng cách, diện tích polygon và IoU đè phủ.
- **`ai/inference/rules.py`**: Chuyên trách các quy tắc an toàn vật lý, bộ lọc hướng đầu xe và heuristics bối cảnh.
- **`ai/inference/mapper.py`**: Chuyên trách thuật toán ghép cặp tổn thất với bộ phận (`map_damages_to_parts`).
- **`ai/inference/visualizer.py`**: Chuyên trách vẽ nhãn Unicode tiếng Việt, tô màu mặt nạ và render side-by-side.
- **`ai/inference/pipeline.py`**: Chỉ đóng vai trò điều phối tinh gọn ($< 250$ dòng).

---

## 2. Thiếu Sót Trong Phân Hệ Đánh Giá AI (Evaluation Gaps)

### 2.1. Thiếu Ground Truth Cho Tập Demo
- Thư mục `demo/input/` chứa 8 ảnh thực tế nhưng **HOÀN TOÀN KHÔNG CÓ FILE NHÃN THẨM ĐỊNH (Ground Truth JSON / YOLO Label)**.
- Khi chạy `random_demo.py`, hệ thống chỉ xuất ra `predictions.json` (dự đoán của AI) mà không thể tính toán tự động xem trong 8 ảnh đó AI đoán đúng bao nhiêu %, sai bao nhiêu %.

### 2.2. Thiếu Kịch Bản Đánh Giá Hồi Quy (Regression Benchmark)
- Dự án chưa có một script tự động chạy trên tập kiểm định để phát hiện hiện tượng "sửa chỗ này hỏng chỗ khác": khi lập trình viên tinh chỉnh một tham số heuristic cho ảnh cửa trước, làm sao biết được ảnh nắp capo có bị giảm độ chính xác hay không?

---

## 3. Thiếu Sót Trong Quản Lý Dataset (Dataset Gaps)

1. **Thiếu dữ liệu cho 2 nhóm tổn thất trong Proposal**:
   - `tear_puncture` (rách cản) và `paint_chipping` (tróc sơn) chưa có ảnh và nhãn huấn luyện.
2. **Thiếu nhãn bộ phận chi tiết**:
   - Thiếu lớp `den_xe` (Đèn pha / Đèn hậu) trong dataset phân đoạn.
   - Lớp `cua_hong` chưa được phân tách thành `cua_truoc` và `cua_sau` ngay từ dữ liệu gốc.

---

## 4. Thiếu Sót Trong Phân Hệ Kiểm Thử (Testing Gaps)

1. **Chưa có Integration Test cho REST API (`ai/api/`)**:
   - Thư mục `ai/api/main.py` có 13.6 KB mã nguồn FastAPI hoàn chỉnh với các endpoint `/predict`, `/change-detection`, `/health`, nhưng **0% test coverage**.
   - Cần bổ sung tệp `ai/tests/test_api.py` sử dụng `httpx.AsyncClient` để kiểm tra việc gửi ảnh multipart và nhận JSON phản hồi.
2. **Chưa có Model Sanity Test**:
   - Thiếu test tự động xác nhận 2 file `.pt` và `.onnx` có thể nạp lên RAM và chạy 1 forward pass mà không báo lỗi thiếu file.
