# 05 - KIỂM KÊ VÀ ĐÁNH GIÁ TRỰC QUAN TẬP ẢNH DEMO (`demo/input/`)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Thư mục kiểm toán:** `demo/input/` và `demo/results/`  
**Nguyên tắc:** Liệt kê danh sách file thực tế 100%, không tự bịa đặt tên file, không tự điền ground truth giả.

---

## 1. Bảng Kiểm Kê Danh Sách Tệp Thực Tế Trong `demo/input/`

Toàn bộ thư mục `demo/input/` hiện có **chính xác 8 tệp**, được lưu dưới đuôi mở rộng `.png` nhưng định dạng nhị phân nội tại (MIME format) là `JPEG`:

| STT | Tên Tệp Thực Tế | Kích Thước File | Độ Phân Giải (W x H) | Định Dạng Nhị Phân | Nhận Xét Tên File |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `image copy 3.png` | 143.7 KB | $667 \times 713$ | JPEG | Đặt tên tự động theo cơ chế trùng lặp của OS |
| **2** | `image copy 4.png` | 673.9 KB | $667 \times 1000$ | JPEG | Đặt tên tự động |
| **3** | `image copy 5.png` | 59.2 KB | $507 \times 338$ | JPEG | Độ phân giải thấp |
| **4** | `image copy 6.png` | 447.2 KB | $1000 \times 558$ | JPEG | Góc chụp siêu cận cảnh |
| **5** | `image copy 7.png` | 678.0 KB | $1000 \times 667$ | JPEG | Ảnh chất lượng cao |
| **6** | `image copy 8.png` | 57.9 KB | $373 \times 500$ | JPEG | Độ phân giải thấp, góc chụp cận cảnh |
| **7** | `image copy.png` | 783.3 KB | $1000 \times 750$ | JPEG | Đặt tên tự động |
| **8** | `image.png` | 878.4 KB | $1000 \times 667$ | JPEG | Ảnh gốc trước khi bị sao chép |

---

## 2. Bảng Đối Chiếu Kết Quả AI Trong `demo/results/predictions.json`

*(Được ghi nhận khách quan từ lần chạy thực nghiệm gần nhất của `TwoStagePipeline`)*

| Tên Ảnh | Trạng Thái (`status`) | Góc Chụp (`shot_type`) | Tổn Thất Phát Hiện | Bộ Phận Phân Đoạn | Kết Quả Ánh Xạ (Mapping) | Đánh Giá Trực Quan Chuyên Môn |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| `image copy 3.png` | `needs_review` | `mid_view` | - `lamp_broken` (0.88)<br>- `scratch` (0.75)<br>- `scratch` (0.50)<br>- `dent` (0.50)<br>- `crack` (0.27) | `Nắp capo` (0.39) | - `lamp_broken` $\rightarrow$ `Đèn xe`<br>- `dent` $\rightarrow$ `Nắp capo`<br>- 3 vết $\rightarrow$ `khong_xac_dinh` | **Chưa hoàn hảo**: Mô hình bỏ sót Cản trước và Đèn xe. Hộp xước chiếm $>30\%$ diện tích ảnh làm kích hoạt cờ cảnh báo `needs_review`. |
| `image copy 4.png` | `completed` | `far_view` | - `dent` (0.87) | `Kính xe`, `Cửa trước`, `Nắp capo`, `Vè che bánh` | - `dent` $\rightarrow$ `Cửa trước` (IoU 0.87) | **Chính xác (Tốt)**: Nhận diện trọn vẹn cánh cửa trước và vết móp sâu ở sườn xe. |
| `image copy 5.png` | `completed` | `mid_view` | Không có detection (0 vết) | `Vè che bánh` (0.43) | Không có mapping | **Đạt**: Xe nguyên vẹn không tổn thất, không phát hiện báo động giả. |
| `image copy 6.png` | `insufficient_data`| `close_up` | - `crack` (0.76) | `Cản trước` (0.31) | - `crack` $\rightarrow$ `khong_xac_dinh` | **An toàn**: Ảnh macro vết nứt hẹp, hệ thống kích hoạt chốt chặn cận cảnh, từ chối gán bừa. |
| `image copy 7.png` | `completed` | `mid_view` | - 3 vết `scratch` (0.93, 0.87, 0.75) | `Cửa trước`, `Nắp capo`, `Kính xe` | - Cả 3 vết $\rightarrow$ `Cửa trước` | **Chính xác (Tốt)**: Gom cụm 3 vết xước trên cùng cánh cửa thành công. |
| `image copy 8.png` | `insufficient_data`| `close_up` | - `scratch` (0.59) | `Vè che bánh` (0.32) | - `scratch` $\rightarrow$ `khong_xac_dinh` | **An toàn**: Ảnh zoom sát vết xước, thiếu thông tin hình học tổng thể nên từ chối gán nhãn. |
| `image copy.png` | `completed` | `far_view` | - `lamp_broken` (0.89) | `Nắp capo`, `Kính xe` | - `lamp_broken` $\rightarrow$ `Đèn xe` | **Cứu bởi Rule**: Mô hình seg bỏ sót cụm đèn, nhưng rule ngữ nghĩa gán thành công vào Đèn xe. |
| `image.png` | `insufficient_data`| `close_up` | - 2 vết `dent` (0.55, 0.43) | `Cản trước` (0.31) | - Cả 2 vết $\rightarrow$ `khong_xac_dinh` | **An toàn**: Ảnh cận cảnh vết móp, yêu cầu chụp lại ảnh góc rộng. |

---

## 3. Tổng Kết Về Bộ Ảnh Demo

1. **Số lượng ảnh quá nhỏ để kết luận độ chính xác**:
   - 8 ảnh chỉ đóng vai trò là bài kiểm thử nhanh (Smoke Test / Sanity Check) để đảm bảo toàn bộ pipeline không bị crash khi chạy từ đầu đến cuối.
   - Không thể dùng 8 ảnh này để suy diễn tỷ lệ chính xác $80\%$ hay $90\%$ cho toàn bộ dự án.
2. **Hiện tượng che lỗi không xuất hiện**:
   - Tỷ lệ hoàn tất phân tích đầy đủ (`completed`) chỉ chiếm $50\%$ (4/8 ảnh).
   - $37.5\%$ (3/8 ảnh) rơi vào trạng thái thiếu dữ liệu (`insufficient_data`).
   - $12.5\%$ (1/8 ảnh) kích hoạt trạng thái cần giám định viên duyệt (`needs_review`).
   - Hệ thống phản ánh trung thực giới hạn vật lý của góc chụp thay vì ép buộc xuất ra kết quả đẹp mắt.
