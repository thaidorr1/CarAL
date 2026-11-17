# 10 - TIẾN ĐỘ XÂY DỰNG BỘ GROUND TRUTH CHO DEMO (SPRINT 4B.1)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4B.1 — Build Ground Truth for Demo  
**Tập dữ liệu:** 8 ảnh tại `demo/input/`  
**Tệp dữ liệu bàn giao:**
- `demo/annotations_ground_truth.json`
- `demo/ground_truth_annotation_guide.md`
- `scratch/validate_ground_truth.py`

---

## 1. Tổng Quan Tiến Độ & Tuân Thủ Nguyên Tắc

Trong Sprint 4B.1, toàn bộ 8 ảnh trong thư mục `demo/input/` đã được lập danh mục, phân tích trực quan chuyên sâu và chuyển đổi thành bộ khung nhãn Ground Truth có cấu trúc chuẩn xác theo đặc tả kỹ thuật `docs/sprint4a_validation/06_ground_truth_design.md`.

### Cam kết bảo vệ tính toàn vẹn:
1. **100% Nhãn ở trạng thái Chờ Duyệt (Pending Review):**  
   Toàn bộ 14 đề xuất bộ phận và 14 đề xuất hư hỏng đều mang nhãn `"verification_status": "pending_review"` và `"is_verified": false`. Không có bất kỳ nhãn nào bị tự động đánh dấu là đã xác minh khi chưa có sự phê duyệt của con người.
2. **Không sửa đổi ảnh gốc hoặc dữ liệu hiện có:**  
   Toàn bộ ảnh gốc trong `demo/input/` được giữ nguyên vẹn 100%. Không can thiệp mã nguồn mô hình, không train lại mạng và không sửa đổi trọng số.
3. **Phân định rõ ràng giữa suy diễn tự động và kiểm định con người:**  
   Đối với các ảnh macro hoặc góc hẹp thiếu thông tin bối cảnh xe, trường `true_part_name` được gán rõ ràng là `"khong_xac_dinh"`, tuân thủ quy tắc Safety Guards của hệ thống.

---

## 2. Bảng Thống Kê Chi Tiết 8 Ảnh Demo

| ID | Tên File Thực Tế | Độ Phân Giải | Góc Chụp (`shot_type`) | Số Bộ Phận Đề Xuất | Số Tổn Thất Đề Xuất | Trạng Thái Thẩm Định | Điểm Nghi Ngờ Cần Con Người Thẩm Định |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `image copy 3.png` | $667 \times 713$ | `front_left_diagonal` | 1 (`nap_capo`) | 5 (`crack` $\times 2$, `scratch` $\times 2$, `dent` $\times 1$) | `pending_review` | - Hộp bao `scratch` [10.5, 464.6, 667.0, 698.2] quá lớn (>30% diện tích ảnh), vi phạm quy tắc hộp bao ôm sát; cần chuyên viên chia nhỏ theo vệt xước.<br>- Đèn pha vỡ nát (`crack`/`lamp_broken`) chưa có mask phân đoạn `den_xe` tương ứng do model seg thiếu lớp này. |
| **2** | `image copy 4.png` | $667 \times 1000$ | `side_right_diagonal` | 4 (`kinh_xe`, `cua_truoc`, `nap_capo`, `ve_che_banh`) | 1 (`dent`) | `pending_review` | - Vết móp lớn trên `cua_truoc` (conf 0.866) cần xác nhận mép biên có lan qua vòm bánh xe (`ve_che_banh`) hay cánh cửa sau (`cua_sau`) hay không. |
| **3** | `image copy 5.png` | $507 \times 338$ | `side_fender_profile` | 1 (`ve_che_banh`) | 0 (Không phát hiện) | `pending_review` | - Độ phân giải ảnh thấp ($507 \times 338$). Chuyên viên cần soi kỹ xem có vết xước dăm nhỏ nào bị bỏ sót hay xe hoàn toàn nguyên vẹn. |
| **4** | `image copy 6.png` | $1000 \times 558$ | `macro_bumper_seam` | 1 (`can_truoc`) | 1 (`crack`) | `pending_review` | - Ảnh siêu cận cảnh vết nứt tại khe nối. Bối cảnh xe bị cắt xén, bộ phận được gán an toàn là `khong_xac_dinh`. Giám định viên cần xác định xem đây là mép cản hay vỏ đèn. |
| **5** | `image copy 7.png` | $1000 \times 667$ | `side_door_panel` | 3 (`cua_truoc`, `nap_capo`, `kinh_xe`) | 3 (`scratch` $\times 3$) | `pending_review` | - Cả 3 vết xước tập trung trên cánh cửa trước. Cần xác định vết xước sâu đã lộ kim loại trần hay chưa để phân định giữa `scratch` và `paint_chipping`. |
| **6** | `image copy 8.png` | $373 \times 500$ | `macro_fender_arch` | 1 (`ve_che_banh`) | 1 (`scratch`) | `pending_review` | - Ảnh cận cảnh vòm bánh xe, thiếu mốc nhận dạng tổng thể. Bộ phận gán `khong_xac_dinh` nhằm chống gán sai. |
| **7** | `image copy.png` | $1000 \times 750$ | `front_left_diagonal` | 2 (`nap_capo`, `kinh_xe`) | 1 (`crack` / `lamp_broken`) | `pending_review` | - Cụm đèn pha vỡ nát. Tổn thất đè lên khu vực cản trước. Cần thẩm định xem có biến dạng móp (`dent`) hoặc xé rách (`tear_puncture`) trên cản trước đi kèm không. |
| **8** | `image.png` | $1000 \times 667$ | `macro_lower_bumper` | 1 (`can_truoc`) | 2 (`dent` $\times 2$) | `pending_review` | - Ảnh chụp góc hẹp mép dưới thân vỏ/cản trước. Gán `khong_xac_dinh` cho bộ phận chờ ảnh toàn cảnh đối chứng. |
| **TỔNG** | **8 Ảnh** | - | - | **14 Phân đoạn** | **14 Hư hỏng** | **100% Pending** | **Tất cả 28 nhãn đều chờ chuyên viên con người thẩm định** |

---

## 3. Kết Quả Kiểm Tra Tính Hợp Lệ Của Dữ Liệu (Automated Validation)

Tập dữ liệu `demo/annotations_ground_truth.json` đã được kiểm chứng tự động bằng script độc lập `scratch/validate_ground_truth.py`. Toàn bộ 8 tiêu chí kiểm tra kỹ thuật đều **ĐẠT (PASS)** tuyệt đối:

```text
=== VALIDATING GROUND TRUTH FILE: demo/annotations_ground_truth.json ===
[PASS] Valid JSON syntax.
[PASS] Top-level verification status: pending_human_review
[PASS] Exactly 8 images found (8).
[PASS] All 8 file names match demo/input/ perfectly with 0 duplicates.
[PASS] 0 labels marked as verified (Total parts: 14, Total damages: 14, all 28 pending review).
[PASS] Damage classes verified against approved taxonomy.
[PASS] Vehicle parts verified against approved taxonomy.
[PASS] Damage-to-Part mapping relationship links correctly.
=== ALL VALIDATION CHECKS PASSED SUCCESSFULLY! ===
```

### Chi tiết các kiểm tra đã vượt qua:
1. **Chuẩn cú pháp JSON:** Phân tích cú pháp thành công với UTF-8, không lỗi mã hóa ký tự.
2. **Số lượng ảnh:** Đúng chính xác 8 ảnh, không thiếu và không vượt quá.
3. **Tính duy nhất của tên tệp:** 100% tên tệp khớp với thư mục vật lý `demo/input/`, không có phần tử trùng lặp.
4. **Phù hợp danh mục hư hỏng (Damage Taxonomy):** Tất cả các lớp tổn thất nằm trong danh mục chuẩn hóa: `{'scratch', 'dent', 'crack', 'tear_puncture', 'paint_chipping', 'khong_xac_dinh'}`.
5. **Phù hợp danh mục bộ phận (Parts Taxonomy):** Tất cả các bộ phận xe nằm trong danh mục chuẩn hóa: `{'can_truoc', 'can_sau', 'nap_capo', 've_che_banh', 'cua_truoc', 'cua_sau', 'kinh_xe', 'den_xe', 'khong_xac_dinh'}`.
6. **Mối quan hệ ánh xạ (Damage-to-Part):** Mỗi tổn thất đều liên kết hợp lệ tới một `part_id` hiện diện trong cùng ảnh, hoặc mang giá trị `true_part_name: "khong_xac_dinh"` khi không đủ điều kiện gán.
7. **Bảo vệ tính khách quan (Zero-Auto-Verify):** Không có bất kỳ nhãn nào mang `is_verified: true`. Đảm bảo tính trung thực và minh bạch tuyệt đối trước khi tiến hành đánh giá hệ thống.

---

## 4. Kế Hoạch Tiếp Theo Cho Sprint 4B.2

1. **Hỗ trợ giao diện thẩm định (Human-in-the-Loop Review):** Cung cấp tài liệu `demo/ground_truth_annotation_guide.md` cho giám định viên hoặc kỹ sư thẩm định rà soát 28 mục nhãn.
2. **Chạy Benchmark Độc Lập:** Khi bộ nhãn được phê duyệt (hoặc sử dụng bộ nhãn sơ bộ có kiểm soát), thực hiện tính toán ma trận nhầm lẫn (Confusion Matrix), Precision, Recall, mAP50 cho Detection và mIoU cho Segmentation trên 8 ảnh demo.
3. **Đánh giá quy tắc Damage-to-Part:** Đo lường độ chính xác của logic ánh xạ hình học và cơ chế Safety Guards trên dữ liệu thực tế.
