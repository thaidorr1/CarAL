# BÁO CÁO KỸ THUẬT: HOÀN THIỆN DANH MỤC PILOT DATASET STAGE 2
**Mã tác vụ:** P1.1.3 — Finalize Pilot Dataset  
**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Ngày thực hiện:** 13/11/2026  
**Chế độ thực thi:** Non-destructive Verification & Draft Finalization  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH HOÀN THIỆN

Dựa trên kết quả từ hai báo cáo trước:
- [`docs/stage2_pilot_independent_audit.md`](file:///d:/AI_CAR/docs/stage2_pilot_independent_audit.md): Xác nhận 80 ảnh đạt chuẩn chất lượng nhưng phát hiện rủi ro rò rỉ dữ liệu giữa ảnh #65 (VAL) và #67 (TRAIN).
- [`docs/stage2_identity_resolution_report.md`](file:///d:/AI_CAR/docs/stage2_identity_resolution_report.md): Chứng minh toán học qua SIFT Homography ($957$ inliers, $94.1\%$) khẳng định `#65 (003947.jpg)` và `#67 (003930.jpg)` là **CONFIRMED SAME VEHICLE**; đồng thời xác định `#70 (003074.jpg)` là ảnh macro crop vỡ kính không đủ tiêu chuẩn nhận diện thân vỏ xe.

Tác vụ **P1.1.3** tiến hành giải quyết dứt điểm hai khiếm khuyết trên nhằm hoàn thiện bộ dữ liệu **80 ảnh Pilot Dataset (60 TRAIN / 20 VAL)** đạt chuẩn **Zero Data Leakage 100%** và sẵn sàng cho việc gán nhãn chuyên gia.

---

## 2. KẾT QUẢ KIỂM TRA TOÀN DIỆN ỨNG VIÊN `000072.jpg` (THAY THẾ CHO #67 TRONG TRAIN)

### 2.1. Thông số vật lý và đặc trưng xe
- **Đường dẫn tệp:** `ai/data/yolo/images/train/000072.jpg` (Tồn tại, nguyên vẹn).
- **Kích thước khung hình:** $1000 \times 667$ pixels (tỷ lệ chuẩn $1.50:1$).
- **Độ nét (Laplacian Variance):** $1128.5$ (vượt xa ngưỡng sàn 250).
- **Độ sáng trung bình (Brightness):** $87.0$ (ánh sáng tự nhiên sắc nét).
- **Mã băm SHA256:** `cd1b25a17495a2d63209f75dd7ffbd60dd09fd9f55f80653d35b1beb6a6144ab`.
- **Đặc điểm xe nhận dạng trực quan:**
  - Kiểu thân xe: Sedan 4 cửa màu xám chì kim loại (`[82, 87, 88]`).
  - Góc chụp: Xiên $45^\circ$ sau (Rear-3/4 View), nhìn rõ cụm đèn hậu, cản sau, vè hông sau và cửa sau.
  - Tổn thất CarDD: Vết cào xước thân xe (`scratch`, conf 0.867).

### 2.2. Kiểm tra đối chiếu với toàn bộ 79 ảnh Pilot còn lại
Đã chạy thuật toán so khớp đa chiều với 79 ảnh trong Manifest:
- **Trùng lặp mã băm (SHA256 Collisions):** **0** (File độc nhất 100%).
- **Khoảng cách Perceptual Hash nhỏ nhất (Min dHash Distance):** **21** (với `003143.jpg`). Toàn bộ $\ge 21$ chứng minh cấu trúc không gian ảnh hoàn toàn khác biệt.
- **Điểm đặc trưng SIFT trùng khớp cao nhất:** **27 điểm** (với `003378.jpg`), chỉ là các điểm ngẫu nhiên ở bề mặt mặt đường $\ll 100$.

> [!TIP]
> **KẾT LUẬN VỀ `000072.jpg`:**  
> **CONFIRMED DISTINCT VEHICLE (Zero Leakage Risk)**. Ảnh hoàn toàn độc lập, chất lượng cao, cực kỳ lý tưởng để thay thế `#67 (003930.jpg)` trong tập **TRAIN**.

---

## 3. SÀNG LỌC VÀ LỰA CHỌN ỨNG VIÊN THAY THẾ CHO `003074.jpg` (TẬP VAL)

### 3.1. Hạn chế của ảnh cũ `003074.jpg`
- Là ảnh crop cận cảnh (macro) góc vỡ kính, kích thước hẹp $1000 \times 484$ px.
- Mô hình nhận diện phát hiện **0 bộ phận thân vỏ** (thiếu cản sau, thiếu vè bánh, thiếu cửa).
- Không phù hợp với tiêu chuẩn đánh giá mô hình phân đoạn ngoại thất xe trong tập kiểm tra (VAL).

### 3.2. Quá trình sàng lọc từ kho dữ liệu CarDD
Từ kho **122 ảnh góc xiên $45^\circ$ sau** chưa từng xuất hiện trong Manifest, chúng tôi áp dụng bộ lọc vật lý (độ phân giải $\ge 900 \times 600$, độ nét $\ge 400$, ánh sáng $60 - 200$) và đánh giá qua mô hình nhận diện bộ phận ngoại thất.

### 3.3. Ứng viên xuất sắc nhất: `003101.jpg`
- **Đường dẫn tệp:** `ai/data/yolo/images/val/003101.jpg` (Thuộc đúng phân tập `val` gốc của CarDD).
- **Kích thước khung hình:** $1000 \times 667$ pixels.
- **Độ nét (Laplacian Variance):** **$1689.8$** (độ tương phản và độ sắc nét rất cao).
- **Độ sáng trung bình (Brightness):** $143.9$ (ánh sáng ban ngày hoàn hảo).
- **Mã băm SHA256:** `4f42ee133bce5073e3ba08a3ef754ea6514b786a72f1b14bee7681499af381b4`.
- **Bộ phận ngoại thất nhận diện đầy đủ:**
  - `can_sau`: Cản sau nhìn thấy toàn bộ góc mép và cảm biến.
  - `ve_che_banh`: Vè hông sau ôm trọn vòm bánh xe.
  - `cua_sau` và `cua_truoc`: Thấy rõ đường chia rãnh giữa hai cánh cửa.
  - `kinh_xe`: Kính sau và kính cửa hông.
  - `den_xe`: Cụm đèn hậu sắc cạnh.
  - Tổn thất CarDD: Rạn vỡ kính sau (`glass_shatter`).

### 3.4. Kiểm tra đối chiếu độc lập của `003101.jpg`
- **SHA256 Collisions:** **0**.
- **Min dHash Distance:** **23** (với `000753.jpg`).
- **Max SIFT Good Matches:** **42** (với `002394.jpg`), không hình thành bất kỳ cụm homography inlier nào.

> [!TIP]
> **KẾT LUẬN VỀ `003101.jpg`:**  
> **CONFIRMED DISTINCT VEHICLE (Zero Leakage Risk)**. Ảnh toàn cảnh xuất sắc, nguồn gốc `val` tự nhiên, hoàn hảo 100% để thay thế vị trí `#70 (003074.jpg)` trong tập **VAL**.

---

## 4. BẢNG TỔNG HỢP SO SÁNH TRƯỚC VÀ SAU KHI ĐIỀU CHỈNH

| Hạng mục đối chiếu | Phiên bản Pilot cũ (P1.1) | Phiên bản Pilot Đề Xuất Hoàn Thiện (P1.1.3) | Đánh giá cải thiện |
| :--- | :--- | :--- | :---: |
| **Vị trí #67 (TRAIN)** | `003930.jpg` *(Trùng xe #65)* | **`000072.jpg`** *(Xe xám chì độc lập)* | **Triệt tiêu 100% rò rỉ Train/Val** |
| **Vị trí #70 (VAL)** | `003074.jpg` *(Macro vỡ kính 0 bộ phận)* | **`003101.jpg`** *(Xe toàn cảnh 5 bộ phận)* | **Chuẩn hóa chất lượng kiểm định Val** |
| **Tổng số lượng ảnh** | 80 ảnh | 80 ảnh | Giữ nguyên $100\%$ |
| **Cơ cấu Train / Val** | 60 Train / 20 Val ($75\% / 25\%$) | 60 Train / 20 Val ($75\% / 25\%$) | Cân bằng tuyệt đối |
| **Số lượng xe độc lập** | 79 xe (1 xe bị chụp 2 lần) | **80 xe độc lập hoàn toàn** | Đạt chuẩn Zero Leakage |
| **Rò rỉ dữ liệu (Leakage)** | Có rủi ro (Cặp #65 vs #67) | **0% (Zero Data Leakage)** | **HOÀN HẢO** |

---

## 5. MA TRẬN PHÂN CHIA VÀ ĐỘ PHỦ 8 LỚP MỤC TIÊU DỰ KIẾN MỚI

### 5.1. Ma trận Góc nhìn $\times$ Phân tập (Viewpoint $\times$ Split)
| Nhóm góc nhìn | Toàn bộ Pilot | Tập TRAIN (75%) | Tập VAL (25%) | Ghi chú chất lượng |
| :--- | :---: | :---: | :---: | :--- |
| **Góc trước (Front View)** | 16 | 12 | 4 | Thấy rõ cản trước, nắp capo, đèn pha, kính lái |
| **Góc sau (Rear View)** | 16 | 12 | 4 | Thấy rõ cản sau, đèn hậu, kính sau |
| **Góc bên hông (Side View)** | 16 | 12 | 4 | Thấy rõ cửa trước, cửa sau, vè bánh |
| **Góc xiên 45° trước (Front 3/4)** | 16 | 12 | 4 | Góc liên hoàn 6 bộ phận ngoại thất |
| **Góc xiên 45° sau (Rear 3/4)** | 10 | 8 | 2 | Đã chuẩn hóa bằng `000072.jpg` và `003101.jpg` |
| **Trung cảnh chi tiết (Medium Shot)** | 6 | 4 | 2 | Cận cảnh chi tiết cụm góc đầu/đuôi xe |
| **TỔNG CỘNG** | **80** | **60** | **20** | **100% xe độc lập, không rò rỉ** |

### 5.2. Thống kê độ phủ 8 lớp mục tiêu
| Mã lớp | Tên bộ phận ngoại thất | Số ảnh Train (60) | Số ảnh Val (20) | Tổng ảnh phủ (80) | Tỷ lệ bao phủ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `can_truoc` | Cản trước | 28 | 10 | **38** | 47.5% |
| `can_sau` | Cản sau | 20 | 6 | **26** | 32.5% |
| `nap_capo` | Nắp capo | 28 | 10 | **38** | 47.5% |
| `ve_che_banh` | Vè che bánh | 36 | 12 | **48** | 60.0% |
| `cua_truoc` | Cửa trước | 24 | 8 | **32** | 40.0% |
| `cua_sau` | Cửa sau | 20 | 6 | **26** | 32.5% |
| `kinh_xe` | Kính xe | 56 | 18 | **74** | 92.5% |
| `den_xe` | Đèn xe | 48 | 16 | **64** | 80.0% |

---

## 6. ĐÁNH GIÁ RỦI RO CÒN TỒN TẠI VÀ BIỆN PHÁP KIỂM SOÁT

1. **Rủi ro rò rỉ dữ liệu (Data Leakage):**
   - *Đánh giá:* **ĐÃ TRIỆT TIÊU HOÀN TOÀN (0%)**. Việc thay thế `#67` bằng `000072.jpg` và `#70` bằng `003101.jpg` đảm bảo 80 ảnh thuộc về đúng 80 xe độc lập.
2. **Rủi ro chất lượng gán nhãn thủ công (Human Annotation Drift):**
   - *Biện pháp kiểm soát:* Trong giai đoạn P1.2, người gán nhãn phải dùng công cụ AnyLabeling có tích hợp AI Assistant (MobileSAM point-prompting). Cần đặt positive point vào giữa mảng panel và negative point vào phần kính/phản quang để polygon bám chuẩn đường mép dập nổi kim loại.

---

## 7. KẾT LUẬN VÀ TRÌNH PHÊ DUYỆT

- Bản dự thảo Manifest mới đã được chuẩn bị và lưu trữ tạm thời tại `scratch/pilot_manifest_proposed.csv`.
- Theo đúng quy định ràng buộc, **tệp chính thức [`ai/data/parts_pilot/pilot_manifest.csv`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_manifest.csv) và ảnh Contact Sheet chưa bị chỉnh sửa**.

Kính trình người dùng phê duyệt cập nhật chính thức hai vị trí:
1. Vị trí #67 (TRAIN): Thay `003930.jpg` bằng **`000072.jpg`**
2. Vị trí #70 (VAL): Thay `003074.jpg` bằng **`003101.jpg`**

Sau khi nhận được lệnh phê duyệt, tôi sẽ lập tức đồng bộ hóa tệp Manifest và kết xuất lại bảng ảnh Contact Sheet 80 ô hoàn chỉnh.
