# DANH SÁCH 30 ẢNH PILOT ƯU TIÊN KIỂM DUYỆT TRÊN LABELME
## (STAGE 2 STEP 4: PILOT PRIORITY ANNOTATION CHECKLIST)

**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Mục tiêu:** Cung cấp danh sách 30 ảnh quan trọng nhất để người dùng thẩm định và hoàn thiện nhãn Ground Truth bằng LabelMe trước khi Fine-tune mô hình ở Bước 4.  
**Tập trung giải quyết:** Lớp `3: ve_che_banh` (vè che bánh / tai xe) và các góc chụp cận cảnh va chạm bị giảm độ tự tin ở Bước 3.  

---

## 1. HƯỚNG DẪN KHỞI CHẠY NHANH LABELME

1. Double click file **`AutoClaimAI_LabelMe_Pilot.bat`** trên Desktop (hoặc tại gốc dự án [`launch_labelme_pilot.bat`](file:///d:/AI_CAR/launch_labelme_pilot.bat)).
2. Trong giao diện LabelMe, danh sách ảnh nằm ở cột bên phải. Tìm đến các mã ảnh trong bảng dưới đây theo thứ tự.
3. **Thao tác nhanh:**
   - **Kéo đỉnh:** Click chuột vào đỉnh polygon AI đã tạo nháp để kéo ôm sát đường gân tôn / rãnh cửa / vòm lốp.
   - **Vẽ mới (`Ctrl + N`):** Vẽ thêm `den_xe` (đèn pha / đèn hậu) và `ve_che_banh` (vòm che bánh xe) với 6 – 12 đỉnh đơn giản.
   - **Sửa nhãn / Xóa:** Nhấn phím `Delete` nếu có polygon nhận diện nhầm; nhấp đúp vào polygon để đổi tên lớp.
   - **Lưu (`Ctrl + S`):** Lưu trực tiếp vào `ai/data/parts_pilot/annotations_raw/`.

---

## 2. BẢNG 30 ẢNH ƯU TIÊN (20 VAL GOLDEN SET + 10 TRAIN FENDER & CLOSE-UP)

| STT | Mã ảnh | Tập | Góc nhìn | Gợi ý AI nháp sẵn (`annotations_raw`) | Trọng tâm kiểm duyệt & Vẽ bổ sung |
| :-: | :--- | :-: | :--- | :--- | :--- |
| **01** | `003848.jpg` | **VAL** | Góc trước cận cảnh | `can_truoc`, `den_xe` | **ĐÃ DUYỆT ĐẠT CHUẨN** (Mẫu kiểm thử số 1) |
| **02** | `000821.jpg` | **VAL** | Góc trước cận cảnh | `can_truoc`, `nap_capo` | Kéo viền cản trước, vẽ thêm cụm `den_xe` hai bên |
| **03** | `003126.jpg` | **VAL** | Góc trước cận cảnh | `can_truoc`, `nap_capo` | Chỉnh mép ca-pô, vẽ thêm 2 chóa `den_xe` pha |
| **04** | `001479.jpg` | **VAL** | Góc trước | *(Trống)* | Vẽ mới ba-đờ-sốc `can_truoc`, `nap_capo`, `den_xe` |
| **05** | `001489.jpg` | **VAL** | Góc sau cận cảnh | `can_sau` | Kéo mép cản sau, vẽ thêm 2 `den_xe` hậu và `kinh_xe` |
| **06** | `000767.jpg` | **VAL** | Góc sau | `can_sau`, `nap_capo` | Đổi `nap_capo` thành `kinh_xe` (nhầm), vẽ thêm đèn hậu |
| **07** | `003352.jpg` | **VAL** | Góc sau | `can_sau`, `kinh_xe` | Kéo sát viền kính chắn gió sau, vẽ cụm `den_xe` hậu |
| **08** | `001948.jpg` | **VAL** | Góc sau | *(Trống)* | Vẽ mới cản sau `can_sau`, `kinh_xe`, `den_xe` |
| **09** | `003916.jpg` | **VAL** | Góc bên hông | `ve_che_banh`, `nap_capo`, `kinh_xe` | **Kéo ôm khít vòm bánh `ve_che_banh`**, thêm 2 cửa hông |
| **10** | `001081.jpg` | **VAL** | Góc bên hông | `ve_che_banh`, `cua_truoc` | **Kéo chuẩn đường cong `ve_che_banh`**, vẽ thêm `cua_sau` |
| **11** | `000728.jpg` | **VAL** | Góc bên hông | `ve_che_banh` | **Kéo chuẩn vòm lốp `ve_che_banh`**, vẽ thêm 2 cửa hông |
| **12** | `000714.jpg` | **VAL** | Góc bên hông | `cua_truoc`, `kinh_xe` | **Vẽ mới `ve_che_banh` trước & sau**, thêm `cua_sau` |
| **13** | `003881.jpg` | **VAL** | Góc xiên 45° trước | `cua_sau`, `can_sau`, `nap_capo`, `kinh_xe` | Đổi `can_sau` thành `can_truoc`, **vẽ thêm `ve_che_banh`**, `den_xe` |
| **14** | `000634.jpg` | **VAL** | Góc xiên 45° trước | `ve_che_banh`, `nap_capo`, `kinh_xe`, `can_sau` | Đổi `can_sau` thành `can_truoc`, **kéo ôm sát `ve_che_banh`** |
| **15** | `000177.jpg` | **VAL** | Góc xiên 45° trước | `ve_che_banh`, `cua_truoc`, `nap_capo`, `kinh_xe` | **Kéo chuẩn mép tai xe `ve_che_banh`**, vẽ thêm đèn pha |
| **16** | `000312.jpg` | **VAL** | Góc xiên 45° trước | `ve_che_banh`, `cua_truoc`, `nap_capo`, `kinh_xe` | **Kéo chuẩn vòm tai xe `ve_che_banh`**, vẽ thêm `den_xe` |
| **17** | `003947.jpg` | **VAL** | Góc xiên 45° sau | `cua_truoc`, `kinh_xe` | Đổi `cua_truoc` thành `cua_sau`, vẽ `can_sau` và `den_xe` |
| **18** | `003101.jpg` | **VAL** | Góc xiên 45° sau | `ve_che_banh`, `cua_truoc`, `can_sau`, `kinh_xe` | **Kéo chuẩn vè che bánh sau `ve_che_banh`**, thêm đèn hậu |
| **19** | `000052.jpg` | **VAL** | Trung cảnh tai nạn | `kinh_xe` | Vẽ cản trước `can_truoc`, ca-pô `nap_capo`, `ve_che_banh` |
| **20** | `003095.jpg` | **VAL** | Trung cảnh tai nạn | `cua_truoc` | Kéo chỉnh đầu xe: cản, nắp ca-pô, vè bánh và đèn |
| **21** | `000011.jpg` | **TRAIN** | Góc bên hông | `cua_truoc`, `kinh_xe`, `ve_che_banh` | **Kéo sát mép tôn vòm lốp `ve_che_banh`**, vẽ thêm `cua_sau` |
| **22** | `001076.jpg` | **TRAIN** | Góc bên hông | `cua_truoc`, `kinh_xe`, `ve_che_banh` | **Kéo chuẩn vè bánh `ve_che_banh`**, chỉnh mép gioăng cửa |
| **23** | `000602.jpg` | **TRAIN** | Góc bên hông | `cua_truoc`, `ve_che_banh` | **Kéo chuẩn vè che bánh trước `ve_che_banh`**, thêm kính xe |
| **24** | `001012.jpg` | **TRAIN** | Góc bên hông | `cua_truoc`, `ve_che_banh` | **Kéo chuẩn vè bánh sau `ve_che_banh`**, chỉnh khe rãnh cửa |
| **25** | `003262.jpg` | **TRAIN** | Góc xiên 45° trước | `cua_truoc`, `nap_capo`, `can_sau`, `kinh_xe` | Đổi `can_sau` thành `can_truoc`, **vẽ thêm `ve_che_banh` tai xe** |
| **26** | `003312.jpg` | **TRAIN** | Góc xiên 45° trước | `ve_che_banh`, `nap_capo`, `kinh_xe`, `can_sau` | **Kéo ôm khít `ve_che_banh`**, đổi `can_sau` thành `can_truoc` |
| **27** | `003931.jpg` | **TRAIN** | Góc xiên 45° trước | `ve_che_banh`, `cua_truoc`, `cua_sau`, `nap_capo` | **Kéo chuẩn viền vòm lốp `ve_che_banh`**, thêm đèn pha |
| **28** | `003773.jpg` | **TRAIN** | Góc xiên 45° trước | `ve_che_banh`, `cua_sau`, `can_sau`, `nap_capo` | **Kéo chuẩn đa giác `ve_che_banh`**, chỉnh mép cản và đèn |
| **29** | `003054.jpg` | **TRAIN** | Góc xiên 45° sau | `can_sau`, `kinh_xe`, `ve_che_banh` | **Kéo chuẩn vè bánh sau `ve_che_banh` vùng va chạm** |
| **30** | `000826.jpg` | **TRAIN** | Góc xiên 45° sau | `cua_truoc`, `ve_che_banh` | **Kéo chuẩn vè che bánh sau `ve_che_banh`**, vẽ thêm đèn hậu |

---

*Lưu ý: Sau khi bạn lưu các file JSON trong danh sách trên, chạy lệnh cập nhật tracking:*  
`python ai/scripts/manage_annotation_tracking.py --sync`
