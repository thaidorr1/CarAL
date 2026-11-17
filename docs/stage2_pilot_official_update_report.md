# BÁO CÁO CẬP NHẬT CHÍNH THỨC: HOÀN TẤT PILOT DATASET STAGE 2
**Mã tác vụ:** P1.1.4 — Official Pilot Dataset Update  
**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Ngày thực hiện:** 13/11/2026  
**Trạng thái phê duyệt:** ĐÃ PHÊ DUYỆT BỞI USER  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH THỰC THI

Sau khi phương án xử lý rò rỉ dữ liệu và chuẩn hóa chất lượng kiểm thử trong báo cáo [`docs/stage2_pilot_finalization_report.md`](file:///d:/AI_CAR/docs/stage2_pilot_finalization_report.md) được phê duyệt:
- **Vị trí #67 (TRAIN):** Thay thế `003930.jpg` bằng **`000072.jpg`** nhằm triệt tiêu hoàn toàn rủi ro trùng xe với ảnh `#65 (003947.jpg, VAL)`.
- **Vị trí #70 (VAL):** Thay thế `003074.jpg` bằng **`003101.jpg`** nhằm thay thế ảnh macro crop kính vỡ (không có bộ phận xe) bằng ảnh toàn cảnh ngoại thất xe chất lượng cao.

Tác vụ **P1.1.4** đã tiến hành sao lưu an toàn, cập nhật chính thức tệp Manifest, kết xuất lại bảng ảnh kiểm tra trực quan Contact Sheet và vượt qua bộ kiểm thử tự động 8 tiêu chí khắt khe.

---

## 2. QUY TRÌNH SAO LƯU VÀ CẬP NHẬT CHÍNH THỨC

### 2.1. Sao lưu dữ liệu phiên bản trước (Safe Backup)
Trước khi ghi đè, toàn bộ phiên bản V1 đã được sao lưu nguyên vẹn:
1. `ai/data/parts_pilot/pilot_manifest_v1_backup.csv` (Sao lưu từ `pilot_manifest.csv`)
2. `ai/data/parts_pilot/pilot_contact_sheet_v1_backup.jpg` (Sao lưu từ `pilot_contact_sheet.jpg`)

### 2.2. Chi tiết hai bản ghi được cập nhật chính thức
| Vị trí | Tên tệp mới | Phân tập | Nguồn gốc ảnh | Vehicle ID | Kích thước & Độ nét | Bộ phận nhận diện |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **#67** | **`000072.jpg`** | **TRAIN** | `ai/data/yolo/images/train/000072.jpg` | `VEH_000072` | $1000 \times 667$<br>Sharpness=$1128.5$ | `can_sau`, `ve_che_banh`, `cua_sau`, `den_xe`, `kinh_xe` |
| **#70** | **`003101.jpg`** | **VAL** | `ai/data/yolo/images/val/003101.jpg` | `VEH_003101` | $1000 \times 667$<br>Sharpness=$1689.8$ | `can_sau`, `ve_che_banh`, `cua_sau`, `cua_truoc`, `den_xe`, `kinh_xe` |

---

## 3. KẾT QUẢ KIỂM TOÁN TỰ ĐỘNG (FINAL VERIFICATION SUITE — 8/8 PASS)

Hệ thống đã chạy bộ test tự động [`scratch/final_verification_test.py`](file:///d:/AI_CAR/scratch/final_verification_test.py) trên tệp Manifest và Contact Sheet vừa cập nhật:

| STT | Tiêu chí kiểm tra bất biến (Invariant Test) | Giá trị kỳ vọng | Giá trị thực tế | Kết quả |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Tổng số lượng bản ghi Manifest** | Đúng 80 dòng | 80 dòng | <span style="color:green">**PASS**</span> |
| **2** | **Cơ cấu phân chia Train / Val** | 60 Train / 20 Val ($75\% / 25\%$) | 60 Train / 20 Val | <span style="color:green">**PASS**</span> |
| **3** | **Tính sẵn sàng của tệp ảnh (Existence)** | 0 ảnh bị thiếu (80/80) | 0 ảnh thiếu | <span style="color:green">**PASS**</span> |
| **4** | **Tính độc nhất dữ liệu nhị phân (SHA256)** | 80 mã SHA256 độc nhất (0 duplicate) | 80 mã độc nhất | <span style="color:green">**PASS**</span> |
| **5** | **Rò rỉ dữ liệu phương tiện (Vehicle Leakage)** | $\text{Train} \cap \text{Val} = \emptyset$ (0 xe) | Đúng 0 xe trùng | <span style="color:green">**PASS**</span> |
| **6a**| **Xác thực vị trí #67** | `000072.jpg` (TRAIN) | Chính xác #67 | <span style="color:green">**PASS**</span> |
| **6b**| **Xác thực vị trí #70** | `003101.jpg` (VAL) | Chính xác #70 | <span style="color:green">**PASS**</span> |
| **7** | **Thông số kỹ thuật Contact Sheet** | $2.400 \times 1.640$ pixels, RGB | $2.400 \times 1.640$ | <span style="color:green">**PASS**</span> |

---

## 4. MA TRẬN PHÂN BỐ VÀ ĐỘ PHỦ 8 LỚP MỤC TIÊU HOÀN THIỆN

### 4.1. Phân bổ theo nhóm góc nhìn (Viewpoint $\times$ Split)
```text
┌───────────────────────┬───────────┬──────────┬─────────┐
│ Nhóm góc nhìn xe      │ Tổng cộng │ Tập TRAIN│ Tập VAL │
├───────────────────────┼───────────┼──────────┼─────────┤
│ Góc trước (Front)     │    16     │    12    │    4    │
│ Góc sau (Rear)        │    16     │    12    │    4    │
│ Góc bên hông (Side)   │    16     │    12    │    4    │
│ Góc xiên 45° trước    │    16     │    12    │    4    │
│ Góc xiên 45° sau      │    10     │     8    │    2    │
│ Trung cảnh chi tiết   │     6     │     4    │    2    │
├───────────────────────┼───────────┼──────────┼─────────┤
│ TỔNG CỘNG             │    80     │    60    │   20    │
└───────────────────────┴───────────┴──────────┴─────────┘
```

### 4.2. Độ phủ chi tiết 8 lớp bộ phận xe
Mọi lớp bộ phận đều đạt số lượng mẫu dồi dào, vượt ngưỡng kiểm soát chất lượng ($\ge 25$ ảnh/lớp):
1. `can_truoc` (Cản trước): **38 ảnh** (Train: 28, Val: 10)
2. `can_sau` (Cản sau): **26 ảnh** (Train: 20, Val: 6)
3. `nap_capo` (Nắp capo): **38 ảnh** (Train: 28, Val: 10)
4. `ve_che_banh` (Vè che bánh): **48 ảnh** (Train: 36, Val: 12)
5. `cua_truoc` (Cửa trước): **32 ảnh** (Train: 24, Val: 8)
6. `cua_sau` (Cửa sau): **26 ảnh** (Train: 20, Val: 6)
7. `kinh_xe` (Kính xe): **74 ảnh** (Train: 56, Val: 18)
8. `den_xe` (Đèn xe): **64 ảnh** (Train: 48, Val: 16)

---

## 5. DANH MỤC TỆP ĐÃ ĐỒNG BỘ VÀ LƯU KHO

1. **Manifest chính thức đã cập nhật:**
   [`ai/data/parts_pilot/pilot_manifest.csv`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_manifest.csv) (80 bản ghi chuẩn hóa).
2. **Contact Sheet chính thức đã kết xuất:**
   [`ai/data/parts_pilot/pilot_contact_sheet.jpg`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_contact_sheet.jpg) (Lưới $10 \times 8$ ô, $2.400 \times 1.640$ pixels).
3. **Các tệp sao lưu an toàn:**
   - [`ai/data/parts_pilot/pilot_manifest_v1_backup.csv`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_manifest_v1_backup.csv)
   - [`ai/data/parts_pilot/pilot_contact_sheet_v1_backup.jpg`](file:///d:/AI_CAR/ai/data/parts_pilot/pilot_contact_sheet_v1_backup.jpg)

---

## 6. KẾT LUẬN VÀ TRẠNG THÁI SẴN SÀNG

- Toàn bộ các yêu cầu của **P1.1.4** đã hoàn thành trọn vẹn.
- Dữ liệu gốc, mô hình, trọng số và nhãn annotation cũ được bảo tồn tuyệt đối.
- **Pilot Dataset chính thức đạt trạng thái:**
  > **READY FOR HUMAN ANNOTATION (STAGE P1.2)** — Sẵn sàng đưa 80 ảnh vào công cụ gán nhãn bán tự động (AnyLabeling / CVAT) kết hợp MobileSAM AI Assistant.
