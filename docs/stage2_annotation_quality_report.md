# BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG GÁN NHÃN (STAGE 2 QUALITY CONTROL)
**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Thư mục kiểm toán:** `ai\data\parts_pilot\annotations_raw`  
**Thời điểm thực thi:** 2026-11-14 23:06:00  

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM TOÁN CHẤT LƯỢNG

| Chỉ số kiểm soát | Giá trị đo được | Đánh giá chất lượng |
| :--- | :---: | :--- |
| **Tổng số ảnh Pilot** | **80** | 60 TRAIN / 20 VAL |
| **Số file JSON đã tạo** | **80** | 100.0% tiến độ |
| **Tổng số polygon ngoại thất** | **230** | Trung bình 2.9 polygon/ảnh |
| **Số file JSON hợp lệ tuyệt đối** | **75** | Đạt 93.8% |
| **Tổng số lỗi hình học / nhãn** | **5** | Cần chỉnh sửa |

---

## 2. PHÂN BỐ SỐ LƯỢNG POLYGON THEO 8 LỚP

| Class ID | Tên bộ phận ngoại thất | Số lượng polygon | Tỷ lệ % |
| :---: | :--- | :---: | :---: |
| 0 | `can_truoc` | 13 | 5.7% |
| 1 | `can_sau` | 40 | 17.4% |
| 2 | `nap_capo` | 40 | 17.4% |
| 3 | `ve_che_banh` | 48 | 20.9% |
| 4 | `cua_truoc` | 26 | 11.3% |
| 5 | `cua_sau` | 15 | 6.5% |
| 6 | `kinh_xe` | 47 | 20.4% |
| 7 | `den_xe` | 1 | 0.4% |

---

## 3. DANH SÁCH CHI TIẾT CÁC VẤN ĐỀ CẦN XỬ LÝ (ACTION ITEMS)

### Ảnh #13: `001479.jpg` (VAL) — Trạng thái: `PRE_LABELED`
- **[EMPTY_ANNOTATION]**: Tệp JSON không chứa đa giác nào (0 shapes)

### Ảnh #28: `001537.jpg` (TRAIN) — Trạng thái: `PRE_LABELED`
- **[EMPTY_ANNOTATION]**: Tệp JSON không chứa đa giác nào (0 shapes)

### Ảnh #29: `001948.jpg` (VAL) — Trạng thái: `PRE_LABELED`
- **[EMPTY_ANNOTATION]**: Tệp JSON không chứa đa giác nào (0 shapes)

### Ảnh #36: `000724.jpg` (TRAIN) — Trạng thái: `PRE_LABELED`
- **[EMPTY_ANNOTATION]**: Tệp JSON không chứa đa giác nào (0 shapes)

### Ảnh #40: `000731.jpg` (TRAIN) — Trạng thái: `PRE_LABELED`
- **[EMPTY_ANNOTATION]**: Tệp JSON không chứa đa giác nào (0 shapes)
