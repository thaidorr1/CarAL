# BÁO CÁO THIẾT LẬP CÔNG CỤ VÀ QUY TRÌNH GÁN NHÃN THỦ CÔNG
## (STAGE 2 HUMAN ANNOTATION TOOL SETUP REPORT)

**Mã tác vụ:** P1.2.1 — Build Human Annotation Workflow  
**Dự án:** AutoClaim-AI (`D:\AI_CAR`)  
**Ngày thực hiện:** 13/11/2026  
**Trạng thái thực thi:** THIẾT LẬP VÀ KIỂM THỬ THÀNH CÔNG (READY FOR AUDIT)  

---

## 1. TỔNG QUAN HẠNG MỤC HOÀN THÀNH

Nhằm phục vụ giai đoạn gán nhãn Ground Truth chuẩn mực cho **80 ảnh Pilot Dataset (60 TRAIN / 20 VAL)** mà không dùng pseudo-label hay polygon sinh tự động, tác vụ **P1.2.1** đã hoàn thành toàn diện 6 hợp phần cốt lõi:

1. **Khảo sát môi trường và tương thích:** Khảo sát Python 3.13 và định dạng YOLOv8-seg.
2. **Cấu trúc thư mục chuyên biệt:** Thiết lập không gian lưu trữ nhãn thô, nhãn kiểm duyệt và xuất mô hình tách biệt hoàn toàn với dataset gốc.
3. **Cấu hình chuẩn hóa 8 lớp:** Khóa chặt danh mục 8 lớp ngoại thất trong `labels.txt` và `class_mapping.yaml`.
4. **Bộ công cụ quản lý vòng đời và kiểm duyệt:** Tạo file tracking 80 ảnh với 6 trạng thái (`NOT_STARTED` đến `APPROVED`).
5. **Bộ kiểm soát chất lượng tự động (Automated QC Validator):** Phát hiện 8 loại lỗi hình học và nhãn sai trước khi phê duyệt.
6. **Công cụ chuyển đổi sang YOLOv8-seg an toàn:** Chỉ xuất những ảnh đã được `APPROVED` sang thư mục riêng.

---

## 2. KẾT QUẢ KHẢO SÁT MÔI TRƯỜNG VÀ ĐỊNH DẠNG CHUẨN

- **Môi trường Python:** Python 3.13.5 (AMD64) trên hệ điều hành Windows.
- **Trạng thái LabelMe:** Chưa cài đặt sẵn trong môi trường Python hệ thống. Đã xây dựng script launcher hỗ trợ cả cài đặt `pip install pyqt6 labelme` hoặc chạy bản Portable `labelme.exe`.
- **Định dạng YOLO Segmentation chuẩn của dự án:**
  $$\text{Format: } \langle\text{class\_id}\rangle \quad x_1 \quad y_1 \quad x_2 \quad y_2 \quad \dots \quad x_n \quad y_n$$
  Trong đó tọa độ được chuẩn hóa về $[0.0, 1.0]$ với độ chính xác 6 chữ số thập phân, khớp với Ultralytics YOLOv8-seg.

---

## 3. CẤU TRÚC THƯ MỤC VÀ TỆP CẤU HÌNH ĐÃ THIẾT LẬP

```text
ai/data/parts_pilot/
├── annotation_config/
│   ├── labels.txt                 # Danh sách 8 lớp khóa cứng cho LabelMe
│   └── class_mapping.yaml         # Bảng ánh xạ ID (0..7) và tên tiếng Việt
├── annotations_raw/               # Nơi lưu JSON thô sau khi vẽ tay
├── annotations_reviewed/          # Nơi lưu JSON sau khi QA thẩm định đạt chuẩn
├── annotation_exports/            # Nơi xuất YOLOv8-seg độc lập (chỉ ảnh APPROVED)
│   ├── dataset_parts_pilot.yaml   # Cấu hình huấn luyện YOLOv8-seg
│   ├── images/ (train, val)
│   └── labels/ (train, val)
├── images/                        # 80 ảnh Pilot được hardlink trực tiếp (0 duplicate byte)
├── annotation_tracking.csv        # Bảng theo dõi tiến độ vòng đời 80 ảnh
├── pilot_manifest.csv             # Manifest chính thức (60 TRAIN / 20 VAL)
└── pilot_contact_sheet.jpg        # Bảng ảnh trực quan 80 ô ($2.400 \times 1.640$ px)
```

---

## 4. BỘ CÔNG CỤ VÀ SCRIPT ĐÃ PHÁT TRIỂN

| STT | Script thực thi | Đường dẫn tệp | Chức năng chính |
| :---: | :--- | :--- | :--- |
| **1** | **Annotation Launcher** | [`ai/scripts/launch_annotation_tool.py`](file:///d:/AI_CAR/ai/scripts/launch_annotation_tool.py) | Khởi chạy LabelMe trỏ đúng vào 80 ảnh Pilot, nạp file nhãn 8 lớp, khóa cờ `--nodata` để tối ưu dung lượng JSON (<15KB). |
| **2** | **Tracking Manager** | [`ai/scripts/manage_annotation_tracking.py`](file:///d:/AI_CAR/ai/scripts/manage_annotation_tracking.py) | Báo cáo tiến độ (`--summary`), cập nhật trạng thái thẩm định (`--update`), quét đồng bộ số polygon tự động (`--sync`). |
| **3** | **Automated QC Validator** | [`ai/scripts/validate_annotations.py`](file:///d:/AI_CAR/ai/scripts/validate_annotations.py) | Kiểm toán tự động 8 tiêu chí kỹ thuật: Cú pháp JSON, biên ảnh, 8 class hợp lệ, $\ge 3$ đỉnh, tính diện tích Shoelace, trùng lặp IoU $\ge 0.90$. |
| **4** | **YOLO Exporter** | [`ai/scripts/export_pilot_to_yolo.py`](file:///d:/AI_CAR/ai/scripts/export_pilot_to_yolo.py) | Chuyển đổi nhãn đã `APPROVED` sang định dạng YOLOv8-seg, chuẩn hóa tọa độ, không ghi đè dữ liệu cũ. |

---

## 5. KẾT QUẢ KIỂM THỬ BẢO TOÀN DỮ LIỆU (5/5 TEST PASSED)

Đã chạy kiểm thử tự động toàn diện qua kịch bản [`scratch/test_pipeline_and_tools.py`](file:///d:/AI_CAR/scratch/test_pipeline_and_tools.py):

| STT | Hạng mục kiểm thử | Dữ liệu kiểm tra | Kết quả thực tế | Đánh giá |
| :---: | :--- | :--- | :---: | :---: |
| **1** | Kiểm tra cấu trúc thư mục | 5 thư mục thành phần tại `parts_pilot/` | 5/5 tồn tại đầy đủ | <span style="color:green">**PASS**</span> |
| **2** | Kiểm tra cấu hình 8 class | Đối soát `labels.txt` | Đúng 8 lớp chuẩn | <span style="color:green">**PASS**</span> |
| **3** | Kiểm thử QC Validator | Dữ liệu giả lập lỗi cố ý (sai class, thiếu điểm, tràn biên, diện tích $= 0$) | Phát hiện chuẩn xác 100% các lỗi vi phạm | <span style="color:green">**PASS**</span> |
| **4** | Kiểm thử YOLO Exporter | Dữ liệu giả lập đạt chuẩn | Xuất đúng format `<cid> x1 y1...`, tọa độ $[0, 1]$, tạo đủ YAML | <span style="color:green">**PASS**</span> |
| **5** | Bảo toàn dữ liệu gốc | Đối soát SHA256 và `pilot_manifest.csv` | 0 ảnh gốc bị thay đổi, 80 ảnh nguyên vẹn | <span style="color:green">**PASS**</span> |

---

## 6. TÀI LIỆU HƯỚNG DẪN ĐÍNH KÈM

Đã biên soạn cẩm nang hướng dẫn đầy đủ bằng tiếng Việt tại:  
👉 [`docs/HUMAN_ANNOTATION_GUIDE.md`](file:///d:/AI_CAR/docs/HUMAN_ANNOTATION_GUIDE.md)  
Bao gồm:
- Hướng dẫn cài đặt và mở công cụ.
- Quy tắc vẽ đa giác bám sát gân dập nổi cơ khí.
- Cách tách biệt cửa trước/cửa sau, cách khoanh vùng đèn xe và kính xe.
- Quy trình nghiệm thu, phê duyệt và các lỗi thường gặp cần tránh.

---

## 7. KẾT LUẬN VÀ TRẠNG THÁI DỪNG HỆ THỐNG

- **Tuân thủ nghiêm ngặt ràng buộc:**
  - Không sửa ảnh gốc.
  - Không sửa Manifest chính thức.
  - Không retrain hoặc thay đổi mô hình.
  - Không tạo pseudo-label tự động hàng loạt.
  - Không đưa nhãn chưa duyệt vào Ground Truth.
- **Hệ thống hiện tại đã dừng và sẵn sàng:** Chờ người dùng kiểm tra công cụ và phê duyệt trước khi chuyển sang giai đoạn gán nhãn thực tế trên 80 ảnh Pilot.
