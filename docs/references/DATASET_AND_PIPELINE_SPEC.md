# TÀI LIỆU THAM KHẢO DỮ LIỆU & ĐẶC TẢ PIPELINE (AUTOCLAIM AI - C1SE.79)

## 1. Tập dữ liệu Nhận diện Hư hại Xe (CarDD - Stage 1)
- **Đường dẫn cấu hình**: `ai/configs/cardd_seg.yaml`
- **Đường dẫn dữ liệu**: `ai/data/yolo/`
- **Trọng số mô hình**: `ai/models/weights/damage_best.pt`
- **Danh mục 6 lớp hư hại chuẩn**:
  | Class ID | Mã kỹ thuật | Tên tiếng Việt | Nhóm vật liệu hợp lệ |
  | :---: | :--- | :--- | :--- |
  | `0` | `dent` | Móp, lõm thân vỏ | Thân vỏ tôn/nhựa (`can_truoc`, `can_sau`, `nap_capo`, `ve_che_banh`, `cua_hong`) |
  | `1` | `scratch` | Trầy, xước sơn | Thân vỏ tôn/nhựa, vỏ gương (`kinh_chieu_hau`) |
  | `2` | `crack` | Nứt, rách | Cản nhựa, thân vỏ, kính |
  | `3` | `glass_shatter` | Vỡ kính | Nhóm kính (`kinh_lai`, `kinh_chieu_hau`) hoặc chóa đèn pha (`den_xe` qua *Headlamp Shatter Guard*) |
  | `4` | `lamp_broken` | Vỡ đèn xe | Cụm đèn xe (`den_xe`) |
  | `5` | `tire_flat` | Xẹp/hỏng lốp | Bánh xe / Lốp xe (`wheel` / `tire`) |

---

## 2. Tập dữ liệu Phân đoạn Bộ phận Xe (7 Bộ phận chuẩn - Stage 2)
- **Tài liệu đặc tả gốc**: `docs/references/Phan_doan_bo_phan_xe.docx`
- **Đường dẫn cấu hình**: `ai/configs/parts_seg.yaml`
- **Đường dẫn dữ liệu**: `ai/data/parts/`
- **Trọng số mô hình**: `ai/models/weights/parts_seg.pt`
- **Danh mục 7 bộ phận xe chuẩn hóa**:
  | Class ID | Mã kỹ thuật (`part_key`) | Tên hiển thị (`part_name`) | Mô tả nhận diện |
  | :---: | :--- | :--- | :--- |
  | `0` | `can_truoc` | `cản trước` | Phần nhựa/kim loại phía trước thấp nhất đầu xe, nằm dưới lưới tản nhiệt và cụm đèn pha |
  | `1` | `can_sau` | `cản sau` | Phần nhựa/kim loại phía dưới cùng đuôi xe, kéo dài sau vòm bánh sau |
  | `2` | `nap_capo` | `nắp capo` | Tấm kim loại lớn che khoang động cơ phía trước kính chắn gió |
  | `3` | `ve_che_banh` | `vè che bánh` | Phần thân vỏ bao quanh phía trên hốc bánh xe (tai xe / hông vè) |
  | `4` | `cua_hong` | `cửa hông` | Cửa trước và cửa sau hai bên thân xe |
  | `5` | `kinh_chieu_hau` | `kính chiếu hậu` | Gương chiếu hậu gắn hai bên thân/cửa trước |
  | `6` | `kinh_lai` | `kính lái` | Kính chắn gió lớn phía trước cabin |
