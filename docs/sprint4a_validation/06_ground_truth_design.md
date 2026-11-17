# 06 - THIẾT KẾ BỘ GROUND TRUTH CHUẨN ĐỊNH LƯỢNG (GROUND TRUTH SPECIFICATION)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 4A — AI Data Validation & Evaluation Baseline  
**Mục tiêu:** Thiết lập quy chuẩn kỹ thuật (Protocol & Schema) để xây dựng bộ nhãn chuẩn thẩm định cho Damage Detection, Parts Segmentation và Damage-to-Part Mapping.

---

## 1. Quy Ước Lớp Đối Tượng Chuẩn Hóa (Standard Taxonomy)

### 1.1. Danh mục 5 Lớp Hư Hỏng Chuẩn (Damage Classes)
1. `scratch`: Vết cào xước làm xước lớp sơn bóng (Clear coat) hoặc lộ lớp sơn lót.
2. `dent`: Vết móp méo, lõm biến dạng bề mặt kim loại tấm hoặc nhựa cứng.
3. `crack`: Rạn nứt kính, vỡ choá đèn pha/hậu hoặc nứt cản nhựa.
4. `tear_puncture`: Rách thủng vỏ cản nhựa, nắp capo hoặc rách toác thân vỏ.
5. `paint_chipping`: Bong tróc, bong tróc từng mảng sơn để lộ kim loại trần.

### 1.2. Danh mục 8 Bộ Phận Ngoại Thất Chuẩn (Vehicle Parts Classes)
1. `can_truoc`: Cản trước (Front Bumper) bao gồm cả lưới tản nhiệt gắn liền.
2. `can_sau`: Cản sau (Rear Bumper).
3. `nap_capo`: Nắp ca-pô khoang động cơ (Hood / Bonnet).
4. `ve_che_banh`: Vè che bánh xe (Fender / Quarter Panel).
5. `cua_truoc`: Cánh cửa trước (Trái / Phải).
6. `cua_sau`: Cánh cửa sau (Trái / Phải).
7. `kinh_xe`: Kính chắn gió trước/sau và kính cửa hông.
8. `den_xe`: Cụm đèn pha, đèn gầm, đèn hậu và đèn xi-nhan.

---

## 2. Quy Tắc Gán Nhãn Hình Học (Annotation Rules)

### 2.1. Quy tắc vẽ Bounding Box (Hộp bao tổn thất)
- Hộp bao phải **ôm sát tối đa** mép ngoài của vết móp/xước (Tight Bounding Box).
- Tỷ lệ diện tích nền (Background) bên trong hộp bao không được vượt quá $20\%$.
- Với vết xước dài uốn cong chéo, **nghiêm cấm** vẽ một hộp chữ nhật khổng lồ trùm cả xe; bắt buộc phải chia thành các đoạn hộp xước nhỏ liền kề bám theo đường xước.
- Kích thước hộp bao tối thiểu: $10 \times 10$ pixels.

### 2.2. Quy tắc vẽ Polygon (Đa giác phân đoạn bộ phận)
- **TUYỆT ĐỐI KHÔNG DÙNG HỘP CHỮ NHẬT 4 ĐIỂM ĐỂ LÀM MASK BỘ PHẬN**.
- Đa giác phải có tối thiểu **16 đến 40 điểm đỉnh** bám sát theo các đường gân dập nổi và khe hở mép nối (Panel Gaps) của từng cánh cửa, cản xe, vè xe.
- Giữa hai bộ phận liền kề (ví dụ: Cửa trước và Cửa sau), hai đa giác không được phép chồng lấn lên nhau (Zero Overlap at Panel Seams).

### 2.3. Quy tắc Ánh xạ Damage $\rightarrow$ Part (Mapping Ground Truth)
- Mỗi vết hư hỏng chỉ được liên kết với bộ phận mà bề mặt vật lý của nó trực tiếp bị tổn hại.
- Nếu vết xước nằm đè lên đường khe hở giữa 2 cánh cửa, vết xước đó phải được chia thành 2 annotation độc lập tương ứng với 2 bộ phận.
- Nếu ảnh quá cận cảnh và người kiểm duyệt không thể nhận diện được bộ phận bằng mắt thường, gán `located_on_part: "khong_xac_dinh"`.

---

## 3. Cấu Trúc Schema JSON Chuẩn Mẫu Cho `demo/annotations_ground_truth.json`

*(Định nghĩa cấu trúc dữ liệu - Không tự ý điền nhãn giả khi chưa có chuyên viên giám định phê duyệt)*

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AutoClaimAI_GroundTruth",
  "dataset_version": "1.0-sprint4",
  "images": [
    {
      "image_id": 1,
      "file_name": "image copy 4.png",
      "width": 667,
      "height": 1000,
      "view_perspective": "side_right_diagonal",
      "is_closeup": false,
      "parts_ground_truth": [
        {
          "part_id": 1,
          "part_label": "cua_truoc",
          "polygon": [
            [120.5, 340.2],
            [310.4, 345.1],
            [305.2, 720.0],
            [115.0, 715.4]
          ]
        }
      ],
      "damages_ground_truth": [
        {
          "damage_id": 1,
          "damage_type": "dent",
          "bbox_xyxy": [150.2, 410.5, 230.8, 520.1],
          "true_part_id": 1,
          "true_part_name": "cua_truoc",
          "severity": "medium",
          "is_ambiguous": false
        }
      ]
    }
  ]
}
```

---

## 4. Chiến Lược Phân Chia Dữ Liệu Chống Rò Rỉ (Anti-Leakage Strategy)

Để ngăn chặn hiện tượng rò rỉ dữ liệu giữa tập Train và Test:
1. **Chia tách theo Định danh Xe (Vehicle/Case-level Split)**:
   - Toàn bộ các góc chụp của cùng một chiếc xe (hoặc cùng một hồ sơ bồi thường) phải nằm hoàn toàn trong tập Train HOẶC tập Test.
   - Tuyệt đối không phân chia ngẫu nhiên theo ảnh đơn lẻ.
2. **Quy mô tập Test khuyến nghị**:
   - Tối thiểu **150 đến 200 ảnh xe thực tế có nhãn chuẩn Ground Truth** bao gồm đủ 8 góc nhìn: chính diện, sau xe, sườn trái, sườn phải, 4 góc xiên chéo, và ảnh cận cảnh.
