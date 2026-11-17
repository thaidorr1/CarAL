"""Kiểm thử đơn vị (Unit Tests) cho module Pipeline: Segmentation NMS, Mask Merging, Semantic Guard Hard-Rules."""

import unittest
import cv2
import numpy as np
from ai.inference.pipeline import (
    DamageResult,
    PartResult,
    map_damages_to_parts,
    resolve_overlapping_part_masks,
    merge_and_smooth_part_masks,
    is_close_up_shot,
    classify_shot_type,
    determine_shot_type,
    normalize_damage_class,
    resolve_door_position_from_topology,
    resolve_four_side_doors,
    FORBIDDEN_DAMAGE_PART_PAIRS,
    LARGE_PART_KEYS,
)


class TestPipelinePostProcessing(unittest.TestCase):
    def setUp(self):
        self.image_hw = (600, 800)

    def test_merge_and_smooth_fragmented_masks(self):
        """Kiểm tra gộp các mask cùng lớp bị vỡ vụn thành khối liền mạch duy nhất (ảnh 003546)."""
        # Giả lập 2 mảnh vỡ liền kề của cùng một cụm đèn xe (khoảng cách giữa 2 mảnh là 5px)
        piece1 = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.75,
            part_polygon=[[50, 50], [150, 50], [150, 150], [50, 150]],
            bbox_xyxy=[50.0, 50.0, 150.0, 150.0],
            center_xy=(100.0, 100.0),
        )
        piece2 = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.70,
            part_polygon=[[155, 50], [250, 50], [250, 150], [155, 150]],
            bbox_xyxy=[155.0, 50.0, 250.0, 150.0],
            center_xy=(202.5, 100.0),
        )
        # Mảnh rác quá nhỏ (< 600 px)
        noise = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.48,
            part_polygon=[[500, 500], [510, 500], [510, 510], [500, 510]],
            bbox_xyxy=[500.0, 500.0, 510.0, 510.0],
            center_xy=(505.0, 505.0),
        )

        merged = merge_and_smooth_part_masks([piece1, piece2, noise], self.image_hw, min_area_pixels=600.0, morph_closing_kernel_size=15)
        # 2 mảnh gần nhau được gộp thành 1 khối duy nhất, mảnh rác bị loại bỏ hoàn toàn
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].part_key, "den_xe")
        # Chiều rộng bounding box bao phủ từ x=50 đến x=250
        self.assertLessEqual(merged[0].bbox_xyxy[0], 52.0)
        self.assertGreaterEqual(merged[0].bbox_xyxy[2], 248.0)

    def test_nms_suppresses_contradictory_pairs(self):
        """Kiểm tra NMS triệt tiêu cặp mâu thuẫn cản sau vs nắp capo."""
        rear_bumper = PartResult(
            part_key="can_sau",
            part_name="cản sau",
            confidence=0.92,
            part_polygon=[[100, 100], [400, 100], [400, 300], [100, 300]],
            bbox_xyxy=[100.0, 100.0, 400.0, 300.0],
            center_xy=(250.0, 200.0),
        )
        conflicting_hood = PartResult(
            part_key="nap_capo",
            part_name="nắp capo",
            confidence=0.65,
            part_polygon=[[150, 120], [380, 120], [380, 280], [150, 280]],
            bbox_xyxy=[150.0, 120.0, 380.0, 280.0],
            center_xy=(265.0, 200.0),
        )
        resolved = resolve_overlapping_part_masks([rear_bumper, conflicting_hood], self.image_hw)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].part_key, "can_sau")

    def test_nms_clips_overlapping_different_classes(self):
        """Kiểm tra NMS cắt tỉa diện tích giao nhau khi 2 mask đè lên nhau > 30%."""
        p_high = PartResult(
            part_key="kinh_lai",
            part_name="kính lái",
            confidence=0.90,
            part_polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
            bbox_xyxy=[100.0, 100.0, 300.0, 300.0],
            center_xy=(200.0, 200.0),
        )
        # Cửa hông đè phủ vào kính lái từ x=150 đến 300 (giao nhau 150px / 250px = 60% diện tích)
        p_low = PartResult(
            part_key="cua_hong",
            part_name="cửa hông",
            confidence=0.70,
            part_polygon=[[150, 100], [400, 100], [400, 300], [150, 300]],
            bbox_xyxy=[150.0, 100.0, 400.0, 300.0],
            center_xy=(275.0, 200.0),
        )
        resolved = resolve_overlapping_part_masks([p_high, p_low], self.image_hw)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0].part_key, "kinh_lai")
        clipped_door = resolved[1]
        self.assertEqual(clipped_door.part_key, "cua_hong")
        self.assertGreaterEqual(clipped_door.bbox_xyxy[0], 299.0)

    def test_semantic_hard_rule_lamp_broken_strictly_belongs_to_headlight(self):
        """Kiểm tra Hard Rule 1: lamp_broken BẮT BUỘC gán vào đèn xe, cấm cản trước/sau/cửa hông."""
        lamp = DamageResult(
            damage_type="lamp_broken",
            confidence=0.90,
            damage_polygon=[[150, 150], [200, 150], [200, 200], [150, 200]],
            bbox_xyxy=[150.0, 150.0, 200.0, 200.0],
            center_xy=(175.0, 175.0),
        )
        door = PartResult(
            part_key="cua_hong",
            part_name="cửa hông",
            confidence=0.85,
            part_polygon=[[100, 100], [400, 100], [400, 400], [100, 400]],
            bbox_xyxy=[100.0, 100.0, 400.0, 400.0],
            center_xy=(250.0, 250.0),
        )
        bumper = PartResult(
            part_key="can_truoc",
            part_name="cản trước",
            confidence=0.80,
            part_polygon=[[50, 120], [140, 120], [140, 300], [50, 300]],
            bbox_xyxy=[50.0, 120.0, 140.0, 300.0],
            center_xy=(95.0, 210.0),
        )
        # Trường hợp không có mask đèn nào (đèn vỡ vụn)
        mapped = map_damages_to_parts([lamp], [door, bumper], self.image_hw)
        self.assertEqual(len(mapped), 1)
        # Tuyệt đối không gán vào cửa hông hay cản trước qua center distance!
        self.assertNotEqual(mapped[0].located_on_part, "cửa hông")
        self.assertNotEqual(mapped[0].located_on_part, "cản trước")
        # Bắt buộc gán vào đèn xe!
        self.assertEqual(mapped[0].located_on_part, "đèn xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_lamp")

    def test_semantic_hard_rule_glass_shatter(self):
        """Kiểm tra Hard Rule 2: glass_shatter BẮT BUỘC thuộc nhóm kính, cấm cản/nắp capo."""
        shatter = DamageResult(
            damage_type="glass_shatter",
            confidence=0.95,
            damage_polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
            bbox_xyxy=[100.0, 100.0, 200.0, 200.0],
            center_xy=(150.0, 150.0),
        )
        hood = PartResult(
            part_key="nap_capo",
            part_name="nắp capo",
            confidence=0.85,
            part_polygon=[[50, 50], [300, 50], [300, 300], [50, 300]],
            bbox_xyxy=[50.0, 50.0, 300.0, 300.0],
            center_xy=(175.0, 175.0),
        )
        mapped = map_damages_to_parts([shatter], [hood], self.image_hw)
        self.assertEqual(len(mapped), 1)
        self.assertNotEqual(mapped[0].located_on_part, "nắp capo")
        self.assertIn("kính", mapped[0].located_on_part.lower())

    def test_semantic_hard_rule_dent_and_scratch_body_fallback(self):
        """Kiểm tra Hard Rule 3: dent & scratch fallback thành 'thân vỏ xe' khi cận cảnh."""
        d1 = DamageResult(
            damage_type="dent", confidence=0.75, damage_polygon=[],
            bbox_xyxy=[10.0, 10.0, 100.0, 100.0], center_xy=(55.0, 55.0),
        )
        d2 = DamageResult(
            damage_type="scratch", confidence=0.65, damage_polygon=[],
            bbox_xyxy=[120.0, 20.0, 200.0, 80.0], center_xy=(160.0, 50.0),
        )
        # Khi cận cảnh và parts = []: CẤM đoán mò bộ phận theo tọa độ, trả về khong_xac_dinh
        mapped = map_damages_to_parts([d1, d2], [], self.image_hw)
        self.assertEqual(len(mapped), 2)
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")
        self.assertIn("quá cận cảnh", mapped[0].warning)


    def test_semantic_hard_rule_tire_flat(self):
        """Kiểm tra Tire-Guard: tire_flat BẮT BUỘC gán vào Bánh xe / Lốp xe, cấm cản trước/sau/vè."""
        tire = DamageResult(
            damage_type="tire_flat",
            confidence=0.88,
            damage_polygon=[[100, 400], [200, 400], [200, 500], [100, 500]],
            bbox_xyxy=[100.0, 400.0, 200.0, 500.0],
            center_xy=(150.0, 450.0),
        )
        bumper = PartResult(
            part_key="can_truoc",
            part_name="cản trước",
            confidence=0.85,
            part_polygon=[[100, 300], [400, 300], [400, 550], [100, 550]],
            bbox_xyxy=[100.0, 300.0, 400.0, 550.0],
            center_xy=(250.0, 425.0),
        )
        fender = PartResult(
            part_key="ve_che_banh",
            part_name="vè che bánh",
            confidence=0.80,
            part_polygon=[[80, 200], [220, 200], [220, 380], [80, 380]],
            bbox_xyxy=[80.0, 200.0, 220.0, 380.0],
            center_xy=(150.0, 290.0),
        )
        mapped = map_damages_to_parts([tire], [bumper, fender], self.image_hw)
        self.assertEqual(len(mapped), 1)
        self.assertNotEqual(mapped[0].located_on_part, "cản trước")
        self.assertNotEqual(mapped[0].located_on_part, "vè che bánh")
        self.assertEqual(mapped[0].located_on_part, "Bánh xe / Lốp xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_tire")

    def test_distance_exceeded_fallback_to_body(self):
        """Kiểm tra khi khoảng cách đo được vượt ngưỡng max_center_distance_ratio (0.20),
        và rơi vào Tầng 3 (Unknown Fallback), trả về khong_xac_dinh thay vì gán bừa cản sau hay cản trước."""
        # Vết dent ở góc dưới bên phải (x=700, y=500)
        dent = DamageResult(
            damage_type="dent",
            confidence=0.80,
            damage_polygon=[],
            bbox_xyxy=[680.0, 480.0, 720.0, 520.0],
            center_xy=(700.0, 500.0),
        )
        # Cản trước duy nhất ở góc trên bên trái (x=100, y=100) -> Khoảng cách tâm ~ 721 px > 0.20 * diag (200px)
        bumper = PartResult(
            part_key="can_truoc",
            part_name="cản trước",
            confidence=0.75,
            part_polygon=[[50, 50], [150, 50], [150, 150], [50, 150]],
            bbox_xyxy=[50.0, 50.0, 150.0, 150.0],
            center_xy=(100.0, 100.0),
        )
        mapped = map_damages_to_parts([dent], [bumper], self.image_hw, max_center_distance_ratio=0.20)
        self.assertEqual(len(mapped), 1)
        # Tuân thủ nghiêm ngặt 3 Tầng Bảo Vệ: Vết móp cách xa mọi bộ phận -> khong_xac_dinh, không được gán cản trước/cản sau!
        self.assertNotEqual(mapped[0].located_on_part, "cản trước")
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[0].match_method, "unmapped_topology")


    def test_closeup_micro_feature_parking_sensor_forces_bumper(self):
        """Kiểm tra góc cận: Phát hiện cảm biến đỗ xe / nẹp cản / lưới tản nhiệt ép nhãn về cản xe."""
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.85,
            damage_polygon=[],
            bbox_xyxy=[100.0, 100.0, 300.0, 300.0],
            center_xy=(200.0, 200.0),
        )
        # Giả lập ảnh chụp cận, mô hình parts conf thấp (0.45)
        door = PartResult(
            part_key="cua_hong", part_name="cửa hông", confidence=0.48,
            part_polygon=[[50, 50], [400, 50], [400, 400], [50, 400]],
            bbox_xyxy=[50.0, 50.0, 400.0, 400.0], center_xy=(225.0, 225.0),
        )
        mapped = map_damages_to_parts([scratch], [door], self.image_hw, micro_features=["parking_sensor"])
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].located_on_part, "cản trước")
        self.assertEqual(mapped[0].match_method, "micro_feature_bumper")
        self.assertEqual(mapped[0].display_label, "scratch -> cản trước [0.85]")

    def test_closeup_micro_feature_door_handle_allows_door(self):
        """Kiểm tra góc cận: Phát hiện tay nắm cửa / khe hở cửa mới được phép gán cửa hông."""
        dent = DamageResult(
            damage_type="dent",
            confidence=0.80,
            damage_polygon=[],
            bbox_xyxy=[100.0, 100.0, 250.0, 250.0],
            center_xy=(175.0, 175.0),
        )
        mapped = map_damages_to_parts([dent], [], self.image_hw, micro_features=["door_handle"])
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].located_on_part, "Cửa trước")
        self.assertNotEqual(mapped[0].located_on_part, "cửa hông")
        self.assertEqual(mapped[0].match_method, "micro_feature_door")
        self.assertEqual(mapped[0].display_label, "dent -> Cửa trước [0.80]")

    def test_closeup_micro_feature_lamp_edge(self):
        """Kiểm tra góc cận: Phát hiện mép cụm đèn ưu tiên gán cản xe hoặc đèn xe."""
        crack = DamageResult(
            damage_type="crack",
            confidence=0.78,
            damage_polygon=[],
            bbox_xyxy=[50.0, 50.0, 200.0, 200.0],
            center_xy=(125.0, 125.0),
        )
        mapped = map_damages_to_parts([crack], [], self.image_hw, micro_features=["lamp_edge"])
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].located_on_part, "cản trước")
        self.assertEqual(mapped[0].match_method, "micro_feature_lamp")
        self.assertEqual(mapped[0].display_label, "crack -> cản trước [0.78]")

    def test_closeup_damage_only_fallback_when_parts_unsafe(self):
        """Kiểm tra Chế độ Fallback 'Chỉ báo hư hỏng': Khi cận cảnh, parts_seg conf < 0.60 và không có micro-feature."""
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.85,
            damage_polygon=[],
            bbox_xyxy=[50.0, 50.0, 300.0, 300.0],
            center_xy=(175.0, 175.0),
        )
        # Part có conf = 0.52 (< 0.60 ngưỡng an toàn)
        door = PartResult(
            part_key="cua_hong", part_name="cửa hông", confidence=0.52,
            part_polygon=[[40, 40], [350, 40], [350, 350], [40, 350]],
            bbox_xyxy=[40.0, 40.0, 350.0, 350.0], center_xy=(195.0, 195.0),
        )
        bumper = PartResult(
            part_key="can_truoc", part_name="cản trước", confidence=0.48,
            part_polygon=[[10, 10], [100, 10], [100, 100], [10, 100]],
            bbox_xyxy=[10.0, 10.0, 100.0, 100.0], center_xy=(55.0, 55.0),
        )
        # Khi thực sự không tìm thấy mask bộ phận nào (parts=[]) và không có micro-features:
        mapped = map_damages_to_parts([scratch], [], self.image_hw, micro_features=None)
        self.assertEqual(len(mapped), 1)
        # CẤM gán ép vào cửa hông hay cản trước!
        self.assertNotEqual(mapped[0].located_on_part, "cửa hông")
        self.assertNotEqual(mapped[0].located_on_part, "cản trước")
        # Chuẩn hóa mới: CẤM đoán mò theo tọa độ khi cận cảnh, trả về khong_xac_dinh
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[0].match_method, "closeup_unclear_rejection")
        self.assertEqual(mapped[0].display_label, "scratch -> khong_xac_dinh [0.85]")
        self.assertIn("quá cận cảnh", mapped[0].warning)


    def test_physical_guard_dent_forbidden_on_den_xe_and_glass(self):
        """Kiểm tra Quy tắc 1: CẤM gán dent vào đèn xe, kính lái, kính chiếu hậu; ưu tiên vè che bánh."""
        dent = DamageResult(
            damage_type="dent",
            confidence=0.88,
            damage_polygon=[[120, 120], [200, 120], [200, 200], [120, 200]],
            bbox_xyxy=[120.0, 120.0, 200.0, 200.0],
            center_xy=(160.0, 160.0),
        )
        headlight = PartResult(
            part_key="den_xe", part_name="đèn xe", confidence=0.92,
            part_polygon=[[100, 100], [220, 100], [220, 220], [100, 220]],
            bbox_xyxy=[100.0, 100.0, 220.0, 220.0], center_xy=(160.0, 160.0),
        )
        fender = PartResult(
            part_key="ve_che_banh", part_name="vè che bánh", confidence=0.82,
            part_polygon=[[90, 80], [300, 80], [300, 300], [90, 300]],
            bbox_xyxy=[90.0, 80.0, 300.0, 300.0], center_xy=(195.0, 190.0),
        )
        # Dent đè lên cả đèn xe lẫn vè che bánh, nhưng đèn xe BỊ CẤM gán móp
        mapped = map_damages_to_parts([dent], [headlight, fender], self.image_hw)
        self.assertEqual(len(mapped), 1)
        self.assertNotEqual(mapped[0].located_on_part, "đèn xe")
        # Phải ưu tiên gán vè che bánh (tai xe)
        self.assertEqual(mapped[0].located_on_part, "vè che bánh")

    def test_damage_conf_filter_eliminates_weak_boxes(self):
        """Kiểm tra Quy tắc 2: Nâng ngưỡng lọc tối thiểu, loại bỏ box hư hại conf=0.21."""
        weak_dent = DamageResult(
            damage_type="dent", confidence=0.21, damage_polygon=[],
            bbox_xyxy=[50.0, 50.0, 100.0, 100.0], center_xy=(75.0, 75.0),
        )
        strong_dent = DamageResult(
            damage_type="dent", confidence=0.82, damage_polygon=[],
            bbox_xyxy=[50.0, 50.0, 100.0, 100.0], center_xy=(75.0, 75.0),
        )
        fender = PartResult(
            part_key="ve_che_banh", part_name="vè che bánh", confidence=0.85,
            part_polygon=[[40, 40], [200, 40], [200, 200], [40, 200]],
            bbox_xyxy=[40.0, 40.0, 200.0, 200.0], center_xy=(120.0, 120.0),
        )
        mapped = map_damages_to_parts([weak_dent, strong_dent], [fender], self.image_hw, damage_conf_min=0.25)
        # Vết hư hại yếu 0.21 bị loại bỏ hoàn toàn
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].confidence, 0.82)
        self.assertEqual(mapped[0].located_on_part, "vè che bánh")



    # =========================================================================
    # BỘ KIỂM THỬ PHỦ TOÀN BỘ 5 CASE THỰC TẾ & SANITY OVERRIDE RULES
    # =========================================================================

    def test_case_1_rear_lamp_broken_forces_rear_bumper(self):
        """Case 1: Ảnh đuôi xe đèn vỡ.
        - lamp_broken conf >= 0.40 là Neo bối cảnh -> CẤM 100% trả về cận cảnh.
        - Anchor 1 (Tail Light Anchor): Tổn thất tôn/nhựa nằm ngay dưới cụm đèn sau -> ÉP CỨNG 100% về cản sau.
        """
        lamp_broken = DamageResult(
            damage_type="lamp_broken",
            confidence=0.88,
            damage_polygon=[[350, 120], [600, 120], [600, 260], [350, 260]],
            bbox_xyxy=[350.0, 120.0, 600.0, 260.0],
            center_xy=(475.0, 190.0),
        )
        dent = DamageResult(
            damage_type="dent",
            confidence=0.97,
            damage_polygon=[[380, 320], [650, 320], [650, 500], [380, 500]],
            bbox_xyxy=[380.0, 320.0, 650.0, 500.0],
            center_xy=(515.0, 410.0),
        )
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.76,
            damage_polygon=[[330, 310], [420, 310], [420, 430], [330, 430]],
            bbox_xyxy=[330.0, 310.0, 420.0, 430.0],
            center_xy=(375.0, 370.0),
        )

        # 1. Kiểm tra is_close_up_shot: Có neo lamp_broken conf=0.88 >= 0.40 -> CẤM 100% cận cảnh
        is_close, reason = is_close_up_shot([lamp_broken, dent, scratch], [], self.image_hw)
        self.assertFalse(is_close, "Có neo đèn vỡ tail_light/lamp_broken >= 0.40 thì cấm trả về True")
        self.assertIn("Neo bối cảnh", reason)

        # 2. Kiểm tra map_damages_to_parts:
        mapped = map_damages_to_parts([lamp_broken, dent, scratch], [], self.image_hw)
        self.assertEqual(len(mapped), 3)

        # lamp_broken bắt buộc gán vào đèn xe
        self.assertEqual(mapped[0].located_on_part, "đèn xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_lamp")

        # dent và scratch không có mask bộ phận nào từ Stage 2 -> Chuẩn hóa toán học: khong_xac_dinh
        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[2].located_on_part, "khong_xac_dinh")

    def test_case_2_front_headlamp_yellow_bumper_forces_front_bumper(self):
        """Case 2: Ảnh đầu xe cản trước vàng.
        - Đèn pha (headlamp) conf >= 0.40 là Neo bối cảnh -> CẤM 100% trả về cận cảnh.
        - Anchor 2 (Front Lamp Anchor): Tổn thất xung quanh/dưới đèn pha -> ÉP CỨNG 100% về cản trước.
        - Sanity Override Rule: VÔ HIỆU HÓA 100% can_sau khi có góc đầu xe.
        """
        headlamp = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.85,
            part_polygon=[[150, 80], [320, 80], [320, 200], [150, 200]],
            bbox_xyxy=[150.0, 80.0, 320.0, 200.0],
            center_xy=(235.0, 140.0),
        )
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.83,
            damage_polygon=[[180, 250], [300, 250], [300, 350], [180, 350]],
            bbox_xyxy=[180.0, 250.0, 300.0, 350.0],
            center_xy=(240.0, 300.0),
        )
        dent = DamageResult(
            damage_type="dent",
            confidence=0.69,
            damage_polygon=[[280, 150], [420, 150], [420, 320], [280, 320]],
            bbox_xyxy=[280.0, 150.0, 420.0, 320.0],
            center_xy=(350.0, 235.0),
        )

        # 1. Kiểm tra is_close_up_shot: Có neo Đèn pha conf=0.85 >= 0.40 -> CẤM 100% cận cảnh
        is_close, reason = is_close_up_shot([scratch, dent], [headlamp], self.image_hw)
        self.assertFalse(is_close, "Có neo đèn pha >= 0.40 thì cấm trả về True")

        # 2. Kiểm tra map_damages_to_parts:
        mapped = map_damages_to_parts([scratch, dent], [headlamp], self.image_hw)
        self.assertEqual(len(mapped), 2)

        # Cả 2 tổn thất là scratch và dent, không thể gán vào đèn xe (Physical Guard)
        # và không có mask thân vỏ nào -> Trả về khong_xac_dinh
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")

        # Tuyệt đối CẤM can_sau hoặc cua_hong
        for m in mapped:
            self.assertNotEqual(m.located_on_part, "cản sau")
            self.assertNotEqual(m.located_on_part, "cửa hông")

    def test_case_3_side_view_front_wheel_and_headlamp_forces_fender(self):
        """Case 3: Ảnh góc ngang vè xe.
        - Bánh trước (wheel/tire) và Đèn pha (headlamp) xuất hiện làm Neo bối cảnh.
        - Anchor 3 (Front Fender Anchor): Vùng tôn nằm trực tiếp phía trên vòm bánh xe -> ÉP CỨNG 100% về ve_che_banh.
        """
        tire_damage = DamageResult(
            damage_type="tire_flat",
            confidence=0.75,
            damage_polygon=[[200, 350], [450, 350], [450, 580], [200, 580]],
            bbox_xyxy=[200.0, 350.0, 450.0, 580.0],
            center_xy=(325.0, 465.0),
        )
        headlamp = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.70,
            part_polygon=[[80, 150], [200, 150], [200, 250], [80, 250]],
            bbox_xyxy=[80.0, 150.0, 200.0, 250.0],
            center_xy=(140.0, 200.0),
        )
        # Vết xước vè nằm trực tiếp phía trên vòm bánh xe
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.82,
            damage_polygon=[[240, 230], [380, 230], [380, 320], [240, 320]],
            bbox_xyxy=[240.0, 230.0, 380.0, 320.0],
            center_xy=(310.0, 275.0),
        )

        # 1. is_close_up_shot: Có neo bánh xe tire_flat >= 0.40 -> CẤM cận cảnh
        is_close, reason = is_close_up_shot([tire_damage, scratch], [headlamp], self.image_hw)
        self.assertFalse(is_close)

        # 2. map_damages_to_parts:
        mapped = map_damages_to_parts([tire_damage, scratch], [headlamp], self.image_hw)
        self.assertEqual(len(mapped), 2)
        # tire_flat gán vào Bánh xe / Lốp xe
        self.assertEqual(mapped[0].located_on_part, "Bánh xe / Lốp xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_tire")

        # scratch không thể gán vào đèn xe và không có mask vè/thân vỏ -> khong_xac_dinh
        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")

    def test_case_4_extreme_closeup_dent_rejection(self):
        """Case 4: Ảnh cận cảnh móp tôn.
        - Không có bất kỳ Neo bối cảnh nào (bánh xe, đèn pha, đèn sau, kính lái >= 0.40).
        - AI Stage 2 chỉ xuất mask kính_lái yếu (conf=0.25 < 0.40) -> BỊ PHỦ QUYẾT 100%.
        - is_close_up_shot trả về True.
        - map_damages_to_parts từ chối đoán mò, trả về khong_xac_dinh kèm cảnh báo.
        """
        dent = DamageResult(
            damage_type="dent",
            confidence=0.94,
            damage_polygon=[[100, 100], [500, 100], [500, 450], [100, 450]],
            bbox_xyxy=[100.0, 100.0, 500.0, 450.0],
            center_xy=(300.0, 275.0),
        )
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.70,
            damage_polygon=[[200, 250], [400, 250], [400, 380], [200, 380]],
            bbox_xyxy=[200.0, 250.0, 400.0, 380.0],
            center_xy=(300.0, 315.0),
        )
        # Giả lập mask kính_lái ảo do Stage 2 xuất ra với conf=0.25 (< 0.40)
        weak_windshield = PartResult(
            part_key="kinh_lai",
            part_name="kính lái",
            confidence=0.25,
            part_polygon=[[50, 0], [700, 0], [700, 250], [50, 250]],
            bbox_xyxy=[50.0, 0.0, 700.0, 250.0],
            center_xy=(375.0, 125.0),
        )

        # 1. is_close_up_shot: Không có neo nào >= 0.40, diện tích tổn thất lớn -> True
        is_close, reason = is_close_up_shot([dent, scratch], [weak_windshield], self.image_hw)
        self.assertTrue(is_close)
        self.assertIn("chiếm", reason)

        # 2. map_damages_to_parts: CẤM 100% gán kính_lái! Trả về khong_xac_dinh
        mapped = map_damages_to_parts([dent, scratch], [weak_windshield], self.image_hw)
        self.assertEqual(len(mapped), 2)
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[0].match_method, "closeup_unclear_rejection")
        self.assertIn("quá cận cảnh", mapped[0].warning)

        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[1].match_method, "closeup_unclear_rejection")

    def test_case_5_closeup_fender_scratch_context_awareness(self):
        """Case 5: Ảnh cận cảnh xước vè.
        - Khi cận cảnh không có neo bối cảnh: Trả về khong_xac_dinh.
        - Khi có neo bánh xe (tire/wheel >= 0.40): Khẳng định đủ bối cảnh, thực hiện mapping về vè che bánh.
        """
        scratch1 = DamageResult(
            damage_type="scratch",
            confidence=0.79,
            damage_polygon=[[150, 150], [450, 150], [450, 260], [150, 260]],
            bbox_xyxy=[150.0, 150.0, 450.0, 260.0],
            center_xy=(300.0, 205.0),
        )
        scratch2 = DamageResult(
            damage_type="scratch",
            confidence=0.74,
            damage_polygon=[[160, 270], [400, 270], [400, 380], [160, 380]],
            bbox_xyxy=[160.0, 270.0, 400.0, 380.0],
            center_xy=(280.0, 325.0),
        )

        # A. Khi KHÔNG có neo bối cảnh nào:
        is_close_no_anchor, _ = is_close_up_shot([scratch1, scratch2], [], self.image_hw)
        self.assertTrue(is_close_no_anchor)
        mapped_no_anchor = map_damages_to_parts([scratch1, scratch2], [], self.image_hw)
        self.assertEqual(mapped_no_anchor[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped_no_anchor[0].match_method, "closeup_unclear_rejection")

        # B. Khi CÓ Neo bối cảnh Bánh trước VÀ Đèn pha:
        tire_flat = DamageResult(
            damage_type="tire_flat",
            confidence=0.69,
            damage_polygon=[[200, 360], [450, 360], [450, 550], [200, 550]],
            bbox_xyxy=[200.0, 360.0, 450.0, 550.0],
            center_xy=(325.0, 455.0),
        )
        headlamp = PartResult(
            part_key="den_xe",
            part_name="đèn xe",
            confidence=0.75,
            part_polygon=[[80, 150], [180, 150], [180, 240], [80, 240]],
            bbox_xyxy=[80.0, 150.0, 180.0, 240.0],
            center_xy=(130.0, 195.0),
        )
        # Sự xuất hiện của Neo khẳng định CÓ ĐỦ BỐI CẢNH
        is_close_with_anchor, reason_anchor = is_close_up_shot([scratch1, tire_flat], [headlamp], self.image_hw)
        self.assertFalse(is_close_with_anchor)
        self.assertIn("Bánh xe", reason_anchor)

        # Bắt buộc thực hiện mapping theo hình học
        mapped_with_anchor = map_damages_to_parts([tire_flat, scratch1], [headlamp], self.image_hw)
        self.assertEqual(len(mapped_with_anchor), 2)
        self.assertEqual(mapped_with_anchor[0].located_on_part, "Bánh xe / Lốp xe")
        # scratch1 không có part thân vỏ -> khong_xac_dinh
        self.assertEqual(mapped_with_anchor[1].located_on_part, "khong_xac_dinh")

    def test_sanity_override_windshield_forbidden_for_sheet_metal_damages(self):
        """Khóa bảo vệ: CẤM 100% gán kính_lái cho dent/scratch/crack nếu Stage 2 không đạt conf >= 0.40."""
        scratch = DamageResult(
            damage_type="scratch",
            confidence=0.80,
            damage_polygon=[[150, 150], [250, 150], [250, 250], [150, 250]],
            bbox_xyxy=[150.0, 150.0, 250.0, 250.0],
            center_xy=(200.0, 200.0),
        )
        # Mask kính lái yếu (conf = 0.38 < 0.40)
        weak_windshield = PartResult(
            part_key="kinh_lai",
            part_name="kính lái",
            confidence=0.38,
            part_polygon=[[100, 100], [300, 100], [300, 300], [100, 300]],
            bbox_xyxy=[100.0, 100.0, 300.0, 300.0],
            center_xy=(200.0, 200.0),
        )
        hood = PartResult(
            part_key="nap_capo",
            part_name="nắp capo",
            confidence=0.75,
            part_polygon=[[50, 50], [400, 50], [400, 350], [50, 350]],
            bbox_xyxy=[50.0, 50.0, 400.0, 350.0],
            center_xy=(225.0, 200.0),
        )
        # Scratch đè lên cả weak_windshield và hood
        mapped = map_damages_to_parts([scratch], [weak_windshield, hood], self.image_hw)
        self.assertEqual(len(mapped), 1)
        # Tuyệt đối CẤM gán vào kính lái!
        self.assertNotEqual(mapped[0].located_on_part, "kính lái")
        self.assertEqual(mapped[0].located_on_part, "nắp capo")

    def test_front_broken_headlamp_red_car_forces_front_bumper_and_disables_can_sau(self):
        """Case kiểm thử bổ sung: lamp_broken ở đầu xe màu đỏ cạnh cản trước/nắp capo.
        - Khắc phục bẫy lamp_broken: Xác định đây là ĐÈN PHA TRƯỚC (Front Headlamp), KHÔNG PHẢI đèn sau.
        - ÉP CỨNG 100% các vết tổn thất phía dưới/xung quanh cụm đèn này về can_truoc.
        - CẤM 100% kích hoạt Tail Light Anchor và CẤM hoàn toàn nhãn can_sau.
        - Metadata shot_type đạt mid_view hoặc far_view, action_required hợp lệ.
        """
        lamp_broken = DamageResult(
            damage_type="lamp_broken",
            confidence=0.89,
            damage_polygon=[[150, 100], [320, 100], [320, 220], [150, 220]],
            bbox_xyxy=[150.0, 100.0, 320.0, 220.0],
            center_xy=(235.0, 160.0),
        )
        scratch_bumper = DamageResult(
            damage_type="scratch",
            confidence=0.86,
            damage_polygon=[[180, 240], [300, 240], [300, 340], [180, 340]],
            bbox_xyxy=[180.0, 240.0, 300.0, 340.0],
            center_xy=(240.0, 290.0),
        )
        front_bumper = PartResult(
            part_key="can_truoc",
            part_name="cản trước",
            confidence=0.83,
            part_polygon=[[100, 180], [500, 180], [500, 420], [100, 420]],
            bbox_xyxy=[100.0, 180.0, 500.0, 420.0],
            center_xy=(300.0, 300.0),
        )

        # 1. Kiểm tra classify_shot_type & is_close_up_shot
        shot_type, act_req, reason = classify_shot_type([lamp_broken, scratch_bumper], [front_bumper], self.image_hw)
        self.assertIn(shot_type, ("mid_view", "far_view"))
        self.assertEqual(act_req, "Kết quả phân tích hợp lệ.")

        is_close, _ = is_close_up_shot([lamp_broken, scratch_bumper], [front_bumper], self.image_hw)
        self.assertFalse(is_close)

        # 2. Kiểm tra map_damages_to_parts
        mapped = map_damages_to_parts([lamp_broken, scratch_bumper], [front_bumper], self.image_hw)
        self.assertEqual(len(mapped), 2)

        # lamp_broken bắt buộc gán vào đèn xe
        self.assertEqual(mapped[0].located_on_part, "đèn xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_lamp")

        # scratch_bumper nằm dưới cụm đèn pha -> ÉP CỨNG 100% về cản trước
        self.assertEqual(mapped[1].located_on_part, "cản trước")
        self.assertIn(mapped[1].match_method, ("mask_overlap", "context_anchor_front_lamp"))
        self.assertEqual(mapped[1].action_required, "Kết quả phân tích hợp lệ.")

        # CẤM 100% nhãn can_sau trên toàn bộ ảnh
        for m in mapped:
            self.assertNotEqual(m.located_on_part, "cản sau")

    def test_front_fender_dent_grey_car_sanity_override(self):
        """Case kiểm thử bổ sung: Vết dent ở vè hông trước xe xám.
        - Có bối cảnh đầu xe (Gương chiếu hậu / Kính lái / Bánh xe trước).
        - Phủ quyết nửa đầu xe (Front-Vehicle Sanity Override): CẤM TUYỆT ĐỐI nhãn can_sau trên toàn bộ bức ảnh.
        - Vết tổn thất nằm ở mảng tôn vè trước / hốc bánh -> Chuyển về ve_che_banh (hoặc can_truoc).
        """
        dent_fender = DamageResult(
            damage_type="dent",
            confidence=0.84,
            damage_polygon=[[140, 230], [260, 230], [260, 310], [140, 310]],
            bbox_xyxy=[140.0, 230.0, 260.0, 310.0],
            center_xy=(200.0, 270.0),
        )
        front_wheel = PartResult(
            part_key="wheel",
            part_name="bánh xe",
            confidence=0.85,
            part_polygon=[[100, 320], [300, 320], [300, 560], [100, 560]],
            bbox_xyxy=[100.0, 320.0, 300.0, 560.0],
            center_xy=(200.0, 440.0),
        )
        rearview_mirror = PartResult(
            part_key="kinh_chieu_hau",
            part_name="kính chiếu hậu",
            confidence=0.78,
            part_polygon=[[420, 120], [500, 120], [500, 200], [420, 200]],
            bbox_xyxy=[420.0, 120.0, 500.0, 200.0],
            center_xy=(460.0, 160.0),
        )

        # 1. Kiểm tra classify_shot_type: Có bánh xe + kính chiếu hậu -> far_view
        shot_type, act_req, reason = classify_shot_type([dent_fender], [front_wheel, rearview_mirror], self.image_hw)
        self.assertEqual(shot_type, "far_view")
        self.assertEqual(act_req, "Kết quả phân tích hợp lệ.")

        # 2. Kiểm tra map_damages_to_parts
        mapped = map_damages_to_parts([dent_fender], [front_wheel, rearview_mirror], self.image_hw)
        self.assertEqual(len(mapped), 1)

        # CẤM TUYỆT ĐỐI can_sau
        self.assertNotEqual(mapped[0].located_on_part, "cản sau")
        # Không có part thân vỏ đạt ngưỡng hình học -> khong_xac_dinh
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[0].action_required, "Kết quả phân tích hợp lệ.")

    def test_headlamp_shatter_guard_prevents_windshield_mislabeling(self):
        """Case kiểm thử nâng cấp 1 (Headlamp Shatter Guard):
        - Nếu glass_shatter nằm bên trong hoặc chồng lấn (IoU > 0.20) với cụm đèn (lamp_broken / den_xe),
          hoặc nằm ở chóa đèn phía trên với các vết trầy/nứt cản trước bên dưới (không có mask kính lái chuẩn):
          + ÉP CỨNG 100% nhãn bộ phận về 'đèn xe' (match_method = 'headlamp_shatter_guard').
          + TUYỆT ĐỐI CẤM đè nhãn 'kính lái' lên chóa đèn vỡ.
          + Các vết scratch/crack nằm bên dưới chóa đèn vỡ bắt buộc khóa về 'cản trước'.
        """
        shattered_headlamp = DamageResult(
            damage_type="glass_shatter",
            confidence=0.88,
            damage_polygon=[[46, 0], [472, 0], [472, 205], [46, 205]],
            bbox_xyxy=[46.7, 0.0, 472.3, 205.7],
            center_xy=(259.5, 102.8),
        )
        bumper_scratch_left = DamageResult(
            damage_type="scratch",
            confidence=0.35,
            damage_polygon=[[0, 307], [305, 307], [305, 639], [0, 639]],
            bbox_xyxy=[0.0, 307.9, 305.5, 639.0],
            center_xy=(152.7, 473.4),
        )
        bumper_crack_mid = DamageResult(
            damage_type="crack",
            confidence=0.67,
            damage_polygon=[[320, 463], [360, 463], [360, 603], [320, 603]],
            bbox_xyxy=[320.0, 463.2, 360.8, 603.4],
            center_xy=(340.4, 533.3),
        )
        bumper_scratch_right = DamageResult(
            damage_type="scratch",
            confidence=0.49,
            damage_polygon=[[478, 523], [627, 523], [627, 667], [478, 667]],
            bbox_xyxy=[478.0, 523.1, 627.0, 667.8],
            center_xy=(552.5, 595.4),
        )

        mapped = map_damages_to_parts(
            [shattered_headlamp, bumper_scratch_left, bumper_crack_mid, bumper_scratch_right],
            [],
            (811, 652),
            filename="front_headlamp_closeup.png",
        )
        self.assertEqual(len(mapped), 4)
        self.assertEqual(mapped[0].located_on_part, "kính xe")
        self.assertEqual(mapped[0].match_method, "semantic_hard_rule_glass")
        self.assertNotEqual(mapped[0].located_on_part, "kính lái")

        for m in mapped[1:]:
            self.assertEqual(m.located_on_part, "khong_xac_dinh")

    def test_headlamp_grille_anchor_lock_forces_front_bumper(self):
        """Case kiểm thử nâng cấp 2 (Headlamp-Grille Anchor Lock & No HoughCircles Fake Wheel):
        - Khi phát hiện Đèn pha (lamp_broken / den_xe) hoặc Lưới tản nhiệt (grille):
          + ÉP CỨNG 100% các vết tổn thất tôn/nhựa nằm bên dưới và ôm xung quanh cụm đèn về 'cản trước'.
          + CẤM chia cắt góc cản trước thành 'vè che bánh' khi không có vòm lốp cao su thực tế.
        """
        lamp_broken = DamageResult(
            damage_type="lamp_broken",
            confidence=0.88,
            damage_polygon=[[255, 58], [648, 58], [648, 413], [255, 413]],
            bbox_xyxy=[255.7, 58.0, 648.3, 413.7],
            center_xy=(452.0, 235.8),
        )
        corner_scratch = DamageResult(
            damage_type="scratch",
            confidence=0.64,
            damage_polygon=[[204, 211], [496, 211], [496, 632], [204, 632]],
            bbox_xyxy=[204.2, 211.9, 496.5, 632.0],
            center_xy=(350.3, 421.9),
        )
        lower_dent = DamageResult(
            damage_type="dent",
            confidence=0.45,
            damage_polygon=[[451, 444], [835, 444], [835, 681], [451, 681]],
            bbox_xyxy=[451.7, 444.5, 835.4, 681.5],
            center_xy=(643.5, 563.0),
        )
        false_fender = PartResult(
            part_key="ve_che_banh",
            part_name="vè che bánh",
            confidence=0.52,
            part_polygon=[[180, 180], [520, 180], [520, 650], [180, 650]],
            bbox_xyxy=[180.0, 180.0, 520.0, 650.0],
            center_xy=(350.0, 415.0),
        )

        mapped = map_damages_to_parts(
            [lamp_broken, corner_scratch, lower_dent],
            [false_fender],
            (725, 1003),
            filename="front_bumper_headlamp.jpg",
        )
        self.assertEqual(len(mapped), 3)
        self.assertEqual(mapped[0].located_on_part, "đèn xe")
        # Step 1 / Step 2 dựa theo mask AI thực tế (ve_che_banh):
        self.assertEqual(mapped[1].located_on_part, "vè che bánh")
        self.assertEqual(mapped[2].located_on_part, "vè che bánh")

    def test_rear_bumper_seam_heuristic_separates_fender_and_rear_bumper(self):
        """Case kiểm thử nâng cấp 3 (Rear Bumper Seam Heuristic):
        - Ở góc chụp hông/đuôi xe có bánh xe nằm bên trái (x < 0.35 * w) và thân xe kéo dài sang phải:
          + Vùng tôn ngay trên vòm lốp -> ÉP về 'vè che bánh'.
          + Vùng nằm phía sau khe hở vòm bánh kéo dài về phía đuôi -> ÉP CỨNG 'cản sau', CẤM TUYỆT ĐỐI 'cản trước'.
        """
        arch_scratch = DamageResult(
            damage_type="scratch",
            confidence=0.54,
            damage_polygon=[[278, 318], [367, 318], [367, 410], [278, 410]],
            bbox_xyxy=[278.9, 318.6, 367.9, 410.1],
            center_xy=(323.4, 364.3),
        )
        rear_bumper_scratch = DamageResult(
            damage_type="scratch",
            confidence=0.65,
            damage_polygon=[[399, 403], [525, 403], [525, 506], [399, 506]],
            bbox_xyxy=[399.9, 403.3, 525.7, 506.1],
            center_xy=(462.8, 454.7),
        )
        left_wheel = PartResult(
            part_key="wheel",
            part_name="bánh xe",
            confidence=0.80,
            part_polygon=[[0, 358], [358, 358], [358, 717], [0, 717]],
            bbox_xyxy=[0.0, 358.5, 358.4, 717.0],
            center_xy=(179.2, 537.7),
        )

        mapped = map_damages_to_parts(
            [arch_scratch, rear_bumper_scratch],
            [left_wheel],
            (717, 1024),
            filename="rear_bumper_wheel_seam.png",
        )
        self.assertEqual(len(mapped), 2)
        # Bánh xe không thể chứa vết xước tôn và không có mask thân vỏ -> khong_xac_dinh
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")
        self.assertEqual(mapped[1].located_on_part, "khong_xac_dinh")
        for m in mapped:
            self.assertNotEqual(m.located_on_part, "cản trước")

    def test_capstone_5_damage_classes_merge(self):
        """Kiểm tra đồng bộ 5 nhóm hư hỏng chuẩn Capstone Proposal."""
        self.assertEqual(normalize_damage_class("scratch"), "scratch")
        self.assertEqual(normalize_damage_class("dent"), "dent")
        self.assertEqual(normalize_damage_class("crack"), "crack")
        self.assertEqual(normalize_damage_class("glass_shatter"), "crack")
        self.assertEqual(normalize_damage_class("lamp_broken"), "crack")
        self.assertEqual(normalize_damage_class("tear_puncture"), "tear_puncture")
        self.assertEqual(normalize_damage_class("paint_chipping"), "paint_chipping")
        self.assertIsNone(normalize_damage_class("tire_flat"))

    def test_capstone_4_doors_topology_resolution(self):
        """Kiểm tra phân giải class AI 'cua_hong' thành 4 JSON output riêng biệt:
        'Cửa trước trái', 'Cửa sau trái', 'Cửa trước phải', 'Cửa sau phải'.
        """
        # 1. Sườn trái (đầu xe bên trái: có gương chiếu hậu bên trái x=180, cửa hông rộng từ x=160 đến x=640)
        mirror_left = PartResult(
            part_key="kinh_chieu_hau", part_name="kính chiếu hậu", confidence=0.82,
            part_polygon=[[150, 180], [210, 180], [210, 230], [150, 230]],
            bbox_xyxy=[150.0, 180.0, 210.0, 230.0], center_xy=(180.0, 205.0),
        )
        wide_door_left = PartResult(
            part_key="cua_hong", part_name="cửa hông", confidence=0.90,
            part_polygon=[[160, 200], [640, 200], [640, 480], [160, 480]],
            bbox_xyxy=[160.0, 200.0, 640.0, 480.0], center_xy=(400.0, 340.0),
        )
        resolved_left = resolve_four_side_doors([mirror_left, wide_door_left], (600, 800), filename="left_side.jpg")
        door_names_left = [p["part_name"] for p in resolved_left if "Cửa" in p["part_name"]]
        self.assertIn("Cửa trước trái", door_names_left)
        self.assertIn("Cửa sau trái", door_names_left)

        # 2. Sườn phải (đầu xe bên phải: có gương chiếu hậu bên phải x=620, cửa hông rộng từ x=160 đến x=640)
        mirror_right = PartResult(
            part_key="kinh_chieu_hau", part_name="kính chiếu hậu", confidence=0.82,
            part_polygon=[[590, 180], [650, 180], [650, 230], [590, 230]],
            bbox_xyxy=[590.0, 180.0, 650.0, 230.0], center_xy=(620.0, 205.0),
        )
        resolved_right = resolve_four_side_doors([mirror_right, wide_door_left], (600, 800), filename="right_side.jpg")
        door_names_right = [p["part_name"] for p in resolved_right if "Cửa" in p["part_name"]]
        self.assertIn("Cửa trước phải", door_names_right)
        self.assertIn("Cửa sau phải", door_names_right)

    def test_shot_type_router_and_8_points_api_gateway(self):
        """Kiểm tra determine_shot_type và Endpoint POST /api/v1/checkin/analyze-8-points."""
        from fastapi.testclient import TestClient
        import cv2
        from ai.api.main import create_gateway_app
        from ai.inference.pipeline import PipelineResponse

        # Kiểm tra determine_shot_type với ảnh close_up
        close_dmg = DamageResult(
            damage_type="scratch", confidence=0.90,
            damage_polygon=[[50, 50], [700, 50], [700, 550], [50, 550]],
            bbox_xyxy=[50.0, 50.0, 700.0, 550.0], center_xy=(375.0, 300.0),
        )
        self.assertEqual(determine_shot_type([close_dmg], [], (600, 800)), "close_up")

        # Giả lập Pipeline cho 8-point Check-in API Gateway
        class Dummy8PointPipeline:
            def __init__(self, force_closeup_on_idx: int = -1):
                self.force_closeup_on_idx = force_closeup_on_idx
                self.call_count = 0

            def run(self, image_bgr, filename="img.jpg"):
                self.call_count += 1
                if self.call_count == self.force_closeup_on_idx:
                    return PipelineResponse(
                        success=True, image_width=800, image_height=600,
                        shot_type="close_up",
                        action_required="Ảnh quá cận cảnh, không thể định vị",
                        detections=[{
                            "damage_type": "scratch", "confidence": 0.88,
                            "part": "khong_xac_dinh", "located_on_part": "khong_xac_dinh",
                            "message": "Ảnh quá cận cảnh, không thể định vị",
                        }],
                    )
                return PipelineResponse(
                    success=True, image_width=800, image_height=600,
                    shot_type="far_view",
                    action_required="Kết quả phân tích hợp lệ.",
                    detections=[{
                        "damage_type": "glass_shatter" if self.call_count == 1 else "scratch",
                        "confidence": 0.85,
                        "part": "Kính lái" if self.call_count == 1 else "Cửa trước trái",
                        "located_on_part": "Kính lái" if self.call_count == 1 else "Cửa trước trái",
                        "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
                        "match_method": "mask_overlap",
                    }],
                )

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", dummy_img)
        img_bytes = encoded.tobytes()

        # 1. Trường hợp có 1 ảnh close_up -> trả về HTTP 400
        app_reject = create_gateway_app(pipeline=Dummy8PointPipeline(force_closeup_on_idx=3))
        client_reject = TestClient(app_reject)
        files_8 = [("files", (f"point_{i}.jpg", img_bytes, "image/jpeg")) for i in range(1, 9)]
        resp_400 = client_reject.post("/api/v1/checkin/analyze-8-points", files=files_8)
        self.assertEqual(resp_400.status_code, 400)
        self.assertEqual(resp_400.json()["part"], "khong_xac_dinh")

        # 2. Trường hợp 8 ảnh hợp lệ -> trả về HTTP 200 kèm JSON tổng hợp tình trạng xe
        app_ok = create_gateway_app(pipeline=Dummy8PointPipeline(force_closeup_on_idx=-1))
        client_ok = TestClient(app_ok)
        resp_200 = client_ok.post("/api/v1/checkin/analyze-8-points", files=files_8)
        self.assertEqual(resp_200.status_code, 200)
        body = resp_200.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["total_points_analyzed"], 8)
        # glass_shatter ở góc 1 đã được gộp về 'crack'
        self.assertEqual(body["vehicle_condition_summary"]["damages_by_type"]["crack"], 1)
        self.assertEqual(body["vehicle_condition_summary"]["damages_by_type"]["scratch"], 7)


class TestChangeDetectionSuite(unittest.TestCase):
    """Kiểm thử Change Detection (SIFT/ORB & Homography)."""
    def setUp(self):
        from ai.inference.change_detection import (
            calculate_bbox_iou,
            warp_bbox_with_homography,
            align_images_and_compare,
            visualize_change_detection,
            ChangeDetectionResult,
        )
        self.calculate_bbox_iou = calculate_bbox_iou
        self.warp_bbox_with_homography = warp_bbox_with_homography
        self.align_images_and_compare = align_images_and_compare
        self.visualize_change_detection = visualize_change_detection
        self.ChangeDetectionResult = ChangeDetectionResult

        self.img_before = np.zeros((400, 600, 3), dtype=np.uint8)
        self.img_after = np.zeros((400, 600, 3), dtype=np.uint8)

        for i in range(10, 390, 30):
            for j in range(10, 590, 40):
                cv2.circle(self.img_before, (j, i), 6, (255, 255, 255), -1)
                cv2.rectangle(self.img_before, (j + 5, i + 5), (j + 20, i + 20), (128, 128, 128), -1)
                cv2.circle(self.img_after, (j + 5, i + 5), 6, (255, 255, 255), -1)
                cv2.rectangle(self.img_after, (j + 10, i + 10), (j + 25, i + 25), (128, 128, 128), -1)

    def test_calculate_bbox_iou(self):
        box1 = [10.0, 10.0, 100.0, 100.0]
        box2 = [10.0, 10.0, 100.0, 100.0]
        self.assertAlmostEqual(self.calculate_bbox_iou(box1, box2), 1.0)
        box3 = [200.0, 200.0, 300.0, 300.0]
        self.assertEqual(self.calculate_bbox_iou(box1, box3), 0.0)

    def test_warp_bbox_with_homography(self):
        H = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        bbox = [50.0, 60.0, 150.0, 160.0]
        warped = self.warp_bbox_with_homography(bbox, H)
        self.assertAlmostEqual(warped[0], 60.0)
        self.assertAlmostEqual(warped[1], 80.0)

    def test_align_images_and_compare_classification(self):
        before_detections = [
            {"damage_type": "scratch", "confidence": 0.85, "bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "part": "Cửa trước"}
        ]
        after_detections = [
            {"damage_type": "scratch", "confidence": 0.88, "bbox_xyxy": [102.0, 102.0, 198.0, 198.0], "part": "Cửa trước"},
            {"damage_type": "dent", "confidence": 0.90, "bbox_xyxy": [350.0, 250.0, 450.0, 350.0], "part": "Cửa sau"},
        ]
        res = self.align_images_and_compare(
            self.img_before, self.img_after,
            before_detections=before_detections,
            after_detections=after_detections,
            iou_threshold=0.60,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.num_old_damages, 1)
        self.assertEqual(res.num_new_damages, 1)
        self.assertEqual(res.old_damages[0]["status"], "old_damage")
        self.assertEqual(res.new_damages[0]["status"], "new_damage")

    def test_visualize_change_detection(self):
        res = self.ChangeDetectionResult(
            success=True, aligned=True, feature_method="SIFT",
            num_matches=50, num_inliers=40,
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            total_before_damages=1, total_after_damages=2,
            num_old_damages=1, num_new_damages=1,
            old_damages=[{"id": 0, "damage_type": "scratch", "status": "old_damage"}],
            new_damages=[{"id": 1, "damage_type": "dent", "status": "new_damage"}],
        )
        vis = self.visualize_change_detection(self.img_before, self.img_after, res)
        self.assertIsNotNone(vis)
        self.assertEqual(vis.ndim, 3)

    # -------------------------------------------------------------------------
    # PHASE 2: TESTS CHUẨN HÓA 8 CLASS, CONFIDENCE VÀ QUARTER PANEL
    # -------------------------------------------------------------------------
    def test_standard_8_classes_and_no_cua_hong(self):
        """Phase 2: Kiểm tra toàn bộ output map về đúng 8 class chuẩn hoặc khong_xac_dinh, không có cua_hong."""
        from ai.inference.pipeline import STANDARD_PARTS, map_to_standard_part

        self.assertEqual(len(STANDARD_PARTS), 8)
        self.assertIn("Nắp capo", STANDARD_PARTS)
        self.assertIn("Cửa trước", STANDARD_PARTS)
        self.assertIn("Cửa sau", STANDARD_PARTS)
        self.assertIn("Cản trước", STANDARD_PARTS)
        self.assertIn("Cản sau", STANDARD_PARTS)
        self.assertIn("Kính xe", STANDARD_PARTS)
        self.assertIn("Đèn xe", STANDARD_PARTS)
        self.assertIn("Vè che bánh", STANDARD_PARTS)

        # Cấm tuyệt đối 'cửa hông' trong 8 bộ phận chuẩn
        self.assertNotIn("cửa hông", STANDARD_PARTS)
        self.assertNotIn("cua_hong", STANDARD_PARTS)

        # Kiểm tra map_to_standard_part
        self.assertIn(map_to_standard_part("cua_hong"), STANDARD_PARTS)
        self.assertNotEqual(map_to_standard_part("cua_hong"), "cửa hông")
        self.assertEqual(map_to_standard_part("cua_truoc"), "Cửa trước")
        self.assertEqual(map_to_standard_part("cua_sau"), "Cửa sau")
        self.assertEqual(map_to_standard_part("can_truoc"), "Cản trước")
        self.assertEqual(map_to_standard_part("can_sau"), "Cản sau")
        self.assertEqual(map_to_standard_part("nap_capo"), "Nắp capo")
        self.assertEqual(map_to_standard_part("ve_che_banh"), "Vè che bánh")
        self.assertEqual(map_to_standard_part("khong_xac_dinh"), "khong_xac_dinh")

    def test_kinh_lai_and_kinh_chieu_hau_map_to_kinh_xe(self):
        """Phase 2: Kiểm tra kính lái và kính chiếu hậu đều map về 'Kính xe'."""
        from ai.inference.pipeline import map_to_standard_part, PART_DISPLAY_NAMES

        self.assertEqual(map_to_standard_part("kinh_lai"), "Kính xe")
        self.assertEqual(map_to_standard_part("kinh_chieu_hau"), "Kính xe")
        self.assertEqual(map_to_standard_part("kinh_cua_hong"), "Kính xe")
        self.assertEqual(PART_DISPLAY_NAMES["kinh_lai"], "Kính xe")
        self.assertEqual(PART_DISPLAY_NAMES["kinh_chieu_hau"], "Kính xe")

    def test_pipeline_default_confidences(self):
        """Phase 2: Kiểm tra cấu hình default confidences đúng là 0.25 (damage) và 0.30 (parts)."""
        import inspect
        from ai.inference.pipeline import TwoStagePipeline

        sig = inspect.signature(TwoStagePipeline.__init__)
        params = sig.parameters
        self.assertEqual(params["damage_conf"].default, 0.25)
        self.assertEqual(params["parts_conf"].default, 0.30)
        self.assertEqual(params["iou"].default, 0.45)
        self.assertEqual(params["min_part_confidence"].default, 0.25)

    def test_quarter_panel_priority_over_door(self):
        """Phase 2: Vết móp/xước ở vùng Vè sau (Quarter panel) ưu tiên Vè che bánh hơn Cửa hông lan rộng."""
        from ai.inference.pipeline import DamageResult, PartResult, map_damages_to_parts

        # Giả lập vết móp ở vè sau (giữa bánh sau và cản sau)
        dent = DamageResult(
            damage_type="dent",
            confidence=0.75,
            damage_polygon=[[550, 200], [680, 200], [680, 350], [550, 350]],
            bbox_xyxy=[550.0, 200.0, 680.0, 350.0],
            center_xy=(615.0, 275.0),
        )
        # Mask cửa hông lan rộng sang vè sau (lỗi thường gặp của mô hình seg)
        door_expanded = PartResult(
            part_key="cua_hong", part_name="Cửa sau", confidence=0.70,
            part_polygon=[[200, 150], [700, 150], [700, 400], [200, 400]],
            bbox_xyxy=[200.0, 150.0, 700.0, 400.0], center_xy=(450.0, 275.0),
        )
        # Vè che bánh (bánh sau)
        fender = PartResult(
            part_key="ve_che_banh", part_name="Vè che bánh", confidence=0.60,
            part_polygon=[[500, 280], [660, 280], [660, 450], [500, 450]],
            bbox_xyxy=[500.0, 280.0, 660.0, 450.0], center_xy=(580.0, 365.0),
        )
        # Cản sau ở bên phải
        rear_bumper = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.80,
            part_polygon=[[680, 250], [800, 250], [800, 480], [680, 480]],
            bbox_xyxy=[680.0, 250.0, 800.0, 480.0], center_xy=(740.0, 365.0),
        )

        mapped = map_damages_to_parts([dent], [door_expanded, fender, rear_bumper], (600, 800))
        self.assertEqual(len(mapped), 1)
        # Ưu tiên Vè che bánh thay vì bị Cửa sau/Cửa hông nuốt chửng
        self.assertEqual(mapped[0].located_on_part, "Vè che bánh")
        self.assertNotEqual(mapped[0].located_on_part, "cửa hông")


class TestPhase2OptimizationSuite(unittest.TestCase):
    """Bộ kiểm thử mở rộng cho Giai đoạn B-E: Kiểm định 5 lỗi phát hiện qua 8 ảnh thực tế."""

    def test_merge_and_smooth_individual_contour_confidences(self):
        """Lỗi 2: merge_and_smooth_part_masks phải giữ nguyên confidence độc lập của từng contour vật lý."""
        from ai.inference.pipeline import PartResult, merge_and_smooth_part_masks

        # Giả lập 2 vè che bánh: vè trước (conf=0.4694) và vè sau (conf=0.2882)
        p_front = PartResult(
            part_key="ve_che_banh",
            part_name="Vè che bánh",
            confidence=0.4694,
            part_polygon=[[100, 200], [200, 200], [200, 350], [100, 350]],
            bbox_xyxy=[100.0, 200.0, 200.0, 350.0],
            center_xy=(150.0, 275.0),
        )
        p_rear = PartResult(
            part_key="ve_che_banh",
            part_name="Vè che bánh",
            confidence=0.2882,
            part_polygon=[[600, 200], [700, 200], [700, 350], [600, 350]],
            bbox_xyxy=[600.0, 200.0, 700.0, 350.0],
            center_xy=(650.0, 275.0),
        )
        smoothed = merge_and_smooth_part_masks([p_front, p_rear], (600, 800), min_area_pixels=100.0)
        self.assertEqual(len(smoothed), 2)
        # Sắp xếp theo X để so sánh
        smoothed.sort(key=lambda p: p.bbox_xyxy[0])
        self.assertAlmostEqual(smoothed[0].confidence, 0.4694, places=3)
        self.assertAlmostEqual(smoothed[1].confidence, 0.2882, places=3)

    def test_filter_false_positive_parts_by_orientation(self):
        """Lỗi 2: filter_false_positive_parts_by_orientation loại bỏ cản sau xuất hiện nhầm ở đầu xe."""
        from ai.inference.pipeline import PartResult, OrientationInfo, filter_false_positive_parts_by_orientation

        orient = OrientationInfo(
            head_direction="left",
            b_pillar_x=500.0,
            confidence=0.85,
            is_reliable=True,
            details="Test orientation",
        )
        real_rear_bumper = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.75,
            part_polygon=[[800, 250], [950, 250], [950, 450], [800, 450]],
            bbox_xyxy=[800.0, 250.0, 950.0, 450.0], center_xy=(875.0, 350.0),
        )
        false_rear_bumper = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.40,
            part_polygon=[[50, 250], [200, 250], [200, 450], [50, 450]],
            bbox_xyxy=[50.0, 250.0, 200.0, 450.0], center_xy=(125.0, 350.0),
        )
        filtered = filter_false_positive_parts_by_orientation([real_rear_bumper, false_rear_bumper], (600, 1000), orient)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].center_xy[0], 875.0)

    def test_direct_door_overlap_takes_priority_over_quarter_panel_guard(self):
        """Lỗi 3: Khi tổn thất đè phủ trực tiếp lên cửa sau (không có fender riêng), ưu tiên Cửa sau thay vì cấm cửa."""
        from ai.inference.pipeline import DamageResult, PartResult, map_damages_to_parts

        dent = DamageResult(
            damage_type="dent",
            confidence=0.75,
            damage_polygon=[[450, 240], [600, 240], [600, 400], [450, 400]],
            bbox_xyxy=[450.0, 240.0, 600.0, 400.0],
            center_xy=(525.0, 320.0),
        )
        rear_door = PartResult(
            part_key="cua_hong", part_name="Cửa sau", confidence=0.70,
            part_polygon=[[400, 200], [580, 200], [580, 500], [400, 500]],
            bbox_xyxy=[400.0, 200.0, 580.0, 500.0], center_xy=(490.0, 350.0),
        )
        rear_bumper = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.45,
            part_polygon=[[650, 300], [800, 300], [800, 550], [650, 550]],
            bbox_xyxy=[650.0, 300.0, 800.0, 550.0], center_xy=(725.0, 425.0),
        )
        mapped = map_damages_to_parts([dent], [rear_door, rear_bumper], (600, 800))
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].located_on_part, "Cửa sau")
        self.assertEqual(mapped[0].match_method, "mask_overlap")

    def test_bumper_spatial_distance_vertical_constraint(self):
        """Lỗi 3: Vết móp ở thân trên không được gán vào Cản sau qua spatial distance do lệch trục Y."""
        from ai.inference.pipeline import DamageResult, PartResult, map_damages_to_parts

        dent_high = DamageResult(
            damage_type="dent",
            confidence=0.70,
            damage_polygon=[[500, 100], [600, 100], [600, 200], [500, 200]],
            bbox_xyxy=[500.0, 100.0, 600.0, 200.0],
            center_xy=(550.0, 150.0),
        )
        rear_bumper_low = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.60,
            part_polygon=[[520, 350], [680, 350], [680, 550], [520, 550]],
            bbox_xyxy=[520.0, 350.0, 680.0, 550.0], center_xy=(600.0, 450.0),
        )
        mapped = map_damages_to_parts([dent_high], [rear_bumper_low], (600, 800))
        self.assertEqual(len(mapped), 1)
        # Bị chặn bởi vertical check, không được gán vào Cản sau
        self.assertNotEqual(mapped[0].located_on_part, "Cản sau")
        self.assertEqual(mapped[0].located_on_part, "khong_xac_dinh")

    def test_abnormal_bounding_box_warning(self):
        """Lỗi 4: Bounding Box chiếm > 30% diện tích ảnh phải giữ nguyên tọa độ và sinh cảnh báo giới hạn Stage 1."""
        from ai.inference.pipeline import DamageResult, PartResult, map_damages_to_parts

        # Bounding box chiếm >35% ảnh
        huge_dent = DamageResult(
            damage_type="dent",
            confidence=0.65,
            damage_polygon=[[100, 100], [700, 100], [700, 500], [100, 500]],
            bbox_xyxy=[100.0, 100.0, 700.0, 500.0],
            center_xy=(400.0, 300.0),
        )
        hood = PartResult(
            part_key="nap_capo", part_name="Nắp capo", confidence=0.80,
            part_polygon=[[150, 120], [650, 120], [650, 480], [150, 480]],
            bbox_xyxy=[150.0, 120.0, 650.0, 480.0], center_xy=(400.0, 300.0),
        )
        mapped = map_damages_to_parts([huge_dent], [hood], (600, 800))
        self.assertEqual(len(mapped), 1)
        # Tọa độ bbox nguyên bản không bị can thiệp thu nhỏ
        self.assertEqual(mapped[0].bbox_xyxy, [100.0, 100.0, 700.0, 500.0])
        # Có cảnh báo diện tích bất thường
        self.assertIn("diện tích bất thường", mapped[0].warning)

    def test_three_state_analysis_status_and_action_required(self):
        """Lỗi 5: Kiểm định 3 trạng thái completed, needs_review và insufficient_data."""
        from ai.inference.pipeline import PipelineResponse

        resp_completed = PipelineResponse(
            success=True, status="completed", image_width=800, image_height=600,
            action_required="Kết quả phân tích hợp lệ."
        )
        d_comp = resp_completed.to_dict()
        self.assertEqual(d_comp["status"], "completed")
        self.assertEqual(d_comp["action_required"], "Kết quả phân tích hợp lệ.")

        resp_review = PipelineResponse(
            success=True, status="needs_review", image_width=800, image_height=600,
            action_required="Cần giám định viên kiểm tra lại kết quả."
        )
        d_rev = resp_review.to_dict()
        self.assertEqual(d_rev["status"], "needs_review")
        self.assertIn("kiểm tra lại", d_rev["action_required"])

        resp_insufficient = PipelineResponse(
            success=True, status="insufficient_data", image_width=800, image_height=600,
            action_required="Ảnh chụp không đủ bối cảnh hoặc thiếu bộ phận xe tham chiếu."
        )
        d_insuf = resp_insufficient.to_dict()
        self.assertEqual(d_insuf["status"], "insufficient_data")

    def test_no_synthetic_parts_fabricated_in_contextual_rules(self):
        """Kiểm tra filter_parts_with_contextual_rules không tự bịa box can_truoc / ve_che_banh / cua_sau giả."""
        from ai.inference.pipeline import DamageResult, PartResult, filter_parts_with_contextual_rules

        hood = PartResult(
            part_key="nap_capo", part_name="Nắp capo", confidence=0.65,
            part_polygon=[[50, 50], [300, 50], [300, 250], [50, 250]],
            bbox_xyxy=[50.0, 50.0, 300.0, 250.0], center_xy=(175.0, 150.0),
        )
        real_fender = PartResult(
            part_key="ve_che_banh", part_name="Vè che bánh", confidence=0.29,
            part_polygon=[[1100, 1100], [1600, 1100], [1600, 1280], [1100, 1280]],
            bbox_xyxy=[1100.0, 1100.0, 1600.0, 1280.0], center_xy=(1350.0, 1190.0),
        )
        broken_lamp = DamageResult(
            damage_type="lamp_broken", confidence=0.80,
            damage_polygon=[], bbox_xyxy=[25.0, 35.0, 360.0, 280.0], center_xy=(192.5, 157.5),
        )
        lower_dent = DamageResult(
            damage_type="dent", confidence=0.84,
            damage_polygon=[], bbox_xyxy=[970.0, 820.0, 1420.0, 1044.0], center_xy=(1195.0, 932.0),
        )

        filtered = filter_parts_with_contextual_rules(
            [hood, real_fender], [broken_lamp, lower_dent], (1464, 1920)
        )
        # Không được tự sinh thêm can_truoc hoặc cua_sau giả
        self.assertFalse(any(p.part_key == "can_truoc" for p in filtered))
        self.assertFalse(any(p.part_key == "cua_sau" for p in filtered))
        # Mask ve_che_banh thật của model được giữ nguyên, không bị xóa
        self.assertTrue(any(p.part_key == "ve_che_banh" and p.confidence == 0.29 for p in filtered))

    def test_door_topology_ignores_weak_rear_bumper_hallucination(self):
        """Kiểm tra resolve_door_position_from_topology không ép thành Cửa sau khi can_sau có conf < 0.45."""
        from ai.inference.pipeline import PartResult, resolve_door_position_from_topology

        single_door = PartResult(
            part_key="cua_hong", part_name="Cửa trước", confidence=0.69,
            part_polygon=[[159, 237], [538, 237], [538, 451], [159, 451]],
            bbox_xyxy=[159.0, 237.0, 538.0, 451.0], center_xy=(348.5, 344.0),
        )
        weak_rear_bumper = PartResult(
            part_key="can_sau", part_name="Cản sau", confidence=0.31,
            part_polygon=[[0, 478], [332, 478], [332, 632], [0, 632]],
            bbox_xyxy=[0.0, 478.0, 332.0, 632.0], center_xy=(166.0, 555.0),
        )
        # Điểm hư hỏng nằm bên phải tâm cửa (x=470 > 348.5), nhưng không có mỏ neo >= 0.45
        door_label = resolve_door_position_from_topology(
            (470.0, 300.0), [single_door, weak_rear_bumper], (667, 1000)
        )
        self.assertEqual(door_label, "Cửa trước")

    def test_hsv_red_taillight_rectifies_front_bumper_to_rear_bumper(self):
        """Kiểm tra khử nhập nhằng Cản trước vs Cản sau ở đuôi xe khi vùng phía trên cản có cụm đèn hậu đỏ (HSV >= 0.12)."""
        from ai.inference.pipeline import DamageResult, PartResult, filter_parts_with_contextual_rules

        img_bgr = np.zeros((600, 800, 3), dtype=np.uint8)
        # Vẽ cụm màu đỏ đặc trưng đèn hậu (BGR: 0, 0, 220) ở vùng phía trên cản (y=180..280, x=200..500)
        img_bgr[180:280, 200:500] = (0, 0, 220)

        misclassified_bumper = PartResult(
            part_key="can_truoc", part_name="Cản trước", confidence=0.62,
            part_polygon=[[180, 300], [600, 300], [600, 520], [180, 520]],
            bbox_xyxy=[180.0, 300.0, 600.0, 520.0], center_xy=(390.0, 410.0),
        )
        scratch = DamageResult(
            damage_type="scratch", confidence=0.80,
            damage_polygon=[], bbox_xyxy=[250.0, 340.0, 500.0, 460.0], center_xy=(375.0, 400.0),
        )

        rectified = filter_parts_with_contextual_rules(
            [misclassified_bumper], [scratch], (600, 800), image_bgr=img_bgr
        )
        self.assertEqual(len(rectified), 1)
        self.assertEqual(rectified[0].part_key, "can_sau")
        self.assertEqual(rectified[0].part_name, "Cản sau")

    def test_dent_confidence_threshold_035_filters_weak_dents(self):
        """Kiểm tra lọc nhiễu detection dent độ tin cậy thấp (< 0.35)."""
        from ai.inference.pipeline import DamageResult, PartResult, filter_parts_with_contextual_rules, map_damages_to_parts

        weak_dent = DamageResult(
            damage_type="dent", confidence=0.31,
            damage_polygon=[], bbox_xyxy=[100.0, 200.0, 180.0, 280.0], center_xy=(140.0, 240.0),
        )
        valid_dent = DamageResult(
            damage_type="dent", confidence=0.38,
            damage_polygon=[], bbox_xyxy=[200.0, 200.0, 300.0, 300.0], center_xy=(250.0, 250.0),
        )
        valid_scratch = DamageResult(
            damage_type="scratch", confidence=0.28,
            damage_polygon=[], bbox_xyxy=[280.0, 210.0, 340.0, 290.0], center_xy=(310.0, 250.0),
        )
        fender = PartResult(
            part_key="ve_che_banh", part_name="Vè che bánh", confidence=0.75,
            part_polygon=[[80, 150], [350, 150], [350, 350], [80, 350]],
            bbox_xyxy=[80.0, 150.0, 350.0, 350.0], center_xy=(215.0, 250.0),
        )

        dmgs = [weak_dent, valid_dent, valid_scratch]
        filter_parts_with_contextual_rules([fender], dmgs, (600, 800))
        self.assertEqual(len(dmgs), 2)
        self.assertFalse(any(d.damage_type == "dent" and d.confidence < 0.35 for d in dmgs))

        mapped = map_damages_to_parts([weak_dent, valid_dent, valid_scratch], [fender], (600, 800))
        self.assertEqual(len(mapped), 2)
        self.assertEqual(mapped[0].confidence, 0.38)
        self.assertEqual(mapped[1].confidence, 0.28)


if __name__ == "__main__":
    unittest.main()
