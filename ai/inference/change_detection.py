"""Module Change Detection (So sánh trước - sau thuê xe) dựa trên SIFT/ORB và Homography.

Trọng tâm Sprint 2 theo Capstone Proposal:
- Căn chỉnh góc chụp giữa ảnh Check-in (Before) và Check-out (After) bằng SIFT/ORB + RANSAC findHomography.
- Bẻ cong (warp) ảnh hoặc chuyển đổi hệ tọa độ bounding box tổn thất từ After về Before.
- So sánh IoU giữa các vết tổn thất:
    + IoU >= 0.6: 'old_damage' (Tổn thất cũ đã tồn tại trước bàn giao xe).
    + IoU < 0.6:  'new_damage' (Tổn thất mới phát sinh - Cần lập biên bản bồi thường).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import supervision as sv
except ImportError:  # pragma: no cover
    sv = None


@dataclass
class DamageComparisonItem:
    """Chi tiết so sánh cho từng vết tổn thất trên ảnh After."""
    after_damage_id: int
    damage_type: str
    confidence: float
    located_on_part: str
    original_bbox_after: List[float]       # [x1, y1, x2, y2] trên ảnh After
    warped_bbox_in_before: List[float]     # [x1, y1, x2, y2] sau khi warp về Before
    best_matching_before_id: Optional[int] # ID của vết tổn thất khớp nhất trên Before
    best_matching_before_type: Optional[str]
    best_matching_before_bbox: Optional[List[float]]
    iou_with_before: float
    status: str                            # 'old_damage' hoặc 'new_damage'
    message: str                           # Mô tả kết quả


@dataclass
class ChangeDetectionResult:
    """Kết quả tổng hợp từ thuật toán so sánh Change Detection."""
    success: bool
    aligned: bool
    feature_method: str                    # 'SIFT' hoặc 'ORB'
    num_matches: int
    num_inliers: int
    homography_matrix: Optional[List[List[float]]]
    total_before_damages: int
    total_after_damages: int
    num_old_damages: int
    num_new_damages: int
    comparison_items: List[DamageComparisonItem] = field(default_factory=list)
    old_damages: List[Dict[str, Any]] = field(default_factory=list)
    new_damages: List[Dict[str, Any]] = field(default_factory=list)
    warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "aligned": self.aligned,
            "feature_method": self.feature_method,
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
            "homography_matrix": self.homography_matrix,
            "total_before_damages": self.total_before_damages,
            "total_after_damages": self.total_after_damages,
            "num_old_damages": self.num_old_damages,
            "num_new_damages": self.num_new_damages,
            "old_damages": self.old_damages,
            "new_damages": self.new_damages,
            "warning": self.warning,
        }


def calculate_bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 Bounding Box [x1, y1, x2, y2]."""
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    xa1, ya1, xa2, ya2 = box_a[:4]
    xb1, yb1, xb2, yb2 = box_b[:4]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def warp_bbox_with_homography(
    bbox: Sequence[float],
    H: np.ndarray,
    clip_hw: Optional[Tuple[int, int]] = None,
) -> List[float]:
    """Chuyển đổi 4 đỉnh của bounding box [x1, y1, x2, y2] qua ma trận Homography H
    và tạo lại bounding box bao ngoài (Axis-Aligned Bounding Box) trên hệ tọa độ mới.
    """
    if len(bbox) < 4 or H is None:
        return [float(v) for v in bbox[:4]]

    x1, y1, x2, y2 = bbox[:4]
    pts = np.array([
        [[x1, y1]],
        [[x2, y1]],
        [[x2, y2]],
        [[x1, y2]],
    ], dtype=np.float32)

    try:
        warped_pts = cv2.perspectiveTransform(pts, H)
        xs = warped_pts[:, 0, 0]
        ys = warped_pts[:, 0, 1]

        wx1 = float(np.min(xs))
        wy1 = float(np.min(ys))
        wx2 = float(np.max(xs))
        wy2 = float(np.max(ys))

        if clip_hw is not None:
            h, w = clip_hw
            wx1 = max(0.0, min(float(w), wx1))
            wy1 = max(0.0, min(float(h), wy1))
            wx2 = max(0.0, min(float(w), wx2))
            wy2 = max(0.0, min(float(h), wy2))

        return [round(wx1, 1), round(wy1, 1), round(wx2, 1), round(wy2, 1)]
    except Exception:
        return [float(v) for v in bbox[:4]]


def extract_features_and_compute_homography(
    img_before_gray: np.ndarray,
    img_after_gray: np.ndarray,
    method: str = "SIFT",
    min_inliers: int = 10,
    ransac_reproj_threshold: float = 5.0,
) -> Tuple[Optional[np.ndarray], int, int, str]:
    """Tìm keypoints, descriptors và ước tính ma trận Homography H (After -> Before).
    Trả về: (H, num_good_matches, num_inliers, method_used)
    """
    if (
        img_before_gray is None
        or img_after_gray is None
        or img_before_gray.size == 0
        or img_after_gray.size == 0
    ):
        return None, 0, 0, method.upper()

    method_upper = method.upper()
    detector = None

    if method_upper == "SIFT":
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
        except Exception:
            detector = cv2.ORB_create(nfeatures=2000)
            method_upper = "ORB"
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        method_upper = "ORB"

    kp1, desc1 = detector.detectAndCompute(img_before_gray, None)
    kp2, desc2 = detector.detectAndCompute(img_after_gray, None)

    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0, method_upper

    good_matches = []
    if method_upper == "SIFT":
        # FLANN matcher cho float descriptors
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            knn_matches = matcher.knnMatch(desc2, desc1, k=2)  # query: After (2), train: Before (1)
            for m_pair in knn_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        except Exception:
            pass
    else:
        # BFMatcher với NORM_HAMMING cho binary descriptors (ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            knn_matches = bf.knnMatch(desc2, desc1, k=2)
            for m_pair in knn_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        except Exception:
            pass

    num_matches = len(good_matches)
    if num_matches < 4:
        return None, num_matches, 0, method_upper

    pts_after = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_before = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(pts_after, pts_before, cv2.RANSAC, ransac_reproj_threshold)
    except cv2.error:
        return None, num_matches, 0, method_upper

    num_inliers = int(np.sum(mask)) if mask is not None else 0

    if H is None or num_inliers < min_inliers:
        return None, num_matches, num_inliers, method_upper

    return H, num_matches, num_inliers, method_upper


def load_image_bgr(img_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Tải và chuẩn hóa ảnh BGR."""
    if isinstance(img_input, np.ndarray):
        return img_input
    path_str = str(img_input)
    img = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc file ảnh: {path_str}")
    return img


def align_images_and_compare(
    img_before_path: Union[str, Path, np.ndarray],
    img_after_path: Union[str, Path, np.ndarray],
    pipeline: Optional[Any] = None,
    before_detections: Optional[List[Dict[str, Any]]] = None,
    after_detections: Optional[List[Dict[str, Any]]] = None,
    iou_threshold: float = 0.60,
    feature_method: str = "SIFT",
    warp_after_image: bool = True,
) -> ChangeDetectionResult:
    """Hàm trọng tâm Sprint 2: So sánh Check-in (Before) và Check-out (After).
    
    Quy trình:
    1. Đọc ảnh Before & After.
    2. Chạy SIFT/ORB + RANSAC cv2.findHomography để tìm ma trận H căn chỉnh After -> Before.
    3. Lấy danh sách tổn thất (móp/xước) từ Before và After (tự chạy Pipeline nếu chưa truyền vào).
    4. Bẻ cong tọa độ bounding box của các vết trên ảnh After về hệ tọa độ của ảnh Before.
    5. Tính IoU giữa mỗi vết After (đã căn chỉnh) với các vết trên ảnh Before:
       - IoU >= 0.60 -> 'old_damage' (Tổn thất cũ đã có từ trước).
       - IoU < 0.60  -> 'new_damage' (Tổn thất mới phát sinh trong quá trình thuê).
    6. Trả về kết quả phân loại chi tiết.
    """
    img_before = load_image_bgr(img_before_path)
    img_after = load_image_bgr(img_after_path)

    h_before, w_before = img_before.shape[:2]
    h_after, w_after = img_after.shape[:2]

    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    # 1. Tìm Homography H (After -> Before)
    H, num_matches, num_inliers, used_method = extract_features_and_compute_homography(
        gray_before, gray_after, method=feature_method
    )

    aligned = (H is not None)
    warning_msg = ""
    if not aligned:
        warning_msg = (
            f"Không đủ đặc trưng tương đồng ({num_matches} matches, {num_inliers} inliers) "
            "để căn chỉnh phối cảnh; áp dụng so sánh tọa độ trực tiếp không nắn chỉnh."
        )

    # 2. Lấy danh sách hư hại trên ảnh Before & After
    det_before: List[Dict[str, Any]] = []
    det_after: List[Dict[str, Any]] = []

    if before_detections is not None:
        det_before = list(before_detections)
    elif pipeline is not None:
        res_before = pipeline.run(img_before)
        det_before = getattr(res_before, "detections", [])

    if after_detections is not None:
        det_after = list(after_detections)
    elif pipeline is not None:
        res_after = pipeline.run(img_after)
        det_after = getattr(res_after, "detections", [])

    comparison_items: List[DamageComparisonItem] = []
    old_damages: List[Dict[str, Any]] = []
    new_damages: List[Dict[str, Any]] = []

    # Chuẩn bị warped boxes cho tất cả vết After
    warped_boxes_list: List[List[float]] = []
    for d_after in det_after:
        raw_box_after = d_after.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        if aligned and H is not None:
            warped_box = warp_bbox_with_homography(raw_box_after, H, clip_hw=(h_before, w_before))
        else:
            warped_box = [float(v) for v in raw_box_after]
        warped_boxes_list.append(warped_box)

    # Tính ma trận IoU hàng loạt bằng supervision (sv.box_iou_batch) nếu khả dụng
    iou_matrix: Optional[np.ndarray] = None
    if sv is not None and det_after and det_before:
        try:
            after_arr = np.array([b[:4] for b in warped_boxes_list], dtype=np.float32)
            before_arr = np.array(
                [d.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])[:4] for d in det_before],
                dtype=np.float32,
            )
            iou_matrix = sv.box_iou_batch(after_arr, before_arr)
        except Exception:
            iou_matrix = None

    # 3. Duyệt từng vết trên ảnh After để so khớp với Before
    for idx_after, d_after in enumerate(det_after):
        raw_box_after = d_after.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        dmg_type = d_after.get("damage_type", "unknown")
        confidence = float(d_after.get("confidence", 0.0))
        part_name = d_after.get("part") or d_after.get("located_on_part") or "khong_xac_dinh"
        warped_box = warped_boxes_list[idx_after]

        best_iou = 0.0
        best_before_idx: Optional[int] = None
        best_before_type: Optional[str] = None
        best_before_box: Optional[List[float]] = None

        # Tìm vết tổn thất tương ứng nhất trên Before
        if iou_matrix is not None and iou_matrix.shape[1] > 0:
            row_ious = iou_matrix[idx_after]
            max_idx = int(np.argmax(row_ious))
            max_val = float(row_ious[max_idx])
            if max_val > 0.0:
                best_iou = max_val
                best_before_idx = max_idx
                best_before_type = det_before[max_idx].get("damage_type")
                best_before_box = det_before[max_idx].get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        else:
            for idx_before, d_before in enumerate(det_before):
                box_before = d_before.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
                iou_val = calculate_bbox_iou(warped_box, box_before)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_before_idx = idx_before
                    best_before_type = d_before.get("damage_type")
                    best_before_box = box_before

        # Phân loại: IoU >= 0.6 -> old_damage; IoU < 0.6 -> new_damage
        is_old = (best_iou >= iou_threshold)
        status_str = "old_damage" if is_old else "new_damage"
        msg = (
            f"Vết tổn thất cũ đã có trước khi thuê (IoU: {best_iou:.2f} >= {iou_threshold:.2f})"
            if is_old
            else f"Vết tổn thất MỚI phát sinh trong chuyến đi (IoU: {best_iou:.2f} < {iou_threshold:.2f})"
        )

        item = DamageComparisonItem(
            after_damage_id=idx_after,
            damage_type=dmg_type,
            confidence=confidence,
            located_on_part=part_name,
            original_bbox_after=raw_box_after,
            warped_bbox_in_before=warped_box,
            best_matching_before_id=best_before_idx,
            best_matching_before_type=best_before_type,
            best_matching_before_bbox=best_before_box,
            iou_with_before=round(best_iou, 4),
            status=status_str,
            message=msg,
        )
        comparison_items.append(item)

        item_dict = {
            "id": idx_after,
            "damage_type": dmg_type,
            "confidence": confidence,
            "located_on_part": part_name,
            "bbox_after": raw_box_after,
            "warped_bbox_before": warped_box,
            "matched_before_id": best_before_idx,
            "iou": round(best_iou, 4),
            "status": status_str,
            "message": msg,
        }

        if is_old:
            old_damages.append(item_dict)
        else:
            new_damages.append(item_dict)

    h_matrix_list = H.tolist() if H is not None else None

    return ChangeDetectionResult(
        success=True,
        aligned=aligned,
        feature_method=used_method,
        num_matches=num_matches,
        num_inliers=num_inliers,
        homography_matrix=h_matrix_list,
        total_before_damages=len(det_before),
        total_after_damages=len(det_after),
        num_old_damages=len(old_damages),
        num_new_damages=len(new_damages),
        comparison_items=comparison_items,
        old_damages=old_damages,
        new_damages=new_damages,
        warning=warning_msg,
    )


def visualize_change_detection(
    img_before_path: Union[str, Path, np.ndarray],
    img_after_path: Union[str, Path, np.ndarray],
    result: ChangeDetectionResult,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Tạo ảnh trực quan so sánh Change Detection:
    Ghép side-by-side Before và After:
    - Vết cũ (old_damage): Đóng khung màu XANH LÁ CÂY (GREEN).
    - Vết mới (new_damage): Đóng khung màu ĐỎ (RED) nổi bật.
    """
    img_before = load_image_bgr(img_before_path).copy()
    img_after = load_image_bgr(img_after_path).copy()

    # Vẽ các tổn thất trên ảnh Before (màu vàng nhạt)
    for item in result.comparison_items:
        if item.best_matching_before_bbox:
            bx1, by1, bx2, by2 = [int(v) for v in item.best_matching_before_bbox]
            cv2.rectangle(img_before, (bx1, by1), (bx2, by2), (0, 215, 255), 2)
            cv2.putText(
                img_before,
                f"Before: {item.best_matching_before_type}",
                (bx1, max(18, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                2,
            )

    # Vẽ các tổn thất trên ảnh After
    for item in result.comparison_items:
        x1, y1, x2, y2 = [int(v) for v in item.original_bbox_after]
        if item.status == "old_damage":
            color = (0, 255, 0)      # Xanh lá: Tổn thất cũ
            label = f"[OLD] {item.damage_type} (IoU={item.iou_with_before:.2f})"
        else:
            color = (0, 0, 255)      # Đỏ: Tổn thất mới phát sinh
            label = f"[NEW] {item.damage_type} ({item.located_on_part})"

        cv2.rectangle(img_after, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img_after,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Ghép ngang 2 ảnh (Side-by-side)
    h1, w1 = img_before.shape[:2]
    h2, w2 = img_after.shape[:2]
    target_h = max(h1, h2)

    # Scale về cùng chiều cao nếu khác nhau
    if h1 != target_h:
        w1_new = int(w1 * target_h / h1)
        img_before = cv2.resize(img_before, (w1_new, target_h))
    if h2 != target_h:
        w2_new = int(w2 * target_h / h2)
        img_after = cv2.resize(img_after, (w2_new, target_h))

    # Thêm tiêu đề cho từng khung ảnh
    cv2.putText(img_before, "CHECK-IN (BEFORE)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(img_after, "CHECK-OUT (AFTER)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    canvas = np.hstack([img_before, img_after])

    # Banner tóm tắt phía dưới
    banner = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    summary_txt = f"Tong hop: {result.num_old_damages} ton that cu | {result.num_new_damages} TON THAT MOI PHAT SINH | Alignment: {'OK' if result.aligned else 'Fallback'}"
    cv2.putText(banner, summary_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    final_vis = np.vstack([canvas, banner])

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), final_vis)

    return final_vis
