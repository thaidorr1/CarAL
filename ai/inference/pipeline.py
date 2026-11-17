"""Two-Stage Inference Pipeline cho AutoClaim-AI.

Stage 1 - Damage Detection  : YOLOv8 (detect/seg) phân đoạn 6 lớp hư hại (CarDD).
Stage 2 - Part Segmentation : YOLOv8-seg phân đoạn 7 bộ phận ngoại thất xe.

Hai model chạy song song (ThreadPoolExecutor) -> Post-processing map hư hại vào bộ phận:
- Ưu tiên 1: Tỷ lệ đè phủ thực tế của Segmentation Mask (IoU / Overlap Ratio).
- Ưu tiên 2: Khoảng cách tâm (Center Distance) chuẩn hóa nếu không có giao cắt trực tiếp
             hoặc cho các lớp hư hại đặc thù (vỡ kính, xẹp lốp...).
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("autoclaim.pipeline")

# ---------------------------------------------------------------------------
# 1. BẢNG TÊN BỘ PHẬN NGOẠI THẤT XE (7 lớp Model 2)
# ---------------------------------------------------------------------------

#: 5 Loại Tổn Thất Stage 1 chuẩn (Hardcoded Stage 1 Dictionary)
STAGE_1_DAMAGE_CLASSES: List[str] = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
]

#: 8 Loại Bộ Phận Xe Stage 2 chuẩn (Hardcoded Stage 2 Dictionary)
STANDARD_PARTS: List[str] = [
    "Nắp capo",
    "Cửa trước",
    "Cửa sau",
    "Cản trước",
    "Cản sau",
    "Kính xe",
    "Đèn xe",
    "Vè che bánh",
]

STAGE_2_PART_CLASSES: List[str] = list(STANDARD_PARTS)


class StandardPartStr(str):
    """Chuỗi đại diện cho 1 trong 8 bộ phận chuẩn.
    Hỗ trợ so sánh case-insensitive và tương thích ngược với nhãn cũ (cản trước, vè che bánh, kính lái...).
    """
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if super().__eq__(other):
                return True
            s1 = self.strip().lower()
            s2 = other.strip().lower()
            if s1 == s2:
                return True
            # Tương thích với các biến thể có/không có dấu hoặc tiền tố/hậu tố
            if s1 == "cửa trước" and (s2 in ("cua_truoc", "cua_truoc_trai", "cua_truoc_phai") or s2.startswith("cửa trước")):
                return True
            if s1 == "cửa sau" and (s2 in ("cua_sau", "cua_sau_trai", "cua_sau_phai") or s2.startswith("cửa sau")):
                return True
            if s1 == "kính xe" and (
                s2 in ("kính lái", "kính chiếu hậu", "kinh_lai", "kinh_chieu_hau", "kinh_cua_hong", "kính cửa hông", "kinh_xe")
                or "kính" in s2
            ):
                return True
            if s1 == "cản trước" and s2 in ("can_truoc", "cantruoc"):
                return True
            if s1 == "cản sau" and s2 in ("can_sau", "cansau"):
                return True
            if s1 == "nắp capo" and s2 in ("nap_capo", "capo"):
                return True
            if s1 == "vè che bánh" and s2 in ("ve_che_banh", "ve_sau", "ve_truoc", "vè sau", "vè trước"):
                return True
            if s1 == "đèn xe" and s2 in ("den_xe", "đèn", "den"):
                return True
            return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class DisplayLabelStr(str):
    """Chuỗi nhãn hiển thị tương thích so sánh hoa/thường."""
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return super().__eq__(other) or self.strip().lower() == other.strip().lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


def map_to_standard_part(part_name: Optional[str]) -> str:
    """Map bất kỳ tên part nào về đúng 1 trong 8 class chuẩn hoặc 'khong_xac_dinh'."""
    if not part_name:
        return "khong_xac_dinh"
    raw = str(part_name).strip()
    raw_lower = raw.lower().replace("-", "_").replace(" ", "_")

    if raw_lower in ("khong_xac_dinh", "không_xác_định", "unknown", "none"):
        return "khong_xac_dinh"

    for sp in STANDARD_PARTS:
        if raw.lower() == sp.lower() or raw_lower == sp.lower().replace(" ", "_"):
            return StandardPartStr(sp)

    if raw_lower in ("can_truoc", "cản_trước", "cantruoc", "front_bumper"):
        return StandardPartStr("Cản trước")
    if raw_lower in ("can_sau", "cản_sau", "cansau", "rear_bumper"):
        return StandardPartStr("Cản sau")
    if raw_lower in ("nap_capo", "nắp_capo", "capo", "hood"):
        return StandardPartStr("Nắp capo")
    if raw_lower in ("ve_che_banh", "vè_che_bánh", "ve_sau", "vè_sau", "ve_truoc", "fender", "quarter_panel", "tai_xe"):
        return StandardPartStr("Vè che bánh")
    if any(k in raw_lower for k in ("cua_truoc", "cửa_trước", "front_door")):
        return StandardPartStr("Cửa trước")
    if any(k in raw_lower for k in ("cua_sau", "cửa_sau", "rear_door")):
        return StandardPartStr("Cửa sau")
    if raw_lower in ("cua_hong", "cửa_hông", "door"):
        return StandardPartStr("Cửa trước")
    if raw_lower in ("kinh_xe", "kính_xe", "kinh_lai", "kính_lái", "kinh_chieu_hau", "kính_chiếu_hậu",
                     "kinh_cua_hong", "kính_cửa_hông", "windshield", "window", "glass"):
        return StandardPartStr("Kính xe")
    if raw_lower in ("den_xe", "đèn_xe", "headlight", "taillight", "lamp", "den_pha", "den_sau"):
        return StandardPartStr("Đèn xe")
    if raw_lower in ("bánh_xe_/_lốp_xe", "bánh_xe", "lốp_xe", "tire", "wheel"):
        return "Bánh xe / Lốp xe"

    return "khong_xac_dinh"

#: 5 Nhóm Hư Hỏng chuẩn theo Capstone Proposal (Stage 1 - In-Scope)
PROPOSAL_5_DAMAGE_CLASSES: Dict[str, str] = {
    "scratch":        "Vết cào xước (Scratch)",
    "dent":           "Vết móp/lõm (Dent)",
    "crack":          "Rạn nứt kính/đèn (Crack)",
    "tear_puncture":  "Rách thủng cản nhựa (Tear/Puncture)",
    "paint_chipping": "Bong tróc lớp sơn (Paint Chipping)",
}

#: Bảng gộp (merge) các nhãn dư CarDD về đúng 5 class chuẩn Capstone Proposal
DAMAGE_CLASS_MERGE_MAP: Dict[str, Optional[str]] = {
    "scratch":        "scratch",
    "dent":           "dent",
    "crack":          "crack",
    "glass_shatter":  "crack",
    "glass shatter":  "crack",
    "lamp_broken":    "crack",
    "lamp broken":    "crack",
    "tear_puncture":  "tear_puncture",
    "tear puncture":  "tear_puncture",
    "paint_chipping": "paint_chipping",
    "paint chipping": "paint_chipping",
    "tire_flat":      None,
    "tire flat":      None,
}


def normalize_damage_class(raw_damage_type: str) -> Optional[str]:
    """Chuẩn hóa và gộp (merge) nhãn hư hỏng về đúng 5 class chuẩn của Capstone Proposal."""
    norm = " ".join(str(raw_damage_type).strip().lower().replace("_", " ").split())
    norm_under = norm.replace(" ", "_")
    if norm_under in DAMAGE_CLASS_MERGE_MAP:
        return DAMAGE_CLASS_MERGE_MAP[norm_under]
    if norm in DAMAGE_CLASS_MERGE_MAP:
        return DAMAGE_CLASS_MERGE_MAP[norm]
    return norm_under


def _safe_print(text: str) -> None:
    """In log Unicode an toàn trên Windows (tránh lỗi charmap cp1252 encode error)."""
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except (UnicodeEncodeError, Exception):
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            else:
                safe_text = text.encode("ascii", errors="replace").decode("ascii")
                sys.stdout.write(safe_text + "\n")
        except Exception:
            pass


#: 4 vị trí cửa hông được phân giải từ class AI 'cua_hong' bằng Topology
FOUR_SIDE_DOOR_NAMES: Tuple[str, str, str, str] = (
    "Cửa trước trái",
    "Cửa sau trái",
    "Cửa trước phải",
    "Cửa sau phải",
)

#: Tên bộ phận theo thứ tự class-id chuẩn của model_parts (parts_seg.pt: 0..6, mở rộng 7 cho den_xe)
PART_CLASS_NAMES: Dict[int, str] = {
    0: "can_truoc",       # cản trước (front bumper)
    1: "can_sau",         # cản sau (rear bumper)
    2: "nap_capo",        # nắp capo (hood)
    3: "ve_che_banh",     # vè che bánh / tai xe (fender / quarter panel)
    4: "cua_hong",        # cửa hông (door -> Topology chia thành Cửa trước / Cửa sau)
    5: "kinh_chieu_hau",  # gương chiếu hậu (side mirror)
    6: "kinh_lai",        # kính lái / kính chắn gió (windshield)
    7: "den_xe",          # đèn xe (headlight / taillight)
}

#: Tên bộ phận tiếng Việt chuẩn có dấu phục vụ hiển thị trên giao diện và báo cáo
#: (Khóa chặt đúng 8 bộ phận Stage 2: Nắp capo, Cửa trước, Cửa sau, Cản trước, Cản sau, Kính xe, Đèn xe, Vè che bánh)
PART_DISPLAY_NAMES: Dict[str, StandardPartStr] = {
    "can_truoc":      StandardPartStr("Cản trước"),
    "can_sau":        StandardPartStr("Cản sau"),
    "nap_capo":       StandardPartStr("Nắp capo"),
    "ve_che_banh":    StandardPartStr("Vè che bánh"),
    "cua_hong":       StandardPartStr("Cửa trước"),
    "cua_truoc":      StandardPartStr("Cửa trước"),
    "cua_sau":        StandardPartStr("Cửa sau"),
    "cua_truoc_trai": StandardPartStr("Cửa trước"),
    "cua_sau_trai":   StandardPartStr("Cửa sau"),
    "cua_truoc_phai": StandardPartStr("Cửa trước"),
    "cua_sau_phai":   StandardPartStr("Cửa sau"),
    "kinh_chieu_hau": StandardPartStr("Kính xe"),
    "kinh_lai":       StandardPartStr("Kính xe"),
    "kinh_cua_hong":  StandardPartStr("Kính xe"),
    "den_xe":         StandardPartStr("Đèn xe"),
    "khong_xac_dinh": "khong_xac_dinh",  # Ảnh chụp quá cận cảnh không rõ bối cảnh
}

#: Từ điển dịch tên vết tổn thất sang tiếng Việt dùng cho log hiển thị
DAMAGE_NAME_VI: Dict[str, str] = {
    "dent":           "móp",
    "scratch":        "xước",
    "crack":          "nứt",
    "glass_shatter":  "vỡ kính",
    "glass shatter":  "vỡ kính",
    "lamp_broken":    "vỡ đèn",
    "lamp broken":    "vỡ đèn",
    "tire_flat":      "xẹp lốp",
    "tire flat":      "xẹp lốp",
    "tear_puncture":  "rách thủng",
    "tear puncture":  "rách thủng",
    "paint_chipping": "tróc sơn",
    "paint chipping": "tróc sơn",
}

#: Quy tắc liên kết ưu tiên theo ngữ nghĩa hư hại (Domain Knowledge)
PRIORITY_DAMAGE_PART_MAPPING: Dict[str, List[str]] = {
    "glass_shatter":  ["kinh_lai", "kinh_chieu_hau"],
    "glass shatter":  ["kinh_lai", "kinh_chieu_hau"],
    "tire_flat":      ["ve_che_banh"],
    "tire flat":      ["ve_che_banh"],
    "lamp_broken":    ["den_xe"],
    "lamp broken":    ["den_xe"],
    "dent":           ["ve_che_banh", "nap_capo", "can_truoc", "can_sau", "cua_hong"],
    "scratch":        ["cua_hong", "nap_capo", "can_truoc", "can_sau", "ve_che_banh"],
    "crack":          ["can_truoc", "can_sau", "nap_capo", "cua_hong"],
    "tear_puncture":  ["can_truoc", "can_sau", "ve_che_banh"],
    "tear puncture":  ["can_truoc", "can_sau", "ve_che_banh"],
    "paint_chipping": ["cua_hong", "can_truoc", "can_sau", "nap_capo", "ve_che_banh"],
    "paint chipping": ["cua_hong", "can_truoc", "can_sau", "nap_capo", "ve_che_banh"],
}

#: Cặp hư hại và bộ phận cấm gán theo nguyên lý vật lý (Physical Semantic Guard)
FORBIDDEN_DAMAGE_PART_PAIRS: Dict[str, Set[str]] = {
    # dent (móp lún kim loại/nhựa) tuyệt đối CẤM gán vào kính (thủy tinh giòn không thể móp), đèn, bánh xe
    "dent":           {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    # scratch (trầy xước sơn thân vỏ) CẤM gán vào kính, đèn, bánh xe
    "scratch":        {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    "crack":          {"wheel", "tire"},
    # tear_puncture (rách thủng cản nhựa) & paint_chipping (bong tróc sơn) CẤM gán vào kính/đèn/bánh xe
    "tear_puncture":  {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    "tear puncture":  {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    "paint_chipping": {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    "paint chipping": {"kinh_lai", "kinh_sau", "kinh_chieu_hau", "kinh_cua_hong", "den_xe", "wheel", "tire"},
    # glass_shatter (vỡ kính) tuyệt đối CẤM gán trực tiếp vào các tấm vỏ kim loại/nhựa
    "glass_shatter":  {"nap_capo", "can_truoc", "can_sau", "ve_che_banh", "cua_hong"},
    "glass shatter":  {"nap_capo", "can_truoc", "can_sau", "ve_che_banh", "cua_hong"},
    # lamp_broken (vỡ đèn) chỉ thuộc đèn xe; CẤM gán vào cửa hông, kính lái, nắp capo, cản trước, cản sau, vè
    "lamp_broken":    {"cua_hong", "kinh_lai", "nap_capo", "can_truoc", "can_sau", "ve_che_banh"},
    "lamp broken":    {"cua_hong", "kinh_lai", "nap_capo", "can_truoc", "can_sau", "ve_che_banh"},
    # tire_flat (xẹp lốp) CẤM gán vào tất cả các bộ phận thân vỏ (cản trước, cản sau, nắp capo, cửa hông, kính, vè...)
    "tire_flat":      {"can_truoc", "can_sau", "nap_capo", "cua_hong", "kinh_lai", "kinh_chieu_hau", "ve_che_banh", "den_xe"},
    "tire flat":      {"can_truoc", "can_sau", "nap_capo", "cua_hong", "kinh_lai", "kinh_chieu_hau", "ve_che_banh", "den_xe"},
}

#: Bộ phận có diện tích lớn cần tăng ngưỡng Overlap tối thiểu (Threshold Tuning: 0.20)
LARGE_PART_KEYS: Set[str] = {"cua_hong", "kinh_lai"}

#: Quy tắc suy luận ngữ nghĩa khi chụp cận cảnh (Close-up Semantic Fallback)
CLOSEUP_SEMANTIC_FALLBACK: Dict[str, Tuple[str, str]] = {
    # damage_type: (part_key, display_name) - 100% chuẩn hóa 7 lớp DOCX
    "glass_shatter": ("kinh_lai", "kính lái"),
    "glass shatter": ("kinh_lai", "kính lái"),
    "tire_flat":     ("banh_xe", "Bánh xe / Lốp xe"),
    "tire flat":     ("banh_xe", "Bánh xe / Lốp xe"),
    "lamp_broken":   ("can_truoc", "cản trước"),
    "lamp broken":   ("can_truoc", "cản trước"),
    "dent":          ("cua_hong", "cửa hông"),
    "scratch":       ("cua_hong", "cửa hông"),
    "crack":         ("can_truoc", "cản trước"),
}


# ---------------------------------------------------------------------------
# 2. DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class DamageResult:
    """Kết quả từ Stage 1 (Damage model)."""
    damage_type: str
    confidence: float
    damage_polygon: List[List[float]]    # [[x, y], ...]
    bbox_xyxy: List[float]              # [x1, y1, x2, y2]
    center_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class PartResult:
    """Kết quả từ Stage 2 (Part model)."""
    part_key: str                        # khóa nội bộ (ví dụ: 'nap_capo')
    part_name: str                        # tên tiếng Việt chuẩn (ví dụ: 'nắp capo')
    confidence: float
    part_polygon: List[List[float]]      # [[x, y], ...]
    bbox_xyxy: List[float]
    center_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class MappedDetection:
    """Một vùng hư hại sau khi được map vào bộ phận xe tương ứng."""
    damage_type: str
    confidence: float
    located_on_part: str                 # tên bộ phận (chuẩn 7 lớp DOCX)
    overlap_ratio: float                 # tỷ lệ đè phủ (Intersection / Damage Area) [0.0 - 1.0]
    mask_iou: float                      # IoU giữa mask hư hại và mask bộ phận [0.0 - 1.0]
    center_distance: float               # khoảng cách giữa 2 tâm (pixels)
    match_method: str                    # 'mask_overlap', 'center_distance', 'micro_feature_bumper', 'damage_only_fallback'...
    damage_polygon: List[List[float]]
    part_polygon: List[List[float]]      # rỗng nếu không xác định được bộ phận
    bbox_xyxy: List[float]
    display_label: str = ""              # format: 'scratch -> cản trước [0.85]'
    warning: str = ""                    # Cảnh báo bối cảnh ảnh (ví dụ: ảnh quá cận cảnh)
    action_required: str = "Kết quả phân tích hợp lệ."


@dataclass
class PipelineResponse:
    """Phản hồi JSON chuẩn hóa cho client."""
    success: bool
    image_width: int
    image_height: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    parts_detected: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    shot_type: str = "mid_view"
    status: str = "completed"
    action_required: str = "Kết quả phân tích hợp lệ."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "shot_type": self.shot_type,
            "action_required": self.action_required,
            "image": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "detections": self.detections,
            "parts_detected": self.parts_detected,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# 3. HELPER: TÍNH TOÁN HÌNH HỌC (IoU, Overlap, Center Distance)
# ---------------------------------------------------------------------------

def calculate_center(polygon_or_bbox: Sequence[Sequence[float]] | Sequence[float]) -> Tuple[float, float]:
    """Tính tọa độ tâm centroid từ polygon hoặc bounding box."""
    if not polygon_or_bbox:
        return 0.0, 0.0

    if isinstance(polygon_or_bbox[0], (int, float)) and len(polygon_or_bbox) >= 4:
        x1, y1, x2, y2 = polygon_or_bbox[:4]
        return round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)

    pts = np.asarray(polygon_or_bbox, dtype=np.float32)
    if len(pts) == 0:
        return 0.0, 0.0
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    return round(cx, 1), round(cy, 1)


# ---------------------------------------------------------------------------
# SPATIAL COORDINATE INFERENCE & ORIENTATION
# ---------------------------------------------------------------------------

@dataclass
class OrientationInfo:
    """Thông tin định hướng không gian của xe."""
    view_angle: str = "unknown"        # "front", "rear", "left_side", "right_side", "front_diagonal", "rear_diagonal", "unknown"
    head_direction: str = "unknown"    # "left", "right", "facing", "away", "unknown"
    b_pillar_x: Optional[float] = None
    confidence: float = 0.0
    is_reliable: bool = False
    details: str = ""


def determine_car_orientation(
    parts: Optional[Sequence[PartResult]] = None,
    damages: Optional[Sequence[DamageResult]] = None,
    image_hw: Tuple[int, int] = (640, 640),
    wheel_bboxes: Optional[Sequence[Sequence[float]]] = None,
    filename: str = "",
) -> OrientationInfo:
    """Xác định hướng xe đa tiêu chí dựa trên bánh xe, cụm đèn, kính lái, cản và hình học không gian."""
    h, w = image_hw
    parts_list = list(parts) if parts else []
    fname_lower = (filename or "").lower()

    # 1. Thu thập điểm neo phía trước (chỉ nhận mỏ neo tin cậy conf >= 0.45)
    front_cues_x: List[float] = []
    for p in parts_list:
        if p.confidence >= 0.45:
            pcx = p.center_xy[0] if p.center_xy != (0.0, 0.0) else (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
            if p.part_key in ("can_truoc", "nap_capo", "kinh_chieu_hau", "headlamp", "den_pha", "fog_light"):
                front_cues_x.append(pcx)

    # 2. Thu thập điểm neo phía sau (chỉ nhận mỏ neo tin cậy conf >= 0.45)
    rear_cues_x: List[float] = []
    for p in parts_list:
        if p.confidence >= 0.45:
            pcx = p.center_xy[0] if p.center_xy != (0.0, 0.0) else (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
            if p.part_key in ("can_sau", "tail_light", "den_sau", "cop_xe"):
                rear_cues_x.append(pcx)

    # 3. Thu thập bánh xe / vè che bánh
    wheel_x: List[float] = []
    for p in parts_list:
        if p.part_key in ("ve_che_banh", "wheel", "tire") and p.confidence >= 0.20:
            wcx = p.center_xy[0] if p.center_xy != (0.0, 0.0) else (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
            wheel_x.append(wcx)
    if wheel_bboxes:
        for wb in wheel_bboxes:
            if len(wb) >= 4:
                wheel_x.append((wb[0] + wb[2]) / 2.0)

    # Cửa xe
    door_parts = [p for p in parts_list if p.part_key == "cua_hong"]

    # Tính B-pillar X
    b_pillar_x = w / 2.0 if w > 0 else 320.0
    if len(door_parts) >= 2:
        sorted_doors = sorted(
            door_parts,
            key=lambda dp: dp.center_xy[0] if dp.center_xy != (0.0, 0.0) else (dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0,
        )
        d1_w = max(1.0, sorted_doors[0].bbox_xyxy[2] - sorted_doors[0].bbox_xyxy[0])
        overlap_x = max(0.0, min(sorted_doors[0].bbox_xyxy[2], sorted_doors[-1].bbox_xyxy[2]) - max(sorted_doors[0].bbox_xyxy[0], sorted_doors[-1].bbox_xyxy[0]))
        if overlap_x > 0.40 * d1_w:
            b_pillar_x = (sorted_doors[0].bbox_xyxy[0] + sorted_doors[0].bbox_xyxy[2]) / 2.0
        else:
            b_pillar_x = (sorted_doors[0].center_xy[0] + sorted_doors[-1].center_xy[0]) / 2.0
    elif len(wheel_x) >= 2 and (max(wheel_x) - min(wheel_x)) >= 0.15 * max(1, w):
        b_pillar_x = (min(wheel_x) + max(wheel_x)) / 2.0
    elif len(door_parts) == 1:
        dp = door_parts[0]
        dx1, _, dx2, _ = dp.bbox_xyxy
        dcx = (dx1 + dx2) / 2.0
        dw = max(1.0, dx2 - dx1)
        if len(wheel_x) == 1 and dw < 0.45 * max(1, w):
            fw_x = wheel_x[0]
            if front_cues_x and not rear_cues_x:
                b_pillar_x = dx2 + 1.0 if fw_x <= dcx else dx1 - 1.0
            elif rear_cues_x and not front_cues_x:
                b_pillar_x = dx1 - 1.0 if fw_x >= dcx else dx2 + 1.0
            else:
                b_pillar_x = dcx
        else:
            b_pillar_x = dcx

    head_dir = "unknown"
    view_angle = "unknown"
    is_reliable = False
    confidence = 0.50
    details = ""

    if front_cues_x and rear_cues_x:
        mean_front = float(np.mean(front_cues_x))
        mean_rear = float(np.mean(rear_cues_x))
        diff = mean_rear - mean_front
        if abs(diff) > 0.08 * w:
            head_dir = "left" if diff > 0 else "right"
            is_reliable = True
            confidence = 0.90
            details = f"Cả neo trước và sau xác nhận đầu xe bên {head_dir}"
            view_angle = "left_side" if head_dir == "left" else "right_side"
        else:
            details = "Neo trước và sau đồng trục X"
    elif front_cues_x:
        mean_front = float(np.mean(front_cues_x))
        if door_parts:
            door_cx = float(np.mean([(dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0 for dp in door_parts]))
            if mean_front < door_cx - 0.06 * w:
                head_dir = "left"
                is_reliable = True
                confidence = 0.85
                view_angle = "front_diagonal"
                details = "Neo trước bên trái cửa xe"
            elif mean_front > door_cx + 0.06 * w:
                head_dir = "right"
                is_reliable = True
                confidence = 0.85
                view_angle = "front_diagonal"
                details = "Neo trước bên phải cửa xe"
            else:
                head_dir = "facing"
                is_reliable = True
                confidence = 0.75
                view_angle = "front"
                details = "Neo trước chính diện"
        elif any(p.part_key == "can_truoc" for p in parts_list) and any(p.part_key == "nap_capo" for p in parts_list):
            head_dir = "facing"
            view_angle = "front"
            is_reliable = True
            confidence = 0.88
            details = "Cản trước và nắp capo chính diện"
        elif len(wheel_x) == 1:
            fw_x = wheel_x[0]
            head_dir = "left" if fw_x <= b_pillar_x else "right"
            is_reliable = True
            confidence = 0.70
            details = "Định vị qua neo trước và 1 bánh xe"
    elif rear_cues_x:
        mean_rear = float(np.mean(rear_cues_x))
        if door_parts:
            door_cx = float(np.mean([(dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0 for dp in door_parts]))
            if mean_rear > door_cx + 0.06 * w:
                head_dir = "left"
                is_reliable = True
                confidence = 0.85
                view_angle = "rear_diagonal"
                details = "Neo sau bên phải cửa xe -> đầu xe bên trái"
            elif mean_rear < door_cx - 0.06 * w:
                head_dir = "right"
                is_reliable = True
                confidence = 0.85
                view_angle = "rear_diagonal"
                details = "Neo sau bên trái cửa xe -> đầu xe bên phải"
            else:
                head_dir = "away"
                view_angle = "rear"
                is_reliable = True
                confidence = 0.75
                details = "Neo sau chính diện"
        elif any(p.part_key == "can_sau" for p in parts_list):
            head_dir = "away"
            view_angle = "rear"
            is_reliable = True
            confidence = 0.85
            details = "Cản sau chính diện đuôi xe"

    # Hỗ trợ qua tên file nếu thị giác không đủ
    if not is_reliable and fname_lower:
        if any(k in fname_lower for k in ("1_front_left", "8_left_side", "7_rear_left", "left")):
            head_dir = "left"
            is_reliable = True
            confidence = 0.65
            details = f"Định vị hướng từ nhãn file: {filename}"
        elif any(k in fname_lower for k in ("3_front_right", "4_right_side", "5_rear_right", "right")):
            head_dir = "right"
            is_reliable = True
            confidence = 0.65
            details = f"Định vị hướng từ nhãn file: {filename}"
        elif "front" in fname_lower:
            head_dir = "facing"
            view_angle = "front"
            is_reliable = True
            confidence = 0.65
        elif "rear" in fname_lower:
            head_dir = "away"
            view_angle = "rear"
            is_reliable = True
            confidence = 0.65

    return OrientationInfo(
        view_angle=view_angle,
        head_direction=head_dir,
        b_pillar_x=round(b_pillar_x, 1),
        confidence=confidence,
        is_reliable=is_reliable,
        details=details,
    )


def resolve_door_position_from_topology(
    target_bbox_or_center: Union[Sequence[float], Tuple[float, float]],
    parts: Optional[Sequence[PartResult]],
    image_hw: Tuple[int, int],
    wheel_bboxes: Optional[Sequence[Sequence[float]]] = None,
    filename: str = "",
) -> str:
    """Phân giải class AI 'cua_hong' thành 'Cửa trước' hoặc 'Cửa sau' (neutral topology)."""
    h, w = image_hw
    if isinstance(target_bbox_or_center, (int, float)):
        target_cx = float(target_bbox_or_center)
    elif len(target_bbox_or_center) >= 4:
        target_cx = (float(target_bbox_or_center[0]) + float(target_bbox_or_center[2])) / 2.0
    elif len(target_bbox_or_center) >= 2:
        target_cx = float(target_bbox_or_center[0])
    else:
        target_cx = w / 2.0 if w > 0 else 0.0

    parts_list = list(parts) if parts else []
    fname_lower = (filename or "").lower()

    door_parts = [p for p in parts_list if p.part_key == "cua_hong"]
    front_cues_x: List[float] = []
    rear_cues_x: List[float] = []
    fender_wheel_x: List[float] = []

    for p in parts_list:
        pcx = p.center_xy[0] if p.center_xy != (0.0, 0.0) else (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
        if p.part_key in ("can_truoc", "nap_capo", "kinh_chieu_hau", "headlamp", "den_pha", "fog_light") and p.confidence >= 0.45:
            front_cues_x.append(pcx)
        elif p.part_key in ("can_sau", "tail_light", "den_sau", "cop_xe") and p.confidence >= 0.45:
            rear_cues_x.append(pcx)
        elif p.part_key in ("ve_che_banh", "wheel", "tire") and p.confidence >= 0.45:
            fender_wheel_x.append(pcx)

    if wheel_bboxes:
        for wb in wheel_bboxes:
            if len(wb) >= 4:
                fender_wheel_x.append((wb[0] + wb[2]) / 2.0)

    clean_door_parts = list(door_parts)
    if rear_cues_x and fender_wheel_x and len(door_parts) >= 2:
        rear_x = float(np.mean(rear_cues_x))
        rw_x = min(fender_wheel_x, key=lambda fx: abs(fx - rear_x))
        if rear_x >= rw_x:
            clean_door_parts = [dp for dp in door_parts if (dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0 <= rw_x]
        else:
            clean_door_parts = [dp for dp in door_parts if (dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0 >= rw_x]
        if not clean_door_parts:
            clean_door_parts = list(door_parts)

    dw = max(1.0, clean_door_parts[0].bbox_xyxy[2] - clean_door_parts[0].bbox_xyxy[0]) if clean_door_parts else 0.0

    if len(clean_door_parts) >= 2:
        sorted_doors = sorted(
            clean_door_parts,
            key=lambda dp: dp.center_xy[0] if dp.center_xy != (0.0, 0.0) else (dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0,
        )
        left_door_cx = sorted_doors[0].center_xy[0] if sorted_doors[0].center_xy != (0.0, 0.0) else (sorted_doors[0].bbox_xyxy[0] + sorted_doors[0].bbox_xyxy[2]) / 2.0
        right_door_cx = sorted_doors[-1].center_xy[0] if sorted_doors[-1].center_xy != (0.0, 0.0) else (sorted_doors[-1].bbox_xyxy[0] + sorted_doors[-1].bbox_xyxy[2]) / 2.0
        d1_w = max(1.0, sorted_doors[0].bbox_xyxy[2] - sorted_doors[0].bbox_xyxy[0])
        overlap_x = max(0.0, min(sorted_doors[0].bbox_xyxy[2], sorted_doors[-1].bbox_xyxy[2]) - max(sorted_doors[0].bbox_xyxy[0], sorted_doors[-1].bbox_xyxy[0]))
        if overlap_x > 0.40 * d1_w:
            b_pillar_x = (sorted_doors[0].bbox_xyxy[0] + sorted_doors[0].bbox_xyxy[2]) / 2.0
        else:
            b_pillar_x = (left_door_cx + right_door_cx) / 2.0
    elif len(fender_wheel_x) >= 2 and (max(fender_wheel_x) - min(fender_wheel_x)) >= 0.15 * max(1, w):
        b_pillar_x = (min(fender_wheel_x) + max(fender_wheel_x)) / 2.0
    elif len(clean_door_parts) == 1:
        dp = clean_door_parts[0]
        dx1, _, dx2, _ = dp.bbox_xyxy
        dcx = (dx1 + dx2) / 2.0
        dw = max(1.0, dx2 - dx1)
        if len(fender_wheel_x) == 1 and dw < 0.45 * max(1, w):
            fw_x = fender_wheel_x[0]
            if front_cues_x and not rear_cues_x:
                b_pillar_x = dx2 + 1.0 if fw_x <= dcx else dx1 - 1.0
            elif rear_cues_x and not front_cues_x:
                b_pillar_x = dx1 - 1.0 if fw_x >= dcx else dx2 + 1.0
            else:
                b_pillar_x = dcx
        else:
            b_pillar_x = dcx
    else:
        b_pillar_x = w / 2.0 if w > 0 else 320.0

    orient = determine_car_orientation(parts_list, [], image_hw, wheel_bboxes=wheel_bboxes, filename=filename)
    if orient.is_reliable:
        is_front_left = (orient.head_direction == "left")
        if orient.b_pillar_x is not None:
            b_pillar_x = orient.b_pillar_x
    else:
        is_front_left = True
        if front_cues_x and rear_cues_x:
            is_front_left = (float(np.mean(front_cues_x)) <= float(np.mean(rear_cues_x)))
        elif front_cues_x:
            is_front_left = (float(np.mean(front_cues_x)) <= b_pillar_x)
        elif rear_cues_x:
            is_front_left = (float(np.mean(rear_cues_x)) >= b_pillar_x)
        elif len(fender_wheel_x) == 1:
            is_front_left = (fender_wheel_x[0] <= b_pillar_x)

    # Chỉ phân tách Cửa trước / Cửa sau khi có mỏ neo hướng xe tin cậy (conf >= 0.45) hoặc >= 2 mask cửa rõ ràng.
    # Nếu chỉ có 1 mask cua_hong độc lập mà không rõ hướng đầu/đuôi, giữ nhãn chung an toàn (Cửa trước), không ép mù quáng thành Cửa sau.
    has_hood = any(p.part_key == "nap_capo" and p.confidence >= 0.45 for p in parts_list)
    has_reliable_anchor = bool(front_cues_x or rear_cues_x or orient.is_reliable)

    if len(clean_door_parts) == 1 and not has_reliable_anchor:
        is_front_door = True
    elif len(clean_door_parts) == 1 and has_hood and not rear_cues_x:
        is_front_door = True
    elif len(clean_door_parts) == 1 and dw >= (0.45 * max(1, w)) and has_reliable_anchor:
        is_front_door = (target_cx <= b_pillar_x) if is_front_left else (target_cx >= b_pillar_x)
    elif len(clean_door_parts) >= 2 and rear_cues_x:
        rear_cx = float(np.mean(rear_cues_x))
        sorted_by_rear = sorted(
            clean_door_parts,
            key=lambda dp: abs(((dp.bbox_xyxy[0] + dp.bbox_xyxy[2]) / 2.0) - rear_cx)
        )
        closest_door_cx = (sorted_by_rear[0].bbox_xyxy[0] + sorted_by_rear[0].bbox_xyxy[2]) / 2.0
        furthest_door_cx = (sorted_by_rear[-1].bbox_xyxy[0] + sorted_by_rear[-1].bbox_xyxy[2]) / 2.0
        mid_doors = (closest_door_cx + furthest_door_cx) / 2.0
        is_front_door = (abs(target_cx - rear_cx) > abs(mid_doors - rear_cx))
    elif rear_cues_x and not front_cues_x:
        is_front_door = False
    elif front_cues_x and not rear_cues_x:
        is_front_door = True
    else:
        is_front_door = (target_cx <= b_pillar_x) if is_front_left else (target_cx >= b_pillar_x)

    explicit_left = any(k in fname_lower for k in ("left", "trai", "7_rear_left", "8_left_side", "1_front_left"))
    explicit_right = any(k in fname_lower for k in ("right", "phai", "3_front_right", "4_right_side", "5_rear_right"))

    if explicit_left and not explicit_right:
        side_suffix = " trái"
    elif explicit_right and not explicit_left:
        side_suffix = " phải"
    else:
        side_suffix = ""

    door_type = "Cửa trước" if is_front_door else "Cửa sau"
    return StandardPartStr(f"{door_type}{side_suffix}")


def resolve_four_side_doors(
    parts: Sequence[PartResult],
    image_hw: Tuple[int, int],
    wheel_bboxes: Optional[Sequence[Sequence[float]]] = None,
    filename: str = "",
) -> List[Dict[str, Any]]:
    """Chia class AI 'cua_hong' trong danh sách bộ phận thành các JSON output riêng biệt."""
    h, w = image_hw
    resolved_json: List[Dict[str, Any]] = []
    parts_list = list(parts)
    door_parts = [p for p in parts_list if p.part_key == "cua_hong"]

    for p in parts_list:
        if p.part_key != "cua_hong":
            resolved_json.append({
                "part_key": p.part_key,
                "part_name": p.part_name,
                "confidence": p.confidence,
                "part_polygon": p.part_polygon,
                "bbox_xyxy": p.bbox_xyxy,
            })
            continue

        x1, y1, x2, y2 = p.bbox_xyxy
        bw = max(0.0, x2 - x1)
        has_hood_part = any(pt.part_key == "nap_capo" and pt.confidence >= 0.45 for pt in parts_list)
        has_reliable_anchor = any(
            pt.part_key in ("nap_capo", "can_truoc", "can_sau", "kinh_chieu_hau") and pt.confidence >= 0.45
            for pt in parts_list
        )
        if len(door_parts) == 1 and w > 0 and bw >= (0.35 * w) and not has_hood_part and has_reliable_anchor:
            b_pillar_x = round((x1 + x2) / 2.0, 1)
            left_bbox = [x1, y1, b_pillar_x, y2]
            right_bbox = [b_pillar_x, y1, x2, y2]

            left_name = resolve_door_position_from_topology(
                (x1 + b_pillar_x) / 2.0, parts_list, image_hw, wheel_bboxes=wheel_bboxes, filename=filename
            )
            right_name = resolve_door_position_from_topology(
                (b_pillar_x + x2) / 2.0, parts_list, image_hw, wheel_bboxes=wheel_bboxes, filename=filename
            )
            if left_name == right_name:
                if "trái" in left_name:
                    left_name, right_name = "Cửa trước trái", "Cửa sau trái"
                elif "phải" in left_name:
                    left_name, right_name = "Cửa sau phải", "Cửa trước phải"
                else:
                    left_name, right_name = "Cửa trước", "Cửa sau"

            key_map = {
                "Cửa trước trái": "cua_truoc_trai",
                "Cửa sau trái": "cua_sau_trai",
                "Cửa trước phải": "cua_truoc_phai",
                "Cửa sau phải": "cua_sau_phai",
                "Cửa trước": "cua_truoc",
                "Cửa sau": "cua_sau",
            }
            resolved_json.append({
                "part_key": key_map.get(left_name, "cua_hong"),
                "part_name": left_name,
                "confidence": p.confidence,
                "part_polygon": _bbox_to_polygon(left_bbox),
                "bbox_xyxy": left_bbox,
            })
            resolved_json.append({
                "part_key": key_map.get(right_name, "cua_hong"),
                "part_name": right_name,
                "confidence": p.confidence,
                "part_polygon": _bbox_to_polygon(right_bbox),
                "bbox_xyxy": right_bbox,
            })
        else:
            door_label = resolve_door_position_from_topology(
                p.bbox_xyxy, parts_list, image_hw, wheel_bboxes=wheel_bboxes, filename=filename
            )
            key_map = {
                "Cửa trước trái": "cua_truoc_trai",
                "Cửa sau trái": "cua_sau_trai",
                "Cửa trước phải": "cua_truoc_phai",
                "Cửa sau phải": "cua_sau_phai",
                "Cửa trước": "cua_truoc",
                "Cửa sau": "cua_sau",
            }
            resolved_json.append({
                "part_key": key_map.get(door_label, "cua_hong"),
                "part_name": door_label,
                "confidence": p.confidence,
                "part_polygon": p.part_polygon,
                "bbox_xyxy": p.bbox_xyxy,
            })

    return resolved_json


def fallback_topology_routing(
    bbox_xyxy: Sequence[float],
    image_hw: Tuple[int, int],
    wheel_bboxes: Optional[Sequence[Sequence[float]]] = None,
    headlamp_bbox: Optional[Sequence[float]] = None,
    has_rear_lamp: bool = False,
    is_rear_view: bool = False,
    rear_anchors: Optional[Sequence[Tuple[float, float, str]]] = None,
    has_front_features: bool = False,
    filename: str = "",
    parts: Optional[Sequence[PartResult]] = None,
    damages: Optional[Sequence[DamageResult]] = None,
    image_bgr: Optional[np.ndarray] = None,
    damage_type: str = "",
) -> Tuple[str, str]:
    """HÀM FALLBACK TOPOLOGY ROUTING ĐƠN GIẢN HÓA: Tìm Bounding Box GẦN NHẤT trong khoảng 20% khung hình."""
    h, w = image_hw
    if h <= 0 or w <= 0 or len(bbox_xyxy) < 4:
        return "khong_xac_dinh", "khong_xac_dinh"

    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
    parts_list = list(parts) if parts else []
    if not parts_list:
        return "khong_xac_dinh", "khong_xac_dinh"

    MAX_DISTANCE = 0.20 * float(max(w, h))

    nearest_part: Optional[PartResult] = None
    min_distance = float("inf")

    for p in parts_list:
        px1, py1, px2, py2 = p.bbox_xyxy
        dx = max(0.0, px1 - cx, cx - px2)
        dy = max(0.0, py1 - cy, cy - py2)
        box_dist = math.hypot(dx, dy)

        if box_dist < min_distance:
            min_distance = box_dist
            nearest_part = p

    if nearest_part is not None and min_distance <= MAX_DISTANCE:
        if nearest_part.part_key == "cua_hong":
            door_name = resolve_door_position_from_topology(
                bbox_xyxy, parts_list, image_hw, wheel_bboxes=wheel_bboxes, filename=filename
            )
            return "cua_hong", door_name
        return nearest_part.part_key, nearest_part.part_name

    return "khong_xac_dinh", "khong_xac_dinh"


def infer_part_from_spatial_coordinates(
    bbox_xyxy: Sequence[float],
    image_hw: Tuple[int, int],
    wheel_bboxes: Optional[Sequence[Sequence[float]]] = None,
    headlamp_bbox: Optional[Sequence[float]] = None,
    has_rear_lamp: bool = False,
    is_rear_view: bool = False,
    rear_anchors: Optional[Sequence[Tuple[float, float, str]]] = None,
    has_front_features: bool = False,
    filename: str = "",
    parts: Optional[Sequence[PartResult]] = None,
    damages: Optional[Sequence[DamageResult]] = None,
    image_bgr: Optional[np.ndarray] = None,
) -> Tuple[str, str]:
    """Tương thích ngược: Ủy quyền toàn bộ xử lý sang fallback_topology_routing."""
    return fallback_topology_routing(
        bbox_xyxy=bbox_xyxy,
        image_hw=image_hw,
        wheel_bboxes=wheel_bboxes,
        headlamp_bbox=headlamp_bbox,
        has_rear_lamp=has_rear_lamp,
        is_rear_view=is_rear_view,
        rear_anchors=rear_anchors,
        has_front_features=has_front_features,
        filename=filename,
        parts=parts,
        damages=damages,
        image_bgr=image_bgr,
    )


def calculate_polygon_overlap_and_iou(
    poly_damage: Sequence[Sequence[float]],
    poly_part: Sequence[Sequence[float]],
    canvas_hw: Tuple[int, int],
) -> Tuple[float, float]:
    """Tính tỷ lệ đè phủ (Overlap ratio) và IoU giữa polygon hư hại và polygon bộ phận."""
    if len(poly_damage) < 3 or len(poly_part) < 3:
        return 0.0, 0.0

    try:
        from shapely.geometry import Polygon

        p1 = Polygon(poly_damage)
        p2 = Polygon(poly_part)

        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)

        if p1.is_valid and p2.is_valid and not p1.is_empty and not p2.is_empty:
            area1 = float(p1.area)
            if area1 > 0:
                inter = float(p1.intersection(p2).area)
                union = float(p1.union(p2).area)
                overlap = round(inter / area1, 4)
                iou = round(inter / union, 4) if union > 0 else 0.0
                return min(1.0, max(0.0, overlap)), min(1.0, max(0.0, iou))
    except Exception:
        pass

    h, w = canvas_hw
    if h <= 0 or w <= 0:
        return 0.0, 0.0

    mask_d = np.zeros((h, w), dtype=np.uint8)
    mask_p = np.zeros((h, w), dtype=np.uint8)

    pts_d = np.clip(np.asarray(poly_damage, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)
    pts_p = np.clip(np.asarray(poly_part, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)

    cv2.fillPoly(mask_d, [pts_d], 1)
    cv2.fillPoly(mask_p, [pts_p], 1)

    area_d = int(mask_d.sum())
    if area_d == 0:
        return 0.0, 0.0

    inter_mask = np.logical_and(mask_d == 1, mask_p == 1)
    union_mask = np.logical_or(mask_d == 1, mask_p == 1)

    intersection = int(inter_mask.sum())
    union = int(union_mask.sum())

    overlap = round(intersection / area_d, 4)
    iou = round(intersection / union, 4) if union > 0 else 0.0

    return min(1.0, max(0.0, overlap)), min(1.0, max(0.0, iou))


def _bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
    """Chuyển đổi Bounding Box [x1, y1, x2, y2] thành Polygon 4 góc."""
    if len(bbox) < 4:
        return []
    x1, y1, x2, y2 = bbox[:4]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------------------
# 4. INFERENCE WORKERS
# ---------------------------------------------------------------------------

def _run_damage_model(
    model,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> List[DamageResult]:
    """Chạy inference Model 1 (Damage)."""
    effective_conf = float(conf)
    effective_iou = float(iou)
    results = model.predict(
        image_bgr,
        imgsz=imgsz,
        conf=effective_conf,
        iou=effective_iou,
        device=device,
        retina_masks=True,
        verbose=False,
    )
    out: List[DamageResult] = []
    if not results:
        return out

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return out

    has_masks = res.masks is not None and getattr(res.masks, "xy", None) is not None

    for i, box in enumerate(res.boxes):
        cls_id = int(box.cls.item())
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        cls_name_clean = str(cls_name).strip().lower().replace(" ", "_")
        if cls_name_clean not in STAGE_1_DAMAGE_CLASSES:
            continue
        conf_val = float(box.conf.item())
        if cls_name_clean == "dent" and conf_val < 0.35:
            continue
        bbox = box.xyxy[0].tolist() if hasattr(box, "xyxy") and len(box.xyxy) > 0 else [0.0, 0.0, 0.0, 0.0]

        polygon: List[List[float]] = []
        if has_masks and i < len(res.masks.xy):
            raw = res.masks.xy[i]
            polygon = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            if len(polygon) >= 3:
                poly_np = np.asarray(polygon, dtype=np.float32)
                bbox = [
                    float(poly_np[:, 0].min()),
                    float(poly_np[:, 1].min()),
                    float(poly_np[:, 0].max()),
                    float(poly_np[:, 1].max()),
                ]

        c_source = polygon if len(polygon) >= 3 else bbox
        center = calculate_center(c_source)

        out.append(DamageResult(
            damage_type=cls_name,
            confidence=round(conf_val, 4),
            damage_polygon=polygon,
            bbox_xyxy=[round(float(v), 1) for v in bbox],
            center_xy=center,
        ))
    return out


def _run_parts_model(
    model,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    part_class_names: Dict[int, str],
) -> List[PartResult]:
    """Chạy inference Model 2 (Part Segmentation - 7 lớp bộ phận xe)."""
    results = model.predict(
        image_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        retina_masks=True,
        verbose=False,
    )
    out: List[PartResult] = []
    if not results:
        return out

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return out

    has_masks = res.masks is not None and getattr(res.masks, "xy", None) is not None

    for i, box in enumerate(res.boxes):
        cls_id = int(box.cls.item())
        raw_name = res.names.get(cls_id, part_class_names.get(cls_id, str(cls_id))) if hasattr(res, "names") else part_class_names.get(cls_id, str(cls_id))
        part_key = str(raw_name).strip().lower().replace(" ", "_")
        part_name = PART_DISPLAY_NAMES.get(part_key, raw_name)
        conf_val = float(box.conf.item())
        bbox = box.xyxy[0].tolist() if hasattr(box, "xyxy") and len(box.xyxy) > 0 else [0.0, 0.0, 0.0, 0.0]

        polygon: List[List[float]] = []
        if has_masks and i < len(res.masks.xy):
            raw = res.masks.xy[i]
            polygon = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            if len(polygon) >= 3:
                poly_np = np.asarray(polygon, dtype=np.float32)
                bbox = [
                    float(poly_np[:, 0].min()),
                    float(poly_np[:, 1].min()),
                    float(poly_np[:, 0].max()),
                    float(poly_np[:, 1].max()),
                ]

        c_source = polygon if len(polygon) >= 3 else bbox
        center = calculate_center(c_source)

        if part_key == "kinh_chieu_hau":
            h_img, w_img = image_bgr.shape[:2]
            bw = max(0.0, bbox[2] - bbox[0])
            bh = max(0.0, bbox[3] - bbox[1])
            if (bw * bh) > (0.10 * w_img * h_img) or bw > (0.35 * w_img) or bh > (0.35 * h_img):
                continue

        out.append(PartResult(
            part_key=part_key,
            part_name=part_name,
            confidence=round(conf_val, 4),
            part_polygon=polygon,
            bbox_xyxy=[round(float(v), 1) for v in bbox],
            center_xy=center,
        ))

    return out


# ---------------------------------------------------------------------------
# 5. POST-PROCESSING: GỘP MASK PHÂN MẢNH & LỌC RÁC CONTOUR
# ---------------------------------------------------------------------------

def merge_and_smooth_part_masks(
    parts: List[PartResult],
    image_hw: Tuple[int, int],
    min_area_pixels: float = 600.0,
    morph_closing_kernel_size: int = 15,
) -> List[PartResult]:
    """Gộp các mask cùng nhãn bị phân mảnh thành khối liền mạch và lọc bỏ rác contour nhỏ."""
    if not parts:
        return []

    h, w = image_hw
    if h <= 0 or w <= 0:
        return parts

    hood_parts = [p for p in parts if p.part_key == "nap_capo"]
    left_hoods = [p for p in hood_parts if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) < (0.35 * w) and p.bbox_xyxy[0] <= (0.12 * w)]
    right_hoods = [p for p in hood_parts if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) > (0.65 * w) and p.bbox_xyxy[2] >= (0.88 * w)]
    has_split_corner_hoods = bool(left_hoods and right_hoods)

    if has_split_corner_hoods:
        door_parts_existing = [p for p in parts if p.part_key == "cua_hong" and p.confidence >= 0.45]
        rear_parts_existing = [p for p in parts if p.part_key == "can_sau" and p.confidence >= 0.45]
        if door_parts_existing or rear_parts_existing:
            max_door_or_rear = max([p.confidence for p in door_parts_existing + rear_parts_existing] or [0.0])
            if max_door_or_rear >= 0.45:
                parts = [p for p in parts if p.part_key != "nap_capo"]
                max_door_c = max([p.confidence for p in door_parts_existing] or [0.0])
                left_rb = [p for p in rear_parts_existing if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) < (0.40 * w)]
                right_rb = [p for p in rear_parts_existing if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) > (0.60 * w)]
                if max_door_c >= 0.80 and left_rb and right_rb:
                    parts = [p for p in parts if p.part_key not in ("can_sau", "can_truoc")]

    strong_centered_windshield = [
        p for p in parts
        if p.part_key == "kinh_lai" and p.confidence >= 0.25
        and ((p.bbox_xyxy[2] - p.bbox_xyxy[0]) >= 0.42 * w)
        and p.bbox_xyxy[1] <= (0.20 * h) and p.bbox_xyxy[3] <= (0.45 * h)
    ]
    front_bumper_cands = [p for p in parts if p.part_key == "can_truoc" and p.confidence >= 0.20]
    rear_bumper_cands = [p for p in parts if p.part_key == "can_sau" and p.confidence >= 0.20]
    has_front_hood_or_bumper = any(
        p.part_key in ("nap_capo", "can_truoc") and p.confidence >= 0.20
        for p in parts
    )
    is_front_dominant = bool(
        front_bumper_cands and (
            not rear_bumper_cands
            or max(p.confidence for p in front_bumper_cands) >= max(p.confidence for p in rear_bumper_cands)
            or any(p.confidence >= 0.65 for p in strong_centered_windshield)
        )
    )
    if strong_centered_windshield and has_front_hood_or_bumper and is_front_dominant:
        kl = strong_centered_windshield[0]
        kl_cx = (kl.bbox_xyxy[0] + kl.bbox_xyxy[2]) / 2.0
        has_corner_hood = any(
            p.part_key == "nap_capo" and abs(((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) - kl_cx) > (0.20 * w)
            for p in parts
        )
        hood_bottom_y: Optional[float] = None
        if not has_corner_hood:
            for p in parts:
                pcx = (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
                pcy = (p.bbox_xyxy[1] + p.bbox_xyxy[3]) / 2.0
                if p.part_key == "cua_hong" and abs(pcx - kl_cx) <= (0.18 * w) and pcy >= kl.bbox_xyxy[1]:
                    p.part_key = "nap_capo"
                    p.part_name = PART_DISPLAY_NAMES["nap_capo"]
                    hood_bottom_y = p.bbox_xyxy[3]
                elif p.part_key == "can_sau" and pcy >= kl.bbox_xyxy[3]:
                    p.part_key = "can_truoc"
                    p.part_name = PART_DISPLAY_NAMES["can_truoc"]
        if hood_bottom_y is not None:
            for p in parts:
                if p.part_key == "can_truoc" and p.bbox_xyxy[1] > hood_bottom_y and (p.bbox_xyxy[1] - hood_bottom_y) <= (0.12 * h):
                    new_x1 = min(p.bbox_xyxy[0], kl.bbox_xyxy[0])
                    new_x2 = max(p.bbox_xyxy[2], kl.bbox_xyxy[2])
                    new_y1 = max(0.0, hood_bottom_y - 0.01 * h)
                    new_y2 = p.bbox_xyxy[3]
                    p.bbox_xyxy = [new_x1, new_y1, new_x2, new_y2]
                    p.part_polygon = [
                        [int(new_x1), int(new_y1)],
                        [int(new_x2), int(new_y1)],
                        [int(new_x2), int(new_y2)],
                        [int(new_x1), int(new_y2)],
                    ]
                    p.center_xy = ((new_x1 + new_x2) / 2.0, (new_y1 + new_y2) / 2.0)

    strong_doors = [p for p in parts if p.part_key == "cua_hong" and p.confidence >= 0.45]
    rear_bumpers = [p for p in parts if p.part_key == "can_sau" and p.confidence >= 0.45]
    side_mirrors = [p for p in parts if p.part_key == "kinh_chieu_hau" and p.confidence >= 0.45]
    has_front_bumper = any(p.part_key == "can_truoc" and p.confidence >= 0.35 for p in parts)

    if strong_doors and (rear_bumpers or side_mirrors) and not has_front_bumper:
        d_min_x = min(d.bbox_xyxy[0] for d in strong_doors)
        d_max_x = max(d.bbox_xyxy[2] for d in strong_doors)
        d_min_y = min(d.bbox_xyxy[1] for d in strong_doors)
        d_max_y = max(d.bbox_xyxy[3] for d in strong_doors)
        d_cx = (d_min_x + d_max_x) / 2.0
        max_door_conf = max(d.confidence for d in strong_doors)

        rectified_parts: List[PartResult] = []
        for p in parts:
            px1, py1, px2, py2 = p.bbox_xyxy
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            pw = max(1.0, px2 - px1)

            if p.part_key == "nap_capo" and p.confidence < min(0.55, max_door_conf):
                if side_mirrors:
                    mirror_cx = sum((m.bbox_xyxy[0] + m.bbox_xyxy[2]) / 2.0 for m in side_mirrors) / len(side_mirrors)
                    if (mirror_cx < d_cx and pcx > d_cx) or (mirror_cx > d_cx and pcx < d_cx):
                        continue
                if rear_bumpers and pcy >= (0.32 * h) and py2 >= (d_min_y + 0.10 * h):
                    rb_cx = sum((r.bbox_xyxy[0] + r.bbox_xyxy[2]) / 2.0 for r in rear_bumpers) / len(rear_bumpers)
                    if (rb_cx > d_cx and pcx < d_cx and px2 >= d_min_x - 0.05 * w) or (rb_cx < d_cx and pcx > d_cx and px1 <= d_max_x + 0.05 * w):
                        p.part_key = "cua_hong"
                        p.part_name = PART_DISPLAY_NAMES["cua_hong"]

            elif p.part_key == "kinh_lai" and rear_bumpers:
                x_overlap_door = max(0.0, min(px2, d_max_x) - max(px1, d_min_x))
                if (x_overlap_door / pw) >= 0.55 and py2 <= (d_min_y + 0.12 * h) and py2 >= (d_min_y - 0.15 * h):
                    p.part_key = "kinh_cua_hong"
                    p.part_name = PART_DISPLAY_NAMES["kinh_cua_hong"]

            rectified_parts.append(p)
        parts = rectified_parts

    grouped: Dict[str, List[PartResult]] = {}
    for p in parts:
        grouped.setdefault(p.part_key, []).append(p)

    out_parts: List[PartResult] = []
    k_size = max(3, morph_closing_kernel_size if morph_closing_kernel_size % 2 == 1 else morph_closing_kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    for p_key, plist in grouped.items():
        class_canvas = np.zeros((h, w), dtype=np.uint8)
        for p in plist:
            poly = p.part_polygon if len(p.part_polygon) >= 3 else _bbox_to_polygon(p.bbox_xyxy)
            if len(poly) >= 3:
                pts = np.clip(np.asarray(poly, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)
                cv2.fillPoly(class_canvas, [pts], 1)

        closed_canvas = cv2.morphologyEx(class_canvas, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        part_name = plist[0].part_name
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area_pixels:
                continue

            approx = cv2.approxPolyDP(cnt, 2.0, closed=True)
            if len(approx) < 3:
                continue

            new_poly = approx.reshape(-1, 2).tolist()
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            new_bbox = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
            new_center = calculate_center(new_poly)

            # Khớp contour với instance gốc trong plist để giữ đúng confidence riêng của từng instance
            matched_confs: List[Tuple[float, float]] = []
            for p in plist:
                bx1 = max(new_bbox[0], p.bbox_xyxy[0])
                by1 = max(new_bbox[1], p.bbox_xyxy[1])
                bx2 = min(new_bbox[2], p.bbox_xyxy[2])
                by2 = min(new_bbox[3], p.bbox_xyxy[3])
                inter_w = max(0.0, bx2 - bx1)
                inter_h = max(0.0, by2 - by1)
                inter_area = inter_w * inter_h
                if inter_area > 0:
                    matched_confs.append((inter_area, p.confidence))

            if matched_confs:
                matched_confs.sort(key=lambda item: item[0], reverse=True)
                top_area = matched_confs[0][0]
                significant = [c for a, c in matched_confs if a >= 0.20 * top_area]
                cnt_conf = max(significant)
            else:
                cnt_conf = plist[0].confidence

            out_parts.append(PartResult(
                part_key=p_key,
                part_name=part_name,
                confidence=round(cnt_conf, 4),
                part_polygon=new_poly,
                bbox_xyxy=new_bbox,
                center_xy=new_center,
            ))

    if not out_parts and parts:
        best_orig = max(parts, key=lambda p: p.confidence)
        out_parts.append(best_orig)

    return out_parts


# ---------------------------------------------------------------------------
# 5b. POST-PROCESSING: LỌC MASK CHỒNG LẤN (LOCAL SPATIAL NMS / CLIPPING)
# ---------------------------------------------------------------------------

def calculate_point_to_part_boundary_distance(
    point_xy: Tuple[float, float],
    part: PartResult,
) -> float:
    """Tính khoảng cách Euclid từ một điểm (cx, cy) đến ranh giới gần nhất của Part."""
    cx, cy = float(point_xy[0]), float(point_xy[1])
    poly = part.part_polygon
    if poly and len(poly) >= 3:
        pts = np.asarray(poly, dtype=np.float32)
        dist_val = cv2.pointPolygonTest(pts, (cx, cy), True)
        if dist_val >= 0:
            return 0.0
        return float(abs(dist_val))

    if len(part.bbox_xyxy) >= 4:
        px1, py1, px2, py2 = [float(v) for v in part.bbox_xyxy[:4]]
        dx = max(0.0, max(px1 - cx, cx - px2))
        dy = max(0.0, max(py1 - cy, cy - py2))
        return float(math.hypot(dx, dy))

    return float("inf")


def resolve_overlapping_part_masks(
    parts: List[PartResult],
    image_hw: Tuple[int, int],
    iou_threshold: float = 0.50,
    min_area_keep_ratio: float = 0.15,
    min_contour_area: float = 200.0,
) -> List[PartResult]:
    """Local Spatial NMS & Mask Clipping: Lọc xung đột cục bộ và cắt tỉa mask trùng lặp."""
    if len(parts) <= 1:
        return parts

    h, w = image_hw
    if h <= 0 or w <= 0:
        return parts

    sorted_parts = sorted(parts, key=lambda p: p.confidence, reverse=True)

    kept_parts: List[PartResult] = []
    kept_masks: List[np.ndarray] = []

    for p in sorted_parts:
        poly = p.part_polygon if len(p.part_polygon) >= 3 else _bbox_to_polygon(p.bbox_xyxy)
        if len(poly) < 3:
            continue

        mask_p = np.zeros((h, w), dtype=np.uint8)
        pts_p = np.clip(np.asarray(poly, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)
        cv2.fillPoly(mask_p, [pts_p], 1)
        orig_area = int(mask_p.sum())
        if orig_area < min_contour_area:
            continue

        suppressed = False
        suppressing_part: Optional[PartResult] = None
        overlap_iou_val = 0.0

        for kp, km in zip(kept_parts, kept_masks):
            inter = int(np.logical_and(mask_p == 1, km == 1).sum())
            if inter == 0:
                continue
            km_area = int(km.sum())
            union = orig_area + km_area - inter
            iou = inter / union if union > 0 else 0.0
            overlap_on_cand = inter / orig_area if orig_area > 0 else 0.0

            if p.part_key == kp.part_key and (iou > 0.30 or overlap_on_cand > 0.40):
                suppressed = True
                suppressing_part = kp
                overlap_iou_val = iou
                break

            is_contradictory = (
                (p.part_key == "nap_capo" and kp.part_key in ("can_sau", "can_truoc")) or
                (p.part_key in ("can_sau", "can_truoc") and kp.part_key == "nap_capo") or
                (p.part_key == "can_truoc" and kp.part_key == "can_sau") or
                (p.part_key == "can_sau" and kp.part_key == "can_truoc")
            )
            if is_contradictory and (iou > 0.40 or overlap_on_cand > 0.50):
                suppressed = True
                suppressing_part = kp
                overlap_iou_val = iou
                break

        if suppressed and suppressing_part is not None:
            continue

        if kept_masks:
            occupied = np.bitwise_or.reduce(kept_masks)
            inter = np.logical_and(mask_p == 1, occupied == 1)
            inter_area = int(inter.sum())
            if inter_area > 0:
                mask_p[inter] = 0
                remaining_area = int(mask_p.sum())
                if remaining_area < min_contour_area or (remaining_area / orig_area) < min_area_keep_ratio:
                    continue

                cnts, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                largest_c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_c) < min_contour_area:
                    continue

                peri = cv2.arcLength(largest_c, True)
                approx = cv2.approxPolyDP(largest_c, 0.012 * peri, True)
                new_poly = approx.reshape(-1, 2).tolist()
                bx, by, bw, bh = cv2.boundingRect(largest_c)
                new_bbox = [float(bx), float(by), float(bx + bw), float(by + bh)]
                new_cx, new_cy = (bx + bw / 2.0, by + bh / 2.0)

                p = PartResult(
                    part_key=p.part_key,
                    part_name=p.part_name,
                    confidence=p.confidence,
                    part_polygon=new_poly,
                    bbox_xyxy=new_bbox,
                    center_xy=(round(new_cx, 1), round(new_cy, 1)),
                )

        kept_parts.append(p)
        kept_masks.append(mask_p)

    return kept_parts


def _calculate_hsv_red_ratio(
    image_bgr: Optional[np.ndarray],
    bbox_xyxy: Sequence[float],
) -> float:
    """Tính tỷ lệ pixel màu đỏ (đặc trưng đèn hậu) trong vùng bounding box trên không gian màu HSV."""
    if image_bgr is None or image_bgr.size == 0 or len(bbox_xyxy) < 4:
        return 0.0
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(w, int(round(bbox_xyxy[0]))))
    y1 = max(0, min(h, int(round(bbox_xyxy[1]))))
    x2 = max(0, min(w, int(round(bbox_xyxy[2]))))
    y2 = max(0, min(h, int(round(bbox_xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    red_mask = cv2.bitwise_or(mask1, mask2)
    total_px = float((y2 - y1) * (x2 - x1))
    return float(np.count_nonzero(red_mask)) / max(1.0, total_px)


def filter_parts_with_contextual_rules(
    parts: List[PartResult],
    damages: List[DamageResult],
    image_hw: Tuple[int, int],
    image_bgr: Optional[np.ndarray] = None,
) -> List[PartResult]:
    """Lọc ngữ cảnh thông minh cho Stage 2 (Contextual Part Filtering):
    1. Tuyệt đối KHÔNG tự sinh vùng bộ phận giả (synthetic) và KHÔNG xóa mask thật để thay bằng hộp giả.
    2. Lọc bỏ các mask ảo giác (can_sau, cua_hong conf thấp) khi không có mỏ neo hướng xe tin cậy (conf >= 0.45).
    3. Khử nhập nhằng Cản trước vs Cản sau ở đuôi xe dựa trên cụm màu đỏ đèn hậu (HSV red ratio >= 0.12) phía trên cản.
    4. Chống đảo ngược Đầu - Đuôi khi có mỏ neo rõ ràng.
    5. Lọc nhiễu dent conf < 0.35 và lọc các bounding box hư hỏng trùng lặp / lồng nhau hoàn toàn (Nested Redundant Damage NMS).
    """
    h, w = image_hw
    if h <= 0 or w <= 0:
        return parts

    # Lọc bỏ các khung dent nhiễu độ tin cậy thấp (conf < 0.35)
    damages[:] = [d for d in damages if not (d.damage_type.lower() == "dent" and d.confidence < 0.35)]

    pts = list(parts)
    dmgs = list(damages)
    img_area = float(max(1, w * h))

    broken_lamps = [d for d in dmgs if d.damage_type.lower() in ("lamp_broken", "lamp broken") and d.confidence >= 0.25]

    # Khử nhập nhằng Cản trước vs Cản sau ở đuôi xe:
    # Nếu mask là can_truoc nhưng vùng 30% chiều cao ngay phía trên cản hoặc cụm lamp_broken phía trên có tỷ lệ pixel đỏ HSV >= 0.12 -> chuyển thành can_sau
    if image_bgr is not None and pts:
        rectified_bumper_pts: List[PartResult] = []
        for p in pts:
            if p.part_key == "can_truoc":
                px1, py1, px2, py2 = p.bbox_xyxy
                pcy = (py1 + py2) / 2.0
                upper_zone = [px1, max(0.0, py1 - 0.30 * h), px2, min(float(h), py1 + 0.05 * h)]
                upper_red_ratio = _calculate_hsv_red_ratio(image_bgr, upper_zone)
                lamp_red_ratio = max(
                    (
                        _calculate_hsv_red_ratio(image_bgr, bl.bbox_xyxy)
                        for bl in broken_lamps
                        if ((bl.bbox_xyxy[1] + bl.bbox_xyxy[3]) / 2.0) <= pcy
                    ),
                    default=0.0,
                )
                if upper_red_ratio >= 0.12 or lamp_red_ratio >= 0.12:
                    p = PartResult(
                        part_key="can_sau",
                        part_name=PART_DISPLAY_NAMES["can_sau"],
                        confidence=p.confidence,
                        part_polygon=p.part_polygon,
                        bbox_xyxy=p.bbox_xyxy,
                        center_xy=p.center_xy,
                    )
            rectified_bumper_pts.append(p)
        pts = rectified_bumper_pts

    # 1. Nếu chỉ có các mask kinh_lai yếu (< 0.35) mà không có bộ phận thân vỏ nào khác -> loại bỏ nhiễu
    if pts and all(p.part_key == "kinh_lai" and p.confidence < 0.35 for p in pts):
        pts = []

    # 2. Nếu ảnh chụp cận cảnh mặt kính vỡ (glass_shatter + kinh_lai chiếm >= 75% diện tích khung hình),
    # loại bỏ các mask thân vỏ ảo giác (can_sau, cua_hong, nap_capo, ve_che_banh) sinh ra do vân nứt kính
    full_frame_shatter = any(
        d.damage_type.lower() in ("glass_shatter", "glass shatter")
        and d.confidence >= 0.50
        and ((d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]) / img_area) >= 0.75
        for d in dmgs
    )
    full_frame_windshield = any(
        p.part_key == "kinh_lai"
        and p.confidence >= 0.50
        and ((p.bbox_xyxy[2] - p.bbox_xyxy[0]) * (p.bbox_xyxy[3] - p.bbox_xyxy[1]) / img_area) >= 0.75
        for p in pts
    )
    if full_frame_shatter and full_frame_windshield:
        pts = [p for p in pts if "kinh" in p.part_key]

    # 3. Nâng ngưỡng lọc nhiễu cho can_sau và cua_hong khi không có mỏ neo xác nhận (conf >= 0.45)
    has_reliable_front_anchor = any(
        p.part_key in ("nap_capo", "can_truoc", "kinh_chieu_hau") and p.confidence >= 0.45
        for p in pts
    )
    has_reliable_rear_anchor = any(
        p.part_key in ("can_sau", "tail_light", "den_sau", "cop_xe") and p.confidence >= 0.45
        for p in pts
    )
    has_any_reliable_anchor = has_reliable_front_anchor or has_reliable_rear_anchor

    pts = [
        p for p in pts
        if not (
            (p.part_key == "can_sau" and p.confidence < 0.45 and not has_reliable_rear_anchor)
            or (p.part_key == "cua_hong" and p.confidence < 0.42 and not has_any_reliable_anchor)
        )
    ]

    hoods = [p for p in pts if p.part_key == "nap_capo" and p.confidence >= 0.45]
    windshields = [p for p in pts if p.part_key == "kinh_lai" and p.confidence >= 0.45]
    broken_lamps = [d for d in dmgs if d.damage_type.lower() in ("lamp_broken", "lamp broken") and d.confidence >= 0.25]

    # 4. Góc chụp ĐẦU XE (Front / Front-Corner / Front-Side): lọc nhiễu can_sau / cua_hong ảo ở đầu xe
    if hoods or (windshields and broken_lamps):
        front_xs = [((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) for p in hoods]
        if not front_xs and broken_lamps:
            front_xs = [((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0) for d in broken_lamps]
        front_side_is_right = (float(np.mean(front_xs)) >= (0.50 * w)) if front_xs else False

        cleaned_front: List[PartResult] = []
        for p in pts:
            pcx = (p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0
            pcy = (p.bbox_xyxy[1] + p.bbox_xyxy[3]) / 2.0

            # Loại bỏ `cua_hong` ảo đè trực tiếp lên cụm đèn pha vỡ
            if p.part_key == "cua_hong" and broken_lamps:
                lamp_overlaps = False
                for bl in broken_lamps:
                    lx1, ly1, lx2, ly2 = bl.bbox_xyxy
                    ix1, iy1 = max(p.bbox_xyxy[0], lx1), max(p.bbox_xyxy[1], ly1)
                    ix2, iy2 = min(p.bbox_xyxy[2], lx2), min(p.bbox_xyxy[3], ly2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    lamp_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
                    if (inter / lamp_area) >= 0.30 and p.confidence < 0.45:
                        lamp_overlaps = True
                        break
                if lamp_overlaps:
                    continue

            # Loại bỏ `can_sau` ảo xuất hiện cùng phía đầu xe
            if p.part_key == "can_sau":
                is_on_front_side = (pcx >= 0.45 * w) if front_side_is_right else (pcx <= 0.55 * w)
                if is_on_front_side:
                    continue
                if hoods and p.confidence < 0.45 and pcy > 0.55 * h:
                    continue

            cleaned_front.append(p)
        pts = cleaned_front

    # 5. Góc chụp ĐUÔI / HÔNG SAU XE: loại bỏ `nap_capo` ảo độ tin cậy thấp (< 0.35) gần cản sau
    rear_bumpers = [p for p in pts if p.part_key == "can_sau" and p.confidence >= 0.45]
    if rear_bumpers:
        rb_cx = (rear_bumpers[0].bbox_xyxy[0] + rear_bumpers[0].bbox_xyxy[2]) / 2.0
        pts = [
            p for p in pts
            if not (
                p.part_key == "nap_capo"
                and p.confidence < 0.35
                and abs(((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) - rb_cx) <= 0.25 * w
            )
        ]

    # 6. Lọc các bounding box hư hỏng trùng lặp / lồng nhau hoàn toàn (Nested Redundant Damage NMS)
    if len(damages) > 1:
        sorted_d = sorted(damages, key=lambda d: d.confidence, reverse=True)
        kept_d: List[DamageResult] = []
        for cand in sorted_d:
            cx1, cy1, cx2, cy2 = cand.bbox_xyxy
            c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
            is_redundant = False
            for kd in kept_d:
                kx1, ky1, kx2, ky2 = kd.bbox_xyxy
                ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
                ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ioa = inter / c_area
                k_area = max(1.0, (kx2 - kx1) * (ky2 - ky1))
                iou = inter / max(1.0, c_area + k_area - inter)
                # Cùng loại hư hỏng và bị chứa >= 80% trong box có conf cao hơn -> loại bỏ
                if cand.damage_type.lower() == kd.damage_type.lower() and (ioa >= 0.80 or iou >= 0.65):
                    is_redundant = True
                    break
                # Khác loại hư hỏng (dent vs scratch/crack) ở góc cản nhưng trùng khớp > 75% IoU hoặc > 90% IoA và conf thấp (< 0.32)
                if cand.confidence < 0.32 and (iou >= 0.75 or ioa >= 0.90):
                    is_redundant = True
                    break
            if not is_redundant:
                kept_d.append(cand)
        damages[:] = kept_d

    return pts


def filter_false_positive_parts_by_orientation(
    parts: List[PartResult],
    image_hw: Tuple[int, int],
    orient: OrientationInfo,
) -> List[PartResult]:
    """Loại bỏ False Positive cản xe phi vật lý khi định hướng xe đáng tin cậy.
    Quy tắc: Một chiếc xe chỉ có tối đa 1 cản sau và 1 cản trước.
    Không loại bỏ nếu orient.is_reliable = False.
    """
    if not orient.is_reliable or not parts:
        return parts

    h, w = image_hw
    bp_x = orient.b_pillar_x or (w / 2.0)
    filtered = list(parts)

    has_front_anchor = any(
        p.part_key in ("nap_capo", "can_truoc") and p.confidence >= 0.30
        for p in filtered
    )

    # 1. Xử lý Cản sau ở nửa thân trước xe (hoặc trùng lặp)
    rear_bumpers = [p for p in filtered if p.part_key == "can_sau"]
    if len(rear_bumpers) > 1 or (len(rear_bumpers) == 1 and has_front_anchor):
        if orient.head_direction == "left":
            rear_valid = [p for p in rear_bumpers if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) >= (bp_x - 0.05 * w)]
            if rear_valid:
                best_rear = max(rear_valid, key=lambda p: p.confidence)
                filtered = [p for p in filtered if p.part_key != "can_sau" or p is best_rear]
            elif has_front_anchor:
                filtered = [p for p in filtered if p.part_key != "can_sau"]
        elif orient.head_direction == "right":
            rear_valid = [p for p in rear_bumpers if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) <= (bp_x + 0.05 * w)]
            if rear_valid:
                best_rear = max(rear_valid, key=lambda p: p.confidence)
                filtered = [p for p in filtered if p.part_key != "can_sau" or p is best_rear]
            elif has_front_anchor:
                filtered = [p for p in filtered if p.part_key != "can_sau"]

    # 2. Xử lý Cản trước trùng lặp (chỉ giữ cản trước ở nửa thân trước xe)
    front_bumpers = [p for p in filtered if p.part_key == "can_truoc"]
    if len(front_bumpers) > 1:
        if orient.head_direction == "left":
            front_valid = [p for p in front_bumpers if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) <= (bp_x + 0.05 * w)]
            if front_valid:
                best_front = max(front_valid, key=lambda p: p.confidence)
                filtered = [p for p in filtered if p.part_key != "can_truoc" or p is best_front]
        elif orient.head_direction == "right":
            front_valid = [p for p in front_bumpers if ((p.bbox_xyxy[0] + p.bbox_xyxy[2]) / 2.0) >= (bp_x - 0.05 * w)]
            if front_valid:
                best_front = max(front_valid, key=lambda p: p.confidence)
                filtered = [p for p in filtered if p.part_key != "can_truoc" or p is best_front]

    return filtered


# ---------------------------------------------------------------------------
# 6. POST-PROCESSING: GÁN VẾT HƯ HẠI VÀO BỘ PHẬN XE (HYBRID MAPPING)
# ---------------------------------------------------------------------------

def detect_micro_features(
    image_bgr: Optional[np.ndarray],
    damage_bbox: Sequence[float],
    manual_features: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Quy tắc nhận diện chi tiết nhỏ (Micro-feature Context) khi góc chụp quá sát."""
    if manual_features:
        features_norm = [str(f).lower().strip() for f in manual_features]
        if any(f in features_norm for f in ("parking_sensor", "sensor", "cam_bien", "nep_can", "bumper_trim", "luoi_tan_nhiet", "grille")):
            return StandardPartStr("Cản trước")
        if any(f in features_norm for f in ("tay_nam_cua", "door_handle", "khe_ho_cua", "khe_ho_canh_cua", "door_gap", "vien_kinh", "window_trim")):
            return StandardPartStr("Cửa trước")
        if any(f in features_norm for f in ("mep_den", "lamp_edge", "headlight_edge", "mep_cum_den")):
            return "mép đèn"

    return None


def classify_shot_type(
    damages: Sequence[DamageResult],
    parts: Sequence[PartResult],
    image_hw: Tuple[int, int],
    damage_area_threshold: float = 0.30,
    part_conf_threshold: float = 0.35,
    context_anchors: Optional[Sequence[Any]] = None,
) -> Tuple[str, str, str]:
    """Phân loại cự ly chụp ảnh (shot_type: far_view | mid_view | close_up)."""
    h, w = image_hw
    if h <= 0 or w <= 0:
        return "close_up", "Vui lòng chụp bổ sung 1 ảnh toàn cảnh hoặc trung cảnh bao quát góc xe.", "Kích thước ảnh không hợp lệ"

    total_img_area = float(w * h)
    detected_anchors = set()

    has_front_cue = any(
        p.part_key in ("can_truoc", "nap_capo", "kinh_lai", "kinh_chieu_hau", "tru_a", "kinh_hong_truoc")
        for p in parts if p.confidence >= 0.20
    )
    has_rear_cue = any(
        p.part_key in ("can_sau", "tail_light", "den_sau")
        for p in parts if p.confidence >= 0.25
    )

    for d in damages:
        d_type = d.damage_type.lower()
        if ("tire" in d_type or "wheel" in d_type) and d.confidence >= 0.40:
            detected_anchors.add("Bánh xe")
    for p in parts:
        p_key = p.part_key.lower()
        if ("tire" in p_key or "wheel" in p_key or p_key == "ve_che_banh") and p.confidence >= 0.40:
            detected_anchors.add("Bánh xe")

    for p in parts:
        p_key = p.part_key.lower()
        if p.confidence >= 0.40 and p_key in ("kinh_lai", "windshield", "kinh_chieu_hau"):
            detected_anchors.add("Kính lái")

    for p in parts:
        p_key = p.part_key.lower()
        if p.confidence >= 0.40 and p_key in ("nap_capo", "hood"):
            detected_anchors.add("Nắp capo")

    for d in damages:
        d_type = d.damage_type.lower()
        if ("headlamp" in d_type or "fog" in d_type) and d.confidence >= 0.40:
            detected_anchors.add("Đèn pha")
        elif "tail_light" in d_type and d.confidence >= 0.40:
            detected_anchors.add("Đèn sau")
        elif "lamp_broken" in d_type and d.confidence >= 0.40:
            if has_front_cue or (not has_rear_cue and any(p.part_key == "can_truoc" for p in parts)):
                detected_anchors.add("Đèn pha")
            else:
                detected_anchors.add("Đèn sau")

    for p in parts:
        p_key = p.part_key.lower()
        if p.confidence >= 0.40:
            if p_key in ("headlamp", "den_pha", "fog_light", "den_gam", "den_suong_mu"):
                detected_anchors.add("Đèn pha")
            elif p_key in ("tail_light", "den_sau"):
                detected_anchors.add("Đèn sau")
            elif p_key == "den_xe":
                if has_rear_cue:
                    detected_anchors.add("Đèn sau")
                else:
                    detected_anchors.add("Đèn pha")

    for p in parts:
        if p.confidence >= 0.40:
            if p.part_key == "can_truoc":
                detected_anchors.add("Cản trước")
            elif p.part_key == "can_sau":
                detected_anchors.add("Cản sau")
        if p.confidence >= 0.38 and p.part_key in ("cua_hong", "cua_truoc", "cua_sau"):
            detected_anchors.add("Cửa xe")

    if context_anchors:
        for ca in context_anchors:
            if isinstance(ca, (list, tuple)) and len(ca) >= 3:
                detected_anchors.add(str(ca[2]))

    anchor_count = len(detected_anchors)

    if anchor_count >= 2:
        return "far_view", "Kết quả phân tích hợp lệ.", f"Ảnh có {anchor_count} Neo bối cảnh ({', '.join(sorted(detected_anchors))})"
    if anchor_count == 1:
        return "mid_view", "Kết quả phân tích hợp lệ.", f"Ảnh có 1 Neo bối cảnh ({list(detected_anchors)[0]})"

    max_dmg_area = 0.0
    total_dmg_area = 0.0
    for d in damages:
        b = d.bbox_xyxy
        if len(b) >= 4:
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            max_dmg_area = max(max_dmg_area, area)
            total_dmg_area += area

    if (max_dmg_area / total_img_area) > damage_area_threshold or (total_dmg_area / total_img_area) > damage_area_threshold or (max_dmg_area / total_img_area) >= 0.25:
        reason = "Vết tổn thất chiếm diện tích lớn mà không có điểm neo"
    else:
        reason = "Không có Neo bối cảnh"

    return "close_up", "Ảnh quá cận cảnh, không thể định vị", reason


def determine_shot_type(
    damages: Optional[Sequence[DamageResult]] = None,
    parts: Optional[Sequence[PartResult]] = None,
    image_hw: Tuple[int, int] = (640, 640),
    damage_area_threshold: float = 0.30,
    part_conf_threshold: float = 0.35,
    context_anchors: Optional[Sequence[Any]] = None,
) -> str:
    """Shot Type Router: Phân loại cự ly chụp ảnh."""
    if isinstance(damages, np.ndarray):
        image_hw = damages.shape[:2]
        damages = []
    shot_type, _, _ = classify_shot_type(
        damages or [],
        parts or [],
        image_hw,
        damage_area_threshold=damage_area_threshold,
        part_conf_threshold=part_conf_threshold,
        context_anchors=context_anchors,
    )
    return shot_type


def is_close_up_shot(
    damages: Sequence[DamageResult],
    parts: Sequence[PartResult],
    image_hw: Tuple[int, int],
    damage_area_threshold: float = 0.30,
    part_conf_threshold: float = 0.35,
    context_anchors: Optional[Sequence[Any]] = None,
) -> Tuple[bool, str]:
    """Kiểm tra có phải ảnh cận cảnh thiếu neo hay không."""
    shot_type, _, reason = classify_shot_type(
        damages, parts, image_hw, damage_area_threshold, part_conf_threshold, context_anchors
    )
    return (shot_type == "close_up", reason)


def map_damages_to_parts(
    damages: List[DamageResult],
    parts: List[PartResult],
    image_hw: Tuple[int, int],
    image_bgr: Optional[np.ndarray] = None,
    micro_features: Optional[List[str]] = None,
    damage_conf_min: float = 0.25,
    overlap_threshold: float = 0.15,
    max_center_distance_ratio: float = 0.20,
    min_part_confidence: float = 0.20,
    closeup_area_threshold: float = 0.30,
    closeup_conf_threshold: float = 0.35,
    filename: str = "",
) -> List[MappedDetection]:
    """Generalized Mathematical Matching: Gán từng vết tổn thất vào bộ phận xe theo 2 bước."""
    mapped: List[MappedDetection] = []
    h, w = image_hw
    if h <= 0 or w <= 0:
        return mapped

    img_diagonal = float(math.hypot(w, h))
    max_allowed_distance = float(max_center_distance_ratio * img_diagonal)
    distance_tolerance = float(0.03 * img_diagonal)

    damages_filtered = [
        d for d in damages
        if d.confidence >= (max(damage_conf_min, 0.35) if d.damage_type.lower() == "dent" else (max(damage_conf_min, 0.85) if d.damage_type.lower() == "glass_shatter" else damage_conf_min))
    ]
    valid_parts = [p for p in parts if p.confidence >= min_part_confidence]

    shot_type, _, _ = classify_shot_type(
        damages_filtered if damages_filtered else damages,
        parts,
        image_hw,
        damage_area_threshold=closeup_area_threshold,
        part_conf_threshold=closeup_conf_threshold,
    )
    is_closeup_shot = (shot_type == "close_up")

    BODY_PART_KEYS = {
        "cua_hong", "cua_truoc", "cua_sau", "cua_truoc_trai", "cua_sau_trai",
        "cua_truoc_phai", "cua_sau_phai", "can_truoc", "can_sau", "ve_che_banh", "nap_capo"
    }

    part_masks: List[np.ndarray] = []
    for p in valid_parts:
        poly_p = p.part_polygon if len(p.part_polygon) >= 3 else _bbox_to_polygon(p.bbox_xyxy)
        m = np.zeros((h, w), dtype=np.uint8)
        if len(poly_p) >= 3:
            pts = np.clip(np.asarray(poly_p, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], 1)
        part_masks.append(m)

    for i, dmg in enumerate(damages_filtered):
        dmg_id = i + 1
        dmg_type_norm = dmg.damage_type.lower()
        forbidden_parts = set(FORBIDDEN_DAMAGE_PART_PAIRS.get(dmg_type_norm, set()))
        is_quarter_panel_zone = False

        poly_d = dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy)
        dmg_mask = np.zeros((h, w), dtype=np.uint8)
        if len(poly_d) >= 3:
            pts_d = np.clip(np.asarray(poly_d, dtype=np.int32), [0, 0], [w - 1, h - 1]).reshape(-1, 1, 2)
            cv2.fillPoly(dmg_mask, [pts_d], 1)
        else:
            bx1, by1, bx2, by2 = [int(v) for v in dmg.bbox_xyxy[:4]]
            dmg_mask[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)] = 1

        dmg_area = float(dmg_mask.sum())
        if dmg_area <= 0:
            bw = max(1.0, dmg.bbox_xyxy[2] - dmg.bbox_xyxy[0])
            bh = max(1.0, dmg.bbox_xyxy[3] - dmg.bbox_xyxy[1])
            dmg_area = float(bw * bh)

        # Cảnh báo Bounding Box diện tích bất thường (> 30% diện tích ảnh - giới hạn mô hình Stage 1)
        dmg_w = max(1.0, dmg.bbox_xyxy[2] - dmg.bbox_xyxy[0])
        dmg_h = max(1.0, dmg.bbox_xyxy[3] - dmg.bbox_xyxy[1])
        dmg_box_area = dmg_w * dmg_h
        is_abnormal_bbox = (dmg_box_area > (0.30 * w * h))
        warning_bbox_msg = "Cảnh báo: Bounding Box có diện tích bất thường (>30% diện tích ảnh), giới hạn của mô hình Object Detection Stage 1" if is_abnormal_bbox else ""

        # Kiểm tra bằng chứng trực tiếp với cửa xe (Direct mask overlap)
        has_direct_door_overlap = False
        for p, lm in zip(valid_parts, part_masks):
            if p.part_key == "cua_hong":
                door_ioa = float(np.count_nonzero(dmg_mask & lm)) / max(1.0, dmg_area)
                if door_ioa >= overlap_threshold:
                    has_direct_door_overlap = True
                    break

        rear_bumper_parts = [p for p in valid_parts if p.part_key == "can_sau"]
        fender_parts = [p for p in valid_parts if p.part_key == "ve_che_banh"]
        if rear_bumper_parts and dmg_type_norm in ("dent", "scratch", "crack", "paint_chipping"):
            rb_box = rear_bumper_parts[0].bbox_xyxy
            rb_cx = (rb_box[0] + rb_box[2]) / 2.0
            is_rear_on_right = (rb_cx >= w / 2.0)
            dcx = dmg.center_xy[0]

            if fender_parts:
                f_box = fender_parts[0].bbox_xyxy
                f_cx = (f_box[0] + f_box[2]) / 2.0
                f_w = max(10.0, f_box[2] - f_box[0])
                if is_rear_on_right and dcx >= (f_cx - 0.25 * f_w) and dcx <= rb_box[2]:
                    is_quarter_panel_zone = True
                elif not is_rear_on_right and dcx <= (f_cx + 0.25 * f_w) and dcx >= rb_box[0]:
                    is_quarter_panel_zone = True
                # Khi có vè che bánh, ưu tiên vè che bánh trên vùng hông sau thay vì để mask cửa lan rộng lấn át
                if is_quarter_panel_zone:
                    forbidden_parts.add("cua_hong")
            else:
                # Không có vè che bánh: chỉ coi là quarter panel zone nếu không có đè phủ trực tiếp với cửa
                if is_rear_on_right and (dcx >= (rb_box[0] - 0.12 * w) or dmg.bbox_xyxy[2] >= rb_box[0]):
                    is_quarter_panel_zone = True
                elif not is_rear_on_right and (dcx <= (rb_box[2] + 0.12 * w) or dmg.bbox_xyxy[0] <= rb_box[2]):
                    is_quarter_panel_zone = True
                if is_quarter_panel_zone and not has_direct_door_overlap:
                    forbidden_parts.add("cua_hong")

        # Quy tắc vùng đứng (Vertical Zone Rule): Tổn thất ở nửa dưới đầu xe hoặc dưới đèn pha vỡ
        # bắt buộc thuộc về Cản trước / Vè che bánh, KHÔNG được kéo lên Nắp capo hay gán nhầm Cửa
        broken_lamps_all = [
            d for d in damages_filtered
            if d.damage_type.lower() in ("lamp_broken", "lamp broken") and d.confidence >= 0.25
        ]
        hood_parts_all = [p for p in valid_parts if p.part_key == "nap_capo" and p.confidence >= 0.30]
        dcy = dmg.center_xy[1]
        if dmg_type_norm in ("dent", "scratch", "crack", "tear_puncture", "paint_chipping"):
            if broken_lamps_all:
                min_lamp_y2 = min(bl.bbox_xyxy[3] for bl in broken_lamps_all)
                if dcy >= (min_lamp_y2 - 0.06 * h) or dmg.bbox_xyxy[1] >= (min_lamp_y2 - 0.08 * h):
                    forbidden_parts.update({"nap_capo", "cua_hong", "can_sau", "kinh_lai"})
            elif hood_parts_all:
                max_hood_y2 = max(hp.bbox_xyxy[3] for hp in hood_parts_all)
                if dcy >= max_hood_y2 and not has_direct_door_overlap:
                    forbidden_parts.add("nap_capo")

        orient = determine_car_orientation(valid_parts, damages_filtered, image_hw, filename=filename)
        if orient.is_reliable:
            bp_x = orient.b_pillar_x or (w / 2.0)
            if orient.head_direction == "left":
                if dmg.center_xy[0] > (bp_x + 0.05 * w):
                    forbidden_parts.add("can_truoc")
                    forbidden_parts.add("nap_capo")
                elif dmg.center_xy[0] < (bp_x - 0.05 * w):
                    forbidden_parts.add("can_sau")
            elif orient.head_direction == "right":
                if dmg.center_xy[0] < (bp_x - 0.05 * w):
                    forbidden_parts.add("can_truoc")
                    forbidden_parts.add("nap_capo")
                elif dmg.center_xy[0] > (bp_x + 0.05 * w):
                    forbidden_parts.add("can_sau")

        if micro_features:
            micro_feat = detect_micro_features(
                image_bgr=image_bgr,
                damage_bbox=dmg.bbox_xyxy,
                manual_features=micro_features,
            )
            if micro_feat is not None:
                features_norm = [str(f).lower().strip() for f in micro_features]
                if any(f in features_norm for f in ("parking_sensor", "sensor", "cam_bien", "nep_can", "bumper_trim", "luoi_tan_nhiet", "grille")):
                    located_on = StandardPartStr("Cản trước")
                    match_method = "micro_feature_bumper"
                elif any(f in features_norm for f in ("tay_nam_cua", "door_handle", "khe_ho_cua", "khe_ho_canh_cua", "door_gap", "vien_kinh", "window_trim")):
                    located_on = StandardPartStr("Cửa trước")
                    match_method = "micro_feature_door"
                elif any(f in features_norm for f in ("mep_den", "lamp_edge", "headlight_edge", "mep_cum_den")):
                    located_on = StandardPartStr("Cản trước")
                    match_method = "micro_feature_lamp"
                else:
                    located_on = map_to_standard_part(micro_feat)
                    match_method = "micro_feature_fallback"
                disp_label = DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]")
                mapped.append(MappedDetection(
                    damage_type=dmg.damage_type,
                    confidence=dmg.confidence,
                    located_on_part=located_on,
                    overlap_ratio=1.0,
                    mask_iou=1.0,
                    center_distance=0.0,
                    match_method=match_method,
                    damage_polygon=dmg.damage_polygon,
                    part_polygon=dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy),
                    bbox_xyxy=dmg.bbox_xyxy,
                    display_label=disp_label,
                ))
                continue

        if dmg_type_norm in ("lamp_broken", "lamp broken"):
            lamp_parts = [p for p in valid_parts if p.part_key == "den_xe"]
            best_lamp: Optional[PartResult] = None
            best_lamp_score = -1.0
            best_lamp_ioa = 0.0
            for lp, lm in zip(valid_parts, part_masks):
                if lp.part_key != "den_xe":
                    continue
                inter_area = float(np.count_nonzero(dmg_mask & lm))
                ioa = inter_area / dmg_area
                score = ioa * lp.confidence
                if score > best_lamp_score:
                    best_lamp_score = score
                    best_lamp_ioa = ioa
                    best_lamp = lp

            located_on = StandardPartStr("Đèn xe")
            if best_lamp is not None and best_lamp_ioa >= overlap_threshold:
                match_method = "mask_overlap"
                part_poly_out = best_lamp.part_polygon
                dist_val = calculate_point_to_part_boundary_distance(dmg.center_xy, best_lamp)
            elif lamp_parts:
                best_lamp_dist = min(lamp_parts, key=lambda lp: calculate_point_to_part_boundary_distance(dmg.center_xy, lp))
                dist_val = calculate_point_to_part_boundary_distance(dmg.center_xy, best_lamp_dist)
                match_method = "spatial_distance"
                part_poly_out = best_lamp_dist.part_polygon
            else:
                match_method = "semantic_hard_rule_lamp"
                part_poly_out = dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy)
                dist_val = 0.0
                best_lamp_ioa = 1.0

            mapped.append(MappedDetection(
                damage_type=dmg.damage_type,
                confidence=dmg.confidence,
                located_on_part=located_on,
                overlap_ratio=best_lamp_ioa,
                mask_iou=best_lamp_ioa,
                center_distance=round(dist_val, 1),
                match_method=match_method,
                damage_polygon=dmg.damage_polygon,
                part_polygon=part_poly_out,
                bbox_xyxy=dmg.bbox_xyxy,
                display_label=DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]"),
            ))
            continue

        if dmg_type_norm in ("tire_flat", "tire flat"):
            located_on = "Bánh xe / Lốp xe"
            mapped.append(MappedDetection(
                damage_type=dmg.damage_type,
                confidence=dmg.confidence,
                located_on_part=located_on,
                overlap_ratio=1.0,
                mask_iou=1.0,
                center_distance=0.0,
                match_method="semantic_hard_rule_tire",
                damage_polygon=dmg.damage_polygon,
                part_polygon=dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy),
                bbox_xyxy=dmg.bbox_xyxy,
                display_label=DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]"),
            ))
            continue

        if dmg_type_norm in ("glass_shatter", "glass shatter"):
            glass_parts = [p for p in valid_parts if "kinh" in p.part_key]
            best_glass: Optional[PartResult] = None
            best_glass_score = -1.0
            best_glass_ioa = 0.0
            for gp, gm in zip(valid_parts, part_masks):
                if "kinh" not in gp.part_key:
                    continue
                inter_area = float(np.count_nonzero(dmg_mask & gm))
                ioa = inter_area / dmg_area
                score = ioa * gp.confidence
                if score > best_glass_score:
                    best_glass_score = score
                    best_glass_ioa = ioa
                    best_glass = gp

            located_on = StandardPartStr("Kính xe")
            if best_glass is not None and best_glass_ioa >= overlap_threshold:
                match_method = "mask_overlap"
                part_poly_out = best_glass.part_polygon
                dist_val = calculate_point_to_part_boundary_distance(dmg.center_xy, best_glass)
            elif glass_parts:
                best_glass_dist = min(glass_parts, key=lambda gp: calculate_point_to_part_boundary_distance(dmg.center_xy, gp))
                dist_val = calculate_point_to_part_boundary_distance(dmg.center_xy, best_glass_dist)
                match_method = "spatial_distance"
                part_poly_out = best_glass_dist.part_polygon
            else:
                match_method = "semantic_hard_rule_glass"
                part_poly_out = dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy)
                dist_val = 0.0
                best_glass_ioa = 1.0

            mapped.append(MappedDetection(
                damage_type=dmg.damage_type,
                confidence=dmg.confidence,
                located_on_part=located_on,
                overlap_ratio=best_glass_ioa,
                mask_iou=best_glass_ioa,
                center_distance=round(dist_val, 1),
                match_method=match_method,
                damage_polygon=dmg.damage_polygon,
                part_polygon=part_poly_out,
                bbox_xyxy=dmg.bbox_xyxy,
                display_label=DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]"),
            ))
            continue

        # STEP 1 – PRIMARY MATCH: MASK OVERLAP (IoA >= 0.15)
        step1_candidates: List[Tuple[float, float, PartResult, int]] = []
        log_candidate_ioas: List[str] = []

        for p_idx, (p, p_mask) in enumerate(zip(valid_parts, part_masks)):
            inter_area = float(np.count_nonzero(dmg_mask & p_mask))
            ioa = inter_area / dmg_area
            score = ioa * p.confidence
            # Ưu tiên Vè che bánh nếu vết tổn thất nằm ngay tại hốc vè trước/sau khi mask cửa bị tràn sang vè
            if p.part_key == "ve_che_banh" and ioa >= 0.25:
                score *= 1.45
            p_display = p.part_name if p.part_name else PART_DISPLAY_NAMES.get(p.part_key, p.part_key)

            if ioa >= 0.01:
                log_candidate_ioas.append(f"{p_display}: IoA={ioa*100:.1f}%, conf={p.confidence:.2f}, score={score:.3f}")

            if ioa >= overlap_threshold and p.part_key not in forbidden_parts:
                step1_candidates.append((score, ioa, p, p_idx))

        if step1_candidates:
            step1_candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_ioa, best_part, best_pidx = step1_candidates[0]
            if best_part.part_key == "cua_hong":
                door_res = resolve_door_position_from_topology(dmg.center_xy, valid_parts, image_hw, filename=filename)
                located_on = map_to_standard_part(door_res)
            else:
                p_raw = best_part.part_name if best_part.part_name else PART_DISPLAY_NAMES.get(best_part.part_key, best_part.part_key)
                located_on = map_to_standard_part(p_raw)
            match_method = "mask_overlap"
            dist_val = calculate_point_to_part_boundary_distance(dmg.center_xy, best_part)

            part_poly_out = best_part.part_polygon if len(best_part.part_polygon) >= 3 else _bbox_to_polygon(best_part.bbox_xyxy)
            disp_label = DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]")
            step1_warning = warning_bbox_msg
            step1_action = "Cần kiểm tra: Bounding Box có kích thước bất thường do giới hạn Object Detection Stage 1" if is_abnormal_bbox else "Kết quả phân tích hợp lệ."
            mapped.append(MappedDetection(
                damage_type=dmg.damage_type,
                confidence=dmg.confidence,
                located_on_part=located_on,
                overlap_ratio=round(best_ioa, 4),
                mask_iou=round(best_ioa, 4),
                center_distance=round(dist_val, 1),
                match_method=match_method,
                damage_polygon=dmg.damage_polygon,
                part_polygon=part_poly_out,
                bbox_xyxy=dmg.bbox_xyxy,
                display_label=disp_label,
                warning=step1_warning,
                action_required=step1_action,
            ))
            continue

        # STEP 2 – SECONDARY MATCH: SPATIAL DISTANCE
        if is_quarter_panel_zone:
            forbidden_parts.update({"ve_che_banh", "can_sau", "can_truoc", "cua_hong"})

        step2_candidates: List[Tuple[float, PartResult]] = []
        log_candidate_dists: List[str] = []

        for p in valid_parts:
            if p.part_key in forbidden_parts:
                continue
            # Vertical constraint for bumpers: cản trước/cản sau không thể bắt tổn thất ở quá cao/thấp ngoài phạm vi cản
            if p.part_key in ("can_truoc", "can_sau"):
                if dmg.bbox_xyxy[3] < (p.bbox_xyxy[1] - 0.08 * h) or dmg.bbox_xyxy[1] > (p.bbox_xyxy[3] + 0.08 * h):
                    continue
            dist = calculate_point_to_part_boundary_distance(dmg.center_xy, p)
            p_display = p.part_name if p.part_name else PART_DISPLAY_NAMES.get(p.part_key, p.part_key)
            log_candidate_dists.append(f"{p_display}: dist={dist:.1f}px")

            if dist < max_allowed_distance:
                step2_candidates.append((dist, p))

        if step2_candidates:
            step2_candidates.sort(key=lambda item: item[0])
            min_dist = step2_candidates[0][0]

            equivalent_candidates = [
                item for item in step2_candidates
                if item[0] <= (min_dist + distance_tolerance)
            ]

            if dmg_type_norm in ("dent", "scratch"):
                body_equivs = [item for item in equivalent_candidates if item[1].part_key in BODY_PART_KEYS]
                if body_equivs:
                    body_equivs.sort(key=lambda item: (item[1].confidence, -item[0]), reverse=True)
                    chosen_dist, chosen_part = body_equivs[0]
                else:
                    chosen_dist, chosen_part = step2_candidates[0]
            else:
                equivalent_candidates.sort(key=lambda item: (item[0], -item[1].confidence))
                chosen_dist, chosen_part = equivalent_candidates[0]

            if chosen_part.part_key == "cua_hong":
                door_res = resolve_door_position_from_topology(dmg.center_xy, valid_parts, image_hw, filename=filename)
                located_on = map_to_standard_part(door_res)
            else:
                p_raw = chosen_part.part_name if chosen_part.part_name else PART_DISPLAY_NAMES.get(chosen_part.part_key, chosen_part.part_key)
                located_on = map_to_standard_part(p_raw)
            match_method = "spatial_distance"
            part_poly_out = chosen_part.part_polygon if len(chosen_part.part_polygon) >= 3 else _bbox_to_polygon(chosen_part.bbox_xyxy)

            disp_label = DisplayLabelStr(f"{dmg.damage_type} -> {located_on} [{dmg.confidence:.2f}]")
            step2_warning = warning_bbox_msg
            step2_action = "Cần kiểm tra: Bounding Box có kích thước bất thường do giới hạn Object Detection Stage 1" if is_abnormal_bbox else "Kết quả phân tích hợp lệ."
            mapped.append(MappedDetection(
                damage_type=dmg.damage_type,
                confidence=dmg.confidence,
                located_on_part=located_on,
                overlap_ratio=1.0,
                mask_iou=1.0,
                center_distance=round(chosen_dist, 1),
                match_method=match_method,
                damage_polygon=dmg.damage_polygon,
                part_polygon=part_poly_out,
                bbox_xyxy=dmg.bbox_xyxy,
                display_label=disp_label,
                warning=step2_warning,
                action_required=step2_action,
            ))
            continue

        # FALLBACK: KHÔNG THỎA MÃN CẢ 2 STEP -> GÁN 'khong_xac_dinh'
        located_on = "khong_xac_dinh"
        if is_quarter_panel_zone:
            match_method = "quarter_panel_inspection_required"
            warning_msg = "Cần kiểm tra: Vết tổn thất nằm trên vùng hông sau (rear quarter panel) ngoài 8 lớp chuẩn"
            if warning_bbox_msg:
                warning_msg += f"; {warning_bbox_msg}"
            action_req_msg = "Kiểm tra thủ công vùng thân hông sau"
        else:
            match_method = "closeup_unclear_rejection" if is_closeup_shot else "unmapped_topology"
            warning_msg = "Ảnh quá cận cảnh, không thể định vị" if is_closeup_shot else warning_bbox_msg
            action_req_msg = "Ảnh quá cận cảnh, không thể định vị" if is_closeup_shot else (
                "Cần kiểm tra: Bounding Box có kích thước bất thường do giới hạn Object Detection Stage 1" if is_abnormal_bbox else "Kết quả phân tích hợp lệ."
            )
        disp_label = DisplayLabelStr(f"{dmg.damage_type} -> khong_xac_dinh [{dmg.confidence:.2f}]")
        mapped.append(MappedDetection(
            damage_type=dmg.damage_type,
            confidence=dmg.confidence,
            located_on_part=located_on,
            overlap_ratio=0.0,
            mask_iou=0.0,
            center_distance=0.0,
            match_method=match_method,
            damage_polygon=dmg.damage_polygon,
            part_polygon=dmg.damage_polygon if len(dmg.damage_polygon) >= 3 else _bbox_to_polygon(dmg.bbox_xyxy),
            bbox_xyxy=dmg.bbox_xyxy,
            display_label=disp_label,
            warning=warning_msg,
            action_required=action_req_msg,
        ))

    return mapped


# ---------------------------------------------------------------------------
# 6. TWO-STAGE PIPELINE
# ---------------------------------------------------------------------------

class TwoStagePipeline:
    """Pipeline 2 mô hình chạy song song: Nhận diện Hư hại + Định vị Bộ phận xe."""

    def __init__(
        self,
        damage_model_path: Union[str, Path] = "ai/models/weights/damage_best.pt",
        parts_model_path: Union[str, Path] = "ai/models/weights/parts_seg.pt",
        damage_conf: float = 0.25,
        parts_conf: float = 0.30,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        overlap_threshold: float = 0.15,
        max_center_distance_ratio: float = 0.20,
        min_part_confidence: float = 0.25,
        min_part_area_pixels: float = 600.0,
        closeup_area_threshold: float = 0.35,
        closeup_conf_threshold: float = 0.30,
        part_class_names: Optional[Dict[int, str]] = None,
        max_workers: int = 1,
    ):
        from ultralytics import YOLO

        self.device = self._resolve_device(device)

        damage_path = Path(damage_model_path).resolve()
        parts_path = Path(parts_model_path).resolve()

        if not damage_path.is_file():
            alt_dmg = damage_path.parent / "weights" / damage_path.name
            if alt_dmg.is_file():
                damage_path = alt_dmg.resolve()
        if not parts_path.is_file():
            alt_parts = parts_path.parent / "weights" / parts_path.name
            if alt_parts.is_file():
                parts_path = alt_parts.resolve()

        if not damage_path.is_file():
            raise FileNotFoundError(f"Tệp trọng số Damage model không tồn tại: {damage_path}")
        if not parts_path.is_file():
            raise FileNotFoundError(f"Tệp trọng số Parts model không tồn tại: {parts_path}")

        self.damage_path_str = str(damage_path)
        self.parts_path_str = str(parts_path)

        logger.info("Đang nạp Model 1 (Damage): %s", damage_path)
        self.model_damage = YOLO(str(damage_path))
        try:
            self.model_damage.model.names = {
                0: "dent",
                1: "scratch",
                2: "crack",
                3: "glass_shatter",
                4: "lamp_broken",
            }
        except Exception:
            pass

        logger.info("Đang nạp Model 2 (Parts): %s", parts_path)
        self.model_parts = YOLO(str(parts_path))
        try:
            if len(getattr(self.model_parts, "names", {})) == 8:
                self.model_parts.model.names = {
                    0: "can_truoc",
                    1: "can_sau",
                    2: "nap_capo",
                    3: "ve_che_banh",
                    4: "cua_truoc",
                    5: "cua_sau",
                    6: "kinh_xe",
                    7: "den_xe",
                }
        except Exception:
            pass

        self.damage_conf = float(damage_conf)
        self.parts_conf = float(parts_conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.overlap_threshold = float(overlap_threshold)
        self.max_center_distance_ratio = float(max_center_distance_ratio)
        self.min_part_confidence = float(min_part_confidence)
        self.min_part_area_pixels = float(min_part_area_pixels)
        self.closeup_area_threshold = float(closeup_area_threshold)
        self.closeup_conf_threshold = float(closeup_conf_threshold)
        self.part_class_names = part_class_names or PART_CLASS_NAMES
        self.max_workers = int(max_workers)

    def run(
        self,
        image_input: Union[bytes, np.ndarray, str, Path],
        filename: str = "image.jpg",
        micro_features: Optional[List[str]] = None,
    ) -> PipelineResponse:
        """Thực thi pipeline trên ảnh đầu vào và trả về PipelineResponse."""
        image_bgr, width, height = self._load_image_input(image_input)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fut_damage = pool.submit(
                _run_damage_model,
                self.model_damage, image_bgr,
                self.damage_conf, self.iou, self.imgsz, self.device,
            )
            fut_parts = pool.submit(
                _run_parts_model,
                self.model_parts, image_bgr,
                min(self.parts_conf, self.min_part_confidence), self.iou, self.imgsz, self.device,
                self.part_class_names,
            )
            damages: List[DamageResult] = fut_damage.result()
            parts_raw: List[PartResult] = fut_parts.result()

        parts_contextual = filter_parts_with_contextual_rules(
            parts_raw,
            damages,
            image_hw=(height, width),
            image_bgr=image_bgr,
        )

        parts_merged = merge_and_smooth_part_masks(
            parts_contextual,
            image_hw=(height, width),
            min_area_pixels=self.min_part_area_pixels,
            morph_closing_kernel_size=15,
        )

        parts = resolve_overlapping_part_masks(
            parts_merged,
            image_hw=(height, width),
            iou_threshold=0.30,
        )

        if isinstance(image_input, (str, Path)):
            filename = Path(image_input).name

        orient_run = determine_car_orientation(parts, damages, (height, width), filename=filename)
        parts = filter_false_positive_parts_by_orientation(parts, (height, width), orient_run)
        parts = filter_parts_with_contextual_rules(parts, damages, image_hw=(height, width), image_bgr=image_bgr)

        shot_type_global = determine_shot_type(
            damages, parts, (height, width),
            damage_area_threshold=self.closeup_area_threshold,
            part_conf_threshold=self.closeup_conf_threshold,
        )
        _, action_req_global, shot_reason_global = classify_shot_type(
            damages, parts, (height, width),
            damage_area_threshold=self.closeup_area_threshold,
            part_conf_threshold=self.closeup_conf_threshold,
        )
        is_closeup_global = (shot_type_global == "close_up")

        if is_closeup_global and not micro_features:
            action_req_global = "Ảnh quá cận cảnh, không thể định vị"
            mapped = [
                MappedDetection(
                    damage_type=normalize_damage_class(d.damage_type) or d.damage_type,
                    confidence=d.confidence,
                    located_on_part="khong_xac_dinh",
                    overlap_ratio=0.0,
                    mask_iou=0.0,
                    center_distance=0.0,
                    match_method="closeup_unclear_rejection",
                    damage_polygon=d.damage_polygon,
                    part_polygon=[],
                    bbox_xyxy=d.bbox_xyxy,
                    display_label=f"{normalize_damage_class(d.damage_type) or d.damage_type} -> khong_xac_dinh [{d.confidence:.2f}]",
                    warning="Ảnh quá cận cảnh, không thể định vị",
                    action_required="Ảnh quá cận cảnh, không thể định vị",
                )
                for d in damages
                if d.confidence >= (max(self.damage_conf, 0.35) if d.damage_type.lower() == "dent" else (max(self.damage_conf, 0.85) if d.damage_type.lower() == "glass_shatter" else self.damage_conf))
                and normalize_damage_class(d.damage_type) is not None
            ]
        else:
            mapped = map_damages_to_parts(
                damages, parts,
                image_hw=(height, width),
                image_bgr=image_bgr,
                micro_features=micro_features,
                damage_conf_min=self.damage_conf,
                overlap_threshold=self.overlap_threshold,
                max_center_distance_ratio=self.max_center_distance_ratio,
                min_part_confidence=self.min_part_confidence,
                closeup_area_threshold=self.closeup_area_threshold,
                closeup_conf_threshold=self.closeup_conf_threshold,
                filename=filename,
            )

        # Xác định chất lượng phân tích (Analysis Quality Status: completed, needs_review, insufficient_data)
        if is_closeup_global or (len(damages) > 0 and len(parts) == 0):
            analysis_status = "insufficient_data"
            action_req_global = "Ảnh chụp không đủ bối cảnh hoặc thiếu bộ phận xe tham chiếu; yêu cầu chụp lại ảnh toàn cảnh rõ nét hơn."
        else:
            has_unmapped = any(m.located_on_part == "khong_xac_dinh" for m in mapped)
            has_warning = any(bool(m.warning) for m in mapped)
            has_abnormal_bbox = any("diện tích bất thường" in (m.warning or "") for m in mapped)
            has_quarter_panel = any(m.match_method == "quarter_panel_inspection_required" for m in mapped)

            if has_unmapped or has_warning or has_abnormal_bbox or has_quarter_panel:
                analysis_status = "needs_review"
                reasons = []
                if has_unmapped or has_quarter_panel:
                    reasons.append("tồn tại tổn thất chưa định vị chính xác hoặc thuộc vùng hông sau")
                if has_abnormal_bbox:
                    reasons.append("bounding box tổn thất lớn bất thường do giới hạn Stage 1 detect")
                elif has_warning:
                    reasons.append("cảnh báo vị trí/bối cảnh cần kiểm tra")
                action_req_global = f"Cần giám định viên kiểm tra lại kết quả ({'; '.join(reasons)})."
            else:
                analysis_status = "completed"
                action_req_global = "Kết quả phân tích hợp lệ."

        detections_json = [
            {
                "damage_type": normalize_damage_class(m.damage_type) or m.damage_type,
                "raw_damage_type": m.damage_type,
                "confidence": m.confidence,
                "part": map_to_standard_part(m.located_on_part),
                "located_on_part": map_to_standard_part(m.located_on_part),
                "message": (
                    "Ảnh quá cận cảnh, không thể định vị" if is_closeup_global
                    else ("Cần kiểm tra: Không thể xác định bộ phận tương ứng." if m.located_on_part == "khong_xac_dinh" else "Kết quả phân tích hợp lệ.")
                ),
                "display_label": m.display_label if m.display_label else f"{normalize_damage_class(m.damage_type) or m.damage_type} -> {m.located_on_part} [{m.confidence:.2f}]",
                "action_required": m.action_required,
                "overlap_ratio": m.overlap_ratio,
                "mask_iou": m.mask_iou,
                "center_distance": m.center_distance,
                "match_method": m.match_method,
                "damage_polygon": m.damage_polygon,
                "part_polygon": m.part_polygon,
                "bbox_xyxy": m.bbox_xyxy,
                "warning": m.warning,
            }
            for m in mapped
            if normalize_damage_class(m.damage_type) is not None or m.damage_type.lower() in ("tire_flat", "tire flat")
        ]

        wheel_bboxes_run = [
            p.bbox_xyxy for p in parts if p.part_key in ("ve_che_banh", "wheel", "tire") and p.confidence >= 0.20
        ]
        parts_json = resolve_four_side_doors(
            parts,
            image_hw=(height, width),
            wheel_bboxes=wheel_bboxes_run,
            filename=filename,
        )

        return PipelineResponse(
            success=True,
            status=analysis_status,
            image_width=width,
            image_height=height,
            shot_type=shot_type_global,
            action_required=action_req_global,
            detections=detections_json,
            parts_detected=parts_json,
            metadata={
                "damage_model": self.damage_path_str,
                "parts_model": self.parts_path_str,
                "damage_conf": self.damage_conf,
                "parts_conf": self.parts_conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
                "device": self.device,
                "status": analysis_status,
                "analysis_status": analysis_status,
                "overlap_threshold": self.overlap_threshold,
                "max_center_distance_ratio": self.max_center_distance_ratio,
                "total_damages_raw": len(damages),
                "total_parts_raw": len(parts_raw),
                "total_parts_nms": len(parts),
                "total_mapped": len(mapped),
                "closeup_fallback_triggered": any("fallback" in m.match_method or "closeup" in m.match_method for m in mapped),
                "shot_type": shot_type_global,
                "part": "khong_xac_dinh" if is_closeup_global else "determined",
                "message": "Ảnh quá cận cảnh, không thể định vị" if is_closeup_global else ("Cần kiểm tra lại kết quả." if analysis_status == "needs_review" else "Kết quả phân tích hợp lệ."),
                "action_required": action_req_global,
                "is_close_up_shot": is_closeup_global,
                "closeup_reason": shot_reason_global,
                "closeup_warning": "Ảnh quá cận cảnh, không thể định vị" if is_closeup_global else "",
            },
        )

    def _load_image_input(
        self, image_input: Union[bytes, np.ndarray, str, Path]
    ) -> Tuple[np.ndarray, int, int]:
        """Chuẩn hóa ảnh đầu vào thành numpy BGR."""
        if isinstance(image_input, np.ndarray):
            h, w = image_input.shape[:2]
            return image_input, w, h

        if isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(bytes(image_input), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể giải mã dữ liệu bytes thành hình ảnh.")
            h, w = img.shape[:2]
            return img, w, h

        path = Path(image_input).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tệp ảnh không tồn tại: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Không thể đọc file ảnh: {path}")
        h, w = img.shape[:2]
        return img, w, h

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Lựa chọn thiết bị suy luận an toàn."""
        if requested and requested != "auto":
            return requested
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def visualize(
        self,
        image_input: Union[bytes, np.ndarray, str, Path],
        response: PipelineResponse,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Vẽ Bounding Box / Polygon lên ảnh."""
        img_bgr, _, _ = self._load_image_input(image_input)
        annotated = draw_annotations(img_bgr, response.detections, response.parts_detected)
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), annotated)
        return annotated


def _get_unicode_font(font_size_px: int):
    """Lấy font PIL thống nhất giữa bước đo kích thước và bước vẽ chữ."""
    try:
        from PIL import ImageFont

        for fp in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
            if os.path.exists(fp):
                return ImageFont.truetype(fp, font_size_px)
    except Exception:
        pass
    return None


def _render_unicode_label(
    canvas: np.ndarray,
    text: str,
    bg_x1: int,
    bg_y1: int,
    bg_x2: int,
    bg_y2: int,
    bg_color_bgr: Tuple[int, int, int],
    font_size_px: int,
) -> None:
    """Vẽ nhãn chữ tiếng Việt có dấu sắc nét bằng PIL (fallback về cv2.putText nếu thiếu PIL)."""
    h_c, w_c = canvas.shape[:2]
    bg_x1 = max(2, min(w_c - 10, bg_x1))
    bg_y1 = max(2, min(h_c - 10, bg_y1))
    bg_x2 = max(bg_x1 + 8, min(w_c - 2, bg_x2))
    bg_y2 = max(bg_y1 + 8, min(h_c - 2, bg_y2))

    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color_bgr, -1)
    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), max(1, font_size_px // 14))
    font = _get_unicode_font(font_size_px)
    if font is not None:
        try:
            from PIL import Image, ImageDraw

            roi = canvas[bg_y1:bg_y2, bg_x1:bg_x2]
            if roi.size == 0:
                return
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            bbox_t = draw.textbbox((0, 0), text, font=font)
            tw = bbox_t[2] - bbox_t[0]
            th = bbox_t[3] - bbox_t[1]
            tx = max(4, (bg_x2 - bg_x1 - tw) // 2 - bbox_t[0])
            ty = max(1, (bg_y2 - bg_y1 - th) // 2 - bbox_t[1])
            draw.text((tx, ty), text, font=font, fill=(255, 255, 255))
            canvas[bg_y1:bg_y2, bg_x1:bg_x2] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    font_scale = max(0.45, font_size_px / 28.0)
    thickness = max(1, int(round(font_scale * 2.0)))
    pad = max(4, int(round(font_scale * 8.0)))
    cv2.putText(
        canvas, text, (bg_x1 + pad, bg_y2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_annotations(
    image_bgr: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    parts: Optional[Sequence[Dict[str, Any]]] = None,
    draw_part_masks: bool = False,
) -> np.ndarray:
    """Vẽ bounding box vết hư hỏng và nhãn bộ phận lên ảnh, tích hợp chống đè chữ và tỉ lệ chữ tự động theo độ phân giải."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    max_dim = float(max(w, h))

    # Tỉ lệ động theo độ phân giải ảnh (max(w, h) / 1000.0) giúp chữ trên ảnh 3K-4K rõ nét tuyệt đối
    scale_factor = max(0.68, max_dim / 1000.0)
    font_size_px = max(15, int(round(20.0 * scale_factor)))
    font_scale = max(0.50, 0.62 * scale_factor)
    thickness = max(1, int(round(2.0 * scale_factor)))
    box_thick = max(2, int(round(3.0 * scale_factor)))
    pad_x = max(8, int(round(12.0 * scale_factor)))
    pad_y = max(5, int(round(7.0 * scale_factor)))

    pil_font = _get_unicode_font(font_size_px)

    # 1. Chỉ vẽ vùng bộ phận xe (Parts) khi bật cờ draw_part_masks=True (dành cho ảnh xuất riêng parts_segmentation)
    if draw_part_masks and parts:
        overlay = canvas.copy()
        part_colors = [
            (255, 180, 50),
            (50, 200, 100),
            (200, 100, 255),
            (50, 220, 220),
            (180, 180, 50),
            (100, 150, 255),
        ]
        for idx, p in enumerate(parts):
            conf = float(p.get("confidence", 0.0))
            if conf < 0.35:
                continue
            color = part_colors[idx % len(part_colors)]
            poly = p.get("part_polygon", [])
            bbox = p.get("bbox_xyxy", [])
            p_raw = p.get("part_name", p.get("part_key", "part"))
            p_name = map_to_standard_part(p_raw)
            if p_name == "khong_xac_dinh":
                p_name = p_raw

            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(canvas, [pts], True, color, max(2, box_thick // 2))
            elif len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(2, box_thick // 2))

            if len(bbox) == 4:
                bx1, by1 = int(bbox[0]), int(bbox[1])
                label_p = f"{p_name} ({conf:.2f})"
                if pil_font is not None:
                    tb_p = pil_font.getbbox(label_p)
                    p_w = int(tb_p[2] - tb_p[0]) + pad_x * 2
                    p_h = int(tb_p[3] - tb_p[1]) + pad_y * 2
                else:
                    p_w = int(len(label_p) * font_size_px * 0.62) + pad_x * 2
                    p_h = font_size_px + pad_y * 2
                px1 = max(5, min(w - p_w - 5, bx1 + 4))
                py1 = max(5, min(h - p_h - 5, by1 + 4))
                _render_unicode_label(canvas, label_p, px1, py1, px1 + p_w, py1 + p_h, color, font_size_px)

        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # 2. Vẽ các vết hư hỏng (Damages):
    # Vẽ TẤT CẢ khung chữ nhật (bounding box) ở Lớp 1 trước, sau đó mới vẽ TẤT CẢ nhãn chữ ở Lớp 2 trên cùng
    # để không bao giờ có chuyện đường viền của box sau cắt ngang đè lên chữ của box trước!
    damage_colors = {
        "scratch": (0, 140, 255),         # Cam đậm
        "dent": (0, 0, 235),              # Đỏ
        "crack": (200, 0, 200),           # Tím hồng
        "lamp_broken": (180, 0, 220),     # Tím đậm
        "glass_shatter": (220, 0, 180),   # Tím
        "tear_puncture": (0, 200, 200),   # Vàng sẫm
        "paint_chipping": (200, 160, 0),  # Xanh lam
    }

    label_items: List[Tuple[int, int, int, int, str, Tuple[int, int, int]]] = []

    # LỚP 1: Vẽ đường viền polygon và bounding box
    for det in detections:
        dmg_type = str(det.get("raw_damage_type") or det.get("damage_type", "damage")).lower()
        color = damage_colors.get(dmg_type, damage_colors.get(str(det.get("damage_type", "")).lower(), (0, 0, 235)))
        bbox = det.get("bbox_xyxy", [])
        poly = det.get("damage_polygon", [])
        raw_label = det.get("display_label") or ""
        located_std = map_to_standard_part(det.get("located_on_part", "khong_xac_dinh"))
        if not raw_label or "cua_hong" in raw_label.lower() or "cửa hông" in raw_label.lower():
            label = f"{dmg_type} -> {located_std} [{det.get('confidence', 0.0):.2f}]"
        else:
            label = raw_label

        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, color, max(2, box_thick - 1))

        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thick)
            label_items.append((x1, y1, x2, y2, label, color))

    # LỚP 2: Tính toán vị trí chống đè chữ và vẽ toàn bộ hộp nhãn lên trên cùng
    drawn_text_boxes: List[Tuple[int, int, int, int]] = []
    margin = max(4, int(round(4.0 * scale_factor)))

    for x1, y1, x2, y2, label, color in label_items:
        if pil_font is not None:
            tb = pil_font.getbbox(label)
            tw = int(tb[2] - tb[0])
            th = int(tb[3] - tb[1])
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        box_w = tw + pad_x * 2 + max(10, int(round(10.0 * scale_factor)))
        box_h = th + pad_y * 2 + max(4, int(round(4.0 * scale_factor)))
        tx1 = max(6, min(max(6, w - box_w - 6), x1))
        bg_x2 = min(w - 4, tx1 + box_w)
        if bg_x2 - tx1 < box_w:
            tx1 = max(2, bg_x2 - box_w)

        placed = False

        # Hướng 1: Thử dịch dần lên phía trên
        curr_y = y1 - box_h - margin
        while curr_y >= 6:
            curr_y2 = curr_y + box_h
            collision = any(
                not (bg_x2 + margin < dt[0] or tx1 - margin > dt[2] or curr_y2 + margin < dt[1] or curr_y - margin > dt[3])
                for dt in drawn_text_boxes
            )
            if not collision:
                bg_y1 = curr_y
                bg_y2 = curr_y2
                placed = True
                break
            curr_y -= (box_h + margin + 2)

        # Hướng 2: Nếu phía trên hết chỗ, dịch dần xuống phía dưới
        if not placed:
            curr_y = y2 + margin
            while curr_y + box_h <= h - 6:
                curr_y2 = curr_y + box_h
                collision = any(
                    not (bg_x2 + margin < dt[0] or tx1 - margin > dt[2] or curr_y2 + margin < dt[1] or curr_y - margin > dt[3])
                    for dt in drawn_text_boxes
                )
                if not collision:
                    bg_y1 = curr_y
                    bg_y2 = curr_y2
                    placed = True
                    break
                curr_y += (box_h + margin + 2)

        # Hướng 3: Đặt bên trong góc trên của bounding box nếu cả trên và dưới đều chạm biên ảnh
        if not placed:
            curr_y = max(6, y1 + margin)
            while curr_y + box_h <= h - 6:
                curr_y2 = curr_y + box_h
                collision = any(
                    not (bg_x2 + margin < dt[0] or tx1 - margin > dt[2] or curr_y2 + margin < dt[1] or curr_y - margin > dt[3])
                    for dt in drawn_text_boxes
                )
                if not collision:
                    bg_y1 = curr_y
                    bg_y2 = curr_y2
                    placed = True
                    break
                curr_y += (box_h + margin + 2)

        if not placed:
            bg_y1 = max(6, min(h - box_h - 6, y1 - box_h - 2))
            bg_y2 = bg_y1 + box_h

        drawn_text_boxes.append((tx1, bg_y1, bg_x2, bg_y2))

        if abs(bg_y1 - y1) > int(30 * scale_factor) and abs(bg_y2 - y1) > int(20 * scale_factor):
            anchor_x = int((x1 + x2) / 2)
            anchor_y = y1 if bg_y1 < y1 else y2
            label_anchor_x = int((tx1 + bg_x2) / 2)
            label_anchor_y = bg_y2 if bg_y1 < y1 else bg_y1
            cv2.line(canvas, (label_anchor_x, label_anchor_y), (anchor_x, anchor_y), color, max(1, box_thick // 2), cv2.LINE_AA)

        _render_unicode_label(canvas, label, tx1, bg_y1, bg_x2, bg_y2, color, font_size_px)

    return canvas


# Alias tương thích ngược
Pipeline = TwoStagePipeline