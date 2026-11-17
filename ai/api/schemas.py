"""Pydantic Schemas cho AutoClaim-AI API Gateway."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DamageType(str, Enum):
    """Phân loại 5 nhóm hư hỏng chuẩn Capstone Proposal."""
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    TEAR_PUNCTURE = "tear_puncture"
    PAINT_CHIPPING = "paint_chipping"
    OTHER = "other"


class CarPartType(str, Enum):
    """8 Vùng thân vỏ tiêu chuẩn."""
    CAN_TRUOC = "can_truoc"
    CAN_SAU = "can_sau"
    NAP_CAPO = "nap_capo"
    VE_CHE_BANH = "ve_che_banh"
    CUA_TRUOC_TRAI = "cua_truoc_trai"
    CUA_SAU_TRAI = "cua_sau_trai"
    CUA_TRUOC_PHAI = "cua_truoc_phai"
    CUA_SAU_PHAI = "cua_sau_phai"
    CUA_TRUOC = "cua_truoc"
    CUA_SAU = "cua_sau"
    KINH_CHIEU_HAU = "kinh_chieu_hau"
    KINH_LAI = "kinh_lai"
    DEN_XE = "den_xe"
    UNKNOWN = "khong_xac_dinh"


class DamageItemSchema(BaseModel):
    """Chi tiết 1 vùng hư hỏng."""
    model_config = ConfigDict(extra="ignore")
    damage_type: str = Field(..., description="Loại hư hỏng")
    confidence: float = Field(..., description="Độ tin cậy của mô hình")
    part: str = Field(..., description="Bộ phận bị ảnh hưởng")
    located_on_part: str = Field(..., description="Tên bộ phận hiển thị")
    bbox_xyxy: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    match_method: str = Field(..., description="Phương pháp mapping (mask_overlap, spatial_fallback...)")
    display_label: Optional[str] = Field(None, description="Nhãn hiển thị trực quan")
    warning: Optional[str] = Field(None, description="Cảnh báo chất lượng hoặc cự ly")


class PointAnalysisResult(BaseModel):
    """Kết quả phân tích cho 1 góc chụp trong bộ 8 góc."""
    model_config = ConfigDict(extra="ignore")
    point_index: int = Field(..., description="Chỉ số góc chụp (1 đến 8)")
    point_label: str = Field(..., description="Tên góc chụp (e.g. 1_front, 2_front_right...)")
    shot_type: str = Field(..., description="Cự ly chụp: mid_view, far_view hoặc close_up")
    action_required: str = Field(..., description="Yêu cầu hành động nếu có lỗi cự ly")
    damages: List[DamageItemSchema] = Field(default_factory=list, description="Danh sách tổn thất")
    parts: List[Dict[str, Any]] = Field(default_factory=list, description="Danh sách bộ phận nhận diện")


class Checkin8PointsResponse(BaseModel):
    """Response chuẩn cho API check-in 8 góc chụp 360."""
    model_config = ConfigDict(extra="ignore")
    success: bool = Field(True, description="Trạng thái thực thi")
    message: str = Field(..., description="Thông báo kết quả")
    total_points_analyzed: int = Field(8, description="Tổng số góc chụp đã phân tích")
    points: List[PointAnalysisResult] = Field(default_factory=list, description="Chi tiết 8 góc chụp")
    vehicle_condition_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Tổng hợp tình trạng xe toàn diện"
    )


class ChangeDetectionItemSchema(BaseModel):
    """Chi tiết 1 tổn thất trong so sánh trước sau (Check-in vs Check-out)."""
    model_config = ConfigDict(extra="ignore")
    id: int
    damage_type: str
    confidence: float
    located_on_part: str
    bbox_after: List[float]
    warped_bbox_before: Optional[List[float]] = None
    matched_before_id: Optional[int] = None
    iou: float
    status: str = Field(..., description="'old_damage' hoặc 'new_damage'")
    message: str


class ChangeDetectionResponse(BaseModel):
    """Response chuẩn cho Change Detection."""
    model_config = ConfigDict(extra="ignore")
    success: bool = True
    aligned: bool = Field(..., description="Căn chỉnh Homography thành công hay fallback")
    num_old_damages: int = Field(..., description="Số tổn thất cũ đã có từ trước")
    num_new_damages: int = Field(..., description="Số tổn thất mới phát sinh cần bồi thường")
    old_damages: List[ChangeDetectionItemSchema] = Field(default_factory=list)
    new_damages: List[ChangeDetectionItemSchema] = Field(default_factory=list)
    warning: Optional[str] = None
