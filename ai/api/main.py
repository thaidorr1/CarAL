"""FastAPI Gateway cho Ứng dụng Di động AutoClaim-AI (Check-in 360 - 8 Góc Chụp).

Endpoint trọng tâm:
- POST /api/v1/checkin/analyze-8-points
  + Nhận đầu vào: 8 file ảnh tương ứng 8 góc chụp 360 độ quanh xe.
  + Gọi Shot Type Router (`determine_shot_type`) để kiểm tra cự ly từng ảnh.
  + Nếu có bất kỳ ảnh nào là `close_up` (quá cận cảnh): trả về lỗi HTTP 400 (Yêu cầu chụp xa hơn).
  + Nếu toàn bộ 8 ảnh hợp lệ (`mid_view` / `far_view`): thực thi Two-Stage Inference Pipeline,
    đồng bộ 5 nhóm hư hỏng chuẩn Proposal & 8 vùng thân vỏ (phân giải 4 cửa hông),
    và gộp kết quả trả về 1 JSON tổng hợp tình trạng xe.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

AI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_ROOT.parent
for _p in (str(AI_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.pipeline import (
    PROPOSAL_5_DAMAGE_CLASSES,
    FOUR_SIDE_DOOR_NAMES,
    TwoStagePipeline,
    determine_shot_type,
    normalize_damage_class,
)

logger = logging.getLogger("autoclaim.api_gateway")

#: 8 góc chụp 360 độ tiêu chuẩn quanh xe khi Check-in trên ứng dụng di động
CHECKIN_8_POINTS_ORDER: Tuple[str, ...] = (
    "1_front",        # Góc 1: Chính diện đầu xe (0°)
    "2_front_right",  # Góc 2: Chéo trước phải (45°)
    "3_right_side",   # Góc 3: Sườn phải xe (90°)
    "4_rear_right",   # Góc 4: Chéo sau phải (135°)
    "5_rear",         # Góc 5: Chính diện đuôi xe (180°)
    "6_rear_left",    # Góc 6: Chéo sau trái (225°)
    "7_left_side",    # Góc 7: Sườn trái xe (270°)
    "8_front_left",   # Góc 8: Chéo trước trái (315°)
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MiB mỗi ảnh
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Giải mã bytes ảnh sang mảng OpenCV BGR."""
    if not raw_bytes:
        raise ValueError("File ảnh rỗng (0 bytes).")
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Không thể giải mã dữ liệu hình ảnh.")
    return img


def create_gateway_app(pipeline: Optional[Any] = None) -> FastAPI:
    """Khởi tạo FastAPI application cho Mobile API Gateway."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline_lock = Lock()
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            default_dmg = AI_ROOT / "models" / "weights" / "damage_best.pt"
            default_parts = AI_ROOT / "models" / "weights" / "parts_seg.pt"
            dmg_path = Path(os.environ.get("CARDD_MODEL", str(default_dmg)))
            parts_path = Path(os.environ.get("CARDD_PARTS_MODEL", str(default_parts)))

            if dmg_path.is_file() and parts_path.is_file():
                try:
                    app.state.pipeline = TwoStagePipeline(
                        damage_model_path=dmg_path,
                        parts_model_path=parts_path,
                        damage_conf=float(os.environ.get("CARDD_CONF", "0.25")),
                        parts_conf=float(os.environ.get("CARDD_PARTS_CONF", "0.20")),
                        iou=float(os.environ.get("CARDD_IOU", "0.60")),
                        imgsz=int(os.environ.get("CARDD_IMGSZ", "640")),
                        device=os.environ.get("CARDD_DEVICE", "auto"),
                    )
                    logger.info(
                        "Loaded TwoStagePipeline for 8-Point Check-in: damage=%s, parts=%s",
                        dmg_path,
                        parts_path,
                    )
                except Exception as exc:
                    logger.warning("Không thể khởi tạo TwoStagePipeline tự động: %s", exc)
                    app.state.pipeline = None
            else:
                app.state.pipeline = None
        yield

    app = FastAPI(
        title="AutoClaim-AI Mobile Check-in Gateway API",
        version="1.0.0",
        description=(
            "API Gateway cho Ứng dụng Di động: Giám định 8 góc chụp 360 độ (Stage 1 Damage 5 Classes "
            "+ Stage 2 Parts 8 Regions & 4 Doors Topology)."
        ),
        lifespan=lifespan,
    )
    app.state.pipeline_lock = Lock()
    app.state.pipeline = pipeline

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["System"])
    def health_check() -> Dict[str, Any]:
        pipe = getattr(app.state, "pipeline", None)
        return {
            "status": "ready",
            "pipeline_loaded": pipe is not None,
            "supported_damage_classes": list(PROPOSAL_5_DAMAGE_CLASSES.keys()),
            "supported_door_positions": list(FOUR_SIDE_DOOR_NAMES),
        }

    @app.post(
        "/api/v1/checkin/analyze-8-points",
        tags=["Check-in 360"],
        summary="Phân tích 8 ảnh góc chụp 360 quanh xe và tổng hợp tình trạng hư hỏng",
    )
    async def analyze_8_points(
        request: Request,
        files: Optional[List[UploadFile]] = File(
            default=None,
            description="Danh sách 8 file ảnh tương ứng 8 góc chụp 360 độ quanh xe",
        ),
    ) -> JSONResponse:
        """Nhận 8 ảnh 360 quanh xe, kiểm tra cự ly bằng `determine_shot_type`,
        từ chối HTTP 400 nếu có ảnh `close_up`, hoặc chạy Inference Pipeline và gộp kết quả.
        """
        upload_list: List[UploadFile] = []
        if files:
            upload_list = list(files)
        else:
            form = await request.form()
            for _, val in form.multi_items():
                if hasattr(val, "read") and hasattr(val, "filename"):
                    upload_list.append(val)

        if len(upload_list) != 8:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_code": "INVALID_IMAGE_COUNT",
                    "message": f"Yêu cầu cung cấp đủ 8 file ảnh tương ứng 8 góc chụp 360 (nhận được {len(upload_list)} ảnh).",
                    "required_points": list(CHECKIN_8_POINTS_ORDER),
                },
            )

        pipe: Optional[TwoStagePipeline] = getattr(app.state, "pipeline", None)
        if pipe is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "success": False,
                    "error_code": "PIPELINE_NOT_READY",
                    "message": "Two-Stage Inference Pipeline chưa sẵn sàng.",
                },
            )

        point_results: List[Dict[str, Any]] = []
        closeup_rejections: List[Dict[str, Any]] = []
        aggregated_damages: List[Dict[str, Any]] = []
        damages_by_part: Dict[str, List[Dict[str, Any]]] = {}
        damages_by_type: Dict[str, int] = {k: 0 for k in PROPOSAL_5_DAMAGE_CLASSES}

        for idx, (point_key, upload) in enumerate(zip(CHECKIN_8_POINTS_ORDER, upload_list), start=1):
            fname = Path(upload.filename or f"{point_key}.jpg").name
            ext = Path(fname).suffix.lower()
            if ext and ext not in ALLOWED_EXTENSIONS:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error_code": "UNSUPPORTED_MEDIA_TYPE",
                        "message": f"Ảnh số {idx} ({fname}) có định dạng '{ext}' không hợp lệ.",
                    },
                )

            raw_bytes = await upload.read(MAX_UPLOAD_BYTES + 1)
            if not raw_bytes or len(raw_bytes) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error_code": "INVALID_IMAGE_PAYLOAD",
                        "message": f"Ảnh số {idx} ({fname}) rỗng hoặc vượt quá giới hạn 20MB.",
                    },
                )

            try:
                img_bgr = _decode_image_bytes(raw_bytes)
            except ValueError as err:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error_code": "IMAGE_DECODE_FAILED",
                        "message": f"Không thể giải mã ảnh số {idx} ({fname}): {err}",
                    },
                )

            # Thực thi Pipeline & Shot Type Router
            infer_filename = f"{point_key}_{fname}"
            with app.state.pipeline_lock:
                resp = pipe.run(img_bgr, filename=infer_filename)
                resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)

            # Kiểm tra cự ly ảnh qua determine_shot_type / resp.shot_type
            shot_type = resp_dict.get("shot_type") or determine_shot_type(img_bgr)
            if shot_type == "close_up":
                closeup_rejections.append({
                    "point_index": idx,
                    "point_name": point_key,
                    "filename": fname,
                    "shot_type": "close_up",
                    "part": "khong_xac_dinh",
                    "message": "Ảnh quá cận cảnh, không thể định vị. Yêu cầu chụp xa hơn.",
                })
                continue

            detections = resp_dict.get("detections", [])
            normalized_dets: List[Dict[str, Any]] = []
            for d in detections:
                raw_dmg = d.get("damage_type", "")
                canon_dmg = normalize_damage_class(raw_dmg)
                if canon_dmg is None:
                    continue

                part_label = d.get("located_on_part") or d.get("part") or "khong_xac_dinh"
                det_entry = {
                    "point_index": idx,
                    "point_name": point_key,
                    "filename": fname,
                    "damage_type": canon_dmg,
                    "damage_label_vi": PROPOSAL_5_DAMAGE_CLASSES.get(canon_dmg, canon_dmg),
                    "part": part_label,
                    "located_on_part": part_label,
                    "confidence": d.get("confidence", 0.0),
                    "bbox_xyxy": d.get("bbox_xyxy", []),
                    "match_method": d.get("match_method", ""),
                }
                normalized_dets.append(det_entry)
                aggregated_damages.append(det_entry)
                damages_by_type[canon_dmg] = damages_by_type.get(canon_dmg, 0) + 1
                damages_by_part.setdefault(part_label, []).append(det_entry)

            point_results.append({
                "point_index": idx,
                "point_name": point_key,
                "filename": fname,
                "shot_type": shot_type,
                "parts_detected": resp_dict.get("parts_detected", []),
                "detections": normalized_dets,
            })

        # Nếu có bất kỳ ảnh nào bị phát hiện là close_up -> Trả về lỗi HTTP 400 (Yêu cầu chụp xa hơn)
        if closeup_rejections:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_code": "CLOSE_UP_SHOT_REJECTED",
                    "part": "khong_xac_dinh",
                    "message": "Ảnh quá cận cảnh, không thể định vị. Vui lòng chụp xa hơn (trung cảnh hoặc toàn cảnh) tại các góc bị từ chối.",
                    "rejected_points": closeup_rejections,
                },
            )

        # Gộp kết quả trả về 1 cục JSON tổng hợp tình trạng xe
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "total_points_analyzed": 8,
                "vehicle_condition_summary": {
                    "has_damage": len(aggregated_damages) > 0,
                    "total_damages": len(aggregated_damages),
                    "Affected_parts_count": len(damages_by_part),
                    "affected_parts": list(damages_by_part.keys()),
                    "damages_by_type": damages_by_type,
                    "damages_by_part": {
                        part_name: [
                            {
                                "damage_type": item["damage_type"],
                                "confidence": item["confidence"],
                                "point_name": item["point_name"],
                                "bbox_xyxy": item["bbox_xyxy"],
                            }
                            for item in items
                        ]
                        for part_name, items in damages_by_part.items()
                    },
                },
                "detections": aggregated_damages,
                "points_detail": point_results,
            },
        )

    return app


app = create_gateway_app()
