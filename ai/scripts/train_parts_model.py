"""Script Huấn Luyện Lại Mô Hình Phân Đoạn Bộ Phận Xe (YOLOv8-Seg Parts Model).
Chuẩn hóa 7 lớp DOCX:
  0: can_truoc
  1: can_sau
  2: nap_capo
  3: ve_che_banh
  4: cua_hong
  5: kinh_chieu_hau
  6: kinh_lai

Yêu cầu cấu hình kỹ thuật:
- Backbone: yolov8s-seg.pt (hoặc yolov8m-seg.pt, yolov8l-seg.pt) Instance Segmentation
- imgsz: 640
- epochs: 100
- batch: 16 (tự động điều chỉnh theo VRAM GPU)
- patience: 20
- lr0: 0.01
- optimizer: 'AdamW'
- Augmentation đặc thù cận cảnh xe: scale=0.6, mosaic=1.0, mixup=0.15, degrees=10.0, translate=0.20, erasing=0.20
- Đánh giá chất lượng sau huấn luyện bằng mAP50-95(mask) và xuất Confusion Matrix chi tiết 7 lớp.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Dict, Any
import numpy as np
import torch
import yaml

# Thêm đường dẫn project
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import AI_ROOT, select_device

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_parts_model")

DOCX_7_CLASSES = {
    0: "can_truoc",
    1: "can_sau",
    2: "nap_capo",
    3: "ve_che_banh",
    4: "cua_hong",
    5: "kinh_chieu_hau",
    6: "kinh_lai",
}

def resolve_dataset_yaml(data_yaml_path: Path, run_dir: Path, scale: float = 0.6, mosaic: float = 1.0, mixup: float = 0.15, degrees: float = 10.0) -> Path:
    """Xác thực và chuẩn bị file cấu hình dataset YAML với 7 lớp chuẩn."""
    if not data_yaml_path.is_file():
        alt1 = AI_ROOT / "configs" / data_yaml_path.name
        alt2 = AI_ROOT / "data/parts" / data_yaml_path.name
        if alt1.is_file():
            data_yaml_path = alt1
        elif alt2.is_file():
            data_yaml_path = alt2
        else:
            raise FileNotFoundError(f"Không tìm thấy cấu hình dataset: {data_yaml_path}")

    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", AI_ROOT / "data/parts"))
    if not root.is_absolute():
        root = AI_ROOT / root

    # Khóa cứng cấu trúc 7 lớp
    cfg["path"] = str(root.resolve()).replace("\\", "/")
    cfg["train"] = str(cfg.get("train", "images/train")).replace("\\", "/")
    cfg["val"] = str(cfg.get("val", "images/val")).replace("\\", "/")
    cfg["test"] = str(cfg.get("test", "images/val")).replace("\\", "/")
    cfg["names"] = DOCX_7_CLASSES
    cfg["nc"] = 7

    # Augmentation
    cfg["mixup"] = mixup
    cfg["mosaic"] = mosaic
    cfg["scale"] = scale
    cfg["degrees"] = degrees

    target_cfg = run_dir / "parts_7_classes_dataset.yaml"
    target_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(target_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return target_cfg.resolve()

def get_or_download_backbone(backbone_name: str = "yolov8s-seg.pt") -> Path:
    """Tìm hoặc tải pretrained weights yolov8s-seg.pt / yolov8m-seg.pt / yolov8l-seg.pt."""
    candidates = [
        AI_ROOT / "models" / backbone_name,
        AI_ROOT / backbone_name,
        Path(backbone_name).resolve(),
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 1_000_000:
            logger.info("Đã tìm thấy pretrained backbone weights: %s", c)
            return c.resolve()

    target = AI_ROOT / "models" / backbone_name
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Chưa có %s tại local. Đang tự động tải về %s ...", backbone_name, target)

    from ultralytics import YOLO
    _ = YOLO(backbone_name)  # Ultralytics tự tải về
    root_candidate = Path(backbone_name)
    if root_candidate.is_file():
        shutil.copy2(root_candidate, target)
        try:
            root_candidate.unlink()
        except Exception:
            pass
    return target.resolve() if target.is_file() else Path(backbone_name).resolve()

def evaluate_model_and_confusion_matrix(model, data_yaml: Path, report_dir: Path) -> Dict[str, Any]:
    """Đánh giá mô hình chi tiết mAP50-95(mask) và xuất Confusion Matrix cho 7 lớp."""
    from ultralytics import YOLO
    logger.info("Bắt đầu đánh giá kiểm định (Validation) trên tập 7 lớp...")

    val_results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        batch=8,
        plots=True,
        save_json=False
    )

    report = {
        "overall_metrics": {},
        "per_class_mask_metrics": {},
        "confusion_matrix": {}
    }

    # Trích xuất mAP Box và Mask
    try:
        report["overall_metrics"] = {
            "mAP50_mask": round(float(val_results.seg.map50), 4),
            "mAP50_95_mask": round(float(val_results.seg.map), 4),
            "mAP50_box": round(float(val_results.box.map50), 4),
            "mAP50_95_box": round(float(val_results.box.map), 4),
            "precision_mask": round(float(val_results.seg.mp), 4),
            "recall_mask": round(float(val_results.seg.mr), 4),
        }
    except Exception as e:
        logger.warning("Không thể đọc chỉ số tổng hợp qua API val_results: %s", e)

    # Chi tiết từng lớp
    try:
        class_maps = val_results.seg.maps
        for idx, cname in DOCX_7_CLASSES.items():
            if idx < len(class_maps):
                report["per_class_mask_metrics"][cname] = {
                    "class_id": idx,
                    "mAP50_95": round(float(class_maps[idx]), 4)
                }
    except Exception as e:
        logger.warning("Không thể đọc per_class_mask_metrics: %s", e)

    # Confusion Matrix
    try:
        cm = val_results.confusion_matrix
        if cm is not None:
            matrix_data = cm.matrix
            report["confusion_matrix"]["matrix"] = matrix_data.tolist()
            report["confusion_matrix"]["names"] = list(DOCX_7_CLASSES.values()) + ["background"]

            # Copy ảnh confusion matrix sang reports
            save_dir = Path(val_results.save_dir)
            cm_img = save_dir / "confusion_matrix.png"
            cm_norm_img = save_dir / "confusion_matrix_normalized.png"
            if cm_img.is_file():
                shutil.copy2(cm_img, report_dir / "confusion_matrix_7_parts.png")
            if cm_norm_img.is_file():
                shutil.copy2(cm_norm_img, report_dir / "confusion_matrix_normalized_7_parts.png")
    except Exception as e:
        logger.warning("Không thể trích xuất confusion matrix: %s", e)

    # Ghi file JSON báo cáo
    report_file = report_dir / "parts_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Đã xuất báo cáo đánh giá chất lượng tại: %s", report_file)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backbone",
        default="yolov8s-seg.pt",
        choices=["yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8n-seg.pt"],
        help="Backbone segmentation weights (mặc định yolov8s-seg.pt)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Đường dẫn trọng số checkpoint tùy chọn (nếu chỉ định sẽ bỏ qua backbone tải sẵn)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=AI_ROOT / "configs/parts.yaml",
        help="File cấu hình YAML cho parts",
    )
    p.add_argument("--epochs", type=int, default=100, help="Số lượt huấn luyện (epochs)")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train/val")
    p.add_argument("--batch", type=int, default=16, help="Kích thước batch size")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--lr0", type=float, default=0.01, help="Tốc độ học khởi đầu lr0")
    p.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD", "Adam"], help="Thuật toán tối ưu")
    p.add_argument("--device", default="auto", help="Thiết bị tính toán: auto, 0, cpu")
    p.add_argument("--workers", type=int, default=2, help="Số tiến trình nạp dữ liệu dataloader")
    p.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Biên độ scale jitter/zoom cận cảnh đa quy mô (0.5 - 1.5)",
    )
    p.add_argument("--mosaic", type=float, default=1.0, help="Tỷ lệ Mosaic augmentation")
    p.add_argument("--mixup", type=float, default=0.15, help="Tỷ lệ Mixup augmentation")
    p.add_argument("--degrees", type=float, default=10.0, help="Góc xoay camera ngẫu nhiên")
    p.add_argument("--translate", type=float, default=0.20, help="Dịch chuyển khung hình")
    p.add_argument(
        "--name",
        default="parts_closeup_yolov8s_seg_100epochs",
        help="Tên thư mục lưu kết quả huấn luyện trong ai/runs/parts/",
    )
    p.add_argument(
        "--export-target",
        type=Path,
        default=AI_ROOT / "models/weights/parts_seg.pt",
        help="Đích đến để tự động lưu checkpoint best.pt cho TwoStagePipeline",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Chỉ thực hiện validation & xuất confusion matrix mà không train",
    )
    a = p.parse_args()

    from ultralytics import YOLO

    report_dir = AI_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_dir = AI_ROOT / "runs/parts" / a.name

    resolved_yaml = resolve_dataset_yaml(
        a.data, run_dir,
        scale=a.scale, mosaic=a.mosaic, mixup=a.mixup, degrees=a.degrees
    )
    device_selected = select_device(a.device)

    # Chế độ chỉ đánh giá
    if a.eval_only:
        model_eval_path = a.model or AI_ROOT / "models/weights/parts_seg.pt"
        if not model_eval_path.is_file():
            logger.error("Không tìm thấy model để đánh giá tại: %s", model_eval_path)
            sys.exit(1)
        logger.info("Chạy chế độ EVAL-ONLY với mô hình: %s", model_eval_path)
        eval_model = YOLO(str(model_eval_path))
        report = evaluate_model_and_confusion_matrix(eval_model, resolved_yaml, report_dir)
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ KIỂM ĐỊNH MÔ HÌNH BỘ PHẬN XE (7 LỚP DOCX):")
        print("=" * 60)
        print(json.dumps(report.get("overall_metrics", {}), indent=2))
        return

    # Xác định weights khởi tạo
    if a.model and Path(a.model).is_file():
        weights_path = Path(a.model).resolve()
    else:
        weights_path = get_or_download_backbone(a.backbone)

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU HUẤN LUYỆN YOLOV8-SEG (7 LỚP PHÂN ĐOẠN BỘ PHẬN XE - GIA ĐOẠN CẬN CẢNH)")
    logger.info("Backbone:    %s", weights_path)
    logger.info("Dataset:     %s", resolved_yaml)
    logger.info("Epochs:      %d | Patience: %d", a.epochs, a.patience)
    logger.info("Batch size:  %d | Img size: %d", a.batch, a.imgsz)
    logger.info("Optimizer:   %s | lr0: %.4f", a.optimizer, a.lr0)
    logger.info("Device:      %s", device_selected)
    logger.info("Augment:     scale=%.2f, mosaic=%.2f, mixup=%.2f, degrees=%.1f, translate=%.2f",
                a.scale, a.mosaic, a.mixup, a.degrees, a.translate)
    logger.info("=" * 70)

    model = YOLO(str(weights_path))

    # Huấn luyện kèm cơ chế tự động hạ batch size nếu gặp Out Of Memory
    current_batch = a.batch
    trained_successfully = False

    while current_batch >= 4:
        try:
            logger.info("Đang bắt đầu train với batch size: %d ...", current_batch)
            model.train(
                data=str(resolved_yaml),
                epochs=a.epochs,
                imgsz=a.imgsz,
                batch=current_batch,
                device=device_selected,
                workers=a.workers,
                patience=a.patience,
                lr0=a.lr0,
                optimizer=a.optimizer,
                project=str(run_dir.parent),
                name=a.name,
                exist_ok=True,
                plots=True,
                # Kỹ thuật Data Augmentation đặc thù góc hẹp & cận cảnh
                scale=a.scale,
                mosaic=a.mosaic,
                mixup=a.mixup,
                degrees=a.degrees,
                translate=a.translate,
                shear=2.0,
                perspective=0.0005,
                erasing=0.20,
            )
            trained_successfully = True
            break
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA Out of Memory với batch size %d! Đang tự động hạ batch size...", current_batch)
            torch.cuda.empty_cache()
            current_batch //= 2
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.warning("Phát hiện lỗi OOM: %s. Đang hạ batch size từ %d...", e, current_batch)
                torch.cuda.empty_cache()
                current_batch //= 2
            else:
                logger.error("Lỗi huấn luyện: %s", e, exc_info=True)
                raise e

    if not trained_successfully:
        logger.error("Không thể hoàn tất huấn luyện ngay cả với batch size nhỏ nhất.")
        sys.exit(1)

    # Lấy checkpoint tốt nhất
    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer else run_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights/best.pt"

    if best_pt.is_file():
        logger.info("Huấn luyện hoàn tất xuất sắc! Best checkpoint: %s", best_pt)
        target_path = a.export_target.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, target_path)
        logger.info("Đã tự động cập nhật checkpoint chuẩn vào: %s", target_path)

        # Chạy kiểm định và trích xuất Confusion Matrix cho mô hình tốt nhất
        best_model = YOLO(str(best_pt))
        report = evaluate_model_and_confusion_matrix(best_model, resolved_yaml, report_dir)
        print("\n" + "=" * 70)
        print("TỔNG HỢP CHẤT LƯỢNG MÔ HÌNH SAU HUẤN LUYỆN (mAP50-95 & METRICS):")
        print("=" * 70)
        print(json.dumps(report.get("overall_metrics", {}), indent=2))
        print("=" * 70)
    else:
        logger.warning("Không tìm thấy checkpoint best.pt tại %s", best_pt)

if __name__ == "__main__":
    main()
