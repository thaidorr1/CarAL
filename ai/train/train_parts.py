"""Train Car Part Segmentation (Model 2: 7 lớp bộ phận xe ngoại thất).
Sử dụng YOLOv8-Seg (mặc định yolov8n-seg.pt hoặc yolov8s-seg.pt).
Tự động lưu/copy checkpoint best.pt vào ai/models/parts_seg.pt để TwoStagePipeline sử dụng.
"""
import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Optional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime import AI_ROOT, select_device

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_parts")

PART_CLASSES = [
    "can_truoc",
    "can_sau",
    "nap_capo",
    "ve_che_banh",
    "cua_hong",
    "kinh_chieu_hau",
    "kinh_lai",
    "den_xe",
]


def resolve_parts_dataset(data_yaml_path: Path, run_dir: Path) -> Path:
    """Resolve YAML dataset cho parts, xác thực đường dẫn và kiểm tra thư mục dataset."""
    if not data_yaml_path.is_file():
        # Fallback thử tìm trong configs hoặc data/parts
        alt1 = AI_ROOT / "configs" / data_yaml_path.name
        alt2 = AI_ROOT / "data/parts" / data_yaml_path.name
        if alt1.is_file():
            data_yaml_path = alt1
        elif alt2.is_file():
            data_yaml_path = alt2
        else:
            raise FileNotFoundError(f"Config file not found: {data_yaml_path}")

    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", AI_ROOT / "data/parts"))
    if not root.is_absolute():
        root = AI_ROOT / root

    # Tự động đảm bảo thư mục split tồn tại
    for split in ("train", "val", "test"):
        split_rel = cfg.get(split, f"images/{split}")
        img_dir = root / split_rel
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    cfg["path"] = str(root.resolve()).replace("\\", "/")
    cfg["train"] = str(cfg.get("train", "images/train")).replace("\\", "/")
    cfg["val"] = str(cfg.get("val", "images/val")).replace("\\", "/")
    cfg["test"] = str(cfg.get("test", "images/val")).replace("\\", "/")
    cfg["names"] = {i: name for i, name in enumerate(PART_CLASSES)}

    resolved_file = run_dir / "dataset_parts.yaml"
    resolved_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_file.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return resolved_file.resolve()


def find_or_download_weights(model_arg: Optional[Path]) -> Path:
    """Tìm kiếm file weights hoặc tự động fallback sang ai/models/ hoặc tải về."""
    if model_arg and Path(model_arg).is_file():
        return Path(model_arg).resolve()

    # Danh sách các vị trí ưu tiên tìm kiếm
    candidates = [
        AI_ROOT / "models/yolov8n-seg.pt",
        AI_ROOT / "models/weights/parts_seg.pt",
        AI_ROOT / "yolov8n-seg.pt",
        Path("yolov8n-seg.pt").resolve(),
    ]

    if model_arg:
        candidates.insert(0, AI_ROOT / Path(model_arg).name)

    for c in candidates:
        if c.is_file():
            logger.info("Đã tìm thấy weights: %s", c)
            return c.resolve()

    # Nếu chưa có, tự động tải yolov8n-seg.pt về ai/models/
    target = AI_ROOT / "models/yolov8n-seg.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Chưa có weights sẵn, đang tự động tải yolov8n-seg.pt về %s ...", target)
    from ultralytics import YOLO
    temp_model = YOLO("yolov8n-seg.pt")  # Ultralytics tự tải về thư mục hiện hành
    shutil.copy2("yolov8n-seg.pt", target)
    return target.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Pretrained weights YOLOv8-Seg (mặc định tự tìm ai/models/parts_seg.pt hoặc yolov8n-seg.pt)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=AI_ROOT / "configs/parts.yaml",
        help="File cấu hình dataset bộ phận xe (YAML)",
    )
    p.add_argument("--epochs", type=int, default=50, help="Số lượng epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh inference/train")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--device", default="auto", help="Thiết bị: auto, 0, cpu...")
    p.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument(
        "--name",
        default="parts_seg_50epochs",
        help="Tên folder lưu kết quả trong runs/parts/",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--export-target",
        type=Path,
        default=AI_ROOT / "models/weights/parts_seg.pt",
        help="Đích đến tự động copy best.pt cho TwoStagePipeline",
    )
    p.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Ghi đè/tiếp tục trong run dir nếu đã tồn tại",
    )
    p.add_argument("--scale", type=float, default=0.7, help="Tỷ lệ scale jitter / zoom cận cảnh (0.5 - 1.5)")
    p.add_argument("--mosaic", type=float, default=1.0, help="Hệ số Mosaic augmentation đa quy mô")
    p.add_argument("--mixup", type=float, default=0.15, help="Hệ số Mixup augmentation kết hợp mẫu")
    p.add_argument("--degrees", type=float, default=10.0, help="Góc xoay augmentation ngẫu nhiên")
    p.add_argument("--translate", type=float, default=0.2, help="Dịch chuyển khung hình mô phỏng ảnh lệch tâm")
    p.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate for fine-tuning")
    p.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    a = p.parse_args()

    if a.epochs <= 0 or a.imgsz <= 0 or a.batch <= 0:
        p.error("epochs, imgsz và batch phải là số nguyên dương")

    model_path = find_or_download_weights(a.model)

    run_dir = AI_ROOT / "runs/parts" / a.name
    # KHÔNG ngắt lỗi nếu run dir tồn tại, tự động cho phép exist_ok

    from ultralytics import YOLO

    logger.info("Đang nạp model weights: %s", model_path)
    model = YOLO(str(model_path))

    resolved_data_yaml = resolve_parts_dataset(a.data, run_dir)
    device_selected = select_device(a.device)
    logger.info("Khởi động huấn luyện trên thiết bị: %s", device_selected)
    logger.info(
        "Augmentations cận cảnh: scale=%.2f, mosaic=%.2f, mixup=%.2f, translate=%.2f, degrees=%.1f",
        a.scale, a.mosaic, a.mixup, a.translate, a.degrees,
    )

    model.train(
        data=str(resolved_data_yaml),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=device_selected,
        workers=a.workers,
        cache=False,
        patience=a.patience,
        project=str(run_dir.parent),
        name=a.name,
        exist_ok=True,
        seed=a.seed,
        plots=True,
        # Data Augmentation chuyên sâu cho cận cảnh (Close-up & Multi-scale)
        scale=a.scale,
        mosaic=a.mosaic,
        mixup=a.mixup,
        translate=a.translate,
        degrees=a.degrees,
        lr0=a.lr0,
        lrf=a.lrf,
        perspective=0.0005,
        shear=2.0,
        erasing=0.2,
    )

    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer else run_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights/best.pt"

    if best_pt.is_file():
        logger.info("Huấn luyện hoàn tất! Validation best checkpoint: %s", best_pt)
        target_path = a.export_target.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, target_path)
        logger.info("Đã tự động cập nhật checkpoint cho TwoStagePipeline tại: %s", target_path)
    else:
        logger.warning("Không tìm thấy best.pt tại %s để tự động export.", best_pt)


if __name__ == "__main__":
    main()
