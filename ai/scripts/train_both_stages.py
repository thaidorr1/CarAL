"""Script Huấn Luyện Cả 2 Stage (Damage Detection & Parts Segmentation) - AI_CAR.

Mục tiêu kỹ thuật:
1. Quản lý đường dẫn weights cố định:
   - Bắt buộc trỏ đường dẫn local: YOLO("ai/models/yolov8s.pt") cho Stage 1.
   - Bắt buộc trỏ đường dẫn local: YOLO("ai/models/yolov8s-seg.pt") cho Stage 2.
   - Tuyệt đối KHÔNG truyền tên weights ngắn (như "yolov8s.pt") để chống tự động download weights rác ra root.
2. Stage 1 (Damage Detection):
   - Bổ sung negative samples (ảnh lốp xe/bánh xe bình thường với file label rỗng 0 bytes)
     để triệt tiêu hoàn toàn hiện tượng báo tire_flat giả trên bánh xe bình thường.
   - imgsz = 800, cls_loss (cls) = 1.5. Ngưỡng conf mặc định cho tire_flat >= 0.55.
   - Checkpoint xuất trực tiếp về ai/models/damage_best.pt (đồng thời đồng bộ sang ai/models/damage_seg.pt).
3. Stage 2 (Parts Segmentation):
   - Sử dụng tập dữ liệu parts bổ sung các mảng crop cận cảnh/trung cảnh (ai/configs/data_parts.yaml).
   - imgsz = 800, seg_loss gain (box) = 12.0.
   - Checkpoint xuất trực tiếp về ai/models/parts_seg.pt.
4. Tự động kích hoạt kiểm định toàn diện (Batch Audit):
   - Chạy ai/scripts/audit_full_dataset.py.
   - Đảm bảo tỷ lệ vi phạm quy tắc vật lý đạt 0.00%.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import yaml

# Cấu hình encoding UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Định vị thư mục gốc AI_ROOT
AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from runtime import select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_both_stages")

# 1. ĐƯỜNG DẪN CỐ ĐỊNH LOCAL CHO PRETRAINED BACKBONES
STAGE1_LOCAL_BACKBONE = (AI_ROOT / "models" / "yolov8s.pt").resolve()
STAGE2_LOCAL_BACKBONE = (AI_ROOT / "models" / "yolov8s-seg.pt").resolve()

# ĐƯỜNG DẪN ĐÍCH XUẤT WEIGHTS
STAGE1_EXPORT_TARGET = (AI_ROOT / "models" / "weights" / "damage_best.pt").resolve()
STAGE1_SEG_TARGET = (AI_ROOT / "models" / "weights" / "damage_seg.pt").resolve()
STAGE2_EXPORT_TARGET = (AI_ROOT / "models" / "weights" / "parts_seg.pt").resolve()


def ensure_local_backbones():
    """Kiểm tra và đảm bảo trọng số backbone tồn tại trong ai/models, cấm lưu ở root."""
    for name, path in [
        ("Stage 1 (yolov8s.pt)", STAGE1_LOCAL_BACKBONE),
        ("Stage 2 (yolov8s-seg.pt)", STAGE2_LOCAL_BACKBONE),
    ]:
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise FileNotFoundError(
                f"LỖI: Không tìm thấy trọng số backbone cố định cho {name} tại {path}!\n"
                f"Yêu cầu lưu file vào ai/models/, không được để ở thư mục gốc."
            )
        logger.info("Xác nhận backbone hợp lệ cho %s: %s (%d bytes)", name, path, path.stat().st_size)


def inject_normal_tire_negative_samples(max_negatives: int = 60) -> int:
    """Bổ sung các ảnh lốp xe/vè bánh bình thường làm Negative Samples cho Stage 1.
    
    YOLOv8 xử lý file nhãn .txt rỗng (0 bytes) như là background sample (không có tổn thất),
    giúp mô hình học được lốp xe nguyên vẹn KHÔNG phải là tire_flat (xẹp lốp).
    """
    yolo_img_dir = AI_ROOT / "data/yolo/images/train"
    yolo_lbl_dir = AI_ROOT / "data/yolo/labels/train"
    parts_img_dir = AI_ROOT / "data/parts/images/train"
    parts_lbl_dir = AI_ROOT / "data/parts/labels/train"

    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Đếm số negative samples đã inject trước đó
    existing_negs = list(yolo_lbl_dir.glob("neg_wheel_*.txt"))
    if len(existing_negs) >= max_negatives:
        logger.info("Đã có sẵn %d negative samples trong tập train YOLO. Bỏ qua inject.", len(existing_negs))
        return len(existing_negs)

    # Tìm các crop vè che bánh / vòm bánh xe trong tập parts
    fender_crops = []
    if parts_lbl_dir.is_dir():
        for f in parts_lbl_dir.glob("closeup_crop_*.txt"):
            try:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    if line.strip().startswith("3 "):  # Lớp 3 là ve_che_banh (chứa vòm bánh xe & lốp xe)
                        img_path = parts_img_dir / (f.stem + ".jpg")
                        if img_path.is_file():
                            fender_crops.append(img_path)
                        break
            except Exception:
                continue

    logger.info("Tìm thấy %d crop bánh xe/vè che bánh bình thường để tạo negative samples.", len(fender_crops))
    injected_count = 0

    for crop_img in fender_crops[:max_negatives]:
        target_img_name = f"neg_wheel_{crop_img.name}"
        target_img_path = yolo_img_dir / target_img_name
        target_lbl_path = yolo_lbl_dir / f"neg_wheel_{crop_img.stem}.txt"

        if not target_img_path.is_file():
            shutil.copy2(crop_img, target_img_path)
        # Tạo file label rỗng (0 bytes) đại diện cho negative background
        target_lbl_path.write_text("", encoding="utf-8")
        injected_count += 1

    total_negs = len(list(yolo_lbl_dir.glob("neg_wheel_*.txt")))
    logger.info("Hoàn tất tạo negative samples: Đã bổ sung %d mẫu lốp xe bình thường (Tổng: %d).", injected_count, total_negs)
    return total_negs


def train_stage1_damage(
    data_yaml: Path,
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 800,
    device: str = "0",
    name: str = "damage_yolov8s_800",
) -> Path:
    """Huấn luyện Stage 1: Damage Detection với cls_loss = 1.5, imgsz = 800."""
    from ultralytics import YOLO

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU HUẤN LUYỆN STAGE 1: DAMAGE DETECTION (YOLOv8s)")
    logger.info("Backbone:    %s", STAGE1_LOCAL_BACKBONE)
    logger.info("Dataset:     %s", data_yaml)
    logger.info("Epochs:      %d | Batch: %d | Imgsz: %d", epochs, batch, imgsz)
    logger.info("Cls Loss:    1.5 (Tăng phạt phân loại nhầm background / lốp xe)")
    logger.info("Device:      %s", device)
    logger.info("=" * 70)

    # BẮT BUỘC KHỞI TẠO TỪ ĐƯỜNG DẪN LOCAL CỐ ĐỊNH
    model = YOLO(str(STAGE1_LOCAL_BACKBONE))

    run_project = AI_ROOT / "runs/damage"
    run_dir = run_project / name

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        cls=1.5,             # Tăng trọng số cls_loss lên 1.5
        device=device,
        project=str(run_project),
        name=name,
        exist_ok=True,
        plots=True,
        scale=0.5,
        mosaic=1.0,
        mixup=0.15,
        degrees=10.0,
    )

    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer else run_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights/best.pt"

    stage1_metrics = {}
    if hasattr(model, "trainer") and hasattr(model.trainer, "metrics"):
        try:
            for k, v in model.trainer.metrics.items():
                stage1_metrics[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
        except Exception:
            pass

    if best_pt.is_file():
        logger.info("Stage 1 hoàn tất thành công! Best checkpoint: %s", best_pt)
        # Lưu vào damage_best.pt theo đúng chỉ định
        STAGE1_EXPORT_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, STAGE1_EXPORT_TARGET)
        logger.info("Đã xuất weights Stage 1 thành công: %s", STAGE1_EXPORT_TARGET)
        # Đồng bộ sang damage_seg.pt để tương thích toàn diện pipeline
        shutil.copy2(best_pt, STAGE1_SEG_TARGET)
        logger.info("Đã đồng bộ sang: %s", STAGE1_SEG_TARGET)
        return STAGE1_EXPORT_TARGET, stage1_metrics
    else:
        raise FileNotFoundError(f"Không tìm thấy best.pt sau khi train Stage 1 tại {best_pt}")


def train_stage2_parts(
    data_yaml: Path,
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 800,
    device: str = "0",
    name: str = "parts_yolov8s_seg_800",
) -> Path:
    """Huấn luyện Stage 2: Parts Segmentation với seg_loss gain (box) = 12.0, imgsz = 800."""
    from ultralytics import YOLO

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU HUẤN LUYỆN STAGE 2: PARTS SEGMENTATION (YOLOv8s-Seg)")
    logger.info("Backbone:    %s", STAGE2_LOCAL_BACKBONE)
    logger.info("Dataset:     %s (Bao gồm crop cận cảnh/trung cảnh)", data_yaml)
    logger.info("Epochs:      %d | Batch: %d | Imgsz: %d", epochs, batch, imgsz)
    logger.info("Seg Loss:    12.0 (Trọng số phạt phân đoạn chi tiết bộ phận xe)")
    logger.info("Device:      %s", device)
    logger.info("=" * 70)

    # BẮT BUỘC KHỞI TẠO TỪ ĐƯỜNG DẪN LOCAL CỐ ĐỊNH
    model = YOLO(str(STAGE2_LOCAL_BACKBONE))

    run_project = AI_ROOT / "runs/parts"
    run_dir = run_project / name

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        box=12.0,            # seg_loss gain = 12.0
        cls=1.5,
        device=device,
        project=str(run_project),
        name=name,
        exist_ok=True,
        plots=True,
        scale=0.6,
        mosaic=1.0,
        mixup=0.15,
        degrees=10.0,
    )

    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer else run_dir / "weights/best.pt"
    if not best_pt.is_file():
        best_pt = run_dir / "weights/best.pt"

    stage2_metrics = {}
    if hasattr(model, "trainer") and hasattr(model.trainer, "metrics"):
        try:
            for k, v in model.trainer.metrics.items():
                stage2_metrics[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
        except Exception:
            pass

    if best_pt.is_file():
        logger.info("Stage 2 hoàn tất thành công! Best checkpoint: %s", best_pt)
        STAGE2_EXPORT_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, STAGE2_EXPORT_TARGET)
        logger.info("Đã xuất weights Stage 2 thành công: %s", STAGE2_EXPORT_TARGET)
        return STAGE2_EXPORT_TARGET, stage2_metrics
    else:
        raise FileNotFoundError(f"Không tìm thấy best.pt sau khi train Stage 2 tại {best_pt}")


def run_batch_audit_and_retest(limit: int = 150) -> Dict:
    """Tự động chạy audit_full_dataset.py và kiểm tra tỷ lệ vi phạm quy tắc vật lý."""
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU KIỂM THỬ BATCH AUDIT VỚI TRỌNG SỐ MỚI (LIMIT = %d)", limit)
    logger.info("=" * 70)

    audit_script = AI_ROOT / "scripts/audit_full_dataset.py"
    cmd = [
        sys.executable,
        str(audit_script),
        "--limit", str(limit),
        "--damage-model", str(STAGE1_EXPORT_TARGET if STAGE1_EXPORT_TARGET.is_file() else STAGE1_SEG_TARGET),
        "--parts-model", str(STAGE2_EXPORT_TARGET),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        logger.error("Audit script gặp lỗi:\n%s", res.stderr)
        raise RuntimeError(f"Audit script failed with exit code {res.returncode}")

    report_path = AI_ROOT / "reports/full_dataset_audit_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy báo cáo audit tại {report_path}")

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report_data.get("summary", {})

    print("\n" + "=" * 70)
    print("BÁO CÁO KIỂM TOÁN TỔNG HỢP SAU KHI NÂNG CẤP WEIGHTS (BATCH AUDIT):")
    print("=" * 70)
    print(f"- Tổng số ảnh kiểm định:          {summary.get('total_images_scanned', 0)}")
    print(f"- Tổng số vết tổn thất phát hiện: {summary.get('total_damages_detected', 0)}")
    print(f"- Tỷ lệ vi phạm quy tắc vật lý:   {summary.get('physical_rule_violation_rate', '0.00%')}")
    print(f"- Tỷ lệ mapping thành công:       {summary.get('mapping_success_rate', '0.00%')}")
    print(f"- Phân bố Shot Type:              {summary.get('shot_type_distribution', {})}")
    print(f"- Chi tiết vi phạm theo Flag:     {summary.get('flag_violations', {})}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-stage1", type=int, default=30, help="Số epochs cho Stage 1 (mặc định: 30)")
    parser.add_argument("--epochs-stage2", type=int, default=30, help="Số epochs cho Stage 2 (mặc định: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (mặc định: 8)")
    parser.add_argument("--imgsz", type=int, default=800, help="Kích thước ảnh imgsz (mặc định: 800)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị: '0', 'cpu', 'auto'")
    parser.add_argument("--max-negatives", type=int, default=60, help="Số lượng negative samples lốp xe cần bổ sung")
    parser.add_argument("--skip-stage1", action="store_true", help="Bỏ qua huấn luyện Stage 1")
    parser.add_argument("--skip-stage2", action="store_true", help="Bỏ qua huấn luyện Stage 2")
    parser.add_argument("--audit-only", action="store_true", help="Chỉ chạy chuẩn bị dữ liệu và Audit kiểm định")
    parser.add_argument("--audit-limit", type=int, default=150, help="Số lượng ảnh tối đa để audit")
    args = parser.parse_args()

    # 1. Quản lý đường dẫn backbone cố định
    ensure_local_backbones()

    # 2. Bổ sung negative samples cho Stage 1
    inject_normal_tire_negative_samples(max_negatives=args.max_negatives)

    device_selected = select_device(args.device)
    data_stage1 = AI_ROOT / "configs/data.yaml"
    data_stage2 = AI_ROOT / "configs/data_parts.yaml"

    stage1_res_metrics = {}
    stage2_res_metrics = {}

    if not args.audit_only:
        # Huấn luyện Stage 1
        if not args.skip_stage1:
            _, stage1_res_metrics = train_stage1_damage(
                data_yaml=data_stage1,
                epochs=args.epochs_stage1,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device_selected,
            )

        # Huấn luyện Stage 2
        if not args.skip_stage2:
            _, stage2_res_metrics = train_stage2_parts(
                data_yaml=data_stage2,
                epochs=args.epochs_stage2,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device_selected,
            )

    # 3. Thực thi Batch Audit và Re-test
    summary = run_batch_audit_and_retest(limit=args.audit_limit)
    violation_rate = summary.get("physical_rule_violation_rate", "0.00%")

    # 4. Chạy toàn bộ Test Suite nghiệm thu
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU CHẠY TOÀN BỘ TEST SUITE AI/TESTS/ ĐỂ NGHIỆM THU")
    logger.info("=" * 70)
    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", "ai/tests", "-p", "test_*.py"]
    test_run = subprocess.run(test_cmd, capture_output=True, text=True, encoding="utf-8")
    tests_passed = (test_run.returncode == 0)
    if tests_passed:
        logger.info("XÁC NHẬN: 100% TOÀN BỘ CÁC UNIT TEST ĐỀU PASS THÀNH CÔNG!")
    else:
        logger.error("CẢNH BÁO: Phát hiện lỗi trong test suite:\n%s", test_run.stderr)

    # 5. Xuất báo cáo so sánh tổng hợp Retrain
    comparison_report = {
        "timestamp": json.loads((AI_ROOT / "reports/full_dataset_audit_report.json").read_text(encoding="utf-8")).get("audit_metadata", {}).get("timestamp", ""),
        "training_config": {
            "stage1_epochs": args.epochs_stage1,
            "stage2_epochs": args.epochs_stage2,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "cls_loss_stage1": 1.5,
            "seg_loss_stage2": 12.0,
            "device": device_selected,
        },
        "stage1_damage_metrics": stage1_res_metrics,
        "stage2_parts_metrics": stage2_res_metrics,
        "audit_summary": summary,
        "test_suite_passed": tests_passed,
    }
    comp_file = AI_ROOT / "reports/retrain_comparison_report.json"
    comp_file.write_text(json.dumps(comparison_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã xuất báo cáo so sánh Retrain tại: %s", comp_file)

    if violation_rate == "0.00%" and tests_passed:
        logger.info("XÁC NHẬN HOÀN TẤT: Tỷ lệ vi phạm quy tắc vật lý đạt chuẩn tuyệt đối 0.00% và 100% test suite pass!")
    else:
        logger.warning("CẢNH BÁO: Tỷ lệ vi phạm: %s | Tests passed: %s", violation_rate, tests_passed)


if __name__ == "__main__":
    main()
