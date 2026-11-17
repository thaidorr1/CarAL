"""
Training script for AutoClaim-AI Stage 2 Backbone (YOLOv8s-seg on carparts-seg 8-class).
"""

import os
import sys
import shutil
import time
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = WORKSPACE_ROOT / "configs" / "data_parts_carparts_8cls.yaml"
BASE_WEIGHTS = WORKSPACE_ROOT / "models" / "yolov8s-seg.pt"
RUNS_DIR = WORKSPACE_ROOT / "runs"
RUN_NAME = "parts_backbone_8cls"
DEST_WEIGHTS = WORKSPACE_ROOT / "models" / "weights" / "parts_carparts_backbone_best.pt"

def train_backbone():
    print("=" * 65)
    print("      AUTOCLAIM-AI — STAGE 2 BACKBONE TRAINING (YOLOv8s-seg)")
    print("=" * 65)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[THIẾT BỊ] GPU: {gpu_name} ({vram:.2f} GB VRAM) — Device ID: {device}")
    else:
        print("[THIẾT BỊ] CPU (Cảnh báo: Tốc độ sẽ chậm hơn đáng kể)")
        
    print(f"[CẤU HÌNH DỮ LIỆU] {DATA_CONFIG}")
    print(f"[TRỌNG SỐ KHỞI TẠO] {BASE_WEIGHTS}")
    print(f"[NƠI LƯU RUNS] {RUNS_DIR / RUN_NAME}")
    print(f"[TRỌNG SỐ ĐÍCH] {DEST_WEIGHTS}")
    
    model = YOLO(str(BASE_WEIGHTS))
    
    batch_size = 16 if torch.cuda.is_available() else 4
    
    start_time = time.time()
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=60,
        patience=15,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=2,
        cos_lr=True,
        scale=0.6,
        degrees=10.0,
        mosaic=1.0,
        fliplr=0.5,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=42,
        save=True,
        plots=True,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n[HOÀN THÀNH HUẤN LUYỆN] Thời gian thực thi: {elapsed/60:.2f} phút")
    
    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if best_pt.exists():
        DEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, DEST_WEIGHTS)
        print(f"[OK] Đã sao chép trọng số tốt nhất sang: {DEST_WEIGHTS}")
        print(f"Kích thước file: {DEST_WEIGHTS.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"[LỖI] Không tìm thấy file {best_pt}")

if __name__ == "__main__":
    train_backbone()
