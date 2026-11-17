"""
Evaluation script for AutoClaim-AI Stage 2 Backbone on the Test split (276 images).
Records Box & Mask mAP@50 and mAP@50-95 overall and per-class.
"""

import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = WORKSPACE_ROOT / "configs" / "data_parts_carparts_8cls.yaml"
WEIGHTS_PATH = WORKSPACE_ROOT / "models" / "weights" / "parts_carparts_backbone_best.pt"

CLASS_NAMES = [
    "can_truoc",
    "can_sau",
    "nap_capo",
    "ve_che_banh",
    "cua_truoc",
    "cua_sau",
    "kinh_xe",
    "den_xe"
]

def evaluate_test():
    print("=" * 65)
    print("      EVALUATING STAGE 2 BACKBONE ON TEST SET (276 IMAGES)")
    print("=" * 65)
    print(f"Model: {WEIGHTS_PATH}")
    print(f"Data:  {DATA_CONFIG}")
    
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Weights not found at {WEIGHTS_PATH}")
        sys.exit(1)
        
    model = YOLO(str(WEIGHTS_PATH))
    
    # Run validation on test split
    metrics = model.val(
        data=str(DATA_CONFIG),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        verbose=True,
        save_json=False
    )
    
    print("\n" + "=" * 65)
    print("                   TEST SET EVALUATION RESULTS")
    print("=" * 65)
    
    # Overall metrics
    box_map50 = metrics.box.map50
    box_map = metrics.box.map
    mask_map50 = metrics.seg.map50
    mask_map = metrics.seg.map
    
    print(f"Overall Box  mAP@50:    {box_map50 * 100:.2f}%")
    print(f"Overall Box  mAP@50-95: {box_map * 100:.2f}%")
    print(f"Overall Mask mAP@50:    {mask_map50 * 100:.2f}%")
    print(f"Overall Mask mAP@50-95: {mask_map * 100:.2f}%")
    
    # Per-class metrics
    print("\nChi tiết từng lớp (Per-Class Performance):")
    print(f"{'ID':<3} {'Tên Lớp':<15} {'Box mAP50':<12} {'Box mAP50-95':<14} {'Mask mAP50':<12} {'Mask mAP50-95':<14}")
    print("-" * 72)
    
    # Active classes mapping
    active_indices = list(metrics.box.ap_class_index)
    
    per_class_results = {}
    for i, cname in enumerate(CLASS_NAMES):
        if i not in active_indices:
            b50 = 0.0
            b95 = 0.0
            m50 = 0.0
            m95 = 0.0
            note = "(0 mẫu)"
        else:
            idx = active_indices.index(i)
            b50 = metrics.box.class_result(idx)[2] * 100
            b95 = metrics.box.class_result(idx)[3] * 100
            m50 = metrics.seg.class_result(idx)[2] * 100
            m95 = metrics.seg.class_result(idx)[3] * 100
            note = ""
                
        per_class_results[cname] = {
            "id": i,
            "box_map50": round(b50, 2),
            "box_map50_95": round(b95, 2),
            "mask_map50": round(m50, 2),
            "mask_map50_95": round(m95, 2)
        }
        print(f"{i:<3} {cname:<15} {b50:6.2f}%      {b95:6.2f}%        {m50:6.2f}%      {m95:6.2f}% {note}")
        
    out_json = WORKSPACE_ROOT / "runs" / "parts_backbone_8cls" / "test_eval_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "overall": {
                "box_map50": round(box_map50 * 100, 2),
                "box_map50_95": round(box_map * 100, 2),
                "mask_map50": round(mask_map50 * 100, 2),
                "mask_map50_95": round(mask_map * 100, 2)
            },
            "classes": per_class_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OK] Metrics saved to {out_json}")
    return metrics

if __name__ == "__main__":
    evaluate_test()
