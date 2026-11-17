"""
Validation script for AutoClaim-AI Stage 2 Backbone on 80 Pilot images.
Analyzes detection frequency and renders a 12-sample evaluation grid.
"""

import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = WORKSPACE_ROOT / "models" / "weights" / "parts_carparts_backbone_best.pt"
PILOT_IMG_DIR = WORKSPACE_ROOT / "data" / "parts_pilot" / "images"
OUT_GRID = WORKSPACE_ROOT.parent / "docs" / "references" / "pilot_backbone_eval_grid.jpg"

CLASS_NAMES = {
    0: "can_truoc",
    1: "can_sau",
    2: "nap_capo",
    3: "ve_che_banh",
    4: "cua_truoc",
    5: "cua_sau",
    6: "kinh_xe",
    7: "den_xe"
}

CLASS_COLORS = {
    0: (0, 140, 255),   # Orange: can_truoc
    1: (0, 0, 255),     # Red: can_sau
    2: (255, 0, 255),   # Magenta: nap_capo
    3: (255, 255, 0),   # Cyan: ve_che_banh
    4: (0, 255, 0),     # Bright Green: cua_truoc
    5: (0, 180, 100),   # Teal: cua_sau
    6: (255, 180, 50),  # Sky blue: kinh_xe
    7: (0, 255, 255),   # Yellow: den_xe
}

def run_pilot_evaluation():
    print("=" * 65)
    print("      EVALUATING STAGE 2 BACKBONE ON 80 PILOT IMAGES")
    print("=" * 65)
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Images:  {PILOT_IMG_DIR}")
    
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
        sys.exit(1)
        
    model = YOLO(str(WEIGHTS_PATH))
    
    pilot_images = sorted(glob.glob(str(PILOT_IMG_DIR / "*.jpg")))
    print(f"Total Pilot images: {len(pilot_images)}")
    
    # Run inference on all 80
    class_detections = defaultdict(int)
    images_with_detections = 0
    total_polygons = 0
    
    image_results = {}
    
    for img_path in pilot_images:
        filename = os.path.basename(img_path)
        res = model.predict(img_path, conf=0.35, iou=0.5, verbose=False)[0]
        
        shapes = []
        if res.masks is not None and len(res.masks) > 0:
            images_with_detections += 1
            clses = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            masks_xy = res.masks.xy
            
            for i, pts in enumerate(masks_xy):
                cid = clses[i]
                conf = confs[i]
                class_detections[cid] += 1
                total_polygons += 1
                shapes.append({
                    "cid": cid,
                    "conf": float(conf),
                    "pts": pts
                })
                
        image_results[filename] = {
            "path": img_path,
            "shapes": shapes
        }
        
    print("\n" + "=" * 65)
    print("            PILOT 80 IMAGES DETECTION SUMMARY")
    print("=" * 65)
    print(f"Số ảnh có phát hiện ít nhất 1 bộ phận: {images_with_detections} / {len(pilot_images)} ({images_with_detections/len(pilot_images)*100:.1f}%)")
    print(f"Tổng số polygon phát hiện: {total_polygons} (Trung bình {total_polygons/len(pilot_images):.1f} parts/ảnh)")
    print("\nTần suất phát hiện theo 8 lớp:")
    for cid in range(8):
        cname = CLASS_NAMES[cid]
        cnt = class_detections[cid]
        pct = (cnt / total_polygons * 100) if total_polygons > 0 else 0
        print(f"  Class {cid} ({cname:12s}): {cnt:4d} instances ({pct:5.1f}%)")
        
    # Render 12 representative images
    # We want: 003848.jpg (test sample), diverse angles, front, rear, side, heavy damage
    priority_stems = [
        "003848", "001977", "003409", "000821",
        "001489", "002636", "003359", "002464",
        "000714", "001081", "003773", "003947"
    ]
    
    selected_files = []
    for stem in priority_stems:
        fname = f"{stem}.jpg"
        if fname in image_results:
            selected_files.append(fname)
            
    # If not enough, fill with others
    if len(selected_files) < 12:
        for fname in sorted(image_results.keys()):
            if fname not in selected_files:
                selected_files.append(fname)
            if len(selected_files) == 12:
                break
                
    selected_files = selected_files[:12]
    print(f"\n12 ảnh tiêu biểu được chọn cho Grid: {', '.join(selected_files)}")
    
    # Render tiles
    target_w, target_h = 480, 360
    tiles = []
    
    for fname in selected_files:
        info = image_results[fname]
        img = cv2.imread(info["path"])
        h, w = img.shape[:2]
        
        overlay = img.copy()
        for s in info["shapes"]:
            cid = s["cid"]
            pts = s["pts"].astype(np.int32)
            color = CLASS_COLORS.get(cid, (200, 200, 200))
            cv2.fillPoly(overlay, [pts], color)
            
        alpha = 0.40
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        for s in info["shapes"]:
            cid = s["cid"]
            conf = s["conf"]
            pts = s["pts"].astype(np.int32)
            color = CLASS_COLORS.get(cid, (200, 200, 200))
            cv2.polylines(blended, [pts], True, color, 2, cv2.LINE_AA)
            
            # Text badge
            m = cv2.moments(pts)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                cx, cy = pts[0]
                
            badge = f"{CLASS_NAMES[cid]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(blended, (cx - 2, cy - th - 3), (cx + tw + 2, cy + 2), (15, 15, 15), -1)
            cv2.putText(blended, badge, (cx, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
        # Top banner
        cv2.rectangle(blended, (0, 0), (w, 28), (25, 25, 25), -1)
        bar_text = f"{fname} | {len(info['shapes'])} detections"
        cv2.putText(blended, bar_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        
        resized = cv2.resize(blended, (target_w, target_h), interpolation=cv2.INTER_AREA)
        tiles.append(resized)
        
    # Grid 4 cols x 3 rows
    row1 = np.hstack(tiles[0:4])
    row2 = np.hstack(tiles[4:8])
    row3 = np.hstack(tiles[8:12])
    grid = np.vstack([row1, row2, row3])
    
    # Title Header Banner
    header = np.zeros((52, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (18, 18, 18)
    title = "AutoClaim-AI Stage 2 Backbone Evaluation -- 12 Representative Pilot Accident Samples (conf=0.35)"
    cv2.putText(header, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 250, 180), 2, cv2.LINE_AA)
    
    final_grid = np.vstack([header, grid])
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_GRID), final_grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n[OK] Saved evaluation grid to: {OUT_GRID}")
    print(f"Dimensions: {final_grid.shape[1]}x{final_grid.shape[0]} px, Size: {OUT_GRID.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    run_pilot_evaluation()
